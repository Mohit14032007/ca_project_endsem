#pragma once

#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "prefetch/prefetch_config.h"
#include "prefetch/predictor_interface.h"
#include "prefetch/predictor_registry.h"
#include "prefetch/prefetch_candidate_queue.h"

namespace llm_system {

class Device;

/**
 * Event-level live prediction metrics.
 */
struct LiveEventMetrics {
    int event_id = 0;
    int request_id = 0;
    int token_id = 0;
    int source_stage_id = -1;
    int target_stage_id = 0;
    int horizon = 1;
    int budget = 4;
    std::string predictor_type;
    std::vector<int> source_experts;
    std::vector<int> actual_target_experts;
    std::vector<int> predicted_candidates;
    std::vector<double> candidate_scores;
    int hits = 0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    double hit_rate = 0.0;
    double mrr = 0.0;
    double ndcg = 0.0;
    double prediction_latency_ns = 0.0;
    double update_latency_ns = 0.0;
};

/**
 * Pending prediction record held between stage routing start and stage router resolution.
 */
struct PendingStagePrediction {
    int event_id = 0;
    int request_id = 0;
    int seq_id = 0;
    int token_id = 0;
    int source_stage_id = -1;
    int target_stage_id = 0;
    std::vector<int> source_experts;
    PredictionResult result;
    std::chrono::high_resolution_clock::time_point t_predict_start;
    double prediction_latency_ns = 0.0;
};

/**
 * Metric accumulator for live prediction statistics.
 */
struct LiveMetricAccumulator {
    int count = 0;
    int total_hits = 0;
    double sum_precision = 0.0;
    double sum_recall = 0.0;
    double sum_f1 = 0.0;
    double sum_hit_rate = 0.0;
    double sum_mrr = 0.0;
    double sum_ndcg = 0.0;
    double sum_pred_latency_ns = 0.0;
    double sum_upd_latency_ns = 0.0;

    void add(const LiveEventMetrics& m) {
        count = count + 1;
        total_hits = total_hits + m.hits;
        sum_precision = sum_precision + m.precision;
        sum_recall = sum_recall + m.recall;
        sum_f1 = sum_f1 + m.f1;
        sum_hit_rate = sum_hit_rate + m.hit_rate;
        sum_mrr = sum_mrr + m.mrr;
        sum_ndcg = sum_ndcg + m.ndcg;
        sum_pred_latency_ns = sum_pred_latency_ns + m.prediction_latency_ns;
        sum_upd_latency_ns = sum_upd_latency_ns + m.update_latency_ns;
    }

    double mean_precision() const {
        if (count > 0) {
            return sum_precision / static_cast<double>(count);
        }
        return 0.0;
    }

    double mean_recall() const {
        if (count > 0) {
            return sum_recall / static_cast<double>(count);
        }
        return 0.0;
    }

    double mean_f1() const {
        if (count > 0) {
            return sum_f1 / static_cast<double>(count);
        }
        return 0.0;
    }

    double mean_hit_rate() const {
        if (count > 0) {
            return sum_hit_rate / static_cast<double>(count);
        }
        return 0.0;
    }

    double mean_mrr() const {
        if (count > 0) {
            return sum_mrr / static_cast<double>(count);
        }
        return 0.0;
    }

    double mean_ndcg() const {
        if (count > 0) {
            return sum_ndcg / static_cast<double>(count);
        }
        return 0.0;
    }

    double mean_pred_latency_ns() const {
        if (count > 0) {
            return sum_pred_latency_ns / static_cast<double>(count);
        }
        return 0.0;
    }

    double mean_upd_latency_ns() const {
        if (count > 0) {
            return sum_upd_latency_ns / static_cast<double>(count);
        }
        return 0.0;
    }
};

/**
 * Central engine managing live predictor lifecycle, causal hooks, logging,
 * and metric evaluation inside LLMSimulator.
 */
class LivePredictionEngine {
public:
    using Ptr = std::shared_ptr<LivePredictionEngine>;

    LivePredictionEngine(
        const PrefetchConfig& config,
        int num_stages,
        int num_experts,
        int top_k,
        const std::string& output_dir = "results/phase_2c_prediction_only");

    ~LivePredictionEngine();

    // Causal Hooks invoked by BatchedSequence::update_expert
    void on_request_start(int request_id);
    void on_stage_routing_start(int request_id, int token_id, int target_stage_id, int seq_id = 0);
    void on_stage_routing_complete(
        int request_id,
        int token_id,
        int target_stage_id,
        const std::vector<int>& actual_target_experts,
        int seq_id = 0,
        double sim_time_ns = 0.0);

    // Finalization: closes logs, computes overhead, writes summary JSON
    void finalize(double total_host_execution_seconds = 0.0);

    // Phase 4D.2 Candidate Dispatch to Device Transfer Engine
    int dispatch_candidates(std::shared_ptr<Device> device, double current_time_ns, int budget = -1);

    // Phase 4D.4 PABI Pin Expiration Coordination
    int expire_pins_for_stage(std::shared_ptr<Device> device, int target_stage_id);

    // Accessors & Diagnostics
    IPredictor::Ptr get_predictor() const { return predictor_; }
    const LiveMetricAccumulator& get_overall_accumulator() const { return acc_overall_; }
    int get_prediction_count() const { return next_event_id_; }
    int get_update_count() const { return update_events_count_; }
    int get_invalid_count() const { return invalid_events_count_; }
    std::string get_csv_path() const { return csv_log_path_; }
    std::string get_summary_path() const { return summary_json_path_; }
    PrefetchCandidateQueue::Ptr get_candidate_queue() const { return candidate_queue_; }
    SchedulingMode get_scheduling_mode() const { return config_.scheduling_mode; }
    void set_scheduling_mode(SchedulingMode mode) { config_.scheduling_mode = mode; }
    void set_horizon(int h) { horizon_ = h; config_.prediction_horizon = h; }
    int get_horizon() const { return horizon_; }

private:
    PrefetchConfig config_;
    int num_stages_;
    int num_experts_;
    int top_k_;
    int horizon_;
    int budget_;
    std::string output_dir_;
    std::string csv_log_path_;
    std::string summary_json_path_;

    IPredictor::Ptr predictor_;
    PrefetchCandidateQueue::Ptr candidate_queue_ = nullptr;
    std::ofstream csv_file_;

    int next_event_id_ = 0;
    int update_events_count_ = 0;
    int invalid_events_count_ = 0;

    std::unordered_set<int> seen_requests_;

    static uint64_t make_key(int req, int tok, int stage) {
        return (static_cast<uint64_t>(req) << 36) |
               ((static_cast<uint64_t>(tok) & 0x3FFFFULL) << 18) |
               (static_cast<uint64_t>(stage) & 0x3FFFFULL);
    }

    // Pending predictions: key = (request_id, token_id, target_stage_id)
    std::unordered_map<uint64_t, PendingStagePrediction> pending_predictions_;

    // History of actual experts per stage for current sequences:
    // key = (request_id, token_id, stage_id) -> vector of expert IDs
    std::unordered_map<uint64_t, std::vector<int>> stage_actual_experts_;

    // Stratified metric accumulators
    LiveMetricAccumulator acc_overall_;
    LiveMetricAccumulator acc_cold_;
    LiveMetricAccumulator acc_warm_;
    LiveMetricAccumulator acc_steady_;

    bool finalized_ = false;

    LiveEventMetrics compute_metrics(
        int event_id,
        int request_id,
        int token_id,
        int source_stage,
        int target_stage,
        const std::vector<int>& source_experts,
        const std::vector<int>& actual_target_experts,
        const PredictionResult& pred_result,
        double pred_latency_ns,
        double upd_latency_ns);
};

} // namespace llm_system

