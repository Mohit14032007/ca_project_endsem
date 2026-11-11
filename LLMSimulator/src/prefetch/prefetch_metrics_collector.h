#pragma once

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <iostream>

namespace llm_system {

/**
 * Event types tracked by the centralized PrefetchMetricsCollector.
 */
enum class CollectorEventType {
    PREDICTION_GENERATED,
    CANDIDATE_ADMITTED,
    CANDIDATE_REJECTED,
    CANDIDATE_DISPATCHED,
    TRANSFER_STARTED,
    TRANSFER_PAUSED,
    TRANSFER_RESUMED,
    TRANSFER_COMPLETED,
    TRANSFER_CANCELLED,
    HBM_RESIDENT,
    HBM_EVICTED,
    EXPERT_DEMANDED,
    DEMAND_ATTACHED_TO_PREFETCH,
    DEMAND_TRANSFER_STARTED,
    EXPERT_READY,
    REQUEST_STARTED,
    REQUEST_COMPLETED,
    TOKEN_STARTED,
    TOKEN_COMPLETED
};

std::string to_string(CollectorEventType event_type);

struct CollectorPredictionRecord {
    uint64_t event_id = 0;
    int request_id = 0;
    int seq_id = 0;
    int token_id = 0;
    int source_stage = -1;
    int target_stage = -1;
    int horizon = 1;
    std::string predictor_type;
    int budget = 0;
    int candidate_rank = 0;
    int predicted_expert_id = -1;
    double confidence_score = 0.0;
    std::vector<int> actual_target_experts;
    int is_hit = 0;
    double predict_latency_ns = 0.0;
    double update_latency_ns = 0.0;
    double timestamp_ns = 0.0;
};

struct CollectorCandidateRecord {
    uint64_t candidate_id = 0;
    std::string event_type; // "ADMITTED", "REJECTED", "DISPATCHED", "EXPIRED", "CANCELLED"
    int request_id = 0;
    int seq_id = 0;
    int token_id = 0;
    int source_stage = -1;
    int target_stage = -1;
    int expert_id = -1;
    int prediction_rank = 0;
    double prediction_score = 0.0;
    std::string reason;
    double timestamp_ns = 0.0;
};

struct CollectorTransferRecord {
    uint64_t transfer_id = 0;
    uint64_t candidate_id = 0;
    int request_id = 0;
    int seq_id = 0;
    int token_id = 0;
    int target_stage = -1;
    int expert_id = -1;
    std::string transfer_class; // "PREFETCH" or "DEMAND"
    int priority = 0;
    uint64_t size_bytes = 0;
    double queue_submit_ts = 0.0;
    double service_start_ts = -1.0;
    double completion_ts = -1.0;
    double duration_ns = 0.0;
    double paused_duration_ns = 0.0;
    std::string outcome; // "DEMANDED", "UNUSED", "CANCELLED", "EVICTED_BEFORE_DEMAND", "STILL_IN_FLIGHT_AT_END"
};

struct CollectorDemandRecord {
    uint64_t demand_id = 0;
    int request_id = 0;
    int seq_id = 0;
    int token_id = 0;
    int decode_step = -1;
    int layer_id = -1;
    int expert_id = -1;
    double demand_ts = 0.0;
    double ready_ts = 0.0;
    double ddr5_wait_ns = 0.0;
    double hbm_access_ns = 0.0;
    std::string residency_status; // "RESIDENT_HIT", "TIMELY_PREFETCH", "LATE_IN_FLIGHT", "DEMAND_MISS"
    uint64_t transfer_id = 0;
    uint64_t candidate_id = 0;
};

struct CollectorRequestLatencyRecord {
    int request_id = 0;
    int seq_id = 0;
    int prompt_len = 0;
    int gen_tokens = 0;
    double arrival_ts = 0.0;
    double start_ts = 0.0;
    double end_ts = 0.0;
    double latency_ns = 0.0;
};

struct CollectorTokenLatencyRecord {
    int request_id = 0;
    int seq_id = 0;
    int token_id = 0;
    int gen_step = 0;
    double token_start_ts = 0.0;
    double token_end_ts = 0.0;
    double token_duration_ns = 0.0;
    double cumulative_latency_ns = 0.0;
};

struct LatencyQuantiles {
    uint64_t count = 0;
    double mean_ns = 0.0;
    double p50_ns = 0.0;
    double p90_ns = 0.0;
    double p95_ns = 0.0;
    double p99_ns = 0.0;
    double max_ns = 0.0;
};

struct AggregatedPrefetchMetrics {
    // 5.A Prefetch Coverage
    uint64_t total_actual_demands = 0;
    uint64_t demands_with_prediction = 0;
    uint64_t demands_without_prediction = 0;
    uint64_t prefetch_assisted_demands = 0; // timely + late
    double overall_coverage = 0.0;
    double predicted_coverage = 0.0;
    double completed_prefetch_utilization = 0.0;

    // Unique Expert Invocations (deduplicated across W1, W3, W2 projections)
    uint64_t unique_actual_demands = 0;
    uint64_t unique_demands_with_prediction = 0;
    uint64_t unique_demands_without_prediction = 0;
    uint64_t unique_prefetch_assisted_demands = 0;
    uint64_t unique_timely_resident_hits = 0;
    uint64_t unique_late_inflight_attachments = 0;
    uint64_t unique_resident_hits = 0;
    uint64_t unique_fresh_demand_misses = 0;
    double unique_overall_coverage = 0.0;
    double unique_predicted_coverage = 0.0;
    double unique_overall_hbm_hit_rate = 0.0;
    double unique_prefetch_assisted_hit_rate = 0.0;
    double unique_fresh_demand_miss_rate = 0.0;
    double unique_inflight_attachment_rate = 0.0;
    double unique_timely_rate_of_demands = 0.0;
    double unique_late_rate_of_demands = 0.0;

    // 5.B Timely and Late Rates
    uint64_t timely_resident_hits = 0;
    uint64_t late_inflight_attachments = 0;
    double timely_rate_of_demands = 0.0;
    double late_rate_of_demands = 0.0;
    double timely_rate_of_dispatched = 0.0;
    double late_rate_of_dispatched = 0.0;
    uint64_t total_dispatched_prefetches = 0;
    uint64_t candidates_rejected_admission = 0;
    uint64_t duplicate_prefetches_skipped = 0;
    uint64_t transfers_cancelled = 0;
    double avg_remaining_late_wait_ns = 0.0;

    // 5.C Unused Prefetches
    uint64_t prefetches_completed_and_used = 0;
    uint64_t prefetches_completed_unused = 0;
    uint64_t prefetches_evicted_before_demand = 0;
    uint64_t prefetches_cancelled_inflight = 0;
    uint64_t prefetches_still_inflight_at_end = 0;
    uint64_t total_completed_prefetches = 0;
    uint64_t completed_unused_bytes = 0;
    uint64_t cancelled_partial_bytes = 0;
    uint64_t total_wasted_speculative_bytes = 0;

    // 5.D HBM Hit Rate
    uint64_t total_hbm_resident_hits = 0;
    uint64_t fresh_demand_misses = 0;
    double overall_hbm_hit_rate = 0.0;
    double prefetch_assisted_hit_rate = 0.0;
    double fresh_demand_miss_rate = 0.0;
    double inflight_attachment_rate = 0.0;

    // Physical Coalesced Demands & Hits (Phase 1.6 Ground Truth)
    uint64_t coalesced_actual_demands = 0;
    uint64_t routed_actual_demands = 0;
    uint64_t tensor_actual_demands = 0;
    uint64_t invalid_decode_step_demands = 0;
    uint64_t coalesced_resident_hits = 0;
    uint64_t coalesced_timely_resident_hits = 0;
    uint64_t coalesced_late_inflight_attachments = 0;
    uint64_t coalesced_fresh_demand_misses = 0;
    double coalesced_hbm_hit_rate = 0.0;

    // HBM Eviction Classification & Pollution Metrics (Point 4)
    uint64_t total_evictions = 0;
    uint64_t unconsumed_prefetch_evictions = 0;
    uint64_t demand_resident_evictions = 0;
    uint64_t pabi_speculative_evictions = 0;
    uint64_t prefetch_induced_evictions = 0;
    uint64_t reloaded_experts_count = 0;
    long long effective_capacity_bytes = 0;
    std::string eviction_policy = "lru";
    int fa_lru_window_size = 16;

    // 5.E Additional Memory Traffic
    uint64_t speculative_bytes_transferred = 0;
    uint64_t demand_bytes_transferred = 0;
    uint64_t total_ddr5_bytes_transferred = 0;
    uint64_t baseline_demand_bytes = 0;
    double traffic_overhead_ratio = 0.0;

    // 5.F Exposed Memory Stall Time
    double aggregate_demand_wait_ns = 0.0;
    double exposed_gpu_stall_ns = 0.0;
    double stall_hidden_by_prefetch_ns = 0.0;
    double wait_per_token_ns = 0.0;
    double wait_per_unique_demand_ns = 0.0;
    double total_simulated_execution_ns = 0.0;

    // 5.G Expert-Load Latency
    double mean_expert_load_latency_ns = 0.0;
    double p50_expert_load_latency_ns = 0.0;
    double p90_expert_load_latency_ns = 0.0;
    double p95_expert_load_latency_ns = 0.0;
    double p99_expert_load_latency_ns = 0.0;
    double max_expert_load_latency_ns = 0.0;

    // 5.H End-to-End and Per-Token Latency
    double mean_request_latency_ns = 0.0;
    double p50_request_latency_ns = 0.0;
    double p90_request_latency_ns = 0.0;
    double p95_request_latency_ns = 0.0;
    double p99_request_latency_ns = 0.0;
    double max_request_latency_ns = 0.0;

    double mean_token_latency_ns = 0.0;
    double p50_token_latency_ns = 0.0;
    double p90_token_latency_ns = 0.0;
    double p95_token_latency_ns = 0.0;
    double p99_token_latency_ns = 0.0;
    double max_token_latency_ns = 0.0;
    double simulated_tokens_per_second = 0.0;

    // Category-Separated Latency Distributions (Metric 7)
    LatencyQuantiles expert_latency_overall;
    LatencyQuantiles expert_latency_resident_hit;
    LatencyQuantiles expert_latency_timely_prefetch;
    LatencyQuantiles expert_latency_late_attachment;
    LatencyQuantiles expert_latency_demand_miss;

    // Unique-level Latency Distributions
    LatencyQuantiles unique_expert_latency_overall;
    LatencyQuantiles unique_expert_latency_resident_hit;
    LatencyQuantiles unique_expert_latency_timely_prefetch;
    LatencyQuantiles unique_expert_latency_late_attachment;
    LatencyQuantiles unique_expert_latency_demand_miss;

    // Token and Request Full Quantiles (Metric 8)
    LatencyQuantiles token_latency_quantiles;
    LatencyQuantiles request_latency_quantiles;
};

struct Key3TupleHash {
    std::size_t operator()(const std::tuple<int, int, int>& k) const {
        std::size_t h1 = std::hash<int>{}(std::get<0>(k));
        std::size_t h2 = std::hash<int>{}(std::get<1>(k));
        std::size_t h3 = std::hash<int>{}(std::get<2>(k));
        std::size_t seed = h1;
        seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

/**
 * Centralized, purely observational prefetch metrics collector.
 */
class PrefetchMetricsCollector {
public:
    static PrefetchMetricsCollector& getInstance();

    // Configuration & Metadata
    void set_output_dir(const std::string& dir);
    std::string get_output_dir() const;
    void set_experiment_metadata(const std::string& dataset, int batch_size,
                                 const std::string& predictor, int horizon, int budget,
                                 const std::string& config_label);
    void set_enabled(bool enabled);
    bool is_enabled() const;

    // Lifecycle Event Observation Hooks
    void record_prediction(int request_id, int seq_id, int token_id,
                           int source_stage, int target_stage, int horizon,
                           const std::string& predictor_type, int budget,
                           int rank, int predicted_expert, double confidence,
                           const std::vector<int>& actual_experts, int is_hit,
                           double pred_latency_ns, double upd_latency_ns,
                           double timestamp_ns);

    void record_candidate_admitted(uint64_t candidate_id, int request_id, int seq_id, int token_id,
                                   int source_stage, int target_stage, int expert_id,
                                   int rank, double score, double timestamp_ns);

    void record_candidate_rejected(int request_id, int seq_id, int token_id,
                                   int source_stage, int target_stage, int expert_id,
                                   const std::string& reason, double timestamp_ns);

    void record_candidate_dispatched(uint64_t candidate_id, int request_id, int seq_id, int token_id,
                                     int target_stage, int expert_id, double timestamp_ns);

    void record_transfer_submitted(uint64_t transfer_id, uint64_t candidate_id,
                                   int request_id, int seq_id, int token_id,
                                   int target_stage, int expert_id,
                                   const std::string& transfer_class, int priority,
                                   uint64_t bytes, double queue_submit_ts);

    void record_transfer_started(uint64_t transfer_id, double service_start_ts);

    void record_transfer_paused(uint64_t transfer_id, double paused_ts, double remaining_bytes);

    void record_transfer_resumed(uint64_t transfer_id, double resumed_ts);

    void record_transfer_completed(uint64_t transfer_id, double completion_ts);

    void record_transfer_cancelled(uint64_t transfer_id, double cancelled_ts, uint64_t transferred_bytes);

    void record_transfer_outcome(uint64_t transfer_id, const std::string& outcome);

    void record_hbm_resident(int layer_id, int expert_id, uint64_t transfer_id,
                             const std::string& classification, double timestamp_ns);

    void record_hbm_evicted(int layer_id, int expert_id, const std::string& reason, double timestamp_ns);

    void record_prefetch_induced_eviction(int layer_id, int expert_id, double timestamp_ns);

    void set_hbm_metrics(long long capacity_bytes, const std::string& policy_name, int window_size, uint64_t reloads);
    void set_current_decode_step(int step) { current_decode_step_ = step; }

    void record_expert_demanded(uint64_t demand_id, int request_id, int seq_id, int token_id,
                                int layer_id, int expert_id, double demand_ts, double ready_ts,
                                double ddr5_wait_ns, double hbm_access_ns,
                                const std::string& residency_status, uint64_t transfer_id, uint64_t candidate_id);

    void record_expert_demanded(uint64_t demand_id, int request_id, int seq_id, int token_id,
                                int layer_id, int expert_id, double demand_ts, double ready_ts,
                                double ddr5_wait_ns, double hbm_access_ns,
                                const std::string& residency_status, uint64_t transfer_id, uint64_t candidate_id,
                                int decode_step);

    void record_request_start(int request_id, int seq_id, int prompt_len, double arrival_ts, double start_ts);

    void record_request_end(int request_id, int seq_id, int gen_tokens, double end_ts);

    void record_token_start(int request_id, int seq_id, int token_id, int gen_step, double start_ts);

    void record_token_end(int request_id, int seq_id, int token_id, int gen_step, double end_ts);

    void record_exposed_stall(double stall_start_ts, double stall_duration_ns);

    // Final Processing & Artifact Generation
    void finalize(double final_device_time_ns, uint64_t baseline_demand_bytes = 0);
    const AggregatedPrefetchMetrics& get_metrics() const;
    const std::vector<CollectorDemandRecord>& get_demands() const { return demands_; }
    int get_current_decode_step() const { return current_decode_step_; }
    int lookup_token_gen_step(int request_id, int seq_id, int token_id) const;
    void dump_artifacts();
    void dump_unified_event_log(const std::string& path);
    void reset();

    // Helper for computing percentiles and quantiles
    static double compute_percentile(const std::vector<double>& sorted_data, double percentile);
    static LatencyQuantiles compute_quantiles(std::vector<double>& data);

private:
    PrefetchMetricsCollector() = default;
    ~PrefetchMetricsCollector() = default;

    PrefetchMetricsCollector(const PrefetchMetricsCollector&) = delete;
    PrefetchMetricsCollector& operator=(const PrefetchMetricsCollector&) = delete;

    bool enabled_ = true;
    std::string output_dir_ = "./results/prefetch_metrics";

    std::string dataset_name_ = "unspecified";
    int batch_size_ = 1;
    std::string predictor_type_ = "none";
    int horizon_ = 1;
    int budget_ = 4;
    std::string config_label_ = "default";

    uint64_t next_demand_id_ = 1;

    // Raw Event Storage
    std::vector<CollectorPredictionRecord> predictions_;
    std::vector<CollectorCandidateRecord> candidates_;
    std::vector<CollectorTransferRecord> transfers_;
    std::unordered_map<uint64_t, size_t> transfer_index_map_;
    std::vector<CollectorDemandRecord> demands_;
    std::map<std::pair<int, int>, CollectorRequestLatencyRecord> requests_;
    std::vector<CollectorTokenLatencyRecord> tokens_;
    std::unordered_map<std::tuple<int, int, int>, double, struct Key3TupleHash> token_start_times_;

    // Exposed execution stall tracking
    double accumulated_exposed_stall_ns_ = 0.0;
    int current_decode_step_ = -1;
    std::unordered_map<std::tuple<int, int, int>, int, struct Key3TupleHash> token_to_gen_step_;

    // Point 4: HBM Eviction & Reload tracking state
    uint64_t total_evictions_ = 0;
    uint64_t unconsumed_prefetch_evictions_ = 0;
    uint64_t demand_resident_evictions_ = 0;
    uint64_t pabi_speculative_evictions_ = 0;
    uint64_t prefetch_induced_evictions_ = 0;
    uint64_t reloaded_experts_count_ = 0;
    long long effective_capacity_bytes_ = 0;
    std::string eviction_policy_name_ = "lru";
    int fa_lru_window_size_ = 16;

    // Final Aggregates
    AggregatedPrefetchMetrics metrics_;
    bool finalized_ = false;
};

} // namespace llm_system
