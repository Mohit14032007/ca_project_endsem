#include "prefetch/live_prediction_engine.h"
#include "prefetch/prefetch_metrics_collector.h"
#include "hardware/device.h"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace llm_system {

LivePredictionEngine::LivePredictionEngine(
    const PrefetchConfig& config,
    int num_stages,
    int num_experts,
    int top_k,
    const std::string& output_dir)
    : config_(config),
      num_stages_(num_stages),
      num_experts_(num_experts),
      top_k_(top_k),
      horizon_(config.prediction_horizon),
      budget_(config.budget),
      output_dir_(output_dir) {
    if (horizon_ <= 0) {
        horizon_ = 1;
    }
    if (budget_ <= 0) {
        budget_ = 4;
    }

    std::filesystem::create_directories(output_dir_);
    csv_log_path_ = output_dir_ + "/live_predictions.csv";
    summary_json_path_ = output_dir_ + "/live_prediction_summary.json";

    csv_file_.open(csv_log_path_, std::ios::out | std::ios::trunc);
    if (csv_file_.is_open()) {
        csv_file_ << "event_id,request_id,token_id,source_stage_id,target_stage_id,horizon,"
                  << "predictor_type,budget,candidate_rank,predicted_expert_id,predicted_score,"
                  << "actual_target_experts,is_hit,prediction_latency_ns,update_latency_ns\n";
    }

    predictor_ = PredictorRegistry::create_predictor(
        config_.predictor_type, config_, num_stages_, num_experts_, top_k_);

    if (config_.queue_enabled) {
        std::string q_dir = output_dir_ + "/queue";
        const char* q_env = std::getenv("PREFETCH_QUEUE_OUTPUT_DIR");
        if (q_env != nullptr && std::string(q_env) != "") {
            q_dir = q_env;
        }
        candidate_queue_ = std::make_shared<PrefetchCandidateQueue>(
            config_, num_stages_, num_experts_, q_dir);
    }

    std::cout << "[LIVE_PREDICTION_ENGINE] Initialized with predictor: " 
              << predictor_->get_name() << " (Budget=" << budget_ 
              << ", Horizon=" << horizon_ << ", Stages=" << num_stages_ 
              << ", Experts=" << num_experts_ << ")\n";
    std::cout << "[LIVE_PREDICTION_ENGINE] Logging events to: " << csv_log_path_ << "\n";
}

LivePredictionEngine::~LivePredictionEngine() {
    if (!finalized_) {
        finalize(0.0);
    }
}

void LivePredictionEngine::on_request_start(int request_id) {
    if (predictor_) {
        predictor_->reset(request_id);
    }
    if (candidate_queue_) {
        candidate_queue_->purge_request(request_id);
    }
    seen_requests_.insert(request_id);
}

void LivePredictionEngine::on_stage_routing_start(
    int request_id,
    int token_id,
    int target_stage_id,
    int seq_id) {
    if (!predictor_) {
        return;
    }

    // Reset request on cold-start (token 0, stage 0) or if first seen
    if (token_id == 0 && target_stage_id == 0) {
        on_request_start(request_id);
    } else {
        if (seen_requests_.find(request_id) == seen_requests_.end()) {
            on_request_start(request_id);
        }
    }

    // In forward_lookahead mode, predictions are produced in on_stage_routing_complete
    // after source stage routing is known.
    if (config_.scheduling_mode == SchedulingMode::FORWARD_LOOKAHEAD) {
        return;
    }

    int source_stage_id = -1;
    std::vector<int> source_experts;
    if (target_stage_id >= horizon_) {
        source_stage_id = target_stage_id - horizon_;
        uint64_t src_key = make_key(request_id, token_id, source_stage_id);
        auto it = stage_actual_experts_.find(src_key);
        if (it != stage_actual_experts_.end()) {
            source_experts = it->second;
        }
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    PredictionResult result = predictor_->predict(
        request_id, token_id, source_stage_id, target_stage_id, source_experts, budget_);
    auto t_end = std::chrono::high_resolution_clock::now();

    double latency_ns = std::chrono::duration<double, std::nano>(t_end - t_start).count();

    PendingStagePrediction pending;
    pending.event_id = next_event_id_;
    next_event_id_ = next_event_id_ + 1;
    pending.request_id = request_id;
    pending.token_id = token_id;
    pending.source_stage_id = source_stage_id;
    pending.target_stage_id = target_stage_id;
    pending.source_experts = source_experts;
    pending.result = result;
    pending.t_predict_start = t_start;
    pending.prediction_latency_ns = latency_ns;

    uint64_t pred_key = make_key(request_id, token_id, target_stage_id);
    pending_predictions_[pred_key] = pending;

    if (candidate_queue_) {
        candidate_queue_->enqueue_candidates(
            request_id, token_id, source_stage_id, target_stage_id, horizon_, result);
    }
}

void LivePredictionEngine::on_stage_routing_complete(
    int request_id,
    int token_id,
    int target_stage_id,
    const std::vector<int>& actual_target_experts,
    int seq_id,
    double sim_time_ns) {
    if (!predictor_) {
        return;
    }

    uint64_t actual_key = make_key(request_id, token_id, target_stage_id);
    stage_actual_experts_[actual_key] = actual_target_experts;

    if (candidate_queue_) {
        candidate_queue_->reconcile_stage(
            request_id, token_id, target_stage_id, actual_target_experts);
    }

    auto it = pending_predictions_.find(actual_key);
    if (it != pending_predictions_.end()) {
        PendingStagePrediction pending = it->second;
        pending_predictions_.erase(it);

        auto t_up_start = std::chrono::high_resolution_clock::now();
        predictor_->update(
            request_id,
            token_id,
            pending.source_stage_id,
            target_stage_id,
            pending.source_experts,
            actual_target_experts);
        auto t_up_end = std::chrono::high_resolution_clock::now();

        double upd_latency_ns = std::chrono::duration<double, std::nano>(t_up_end - t_up_start).count();
        update_events_count_ = update_events_count_ + 1;

        LiveEventMetrics m = compute_metrics(
            pending.event_id,
            request_id,
            token_id,
            pending.source_stage_id,
            target_stage_id,
            pending.source_experts,
            actual_target_experts,
            pending.result,
            pending.prediction_latency_ns,
            upd_latency_ns);

        acc_overall_.add(m);
        if (token_id == 0) {
            acc_cold_.add(m);
        } else {
            if (token_id < 4) {
                acc_warm_.add(m);
            } else {
                acc_steady_.add(m);
            }
        }

        if (csv_file_.is_open()) {
            std::ostringstream act_ss;
            for (size_t i = 0; i < actual_target_experts.size(); ++i) {
                act_ss << actual_target_experts[i];
                if (i + 1 < actual_target_experts.size()) {
                    act_ss << ";";
                }
            }
            std::string act_str = act_ss.str();

            std::unordered_set<int> act_set(
                actual_target_experts.begin(), actual_target_experts.end());

            const auto& cands = pending.result.candidate_experts;
            const auto& scores = pending.result.scores;

            if (cands.empty()) {
                csv_file_ << pending.event_id << ","
                          << request_id << ","
                          << token_id << ","
                          << pending.source_stage_id << ","
                          << target_stage_id << ","
                          << horizon_ << ","
                          << predictor_->get_name() << ","
                          << budget_ << ","
                          << 0 << ","
                          << -1 << ","
                          << 0.0 << ","
                          << "\"" << act_str << "\","
                          << 0 << ","
                          << std::fixed << std::setprecision(2) << pending.prediction_latency_ns << ","
                          << upd_latency_ns << "\n";
            } else {
                for (size_t rank = 0; rank < cands.size(); ++rank) {
                    int cand = cands[rank];
                    double sc = 0.0;
                    if (rank < scores.size()) {
                        sc = scores[rank].score;
                    }
                    int hit_flag = 0;
                    if (act_set.find(cand) != act_set.end()) {
                        hit_flag = 1;
                    }

                    csv_file_ << pending.event_id << ","
                              << request_id << ","
                              << token_id << ","
                              << pending.source_stage_id << ","
                              << target_stage_id << ","
                              << horizon_ << ","
                              << predictor_->get_name() << ","
                              << budget_ << ","
                              << (rank + 1) << ","
                              << cand << ","
                              << std::fixed << std::setprecision(8) << sc << ","
                              << "\"" << act_str << "\","
                              << hit_flag << ","
                              << std::fixed << std::setprecision(2) << pending.prediction_latency_ns << ","
                              << upd_latency_ns << "\n";
                }
            }
        }
    } else {
        if (config_.scheduling_mode == SchedulingMode::LEGACY_CURRENT_STAGE) {
            invalid_events_count_ = invalid_events_count_ + 1;
        }
    }

    // In forward_lookahead mode: trigger forward prediction for future stage
    // source_stage_id = target_stage_id (the stage that just finished routing)
    // future_target_stage_id = target_stage_id + horizon_
    if (config_.scheduling_mode == SchedulingMode::FORWARD_LOOKAHEAD) {
        int future_target_stage_id = target_stage_id + horizon_;
        if (future_target_stage_id < num_stages_) {
            int source_stage_id = target_stage_id;
            const auto& source_experts = actual_target_experts;

            auto t_start = std::chrono::high_resolution_clock::now();
            PredictionResult result = predictor_->predict(
                request_id, token_id, source_stage_id, future_target_stage_id, source_experts, budget_);
            auto t_end = std::chrono::high_resolution_clock::now();

            double latency_ns = std::chrono::duration<double, std::nano>(t_end - t_start).count();

            PendingStagePrediction forward_pending;
            forward_pending.event_id = next_event_id_;
            next_event_id_ = next_event_id_ + 1;
            forward_pending.request_id = request_id;
            forward_pending.seq_id = seq_id;
            forward_pending.token_id = token_id;
            forward_pending.source_stage_id = source_stage_id;
            forward_pending.target_stage_id = future_target_stage_id;
            forward_pending.source_experts = source_experts;
            forward_pending.result = result;
            forward_pending.t_predict_start = t_start;
            forward_pending.prediction_latency_ns = latency_ns;

            uint64_t pred_key = make_key(request_id, token_id, future_target_stage_id);
            pending_predictions_[pred_key] = forward_pending;

            if (candidate_queue_) {
                candidate_queue_->enqueue_candidates(
                    request_id, token_id, source_stage_id, future_target_stage_id, horizon_, result, seq_id, sim_time_ns);
            }

            for (size_t rank = 0; rank < result.candidate_experts.size(); ++rank) {
                int cand_exp = result.candidate_experts[rank];
                double sc = 0.0;
                if (rank < result.scores.size()) {
                    sc = result.scores[rank].score;
                }
                PrefetchMetricsCollector::getInstance().record_prediction(
                    request_id, seq_id, token_id, source_stage_id, future_target_stage_id,
                    horizon_, predictor_->get_name(), budget_, static_cast<int>(rank + 1),
                    cand_exp, sc, actual_target_experts, 0, latency_ns, 0.0, sim_time_ns);
            }
        }
    }
}

LiveEventMetrics LivePredictionEngine::compute_metrics(
    int event_id,
    int request_id,
    int token_id,
    int source_stage,
    int target_stage,
    const std::vector<int>& source_experts,
    const std::vector<int>& actual_target_experts,
    const PredictionResult& pred_result,
    double pred_latency_ns,
    double upd_latency_ns) {
    LiveEventMetrics m;
    m.event_id = event_id;
    m.request_id = request_id;
    m.token_id = token_id;
    m.source_stage_id = source_stage;
    m.target_stage_id = target_stage;
    m.horizon = horizon_;
    m.budget = budget_;
    m.predictor_type = predictor_->get_name();
    m.source_experts = source_experts;
    m.actual_target_experts = actual_target_experts;
    m.predicted_candidates = pred_result.candidate_experts;
    m.prediction_latency_ns = pred_latency_ns;
    m.update_latency_ns = upd_latency_ns;

    for (const auto& cs : pred_result.scores) {
        m.candidate_scores.push_back(cs.score);
    }

    std::unordered_set<int> act_set(
        actual_target_experts.begin(), actual_target_experts.end());
    m.hits = 0;
    int first_hit_rank = -1;

    for (size_t i = 0; i < m.predicted_candidates.size(); ++i) {
        int cand = m.predicted_candidates[i];
        if (act_set.find(cand) != act_set.end()) {
            m.hits = m.hits + 1;
            if (first_hit_rank < 0) {
                first_hit_rank = static_cast<int>(i + 1);
            }
        }
    }

    if (!m.predicted_candidates.empty()) {
        m.precision = static_cast<double>(m.hits) / static_cast<double>(m.predicted_candidates.size());
    } else {
        m.precision = 0.0;
    }

    if (!actual_target_experts.empty()) {
        m.recall = static_cast<double>(m.hits) / static_cast<double>(actual_target_experts.size());
    } else {
        m.recall = 0.0;
    }

    if ((m.precision + m.recall) > 0.0) {
        m.f1 = (2.0 * m.precision * m.recall) / (m.precision + m.recall);
    } else {
        m.f1 = 0.0;
    }

    if (m.hits > 0) {
        m.hit_rate = 1.0;
    } else {
        m.hit_rate = 0.0;
    }

    if (first_hit_rank > 0) {
        m.mrr = 1.0 / static_cast<double>(first_hit_rank);
    } else {
        m.mrr = 0.0;
    }

    // nDCG calculation
    double dcg = 0.0;
    for (size_t i = 0; i < m.predicted_candidates.size(); ++i) {
        int cand = m.predicted_candidates[i];
        if (act_set.find(cand) != act_set.end()) {
            double rank_discount = std::log2(static_cast<double>(i + 2));
            dcg = dcg + (1.0 / rank_discount);
        }
    }

    size_t ideal_hits = m.predicted_candidates.size();
    if (actual_target_experts.size() < ideal_hits) {
        ideal_hits = actual_target_experts.size();
    }

    double idcg = 0.0;
    for (size_t i = 0; i < ideal_hits; ++i) {
        double rank_discount = std::log2(static_cast<double>(i + 2));
        idcg = idcg + (1.0 / rank_discount);
    }

    if (idcg > 0.0) {
        m.ndcg = dcg / idcg;
    } else {
        m.ndcg = 0.0;
    }

    return m;
}

void LivePredictionEngine::finalize(double total_host_execution_seconds) {
    if (finalized_) {
        return;
    }
    finalized_ = true;

    if (candidate_queue_) {
        candidate_queue_->finalize();
    }

    if (csv_file_.is_open()) {
        csv_file_.flush();
        csv_file_.close();
    }

    double total_pred_time_sec = (acc_overall_.sum_pred_latency_ns + acc_overall_.sum_upd_latency_ns) / 1e9;
    double overhead_pct = 0.0;
    if (total_host_execution_seconds > 0.0) {
        overhead_pct = (total_pred_time_sec / total_host_execution_seconds) * 100.0;
    }

    // Write summary JSON
    std::ofstream json_out(summary_json_path_);
    if (json_out.is_open()) {
        json_out << "{\n"
                 << "  \"predictor\": \"" << predictor_->get_name() << "\",\n"
                 << "  \"budget\": " << budget_ << ",\n"
                 << "  \"horizon\": " << horizon_ << ",\n"
                 << "  \"total_prediction_events\": " << next_event_id_ << ",\n"
                 << "  \"total_update_events\": " << update_events_count_ << ",\n"
                 << "  \"invalid_events\": " << invalid_events_count_ << ",\n"
                 << "  \"total_host_seconds\": " << total_host_execution_seconds << ",\n"
                 << "  \"prediction_overhead_percent\": " << overhead_pct << ",\n"
                 << "  \"overall\": {\n"
                 << "    \"events\": " << acc_overall_.count << ",\n"
                 << "    \"precision\": " << acc_overall_.mean_precision() << ",\n"
                 << "    \"recall\": " << acc_overall_.mean_recall() << ",\n"
                 << "    \"f1\": " << acc_overall_.mean_f1() << ",\n"
                 << "    \"hit_rate\": " << acc_overall_.mean_hit_rate() << ",\n"
                 << "    \"mrr\": " << acc_overall_.mean_mrr() << ",\n"
                 << "    \"ndcg\": " << acc_overall_.mean_ndcg() << ",\n"
                 << "    \"mean_pred_latency_ns\": " << acc_overall_.mean_pred_latency_ns() << ",\n"
                 << "    \"mean_upd_latency_ns\": " << acc_overall_.mean_upd_latency_ns() << "\n"
                 << "  },\n"
                 << "  \"cold_start\": {\n"
                 << "    \"events\": " << acc_cold_.count << ",\n"
                 << "    \"precision\": " << acc_cold_.mean_precision() << ",\n"
                 << "    \"recall\": " << acc_cold_.mean_recall() << ",\n"
                 << "    \"f1\": " << acc_cold_.mean_f1() << ",\n"
                 << "    \"hit_rate\": " << acc_cold_.mean_hit_rate() << ",\n"
                 << "    \"mrr\": " << acc_cold_.mean_mrr() << ",\n"
                 << "    \"ndcg\": " << acc_cold_.mean_ndcg() << "\n"
                 << "  },\n"
                 << "  \"steady_state\": {\n"
                 << "    \"events\": " << acc_steady_.count << ",\n"
                 << "    \"precision\": " << acc_steady_.mean_precision() << ",\n"
                 << "    \"recall\": " << acc_steady_.mean_recall() << ",\n"
                 << "    \"f1\": " << acc_steady_.mean_f1() << ",\n"
                 << "    \"hit_rate\": " << acc_steady_.mean_hit_rate() << ",\n"
                 << "    \"mrr\": " << acc_steady_.mean_mrr() << ",\n"
                 << "    \"ndcg\": " << acc_steady_.mean_ndcg() << "\n"
                 << "  }\n"
                 << "}\n";
        json_out.close();
    }

    std::cout << "\n======================================================================\n";
    std::cout << "LIVE PREDICTION SUMMARY: " << predictor_->get_name() 
              << " (Budget=" << budget_ << ", Horizon=" << horizon_ << ")\n";
    std::cout << "======================================================================\n";
    std::cout << "Total Predictions: " << next_event_id_ 
              << " | Total Updates: " << update_events_count_
              << " | Invalid/Skipped: " << invalid_events_count_ << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Metric         | Overall   | Cold (T=0)| Warm (1-3)| Steady (>=4)\n";
    std::cout << "---------------+-----------+-----------+-----------+-------------\n";
    std::cout << "Events         | " << std::setw(9) << acc_overall_.count << " | "
              << std::setw(9) << acc_cold_.count << " | "
              << std::setw(9) << acc_warm_.count << " | "
              << std::setw(11) << acc_steady_.count << "\n";
    std::cout << "Precision@B    | " << std::setw(9) << acc_overall_.mean_precision() << " | "
              << std::setw(9) << acc_cold_.mean_precision() << " | "
              << std::setw(9) << acc_warm_.mean_precision() << " | "
              << std::setw(11) << acc_steady_.mean_precision() << "\n";
    std::cout << "Recall@B       | " << std::setw(9) << acc_overall_.mean_recall() << " | "
              << std::setw(9) << acc_cold_.mean_recall() << " | "
              << std::setw(9) << acc_warm_.mean_recall() << " | "
              << std::setw(11) << acc_steady_.mean_recall() << "\n";
    std::cout << "F1@B           | " << std::setw(9) << acc_overall_.mean_f1() << " | "
              << std::setw(9) << acc_cold_.mean_f1() << " | "
              << std::setw(9) << acc_warm_.mean_f1() << " | "
              << std::setw(11) << acc_steady_.mean_f1() << "\n";
    std::cout << "Hit Rate       | " << std::setw(9) << acc_overall_.mean_hit_rate() << " | "
              << std::setw(9) << acc_cold_.mean_hit_rate() << " | "
              << std::setw(9) << acc_warm_.mean_hit_rate() << " | "
              << std::setw(11) << acc_steady_.mean_hit_rate() << "\n";
    std::cout << "MRR            | " << std::setw(9) << acc_overall_.mean_mrr() << " | "
              << std::setw(9) << acc_cold_.mean_mrr() << " | "
              << std::setw(9) << acc_warm_.mean_mrr() << " | "
              << std::setw(11) << acc_steady_.mean_mrr() << "\n";
    std::cout << "nDCG@B         | " << std::setw(9) << acc_overall_.mean_ndcg() << " | "
              << std::setw(9) << acc_cold_.mean_ndcg() << " | "
              << std::setw(9) << acc_warm_.mean_ndcg() << " | "
              << std::setw(11) << acc_steady_.mean_ndcg() << "\n";
    std::cout << "Pred Lat (ns)  | " << std::setw(9) << acc_overall_.mean_pred_latency_ns() << " | "
              << std::setw(9) << acc_cold_.mean_pred_latency_ns() << " | "
              << std::setw(9) << acc_warm_.mean_pred_latency_ns() << " | "
              << std::setw(11) << acc_steady_.mean_pred_latency_ns() << "\n";
    std::cout << "Upd Lat (ns)   | " << std::setw(9) << acc_overall_.mean_upd_latency_ns() << " | "
              << std::setw(9) << acc_cold_.mean_upd_latency_ns() << " | "
              << std::setw(9) << acc_warm_.mean_upd_latency_ns() << " | "
              << std::setw(11) << acc_steady_.mean_upd_latency_ns() << "\n";
    std::cout << "Host Overhead  | " << std::setw(8) << overhead_pct << "%\n";
    std::cout << "======================================================================\n";
    std::cout << "[LIVE_PREDICTION_ENGINE] Summaries written to: " << summary_json_path_ << "\n";
}

int LivePredictionEngine::dispatch_candidates(std::shared_ptr<Device> device, double current_time_ns, int budget) {
    if (device == nullptr || candidate_queue_ == nullptr) {
        return 0;
    }
    int max_candidates = budget;
    if (max_candidates < 0) {
        max_candidates = budget_;
    }
    return device->dispatch_prefetch_candidates(candidate_queue_, current_time_ns, max_candidates);
}

int LivePredictionEngine::expire_pins_for_stage(std::shared_ptr<Device> device, int target_stage_id) {
    if (device == nullptr) {
        return 0;
    }
    return device->expire_pins_for_stage(target_stage_id);
}

} // namespace llm_system

