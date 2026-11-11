#include "prefetch/prefetch_metrics_collector.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

namespace llm_system {

std::string to_string(CollectorEventType event_type) {
    if (event_type == CollectorEventType::PREDICTION_GENERATED) {
        return "PREDICTION_GENERATED";
    }
    if (event_type == CollectorEventType::CANDIDATE_ADMITTED) {
        return "CANDIDATE_ADMITTED";
    }
    if (event_type == CollectorEventType::CANDIDATE_REJECTED) {
        return "CANDIDATE_REJECTED";
    }
    if (event_type == CollectorEventType::CANDIDATE_DISPATCHED) {
        return "CANDIDATE_DISPATCHED";
    }
    if (event_type == CollectorEventType::TRANSFER_STARTED) {
        return "TRANSFER_STARTED";
    }
    if (event_type == CollectorEventType::TRANSFER_PAUSED) {
        return "TRANSFER_PAUSED";
    }
    if (event_type == CollectorEventType::TRANSFER_RESUMED) {
        return "TRANSFER_RESUMED";
    }
    if (event_type == CollectorEventType::TRANSFER_COMPLETED) {
        return "TRANSFER_COMPLETED";
    }
    if (event_type == CollectorEventType::TRANSFER_CANCELLED) {
        return "TRANSFER_CANCELLED";
    }
    if (event_type == CollectorEventType::HBM_RESIDENT) {
        return "HBM_RESIDENT";
    }
    if (event_type == CollectorEventType::HBM_EVICTED) {
        return "HBM_EVICTED";
    }
    if (event_type == CollectorEventType::EXPERT_DEMANDED) {
        return "EXPERT_DEMANDED";
    }
    if (event_type == CollectorEventType::DEMAND_ATTACHED_TO_PREFETCH) {
        return "DEMAND_ATTACHED_TO_PREFETCH";
    }
    if (event_type == CollectorEventType::DEMAND_TRANSFER_STARTED) {
        return "DEMAND_TRANSFER_STARTED";
    }
    if (event_type == CollectorEventType::EXPERT_READY) {
        return "EXPERT_READY";
    }
    if (event_type == CollectorEventType::REQUEST_STARTED) {
        return "REQUEST_STARTED";
    }
    if (event_type == CollectorEventType::REQUEST_COMPLETED) {
        return "REQUEST_COMPLETED";
    }
    if (event_type == CollectorEventType::TOKEN_STARTED) {
        return "TOKEN_STARTED";
    }
    if (event_type == CollectorEventType::TOKEN_COMPLETED) {
        return "TOKEN_COMPLETED";
    }
    return "UNKNOWN";
}

PrefetchMetricsCollector& PrefetchMetricsCollector::getInstance() {
    static PrefetchMetricsCollector instance;
    return instance;
}

void PrefetchMetricsCollector::set_output_dir(const std::string& dir) {
    output_dir_ = dir;
}

std::string PrefetchMetricsCollector::get_output_dir() const {
    return output_dir_;
}

void PrefetchMetricsCollector::set_experiment_metadata(
    const std::string& dataset,
    int batch_size,
    const std::string& predictor,
    int horizon,
    int budget,
    const std::string& config_label) {
    dataset_name_ = dataset;
    batch_size_ = batch_size;
    predictor_type_ = predictor;
    horizon_ = horizon;
    budget_ = budget;
    config_label_ = config_label;
}

void PrefetchMetricsCollector::set_enabled(bool enabled) {
    enabled_ = enabled;
}

bool PrefetchMetricsCollector::is_enabled() const {
    return enabled_;
}

void PrefetchMetricsCollector::record_prediction(
    int request_id, int seq_id, int token_id,
    int source_stage, int target_stage, int horizon,
    const std::string& predictor_type, int budget,
    int rank, int predicted_expert, double confidence,
    const std::vector<int>& actual_experts, int is_hit,
    double pred_latency_ns, double upd_latency_ns,
    double timestamp_ns) {
    if (!enabled_) {
        return;
    }

    CollectorPredictionRecord rec;
    rec.event_id = predictions_.size() + 1;
    rec.request_id = request_id;
    rec.seq_id = seq_id;
    rec.token_id = token_id;
    rec.source_stage = source_stage;
    rec.target_stage = target_stage;
    rec.horizon = horizon;
    rec.predictor_type = predictor_type;
    rec.budget = budget;
    rec.candidate_rank = rank;
    rec.predicted_expert_id = predicted_expert;
    rec.confidence_score = confidence;
    rec.actual_target_experts = actual_experts;
    rec.is_hit = is_hit;
    rec.predict_latency_ns = pred_latency_ns;
    rec.update_latency_ns = upd_latency_ns;
    rec.timestamp_ns = timestamp_ns;
    predictions_.push_back(rec);
}

void PrefetchMetricsCollector::record_candidate_admitted(
    uint64_t candidate_id, int request_id, int seq_id, int token_id,
    int source_stage, int target_stage, int expert_id,
    int rank, double score, double timestamp_ns) {
    if (!enabled_) {
        return;
    }

    CollectorCandidateRecord rec;
    rec.candidate_id = candidate_id;
    rec.event_type = "ADMITTED";
    rec.request_id = request_id;
    rec.seq_id = seq_id;
    rec.token_id = token_id;
    rec.source_stage = source_stage;
    rec.target_stage = target_stage;
    rec.expert_id = expert_id;
    rec.prediction_rank = rank;
    rec.prediction_score = score;
    rec.reason = "ADMITTED";
    rec.timestamp_ns = timestamp_ns;
    candidates_.push_back(rec);
}

void PrefetchMetricsCollector::record_candidate_rejected(
    int request_id, int seq_id, int token_id,
    int source_stage, int target_stage, int expert_id,
    const std::string& reason, double timestamp_ns) {
    if (!enabled_) {
        return;
    }

    metrics_.candidates_rejected_admission = metrics_.candidates_rejected_admission + 1;

    CollectorCandidateRecord rec;
    rec.candidate_id = 0;
    rec.event_type = "REJECTED";
    rec.request_id = request_id;
    rec.seq_id = seq_id;
    rec.token_id = token_id;
    rec.source_stage = source_stage;
    rec.target_stage = target_stage;
    rec.expert_id = expert_id;
    rec.reason = reason;
    rec.timestamp_ns = timestamp_ns;
    candidates_.push_back(rec);
}

void PrefetchMetricsCollector::record_candidate_dispatched(
    uint64_t candidate_id, int request_id, int seq_id, int token_id,
    int target_stage, int expert_id, double timestamp_ns) {
    if (!enabled_) {
        return;
    }

    CollectorCandidateRecord rec;
    rec.candidate_id = candidate_id;
    rec.event_type = "DISPATCHED";
    rec.request_id = request_id;
    rec.seq_id = seq_id;
    rec.token_id = token_id;
    rec.target_stage = target_stage;
    rec.expert_id = expert_id;
    rec.reason = "DISPATCHED_TO_DDR5";
    rec.timestamp_ns = timestamp_ns;
    candidates_.push_back(rec);
}

void PrefetchMetricsCollector::record_transfer_submitted(
    uint64_t transfer_id, uint64_t candidate_id,
    int request_id, int seq_id, int token_id,
    int target_stage, int expert_id,
    const std::string& transfer_class, int priority,
    uint64_t bytes, double queue_submit_ts) {
    if (!enabled_) {
        return;
    }

    CollectorTransferRecord rec;
    rec.transfer_id = transfer_id;
    rec.candidate_id = candidate_id;
    rec.request_id = request_id;
    rec.seq_id = seq_id;
    rec.token_id = token_id;
    rec.target_stage = target_stage;
    rec.expert_id = expert_id;
    rec.transfer_class = transfer_class;
    rec.priority = priority;
    rec.size_bytes = bytes;
    rec.queue_submit_ts = queue_submit_ts;
    rec.service_start_ts = -1.0;
    rec.completion_ts = -1.0;
    rec.duration_ns = 0.0;
    rec.paused_duration_ns = 0.0;
    rec.outcome = "UNDECIDED";

    transfer_index_map_[transfer_id] = transfers_.size();
    transfers_.push_back(rec);
}

void PrefetchMetricsCollector::record_transfer_started(
    uint64_t transfer_id, double service_start_ts) {
    if (!enabled_) {
        return;
    }

    auto it = transfer_index_map_.find(transfer_id);
    if (it != transfer_index_map_.end()) {
        size_t idx = it->second;
        if (transfers_[idx].service_start_ts < 0.0) {
            transfers_[idx].service_start_ts = service_start_ts;
        }
    }
}

void PrefetchMetricsCollector::record_transfer_paused(
    uint64_t transfer_id, double paused_ts, double remaining_bytes) {
    if (!enabled_) {
        return;
    }

    (void)paused_ts;
    (void)remaining_bytes;
}

void PrefetchMetricsCollector::record_transfer_resumed(
    uint64_t transfer_id, double resumed_ts) {
    if (!enabled_) {
        return;
    }

    (void)transfer_id;
    (void)resumed_ts;
}

void PrefetchMetricsCollector::record_transfer_completed(
    uint64_t transfer_id, double completion_ts) {
    if (!enabled_) {
        return;
    }

    auto it = transfer_index_map_.find(transfer_id);
    if (it != transfer_index_map_.end()) {
        size_t idx = it->second;
        transfers_[idx].completion_ts = completion_ts;
        if (transfers_[idx].service_start_ts >= 0.0) {
            transfers_[idx].duration_ns = completion_ts - transfers_[idx].service_start_ts;
        } else {
            transfers_[idx].duration_ns = completion_ts - transfers_[idx].queue_submit_ts;
        }
        if (transfers_[idx].outcome == "UNDECIDED") {
            transfers_[idx].outcome = "COMPLETED_UNUSED";
        }
    }
}

void PrefetchMetricsCollector::record_transfer_cancelled(
    uint64_t transfer_id, double cancelled_ts, uint64_t transferred_bytes) {
    if (!enabled_) {
        return;
    }

    auto it = transfer_index_map_.find(transfer_id);
    if (it != transfer_index_map_.end()) {
        size_t idx = it->second;
        transfers_[idx].outcome = "CANCELLED";
        transfers_[idx].completion_ts = cancelled_ts;
        metrics_.cancelled_partial_bytes = metrics_.cancelled_partial_bytes + transferred_bytes;
    }
}

void PrefetchMetricsCollector::record_transfer_outcome(
    uint64_t transfer_id, const std::string& outcome) {
    if (!enabled_) {
        return;
    }

    auto it = transfer_index_map_.find(transfer_id);
    if (it != transfer_index_map_.end()) {
        size_t idx = it->second;
        transfers_[idx].outcome = outcome;
    }
}

void PrefetchMetricsCollector::record_hbm_resident(
    int layer_id, int expert_id, uint64_t transfer_id,
    const std::string& classification, double timestamp_ns) {
    if (!enabled_) {
        return;
    }

    (void)layer_id;
    (void)expert_id;
    (void)transfer_id;
    (void)classification;
    (void)timestamp_ns;
}

void PrefetchMetricsCollector::record_hbm_evicted(
    int layer_id, int expert_id, const std::string& reason, double timestamp_ns) {
    if (!enabled_) {
        return;
    }

    (void)layer_id;
    (void)expert_id;
    (void)timestamp_ns;

    total_evictions_ = total_evictions_ + 1;
    if (reason == "UNCONSUMED_PREFETCH") {
        unconsumed_prefetch_evictions_ = unconsumed_prefetch_evictions_ + 1;
    } else if (reason == "DEMAND_RESIDENT") {
        demand_resident_evictions_ = demand_resident_evictions_ + 1;
    } else if (reason == "PABI_SPECULATIVE") {
        pabi_speculative_evictions_ = pabi_speculative_evictions_ + 1;
    } else {
        demand_resident_evictions_ = demand_resident_evictions_ + 1;
    }
}

void PrefetchMetricsCollector::record_prefetch_induced_eviction(
    int layer_id, int expert_id, double timestamp_ns) {
    if (!enabled_) {
        return;
    }

    (void)layer_id;
    (void)expert_id;
    (void)timestamp_ns;

    prefetch_induced_evictions_ = prefetch_induced_evictions_ + 1;
}

void PrefetchMetricsCollector::set_hbm_metrics(
    long long capacity_bytes, const std::string& policy_name, int window_size, uint64_t reloads) {
    effective_capacity_bytes_ = capacity_bytes;
    eviction_policy_name_ = policy_name;
    fa_lru_window_size_ = window_size;
    reloaded_experts_count_ = reloads;
}

void PrefetchMetricsCollector::record_expert_demanded(
    uint64_t demand_id, int request_id, int seq_id, int token_id,
    int layer_id, int expert_id, double demand_ts, double ready_ts,
    double ddr5_wait_ns, double hbm_access_ns,
    const std::string& residency_status, uint64_t transfer_id, uint64_t candidate_id) {
    record_expert_demanded(demand_id, request_id, seq_id, token_id, layer_id, expert_id,
                           demand_ts, ready_ts, ddr5_wait_ns, hbm_access_ns,
                           residency_status, transfer_id, candidate_id, -1);
}

void PrefetchMetricsCollector::record_expert_demanded(
    uint64_t demand_id, int request_id, int seq_id, int token_id,
    int layer_id, int expert_id, double demand_ts, double ready_ts,
    double ddr5_wait_ns, double hbm_access_ns,
    const std::string& residency_status, uint64_t transfer_id, uint64_t candidate_id,
    int decode_step) {
    if (!enabled_) {
        return;
    }

    uint64_t d_id = demand_id;
    if (d_id == 0) {
        d_id = next_demand_id_;
        next_demand_id_ = next_demand_id_ + 1;
    }

    int d_step = decode_step;
    if (d_step < 0) {
        auto t_key = std::make_tuple(request_id, seq_id, token_id);
        auto t_it = token_to_gen_step_.find(t_key);
        if (t_it != token_to_gen_step_.end()) {
            d_step = t_it->second;
        } else if (current_decode_step_ >= 0) {
            d_step = current_decode_step_;
        } else {
            d_step = -1;
        }
    }

    CollectorDemandRecord rec;
    rec.demand_id = d_id;
    rec.request_id = request_id;
    rec.seq_id = seq_id;
    rec.token_id = token_id;
    rec.decode_step = d_step;
    rec.layer_id = layer_id;
    rec.expert_id = expert_id;
    rec.demand_ts = demand_ts;
    rec.ready_ts = ready_ts;
    rec.ddr5_wait_ns = ddr5_wait_ns;
    rec.hbm_access_ns = hbm_access_ns;
    rec.residency_status = residency_status;
    rec.transfer_id = transfer_id;
    rec.candidate_id = candidate_id;
    demands_.push_back(rec);
}

void PrefetchMetricsCollector::record_request_start(
    int request_id, int seq_id, int prompt_len, double arrival_ts, double start_ts) {
    if (!enabled_) {
        return;
    }

    std::pair<int, int> key = {request_id, seq_id};
    CollectorRequestLatencyRecord rec;
    rec.request_id = request_id;
    rec.seq_id = seq_id;
    rec.prompt_len = prompt_len;
    rec.gen_tokens = 0;
    rec.arrival_ts = arrival_ts;
    rec.start_ts = start_ts;
    rec.end_ts = 0.0;
    rec.latency_ns = 0.0;
    requests_[key] = rec;
}

void PrefetchMetricsCollector::record_request_end(
    int request_id, int seq_id, int gen_tokens, double end_ts) {
    if (!enabled_) {
        return;
    }

    std::pair<int, int> key = {request_id, seq_id};
    auto it = requests_.find(key);
    if (it != requests_.end()) {
        it->second.gen_tokens = gen_tokens;
        it->second.end_ts = end_ts;
        it->second.latency_ns = end_ts - it->second.start_ts;
    }
}

int PrefetchMetricsCollector::lookup_token_gen_step(int request_id, int seq_id, int token_id) const {
    auto key = std::make_tuple(request_id, seq_id, token_id);
    auto it = token_to_gen_step_.find(key);
    if (it != token_to_gen_step_.end()) {
        return it->second;
    }
    return -1;
}

void PrefetchMetricsCollector::record_token_start(
    int request_id, int seq_id, int token_id, int gen_step, double start_ts) {
    if (!enabled_) {
        return;
    }

    current_decode_step_ = gen_step;
    auto key = std::make_tuple(request_id, seq_id, token_id);
    token_start_times_[key] = start_ts;
    token_to_gen_step_[key] = gen_step;
}

void PrefetchMetricsCollector::record_token_end(
    int request_id, int seq_id, int token_id, int gen_step, double end_ts) {
    if (!enabled_) {
        return;
    }

    auto key = std::make_tuple(request_id, seq_id, token_id);
    double start_ts = end_ts;
    auto it = token_start_times_.find(key);
    if (it != token_start_times_.end()) {
        start_ts = it->second;
    }

    CollectorTokenLatencyRecord rec;
    rec.request_id = request_id;
    rec.seq_id = seq_id;
    rec.token_id = token_id;
    rec.gen_step = gen_step;
    rec.token_start_ts = start_ts;
    rec.token_end_ts = end_ts;
    rec.token_duration_ns = end_ts - start_ts;
    rec.cumulative_latency_ns = 0.0;
    tokens_.push_back(rec);
}

void PrefetchMetricsCollector::record_exposed_stall(
    double stall_start_ts, double stall_duration_ns) {
    if (!enabled_) {
        return;
    }

    (void)stall_start_ts;
    accumulated_exposed_stall_ns_ = accumulated_exposed_stall_ns_ + stall_duration_ns;
}

double PrefetchMetricsCollector::compute_percentile(
    const std::vector<double>& sorted_data, double percentile) {
    if (sorted_data.empty()) {
        return 0.0;
    }
    if (percentile <= 0.0) {
        return sorted_data.front();
    }
    if (percentile >= 1.0) {
        return sorted_data.back();
    }

    double rank = percentile * static_cast<double>(sorted_data.size() - 1);
    size_t lower_idx = static_cast<size_t>(std::floor(rank));
    size_t upper_idx = static_cast<size_t>(std::ceil(rank));

    if (lower_idx == upper_idx) {
        return sorted_data[lower_idx];
    }

    double frac = rank - static_cast<double>(lower_idx);
    return sorted_data[lower_idx] + frac * (sorted_data[upper_idx] - sorted_data[lower_idx]);
}

LatencyQuantiles PrefetchMetricsCollector::compute_quantiles(std::vector<double>& data) {
    LatencyQuantiles q;
    if (data.empty()) {
        return q;
    }
    std::sort(data.begin(), data.end());
    q.count = data.size();
    double sum = 0.0;
    for (double v : data) {
        sum = sum + v;
    }
    q.mean_ns = sum / static_cast<double>(data.size());
    q.p50_ns = compute_percentile(data, 0.50);
    q.p90_ns = compute_percentile(data, 0.90);
    q.p95_ns = compute_percentile(data, 0.95);
    q.p99_ns = compute_percentile(data, 0.99);
    q.max_ns = data.back();
    return q;
}

void PrefetchMetricsCollector::finalize(
    double final_device_time_ns, uint64_t baseline_demand_bytes) {
    if (finalized_) {
        return;
    }

    metrics_.total_actual_demands = demands_.size();
    metrics_.total_simulated_execution_ns = final_device_time_ns;
    metrics_.exposed_gpu_stall_ns = accumulated_exposed_stall_ns_;
    metrics_.baseline_demand_bytes = baseline_demand_bytes;

    std::vector<double> expert_load_latencies;
    expert_load_latencies.reserve(demands_.size());
    std::vector<double> res_hit_latencies;
    std::vector<double> timely_pref_latencies;
    std::vector<double> late_att_latencies;
    std::vector<double> demand_miss_latencies;

    // 1. Process Projection-Level Expert Demands
    for (const auto& dem : demands_) {
        double lat = dem.ready_ts - dem.demand_ts;
        if (lat < 0.0) {
            lat = 0.0;
        }
        expert_load_latencies.push_back(lat);

        metrics_.aggregate_demand_wait_ns = metrics_.aggregate_demand_wait_ns + dem.ddr5_wait_ns;

        if (dem.residency_status == "TIMELY_PREFETCH") {
            metrics_.timely_resident_hits = metrics_.timely_resident_hits + 1;
            metrics_.total_hbm_resident_hits = metrics_.total_hbm_resident_hits + 1;
            timely_pref_latencies.push_back(lat);
        } else if (dem.residency_status == "LATE_IN_FLIGHT") {
            metrics_.late_inflight_attachments = metrics_.late_inflight_attachments + 1;
            late_att_latencies.push_back(lat);
        } else if (dem.residency_status == "RESIDENT_HIT") {
            metrics_.total_hbm_resident_hits = metrics_.total_hbm_resident_hits + 1;
            res_hit_latencies.push_back(lat);
        } else if (dem.residency_status == "DEMAND_MISS") {
            metrics_.fresh_demand_misses = metrics_.fresh_demand_misses + 1;
            demand_miss_latencies.push_back(lat);
        }

        if (dem.candidate_id > 0) {
            metrics_.demands_with_prediction = metrics_.demands_with_prediction + 1;
        } else {
            metrics_.demands_without_prediction = metrics_.demands_without_prediction + 1;
        }
    }

    // 2. Process Unique Expert Demands (deduplicated across projections W1, W3, W2)
    std::map<std::tuple<int, int, int, int, int>, CollectorDemandRecord> unique_demands_map;
    for (const auto& dem : demands_) {
        auto key = std::make_tuple(dem.request_id, dem.seq_id, dem.token_id, dem.layer_id, dem.expert_id);
        if (unique_demands_map.find(key) == unique_demands_map.end()) {
            unique_demands_map[key] = dem;
        }
    }

    metrics_.unique_actual_demands = unique_demands_map.size();
    std::vector<double> u_overall_lats;
    std::vector<double> u_res_lats;
    std::vector<double> u_timely_lats;
    std::vector<double> u_late_lats;
    std::vector<double> u_miss_lats;

    for (const auto& kv : unique_demands_map) {
        const auto& dem = kv.second;
        double lat = dem.ready_ts - dem.demand_ts;
        if (lat < 0.0) {
            lat = 0.0;
        }
        u_overall_lats.push_back(lat);

        if (dem.residency_status == "TIMELY_PREFETCH") {
            metrics_.unique_timely_resident_hits = metrics_.unique_timely_resident_hits + 1;
            u_timely_lats.push_back(lat);
        } else if (dem.residency_status == "LATE_IN_FLIGHT") {
            metrics_.unique_late_inflight_attachments = metrics_.unique_late_inflight_attachments + 1;
            u_late_lats.push_back(lat);
        } else if (dem.residency_status == "RESIDENT_HIT") {
            metrics_.unique_resident_hits = metrics_.unique_resident_hits + 1;
            u_res_lats.push_back(lat);
        } else if (dem.residency_status == "DEMAND_MISS") {
            metrics_.unique_fresh_demand_misses = metrics_.unique_fresh_demand_misses + 1;
            u_miss_lats.push_back(lat);
        }

        if (dem.candidate_id > 0) {
            metrics_.unique_demands_with_prediction = metrics_.unique_demands_with_prediction + 1;
        } else {
            metrics_.unique_demands_without_prediction = metrics_.unique_demands_without_prediction + 1;
        }
    }

    metrics_.unique_expert_latency_overall = compute_quantiles(u_overall_lats);
    metrics_.unique_expert_latency_resident_hit = compute_quantiles(u_res_lats);
    metrics_.unique_expert_latency_timely_prefetch = compute_quantiles(u_timely_lats);
    metrics_.unique_expert_latency_late_attachment = compute_quantiles(u_late_lats);
    metrics_.unique_expert_latency_demand_miss = compute_quantiles(u_miss_lats);

    metrics_.unique_prefetch_assisted_demands =
        metrics_.unique_timely_resident_hits + metrics_.unique_late_inflight_attachments;

    if (metrics_.unique_actual_demands > 0) {
        metrics_.unique_overall_coverage =
            static_cast<double>(metrics_.unique_prefetch_assisted_demands) / static_cast<double>(metrics_.unique_actual_demands);
        metrics_.unique_overall_hbm_hit_rate =
            static_cast<double>(metrics_.unique_resident_hits + metrics_.unique_timely_resident_hits) / static_cast<double>(metrics_.unique_actual_demands);
        metrics_.unique_prefetch_assisted_hit_rate =
            static_cast<double>(metrics_.unique_timely_resident_hits) / static_cast<double>(metrics_.unique_actual_demands);
        metrics_.unique_fresh_demand_miss_rate =
            static_cast<double>(metrics_.unique_fresh_demand_misses) / static_cast<double>(metrics_.unique_actual_demands);
        metrics_.unique_inflight_attachment_rate =
            static_cast<double>(metrics_.unique_late_inflight_attachments) / static_cast<double>(metrics_.unique_actual_demands);
        metrics_.unique_timely_rate_of_demands =
            static_cast<double>(metrics_.unique_timely_resident_hits) / static_cast<double>(metrics_.unique_actual_demands);
        metrics_.unique_late_rate_of_demands =
            static_cast<double>(metrics_.unique_late_inflight_attachments) / static_cast<double>(metrics_.unique_actual_demands);
        metrics_.wait_per_unique_demand_ns =
            metrics_.aggregate_demand_wait_ns / static_cast<double>(metrics_.unique_actual_demands);
    }

    if (metrics_.unique_demands_with_prediction > 0) {
        metrics_.unique_predicted_coverage =
            static_cast<double>(metrics_.unique_prefetch_assisted_demands) / static_cast<double>(metrics_.unique_demands_with_prediction);
    }

    // 2b. Process Stage-Coalesced Demands (Physical Demands across Serving Batch)
    std::map<std::tuple<int, int, int>, CollectorDemandRecord> stage_coalesced_demands;
    uint64_t invalid_decode_step_count = 0;
    int invalid_seq_counter = -1;
    for (const auto& dem : demands_) {
        int step = dem.decode_step;
        if (step < 0) {
            invalid_decode_step_count = invalid_decode_step_count + 1;
            step = invalid_seq_counter;
            invalid_seq_counter = invalid_seq_counter - 1;
        }
        auto coalesced_key = std::make_tuple(step, dem.layer_id, dem.expert_id);
        if (stage_coalesced_demands.find(coalesced_key) == stage_coalesced_demands.end()) {
            stage_coalesced_demands[coalesced_key] = dem;
        }
    }

    metrics_.coalesced_actual_demands = stage_coalesced_demands.size();
    metrics_.invalid_decode_step_demands = invalid_decode_step_count;
    metrics_.routed_actual_demands = metrics_.unique_actual_demands;
    metrics_.tensor_actual_demands = metrics_.total_actual_demands;

    for (const auto& kv : stage_coalesced_demands) {
        const auto& dem = kv.second;
        if (dem.residency_status == "TIMELY_PREFETCH") {
            metrics_.coalesced_timely_resident_hits = metrics_.coalesced_timely_resident_hits + 1;
        } else if (dem.residency_status == "LATE_IN_FLIGHT") {
            metrics_.coalesced_late_inflight_attachments = metrics_.coalesced_late_inflight_attachments + 1;
        } else if (dem.residency_status == "RESIDENT_HIT") {
            metrics_.coalesced_resident_hits = metrics_.coalesced_resident_hits + 1;
        } else if (dem.residency_status == "DEMAND_MISS") {
            metrics_.coalesced_fresh_demand_misses = metrics_.coalesced_fresh_demand_misses + 1;
        }
    }

    if (metrics_.coalesced_actual_demands > 0) {
        metrics_.coalesced_hbm_hit_rate =
            static_cast<double>(metrics_.coalesced_resident_hits + metrics_.coalesced_timely_resident_hits) /
            static_cast<double>(metrics_.coalesced_actual_demands);
    }

    metrics_.total_evictions = total_evictions_;
    metrics_.unconsumed_prefetch_evictions = unconsumed_prefetch_evictions_;
    metrics_.demand_resident_evictions = demand_resident_evictions_;
    metrics_.pabi_speculative_evictions = pabi_speculative_evictions_;
    metrics_.prefetch_induced_evictions = prefetch_induced_evictions_;
    metrics_.reloaded_experts_count = reloaded_experts_count_;
    metrics_.effective_capacity_bytes = effective_capacity_bytes_;
    metrics_.eviction_policy = eviction_policy_name_;
    metrics_.fa_lru_window_size = fa_lru_window_size_;

    // 3. Process Transfers
    for (const auto& tr : transfers_) {
        if (tr.transfer_class == "PREFETCH") {
            metrics_.total_dispatched_prefetches = metrics_.total_dispatched_prefetches + 1;
            metrics_.speculative_bytes_transferred = metrics_.speculative_bytes_transferred + tr.size_bytes;

            if (tr.outcome == "DEMANDED") {
                metrics_.prefetches_completed_and_used = metrics_.prefetches_completed_and_used + 1;
                metrics_.total_completed_prefetches = metrics_.total_completed_prefetches + 1;
            } else if (tr.outcome == "UNUSED" || tr.outcome == "COMPLETED_UNUSED") {
                metrics_.prefetches_completed_unused = metrics_.prefetches_completed_unused + 1;
                metrics_.completed_unused_bytes = metrics_.completed_unused_bytes + tr.size_bytes;
                metrics_.total_completed_prefetches = metrics_.total_completed_prefetches + 1;
            } else if (tr.outcome == "EVICTED_BEFORE_DEMAND") {
                metrics_.prefetches_evicted_before_demand = metrics_.prefetches_evicted_before_demand + 1;
                metrics_.completed_unused_bytes = metrics_.completed_unused_bytes + tr.size_bytes;
                metrics_.total_completed_prefetches = metrics_.total_completed_prefetches + 1;
            } else if (tr.outcome == "CANCELLED") {
                metrics_.prefetches_cancelled_inflight = metrics_.prefetches_cancelled_inflight + 1;
                metrics_.transfers_cancelled = metrics_.transfers_cancelled + 1;
            } else if (tr.outcome == "STILL_IN_FLIGHT_AT_END") {
                metrics_.prefetches_still_inflight_at_end = metrics_.prefetches_still_inflight_at_end + 1;
            }
        } else {
            metrics_.demand_bytes_transferred = metrics_.demand_bytes_transferred + tr.size_bytes;
        }
    }

    metrics_.total_ddr5_bytes_transferred =
        metrics_.speculative_bytes_transferred + metrics_.demand_bytes_transferred;
    metrics_.total_wasted_speculative_bytes =
        metrics_.completed_unused_bytes + metrics_.cancelled_partial_bytes;

    // 4. Derived Coverage & Rates (Projection Level)
    metrics_.prefetch_assisted_demands =
        metrics_.timely_resident_hits + metrics_.late_inflight_attachments;

    if (metrics_.total_actual_demands > 0) {
        metrics_.overall_coverage =
            static_cast<double>(metrics_.prefetch_assisted_demands) / static_cast<double>(metrics_.total_actual_demands);
        metrics_.timely_rate_of_demands =
            static_cast<double>(metrics_.timely_resident_hits) / static_cast<double>(metrics_.total_actual_demands);
        metrics_.late_rate_of_demands =
            static_cast<double>(metrics_.late_inflight_attachments) / static_cast<double>(metrics_.total_actual_demands);
        metrics_.overall_hbm_hit_rate =
            static_cast<double>(metrics_.total_hbm_resident_hits) / static_cast<double>(metrics_.total_actual_demands);
        metrics_.prefetch_assisted_hit_rate =
            static_cast<double>(metrics_.timely_resident_hits) / static_cast<double>(metrics_.total_actual_demands);
        metrics_.fresh_demand_miss_rate =
            static_cast<double>(metrics_.fresh_demand_misses) / static_cast<double>(metrics_.total_actual_demands);
        metrics_.inflight_attachment_rate =
            static_cast<double>(metrics_.late_inflight_attachments) / static_cast<double>(metrics_.total_actual_demands);
    }

    if (metrics_.demands_with_prediction > 0) {
        metrics_.predicted_coverage =
            static_cast<double>(metrics_.prefetch_assisted_demands) / static_cast<double>(metrics_.demands_with_prediction);
    }

    if (metrics_.total_completed_prefetches > 0) {
        metrics_.completed_prefetch_utilization =
            static_cast<double>(metrics_.prefetches_completed_and_used) / static_cast<double>(metrics_.total_completed_prefetches);
    }

    if (metrics_.total_dispatched_prefetches > 0) {
        metrics_.timely_rate_of_dispatched =
            static_cast<double>(metrics_.timely_resident_hits) / static_cast<double>(metrics_.total_dispatched_prefetches);
        metrics_.late_rate_of_dispatched =
            static_cast<double>(metrics_.late_inflight_attachments) / static_cast<double>(metrics_.total_dispatched_prefetches);
    }

    if (baseline_demand_bytes > 0) {
        metrics_.traffic_overhead_ratio =
            static_cast<double>(metrics_.total_ddr5_bytes_transferred - baseline_demand_bytes) / static_cast<double>(baseline_demand_bytes);
    } else {
        metrics_.traffic_overhead_ratio = 0.0;
    }

    // 5. Quantiles for Expert Load Latency (Projection Level)
    metrics_.expert_latency_overall = compute_quantiles(expert_load_latencies);
    metrics_.expert_latency_resident_hit = compute_quantiles(res_hit_latencies);
    metrics_.expert_latency_timely_prefetch = compute_quantiles(timely_pref_latencies);
    metrics_.expert_latency_late_attachment = compute_quantiles(late_att_latencies);
    metrics_.expert_latency_demand_miss = compute_quantiles(demand_miss_latencies);

    metrics_.mean_expert_load_latency_ns = metrics_.expert_latency_overall.mean_ns;
    metrics_.p50_expert_load_latency_ns = metrics_.expert_latency_overall.p50_ns;
    metrics_.p90_expert_load_latency_ns = metrics_.expert_latency_overall.p90_ns;
    metrics_.p95_expert_load_latency_ns = metrics_.expert_latency_overall.p95_ns;
    metrics_.p99_expert_load_latency_ns = metrics_.expert_latency_overall.p99_ns;
    metrics_.max_expert_load_latency_ns = metrics_.expert_latency_overall.max_ns;

    // 6. Quantiles for Request Latency
    std::vector<double> req_latencies;
    for (const auto& kv : requests_) {
        if (kv.second.latency_ns > 0.0) {
            req_latencies.push_back(kv.second.latency_ns);
        }
    }
    metrics_.request_latency_quantiles = compute_quantiles(req_latencies);
    metrics_.mean_request_latency_ns = metrics_.request_latency_quantiles.mean_ns;
    metrics_.p50_request_latency_ns = metrics_.request_latency_quantiles.p50_ns;
    metrics_.p90_request_latency_ns = metrics_.request_latency_quantiles.p90_ns;
    metrics_.p95_request_latency_ns = metrics_.request_latency_quantiles.p95_ns;
    metrics_.p99_request_latency_ns = metrics_.request_latency_quantiles.p99_ns;
    metrics_.max_request_latency_ns = metrics_.request_latency_quantiles.max_ns;

    // 7. Quantiles for Token Latency
    std::vector<double> tok_latencies;
    for (const auto& tok : tokens_) {
        if (tok.token_duration_ns > 0.0) {
            tok_latencies.push_back(tok.token_duration_ns);
        }
    }
    metrics_.token_latency_quantiles = compute_quantiles(tok_latencies);
    metrics_.mean_token_latency_ns = metrics_.token_latency_quantiles.mean_ns;
    metrics_.p50_token_latency_ns = metrics_.token_latency_quantiles.p50_ns;
    metrics_.p90_token_latency_ns = metrics_.token_latency_quantiles.p90_ns;
    metrics_.p95_token_latency_ns = metrics_.token_latency_quantiles.p95_ns;
    metrics_.p99_token_latency_ns = metrics_.token_latency_quantiles.p99_ns;
    metrics_.max_token_latency_ns = metrics_.token_latency_quantiles.max_ns;

    if (final_device_time_ns > 0.0 && !tokens_.empty()) {
        double sim_sec = final_device_time_ns / 1e9;
        metrics_.simulated_tokens_per_second = static_cast<double>(tokens_.size()) / sim_sec;
    }

    if (!tokens_.empty()) {
        metrics_.wait_per_token_ns = metrics_.aggregate_demand_wait_ns / static_cast<double>(tokens_.size());
    }

    if (metrics_.late_inflight_attachments > 0) {
        double sum_late_wait = 0.0;
        for (double w : late_att_latencies) {
            sum_late_wait = sum_late_wait + w;
        }
        metrics_.avg_remaining_late_wait_ns = sum_late_wait / static_cast<double>(metrics_.late_inflight_attachments);
    }

    double base_xfer_ns = 1376256.0;
    if (!demand_miss_latencies.empty()) {
        base_xfer_ns = metrics_.expert_latency_demand_miss.mean_ns;
    }
    double hidden_stall = static_cast<double>(metrics_.timely_resident_hits) * base_xfer_ns;
    for (double w : late_att_latencies) {
        if (base_xfer_ns > w) {
            hidden_stall = hidden_stall + (base_xfer_ns - w);
        }
    }
    metrics_.stall_hidden_by_prefetch_ns = hidden_stall;

    finalized_ = true;
}

const AggregatedPrefetchMetrics& PrefetchMetricsCollector::get_metrics() const {
    return metrics_;
}

void PrefetchMetricsCollector::dump_artifacts() {
    if (!enabled_) {
        return;
    }

    std::filesystem::create_directories(output_dir_);

    // 1. prediction_events.csv
    {
        std::ofstream csv(output_dir_ + "/prediction_events.csv");
        if (csv.is_open()) {
            csv << "event_id,request_id,seq_id,token_id,source_stage,target_stage,horizon,"
                << "predictor_type,budget,candidate_rank,predicted_expert_id,confidence_score,"
                << "actual_target_experts,is_hit,predict_latency_ns,update_latency_ns,timestamp_ns\n";
            for (const auto& p : predictions_) {
                std::ostringstream act_ss;
                for (size_t i = 0; i < p.actual_target_experts.size(); ++i) {
                    act_ss << p.actual_target_experts[i];
                    if (i + 1 < p.actual_target_experts.size()) {
                        act_ss << ";";
                    }
                }
                csv << p.event_id << ","
                    << p.request_id << ","
                    << p.seq_id << ","
                    << p.token_id << ","
                    << p.source_stage << ","
                    << p.target_stage << ","
                    << p.horizon << ","
                    << p.predictor_type << ","
                    << p.budget << ","
                    << p.candidate_rank << ","
                    << p.predicted_expert_id << ","
                    << std::fixed << std::setprecision(6) << p.confidence_score << ","
                    << "\"" << act_ss.str() << "\","
                    << p.is_hit << ","
                    << std::fixed << std::setprecision(2) << p.predict_latency_ns << ","
                    << p.update_latency_ns << ","
                    << p.timestamp_ns << "\n";
            }
        }
    }

    // 2. candidate_events.csv
    {
        std::ofstream csv(output_dir_ + "/candidate_events.csv");
        if (csv.is_open()) {
            csv << "candidate_id,event_type,request_id,seq_id,token_id,source_stage,target_stage,"
                << "expert_id,prediction_rank,prediction_score,reason,timestamp_ns\n";
            for (const auto& c : candidates_) {
                csv << c.candidate_id << ","
                    << c.event_type << ","
                    << c.request_id << ","
                    << c.seq_id << ","
                    << c.token_id << ","
                    << c.source_stage << ","
                    << c.target_stage << ","
                    << c.expert_id << ","
                    << c.prediction_rank << ","
                    << std::fixed << std::setprecision(6) << c.prediction_score << ","
                    << c.reason << ","
                    << std::fixed << std::setprecision(2) << c.timestamp_ns << "\n";
            }
        }
    }

    // 3. prefetch_transfer_events.csv
    {
        std::ofstream csv(output_dir_ + "/prefetch_transfer_events.csv");
        if (csv.is_open()) {
            csv << "transfer_id,candidate_id,request_id,seq_id,token_id,target_stage,expert_id,"
                << "transfer_class,priority,size_bytes,queue_submit_ts,service_start_ts,completion_ts,"
                << "duration_ns,paused_duration_ns,outcome\n";
            for (const auto& t : transfers_) {
                csv << t.transfer_id << ","
                    << t.candidate_id << ","
                    << t.request_id << ","
                    << t.seq_id << ","
                    << t.token_id << ","
                    << t.target_stage << ","
                    << t.expert_id << ","
                    << t.transfer_class << ","
                    << t.priority << ","
                    << t.size_bytes << ","
                    << std::fixed << std::setprecision(2) << t.queue_submit_ts << ","
                    << t.service_start_ts << ","
                    << t.completion_ts << ","
                    << t.duration_ns << ","
                    << t.paused_duration_ns << ","
                    << t.outcome << "\n";
            }
        }
    }

    // 4. expert_demand_events.csv
    {
        std::ofstream csv(output_dir_ + "/expert_demand_events.csv");
        if (csv.is_open()) {
            csv << "demand_id,request_id,seq_id,token_id,layer_id,expert_id,demand_ts,ready_ts,"
                << "ddr5_wait_ns,hbm_access_ns,residency_status,transfer_id,candidate_id\n";
            for (const auto& d : demands_) {
                csv << d.demand_id << ","
                    << d.request_id << ","
                    << d.seq_id << ","
                    << d.token_id << ","
                    << d.layer_id << ","
                    << d.expert_id << ","
                    << std::fixed << std::setprecision(2) << d.demand_ts << ","
                    << d.ready_ts << ","
                    << d.ddr5_wait_ns << ","
                    << d.hbm_access_ns << ","
                    << d.residency_status << ","
                    << d.transfer_id << ","
                    << d.candidate_id << "\n";
            }
        }
    }

    // 5. request_latency.csv
    {
        std::ofstream csv(output_dir_ + "/request_latency.csv");
        if (csv.is_open()) {
            csv << "request_id,seq_id,prompt_len,gen_tokens,arrival_ts,start_ts,end_ts,latency_ns\n";
            for (const auto& kv : requests_) {
                const auto& r = kv.second;
                csv << r.request_id << ","
                    << r.seq_id << ","
                    << r.prompt_len << ","
                    << r.gen_tokens << ","
                    << std::fixed << std::setprecision(2) << r.arrival_ts << ","
                    << r.start_ts << ","
                    << r.end_ts << ","
                    << r.latency_ns << "\n";
            }
        }
    }

    // 6. token_latency.csv
    {
        std::ofstream csv(output_dir_ + "/token_latency.csv");
        if (csv.is_open()) {
            csv << "request_id,seq_id,token_id,gen_step,token_start_ts,token_end_ts,token_duration_ns\n";
            for (const auto& tok : tokens_) {
                csv << tok.request_id << ","
                    << tok.seq_id << ","
                    << tok.token_id << ","
                    << tok.gen_step << ","
                    << std::fixed << std::setprecision(2) << tok.token_start_ts << ","
                    << tok.token_end_ts << ","
                    << tok.token_duration_ns << "\n";
            }
        }
    }

    // 7. prefetch_metrics.csv
    {
        std::ofstream csv(output_dir_ + "/prefetch_metrics.csv");
        if (csv.is_open()) {
            csv << "config_label,dataset,total_demands,unique_demands,overall_coverage,unique_coverage,"
                << "predicted_coverage,unique_predicted_coverage,prefetch_utilization,"
                << "timely_rate_demands,unique_timely_rate_demands,late_rate_demands,unique_late_rate_demands,"
                << "timely_rate_dispatched,late_rate_dispatched,"
                << "hbm_hit_rate,unique_hbm_hit_rate,fresh_demand_miss_rate,unique_fresh_demand_miss_rate,"
                << "inflight_attachment_rate,unique_inflight_attachment_rate,"
                << "speculative_bytes,demand_bytes,total_ddr5_bytes,wasted_speculative_bytes,traffic_overhead_ratio,"
                << "aggregate_demand_wait_ns,exposed_gpu_stall_ns,stall_hidden_by_prefetch_ns,wait_per_token_ns,"
                << "wait_per_unique_demand_ns,avg_remaining_late_wait_ns,total_simulated_ns,"
                << "mean_expert_load_latency_ns,p50_expert_load_latency_ns,p90_expert_load_latency_ns,p95_expert_load_latency_ns,p99_expert_load_latency_ns,max_expert_load_latency_ns,"
                << "mean_request_latency_ns,p50_request_latency_ns,p90_request_latency_ns,p95_request_latency_ns,p99_request_latency_ns,max_request_latency_ns,"
                << "mean_token_latency_ns,p50_token_latency_ns,p90_token_latency_ns,p95_token_latency_ns,p99_token_latency_ns,max_token_latency_ns,"
                << "simulated_tokens_per_second,"
                << "coalesced_demands,routed_demands,tensor_demands,invalid_decode_step_demands,coalesced_hbm_hit_rate,total_evictions,unconsumed_prefetch_evictions,"
                << "demand_resident_evictions,pabi_speculative_evictions,prefetch_induced_evictions,reloaded_experts_count,"
                << "effective_capacity_bytes,eviction_policy,fa_lru_window_size\n";

            csv << config_label_ << ","
                << dataset_name_ << ","
                << metrics_.total_actual_demands << ","
                << metrics_.unique_actual_demands << ","
                << std::fixed << std::setprecision(6) << metrics_.overall_coverage << ","
                << metrics_.unique_overall_coverage << ","
                << metrics_.predicted_coverage << ","
                << metrics_.unique_predicted_coverage << ","
                << metrics_.completed_prefetch_utilization << ","
                << metrics_.timely_rate_of_demands << ","
                << metrics_.unique_timely_rate_of_demands << ","
                << metrics_.late_rate_of_demands << ","
                << metrics_.unique_late_rate_of_demands << ","
                << metrics_.timely_rate_of_dispatched << ","
                << metrics_.late_rate_of_dispatched << ","
                << metrics_.overall_hbm_hit_rate << ","
                << metrics_.unique_overall_hbm_hit_rate << ","
                << metrics_.fresh_demand_miss_rate << ","
                << metrics_.unique_fresh_demand_miss_rate << ","
                << metrics_.inflight_attachment_rate << ","
                << metrics_.unique_inflight_attachment_rate << ","
                << metrics_.speculative_bytes_transferred << ","
                << metrics_.demand_bytes_transferred << ","
                << metrics_.total_ddr5_bytes_transferred << ","
                << metrics_.total_wasted_speculative_bytes << ","
                << std::fixed << std::setprecision(4) << metrics_.traffic_overhead_ratio << ","
                << std::fixed << std::setprecision(2) << metrics_.aggregate_demand_wait_ns << ","
                << metrics_.exposed_gpu_stall_ns << ","
                << metrics_.stall_hidden_by_prefetch_ns << ","
                << metrics_.wait_per_token_ns << ","
                << metrics_.wait_per_unique_demand_ns << ","
                << metrics_.avg_remaining_late_wait_ns << ","
                << metrics_.total_simulated_execution_ns << ","
                << metrics_.mean_expert_load_latency_ns << ","
                << metrics_.p50_expert_load_latency_ns << ","
                << metrics_.p90_expert_load_latency_ns << ","
                << metrics_.p95_expert_load_latency_ns << ","
                << metrics_.p99_expert_load_latency_ns << ","
                << metrics_.max_expert_load_latency_ns << ","
                << metrics_.mean_request_latency_ns << ","
                << metrics_.p50_request_latency_ns << ","
                << metrics_.p90_request_latency_ns << ","
                << metrics_.p95_request_latency_ns << ","
                << metrics_.p99_request_latency_ns << ","
                << metrics_.max_request_latency_ns << ","
                << metrics_.mean_token_latency_ns << ","
                << metrics_.p50_token_latency_ns << ","
                << metrics_.p90_token_latency_ns << ","
                << metrics_.p95_token_latency_ns << ","
                << metrics_.p99_token_latency_ns << ","
                << metrics_.max_token_latency_ns << ","
                << std::fixed << std::setprecision(2) << metrics_.simulated_tokens_per_second << ","
                << metrics_.coalesced_actual_demands << ","
                << metrics_.routed_actual_demands << ","
                << metrics_.tensor_actual_demands << ","
                << metrics_.invalid_decode_step_demands << ","
                << std::fixed << std::setprecision(6) << metrics_.coalesced_hbm_hit_rate << ","
                << metrics_.total_evictions << ","
                << metrics_.unconsumed_prefetch_evictions << ","
                << metrics_.demand_resident_evictions << ","
                << metrics_.pabi_speculative_evictions << ","
                << metrics_.prefetch_induced_evictions << ","
                << metrics_.reloaded_experts_count << ","
                << metrics_.effective_capacity_bytes << ",\""
                << metrics_.eviction_policy << "\","
                << metrics_.fa_lru_window_size << "\n";
        }
    }

    // 8. experiment_summary.json
    {
        std::ofstream json(output_dir_ + "/experiment_summary.json");
        if (json.is_open()) {
            json << "{\n"
                 << "  \"experiment_metadata\": {\n"
                 << "    \"config_label\": \"" << config_label_ << "\",\n"
                 << "    \"dataset\": \"" << dataset_name_ << "\",\n"
                 << "    \"batch_size\": " << batch_size_ << ",\n"
                 << "    \"predictor_type\": \"" << predictor_type_ << "\",\n"
                 << "    \"horizon\": " << horizon_ << ",\n"
                 << "    \"budget\": " << budget_ << "\n"
                 << "  },\n"
                 << "  \"prefetch_coverage\": {\n"
                 << "    \"total_actual_demands\": " << metrics_.total_actual_demands << ",\n"
                 << "    \"unique_actual_demands\": " << metrics_.unique_actual_demands << ",\n"
                 << "    \"demands_with_prediction\": " << metrics_.demands_with_prediction << ",\n"
                 << "    \"unique_demands_with_prediction\": " << metrics_.unique_demands_with_prediction << ",\n"
                 << "    \"demands_without_prediction\": " << metrics_.demands_without_prediction << ",\n"
                 << "    \"unique_demands_without_prediction\": " << metrics_.unique_demands_without_prediction << ",\n"
                 << "    \"prefetch_assisted_demands\": " << metrics_.prefetch_assisted_demands << ",\n"
                 << "    \"unique_prefetch_assisted_demands\": " << metrics_.unique_prefetch_assisted_demands << ",\n"
                 << "    \"overall_coverage\": " << std::fixed << std::setprecision(6) << metrics_.overall_coverage << ",\n"
                 << "    \"unique_overall_coverage\": " << metrics_.unique_overall_coverage << ",\n"
                 << "    \"predicted_coverage\": " << metrics_.predicted_coverage << ",\n"
                 << "    \"unique_predicted_coverage\": " << metrics_.unique_predicted_coverage << ",\n"
                 << "    \"completed_prefetch_utilization\": " << metrics_.completed_prefetch_utilization << "\n"
                 << "  },\n"
                 << "  \"timely_and_late_rates\": {\n"
                 << "    \"timely_resident_hits\": " << metrics_.timely_resident_hits << ",\n"
                 << "    \"unique_timely_resident_hits\": " << metrics_.unique_timely_resident_hits << ",\n"
                 << "    \"late_inflight_attachments\": " << metrics_.late_inflight_attachments << ",\n"
                 << "    \"unique_late_inflight_attachments\": " << metrics_.unique_late_inflight_attachments << ",\n"
                 << "    \"timely_rate_of_demands\": " << metrics_.timely_rate_of_demands << ",\n"
                 << "    \"unique_timely_rate_of_demands\": " << metrics_.unique_timely_rate_of_demands << ",\n"
                 << "    \"late_rate_of_demands\": " << metrics_.late_rate_of_demands << ",\n"
                 << "    \"unique_late_rate_of_demands\": " << metrics_.unique_late_rate_of_demands << ",\n"
                 << "    \"timely_rate_of_dispatched\": " << metrics_.timely_rate_of_dispatched << ",\n"
                 << "    \"late_rate_of_dispatched\": " << metrics_.late_rate_of_dispatched << ",\n"
                 << "    \"total_dispatched_prefetches\": " << metrics_.total_dispatched_prefetches << ",\n"
                 << "    \"candidates_rejected_admission\": " << metrics_.candidates_rejected_admission << ",\n"
                 << "    \"duplicate_prefetches_skipped\": " << metrics_.duplicate_prefetches_skipped << ",\n"
                 << "    \"transfers_cancelled\": " << metrics_.transfers_cancelled << ",\n"
                 << "    \"avg_remaining_late_wait_ns\": " << std::fixed << std::setprecision(2) << metrics_.avg_remaining_late_wait_ns << "\n"
                 << "  },\n"
                 << "  \"unused_prefetches\": {\n"
                 << "    \"prefetches_completed_and_used\": " << metrics_.prefetches_completed_and_used << ",\n"
                 << "    \"prefetches_completed_unused\": " << metrics_.prefetches_completed_unused << ",\n"
                 << "    \"prefetches_evicted_before_demand\": " << metrics_.prefetches_evicted_before_demand << ",\n"
                 << "    \"prefetches_cancelled_inflight\": " << metrics_.prefetches_cancelled_inflight << ",\n"
                 << "    \"prefetches_still_inflight_at_end\": " << metrics_.prefetches_still_inflight_at_end << ",\n"
                 << "    \"completed_unused_bytes\": " << metrics_.completed_unused_bytes << ",\n"
                 << "    \"cancelled_partial_bytes\": " << metrics_.cancelled_partial_bytes << ",\n"
                 << "    \"total_wasted_speculative_bytes\": " << metrics_.total_wasted_speculative_bytes << "\n"
                 << "  },\n"
                 << "  \"hbm_hit_rates\": {\n"
                 << "    \"total_hbm_resident_hits\": " << metrics_.total_hbm_resident_hits << ",\n"
                 << "    \"unique_resident_hits\": " << metrics_.unique_resident_hits << ",\n"
                 << "    \"fresh_demand_misses\": " << metrics_.fresh_demand_misses << ",\n"
                 << "    \"unique_fresh_demand_misses\": " << metrics_.unique_fresh_demand_misses << ",\n"
                 << "    \"overall_hbm_hit_rate\": " << metrics_.overall_hbm_hit_rate << ",\n"
                 << "    \"unique_overall_hbm_hit_rate\": " << metrics_.unique_overall_hbm_hit_rate << ",\n"
                 << "    \"prefetch_assisted_hit_rate\": " << metrics_.prefetch_assisted_hit_rate << ",\n"
                 << "    \"unique_prefetch_assisted_hit_rate\": " << metrics_.unique_prefetch_assisted_hit_rate << ",\n"
                 << "    \"fresh_demand_miss_rate\": " << metrics_.fresh_demand_miss_rate << ",\n"
                 << "    \"unique_fresh_demand_miss_rate\": " << metrics_.unique_fresh_demand_miss_rate << ",\n"
                 << "    \"inflight_attachment_rate\": " << metrics_.inflight_attachment_rate << ",\n"
                 << "    \"unique_inflight_attachment_rate\": " << metrics_.unique_inflight_attachment_rate << "\n"
                 << "  },\n"
                 << "  \"memory_traffic\": {\n"
                 << "    \"speculative_bytes_transferred\": " << metrics_.speculative_bytes_transferred << ",\n"
                 << "    \"demand_bytes_transferred\": " << metrics_.demand_bytes_transferred << ",\n"
                 << "    \"total_ddr5_bytes_transferred\": " << metrics_.total_ddr5_bytes_transferred << ",\n"
                 << "    \"baseline_demand_bytes\": " << metrics_.baseline_demand_bytes << ",\n"
                 << "    \"traffic_overhead_ratio\": " << std::fixed << std::setprecision(4) << metrics_.traffic_overhead_ratio << "\n"
                 << "  },\n"
                 << "  \"exposed_stall_and_execution\": {\n"
                 << "    \"aggregate_demand_wait_ns\": " << std::fixed << std::setprecision(2) << metrics_.aggregate_demand_wait_ns << ",\n"
                 << "    \"exposed_gpu_stall_ns\": " << metrics_.exposed_gpu_stall_ns << ",\n"
                 << "    \"stall_hidden_by_prefetch_ns\": " << metrics_.stall_hidden_by_prefetch_ns << ",\n"
                 << "    \"wait_per_token_ns\": " << metrics_.wait_per_token_ns << ",\n"
                 << "    \"wait_per_unique_demand_ns\": " << metrics_.wait_per_unique_demand_ns << ",\n"
                 << "    \"total_simulated_execution_ns\": " << metrics_.total_simulated_execution_ns << "\n"
                 << "  },\n"
                 << "  \"expert_load_latency\": {\n"
                 << "    \"mean_ns\": " << metrics_.mean_expert_load_latency_ns << ",\n"
                 << "    \"p50_ns\": " << metrics_.p50_expert_load_latency_ns << ",\n"
                 << "    \"p90_ns\": " << metrics_.p90_expert_load_latency_ns << ",\n"
                 << "    \"p95_ns\": " << metrics_.p95_expert_load_latency_ns << ",\n"
                 << "    \"p99_ns\": " << metrics_.p99_expert_load_latency_ns << ",\n"
                 << "    \"max_ns\": " << metrics_.max_expert_load_latency_ns << "\n"
                 << "  },\n"
                 << "  \"expert_load_latency_distributions\": {\n"
                 << "    \"overall\": {\"count\": " << metrics_.expert_latency_overall.count
                 << ", \"mean_ns\": " << metrics_.expert_latency_overall.mean_ns
                 << ", \"p50_ns\": " << metrics_.expert_latency_overall.p50_ns
                 << ", \"p90_ns\": " << metrics_.expert_latency_overall.p90_ns
                 << ", \"p95_ns\": " << metrics_.expert_latency_overall.p95_ns
                 << ", \"p99_ns\": " << metrics_.expert_latency_overall.p99_ns
                 << ", \"max_ns\": " << metrics_.expert_latency_overall.max_ns << "},\n"
                 << "    \"resident_hits\": {\"count\": " << metrics_.expert_latency_resident_hit.count
                 << ", \"mean_ns\": " << metrics_.expert_latency_resident_hit.mean_ns
                 << ", \"p50_ns\": " << metrics_.expert_latency_resident_hit.p50_ns
                 << ", \"p90_ns\": " << metrics_.expert_latency_resident_hit.p90_ns
                 << ", \"p95_ns\": " << metrics_.expert_latency_resident_hit.p95_ns
                 << ", \"p99_ns\": " << metrics_.expert_latency_resident_hit.p99_ns
                 << ", \"max_ns\": " << metrics_.expert_latency_resident_hit.max_ns << "},\n"
                 << "    \"timely_prefetch\": {\"count\": " << metrics_.expert_latency_timely_prefetch.count
                 << ", \"mean_ns\": " << metrics_.expert_latency_timely_prefetch.mean_ns
                 << ", \"p50_ns\": " << metrics_.expert_latency_timely_prefetch.p50_ns
                 << ", \"p90_ns\": " << metrics_.expert_latency_timely_prefetch.p90_ns
                 << ", \"p95_ns\": " << metrics_.expert_latency_timely_prefetch.p95_ns
                 << ", \"p99_ns\": " << metrics_.expert_latency_timely_prefetch.p99_ns
                 << ", \"max_ns\": " << metrics_.expert_latency_timely_prefetch.max_ns << "},\n"
                 << "    \"late_attachment\": {\"count\": " << metrics_.expert_latency_late_attachment.count
                 << ", \"mean_ns\": " << metrics_.expert_latency_late_attachment.mean_ns
                 << ", \"p50_ns\": " << metrics_.expert_latency_late_attachment.p50_ns
                 << ", \"p90_ns\": " << metrics_.expert_latency_late_attachment.p90_ns
                 << ", \"p95_ns\": " << metrics_.expert_latency_late_attachment.p95_ns
                 << ", \"p99_ns\": " << metrics_.expert_latency_late_attachment.p99_ns
                 << ", \"max_ns\": " << metrics_.expert_latency_late_attachment.max_ns << "},\n"
                 << "    \"demand_miss\": {\"count\": " << metrics_.expert_latency_demand_miss.count
                 << ", \"mean_ns\": " << metrics_.expert_latency_demand_miss.mean_ns
                 << ", \"p50_ns\": " << metrics_.expert_latency_demand_miss.p50_ns
                 << ", \"p90_ns\": " << metrics_.expert_latency_demand_miss.p90_ns
                 << ", \"p95_ns\": " << metrics_.expert_latency_demand_miss.p95_ns
                 << ", \"p99_ns\": " << metrics_.expert_latency_demand_miss.p99_ns
                 << ", \"max_ns\": " << metrics_.expert_latency_demand_miss.max_ns << "}\n"
                 << "  },\n"
                 << "  \"unique_expert_load_latency_distributions\": {\n"
                 << "    \"overall\": {\"count\": " << metrics_.unique_expert_latency_overall.count
                 << ", \"mean_ns\": " << metrics_.unique_expert_latency_overall.mean_ns
                 << ", \"p50_ns\": " << metrics_.unique_expert_latency_overall.p50_ns
                 << ", \"p90_ns\": " << metrics_.unique_expert_latency_overall.p90_ns
                 << ", \"p95_ns\": " << metrics_.unique_expert_latency_overall.p95_ns
                 << ", \"p99_ns\": " << metrics_.unique_expert_latency_overall.p99_ns
                 << ", \"max_ns\": " << metrics_.unique_expert_latency_overall.max_ns << "},\n"
                 << "    \"resident_hits\": {\"count\": " << metrics_.unique_expert_latency_resident_hit.count
                 << ", \"mean_ns\": " << metrics_.unique_expert_latency_resident_hit.mean_ns
                 << ", \"p50_ns\": " << metrics_.unique_expert_latency_resident_hit.p50_ns
                 << ", \"p90_ns\": " << metrics_.unique_expert_latency_resident_hit.p90_ns
                 << ", \"p95_ns\": " << metrics_.unique_expert_latency_resident_hit.p95_ns
                 << ", \"p99_ns\": " << metrics_.unique_expert_latency_resident_hit.p99_ns
                 << ", \"max_ns\": " << metrics_.unique_expert_latency_resident_hit.max_ns << "},\n"
                 << "    \"timely_prefetch\": {\"count\": " << metrics_.unique_expert_latency_timely_prefetch.count
                 << ", \"mean_ns\": " << metrics_.unique_expert_latency_timely_prefetch.mean_ns
                 << ", \"p50_ns\": " << metrics_.unique_expert_latency_timely_prefetch.p50_ns
                 << ", \"p90_ns\": " << metrics_.unique_expert_latency_timely_prefetch.p90_ns
                 << ", \"p95_ns\": " << metrics_.unique_expert_latency_timely_prefetch.p95_ns
                 << ", \"p99_ns\": " << metrics_.unique_expert_latency_timely_prefetch.p99_ns
                 << ", \"max_ns\": " << metrics_.unique_expert_latency_timely_prefetch.max_ns << "},\n"
                 << "    \"late_attachment\": {\"count\": " << metrics_.unique_expert_latency_late_attachment.count
                 << ", \"mean_ns\": " << metrics_.unique_expert_latency_late_attachment.mean_ns
                 << ", \"p50_ns\": " << metrics_.unique_expert_latency_late_attachment.p50_ns
                 << ", \"p90_ns\": " << metrics_.unique_expert_latency_late_attachment.p90_ns
                 << ", \"p95_ns\": " << metrics_.unique_expert_latency_late_attachment.p95_ns
                 << ", \"p99_ns\": " << metrics_.unique_expert_latency_late_attachment.p99_ns
                 << ", \"max_ns\": " << metrics_.unique_expert_latency_late_attachment.max_ns << "},\n"
                 << "    \"demand_miss\": {\"count\": " << metrics_.unique_expert_latency_demand_miss.count
                 << ", \"mean_ns\": " << metrics_.unique_expert_latency_demand_miss.mean_ns
                 << ", \"p50_ns\": " << metrics_.unique_expert_latency_demand_miss.p50_ns
                 << ", \"p90_ns\": " << metrics_.unique_expert_latency_demand_miss.p90_ns
                 << ", \"p95_ns\": " << metrics_.unique_expert_latency_demand_miss.p95_ns
                 << ", \"p99_ns\": " << metrics_.unique_expert_latency_demand_miss.p99_ns
                 << ", \"max_ns\": " << metrics_.unique_expert_latency_demand_miss.max_ns << "}\n"
                 << "  },\n"
                 << "  \"request_latency\": {\n"
                 << "    \"count\": " << metrics_.request_latency_quantiles.count << ",\n"
                 << "    \"mean_ns\": " << metrics_.request_latency_quantiles.mean_ns << ",\n"
                 << "    \"p50_ns\": " << metrics_.request_latency_quantiles.p50_ns << ",\n"
                 << "    \"p90_ns\": " << metrics_.request_latency_quantiles.p90_ns << ",\n"
                 << "    \"p95_ns\": " << metrics_.request_latency_quantiles.p95_ns << ",\n"
                 << "    \"p99_ns\": " << metrics_.request_latency_quantiles.p99_ns << ",\n"
                 << "    \"max_ns\": " << metrics_.request_latency_quantiles.max_ns << "\n"
                 << "  },\n"
                 << "  \"token_latency\": {\n"
                 << "    \"count\": " << metrics_.token_latency_quantiles.count << ",\n"
                 << "    \"mean_ns\": " << metrics_.token_latency_quantiles.mean_ns << ",\n"
                 << "    \"p50_ns\": " << metrics_.token_latency_quantiles.p50_ns << ",\n"
                 << "    \"p90_ns\": " << metrics_.token_latency_quantiles.p90_ns << ",\n"
                 << "    \"p95_ns\": " << metrics_.token_latency_quantiles.p95_ns << ",\n"
                 << "    \"p99_ns\": " << metrics_.token_latency_quantiles.p99_ns << ",\n"
                 << "    \"max_ns\": " << metrics_.token_latency_quantiles.max_ns << ",\n"
                 << "    \"simulated_tokens_per_second\": " << metrics_.simulated_tokens_per_second << "\n"
                 << "  },\n"
                 << "  \"hbm_management\": {\n"
                 << "    \"eviction_policy\": \"" << metrics_.eviction_policy << "\",\n"
                 << "    \"fa_lru_window_size\": " << metrics_.fa_lru_window_size << ",\n"
                 << "    \"effective_capacity_bytes\": " << metrics_.effective_capacity_bytes << ",\n"
                 << "    \"coalesced_actual_demands\": " << metrics_.coalesced_actual_demands << ",\n"
                 << "    \"routed_actual_demands\": " << metrics_.routed_actual_demands << ",\n"
                 << "    \"tensor_actual_demands\": " << metrics_.tensor_actual_demands << ",\n"
                 << "    \"invalid_decode_step_demands\": " << metrics_.invalid_decode_step_demands << ",\n"
                 << "    \"coalesced_resident_hits\": " << metrics_.coalesced_resident_hits << ",\n"
                 << "    \"coalesced_timely_resident_hits\": " << metrics_.coalesced_timely_resident_hits << ",\n"
                 << "    \"coalesced_late_inflight_attachments\": " << metrics_.coalesced_late_inflight_attachments << ",\n"
                 << "    \"coalesced_fresh_demand_misses\": " << metrics_.coalesced_fresh_demand_misses << ",\n"
                 << "    \"coalesced_hbm_hit_rate\": " << std::fixed << std::setprecision(6) << metrics_.coalesced_hbm_hit_rate << ",\n"
                 << "    \"total_evictions\": " << metrics_.total_evictions << ",\n"
                 << "    \"unconsumed_prefetch_evictions\": " << metrics_.unconsumed_prefetch_evictions << ",\n"
                 << "    \"demand_resident_evictions\": " << metrics_.demand_resident_evictions << ",\n"
                 << "    \"pabi_speculative_evictions\": " << metrics_.pabi_speculative_evictions << ",\n"
                 << "    \"prefetch_induced_evictions\": " << metrics_.prefetch_induced_evictions << ",\n"
                 << "    \"reloaded_experts_count\": " << metrics_.reloaded_experts_count << "\n"
                 << "  }\n"
                 << "}\n";
        }
    }

    // 8b. prefetch_summary.json (canonical Point 4 JSON summary)
    try {
        std::filesystem::copy_file(
            output_dir_ + "/experiment_summary.json",
            output_dir_ + "/prefetch_summary.json",
            std::filesystem::copy_options::overwrite_existing);
    } catch (...) {
    }

    dump_unified_event_log(output_dir_ + "/prefetch_events.csv");
}

void PrefetchMetricsCollector::dump_unified_event_log(const std::string& path) {
    if (!enabled_) {
        return;
    }

    struct UnifiedEventItem {
        double timestamp_ns = 0.0;
        std::string event_type;
        int request_id = 0;
        int seq_id = 0;
        int token_id = 0;
        int layer_id = -1;
        int expert_id = -1;
        uint64_t candidate_id = 0;
        uint64_t transfer_id = 0;
        std::string details;
    };

    std::vector<UnifiedEventItem> items;
    size_t est_size = predictions_.size() + candidates_.size() + (transfers_.size() * 2) + demands_.size() + (tokens_.size() * 2);
    items.reserve(est_size);

    for (const auto& p : predictions_) {
        UnifiedEventItem it;
        it.timestamp_ns = p.timestamp_ns;
        it.event_type = "PREDICTION";
        it.request_id = p.request_id;
        it.seq_id = p.seq_id;
        it.token_id = p.token_id;
        it.layer_id = p.target_stage;
        it.expert_id = p.predicted_expert_id;
        it.details = "rank=" + std::to_string(p.candidate_rank) + ";conf=" + std::to_string(p.confidence_score) + ";hit=" + std::to_string(p.is_hit);
        items.push_back(it);
    }

    for (const auto& c : candidates_) {
        UnifiedEventItem it;
        it.timestamp_ns = c.timestamp_ns;
        it.event_type = "CANDIDATE_" + c.event_type;
        it.request_id = c.request_id;
        it.seq_id = c.seq_id;
        it.token_id = c.token_id;
        it.layer_id = c.target_stage;
        it.expert_id = c.expert_id;
        it.candidate_id = c.candidate_id;
        it.details = "reason=" + c.reason;
        items.push_back(it);
    }

    for (const auto& tr : transfers_) {
        UnifiedEventItem it;
        it.timestamp_ns = tr.queue_submit_ts;
        it.event_type = "TRANSFER_SUBMITTED";
        it.request_id = tr.request_id;
        it.seq_id = tr.seq_id;
        it.token_id = tr.token_id;
        it.layer_id = tr.target_stage;
        it.expert_id = tr.expert_id;
        it.candidate_id = tr.candidate_id;
        it.transfer_id = tr.transfer_id;
        it.details = "class=" + tr.transfer_class + ";bytes=" + std::to_string(tr.size_bytes);
        items.push_back(it);

        if (tr.service_start_ts >= 0.0) {
            UnifiedEventItem st;
            st.timestamp_ns = tr.service_start_ts;
            st.event_type = "TRANSFER_STARTED";
            st.request_id = tr.request_id;
            st.seq_id = tr.seq_id;
            st.token_id = tr.token_id;
            st.layer_id = tr.target_stage;
            st.expert_id = tr.expert_id;
            st.transfer_id = tr.transfer_id;
            items.push_back(st);
        }

        if (tr.completion_ts >= 0.0) {
            UnifiedEventItem comp;
            comp.timestamp_ns = tr.completion_ts;
            comp.event_type = "TRANSFER_COMPLETED";
            comp.request_id = tr.request_id;
            comp.seq_id = tr.seq_id;
            comp.token_id = tr.token_id;
            comp.layer_id = tr.target_stage;
            comp.expert_id = tr.expert_id;
            comp.transfer_id = tr.transfer_id;
            comp.details = "outcome=" + tr.outcome;
            items.push_back(comp);
        }
    }

    for (const auto& dem : demands_) {
        UnifiedEventItem it;
        it.timestamp_ns = dem.demand_ts;
        it.event_type = "EXPERT_DEMANDED";
        it.request_id = dem.request_id;
        it.seq_id = dem.seq_id;
        it.token_id = dem.token_id;
        it.layer_id = dem.layer_id;
        it.expert_id = dem.expert_id;
        it.candidate_id = dem.candidate_id;
        it.transfer_id = dem.transfer_id;
        it.details = "status=" + dem.residency_status + ";wait_ns=" + std::to_string(dem.ddr5_wait_ns) + ";ready_ts=" + std::to_string(dem.ready_ts);
        items.push_back(it);
    }

    for (const auto& tok : tokens_) {
        UnifiedEventItem st;
        st.timestamp_ns = tok.token_start_ts;
        st.event_type = "TOKEN_STARTED";
        st.request_id = tok.request_id;
        st.seq_id = tok.seq_id;
        st.token_id = tok.token_id;
        st.details = "step=" + std::to_string(tok.gen_step);
        items.push_back(st);

        UnifiedEventItem ed;
        ed.timestamp_ns = tok.token_end_ts;
        ed.event_type = "TOKEN_COMPLETED";
        ed.request_id = tok.request_id;
        ed.seq_id = tok.seq_id;
        ed.token_id = tok.token_id;
        ed.details = "duration_ns=" + std::to_string(tok.token_duration_ns);
        items.push_back(ed);
    }

    for (const auto& kv : requests_) {
        UnifiedEventItem st;
        st.timestamp_ns = kv.second.start_ts;
        st.event_type = "REQUEST_STARTED";
        st.request_id = kv.second.request_id;
        st.seq_id = kv.second.seq_id;
        st.details = "prompt_len=" + std::to_string(kv.second.prompt_len);
        items.push_back(st);

        if (kv.second.end_ts > kv.second.start_ts) {
            UnifiedEventItem ed;
            ed.timestamp_ns = kv.second.end_ts;
            ed.event_type = "REQUEST_COMPLETED";
            ed.request_id = kv.second.request_id;
            ed.seq_id = kv.second.seq_id;
            ed.details = "gen_tokens=" + std::to_string(kv.second.gen_tokens) + ";latency_ns=" + std::to_string(kv.second.latency_ns);
            items.push_back(ed);
        }
    }

    std::sort(items.begin(), items.end(), [](const UnifiedEventItem& a, const UnifiedEventItem& b) {
        if (a.timestamp_ns != b.timestamp_ns) {
            return a.timestamp_ns < b.timestamp_ns;
        }
        return a.event_type < b.event_type;
    });

    std::ofstream csv(path);
    if (csv.is_open()) {
        csv << "event_id,timestamp_ns,event_type,request_id,seq_id,token_id,layer_id,expert_id,candidate_id,transfer_id,details\n";
        uint64_t eid = 1;
        for (const auto& ev : items) {
            csv << eid << ","
                << std::fixed << std::setprecision(2) << ev.timestamp_ns << ","
                << ev.event_type << ","
                << ev.request_id << ","
                << ev.seq_id << ","
                << ev.token_id << ","
                << ev.layer_id << ","
                << ev.expert_id << ","
                << ev.candidate_id << ","
                << ev.transfer_id << ","
                << "\"" << ev.details << "\"\n";
            eid = eid + 1;
        }
    }
}

void PrefetchMetricsCollector::reset() {
    predictions_.clear();
    candidates_.clear();
    transfers_.clear();
    transfer_index_map_.clear();
    demands_.clear();
    requests_.clear();
    tokens_.clear();
    token_start_times_.clear();
    token_to_gen_step_.clear();
    accumulated_exposed_stall_ns_ = 0.0;
    current_decode_step_ = -1;
    total_evictions_ = 0;
    unconsumed_prefetch_evictions_ = 0;
    demand_resident_evictions_ = 0;
    pabi_speculative_evictions_ = 0;
    prefetch_induced_evictions_ = 0;
    reloaded_experts_count_ = 0;
    effective_capacity_bytes_ = 0;
    eviction_policy_name_ = "lru";
    fa_lru_window_size_ = 16;
    metrics_ = AggregatedPrefetchMetrics();
    finalized_ = false;
    next_demand_id_ = 1;
}

} // namespace llm_system
