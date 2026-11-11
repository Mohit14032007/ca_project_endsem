#include "prefetch/prefetch_candidate_queue.h"
#include "prefetch/prefetch_metrics_collector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_set>

namespace llm_system {

std::string to_string(CandidateState state) {
    switch (state) {
        case CandidateState::ADMITTED: return "ADMITTED";
        case CandidateState::READY: return "READY";
        case CandidateState::RECONCILED_HIT: return "RECONCILED_HIT";
        case CandidateState::EXPIRED_MISS: return "EXPIRED_MISS";
        case CandidateState::CANCELLED: return "CANCELLED";
        case CandidateState::EVICTED: return "EVICTED";
        default: return "UNKNOWN";
    }
}

PrefetchCandidateQueue::PrefetchCandidateQueue(
    const PrefetchConfig& config,
    int num_stages,
    int num_experts,
    const std::string& output_dir)
    : config_(config),
      num_stages_(num_stages),
      num_experts_(num_experts),
      capacity_(config.queue_capacity),
      priority_policy_(config.priority_policy),
      duplicate_policy_(config.duplicate_policy),
      overflow_policy_(config.overflow_policy),
      output_dir_(output_dir) {

    std::filesystem::create_directories(output_dir_);
    csv_log_path_ = output_dir_ + "/prefetch_queue_events.csv";
    summary_json_path_ = output_dir_ + "/prefetch_queue_summary.json";

    csv_file_.open(csv_log_path_, std::ios::out | std::ios::trunc);
    if (csv_file_.is_open()) {
        csv_file_ << "candidate_id,request_id,token_id,source_stage_id,target_stage_id,"
                  << "expert_id,prediction_rank,prediction_score,horizon,insertion_seq,"
                  << "event_type,state\n";
    }
}

PrefetchCandidateQueue::~PrefetchCandidateQueue() {
    if (!finalized_) {
        finalize("");
    }
}

size_t PrefetchCandidateQueue::size() const {
    return active_candidates_.size();
}

bool PrefetchCandidateQueue::empty() const {
    return active_candidates_.empty();
}

int PrefetchCandidateQueue::get_capacity() const {
    return capacity_;
}

const QueueMetrics& PrefetchCandidateQueue::get_metrics() const {
    return metrics_;
}

std::string PrefetchCandidateQueue::get_csv_path() const {
    return csv_log_path_;
}

std::string PrefetchCandidateQueue::get_summary_path() const {
    return summary_json_path_;
}

bool PrefetchCandidateQueue::validate_candidate(const PrefetchCandidate& cand) {
    if (cand.request_id < 0) {
        metrics_.invalid_rejected = metrics_.invalid_rejected + 1;
        return false;
    }
    if (cand.token_id < 0) {
        metrics_.invalid_rejected = metrics_.invalid_rejected + 1;
        return false;
    }
    if (cand.target_stage_id < 0) {
        metrics_.invalid_rejected = metrics_.invalid_rejected + 1;
        return false;
    }
    if (cand.expert_id < 0) {
        metrics_.invalid_rejected = metrics_.invalid_rejected + 1;
        return false;
    }
    if (cand.expert_id >= num_experts_) {
        metrics_.invalid_rejected = metrics_.invalid_rejected + 1;
        return false;
    }
    if (cand.prediction_rank < 1) {
        metrics_.invalid_rejected = metrics_.invalid_rejected + 1;
        return false;
    }
    if (std::isnan(cand.prediction_score)) {
        metrics_.invalid_rejected = metrics_.invalid_rejected + 1;
        return false;
    }
    if (std::isinf(cand.prediction_score)) {
        metrics_.invalid_rejected = metrics_.invalid_rejected + 1;
        return false;
    }
    return true;
}

void PrefetchCandidateQueue::update_occupancy_sample() {
    int cur_occ = static_cast<int>(active_candidates_.size());
    if (cur_occ > metrics_.queue_high_water_mark) {
        metrics_.queue_high_water_mark = cur_occ;
    }
    metrics_.sum_queue_occupancy = metrics_.sum_queue_occupancy + static_cast<uint64_t>(cur_occ);
    metrics_.queue_sample_count = metrics_.queue_sample_count + 1;
}

void PrefetchCandidateQueue::log_event(const PrefetchCandidate& cand, const std::string& event_type) {
    if (csv_file_.is_open()) {
        csv_file_ << cand.candidate_id << ","
                  << cand.request_id << ","
                  << cand.token_id << ","
                  << cand.source_stage_id << ","
                  << cand.target_stage_id << ","
                  << cand.expert_id << ","
                  << cand.prediction_rank << ","
                  << std::fixed << std::setprecision(6) << cand.prediction_score << ","
                  << cand.prediction_horizon << ","
                  << cand.insertion_sequence << ","
                  << event_type << ","
                  << to_string(cand.state) << "\n";
    }
}

bool PrefetchCandidateQueue::enqueue_candidate(const PrefetchCandidate& candidate) {
    metrics_.total_candidates_received = metrics_.total_candidates_received + 1;

    if (!validate_candidate(candidate)) {
        return false;
    }

    CandidateKey key = candidate.get_key();
    auto it = active_candidates_.find(key);
    if (it != active_candidates_.end()) {
        // Candidate is duplicate of an active entry
        if (duplicate_policy_ == DuplicatePolicy::RETAIN_HIGHEST_PRIORITY) {
            CandidatePriorityComparator cmp(priority_policy_);
            if (cmp(candidate, it->second)) {
                // Incoming candidate has higher priority: update existing entry
                it->second.prediction_score = candidate.prediction_score;
                it->second.prediction_rank = candidate.prediction_rank;
                it->second.source_stage_id = candidate.source_stage_id;
                log_event(it->second, "DUPLICATE_UPDATED");
                return true;
            } else {
                metrics_.duplicates_suppressed = metrics_.duplicates_suppressed + 1;
                log_event(candidate, "DUPLICATE_SUPPRESSED");
                PrefetchMetricsCollector::getInstance().record_candidate_rejected(
                    candidate.request_id, candidate.seq_id, candidate.token_id,
                    candidate.source_stage_id, candidate.target_stage_id, candidate.expert_id,
                    "DUPLICATE_SUPPRESSED", candidate.admission_timestamp_ns);
                return false;
            }
        } else {
            metrics_.duplicates_suppressed = metrics_.duplicates_suppressed + 1;
            log_event(candidate, "DUPLICATE_REJECTED");
            PrefetchMetricsCollector::getInstance().record_candidate_rejected(
                candidate.request_id, candidate.seq_id, candidate.token_id,
                candidate.source_stage_id, candidate.target_stage_id, candidate.expert_id,
                "DUPLICATE_REJECTED", candidate.admission_timestamp_ns);
            return false;
        }
    }

    // Capacity enforcement
    if (active_candidates_.size() >= static_cast<size_t>(capacity_)) {
        if (overflow_policy_ == OverflowPolicy::REJECT_NEW) {
            metrics_.capacity_rejected = metrics_.capacity_rejected + 1;
            log_event(candidate, "OVERFLOW_REJECTED");
            PrefetchMetricsCollector::getInstance().record_candidate_rejected(
                candidate.request_id, candidate.seq_id, candidate.token_id,
                candidate.source_stage_id, candidate.target_stage_id, candidate.expert_id,
                "OVERFLOW_REJECTED", candidate.admission_timestamp_ns);
            return false;
        } else {
            // REJECT_LOWEST_PRIORITY: Find lowest priority entry currently in queue
            CandidatePriorityComparator cmp(priority_policy_);
            auto lowest_it = active_candidates_.begin();
            for (auto iter = active_candidates_.begin(); iter != active_candidates_.end(); ++iter) {
                if (cmp(lowest_it->second, iter->second)) {
                    lowest_it = iter;
                }
            }

            if (cmp(candidate, lowest_it->second)) {
                // Incoming candidate has strictly higher priority than lowest
                lowest_it->second.state = CandidateState::EVICTED;
                metrics_.overflow_evicted = metrics_.overflow_evicted + 1;
                log_event(lowest_it->second, "OVERFLOW_EVICTED");

                CandidateKey evict_key = lowest_it->first;
                active_candidates_.erase(lowest_it);

                // Clean stage_index_ for evicted candidate
                uint64_t evict_stg_key = make_stage_key(
                    evict_key.request_id, evict_key.token_id, evict_key.target_stage_id);
                auto stg_it = stage_index_.find(evict_stg_key);
                if (stg_it != stage_index_.end()) {
                    auto& vec = stg_it->second;
                    vec.erase(std::remove(vec.begin(), vec.end(), evict_key), vec.end());
                }
            } else {
                // Incoming candidate is lower or equal priority
                metrics_.capacity_rejected = metrics_.capacity_rejected + 1;
                log_event(candidate, "OVERFLOW_REJECTED");
                PrefetchMetricsCollector::getInstance().record_candidate_rejected(
                    candidate.request_id, candidate.seq_id, candidate.token_id,
                    candidate.source_stage_id, candidate.target_stage_id, candidate.expert_id,
                    "OVERFLOW_REJECTED", candidate.admission_timestamp_ns);
                return false;
            }
        }
    }

    // Admit new candidate
    PrefetchCandidate admitted = candidate;
    admitted.candidate_id = next_candidate_id_;
    next_candidate_id_ = next_candidate_id_ + 1;
    admitted.insertion_sequence = next_insertion_seq_;
    next_insertion_seq_ = next_insertion_seq_ + 1;
    admitted.state = CandidateState::ADMITTED;

    if (candidate.admission_timestamp_ns > 0.0) {
        admitted.admission_timestamp_ns = candidate.admission_timestamp_ns;
    } else {
        admitted.admission_timestamp_ns = 0.0;
    }

    active_candidates_[key] = admitted;

    uint64_t stg_key = make_stage_key(admitted.request_id, admitted.token_id, admitted.target_stage_id);
    stage_index_[stg_key].push_back(key);
    request_index_[admitted.request_id].push_back(key);

    metrics_.candidates_admitted = metrics_.candidates_admitted + 1;
    update_occupancy_sample();
    log_event(admitted, "ADMIT");
    PrefetchMetricsCollector::getInstance().record_candidate_admitted(
        admitted.candidate_id, admitted.request_id, admitted.seq_id, admitted.token_id,
        admitted.source_stage_id, admitted.target_stage_id, admitted.expert_id,
        admitted.prediction_rank, admitted.prediction_score, admitted.admission_timestamp_ns);
    return true;
}

int PrefetchCandidateQueue::enqueue_candidates(
    int request_id,
    int token_id,
    int source_stage_id,
    int target_stage_id,
    int horizon,
    const PredictionResult& result,
    int seq_id,
    double sim_time_ns) {

    int admitted_count = 0;
    const auto& cands = result.candidate_experts;
    const auto& scores = result.scores;

    for (size_t rank = 0; rank < cands.size(); ++rank) {
        PrefetchCandidate cand;
        cand.request_id = request_id;
        cand.seq_id = seq_id;
        cand.token_id = token_id;
        cand.source_stage_id = source_stage_id;
        cand.target_stage_id = target_stage_id;
        cand.expert_id = cands[rank];
        cand.prediction_rank = static_cast<int>(rank + 1);
        cand.prediction_horizon = horizon;
        cand.admission_timestamp_ns = sim_time_ns;

        if (rank < scores.size()) {
            cand.prediction_score = scores[rank].score;
        } else {
            cand.prediction_score = 0.0;
        }

        if (enqueue_candidate(cand)) {
            admitted_count = admitted_count + 1;
        }
    }

    return admitted_count;
}

int PrefetchCandidateQueue::reconcile_stage(
    int request_id,
    int token_id,
    int target_stage_id,
    const std::vector<int>& actual_target_experts) {

    uint64_t stg_key = make_stage_key(request_id, token_id, target_stage_id);
    auto stg_it = stage_index_.find(stg_key);
    if (stg_it == stage_index_.end()) {
        return 0;
    }

    std::unordered_set<int> act_set(actual_target_experts.begin(), actual_target_experts.end());
    std::vector<CandidateKey> keys = stg_it->second;
    stage_index_.erase(stg_it);

    int reconciled_count = 0;
    for (const auto& key : keys) {
        auto it = active_candidates_.find(key);
        if (it != active_candidates_.end()) {
            PrefetchCandidate cand = it->second;
            if (act_set.find(cand.expert_id) != act_set.end()) {
                cand.state = CandidateState::RECONCILED_HIT;
                metrics_.reconciled_hits = metrics_.reconciled_hits + 1;
                log_event(cand, "RECONCILED_HIT");
            } else {
                cand.state = CandidateState::EXPIRED_MISS;
                metrics_.expired_misses = metrics_.expired_misses + 1;
                log_event(cand, "EXPIRED_MISS");
            }
            active_candidates_.erase(it);
            reconciled_count = reconciled_count + 1;
        }
    }

    update_occupancy_sample();
    return reconciled_count;
}

int PrefetchCandidateQueue::purge_request(int request_id) {
    auto req_it = request_index_.find(request_id);
    if (req_it == request_index_.end()) {
        return 0;
    }

    std::vector<CandidateKey> keys = req_it->second;
    request_index_.erase(req_it);

    int purged_count = 0;
    for (const auto& key : keys) {
        auto it = active_candidates_.find(key);
        if (it != active_candidates_.end()) {
            PrefetchCandidate cand = it->second;
            cand.state = CandidateState::CANCELLED;
            metrics_.candidates_cancelled = metrics_.candidates_cancelled + 1;
            log_event(cand, "CANCELLED");

            // Clean from stage_index_
            uint64_t stg_key = make_stage_key(cand.request_id, cand.token_id, cand.target_stage_id);
            auto s_it = stage_index_.find(stg_key);
            if (s_it != stage_index_.end()) {
                auto& vec = s_it->second;
                vec.erase(std::remove(vec.begin(), vec.end(), key), vec.end());
            }

            active_candidates_.erase(it);
            purged_count = purged_count + 1;
        }
    }

    update_occupancy_sample();
    return purged_count;
}

std::vector<PrefetchCandidate> PrefetchCandidateQueue::select_ready_candidates(int max_count) {
    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<PrefetchCandidate> candidates;
    candidates.reserve(active_candidates_.size());
    for (const auto& kv : active_candidates_) {
        if (kv.second.state == CandidateState::ADMITTED) {
            candidates.push_back(kv.second);
        }
    }

    CandidatePriorityComparator cmp(priority_policy_);
    std::sort(candidates.begin(), candidates.end(), cmp);

    int count_to_take = static_cast<int>(candidates.size());
    if (max_count >= 0 && max_count < count_to_take) {
        count_to_take = max_count;
    }

    std::vector<PrefetchCandidate> selected;
    selected.reserve(count_to_take);
    for (int i = 0; i < count_to_take; ++i) {
        selected.push_back(candidates[i]);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double lat_ns = std::chrono::duration<double, std::nano>(t_end - t_start).count();

    metrics_.total_selection_latency_ns = metrics_.total_selection_latency_ns + lat_ns;
    metrics_.selection_call_count = metrics_.selection_call_count + 1;
    metrics_.candidates_selected = metrics_.candidates_selected + count_to_take;

    return selected;
}

bool PrefetchCandidateQueue::mark_candidate_dispatched(uint64_t candidate_id, double dispatch_time_ns) {
    for (auto& kv : active_candidates_) {
        if (kv.second.candidate_id == candidate_id) {
            kv.second.state = CandidateState::READY;
            log_event(kv.second, "DISPATCHED");
            double ts = kv.second.admission_timestamp_ns;
            if (dispatch_time_ns >= 0.0) {
                ts = dispatch_time_ns;
            }
            PrefetchMetricsCollector::getInstance().record_candidate_dispatched(
                candidate_id, kv.second.request_id, kv.second.seq_id, kv.second.token_id,
                kv.second.target_stage_id, kv.second.expert_id, ts);
            return true;
        }
    }
    return false;
}

void PrefetchCandidateQueue::finalize(const std::string& custom_dir) {
    if (finalized_) {
        return;
    }
    finalized_ = true;

    if (!custom_dir.empty()) {
        std::filesystem::create_directories(custom_dir);
        summary_json_path_ = custom_dir + "/prefetch_queue_summary.json";
    }

    if (csv_file_.is_open()) {
        csv_file_.flush();
        csv_file_.close();
    }

    std::ofstream json_out(summary_json_path_, std::ios::out | std::ios::trunc);
    if (json_out.is_open()) {
        json_out << "{\n"
                 << "  \"queue_capacity\": " << capacity_ << ",\n"
                 << "  \"priority_policy\": \"" << to_string(priority_policy_) << "\",\n"
                 << "  \"duplicate_policy\": \"" << to_string(duplicate_policy_) << "\",\n"
                 << "  \"overflow_policy\": \"" << to_string(overflow_policy_) << "\",\n"
                 << "  \"total_candidates_received\": " << metrics_.total_candidates_received << ",\n"
                 << "  \"candidates_admitted\": " << metrics_.candidates_admitted << ",\n"
                 << "  \"duplicates_suppressed\": " << metrics_.duplicates_suppressed << ",\n"
                 << "  \"invalid_rejected\": " << metrics_.invalid_rejected << ",\n"
                 << "  \"capacity_rejected\": " << metrics_.capacity_rejected << ",\n"
                 << "  \"overflow_evicted\": " << metrics_.overflow_evicted << ",\n"
                 << "  \"reconciled_hits\": " << metrics_.reconciled_hits << ",\n"
                 << "  \"expired_misses\": " << metrics_.expired_misses << ",\n"
                 << "  \"candidates_cancelled\": " << metrics_.candidates_cancelled << ",\n"
                 << "  \"candidates_selected\": " << metrics_.candidates_selected << ",\n"
                 << "  \"queue_high_water_mark\": " << metrics_.queue_high_water_mark << ",\n"
                 << "  \"mean_queue_occupancy\": " << metrics_.mean_queue_occupancy() << ",\n"
                 << "  \"mean_selection_latency_ns\": " << metrics_.mean_selection_latency_ns() << "\n"
                 << "}\n";
        json_out.close();
    }
}

} // namespace llm_system

