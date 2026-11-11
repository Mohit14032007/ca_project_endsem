#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "prefetch/prefetch_config.h"
#include "prefetch/predictor_interface.h"

namespace llm_system {

/**
 * Lifecycle state of a prefetch candidate.
 * Strictly avoids simulated memory transfers or fake HBM resident states.
 */
enum class CandidateState {
    ADMITTED,       // Admitted to queue and pending scheduling
    READY,          // Prioritized and ready for future Phase 4 transfer selection
    RECONCILED_HIT, // Authoritatively verified as an actual routing hit at target stage
    EXPIRED_MISS,   // Reached target stage without router selection (false positive)
    CANCELLED,      // Cancelled due to request reset or preemption
    EVICTED         // Evicted due to queue capacity overflow
};

std::string to_string(CandidateState state);

/**
 * Composite key uniquely identifying a candidate prefetch target.
 * Completely immune to cross-request, cross-token, or cross-stage collisions.
 */
struct CandidateKey {
    int request_id = 0;
    int token_id = 0;
    int target_stage_id = 0;
    int expert_id = 0;

    bool operator==(const CandidateKey& o) const {
        if (request_id != o.request_id) {
            return false;
        }
        if (token_id != o.token_id) {
            return false;
        }
        if (target_stage_id != o.target_stage_id) {
            return false;
        }
        if (expert_id != o.expert_id) {
            return false;
        }
        return true;
    }
};

struct CandidateKeyHash {
    std::size_t operator()(const CandidateKey& k) const {
        std::size_t h1 = std::hash<int>{}(k.request_id);
        std::size_t h2 = std::hash<int>{}(k.token_id);
        std::size_t h3 = std::hash<int>{}(k.target_stage_id);
        std::size_t h4 = std::hash<int>{}(k.expert_id);
        std::size_t seed = h1;
        seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= h4 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

/**
 * Explicit candidate entry representation in the prefetch queue.
 */
struct PrefetchCandidate {
    uint64_t candidate_id = 0;          // Unique monotonic candidate ID
    int request_id = 0;
    int seq_id = 0;
    int token_id = 0;
    int source_stage_id = -1;
    int target_stage_id = 0;
    int expert_id = 0;
    int prediction_rank = 0;            // 1-indexed rank from predictor (1 to B)
    double prediction_score = 0.0;      // Associated prediction probability or score
    int prediction_horizon = 1;         // Horizon H
    uint64_t insertion_sequence = 0;    // Insertion sequence number for deterministic tie-breaking
    CandidateState state = CandidateState::ADMITTED;
    double admission_timestamp_ns = 0.0;

    CandidateKey get_key() const {
        CandidateKey k;
        k.request_id = request_id;
        k.token_id = token_id;
        k.target_stage_id = target_stage_id;
        k.expert_id = expert_id;
        return k;
    }
};

/**
 * Deterministic candidate priority ordering functor.
 * Strict priority:
 * 1. Target deadline urgency (earlier token, then earlier target stage)
 * 2. Prediction score (higher confidence preferred)
 * 3. Prediction rank (lower rank 1..B preferred)
 * 4. Insertion sequence (earlier admitted candidate preferred)
 * 5. Expert ID (deterministic tie-breaker)
 */
struct CandidatePriorityComparator {
    PrefetchPriorityPolicy policy = PrefetchPriorityPolicy::DEADLINE_SCORE_RANK;

    CandidatePriorityComparator() = default;
    explicit CandidatePriorityComparator(PrefetchPriorityPolicy p) : policy(p) {}

    bool operator()(const PrefetchCandidate& a, const PrefetchCandidate& b) const {
        if (policy == PrefetchPriorityPolicy::DEADLINE_ONLY) {
            int64_t deadline_a = static_cast<int64_t>(a.token_id) * 1000000LL + a.target_stage_id;
            int64_t deadline_b = static_cast<int64_t>(b.token_id) * 1000000LL + b.target_stage_id;
            if (deadline_a != deadline_b) {
                return deadline_a < deadline_b;
            }
            if (a.insertion_sequence != b.insertion_sequence) {
                return a.insertion_sequence < b.insertion_sequence;
            }
            return a.expert_id < b.expert_id;
        }

        if (policy == PrefetchPriorityPolicy::SCORE_ONLY) {
            if (std::abs(a.prediction_score - b.prediction_score) > 1e-9) {
                return a.prediction_score > b.prediction_score;
            }
            if (a.prediction_rank != b.prediction_rank) {
                return a.prediction_rank < b.prediction_rank;
            }
            if (a.insertion_sequence != b.insertion_sequence) {
                return a.insertion_sequence < b.insertion_sequence;
            }
            return a.expert_id < b.expert_id;
        }

        // Default: DEADLINE_SCORE_RANK
        int64_t deadline_a = static_cast<int64_t>(a.token_id) * 1000000LL + a.target_stage_id;
        int64_t deadline_b = static_cast<int64_t>(b.token_id) * 1000000LL + b.target_stage_id;
        if (deadline_a != deadline_b) {
            return deadline_a < deadline_b;
        }
        if (std::abs(a.prediction_score - b.prediction_score) > 1e-9) {
            return a.prediction_score > b.prediction_score;
        }
        if (a.prediction_rank != b.prediction_rank) {
            return a.prediction_rank < b.prediction_rank;
        }
        if (a.insertion_sequence != b.insertion_sequence) {
            return a.insertion_sequence < b.insertion_sequence;
        }
        return a.expert_id < b.expert_id;
    }
};

/**
 * Metric tracking structure for queue admission, occupancy, and reconciliation.
 */
struct QueueMetrics {
    int total_candidates_received = 0;
    int candidates_admitted = 0;
    int duplicates_suppressed = 0;
    int invalid_rejected = 0;
    int capacity_rejected = 0;
    int overflow_evicted = 0;
    int reconciled_hits = 0;
    int expired_misses = 0;
    int candidates_cancelled = 0;
    int candidates_selected = 0;
    int queue_high_water_mark = 0;
    int queue_sample_count = 0;
    uint64_t sum_queue_occupancy = 0;
    double total_selection_latency_ns = 0.0;
    int selection_call_count = 0;

    double mean_queue_occupancy() const {
        if (queue_sample_count > 0) {
            return static_cast<double>(sum_queue_occupancy) / static_cast<double>(queue_sample_count);
        }
        return 0.0;
    }

    double mean_selection_latency_ns() const {
        if (selection_call_count > 0) {
            return total_selection_latency_ns / static_cast<double>(selection_call_count);
        }
        return 0.0;
    }
};

/**
 * PrefetchCandidateQueue manages candidate entry priority, duplicate suppression,
 * capacity limits, lifecycle transitions, and target-stage reconciliation.
 */
class PrefetchCandidateQueue {
public:
    using Ptr = std::shared_ptr<PrefetchCandidateQueue>;

    PrefetchCandidateQueue(
        const PrefetchConfig& config,
        int num_stages,
        int num_experts,
        const std::string& output_dir = "results/phase_3_prefetch_scheduling");

    ~PrefetchCandidateQueue();

    // Ingestion & Admission
    bool enqueue_candidate(const PrefetchCandidate& candidate);
    int enqueue_candidates(
        int request_id,
        int token_id,
        int source_stage_id,
        int target_stage_id,
        int horizon,
        const PredictionResult& result,
        int seq_id = 0,
        double sim_time_ns = 0.0);

    // Target-Stage Reconciliation (Invoked upon authoritative routing resolution)
    int reconcile_stage(
        int request_id,
        int token_id,
        int target_stage_id,
        const std::vector<int>& actual_target_experts);

    // Request Cleanup
    int purge_request(int request_id);

    // Future Phase 4 Transfer Interface
    std::vector<PrefetchCandidate> select_ready_candidates(int max_count);
    bool mark_candidate_dispatched(uint64_t candidate_id, double dispatch_time_ns = -1.0);

    // Finalization & Diagnostics
    void finalize(const std::string& custom_dir = "");

    size_t size() const;
    bool empty() const;
    int get_capacity() const;
    const QueueMetrics& get_metrics() const;
    std::string get_csv_path() const;
    std::string get_summary_path() const;

private:
    PrefetchConfig config_;
    int num_stages_;
    int num_experts_;
    int capacity_;
    PrefetchPriorityPolicy priority_policy_;
    DuplicatePolicy duplicate_policy_;
    OverflowPolicy overflow_policy_;
    std::string output_dir_;
    std::string csv_log_path_;
    std::string summary_json_path_;

    std::ofstream csv_file_;
    QueueMetrics metrics_;

    uint64_t next_candidate_id_ = 1;
    uint64_t next_insertion_seq_ = 1;
    bool finalized_ = false;

    // Active candidates indexed by composite key
    std::unordered_map<CandidateKey, PrefetchCandidate, CandidateKeyHash> active_candidates_;

    // Fast mapping from stage composite (request_id, token_id, target_stage_id) to candidate keys
    static uint64_t make_stage_key(int req, int tok, int stage) {
        return (static_cast<uint64_t>(req) << 36) |
               ((static_cast<uint64_t>(tok) & 0x3FFFFULL) << 18) |
               (static_cast<uint64_t>(stage) & 0x3FFFFULL);
    }
    std::unordered_map<uint64_t, std::vector<CandidateKey>> stage_index_;

    // Fast mapping from request_id to candidate keys
    std::unordered_map<int, std::vector<CandidateKey>> request_index_;

    void log_event(const PrefetchCandidate& cand, const std::string& event_type);
    void update_occupancy_sample();
    bool validate_candidate(const PrefetchCandidate& cand);
};

} // namespace llm_system

