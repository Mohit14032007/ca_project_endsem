#pragma once

#include <string>
#include <stdexcept>

namespace YAML {
class Node;
}

namespace llm_system {

/**
 * Operating mode of the prefetching engine.
 */
enum class PrefetchMode {
    NONE,               // No prediction or prefetching (baseline behavior)
    PREDICTION_ONLY,    // Predictor executes and logs metrics; zero memory transfers issued
    HARDWARE_SYNC,      // Future experimental mode: blocking transfer simulation
    HARDWARE_ASYNC      // Future experimental mode: event-driven asynchronous transfer engine
};

/**
 * Recognized expert predictor identifiers.
 */
enum class PredictorType {
    NONE,               // No-op predictor (Phase 1 default)
    ST_MOE,             // Spatio-Temporal MoE predictor (Zhao et al.)
    CSTP,               // Confidence- and Frequency-Aware Spatio-Temporal Predictor
    ACSH_GLOBAL,        // Adaptive Confidence & Statistical Hybrid (Global EMA)
    ACSH_STAGE_AWARE,   // Adaptive Confidence & Statistical Hybrid (Per-Stage EMA)
    ACSH_FAST_RECOVERY, // Adaptive Confidence & Statistical Hybrid (Fast Cooldown)
    FREQUENCY,          // Empirical expert frequency baseline
    TEMPORAL,           // Temporal-only history baseline
    SPATIAL,            // Spatial-only transition baseline
    RANDOM,             // Uniform random candidate baseline
    ORACLE              // Theoretical upper-bound evaluation oracle (non-deployable)
};

/**
 * Candidate expert selection policies.
 */
enum class CandidateSelectionPolicy {
    TOP_K,              // Select top-B highest scored candidates
    THRESHOLD_FILTERED, // Select candidates exceeding a confidence score threshold
    ADAPTIVE            // Dynamically sized candidate set based on score distribution
};

/**
 * HBM expert cache eviction policies.
 */
enum class EvictionPolicy {
    LRU,                // Standard Least Recently Used eviction (baseline)
    LRU_PINNED,         // LRU with pin protection for in-flight/active experts
    PREFETCH_AWARE,     // Prefetch-guided eviction policy
    PABI,               // Predictor-Aware Buffer Insertion (Phase 4D.3)
    FA_LRU              // Frequency-Aware LRU with bottom-W window and lazy decay (Point 4)
};

/**
 * Memory transfer simulation models.
 */
enum class TransferModel {
    EVENT_DRIVEN,       // Calibrated event-driven transfer latency model
    RAMULATOR_FULL,     // Cycle-by-cycle Ramulator2 execution
    FIXED_LATENCY       // Static analytical latency per expert transfer
};

/**
 * Prefetch candidate priority policies.
 */
enum class PrefetchPriorityPolicy {
    DEADLINE_SCORE_RANK, // Deadline urgency first, then score, rank, insertion, expert_id
    DEADLINE_ONLY,       // Deadline urgency only, then insertion, expert_id
    SCORE_ONLY           // Score only, then rank, insertion, expert_id
};

/**
 * Candidate duplicate policies.
 */
enum class DuplicatePolicy {
    RETAIN_HIGHEST_PRIORITY, // Retain highest score / priority candidate
    REJECT_DUPLICATE         // Reject new duplicate immediately
};

/**
 * Queue capacity overflow policies.
 */
enum class OverflowPolicy {
    REJECT_LOWEST_PRIORITY,  // Evict lowest priority candidate if new candidate has higher priority
    REJECT_NEW               // Reject new candidate when queue is full
};

/**
 * Temporal scheduling mode of prefetch candidates.
 */
enum class SchedulingMode {
    LEGACY_CURRENT_STAGE, // Legacy co-located dispatch (dispatches at stage L for stage L using historical L-H)
    FORWARD_LOOKAHEAD     // Phase 2C forward-lookahead (uses routing of stage L to predict stage L+H)
};

// String conversion helpers
std::string to_string(PrefetchMode mode);
std::string to_string(PredictorType type);
std::string to_string(CandidateSelectionPolicy policy);
std::string to_string(EvictionPolicy policy);
std::string to_string(TransferModel model);
std::string to_string(PrefetchPriorityPolicy policy);
std::string to_string(DuplicatePolicy policy);
std::string to_string(OverflowPolicy policy);
std::string to_string(SchedulingMode mode);

// String parsing helpers (throw std::invalid_argument on unrecognized input)
PrefetchMode parse_prefetch_mode(const std::string& str);
PredictorType parse_predictor_type(const std::string& str);
CandidateSelectionPolicy parse_candidate_selection_policy(const std::string& str);
EvictionPolicy parse_eviction_policy(const std::string& str);
TransferModel parse_transfer_model(const std::string& str);
PrefetchPriorityPolicy parse_priority_policy(const std::string& str);
DuplicatePolicy parse_duplicate_policy(const std::string& str);
OverflowPolicy parse_overflow_policy(const std::string& str);
SchedulingMode parse_scheduling_mode(const std::string& str);

/**
 * Comprehensive configuration structure for the prefetching framework.
 */
struct PrefetchConfig {
    // Global toggle
    bool enabled = false;

    // Operating mode
    PrefetchMode mode = PrefetchMode::NONE;

    // Predictor configuration
    PredictorType predictor_type = PredictorType::NONE;
    double alpha = 0.5;                  // CSTP spatial/temporal balance in [0.0, 1.0]
    double smoothing_lambda = 1.0;       // Dirichlet prior smoothing constant (> 0.0)
    int confidence_threshold = 2;        // ST-MoE saturating counter threshold (>= 0)
    bool online_adaptation = true;       // Dynamic update of prediction tables

    // ACSH hybrid parameters
    double acsh_beta = 0.8;              // EMA momentum for reliability tracking
    double acsh_tau = 0.2;               // Temperature for softmax reliability weighting
    double acsh_temp_c = 1.0;            // Softmax temperature for CSTP component scores
    double acsh_temp_m = 1.0;            // Softmax temperature for ST-MoE component scores
    double acsh_initial_reliability = 0.5;// Initial prior reliability value
    int acsh_shift_window = 3;           // Fast recovery error window length
    double acsh_shift_threshold = 0.75;  // Fast recovery error trigger threshold
    double acsh_fast_beta = 0.2;         // Fast recovery acceleration momentum
    int acsh_fast_cooldown_steps = 3;    // Fast recovery cooldown steps

    // Random predictor seed
    unsigned int random_seed = 42;

    // Candidate selection configuration
    CandidateSelectionPolicy candidate_selection_policy = CandidateSelectionPolicy::TOP_K;
    int budget = 4;                      // Maximum prefetch candidate budget B (>= 0)

    // Scheduling configuration
    SchedulingMode scheduling_mode = SchedulingMode::LEGACY_CURRENT_STAGE; // Default to legacy co-located mode for backward compatibility
    int prediction_horizon = 1;          // Inter-layer prediction horizon H (>= 1)
    int lookahead_tokens = 0;            // Inter-token lookahead (>= 0)
    bool demand_priority = true;         // Demand misses take precedence over prefetch requests
    bool queue_enabled = true;           // Prefetch candidate queue enabled
    int queue_capacity = 1024;           // Maximum candidate queue capacity (>= 1)
    PrefetchPriorityPolicy priority_policy = PrefetchPriorityPolicy::DEADLINE_SCORE_RANK;
    DuplicatePolicy duplicate_policy = DuplicatePolicy::RETAIN_HIGHEST_PRIORITY;
    OverflowPolicy overflow_policy = OverflowPolicy::REJECT_LOWEST_PRIORITY;

    // Memory transfer configuration
    TransferModel transfer_model = TransferModel::EVENT_DRIVEN;
    bool asynchronous = true;            // Overlap transfer latency with computation
    int max_outstanding_transfers = 4;   // Queue capacity for concurrent transfers (>= 1)
    std::string calibration_file = "";   // Path to Ramulator transfer calibration data

    // HBM cache configuration
    EvictionPolicy eviction_policy = EvictionPolicy::LRU;
    long long reserved_prefetch_bytes = 0; // Dedicated prefetch buffer capacity (>= 0)
    int fa_lru_window_size = 16;           // Bottom-W LRU candidate window size (>= 1)

    // PABI Configuration (Phase 4D.3)
    bool pabi_quarantine_enabled = true;
    long long pabi_speculative_buffer_capacity_bytes = 16000000000LL; // 16 GB default (20% of 80 GB)
    double pabi_speculative_fraction = 0.20; // 20% default fraction of total HBM capacity

    // PABI Pinning Configuration (Phase 4D.4)
    bool pabi_pinning_enabled = false;
    int pabi_max_pinned_experts = 4;
    std::string pabi_pin_selection_policy = "top_ranked";
    std::string pabi_pin_expiration_policy = "stage_deadline";

    // Validation
    void validate() const;

    // Parsing from YAML
    static PrefetchConfig parse_yaml(const YAML::Node& node);
};

} // namespace llm_system
