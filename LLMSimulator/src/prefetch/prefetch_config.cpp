#include "prefetch/prefetch_config.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <yaml-cpp/yaml.h>

namespace llm_system {

static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return s;
}

std::string to_string(PrefetchMode mode) {
    switch (mode) {
        case PrefetchMode::NONE: return "none";
        case PrefetchMode::PREDICTION_ONLY: return "prediction_only";
        case PrefetchMode::HARDWARE_SYNC: return "hardware_sync";
        case PrefetchMode::HARDWARE_ASYNC: return "hardware_async";
        default: return "unknown";
    }
}

std::string to_string(PredictorType type) {
    switch (type) {
        case PredictorType::NONE: return "none";
        case PredictorType::ST_MOE: return "st_moe";
        case PredictorType::CSTP: return "cstp";
        case PredictorType::ACSH_GLOBAL: return "acsh_global";
        case PredictorType::ACSH_STAGE_AWARE: return "acsh_stage_aware";
        case PredictorType::ACSH_FAST_RECOVERY: return "acsh_fast_recovery";
        case PredictorType::FREQUENCY: return "frequency";
        case PredictorType::TEMPORAL: return "temporal";
        case PredictorType::SPATIAL: return "spatial";
        case PredictorType::RANDOM: return "random";
        case PredictorType::ORACLE: return "oracle";
        default: return "unknown";
    }
}

std::string to_string(CandidateSelectionPolicy policy) {
    switch (policy) {
        case CandidateSelectionPolicy::TOP_K: return "top_k";
        case CandidateSelectionPolicy::THRESHOLD_FILTERED: return "threshold_filtered";
        case CandidateSelectionPolicy::ADAPTIVE: return "adaptive";
        default: return "unknown";
    }
}

std::string to_string(EvictionPolicy policy) {
    switch (policy) {
        case EvictionPolicy::LRU: return "lru";
        case EvictionPolicy::LRU_PINNED: return "lru_pinned";
        case EvictionPolicy::PREFETCH_AWARE: return "prefetch_aware";
        case EvictionPolicy::PABI: return "pabi";
        case EvictionPolicy::FA_LRU: return "fa_lru";
        default: return "unknown";
    }
}

std::string to_string(TransferModel model) {
    switch (model) {
        case TransferModel::EVENT_DRIVEN: return "event_driven";
        case TransferModel::RAMULATOR_FULL: return "ramulator_full";
        case TransferModel::FIXED_LATENCY: return "fixed_latency";
        default: return "unknown";
    }
}

std::string to_string(PrefetchPriorityPolicy policy) {
    switch (policy) {
        case PrefetchPriorityPolicy::DEADLINE_SCORE_RANK: return "deadline_score_rank";
        case PrefetchPriorityPolicy::DEADLINE_ONLY: return "deadline_only";
        case PrefetchPriorityPolicy::SCORE_ONLY: return "score_only";
        default: return "unknown";
    }
}

std::string to_string(DuplicatePolicy policy) {
    switch (policy) {
        case DuplicatePolicy::RETAIN_HIGHEST_PRIORITY: return "retain_highest_priority";
        case DuplicatePolicy::REJECT_DUPLICATE: return "reject_duplicate";
        default: return "unknown";
    }
}

std::string to_string(OverflowPolicy policy) {
    switch (policy) {
        case OverflowPolicy::REJECT_LOWEST_PRIORITY: return "reject_lowest_priority";
        case OverflowPolicy::REJECT_NEW: return "reject_new";
        default: return "unknown";
    }
}

std::string to_string(SchedulingMode mode) {
    if (mode == SchedulingMode::FORWARD_LOOKAHEAD) {
        return "forward_lookahead";
    }
    if (mode == SchedulingMode::LEGACY_CURRENT_STAGE) {
        return "legacy_current_stage";
    }
    return "unknown";
}

SchedulingMode parse_scheduling_mode(const std::string& str) {
    std::string s = to_lower(str);
    if (s == "forward_lookahead" || s == "forward" || s == "lookahead") {
        return SchedulingMode::FORWARD_LOOKAHEAD;
    }
    if (s == "legacy_current_stage" || s == "legacy" || s == "co_located" || s == "current_stage") {
        return SchedulingMode::LEGACY_CURRENT_STAGE;
    }
    throw std::invalid_argument("[CONFIG_ERROR] Unrecognized scheduling mode: '" + str + 
                                "'. Supported modes: ['forward_lookahead', 'legacy_current_stage']");
}

PrefetchMode parse_prefetch_mode(const std::string& str) {
    std::string s = to_lower(str);
    if (s == "none") return PrefetchMode::NONE;
    if (s == "prediction_only") return PrefetchMode::PREDICTION_ONLY;
    if (s == "hardware_sync") return PrefetchMode::HARDWARE_SYNC;
    if (s == "hardware_async") return PrefetchMode::HARDWARE_ASYNC;
    throw std::invalid_argument("[CONFIG_ERROR] Unrecognized prefetch mode: '" + str + 
                                "'. Supported modes: ['none', 'prediction_only', 'hardware_sync', 'hardware_async']");
}

PredictorType parse_predictor_type(const std::string& str) {
    std::string s = to_lower(str);
    if (s == "none") return PredictorType::NONE;
    if (s == "st_moe") return PredictorType::ST_MOE;
    if (s == "cstp") return PredictorType::CSTP;
    if (s == "acsh_global") return PredictorType::ACSH_GLOBAL;
    if (s == "acsh_stage_aware") return PredictorType::ACSH_STAGE_AWARE;
    if (s == "acsh_fast_recovery") return PredictorType::ACSH_FAST_RECOVERY;
    if (s == "frequency") return PredictorType::FREQUENCY;
    if (s == "temporal") return PredictorType::TEMPORAL;
    if (s == "spatial") return PredictorType::SPATIAL;
    if (s == "random") return PredictorType::RANDOM;
    if (s == "oracle") return PredictorType::ORACLE;
    throw std::invalid_argument("[CONFIG_ERROR] Unrecognized predictor type: '" + str + 
                                "'. Supported types: ['none', 'st_moe', 'cstp', 'acsh_global', 'acsh_stage_aware', "
                                "'acsh_fast_recovery', 'frequency', 'temporal', 'spatial', 'random', 'oracle']");
}

CandidateSelectionPolicy parse_candidate_selection_policy(const std::string& str) {
    std::string s = to_lower(str);
    if (s == "top_k") return CandidateSelectionPolicy::TOP_K;
    if (s == "threshold_filtered") return CandidateSelectionPolicy::THRESHOLD_FILTERED;
    if (s == "adaptive") return CandidateSelectionPolicy::ADAPTIVE;
    throw std::invalid_argument("[CONFIG_ERROR] Unrecognized candidate selection policy: '" + str + 
                                "'. Supported policies: ['top_k', 'threshold_filtered', 'adaptive']");
}

EvictionPolicy parse_eviction_policy(const std::string& str) {
    std::string s = to_lower(str);
    if (s == "lru") return EvictionPolicy::LRU;
    if (s == "lru_pinned") return EvictionPolicy::LRU_PINNED;
    if (s == "prefetch_aware") return EvictionPolicy::PREFETCH_AWARE;
    if (s == "pabi") return EvictionPolicy::PABI;
    if (s == "fa_lru" || s == "falru" || s == "frequency_aware_lru") return EvictionPolicy::FA_LRU;
    throw std::invalid_argument("[CONFIG_ERROR] Unrecognized eviction policy: '" + str + 
                                "'. Supported policies: ['lru', 'lru_pinned', 'prefetch_aware', 'pabi', 'fa_lru']");
}

TransferModel parse_transfer_model(const std::string& str) {
    std::string s = to_lower(str);
    if (s == "event_driven") return TransferModel::EVENT_DRIVEN;
    if (s == "ramulator_full") return TransferModel::RAMULATOR_FULL;
    if (s == "fixed_latency") return TransferModel::FIXED_LATENCY;
    throw std::invalid_argument("[CONFIG_ERROR] Unrecognized transfer model: '" + str + 
                                "'. Supported models: ['event_driven', 'ramulator_full', 'fixed_latency']");
}

PrefetchPriorityPolicy parse_priority_policy(const std::string& str) {
    std::string s = to_lower(str);
    if (s == "deadline_score_rank") return PrefetchPriorityPolicy::DEADLINE_SCORE_RANK;
    if (s == "deadline_only") return PrefetchPriorityPolicy::DEADLINE_ONLY;
    if (s == "score_only") return PrefetchPriorityPolicy::SCORE_ONLY;
    throw std::invalid_argument("[CONFIG_ERROR] Unrecognized priority policy: '" + str + 
                                "'. Supported policies: ['deadline_score_rank', 'deadline_only', 'score_only']");
}

DuplicatePolicy parse_duplicate_policy(const std::string& str) {
    std::string s = to_lower(str);
    if (s == "retain_highest_priority") return DuplicatePolicy::RETAIN_HIGHEST_PRIORITY;
    if (s == "reject_duplicate") return DuplicatePolicy::REJECT_DUPLICATE;
    throw std::invalid_argument("[CONFIG_ERROR] Unrecognized duplicate policy: '" + str + 
                                "'. Supported policies: ['retain_highest_priority', 'reject_duplicate']");
}

OverflowPolicy parse_overflow_policy(const std::string& str) {
    std::string s = to_lower(str);
    if (s == "reject_lowest_priority") return OverflowPolicy::REJECT_LOWEST_PRIORITY;
    if (s == "reject_new") return OverflowPolicy::REJECT_NEW;
    throw std::invalid_argument("[CONFIG_ERROR] Unrecognized overflow policy: '" + str + 
                                "'. Supported policies: ['reject_lowest_priority', 'reject_new']");
}

void PrefetchConfig::validate() const {
    // Numeric bounds validation
    if (budget < 0) {
        throw std::invalid_argument("[CONFIG_ERROR] candidate_selection.budget cannot be negative, got " + std::to_string(budget));
    }
    if (prediction_horizon < 1) {
        throw std::invalid_argument("[CONFIG_ERROR] scheduling.prediction_horizon must be >= 1, got " + std::to_string(prediction_horizon));
    }
    if (lookahead_tokens < 0) {
        throw std::invalid_argument("[CONFIG_ERROR] scheduling.lookahead_tokens cannot be negative, got " + std::to_string(lookahead_tokens));
    }
    if (queue_capacity <= 0) {
        throw std::invalid_argument("[CONFIG_ERROR] scheduling.queue_capacity must be > 0, got " + std::to_string(queue_capacity));
    }
    if (alpha < 0.0 || alpha > 1.0) {
        throw std::invalid_argument("[CONFIG_ERROR] predictor.alpha must be in [0.0, 1.0], got " + std::to_string(alpha));
    }
    if (smoothing_lambda <= 0.0) {
        throw std::invalid_argument("[CONFIG_ERROR] predictor.smoothing_lambda must be > 0.0, got " + std::to_string(smoothing_lambda));
    }
    if (confidence_threshold < 0) {
        throw std::invalid_argument("[CONFIG_ERROR] predictor.confidence_threshold cannot be negative, got " + std::to_string(confidence_threshold));
    }
    if (max_outstanding_transfers <= 0) {
        throw std::invalid_argument("[CONFIG_ERROR] transfer.max_outstanding_transfers must be >= 1, got " + std::to_string(max_outstanding_transfers));
    }
    if (reserved_prefetch_bytes < 0) {
        throw std::invalid_argument("[CONFIG_ERROR] hbm.reserved_prefetch_bytes cannot be negative, got " + std::to_string(reserved_prefetch_bytes));
    }
    if (fa_lru_window_size < 1) {
        throw std::invalid_argument("[CONFIG_ERROR] fa_lru_window_size must be >= 1, got " + std::to_string(fa_lru_window_size));
    }
    if (pabi_speculative_buffer_capacity_bytes < 0) {
        throw std::invalid_argument("[CONFIG_ERROR] pabi.speculative_buffer_capacity_bytes cannot be negative, got " + std::to_string(pabi_speculative_buffer_capacity_bytes));
    }
    if (pabi_speculative_fraction < 0.0 || pabi_speculative_fraction > 1.0) {
        throw std::invalid_argument("[CONFIG_ERROR] pabi.speculative_fraction must be in [0.0, 1.0], got " + std::to_string(pabi_speculative_fraction));
    }
    if (pabi_max_pinned_experts < 0) {
        throw std::invalid_argument("[CONFIG_ERROR] pabi.max_pinned_experts cannot be negative, got " + std::to_string(pabi_max_pinned_experts));
    }
    if (pabi_pin_selection_policy != "top_ranked") {
        throw std::invalid_argument("[CONFIG_ERROR] unrecognized pabi.pin_selection_policy: '" + pabi_pin_selection_policy + "', must be 'top_ranked'");
    }
    if (pabi_pin_expiration_policy != "stage_deadline") {
        throw std::invalid_argument("[CONFIG_ERROR] unrecognized pabi.pin_expiration_policy: '" + pabi_pin_expiration_policy + "', must be 'stage_deadline'");
    }

    // Logical mode and consistency validation
    if (!enabled) {
        if (mode != PrefetchMode::NONE) {
            throw std::invalid_argument(
                "[CONFIG_ERROR] Prefetching is disabled (enabled: false), but mode is '" + 
                to_string(mode) + "'. Set enabled: true to activate prefetching, or set mode: 'none'.");
        }
        if (predictor_type != PredictorType::NONE) {
            throw std::invalid_argument(
                "[CONFIG_ERROR] Prefetching is disabled (enabled: false), but predictor.type is '" + 
                to_string(predictor_type) + "'. Set enabled: true to activate predictor, or set predictor.type: 'none'.");
        }
        return; // Disabled configuration is valid
    }

    // If enabled, mode must be active
    if (mode == PrefetchMode::NONE) {
        throw std::invalid_argument(
            "[CONFIG_ERROR] Prefetching is enabled (enabled: true), but mode is set to 'none'. "
            "Please select an active operating mode: ['prediction_only', 'hardware_sync', 'hardware_async']");
    }

    // Oracle predictor safety check: oracle is strictly an evaluation-only upper bound
    if (predictor_type == PredictorType::ORACLE && mode != PrefetchMode::PREDICTION_ONLY) {
        throw std::invalid_argument(
            "[CONFIG_ERROR] Predictor 'oracle' can only be executed in 'prediction_only' mode. "
            "It cannot be combined with hardware prefetching modes ('hardware_sync', 'hardware_async') "
            "because it represents a non-causal theoretical upper bound.");
    }
}

PrefetchConfig PrefetchConfig::parse_yaml(const YAML::Node& node) {
    PrefetchConfig config;
    if (!node || node.IsNull()) {
        return config; // Return disabled default
    }

    if (node["enabled"]) {
        config.enabled = node["enabled"].as<bool>();
    }

    if (node["mode"]) {
        config.mode = parse_prefetch_mode(node["mode"].as<std::string>());
    }

    if (node["predictor"]) {
        const auto& p_node = node["predictor"];
        if (p_node["type"]) {
            config.predictor_type = parse_predictor_type(p_node["type"].as<std::string>());
        }
        if (p_node["alpha"]) {
            config.alpha = p_node["alpha"].as<double>();
        }
        if (p_node["smoothing_lambda"]) {
            config.smoothing_lambda = p_node["smoothing_lambda"].as<double>();
        }
        if (p_node["confidence_threshold"]) {
            config.confidence_threshold = p_node["confidence_threshold"].as<int>();
        }
        if (p_node["online_adaptation"]) {
            config.online_adaptation = p_node["online_adaptation"].as<bool>();
        }
        if (p_node["beta"]) {
            config.acsh_beta = p_node["beta"].as<double>();
        }
        if (p_node["tau"]) {
            config.acsh_tau = p_node["tau"].as<double>();
        }
        if (p_node["temp_c"]) {
            config.acsh_temp_c = p_node["temp_c"].as<double>();
        }
        if (p_node["temp_m"]) {
            config.acsh_temp_m = p_node["temp_m"].as<double>();
        }
        if (p_node["initial_reliability"]) {
            config.acsh_initial_reliability = p_node["initial_reliability"].as<double>();
        }
        if (p_node["shift_window"]) {
            config.acsh_shift_window = p_node["shift_window"].as<int>();
        }
        if (p_node["shift_threshold"]) {
            config.acsh_shift_threshold = p_node["shift_threshold"].as<double>();
        }
        if (p_node["fast_beta"]) {
            config.acsh_fast_beta = p_node["fast_beta"].as<double>();
        }
        if (p_node["fast_cooldown_steps"]) {
            config.acsh_fast_cooldown_steps = p_node["fast_cooldown_steps"].as<int>();
        }
        if (p_node["random_seed"]) {
            config.random_seed = p_node["random_seed"].as<unsigned int>();
        }
        if (p_node["budget"]) {
            config.budget = p_node["budget"].as<int>();
        }
        if (p_node["prediction_horizon"]) {
            config.prediction_horizon = p_node["prediction_horizon"].as<int>();
        }
    }

    if (node["candidate_selection"]) {
        const auto& cs_node = node["candidate_selection"];
        if (cs_node["policy"]) {
            config.candidate_selection_policy = parse_candidate_selection_policy(cs_node["policy"].as<std::string>());
        }
        if (cs_node["budget"]) {
            config.budget = cs_node["budget"].as<int>();
        }
    }

    if (node["scheduling"]) {
        const auto& s_node = node["scheduling"];
        if (s_node["prediction_horizon"]) {
            config.prediction_horizon = s_node["prediction_horizon"].as<int>();
        }
        if (s_node["scheduling_mode"]) {
            config.scheduling_mode = parse_scheduling_mode(s_node["scheduling_mode"].as<std::string>());
        }
        if (s_node["lookahead_tokens"]) {
            config.lookahead_tokens = s_node["lookahead_tokens"].as<int>();
        }
        if (s_node["demand_priority"]) {
            config.demand_priority = s_node["demand_priority"].as<bool>();
        }
        if (s_node["queue_enabled"]) {
            config.queue_enabled = s_node["queue_enabled"].as<bool>();
        }
        if (s_node["queue_capacity"]) {
            config.queue_capacity = s_node["queue_capacity"].as<int>();
        }
        if (s_node["priority_policy"]) {
            config.priority_policy = parse_priority_policy(s_node["priority_policy"].as<std::string>());
        }
        if (s_node["duplicate_policy"]) {
            config.duplicate_policy = parse_duplicate_policy(s_node["duplicate_policy"].as<std::string>());
        }
        if (s_node["overflow_policy"]) {
            config.overflow_policy = parse_overflow_policy(s_node["overflow_policy"].as<std::string>());
        }
    }

    if (node["transfer"]) {
        const auto& t_node = node["transfer"];
        if (t_node["model"]) {
            config.transfer_model = parse_transfer_model(t_node["model"].as<std::string>());
        }
        if (t_node["asynchronous"]) {
            config.asynchronous = t_node["asynchronous"].as<bool>();
        }
        if (t_node["max_outstanding_transfers"]) {
            config.max_outstanding_transfers = t_node["max_outstanding_transfers"].as<int>();
        }
        if (t_node["calibration_file"]) {
            config.calibration_file = t_node["calibration_file"].as<std::string>();
        }
    }

    if (node["eviction_policy"]) {
        config.eviction_policy = parse_eviction_policy(node["eviction_policy"].as<std::string>());
    }

    if (node["fa_lru_window_size"]) {
        config.fa_lru_window_size = node["fa_lru_window_size"].as<int>();
    }

    if (node["pabi"]) {
        const auto& pabi_node = node["pabi"];
        if (pabi_node["quarantine_enabled"]) {
            config.pabi_quarantine_enabled = pabi_node["quarantine_enabled"].as<bool>();
        }
        if (pabi_node["speculative_buffer_capacity_bytes"]) {
            config.pabi_speculative_buffer_capacity_bytes = pabi_node["speculative_buffer_capacity_bytes"].as<long long>();
        }
        if (pabi_node["speculative_fraction"]) {
            config.pabi_speculative_fraction = pabi_node["speculative_fraction"].as<double>();
        }
        if (pabi_node["pinning_enabled"]) {
            config.pabi_pinning_enabled = pabi_node["pinning_enabled"].as<bool>();
        }
        if (pabi_node["max_pinned_experts"]) {
            config.pabi_max_pinned_experts = pabi_node["max_pinned_experts"].as<int>();
        }
        if (pabi_node["pin_selection_policy"]) {
            config.pabi_pin_selection_policy = pabi_node["pin_selection_policy"].as<std::string>();
        }
        if (pabi_node["pin_expiration_policy"]) {
            config.pabi_pin_expiration_policy = pabi_node["pin_expiration_policy"].as<std::string>();
        }
    }

    if (node["hbm"]) {
        const auto& h_node = node["hbm"];
        if (h_node["eviction_policy"]) {
            config.eviction_policy = parse_eviction_policy(h_node["eviction_policy"].as<std::string>());
        }
        if (h_node["reserved_prefetch_bytes"]) {
            config.reserved_prefetch_bytes = h_node["reserved_prefetch_bytes"].as<long long>();
        }
        if (h_node["fa_lru_window_size"]) {
            config.fa_lru_window_size = h_node["fa_lru_window_size"].as<int>();
        }
        if (h_node["pabi"]) {
            const auto& pabi_node = h_node["pabi"];
            if (pabi_node["quarantine_enabled"]) {
                config.pabi_quarantine_enabled = pabi_node["quarantine_enabled"].as<bool>();
            }
            if (pabi_node["speculative_buffer_capacity_bytes"]) {
                config.pabi_speculative_buffer_capacity_bytes = pabi_node["speculative_buffer_capacity_bytes"].as<long long>();
            }
            if (pabi_node["speculative_fraction"]) {
                config.pabi_speculative_fraction = pabi_node["speculative_fraction"].as<double>();
            }
            if (pabi_node["pinning_enabled"]) {
                config.pabi_pinning_enabled = pabi_node["pinning_enabled"].as<bool>();
            }
            if (pabi_node["max_pinned_experts"]) {
                config.pabi_max_pinned_experts = pabi_node["max_pinned_experts"].as<int>();
            }
            if (pabi_node["pin_selection_policy"]) {
                config.pabi_pin_selection_policy = pabi_node["pin_selection_policy"].as<std::string>();
            }
            if (pabi_node["pin_expiration_policy"]) {
                config.pabi_pin_expiration_policy = pabi_node["pin_expiration_policy"].as<std::string>();
            }
        }
    }

    const char* env_sched_mode = std::getenv("PREFETCH_SCHEDULING_MODE");
    if (env_sched_mode != nullptr) {
        config.scheduling_mode = parse_scheduling_mode(std::string(env_sched_mode));
    }

    const char* env_horizon = std::getenv("PREFETCH_PREDICTION_HORIZON");
    if (env_horizon != nullptr) {
        config.prediction_horizon = std::stoi(std::string(env_horizon));
    }

    const char* env_eviction = std::getenv("EVICTION_POLICY");
    if (env_eviction != nullptr) {
        config.eviction_policy = parse_eviction_policy(std::string(env_eviction));
    }

    const char* env_window = std::getenv("FA_LRU_WINDOW_SIZE");
    if (env_window != nullptr) {
        config.fa_lru_window_size = std::stoi(std::string(env_window));
    }

    return config;
}

} // namespace llm_system
