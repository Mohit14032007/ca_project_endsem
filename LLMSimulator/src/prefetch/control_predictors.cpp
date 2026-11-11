#include "prefetch/control_predictors.h"

#include <algorithm>

namespace llm_system {

// -----------------------------------------------------------------------------
// FrequencyPredictor Implementation
// -----------------------------------------------------------------------------

FrequencyPredictor::FrequencyPredictor(int num_moe_stages, int num_routed_experts)
    : num_moe_stages_(num_moe_stages), num_routed_experts_(num_routed_experts) {
    frequency_counts_.resize(
        num_moe_stages_, std::vector<double>(num_routed_experts_, 0.0));
}

void FrequencyPredictor::reset(int request_id) {
    (void)request_id;
    // Marginal frequency is a global empirical prior; request reset does not clear it
}

void FrequencyPredictor::set_frequency_count(int stage_id, int expert_id, double count) {
    if (stage_id >= 0 && stage_id < num_moe_stages_) {
        if (expert_id >= 0 && expert_id < num_routed_experts_) {
            frequency_counts_[stage_id][expert_id] = count;
        }
    }
}

double FrequencyPredictor::get_frequency_count(int stage_id, int expert_id) const {
    if (stage_id >= 0 && stage_id < num_moe_stages_) {
        if (expert_id >= 0 && expert_id < num_routed_experts_) {
            return frequency_counts_[stage_id][expert_id];
        }
    }
    return 0.0;
}

void FrequencyPredictor::update(
    int request_id,
    int token_id,
    int source_stage_id,
    int target_stage_id,
    const std::vector<int>& source_experts,
    const std::vector<int>& actual_target_experts) {
    (void)request_id;
    (void)token_id;
    (void)source_stage_id;

    if (target_stage_id >= 0 && target_stage_id < num_moe_stages_) {
        for (int v : actual_target_experts) {
            if (v >= 0 && v < num_routed_experts_) {
                frequency_counts_[target_stage_id][v] =
                    frequency_counts_[target_stage_id][v] + 1.0;
            }
        }
    }
}

PredictionResult FrequencyPredictor::predict(
    int request_id,
    int token_id,
    int source_stage_id,
    int target_stage_id,
    const std::vector<int>& source_experts,
    int budget) {
    (void)source_experts;

    PredictionResult result;
    result.request_id = request_id;
    result.token_id = token_id;
    result.source_stage = source_stage_id;
    result.current_stage = source_stage_id;
    result.target_stage = target_stage_id;
    result.horizon = target_stage_id - source_stage_id;

    if (budget <= 0) {
        result.finalize_and_sort(0);
        return result;
    }

    result.scores.reserve(num_routed_experts_);
    if (target_stage_id >= 0 && target_stage_id < num_moe_stages_) {
        const auto& counts = frequency_counts_[target_stage_id];
        for (int v = 0; v < num_routed_experts_; ++v) {
            CandidateScore cs;
            cs.expert_id = v;
            cs.score = counts[v];
            result.scores.push_back(cs);
        }
    } else {
        for (int v = 0; v < num_routed_experts_; ++v) {
            CandidateScore cs;
            cs.expert_id = v;
            cs.score = 0.0;
            result.scores.push_back(cs);
        }
    }

    result.finalize_and_sort(budget);
    return result;
}

// -----------------------------------------------------------------------------
// TemporalOnlyPredictor Implementation (CSTP with alpha = 0.0)
// -----------------------------------------------------------------------------

TemporalOnlyPredictor::TemporalOnlyPredictor(
    int num_moe_stages,
    int num_routed_experts,
    int top_k,
    double smoothing_lambda,
    double prior_epsilon,
    bool online_adaptation) {
    inner_cstp_ = std::make_shared<CSTPPredictor>(
        num_moe_stages,
        num_routed_experts,
        top_k,
        0.0, // alpha = 0.0 enforces pure temporal prediction
        smoothing_lambda,
        prior_epsilon,
        online_adaptation,
        1);
}

void TemporalOnlyPredictor::reset(int request_id) {
    inner_cstp_->reset(request_id);
}

void TemporalOnlyPredictor::update(
    int request_id,
    int token_id,
    int source_stage_id,
    int target_stage_id,
    const std::vector<int>& source_experts,
    const std::vector<int>& actual_target_experts) {
    inner_cstp_->update(
        request_id, token_id, source_stage_id, target_stage_id,
        source_experts, actual_target_experts);
}

PredictionResult TemporalOnlyPredictor::predict(
    int request_id,
    int token_id,
    int source_stage_id,
    int target_stage_id,
    const std::vector<int>& source_experts,
    int budget) {
    PredictionResult res = inner_cstp_->predict(
        request_id, token_id, source_stage_id, target_stage_id,
        source_experts, budget);
    return res;
}

// -----------------------------------------------------------------------------
// SpatialOnlyPredictor Implementation (CSTP with alpha = 1.0)
// -----------------------------------------------------------------------------

SpatialOnlyPredictor::SpatialOnlyPredictor(
    int num_moe_stages,
    int num_routed_experts,
    int top_k,
    double smoothing_lambda,
    double prior_epsilon,
    bool online_adaptation,
    int prediction_horizon) {
    inner_cstp_ = std::make_shared<CSTPPredictor>(
        num_moe_stages,
        num_routed_experts,
        top_k,
        1.0, // alpha = 1.0 enforces pure spatial prediction
        smoothing_lambda,
        prior_epsilon,
        online_adaptation,
        prediction_horizon);
}

void SpatialOnlyPredictor::reset(int request_id) {
    inner_cstp_->reset(request_id);
}

void SpatialOnlyPredictor::update(
    int request_id,
    int token_id,
    int source_stage_id,
    int target_stage_id,
    const std::vector<int>& source_experts,
    const std::vector<int>& actual_target_experts) {
    inner_cstp_->update(
        request_id, token_id, source_stage_id, target_stage_id,
        source_experts, actual_target_experts);
}

PredictionResult SpatialOnlyPredictor::predict(
    int request_id,
    int token_id,
    int source_stage_id,
    int target_stage_id,
    const std::vector<int>& source_experts,
    int budget) {
    PredictionResult res = inner_cstp_->predict(
        request_id, token_id, source_stage_id, target_stage_id,
        source_experts, budget);
    return res;
}

// -----------------------------------------------------------------------------
// RandomPredictor Implementation
// -----------------------------------------------------------------------------

RandomPredictor::RandomPredictor(int num_routed_experts, unsigned int seed)
    : num_routed_experts_(num_routed_experts), initial_seed_(seed), rng_(seed) {
}

void RandomPredictor::set_seed(unsigned int seed) {
    initial_seed_ = seed;
    rng_.seed(seed);
}

void RandomPredictor::reset(int request_id) {
    (void)request_id;
    // Stateless with respect to requests
}

void RandomPredictor::update(
    int request_id,
    int token_id,
    int source_stage_id,
    int target_stage_id,
    const std::vector<int>& source_experts,
    const std::vector<int>& actual_target_experts) {
    (void)request_id;
    (void)token_id;
    (void)source_stage_id;
    (void)target_stage_id;
    (void)source_experts;
    (void)actual_target_experts;
}

PredictionResult RandomPredictor::predict(
    int request_id,
    int token_id,
    int source_stage_id,
    int target_stage_id,
    const std::vector<int>& source_experts,
    int budget) {
    (void)source_experts;

    PredictionResult result;
    result.request_id = request_id;
    result.token_id = token_id;
    result.source_stage = source_stage_id;
    result.current_stage = source_stage_id;
    result.target_stage = target_stage_id;
    result.horizon = target_stage_id - source_stage_id;

    if (budget <= 0) {
        result.finalize_and_sort(0);
        return result;
    }

    // Generate full list of expert candidates
    std::vector<int> all_experts(num_routed_experts_);
    for (int i = 0; i < num_routed_experts_; ++i) {
        all_experts[i] = i;
    }

    // Shuffle deterministically using internal rng
    std::shuffle(all_experts.begin(), all_experts.end(), rng_);

    int effective_budget = budget;
    if (num_routed_experts_ < effective_budget) {
        effective_budget = num_routed_experts_;
    }

    result.scores.reserve(effective_budget);
    for (int i = 0; i < effective_budget; ++i) {
        CandidateScore cs;
        cs.expert_id = all_experts[i];
        // Score assigned based on reverse draw order so finalize_and_sort preserves chosen order
        cs.score = static_cast<double>(effective_budget - i);
        result.scores.push_back(cs);
    }

    result.finalize_and_sort(budget);
    return result;
}

} // namespace llm_system
