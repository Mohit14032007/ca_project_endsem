#pragma once

#include "prefetch/predictor_interface.h"

namespace llm_system {

/**
 * Baseline no-op predictor implementation.
 * Used when prefetching is disabled or in baseline evaluation mode.
 */
class NonePredictor : public IPredictor {
public:
    NonePredictor() = default;
    ~NonePredictor() override = default;

    std::string get_name() const override { return "none"; }
    PredictorType get_type() const override { return PredictorType::NONE; }
    bool is_implemented() const override { return true; }
    bool is_evaluation_only() const override { return false; }

    void reset(int request_id) override;

    void update(
        int request_id,
        int token_id,
        int source_stage_id,
        int target_stage_id,
        const std::vector<int>& source_experts,
        const std::vector<int>& actual_target_experts) override;

    PredictionResult predict(
        int request_id,
        int token_id,
        int source_stage_id,
        int target_stage_id,
        const std::vector<int>& source_experts,
        int budget) override;
};

} // namespace llm_system
