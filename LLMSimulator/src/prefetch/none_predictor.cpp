#include "prefetch/none_predictor.h"

namespace llm_system {

void NonePredictor::reset(int request_id) {
    (void)request_id;
    // Stateless: no per-request state to clear
}

void NonePredictor::update(
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
    // Baseline: routing decisions are observed but not recorded
}

PredictionResult NonePredictor::predict(
    int request_id,
    int token_id,
    int source_stage_id,
    int target_stage_id,
    const std::vector<int>& source_experts,
    int budget) {
    (void)source_experts;
    (void)budget;

    PredictionResult result;
    result.request_id = request_id;
    result.token_id = token_id;
    result.source_stage = source_stage_id;
    result.current_stage = source_stage_id;
    result.target_stage = target_stage_id;
    result.horizon = target_stage_id - source_stage_id;
    result.is_valid = true;
    result.candidate_experts.clear();
    result.scores.clear();

    return result;
}

} // namespace llm_system
