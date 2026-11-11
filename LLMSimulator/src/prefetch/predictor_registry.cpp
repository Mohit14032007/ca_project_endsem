#include "prefetch/predictor_registry.h"

#include <stdexcept>

#include "prefetch/acsh_predictor.h"
#include "prefetch/control_predictors.h"
#include "prefetch/cstp_predictor.h"
#include "prefetch/none_predictor.h"
#include "prefetch/st_moe_predictor.h"

namespace llm_system {

bool PredictorRegistry::is_known_identifier(const std::string& name) {
    try {
        parse_predictor_type(name);
        return true;
    } catch (...) {
        return false;
    }
}

bool PredictorRegistry::is_implemented(PredictorType type) {
    switch (type) {
        case PredictorType::NONE:
        case PredictorType::ST_MOE:
        case PredictorType::CSTP:
        case PredictorType::ACSH_GLOBAL:
        case PredictorType::ACSH_STAGE_AWARE:
        case PredictorType::ACSH_FAST_RECOVERY:
        case PredictorType::FREQUENCY:
        case PredictorType::TEMPORAL:
        case PredictorType::SPATIAL:
        case PredictorType::RANDOM:
            return true;

        case PredictorType::ORACLE:
        default:
            return false;
    }
}

bool PredictorRegistry::is_evaluation_only(PredictorType type) {
    return type == PredictorType::ORACLE;
}

IPredictor::Ptr PredictorRegistry::create_predictor(
    PredictorType type,
    const PrefetchConfig& config,
    int num_stages,
    int num_experts,
    int top_k) {
    switch (type) {
        case PredictorType::NONE:
            return std::make_shared<NonePredictor>();

        case PredictorType::ST_MOE:
            return std::make_shared<STMoEPredictor>(
                num_stages,
                num_experts,
                top_k,
                top_k,
                config.confidence_threshold,
                config.online_adaptation,
                config.prediction_horizon);

        case PredictorType::CSTP:
            return std::make_shared<CSTPPredictor>(
                num_stages,
                num_experts,
                top_k,
                config.alpha,
                config.smoothing_lambda,
                0.1,
                config.online_adaptation,
                config.prediction_horizon);

        case PredictorType::ACSH_GLOBAL:
        case PredictorType::ACSH_STAGE_AWARE:
        case PredictorType::ACSH_FAST_RECOVERY: {
            auto cstp_inst = std::make_shared<CSTPPredictor>(
                num_stages,
                num_experts,
                top_k,
                config.alpha,
                config.smoothing_lambda,
                0.1,
                config.online_adaptation,
                config.prediction_horizon);

            auto st_inst = std::make_shared<STMoEPredictor>(
                num_stages,
                num_experts,
                top_k,
                top_k,
                config.confidence_threshold,
                config.online_adaptation,
                config.prediction_horizon);

            return std::make_shared<ACSHPredictor>(
                type,
                cstp_inst,
                st_inst,
                config.acsh_beta,
                config.acsh_tau,
                config.acsh_temp_c,
                config.acsh_temp_m,
                config.acsh_initial_reliability,
                config.acsh_shift_window,
                config.acsh_shift_threshold,
                config.acsh_fast_beta,
                config.acsh_fast_cooldown_steps);
        }

        case PredictorType::FREQUENCY:
            return std::make_shared<FrequencyPredictor>(num_stages, num_experts);

        case PredictorType::TEMPORAL:
            return std::make_shared<TemporalOnlyPredictor>(
                num_stages,
                num_experts,
                top_k,
                config.smoothing_lambda,
                0.1,
                config.online_adaptation);

        case PredictorType::SPATIAL:
            return std::make_shared<SpatialOnlyPredictor>(
                num_stages,
                num_experts,
                top_k,
                config.smoothing_lambda,
                0.1,
                config.online_adaptation,
                config.prediction_horizon);

        case PredictorType::RANDOM:
            return std::make_shared<RandomPredictor>(num_experts, config.random_seed);

        case PredictorType::ORACLE:
            throw std::runtime_error(
                "[PREDICTOR_REGISTRY_ERROR] Predictor 'oracle' is strictly an evaluation-only "
                "theoretical upper bound and is NOT implemented as a deployable prefetcher. "
                "It cannot be instantiated.");

        default:
            throw std::runtime_error("[PREDICTOR_REGISTRY_ERROR] Unknown or unhandled predictor type.");
    }
}

} // namespace llm_system
