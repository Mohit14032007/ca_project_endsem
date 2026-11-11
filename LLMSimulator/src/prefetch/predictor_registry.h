#pragma once

#include <memory>
#include <string>

#include "prefetch/prefetch_config.h"
#include "prefetch/predictor_interface.h"

namespace llm_system {

/**
 * Factory and registry for expert predictors.
 * Encapsulates instantiation, availability checks, and evaluation-only restrictions.
 */
class PredictorRegistry {
public:
    /**
     * Creates an IPredictor instance matching the configured PredictorType.
     * Throws std::runtime_error if the predictor is unimplemented, unknown, or invalid.
     */
    static IPredictor::Ptr create_predictor(
        PredictorType type,
        const PrefetchConfig& config,
        int num_stages,
        int num_experts,
        int top_k);

    /**
     * Checks if a string identifier corresponds to a known predictor enum.
     */
    static bool is_known_identifier(const std::string& name);

    /**
     * Checks if a predictor type is implemented and available for instantiation.
     */
    static bool is_implemented(PredictorType type);

    /**
     * Checks if a predictor type is strictly an evaluation-only oracle.
     */
    static bool is_evaluation_only(PredictorType type);
};

} // namespace llm_system

