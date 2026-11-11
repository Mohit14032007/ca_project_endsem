#pragma once

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "prefetch/cstp_predictor.h"
#include "prefetch/predictor_interface.h"

namespace llm_system {

/**
 * Frequency-Only Control Predictor.
 * Ranks candidate experts strictly according to observed historical occurrence
 * counts at the target MoE stage.
 */
class FrequencyPredictor : public IPredictor {
public:
    FrequencyPredictor(int num_moe_stages, int num_routed_experts);
    ~FrequencyPredictor() override = default;

    std::string get_name() const override { return "frequency"; }
    PredictorType get_type() const override { return PredictorType::FREQUENCY; }
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

    void set_frequency_count(int stage_id, int expert_id, double count);
    double get_frequency_count(int stage_id, int expert_id) const;

private:
    int num_moe_stages_;
    int num_routed_experts_;
    std::vector<std::vector<double>> frequency_counts_;
};

/**
 * Temporal-Only Control Predictor.
 * Relies strictly on temporal autocorrelation across tokens at the target stage,
 * mathematically equivalent to CSTP with alpha = 0.0.
 */
class TemporalOnlyPredictor : public IPredictor {
public:
    TemporalOnlyPredictor(
        int num_moe_stages,
        int num_routed_experts,
        int top_k = 8,
        double smoothing_lambda = 1.0,
        double prior_epsilon = 0.1,
        bool online_adaptation = true);
    ~TemporalOnlyPredictor() override = default;

    std::string get_name() const override { return "temporal"; }
    PredictorType get_type() const override { return PredictorType::TEMPORAL; }
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

    std::shared_ptr<CSTPPredictor> get_inner_cstp() const { return inner_cstp_; }

private:
    std::shared_ptr<CSTPPredictor> inner_cstp_;
};

/**
 * Spatial-Only Control Predictor.
 * Relies strictly on cross-layer co-activation transitions from the source stage,
 * mathematically equivalent to CSTP with alpha = 1.0.
 */
class SpatialOnlyPredictor : public IPredictor {
public:
    SpatialOnlyPredictor(
        int num_moe_stages,
        int num_routed_experts,
        int top_k = 8,
        double smoothing_lambda = 1.0,
        double prior_epsilon = 0.1,
        bool online_adaptation = true,
        int prediction_horizon = 1);
    ~SpatialOnlyPredictor() override = default;

    std::string get_name() const override { return "spatial"; }
    PredictorType get_type() const override { return PredictorType::SPATIAL; }
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

    std::shared_ptr<CSTPPredictor> get_inner_cstp() const { return inner_cstp_; }

private:
    std::shared_ptr<CSTPPredictor> inner_cstp_;
};

/**
 * Random Control Predictor.
 * Selects candidate experts uniformly at random without replacement.
 * Uses a deterministic seed for exact reproducibility.
 */
class RandomPredictor : public IPredictor {
public:
    RandomPredictor(int num_routed_experts, unsigned int seed = 42);
    ~RandomPredictor() override = default;

    std::string get_name() const override { return "random"; }
    PredictorType get_type() const override { return PredictorType::RANDOM; }
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

    void set_seed(unsigned int seed);

private:
    int num_routed_experts_;
    unsigned int initial_seed_;
    std::mt19937 rng_;
};

} // namespace llm_system

