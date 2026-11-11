#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "prefetch/prefetch_config.h"
#include "prefetch/predictor_interface.h"

namespace llm_system {

/**
 * Proposed CSTP (Confidence- and Frequency-Aware Spatio-Temporal Predictor).
 *
 * Combines:
 * 1. Dirichlet-smoothed spatial conditional transition probabilities across layers.
 * 2. Dirichlet-smoothed temporal autocorrelation probabilities across tokens.
 * 3. Target-layer empirical marginal frequency priors.
 * 4. Composite convex weighting with parameter alpha in [0.0, 1.0].
 * 5. Configurable smoothing constants lambda and prior_epsilon.
 */
class CSTPPredictor : public IPredictor {
public:
    CSTPPredictor(
        int num_moe_stages,
        int num_routed_experts,
        int top_k = 8,
        double alpha = 0.5,
        double smoothing_lambda = 1.0,
        double prior_epsilon = 0.1,
        bool online_adaptation = true,
        int prediction_horizon = 1);

    ~CSTPPredictor() override = default;

    // Identification metadata
    std::string get_name() const override { return "cstp"; }
    PredictorType get_type() const override { return PredictorType::CSTP; }
    bool is_implemented() const override { return true; }
    bool is_evaluation_only() const override { return false; }

    // Request lifecycle reset (purges temporal history table for request_id)
    void reset(int request_id) override;

    // Post-routing feedback update
    void update(
        int request_id,
        int token_id,
        int source_stage_id,
        int target_stage_id,
        const std::vector<int>& source_experts,
        const std::vector<int>& actual_target_experts) override;

    // Budget-constrained prefetch prediction
    PredictionResult predict(
        int request_id,
        int token_id,
        int source_stage_id,
        int target_stage_id,
        const std::vector<int>& source_experts,
        int budget) override;

    // Scoring components and inspection helpers
    std::vector<double> get_marginal_prior(int stage_id) const;
    std::vector<double> compute_spatial_score(
        int source_stage_id,
        int target_stage_id,
        const std::vector<int>& source_experts) const;
    std::vector<double> compute_temporal_score(
        int request_id,
        int target_stage_id) const;

    // Manipulation helpers for testing
    void set_marginal_count(int stage_id, int expert_id, double count);
    void set_spatial_count(int source_stage_id, int source_expert, int target_expert, double count);
    void set_temporal_count(int stage_id, int prev_expert, int curr_expert, double count);
    void set_history_entry(int request_id, int stage_id, const std::vector<int>& experts);

    void set_alpha(double alpha) { alpha_ = alpha; }
    double get_alpha() const { return alpha_; }
    int get_num_routed_experts() const { return num_routed_experts_; }
    int get_num_moe_stages() const { return num_moe_stages_; }

private:
    int num_moe_stages_;
    int num_routed_experts_;
    int top_k_;
    double alpha_;
    double smoothing_lambda_;
    double prior_epsilon_;
    bool online_adaptation_;
    int prediction_horizon_;

    // Marginal occurrence counts: [stage_id][expert_id]
    std::vector<std::vector<double>> stage_marginal_counts_;

    // Spatial co-activation counts: [source_stage_id][source_expert][dest_expert]
    std::vector<std::vector<std::vector<double>>> spatial_counts_;

    // Temporal transition counts: [stage_id][prev_expert][curr_expert]
    std::vector<std::vector<std::vector<double>>> temporal_counts_;

    // History Table: request_id -> stage_id -> list of actual expert IDs from previous token
    std::unordered_map<int, std::unordered_map<int, std::vector<int>>> history_table_;
};

} // namespace llm_system

