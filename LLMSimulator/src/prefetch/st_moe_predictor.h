#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "prefetch/prefetch_config.h"
#include "prefetch/predictor_interface.h"

namespace llm_system {

/**
 * Paper-Faithful ST-MoE Expert Predictor Implementation.
 *
 * Reference:
 * "A Spatio-Temporal Expert Prefetching Framework for Efficient MoE-based LLM Inference"
 * Yingnan Zhao, Razvan Bunescu, Ahmed Louri, Avinash Karanth, and Ke Wang.
 * arXiv:2606.15453v1 [cs.AR], 13 Jun 2026.
 *
 * Core Components:
 * - Cross-Layer Correlation Table (CCT): Tracks destination candidate profiles with 2-bit
 *   saturating counters (00=0, 01=1, 10=2, 11=3). Initialized to decimal 2 (binary 10).
 * - History Table (HT): Request-isolated temporal tracking from previous token executions.
 * - Score Aggregation (Equation 1): Score(e) = CCT(e) + HT(e), where HT contribution is +2.
 * - Online Adaptation (Algorithm 3): Counter increment on match, decrement on mismatch,
 *   and deterministic replacement of zero-confidence candidates.
 */
class STMoEPredictor : public IPredictor {
public:
    STMoEPredictor(
        int num_moe_stages,
        int num_routed_experts,
        int top_k = 8,
        int cct_capacity = 8,
        int confidence_threshold = 2,
        bool online_adaptation = true,
        int prediction_horizon = 1);

    ~STMoEPredictor() override = default;

    // Identification metadata
    std::string get_name() const override { return "st_moe"; }
    PredictorType get_type() const override { return PredictorType::ST_MOE; }
    bool is_implemented() const override { return true; }
    bool is_evaluation_only() const override { return false; }

    // Request lifecycle reset (purges temporal history table for request_id)
    void reset(int request_id) override;

    // Post-routing feedback update (Algorithm 3)
    void update(
        int request_id,
        int token_id,
        int source_stage_id,
        int target_stage_id,
        const std::vector<int>& source_experts,
        const std::vector<int>& actual_target_experts) override;

    // Budget-constrained prefetch prediction (Algorithm 2 & Equation 1)
    PredictionResult predict(
        int request_id,
        int token_id,
        int source_stage_id,
        int target_stage_id,
        const std::vector<int>& source_experts,
        int budget) override;

    // Explicit test inspection and manipulation helpers
    void set_cct_entry(int source_stage_id, int source_expert, int dest_expert, int confidence);
    int get_cct_entry(int source_stage_id, int source_expert, int dest_expert) const;
    void set_history_entry(int request_id, int stage_id, const std::vector<int>& experts);
    std::vector<int> get_history_entry(int request_id, int stage_id) const;

    // Score computation helper
    std::vector<double> compute_confidence_scores(
        int request_id,
        int source_stage_id,
        int target_stage_id,
        const std::vector<int>& source_experts) const;

    // Algorithm 2 Line 10 threshold filter { e | ConfScore[e] >= threshold }
    std::vector<int> predict_threshold(
        int request_id,
        int source_stage_id,
        int target_stage_id,
        const std::vector<int>& source_experts,
        int threshold) const;

private:
    int num_moe_stages_;
    int num_routed_experts_;
    int top_k_;
    int cct_capacity_;
    int confidence_threshold_;
    bool online_adaptation_;
    int prediction_horizon_;

    // CCT: [source_stage_id][source_expert] -> {dest_expert: counter (0..3)}
    std::vector<std::vector<std::unordered_map<int, int>>> cct_;

    // History Table: request_id -> stage_id -> list of actual expert IDs from previous token
    std::unordered_map<int, std::unordered_map<int, std::vector<int>>> history_table_;
};

} // namespace llm_system

