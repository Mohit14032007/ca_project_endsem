#pragma once

#include <cmath>
#include <deque>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "prefetch/cstp_predictor.h"
#include "prefetch/prefetch_config.h"
#include "prefetch/predictor_interface.h"
#include "prefetch/st_moe_predictor.h"

namespace llm_system {

/**
 * Adaptive Confidence and Statistical Hybrid (ACSH) Predictor.
 *
 * Combines Bayesian probabilistic state from CSTP with bounded saturating confidence
 * counters from ST-MoE using causal online reliability estimation and adaptive softmax fusion.
 *
 * Supported Variants:
 * - ACSH_GLOBAL: Shares a single global EMA reliability pair per request.
 * - ACSH_STAGE_AWARE: Tracks decoupled EMA reliability pairs per (target_stage, horizon).
 * - ACSH_FAST_RECOVERY: Adds online error burst detection and accelerated beta adaptation.
 */
class ACSHPredictor : public IPredictor {
public:
    struct PendingRecord {
        std::vector<int> cstp_top_b;
        std::vector<int> st_top_b;
        std::vector<int> acsh_top_b;
        double w_c = 0.5;
        double w_m = 0.5;
        int horizon = 1;
        int budget = 0;
    };

    ACSHPredictor(
        PredictorType variant,
        std::shared_ptr<CSTPPredictor> cstp,
        std::shared_ptr<STMoEPredictor> st_moe,
        double beta = 0.8,
        double tau = 0.2,
        double temp_c = 1.0,
        double temp_m = 1.0,
        double initial_reliability = 0.5,
        int shift_window = 3,
        double shift_threshold = 0.75,
        double fast_beta = 0.2,
        int fast_cooldown_steps = 3);

    ~ACSHPredictor() override = default;

    // Identification metadata
    std::string get_name() const override;
    PredictorType get_type() const override { return variant_; }
    bool is_implemented() const override { return true; }
    bool is_evaluation_only() const override { return false; }

    // Request lifecycle reset (purges request-level reliability tracking and resets components)
    void reset(int request_id) override;

    // Post-routing feedback update
    void update(
        int request_id,
        int token_id,
        int source_stage_id,
        int target_stage_id,
        const std::vector<int>& source_experts,
        const std::vector<int>& actual_target_experts) override;

    // Adaptive prefetch prediction
    PredictionResult predict(
        int request_id,
        int token_id,
        int source_stage_id,
        int target_stage_id,
        const std::vector<int>& source_experts,
        int budget) override;

    // Inspection helpers
    std::pair<double, double> get_current_reliability(
        int request_id, int target_stage_id, int horizon) const;
    std::pair<double, double> get_fusion_weights(
        int request_id, int target_stage_id, int horizon) const;
    int get_shift_detected_count() const { return shift_detected_count_; }

    std::shared_ptr<CSTPPredictor> get_cstp() const { return cstp_; }
    std::shared_ptr<STMoEPredictor> get_st_moe() const { return st_moe_; }

private:
    double calculate_recall_utility(
        const std::vector<int>& predicted,
        const std::vector<int>& actual) const;

    PredictorType variant_;
    std::shared_ptr<CSTPPredictor> cstp_;
    std::shared_ptr<STMoEPredictor> st_moe_;

    double beta_;
    double tau_;
    double temp_c_;
    double temp_m_;
    double initial_reliability_;
    int shift_window_;
    double shift_threshold_;
    double fast_beta_;
    int fast_cooldown_steps_;

    int num_routed_experts_;
    int num_moe_stages_;
    int shift_detected_count_ = 0;

    // Global variant reliability: request_id -> (R_C, R_M)
    mutable std::unordered_map<int, double> rel_c_global_;
    mutable std::unordered_map<int, double> rel_m_global_;

    // Stage-aware / Fast-recovery reliability:
    // request_id -> map of key(target_stage, horizon) -> double
    // Key encoding: (target_stage << 16) | horizon
    mutable std::unordered_map<int, std::unordered_map<long long, double>> rel_c_stage_;
    mutable std::unordered_map<int, std::unordered_map<long long, double>> rel_m_stage_;

    // Fast recovery state tracking:
    // request_id -> map of key -> deque of recent errors
    std::unordered_map<int, std::unordered_map<long long, std::deque<double>>> recent_errors_;
    // request_id -> map of key -> cooldown remaining
    std::unordered_map<int, std::unordered_map<long long, int>> fast_cooldown_;

    // Pending prediction records for causal evaluation in update()
    // Key encoding: ((long long)request_id << 32) | ((long long)target_stage_id << 16) | (long long)token_id
    std::unordered_map<long long, PendingRecord> pending_predictions_;

    static long long make_stage_key(int target_stage, int horizon) {
        return (static_cast<long long>(target_stage) << 16) | (static_cast<long long>(horizon) & 0xFFFFLL);
    }

    static long long make_pred_key(int request_id, int target_stage, int token_id) {
        return (static_cast<long long>(request_id) << 36) |
               ((static_cast<long long>(target_stage) & 0x3FFFFLL) << 18) |
               (static_cast<long long>(token_id) & 0x3FFFFLL);
    }
};

} // namespace llm_system

