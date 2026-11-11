#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "prefetch/prefetch_config.h"

namespace llm_system {

/**
 * Scored candidate expert structure.
 * Represents an expert candidate with an associated metric:
 * - ST-MoE: Integer confidence counter (0..E)
 * - CSTP: Composite Bayesian probability score in [0.0, 1.0]
 * - ACSH: Softmax-normalized hybrid score in [0.0, 1.0]
 * - Frequency: Historical occurrence count (>= 0)
 * - Random: Uniform pseudo-random draw
 */
struct CandidateScore {
    int expert_id = -1;
    double score = 0.0;
};

/**
 * Standard typed result structure for expert predictions.
 */
struct PredictionResult {
    int request_id = 0;                     // Unique request / sequence ID
    int token_id = 0;                       // Token position in sequence
    int source_stage = 0;                   // Source MoE stage generating prediction
    int current_stage = 0;                  // Backwards-compatible alias for source_stage
    int target_stage = 0;                   // Destination MoE stage being predicted
    int horizon = 1;                        // Horizon distance (target_stage - source_stage)
    std::vector<int> candidate_experts;     // Predicted expert IDs (at most budget B)
    std::vector<CandidateScore> scores;     // Candidate scores sorted by contract
    bool is_valid = false;                  // Status flag indicating prediction validity

    /**
     * Enforces the common candidate ordering contract:
     * 1. Resolves and deduplicates candidates (keeps highest score per expert ID).
     * 2. Sanitizes non-finite scores (NaN, +/-Inf) to -infinity.
     * 3. Sorts deterministically: descending score, then ascending expert ID.
     * 4. Truncates candidate list and scores to budget B.
     * 5. Synchronizes candidate_experts and scores vectors.
     * 6. Marks is_valid = true.
     */
    void finalize_and_sort(int budget) {
        is_valid = true;

        if (budget <= 0) {
            candidate_experts.clear();
            scores.clear();
            return;
        }

        if (scores.empty()) {
            candidate_experts.clear();
            return;
        }

        // Step 1: Deduplicate while sanitizing non-finite values (keep highest score per expert)
        std::unordered_map<int, double> best_scores;
        for (const auto& item : scores) {
            double clean_score = item.score;
            if (std::isnan(clean_score) || std::isinf(clean_score)) {
                clean_score = -std::numeric_limits<double>::infinity();
            }

            auto it = best_scores.find(item.expert_id);
            if (it == best_scores.end()) {
                best_scores[item.expert_id] = clean_score;
            } else {
                if (clean_score > it->second) {
                    it->second = clean_score;
                }
            }
        }

        std::vector<CandidateScore> unique_candidates;
        unique_candidates.reserve(best_scores.size());
        for (const auto& pair : best_scores) {
            CandidateScore cs;
            cs.expert_id = pair.first;
            cs.score = pair.second;
            unique_candidates.push_back(cs);
        }

        // Step 2: Deterministic sorting
        // Primary key: score descending
        // Secondary key: expert_id ascending (deterministic tie-breaker)
        std::sort(
            unique_candidates.begin(),
            unique_candidates.end(),
            [](const CandidateScore& a, const CandidateScore& b) {
                if (a.score != b.score) {
                    return a.score > b.score;
                }
                return a.expert_id < b.expert_id;
            });

        // Step 3: Truncate to budget
        int effective_budget = budget;
        if ((int)unique_candidates.size() < effective_budget) {
            effective_budget = (int)unique_candidates.size();
        }

        scores.clear();
        scores.reserve(effective_budget);
        candidate_experts.clear();
        candidate_experts.reserve(effective_budget);

        for (int i = 0; i < effective_budget; ++i) {
            scores.push_back(unique_candidates[i]);
            candidate_experts.push_back(unique_candidates[i].expert_id);
        }
    }
};

/**
 * Common abstract interface for all MoE expert predictors.
 *
 * Causal Separation Contract:
 * - At prediction time: predict() receives strictly already-observed tokens/stages
 *   and source experts. Actual target-stage routing decisions MUST NEVER be accessed.
 * - After target stage execution: update() receives the ground-truth routing decisions
 *   for the (source_stage_id -> target_stage_id) transition to update internal state.
 */
class IPredictor {
public:
    using Ptr = std::shared_ptr<IPredictor>;

    virtual ~IPredictor() = default;

    // Identification metadata
    virtual std::string get_name() const = 0;
    virtual PredictorType get_type() const = 0;
    virtual bool is_implemented() const = 0;
    virtual bool is_evaluation_only() const { return false; }

    // Request lifecycle reset (guarantees cross-request isolation)
    virtual void reset(int request_id) = 0;

    /**
     * Updates internal prediction state with newly observed routing decisions.
     * Strictly called AFTER target stage routing is revealed (causally valid).
     *
     * @param request_id Unique sequence request identifier.
     * @param token_id Token position in sequence.
     * @param source_stage_id Predecessor stage index where source_experts were activated.
     * @param target_stage_id Target stage index that just completed execution.
     * @param source_experts Experts that were selected at source_stage_id.
     * @param actual_target_experts Actual experts selected at target_stage_id.
     */
    virtual void update(
        int request_id,
        int token_id,
        int source_stage_id,
        int target_stage_id,
        const std::vector<int>& source_experts,
        const std::vector<int>& actual_target_experts) = 0;

    /**
     * Predicts candidate expert IDs for a target MoE stage.
     * Strictly causal: does not access future routing or actual target experts.
     *
     * @param request_id Unique sequence request identifier.
     * @param token_id Current token index.
     * @param source_stage_id Predecessor stage generating the prediction.
     * @param target_stage_id Destination MoE stage to prefetch for.
     * @param source_experts Experts observed at source_stage_id.
     * @param budget Maximum candidate budget (B).
     */
    virtual PredictionResult predict(
        int request_id,
        int token_id,
        int source_stage_id,
        int target_stage_id,
        const std::vector<int>& source_experts,
        int budget) = 0;
};

} // namespace llm_system
