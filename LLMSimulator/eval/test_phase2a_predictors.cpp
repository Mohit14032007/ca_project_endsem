#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "prefetch/acsh_predictor.h"
#include "prefetch/control_predictors.h"
#include "prefetch/cstp_predictor.h"
#include "prefetch/none_predictor.h"
#include "prefetch/prefetch_config.h"
#include "prefetch/predictor_interface.h"
#include "prefetch/predictor_registry.h"
#include "prefetch/st_moe_predictor.h"

// Simple testing framework
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg)                                                 \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "  [FAIL] " << __FUNCTION__ << " line " << __LINE__   \
                      << ": " << (msg) << std::endl;                           \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define RUN_TEST(fn)                                                           \
    do {                                                                       \
        std::cout << "[RUNNING] " << #fn << "...\n";                           \
        if (fn()) {                                                            \
            std::cout << "  [PASS] " << #fn << "\n";                           \
            tests_passed++;                                                    \
        } else {                                                               \
            std::cout << "  [FAIL] " << #fn << "\n";                           \
            tests_failed++;                                                    \
        }                                                                      \
    } while (0)

namespace llm_system {

// =============================================================================
// Group 1: Interface & Candidate Ordering Contract Tests
// =============================================================================

bool test_interface_candidate_ordering_descending() {
    PredictionResult res;
    res.scores = {
        CandidateScore{10, 1.5},
        CandidateScore{20, 4.2},
        CandidateScore{30, 0.8},
        CandidateScore{40, 3.1}
    };
    res.finalize_and_sort(4);

    TEST_ASSERT(res.is_valid, "Must be valid");
    TEST_ASSERT(res.candidate_experts.size() == 4, "Must have 4 candidates");
    TEST_ASSERT(res.candidate_experts[0] == 20, "Highest score (4.2) must be first");
    TEST_ASSERT(res.candidate_experts[1] == 40, "Second score (3.1) must be second");
    TEST_ASSERT(res.candidate_experts[2] == 10, "Third score (1.5) must be third");
    TEST_ASSERT(res.candidate_experts[3] == 30, "Lowest score (0.8) must be fourth");
    return true;
}

bool test_interface_candidate_ordering_tie_breaking() {
    PredictionResult res;
    // Tied scores of 2.0: expert 50, expert 12, expert 88, expert 5
    res.scores = {
        CandidateScore{50, 2.0},
        CandidateScore{12, 2.0},
        CandidateScore{88, 2.0},
        CandidateScore{5, 2.0}
    };
    res.finalize_and_sort(4);

    TEST_ASSERT(res.candidate_experts.size() == 4, "Must have 4 candidates");
    TEST_ASSERT(res.candidate_experts[0] == 5, "Lowest expert ID (5) must break tie first");
    TEST_ASSERT(res.candidate_experts[1] == 12, "Expert 12 second");
    TEST_ASSERT(res.candidate_experts[2] == 50, "Expert 50 third");
    TEST_ASSERT(res.candidate_experts[3] == 88, "Expert 88 fourth");
    return true;
}

bool test_interface_candidate_ordering_duplicate_expert_ids() {
    PredictionResult res;
    // Duplicate expert 7 with scores 1.0 and 5.0; duplicate expert 9 with scores 3.0 and 2.0
    res.scores = {
        CandidateScore{7, 1.0},
        CandidateScore{9, 3.0},
        CandidateScore{7, 5.0},
        CandidateScore{9, 2.0}
    };
    res.finalize_and_sort(4);

    TEST_ASSERT(res.candidate_experts.size() == 2, "Duplicate expert IDs must be collapsed");
    TEST_ASSERT(res.candidate_experts[0] == 7, "Expert 7 highest score (5.0) should be first");
    TEST_ASSERT(res.candidate_experts[1] == 9, "Expert 9 highest score (3.0) should be second");
    TEST_ASSERT(res.scores[0].score == 5.0, "Score for expert 7 must be 5.0");
    TEST_ASSERT(res.scores[1].score == 3.0, "Score for expert 9 must be 3.0");
    return true;
}

bool test_interface_candidate_ordering_invalid_scores() {
    PredictionResult res;
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();

    res.scores = {
        CandidateScore{1, 2.5},
        CandidateScore{2, nan_val},
        CandidateScore{3, inf_val},
        CandidateScore{4, 1.0}
    };
    res.finalize_and_sort(4);

    TEST_ASSERT(res.candidate_experts.size() == 4, "All candidates must be retained");
    TEST_ASSERT(res.candidate_experts[0] == 1, "Finite 2.5 must be first");
    TEST_ASSERT(res.candidate_experts[1] == 4, "Finite 1.0 must be second");
    // Non-finite sanitized to -infinity, tie broken by ascending ID (2 before 3)
    TEST_ASSERT(res.candidate_experts[2] == 2, "Sanitized NaN must be placed after finite");
    TEST_ASSERT(res.candidate_experts[3] == 3, "Sanitized Inf must be placed after finite");
    return true;
}

bool test_interface_budget_boundaries() {
    PredictionResult res;
    res.scores = {
        CandidateScore{1, 10.0},
        CandidateScore{2, 20.0},
        CandidateScore{3, 30.0}
    };

    // Budget = 0
    res.finalize_and_sort(0);
    TEST_ASSERT(res.candidate_experts.empty(), "Budget 0 must yield empty candidates");
    TEST_ASSERT(res.scores.empty(), "Budget 0 must yield empty scores");

    // Negative budget
    res.scores = { CandidateScore{1, 10.0} };
    res.finalize_and_sort(-5);
    TEST_ASSERT(res.candidate_experts.empty(), "Negative budget must yield empty candidates");

    // Budget > available
    res.scores = { CandidateScore{1, 10.0}, CandidateScore{2, 20.0} };
    res.finalize_and_sort(10);
    TEST_ASSERT(res.candidate_experts.size() == 2, "Budget 10 with 2 items yields 2 candidates");
    TEST_ASSERT(res.candidate_experts[0] == 2, "Sorted order preserved");
    TEST_ASSERT(res.candidate_experts[1] == 1, "Sorted order preserved");
    return true;
}

bool test_interface_request_isolation() {
    STMoEPredictor pred(4, 16, 2, 2, 2);
    pred.set_history_entry(101, 1, {4, 8});

    // Request 101 sees temporal history
    std::vector<int> hist101 = pred.get_history_entry(101, 1);
    TEST_ASSERT(hist101.size() == 2, "Req 101 must have 2 history entries");
    TEST_ASSERT(hist101[0] == 4, "Req 101 history expert 0 must be 4");

    // Request 202 must be completely empty
    std::vector<int> hist202 = pred.get_history_entry(202, 1);
    TEST_ASSERT(hist202.empty(), "Req 202 must have empty history");

    // Resetting req 101 clears its state without affecting req 202
    pred.reset(101);
    TEST_ASSERT(pred.get_history_entry(101, 1).empty(), "Req 101 must be cleared after reset");
    return true;
}

bool test_interface_multi_stage_horizon() {
    NonePredictor pred;
    PredictionResult res = pred.predict(1, 0, 2, 5, {0, 1}, 2);
    TEST_ASSERT(res.source_stage == 2, "source_stage must be 2");
    TEST_ASSERT(res.current_stage == 2, "current_stage alias must be 2");
    TEST_ASSERT(res.target_stage == 5, "target_stage must be 5");
    TEST_ASSERT(res.horizon == 3, "horizon must be 5 - 2 = 3");
    return true;
}

// =============================================================================
// Group 2: ST-MoE Predictor Tests (Faithful to Zhao et al. and st_moe_paper.py)
// =============================================================================

bool test_st_moe_counter_initialization() {
    STMoEPredictor pred(4, 16, 2, 2, 2);
    // Unset entry must return 0
    TEST_ASSERT(pred.get_cct_entry(0, 1, 5) == 0, "Unset entry must be 0");

    // Explicit set to decimal 2 (binary 10)
    pred.set_cct_entry(0, 1, 5, 2);
    TEST_ASSERT(pred.get_cct_entry(0, 1, 5) == 2, "Entry must be 2");
    return true;
}

bool test_st_moe_counter_increment_and_saturation() {
    STMoEPredictor pred(4, 16, 2, 2, 2, true);
    pred.set_cct_entry(0, 1, 5, 2);

    // Match with actual target expert 5 -> increment 2 to 3
    pred.update(1, 0, 0, 1, {1}, {5});
    TEST_ASSERT(pred.get_cct_entry(0, 1, 5) == 3, "2 + 1 must be 3");

    // Match again -> must saturate at ceiling 3
    pred.update(1, 1, 0, 1, {1}, {5});
    TEST_ASSERT(pred.get_cct_entry(0, 1, 5) == 3, "Must saturate at 3");
    return true;
}

bool test_st_moe_counter_decrement_and_floor() {
    STMoEPredictor pred(4, 16, 2, 2, 2, true);
    pred.set_cct_entry(0, 1, 5, 3);

    // Mismatch (actual target does not contain 5) -> decrement 3 to 2
    pred.update(1, 0, 0, 1, {1}, {9});
    TEST_ASSERT(pred.get_cct_entry(0, 1, 5) == 2, "3 - 1 must be 2");

    // Mismatch again -> decrement 2 to 1
    pred.update(1, 1, 0, 1, {1}, {9});
    TEST_ASSERT(pred.get_cct_entry(0, 1, 5) == 1, "2 - 1 must be 1");

    // Mismatch again -> decrement 1 to 0
    pred.update(1, 2, 0, 1, {1}, {9});
    TEST_ASSERT(pred.get_cct_entry(0, 1, 5) == 0, "1 - 1 must be 0");
    return true;
}

bool test_st_moe_zero_confidence_replacement() {
    // When ConfScore == 0 upon mismatch, candidate is replaced by an un-stored actual target expert
    STMoEPredictor pred(4, 16, 2, 2, 2, true);
    pred.set_cct_entry(0, 1, 5, 0); // score 0, eligible for replacement
    pred.set_cct_entry(0, 1, 6, 2); // score 2, will decrement to 1

    // Actual target experts: 7 and 8
    pred.update(1, 0, 0, 1, {1}, {7, 8});

    // Candidate 5 should be replaced by lowest un-stored actual target (7), initialized to 2
    TEST_ASSERT(pred.get_cct_entry(0, 1, 5) == 0, "Candidate 5 should be evicted");
    TEST_ASSERT(pred.get_cct_entry(0, 1, 7) == 2, "Candidate 7 inserted with confidence 2");
    TEST_ASSERT(pred.get_cct_entry(0, 1, 6) == 1, "Candidate 6 decremented to 1");
    return true;
}

bool test_st_moe_spatial_score_accumulation() {
    // Algorithm 2 Lines 3-5: Multiple source experts accumulate confidence for candidate
    STMoEPredictor pred(4, 16, 2, 2, 2);
    pred.set_cct_entry(0, 1, 10, 3);
    pred.set_cct_entry(0, 2, 10, 2);

    std::vector<double> scores = pred.compute_confidence_scores(1, 0, 1, {1, 2});
    TEST_ASSERT(scores[10] == 5.0, "Candidate 10 accumulated score must be 3 + 2 = 5");
    return true;
}

bool test_st_moe_equation1_spatio_temporal_combination() {
    // Equation 1: Score(e) = CCT(e) + HT(e), where HT(e) = 2
    STMoEPredictor pred(4, 16, 2, 2, 2);
    pred.set_cct_entry(0, 1, 7, 3);
    pred.set_history_entry(1, 1, {7});

    std::vector<double> scores = pred.compute_confidence_scores(1, 0, 1, {1});
    TEST_ASSERT(scores[7] == 5.0, "Score for candidate 7 must be spatial 3 + temporal 2 = 5");
    return true;
}

bool test_st_moe_threshold_filtering() {
    // Algorithm 2 Line 10: PrefetchSet = { k | ConfScore[k] >= threshold }
    STMoEPredictor pred(4, 16, 2, 2, 2);
    pred.set_cct_entry(0, 1, 4, 1); // Below threshold 2
    pred.set_cct_entry(0, 1, 5, 2); // Meets threshold 2
    pred.set_cct_entry(0, 1, 6, 3); // Exceeds threshold 2

    std::vector<int> filtered = pred.predict_threshold(1, 0, 1, {1}, 2);
    TEST_ASSERT(filtered.size() == 2, "Filtered set must contain exactly 2 experts");
    TEST_ASSERT(filtered[0] == 6, "Expert 6 (score 3) must be first");
    TEST_ASSERT(filtered[1] == 5, "Expert 5 (score 2) must be second");
    return true;
}

bool test_st_moe_budget_prediction_tie_breaking() {
    STMoEPredictor pred(4, 16, 2, 4, 2);
    pred.set_cct_entry(0, 1, 4, 3);
    pred.set_cct_entry(0, 1, 8, 3); // Tied with 4 at score 3
    pred.set_cct_entry(0, 1, 2, 2);

    PredictionResult res1 = pred.predict(1, 0, 0, 1, {1}, 1);
    TEST_ASSERT(res1.candidate_experts.size() == 1, "Budget 1");
    TEST_ASSERT(res1.candidate_experts[0] == 4, "Tied experts 4 and 8: 4 has lower ID");

    PredictionResult res2 = pred.predict(1, 0, 0, 1, {1}, 2);
    TEST_ASSERT(res2.candidate_experts.size() == 2, "Budget 2");
    TEST_ASSERT(res2.candidate_experts[0] == 4, "Expert 4 first");
    TEST_ASSERT(res2.candidate_experts[1] == 8, "Expert 8 second");
    return true;
}

// =============================================================================
// Group 3: CSTP Predictor Tests (Faithful to cstp_predictor.py)
// =============================================================================

bool test_cstp_marginal_prior_uniform_on_cold_start() {
    CSTPPredictor pred(4, 8, 2, 0.5, 1.0, 0.1);
    std::vector<double> prior = pred.get_marginal_prior(1);
    TEST_ASSERT(prior.size() == 8, "Size must be 8");
    double expected_uniform = 1.0 / 8.0;
    for (int e = 0; e < 8; ++e) {
        TEST_ASSERT(std::abs(prior[e] - expected_uniform) < 1e-5, "Prior must be uniform");
    }
    return true;
}

bool test_cstp_marginal_prior_with_observations() {
    CSTPPredictor pred(4, 8, 2, 0.5, 1.0, 0.1);
    pred.set_marginal_count(2, 3, 2.0); // Observe expert 3 twice at stage 2

    std::vector<double> prior = pred.get_marginal_prior(2);
    TEST_ASSERT(prior[3] > prior[0], "Observed expert 3 prior must be higher than expert 0");
    TEST_ASSERT(prior[3] > prior[1], "Observed expert 3 prior must be higher than expert 1");

    double sum = 0.0;
    for (double p : prior) {
        sum = sum + p;
    }
    TEST_ASSERT(std::abs(sum - 1.0) < 1e-5, "Prior probabilities must sum to 1.0");
    return true;
}

bool test_cstp_spatial_conditional_scoring() {
    CSTPPredictor pred(4, 8, 2, 0.5, 1.0, 0.1);
    // Stage 0 expert 1 strongly co-activates with stage 1 expert 5
    pred.set_spatial_count(0, 1, 5, 10.0);

    std::vector<double> score = pred.compute_spatial_score(0, 1, {1});
    int best_expert = 0;
    double max_score = score[0];
    for (int e = 1; e < 8; ++e) {
        if (score[e] > max_score) {
            max_score = score[e];
            best_expert = e;
        }
    }
    TEST_ASSERT(best_expert == 5, "Expert 5 must achieve highest spatial score");
    return true;
}

bool test_cstp_temporal_scoring_cold_start() {
    CSTPPredictor pred(4, 8, 2, 0.5, 1.0, 0.1);
    std::vector<double> prior = pred.get_marginal_prior(2);
    std::vector<double> score = pred.compute_temporal_score(1, 2);

    for (int e = 0; e < 8; ++e) {
        TEST_ASSERT(std::abs(score[e] - prior[e]) < 1e-5, "Cold start temporal score must equal prior");
    }
    return true;
}

bool test_cstp_alpha_boundaries_spatial_and_temporal() {
    CSTPPredictor pred(4, 8, 2, 0.5, 1.0, 0.1);
    pred.set_spatial_count(0, 2, 7, 20.0);
    pred.set_temporal_count(1, 4, 0, 20.0);
    pred.set_history_entry(1, 1, {4});

    // Pure spatial mode: alpha = 1.0
    pred.set_alpha(1.0);
    PredictionResult res_spatial = pred.predict(1, 0, 0, 1, {2}, 1);
    TEST_ASSERT(res_spatial.candidate_experts[0] == 7, "Alpha 1.0 must select spatial expert 7");

    // Pure temporal mode: alpha = 0.0
    pred.set_alpha(0.0);
    PredictionResult res_temporal = pred.predict(1, 0, 0, 1, {2}, 1);
    TEST_ASSERT(res_temporal.candidate_experts[0] == 0, "Alpha 0.0 must select temporal expert 0");
    return true;
}

bool test_cstp_multi_horizon_transitions() {
    CSTPPredictor pred(6, 16, 2, 1.0, 1.0, 0.1);
    // Set transition for horizon 2 (stage 0 -> stage 2)
    pred.set_spatial_count(0, 1, 12, 15.0);

    PredictionResult res = pred.predict(1, 0, 0, 2, {1}, 1);
    TEST_ASSERT(res.horizon == 2, "Horizon must be 2");
    TEST_ASSERT(res.candidate_experts[0] == 12, "Must predict destination expert 12 across horizon 2");
    return true;
}

bool test_cstp_online_adaptation() {
    CSTPPredictor pred(4, 8, 2, 0.5, 1.0, 0.1, true);

    // Initial prior is uniform
    std::vector<double> prior_before = pred.get_marginal_prior(1);

    // Update with actual target expert 6
    pred.update(1, 0, 0, 1, {2}, {6});

    std::vector<double> prior_after = pred.get_marginal_prior(1);
    TEST_ASSERT(prior_after[6] > prior_before[6], "Online update must increase expert 6 marginal prior");
    return true;
}

// =============================================================================
// Group 4: ACSH Predictor Tests (Faithful to acsh_predictor.py & test_acsh_validation.py)
// =============================================================================

bool test_acsh_initial_weights_are_equal() {
    auto cstp = std::make_shared<CSTPPredictor>(4, 16, 2, 0.5, 1.0, 0.1);
    auto st_moe = std::make_shared<STMoEPredictor>(4, 16, 2, 2, 2);
    ACSHPredictor acsh(PredictorType::ACSH_STAGE_AWARE, cstp, st_moe, 0.8, 0.2);

    std::pair<double, double> w = acsh.get_fusion_weights(1, 1, 1);
    TEST_ASSERT(std::abs(w.first - 0.5) < 1e-4, "Initial CSTP weight must be 0.5");
    TEST_ASSERT(std::abs(w.second - 0.5) < 1e-4, "Initial ST-MoE weight must be 0.5");
    TEST_ASSERT(std::abs((w.first + w.second) - 1.0) < 1e-4, "Weights must sum to 1.0");
    return true;
}

bool test_acsh_cstp_higher_reliability_adapts_weights() {
    auto cstp = std::make_shared<CSTPPredictor>(4, 16, 2, 1.0, 1.0, 0.1);
    auto st_moe = std::make_shared<STMoEPredictor>(4, 16, 2, 2, 2);

    // CSTP prefers [5, 6], ST-MoE prefers [10, 11]
    cstp->set_spatial_count(0, 0, 5, 10.0);
    cstp->set_spatial_count(0, 1, 6, 10.0);
    st_moe->set_cct_entry(0, 0, 10, 3);
    st_moe->set_cct_entry(0, 1, 11, 3);

    ACSHPredictor acsh(PredictorType::ACSH_STAGE_AWARE, cstp, st_moe, 0.8, 0.2);

    PredictionResult preds = acsh.predict(1, 0, 0, 1, {0, 1}, 2);
    TEST_ASSERT(preds.candidate_experts.size() == 2, "Budget 2");

    // Actual target matches CSTP [5, 6]
    acsh.update(1, 0, 0, 1, {0, 1}, {5, 6});

    std::pair<double, double> w = acsh.get_fusion_weights(1, 1, 1);
    std::pair<double, double> r = acsh.get_current_reliability(1, 1, 1);

    TEST_ASSERT(r.first > r.second, "CSTP reliability must exceed ST-MoE reliability");
    TEST_ASSERT(w.first > 0.5, "CSTP weight must adapt to > 0.5");
    TEST_ASSERT(w.second < 0.5, "ST-MoE weight must adapt to < 0.5");
    TEST_ASSERT(std::abs((w.first + w.second) - 1.0) < 1e-4, "Weights must sum to 1.0");
    return true;
}

bool test_acsh_st_moe_higher_reliability_adapts_weights() {
    auto cstp = std::make_shared<CSTPPredictor>(4, 16, 2, 1.0, 1.0, 0.1);
    auto st_moe = std::make_shared<STMoEPredictor>(4, 16, 2, 2, 2);

    cstp->set_spatial_count(0, 0, 5, 10.0);
    cstp->set_spatial_count(0, 1, 6, 10.0);
    st_moe->set_cct_entry(0, 0, 10, 3);
    st_moe->set_cct_entry(0, 1, 11, 3);

    ACSHPredictor acsh(PredictorType::ACSH_STAGE_AWARE, cstp, st_moe, 0.8, 0.2);

    acsh.predict(1, 0, 0, 1, {0, 1}, 2);

    // Actual target matches ST-MoE [10, 11]
    acsh.update(1, 0, 0, 1, {0, 1}, {10, 11});

    std::pair<double, double> w = acsh.get_fusion_weights(1, 1, 1);
    std::pair<double, double> r = acsh.get_current_reliability(1, 1, 1);

    TEST_ASSERT(r.second > r.first, "ST-MoE reliability must exceed CSTP reliability");
    TEST_ASSERT(w.second > 0.5, "ST-MoE weight must adapt to > 0.5");
    TEST_ASSERT(w.first < 0.5, "CSTP weight must adapt to < 0.5");
    return true;
}

bool test_acsh_stage_aware_isolation() {
    auto cstp = std::make_shared<CSTPPredictor>(4, 16, 2, 1.0, 1.0, 0.1);
    auto st_moe = std::make_shared<STMoEPredictor>(4, 16, 2, 2, 2);
    cstp->set_spatial_count(0, 0, 5, 10.0);

    ACSHPredictor acsh(PredictorType::ACSH_STAGE_AWARE, cstp, st_moe, 0.8, 0.2);

    acsh.predict(1, 0, 0, 1, {0}, 2);
    acsh.update(1, 0, 0, 1, {0}, {5, 6});

    // Stage 1 adapted to favor CSTP
    std::pair<double, double> w1 = acsh.get_fusion_weights(1, 1, 1);
    TEST_ASSERT(w1.first > 0.5, "Stage 1 CSTP weight must adapt to > 0.5");

    // Stage 2 must remain at initial 50/50
    std::pair<double, double> w2 = acsh.get_fusion_weights(1, 2, 1);
    TEST_ASSERT(std::abs(w2.first - 0.5) < 1e-4, "Stage 2 CSTP weight must remain 0.5");
    TEST_ASSERT(std::abs(w2.second - 0.5) < 1e-4, "Stage 2 ST-MoE weight must remain 0.5");
    return true;
}

bool test_acsh_global_variant_sharing() {
    auto cstp = std::make_shared<CSTPPredictor>(4, 16, 2, 1.0, 1.0, 0.1);
    auto st_moe = std::make_shared<STMoEPredictor>(4, 16, 2, 2, 2);
    cstp->set_spatial_count(0, 0, 5, 10.0);

    ACSHPredictor acsh(PredictorType::ACSH_GLOBAL, cstp, st_moe, 0.8, 0.2);

    acsh.predict(1, 0, 0, 1, {0}, 2);
    acsh.update(1, 0, 0, 1, {0}, {5, 6});

    // In global variant, stage 2 sees adapted weights immediately
    std::pair<double, double> w2 = acsh.get_fusion_weights(1, 2, 1);
    TEST_ASSERT(w2.first > 0.5, "Global variant: stage 2 sees adapted CSTP weight > 0.5");
    return true;
}

bool test_acsh_fast_recovery_trigger() {
    auto cstp = std::make_shared<CSTPPredictor>(4, 16, 2, 1.0, 1.0, 0.1);
    auto st_moe = std::make_shared<STMoEPredictor>(4, 16, 2, 2, 2);

    ACSHPredictor acsh(
        PredictorType::ACSH_FAST_RECOVERY, cstp, st_moe,
        0.8, 0.2, 1.0, 1.0, 0.5, 2, 0.8, 0.1, 3);

    // Feed 2 consecutive complete misses
    std::vector<std::vector<int>> targets = {{14, 15}, {12, 13}};
    for (int tok = 0; tok < 2; ++tok) {
        acsh.predict(1, tok, 0, 1, {0, 1}, 2);
        acsh.update(1, tok, 0, 1, {0, 1}, targets[tok]);
    }

    TEST_ASSERT(acsh.get_shift_detected_count() > 0, "Fast recovery shift detection must trigger");
    return true;
}

// =============================================================================
// Group 5: Control Predictors Tests
// =============================================================================

bool test_frequency_predictor_ranking() {
    FrequencyPredictor pred(4, 16);
    pred.set_frequency_count(1, 8, 50.0);
    pred.set_frequency_count(1, 3, 20.0);
    pred.set_frequency_count(1, 14, 80.0);

    PredictionResult res = pred.predict(1, 0, 0, 1, {}, 3);
    TEST_ASSERT(res.candidate_experts.size() == 3, "Budget 3");
    TEST_ASSERT(res.candidate_experts[0] == 14, "Highest frequency 14 first");
    TEST_ASSERT(res.candidate_experts[1] == 8, "Second frequency 8 second");
    TEST_ASSERT(res.candidate_experts[2] == 3, "Third frequency 3 third");
    return true;
}

bool test_temporal_only_predictor() {
    TemporalOnlyPredictor pred(4, 16, 2);
    TEST_ASSERT(pred.get_name() == "temporal", "Name must be temporal");
    TEST_ASSERT(pred.get_type() == PredictorType::TEMPORAL, "Type must be TEMPORAL");
    TEST_ASSERT(pred.get_inner_cstp()->get_alpha() == 0.0, "Inner CSTP alpha must be 0.0");
    return true;
}

bool test_spatial_only_predictor() {
    SpatialOnlyPredictor pred(4, 16, 2);
    TEST_ASSERT(pred.get_name() == "spatial", "Name must be spatial");
    TEST_ASSERT(pred.get_type() == PredictorType::SPATIAL, "Type must be SPATIAL");
    TEST_ASSERT(pred.get_inner_cstp()->get_alpha() == 1.0, "Inner CSTP alpha must be 1.0");
    return true;
}

bool test_random_predictor_deterministic_and_unique() {
    RandomPredictor pred1(32, 12345);
    RandomPredictor pred2(32, 12345);

    PredictionResult res1 = pred1.predict(1, 0, 0, 1, {}, 4);
    PredictionResult res2 = pred2.predict(1, 0, 0, 1, {}, 4);

    TEST_ASSERT(res1.candidate_experts.size() == 4, "Budget 4");
    TEST_ASSERT(res2.candidate_experts.size() == 4, "Budget 4");

    // Exact deterministic reproducibility
    for (int i = 0; i < 4; ++i) {
        TEST_ASSERT(res1.candidate_experts[i] == res2.candidate_experts[i], "Predictions must match exactly");
    }

    // Verify uniqueness (no duplicate candidates)
    std::unordered_set<int> unique_cands(
        res1.candidate_experts.begin(), res1.candidate_experts.end());
    TEST_ASSERT(unique_cands.size() == 4, "Random candidate draw must have no duplicates");
    return true;
}

// =============================================================================
// Group 6: Predictor Registry Tests
// =============================================================================

bool test_registry_create_all_supported_predictors() {
    PrefetchConfig cfg;
    cfg.enabled = true;
    cfg.mode = PrefetchMode::PREDICTION_ONLY;

    std::vector<PredictorType> supported_types = {
        PredictorType::NONE,
        PredictorType::ST_MOE,
        PredictorType::CSTP,
        PredictorType::ACSH_GLOBAL,
        PredictorType::ACSH_STAGE_AWARE,
        PredictorType::ACSH_FAST_RECOVERY,
        PredictorType::FREQUENCY,
        PredictorType::TEMPORAL,
        PredictorType::SPATIAL,
        PredictorType::RANDOM
    };

    for (auto type : supported_types) {
        TEST_ASSERT(PredictorRegistry::is_implemented(type), "Type must be implemented");
        IPredictor::Ptr inst = PredictorRegistry::create_predictor(type, cfg, 4, 16, 2);
        TEST_ASSERT(inst != nullptr, "Instance must not be null");
        TEST_ASSERT(inst->get_type() == type, "Instance type must match");
        TEST_ASSERT(inst->is_implemented(), "Instance must report implemented");
        TEST_ASSERT(!inst->is_evaluation_only(), "Deployable predictor must not be eval-only");
    }
    return true;
}

bool test_registry_oracle_rejected() {
    PrefetchConfig cfg;
    cfg.enabled = true;
    cfg.mode = PrefetchMode::PREDICTION_ONLY;

    TEST_ASSERT(!PredictorRegistry::is_implemented(PredictorType::ORACLE), "Oracle must not be implemented");
    TEST_ASSERT(PredictorRegistry::is_evaluation_only(PredictorType::ORACLE), "Oracle is evaluation-only");

    bool threw = false;
    try {
        PredictorRegistry::create_predictor(PredictorType::ORACLE, cfg, 4, 16, 2);
    } catch (const std::runtime_error& e) {
        threw = true;
        std::string msg = e.what();
        TEST_ASSERT(msg.find("evaluation-only") != std::string::npos, "Error message must cite evaluation-only status");
    }
    TEST_ASSERT(threw, "create_predictor must throw for Oracle");
    return true;
}

bool test_registry_unknown_rejected() {
    TEST_ASSERT(!PredictorRegistry::is_known_identifier("non_existent_predictor_xyz"), "Unknown identifier must fail check");
    bool threw = false;
    try {
        parse_predictor_type("non_existent_predictor_xyz");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    TEST_ASSERT(threw, "parse_predictor_type must throw for unknown name");
    return true;
}

} // namespace llm_system

int main() {
    std::cout << "======================================================================\n";
    std::cout << "PHASE 2A: C++ PREDICTOR IMPLEMENTATION TEST SUITE\n";
    std::cout << "======================================================================\n\n";

    std::cout << "--- Group 1: Interface & Candidate Ordering Contract Tests ---\n";
    RUN_TEST(llm_system::test_interface_candidate_ordering_descending);
    RUN_TEST(llm_system::test_interface_candidate_ordering_tie_breaking);
    RUN_TEST(llm_system::test_interface_candidate_ordering_duplicate_expert_ids);
    RUN_TEST(llm_system::test_interface_candidate_ordering_invalid_scores);
    RUN_TEST(llm_system::test_interface_budget_boundaries);
    RUN_TEST(llm_system::test_interface_request_isolation);
    RUN_TEST(llm_system::test_interface_multi_stage_horizon);

    std::cout << "\n--- Group 2: ST-MoE Predictor Tests ---\n";
    RUN_TEST(llm_system::test_st_moe_counter_initialization);
    RUN_TEST(llm_system::test_st_moe_counter_increment_and_saturation);
    RUN_TEST(llm_system::test_st_moe_counter_decrement_and_floor);
    RUN_TEST(llm_system::test_st_moe_zero_confidence_replacement);
    RUN_TEST(llm_system::test_st_moe_spatial_score_accumulation);
    RUN_TEST(llm_system::test_st_moe_equation1_spatio_temporal_combination);
    RUN_TEST(llm_system::test_st_moe_threshold_filtering);
    RUN_TEST(llm_system::test_st_moe_budget_prediction_tie_breaking);

    std::cout << "\n--- Group 3: CSTP Predictor Tests ---\n";
    RUN_TEST(llm_system::test_cstp_marginal_prior_uniform_on_cold_start);
    RUN_TEST(llm_system::test_cstp_marginal_prior_with_observations);
    RUN_TEST(llm_system::test_cstp_spatial_conditional_scoring);
    RUN_TEST(llm_system::test_cstp_temporal_scoring_cold_start);
    RUN_TEST(llm_system::test_cstp_alpha_boundaries_spatial_and_temporal);
    RUN_TEST(llm_system::test_cstp_multi_horizon_transitions);
    RUN_TEST(llm_system::test_cstp_online_adaptation);

    std::cout << "\n--- Group 4: ACSH Hybrid Predictor Tests ---\n";
    RUN_TEST(llm_system::test_acsh_initial_weights_are_equal);
    RUN_TEST(llm_system::test_acsh_cstp_higher_reliability_adapts_weights);
    RUN_TEST(llm_system::test_acsh_st_moe_higher_reliability_adapts_weights);
    RUN_TEST(llm_system::test_acsh_stage_aware_isolation);
    RUN_TEST(llm_system::test_acsh_global_variant_sharing);
    RUN_TEST(llm_system::test_acsh_fast_recovery_trigger);

    std::cout << "\n--- Group 5: Control Predictors Tests ---\n";
    RUN_TEST(llm_system::test_frequency_predictor_ranking);
    RUN_TEST(llm_system::test_temporal_only_predictor);
    RUN_TEST(llm_system::test_spatial_only_predictor);
    RUN_TEST(llm_system::test_random_predictor_deterministic_and_unique);

    std::cout << "\n--- Group 6: Predictor Registry Tests ---\n";
    RUN_TEST(llm_system::test_registry_create_all_supported_predictors);
    RUN_TEST(llm_system::test_registry_oracle_rejected);
    RUN_TEST(llm_system::test_registry_unknown_rejected);

    std::cout << "\n======================================================================\n";
    std::cout << "TEST SUMMARY: " << tests_passed << " PASSED, " << tests_failed << " FAILED\n";
    std::cout << "======================================================================\n";

    if (tests_failed == 0) {
        return 0;
    } else {
        return 1;
    }
}

