#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "prefetch/acsh_predictor.h"
#include "prefetch/cstp_predictor.h"
#include "prefetch/live_prediction_engine.h"
#include "prefetch/prefetch_candidate_queue.h"
#include "prefetch/prefetch_config.h"
#include "prefetch/predictor_interface.h"
#include "prefetch/predictor_registry.h"
#include "prefetch/st_moe_predictor.h"
#include "scheduler/scheduler.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg)                                                 \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "  [FAIL] " << __FUNCTION__ << " line " << __LINE__   \
                      << ": " << (msg) << std::endl;                           \
            tests_failed++;                                                    \
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

// Test 1: Empty queue behavior
bool test_1_empty_queue() {
    PrefetchConfig config;
    config.queue_capacity = 10;
    std::string test_dir = "results/phase_3_prefetch_scheduling/test1_empty";
    PrefetchCandidateQueue queue(config, 10, 64, test_dir);

    TEST_ASSERT(queue.size() == 0, "Initial queue size must be 0");
    TEST_ASSERT(queue.empty(), "Initial queue must be empty");
    TEST_ASSERT(queue.get_capacity() == 10, "Capacity must be 10");

    auto selected = queue.select_ready_candidates(5);
    TEST_ASSERT(selected.empty(), "Selected candidates must be empty");

    int reconciled = queue.reconcile_stage(0, 0, 0, {1, 2});
    TEST_ASSERT(reconciled == 0, "Reconciled count must be 0 for empty queue");

    int purged = queue.purge_request(0);
    TEST_ASSERT(purged == 0, "Purged count must be 0 for empty queue");
    return true;
}

// Test 2: Single candidate admission
bool test_2_single_candidate_admission() {
    PrefetchConfig config;
    std::string test_dir = "results/phase_3_prefetch_scheduling/test2_single";
    PrefetchCandidateQueue queue(config, 10, 64, test_dir);

    PrefetchCandidate c;
    c.request_id = 0;
    c.token_id = 0;
    c.source_stage_id = 0;
    c.target_stage_id = 1;
    c.expert_id = 5;
    c.prediction_rank = 1;
    c.prediction_score = 0.9;
    c.prediction_horizon = 1;

    bool ok = queue.enqueue_candidate(c);
    TEST_ASSERT(ok, "Candidate must be admitted");
    TEST_ASSERT(queue.size() == 1, "Queue size must be 1");
    TEST_ASSERT(!queue.empty(), "Queue must not be empty");

    const auto& m = queue.get_metrics();
    TEST_ASSERT(m.candidates_admitted == 1, "candidates_admitted must be 1");
    TEST_ASSERT(m.total_candidates_received == 1, "total_candidates_received must be 1");
    return true;
}

// Test 3: Multiple candidate admission
bool test_3_multiple_candidate_admission() {
    PrefetchConfig config;
    std::string test_dir = "results/phase_3_prefetch_scheduling/test3_multi";
    PrefetchCandidateQueue queue(config, 10, 64, test_dir);

    for (int exp = 10; exp < 14; exp = exp + 1) {
        PrefetchCandidate c;
        c.request_id = 0;
        c.token_id = 0;
        c.source_stage_id = 1;
        c.target_stage_id = 2;
        c.expert_id = exp;
        c.prediction_rank = exp - 9;
        c.prediction_score = 0.8 - 0.1 * static_cast<double>(exp - 10);
        c.prediction_horizon = 1;
        bool ok = queue.enqueue_candidate(c);
        TEST_ASSERT(ok, "Candidate must be admitted");
    }

    TEST_ASSERT(queue.size() == 4, "Queue size must be 4");
    TEST_ASSERT(queue.get_metrics().candidates_admitted == 4, "Admitted must be 4");
    return true;
}

// Test 4: Duplicate suppression
bool test_4_duplicate_suppression() {
    PrefetchConfig config;
    std::string test_dir = "results/phase_3_prefetch_scheduling/test4_dup";
    PrefetchCandidateQueue queue(config, 10, 64, test_dir);

    PrefetchCandidate c1;
    c1.request_id = 0;
    c1.token_id = 0;
    c1.target_stage_id = 1;
    c1.expert_id = 5;
    c1.prediction_rank = 1;
    c1.prediction_score = 0.9;

    bool ok1 = queue.enqueue_candidate(c1);
    TEST_ASSERT(ok1, "First submission must succeed");

    // Re-submit duplicate with equal or lower priority
    PrefetchCandidate c2 = c1;
    c2.prediction_score = 0.8;
    bool ok2 = queue.enqueue_candidate(c2);
    TEST_ASSERT(!ok2, "Duplicate submission must be rejected / suppressed");
    TEST_ASSERT(queue.size() == 1, "Queue size must remain 1");
    TEST_ASSERT(queue.get_metrics().duplicates_suppressed == 1, "duplicates_suppressed must be 1");
    return true;
}

// Test 5: Duplicate candidate with higher score
bool test_5_duplicate_candidate_higher_score() {
    PrefetchConfig config;
    config.duplicate_policy = DuplicatePolicy::RETAIN_HIGHEST_PRIORITY;
    std::string test_dir = "results/phase_3_prefetch_scheduling/test5_dup_high";
    PrefetchCandidateQueue queue(config, 10, 64, test_dir);

    PrefetchCandidate c1;
    c1.request_id = 0;
    c1.token_id = 0;
    c1.target_stage_id = 1;
    c1.expert_id = 5;
    c1.prediction_rank = 3;
    c1.prediction_score = 0.4;
    queue.enqueue_candidate(c1);

    // Re-submit with higher score and better rank
    PrefetchCandidate c2 = c1;
    c2.prediction_rank = 1;
    c2.prediction_score = 0.95;
    bool ok2 = queue.enqueue_candidate(c2);
    TEST_ASSERT(ok2, "Duplicate update with higher priority must succeed");
    TEST_ASSERT(queue.size() == 1, "Queue size must remain 1");

    auto selected = queue.select_ready_candidates(1);
    TEST_ASSERT(selected.size() == 1, "Must select 1 candidate");
    TEST_ASSERT(selected[0].prediction_score == 0.95, "Score must be updated to 0.95");
    TEST_ASSERT(selected[0].prediction_rank == 1, "Rank must be updated to 1");
    return true;
}

// Test 6: Deterministic priority ordering
bool test_6_deterministic_priority_ordering() {
    PrefetchConfig config;
    std::string test_dir = "results/phase_3_prefetch_scheduling/test6_priority";
    PrefetchCandidateQueue queue(config, 10, 64, test_dir);

    PrefetchCandidate c1;
    c1.request_id = 0; c1.token_id = 0; c1.target_stage_id = 1;
    c1.expert_id = 1; c1.prediction_rank = 3; c1.prediction_score = 0.2;

    PrefetchCandidate c2;
    c2.request_id = 0; c2.token_id = 0; c2.target_stage_id = 1;
    c2.expert_id = 2; c2.prediction_rank = 1; c2.prediction_score = 0.8;

    PrefetchCandidate c3;
    c3.request_id = 0; c3.token_id = 0; c3.target_stage_id = 1;
    c3.expert_id = 3; c3.prediction_rank = 2; c3.prediction_score = 0.5;

    queue.enqueue_candidate(c1);
    queue.enqueue_candidate(c2);
    queue.enqueue_candidate(c3);

    auto sel = queue.select_ready_candidates(3);
    TEST_ASSERT(sel.size() == 3, "Must select 3 candidates");
    TEST_ASSERT(sel[0].expert_id == 2, "First must be highest score (exp 2, score 0.8)");
    TEST_ASSERT(sel[1].expert_id == 3, "Second must be score 0.5 (exp 3)");
    TEST_ASSERT(sel[2].expert_id == 1, "Third must be score 0.2 (exp 1)");
    return true;
}

// Test 7: Different target-stage deadlines
bool test_7_different_target_stage_deadlines() {
    PrefetchConfig config;
    std::string test_dir = "results/phase_3_prefetch_scheduling/test7_deadline";
    PrefetchCandidateQueue queue(config, 10, 64, test_dir);

    // Candidate for stage 5 with high score
    PrefetchCandidate c_late;
    c_late.request_id = 0; c_late.token_id = 0; c_late.target_stage_id = 5;
    c_late.expert_id = 1; c_late.prediction_rank = 1; c_late.prediction_score = 0.99;

    // Candidate for stage 2 with low score
    PrefetchCandidate c_early;
    c_early.request_id = 0; c_early.token_id = 0; c_early.target_stage_id = 2;
    c_early.expert_id = 2; c_early.prediction_rank = 4; c_early.prediction_score = 0.10;

    queue.enqueue_candidate(c_late);
    queue.enqueue_candidate(c_early);

    auto sel = queue.select_ready_candidates(2);
    TEST_ASSERT(sel.size() == 2, "Must select 2");
    TEST_ASSERT(sel[0].expert_id == 2, "Earlier deadline stage 2 must precede stage 5");
    TEST_ASSERT(sel[1].expert_id == 1, "Later deadline stage 5 comes second");
    return true;
}

// Test 8: Queue capacity enforcement
bool test_8_queue_capacity_enforcement() {
    PrefetchConfig config;
    config.queue_capacity = 3;
    config.overflow_policy = OverflowPolicy::REJECT_NEW;
    std::string test_dir = "results/phase_3_prefetch_scheduling/test8_cap";
    PrefetchCandidateQueue queue(config, 10, 64, test_dir);

    for (int i = 0; i < 3; i = i + 1) {
        PrefetchCandidate c;
        c.request_id = 0; c.token_id = 0; c.target_stage_id = 1;
        c.expert_id = i; c.prediction_rank = i + 1; c.prediction_score = 0.8;
        TEST_ASSERT(queue.enqueue_candidate(c), "Must admit up to capacity");
    }

    TEST_ASSERT(queue.size() == 3, "Queue size must be 3");

    // 4th candidate must be rejected
    PrefetchCandidate c4;
    c4.request_id = 0; c4.token_id = 0; c4.target_stage_id = 1;
    c4.expert_id = 3; c4.prediction_rank = 1; c4.prediction_score = 0.99;
    bool ok4 = queue.enqueue_candidate(c4);
    TEST_ASSERT(!ok4, "Must reject when capacity exceeded under REJECT_NEW");
    TEST_ASSERT(queue.size() == 3, "Queue size remains 3");
    TEST_ASSERT(queue.get_metrics().capacity_rejected == 1, "capacity_rejected must be 1");
    return true;
}

// Test 9: Overflow policy reject lowest priority
bool test_9_overflow_policy() {
    PrefetchConfig config;
    config.queue_capacity = 2;
    config.overflow_policy = OverflowPolicy::REJECT_LOWEST_PRIORITY;
    std::string test_dir = "results/phase_3_prefetch_scheduling/test9_overflow";
    PrefetchCandidateQueue queue(config, 10, 64, test_dir);

    PrefetchCandidate c1;
    c1.request_id = 0; c1.token_id = 0; c1.target_stage_id = 1;
    c1.expert_id = 10; c1.prediction_rank = 2; c1.prediction_score = 0.3;

    PrefetchCandidate c2;
    c2.request_id = 0; c2.token_id = 0; c2.target_stage_id = 1;
    c2.expert_id = 20; c2.prediction_rank = 1; c2.prediction_score = 0.6;

    queue.enqueue_candidate(c1);
    queue.enqueue_candidate(c2);
    TEST_ASSERT(queue.size() == 2, "Size is 2");

    // High priority candidate: should evict c1 (score 0.3)
    PrefetchCandidate c3;
    c3.request_id = 0; c3.token_id = 0; c3.target_stage_id = 1;
    c3.expert_id = 30; c3.prediction_rank = 1; c3.prediction_score = 0.9;

    bool ok3 = queue.enqueue_candidate(c3);
    TEST_ASSERT(ok3, "Higher priority candidate must cause eviction and admission");
    TEST_ASSERT(queue.size() == 2, "Size must remain 2");
    TEST_ASSERT(queue.get_metrics().overflow_evicted == 1, "overflow_evicted must be 1");

    auto sel = queue.select_ready_candidates(2);
    TEST_ASSERT(sel[0].expert_id == 30, "Highest priority exp 30 must be present");
    TEST_ASSERT(sel[1].expert_id == 20, "Retained exp 20 must be present");
    return true;
}

// Test 10: Candidate cancellation
bool test_10_candidate_cancellation() {
    PrefetchConfig config;
    std::string test_dir = "results/phase_3_prefetch_scheduling/test10_cancel";
    PrefetchCandidateQueue queue(config, 10, 64, test_dir);

    PrefetchCandidate c1;
    c1.request_id = 7; c1.token_id = 0; c1.target_stage_id = 1;
    c1.expert_id = 1; c1.prediction_rank = 1; c1.prediction_score = 0.8;
    queue.enqueue_candidate(c1);

    PrefetchCandidate c2;
    c2.request_id = 7; c2.token_id = 0; c2.target_stage_id = 2;
    c2.expert_id = 2; c2.prediction_rank = 1; c2.prediction_score = 0.8;
    queue.enqueue_candidate(c2);

    TEST_ASSERT(queue.size() == 2, "Size must be 2");
    int purged = queue.purge_request(7);
    TEST_ASSERT(purged == 2, "Must purge 2 candidates");
    TEST_ASSERT(queue.empty(), "Queue must be empty after purge");
    TEST_ASSERT(queue.get_metrics().candidates_cancelled == 2, "cancelled count must be 2");
    return true;
}

// Test 11: Target-stage reconciliation
bool test_11_target_stage_reconciliation() {
    PrefetchConfig config;
    std::string test_dir = "results/phase_3_prefetch_scheduling/test11_reconcile";
    PrefetchCandidateQueue queue(config, 10, 64, test_dir);

    // Candidates for stage 2: experts 10, 20, 30
    for (int exp : {10, 20, 30}) {
        PrefetchCandidate c;
        c.request_id = 0; c.token_id = 0; c.target_stage_id = 2;
        c.expert_id = exp; c.prediction_rank = 1; c.prediction_score = 0.8;
        queue.enqueue_candidate(c);
    }

    TEST_ASSERT(queue.size() == 3, "Size must be 3");

    // Router actual experts: {10, 25, 30} -> hits: 10, 30; miss: 20
    int rec = queue.reconcile_stage(0, 0, 2, {10, 25, 30});
    TEST_ASSERT(rec == 3, "All 3 stage candidates must be reconciled");
    TEST_ASSERT(queue.empty(), "All stage 2 candidates retired from active queue");

    const auto& m = queue.get_metrics();
    TEST_ASSERT(m.reconciled_hits == 2, "reconciled_hits must be 2");
    TEST_ASSERT(m.expired_misses == 1, "expired_misses must be 1");
    return true;
}

// Test 12: Request cleanup
bool test_12_request_cleanup() {
    PrefetchConfig config;
    std::string test_dir = "results/phase_3_prefetch_scheduling/test12_req_clean";
    PrefetchCandidateQueue queue(config, 10, 64, test_dir);

    // Req 1 and Req 2
    PrefetchCandidate c1;
    c1.request_id = 1; c1.token_id = 0; c1.target_stage_id = 1;
    c1.expert_id = 5; c1.prediction_rank = 1; c1.prediction_score = 0.8;
    queue.enqueue_candidate(c1);

    PrefetchCandidate c2;
    c2.request_id = 2; c2.token_id = 0; c2.target_stage_id = 1;
    c2.expert_id = 8; c2.prediction_rank = 1; c2.prediction_score = 0.8;
    queue.enqueue_candidate(c2);

    TEST_ASSERT(queue.size() == 2, "Size must be 2");

    queue.purge_request(1);
    TEST_ASSERT(queue.size() == 1, "Size must be 1 after purging Req 1");

    auto sel = queue.select_ready_candidates(1);
    TEST_ASSERT(sel[0].request_id == 2, "Req 2 must remain in queue");
    return true;
}

// Test 13: Multi-request isolation
bool test_13_multi_request_isolation() {
    PrefetchConfig config;
    std::string test_dir = "results/phase_3_prefetch_scheduling/test13_isolation";
    PrefetchCandidateQueue queue(config, 10, 64, test_dir);

    // Both requests predict the same expert for same stage
    PrefetchCandidate c1;
    c1.request_id = 1; c1.token_id = 0; c1.target_stage_id = 3;
    c1.expert_id = 15; c1.prediction_rank = 1; c1.prediction_score = 0.7;
    queue.enqueue_candidate(c1);

    PrefetchCandidate c2;
    c2.request_id = 2; c2.token_id = 0; c2.target_stage_id = 3;
    c2.expert_id = 15; c2.prediction_rank = 1; c2.prediction_score = 0.7;
    queue.enqueue_candidate(c2);

    TEST_ASSERT(queue.size() == 2, "Both must be admitted independently (no collision)");

    // Reconcile Req 1 only
    queue.reconcile_stage(1, 0, 3, {15});
    TEST_ASSERT(queue.size() == 1, "Req 2 remains in queue");

    auto sel = queue.select_ready_candidates(1);
    TEST_ASSERT(sel[0].request_id == 2, "Req 2 candidate remains untouched");
    return true;
}

// Test 14: Multi-token isolation
bool test_14_multi_token_isolation() {
    PrefetchConfig config;
    std::string test_dir = "results/phase_3_prefetch_scheduling/test14_token_iso";
    PrefetchCandidateQueue queue(config, 10, 64, test_dir);

    PrefetchCandidate c_tok0;
    c_tok0.request_id = 0; c_tok0.token_id = 0; c_tok0.target_stage_id = 2;
    c_tok0.expert_id = 10; c_tok0.prediction_rank = 1; c_tok0.prediction_score = 0.8;
    queue.enqueue_candidate(c_tok0);

    PrefetchCandidate c_tok1;
    c_tok1.request_id = 0; c_tok1.token_id = 1; c_tok1.target_stage_id = 2;
    c_tok1.expert_id = 10; c_tok1.prediction_rank = 1; c_tok1.prediction_score = 0.8;
    queue.enqueue_candidate(c_tok1);

    TEST_ASSERT(queue.size() == 2, "Both tokens admitted independently");

    queue.reconcile_stage(0, 0, 2, {10});
    TEST_ASSERT(queue.size() == 1, "Token 1 remains active after Token 0 reconciled");
    return true;
}

// Test 15: Same expert across different target stages
bool test_15_same_expert_different_target_stages() {
    PrefetchConfig config;
    std::string test_dir = "results/phase_3_prefetch_scheduling/test15_stages";
    PrefetchCandidateQueue queue(config, 10, 64, test_dir);

    PrefetchCandidate c_stg1;
    c_stg1.request_id = 0; c_stg1.token_id = 0; c_stg1.target_stage_id = 1;
    c_stg1.expert_id = 7; c_stg1.prediction_rank = 1; c_stg1.prediction_score = 0.8;
    queue.enqueue_candidate(c_stg1);

    PrefetchCandidate c_stg5;
    c_stg5.request_id = 0; c_stg5.token_id = 0; c_stg5.target_stage_id = 5;
    c_stg5.expert_id = 7; c_stg5.prediction_rank = 1; c_stg5.prediction_score = 0.8;
    queue.enqueue_candidate(c_stg5);

    TEST_ASSERT(queue.size() == 2, "Both stages admitted independently");

    queue.reconcile_stage(0, 0, 1, {7});
    TEST_ASSERT(queue.size() == 1, "Stage 5 candidate remains active");
    return true;
}

// Test 16: Invalid candidate handling
bool test_16_invalid_candidate_handling() {
    PrefetchConfig config;
    std::string test_dir = "results/phase_3_prefetch_scheduling/test16_invalid";
    PrefetchCandidateQueue queue(config, 10, 64, test_dir);

    // Negative req
    PrefetchCandidate bad1;
    bad1.request_id = -1; bad1.token_id = 0; bad1.target_stage_id = 1; bad1.expert_id = 0; bad1.prediction_rank = 1;
    TEST_ASSERT(!queue.enqueue_candidate(bad1), "Negative request_id must be rejected");

    // Out of range expert
    PrefetchCandidate bad2;
    bad2.request_id = 0; bad2.token_id = 0; bad2.target_stage_id = 1; bad2.expert_id = 64; bad2.prediction_rank = 1;
    TEST_ASSERT(!queue.enqueue_candidate(bad2), "Expert >= num_experts must be rejected");

    // Invalid rank
    PrefetchCandidate bad3;
    bad3.request_id = 0; bad3.token_id = 0; bad3.target_stage_id = 1; bad3.expert_id = 0; bad3.prediction_rank = 0;
    TEST_ASSERT(!queue.enqueue_candidate(bad3), "Rank < 1 must be rejected");

    TEST_ASSERT(queue.empty(), "Queue must remain empty");
    TEST_ASSERT(queue.get_metrics().invalid_rejected == 3, "invalid_rejected must be 3");
    return true;
}

// Test 17: Non-finite score handling
bool test_17_non_finite_score_handling() {
    PrefetchConfig config;
    std::string test_dir = "results/phase_3_prefetch_scheduling/test17_non_finite";
    PrefetchCandidateQueue queue(config, 10, 64, test_dir);

    PrefetchCandidate nan_cand;
    nan_cand.request_id = 0; nan_cand.token_id = 0; nan_cand.target_stage_id = 1;
    nan_cand.expert_id = 5; nan_cand.prediction_rank = 1;
    nan_cand.prediction_score = std::numeric_limits<double>::quiet_NaN();
    TEST_ASSERT(!queue.enqueue_candidate(nan_cand), "NaN score must be rejected");

    PrefetchCandidate inf_cand;
    inf_cand.request_id = 0; inf_cand.token_id = 0; inf_cand.target_stage_id = 1;
    inf_cand.expert_id = 5; inf_cand.prediction_rank = 1;
    inf_cand.prediction_score = std::numeric_limits<double>::infinity();
    TEST_ASSERT(!queue.enqueue_candidate(inf_cand), "Infinity score must be rejected");

    TEST_ASSERT(queue.get_metrics().invalid_rejected == 2, "invalid_rejected must be 2");
    return true;
}

// Test 18: Deterministic repeated execution
bool test_18_deterministic_repeated_execution() {
    PrefetchConfig config;
    std::string dir_a = "results/phase_3_prefetch_scheduling/test18_run_a";
    std::string dir_b = "results/phase_3_prefetch_scheduling/test18_run_b";

    PrefetchCandidateQueue q_a(config, 10, 64, dir_a);
    PrefetchCandidateQueue q_b(config, 10, 64, dir_b);

    for (int i = 0; i < 5; i = i + 1) {
        PrefetchCandidate c;
        c.request_id = 0; c.token_id = 0; c.target_stage_id = i;
        c.expert_id = (i * 7) % 64; c.prediction_rank = 1; c.prediction_score = 0.5 + 0.05 * i;
        q_a.enqueue_candidate(c);
        q_b.enqueue_candidate(c);
    }

    auto sel_a = q_a.select_ready_candidates(5);
    auto sel_b = q_b.select_ready_candidates(5);

    TEST_ASSERT(sel_a.size() == sel_b.size(), "Selection sizes must match");
    for (size_t i = 0; i < sel_a.size(); ++i) {
        TEST_ASSERT(sel_a[i].expert_id == sel_b[i].expert_id, "Expert IDs must match");
        TEST_ASSERT(sel_a[i].target_stage_id == sel_b[i].target_stage_id, "Stage IDs must match");
    }
    return true;
}

// Test 19: Queue-disabled behavior
bool test_19_queue_disabled_behavior() {
    PrefetchConfig config;
    config.enabled = true;
    config.mode = PrefetchMode::PREDICTION_ONLY;
    config.predictor_type = PredictorType::ST_MOE;
    config.queue_enabled = false; // Queue explicitly disabled
    config.budget = 4;
    config.prediction_horizon = 1;

    std::string test_dir = "results/phase_3_prefetch_scheduling/test19_disabled";
    auto engine = std::make_shared<LivePredictionEngine>(config, 5, 32, 2, test_dir);
    TEST_ASSERT(engine->get_candidate_queue() == nullptr, "Candidate queue must be nullptr when disabled");

    engine->on_stage_routing_start(0, 0, 0);
    engine->on_stage_routing_complete(0, 0, 0, {1, 2});
    TEST_ASSERT(engine->get_prediction_count() == 1, "Prediction still functions");
    engine->finalize(0.01);
    return true;
}

// Test 20: Phase 2C regression verification
bool test_20_phase2c_regression() {
    PrefetchConfig config;
    config.enabled = true;
    config.mode = PrefetchMode::PREDICTION_ONLY;
    config.predictor_type = PredictorType::ST_MOE;
    config.queue_enabled = true;
    config.budget = 4;
    config.prediction_horizon = 1;

    std::string test_dir = "results/phase_3_prefetch_scheduling/test20_regression";
    auto engine = std::make_shared<LivePredictionEngine>(config, 5, 32, 2, test_dir);
    TEST_ASSERT(engine->get_candidate_queue() != nullptr, "Queue must be active");

    engine->on_request_start(0);

    // Stage 0
    engine->on_stage_routing_start(0, 0, 0);
    engine->on_stage_routing_complete(0, 0, 0, {3, 7});

    // Stage 1
    engine->on_stage_routing_start(0, 0, 1);
    engine->on_stage_routing_complete(0, 0, 1, {3, 10});

    // Stage 2
    engine->on_stage_routing_start(0, 0, 2);
    engine->on_stage_routing_complete(0, 0, 2, {3, 12});

    TEST_ASSERT(engine->get_prediction_count() == 3, "Must have 3 prediction events");
    TEST_ASSERT(engine->get_update_count() == 3, "Must have 3 update events");

    auto queue = engine->get_candidate_queue();
    TEST_ASSERT(queue != nullptr, "Queue must be non-null");

    const auto& q_metrics = queue->get_metrics();
    TEST_ASSERT(q_metrics.total_candidates_received > 0, "Queue must receive candidates");
    TEST_ASSERT(q_metrics.candidates_admitted > 0, "Queue must admit candidates");

    engine->finalize(0.02);
    TEST_ASSERT(std::filesystem::exists(test_dir + "/live_predictions.csv"), "Live predictions CSV exists");
    TEST_ASSERT(std::filesystem::exists(test_dir + "/live_prediction_summary.json"), "Summary JSON exists");
    return true;
}

} // namespace llm_system

int main() {
    std::cout << "======================================================================\n";
    std::cout << "Phase 3: Prefetch Candidate Queue & Scheduling Test Suite\n";
    std::cout << "Executable: test_prefetch_scheduler\n";
    std::cout << "======================================================================\n";

    RUN_TEST(llm_system::test_1_empty_queue);
    RUN_TEST(llm_system::test_2_single_candidate_admission);
    RUN_TEST(llm_system::test_3_multiple_candidate_admission);
    RUN_TEST(llm_system::test_4_duplicate_suppression);
    RUN_TEST(llm_system::test_5_duplicate_candidate_higher_score);
    RUN_TEST(llm_system::test_6_deterministic_priority_ordering);
    RUN_TEST(llm_system::test_7_different_target_stage_deadlines);
    RUN_TEST(llm_system::test_8_queue_capacity_enforcement);
    RUN_TEST(llm_system::test_9_overflow_policy);
    RUN_TEST(llm_system::test_10_candidate_cancellation);
    RUN_TEST(llm_system::test_11_target_stage_reconciliation);
    RUN_TEST(llm_system::test_12_request_cleanup);
    RUN_TEST(llm_system::test_13_multi_request_isolation);
    RUN_TEST(llm_system::test_14_multi_token_isolation);
    RUN_TEST(llm_system::test_15_same_expert_different_target_stages);
    RUN_TEST(llm_system::test_16_invalid_candidate_handling);
    RUN_TEST(llm_system::test_17_non_finite_score_handling);
    RUN_TEST(llm_system::test_18_deterministic_repeated_execution);
    RUN_TEST(llm_system::test_19_queue_disabled_behavior);
    RUN_TEST(llm_system::test_20_phase2c_regression);

    std::cout << "======================================================================\n";
    std::cout << "Test Summary: " << tests_passed << " Passed, " << tests_failed << " Failed\n";
    std::cout << "======================================================================\n";

    if (tests_failed > 0) {
        return 1;
    }
    return 0;
}
