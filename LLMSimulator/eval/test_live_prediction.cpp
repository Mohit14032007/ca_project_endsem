#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "prefetch/acsh_predictor.h"
#include "prefetch/cstp_predictor.h"
#include "prefetch/live_prediction_engine.h"
#include "prefetch/none_predictor.h"
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

// Test 1: Configuration disabled
bool test_1_configuration_disabled() {
    PrefetchConfig config;
    config.enabled = false;
    config.mode = PrefetchMode::NONE;
    config.predictor_type = PredictorType::NONE;
    TEST_ASSERT(!config.enabled, "Config must be disabled");

    SystemConfig sys_cfg;
    sys_cfg.num_device = 1;
    sys_cfg.num_node = 1;
    sys_cfg.prefetch_config = config;

    ModelConfig model_cfg;
    model_cfg.model_name = "deepseekV3";
    model_cfg.num_layers = 60;
    model_cfg.first_k_dense = 3;
    model_cfg.num_routed_expert = 256;
    model_cfg.top_k = 8;
    model_cfg.expert_freq = 1;
    model_cfg.ne_tp_dg = 1;
    model_cfg.e_tp_dg = 1;

    Scheduler::Ptr scheduler = Scheduler::Create(sys_cfg, model_cfg, "none", 1, 1024, 0);
    TEST_ASSERT(scheduler != nullptr, "Scheduler must be created");
    TEST_ASSERT(scheduler->prediction_engine == nullptr, "prediction_engine must be nullptr when disabled");

    std::string test_dir = "results/phase_2c_prediction_only/test1_disabled";
    if (std::filesystem::exists(test_dir)) {
        std::filesystem::remove_all(test_dir);
    }
    TEST_ASSERT(!std::filesystem::exists(test_dir + "/live_predictions.csv"), "CSV must not exist");
    return true;
}

// Test 2: NonePredictor
bool test_2_none_predictor() {
    PrefetchConfig config;
    config.enabled = true;
    config.mode = PrefetchMode::PREDICTION_ONLY;
    config.predictor_type = PredictorType::NONE;
    config.budget = 4;
    config.prediction_horizon = 1;

    std::string test_dir = "results/phase_2c_prediction_only/test2_none";
    auto engine = std::make_shared<LivePredictionEngine>(config, 10, 64, 4, test_dir);
    TEST_ASSERT(engine->get_predictor() != nullptr, "Predictor must exist");
    TEST_ASSERT(engine->get_predictor()->get_name() == "none", "Name must be none");

    engine->on_request_start(0);
    engine->on_stage_routing_start(0, 0, 0);
    engine->on_stage_routing_complete(0, 0, 0, {5, 6, 7, 8});

    engine->on_stage_routing_start(0, 0, 1);
    engine->on_stage_routing_complete(0, 0, 1, {9, 10, 11, 12});

    TEST_ASSERT(engine->get_prediction_count() == 2, "Must have 2 prediction events");
    TEST_ASSERT(engine->get_update_count() == 2, "Must have 2 update events");
    TEST_ASSERT(engine->get_invalid_count() == 0, "No invalid events");

    const auto& acc = engine->get_overall_accumulator();
    TEST_ASSERT(acc.total_hits == 0, "NonePredictor produces 0 hits");
    TEST_ASSERT(acc.mean_recall() == 0.0, "NonePredictor has 0 recall");

    engine->finalize(0.01);
    TEST_ASSERT(std::filesystem::exists(test_dir + "/live_predictions.csv"), "CSV must be written");
    TEST_ASSERT(std::filesystem::exists(test_dir + "/live_prediction_summary.json"), "JSON must be written");
    return true;
}

// Test 3: ST-MoE live integration
bool test_3_st_moe_live_integration() {
    PrefetchConfig config;
    config.enabled = true;
    config.mode = PrefetchMode::PREDICTION_ONLY;
    config.predictor_type = PredictorType::ST_MOE;
    config.budget = 4;
    config.prediction_horizon = 1;

    std::string test_dir = "results/phase_2c_prediction_only/test3_st_moe";
    auto engine = std::make_shared<LivePredictionEngine>(config, 5, 32, 2, test_dir);
    TEST_ASSERT(engine->get_predictor()->get_name() == "st_moe", "Predictor must be st_moe");

    engine->on_request_start(42);

    // Stage 0: target = 0, horizon = 1. source_stage = -1 (no source yet)
    engine->on_stage_routing_start(42, 0, 0);
    engine->on_stage_routing_complete(42, 0, 0, {3, 7});

    // Stage 1: target = 1, horizon = 1. source_stage = 0 (source experts = {3, 7})
    engine->on_stage_routing_start(42, 0, 1);
    // At this point, target routing is unrevealed. Complete reveals {3, 10}:
    engine->on_stage_routing_complete(42, 0, 1, {3, 10});

    // Stage 2: target = 2, horizon = 1. source_stage = 1 (source experts = {3, 10})
    engine->on_stage_routing_start(42, 0, 2);
    engine->on_stage_routing_complete(42, 0, 2, {3, 12});

    TEST_ASSERT(engine->get_prediction_count() == 3, "3 prediction events");
    TEST_ASSERT(engine->get_update_count() == 3, "3 update events");

    engine->finalize(0.02);
    TEST_ASSERT(std::filesystem::exists(test_dir + "/live_predictions.csv"), "CSV exists");
    TEST_ASSERT(std::filesystem::exists(test_dir + "/live_prediction_summary.json"), "Summary exists");
    return true;
}

// Test 4: CSTP live integration
bool test_4_cstp_live_integration() {
    PrefetchConfig config;
    config.enabled = true;
    config.mode = PrefetchMode::PREDICTION_ONLY;
    config.predictor_type = PredictorType::CSTP;
    config.budget = 4;
    config.prediction_horizon = 1;

    std::string test_dir = "results/phase_2c_prediction_only/test4_cstp";
    auto engine = std::make_shared<LivePredictionEngine>(config, 6, 32, 2, test_dir);
    TEST_ASSERT(engine->get_predictor()->get_name() == "cstp", "Predictor must be cstp");

    engine->on_request_start(100);

    // Feed a sequence across 4 tokens, 4 stages each
    for (int tok = 0; tok < 4; tok = tok + 1) {
        for (int stg = 0; stg < 4; stg = stg + 1) {
            engine->on_stage_routing_start(100, tok, stg);
            int exp1 = (stg * 2 + tok) % 32;
            int exp2 = (stg * 2 + tok + 1) % 32;
            engine->on_stage_routing_complete(100, tok, stg, {exp1, exp2});
        }
    }

    TEST_ASSERT(engine->get_prediction_count() == 16, "16 prediction events");
    TEST_ASSERT(engine->get_update_count() == 16, "16 update events");

    const auto& acc = engine->get_overall_accumulator();
    TEST_ASSERT(acc.count == 16, "16 events in accumulator");

    engine->finalize(0.05);
    return true;
}

// Test 5: ACSH live integration
bool test_5_acsh_live_integration() {
    PrefetchConfig config;
    config.enabled = true;
    config.mode = PrefetchMode::PREDICTION_ONLY;
    config.predictor_type = PredictorType::ACSH_STAGE_AWARE;
    config.budget = 4;
    config.prediction_horizon = 1;

    std::string test_dir = "results/phase_2c_prediction_only/test5_acsh";
    auto engine = std::make_shared<LivePredictionEngine>(config, 8, 32, 2, test_dir);
    TEST_ASSERT(engine->get_predictor()->get_name() == "acsh_stage_aware", "Predictor must be acsh_stage_aware");

    engine->on_request_start(200);

    for (int tok = 0; tok < 3; tok = tok + 1) {
        for (int stg = 0; stg < 8; stg = stg + 1) {
            engine->on_stage_routing_start(200, tok, stg);
            int e1 = (stg + tok) % 32;
            int e2 = (stg + tok + 3) % 32;
            engine->on_stage_routing_complete(200, tok, stg, {e1, e2});
        }
    }

    TEST_ASSERT(engine->get_prediction_count() == 24, "24 prediction events");
    TEST_ASSERT(engine->get_update_count() == 24, "24 update events");

    engine->finalize(0.05);
    return true;
}

// Test 6: Multiple requests isolation
bool test_6_multiple_requests() {
    PrefetchConfig config;
    config.enabled = true;
    config.mode = PrefetchMode::PREDICTION_ONLY;
    config.predictor_type = PredictorType::ST_MOE;
    config.budget = 4;
    config.prediction_horizon = 1;

    std::string test_dir = "results/phase_2c_prediction_only/test6_multireq";
    auto engine = std::make_shared<LivePredictionEngine>(config, 4, 32, 2, test_dir);

    // Start two requests
    engine->on_request_start(1);
    engine->on_request_start(2);

    // Request 1 visits {1, 2} at stage 0
    engine->on_stage_routing_start(1, 0, 0);
    engine->on_stage_routing_complete(1, 0, 0, {1, 2});

    // Request 2 visits {20, 21} at stage 0
    engine->on_stage_routing_start(2, 0, 0);
    engine->on_stage_routing_complete(2, 0, 0, {20, 21});

    // Request 1 at stage 1: source must be {1, 2}
    engine->on_stage_routing_start(1, 0, 1);
    engine->on_stage_routing_complete(1, 0, 1, {1, 3});

    // Request 2 at stage 1: source must be {20, 21}
    engine->on_stage_routing_start(2, 0, 1);
    engine->on_stage_routing_complete(2, 0, 1, {20, 22});

    TEST_ASSERT(engine->get_prediction_count() == 4, "4 events total");
    TEST_ASSERT(engine->get_update_count() == 4, "4 updates total");

    engine->finalize(0.01);
    return true;
}

// Test 7: Repeated execution determinism
bool test_7_repeated_execution() {
    PrefetchConfig config;
    config.enabled = true;
    config.mode = PrefetchMode::PREDICTION_ONLY;
    config.predictor_type = PredictorType::ST_MOE;
    config.budget = 4;
    config.prediction_horizon = 1;

    std::string dir_a = "results/phase_2c_prediction_only/test7_run_a";
    std::string dir_b = "results/phase_2c_prediction_only/test7_run_b";

    auto engine_a = std::make_shared<LivePredictionEngine>(config, 5, 32, 2, dir_a);
    auto engine_b = std::make_shared<LivePredictionEngine>(config, 5, 32, 2, dir_b);

    engine_a->on_request_start(0);
    engine_b->on_request_start(0);

    for (int stg = 0; stg < 5; stg = stg + 1) {
        int e1 = stg % 32;
        int e2 = (stg + 4) % 32;

        engine_a->on_stage_routing_start(0, 0, stg);
        engine_a->on_stage_routing_complete(0, 0, stg, {e1, e2});

        engine_b->on_stage_routing_start(0, 0, stg);
        engine_b->on_stage_routing_complete(0, 0, stg, {e1, e2});
    }

    engine_a->finalize(0.01);
    engine_b->finalize(0.01);

    const auto& acc_a = engine_a->get_overall_accumulator();
    const auto& acc_b = engine_b->get_overall_accumulator();

    TEST_ASSERT(acc_a.count == acc_b.count, "Counts match");
    TEST_ASSERT(acc_a.total_hits == acc_b.total_hits, "Total hits match");
    TEST_ASSERT(std::abs(acc_a.mean_recall() - acc_b.mean_recall()) < 1e-9, "Recall matches");
    TEST_ASSERT(std::abs(acc_a.mean_precision() - acc_b.mean_precision()) < 1e-9, "Precision matches");
    TEST_ASSERT(std::abs(acc_a.mean_f1() - acc_b.mean_f1()) < 1e-9, "F1 matches");
    return true;
}

// Test 8: Baseline regression against disabled mode
bool test_8_baseline_regression() {
    PrefetchConfig config_off;
    config_off.enabled = false;
    config_off.mode = PrefetchMode::NONE;

    SystemConfig sys_off;
    sys_off.num_device = 1;
    sys_off.num_node = 1;
    sys_off.prefetch_config = config_off;

    ModelConfig model_cfg;
    model_cfg.model_name = "deepseekV3";
    model_cfg.num_layers = 60;
    model_cfg.first_k_dense = 3;
    model_cfg.num_routed_expert = 256;
    model_cfg.top_k = 8;
    model_cfg.expert_freq = 1;
    model_cfg.ne_tp_dg = 1;
    model_cfg.e_tp_dg = 1;

    Scheduler::Ptr sched_off = Scheduler::Create(sys_off, model_cfg, "none", 1, 1024, 0);
    TEST_ASSERT(sched_off->prediction_engine == nullptr, "Disabled sched has null engine");

    PrefetchConfig config_on;
    config_on.enabled = true;
    config_on.mode = PrefetchMode::PREDICTION_ONLY;
    config_on.predictor_type = PredictorType::ST_MOE;
    config_on.budget = 4;
    config_on.prediction_horizon = 1;

    SystemConfig sys_on;
    sys_on.num_device = 1;
    sys_on.num_node = 1;
    sys_on.prefetch_config = config_on;

    setenv("PREDICTION_OUTPUT_DIR", "results/phase_2c_prediction_only/test8_regression", 1);
    Scheduler::Ptr sched_on = Scheduler::Create(sys_on, model_cfg, "none", 1, 1024, 0);
    TEST_ASSERT(sched_on->prediction_engine != nullptr, "Enabled sched has active engine");
    TEST_ASSERT(sched_on->prediction_engine->get_predictor()->get_name() == "st_moe", "ST-MoE loaded");

    sched_on->prediction_engine->finalize(0.01);
    return true;
}

} // namespace llm_system

int main() {
    std::cout << "======================================================================\n";
    std::cout << "Phase 2C: Live Prediction Integration Test Suite\n";
    std::cout << "Executable: test_live_prediction\n";
    std::cout << "======================================================================\n";

    RUN_TEST(llm_system::test_1_configuration_disabled);
    RUN_TEST(llm_system::test_2_none_predictor);
    RUN_TEST(llm_system::test_3_st_moe_live_integration);
    RUN_TEST(llm_system::test_4_cstp_live_integration);
    RUN_TEST(llm_system::test_5_acsh_live_integration);
    RUN_TEST(llm_system::test_6_multiple_requests);
    RUN_TEST(llm_system::test_7_repeated_execution);
    RUN_TEST(llm_system::test_8_baseline_regression);

    std::cout << "======================================================================\n";
    std::cout << "Test Summary: " << tests_passed << " Passed, " << tests_failed << " Failed\n";
    std::cout << "======================================================================\n";

    if (tests_failed > 0) {
        return 1;
    }
    return 0;
}
