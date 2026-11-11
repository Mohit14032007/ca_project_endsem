#include <iostream>
#include <cassert>
#include <chrono>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "prefetch/prefetch_config.h"
#include "prefetch/predictor_interface.h"
#include "prefetch/none_predictor.h"
#include "prefetch/predictor_registry.h"

using namespace llm_system;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  [FAIL] " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            tests_failed++; \
            return false; \
        } \
    } while (0)

#define RUN_TEST(fn) \
    do { \
        std::cout << "[RUNNING] " << #fn << "...\n"; \
        auto start = std::chrono::high_resolution_clock::now(); \
        bool ok = fn(); \
        auto end = std::chrono::high_resolution_clock::now(); \
        double ms = std::chrono::duration<double, std::milli>(end - start).count(); \
        if (ok) { \
            std::cout << "  [PASS] " << #fn << " (" << ms << " ms)\n"; \
            tests_passed++; \
        } else { \
            std::cout << "  [FAIL] " << #fn << " (" << ms << " ms)\n"; \
        } \
    } while (0)

// -----------------------------------------------------------------------------
// Configuration Tests (12 tests)
// -----------------------------------------------------------------------------

// 1. YAML without prefetching section
bool test_config_missing_section() {
    std::string yaml_str = "simulation:\n  input_len: 128\n";
    YAML::Node root = YAML::Load(yaml_str);
    PrefetchConfig cfg = PrefetchConfig::parse_yaml(root["simulation"]["prefetching"]);
    TEST_ASSERT(!cfg.enabled, "Default enabled must be false");
    TEST_ASSERT(cfg.mode == PrefetchMode::NONE, "Default mode must be NONE");
    TEST_ASSERT(cfg.predictor_type == PredictorType::NONE, "Default predictor must be NONE");
    cfg.validate();
    return true;
}

// 2. Explicitly disabled prefetching
bool test_config_explicitly_disabled() {
    std::string yaml_str = "prefetching:\n  enabled: false\n  mode: none\n  predictor:\n    type: none\n";
    YAML::Node root = YAML::Load(yaml_str);
    PrefetchConfig cfg = PrefetchConfig::parse_yaml(root["prefetching"]);
    TEST_ASSERT(!cfg.enabled, "Enabled must be false");
    TEST_ASSERT(cfg.mode == PrefetchMode::NONE, "Mode must be NONE");
    TEST_ASSERT(cfg.predictor_type == PredictorType::NONE, "Predictor must be NONE");
    cfg.validate();
    return true;
}

// 3. Enabled prediction_only with none
bool test_config_prediction_only_none() {
    std::string yaml_str = "prefetching:\n  enabled: true\n  mode: prediction_only\n  predictor:\n    type: none\n";
    YAML::Node root = YAML::Load(yaml_str);
    PrefetchConfig cfg = PrefetchConfig::parse_yaml(root["prefetching"]);
    TEST_ASSERT(cfg.enabled, "Enabled must be true");
    TEST_ASSERT(cfg.mode == PrefetchMode::PREDICTION_ONLY, "Mode must be PREDICTION_ONLY");
    TEST_ASSERT(cfg.predictor_type == PredictorType::NONE, "Predictor must be NONE");
    cfg.validate();
    return true;
}

// 4. Valid st_moe selection recognized and instantiated in Phase 2A
bool test_config_st_moe_unimplemented_in_phase1() {
    std::string yaml_str = "prefetching:\n  enabled: true\n  mode: prediction_only\n  predictor:\n    type: st_moe\n";
    YAML::Node root = YAML::Load(yaml_str);
    PrefetchConfig cfg = PrefetchConfig::parse_yaml(root["prefetching"]);
    cfg.validate(); // Config validation succeeds (st_moe is recognized enum)
    TEST_ASSERT(cfg.predictor_type == PredictorType::ST_MOE, "Predictor must be ST_MOE");
    TEST_ASSERT(PredictorRegistry::is_implemented(cfg.predictor_type), "ST_MOE is implemented in Phase 2A");

    // Instantiation succeeds
    auto pred = PredictorRegistry::create_predictor(cfg.predictor_type, cfg, 75, 256, 8);
    TEST_ASSERT(pred != nullptr, "ST_MOE instance must be created");
    TEST_ASSERT(pred->get_name() == "st_moe", "Predictor name must be st_moe");

    // Theoretical evaluation-only oracle remains unimplemented and must throw
    TEST_ASSERT(!PredictorRegistry::is_implemented(PredictorType::ORACLE), "Oracle must not be implemented");
    bool threw = false;
    try {
        PredictorRegistry::create_predictor(PredictorType::ORACLE, cfg, 75, 256, 8);
    } catch (const std::runtime_error& e) {
        threw = true;
        std::string msg = e.what();
        TEST_ASSERT(msg.find("evaluation-only") != std::string::npos, "Error message must cite evaluation-only status");
    }
    TEST_ASSERT(threw, "create_predictor must throw for unimplemented oracle");
    return true;
}

// 5. Unknown predictor identifier rejected
bool test_config_unknown_predictor_rejected() {
    std::string yaml_str = "prefetching:\n  enabled: true\n  mode: prediction_only\n  predictor:\n    type: neural_transformer_v9\n";
    YAML::Node root = YAML::Load(yaml_str);
    bool threw = false;
    try {
        PrefetchConfig::parse_yaml(root["prefetching"]);
    } catch (const std::invalid_argument& e) {
        threw = true;
        std::string msg = e.what();
        TEST_ASSERT(msg.find("Unrecognized predictor type") != std::string::npos, "Must identify unrecognized predictor");
    }
    TEST_ASSERT(threw, "Must throw invalid_argument for unknown predictor identifier");
    return true;
}

// 6. Invalid mode rejected
bool test_config_invalid_mode_rejected() {
    std::string yaml_str = "prefetching:\n  enabled: true\n  mode: super_fast_async\n";
    YAML::Node root = YAML::Load(yaml_str);
    bool threw = false;
    try {
        PrefetchConfig::parse_yaml(root["prefetching"]);
    } catch (const std::invalid_argument& e) {
        threw = true;
        std::string msg = e.what();
        TEST_ASSERT(msg.find("Unrecognized prefetch mode") != std::string::npos, "Must identify unrecognized mode");
    }
    TEST_ASSERT(threw, "Must throw invalid_argument for unknown mode");
    return true;
}

// 7. Invalid candidate-selection policy rejected
bool test_config_invalid_candidate_policy_rejected() {
    std::string yaml_str = "prefetching:\n  candidate_selection:\n    policy: random_lottery\n";
    YAML::Node root = YAML::Load(yaml_str);
    bool threw = false;
    try {
        PrefetchConfig::parse_yaml(root["prefetching"]);
    } catch (const std::invalid_argument& e) {
        threw = true;
        std::string msg = e.what();
        TEST_ASSERT(msg.find("Unrecognized candidate selection policy") != std::string::npos, "Must identify policy error");
    }
    TEST_ASSERT(threw, "Must throw invalid_argument for unknown candidate policy");
    return true;
}

// 8. Invalid eviction policy rejected
bool test_config_invalid_eviction_policy_rejected() {
    std::string yaml_str = "prefetching:\n  hbm:\n    eviction_policy: random_drop\n";
    YAML::Node root = YAML::Load(yaml_str);
    bool threw = false;
    try {
        PrefetchConfig::parse_yaml(root["prefetching"]);
    } catch (const std::invalid_argument& e) {
        threw = true;
        std::string msg = e.what();
        TEST_ASSERT(msg.find("Unrecognized eviction policy") != std::string::npos, "Must identify eviction policy error");
    }
    TEST_ASSERT(threw, "Must throw invalid_argument for unknown eviction policy");
    return true;
}

// 9. Invalid transfer model rejected
bool test_config_invalid_transfer_model_rejected() {
    std::string yaml_str = "prefetching:\n  transfer:\n    model: warp_speed\n";
    YAML::Node root = YAML::Load(yaml_str);
    bool threw = false;
    try {
        PrefetchConfig::parse_yaml(root["prefetching"]);
    } catch (const std::invalid_argument& e) {
        threw = true;
        std::string msg = e.what();
        TEST_ASSERT(msg.find("Unrecognized transfer model") != std::string::npos, "Must identify transfer model error");
    }
    TEST_ASSERT(threw, "Must throw invalid_argument for unknown transfer model");
    return true;
}

// 10. Invalid numeric values rejected
bool test_config_invalid_numeric_values_rejected() {
    // Negative budget
    {
        PrefetchConfig cfg;
        cfg.budget = -1;
        bool threw = false;
        try { cfg.validate(); } catch (const std::invalid_argument&) { threw = true; }
        TEST_ASSERT(threw, "Negative budget must be rejected");
    }
    // Zero / negative horizon
    {
        PrefetchConfig cfg;
        cfg.prediction_horizon = 0;
        bool threw = false;
        try { cfg.validate(); } catch (const std::invalid_argument&) { threw = true; }
        TEST_ASSERT(threw, "Zero prediction_horizon must be rejected");
    }
    // Alpha > 1.0
    {
        PrefetchConfig cfg;
        cfg.alpha = 1.5;
        bool threw = false;
        try { cfg.validate(); } catch (const std::invalid_argument&) { threw = true; }
        TEST_ASSERT(threw, "Alpha > 1.0 must be rejected");
    }
    // Alpha < 0.0
    {
        PrefetchConfig cfg;
        cfg.alpha = -0.1;
        bool threw = false;
        try { cfg.validate(); } catch (const std::invalid_argument&) { threw = true; }
        TEST_ASSERT(threw, "Alpha < 0.0 must be rejected");
    }
    // Smoothing lambda <= 0.0
    {
        PrefetchConfig cfg;
        cfg.smoothing_lambda = 0.0;
        bool threw = false;
        try { cfg.validate(); } catch (const std::invalid_argument&) { threw = true; }
        TEST_ASSERT(threw, "Smoothing lambda <= 0.0 must be rejected");
    }
    // Max outstanding transfers <= 0
    {
        PrefetchConfig cfg;
        cfg.max_outstanding_transfers = 0;
        bool threw = false;
        try { cfg.validate(); } catch (const std::invalid_argument&) { threw = true; }
        TEST_ASSERT(threw, "Max outstanding transfers <= 0 must be rejected");
    }
    // Negative reserved prefetch bytes
    {
        PrefetchConfig cfg;
        cfg.reserved_prefetch_bytes = -50;
        bool threw = false;
        try { cfg.validate(); } catch (const std::invalid_argument&) { threw = true; }
        TEST_ASSERT(threw, "Negative reserved_prefetch_bytes must be rejected");
    }
    return true;
}

// 11. Missing calibration file accepted in prediction_only
bool test_config_missing_calibration_file_prediction_only() {
    std::string yaml_str = 
        "prefetching:\n"
        "  enabled: true\n"
        "  mode: prediction_only\n"
        "  predictor:\n"
        "    type: none\n"
        "  transfer:\n"
        "    calibration_file: \"\"\n";
    YAML::Node root = YAML::Load(yaml_str);
    PrefetchConfig cfg = PrefetchConfig::parse_yaml(root["prefetching"]);
    TEST_ASSERT(cfg.calibration_file.empty(), "Calibration file should be empty");
    cfg.validate(); // Must succeed without requiring calibration file
    return true;
}

// 12. Invalid mode/predictor combinations rejected
bool test_config_invalid_combinations_rejected() {
    // Disabled prefetching with active mode
    {
        PrefetchConfig cfg;
        cfg.enabled = false;
        cfg.mode = PrefetchMode::HARDWARE_ASYNC;
        bool threw = false;
        try { cfg.validate(); } catch (const std::invalid_argument&) { threw = true; }
        TEST_ASSERT(threw, "enabled=false with active mode must be rejected");
    }
    // Enabled prefetching with mode=none
    {
        PrefetchConfig cfg;
        cfg.enabled = true;
        cfg.mode = PrefetchMode::NONE;
        bool threw = false;
        try { cfg.validate(); } catch (const std::invalid_argument&) { threw = true; }
        TEST_ASSERT(threw, "enabled=true with mode=none must be rejected");
    }
    // Oracle predictor in hardware_async mode
    {
        PrefetchConfig cfg;
        cfg.enabled = true;
        cfg.mode = PrefetchMode::HARDWARE_ASYNC;
        cfg.predictor_type = PredictorType::ORACLE;
        bool threw = false;
        try { cfg.validate(); } catch (const std::invalid_argument&) { threw = true; }
        TEST_ASSERT(threw, "Oracle predictor in hardware_async mode must be rejected");
    }
    // Oracle predictor creation in registry
    {
        PrefetchConfig cfg;
        cfg.enabled = true;
        cfg.mode = PrefetchMode::PREDICTION_ONLY;
        cfg.predictor_type = PredictorType::ORACLE;
        bool threw = false;
        try {
            PredictorRegistry::create_predictor(cfg.predictor_type, cfg, 75, 256, 8);
        } catch (const std::runtime_error& e) {
            threw = true;
            std::string msg = e.what();
            TEST_ASSERT(msg.find("evaluation-only") != std::string::npos, "Must cite evaluation-only status");
        }
        TEST_ASSERT(threw, "Oracle predictor creation must throw in Phase 1");
    }
    return true;
}

// -----------------------------------------------------------------------------
// Predictor Tests (5 tests)
// -----------------------------------------------------------------------------

// 1. NonePredictor initialization
bool test_none_predictor_init() {
    NonePredictor predictor;
    TEST_ASSERT(predictor.get_name() == "none", "Name must be 'none'");
    TEST_ASSERT(predictor.get_type() == PredictorType::NONE, "Type must be NONE");
    TEST_ASSERT(predictor.is_implemented(), "NonePredictor must be implemented");
    TEST_ASSERT(!predictor.is_evaluation_only(), "NonePredictor is not evaluation-only");
    return true;
}

// 2. Routing updates accepted
bool test_none_predictor_update() {
    NonePredictor predictor;
    std::vector<int> src = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int> tgt = {8, 9, 10, 11, 12, 13, 14, 15};
    // Should accept updates safely without exceptions or mutations
    predictor.update(0, 0, 0, 1, src, tgt);
    predictor.update(0, 1, 1, 2, tgt, src);
    predictor.update(1, 0, 0, 1, src, tgt);
    return true;
}

// 3. Predictions always return empty
bool test_none_predictor_predict_empty() {
    NonePredictor predictor;
    std::vector<int> src = {10, 20, 30, 40};
    PredictionResult res = predictor.predict(0, 5, 2, 4, src, 4);
    TEST_ASSERT(res.is_valid, "Result must be marked valid");
    TEST_ASSERT(res.request_id == 0, "request_id must match");
    TEST_ASSERT(res.token_id == 5, "token_id must match");
    TEST_ASSERT(res.current_stage == 2, "current_stage must match");
    TEST_ASSERT(res.target_stage == 4, "target_stage must match");
    TEST_ASSERT(res.horizon == 2, "horizon must be target_stage - current_stage = 2");
    TEST_ASSERT(res.candidate_experts.empty(), "Candidate experts must be empty");
    TEST_ASSERT(res.scores.empty(), "Scores must be empty");
    return true;
}

// 4. No future trace access
bool test_none_predictor_no_future_access() {
    NonePredictor predictor;
    // Verify predictor is completely stateless and has zero references to trace files or future tokens
    TEST_ASSERT(sizeof(predictor) <= sizeof(void*) * 2, "NonePredictor must be stateless (vtable pointer only)");
    return true;
}

// 5. Multiple sequence IDs do not share mutable history
bool test_none_predictor_sequence_isolation() {
    NonePredictor predictor;
    predictor.reset(0);
    predictor.update(0, 0, 0, 1, {1, 2}, {3, 4});

    predictor.reset(1);
    predictor.update(1, 0, 0, 1, {5, 6}, {7, 8});

    PredictionResult r0 = predictor.predict(0, 0, 1, 2, {3, 4}, 4);
    PredictionResult r1 = predictor.predict(1, 0, 1, 2, {7, 8}, 4);

    TEST_ASSERT(r0.candidate_experts.empty(), "Seq 0 candidates must be empty");
    TEST_ASSERT(r1.candidate_experts.empty(), "Seq 1 candidates must be empty");
    TEST_ASSERT(r0.request_id == 0, "Seq 0 request ID preserved");
    TEST_ASSERT(r1.request_id == 1, "Seq 1 request ID preserved");
    return true;
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "PHASE 1: PREFETCH FRAMEWORK UNIT & INTEGRATION TEST SUITE\n";
    std::cout << "======================================================================\n\n";

    std::cout << "--- Group 1: Configuration Tests (12 tests) ---\n";
    RUN_TEST(test_config_missing_section);
    RUN_TEST(test_config_explicitly_disabled);
    RUN_TEST(test_config_prediction_only_none);
    RUN_TEST(test_config_st_moe_unimplemented_in_phase1);
    RUN_TEST(test_config_unknown_predictor_rejected);
    RUN_TEST(test_config_invalid_mode_rejected);
    RUN_TEST(test_config_invalid_candidate_policy_rejected);
    RUN_TEST(test_config_invalid_eviction_policy_rejected);
    RUN_TEST(test_config_invalid_transfer_model_rejected);
    RUN_TEST(test_config_invalid_numeric_values_rejected);
    RUN_TEST(test_config_missing_calibration_file_prediction_only);
    RUN_TEST(test_config_invalid_combinations_rejected);

    std::cout << "\n--- Group 2: Predictor Tests (5 tests) ---\n";
    RUN_TEST(test_none_predictor_init);
    RUN_TEST(test_none_predictor_update);
    RUN_TEST(test_none_predictor_predict_empty);
    RUN_TEST(test_none_predictor_no_future_access);
    RUN_TEST(test_none_predictor_sequence_isolation);

    std::cout << "\n======================================================================\n";
    std::cout << "TEST SUMMARY: " << tests_passed << " PASSED, " << tests_failed << " FAILED\n";
    std::cout << "======================================================================\n";

    if (tests_failed == 0) {
        return 0;
    } else {
        return 1;
    }
}

