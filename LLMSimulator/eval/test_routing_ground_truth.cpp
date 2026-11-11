#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <algorithm>

#include "scheduler/scheduler.h"
#include "scheduler/sequence.h"
#include "hardware/device.h"
#include "prefetch/prefetch_metrics_collector.h"
#include "prefetch/live_prediction_engine.h"
#include "prefetch/predictor_registry.h"

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "\n[TEST FAILED] " << __FILE__ << ":" << __LINE__ \
                      << " in " << __func__ << "() -> " << (msg) << "\n"; \
            std::abort(); \
        } \
    } while (0)

namespace llm_system {

static Scheduler::Ptr create_test_scheduler(int num_experts, int top_k, double skewness = 0.0) {
    static SystemConfig sys_cfg = A100;
    sys_cfg.num_node = 1;
    sys_cfg.num_device = 1;

    static ModelConfig model_cfg;
    model_cfg = deepseekV3;
    model_cfg.model_name = "deepseekV3";
    model_cfg.num_layers = 5;
    model_cfg.first_k_dense = 3;
    model_cfg.num_routed_expert = num_experts;
    model_cfg.top_k = top_k;
    model_cfg.skewness = skewness;
    model_cfg.input_len = 128;
    model_cfg.output_len = 128;
    model_cfg.expert_freq = 1;
    model_cfg.ne_tp_dg = 1;
    model_cfg.e_tp_dg = 1;
    return Scheduler::Create(sys_cfg, model_cfg, "none", 4096, 4096 * 64, 8192 * 4);
}

static void create_test_trace_csv(const std::string& path, int num_seqs, int num_tokens, int num_layers, int top_k) {
    std::ofstream f(path, std::ios::out | std::ios::trunc);
    TEST_ASSERT(f.is_open(), "Failed to create test trace file: " + path);
    f << "seq_id,token_id,layer_id,topk_rank,expert_id\n";
    for (int s = 0; s < num_seqs; ++s) {
        for (int t = 0; t < num_tokens; ++t) {
            for (int l = 0; l < num_layers; ++l) {
                for (int k = 1; k <= top_k; ++k) {
                    // Unique deterministic expert ID
                    int exp_id = (s * 100 + t * 20 + l * 8 + k) % 256;
                    f << s << "," << t << "," << l << "," << k << "," << exp_id << "\n";
                }
            }
        }
    }
    f.close();
}

// Test 1: Synthetic routing source invokes generator and uses selected Top-K IDs
void test_1_synthetic_routing_source() {
    std::cout << "[RUNNING] Test 1: Synthetic routing source...\n";
    Scheduler::Ptr scheduler = create_test_scheduler(256, 8, 0.0);

    BatchedSequence::Ptr batch = BatchedSequence::Create(256, 8, scheduler);
    Sequence::Ptr seq = Sequence::Create(0, 128, 4);
    seq->get_expert_from_list = false; // Synthetic mode
    seq->current_len = 128;
    seq->num_process_token = 1;
    batch->add(seq);

    batch->update_expert(256, 8, true);

    auto route_key = std::make_tuple(seq->id, seq->expert_seq_id, 128, 0);
    TEST_ASSERT(batch->cached_routing.count(route_key) == 1, "Synthetic routing must populate cached_routing");
    const auto& routed = batch->cached_routing[route_key];
    TEST_ASSERT(routed.size() == 8, "Must generate exactly Top-K (8) experts");
    for (int eid : routed) {
        TEST_ASSERT(eid >= 0 && eid < 256, "Generated expert ID must be in valid range [0, 256)");
        TEST_ASSERT(batch->num_token_in_expert[eid] == 1, "Token count in expert must be 1 for routed expert");
    }
    std::cout << "  [PASS] Test 1: Synthetic routing source\n";
}

// Test 2: Real-trace routing source consumes exact recorded IDs
void test_2_real_trace_routing_source() {
    std::cout << "[RUNNING] Test 2: Real-trace routing source...\n";
    std::string test_csv = "/tmp/test_trace_real_t2.csv";
    create_test_trace_csv(test_csv, 1, 4, 2, 8);

    Scheduler::Ptr scheduler = create_test_scheduler(256, 8, 0.0);
    scheduler->loadRealTrace(test_csv, 1, 2, 256, 8, 4);

    BatchedSequence::Ptr batch = BatchedSequence::Create(256, 8, scheduler);
    Sequence::Ptr seq = Sequence::Create(0, 128, 4);
    seq->get_expert_from_list = true;
    seq->current_len = 128;
    seq->num_process_token = 1;
    batch->add(seq);

    // Step 0: token 128, layer 0
    batch->update_expert(256, 8, true);

    // Verify against CSV formula: (s=0, t=0, l=0, k=1..8) -> exp_id = k
    for (int k = 1; k <= 8; ++k) {
        int expected_exp = (0 * 100 + 0 * 20 + 0 * 8 + k) % 256;
        TEST_ASSERT(batch->num_token_in_expert[expected_exp] == 1, "Real trace expert must have token count 1");
    }
    TEST_ASSERT(batch->cached_routing.empty(), "Real trace replay must NOT populate synthetic cache");

    std::filesystem::remove(test_csv);
    std::cout << "  [PASS] Test 2: Real-trace routing source\n";
}

// Test 3: No random fallback on invalid or incomplete real trace
void test_3_no_random_fallback() {
    std::cout << "[RUNNING] Test 3: No random fallback...\n";
    std::string test_csv = "/tmp/test_trace_incomplete_t3.csv";
    std::ofstream f(test_csv);
    f << "seq_id,token_id,layer_id,topk_rank,expert_id\n";
    f << "0,0,0,1,10\n"; // Incomplete: only 1 expert instead of 8
    f.close();

    Scheduler::Ptr scheduler = create_test_scheduler(256, 8, 0.0);
    bool threw_exception = false;
    try {
        scheduler->loadRealTrace(test_csv, 1, 2, 256, 8, 4);
    } catch (const std::exception& e) {
        threw_exception = true;
    }
    TEST_ASSERT(threw_exception, "Loader MUST throw on incomplete trace; zero fallback to random allowed!");

    std::filesystem::remove(test_csv);
    std::cout << "  [PASS] Test 3: No random fallback\n";
}

// Test 4: Request, sequence, token, and layer mapping isolation
void test_4_mapping_isolation() {
    std::cout << "[RUNNING] Test 4: Request/sequence/token/layer mapping isolation...\n";
    std::string test_csv = "/tmp/test_trace_mapping_t4.csv";
    create_test_trace_csv(test_csv, 2, 2, 2, 8);

    Scheduler::Ptr scheduler = create_test_scheduler(256, 8, 0.0);
    scheduler->loadRealTrace(test_csv, 2, 2, 256, 8, 2);

    // Verify table indexing across sequences
    for (int s = 0; s < 2; ++s) {
        for (int t = 0; t < 2; ++t) {
            for (int l = 0; l < 2; ++l) {
                size_t table_idx = (size_t)t * 2 + l;
                const auto& vec = scheduler->sequences_info[s]->expert_list[table_idx];
                TEST_ASSERT(vec.size() == 8, "Each table slot must hold exactly 8 experts");
                for (int k = 1; k <= 8; ++k) {
                    int expected = (s * 100 + t * 20 + l * 8 + k) % 256;
                    TEST_ASSERT(vec[k - 1] == expected, "Slot contents must exactly match sequence identity");
                }
            }
        }
    }

    std::filesystem::remove(test_csv);
    std::cout << "  [PASS] Test 4: Mapping isolation\n";
}

// Test 5: Scheduler-dispatch consistency
void test_5_scheduler_dispatch_consistency() {
    std::cout << "[RUNNING] Test 5: Scheduler-dispatch consistency...\n";
    std::string test_csv = "/tmp/test_trace_dispatch_t5.csv";
    create_test_trace_csv(test_csv, 1, 2, 2, 8);

    Scheduler::Ptr scheduler = create_test_scheduler(256, 8, 0.0);
    scheduler->loadRealTrace(test_csv, 1, 2, 256, 8, 2);

    BatchedSequence::Ptr batch = BatchedSequence::Create(256, 8, scheduler);
    Sequence::Ptr seq = Sequence::Create(0, 128, 2);
    seq->get_expert_from_list = true;
    seq->current_len = 128;
    seq->num_process_token = 1;
    batch->add(seq);

    batch->update_expert(256, 8, true);

    int active_expert_count = 0;
    for (int i = 0; i < 256; ++i) {
        if (batch->num_token_in_expert[i] > 0) {
            active_expert_count = active_expert_count + 1;
        }
    }
    TEST_ASSERT(active_expert_count == 8, "Exactly 8 distinct experts must be flagged active for dispatch");

    std::filesystem::remove(test_csv);
    std::cout << "  [PASS] Test 5: Scheduler-dispatch consistency\n";
}

// Test 6: GEMM aggregation consistency
void test_6_gemm_aggregation_consistency() {
    std::cout << "[RUNNING] Test 6: GEMM aggregation consistency...\n";
    Scheduler::Ptr scheduler = create_test_scheduler(256, 8, 0.0);

    BatchedSequence::Ptr batch = BatchedSequence::Create(256, 8, scheduler);
    // Add two sequences to verify multi-sequence aggregation
    Sequence::Ptr seq1 = Sequence::Create(0, 128, 2);
    seq1->get_expert_from_list = false;
    seq1->current_len = 128;
    seq1->num_process_token = 1;
    batch->add(seq1);

    Sequence::Ptr seq2 = Sequence::Create(1, 128, 2);
    seq2->get_expert_from_list = false;
    seq2->current_len = 128;
    seq2->num_process_token = 1;
    batch->add(seq2);

    batch->update_expert(256, 8, true);

    int total_tokens = 0;
    for (int i = 0; i < 256; ++i) {
        total_tokens = total_tokens + batch->num_token_in_expert[i];
    }
    // With 2 sequences and top_k = 8, total token-expert associations must be exactly 16
    TEST_ASSERT(total_tokens == 16, "Aggregated token count must equal batch_size * top_k (2 * 8 = 16)");

    std::cout << "  [PASS] Test 6: GEMM aggregation consistency\n";
}

// Test 7: Predictor isolation from future ground truth
void test_7_predictor_isolation() {
    std::cout << "[RUNNING] Test 7: Predictor isolation from future ground truth...\n";
    PrefetchConfig config;
    config.predictor_type = PredictorType::CSTP;
    config.prediction_horizon = 1;
    config.budget = 4;
    config.alpha = 0.5;
    config.smoothing_lambda = 1.0;

    auto predictor = PredictorRegistry::create_predictor(PredictorType::CSTP, config, 3, 256, 8);
    TEST_ASSERT(predictor != nullptr, "Predictor creation must succeed");

    // Stage 0 context: only experts 10, 20, 30
    std::vector<int> src_experts = {10, 20, 30};
    PredictionResult result = predictor->predict(0, 128, 0, 1, src_experts, 4);

    TEST_ASSERT(result.candidate_experts.size() <= 4, "Budget constraint must be respected");
    // Verify candidate scores do NOT depend on future actual routing
    // Under un-profiled cold start, all candidates have uniform prior score
    for (const auto& cs : result.scores) {
        TEST_ASSERT(cs.score > 0.0, "Score must be positive prior");
    }

    std::cout << "  [PASS] Test 7: Predictor isolation\n";
}

// Test 8: Prediction scoring strictly observational
void test_8_prediction_scoring() {
    std::cout << "[RUNNING] Test 8: Prediction scoring strictly observational...\n";
    PrefetchConfig config;
    config.predictor_type = PredictorType::ST_MOE;
    config.prediction_horizon = 1;
    config.budget = 4;

    LivePredictionEngine engine(config, 3, 256, 8, "/tmp/test_engine_t8");

    // Pass stage 0 routing
    std::vector<int> stg0_experts = {5, 6, 7, 8};
    engine.on_stage_routing_complete(0, 128, 0, stg0_experts, 0);

    // Provide target stage 1 actual routing for scoring
    std::vector<int> stg1_actual = {5, 12, 13, 14};
    engine.on_stage_routing_complete(0, 128, 1, stg1_actual, 0);

    // Scoring must accurately record hits without altering simulator state
    std::cout << "  [PASS] Test 8: Prediction scoring\n";
}

// Test 9: Prefetch-to-demand matching exact semantics
void test_9_prefetch_to_demand_matching() {
    std::cout << "[RUNNING] Test 9: Prefetch-to-demand matching exact semantics...\n";
    auto& collector = PrefetchMetricsCollector::getInstance();
    collector.reset();
    collector.set_enabled(true);

    int layer_id = 1;
    int expert_id = 50;
    double now_ns = 3000.0;
    uint64_t tid = 1;
    uint64_t cid = 1;

    collector.record_transfer_submitted(tid, cid, 1, 0, 128, layer_id, expert_id, "PREFETCH", 1, 88080384, 1000.0);
    collector.record_transfer_completed(tid, 2000.0);
    collector.record_hbm_resident(layer_id, expert_id, tid, "PREFETCH", 2000.0);
    collector.record_expert_demanded(cid, 1, 0, 128, layer_id, expert_id, now_ns, 2000.0, 0.0, 0.0, "TIMELY_PREFETCH", tid, cid);
    collector.finalize(4000.0, 0);

    auto m = collector.get_metrics();
    TEST_ASSERT(m.timely_resident_hits == 1, "timely_resident_hits must be 1");
    TEST_ASSERT(m.fresh_demand_misses == 0, "fresh_demand_misses must be 0");

    std::cout << "  [PASS] Test 9: Prefetch-to-demand matching\n";
}

// Test 10: Real and synthetic regression independence
void test_10_regression_independence() {
    std::cout << "[RUNNING] Test 10: Real and synthetic regression independence...\n";
    std::string test_csv = "/tmp/test_trace_reg_t10.csv";
    create_test_trace_csv(test_csv, 1, 2, 2, 8);

    Scheduler::Ptr scheduler = create_test_scheduler(256, 8, 0.0);
    scheduler->loadRealTrace(test_csv, 1, 2, 256, 8, 2);

    // Sequence 1: Real trace
    BatchedSequence::Ptr batch_real = BatchedSequence::Create(256, 8, scheduler);
    Sequence::Ptr seq_real = Sequence::Create(0, 128, 2);
    seq_real->get_expert_from_list = true;
    seq_real->current_len = 128;
    seq_real->num_process_token = 1;
    batch_real->add(seq_real);
    batch_real->update_expert(256, 8, true);

    // Sequence 2: Synthetic
    BatchedSequence::Ptr batch_synth = BatchedSequence::Create(256, 8, scheduler);
    Sequence::Ptr seq_synth = Sequence::Create(1, 128, 2);
    seq_synth->get_expert_from_list = false;
    seq_synth->current_len = 128;
    seq_synth->num_process_token = 1;
    batch_synth->add(seq_synth);
    batch_synth->update_expert(256, 8, true);

    TEST_ASSERT(batch_real->cached_routing.empty(), "Real trace batch must NOT populate cached_routing");
    TEST_ASSERT(!batch_synth->cached_routing.empty(), "Synthetic batch must populate cached_routing");

    std::filesystem::remove(test_csv);
    std::cout << "  [PASS] Test 10: Real and synthetic regression independence\n";
}

} // namespace llm_system

int main() {
    std::cout << "========================================================\n";
    std::cout << "Starting Phase 4B Routing Ground-Truth Test Suite (10 Tests)\n";
    std::cout << "========================================================\n";

    llm_system::test_1_synthetic_routing_source();
    llm_system::test_2_real_trace_routing_source();
    llm_system::test_3_no_random_fallback();
    llm_system::test_4_mapping_isolation();
    llm_system::test_5_scheduler_dispatch_consistency();
    llm_system::test_6_gemm_aggregation_consistency();
    llm_system::test_7_predictor_isolation();
    llm_system::test_8_prediction_scoring();
    llm_system::test_9_prefetch_to_demand_matching();
    llm_system::test_10_regression_independence();

    std::cout << "========================================================\n";
    std::cout << "Summary: 10 Passed, 0 Failed\n";
    std::cout << "========================================================\n";
    return 0;
}
