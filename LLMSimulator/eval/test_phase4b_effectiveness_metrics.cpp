#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "common/type.h"
#include "hardware/cluster.h"
#include "hardware/device.h"
#include "hardware/hardware_config.h"
#include "module/tensor.h"
#include "prefetch/ddr5_transfer_engine.h"
#include "prefetch/prefetch_candidate_queue.h"
#include "prefetch/prefetch_config.h"
#include "prefetch/prefetch_metrics_collector.h"

using namespace llm_system;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  [FAIL] Line " << __LINE__ << ": " << msg << std::endl; \
            tests_failed = tests_failed + 1; \
            return false; \
        } \
    } while (false)

static std::string get_test_hbm_cfg_path() {
    if (std::filesystem::exists("./dram_config_HBM3_80GB.yaml")) {
        return "./dram_config_HBM3_80GB.yaml";
    }
    if (std::filesystem::exists("LLMSimulator/build/dram_config_HBM3_80GB.yaml")) {
        return "LLMSimulator/build/dram_config_HBM3_80GB.yaml";
    }
    if (std::filesystem::exists("build/dram_config_HBM3_80GB.yaml")) {
        return "build/dram_config_HBM3_80GB.yaml";
    }
    return "dram_config_HBM3_80GB.yaml";
}

static Device::Ptr make_test_device(bool prefetch_enabled, EvictionPolicy policy, double bw_gbps) {
    SystemConfig cfg = A100;
    cfg.hbm_dram_cfg_path = get_test_hbm_cfg_path();
    cfg.use_ramulator = false;
    cfg.use_event_driven_dram = true;
    cfg.ddr5_bandwidth_gbps = bw_gbps;

    cfg.prefetch_config.enabled = prefetch_enabled;
    if (prefetch_enabled) {
        cfg.prefetch_config.mode = PrefetchMode::HARDWARE_ASYNC;
        cfg.prefetch_config.scheduling_mode = SchedulingMode::FORWARD_LOOKAHEAD;
        cfg.prefetch_config.predictor_type = PredictorType::CSTP;
        cfg.prefetch_config.eviction_policy = policy;
        cfg.prefetch_config.budget = 4;
    } else {
        cfg.prefetch_config.mode = PrefetchMode::NONE;
    }

    Cluster::Ptr cluster = Cluster::Create(cfg, nullptr);
    Device::Ptr dev = Device::Create(cfg, 0, cluster);
    return dev;
}

// Case A: Timely prefetch completed before demand arrives
static bool test_case_a_timely_prefetch() {
    std::cout << "[Case A] Timely prefetch hit (0 ns wait)\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    Device::Ptr dev = make_test_device(true, EvictionPolicy::LRU, 64.0);
    int layer_id = 1;
    int expert_id = 7;
    uint64_t bytes = 64000;
    double now_ns = 1000.0;
    dev->set_time(now_ns);

    uint64_t tid = dev->submit_prefetch_transfer(layer_id, expert_id, now_ns, bytes, 1, 0, 10, 1, 0, 100);
    TEST_CHECK(tid > 0, "Valid transfer id required");

    now_ns = 3000.0;
    dev->set_time(now_ns);
    dev->process_completed_prefetches(now_ns);
    TEST_CHECK(dev->has_resident_expert(layer_id, expert_id), "Expert must be resident");

    col.record_expert_demanded(1, 1, 0, 100, layer_id, expert_id, now_ns, now_ns, 0.0, 0.0, "TIMELY_PREFETCH", tid, 10);
    col.finalize(4000.0, 0);

    const auto& m = col.get_metrics();
    TEST_CHECK(m.timely_resident_hits == 1, "timely_resident_hits must be 1");
    TEST_CHECK(m.unique_timely_resident_hits == 1, "unique_timely_resident_hits must be 1");
    TEST_CHECK(m.fresh_demand_misses == 0, "fresh_demand_misses must be 0");
    TEST_CHECK(m.overall_coverage == 1.0, "overall_coverage must be 1.0");
    TEST_CHECK(m.unique_overall_coverage == 1.0, "unique_overall_coverage must be 1.0");

    tests_passed = tests_passed + 1;
    return true;
}

// Case B: Demand arrives while prefetch is in flight (late attachment)
static bool test_case_b_late_inflight_attachment() {
    std::cout << "[Case B] Late in-flight attachment\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    Device::Ptr dev = make_test_device(true, EvictionPolicy::LRU, 64.0);
    int layer_id = 1;
    int expert_id = 12;
    uint64_t bytes = 128000;
    double now_ns = 1000.0;
    dev->set_time(now_ns);

    uint64_t tid = dev->submit_prefetch_transfer(layer_id, expert_id, now_ns, bytes, 1, 0, 20, 1, 0, 101);
    TEST_CHECK(tid > 0, "Valid transfer id required");

    // Demand arrives at 1500 ns (transfer finishes at 3000 ns, 1500 ns remaining)
    now_ns = 1500.0;
    dev->set_time(now_ns);
    uint64_t dem_tid = dev->submit_demand_transfer(layer_id, expert_id, now_ns, bytes, 1, 0, 101);
    TEST_CHECK(dem_tid == tid, "Demand must attach to existing in-flight transfer");

    now_ns = 3500.0;
    dev->set_time(now_ns);
    dev->process_completed_prefetches(now_ns);
    col.finalize(5000.0, 0);

    const auto& m = col.get_metrics();
    TEST_CHECK(m.late_inflight_attachments == 1, "late_inflight_attachments must be 1");
    TEST_CHECK(m.unique_late_inflight_attachments == 1, "unique_late_inflight_attachments must be 1");
    TEST_CHECK(m.avg_remaining_late_wait_ns > 0.0, "avg_remaining_late_wait_ns must be positive");
    TEST_CHECK(m.speculative_bytes_transferred == bytes, "Only speculative bytes issued");
    TEST_CHECK(m.demand_bytes_transferred == 0, "No duplicate demand bytes issued");

    tests_passed = tests_passed + 1;
    return true;
}

// Case C: Speculative prefetch completes but is never demanded
static bool test_case_c_unused_prefetch() {
    std::cout << "[Case C] Unused speculative prefetch\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    Device::Ptr dev = make_test_device(true, EvictionPolicy::LRU, 64.0);
    int layer_id = 2;
    int expert_id = 15;
    uint64_t bytes = 64000;
    double now_ns = 1000.0;
    dev->set_time(now_ns);

    uint64_t tid = dev->submit_prefetch_transfer(layer_id, expert_id, now_ns, bytes, 1, 0, 30, 1, 0, 102);
    TEST_CHECK(tid > 0, "Valid transfer id required");

    now_ns = 3000.0;
    dev->set_time(now_ns);
    dev->process_completed_prefetches(now_ns);
    TEST_CHECK(dev->has_resident_expert(layer_id, expert_id), "Expert must be resident");

    col.record_transfer_outcome(tid, "UNUSED");
    col.finalize(4000.0, 0);

    const auto& m = col.get_metrics();
    TEST_CHECK(m.prefetches_completed_unused == 1, "prefetches_completed_unused must be 1");
    TEST_CHECK(m.completed_unused_bytes == bytes, "completed_unused_bytes must match transfer size");
    TEST_CHECK(m.total_wasted_speculative_bytes == bytes, "total_wasted_speculative_bytes must match transfer size");
    TEST_CHECK(m.prefetches_completed_and_used == 0, "prefetches_completed_and_used must be 0");

    tests_passed = tests_passed + 1;
    return true;
}

// Case D: Candidate admission rejection
static bool test_case_d_candidate_rejection() {
    std::cout << "[Case D] Candidate admission rejection\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    col.record_candidate_rejected(1, 0, 103, 1, 2, 4, "BUDGET_EXCEEDED", 1200.0);
    col.record_candidate_rejected(1, 0, 103, 1, 2, 8, "DUPLICATE_IN_QUEUE", 1205.0);
    col.finalize(2000.0, 0);

    const auto& m = col.get_metrics();
    TEST_CHECK(m.candidates_rejected_admission == 2, "candidates_rejected_admission must be 2");

    tests_passed = tests_passed + 1;
    return true;
}

// Case E: Speculative prefetch cancelled before completion
static bool test_case_e_transfer_cancellation() {
    std::cout << "[Case E] Speculative transfer cancellation\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    uint64_t tid = 55;
    col.record_transfer_submitted(tid, 50, 1, 0, 104, 3, 9, "PREFETCH", 1, 88080384, 1000.0);
    col.record_transfer_cancelled(tid, 1500.0, 32000);
    col.record_transfer_outcome(tid, "CANCELLED");
    col.finalize(3000.0, 0);

    const auto& m = col.get_metrics();
    TEST_CHECK(m.transfers_cancelled == 1, "transfers_cancelled must be 1");
    TEST_CHECK(m.prefetches_cancelled_inflight == 1, "prefetches_cancelled_inflight must be 1");

    tests_passed = tests_passed + 1;
    return true;
}

// Case F: Expert evicted before demand arrives
static bool test_case_f_evicted_before_demand() {
    std::cout << "[Case F] Evicted before demand\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    uint64_t tid = 77;
    col.record_transfer_submitted(tid, 70, 1, 0, 105, 2, 3, "PREFETCH", 1, 64000, 1000.0);
    col.record_transfer_completed(tid, 2000.0);
    col.record_transfer_outcome(tid, "EVICTED_BEFORE_DEMAND");

    // Later, actual demand arrives as DEMAND_MISS
    col.record_expert_demanded(1, 1, 0, 105, 2, 3, 3000.0, 4000.0, 1000.0, 0.0, "DEMAND_MISS", 0, 0);
    col.finalize(5000.0, 0);

    const auto& m = col.get_metrics();
    TEST_CHECK(m.prefetches_evicted_before_demand == 1, "prefetches_evicted_before_demand must be 1");
    TEST_CHECK(m.fresh_demand_misses == 1, "fresh_demand_misses must be 1");

    tests_passed = tests_passed + 1;
    return true;
}

// Case G: Deduplication of multi-projection demands (W1, W3, W2)
static bool test_case_g_projection_deduplication() {
    std::cout << "[Case G] Deduplication of multi-projection demands (W1, W3, W2)\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    // Suppose W1 arrives at 1000 ns, experiences demand miss of 1000 ns (ready at 2000 ns)
    col.record_expert_demanded(1, 1, 0, 106, 1, 5, 1000.0, 2000.0, 1000.0, 0.0, "DEMAND_MISS", 101, 0);
    // W3 arrives at 2050 ns, immediately hits in HBM
    col.record_expert_demanded(2, 1, 0, 106, 1, 5, 2050.0, 2050.0, 0.0, 0.0, "RESIDENT_HIT", 101, 0);
    // W2 arrives at 2100 ns, immediately hits in HBM
    col.record_expert_demanded(3, 1, 0, 106, 1, 5, 2100.0, 2100.0, 0.0, 0.0, "RESIDENT_HIT", 101, 0);

    col.finalize(3000.0, 0);

    const auto& m = col.get_metrics();
    TEST_CHECK(m.total_actual_demands == 3, "total_actual_demands must be 3 (projection-level)");
    TEST_CHECK(m.unique_actual_demands == 1, "unique_actual_demands must be 1 (unique-expert)");
    TEST_CHECK(m.fresh_demand_misses == 1, "fresh_demand_misses must be 1 (projection-level)");
    TEST_CHECK(m.unique_fresh_demand_misses == 1, "unique_fresh_demand_misses must be 1 (unique-expert)");
    TEST_CHECK(m.total_hbm_resident_hits == 2, "total_hbm_resident_hits must be 2 (W3, W2)");
    TEST_CHECK(m.unique_resident_hits == 0, "unique_resident_hits must be 0 (initial miss was first)");

    // Projection hit rate: 2/3 = 66.67%, Unique hit rate: 0/1 = 0%
    TEST_CHECK(std::abs(m.overall_hbm_hit_rate - (2.0 / 3.0)) < 1e-4, "Projection hit rate must be 66.67%");
    TEST_CHECK(m.unique_overall_hbm_hit_rate == 0.0, "Unique hit rate must be 0.0% for cold start");

    tests_passed = tests_passed + 1;
    return true;
}

// Case H: End-of-simulation transfer in-flight accounting
static bool test_case_h_end_of_sim_inflight() {
    std::cout << "[Case H] End-of-simulation in-flight accounting\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    uint64_t tid = 88;
    col.record_transfer_submitted(tid, 80, 1, 0, 107, 3, 2, "PREFETCH", 1, 64000, 1000.0);
    col.record_transfer_started(tid, 1050.0);
    // Simulation ends before completion
    col.record_transfer_outcome(tid, "STILL_IN_FLIGHT_AT_END");
    col.finalize(1200.0, 0);

    const auto& m = col.get_metrics();
    TEST_CHECK(m.prefetches_still_inflight_at_end == 1, "prefetches_still_inflight_at_end must be 1");
    TEST_CHECK(m.total_completed_prefetches == 0, "total_completed_prefetches must be 0");

    tests_passed = tests_passed + 1;
    return true;
}

// Case I: Timeline overlap and stall hidden calculation
static bool test_case_i_stall_hidden_calculation() {
    std::cout << "[Case I] Stall hidden calculation\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    // 1 Demand Miss with 1376256 ns wait (defines base transfer time)
    col.record_expert_demanded(1, 1, 0, 108, 0, 1, 1000.0, 1000.0 + 1376256.0, 1376256.0, 0.0, "DEMAND_MISS", 0, 0);
    // 1 Timely hit: saved 1376256 ns
    col.record_expert_demanded(2, 1, 0, 108, 0, 2, 2000.0, 2000.0, 0.0, 0.0, "TIMELY_PREFETCH", 10, 1);
    // 1 Late attachment: remaining wait 376256 ns => saved 1000000 ns
    col.record_expert_demanded(3, 1, 0, 108, 0, 3, 3000.0, 3000.0 + 376256.0, 376256.0, 0.0, "LATE_IN_FLIGHT", 11, 2);

    col.finalize(5000000.0, 0);

    const auto& m = col.get_metrics();
    double expected_hidden = 1376256.0 + (1376256.0 - 376256.0); // 2376256.0 ns
    TEST_CHECK(std::abs(m.stall_hidden_by_prefetch_ns - expected_hidden) < 1.0, "Stall hidden calculation must match expected sum");

    tests_passed = tests_passed + 1;
    return true;
}

// Case J: Token and request latency quantiles correctly measured
static bool test_case_j_token_latency_quantiles() {
    std::cout << "[Case J] Token and request latency quantiles\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    col.record_request_start(0, 0, 128, 0.0, 1000.0);

    // Token 0: duration 10000 ns
    col.record_token_start(0, 0, 128, 0, 1000.0);
    col.record_token_end(0, 0, 128, 0, 11000.0);

    // Token 1: duration 20000 ns
    col.record_token_start(0, 0, 129, 1, 11000.0);
    col.record_token_end(0, 0, 129, 1, 31000.0);

    // Token 2: duration 30000 ns
    col.record_token_start(0, 0, 130, 2, 31000.0);
    col.record_token_end(0, 0, 130, 2, 61000.0);

    // Token 3: duration 40000 ns
    col.record_token_start(0, 0, 131, 3, 61000.0);
    col.record_token_end(0, 0, 131, 3, 101000.0);

    col.record_request_end(0, 0, 4, 101000.0);

    col.finalize(101000.0, 0);

    const auto& m = col.get_metrics();
    TEST_CHECK(m.token_latency_quantiles.count == 4, "token count must be 4");
    TEST_CHECK(std::abs(m.mean_token_latency_ns - 25000.0) < 1e-3, "mean token latency must be 25000 ns");
    TEST_CHECK(m.max_token_latency_ns == 40000.0, "max token latency must be 40000 ns");
    TEST_CHECK(m.request_latency_quantiles.count == 1, "request count must be 1");
    TEST_CHECK(m.mean_request_latency_ns == 100000.0, "mean request latency must be 100000 ns");
    TEST_CHECK(m.simulated_tokens_per_second > 0.0, "tokens per second must be positive");

    tests_passed = tests_passed + 1;
    return true;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Phase 4B Effectiveness Metrics Unit Test Suite (Cases A-J)\n";
    std::cout << "========================================================\n";

    test_case_a_timely_prefetch();
    test_case_b_late_inflight_attachment();
    test_case_c_unused_prefetch();
    test_case_d_candidate_rejection();
    test_case_e_transfer_cancellation();
    test_case_f_evicted_before_demand();
    test_case_g_projection_deduplication();
    test_case_h_end_of_sim_inflight();
    test_case_i_stall_hidden_calculation();
    test_case_j_token_latency_quantiles();

    std::cout << "========================================================\n";
    std::cout << "Summary: " << tests_passed << " Passed, " << tests_failed << " Failed\n";
    std::cout << "========================================================\n";

    if (tests_failed > 0) {
        return 1;
    }
    return 0;
}
