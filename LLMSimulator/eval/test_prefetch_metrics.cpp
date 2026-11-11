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
        cfg.prefetch_config.pabi_quarantine_enabled = (policy == EvictionPolicy::PABI);
        cfg.prefetch_config.pabi_pinning_enabled = (policy == EvictionPolicy::PABI);
        cfg.prefetch_config.pabi_max_pinned_experts = 4;
        cfg.prefetch_config.budget = 4;
    } else {
        cfg.prefetch_config.mode = PrefetchMode::NONE;
    }

    Cluster::Ptr cluster = Cluster::Create(cfg, nullptr);
    Device::Ptr dev = Device::Create(cfg, 0, cluster);
    return dev;
}

// Scenario 1: Timely prefetch hit (TIMELY_PREFETCH, 0 wait)
static bool test_scenario_1_timely_prefetch() {
    std::cout << "[Test 1] Scenario 1: Timely prefetch hit\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    Device::Ptr dev = make_test_device(true, EvictionPolicy::LRU, 64.0);
    int layer_id = 2;
    int expert_id = 5;
    uint64_t bytes = 64000;
    double now_ns = 1000.0;
    dev->set_time(now_ns);

    uint64_t tid = dev->submit_prefetch_transfer(layer_id, expert_id, now_ns, bytes, 1, 0, 101, 1, 1, 1);
    TEST_CHECK(tid > 0, "Prefetch submission must return valid transfer id");

    // Advance time past prefetch completion (64000 / 64.0 = 1000 ns => finishes at 2000 ns)
    now_ns = 3000.0;
    dev->set_time(now_ns);
    dev->process_completed_prefetches(now_ns);

    TEST_CHECK(dev->has_resident_expert(layer_id, expert_id), "Expert must be resident in HBM");

    // Expert demanded at now_ns = 3000.0
    col.record_expert_demanded(1, 1, 1, 1, layer_id, expert_id, now_ns, now_ns, 0.0, 0.0, "TIMELY_PREFETCH", tid, 101);

    col.finalize(now_ns, bytes);
    const auto& m = col.get_metrics();
    TEST_CHECK(m.timely_resident_hits == 1, "Must record 1 timely prefetch hit");
    TEST_CHECK(m.total_actual_demands == 1, "Must record 1 total expert demand");
    TEST_CHECK(m.mean_expert_load_latency_ns == 0.0, "Timely hit must have 0 load latency");
    TEST_CHECK(m.overall_coverage == 1.0, "Prefetch coverage must be 1.0");

    tests_passed = tests_passed + 1;
    return true;
}

// Scenario 2: Late in-flight attachment (LATE_IN_FLIGHT, remaining wait)
static bool test_scenario_2_late_inflight() {
    std::cout << "[Test 2] Scenario 2: Late in-flight attachment\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    Device::Ptr dev = make_test_device(true, EvictionPolicy::LRU, 64.0);
    int layer_id = 3;
    int expert_id = 7;
    uint64_t bytes = 128000; // Duration = 2000 ns
    double submit_ns = 1000.0;
    dev->set_time(submit_ns);

    uint64_t tid = dev->submit_prefetch_transfer(layer_id, expert_id, submit_ns, bytes, 1, 1, 201, 2, 2, 1);
    TEST_CHECK(tid > 0, "Prefetch submission must succeed");

    // Demand arrives midway at 2000 ns (1000 ns remaining)
    double demand_ns = 2000.0;
    dev->set_time(demand_ns);
    uint64_t dem_tid = dev->submit_demand_transfer(layer_id, expert_id, demand_ns, bytes, 2, 2, 1);
    TEST_CHECK(dem_tid == tid, "Demand transfer should attach to in-flight prefetch transfer");

    bool finished = dev->wait_for_demand_transfer(tid);
    TEST_CHECK(finished, "Wait for demand transfer must finish");

    double completion_ns = 3000.0;
    dev->set_time(completion_ns);
    col.finalize(completion_ns, bytes);

    const auto& m = col.get_metrics();
    TEST_CHECK(m.late_inflight_attachments == 1, "Must record 1 late in-flight hit");
    TEST_CHECK(m.late_rate_of_demands == 1.0, "Late prefetch rate must be 1.0");
    TEST_CHECK(m.exposed_gpu_stall_ns > 0.0 || m.mean_expert_load_latency_ns > 0.0, "Must record positive wait latency");

    tests_passed = tests_passed + 1;
    return true;
}

// Scenario 3: Completed unused prefetch (UNUSED, wasted bytes)
static bool test_scenario_3_unused_prefetch() {
    std::cout << "[Test 3] Scenario 3: Completed unused prefetch\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    Device::Ptr dev = make_test_device(true, EvictionPolicy::LRU, 64.0);
    int layer_id = 4;
    int expert_id = 9;
    uint64_t bytes = 64000;
    double now_ns = 1000.0;
    dev->set_time(now_ns);

    uint64_t tid = dev->submit_prefetch_transfer(layer_id, expert_id, now_ns, bytes, 1, 0, 301, 3, 3, 1);
    TEST_CHECK(tid > 0, "Prefetch submitted");

    now_ns = 5000.0;
    dev->set_time(now_ns);
    dev->process_completed_prefetches(now_ns);

    // End simulation without demanding expert 9
    col.finalize(now_ns, bytes);
    const auto& m = col.get_metrics();
    TEST_CHECK(m.prefetches_completed_unused == 1, "Must mark 1 completed unused prefetch");
    TEST_CHECK(m.total_wasted_speculative_bytes == bytes, "Wasted prefetch bytes must match transfer bytes");

    tests_passed = tests_passed + 1;
    return true;
}

// Scenario 4: Candidate admission rejection (OVERFLOW_REJECTED)
static bool test_scenario_4_admission_rejection() {
    std::cout << "[Test 4] Scenario 4: Candidate admission rejection\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    PrefetchConfig pcfg;
    pcfg.queue_capacity = 2;
    pcfg.overflow_policy = OverflowPolicy::REJECT_NEW;
    PrefetchCandidateQueue queue(pcfg, 8, 32);

    PrefetchCandidate c1;
    c1.candidate_id = 401; c1.request_id = 4; c1.seq_id = 4; c1.token_id = 1; c1.target_stage_id = 1; c1.expert_id = 1; c1.prediction_rank = 1; c1.prediction_score = 0.9;
    PrefetchCandidate c2;
    c2.candidate_id = 402; c2.request_id = 4; c2.seq_id = 4; c2.token_id = 1; c2.target_stage_id = 1; c2.expert_id = 2; c2.prediction_rank = 2; c2.prediction_score = 0.8;
    PrefetchCandidate c3;
    c3.candidate_id = 403; c3.request_id = 4; c3.seq_id = 4; c3.token_id = 1; c3.target_stage_id = 1; c3.expert_id = 3; c3.prediction_rank = 3; c3.prediction_score = 0.7;

    queue.enqueue_candidate(c1);
    queue.enqueue_candidate(c2);
    bool accepted = queue.enqueue_candidate(c3);
    TEST_CHECK(!accepted, "Third candidate must be rejected due to queue overflow");

    col.finalize(2000.0, 0);
    const auto& m = col.get_metrics();
    TEST_CHECK(m.candidates_rejected_admission == 1, "Must record 1 candidate rejected");

    tests_passed = tests_passed + 1;
    return true;
}

// Scenario 5: Transfer cancellation (CANCELLED, partial bytes)
static bool test_scenario_5_cancellation() {
    std::cout << "[Test 5] Scenario 5: Transfer cancellation\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    DDR5TransferEngine engine(64.0);
    uint64_t tid = engine.submit_prefetch_id(5, 1, 12, 1000.0, 128000, 1, 0, 5, 1, 501);
    col.record_transfer_submitted(tid, 501, 5, 5, 1, 1, 12, "PREFETCH", 1, 128000, 1000.0);

    // Cancel at 1500 ns after transferring partial bytes (500 ns * 64 bytes/ns = 32000 bytes)
    engine.run_until(1500.0);
    bool ok = engine.cancel_transfer(tid, 1500.0);
    TEST_CHECK(ok, "Transfer cancellation must succeed");

    const auto* tr = engine.get_transfer(tid);
    TEST_CHECK(tr != nullptr, "Transfer must exist");
    col.record_transfer_cancelled(tid, 1500.0, static_cast<uint64_t>(tr->transferred_bytes));
    col.record_transfer_outcome(tid, "CANCELLED");

    col.finalize(2000.0, 0);
    const auto& m = col.get_metrics();
    TEST_CHECK(m.transfers_cancelled == 1, "Must record 1 cancelled transfer");
    TEST_CHECK(m.cancelled_partial_bytes > 0, "Must record non-zero cancelled transfer bytes");

    tests_passed = tests_passed + 1;
    return true;
}

// Scenario 6: Resident HBM hit (RESIDENT_HIT, 0 DDR5 wait)
static bool test_scenario_6_resident_hit() {
    std::cout << "[Test 6] Scenario 6: Resident HBM hit\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    Device::Ptr dev = make_test_device(true, EvictionPolicy::LRU, 64.0);
    int layer_id = 1;
    int expert_id = 3;
    uint64_t bytes = 64000;
    double now_ns = 1000.0;
    dev->set_time(now_ns);

    // Expert already placed in HBM
    dev->mark_expert_resident(layer_id, expert_id, bytes);
    TEST_CHECK(dev->has_resident_expert(layer_id, expert_id), "Expert must be resident");

    col.record_expert_demanded(0, 6, 6, 1, layer_id, expert_id, now_ns, now_ns, 0.0, 0.0, "RESIDENT_HIT", 0, 0);
    col.finalize(now_ns, bytes);

    const auto& m = col.get_metrics();
    TEST_CHECK(m.total_hbm_resident_hits == 1, "Must record 1 resident hit");
    TEST_CHECK(m.overall_hbm_hit_rate == 1.0, "HBM hit rate must be 1.0");
    TEST_CHECK(m.mean_expert_load_latency_ns == 0.0, "Resident hit must have 0 latency");

    tests_passed = tests_passed + 1;
    return true;
}

// Scenario 7: In-flight demand attachment deduplication
static bool test_scenario_7_demand_deduplication() {
    std::cout << "[Test 7] Scenario 7: In-flight demand attachment deduplication\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    Device::Ptr dev = make_test_device(true, EvictionPolicy::LRU, 64.0);
    int layer_id = 0;
    int expert_id = 14;
    uint64_t bytes = 64000;
    double now_ns = 1000.0;
    dev->set_time(now_ns);

    uint64_t tid1 = dev->submit_demand_transfer(layer_id, expert_id, now_ns, bytes, 7, 7, 1);
    TEST_CHECK(tid1 > 0, "First demand transfer succeeds");

    // Second demand arrives before first finishes
    uint64_t tid2 = dev->submit_demand_transfer(layer_id, expert_id, now_ns + 100.0, bytes, 7, 7, 2);
    TEST_CHECK(tid2 == tid1, "Second demand transfer must attach to first in-flight transfer without creating new id");

    dev->wait_for_demand_transfer(tid1);
    col.finalize(3000.0, bytes);

    const auto& m = col.get_metrics();
    TEST_CHECK(m.fresh_demand_misses == 1, "Only 1 physical demand miss submitted");

    tests_passed = tests_passed + 1;
    return true;
}

// Scenario 8: Fresh demand miss (DEMAND_MISS, full transfer wait)
static bool test_scenario_8_fresh_demand_miss() {
    std::cout << "[Test 8] Scenario 8: Fresh demand miss\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    Device::Ptr dev = make_test_device(true, EvictionPolicy::LRU, 64.0);
    int layer_id = 2;
    int expert_id = 20;
    uint64_t bytes = 64000;
    double now_ns = 1000.0;
    dev->set_time(now_ns);

    uint64_t tid = dev->submit_demand_transfer(layer_id, expert_id, now_ns, bytes, 8, 8, 1);
    TEST_CHECK(tid > 0, "Demand miss transfer submitted");

    dev->wait_for_demand_transfer(tid);
    double finish_ns = now_ns + (static_cast<double>(bytes) / 64.0);
    dev->set_time(finish_ns);
    col.record_exposed_stall(now_ns, static_cast<double>(bytes) / 64.0);

    col.finalize(finish_ns, bytes);
    const auto& m = col.get_metrics();
    TEST_CHECK(m.fresh_demand_misses == 1, "Must record 1 fresh demand miss");
    TEST_CHECK(m.exposed_gpu_stall_ns > 0.0, "Must record exposed stall duration");

    tests_passed = tests_passed + 1;
    return true;
}

// Scenario 9: Evicted before demand (EVICTED_BEFORE_DEMAND)
static bool test_scenario_9_evicted_before_demand() {
    std::cout << "[Test 9] Scenario 9: Evicted before demand\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    // Small cache capacity device to force eviction
    Device::Ptr dev = make_test_device(true, EvictionPolicy::LRU, 64.0);
    dev->cache_capacity_bytes = 64000; // Only holds 1 expert

    int l1 = 0, e1 = 1;
    int l2 = 0, e2 = 2;
    uint64_t bytes = 64000;
    double t = 1000.0;
    dev->set_time(t);

    uint64_t tid1 = dev->submit_prefetch_transfer(l1, e1, t, bytes, 1, 0, 901, 9, 9, 1);
    t = 2500.0;
    dev->set_time(t);
    dev->process_completed_prefetches(t);
    TEST_CHECK(dev->has_resident_expert(l1, e1), "Expert 1 must be resident");

    // Submit second prefetch and let it complete, displacing Expert 1
    uint64_t tid2 = dev->submit_prefetch_transfer(l2, e2, t, bytes, 1, 0, 902, 9, 9, 1);
    t = 4000.0;
    dev->set_time(t);
    dev->process_completed_prefetches(t);

    TEST_CHECK(!dev->has_resident_expert(l1, e1), "Expert 1 must have been evicted");
    TEST_CHECK(dev->has_resident_expert(l2, e2), "Expert 2 must now be resident");

    col.finalize(t, bytes);
    const auto& m = col.get_metrics();
    TEST_CHECK(m.prefetches_evicted_before_demand >= 1, "Must record at least 1 transfer evicted before demand");

    tests_passed = tests_passed + 1;
    return true;
}

// Scenario 10: Re-demand after eviction
static bool test_scenario_10_redemand_after_eviction() {
    std::cout << "[Test 10] Scenario 10: Re-demand after eviction\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    Device::Ptr dev = make_test_device(true, EvictionPolicy::LRU, 64.0);
    dev->cache_capacity_bytes = 64000;

    int l1 = 0, e1 = 10;
    int l2 = 0, e2 = 20;
    uint64_t bytes = 64000;
    double t = 1000.0;
    dev->set_time(t);

    dev->mark_expert_resident(l1, e1, bytes);
    TEST_CHECK(dev->has_resident_expert(l1, e1), "Expert 10 resident");

    // Inserting Expert 20 evicts Expert 10
    dev->mark_expert_resident(l2, e2, bytes);
    TEST_CHECK(!dev->has_resident_expert(l1, e1), "Expert 10 evicted");

    // Re-demand Expert 10 -> triggers fresh demand transfer
    t = 2000.0;
    dev->set_time(t);
    uint64_t tid = dev->submit_demand_transfer(l1, e1, t, bytes, 10, 10, 1);
    TEST_CHECK(tid > 0, "Re-demand generates fresh demand transfer");
    dev->wait_for_demand_transfer(tid);

    col.finalize(4000.0, bytes);
    const auto& m = col.get_metrics();
    TEST_CHECK(m.fresh_demand_misses >= 1, "Must record demand miss for re-demanded expert");

    tests_passed = tests_passed + 1;
    return true;
}

// Scenario 11: Duplicate prediction skip
static bool test_scenario_11_duplicate_prediction_skip() {
    std::cout << "[Test 11] Scenario 11: Duplicate prediction skip\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    PrefetchConfig pcfg;
    pcfg.queue_capacity = 10;
    pcfg.duplicate_policy = DuplicatePolicy::REJECT_DUPLICATE;
    PrefetchCandidateQueue queue(pcfg, 8, 32);

    PrefetchCandidate c1;
    c1.candidate_id = 1101; c1.target_stage_id = 2; c1.expert_id = 8; c1.prediction_rank = 1; c1.prediction_score = 0.5;
    PrefetchCandidate c2;
    c2.candidate_id = 1102; c2.target_stage_id = 2; c2.expert_id = 8; c2.prediction_rank = 1; c2.prediction_score = 0.5;

    bool r1 = queue.enqueue_candidate(c1);
    bool r2 = queue.enqueue_candidate(c2);
    TEST_CHECK(r1, "First candidate enqueued");
    TEST_CHECK(!r2, "Duplicate candidate should be skipped");
    TEST_CHECK(queue.size() == 1, "Queue size must remain 1");

    tests_passed = tests_passed + 1;
    return true;
}

// Scenario 12: Multi-sequence identity isolation
static bool test_scenario_12_multi_sequence_isolation() {
    std::cout << "[Test 12] Scenario 12: Multi-sequence identity isolation\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    Device::Ptr dev = make_test_device(true, EvictionPolicy::LRU, 64.0);

    // Sequence 1 and Sequence 2 access the same layer but different experts
    double t = 1000.0;
    dev->set_time(t);
    uint64_t tid1 = dev->submit_demand_transfer(1, 10, t, 64000, 1, 1, 1);
    uint64_t tid2 = dev->submit_demand_transfer(1, 20, t, 64000, 2, 2, 1);

    TEST_CHECK(tid1 != tid2, "Transfers must have unique IDs");
    dev->wait_for_demand_transfer(tid1);
    dev->wait_for_demand_transfer(tid2);

    col.finalize(3000.0, 128000);
    const auto& m = col.get_metrics();
    TEST_CHECK(m.fresh_demand_misses == 2, "Must record 2 independent demand misses");

    tests_passed = tests_passed + 1;
    return true;
}

// Scenario 13: No-prefetch baseline verification
static bool test_scenario_13_no_prefetch_baseline() {
    std::cout << "[Test 13] Scenario 13: No-prefetch baseline verification\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    Device::Ptr dev = make_test_device(false, EvictionPolicy::LRU, 64.0);
    int layer_id = 1;
    int expert_id = 4;
    uint64_t bytes = 64000;
    double t = 1000.0;
    dev->set_time(t);

    uint64_t dem_tid = dev->submit_demand_transfer(layer_id, expert_id, t, bytes, 13, 13, 1);
    dev->wait_for_demand_transfer(dem_tid);

    col.finalize(3000.0, bytes);
    const auto& m = col.get_metrics();
    TEST_CHECK(m.timely_resident_hits == 0, "No prefetch hits in baseline");
    TEST_CHECK(m.late_inflight_attachments == 0, "No late in-flight hits in baseline");
    TEST_CHECK(m.fresh_demand_misses >= 1, "Must have demand misses");

    tests_passed = tests_passed + 1;
    return true;
}

// Scenario 14: Equivalence Test (Collector enabled vs disabled)
static bool test_scenario_14_equivalence_test() {
    std::cout << "[Test 14] Scenario 14: Equivalence Test (Enabled vs Disabled)\n";
    
    // Run 1: With collector disabled
    Device::Ptr dev_disabled = make_test_device(true, EvictionPolicy::LRU, 64.0);
    PrefetchMetricsCollector::getInstance().set_enabled(false);
    double t1 = 1000.0;
    dev_disabled->set_time(t1);
    uint64_t tid_d1 = dev_disabled->submit_prefetch_transfer(1, 5, t1, 64000, 1, 0, 1401, 14, 14, 1);
    t1 = 3000.0;
    dev_disabled->set_time(t1);
    dev_disabled->process_completed_prefetches(t1);
    bool res_d1 = dev_disabled->has_resident_expert(1, 5);

    // Run 2: With collector enabled
    Device::Ptr dev_enabled = make_test_device(true, EvictionPolicy::LRU, 64.0);
    PrefetchMetricsCollector::getInstance().set_enabled(true);
    PrefetchMetricsCollector::getInstance().reset();
    double t2 = 1000.0;
    dev_enabled->set_time(t2);
    uint64_t tid_e1 = dev_enabled->submit_prefetch_transfer(1, 5, t2, 64000, 1, 0, 1401, 14, 14, 1);
    t2 = 3000.0;
    dev_enabled->set_time(t2);
    dev_enabled->process_completed_prefetches(t2);
    bool res_e1 = dev_enabled->has_resident_expert(1, 5);

    TEST_CHECK(res_d1 == res_e1, "Residency outcome must be identical regardless of collector state");
    TEST_CHECK(tid_d1 == tid_e1, "Transfer IDs must match across enabled/disabled collector runs");

    tests_passed = tests_passed + 1;
    return true;
}

// Artifact dumping and schema verification
static bool test_artifact_dumping_and_schemas() {
    std::cout << "[Test 15] Artifact dumping and schema verification\n";
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    std::string out_dir = "eval_metrics_test_output";
    std::filesystem::remove_all(out_dir);
    col.set_output_dir(out_dir);

    // Record dummy events across all types
    std::vector<int> actuals = {3};
    col.record_prediction(1, 1, 1, 0, 1, 1, "CSTP", 4, 1, 3, 0.95, actuals, 1, 10.0, 5.0, 1000.0);
    col.record_candidate_admitted(1, 1, 1, 1, 0, 1, 3, 1, 0.95, 1050.0);
    col.record_candidate_dispatched(1, 1, 1, 1, 1, 3, 1100.0);
    col.record_transfer_submitted(1, 1, 1, 1, 1, 1, 3, "PREFETCH", 1, 64000, 1100.0);
    col.record_transfer_started(1, 1150.0);
    col.record_transfer_completed(1, 2100.0);
    col.record_hbm_resident(1, 3, 1, "PREFETCH", 2100.0);
    col.record_expert_demanded(1, 1, 1, 1, 1, 3, 2200.0, 2200.0, 0.0, 0.0, "TIMELY_PREFETCH", 1, 1);
    col.record_request_start(1, 1, 32, 1000.0, 1000.0);
    col.record_token_start(1, 1, 1, 0, 1000.0);
    col.record_token_end(1, 1, 1, 0, 2500.0);
    col.record_request_end(1, 1, 1, 2500.0);

    col.finalize(3000.0, 64000);
    col.dump_artifacts();

    std::vector<std::string> required_files = {
        "prediction_events.csv",
        "candidate_events.csv",
        "prefetch_transfer_events.csv",
        "expert_demand_events.csv",
        "request_latency.csv",
        "token_latency.csv",
        "prefetch_metrics.csv",
        "experiment_summary.json"
    };

    for (const auto& fname : required_files) {
        std::string full_path = out_dir + "/" + fname;
        TEST_CHECK(std::filesystem::exists(full_path), "Artifact must exist: " + fname);
        std::ifstream f(full_path);
        std::string first_line;
        std::getline(f, first_line);
        TEST_CHECK(!first_line.empty(), "Artifact must not be empty: " + fname);
    }

    std::filesystem::remove_all(out_dir);
    tests_passed = tests_passed + 1;
    return true;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Starting PrefetchMetricsCollector Test Suite (15 Tests)\n";
    std::cout << "========================================================\n";

    test_scenario_1_timely_prefetch();
    test_scenario_2_late_inflight();
    test_scenario_3_unused_prefetch();
    test_scenario_4_admission_rejection();
    test_scenario_5_cancellation();
    test_scenario_6_resident_hit();
    test_scenario_7_demand_deduplication();
    test_scenario_8_fresh_demand_miss();
    test_scenario_9_evicted_before_demand();
    test_scenario_10_redemand_after_eviction();
    test_scenario_11_duplicate_prediction_skip();
    test_scenario_12_multi_sequence_isolation();
    test_scenario_13_no_prefetch_baseline();
    test_scenario_14_equivalence_test();
    test_artifact_dumping_and_schemas();

    std::cout << "========================================================\n";
    std::cout << "Test Summary: " << tests_passed << " Passed, " << tests_failed << " Failed\n";
    std::cout << "========================================================\n";

    if (tests_failed > 0) {
        return 1;
    }
    return 0;
}
