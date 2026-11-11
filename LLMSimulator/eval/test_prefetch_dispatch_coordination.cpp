#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "common/type.h"
#include "dram/dram_interface.h"
#include "dram/mmap_controller.h"
#include "hardware/cluster.h"
#include "hardware/device.h"
#include "hardware/hardware_config.h"
#include "hardware/layer_impl.h"
#include "module/tensor.h"
#include "prefetch/ddr5_transfer_engine.h"
#include "prefetch/prefetch_candidate_queue.h"
#include "prefetch/live_prediction_engine.h"

using namespace llm_system;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] " << (msg) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while (false)

static std::string get_hbm_cfg_path() {
    if (std::filesystem::exists("./dram_config_HBM3_80GB.yaml")) {
        return "./dram_config_HBM3_80GB.yaml";
    }
    if (std::filesystem::exists("LLMSimulator/build/dram_config_HBM3_80GB.yaml")) {
        return "LLMSimulator/build/dram_config_HBM3_80GB.yaml";
    }
    if (std::filesystem::exists("LLMSimulator/dram_config_HBM3_80GB.yaml")) {
        return "LLMSimulator/dram_config_HBM3_80GB.yaml";
    }
    return "./dram_config_HBM3_80GB.yaml";
}

static Device::Ptr create_test_device(int rank = 0, double bw_gbps = 22.578) {
    SystemConfig cfg = A100;
    cfg.hbm_dram_cfg_path = get_hbm_cfg_path();
    cfg.use_ramulator = true;
    cfg.use_event_driven_dram = true;
    cfg.ddr5_bandwidth_gbps = bw_gbps;
    Cluster::Ptr cluster = Cluster::Create(cfg, nullptr);
    return Device::Create(cfg, rank, cluster);
}

// Test 1: Candidate admission followed by actual dispatch
bool test_1_candidate_admission_and_dispatch() {
    std::cout << "[RUNNING] Test 1: Candidate admission followed by actual dispatch...\n";
    Device::Ptr device = create_test_device(0);

    PrefetchConfig pcfg;
    pcfg.enabled = true;
    pcfg.queue_enabled = true;
    pcfg.budget = 4;
    pcfg.prediction_horizon = 1;

    auto queue = std::make_shared<PrefetchCandidateQueue>(pcfg, 10, 64, "results/phase_4d2/test1");

    for (int i = 0; i < 4; i = i + 1) {
        PrefetchCandidate c;
        c.request_id = 1;
        c.token_id = 0;
        c.source_stage_id = 0;
        c.target_stage_id = 1;
        c.expert_id = 10 + i;
        c.prediction_rank = i + 1;
        c.prediction_score = 0.9 - 0.1 * static_cast<double>(i);
        c.prediction_horizon = 1;
        TEST_ASSERT(queue->enqueue_candidate(c), "Candidate admission failed");
    }

    TEST_ASSERT(queue->size() == 4, "Queue size must be 4");

    int dispatched = device->dispatch_prefetch_candidates(queue, 0.0, 4);
    TEST_ASSERT(dispatched == 4, "Must dispatch 4 candidates");
    TEST_ASSERT(device->prefetch_metrics.candidates_dispatched == 4, "Metric candidates_dispatched must be 4");

    for (int i = 0; i < 4; i = i + 1) {
        TEST_ASSERT(device->is_expert_inflight(1, 10 + i), "Expert must be inflight");
        TEST_ASSERT(!device->is_expert_resident(1, 10 + i), "Expert must not yet be resident");
    }

    TEST_ASSERT(device->ddr5_transfer_engine != nullptr, "Engine must be created");
    TEST_ASSERT(device->ddr5_transfer_engine->has_active_transfer(), "Engine must have active transfer");
    TEST_ASSERT(device->ddr5_transfer_engine->get_prefetch_queue_size() == 3, "Engine must have 3 queued prefetches");

    std::cout << "[PASS] Test 1: Candidate admission followed by actual dispatch verified\n";
    return true;
}

// Test 2: Prefetch submission is asynchronous (non-blocking)
bool test_2_prefetch_submission_asynchronous() {
    std::cout << "[RUNNING] Test 2: Prefetch submission is asynchronous...\n";
    Device::Ptr device = create_test_device(0);

    double submit_time_ns = 500000.0;
    device->set_time(submit_time_ns);

    uint64_t tid = device->submit_prefetch_transfer(1, 20, submit_time_ns);
    TEST_ASSERT(tid != 0, "Prefetch submission must return nonzero transfer ID");
    TEST_ASSERT(device->is_expert_inflight(1, 20), "Expert must be registered in inflight");
    TEST_ASSERT(!device->is_expert_resident(1, 20), "Expert must not be resident immediately");

    const auto* tr = device->ddr5_transfer_engine->get_transfer(tid);
    TEST_ASSERT(tr != nullptr, "Transfer pointer must not be null");
    TEST_ASSERT(tr->state == TransferState::ACTIVE, "Transfer must be ACTIVE immediately");
    TEST_ASSERT(!tr->is_completed(), "Transfer must NOT be completed immediately upon async submission");
    TEST_ASSERT(std::abs(device->ddr5_transfer_engine->get_current_time_ns() - submit_time_ns) < 1e-6,
                "Engine clock must not advance beyond submission timestamp");

    std::cout << "[PASS] Test 2: Prefetch submission is asynchronous verified\n";
    return true;
}

// Test 3: Prefetch completion causes HBM insertion
bool test_3_prefetch_completion_hbm_insertion() {
    std::cout << "[RUNNING] Test 3: Prefetch completion causes HBM insertion...\n";
    Device::Ptr device = create_test_device(0);

    uint64_t tid = device->submit_prefetch_transfer(1, 21, 0.0);
    TEST_ASSERT(tid != 0, "Submission failed");

    // Full duration: 88,080,384 / 22.578 = 3,901,159.71 ns
    // Advance to 4,000,000 ns
    double after_complete_time = 4000000.0;
    device->set_time(after_complete_time);
    device->process_completed_prefetches(after_complete_time);

    TEST_ASSERT(device->is_expert_resident(1, 21), "Expert must be resident in HBM after completion");
    TEST_ASSERT(!device->is_expert_inflight(1, 21), "Expert must no longer be inflight after insertion");
    TEST_ASSERT(device->prefetch_metrics.prefetches_inserted_hbm == 1, "Metric prefetches_inserted_hbm must be 1");
    TEST_ASSERT(device->prefetch_metrics.prefetch_transfers_completed == 1, "Metric prefetch_transfers_completed must be 1");
    TEST_ASSERT(device->prefetch_metrics.prefetch_useful_hits == 1, "Useful hit must be recorded on demand query");

    std::cout << "[PASS] Test 3: Prefetch completion causes HBM insertion verified\n";
    return true;
}

// Test 4: Already resident expert is not prefetched
bool test_4_already_resident_expert_not_prefetched() {
    std::cout << "[RUNNING] Test 4: Already resident expert is not prefetched...\n";
    Device::Ptr device = create_test_device(0);

    device->mark_expert_resident(1, 22, 88080384ULL);
    TEST_ASSERT(device->is_expert_resident(1, 22), "Expert must be resident");

    uint64_t tid = device->submit_prefetch_transfer(1, 22, 0.0);
    TEST_ASSERT(tid == 0, "Submitting prefetch for resident expert must return 0");
    TEST_ASSERT(!device->is_expert_inflight(1, 22), "Resident expert must not be marked inflight");

    PrefetchConfig pcfg;
    pcfg.enabled = true;
    pcfg.queue_enabled = true;
    pcfg.budget = 1;
    auto queue = std::make_shared<PrefetchCandidateQueue>(pcfg, 10, 64, "results/phase_4d2/test4");

    PrefetchCandidate c;
    c.request_id = 1;
    c.token_id = 0;
    c.target_stage_id = 1;
    c.expert_id = 22;
    c.prediction_rank = 1;
    c.prediction_score = 0.95;
    queue->enqueue_candidate(c);

    int dispatched = device->dispatch_prefetch_candidates(queue, 0.0, 1);
    TEST_ASSERT(dispatched == 0, "Must not dispatch resident candidate");
    TEST_ASSERT(device->prefetch_metrics.resident_prefetches_skipped == 1, "Skipped metric must increment");

    std::cout << "[PASS] Test 4: Already resident expert is not prefetched verified\n";
    return true;
}

// Test 5: Already in-flight expert is not prefetched again
bool test_5_already_inflight_expert_not_prefetched() {
    std::cout << "[RUNNING] Test 5: Already in-flight expert is not prefetched again...\n";
    Device::Ptr device = create_test_device(0);

    uint64_t tid1 = device->submit_prefetch_transfer(1, 23, 0.0);
    TEST_ASSERT(tid1 != 0, "First submission must succeed");

    uint64_t tid2 = device->submit_prefetch_transfer(1, 23, 1000.0);
    TEST_ASSERT(tid2 == 0, "Second submission for same inflight expert must return 0");

    PrefetchConfig pcfg;
    pcfg.enabled = true;
    pcfg.queue_enabled = true;
    pcfg.budget = 1;
    auto queue = std::make_shared<PrefetchCandidateQueue>(pcfg, 10, 64, "results/phase_4d2/test5");

    PrefetchCandidate c;
    c.request_id = 2;
    c.token_id = 0;
    c.target_stage_id = 1;
    c.expert_id = 23;
    c.prediction_rank = 1;
    c.prediction_score = 0.99;
    queue->enqueue_candidate(c);

    int dispatched = device->dispatch_prefetch_candidates(queue, 1000.0, 1);
    TEST_ASSERT(dispatched == 0, "Must not dispatch candidate for inflight expert");
    TEST_ASSERT(device->prefetch_metrics.duplicate_prefetches_prevented == 1, "Duplicate metric must increment");

    std::cout << "[PASS] Test 5: Already in-flight expert is not prefetched again verified\n";
    return true;
}

// Test 6: Demand arriving during active prefetch promotes the same transfer
bool test_6_demand_arriving_active_prefetch_promotes() {
    std::cout << "[RUNNING] Test 6: Demand arriving during active prefetch promotes the same transfer...\n";
    Device::Ptr device = create_test_device(0);

    uint64_t tid_pref = device->submit_prefetch_transfer(1, 24, 0.0);
    TEST_ASSERT(tid_pref != 0, "Prefetch submission failed");

    double demand_time_ns = 1000000.0; // 1 ms
    device->set_time(demand_time_ns);

    uint64_t tid_dem = device->submit_demand_transfer(1, 24, demand_time_ns);
    TEST_ASSERT(tid_dem == tid_pref, "Demand must attach to existing prefetch transfer ID");
    TEST_ASSERT(device->prefetch_metrics.demand_inflight_attachments == 1, "Attachment count must be 1");
    TEST_ASSERT(device->prefetch_metrics.prefetch_promotions_to_demand == 1, "Promotion count must be 1");

    const auto* tr = device->ddr5_transfer_engine->get_transfer(tid_dem);
    TEST_ASSERT(tr != nullptr, "Transfer pointer must not be null");
    TEST_ASSERT(tr->transfer_class == TransferClass::DEMAND, "Class must be promoted to DEMAND");
    TEST_ASSERT(tr->state == TransferState::ACTIVE, "Transfer must remain ACTIVE");

    double expected_transferred = demand_time_ns * 22.578;
    TEST_ASSERT(std::abs(tr->transferred_bytes - expected_transferred) < 1.0, "Transferred bytes progress preserved");
    TEST_ASSERT(std::abs(tr->remaining_bytes - (88080384.0 - expected_transferred)) < 1.0, "Remaining bytes preserved");

    std::cout << "[PASS] Test 6: Demand arriving during active prefetch promotes verified\n";
    return true;
}

// Test 7: Demand arriving during queued prefetch promotes the same transfer
bool test_7_demand_arriving_queued_prefetch_promotes() {
    std::cout << "[RUNNING] Test 7: Demand arriving during queued prefetch promotes the same transfer...\n";
    Device::Ptr device = create_test_device(0);

    uint64_t tid1 = device->submit_prefetch_transfer(1, 25, 0.0); // ACTIVE
    uint64_t tid2 = device->submit_prefetch_transfer(1, 26, 0.0); // QUEUED
    TEST_ASSERT(tid1 != 0 && tid2 != 0, "Submissions failed");

    const auto* tr2_before = device->ddr5_transfer_engine->get_transfer(tid2);
    TEST_ASSERT(tr2_before->state == TransferState::QUEUED, "Transfer 2 must initially be QUEUED");

    double demand_time_ns = 500000.0;
    device->set_time(demand_time_ns);

    uint64_t tid_dem = device->submit_demand_transfer(1, 26, demand_time_ns);
    TEST_ASSERT(tid_dem == tid2, "Demand must attach to existing queued transfer ID");
    TEST_ASSERT(device->prefetch_metrics.prefetch_promotions_to_demand == 1, "Promotion count must be 1");

    const auto* tr2_after = device->ddr5_transfer_engine->get_transfer(tid2);
    TEST_ASSERT(tr2_after->transfer_class == TransferClass::DEMAND, "Class must be promoted to DEMAND");
    TEST_ASSERT(tr2_after->state == TransferState::ACTIVE, "Promoted demand must preempt active prefetch and become ACTIVE");

    const auto* tr1_after = device->ddr5_transfer_engine->get_transfer(tid1);
    TEST_ASSERT(tr1_after->state == TransferState::PAUSED, "Active prefetch 1 must be preempted to PAUSED");

    std::cout << "[PASS] Test 7: Demand arriving during queued prefetch promotes verified\n";
    return true;
}

// Test 8: Remaining transfer time is charged correctly
bool test_8_remaining_transfer_time_charged_correctly() {
    std::cout << "[RUNNING] Test 8: Remaining transfer time is charged correctly...\n";
    Device::Ptr device = create_test_device(0);

    uint64_t tid = device->submit_prefetch_transfer(1, 27, 0.0);
    TEST_ASSERT(tid != 0, "Submission failed");

    double demand_time_ns = 1000000.0; // 1 ms
    device->set_time(demand_time_ns);

    uint64_t dem_tid = device->submit_demand_transfer(1, 27, demand_time_ns);
    TEST_ASSERT(dem_tid == tid, "Must attach to existing transfer");

    bool completed = device->wait_for_demand_transfer(dem_tid);
    TEST_ASSERT(completed, "Wait for transfer must succeed");

    const auto* tr = device->ddr5_transfer_engine->get_transfer(dem_tid);
    TEST_ASSERT(tr != nullptr && tr->is_completed(), "Transfer must be completed");

    double remaining_duration_ns = tr->completion_time_ns - demand_time_ns;
    double expected_remaining_ns = (88080384.0 - (1000000.0 * 22.578)) / 22.578;

    TEST_ASSERT(std::abs(remaining_duration_ns - expected_remaining_ns) < 1.0,
                "Remaining transfer duration charged must match remaining bytes / bandwidth");
    TEST_ASSERT(remaining_duration_ns < 3900000.0,
                "Remaining duration must be strictly less than full transfer duration");

    std::cout << "[PASS] Test 8: Remaining transfer time is charged correctly verified (charged "
              << remaining_duration_ns / 1e6 << " ms instead of 3.90116 ms)\n";
    return true;
}

// Test 9: Transfer byte count is not duplicated
bool test_9_transfer_byte_count_not_duplicated() {
    std::cout << "[RUNNING] Test 9: Transfer byte count is not duplicated...\n";
    Device::Ptr device = create_test_device(0);

    uint64_t tid = device->submit_prefetch_transfer(1, 28, 0.0);
    device->submit_demand_transfer(1, 28, 1000000.0);
    device->wait_for_demand_transfer(tid);

    const auto* tr = device->ddr5_transfer_engine->get_transfer(tid);
    TEST_ASSERT(tr != nullptr, "Transfer null");
    TEST_ASSERT(std::abs(tr->transferred_bytes - 88080384.0) < 1e-6, "Transferred bytes must equal exact expert size");

    const auto& metrics = device->ddr5_transfer_engine->get_metrics();
    TEST_ASSERT(metrics.total_bytes_transferred == 88080384ULL, "Total engine transferred bytes must be exactly 88080384");
    TEST_ASSERT(device->ddr5_transfer_engine->verify_conservation_invariants(), "Conservation invariants must hold");

    std::cout << "[PASS] Test 9: Transfer byte count is not duplicated verified\n";
    return true;
}

// Test 10: Multiple requests share one in-flight expert transfer
bool test_10_multi_request_share_inflight_transfer() {
    std::cout << "[RUNNING] Test 10: Multiple requests share one in-flight expert transfer...\n";
    Device::Ptr device = create_test_device(0);

    uint64_t tid1 = device->submit_demand_transfer(2, 10, 0.0);
    uint64_t tid2 = device->submit_demand_transfer(2, 10, 500.0);
    uint64_t tid3 = device->submit_demand_transfer(2, 10, 1000.0);

    TEST_ASSERT(tid1 != 0, "Submission 1 failed");
    TEST_ASSERT(tid1 == tid2, "Request 2 must share transfer ID with Request 1");
    TEST_ASSERT(tid2 == tid3, "Request 3 must share transfer ID with Request 1");
    TEST_ASSERT(device->prefetch_metrics.demand_inflight_attachments == 2, "Attachment count must be 2");

    const auto& metrics = device->ddr5_transfer_engine->get_metrics();
    TEST_ASSERT(metrics.total_demand_submitted == 1, "Only 1 demand transfer must be submitted to engine");

    std::cout << "[PASS] Test 10: Multiple requests share one in-flight expert transfer verified\n";
    return true;
}

// Test 11: Completed transfer before insertion does not cause duplicate loading
bool test_11_completed_transfer_before_insertion_no_duplicate() {
    std::cout << "[RUNNING] Test 11: Completed transfer before insertion does not cause duplicate loading...\n";
    Device::Ptr device = create_test_device(0);

    uint64_t tid = device->submit_prefetch_transfer(2, 11, 0.0);
    TEST_ASSERT(tid != 0, "Submission failed");

    // Advance engine clock to 5 ms: transfer finishes at 3.90116 ms
    device->ddr5_transfer_engine->run_until(5000000.0);
    const auto* tr = device->ddr5_transfer_engine->get_transfer(tid);
    TEST_ASSERT(tr != nullptr && tr->is_completed(), "Transfer must be complete in engine");

    // Demand arrives at 5 ms BEFORE process_completed_prefetches is explicitly called
    device->set_time(5000000.0);
    uint64_t dem_tid = device->submit_demand_transfer(2, 11, 5000000.0);

    TEST_ASSERT(dem_tid == tid, "Demand must attach to existing completed transfer ID, not create duplicate");
    const auto& metrics = device->ddr5_transfer_engine->get_metrics();
    TEST_ASSERT(metrics.total_demand_submitted == 0, "No new demand transfer must be created in engine");
    TEST_ASSERT(metrics.total_prefetch_submitted == 1, "Only initial prefetch exists");

    device->process_completed_prefetches(5000000.0);
    TEST_ASSERT(device->is_expert_resident(2, 11), "Expert must now be resident in HBM");

    std::cout << "[PASS] Test 11: Completed transfer before insertion does not cause duplicate loading verified\n";
    return true;
}

// Test 12: Candidate expiration does not corrupt active transfer state
bool test_12_candidate_expiration_safety() {
    std::cout << "[RUNNING] Test 12: Candidate expiration does not corrupt active transfer state...\n";
    Device::Ptr device = create_test_device(0);

    PrefetchConfig pcfg;
    pcfg.enabled = true;
    pcfg.queue_enabled = true;
    pcfg.budget = 1;
    auto queue = std::make_shared<PrefetchCandidateQueue>(pcfg, 10, 64, "results/phase_4d2/test12");

    PrefetchCandidate c;
    c.request_id = 5;
    c.token_id = 0;
    c.source_stage_id = 0;
    c.target_stage_id = 3;
    c.expert_id = 30;
    c.prediction_rank = 1;
    c.prediction_score = 0.85;
    queue->enqueue_candidate(c);

    int dispatched = device->dispatch_prefetch_candidates(queue, 0.0, 1);
    TEST_ASSERT(dispatched == 1, "Candidate must be dispatched");
    TEST_ASSERT(device->is_expert_inflight(3, 30), "Expert must be inflight");

    // Router arrives at target stage 3 with actual experts {31, 32} (misprediction)
    int expired = queue->reconcile_stage(5, 0, 3, {31, 32});
    TEST_ASSERT(expired == 1, "Candidate must be expired as miss");

    // Verify transfer continues cleanly without memory corruption or abort
    device->set_time(4000000.0);
    device->process_completed_prefetches(4000000.0);

    TEST_ASSERT(device->is_expert_resident(3, 30), "Expert must be inserted into HBM despite candidate expiration");
    TEST_ASSERT(!device->is_expert_inflight(3, 30), "Inflight cleared");

    std::cout << "[PASS] Test 12: Candidate expiration does not corrupt active transfer state verified\n";
    return true;
}

// Test 13: Failed/cancelled transfer never inserts an incomplete expert
bool test_13_failed_cancelled_transfer_never_inserts() {
    std::cout << "[RUNNING] Test 13: Failed/cancelled transfer never inserts an incomplete expert...\n";
    Device::Ptr device = create_test_device(0);

    uint64_t tid = device->submit_prefetch_transfer(2, 13, 0.0);
    TEST_ASSERT(tid != 0, "Submission failed");

    // Cancel transfer midway at 1 ms
    bool cancelled = device->ddr5_transfer_engine->cancel_transfer(tid, 1000000.0);
    TEST_ASSERT(cancelled, "Transfer cancellation must succeed");

    device->set_time(5000000.0);
    device->process_completed_prefetches(5000000.0);

    TEST_ASSERT(!device->is_expert_resident(2, 13), "Cancelled transfer must NEVER insert incomplete expert into HBM");
    TEST_ASSERT(!device->is_expert_inflight(2, 13), "Inflight registry must be cleared of cancelled transfer");
    TEST_ASSERT(device->prefetch_metrics.cancelled_prefetches == 1, "Cancelled prefetch metric must increment");

    std::cout << "[PASS] Test 13: Failed/cancelled transfer never inserts an incomplete expert verified\n";
    return true;
}

// Test 14: LRU behavior remains correct after prefetch insertion
bool test_14_lru_behavior_preserved_after_prefetch() {
    std::cout << "[RUNNING] Test 14: LRU behavior remains correct after prefetch insertion...\n";
    Device::Ptr device = create_test_device(0);

    // Limit cache capacity to exactly 2 experts
    device->cache_capacity_bytes = 2 * 88080384LL;
    device->reset_expert_cache();

    // Prefetch two experts
    device->submit_prefetch_transfer(3, 1, 0.0);
    device->submit_prefetch_transfer(3, 2, 0.0);

    // Run until both complete (2 * 3.90116 ms = 7.80232 ms)
    device->set_time(8000000.0);
    device->process_completed_prefetches(8000000.0);

    TEST_ASSERT(device->is_expert_resident(3, 1), "Expert 1 must be resident");
    TEST_ASSERT(device->is_expert_resident(3, 2), "Expert 2 must be resident");
    TEST_ASSERT(device->resident_bytes == 2 * 88080384LL, "Resident bytes must match capacity");

    // Note: Querying (3, 1) and then (3, 2) makes (3, 1) LRU and (3, 2) MRU
    // Now prefetch a 3rd expert (3, 3)
    device->submit_prefetch_transfer(3, 3, 8000000.0);
    device->set_time(12000000.0);
    device->process_completed_prefetches(12000000.0);

    TEST_ASSERT(!device->is_expert_resident(3, 1), "Expert 1 must be evicted by standard LRU");
    TEST_ASSERT(device->is_expert_resident(3, 2), "Expert 2 must remain resident");
    TEST_ASSERT(device->is_expert_resident(3, 3), "Expert 3 must be resident");
    TEST_ASSERT(device->resident_bytes == 2 * 88080384LL, "Resident bytes must remain bounded by capacity");

    std::cout << "[PASS] Test 14: LRU behavior remains correct after prefetch insertion verified\n";
    return true;
}

// Test 15: Device 0 and Device 1 remain isolated
bool test_15_device_isolation() {
    std::cout << "[RUNNING] Test 15: Device 0 and Device 1 remain isolated...\n";
    Device::Ptr dev0 = create_test_device(0);
    Device::Ptr dev1 = create_test_device(1);

    dev0->submit_prefetch_transfer(4, 1, 0.0);

    TEST_ASSERT(dev0->is_expert_inflight(4, 1), "Dev 0 must have expert inflight");
    TEST_ASSERT(!dev1->is_expert_inflight(4, 1), "Dev 1 must NOT have expert inflight");

    dev0->set_time(4000000.0);
    dev0->process_completed_prefetches(4000000.0);

    TEST_ASSERT(dev0->is_expert_resident(4, 1), "Dev 0 must have expert resident");
    TEST_ASSERT(!dev1->is_expert_resident(4, 1), "Dev 1 must NOT have expert resident");

    std::cout << "[PASS] Test 15: Device 0 and Device 1 remain isolated verified\n";
    return true;
}

// Test 16: Original blocking Ramulator baseline remains unchanged
bool test_16_original_blocking_ramulator_baseline_unchanged() {
    std::cout << "[RUNNING] Test 16: Original blocking Ramulator baseline remains unchanged...\n";
    SystemConfig cfg = A100;
    cfg.hbm_dram_cfg_path = get_hbm_cfg_path();
    cfg.use_ramulator = true;
    cfg.use_event_driven_dram = false; // Baseline default

    Cluster::Ptr cluster = Cluster::Create(cfg, nullptr);
    Device::Ptr device = Device::Create(cfg, 0, cluster);

    TEST_ASSERT(device->ddr5_transfer_engine == nullptr, "Engine must be nullptr when event-driven mode disabled");
    TEST_ASSERT(device->config.use_event_driven_dram == false, "Config flag must remain false");

    std::cout << "[PASS] Test 16: Original blocking Ramulator baseline remains unchanged verified\n";
    return true;
}

// Test 17: Event-driven demand-only mode from Phase 4D.1 continues to pass
bool test_17_demand_only_regression_phase4d1() {
    std::cout << "[RUNNING] Test 17: Event-driven demand-only mode from Phase 4D.1 continues to pass...\n";
    Device::Ptr device = create_test_device(0);

    uint64_t tid = device->submit_demand_transfer(5, 1, 0.0);
    TEST_ASSERT(tid != 0, "Demand submission failed");
    TEST_ASSERT(device->is_expert_inflight(5, 1), "Expert must be inflight");

    bool ok = device->wait_for_demand_transfer(tid);
    TEST_ASSERT(ok, "Demand transfer wait must succeed");

    const auto* tr = device->ddr5_transfer_engine->get_transfer(tid);
    TEST_ASSERT(tr != nullptr && tr->is_completed(), "Transfer must be completed");
    TEST_ASSERT(std::abs(tr->completion_time_ns - 3901159.71) < 1.0, "Latency must be 3.90116 ms");

    device->mark_expert_resident(5, 1, 88080384ULL);
    device->clear_inflight_expert(5, 1);

    TEST_ASSERT(device->is_expert_resident(5, 1), "Expert must be resident");
    TEST_ASSERT(!device->is_expert_inflight(5, 1), "Inflight must be cleared");

    std::cout << "[PASS] Test 17: Event-driven demand-only mode from Phase 4D.1 continues to pass verified\n";
    return true;
}

// Test 18: Demand priority preempts an unrelated active prefetch and preserves its progress
bool test_18_demand_preempts_unrelated_active_prefetch() {
    std::cout << "[RUNNING] Test 18: Demand priority preempts an unrelated active prefetch...\n";
    Device::Ptr device = create_test_device(0);

    // Prefetch P1 starts at t = 0.0 for (6, 1)
    uint64_t tid_p1 = device->submit_prefetch_transfer(6, 1, 0.0);
    TEST_ASSERT(tid_p1 != 0, "P1 submission failed");

    // At t = 1 ms, unrelated demand D1 arrives for (6, 2)
    double demand_time_ns = 1000000.0;
    device->set_time(demand_time_ns);

    uint64_t tid_d1 = device->submit_demand_transfer(6, 2, demand_time_ns);
    TEST_ASSERT(tid_d1 != 0, "D1 submission failed");
    TEST_ASSERT(tid_d1 != tid_p1, "D1 must have distinct transfer ID from P1");

    const auto* tr_p1 = device->ddr5_transfer_engine->get_transfer(tid_p1);
    TEST_ASSERT(tr_p1->state == TransferState::PAUSED, "P1 must be preempted to PAUSED state");
    TEST_ASSERT(std::abs(tr_p1->transferred_bytes - 22578000.0) < 1.0, "P1 progress at preemption preserved");

    const auto* tr_d1 = device->ddr5_transfer_engine->get_transfer(tid_d1);
    TEST_ASSERT(tr_d1->state == TransferState::ACTIVE, "D1 must immediately become ACTIVE");

    // Wait for D1 to complete
    bool d1_ok = device->wait_for_demand_transfer(tid_d1);
    TEST_ASSERT(d1_ok, "D1 wait failed");
    TEST_ASSERT(std::abs(tr_d1->completion_time_ns - 4901159.71) < 1.0,
                "D1 completion must be 1.0 ms + 3.90116 ms = 4.90116 ms");

    device->mark_expert_resident(6, 2, 88080384ULL);
    device->clear_inflight_expert(6, 2);

    // Advance to 8 ms: P1 resumes after D1 and finishes
    device->set_time(8000000.0);
    device->process_completed_prefetches(8000000.0);

    TEST_ASSERT(tr_p1->is_completed(), "P1 must resume and complete after D1 finishes");
    double expected_p1_complete = 4901159.71 + (88080384.0 - 22578000.0) / 22.578;
    TEST_ASSERT(std::abs(tr_p1->completion_time_ns - expected_p1_complete) < 1.0,
                "P1 completion timestamp must be exactly 7.80232 ms");

    TEST_ASSERT(device->is_expert_resident(6, 1), "P1 expert must be resident in HBM");
    TEST_ASSERT(device->is_expert_resident(6, 2), "D1 expert must be resident in HBM");
    TEST_ASSERT(device->ddr5_transfer_engine->verify_conservation_invariants(), "Conservation invariants must hold");

    std::cout << "[PASS] Test 18: Demand priority preempts unrelated prefetch verified (P1 finished at "
              << tr_p1->completion_time_ns / 1e6 << " ms)\n";
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "======================================================================\n";
    std::cout << "PHASE 4D.2: Prediction-Driven Prefetch Dispatch and In-Flight Demand Coordination\n";
    std::cout << "Calibrated Rate: 22.578 B/ns | Serialized Channel Service | 18 Test Cases\n";
    std::cout << "======================================================================\n\n";

    int passed = 0;
    int failed = 0;

    auto run_test = [&](bool (*test_fn)(), const std::string& name) {
        if (test_fn()) {
            passed = passed + 1;
        } else {
            failed = failed + 1;
            std::cerr << "[FAILED] " << name << "\n";
        }
    };

    run_test(test_1_candidate_admission_and_dispatch, "Test 1");
    run_test(test_2_prefetch_submission_asynchronous, "Test 2");
    run_test(test_3_prefetch_completion_hbm_insertion, "Test 3");
    run_test(test_4_already_resident_expert_not_prefetched, "Test 4");
    run_test(test_5_already_inflight_expert_not_prefetched, "Test 5");
    run_test(test_6_demand_arriving_active_prefetch_promotes, "Test 6");
    run_test(test_7_demand_arriving_queued_prefetch_promotes, "Test 7");
    run_test(test_8_remaining_transfer_time_charged_correctly, "Test 8");
    run_test(test_9_transfer_byte_count_not_duplicated, "Test 9");
    run_test(test_10_multi_request_share_inflight_transfer, "Test 10");
    run_test(test_11_completed_transfer_before_insertion_no_duplicate, "Test 11");
    run_test(test_12_candidate_expiration_safety, "Test 12");
    run_test(test_13_failed_cancelled_transfer_never_inserts, "Test 13");
    run_test(test_14_lru_behavior_preserved_after_prefetch, "Test 14");
    run_test(test_15_device_isolation, "Test 15");
    run_test(test_16_original_blocking_ramulator_baseline_unchanged, "Test 16");
    run_test(test_17_demand_only_regression_phase4d1, "Test 17");
    run_test(test_18_demand_preempts_unrelated_active_prefetch, "Test 18");

    std::cout << "\n======================================================================\n";
    std::cout << "TEST SUITE SUMMARY: " << passed << " PASSED, " << failed << " FAILED\n";
    std::cout << "======================================================================\n";

    if (failed > 0) {
        return 1;
    }
    return 0;
}

