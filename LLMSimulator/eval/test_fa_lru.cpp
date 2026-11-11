#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#include <iomanip>
#include "hardware/device.h"
#include "prefetch/prefetch_config.h"
#include "prefetch/prefetch_metrics_collector.h"

using namespace llm_system;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[TEST FAILED] " << (msg) << " at line " << __LINE__ << std::endl; \
            return false; \
        } \
    } while (0)

// ======================================================================
// TASK 1: FREQUENCY DECAY AUDIT TESTS
// ======================================================================

bool test_1_decay_boundary_zero_diff() {
    Device dev;
    ResidentExpertEntry entry;
    entry.access_frequency = 10;
    entry.last_epoch = 5;
    dev.global_epoch = 5; // epoch_diff = 0

    dev.materialize_lazy_decay(entry);
    TEST_ASSERT((entry.access_frequency == 10), "epoch_diff=0 must not modify frequency");
    TEST_ASSERT((entry.last_epoch == 5), "last_epoch remains 5");
    std::cout << "  [PASS] Test 1: Decay boundary with zero epoch difference\n";
    return true;
}

bool test_2_decay_boundary_diff_1_and_2() {
    Device dev;
    ResidentExpertEntry entry;
    entry.access_frequency = 10;
    entry.last_epoch = 0;

    // epoch_diff = 1: floor(10 / 2) = 5
    dev.global_epoch = 1;
    dev.materialize_lazy_decay(entry);
    TEST_ASSERT((entry.access_frequency == 5), "epoch_diff=1: 10 >> 1 = 5");
    TEST_ASSERT((entry.last_epoch == 1), "last_epoch updated to 1");

    // epoch_diff = 2 from epoch 1: floor(5 / 4) = 1 (clamped to max(1, ...))
    dev.global_epoch = 3; // 3 - 1 = 2
    dev.materialize_lazy_decay(entry);
    TEST_ASSERT((entry.access_frequency == 1), "epoch_diff=2 on 5: 5 >> 2 = 1");
    TEST_ASSERT((entry.last_epoch == 3), "last_epoch updated to 3");

    std::cout << "  [PASS] Test 2: Decay boundary with epoch differences 1 and 2\n";
    return true;
}

bool test_3_decay_boundary_diff_30_31_and_large() {
    Device dev;
    ResidentExpertEntry entry;

    // 1. epoch_diff = 30
    entry.access_frequency = 100;
    entry.last_epoch = 0;
    dev.global_epoch = 30;
    dev.materialize_lazy_decay(entry);
    TEST_ASSERT((entry.access_frequency == 1), "epoch_diff=30 must floor at 1");
    TEST_ASSERT((entry.last_epoch == 30), "last_epoch updated to 30");

    // 2. epoch_diff = 31 (shift hazard boundary) -> assigns 1, not 0!
    entry.access_frequency = 1000;
    entry.last_epoch = 0;
    dev.global_epoch = 31;
    dev.materialize_lazy_decay(entry);
    TEST_ASSERT((entry.access_frequency == 1), "epoch_diff=31 must assign 1 directly without undefined shift");
    TEST_ASSERT((entry.last_epoch == 31), "last_epoch updated to 31");

    // 3. Very large epoch_diff (e.g. 1000)
    entry.access_frequency = 50000;
    entry.last_epoch = 10;
    dev.global_epoch = 1010;
    dev.materialize_lazy_decay(entry);
    TEST_ASSERT((entry.access_frequency == 1), "epoch_diff=1000 must assign 1 directly");
    TEST_ASSERT((entry.last_epoch == 1010), "last_epoch updated to 1010");

    std::cout << "  [PASS] Test 3: Decay boundary clamp for epoch_diff 30, 31, and 1000 (assigns 1, shift-safe)\n";
    return true;
}

bool test_4_decay_idempotence_and_underflow() {
    Device dev;
    ResidentExpertEntry entry;
    entry.access_frequency = 8;
    entry.last_epoch = 0;
    dev.global_epoch = 2; // epoch_diff = 2

    dev.materialize_lazy_decay(entry);
    TEST_ASSERT((entry.access_frequency == 2), "First materialization halves twice: 8 >> 2 = 2");
    TEST_ASSERT((entry.last_epoch == 2), "last_epoch updated to 2");

    // Second call in same epoch
    dev.materialize_lazy_decay(entry);
    TEST_ASSERT((entry.access_frequency == 2), "Second materialization in same epoch must be a no-op");
    TEST_ASSERT((entry.last_epoch == 2), "last_epoch remains 2");

    // Underflow guard: global_epoch < entry.last_epoch
    dev.global_epoch = 1; // 1 < 2
    dev.materialize_lazy_decay(entry);
    TEST_ASSERT((entry.access_frequency == 2), "global_epoch < last_epoch must not underflow or corrupt frequency");
    TEST_ASSERT((entry.last_epoch == 2), "last_epoch remains 2");

    std::cout << "  [PASS] Test 4: Decay idempotence in identical epoch and underflow protection\n";
    return true;
}

// ======================================================================
// TASK 4: PINNED-ENTRY PROTECTION & FA-LRU SELECTION TESTS
// ======================================================================

bool test_5_w1_lru_equivalence_no_pinning() {
    Device dev;
    dev.cache_capacity_bytes = 100;
    dev.eviction_policy = EvictionPolicy::FA_LRU;
    dev.fa_lru_window_size = 1;
    dev.reset_expert_cache();

    // Insert (1, 1), (1, 2)
    dev.mark_expert_resident(1, 1, 40);
    dev.mark_expert_resident(1, 2, 40);

    // Give (1, 1) high frequency
    dev.resident_experts[{1, 1}].access_frequency = 50;
    dev.resident_experts[{1, 2}].access_frequency = 1;

    // Because W=1, window only inspects front element (1, 1)
    auto victim_it = dev.select_fa_lru_victim();
    TEST_ASSERT((victim_it != dev.lru_expert_list.end()), "Victim iterator must be valid");
    TEST_ASSERT((std::get<0>(*victim_it) == 1 && std::get<1>(*victim_it) == 1),
                "With W=1, oldest entry (1, 1) must be selected regardless of frequency");

    std::cout << "  [PASS] Test 5: W=1 FA-LRU strict equivalence to LRU\n";
    return true;
}

bool test_6_fa_lru_victim_selection_min_freq() {
    Device dev;
    dev.cache_capacity_bytes = 200;
    dev.eviction_policy = EvictionPolicy::FA_LRU;
    dev.fa_lru_window_size = 4;
    dev.reset_expert_cache();

    // Insert 4 entries of size 40 (total 160 bytes)
    dev.mark_expert_resident(1, 1, 40); // Oldest (front)
    dev.mark_expert_resident(1, 2, 40);
    dev.mark_expert_resident(1, 3, 40);
    dev.mark_expert_resident(1, 4, 40); // Newest (back)

    dev.resident_experts[{1, 1}].access_frequency = 5;
    dev.resident_experts[{1, 2}].access_frequency = 2;
    dev.resident_experts[{1, 3}].access_frequency = 8;
    dev.resident_experts[{1, 4}].access_frequency = 4;

    auto victim_it = dev.select_fa_lru_victim();
    TEST_ASSERT((std::get<0>(*victim_it) == 1 && std::get<1>(*victim_it) == 2),
                "FA-LRU must select entry (1, 2) with lowest frequency F=2");

    std::cout << "  [PASS] Test 6: FA-LRU victim selection by minimum frequency in window\n";
    return true;
}

bool test_7_fa_lru_window_bounding() {
    Device dev;
    dev.cache_capacity_bytes = 200;
    dev.eviction_policy = EvictionPolicy::FA_LRU;
    dev.fa_lru_window_size = 2; // Window covers only bottom 2 entries
    dev.reset_expert_cache();

    dev.mark_expert_resident(1, 1, 40); // Oldest, index 0
    dev.mark_expert_resident(1, 2, 40); // index 1
    dev.mark_expert_resident(1, 3, 40); // index 2 (outside window)
    dev.mark_expert_resident(1, 4, 40); // Newest, index 3 (outside window)

    dev.resident_experts[{1, 1}].access_frequency = 5;
    dev.resident_experts[{1, 2}].access_frequency = 3;
    dev.resident_experts[{1, 3}].access_frequency = 1; // coldest, but outside window
    dev.resident_experts[{1, 4}].access_frequency = 9;

    auto victim_it = dev.select_fa_lru_victim();
    TEST_ASSERT((std::get<0>(*victim_it) == 1 && std::get<1>(*victim_it) == 2),
                "FA-LRU with W=2 must select (1, 2) because (1, 3) is outside candidate window");

    std::cout << "  [PASS] Test 7: FA-LRU window bounding (entries outside W protected)\n";
    return true;
}

bool test_8_fa_lru_strict_tie_breaking() {
    Device dev;
    dev.cache_capacity_bytes = 200;
    dev.eviction_policy = EvictionPolicy::FA_LRU;
    dev.fa_lru_window_size = 4;
    dev.reset_expert_cache();

    dev.mark_expert_resident(1, 1, 40); // Oldest (F=2)
    dev.mark_expert_resident(1, 2, 40); // (F=5)
    dev.mark_expert_resident(1, 3, 40); // (F=2) - tie with (1, 1)
    dev.mark_expert_resident(1, 4, 40); // (F=8)

    dev.resident_experts[{1, 1}].access_frequency = 2;
    dev.resident_experts[{1, 2}].access_frequency = 5;
    dev.resident_experts[{1, 3}].access_frequency = 2;
    dev.resident_experts[{1, 4}].access_frequency = 8;

    auto victim_it = dev.select_fa_lru_victim();
    TEST_ASSERT((std::get<0>(*victim_it) == 1 && std::get<1>(*victim_it) == 1),
                "Tie between (1, 1) and (1, 3) must select older entry (1, 1)");

    std::cout << "  [PASS] Test 8: FA-LRU strict tie-breaking to oldest candidate\n";
    return true;
}

bool test_9_pinned_oldest_with_unpinned_candidates() {
    Device dev;
    dev.cache_capacity_bytes = 200;
    dev.eviction_policy = EvictionPolicy::FA_LRU;
    dev.fa_lru_window_size = 3;
    dev.reset_expert_cache();

    dev.mark_expert_resident(1, 1, 40); // Oldest, pinned!
    dev.mark_expert_resident(1, 2, 40); // Unpinned, F=5
    dev.mark_expert_resident(1, 3, 40); // Unpinned, F=2

    dev.resident_experts[{1, 1}].is_pinned = true;
    dev.resident_experts[{1, 1}].access_frequency = 1;
    dev.resident_experts[{1, 2}].access_frequency = 5;
    dev.resident_experts[{1, 3}].access_frequency = 2;

    auto victim_it = dev.select_fa_lru_victim();
    TEST_ASSERT((victim_it != dev.lru_expert_list.end()), "Valid victim found");
    TEST_ASSERT((std::get<0>(*victim_it) == 1 && std::get<1>(*victim_it) == 3),
                "Pinned oldest entry (1, 1) must NOT be evicted; unpinned (1, 3) with F=2 selected");

    std::cout << "  [PASS] Test 9: Pinned oldest entry protected, unpinned candidate selected\n";
    return true;
}

bool test_10_all_window_candidates_pinned_fallback_unpinned() {
    Device dev;
    dev.cache_capacity_bytes = 200;
    dev.eviction_policy = EvictionPolicy::FA_LRU;
    dev.fa_lru_window_size = 2; // Window covers bottom 2 entries
    dev.reset_expert_cache();

    dev.mark_expert_resident(1, 1, 40); // In window, pinned!
    dev.mark_expert_resident(1, 2, 40); // In window, pinned!
    dev.mark_expert_resident(1, 3, 40); // Beyond window, unpinned!

    dev.resident_experts[{1, 1}].is_pinned = true;
    dev.resident_experts[{1, 2}].is_pinned = true;
    dev.resident_experts[{1, 3}].is_pinned = false;

    auto victim_it = dev.select_fa_lru_victim();
    TEST_ASSERT((victim_it != dev.lru_expert_list.end()), "Valid fallback victim found");
    TEST_ASSERT((std::get<0>(*victim_it) == 1 && std::get<1>(*victim_it) == 3),
                "When all window candidates are pinned, fallback selects first unpinned entry (1, 3)");

    std::cout << "  [PASS] Test 10: All window candidates pinned, fallback to unpinned beyond window\n";
    return true;
}

bool test_11_all_candidates_pinned_no_eviction() {
    Device dev;
    dev.cache_capacity_bytes = 80;
    dev.eviction_policy = EvictionPolicy::FA_LRU;
    dev.fa_lru_window_size = 2;
    dev.reset_expert_cache();

    dev.mark_expert_resident(1, 1, 40);
    dev.mark_expert_resident(1, 2, 40);

    dev.resident_experts[{1, 1}].is_pinned = true;
    dev.resident_experts[{1, 2}].is_pinned = true;

    auto victim_it = dev.select_fa_lru_victim();
    TEST_ASSERT((victim_it == dev.lru_expert_list.end()),
                "When all entries in cache are pinned, select_fa_lru_victim must return end()");

    // Attempting to admit a speculative prefetch when all entries are pinned must reject prefetch!
    dev.mark_expert_resident(1, 3, 40, ExpertClassification::DEMAND_VERIFIED, true); // prefetch admission
    TEST_ASSERT((dev.resident_experts.count({1, 3}) == 0),
                "Prefetch admission rejected when all resident entries are pinned and capacity is full");
    TEST_ASSERT((dev.resident_bytes == 80), "Residency remains capped at 80 bytes");

    std::cout << "  [PASS] Test 11: All candidates pinned protects cache from silent eviction\n";
    return true;
}

// ======================================================================
// TASK 2: DEMAND ACCOUNTING AUDIT TESTS (B=1, B=2, Overlapping, Projections)
// ======================================================================

bool test_12_demand_accounting_scenarios() {
    auto& col = PrefetchMetricsCollector::getInstance();

    // -------------------------------------------------------------
    // Scenario A: B = 1
    // 1 sequence requests (layer 1, expert 3) at decode_step 0. W1, W3, W2 execute.
    // -------------------------------------------------------------
    col.reset();
    col.set_enabled(true);
    col.set_current_decode_step(0);
    // W1, W3, W2
    col.record_expert_demanded(0, 1, 0, 100, 1, 3, 10.0, 10.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    col.record_expert_demanded(0, 1, 0, 100, 1, 3, 11.0, 11.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    col.record_expert_demanded(0, 1, 0, 100, 1, 3, 12.0, 12.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    col.finalize(100.0);

    AggregatedPrefetchMetrics m_a = col.get_metrics();
    TEST_ASSERT((m_a.tensor_actual_demands == 3), "Scenario A: Tensor demands = 3");
    TEST_ASSERT((m_a.routed_actual_demands == 1), "Scenario A: Routed demands = 1");
    TEST_ASSERT((m_a.coalesced_actual_demands == 1), "Scenario A: Coalesced demands = 1");

    // -------------------------------------------------------------
    // Scenario B: B = 2, same expert in same layer & decode step
    // Seq 0 and Seq 1 both request (layer 1, expert 3) in decode_step 0.
    // Expected: Raw tensor events = 6, Routed = 2, Coalesced = 1.
    // -------------------------------------------------------------
    col.reset();
    col.set_enabled(true);
    col.set_current_decode_step(0);
    // Seq 0: W1, W3, W2
    col.record_expert_demanded(0, 1, 0, 100, 1, 3, 10.0, 10.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    col.record_expert_demanded(0, 1, 0, 100, 1, 3, 11.0, 11.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    col.record_expert_demanded(0, 1, 0, 100, 1, 3, 12.0, 12.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    // Seq 1: W1, W3, W2
    col.record_expert_demanded(0, 1, 1, 100, 1, 3, 13.0, 13.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    col.record_expert_demanded(0, 1, 1, 100, 1, 3, 14.0, 14.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    col.record_expert_demanded(0, 1, 1, 100, 1, 3, 15.0, 15.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    col.finalize(100.0);

    AggregatedPrefetchMetrics m_b = col.get_metrics();
    TEST_ASSERT((m_b.tensor_actual_demands == 6), "Scenario B: Tensor demands = 6");
    TEST_ASSERT((m_b.routed_actual_demands == 2), "Scenario B: Routed demands = 2 (2 distinct sequences)");
    TEST_ASSERT((m_b.coalesced_actual_demands == 1), "Scenario B: Coalesced demands = 1 (same expert in same decode step)");

    // -------------------------------------------------------------
    // Scenario C: B = 2, sequences request different experts in same step
    // Seq 0 requests (1, 3), Seq 1 requests (1, 7) in decode_step 0.
    // -------------------------------------------------------------
    col.reset();
    col.set_enabled(true);
    // Seq 0: W1, W3, W2 for (1, 3)
    col.record_expert_demanded(0, 1, 0, 100, 1, 3, 10.0, 10.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    col.record_expert_demanded(0, 1, 0, 100, 1, 3, 11.0, 11.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    col.record_expert_demanded(0, 1, 0, 100, 1, 3, 12.0, 12.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    // Seq 1: W1, W3, W2 for (1, 7)
    col.record_expert_demanded(0, 1, 1, 100, 1, 7, 13.0, 13.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    col.record_expert_demanded(0, 1, 1, 100, 1, 7, 14.0, 14.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    col.record_expert_demanded(0, 1, 1, 100, 1, 7, 15.0, 15.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    col.finalize(100.0);

    AggregatedPrefetchMetrics m_c = col.get_metrics();
    TEST_ASSERT((m_c.tensor_actual_demands == 6), "Scenario C: Tensor demands = 6");
    TEST_ASSERT((m_c.routed_actual_demands == 2), "Scenario C: Routed demands = 2");
    TEST_ASSERT((m_c.coalesced_actual_demands == 2), "Scenario C: Coalesced demands = 2 (2 distinct experts)");

    // -------------------------------------------------------------
    // Scenario D: Two independent requests with overlapping token indices
    // Req 1 has seq 0, token_id=5 at decode_step 0.
    // Req 2 has seq 0, token_id=5 at decode_step 10.
    // -------------------------------------------------------------
    col.reset();
    col.set_enabled(true);
    // Req 1 at decode_step 0:
    col.record_expert_demanded(0, 1, 0, 5, 1, 3, 10.0, 10.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    col.record_expert_demanded(0, 1, 0, 5, 1, 3, 11.0, 11.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    col.record_expert_demanded(0, 1, 0, 5, 1, 3, 12.0, 12.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    // Req 2 at decode_step 10:
    col.record_expert_demanded(0, 2, 0, 5, 1, 3, 50.0, 50.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 10);
    col.record_expert_demanded(0, 2, 0, 5, 1, 3, 51.0, 51.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 10);
    col.record_expert_demanded(0, 2, 0, 5, 1, 3, 52.0, 52.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 10);
    col.finalize(100.0);

    AggregatedPrefetchMetrics m_d = col.get_metrics();
    TEST_ASSERT((m_d.tensor_actual_demands == 6), "Scenario D: Tensor demands = 6");
    TEST_ASSERT((m_d.routed_actual_demands == 2), "Scenario D: Routed demands = 2 (different requests)");
    TEST_ASSERT((m_d.coalesced_actual_demands == 2), "Scenario D: Coalesced demands = 2 (different decode steps 0 and 10)");

    // -------------------------------------------------------------
    // Scenario E: Repeated expert requests in different decode steps
    // Step 0 requests (1, 3). Step 1 requests (1, 3).
    // -------------------------------------------------------------
    col.reset();
    col.set_enabled(true);
    // Step 0:
    col.record_expert_demanded(0, 1, 0, 100, 1, 3, 10.0, 10.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    col.record_expert_demanded(0, 1, 0, 100, 1, 3, 11.0, 11.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    col.record_expert_demanded(0, 1, 0, 100, 1, 3, 12.0, 12.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 0);
    // Step 1:
    col.record_expert_demanded(0, 1, 0, 101, 1, 3, 20.0, 20.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 1);
    col.record_expert_demanded(0, 1, 0, 101, 1, 3, 21.0, 21.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 1);
    col.record_expert_demanded(0, 1, 0, 101, 1, 3, 22.0, 22.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, 1);
    col.finalize(100.0);

    AggregatedPrefetchMetrics m_e = col.get_metrics();
    TEST_ASSERT((m_e.tensor_actual_demands == 6), "Scenario E: Tensor demands = 6");
    TEST_ASSERT((m_e.routed_actual_demands == 2), "Scenario E: Routed demands = 2");
    TEST_ASSERT((m_e.coalesced_actual_demands == 2), "Scenario E: Coalesced demands = 2");

    // -------------------------------------------------------------
    // Scenario F: Multiple sequences with different prompt lengths
    // Seq 0 has prompt_len 10, token_id=10 at gen_step 0.
    // Seq 1 has prompt_len 30, token_id=30 at gen_step 0.
    // Both request (layer 2, expert 5) in decode step 0.
    // Expected: Tensor=6, Routed=2, Coalesced=1.
    // -------------------------------------------------------------
    col.reset();
    col.set_enabled(true);
    col.record_token_start(1, 0, 10, 0, 100.0);
    col.record_token_start(1, 1, 30, 0, 100.0);
    // Seq 0 W1, W3, W2 using 13-arg overload (resolves via token_to_gen_step_ to decode_step 0)
    col.record_expert_demanded(0, 1, 0, 10, 2, 5, 10.0, 10.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0);
    col.record_expert_demanded(0, 1, 0, 10, 2, 5, 11.0, 11.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0);
    col.record_expert_demanded(0, 1, 0, 10, 2, 5, 12.0, 12.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0);
    // Seq 1 W1, W3, W2 using 13-arg overload (resolves via token_to_gen_step_ to decode_step 0)
    col.record_expert_demanded(0, 1, 1, 30, 2, 5, 13.0, 13.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0);
    col.record_expert_demanded(0, 1, 1, 30, 2, 5, 14.0, 14.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0);
    col.record_expert_demanded(0, 1, 1, 30, 2, 5, 15.0, 15.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0);
    col.finalize(100.0);

    AggregatedPrefetchMetrics m_f = col.get_metrics();
    TEST_ASSERT((m_f.tensor_actual_demands == 6), "Scenario F: Tensor demands = 6");
    TEST_ASSERT((m_f.routed_actual_demands == 2), "Scenario F: Routed demands = 2");
    TEST_ASSERT((m_f.coalesced_actual_demands == 1), "Scenario F: Coalesced demands = 1 (different prompt lengths coalesce in step 0)");

    // -------------------------------------------------------------
    // Scenario G: Missing decode-step information (no token start, d_step < 0)
    // Ensures events lacking decode step do not silently collapse together.
    // -------------------------------------------------------------
    col.reset();
    col.set_enabled(true);
    // current_decode_step_ is -1 after reset, no token registered
    col.record_expert_demanded(0, 1, 0, 100, 1, 3, 10.0, 10.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, -1);
    col.record_expert_demanded(0, 2, 0, 200, 1, 3, 20.0, 20.0, 0.0, 0.0, "RESIDENT_HIT", 0, 0, -1);
    col.finalize(100.0);

    AggregatedPrefetchMetrics m_g = col.get_metrics();
    TEST_ASSERT((m_g.invalid_decode_step_demands == 2), "Scenario G: 2 invalid decode step demands flagged");
    TEST_ASSERT((m_g.coalesced_actual_demands == 2), "Scenario G: Invalid events do not collapse together");

    // -------------------------------------------------------------
    // Scenario H: Inspect individual demand records for decode_step identity
    // -------------------------------------------------------------
    const auto& dem_list = col.get_demands();
    TEST_ASSERT((dem_list.size() == 2), "Scenario H: 2 demands recorded");
    TEST_ASSERT((dem_list[0].decode_step == -1), "Scenario H: First demand has decode_step -1");
    TEST_ASSERT((dem_list[1].decode_step == -1), "Scenario H: Second demand has decode_step -1");

    std::cout << "  [PASS] Test 12: Comprehensive Demand Accounting Scenarios A-H Verified\n";
    std::cout << "         [DENOMINATORS] Scenario B (B=2 same expert): Raw Events=" << m_b.tensor_actual_demands
              << ", Routed=" << m_b.routed_actual_demands
              << ", Coalesced=" << m_b.coalesced_actual_demands << "\n";
    std::cout << "         [DENOMINATORS] Scenario D (Overlapping token_idx): Raw Events=" << m_d.tensor_actual_demands
              << ", Routed=" << m_d.routed_actual_demands
              << ", Coalesced=" << m_d.coalesced_actual_demands << "\n";
    std::cout << "         [DENOMINATORS] Scenario F (Diff Prompt Lengths): Raw Events=" << m_f.tensor_actual_demands
              << ", Routed=" << m_f.routed_actual_demands
              << ", Coalesced=" << m_f.coalesced_actual_demands << "\n";
    return true;
}

// ======================================================================
// TASK 3: PABI MUTUALLY EXCLUSIVE EVICTION ACCOUNTING TEST
// ======================================================================

bool test_13_pabi_mutually_exclusive_eviction_accounting() {
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    // Record one of each primary eviction type
    col.record_hbm_evicted(1, 1, "DEMAND_RESIDENT", 10.0);
    col.record_hbm_evicted(1, 2, "UNCONSUMED_PREFETCH", 20.0);
    col.record_hbm_evicted(1, 3, "PABI_SPECULATIVE", 30.0);

    // Also record a secondary prefetch-induced eviction for (1, 1)
    col.record_prefetch_induced_eviction(1, 1, 10.0);

    col.finalize(100.0);
    const auto& m = col.get_metrics();

    TEST_ASSERT((m.demand_resident_evictions == 1), "Demand resident evictions = 1");
    TEST_ASSERT((m.unconsumed_prefetch_evictions == 1), "Unconsumed prefetch evictions = 1");
    TEST_ASSERT((m.pabi_speculative_evictions == 1), "PABI speculative evictions = 1");
    TEST_ASSERT((m.prefetch_induced_evictions == 1), "Prefetch-induced evictions = 1 (secondary)");

    // Primary conservation identity: E_total = E_unconsumed + E_demand + E_pabi
    uint64_t primary_sum = m.unconsumed_prefetch_evictions + m.demand_resident_evictions + m.pabi_speculative_evictions;
    TEST_ASSERT((m.total_evictions == primary_sum), "E_total must strictly equal primary sum");
    TEST_ASSERT((m.total_evictions == 3), "Total evictions = 3");

    std::cout << "  [PASS] Test 13: PABI and primary eviction categories are strictly mutually exclusive (E_total = 3)\n";
    return true;
}

// ======================================================================
// TASK 4: SIMULATOR BATCH DECODE STEP PROPAGATION TEST
// ======================================================================

bool test_14_simulator_batch_decode_step_propagation() {
    auto& col = PrefetchMetricsCollector::getInstance();
    col.reset();
    col.set_enabled(true);

    // Simulate batch B=4 across 2 generation steps (step 0 and step 1)
    int input_len = 128;

    // --- STEP 0 ---
    for (int s = 0; s < 4; s++) {
        col.record_token_start(0, s, input_len + 0, 0, 1000.0);
    }
    // Seq 0 & Seq 1 demand (layer 0, expert 5): W1, W3, W2 each
    for (int p = 0; p < 3; p++) {
        col.record_expert_demanded(0, 0, 0, input_len + 0, 0, 5, 1000.0 + p, 1000.0 + p, 0.0, 0.0, "RESIDENT_HIT", 0, 0);
        col.record_expert_demanded(0, 0, 1, input_len + 0, 0, 5, 1000.0 + p, 1000.0 + p, 0.0, 0.0, "RESIDENT_HIT", 0, 0);
    }
    // Seq 2 demands (layer 0, expert 6): W1, W3, W2
    for (int p = 0; p < 3; p++) {
        col.record_expert_demanded(0, 0, 2, input_len + 0, 0, 6, 1000.0 + p, 1000.0 + p, 0.0, 0.0, "RESIDENT_HIT", 0, 0);
    }
    // Seq 3 demands (layer 0, expert 7): W1, W3, W2
    for (int p = 0; p < 3; p++) {
        col.record_expert_demanded(0, 0, 3, input_len + 0, 0, 7, 1000.0 + p, 1000.0 + p, 0.0, 0.0, "RESIDENT_HIT", 0, 0);
    }

    // --- STEP 1 ---
    for (int s = 0; s < 4; s++) {
        col.record_token_start(0, s, input_len + 1, 1, 2000.0);
    }
    // Seq 0 & Seq 1 demand (layer 0, expert 5): W1, W3, W2 each
    for (int p = 0; p < 3; p++) {
        col.record_expert_demanded(0, 0, 0, input_len + 1, 0, 5, 2000.0 + p, 2000.0 + p, 0.0, 0.0, "RESIDENT_HIT", 0, 0);
        col.record_expert_demanded(0, 0, 1, input_len + 1, 0, 5, 2000.0 + p, 2000.0 + p, 0.0, 0.0, "RESIDENT_HIT", 0, 0);
    }
    // Seq 2 & Seq 3 demand (layer 0, expert 8): W1, W3, W2 each
    for (int p = 0; p < 3; p++) {
        col.record_expert_demanded(0, 0, 2, input_len + 1, 0, 8, 2000.0 + p, 2000.0 + p, 0.0, 0.0, "RESIDENT_HIT", 0, 0);
        col.record_expert_demanded(0, 0, 3, input_len + 1, 0, 8, 2000.0 + p, 2000.0 + p, 0.0, 0.0, "RESIDENT_HIT", 0, 0);
    }

    col.finalize(3000.0);
    AggregatedPrefetchMetrics m = col.get_metrics();

    // Verifications:
    // Tensor events: 12 (step 0) + 12 (step 1) = 24
    TEST_ASSERT((m.tensor_actual_demands == 24), "Test 14: Tensor demands = 24");
    // Routed demands: 4 (step 0) + 4 (step 1) = 8
    TEST_ASSERT((m.routed_actual_demands == 8), "Test 14: Routed demands = 8");
    // Coalesced demands: 3 (step 0: exp 5, 6, 7) + 2 (step 1: exp 5, 8) = 5
    TEST_ASSERT((m.coalesced_actual_demands == 5), "Test 14: Coalesced demands = 5");
    TEST_ASSERT((m.invalid_decode_step_demands == 0), "Test 14: 0 invalid decode step demands");

    // Inspect individual records for correct decode step propagation
    const auto& demands = col.get_demands();
    TEST_ASSERT((demands.size() == 24), "Test 14: 24 total demand records");
    for (size_t i = 0; i < 12; i++) {
        TEST_ASSERT((demands[i].decode_step == 0), "Test 14: Step 0 records must have decode_step 0");
    }
    for (size_t i = 12; i < 24; i++) {
        TEST_ASSERT((demands[i].decode_step == 1), "Test 14: Step 1 records must have decode_step 1");
    }

    std::cout << "  [PASS] Test 14: Simulator Batch Decode Step Propagation (B=4, 2 steps: Tensor=24, Routed=8, Coalesced=5)\n";
    return true;
}

// ======================================================================
// MAIN TEST DRIVER
// ======================================================================

int main() {
    std::cout << "======================================================================\n";
    std::cout << "POINT 4 PHASE 2.6 EXTENDED UNIT TESTS: FA-LRU, ACCOUNTING & PINNING\n";
    std::cout << "======================================================================\n";

    bool all_passed = true;
    if (!test_1_decay_boundary_zero_diff()) all_passed = false;
    if (!test_2_decay_boundary_diff_1_and_2()) all_passed = false;
    if (!test_3_decay_boundary_diff_30_31_and_large()) all_passed = false;
    if (!test_4_decay_idempotence_and_underflow()) all_passed = false;
    if (!test_5_w1_lru_equivalence_no_pinning()) all_passed = false;
    if (!test_6_fa_lru_victim_selection_min_freq()) all_passed = false;
    if (!test_7_fa_lru_window_bounding()) all_passed = false;
    if (!test_8_fa_lru_strict_tie_breaking()) all_passed = false;
    if (!test_9_pinned_oldest_with_unpinned_candidates()) all_passed = false;
    if (!test_10_all_window_candidates_pinned_fallback_unpinned()) all_passed = false;
    if (!test_11_all_candidates_pinned_no_eviction()) all_passed = false;
    if (!test_12_demand_accounting_scenarios()) all_passed = false;
    if (!test_13_pabi_mutually_exclusive_eviction_accounting()) all_passed = false;
    if (!test_14_simulator_batch_decode_step_propagation()) all_passed = false;

    std::cout << "======================================================================\n";
    if (all_passed) {
        std::cout << "[SUCCESS] ALL 14 EXTENDED UNIT TESTS PASSED CLEANLY!\n";
        std::cout << "======================================================================\n";
        return 0;
    } else {
        std::cerr << "[FAILURE] ONE OR MORE UNIT TESTS FAILED!\n";
        std::cout << "======================================================================\n";
        return 1;
    }
}
