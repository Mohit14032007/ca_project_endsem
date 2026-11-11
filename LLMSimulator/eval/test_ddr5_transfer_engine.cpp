#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "prefetch/ddr5_transfer_engine.h"

using namespace llm_system;

namespace {

int g_pass_count = 0;
int g_fail_count = 0;

void record_result(const std::string& test_name, bool passed, const std::string& details = "") {
    if (passed) {
        g_pass_count++;
        std::cout << "[PASS] " << test_name;
        if (!details.empty()) {
            std::cout << " (" << details << ")";
        }
        std::cout << "\n";
    } else {
        g_fail_count++;
        std::cerr << "[FAIL] " << test_name;
        if (!details.empty()) {
            std::cerr << " (" << details << ")";
        }
        std::cerr << "\n";
    }
}

bool approx_equal(double val, double expected, double tolerance_ns = 1.0) {
    double diff = std::abs(val - expected);
    if (diff <= tolerance_ns) {
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Test A: Single Demand Transfer
// ---------------------------------------------------------------------------
void test_a_single_demand() {
    DDR5TransferEngine engine;
    uint64_t expert_bytes = 88080384ULL;
    double expected_completion_ns = static_cast<double>(expert_bytes) / 22.578; // ~3,901,159.71 ns

    bool submitted = engine.submit_demand(1, 0, 10, 0.0, expert_bytes, 101);
    assert(submitted);

    engine.run_until_empty();

    const MemoryTransfer* t = engine.get_transfer(101);
    bool pass = true;
    if (t == nullptr) {
        pass = false;
    } else {
        if (t->state != TransferState::COMPLETED) {
            pass = false;
        }
        if (!approx_equal(t->completion_time_ns, expected_completion_ns, 0.01)) {
            pass = false;
        }
        if (t->transferred_bytes != static_cast<double>(expert_bytes)) {
            pass = false;
        }
        if (t->remaining_bytes != 0.0) {
            pass = false;
        }
    }

    if (!engine.verify_conservation_invariants()) {
        pass = false;
    }

    record_result("Test A: Single demand", pass, "Completed at " + std::to_string(t->completion_time_ns / 1e6) + " ms");
}

// ---------------------------------------------------------------------------
// Test B: Two Simultaneous Demands
// ---------------------------------------------------------------------------
void test_b_two_simultaneous_demands() {
    DDR5TransferEngine engine;
    uint64_t expert_bytes = 88080384ULL;
    double single_t_ns = static_cast<double>(expert_bytes) / 22.578;

    engine.submit_demand(1, 0, 11, 0.0, expert_bytes, 201);
    engine.submit_demand(1, 0, 12, 0.0, expert_bytes, 202);

    engine.run_until_empty();

    const MemoryTransfer* t1 = engine.get_transfer(201);
    const MemoryTransfer* t2 = engine.get_transfer(202);

    bool pass = true;
    if (t1 == nullptr || t2 == nullptr) {
        pass = false;
    } else {
        if (!approx_equal(t1->completion_time_ns, single_t_ns, 0.01)) {
            pass = false;
        }
        if (!approx_equal(t2->completion_time_ns, 2.0 * single_t_ns, 0.01)) {
            pass = false;
        }
        if (t1->state != TransferState::COMPLETED || t2->state != TransferState::COMPLETED) {
            pass = false;
        }
    }

    if (!engine.verify_conservation_invariants()) {
        pass = false;
    }

    record_result("Test B: Two simultaneous demands", pass,
                  "T1=" + std::to_string(t1->completion_time_ns / 1e6) + " ms, T2=" +
                  std::to_string(t2->completion_time_ns / 1e6) + " ms");
}

// ---------------------------------------------------------------------------
// Test C: Four Simultaneous Demands
// ---------------------------------------------------------------------------
void test_c_four_simultaneous_demands() {
    DDR5TransferEngine engine;
    uint64_t expert_bytes = 88080384ULL;
    double single_t_ns = static_cast<double>(expert_bytes) / 22.578;

    engine.submit_demand(1, 0, 13, 0.0, expert_bytes, 301);
    engine.submit_demand(1, 0, 14, 0.0, expert_bytes, 302);
    engine.submit_demand(1, 0, 15, 0.0, expert_bytes, 303);
    engine.submit_demand(1, 0, 16, 0.0, expert_bytes, 304);

    engine.run_until_empty();

    const MemoryTransfer* t1 = engine.get_transfer(301);
    const MemoryTransfer* t2 = engine.get_transfer(302);
    const MemoryTransfer* t3 = engine.get_transfer(303);
    const MemoryTransfer* t4 = engine.get_transfer(304);

    bool pass = true;
    if (t1 == nullptr || t2 == nullptr || t3 == nullptr || t4 == nullptr) {
        pass = false;
    } else {
        // T1 and T4 are directly calibrated; T2 and T3 are analytical FIFO expectations
        if (!approx_equal(t1->completion_time_ns, 1.0 * single_t_ns, 0.01)) {
            pass = false;
        }
        if (!approx_equal(t2->completion_time_ns, 2.0 * single_t_ns, 0.01)) {
            pass = false;
        }
        if (!approx_equal(t3->completion_time_ns, 3.0 * single_t_ns, 0.01)) {
            pass = false;
        }
        if (!approx_equal(t4->completion_time_ns, 4.0 * single_t_ns, 0.01)) {
            pass = false;
        }
    }

    if (!engine.verify_conservation_invariants()) {
        pass = false;
    }

    record_result("Test C: Four simultaneous demands", pass,
                  "T1=" + std::to_string(t1->completion_time_ns / 1e6) + "ms, T2=" +
                  std::to_string(t2->completion_time_ns / 1e6) + "ms, T3=" +
                  std::to_string(t3->completion_time_ns / 1e6) + "ms, T4=" +
                  std::to_string(t4->completion_time_ns / 1e6) + "ms");
}

// ---------------------------------------------------------------------------
// Test D: Two Prefetch Transfers
// ---------------------------------------------------------------------------
void test_d_two_prefetch_transfers() {
    DDR5TransferEngine engine;
    uint64_t expert_bytes = 88080384ULL;
    double single_t_ns = static_cast<double>(expert_bytes) / 22.578;

    engine.submit_prefetch(1, 0, 21, 0.0, expert_bytes, 0, 401);
    engine.submit_prefetch(1, 0, 22, 0.0, expert_bytes, 0, 402);

    engine.run_until_empty();

    const MemoryTransfer* t1 = engine.get_transfer(401);
    const MemoryTransfer* t2 = engine.get_transfer(402);

    bool pass = true;
    if (t1 == nullptr || t2 == nullptr) {
        pass = false;
    } else {
        if (!approx_equal(t1->completion_time_ns, single_t_ns, 0.01)) {
            pass = false;
        }
        if (!approx_equal(t2->completion_time_ns, 2.0 * single_t_ns, 0.01)) {
            pass = false;
        }
    }

    if (!engine.verify_conservation_invariants()) {
        pass = false;
    }

    record_result("Test D: Two prefetch transfers", pass,
                  "P1=" + std::to_string(t1->completion_time_ns / 1e6) + "ms, P2=" +
                  std::to_string(t2->completion_time_ns / 1e6) + "ms");
}

// ---------------------------------------------------------------------------
// Test E: Demand Before Prefetch Service
// ---------------------------------------------------------------------------
void test_e_demand_before_prefetch_service() {
    DDR5TransferEngine engine;
    uint64_t expert_bytes = 88080384ULL;
    double single_t_ns = static_cast<double>(expert_bytes) / 22.578;

    // Submit prefetch first, but at the exact same timestamp submit demand
    // Demand must be serviced ahead of prefetch
    MemoryTransfer pref;
    pref.transfer_id = 501;
    pref.transfer_class = TransferClass::PREFETCH;
    pref.total_bytes = expert_bytes;
    pref.arrival_time_ns = 0.0;

    MemoryTransfer dem;
    dem.transfer_id = 502;
    dem.transfer_class = TransferClass::DEMAND;
    dem.total_bytes = expert_bytes;
    dem.arrival_time_ns = 0.0;

    engine.submit_transfer(pref, 0.0);
    // While pref is active at t=0, submitting demand at t=0 must preempt/replace it immediately
    engine.submit_transfer(dem, 0.0);

    engine.run_until_empty();

    const MemoryTransfer* td = engine.get_transfer(502);
    const MemoryTransfer* tp = engine.get_transfer(501);

    bool pass = true;
    if (td == nullptr || tp == nullptr) {
        pass = false;
    } else {
        // Demand must finish first at 3.901 ms
        if (!approx_equal(td->completion_time_ns, single_t_ns, 0.01)) {
            pass = false;
        }
        // Prefetch must finish second at 7.802 ms
        if (!approx_equal(tp->completion_time_ns, 2.0 * single_t_ns, 0.01)) {
            pass = false;
        }
    }

    if (!engine.verify_conservation_invariants()) {
        pass = false;
    }

    record_result("Test E: Demand before prefetch service", pass,
                  "Demand completed at " + std::to_string(td->completion_time_ns / 1e6) + "ms, Prefetch at " +
                  std::to_string(tp->completion_time_ns / 1e6) + "ms");
}

// ---------------------------------------------------------------------------
// Test F: Demand Preempts Active Prefetch
// ---------------------------------------------------------------------------
void test_f_demand_preempts_active_prefetch() {
    DDR5TransferEngine engine;
    uint64_t expert_bytes = 88080384ULL;
    double single_t_ns = static_cast<double>(expert_bytes) / 22.578; // 3,901,159.71 ns

    // Start prefetch at t = 0
    engine.submit_prefetch(1, 0, 31, 0.0, expert_bytes, 0, 601);

    // At t = 1.0 ms (1,000,000 ns), demand arrives
    double preemption_time_ns = 1000000.0;
    engine.submit_demand(1, 0, 30, preemption_time_ns, expert_bytes, 602);

    const MemoryTransfer* tp_preempted = engine.get_transfer(601);
    assert(tp_preempted != nullptr);

    double expected_transferred_at_preempt = preemption_time_ns * 22.578; // 22,578,000 bytes
    double expected_remaining_at_preempt = static_cast<double>(expert_bytes) - expected_transferred_at_preempt;

    bool pre_check = true;
    if (!approx_equal(tp_preempted->transferred_bytes, expected_transferred_at_preempt, 1.0)) {
        pre_check = false;
    }
    if (!approx_equal(tp_preempted->remaining_bytes, expected_remaining_at_preempt, 1.0)) {
        pre_check = false;
    }
    if (tp_preempted->state != TransferState::PAUSED) {
        pre_check = false;
    }

    engine.run_until_empty();

    const MemoryTransfer* td = engine.get_transfer(602);
    const MemoryTransfer* tp = engine.get_transfer(601);

    bool pass = pre_check;
    if (td == nullptr || tp == nullptr) {
        pass = false;
    } else {
        // Demand started at 1.0 ms and ran for 3.90116 ms -> completed at 4.90116 ms
        double expected_demand_completion = preemption_time_ns + single_t_ns;
        if (!approx_equal(td->completion_time_ns, expected_demand_completion, 0.01)) {
            pass = false;
        }

        // Prefetch resumed at 4.90116 ms with 65,502,384 bytes remaining
        // Remaining prefetch duration = 65,502,384 / 22.578 = 2,901,159.71 ns
        // Prefetch completed at 4.90116 ms + 2.90116 ms = 7.80232 ms
        double expected_prefetch_completion = expected_demand_completion + (expected_remaining_at_preempt / 22.578);
        if (!approx_equal(tp->completion_time_ns, expected_prefetch_completion, 0.01)) {
            pass = false;
        }

        if (tp->transferred_bytes != static_cast<double>(expert_bytes)) {
            pass = false;
        }
        if (engine.get_metrics().prefetch_preemptions != 1) {
            pass = false;
        }
        if (engine.get_metrics().prefetch_resumptions != 1) {
            pass = false;
        }
    }

    if (!engine.verify_conservation_invariants()) {
        pass = false;
    }

    record_result("Test F: Demand preempts active prefetch", pass,
                  "Demand completed at " + std::to_string(td->completion_time_ns / 1e6) + "ms, Prefetch resumed and finished at " +
                  std::to_string(tp->completion_time_ns / 1e6) + "ms");
}

// ---------------------------------------------------------------------------
// Test G: Multiple Demands During Prefetch
// ---------------------------------------------------------------------------
void test_g_multiple_demands_during_prefetch() {
    DDR5TransferEngine engine;
    uint64_t expert_bytes = 88080384ULL;
    double single_t_ns = static_cast<double>(expert_bytes) / 22.578;

    // Start prefetch at t = 0
    engine.submit_prefetch(1, 0, 41, 0.0, expert_bytes, 0, 701);

    // At t = 1.0 ms, Demand 1 arrives
    engine.submit_demand(1, 0, 42, 1000000.0, expert_bytes, 702);

    // At t = 2.0 ms, Demand 2 arrives while Demand 1 is active
    engine.submit_demand(1, 0, 43, 2000000.0, expert_bytes, 703);

    engine.run_until_empty();

    const MemoryTransfer* tp = engine.get_transfer(701);
    const MemoryTransfer* td1 = engine.get_transfer(702);
    const MemoryTransfer* td2 = engine.get_transfer(703);

    bool pass = true;
    if (tp == nullptr || td1 == nullptr || td2 == nullptr) {
        pass = false;
    } else {
        // Demand 1: started at 1.0 ms -> completed at 1.0 ms + single_t_ns = 4.90116 ms
        double exp_d1_end = 1000000.0 + single_t_ns;
        if (!approx_equal(td1->completion_time_ns, exp_d1_end, 0.01)) {
            pass = false;
        }

        // Demand 2: started at 4.90116 ms -> completed at 4.90116 ms + single_t_ns = 8.80232 ms
        double exp_d2_end = exp_d1_end + single_t_ns;
        if (!approx_equal(td2->completion_time_ns, exp_d2_end, 0.01)) {
            pass = false;
        }

        // Prefetch: resumed at 8.80232 ms, transferred remaining bytes (3.90116 - 1.0 = 2.90116 ms)
        // -> completed at 8.80232 + 2.90116 = 11.70348 ms
        double exp_p_end = exp_d2_end + (single_t_ns - 1000000.0);
        if (!approx_equal(tp->completion_time_ns, exp_p_end, 0.01)) {
            pass = false;
        }

        if (tp->transferred_bytes != static_cast<double>(expert_bytes)) {
            pass = false;
        }
    }

    if (!engine.verify_conservation_invariants()) {
        pass = false;
    }

    record_result("Test G: Multiple demands during prefetch", pass,
                  "D1=" + std::to_string(td1->completion_time_ns / 1e6) + "ms, D2=" +
                  std::to_string(td2->completion_time_ns / 1e6) + "ms, Prefetch=" +
                  std::to_string(tp->completion_time_ns / 1e6) + "ms");
}

// ---------------------------------------------------------------------------
// Test H: Repeated Preemption
// ---------------------------------------------------------------------------
void test_h_repeated_preemption() {
    DDR5TransferEngine engine;
    uint64_t expert_bytes = 88080384ULL;
    double single_t_ns = static_cast<double>(expert_bytes) / 22.578;

    // Start prefetch at t = 0
    engine.submit_prefetch(1, 0, 51, 0.0, expert_bytes, 0, 801);

    // At t = 1.0 ms, preempt with D1
    engine.submit_demand(1, 0, 52, 1000000.0, expert_bytes, 802);

    // D1 completes at 1.0 ms + single_t_ns = 4.90116 ms
    // Prefetch resumes at 4.90116 ms
    // Let prefetch run for 1.0 ms (until t = 5.90116 ms), then preempt again with D2!
    double t_second_preempt = 1000000.0 + single_t_ns + 1000000.0;
    engine.submit_demand(1, 0, 53, t_second_preempt, expert_bytes, 803);

    engine.run_until_empty();

    const MemoryTransfer* tp = engine.get_transfer(801);
    const MemoryTransfer* td1 = engine.get_transfer(802);
    const MemoryTransfer* td2 = engine.get_transfer(803);

    bool pass = true;
    if (tp == nullptr || td1 == nullptr || td2 == nullptr) {
        pass = false;
    } else {
        // D2 ends at t_second_preempt + single_t_ns
        double exp_d2_end = t_second_preempt + single_t_ns;
        if (!approx_equal(td2->completion_time_ns, exp_d2_end, 0.01)) {
            pass = false;
        }

        // Prefetch had 2.0 ms transferred across the two intervals (22,578,000 * 2 = 45,156,000 B)
        // Remaining time = single_t_ns - 2.0 ms = 1,901,159.71 ns
        // Completion time = exp_d2_end + (single_t_ns - 2000000.0)
        double exp_p_end = exp_d2_end + (single_t_ns - 2000000.0);
        if (!approx_equal(tp->completion_time_ns, exp_p_end, 0.01)) {
            pass = false;
        }

        if (tp->transferred_bytes != static_cast<double>(expert_bytes)) {
            pass = false;
        }
        if (engine.get_metrics().prefetch_preemptions != 2) {
            pass = false;
        }
        if (engine.get_metrics().prefetch_resumptions != 2) {
            pass = false;
        }
    }

    if (!engine.verify_conservation_invariants()) {
        pass = false;
    }

    record_result("Test H: Repeated preemption", pass,
                  "Prefetch completed after 2 preemptions at " + std::to_string(tp->completion_time_ns / 1e6) + "ms");
}

// ---------------------------------------------------------------------------
// Test I: Exact Timestamp Collision
// ---------------------------------------------------------------------------
void test_i_exact_timestamp_collision() {
    DDR5TransferEngine engine;
    uint64_t expert_bytes = 88080384ULL;
    double single_t_ns = static_cast<double>(expert_bytes) / 22.578;

    // Start prefetch at t = 0. Its predicted completion is exactly single_t_ns
    engine.submit_prefetch(1, 0, 61, 0.0, expert_bytes, 0, 901);

    // Demand arrives at the EXACT completion timestamp of the active prefetch
    double exact_collision_time = single_t_ns;
    engine.submit_demand(1, 0, 62, exact_collision_time, expert_bytes, 902);

    engine.run_until_empty();

    const MemoryTransfer* tp = engine.get_transfer(901);
    const MemoryTransfer* td = engine.get_transfer(902);

    bool pass = true;
    if (tp == nullptr || td == nullptr) {
        pass = false;
    } else {
        // Prefetch must complete cleanly at exact_collision_time
        if (!approx_equal(tp->completion_time_ns, exact_collision_time, 0.001)) {
            pass = false;
        }
        // Demand must start at exact_collision_time and finish at exact_collision_time + single_t_ns
        if (!approx_equal(td->completion_time_ns, 2.0 * single_t_ns, 0.001)) {
            pass = false;
        }
        // Prefetch was NOT preempted because it completed exactly on time
        if (engine.get_metrics().prefetch_preemptions != 0) {
            pass = false;
        }
    }

    if (!engine.verify_conservation_invariants()) {
        pass = false;
    }

    record_result("Test I: Exact timestamp collision", pass,
                  "Prefetch completed at " + std::to_string(tp->completion_time_ns / 1e6) + "ms, Demand at " +
                  std::to_string(td->completion_time_ns / 1e6) + "ms");
}

// ---------------------------------------------------------------------------
// Test J: Stale Completion Event Invalidation
// ---------------------------------------------------------------------------
void test_j_stale_completion_event() {
    DDR5TransferEngine engine;
    uint64_t expert_bytes = 88080384ULL;
    double single_t_ns = static_cast<double>(expert_bytes) / 22.578; // 3.90116 ms

    // Start prefetch at t = 0. An event at 3.90116 ms (generation 2) is scheduled in event_queue_.
    engine.submit_prefetch(1, 0, 71, 0.0, expert_bytes, 0, 1001);

    // At t = 1.0 ms, preempt with Demand.
    // Prefetch state becomes PAUSED, generation becomes 3.
    // The old event at 3.90116 ms remains in event_queue_ with generation 2.
    engine.submit_demand(1, 0, 72, 1000000.0, expert_bytes, 1002);

    // Run past 3.90116 ms (run up to 4.5 ms)
    // At 3.90116 ms, the old event pops from the queue!
    engine.run_until(4500000.0);

    const MemoryTransfer* tp_mid = engine.get_transfer(1001);
    bool pass = true;
    if (tp_mid == nullptr) {
        pass = false;
    } else {
        // Prefetch must NOT have completed! It must still be waiting or paused
        if (tp_mid->state == TransferState::COMPLETED) {
            std::cerr << "[FAIL] Stale event incorrectly completed the paused prefetch!\n";
            pass = false;
        }
        // Verify stale event was recognized and counted
        if (engine.get_metrics().stale_events_ignored == 0) {
            pass = false;
        }
    }

    // Now run to completion
    engine.run_until_empty();

    const MemoryTransfer* tp_final = engine.get_transfer(1001);
    const MemoryTransfer* td_final = engine.get_transfer(1002);

    if (tp_final == nullptr || td_final == nullptr) {
        pass = false;
    } else {
        if (tp_final->state != TransferState::COMPLETED) {
            pass = false;
        }
        // Final completion of prefetch should be at 1.0 ms + 3.90116 ms (demand) + 2.90116 ms (remaining) = 7.80232 ms
        double exp_final = 1000000.0 + single_t_ns + (single_t_ns - 1000000.0);
        if (!approx_equal(tp_final->completion_time_ns, exp_final, 0.01)) {
            pass = false;
        }
    }

    if (!engine.verify_conservation_invariants()) {
        pass = false;
    }

    record_result("Test J: Stale completion event", pass,
                  "Stale events ignored=" + std::to_string(engine.get_metrics().stale_events_ignored) +
                  ", Prefetch correctly delayed to " + std::to_string(tp_final->completion_time_ns / 1e6) + "ms");
}

// ---------------------------------------------------------------------------
// Test K: Fractional Progress
// ---------------------------------------------------------------------------
void test_k_fractional_progress() {
    DDR5TransferEngine engine;
    uint64_t expert_bytes = 88080384ULL;

    // Start prefetch at t = 0
    engine.submit_prefetch(1, 0, 81, 0.0, expert_bytes, 0, 1101);

    // Preempt at an arbitrary non-round timestamp with fractional nanoseconds: 123456.789 ns
    double fractional_preempt_ns = 123456.789;
    engine.submit_demand(1, 0, 82, fractional_preempt_ns, expert_bytes, 1102);

    const MemoryTransfer* tp_preempt = engine.get_transfer(1101);
    double expected_bytes = fractional_preempt_ns * 22.578; // 2,787,407.382042 bytes
    double bytes_at_preempt = 0.0;

    bool pass = true;
    if (tp_preempt == nullptr) {
        pass = false;
    } else {
        bytes_at_preempt = tp_preempt->transferred_bytes;
        if (!approx_equal(tp_preempt->transferred_bytes, expected_bytes, 1e-4)) {
            pass = false;
        }
        double sum = tp_preempt->transferred_bytes + tp_preempt->remaining_bytes;
        double diff = std::abs(static_cast<double>(expert_bytes) - sum);
        if (diff > 1e-6) {
            pass = false;
        }
    }

    engine.run_until_empty();

    if (!engine.verify_conservation_invariants()) {
        pass = false;
    }

    record_result("Test K: Fractional progress", pass,
                  "Transferred at preempt=" + std::to_string(bytes_at_preempt) +
                  " B, conservation delta < 1e-6 B");
}

// ---------------------------------------------------------------------------
// Test L: Invalid Operations and Rejections
// ---------------------------------------------------------------------------
void test_l_invalid_operations() {
    DDR5TransferEngine engine;
    uint64_t expert_bytes = 88080384ULL;

    bool pass = true;

    // 1. Zero-byte transfer rejection
    bool zero_res = engine.submit_demand(1, 0, 91, 0.0, 0ULL, 1201);
    if (zero_res) {
        std::cerr << "[FAIL] Engine accepted zero-byte transfer!\n";
        pass = false;
    }

    // 2. Duplicate transfer ID rejection
    bool d1 = engine.submit_demand(1, 0, 92, 0.0, expert_bytes, 1202);
    if (!d1) {
        pass = false;
    }
    bool d2_dup = engine.submit_demand(1, 0, 93, 0.0, expert_bytes, 1202);
    if (d2_dup) {
        std::cerr << "[FAIL] Engine accepted duplicate transfer ID 1202!\n";
        pass = false;
    }

    // 3. Cancellation of queued transfer
    engine.submit_prefetch(1, 0, 94, 0.0, expert_bytes, 0, 1203);
    bool cancel_queued = engine.cancel_transfer(1203, 0.0);
    if (!cancel_queued) {
        std::cerr << "[FAIL] Failed to cancel queued transfer 1203!\n";
        pass = false;
    }
    const MemoryTransfer* t_canc = engine.get_transfer(1203);
    if (t_canc == nullptr || t_canc->state != TransferState::CANCELLED) {
        pass = false;
    }

    // 4. Cancellation of active transfer
    // 1202 is currently active
    bool cancel_active = engine.cancel_transfer(1202, 1000.0);
    if (!cancel_active) {
        std::cerr << "[FAIL] Failed to cancel active transfer 1202!\n";
        pass = false;
    }
    const MemoryTransfer* t_act_canc = engine.get_transfer(1202);
    if (t_act_canc == nullptr || t_act_canc->state != TransferState::CANCELLED) {
        pass = false;
    }

    // 5. Cancel non-existent transfer
    bool cancel_bad = engine.cancel_transfer(999999, 1000.0);
    if (cancel_bad) {
        pass = false;
    }

    if (!engine.verify_conservation_invariants()) {
        pass = false;
    }

    record_result("Test L: Invalid operations and cancellation", pass,
                  "Zero-byte rejected, duplicate ID rejected, cancellations verified");
}

} // namespace

int main() {
    std::cout << "======================================================================\n";
    std::cout << "PHASE 4C: Calibrated Event-Driven DDR5 Transfer Engine Unit Tests\n";
    std::cout << "Calibrated Rate: 22.578 B/ns (22.578 GB/s) | Serialized Channel Service\n";
    std::cout << "======================================================================\n\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    test_a_single_demand();
    test_b_two_simultaneous_demands();
    test_c_four_simultaneous_demands();
    test_d_two_prefetch_transfers();
    test_e_demand_before_prefetch_service();
    test_f_demand_preempts_active_prefetch();
    test_g_multiple_demands_during_prefetch();
    test_h_repeated_preemption();
    test_i_exact_timestamp_collision();
    test_j_stale_completion_event();
    test_k_fractional_progress();
    test_l_invalid_operations();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;

    std::cout << "\n======================================================================\n";
    std::cout << "TEST SUITE SUMMARY: " << g_pass_count << " PASSED, " << g_fail_count << " FAILED\n";
    std::cout << "Host Execution Wall-Clock Time: " << std::fixed << std::setprecision(3) << elapsed_ms.count() << " ms\n";
    std::cout << "======================================================================\n";

    if (g_fail_count > 0) {
        return 1;
    }
    return 0;
}
