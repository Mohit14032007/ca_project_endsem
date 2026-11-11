#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "hardware/cluster.h"
#include "hardware/device.h"
#include "hardware/hardware_config.h"
#include "prefetch/ddr5_transfer_engine.h"
#include "scheduler/scheduler.h"

using namespace llm_system;

void test_1_multiplier_config_parsing() {
    std::cout << "[RUNNING] Test 1: Config & Environment Multiplier Parsing...\n";

    SystemConfig cfg = A100;
    assert(cfg.compute_time_multiplier == 1.0);
    assert(cfg.clock_continuity == true);

    setenv("COMPUTE_MULTIPLIER", "12.0", 1);
    setenv("CLOCK_CONTINUITY", "1", 1);
    cfg.apply_environment_overrides();
    assert(cfg.compute_time_multiplier == 12.0);
    assert(cfg.clock_continuity == true);

    setenv("COMPUTE_MULTIPLIER", "16.0", 1);
    setenv("CLOCK_CONTINUITY", "0", 1);
    cfg.apply_environment_overrides();
    assert(cfg.compute_time_multiplier == 16.0);
    assert(cfg.clock_continuity == false);

    // Clean up environment
    unsetenv("COMPUTE_MULTIPLIER");
    unsetenv("CLOCK_CONTINUITY");

    std::cout << "  [PASS] Test 1: Multiplier and clock continuity correctly parsed from environment.\n";
}

void test_2_clock_continuity_timebase() {
    std::cout << "[RUNNING] Test 2: Inter-Token Clock Continuity in restartModuleGraph...\n";

    SystemConfig cfg = A100;
    cfg.clock_continuity = true;
    Cluster::Ptr cluster = Cluster::Create(cfg, nullptr);
    Device::Ptr dev = cluster->get_device(0);

    dev->status.device_time = 2500000; // 2.5 ms
    dev->status.low_time = 2500000;
    dev->status.high_time = 2500000;

    cluster->restartModuleGraph();

    assert(dev->status.device_time == 2500000);
    assert(dev->status.low_time == 2500000);
    assert(dev->status.high_time == 2500000);

    // Now test with clock continuity disabled
    cfg.clock_continuity = false;
    Cluster::Ptr cluster_reset = Cluster::Create(cfg, nullptr);
    Device::Ptr dev_reset = cluster_reset->get_device(0);
    dev_reset->status.device_time = 2500000;
    cluster_reset->restartModuleGraph();

    assert(dev_reset->status.device_time == 0);

    std::cout << "  [PASS] Test 2: Monotonic time preserved across module graph restart when enabled.\n";
}

void test_3_transfer_engine_window_overlap() {
    std::cout << "[RUNNING] Test 3: Expanded Compute Window Allows DDR5 Transfer Completion...\n";

    double bw = 22.578; // GB/s = bytes/ns
    uint64_t expert_bytes = 88080384;
    double isolated_transfer_time_ns = static_cast<double>(expert_bytes) / bw; // ~3,901,160 ns

    DDR5TransferEngine engine(bw);

    // Dispatch a speculative prefetch at t = 0
    uint64_t tid = engine.submit_prefetch_id(1, 10, 5, 0.0, expert_bytes);
    assert(tid != 0);

    // Simulation advances through stage computation
    // Baseline compute window: 2 stages * 0.183 ms = 0.366 ms (366,000 ns)
    double t_demand_m1 = 366000.0;
    engine.run_until(t_demand_m1);

    const MemoryTransfer* tr1 = engine.get_transfer(tid);
    assert(tr1 != nullptr);
    assert(!tr1->is_completed()); // Must NOT be completed at M=1
    assert(tr1->completion_time_ns > t_demand_m1);

    // Now consider scaled compute window at M=12: 2 stages * (0.183 ms * 12) = 4.396 ms (4,396,000 ns)
    double t_demand_m12 = 4396000.0;
    engine.run_until(t_demand_m12);

    const MemoryTransfer* tr12 = engine.get_transfer(tid);
    assert(tr12 != nullptr);
    assert(tr12->is_completed()); // Must be COMPLETED at M=12!
    assert(tr12->completion_time_ns <= t_demand_m12);

    std::cout << "  [PASS] Test 3: DDR5 transfer incomplete at M=1 window (366 us), successfully completes at M=12 window (4.396 ms)!\n";
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "PHASE 2F UNIT & INTEGRATION TESTS: COMPUTE MULTIPLIER & CLOCK CONTINUITY\n";
    std::cout << "============================================================\n";

    test_1_multiplier_config_parsing();
    test_2_clock_continuity_timebase();
    test_3_transfer_engine_window_overlap();

    std::cout << "\nALL PHASE 2F UNIT TESTS PASSED SUCCESSFULLY!\n";
    return 0;
}
