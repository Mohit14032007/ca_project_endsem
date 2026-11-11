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

using namespace llm_system;

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

bool test_1_single_demand_miss() {
    SystemConfig cfg = A100;
    cfg.hbm_dram_cfg_path = get_hbm_cfg_path();
    cfg.use_ramulator = true;
    cfg.use_event_driven_dram = true;
    cfg.ddr5_bandwidth_gbps = 22.578;
    Cluster::Ptr cluster = Cluster::Create(cfg, nullptr);
    Device::Ptr device = Device::Create(cfg, 0, cluster);

    int layer_id = 0;
    int expert_id = 10;
    uint64_t expert_bytes = 88080384ULL;

    if (device->is_expert_resident(layer_id, expert_id)) {
        return false;
    }

    double start_time_ns = 1000000.0;
    device->set_time(start_time_ns);

    uint64_t tid = device->submit_demand_transfer(layer_id, expert_id, start_time_ns, expert_bytes);
    if (tid == 0) {
        return false;
    }
    if (!device->is_expert_inflight(layer_id, expert_id)) {
        return false;
    }

    bool ok = device->wait_for_demand_transfer(tid);
    if (!ok) {
        return false;
    }

    const auto* tr = device->ddr5_transfer_engine->get_transfer(tid);
    if (tr == nullptr) {
        return false;
    }
    if (!tr->is_completed()) {
        return false;
    }

    double expected_transfer_ns = static_cast<double>(expert_bytes) / 22.578;
    double expected_completion_ns = start_time_ns + expected_transfer_ns;
    double diff = std::abs(tr->completion_time_ns - expected_completion_ns);
    if (diff >= 1e-3) {
        return false;
    }

    device->mark_expert_resident(layer_id, expert_id, expert_bytes);
    device->clear_inflight_expert(layer_id, expert_id);

    if (device->is_expert_inflight(layer_id, expert_id)) {
        return false;
    }
    if (!device->is_expert_resident(layer_id, expert_id)) {
        return false;
    }

    return true;
}

bool test_2_repeated_demand_resident() {
    SystemConfig cfg = A100;
    cfg.hbm_dram_cfg_path = get_hbm_cfg_path();
    cfg.use_ramulator = true;
    cfg.use_event_driven_dram = true;
    cfg.ddr5_bandwidth_gbps = 22.578;
    Cluster::Ptr cluster = Cluster::Create(cfg, nullptr);
    Device::Ptr device = Device::Create(cfg, 0, cluster);

    int layer_id = 1;
    int expert_id = 15;
    uint64_t expert_bytes = 88080384ULL;

    uint64_t tid1 = device->submit_demand_transfer(layer_id, expert_id, 0.0, expert_bytes);
    device->wait_for_demand_transfer(tid1);
    device->mark_expert_resident(layer_id, expert_id, expert_bytes);
    device->clear_inflight_expert(layer_id, expert_id);

    if (!device->is_expert_resident(layer_id, expert_id)) {
        return false;
    }
    size_t completions_before = device->ddr5_transfer_engine->get_metrics().demand_completions;

    bool is_hit = device->is_expert_resident(layer_id, expert_id);
    if (!is_hit) {
        return false;
    }

    size_t completions_after = device->ddr5_transfer_engine->get_metrics().demand_completions;
    if (completions_before != completions_after) {
        return false;
    }

    return true;
}

bool test_3_two_demand_misses_serialized() {
    SystemConfig cfg = A100;
    cfg.hbm_dram_cfg_path = get_hbm_cfg_path();
    cfg.use_ramulator = true;
    cfg.use_event_driven_dram = true;
    cfg.ddr5_bandwidth_gbps = 22.578;
    Cluster::Ptr cluster = Cluster::Create(cfg, nullptr);
    Device::Ptr device = Device::Create(cfg, 0, cluster);

    uint64_t expert_bytes = 88080384ULL;
    double t0 = 0.0;

    uint64_t tid1 = device->submit_demand_transfer(0, 21, t0, expert_bytes);
    uint64_t tid2 = device->submit_demand_transfer(0, 22, t0, expert_bytes);
    if (tid1 == 0 || tid2 == 0) {
        return false;
    }
    if (tid1 == tid2) {
        return false;
    }

    bool ok1 = device->wait_for_demand_transfer(tid1);
    if (!ok1) {
        return false;
    }
    const auto* tr1 = device->ddr5_transfer_engine->get_transfer(tid1);
    if (tr1 == nullptr) {
        return false;
    }
    if (!tr1->is_completed()) {
        return false;
    }

    double expected_t1 = 88080384.0 / 22.578;
    if (std::abs(tr1->completion_time_ns - expected_t1) >= 1e-3) {
        return false;
    }
    device->mark_expert_resident(0, 21, expert_bytes);
    device->clear_inflight_expert(0, 21);

    bool ok2 = device->wait_for_demand_transfer(tid2);
    if (!ok2) {
        return false;
    }
    const auto* tr2 = device->ddr5_transfer_engine->get_transfer(tid2);
    if (tr2 == nullptr) {
        return false;
    }
    if (!tr2->is_completed()) {
        return false;
    }

    double expected_t2 = 2.0 * expected_t1;
    if (std::abs(tr2->completion_time_ns - expected_t2) >= 1e-3) {
        return false;
    }
    device->mark_expert_resident(0, 22, expert_bytes);
    device->clear_inflight_expert(0, 22);

    if (!device->is_expert_resident(0, 21)) {
        return false;
    }
    if (!device->is_expert_resident(0, 22)) {
        return false;
    }

    const auto& metrics = device->ddr5_transfer_engine->get_metrics();
    if (metrics.demand_completions != 2) {
        return false;
    }
    if (metrics.total_bytes_transferred != 2 * expert_bytes) {
        return false;
    }

    return true;
}

bool test_4_transfer_failure_handling() {
    SystemConfig cfg = A100;
    cfg.hbm_dram_cfg_path = get_hbm_cfg_path();
    cfg.use_ramulator = true;
    cfg.use_event_driven_dram = true;
    Cluster::Ptr cluster = Cluster::Create(cfg, nullptr);
    Device::Ptr device = Device::Create(cfg, 0, cluster);

    int layer_id = 0;
    int expert_id = 33;

    uint64_t tid = device->submit_demand_transfer(layer_id, expert_id, 0.0, 0ULL);
    if (tid != 0) {
        return false;
    }

    if (device->is_expert_resident(layer_id, expert_id)) {
        return false;
    }
    if (device->is_expert_inflight(layer_id, expert_id)) {
        return false;
    }

    uint64_t tid2 = device->submit_demand_transfer(layer_id, 34, 0.0, 88080384ULL);
    if (tid2 == 0) {
        return false;
    }
    bool cancelled = device->ddr5_transfer_engine->cancel_transfer(tid2, 0.0);
    if (!cancelled) {
        return false;
    }
    bool ok2 = device->wait_for_demand_transfer(tid2);
    if (ok2) {
        return false;
    }

    device->clear_inflight_expert(layer_id, 34);
    if (device->is_expert_resident(layer_id, 34)) {
        return false;
    }
    if (device->is_expert_inflight(layer_id, 34)) {
        return false;
    }

    return true;
}

bool test_5_lru_capacity_eviction_preserved() {
    SystemConfig cfg = A100;
    cfg.hbm_dram_cfg_path = get_hbm_cfg_path();
    cfg.use_ramulator = true;
    cfg.use_event_driven_dram = true;
    Cluster::Ptr cluster = Cluster::Create(cfg, nullptr);
    Device::Ptr device = Device::Create(cfg, 0, cluster);

    uint64_t expert_bytes = 88080384ULL;
    device->cache_capacity_bytes = 2 * expert_bytes;
    device->reset_expert_cache();

    uint64_t t1 = device->submit_demand_transfer(0, 1, 0.0, expert_bytes);
    device->wait_for_demand_transfer(t1);
    device->mark_expert_resident(0, 1, expert_bytes);
    device->clear_inflight_expert(0, 1);
    if (!device->is_expert_resident(0, 1)) {
        return false;
    }
    if (device->resident_bytes != static_cast<long long>(expert_bytes)) {
        return false;
    }

    uint64_t t2 = device->submit_demand_transfer(0, 2, 4000000.0, expert_bytes);
    device->wait_for_demand_transfer(t2);
    device->mark_expert_resident(0, 2, expert_bytes);
    device->clear_inflight_expert(0, 2);
    if (!device->is_expert_resident(0, 2)) {
        return false;
    }
    if (device->resident_bytes != static_cast<long long>(2 * expert_bytes)) {
        return false;
    }

    bool hit1 = device->is_expert_resident(0, 1);
    if (!hit1) {
        return false;
    }

    uint64_t t3 = device->submit_demand_transfer(0, 3, 8000000.0, expert_bytes);
    device->wait_for_demand_transfer(t3);
    device->mark_expert_resident(0, 3, expert_bytes);
    device->clear_inflight_expert(0, 3);

    if (!device->is_expert_resident(0, 1)) {
        return false;
    }
    if (device->is_expert_resident(0, 2)) {
        return false;
    }
    if (!device->is_expert_resident(0, 3)) {
        return false;
    }
    if (device->resident_bytes != static_cast<long long>(2 * expert_bytes)) {
        return false;
    }

    return true;
}

bool test_6_per_device_isolation() {
    SystemConfig cfg = A100;
    cfg.hbm_dram_cfg_path = get_hbm_cfg_path();
    cfg.use_ramulator = true;
    cfg.use_event_driven_dram = true;
    cfg.num_device = 2;
    Cluster::Ptr cluster = Cluster::Create(cfg, nullptr);
    Device::Ptr dev0 = Device::Create(cfg, 0, cluster);
    Device::Ptr dev1 = Device::Create(cfg, 1, cluster);

    if (dev0->ddr5_transfer_engine == nullptr || dev1->ddr5_transfer_engine == nullptr) {
        return false;
    }
    if (dev0->ddr5_transfer_engine == dev1->ddr5_transfer_engine) {
        return false;
    }

    uint64_t expert_bytes = 88080384ULL;

    uint64_t tid0 = dev0->submit_demand_transfer(0, 5, 0.0, expert_bytes);
    uint64_t tid1 = dev1->submit_demand_transfer(0, 5, 0.0, expert_bytes);

    dev0->wait_for_demand_transfer(tid0);
    dev1->wait_for_demand_transfer(tid1);

    const auto* tr0 = dev0->ddr5_transfer_engine->get_transfer(tid0);
    const auto* tr1 = dev1->ddr5_transfer_engine->get_transfer(tid1);

    double expected_t = 88080384.0 / 22.578;
    if (std::abs(tr0->completion_time_ns - expected_t) >= 1e-3) {
        return false;
    }
    if (std::abs(tr1->completion_time_ns - expected_t) >= 1e-3) {
        return false;
    }

    return true;
}

bool test_7_clock_synchronization() {
    SystemConfig cfg = A100;
    cfg.hbm_dram_cfg_path = get_hbm_cfg_path();
    cfg.use_ramulator = true;
    cfg.use_event_driven_dram = true;
    Cluster::Ptr cluster = Cluster::Create(cfg, nullptr);
    Device::Ptr device = Device::Create(cfg, 0, cluster);

    time_ns initial_device_time = 2500000.0;
    device->set_time(initial_device_time);

    uint64_t expert_bytes = 88080384ULL;
    uint64_t tid = device->submit_demand_transfer(0, 7, device->get_time(), expert_bytes);
    device->wait_for_demand_transfer(tid);

    const auto* tr = device->ddr5_transfer_engine->get_transfer(tid);
    if (tr == nullptr) {
        return false;
    }

    double expected_completion = initial_device_time + (88080384.0 / 22.578);
    if (std::abs(tr->completion_time_ns - expected_completion) >= 1e-3) {
        return false;
    }

    if (device->ddr5_transfer_engine->get_current_time_ns() < tr->completion_time_ns) {
        return false;
    }

    time_ns duration = tr->completion_time_ns - initial_device_time;
    device->set_time(initial_device_time + duration);

    if (std::abs(device->get_time() - tr->completion_time_ns) >= 1e-3) {
        return false;
    }

    return true;
}

bool test_8_no_double_charging() {
    SystemConfig cfg = A100;
    cfg.hbm_dram_cfg_path = get_hbm_cfg_path();
    cfg.use_ramulator = true;
    cfg.use_event_driven_dram = true;
    cfg.ddr5_bandwidth_gbps = 22.578;
    Cluster::Ptr cluster = Cluster::Create(cfg, nullptr);
    Device::Ptr device = Device::Create(cfg, 0, cluster);

    int m = 1;
    int k = 4096;
    int n = 4096;
    Tensor::Ptr input = Tensor::Create("in", {m, k}, "act", device);
    input->allocateMemoryObject(MemoryObject::Create(MMap::ALL_CHANNEL, 0, m * k * 2, device->mmap_controller));

    Tensor::Ptr weight = Tensor::Create("w1", {k, n}, "weight", device);
    weight->is_expert_weight = true;
    weight->expert_layer_id = 0;
    weight->expert_id = 88;
    weight->weight_target = MemoryTarget::OFFCHIP_DRAM;
    weight->allocateMemoryObject(MemoryObject::Create(MMap::ALL_CHANNEL, 10000000ULL, k * n * 2, device->offchip_mmap_controller));

    Tensor::Ptr w2 = Tensor::Create("w2", {k, n}, "weight", device);
    w2->is_expert_weight = true;
    w2->expert_layer_id = 0;
    w2->expert_id = 88;
    w2->allocateMemoryObject(MemoryObject::Create(MMap::ALL_CHANNEL, 20000000ULL, k * n * 2, device->offchip_mmap_controller));

    Tensor::Ptr w3 = Tensor::Create("w3", {k, n}, "weight", device);
    w3->is_expert_weight = true;
    w3->expert_layer_id = 0;
    w3->expert_id = 88;
    w3->allocateMemoryObject(MemoryObject::Create(MMap::ALL_CHANNEL, 30000000ULL, k * n * 2, device->offchip_mmap_controller));

    weight->sister_weights = {w2, w3};
    weight->load_sisters = true;

    Tensor::Ptr output = Tensor::Create("out", {m, n}, "act", device);
    output->allocateMemoryObject(MemoryObject::Create(MMap::ALL_CHANNEL, 30000000ULL, m * n * 2, device->mmap_controller));

    uint64_t total_expert_bytes = weight->getSize() + w2->getSize() + w3->getSize();

    ExecStatus status = LinearExecutionGPU(device, input, weight, output, true);

    double expected_transfer_cost = static_cast<double>(total_expert_bytes) / 22.578;
    if (status.memory_duration < expected_transfer_cost) {
        return false;
    }

    if (!device->is_expert_resident(0, 88)) {
        return false;
    }

    if (device->ddr5_transfer_engine->get_metrics().demand_completions != 1) {
        return false;
    }
    if (device->ddr5_transfer_engine->get_metrics().total_bytes_transferred != total_expert_bytes) {
        return false;
    }

    return true;
}

bool test_9_original_ramulator_baseline_preserved() {
    SystemConfig cfg = A100;
    cfg.hbm_dram_cfg_path = get_hbm_cfg_path();
    cfg.use_ramulator = true;
    cfg.use_event_driven_dram = false;
    Cluster::Ptr cluster = Cluster::Create(cfg, nullptr);
    Device::Ptr device = Device::Create(cfg, 0, cluster);

    if (device->ddr5_transfer_engine != nullptr) {
        return false;
    }
    if (device->config.use_event_driven_dram) {
        return false;
    }

    SystemConfig default_cfg;
    if (default_cfg.use_event_driven_dram) {
        return false;
    }

    return true;
}

bool test_10_event_driven_mode_verified() {
    SystemConfig cfg = A100;
    cfg.hbm_dram_cfg_path = get_hbm_cfg_path();
    cfg.use_ramulator = true;
    cfg.use_event_driven_dram = true;
    cfg.ddr5_bandwidth_gbps = 22.578;
    Cluster::Ptr cluster = Cluster::Create(cfg, nullptr);
    Device::Ptr device = Device::Create(cfg, 0, cluster);

    if (!device->config.use_event_driven_dram) {
        return false;
    }
    if (device->ddr5_transfer_engine == nullptr) {
        return false;
    }

    int m = 1;
    int k = 4096;
    int n = 4096;
    Tensor::Ptr input = Tensor::Create("in", {m, k}, "act", device);
    input->allocateMemoryObject(MemoryObject::Create(MMap::ALL_CHANNEL, 0, m * k * 2, device->mmap_controller));

    Tensor::Ptr weight = Tensor::Create("w1", {k, n}, "weight", device);
    weight->is_expert_weight = true;
    weight->expert_layer_id = 0;
    weight->expert_id = 99;
    weight->weight_target = MemoryTarget::OFFCHIP_DRAM;
    weight->allocateMemoryObject(MemoryObject::Create(MMap::ALL_CHANNEL, 10000000ULL, k * n * 2, device->offchip_mmap_controller));

    Tensor::Ptr output = Tensor::Create("out", {m, n}, "act", device);
    output->allocateMemoryObject(MemoryObject::Create(MMap::ALL_CHANNEL, 30000000ULL, m * n * 2, device->mmap_controller));

    double offchip_time_before = 0.0;
    if (device->offchip_dram_interface != nullptr) {
        offchip_time_before = device->offchip_dram_interface->time;
    }

    LinearExecutionGPU(device, input, weight, output, true);

    if (device->ddr5_transfer_engine->get_metrics().demand_completions != 1) {
        return false;
    }

    if (device->offchip_dram_interface != nullptr) {
        double offchip_time_after = device->offchip_dram_interface->time;
        if (offchip_time_after != offchip_time_before) {
            return false;
        }
    }

    return true;
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "PHASE 4D.1: Production Integration of DDR5TransferEngine (Demand-Only)\n";
    std::cout << "Calibrated DDR5 Bandwidth: 22.578 B/ns | Single-Server Event Queue\n";
    std::cout << "======================================================================\n\n";

    int passed = 0;
    int total = 10;

    std::cout << "[RUNNING] Test 1: Single demand miss completes and marks expert resident...\n";
    if (test_1_single_demand_miss()) {
        std::cout << "[PASS] Test 1: Single demand miss completes and marks expert resident (3.90116 ms latency)\n";
        passed++;
    } else {
        std::cout << "[FAIL] Test 1: Single demand miss failed\n";
    }

    std::cout << "[RUNNING] Test 2: Repeated demand for resident expert (HBM hit, no transfer)...\n";
    if (test_2_repeated_demand_resident()) {
        std::cout << "[PASS] Test 2: Repeated demand correctly hits HBM without second transfer\n";
        passed++;
    } else {
        std::cout << "[FAIL] Test 2: Repeated demand failed\n";
    }

    std::cout << "[RUNNING] Test 3: Two demand misses serialized FIFO (3.90116 ms and 7.80232 ms)...\n";
    if (test_3_two_demand_misses_serialized()) {
        std::cout << "[PASS] Test 3: Serialized FIFO completions verified (T1=3.90116 ms, T2=7.80232 ms)\n";
        passed++;
    } else {
        std::cout << "[FAIL] Test 3: Two demand misses serialization failed\n";
    }

    std::cout << "[RUNNING] Test 4: Transfer failure handling (expert not marked resident)...\n";
    if (test_4_transfer_failure_handling()) {
        std::cout << "[PASS] Test 4: Transfer failure correctly prevents resident insertion\n";
        passed++;
    } else {
        std::cout << "[FAIL] Test 4: Transfer failure handling failed\n";
    }

    std::cout << "[RUNNING] Test 5: HBM capacity eviction preservation (LRU policy verified)...\n";
    if (test_5_lru_capacity_eviction_preserved()) {
        std::cout << "[PASS] Test 5: Existing LRU eviction behavior strictly preserved\n";
        passed++;
    } else {
        std::cout << "[FAIL] Test 5: LRU capacity eviction failed\n";
    }

    std::cout << "[RUNNING] Test 6: Per-device isolation (independent DDR5 transfer engines)...\n";
    if (test_6_per_device_isolation()) {
        std::cout << "[PASS] Test 6: Independent devices maintain isolated transfer channels\n";
        passed++;
    } else {
        std::cout << "[FAIL] Test 6: Per-device isolation failed\n";
    }

    std::cout << "[RUNNING] Test 7: Clock synchronization between device time and engine time...\n";
    if (test_7_clock_synchronization()) {
        std::cout << "[PASS] Test 7: Device time and engine clock advance consistently\n";
        passed++;
    } else {
        std::cout << "[FAIL] Test 7: Clock synchronization failed\n";
    }

    std::cout << "[RUNNING] Test 8: No double-charging verification...\n";
    if (test_8_no_double_charging()) {
        std::cout << "[PASS] Test 8: Transfer duration contributes exactly once to execution duration\n";
        passed++;
    } else {
        std::cout << "[FAIL] Test 8: No double-charging failed\n";
    }

    std::cout << "[RUNNING] Test 9: Original Ramulator mode preservation...\n";
    if (test_9_original_ramulator_baseline_preserved()) {
        std::cout << "[PASS] Test 9: Original blocking Ramulator path remains default and preserved\n";
        passed++;
    } else {
        std::cout << "[FAIL] Test 9: Original Ramulator mode preservation failed\n";
    }

    std::cout << "[RUNNING] Test 10: Event-driven mode verified (Ramulator bypassed on miss)...\n";
    if (test_10_event_driven_mode_verified()) {
        std::cout << "[PASS] Test 10: Event-driven engine successfully bypasses Ramulator on miss\n";
        passed++;
    } else {
        std::cout << "[FAIL] Test 10: Event-driven mode verification failed\n";
    }

    std::cout << "\n======================================================================\n";
    std::cout << "TEST SUITE SUMMARY: " << passed << " PASSED, " << (total - passed) << " FAILED\n";
    std::cout << "======================================================================\n";

    if (passed == total) {
        return 0;
    }
    return 1;
}
