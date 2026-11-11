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

static Device::Ptr create_test_device(EvictionPolicy policy = EvictionPolicy::PABI,
                                      long long cache_cap = 80000000000LL,
                                      long long spec_cap = 16000000000LL) {
    SystemConfig cfg = A100;
    cfg.hbm_dram_cfg_path = get_hbm_cfg_path();
    cfg.use_ramulator = true;
    cfg.use_event_driven_dram = true;
    cfg.ddr5_bandwidth_gbps = 22.578;
    cfg.prefetch_config.enabled = true;
    cfg.prefetch_config.eviction_policy = policy;
    cfg.prefetch_config.pabi_quarantine_enabled = true;
    cfg.prefetch_config.pabi_speculative_buffer_capacity_bytes = spec_cap;

    Cluster::Ptr cluster = Cluster::Create(cfg, nullptr);
    Device::Ptr dev = Device::Create(cfg, 0, cluster);
    dev->cache_capacity_bytes = cache_cap;
    dev->speculative_buffer_capacity_bytes = spec_cap;
    dev->eviction_policy = policy;
    dev->reset_expert_cache();
    return dev;
}

static const long long EXPERT_SIZE = 88080384LL; // Calibrated DeepSeek-V3 expert

// Test 1: Standard LRU insertion remains unchanged
bool test_1_standard_lru_insertion_unchanged() {
    std::cout << "[RUNNING] Test 1: Standard LRU insertion remains unchanged...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::LRU, 2 * EXPERT_SIZE, 0);

    device->mark_expert_demand(1, 1, EXPERT_SIZE);
    device->mark_expert_demand(1, 2, EXPERT_SIZE);

    TEST_ASSERT(device->resident_bytes == 2 * EXPERT_SIZE, "Resident bytes must be 2 * EXPERT_SIZE");
    TEST_ASSERT(device->lru_expert_list.size() == 2, "LRU list must have 2 entries");
    TEST_ASSERT(device->resident_experts.size() == 2, "Resident experts must have 2 entries");
    TEST_ASSERT(device->is_expert_resident(1, 1), "Expert (1, 1) must be resident");

    // Next insert evicts LRU (1, 2 became LRU because 1, 1 was accessed above)
    device->mark_expert_demand(1, 3, EXPERT_SIZE);
    TEST_ASSERT(!device->is_expert_resident(1, 2), "Expert (1, 2) must be evicted");
    TEST_ASSERT(device->is_expert_resident(1, 1), "Expert (1, 1) must remain resident");
    TEST_ASSERT(device->is_expert_resident(1, 3), "Expert (1, 3) must be resident");
    TEST_ASSERT(device->verify_pabi_invariants(), "PABI invariants must hold");

    std::cout << "[PASS] Test 1: Standard LRU insertion remains unchanged verified\n";
    return true;
}

// Test 2: PABI prefetch insertion enters speculative list
bool test_2_pabi_prefetch_insertion_enters_speculative_list() {
    std::cout << "[RUNNING] Test 2: PABI prefetch insertion enters speculative list...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE);

    device->mark_expert_speculative(1, 10, EXPERT_SIZE);

    TEST_ASSERT(device->speculative_lru_list.size() == 1, "Speculative list size must be 1");
    TEST_ASSERT(device->demand_lru_list.empty(), "Demand list must be empty");
    TEST_ASSERT(device->resident_speculative_bytes == EXPERT_SIZE, "Speculative bytes must equal EXPERT_SIZE");
    TEST_ASSERT(device->resident_demand_bytes == 0, "Demand bytes must be 0");
    TEST_ASSERT(device->resident_bytes == EXPERT_SIZE, "Total resident bytes must equal EXPERT_SIZE");
    TEST_ASSERT(device->pabi_metrics.speculative_insertions == 1, "Speculative insertions metric must be 1");

    auto it = device->resident_experts.find({1, 10});
    TEST_ASSERT(it != device->resident_experts.end(), "Expert must be in resident_experts map");
    TEST_ASSERT(it->second.classification == ExpertClassification::SPECULATIVE, "Classification must be SPECULATIVE");
    TEST_ASSERT(device->verify_pabi_invariants(), "PABI invariants must hold");

    std::cout << "[PASS] Test 2: PABI prefetch insertion enters speculative list verified\n";
    return true;
}

// Test 3: Demand-verified insertion enters demand list
bool test_3_demand_verified_insertion_enters_demand_list() {
    std::cout << "[RUNNING] Test 3: Demand-verified insertion enters demand list...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE);

    device->mark_expert_demand(1, 11, EXPERT_SIZE);

    TEST_ASSERT(device->demand_lru_list.size() == 1, "Demand list size must be 1");
    TEST_ASSERT(device->speculative_lru_list.empty(), "Speculative list must be empty");
    TEST_ASSERT(device->resident_demand_bytes == EXPERT_SIZE, "Demand bytes must equal EXPERT_SIZE");
    TEST_ASSERT(device->resident_speculative_bytes == 0, "Speculative bytes must be 0");
    TEST_ASSERT(device->resident_bytes == EXPERT_SIZE, "Total resident bytes must equal EXPERT_SIZE");
    TEST_ASSERT(device->pabi_metrics.demand_insertions == 1, "Demand insertions metric must be 1");

    auto it = device->resident_experts.find({1, 11});
    TEST_ASSERT(it != device->resident_experts.end(), "Expert must be in resident_experts map");
    TEST_ASSERT(it->second.classification == ExpertClassification::DEMAND_VERIFIED, "Classification must be DEMAND_VERIFIED");
    TEST_ASSERT(device->verify_pabi_invariants(), "PABI invariants must hold");

    std::cout << "[PASS] Test 3: Demand-verified insertion enters demand list verified\n";
    return true;
}

// Test 4: Speculative hit promotes entry to demand list
bool test_4_speculative_hit_promotes_entry_to_demand_list() {
    std::cout << "[RUNNING] Test 4: Speculative hit promotes entry to demand list...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE);

    device->mark_expert_speculative(1, 12, EXPERT_SIZE);
    TEST_ASSERT(device->speculative_lru_list.size() == 1, "Must have 1 speculative entry");

    bool is_hit = device->is_expert_resident(1, 12);
    TEST_ASSERT(is_hit, "Query on speculative resident expert must return true (HIT)");

    TEST_ASSERT(device->speculative_lru_list.empty(), "Speculative list must now be empty");
    TEST_ASSERT(device->demand_lru_list.size() == 1, "Demand list must now contain promoted entry");
    TEST_ASSERT(device->resident_speculative_bytes == 0, "Speculative bytes must be 0");
    TEST_ASSERT(device->resident_demand_bytes == EXPERT_SIZE, "Demand bytes must be EXPERT_SIZE");
    TEST_ASSERT(device->resident_bytes == EXPERT_SIZE, "Total resident bytes must remain EXPERT_SIZE");

    auto it = device->resident_experts.find({1, 12});
    TEST_ASSERT(it != device->resident_experts.end(), "Expert must remain in resident_experts");
    TEST_ASSERT(it->second.classification == ExpertClassification::DEMAND_VERIFIED, "Classification must be DEMAND_VERIFIED");
    TEST_ASSERT(device->pabi_metrics.speculative_promotions == 1, "Promotion metric must be 1");
    TEST_ASSERT(device->pabi_metrics.useful_prefetch_hits == 1, "Useful hit metric must be 1");
    TEST_ASSERT(device->verify_pabi_invariants(), "PABI invariants must hold");

    std::cout << "[PASS] Test 4: Speculative hit promotes entry to demand list verified\n";
    return true;
}

// Test 5: Promotion preserves total resident bytes
bool test_5_promotion_preserves_total_resident_bytes() {
    std::cout << "[RUNNING] Test 5: Promotion preserves total resident bytes...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE);

    device->mark_expert_demand(1, 20, EXPERT_SIZE);
    device->mark_expert_speculative(1, 21, EXPERT_SIZE);

    long long bytes_before = device->resident_bytes;
    TEST_ASSERT(bytes_before == 2 * EXPERT_SIZE, "Initial bytes must be 2 * EXPERT_SIZE");

    device->promote_speculative_to_demand(1, 21);

    long long bytes_after = device->resident_bytes;
    TEST_ASSERT(bytes_before == bytes_after, "Total resident bytes must be exactly preserved across promotion");
    TEST_ASSERT(device->resident_demand_bytes == 2 * EXPERT_SIZE, "Demand bytes must now be 2 * EXPERT_SIZE");
    TEST_ASSERT(device->resident_speculative_bytes == 0, "Speculative bytes must now be 0");
    TEST_ASSERT(device->verify_pabi_invariants(), "PABI invariants must hold");

    std::cout << "[PASS] Test 5: Promotion preserves total resident bytes verified\n";
    return true;
}

// Test 6: Speculative buffer overflow evicts speculative LRU first
bool test_6_speculative_buffer_overflow_evicts_speculative_lru_first() {
    std::cout << "[RUNNING] Test 6: Speculative buffer overflow evicts speculative LRU first...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 10 * EXPERT_SIZE, 2 * EXPERT_SIZE);

    device->mark_expert_speculative(1, 1, EXPERT_SIZE); // LRU in speculative buffer
    device->mark_expert_speculative(1, 2, EXPERT_SIZE); // MRU in speculative buffer
    TEST_ASSERT(device->resident_speculative_bytes == 2 * EXPERT_SIZE, "Buffer must be full");

    // Inserting 3rd speculative expert must evict (1, 1)
    device->mark_expert_speculative(1, 3, EXPERT_SIZE);

    TEST_ASSERT(device->resident_speculative_bytes == 2 * EXPERT_SIZE, "Speculative bytes must stay bounded by 2 * EXPERT_SIZE");
    TEST_ASSERT(!device->is_expert_resident(1, 1), "LRU speculative expert (1, 1) must be evicted");
    TEST_ASSERT(device->is_expert_resident(1, 2), "Expert (1, 2) must remain resident");
    TEST_ASSERT(device->is_expert_resident(1, 3), "Expert (1, 3) must remain resident");
    TEST_ASSERT(device->pabi_metrics.speculative_evictions == 1, "Speculative evictions metric must be 1");
    TEST_ASSERT(device->verify_pabi_invariants(), "PABI invariants must hold");

    std::cout << "[PASS] Test 6: Speculative buffer overflow evicts speculative LRU first verified\n";
    return true;
}

// Test 7: Speculative overflow never evicts demand-verified entries
bool test_7_speculative_overflow_never_evicts_demand_verified_entries() {
    std::cout << "[RUNNING] Test 7: Speculative overflow never evicts demand-verified entries...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 10 * EXPERT_SIZE, 2 * EXPERT_SIZE);

    device->mark_expert_demand(0, 100, EXPERT_SIZE);
    device->mark_expert_demand(0, 101, EXPERT_SIZE);

    for (int i = 0; i < 5; i = i + 1) {
        device->mark_expert_speculative(1, i, EXPERT_SIZE);
    }

    TEST_ASSERT(device->is_expert_resident(0, 100), "Demand expert (0, 100) must NEVER be evicted by speculative overflow");
    TEST_ASSERT(device->is_expert_resident(0, 101), "Demand expert (0, 101) must NEVER be evicted by speculative overflow");
    TEST_ASSERT(device->pabi_metrics.demand_evictions == 0, "Demand evictions must be 0");
    TEST_ASSERT(device->pabi_metrics.speculative_evictions == 3, "Speculative evictions must be 3");
    TEST_ASSERT(device->verify_pabi_invariants(), "PABI invariants must hold");

    std::cout << "[PASS] Test 7: Speculative overflow never evicts demand-verified entries verified\n";
    return true;
}

// Test 8: Total HBM overflow evicts speculative entries first
bool test_8_total_hbm_overflow_evicts_speculative_entries_first() {
    std::cout << "[RUNNING] Test 8: Total HBM overflow evicts speculative entries first...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 3 * EXPERT_SIZE, 2 * EXPERT_SIZE);

    device->mark_expert_demand(0, 1, EXPERT_SIZE);
    device->mark_expert_speculative(1, 10, EXPERT_SIZE); // S1 (LRU spec)
    device->mark_expert_speculative(1, 11, EXPERT_SIZE); // S2 (MRU spec)

    TEST_ASSERT(device->resident_bytes == 3 * EXPERT_SIZE, "HBM must be 100% full");

    // Demand arrives for (0, 2): HBM is full -> S1 must be evicted first, NOT D1!
    device->mark_expert_demand(0, 2, EXPERT_SIZE);

    TEST_ASSERT(device->is_expert_resident(0, 1), "Demand expert (0, 1) must NOT be evicted");
    TEST_ASSERT(device->is_expert_resident(0, 2), "New demand expert (0, 2) must be resident");
    TEST_ASSERT(!device->is_expert_resident(1, 10), "Speculative expert S1 must be evicted to yield room");
    TEST_ASSERT(device->is_expert_resident(1, 11), "Speculative expert S2 must remain resident");
    TEST_ASSERT(device->pabi_metrics.speculative_evictions == 1, "Speculative eviction must occur");
    TEST_ASSERT(device->pabi_metrics.demand_evictions == 0, "Demand eviction must NOT occur");
    TEST_ASSERT(device->verify_pabi_invariants(), "PABI invariants must hold");

    std::cout << "[PASS] Test 8: Total HBM overflow evicts speculative entries first verified\n";
    return true;
}

// Test 9: Demand entries are evicted only when necessary
bool test_9_demand_entries_evicted_only_when_necessary() {
    std::cout << "[RUNNING] Test 9: Demand entries are evicted only when necessary...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 2 * EXPERT_SIZE, 1 * EXPERT_SIZE);

    device->mark_expert_demand(0, 1, EXPERT_SIZE); // LRU demand
    device->mark_expert_demand(0, 2, EXPERT_SIZE); // MRU demand
    TEST_ASSERT(device->resident_speculative_bytes == 0, "No speculative entries exist");

    // Demand arrives for (0, 3) -> since no speculative entries exist, evict demand LRU (0, 1)
    device->mark_expert_demand(0, 3, EXPERT_SIZE);

    TEST_ASSERT(!device->is_expert_resident(0, 1), "Demand LRU (0, 1) must be evicted");
    TEST_ASSERT(device->is_expert_resident(0, 2), "Demand (0, 2) must remain resident");
    TEST_ASSERT(device->is_expert_resident(0, 3), "Demand (0, 3) must be resident");
    TEST_ASSERT(device->pabi_metrics.demand_evictions == 1, "Demand eviction count must be 1");
    TEST_ASSERT(device->verify_pabi_invariants(), "PABI invariants must hold");

    std::cout << "[PASS] Test 9: Demand entries are evicted only when necessary verified\n";
    return true;
}

// Test 10: Speculative entry can be evicted after a false-positive prediction
bool test_10_speculative_entry_evicted_after_false_positive() {
    std::cout << "[RUNNING] Test 10: Speculative entry can be evicted after a false-positive prediction...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 1 * EXPERT_SIZE);

    // False positive prefetch (never demanded)
    device->mark_expert_speculative(2, 99, EXPERT_SIZE);
    TEST_ASSERT(device->is_expert_resident(2, 99), "Pre-check resident");
    // (Note: calling is_expert_resident promoted it, so let's test pure un-demanded eviction)
    device->reset_expert_cache();

    device->mark_expert_speculative(2, 99, EXPERT_SIZE);
    // New prefetch arrives without ever demanding 99
    device->mark_expert_speculative(2, 100, EXPERT_SIZE);

    // (2, 99) was evicted without ever entering demand list!
    auto it99 = device->resident_experts.find({2, 99});
    TEST_ASSERT(it99 == device->resident_experts.end(), "False positive 99 must be evicted");
    TEST_ASSERT(device->demand_lru_list.empty(), "Demand list was never polluted");
    TEST_ASSERT(device->verify_pabi_invariants(), "PABI invariants must hold");

    std::cout << "[PASS] Test 10: Speculative entry can be evicted after a false-positive prediction verified\n";
    return true;
}

// Test 11: Promoted entry follows demand LRU thereafter
bool test_11_promoted_entry_follows_demand_lru_thereafter() {
    std::cout << "[RUNNING] Test 11: Promoted entry follows demand LRU thereafter...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 2 * EXPERT_SIZE, 1 * EXPERT_SIZE);

    device->mark_expert_speculative(1, 5, EXPERT_SIZE);
    device->is_expert_resident(1, 5); // Demanded! Promoted to demand list

    // Now insert demand expert (1, 6) -> (1, 5) is LRU demand, (1, 6) is MRU demand
    device->mark_expert_demand(1, 6, EXPERT_SIZE);

    // Next demand for (1, 7) must evict (1, 5) because it is now demand LRU
    device->mark_expert_demand(1, 7, EXPERT_SIZE);

    TEST_ASSERT(!device->is_expert_resident(1, 5), "Promoted expert (1, 5) must be evicted according to demand LRU");
    TEST_ASSERT(device->is_expert_resident(1, 6), "Expert (1, 6) must remain resident");
    TEST_ASSERT(device->is_expert_resident(1, 7), "Expert (1, 7) must be resident");
    TEST_ASSERT(device->verify_pabi_invariants(), "PABI invariants must hold");

    std::cout << "[PASS] Test 11: Promoted entry follows demand LRU thereafter verified\n";
    return true;
}

// Test 12: Cancelled prefetch never enters HBM
bool test_12_cancelled_prefetch_never_enters_hbm() {
    std::cout << "[RUNNING] Test 12: Cancelled prefetch never enters HBM...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE);

    uint64_t tid = device->submit_prefetch_transfer(3, 1, 0.0);
    TEST_ASSERT(tid != 0, "Submission failed");

    device->ddr5_transfer_engine->cancel_transfer(tid, 500000.0);
    device->set_time(5000000.0);
    device->process_completed_prefetches(5000000.0);

    TEST_ASSERT(!device->is_expert_resident(3, 1), "Cancelled transfer must NEVER enter HBM");
    TEST_ASSERT(device->speculative_lru_list.empty(), "Speculative list must remain empty");
    TEST_ASSERT(device->demand_lru_list.empty(), "Demand list must remain empty");
    TEST_ASSERT(device->resident_bytes == 0, "Resident bytes must be 0");
    TEST_ASSERT(device->verify_pabi_invariants(), "PABI invariants must hold");

    std::cout << "[PASS] Test 12: Cancelled prefetch never enters HBM verified\n";
    return true;
}

// Test 13: Demand-promoted in-flight transfer enters demand list on completion
bool test_13_demand_promoted_inflight_transfer_enters_demand_list() {
    std::cout << "[RUNNING] Test 13: Demand-promoted in-flight transfer enters demand list...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE);

    uint64_t tid = device->submit_prefetch_transfer(3, 2, 0.0);
    TEST_ASSERT(tid != 0, "Submission failed");

    // Demand arrives at 1 ms while in flight
    device->submit_demand_transfer(3, 2, 1000000.0);

    // Complete transfer at 5 ms
    device->set_time(5000000.0);
    device->process_completed_prefetches(5000000.0);

    auto it = device->resident_experts.find({3, 2});
    TEST_ASSERT(it != device->resident_experts.end(), "Expert must be resident");
    TEST_ASSERT(it->second.classification == ExpertClassification::DEMAND_VERIFIED,
                "Promoted transfer must enter DEMAND_VERIFIED, NOT SPECULATIVE");
    TEST_ASSERT(device->demand_lru_list.size() == 1, "Demand list must contain expert");
    TEST_ASSERT(device->speculative_lru_list.empty(), "Speculative list must be empty");
    TEST_ASSERT(device->verify_pabi_invariants(), "PABI invariants must hold");

    std::cout << "[PASS] Test 13: Demand-promoted in-flight transfer enters demand list verified\n";
    return true;
}

// Test 14: Duplicate insertion does not double-count bytes
bool test_14_duplicate_insertion_does_not_double_count_bytes() {
    std::cout << "[RUNNING] Test 14: Duplicate insertion does not double-count bytes...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE);

    device->mark_expert_demand(4, 1, EXPERT_SIZE);
    device->mark_expert_demand(4, 1, EXPERT_SIZE);

    TEST_ASSERT(device->resident_bytes == EXPERT_SIZE, "Resident bytes must NOT double-count");
    TEST_ASSERT(device->demand_lru_list.size() == 1, "Demand list size must remain 1");
    TEST_ASSERT(device->verify_pabi_invariants(), "PABI invariants must hold");

    std::cout << "[PASS] Test 14: Duplicate insertion does not double-count bytes verified\n";
    return true;
}

// Test 15: Cache reset clears PABI state
bool test_15_cache_reset_clears_pabi_state() {
    std::cout << "[RUNNING] Test 15: Cache reset clears PABI state...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE);

    device->mark_expert_demand(5, 1, EXPERT_SIZE);
    device->mark_expert_speculative(5, 2, EXPERT_SIZE);

    device->reset_expert_cache();

    TEST_ASSERT(device->resident_bytes == 0, "Resident bytes must be 0");
    TEST_ASSERT(device->resident_demand_bytes == 0, "Demand bytes must be 0");
    TEST_ASSERT(device->resident_speculative_bytes == 0, "Speculative bytes must be 0");
    TEST_ASSERT(device->demand_lru_list.empty(), "Demand list must be empty");
    TEST_ASSERT(device->speculative_lru_list.empty(), "Speculative list must be empty");
    TEST_ASSERT(device->resident_experts.empty(), "Resident experts map must be empty");
    TEST_ASSERT(device->verify_pabi_invariants(), "PABI invariants must hold");

    std::cout << "[PASS] Test 15: Cache reset clears PABI state verified\n";
    return true;
}

// Test 16: Zero-sized speculative buffer safely rejects speculative insertion
bool test_16_zero_sized_speculative_buffer_rejects_speculative_insertion() {
    std::cout << "[RUNNING] Test 16: Zero-sized speculative buffer safely rejects speculative insertion...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 0);

    device->mark_expert_speculative(6, 1, EXPERT_SIZE);

    TEST_ASSERT(!device->is_expert_resident(6, 1), "Speculative insertion must be rejected when capacity is 0");
    TEST_ASSERT(device->pabi_metrics.speculative_insertions_rejected == 1, "Rejected metric must be 1");
    TEST_ASSERT(device->resident_bytes == 0, "Resident bytes must be 0");
    TEST_ASSERT(device->verify_pabi_invariants(), "PABI invariants must hold");

    std::cout << "[PASS] Test 16: Zero-sized speculative buffer safely rejects speculative insertion verified\n";
    return true;
}

// Test 17: Oversized expert safely rejects speculative insertion
bool test_17_oversized_expert_rejects_speculative_insertion() {
    std::cout << "[RUNNING] Test 17: Oversized expert safely rejects speculative insertion...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, EXPERT_SIZE / 2);

    device->mark_expert_speculative(7, 1, EXPERT_SIZE);

    TEST_ASSERT(!device->is_expert_resident(7, 1), "Oversized expert must be rejected from speculative buffer");
    TEST_ASSERT(device->pabi_metrics.speculative_insertions_rejected == 1, "Rejected metric must be 1");
    TEST_ASSERT(device->resident_bytes == 0, "Resident bytes must be 0");
    TEST_ASSERT(device->verify_pabi_invariants(), "PABI invariants must hold");

    std::cout << "[PASS] Test 17: Oversized expert safely rejects speculative insertion verified\n";
    return true;
}

// Test 18: Invalid configuration is handled safely
bool test_18_invalid_configuration_handled_safely() {
    std::cout << "[RUNNING] Test 18: Invalid configuration is handled safely...\n";

    TEST_ASSERT(parse_eviction_policy("lru") == EvictionPolicy::LRU, "Parse lru");
    TEST_ASSERT(parse_eviction_policy("pabi") == EvictionPolicy::PABI, "Parse pabi");

    bool caught_invalid_policy = false;
    try {
        parse_eviction_policy("non_existent_policy");
    } catch (const std::invalid_argument& e) {
        caught_invalid_policy = true;
    }
    TEST_ASSERT(caught_invalid_policy, "Invalid eviction policy string must throw invalid_argument");

    PrefetchConfig cfg;
    cfg.enabled = true;
    cfg.mode = PrefetchMode::PREDICTION_ONLY;
    cfg.pabi_speculative_fraction = 1.5; // Invalid fraction > 1.0

    bool caught_invalid_fraction = false;
    try {
        cfg.validate();
    } catch (const std::invalid_argument& e) {
        caught_invalid_fraction = true;
    }
    TEST_ASSERT(caught_invalid_fraction, "Fraction > 1.0 must throw invalid_argument");

    std::cout << "[PASS] Test 18: Invalid configuration is handled safely verified\n";
    return true;
}

// Test 19: LRU and PABI produce expected different eviction orders (Controlled Comparison)
bool test_19_lru_vs_pabi_controlled_comparison() {
    std::cout << "[RUNNING] Test 19: LRU vs PABI controlled comparison under mispredictions...\n";

    // Setup: Cache holds 4 experts
    long long cap = 4 * EXPERT_SIZE;
    long long spec_cap = 2 * EXPERT_SIZE;

    // --- Experiment A: Standard LRU ---
    Device::Ptr dev_lru = create_test_device(EvictionPolicy::LRU, cap, 0);

    // Initial demand working set: D1, D2, D3, D4
    dev_lru->mark_expert_demand(0, 1, EXPERT_SIZE);
    dev_lru->mark_expert_demand(0, 2, EXPERT_SIZE);
    dev_lru->mark_expert_demand(0, 3, EXPERT_SIZE);
    dev_lru->mark_expert_demand(0, 4, EXPERT_SIZE);

    // 4 false-positive speculative prefetches arrive: F1, F2, F3, F4
    dev_lru->mark_expert_demand(1, 101, EXPERT_SIZE);
    dev_lru->mark_expert_demand(1, 102, EXPERT_SIZE);
    dev_lru->mark_expert_demand(1, 103, EXPERT_SIZE);
    dev_lru->mark_expert_demand(1, 104, EXPERT_SIZE);

    // Under LRU: The 4 false-positive prefetches COMPLETELY EVICTED the demand working set!
    int lru_hits = 0;
    if (dev_lru->is_expert_resident(0, 1)) lru_hits = lru_hits + 1;
    if (dev_lru->is_expert_resident(0, 2)) lru_hits = lru_hits + 1;
    if (dev_lru->is_expert_resident(0, 3)) lru_hits = lru_hits + 1;
    if (dev_lru->is_expert_resident(0, 4)) lru_hits = lru_hits + 1;

    TEST_ASSERT(lru_hits == 0, "Under LRU, all 4 demand-verified experts must be evicted (0 hits / 100% pollution)");

    // --- Experiment B: PABI Quarantine ---
    Device::Ptr dev_pabi = create_test_device(EvictionPolicy::PABI, cap, spec_cap);

    // Initial demand working set: D1, D2, D3, D4
    dev_pabi->mark_expert_demand(0, 1, EXPERT_SIZE);
    dev_pabi->mark_expert_demand(0, 2, EXPERT_SIZE);
    dev_pabi->mark_expert_demand(0, 3, EXPERT_SIZE);
    dev_pabi->mark_expert_demand(0, 4, EXPERT_SIZE);

    // 4 false-positive speculative prefetches arrive: F1, F2, F3, F4
    dev_pabi->mark_expert_speculative(1, 101, EXPERT_SIZE);
    dev_pabi->mark_expert_speculative(1, 102, EXPERT_SIZE);
    dev_pabi->mark_expert_speculative(1, 103, EXPERT_SIZE);
    dev_pabi->mark_expert_speculative(1, 104, EXPERT_SIZE);

    // Under PABI: The speculative prefetches could NOT evict demand entries!
    // Working set D1, D2, D3, D4 remained 100% resident!
    int pabi_hits = 0;
    if (dev_pabi->is_expert_resident(0, 1)) pabi_hits = pabi_hits + 1;
    if (dev_pabi->is_expert_resident(0, 2)) pabi_hits = pabi_hits + 1;
    if (dev_pabi->is_expert_resident(0, 3)) pabi_hits = pabi_hits + 1;
    if (dev_pabi->is_expert_resident(0, 4)) pabi_hits = pabi_hits + 1;

    TEST_ASSERT(pabi_hits == 4, "Under PABI, all 4 demand-verified experts remained resident (4 hits / 100% hit rate)");
    TEST_ASSERT(dev_pabi->verify_pabi_invariants(), "PABI invariants must hold");

    std::cout << "[PASS] Test 19: LRU vs PABI controlled comparison verified (LRU Hits: " 
              << lru_hits << "/4, PABI Hits: " << pabi_hits << "/4)\n";
    return true;
}

// Test 20: Regression with Phase 4D.1 and 4D.2 tests
bool test_20_regression_phase4d1_and_phase4d2() {
    std::cout << "[RUNNING] Test 20: Regression with Phase 4D.1 and 4D.2...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 80000000000LL, 16000000000LL);

    uint64_t tid = device->submit_prefetch_transfer(1, 30, 0.0);
    TEST_ASSERT(tid != 0, "Prefetch submission failed");

    // Advance to 5 ms
    device->set_time(5000000.0);
    device->process_completed_prefetches(5000000.0);

    TEST_ASSERT(device->is_expert_resident(1, 30), "Prefetch must be completed and promoted on query");
    TEST_ASSERT(device->prefetch_metrics.prefetch_useful_hits == 1, "Useful hit recorded");
    TEST_ASSERT(device->pabi_metrics.speculative_promotions == 1, "PABI promotion recorded");
    TEST_ASSERT(device->verify_pabi_invariants(), "PABI invariants must hold");

    std::cout << "[PASS] Test 20: Regression with Phase 4D.1 and 4D.2 verified\n";
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "======================================================================\n";
    std::cout << "PHASE 4D.3: PABI Speculative HBM Quarantine & Demand Promotion Tests\n";
    std::cout << "Dual Recency Lists | Logical Speculative Quarantine | 20 Test Cases\n";
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

    run_test(test_1_standard_lru_insertion_unchanged, "Test 1");
    run_test(test_2_pabi_prefetch_insertion_enters_speculative_list, "Test 2");
    run_test(test_3_demand_verified_insertion_enters_demand_list, "Test 3");
    run_test(test_4_speculative_hit_promotes_entry_to_demand_list, "Test 4");
    run_test(test_5_promotion_preserves_total_resident_bytes, "Test 5");
    run_test(test_6_speculative_buffer_overflow_evicts_speculative_lru_first, "Test 6");
    run_test(test_7_speculative_overflow_never_evicts_demand_verified_entries, "Test 7");
    run_test(test_8_total_hbm_overflow_evicts_speculative_entries_first, "Test 8");
    run_test(test_9_demand_entries_evicted_only_when_necessary, "Test 9");
    run_test(test_10_speculative_entry_evicted_after_false_positive, "Test 10");
    run_test(test_11_promoted_entry_follows_demand_lru_thereafter, "Test 11");
    run_test(test_12_cancelled_prefetch_never_enters_hbm, "Test 12");
    run_test(test_13_demand_promoted_inflight_transfer_enters_demand_list, "Test 13");
    run_test(test_14_duplicate_insertion_does_not_double_count_bytes, "Test 14");
    run_test(test_15_cache_reset_clears_pabi_state, "Test 15");
    run_test(test_16_zero_sized_speculative_buffer_rejects_speculative_insertion, "Test 16");
    run_test(test_17_oversized_expert_rejects_speculative_insertion, "Test 17");
    run_test(test_18_invalid_configuration_handled_safely, "Test 18");
    run_test(test_19_lru_vs_pabi_controlled_comparison, "Test 19");
    run_test(test_20_regression_phase4d1_and_phase4d2, "Test 20");

    std::cout << "\n======================================================================\n";
    std::cout << "TEST SUITE SUMMARY: " << passed << " PASSED, " << failed << " FAILED\n";
    std::cout << "======================================================================\n";

    if (failed > 0) {
        return 1;
    }
    return 0;
}

