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
                                      long long spec_cap = 16000000000LL,
                                      bool pinning = true,
                                      int max_pinned = 4) {
    SystemConfig cfg = A100;
    cfg.hbm_dram_cfg_path = get_hbm_cfg_path();
    cfg.use_ramulator = true;
    cfg.use_event_driven_dram = true;
    cfg.ddr5_bandwidth_gbps = 22.578;
    cfg.prefetch_config.enabled = true;
    cfg.prefetch_config.eviction_policy = policy;
    cfg.prefetch_config.pabi_quarantine_enabled = true;
    cfg.prefetch_config.pabi_speculative_buffer_capacity_bytes = spec_cap;
    cfg.prefetch_config.pabi_pinning_enabled = pinning;
    cfg.prefetch_config.pabi_max_pinned_experts = max_pinned;
    cfg.prefetch_config.pabi_pin_selection_policy = "top_ranked";
    cfg.prefetch_config.pabi_pin_expiration_policy = "stage_deadline";

    Cluster::Ptr cluster = Cluster::Create(cfg, nullptr);
    Device::Ptr dev = Device::Create(cfg, 0, cluster);
    dev->cache_capacity_bytes = cache_cap;
    dev->speculative_buffer_capacity_bytes = spec_cap;
    dev->eviction_policy = policy;
    dev->pabi_quarantine_enabled = true;
    dev->pinning_enabled = pinning;
    dev->max_pinned_experts = max_pinned;
    return dev;
}

static const long long EXPERT_SIZE = DDR5TransferEngine::DEFAULT_EXPERT_BYTES;

// Test 1: Pinning disabled preserves Phase 4D.3 behavior
bool test_1_pinning_disabled_preserves_pabi_behavior() {
    std::cout << "[RUNNING] Test 1: Pinning disabled preserves Phase 4D.3 behavior...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE, false, 4);

    TEST_ASSERT(!device->is_pinning_active(), "Pinning must not be active when pinning_enabled is false");

    // Insert 2 speculative experts
    device->mark_expert_speculative(1, 1, EXPERT_SIZE);
    device->mark_expert_speculative(1, 2, EXPERT_SIZE);
    TEST_ASSERT(device->resident_speculative_bytes == 2 * EXPERT_SIZE, "Speculative bytes must be 2 * EXPERT_SIZE");
    TEST_ASSERT((device->resident_experts[{1, 1}].is_pinned == false), "Expert (1, 1) must be unpinned");
    TEST_ASSERT((device->resident_experts[{1, 2}].is_pinned == false), "Expert (1, 2) must be unpinned");

    // Third speculative insert evicts LRU (1, 1)
    device->mark_expert_speculative(1, 3, EXPERT_SIZE);
    TEST_ASSERT(!device->is_expert_resident(1, 1), "Expert (1, 1) must be evicted");
    TEST_ASSERT(device->is_expert_resident(1, 2), "Expert (1, 2) must remain");
    TEST_ASSERT(device->is_expert_resident(1, 3), "Expert (1, 3) must remain");
    TEST_ASSERT(device->verify_pabi_invariants(), "PABI invariants must hold");

    std::cout << "[PASS] Test 1: Pinning disabled preserves Phase 4D.3 behavior verified\n";
    return true;
}

// Test 2: LRU policy never activates PABI pinning
bool test_2_lru_policy_never_activates_pinning() {
    std::cout << "[RUNNING] Test 2: LRU policy never activates PABI pinning...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::LRU, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE, true, 4);

    TEST_ASSERT(!device->is_pinning_active(), "Pinning must be inactive under LRU policy");

    PrefetchCandidate c1;
    c1.target_stage_id = 1;
    c1.expert_id = 10;
    c1.prediction_rank = 1;
    c1.prediction_score = 0.95;

    auto selected = device->select_pin_candidates_for_stage(1, {c1});
    TEST_ASSERT(selected.empty(), "Pin selection must be empty under LRU policy");
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants must hold");

    std::cout << "[PASS] Test 2: LRU policy never activates PABI pinning verified\n";
    return true;
}

// Test 3: Top-K selection respects candidate rank
bool test_3_top_k_selection_respects_candidate_rank() {
    std::cout << "[RUNNING] Test 3: Top-K selection respects candidate rank...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 8 * EXPERT_SIZE, 4 * EXPERT_SIZE, true, 2);

    PrefetchCandidate c1;
    c1.target_stage_id = 2;
    c1.expert_id = 10;
    c1.prediction_rank = 3;
    c1.prediction_score = 0.50;

    PrefetchCandidate c2;
    c2.target_stage_id = 2;
    c2.expert_id = 20;
    c2.prediction_rank = 1;
    c2.prediction_score = 0.90;

    PrefetchCandidate c3;
    c3.target_stage_id = 2;
    c3.expert_id = 30;
    c3.prediction_rank = 2;
    c3.prediction_score = 0.70;

    auto selected = device->select_pin_candidates_for_stage(2, {c1, c2, c3});
    TEST_ASSERT(selected.size() == 2, "Must select exactly 2 candidates (budget=2)");
    TEST_ASSERT(selected[0].expert_id == 20, "Rank 1 (expert 20) must be selected first");
    TEST_ASSERT(selected[1].expert_id == 30, "Rank 2 (expert 30) must be selected second");

    std::cout << "[PASS] Test 3: Top-K selection respects candidate rank verified\n";
    return true;
}

// Test 4: Score tie-break is deterministic
bool test_4_score_tie_break_is_deterministic() {
    std::cout << "[RUNNING] Test 4: Score tie-break is deterministic...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 8 * EXPERT_SIZE, 4 * EXPERT_SIZE, true, 2);

    PrefetchCandidate c1;
    c1.target_stage_id = 3;
    c1.expert_id = 10;
    c1.prediction_rank = 1;
    c1.prediction_score = 0.75;

    PrefetchCandidate c2;
    c2.target_stage_id = 3;
    c2.expert_id = 20;
    c2.prediction_rank = 1;
    c2.prediction_score = 0.95;

    PrefetchCandidate c3;
    c3.target_stage_id = 3;
    c3.expert_id = 30;
    c3.prediction_rank = 1;
    c3.prediction_score = 0.60;

    auto selected = device->select_pin_candidates_for_stage(3, {c1, c2, c3});
    TEST_ASSERT(selected.size() == 2, "Must select exactly 2 candidates");
    TEST_ASSERT(selected[0].expert_id == 20, "Expert 20 with score 0.95 must be selected first");
    TEST_ASSERT(selected[1].expert_id == 10, "Expert 10 with score 0.75 must be selected second");

    std::cout << "[PASS] Test 4: Score tie-break is deterministic verified\n";
    return true;
}

// Test 5: Expert ID tie-break is deterministic
bool test_5_expert_id_tie_break_is_deterministic() {
    std::cout << "[RUNNING] Test 5: Expert ID tie-break is deterministic...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 8 * EXPERT_SIZE, 4 * EXPERT_SIZE, true, 2);

    PrefetchCandidate c1;
    c1.target_stage_id = 4;
    c1.expert_id = 45;
    c1.prediction_rank = 1;
    c1.prediction_score = 0.80;

    PrefetchCandidate c2;
    c2.target_stage_id = 4;
    c2.expert_id = 12;
    c2.prediction_rank = 1;
    c2.prediction_score = 0.80;

    PrefetchCandidate c3;
    c3.target_stage_id = 4;
    c3.expert_id = 33;
    c3.prediction_rank = 1;
    c3.prediction_score = 0.80;

    auto selected = device->select_pin_candidates_for_stage(4, {c1, c2, c3});
    TEST_ASSERT(selected.size() == 2, "Must select exactly 2 candidates");
    TEST_ASSERT(selected[0].expert_id == 12, "Expert 12 must be selected first (lowest ID)");
    TEST_ASSERT(selected[1].expert_id == 33, "Expert 33 must be selected second (second lowest ID)");

    std::cout << "[PASS] Test 5: Expert ID tie-break is deterministic verified\n";
    return true;
}

// Test 6: Per-target-stage pin budget is enforced
bool test_6_per_target_stage_pin_budget_enforced() {
    std::cout << "[RUNNING] Test 6: Per-target-stage pin budget is enforced...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 8 * EXPERT_SIZE, 4 * EXPERT_SIZE, true, 2);

    std::vector<PrefetchCandidate> cands;
    for (int i = 0; i < 5; ++i) {
        PrefetchCandidate c;
        c.target_stage_id = 5;
        c.expert_id = 10 + i;
        c.prediction_rank = i + 1;
        c.prediction_score = 0.9 - 0.1 * i;
        cands.push_back(c);
    }

    auto selected = device->select_pin_candidates_for_stage(5, cands);
    TEST_ASSERT(selected.size() == 2, "Budget of 2 must be strictly enforced");
    TEST_ASSERT(device->pabi_metrics.pin_candidates_selected == 2, "Selected metric must be 2");

    std::cout << "[PASS] Test 6: Per-target-stage pin budget is enforced verified\n";
    return true;
}

// Test 7: Selected in-flight candidate receives pin intent
bool test_7_selected_inflight_candidate_receives_pin_intent() {
    std::cout << "[RUNNING] Test 7: Selected in-flight candidate receives pin intent...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 8 * EXPERT_SIZE, 4 * EXPERT_SIZE, true, 2);

    PrefetchCandidate c;
    c.candidate_id = 701;
    c.target_stage_id = 1;
    c.expert_id = 5;
    c.prediction_rank = 1;
    c.prediction_score = 0.95;

    device->select_pin_candidates_for_stage(1, {c});
    TEST_ASSERT(device->pabi_metrics.pin_intents_created == 1, "Pin intent must be created");

    uint64_t tid = device->submit_prefetch_transfer(1, 5, 0.0, EXPERT_SIZE);
    TEST_ASSERT(tid != 0, "Transfer must be submitted");
    TEST_ASSERT((device->inflight_registry[{1, 5}].has_pin_intent == true), "Inflight entry must have pin intent");
    TEST_ASSERT((device->inflight_registry[{1, 5}].pin_expiration_stage_id == 1), "Expiration stage must be 1");
    TEST_ASSERT(device->pabi_metrics.inflight_pin_intents == 1, "Inflight pin metric must be 1");

    std::cout << "[PASS] Test 7: Selected in-flight candidate receives pin intent verified\n";
    return true;
}

// Test 8: Completed transfer receives a speculative resident pin
bool test_8_completed_transfer_receives_speculative_resident_pin() {
    std::cout << "[RUNNING] Test 8: Completed transfer receives a speculative resident pin...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 8 * EXPERT_SIZE, 4 * EXPERT_SIZE, true, 2);

    PrefetchCandidate c;
    c.candidate_id = 801;
    c.target_stage_id = 2;
    c.expert_id = 15;
    c.prediction_rank = 1;
    c.prediction_score = 0.92;

    device->select_pin_candidates_for_stage(2, {c});
    uint64_t tid = device->submit_prefetch_transfer(2, 15, 0.0, EXPERT_SIZE);
    TEST_ASSERT(tid != 0, "Transfer must be submitted");

    device->process_completed_prefetches(4000000.0);
    auto it = device->resident_experts.find({2, 15});
    TEST_ASSERT((it != device->resident_experts.end()), "Expert (2, 15) must be resident");
    TEST_ASSERT((it->second.classification == ExpertClassification::SPECULATIVE), "Must be SPECULATIVE");
    TEST_ASSERT((it->second.is_pinned == true), "Must be pinned");
    TEST_ASSERT((it->second.pin_expiration_stage_id == 2), "Expiration stage must be 2");
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants must hold");

    std::cout << "[PASS] Test 8: Completed transfer receives a speculative resident pin verified\n";
    return true;
}

// Test 9: Pinned expert is skipped during speculative LRU eviction
bool test_9_pinned_expert_skipped_during_speculative_lru_eviction() {
    std::cout << "[RUNNING] Test 9: Pinned expert is skipped during speculative LRU eviction...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 8 * EXPERT_SIZE, 2 * EXPERT_SIZE, true, 2);

    device->mark_expert_speculative(1, 10, EXPERT_SIZE);
    PrefetchCandidate c;
    c.target_stage_id = 1;
    c.expert_id = 10;
    c.prediction_rank = 1;
    c.prediction_score = 0.95;
    device->select_pin_candidates_for_stage(1, {c});
    TEST_ASSERT((device->resident_experts[{1, 10}].is_pinned == true), "Expert (1, 10) must be pinned");

    device->mark_expert_speculative(1, 20, EXPERT_SIZE);
    TEST_ASSERT((device->resident_experts[{1, 20}].is_pinned == false), "Expert (1, 20) must be unpinned");
    TEST_ASSERT(device->resident_speculative_bytes == 2 * EXPERT_SIZE, "Speculative bytes must be 2 * EXPERT_SIZE");

    // Overflow by adding (1, 30): pinned (1, 10) must be skipped, unpinned (1, 20) must be evicted!
    device->mark_expert_speculative(1, 30, EXPERT_SIZE);

    TEST_ASSERT((device->resident_experts.find({1, 10}) != device->resident_experts.end()), "Pinned expert (1, 10) must survive!");
    TEST_ASSERT((device->resident_experts.find({1, 20}) == device->resident_experts.end()), "Unpinned expert (1, 20) must be evicted!");
    TEST_ASSERT((device->resident_experts.find({1, 30}) != device->resident_experts.end()), "New expert (1, 30) must be resident!");
    TEST_ASSERT(device->pabi_metrics.pinned_eviction_skipped > 0, "Pinned eviction skip metric must increment");
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants must hold");

    std::cout << "[PASS] Test 9: Pinned expert is skipped during speculative LRU eviction verified\n";
    return true;
}

// Test 10: Unpinned speculative expert is evicted before pinned speculative expert
bool test_10_unpinned_speculative_evicted_before_pinned() {
    std::cout << "[RUNNING] Test 10: Unpinned speculative expert is evicted before pinned...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 8 * EXPERT_SIZE, 2 * EXPERT_SIZE, true, 2);

    device->mark_expert_speculative(1, 1, EXPERT_SIZE); // unpinned
    device->mark_expert_speculative(1, 2, EXPERT_SIZE); // pinned
    PrefetchCandidate c;
    c.target_stage_id = 1;
    c.expert_id = 2;
    c.prediction_rank = 1;
    c.prediction_score = 0.90;
    device->select_pin_candidates_for_stage(1, {c});

    device->mark_expert_speculative(1, 3, EXPERT_SIZE);
    TEST_ASSERT((device->resident_experts.find({1, 1}) == device->resident_experts.end()), "Unpinned expert 1 evicted");
    TEST_ASSERT((device->resident_experts.find({1, 2}) != device->resident_experts.end()), "Pinned expert 2 survived");
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants must hold");

    std::cout << "[PASS] Test 10: Unpinned speculative expert is evicted before pinned verified\n";
    return true;
}

// Test 11: Pinned expert is skipped during total HBM overflow
bool test_11_pinned_expert_skipped_during_total_hbm_overflow() {
    std::cout << "[RUNNING] Test 11: Pinned expert is skipped during total HBM overflow...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 3 * EXPERT_SIZE, 2 * EXPERT_SIZE, true, 2);

    device->mark_expert_demand(0, 1, EXPERT_SIZE);

    device->mark_expert_speculative(1, 10, EXPERT_SIZE);
    PrefetchCandidate c;
    c.target_stage_id = 1;
    c.expert_id = 10;
    c.prediction_rank = 1;
    c.prediction_score = 0.95;
    device->select_pin_candidates_for_stage(1, {c});

    device->mark_expert_speculative(1, 20, EXPERT_SIZE);
    TEST_ASSERT(device->resident_bytes == 3 * EXPERT_SIZE, "HBM must be 100% full (3 experts)");

    // Demand load (0, 2). Unpinned spec (1, 20) evicted. Pinned (1, 10) survives! Demand (0, 1) survives!
    device->mark_expert_demand(0, 2, EXPERT_SIZE);

    TEST_ASSERT((device->resident_experts.find({1, 20}) == device->resident_experts.end()), "Unpinned spec (1, 20) evicted");
    TEST_ASSERT((device->resident_experts.find({1, 10}) != device->resident_experts.end()), "Pinned spec (1, 10) survived");
    TEST_ASSERT((device->resident_experts.find({0, 1}) != device->resident_experts.end()), "Demand (0, 1) survived");
    TEST_ASSERT((device->resident_experts.find({0, 2}) != device->resident_experts.end()), "New demand (0, 2) resident");
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants must hold");

    std::cout << "[PASS] Test 11: Pinned expert is skipped during total HBM overflow verified\n";
    return true;
}

// Test 12: Incoming speculative insertion is rejected if all eviction candidates are pinned
bool test_12_incoming_speculative_rejected_if_all_pinned() {
    std::cout << "[RUNNING] Test 12: Incoming speculative insertion rejected if all pinned...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE, true, 2);

    device->mark_expert_speculative(1, 1, EXPERT_SIZE);
    device->mark_expert_speculative(1, 2, EXPERT_SIZE);

    PrefetchCandidate c1;
    c1.target_stage_id = 1;
    c1.expert_id = 1;
    c1.prediction_rank = 1;
    c1.prediction_score = 0.9;

    PrefetchCandidate c2;
    c2.target_stage_id = 1;
    c2.expert_id = 2;
    c2.prediction_rank = 2;
    c2.prediction_score = 0.8;

    device->select_pin_candidates_for_stage(1, {c1, c2});
    TEST_ASSERT((device->resident_experts[{1, 1}].is_pinned == true), "Expert 1 must be pinned");
    TEST_ASSERT((device->resident_experts[{1, 2}].is_pinned == true), "Expert 2 must be pinned");

    // Attempt to insert third speculative expert (1, 3). Must be REJECTED!
    device->mark_expert_speculative(1, 3, EXPERT_SIZE);

    TEST_ASSERT((device->resident_experts.find({1, 3}) == device->resident_experts.end()), "Expert 3 must be rejected");
    TEST_ASSERT(device->resident_speculative_bytes == 2 * EXPERT_SIZE, "Speculative bytes must not exceed capacity");
    TEST_ASSERT(device->pabi_metrics.speculative_reject_pinned_capacity == 1, "Reject pinned capacity metric must be 1");
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants must hold");

    std::cout << "[PASS] Test 12: Incoming speculative insertion rejected if all pinned verified\n";
    return true;
}

// Test 13: Demand hit promotes pinned speculative expert and clears its pin
bool test_13_demand_hit_promotes_pinned_speculative_and_clears_pin() {
    std::cout << "[RUNNING] Test 13: Demand hit promotes pinned speculative expert and clears pin...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE, true, 2);

    device->mark_expert_speculative(1, 10, EXPERT_SIZE);
    PrefetchCandidate c;
    c.target_stage_id = 1;
    c.expert_id = 10;
    c.prediction_rank = 1;
    c.prediction_score = 0.95;
    device->select_pin_candidates_for_stage(1, {c});
    TEST_ASSERT((device->resident_experts[{1, 10}].is_pinned == true), "Expert must be pinned");

    bool hit = device->is_expert_resident(1, 10);
    TEST_ASSERT(hit, "Must hit resident expert");
    auto it = device->resident_experts.find({1, 10});
    TEST_ASSERT((it->second.classification == ExpertClassification::DEMAND_VERIFIED), "Classification must be DEMAND_VERIFIED");
    TEST_ASSERT((it->second.is_pinned == false), "Pin must be cleared on promotion");
    TEST_ASSERT(device->pabi_metrics.pins_cleared_on_promotion == 1, "Pins cleared on promotion must be 1");
    TEST_ASSERT(device->pabi_metrics.pinned_expert_demand_hits == 1, "Pinned expert demand hits must be 1");
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants must hold");

    std::cout << "[PASS] Test 13: Demand hit promotes pinned speculative expert and clears pin verified\n";
    return true;
}

// Test 14: In-flight prefetch promoted to demand does not receive speculative pin
bool test_14_inflight_prefetch_promoted_to_demand_no_pin() {
    std::cout << "[RUNNING] Test 14: In-flight prefetch promoted to demand does not receive pin...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE, true, 2);

    PrefetchCandidate c;
    c.target_stage_id = 2;
    c.expert_id = 50;
    c.prediction_rank = 1;
    c.prediction_score = 0.9;
    device->select_pin_candidates_for_stage(2, {c});

    uint64_t tid = device->submit_prefetch_transfer(2, 50, 0.0, EXPERT_SIZE);
    TEST_ASSERT((device->inflight_registry[{2, 50}].has_pin_intent == true), "Must have pin intent");

    device->submit_demand_transfer(2, 50, 1000.0, EXPERT_SIZE);
    TEST_ASSERT((device->inflight_registry[{2, 50}].transfer_class == TransferClass::DEMAND), "Transfer class must be DEMAND");
    TEST_ASSERT((device->inflight_registry[{2, 50}].has_pin_intent == false), "Pin intent must be cleared");

    device->process_completed_prefetches(4000000.0);
    auto it = device->resident_experts.find({2, 50});
    TEST_ASSERT((it != device->resident_experts.end()), "Expert must be resident");
    TEST_ASSERT((it->second.classification == ExpertClassification::DEMAND_VERIFIED), "Must be DEMAND_VERIFIED");
    TEST_ASSERT((it->second.is_pinned == false), "Pin must NOT be applied to demand transfer");
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants must hold");

    std::cout << "[PASS] Test 14: In-flight prefetch promoted to demand does not receive pin verified\n";
    return true;
}

// Test 15: Pin expires at exact target-stage completion
bool test_15_pin_expires_at_exact_target_stage_completion() {
    std::cout << "[RUNNING] Test 15: Pin expires at exact target-stage completion...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE, true, 2);

    device->mark_expert_speculative(2, 5, EXPERT_SIZE);
    PrefetchCandidate c;
    c.target_stage_id = 2;
    c.expert_id = 5;
    c.prediction_rank = 1;
    c.prediction_score = 0.95;
    device->select_pin_candidates_for_stage(2, {c});
    TEST_ASSERT((device->resident_experts[{2, 5}].is_pinned == true), "Expert must be pinned");

    int exp1 = device->expire_pins_for_stage(1);
    TEST_ASSERT(exp1 == 0, "Stage 1 expiration must not affect stage 2 pin");
    TEST_ASSERT((device->resident_experts[{2, 5}].is_pinned == true), "Expert must remain pinned");

    int exp2 = device->expire_pins_for_stage(2);
    TEST_ASSERT(exp2 == 1, "Must expire exactly 1 pin");
    TEST_ASSERT((device->resident_experts[{2, 5}].is_pinned == false), "Pin must be cleared");
    TEST_ASSERT(device->pabi_metrics.pin_expirations == 1, "Pin expirations metric must be 1");
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants must hold");

    std::cout << "[PASS] Test 15: Pin expires at exact target-stage completion verified\n";
    return true;
}

// Test 16: Expiration does not evict the expert immediately
bool test_16_expiration_does_not_evict_expert_immediately() {
    std::cout << "[RUNNING] Test 16: Expiration does not evict expert immediately...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE, true, 2);

    device->mark_expert_speculative(2, 7, EXPERT_SIZE);
    PrefetchCandidate c;
    c.target_stage_id = 2;
    c.expert_id = 7;
    c.prediction_rank = 1;
    c.prediction_score = 0.95;
    device->select_pin_candidates_for_stage(2, {c});

    long long before_bytes = device->resident_speculative_bytes;
    device->expire_pins_for_stage(2);

    TEST_ASSERT((device->resident_experts.find({2, 7}) != device->resident_experts.end()), "Expert must remain in resident_experts");
    TEST_ASSERT(device->resident_speculative_bytes == before_bytes, "Speculative bytes must not change on expiration");
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants must hold");

    std::cout << "[PASS] Test 16: Expiration does not evict expert immediately verified\n";
    return true;
}

// Test 17: Expired expert becomes eligible for ordinary speculative LRU eviction
bool test_17_expired_expert_eligible_for_ordinary_lru_eviction() {
    std::cout << "[RUNNING] Test 17: Expired expert eligible for ordinary LRU eviction...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE, true, 2);

    device->mark_expert_speculative(1, 1, EXPERT_SIZE);
    device->mark_expert_speculative(2, 2, EXPERT_SIZE);

    PrefetchCandidate c;
    c.target_stage_id = 1;
    c.expert_id = 1;
    c.prediction_rank = 1;
    c.prediction_score = 0.95;
    device->select_pin_candidates_for_stage(1, {c});

    device->expire_pins_for_stage(1);

    device->mark_expert_speculative(3, 3, EXPERT_SIZE);
    TEST_ASSERT((device->resident_experts.find({1, 1}) == device->resident_experts.end()), "Expired expert 1 must be evicted");
    TEST_ASSERT((device->resident_experts.find({2, 2}) != device->resident_experts.end()), "Expert 2 remains");
    TEST_ASSERT((device->resident_experts.find({3, 3}) != device->resident_experts.end()), "Expert 3 resident");
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants must hold");

    std::cout << "[PASS] Test 17: Expired expert eligible for ordinary LRU eviction verified\n";
    return true;
}

// Test 18: Transfer completing after target-stage deadline is inserted unpinned
bool test_18_transfer_completing_after_deadline_inserted_unpinned() {
    std::cout << "[RUNNING] Test 18: Transfer completing after deadline is inserted unpinned...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE, true, 2);

    PrefetchCandidate c;
    c.target_stage_id = 1;
    c.expert_id = 18;
    c.prediction_rank = 1;
    c.prediction_score = 0.90;
    device->select_pin_candidates_for_stage(1, {c});

    uint64_t tid = device->submit_prefetch_transfer(1, 18, 0.0, EXPERT_SIZE);
    TEST_ASSERT(tid != 0, "Transfer must be submitted");

    device->expire_pins_for_stage(1);
    TEST_ASSERT((device->inflight_registry[{1, 18}].has_pin_intent == false), "Inflight pin intent must be cleared");

    device->process_completed_prefetches(4000000.0);
    auto it = device->resident_experts.find({1, 18});
    TEST_ASSERT((it != device->resident_experts.end()), "Expert must be resident");
    TEST_ASSERT((it->second.classification == ExpertClassification::SPECULATIVE), "Must be SPECULATIVE");
    TEST_ASSERT((it->second.is_pinned == false), "Expert must be inserted UNPINNED");
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants must hold");

    std::cout << "[PASS] Test 18: Transfer completing after deadline is inserted unpinned verified\n";
    return true;
}

// Test 19: Cancelled transfer clears pin intent
bool test_19_cancelled_transfer_clears_pin_intent() {
    std::cout << "[RUNNING] Test 19: Cancelled transfer clears pin intent...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE, true, 2);

    PrefetchCandidate c;
    c.target_stage_id = 2;
    c.expert_id = 19;
    c.prediction_rank = 1;
    c.prediction_score = 0.9;
    device->select_pin_candidates_for_stage(2, {c});

    uint64_t tid = device->submit_prefetch_transfer(2, 19, 0.0, EXPERT_SIZE);
    TEST_ASSERT((device->active_pin_intents.find({2, 19}) != device->active_pin_intents.end()), "Pin intent must exist");

    device->ddr5_transfer_engine->cancel_transfer(tid, 500.0);
    device->process_completed_prefetches(1000.0);

    TEST_ASSERT(!device->is_expert_inflight(2, 19), "Expert must no longer be in flight");
    TEST_ASSERT((device->active_pin_intents.find({2, 19}) == device->active_pin_intents.end()), "Pin intent must be cleared");
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants must hold");

    std::cout << "[PASS] Test 19: Cancelled transfer clears pin intent verified\n";
    return true;
}

// Test 20: Multiple requests predicting the same expert do not create duplicate pin entries
bool test_20_multiple_requests_same_expert_no_duplicate_pins() {
    std::cout << "[RUNNING] Test 20: Multiple requests predicting same expert no duplicates...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE, true, 2);

    PrefetchCandidate c1;
    c1.request_id = 101;
    c1.target_stage_id = 3;
    c1.expert_id = 8;
    c1.prediction_rank = 1;
    c1.prediction_score = 0.90;

    PrefetchCandidate c2;
    c2.request_id = 102;
    c2.target_stage_id = 3;
    c2.expert_id = 8;
    c2.prediction_rank = 2;
    c2.prediction_score = 0.70;

    auto selected = device->select_pin_candidates_for_stage(3, {c1, c2});
    TEST_ASSERT(selected.size() == 1, "Must select exactly 1 pin candidate (deduplicated)");
    TEST_ASSERT(selected[0].prediction_rank == 1, "Best rank (1) must be retained");
    TEST_ASSERT(device->active_pin_intents.size() == 1, "Active pin intents map must contain exactly 1 entry");

    std::cout << "[PASS] Test 20: Multiple requests predicting same expert no duplicates verified\n";
    return true;
}

// Test 21: Multiple target-stage deadlines are handled deterministically
bool test_21_multiple_target_stage_deadlines_handled_deterministically() {
    std::cout << "[RUNNING] Test 21: Multiple target-stage deadlines handled deterministically...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 8 * EXPERT_SIZE, 4 * EXPERT_SIZE, true, 2);

    device->mark_expert_speculative(1, 10, EXPERT_SIZE);
    PrefetchCandidate c1;
    c1.target_stage_id = 1;
    c1.expert_id = 10;
    c1.prediction_rank = 1;
    c1.prediction_score = 0.9;
    device->select_pin_candidates_for_stage(1, {c1});

    device->mark_expert_speculative(2, 20, EXPERT_SIZE);
    PrefetchCandidate c2;
    c2.target_stage_id = 2;
    c2.expert_id = 20;
    c2.prediction_rank = 1;
    c2.prediction_score = 0.9;
    device->select_pin_candidates_for_stage(2, {c2});

    TEST_ASSERT((device->resident_experts[{1, 10}].is_pinned == true), "Expert (1, 10) pinned");
    TEST_ASSERT((device->resident_experts[{2, 20}].is_pinned == true), "Expert (2, 20) pinned");

    device->expire_pins_for_stage(1);
    TEST_ASSERT((device->resident_experts[{1, 10}].is_pinned == false), "Expert (1, 10) must be unpinned");
    TEST_ASSERT((device->resident_experts[{2, 20}].is_pinned == true), "Expert (2, 20) must remain pinned");

    device->expire_pins_for_stage(2);
    TEST_ASSERT((device->resident_experts[{2, 20}].is_pinned == false), "Expert (2, 20) must be unpinned");
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants must hold");

    std::cout << "[PASS] Test 21: Multiple target-stage deadlines handled deterministically verified\n";
    return true;
}

// Test 22: Skipped/aborted stage cleanup does not leak pins
bool test_22_skipped_aborted_stage_cleanup_does_not_leak_pins() {
    std::cout << "[RUNNING] Test 22: Skipped/aborted stage cleanup does not leak pins...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE, true, 2);

    PrefetchCandidate c;
    c.target_stage_id = 7;
    c.expert_id = 22;
    c.prediction_rank = 1;
    c.prediction_score = 0.9;
    device->select_pin_candidates_for_stage(7, {c});
    TEST_ASSERT((device->active_pin_intents.find({7, 22}) != device->active_pin_intents.end()), "Pin intent exists");

    device->cleanup_unexecuted_stage_pins(7);
    TEST_ASSERT((device->active_pin_intents.find({7, 22}) == device->active_pin_intents.end()), "Pin intent must be cleaned up");
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants must hold");

    std::cout << "[PASS] Test 22: Skipped/aborted stage cleanup does not leak pins verified\n";
    return true;
}

// Test 23: Cache reset clears all resident and in-flight pin state
bool test_23_cache_reset_clears_all_pin_state() {
    std::cout << "[RUNNING] Test 23: Cache reset clears all resident and in-flight pin state...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE, true, 2);

    device->mark_expert_speculative(1, 23, EXPERT_SIZE);
    PrefetchCandidate c;
    c.target_stage_id = 1;
    c.expert_id = 23;
    c.prediction_rank = 1;
    c.prediction_score = 0.9;
    device->select_pin_candidates_for_stage(1, {c});

    TEST_ASSERT(device->active_pin_intents.size() == 1, "Must have 1 pin intent");
    TEST_ASSERT((device->resident_experts[{1, 23}].is_pinned == true), "Must be pinned");

    device->reset_expert_cache();
    TEST_ASSERT(device->active_pin_intents.empty(), "Active pin intents must be empty");
    TEST_ASSERT(device->resident_experts.empty(), "Resident experts must be empty");
    TEST_ASSERT(device->resident_bytes == 0, "Resident bytes must be 0");
    TEST_ASSERT(device->resident_speculative_bytes == 0, "Speculative bytes must be 0");
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants must hold");

    std::cout << "[PASS] Test 23: Cache reset clears all resident and in-flight pin state verified\n";
    return true;
}

// Test 24: Byte-accounting invariants remain valid through pin/unpin/promotion/eviction
bool test_24_byte_accounting_invariants_valid_through_lifecycle() {
    std::cout << "[RUNNING] Test 24: Byte-accounting invariants valid through lifecycle...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 4 * EXPERT_SIZE, 2 * EXPERT_SIZE, true, 2);

    TEST_ASSERT(device->verify_pabi_invariants(), "Initial invariants");

    device->mark_expert_speculative(1, 1, EXPERT_SIZE);
    PrefetchCandidate c1;
    c1.target_stage_id = 1;
    c1.expert_id = 1;
    c1.prediction_rank = 1;
    c1.prediction_score = 0.9;
    device->select_pin_candidates_for_stage(1, {c1});
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants after pin");

    device->mark_expert_speculative(1, 2, EXPERT_SIZE);
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants after 2nd insert");

    device->is_expert_resident(1, 1);
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants after promotion");

    device->mark_expert_demand(0, 3, EXPERT_SIZE);
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants after demand insert");

    device->mark_expert_demand(0, 4, EXPERT_SIZE);
    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants after eviction");

    std::cout << "[PASS] Test 24: Byte-accounting invariants valid through lifecycle verified\n";
    return true;
}

// Test 25: Regression with Phase 4D.1, 4D.2, and 4D.3
bool test_25_regression_phase4d1_4d2_4d3() {
    std::cout << "[RUNNING] Test 25: Regression with Phase 4D.1, 4D.2, and 4D.3...\n";
    Device::Ptr device = create_test_device(EvictionPolicy::PABI, 8 * EXPERT_SIZE, 4 * EXPERT_SIZE, true, 2);

    PrefetchConfig qcfg;
    qcfg.enabled = true;
    qcfg.budget = 4;
    qcfg.prediction_horizon = 1;
    PrefetchCandidateQueue::Ptr queue = std::make_shared<PrefetchCandidateQueue>(qcfg, 5, 32);

    PredictionResult pr;
    pr.target_stage = 1;
    pr.candidate_experts = {10, 11};
    CandidateScore s1; s1.expert_id = 10; s1.score = 0.95;
    CandidateScore s2; s2.expert_id = 11; s2.score = 0.85;
    pr.scores = {s1, s2};

    queue->enqueue_candidates(1, 0, 0, 1, 1, pr);

    int dispatched = device->dispatch_prefetch_candidates(queue, 0.0, 2);
    TEST_ASSERT(dispatched == 2, "Must dispatch 2 candidates");
    TEST_ASSERT((device->inflight_registry[{1, 10}].has_pin_intent == true), "Candidate 10 must have pin intent");
    TEST_ASSERT((device->inflight_registry[{1, 11}].has_pin_intent == true), "Candidate 11 must have pin intent");

    device->process_completed_prefetches(8000000.0);
    TEST_ASSERT((device->resident_experts[{1, 10}].is_pinned == true), "Resident expert 10 must be pinned");
    TEST_ASSERT((device->resident_experts[{1, 11}].is_pinned == true), "Resident expert 11 must be pinned");

    device->is_expert_resident(1, 10);
    TEST_ASSERT((device->resident_experts[{1, 10}].classification == ExpertClassification::DEMAND_VERIFIED), "Expert 10 promoted");
    TEST_ASSERT((device->resident_experts[{1, 10}].is_pinned == false), "Pin cleared on promotion");

    device->expire_pins_for_stage(1);
    TEST_ASSERT((device->resident_experts[{1, 11}].is_pinned == false), "Expert 11 pin expired");

    TEST_ASSERT(device->verify_pabi_invariants(), "Invariants must hold");

    std::cout << "[PASS] Test 25: Regression with Phase 4D.1, 4D.2, and 4D.3 verified\n";
    return true;
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "PHASE 4D.4: Belady-Style Predictor-Score Pinning & Stage Expiration\n";
    std::cout << "Per-Stage Pin Budget | In-Flight Pin Intent | 25 Test Cases\n";
    std::cout << "======================================================================\n\n";

    int passed = 0;
    int failed = 0;

    std::vector<std::pair<std::string, bool(*)()>> tests = {
        {"Test 1: Pinning disabled preserves Phase 4D.3 behavior", test_1_pinning_disabled_preserves_pabi_behavior},
        {"Test 2: LRU policy never activates PABI pinning", test_2_lru_policy_never_activates_pinning},
        {"Test 3: Top-K selection respects candidate rank", test_3_top_k_selection_respects_candidate_rank},
        {"Test 4: Score tie-break is deterministic", test_4_score_tie_break_is_deterministic},
        {"Test 5: Expert ID tie-break is deterministic", test_5_expert_id_tie_break_is_deterministic},
        {"Test 6: Per-target-stage pin budget is enforced", test_6_per_target_stage_pin_budget_enforced},
        {"Test 7: Selected in-flight candidate receives pin intent", test_7_selected_inflight_candidate_receives_pin_intent},
        {"Test 8: Completed transfer receives a speculative resident pin", test_8_completed_transfer_receives_speculative_resident_pin},
        {"Test 9: Pinned expert is skipped during speculative LRU eviction", test_9_pinned_expert_skipped_during_speculative_lru_eviction},
        {"Test 10: Unpinned speculative expert is evicted before pinned", test_10_unpinned_speculative_evicted_before_pinned},
        {"Test 11: Pinned expert is skipped during total HBM overflow", test_11_pinned_expert_skipped_during_total_hbm_overflow},
        {"Test 12: Incoming speculative insertion rejected if all pinned", test_12_incoming_speculative_rejected_if_all_pinned},
        {"Test 13: Demand hit promotes pinned speculative expert and clears pin", test_13_demand_hit_promotes_pinned_speculative_and_clears_pin},
        {"Test 14: In-flight prefetch promoted to demand does not receive pin", test_14_inflight_prefetch_promoted_to_demand_no_pin},
        {"Test 15: Pin expires at exact target-stage completion", test_15_pin_expires_at_exact_target_stage_completion},
        {"Test 16: Expiration does not evict expert immediately", test_16_expiration_does_not_evict_expert_immediately},
        {"Test 17: Expired expert eligible for ordinary LRU eviction", test_17_expired_expert_eligible_for_ordinary_lru_eviction},
        {"Test 18: Transfer completing after deadline is inserted unpinned", test_18_transfer_completing_after_deadline_inserted_unpinned},
        {"Test 19: Cancelled transfer clears pin intent", test_19_cancelled_transfer_clears_pin_intent},
        {"Test 20: Multiple requests predicting same expert no duplicates", test_20_multiple_requests_same_expert_no_duplicate_pins},
        {"Test 21: Multiple target-stage deadlines handled deterministically", test_21_multiple_target_stage_deadlines_handled_deterministically},
        {"Test 22: Skipped/aborted stage cleanup does not leak pins", test_22_skipped_aborted_stage_cleanup_does_not_leak_pins},
        {"Test 23: Cache reset clears all resident and in-flight pin state", test_23_cache_reset_clears_all_pin_state},
        {"Test 24: Byte-accounting invariants valid through lifecycle", test_24_byte_accounting_invariants_valid_through_lifecycle},
        {"Test 25: Regression with Phase 4D.1, 4D.2, and 4D.3", test_25_regression_phase4d1_4d2_4d3}
    };

    for (const auto& t : tests) {
        bool res = false;
        try {
            res = t.second();
        } catch (const std::exception& e) {
            std::cerr << "[EXCEPTION] in " << t.first << ": " << e.what() << "\n";
            res = false;
        }
        if (res) {
            passed = passed + 1;
        } else {
            failed = failed + 1;
        }
    }

    std::cout << "\n======================================================================\n";
    std::cout << "TEST SUITE SUMMARY: " << passed << " PASSED, " << failed << " FAILED\n";
    std::cout << "======================================================================\n";

    if (failed > 0) {
        return 1;
    }
    return 0;
}
