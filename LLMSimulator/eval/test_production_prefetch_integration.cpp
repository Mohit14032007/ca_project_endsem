#include <iostream>
#include <memory>
#include <vector>
#include <cassert>
#include <string>
#include <cmath>
#include <filesystem>

#include "hardware/cluster.h"
#include "hardware/device.h"
#include "hardware/hardware_config.h"
#include "model/model_config.h"
#include "prefetch/prefetch_config.h"
#include "prefetch/live_prediction_engine.h"
#include "prefetch/prefetch_candidate_queue.h"
#include "prefetch/ddr5_transfer_engine.h"
#include "scheduler/scheduler.h"
#include "scheduler/sequence.h"

using namespace llm_system;

#define TEST_ASSERT(cond, msg) \
  do { \
    if (!(cond)) { \
      std::cerr << "[FAIL] Line " << __LINE__ << ": " << msg << std::endl; \
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
  if (std::filesystem::exists("build/dram_config_HBM3_80GB.yaml")) {
    return "build/dram_config_HBM3_80GB.yaml";
  }
  return "dram_config_HBM3_80GB.yaml";
}

static SystemConfig create_test_system_config(PrefetchMode mode, EvictionPolicy policy, bool quarantine, bool pinning) {
  SystemConfig sys_cfg;
  sys_cfg.num_node = 1;
  sys_cfg.num_device = 1;
  sys_cfg.use_event_driven_dram = true;
  sys_cfg.ddr5_bandwidth_gbps = 64.0;
  sys_cfg.hbm_dram_cfg_path = get_hbm_cfg_path();

  sys_cfg.prefetch_config.enabled = true;
  if (mode == PrefetchMode::NONE) {
    sys_cfg.prefetch_config.enabled = false;
  }
  sys_cfg.prefetch_config.mode = mode;
  sys_cfg.prefetch_config.predictor_type = PredictorType::CSTP;
  sys_cfg.prefetch_config.confidence_threshold = 0;
  sys_cfg.prefetch_config.eviction_policy = policy;
  sys_cfg.prefetch_config.pabi_quarantine_enabled = quarantine;
  sys_cfg.prefetch_config.pabi_pinning_enabled = pinning;
  sys_cfg.prefetch_config.pabi_max_pinned_experts = 4;
  sys_cfg.prefetch_config.budget = 4;
  sys_cfg.prefetch_config.prediction_horizon = 1;
  sys_cfg.prefetch_config.lookahead_tokens = 0;
  sys_cfg.prefetch_config.asynchronous = true;
  sys_cfg.prefetch_config.transfer_model = TransferModel::EVENT_DRIVEN;

  return sys_cfg;
}

static Device::Ptr create_test_device(SystemConfig sys, ModelConfig model_cfg, Scheduler::Ptr sched = nullptr) {
  Cluster::Ptr cluster = Cluster::Create(sys, sched);
  Device::Ptr dev = Device::Create(sys, 0, cluster);
  dev->model_config = model_cfg;
  return dev;
}

static ModelConfig create_test_model_config() {
  ModelConfig cfg = deepseekV3;
  cfg.ne_tp_dg = 1;
  cfg.e_tp_dg = 1;
  cfg.num_layers = 5;
  cfg.input_len = 128;
  cfg.output_len = 128;
  return cfg;
}

// Test 1: Verify prediction engine initialization under hardware_async and none
bool test_1_prediction_engine_initialization() {
  std::cout << "[Test 1] Prediction engine initialization under hardware_async vs none\n";
  ModelConfig model_cfg = create_test_model_config();

  // Subtest 1A: hardware_async initializes engine
  SystemConfig sys_async = create_test_system_config(PrefetchMode::HARDWARE_ASYNC, EvictionPolicy::LRU, false, false);
  Scheduler::Ptr sched_async = Scheduler::Create(sys_async, model_cfg, "none", 1, 1024, 1024);
  TEST_ASSERT(sched_async->get_prediction_engine() != nullptr, "hardware_async must initialize prediction engine");
  TEST_ASSERT(sched_async->get_candidate_queue() != nullptr, "candidate queue must be non-null under hardware_async");

  // Subtest 1B: mode none leaves engine null
  SystemConfig sys_none = create_test_system_config(PrefetchMode::NONE, EvictionPolicy::LRU, false, false);
  Scheduler::Ptr sched_none = Scheduler::Create(sys_none, model_cfg, "none", 1, 1024, 1024);
  TEST_ASSERT(sched_none->get_prediction_engine() == nullptr, "mode none must not initialize prediction engine");
  TEST_ASSERT(sched_none->get_candidate_queue() == nullptr, "candidate queue must be null under mode none");

  // Subtest 1C: prediction_only initializes engine
  SystemConfig sys_pred = create_test_system_config(PrefetchMode::PREDICTION_ONLY, EvictionPolicy::LRU, false, false);
  Scheduler::Ptr sched_pred = Scheduler::Create(sys_pred, model_cfg, "none", 1, 1024, 1024);
  TEST_ASSERT(sched_pred->get_prediction_engine() != nullptr, "prediction_only must initialize prediction engine");

  std::cout << "  -> PASS\n";
  return true;
}

// Test 2: Verify candidate dispatch occurs under hardware_async but not prediction_only
bool test_2_candidate_dispatch_modes() {
  std::cout << "[Test 2] Candidate dispatch occurs under hardware_async but not prediction_only\n";
  ModelConfig model_cfg = create_test_model_config();
  model_cfg.num_layers = 5;

  // Subtest 2A: hardware_async dispatches
  SystemConfig sys_async = create_test_system_config(PrefetchMode::HARDWARE_ASYNC, EvictionPolicy::LRU, false, false);
  Scheduler::Ptr sched_async = Scheduler::Create(sys_async, model_cfg, "none", 1, 1024, 1024);
  Device::Ptr dev_async = create_test_device(sys_async, model_cfg, sched_async);
  sched_async->set_device(dev_async);

  BatchedSequence::Ptr batch_async = BatchedSequence::Create(1, 1, sched_async);
  Sequence::Ptr seq_async = Sequence::Create(0, 128, 128);
  seq_async->current_len = 128;
  seq_async->num_process_token = 1;
  seq_async->get_expert_from_list = false;
  batch_async->add(seq_async);

  batch_async->update_expert(model_cfg.num_routed_expert, model_cfg.top_k, true);
  TEST_ASSERT(dev_async->prefetch_metrics.candidates_dispatched > 0, "Candidates must be dispatched under hardware_async");

  // Subtest 2B: prediction_only does not dispatch to device
  SystemConfig sys_pred = create_test_system_config(PrefetchMode::PREDICTION_ONLY, EvictionPolicy::LRU, false, false);
  Scheduler::Ptr sched_pred = Scheduler::Create(sys_pred, model_cfg, "none", 1, 1024, 1024);
  Device::Ptr dev_pred = create_test_device(sys_pred, model_cfg, sched_pred);
  sched_pred->set_device(dev_pred);

  BatchedSequence::Ptr batch_pred = BatchedSequence::Create(1, 1, sched_pred);
  Sequence::Ptr seq_pred = Sequence::Create(0, 128, 128);
  seq_pred->current_len = 128;
  seq_pred->num_process_token = 1;
  seq_pred->get_expert_from_list = false;
  batch_pred->add(seq_pred);

  batch_pred->update_expert(model_cfg.num_routed_expert, model_cfg.top_k, true);
  TEST_ASSERT(dev_pred->prefetch_metrics.candidates_dispatched == 0, "Candidates must NOT be dispatched under prediction_only");
  TEST_ASSERT(dev_pred->prefetch_metrics.prefetch_transfers_submitted == 0, "No prefetch transfers under prediction_only");

  std::cout << "  -> PASS\n";
  return true;
}

// Test 3: Stage deadline expiration at stage completion in production sequence loop
bool test_3_stage_deadline_expiration_sequence_loop() {
  std::cout << "[Test 3] Stage deadline expiration at stage completion in sequence loop\n";
  ModelConfig model_cfg = create_test_model_config();
  model_cfg.num_layers = 5;

  SystemConfig sys = create_test_system_config(PrefetchMode::HARDWARE_ASYNC, EvictionPolicy::PABI, true, true);
  Scheduler::Ptr sched = Scheduler::Create(sys, model_cfg, "none", 1, 1024, 1024);
  Device::Ptr dev = create_test_device(sys, model_cfg, sched);
  sched->set_device(dev);

  BatchedSequence::Ptr batch = BatchedSequence::Create(1, 1, sched);
  Sequence::Ptr seq = Sequence::Create(0, 128, 128);
  seq->current_len = 128;
  seq->num_process_token = 1;
  seq->get_expert_from_list = false;
  batch->add(seq);

  // Stage 0 routes and dispatches
  batch->update_expert(model_cfg.num_routed_expert, model_cfg.top_k, true);
  TEST_ASSERT(batch->last_completed_stage == 0, "last_completed_stage must be 0 after stage 0 routing");

  // Advance time to complete prefetches and process them
  dev->process_completed_prefetches(10000000.0);

  // Stage 1 routes: this should expire stage 0's pins!
  batch->update_expert(model_cfg.num_routed_expert, model_cfg.top_k, true);
  TEST_ASSERT(batch->last_completed_stage == 1, "last_completed_stage must be 1 after stage 1 routing");

  // Next token / step completion update: expires stage 1's pins!
  batch->update(1000);
  TEST_ASSERT(batch->last_completed_stage == -1, "last_completed_stage must be reset to -1 after batch->update()");

  std::cout << "  -> PASS\n";
  return true;
}

// Test 4: Current-stage progression and late transfer unpinned insertion
bool test_4_current_stage_and_late_transfer() {
  std::cout << "[Test 4] Current-stage progression and late transfer unpinned insertion\n";
  ModelConfig model_cfg = create_test_model_config();
  SystemConfig sys = create_test_system_config(PrefetchMode::HARDWARE_ASYNC, EvictionPolicy::PABI, true, true);
  Device::Ptr dev = create_test_device(sys, model_cfg, nullptr);

  // Set stage 0
  dev->set_current_stage(0);
  TEST_ASSERT(dev->current_stage_id == 0, "current_stage_id must be 0");

  // Submit prefetch targeting stage 0 with pin intent
  std::vector<PrefetchCandidate> cands;
  PrefetchCandidate c;
  c.candidate_id = 1001;
  c.target_stage_id = 0;
  c.expert_id = 15;
  c.prediction_score = 0.95;
  c.prediction_rank = 1;
  c.state = CandidateState::ADMITTED;
  cands.push_back(c);

  dev->select_pin_candidates_for_stage(0, cands);
  uint64_t tid = dev->submit_prefetch_transfer(0, 15, 0.0);
  TEST_ASSERT(tid != 0, "Transfer ID must be valid");

  // Now simulate that the execution advances to stage 1 BEFORE transfer finishes
  dev->set_current_stage(1);
  TEST_ASSERT(dev->current_stage_id == 1, "current_stage_id must be 1");

  // Advance time so transfer finishes
  dev->process_completed_prefetches(10000000.0);

  // Expert 15 should now be resident in speculative buffer, but UNPINNED because it was late (deadline 0 < current 1)
  std::pair<int, int> key = {0, 15};
  TEST_ASSERT(dev->resident_experts.find(key) != dev->resident_experts.end(), "Expert must be resident");
  TEST_ASSERT(dev->resident_experts[key].is_pinned == false, "Late transfer must be inserted unpinned");
  TEST_ASSERT(dev->pabi_metrics.pin_expirations > 0, "Late transfer must count as pin expiration");

  std::cout << "  -> PASS\n";
  return true;
}

// Test 5: Exact-deadline transfer completes on time and enters HBM PINNED
bool test_5_exact_deadline_transfer() {
  std::cout << "[Test 5] Exact-deadline transfer completes on time and enters HBM PINNED\n";
  ModelConfig model_cfg = create_test_model_config();
  SystemConfig sys = create_test_system_config(PrefetchMode::HARDWARE_ASYNC, EvictionPolicy::PABI, true, true);
  Device::Ptr dev = create_test_device(sys, model_cfg, nullptr);

  // Current stage is 1
  dev->set_current_stage(1);

  // Candidate has deadline 1
  std::vector<PrefetchCandidate> cands;
  PrefetchCandidate c;
  c.candidate_id = 2001;
  c.target_stage_id = 1;
  c.expert_id = 33;
  c.prediction_score = 0.90;
  c.prediction_rank = 1;
  c.state = CandidateState::ADMITTED;
  cands.push_back(c);

  dev->select_pin_candidates_for_stage(1, cands);
  uint64_t tid = dev->submit_prefetch_transfer(1, 33, 0.0);
  TEST_ASSERT(tid != 0, "Transfer must be submitted");

  // Transfer completes while current_stage_id is still 1
  dev->process_completed_prefetches(10000000.0);

  std::pair<int, int> key = {1, 33};
  TEST_ASSERT(dev->resident_experts.find(key) != dev->resident_experts.end(), "Expert must be resident");
  TEST_ASSERT(dev->resident_experts[key].is_pinned == true, "On-time exact-deadline transfer must be inserted pinned");
  TEST_ASSERT(dev->resident_experts[key].pin_expiration_stage_id == 1, "Expiration stage must be 1");

  std::cout << "  -> PASS\n";
  return true;
}

// Test 6: Multiple target-stage ownership for one physical expert
bool test_6_multiple_target_stage_ownership() {
  std::cout << "[Test 6] Multiple target-stage ownership for one physical expert\n";
  ModelConfig model_cfg = create_test_model_config();
  SystemConfig sys = create_test_system_config(PrefetchMode::HARDWARE_ASYNC, EvictionPolicy::PABI, true, true);
  Device::Ptr dev = create_test_device(sys, model_cfg, nullptr);

  // First intent: target stage 1
  std::vector<PrefetchCandidate> cands1;
  PrefetchCandidate c1;
  c1.candidate_id = 3001;
  c1.target_stage_id = 1;
  c1.expert_id = 77;
  c1.prediction_score = 0.85;
  c1.prediction_rank = 2;
  c1.state = CandidateState::ADMITTED;
  cands1.push_back(c1);
  dev->select_pin_candidates_for_stage(1, cands1);

  // Second intent: targeting same expert with later stage deadline (stage 2)
  std::vector<PrefetchCandidate> cands2;
  PrefetchCandidate c2;
  c2.candidate_id = 3002;
  c2.target_stage_id = 1; // same physical layer 1
  c2.expert_id = 77;
  c2.prediction_score = 0.95;
  c2.prediction_rank = 1;
  c2.state = CandidateState::ADMITTED;
  cands2.push_back(c2);
  dev->select_pin_candidates_for_stage(2, cands2);

  // Submit transfer and complete
  uint64_t tid = dev->submit_prefetch_transfer(1, 77, 0.0);
  dev->set_current_stage(0);
  dev->process_completed_prefetches(10000000.0);

  std::pair<int, int> key = {1, 77};
  TEST_ASSERT(dev->resident_experts.find(key) != dev->resident_experts.end(), "Expert must be resident");
  TEST_ASSERT(dev->resident_experts[key].is_pinned == true, "Expert must be pinned");
  TEST_ASSERT(dev->resident_experts[key].pin_expiration_stage_id == 2, "Expiration must extend to latest stage (2)");

  // Stage 1 expires: expert 77 must NOT be unpinned because it is still needed for stage 2!
  dev->expire_pins_for_stage(1);
  TEST_ASSERT(dev->resident_experts[key].is_pinned == true, "Expert must remain pinned after earlier stage expires");

  // Stage 2 expires: expert 77 should now be unpinned
  dev->expire_pins_for_stage(2);
  TEST_ASSERT(dev->resident_experts[key].is_pinned == false, "Expert must unpin when latest stage expires");

  std::cout << "  -> PASS\n";
  return true;
}

// Test 7: Demand promotion clears pin metadata and active intent
bool test_7_demand_promotion_clears_pins() {
  std::cout << "[Test 7] Demand promotion clears pin metadata and active intent\n";
  ModelConfig model_cfg = create_test_model_config();
  SystemConfig sys = create_test_system_config(PrefetchMode::HARDWARE_ASYNC, EvictionPolicy::PABI, true, true);
  Device::Ptr dev = create_test_device(sys, model_cfg, nullptr);

  std::vector<PrefetchCandidate> cands;
  PrefetchCandidate c;
  c.candidate_id = 4001;
  c.target_stage_id = 0;
  c.expert_id = 99;
  c.prediction_score = 0.98;
  c.prediction_rank = 1;
  c.state = CandidateState::ADMITTED;
  cands.push_back(c);

  dev->select_pin_candidates_for_stage(0, cands);
  uint64_t tid = dev->submit_prefetch_transfer(0, 99, 0.0);
  dev->set_current_stage(0);
  dev->process_completed_prefetches(10000000.0);

  std::pair<int, int> key = {0, 99};
  TEST_ASSERT(dev->resident_experts[key].is_pinned == true, "Must be pinned");
  TEST_ASSERT(dev->active_pin_intents.find(key) != dev->active_pin_intents.end(), "Pin intent must exist");

  // Demand arrives for expert {0, 99}
  bool hit = dev->is_expert_resident(0, 99);
  TEST_ASSERT(hit == true, "Must be a resident hit");
  TEST_ASSERT(dev->resident_experts[key].classification == ExpertClassification::DEMAND_VERIFIED, "Must be promoted to DEMAND_VERIFIED");
  TEST_ASSERT(dev->resident_experts[key].is_pinned == false, "Pin must be cleared on promotion");
  TEST_ASSERT(dev->resident_experts[key].pin_expiration_stage_id == -1, "Expiration stage must be reset");
  TEST_ASSERT(dev->active_pin_intents.find(key) == dev->active_pin_intents.end(), "Active pin intent must be erased");
  TEST_ASSERT(dev->pabi_metrics.pins_cleared_on_promotion > 0, "pins_cleared_on_promotion must be incremented");

  std::cout << "  -> PASS\n";
  return true;
}

// Test 8: Cancellation and cache reset cleanup all pin metadata
bool test_8_cancellation_and_reset_cleanup() {
  std::cout << "[Test 8] Cancellation and cache reset cleanup all pin metadata\n";
  ModelConfig model_cfg = create_test_model_config();
  SystemConfig sys = create_test_system_config(PrefetchMode::HARDWARE_ASYNC, EvictionPolicy::PABI, true, true);
  Device::Ptr dev = create_test_device(sys, model_cfg, nullptr);

  std::vector<PrefetchCandidate> cands;
  PrefetchCandidate c;
  c.candidate_id = 5001;
  c.target_stage_id = 1;
  c.expert_id = 120;
  c.prediction_score = 0.90;
  c.prediction_rank = 1;
  c.state = CandidateState::ADMITTED;
  cands.push_back(c);

  dev->select_pin_candidates_for_stage(1, cands);
  uint64_t tid = dev->submit_prefetch_transfer(1, 120, 0.0);
  std::pair<int, int> key = {1, 120};
  TEST_ASSERT(dev->active_pin_intents.find(key) != dev->active_pin_intents.end(), "Pin intent must exist");

  // Reset cache
  dev->reset_expert_cache();
  TEST_ASSERT(dev->active_pin_intents.empty(), "Active pin intents must be empty after reset");
  TEST_ASSERT(dev->resident_experts.empty(), "Resident experts must be empty after reset");
  TEST_ASSERT(dev->current_stage_id == 0, "current_stage_id must be reset to 0");

  std::cout << "  -> PASS\n";
  return true;
}

int main() {
  std::cout << "======================================================================\n";
  std::cout << "PHASE 4D.5B: PRODUCTION PREFETCH INTEGRATION REGRESSION TEST SUITE\n";
  std::cout << "======================================================================\n";

  bool ok = true;
  if (!test_1_prediction_engine_initialization()) ok = false;
  if (!test_2_candidate_dispatch_modes()) ok = false;
  if (!test_3_stage_deadline_expiration_sequence_loop()) ok = false;
  if (!test_4_current_stage_and_late_transfer()) ok = false;
  if (!test_5_exact_deadline_transfer()) ok = false;
  if (!test_6_multiple_target_stage_ownership()) ok = false;
  if (!test_7_demand_promotion_clears_pins()) ok = false;
  if (!test_8_cancellation_and_reset_cleanup()) ok = false;

  std::cout << "======================================================================\n";
  if (ok) {
    std::cout << "SUCCESS: All 8 Phase 4D.5B integration test cases PASSED!\n";
    std::cout << "======================================================================\n";
    return 0;
  } else {
    std::cerr << "FAILURE: One or more test cases FAILED!\n";
    std::cout << "======================================================================\n";
    return 1;
  }
}
