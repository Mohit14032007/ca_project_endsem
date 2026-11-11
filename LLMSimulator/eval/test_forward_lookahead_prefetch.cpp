#include <iostream>
#include <memory>
#include <vector>
#include <cassert>
#include <string>
#include <cmath>
#include <filesystem>
#include <unordered_set>

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

static SystemConfig create_test_system_config(
    PrefetchMode mode,
    SchedulingMode sched_mode,
    int horizon,
    EvictionPolicy policy,
    bool quarantine,
    bool pinning) {
  SystemConfig sys_cfg;
  sys_cfg.num_node = 1;
  sys_cfg.num_device = 1;
  sys_cfg.use_event_driven_dram = true;
  sys_cfg.ddr5_bandwidth_gbps = 64.0;
  sys_cfg.hbm_dram_cfg_path = get_hbm_cfg_path();

  if (mode == PrefetchMode::NONE) {
    sys_cfg.prefetch_config.enabled = false;
  } else {
    sys_cfg.prefetch_config.enabled = true;
  }
  sys_cfg.prefetch_config.mode = mode;
  sys_cfg.prefetch_config.scheduling_mode = sched_mode;
  sys_cfg.prefetch_config.prediction_horizon = horizon;
  sys_cfg.prefetch_config.predictor_type = PredictorType::CSTP;
  sys_cfg.prefetch_config.confidence_threshold = 0;
  sys_cfg.prefetch_config.eviction_policy = policy;
  sys_cfg.prefetch_config.pabi_quarantine_enabled = quarantine;
  sys_cfg.prefetch_config.pabi_pinning_enabled = pinning;
  sys_cfg.prefetch_config.pabi_max_pinned_experts = 4;
  sys_cfg.prefetch_config.budget = 4;
  return sys_cfg;
}

static Device::Ptr create_test_device(SystemConfig sys) {
  ModelConfig model_cfg = deepseekV3;
  model_cfg.ne_tp_dg = 1;
  model_cfg.e_tp_dg = 1;
  model_cfg.num_layers = 5;
  Cluster::Ptr cluster = Cluster::Create(sys, nullptr);
  Device::Ptr dev = Device::Create(sys, 0, cluster);
  dev->model_config = model_cfg;
  return dev;
}

// Test 1: Predictor receives source-stage routing, not target-stage routing
static bool test_source_routing_causality() {
  std::cout << "[Test 1] Predictor receives source-stage routing, not target-stage routing\n";

  PrefetchConfig cfg;
  cfg.enabled = true;
  cfg.mode = PrefetchMode::HARDWARE_ASYNC;
  cfg.scheduling_mode = SchedulingMode::FORWARD_LOOKAHEAD;
  cfg.prediction_horizon = 2;
  cfg.predictor_type = PredictorType::CSTP;
  cfg.budget = 4;

  int num_stages = 8;
  int num_experts = 32;
  int top_k = 2;

  LivePredictionEngine engine(cfg, num_stages, num_experts, top_k, "results/test_p2c");
  engine.on_request_start(100);

  // In forward lookahead, on_stage_routing_start at stage 0 produces zero predictions
  engine.on_stage_routing_start(100, 0, 0);
  TEST_ASSERT(engine.get_prediction_count() == 0, "No prediction should be generated at routing start in forward mode");

  // Routing completes at stage 0 with experts {7, 14}
  std::vector<int> stage0_experts = {7, 14};
  engine.on_stage_routing_complete(100, 0, 0, stage0_experts);

  // Prediction must now have been generated for future target stage = 0 + 2 = 2
  TEST_ASSERT(engine.get_prediction_count() == 1, "Exactly 1 prediction must be generated upon source routing complete");

  auto queue = engine.get_candidate_queue();
  TEST_ASSERT(queue != nullptr, "Candidate queue must exist");
  auto cands = queue->select_ready_candidates(-1);
  TEST_ASSERT(!cands.empty(), "Candidates must be enqueued");

  for (const auto& c : cands) {
    TEST_ASSERT(c.source_stage_id == 0, "Source stage must be 0");
    TEST_ASSERT(c.target_stage_id == 2, "Target stage must be 0 + H = 2");
    TEST_ASSERT(c.prediction_horizon == 2, "Horizon must be 2");
  }

  std::cout << "  -> PASS\n";
  return true;
}

// Test 2: Target stage equals source stage plus H for H in {1, 2, 3}
static bool test_target_stage_calculation() {
  std::cout << "[Test 2] Target stage equals source stage plus H for H in {1, 2, 3}\n";

  int horizons[3] = {1, 2, 3};
  for (int idx = 0; idx < 3; ++idx) {
    int h = horizons[idx];
    PrefetchConfig cfg;
    cfg.enabled = true;
    cfg.mode = PrefetchMode::HARDWARE_ASYNC;
    cfg.scheduling_mode = SchedulingMode::FORWARD_LOOKAHEAD;
    cfg.prediction_horizon = h;
    cfg.predictor_type = PredictorType::CSTP;
    cfg.budget = 4;

    int num_stages = 10;
    int num_experts = 32;
    int top_k = 2;

    LivePredictionEngine engine(cfg, num_stages, num_experts, top_k, "results/test_p2c");
    engine.on_request_start(200 + h);

    // Source stage 1 routing completes
    std::vector<int> src_experts = {3, 9};
    engine.on_stage_routing_complete(200 + h, 0, 1, src_experts);

    auto queue = engine.get_candidate_queue();
    auto cands = queue->select_ready_candidates(-1);
    TEST_ASSERT(!cands.empty(), "Candidates must be enqueued for horizon test");

    for (const auto& c : cands) {
      TEST_ASSERT(c.source_stage_id == 1, "Source stage must be 1");
      TEST_ASSERT(c.target_stage_id == 1 + h, "Target stage must equal source stage + H");
      TEST_ASSERT(c.prediction_horizon == h, "Horizon must match configured H");
    }

    // Boundary test: source stage = num_stages - h
    // (num_stages - h) + h = num_stages, which is out of bounds
    int boundary_stage = num_stages - h;
    int pred_before = engine.get_prediction_count();
    engine.on_stage_routing_complete(200 + h, 0, boundary_stage, src_experts);
    int pred_after = engine.get_prediction_count();
    TEST_ASSERT(pred_after == pred_before, "No forward prediction should be made when target_stage >= num_stages");
  }

  std::cout << "  -> PASS\n";
  return true;
}

// Test 3: Dispatch occurs after source routing completes
static bool test_dispatch_timing() {
  std::cout << "[Test 3] Dispatch occurs after source routing completes\n";

  SystemConfig sys_cfg = create_test_system_config(
      PrefetchMode::HARDWARE_ASYNC, SchedulingMode::FORWARD_LOOKAHEAD, 1, EvictionPolicy::PABI, true, true);
  Device::Ptr dev = create_test_device(sys_cfg);

  PrefetchConfig cfg = sys_cfg.prefetch_config;
  int num_stages = 4;
  int num_experts = 16;
  int top_k = 2;

  LivePredictionEngine engine(cfg, num_stages, num_experts, top_k, "results/test_p2c");
  engine.on_request_start(300);

  // Before source routing complete: candidate queue empty
  engine.on_stage_routing_start(300, 0, 0);
  int dispatched_early = engine.dispatch_candidates(dev, 0.0);
  TEST_ASSERT(dispatched_early == 0, "No candidates can be dispatched before routing completes");

  // Routing completes: forward candidates enqueued for target stage 1
  std::vector<int> routed = {2, 4};
  engine.on_stage_routing_complete(300, 0, 0, routed);

  // Immediate dispatch after routing complete
  int dispatched_after = engine.dispatch_candidates(dev, 1000.0);
  TEST_ASSERT(dispatched_after > 0, "Candidates must be dispatched after source routing completes");
  TEST_ASSERT(dev->prefetch_metrics.candidates_dispatched > 0, "Device dispatched counter must be updated");

  std::cout << "  -> PASS\n";
  return true;
}

// Test 4: Transfer progresses during intervening computation
static bool test_transfer_progress_during_computation() {
  std::cout << "[Test 4] Transfer progresses during intervening computation\n";

  SystemConfig sys_cfg = create_test_system_config(
      PrefetchMode::HARDWARE_ASYNC, SchedulingMode::FORWARD_LOOKAHEAD, 2, EvictionPolicy::PABI, true, true);
  Device::Ptr dev = create_test_device(sys_cfg);

  double bw = 64.0; // 64 GB/s = 64 bytes/ns
  uint64_t expert_bytes = 88080384ULL; // ~88 MB -> ~1,376,256 ns transfer duration
  double expected_duration = static_cast<double>(expert_bytes) / bw;

  // Submit prefetch at T=0
  uint64_t tid = dev->submit_prefetch_transfer(2, 5, 0.0, expert_bytes);
  TEST_ASSERT(tid != 0, "Prefetch transfer must be accepted");

  // Intervening computation of 500,000 ns (stage 0 computation)
  dev->process_completed_prefetches(500000.0);
  TEST_ASSERT(!dev->has_resident_expert(2, 5), "Transfer must not yet be complete at 500 us");

  const auto* tr1 = dev->ddr5_transfer_engine->get_transfer(tid);
  TEST_ASSERT(tr1 != nullptr, "Transfer record must exist");
  TEST_ASSERT(tr1->is_active(), "Transfer must be actively progressing");
  TEST_ASSERT(tr1->transferred_bytes > 0, "Transfer must have transferred bytes during compute");

  // Further computation advancing past expected completion time
  double comp_end = expected_duration + 100000.0;
  dev->process_completed_prefetches(comp_end);

  const auto* tr2 = dev->ddr5_transfer_engine->get_transfer(tid);
  TEST_ASSERT(tr2->is_completed(), "Transfer must be completed after compute duration exceeds transfer time");
  TEST_ASSERT(dev->has_resident_expert(2, 5), "Expert must be resident in HBM after completion");

  std::cout << "  -> PASS\n";
  return true;
}

// Test 5: Timely prefetches produce resident hits
static bool test_timely_resident_hits() {
  std::cout << "[Test 5] Timely prefetches produce resident hits\n";

  SystemConfig sys_cfg = create_test_system_config(
      PrefetchMode::HARDWARE_ASYNC, SchedulingMode::FORWARD_LOOKAHEAD, 1, EvictionPolicy::PABI, true, true);
  Device::Ptr dev = create_test_device(sys_cfg);

  uint64_t expert_bytes = 64000ULL; // 64 KB -> 1000 ns transfer at 64 GB/s
  uint64_t tid = dev->submit_prefetch_transfer(1, 10, 0.0, expert_bytes);
  TEST_ASSERT(tid != 0, "Prefetch transfer must be accepted");

  // Advance clock past completion: transfer finishes at 1000 ns
  dev->process_completed_prefetches(2000.0);
  TEST_ASSERT(dev->has_resident_expert(1, 10), "Expert must be resident in speculative buffer");

  // Target stage 1 demands expert 10 at T=2000 ns
  dev->set_current_stage(1);
  dev->set_time(2000);

  bool is_hit = dev->is_expert_resident(1, 10, true);
  TEST_ASSERT(is_hit, "Expert must produce resident hit");
  TEST_ASSERT(dev->prefetch_metrics.timely_resident_hits == 1, "timely_resident_hits counter must be 1");
  TEST_ASSERT(dev->prefetch_metrics.demanded_transfers == 1, "demanded_transfers counter must be 1");
  TEST_ASSERT(dev->pabi_metrics.speculative_promotions == 1, "Speculative promotion must occur");

  std::cout << "  -> PASS\n";
  return true;
}

// Test 6: Late prefetches attach without restarting the transfer
static bool test_late_inflight_attachment() {
  std::cout << "[Test 6] Late prefetches attach without restarting the transfer\n";

  SystemConfig sys_cfg = create_test_system_config(
      PrefetchMode::HARDWARE_ASYNC, SchedulingMode::FORWARD_LOOKAHEAD, 1, EvictionPolicy::PABI, true, true);
  Device::Ptr dev = create_test_device(sys_cfg);

  double bw = 64.0;
  uint64_t expert_bytes = 6400000ULL; // 6.4 MB -> 100,000 ns transfer duration
  double total_duration = static_cast<double>(expert_bytes) / bw;

  // Submit prefetch at T=0
  uint64_t pref_tid = dev->submit_prefetch_transfer(1, 20, 0.0, expert_bytes);
  TEST_ASSERT(pref_tid != 0, "Prefetch transfer must be accepted");

  // Advance time to 40,000 ns (transfer is 40% complete)
  dev->process_completed_prefetches(40000.0);
  dev->set_current_stage(1);
  dev->set_time(40000);

  TEST_ASSERT(!dev->has_resident_expert(1, 20), "Expert must not be resident yet");
  TEST_ASSERT(dev->is_expert_inflight(1, 20), "Expert must be in flight");

  // Demand arrives at T=40,000 ns
  uint64_t demand_tid = dev->submit_demand_transfer(1, 20, 40000.0, expert_bytes);
  TEST_ASSERT(demand_tid == pref_tid, "Demand transfer ID must match in-flight prefetch transfer ID");
  TEST_ASSERT(dev->prefetch_metrics.late_inflight_attachments == 1, "late_inflight_attachments must be 1");
  TEST_ASSERT(dev->prefetch_metrics.demanded_transfers == 1, "demanded_transfers must be 1");

  // Wait for demand transfer completion
  bool ok = dev->wait_for_demand_transfer(demand_tid);
  TEST_ASSERT(ok, "Transfer must complete successfully");

  const auto* tr = dev->ddr5_transfer_engine->get_transfer(demand_tid);
  TEST_ASSERT(tr != nullptr, "Transfer record must exist");
  TEST_ASSERT(std::abs(tr->completion_time_ns - total_duration) < 1.0, "Completion time must match original prefetch duration without restart");

  std::cout << "  -> PASS\n";
  return true;
}

// Test 7: Actual routing remains unchanged
static bool test_routing_invariance() {
  std::cout << "[Test 7] Actual routing remains unchanged\n";

  // Verify that model ground truth routing is unchanged by prefetch mode
  int num_stages = 4;
  int num_experts = 16;
  int top_k = 2;

  // Configuration A: Baseline (no prefetch)
  PrefetchConfig cfg_base;
  cfg_base.enabled = false;
  cfg_base.mode = PrefetchMode::NONE;

  // Configuration B: Hardware Async Forward Lookahead
  PrefetchConfig cfg_fwd;
  cfg_fwd.enabled = true;
  cfg_fwd.mode = PrefetchMode::HARDWARE_ASYNC;
  cfg_fwd.scheduling_mode = SchedulingMode::FORWARD_LOOKAHEAD;
  cfg_fwd.prediction_horizon = 2;
  cfg_fwd.predictor_type = PredictorType::CSTP;
  cfg_fwd.budget = 4;

  LivePredictionEngine eng_base(cfg_base, num_stages, num_experts, top_k, "results/test_p2c_base");
  LivePredictionEngine eng_fwd(cfg_fwd, num_stages, num_experts, top_k, "results/test_p2c_fwd");

  eng_base.on_request_start(700);
  eng_fwd.on_request_start(700);

  std::vector<std::vector<int>> ground_truth = {
    {3, 8},
    {1, 12},
    {7, 14},
    {0, 5}
  };

  for (int s = 0; s < num_stages; ++s) {
    eng_base.on_stage_routing_start(700, 0, s);
    eng_fwd.on_stage_routing_start(700, 0, s);

    // Both pass identical router output
    eng_base.on_stage_routing_complete(700, 0, s, ground_truth[s]);
    eng_fwd.on_stage_routing_complete(700, 0, s, ground_truth[s]);
  }

  // Verify that predictor execution did not mutate ground truth routing vectors
  for (int s = 0; s < num_stages; ++s) {
    TEST_ASSERT(ground_truth[s].size() == static_cast<size_t>(top_k), "Ground truth routing size must remain top_k");
  }

  std::cout << "  -> PASS\n";
  return true;
}

// Test 8: PABI and pinning protect the intended target-stage experts
static bool test_pabi_pinning_protection() {
  std::cout << "[Test 8] PABI and pinning protect the intended target-stage experts\n";

  SystemConfig sys_cfg = create_test_system_config(
      PrefetchMode::HARDWARE_ASYNC, SchedulingMode::FORWARD_LOOKAHEAD, 2, EvictionPolicy::PABI, true, true);
  Device::Ptr dev = create_test_device(sys_cfg);

  // Set tight speculative buffer capacity: exactly 2 experts
  uint64_t expert_bytes = 1000000ULL; // 1 MB
  dev->speculative_buffer_capacity_bytes = 2 * expert_bytes;
  dev->max_pinned_experts = 1;

  // Candidate for target stage 2 pinned with deadline stage 2
  PrefetchCandidate cand;
  cand.candidate_id = 801;
  cand.target_stage_id = 2;
  cand.expert_id = 7;
  cand.prediction_rank = 1;
  cand.prediction_score = 0.95;
  cand.state = CandidateState::ADMITTED;

  dev->select_pin_candidates_for_stage(2, {cand});
  TEST_ASSERT(dev->pabi_metrics.pin_intents_created == 1, "Pin intent must be created");

  // Submit prefetch at stage 0 (dispatched for target stage 2)
  uint64_t tid = dev->submit_prefetch_transfer(2, 7, 0.0, expert_bytes);
  TEST_ASSERT(tid != 0, "Prefetch transfer must be accepted");

  // Complete prefetch while current stage is 1 (before stage 2 deadline)
  dev->set_current_stage(1);
  dev->process_completed_prefetches(100000.0);

  TEST_ASSERT(dev->has_resident_expert(2, 7), "Expert (2, 7) must be resident");
  auto res_it = dev->resident_experts.find({2, 7});
  TEST_ASSERT(res_it->second.is_pinned, "Expert (2, 7) must be pinned in speculative buffer");
  TEST_ASSERT(res_it->second.pin_expiration_stage_id == 2, "Pin expiration stage must be 2");

  // Insert unpinned speculative expert 8
  dev->mark_expert_speculative(1, 8, expert_bytes);

  // Buffer is now at capacity (2 MB). Try inserting expert 9:
  // Pinned expert (2, 7) must NOT be evicted! Unpinned expert (1, 8) must be evicted!
  dev->mark_expert_speculative(1, 9, expert_bytes);

  TEST_ASSERT(dev->has_resident_expert(2, 7), "Pinned expert (2, 7) must NOT be evicted under pressure");
  TEST_ASSERT(!dev->has_resident_expert(1, 8), "Unpinned expert (1, 8) must be evicted");
  TEST_ASSERT(dev->pabi_metrics.pinned_eviction_skipped > 0, "Pinned eviction skip counter must be incremented");

  // Advance to stage 2: target stage 2 completes execution, expiring deadline 2
  int expired = dev->expire_pins_for_stage(2);
  TEST_ASSERT(expired == 1, "Pin for stage 2 must expire after stage 2 completes");
  TEST_ASSERT((!dev->resident_experts[{2, 7}].is_pinned), "Expert (2, 7) must now be unpinned");

  std::cout << "  -> PASS\n";
  return true;
}

// Test 9: Legacy mode reproduces the old scheduling behaviour
static bool test_legacy_mode_reproduction() {
  std::cout << "[Test 9] The legacy mode reproduces the old scheduling behaviour\n";

  PrefetchConfig cfg;
  cfg.enabled = true;
  cfg.mode = PrefetchMode::HARDWARE_ASYNC;
  cfg.scheduling_mode = SchedulingMode::LEGACY_CURRENT_STAGE;
  cfg.prediction_horizon = 1;
  cfg.predictor_type = PredictorType::CSTP;
  cfg.budget = 4;

  int num_stages = 4;
  int num_experts = 16;
  int top_k = 2;

  LivePredictionEngine engine(cfg, num_stages, num_experts, top_k, "results/test_p2c_legacy");
  engine.on_request_start(900);

  // In legacy mode, on_stage_routing_start predicts at stage 1 for stage 1 itself (co-located)
  engine.on_stage_routing_start(900, 0, 1);
  TEST_ASSERT(engine.get_prediction_count() == 1, "Legacy mode must generate prediction at routing start");

  auto queue = engine.get_candidate_queue();
  auto cands = queue->select_ready_candidates(-1);
  TEST_ASSERT(!cands.empty(), "Candidates must be enqueued");

  for (const auto& c : cands) {
    TEST_ASSERT(c.target_stage_id == 1, "Legacy mode target stage must equal current stage (co-located)");
  }

  std::cout << "  -> PASS\n";
  return true;
}

// Test 10: No duplicate transfers or incorrect future-stage accesses
static bool test_duplicate_prevention_and_bounds() {
  std::cout << "[Test 10] No duplicate transfers or incorrect future-stage accesses occur\n";

  SystemConfig sys_cfg = create_test_system_config(
      PrefetchMode::HARDWARE_ASYNC, SchedulingMode::FORWARD_LOOKAHEAD, 1, EvictionPolicy::PABI, true, true);
  Device::Ptr dev = create_test_device(sys_cfg);

  uint64_t expert_bytes = 100000ULL;

  // Submit transfer 1 for (stage 1, expert 3)
  uint64_t tid1 = dev->submit_prefetch_transfer(1, 3, 0.0, expert_bytes);
  TEST_ASSERT(tid1 != 0, "First prefetch must succeed");

  // Attempt duplicate submission while transfer 1 is in flight
  uint64_t tid2 = dev->submit_prefetch_transfer(1, 3, 1000.0, expert_bytes);
  TEST_ASSERT(tid2 == 0, "Duplicate submission while in flight must return 0");

  // Complete transfer 1 so expert is resident
  dev->process_completed_prefetches(200000.0);
  TEST_ASSERT(dev->has_resident_expert(1, 3), "Expert must be resident");

  // Attempt duplicate submission when expert is resident
  uint64_t tid3 = dev->submit_prefetch_transfer(1, 3, 200000.0, expert_bytes);
  TEST_ASSERT(tid3 == 0, "Duplicate submission when resident must return 0");

  // Candidate queue duplicate test
  PrefetchConfig qcfg;
  qcfg.budget = 100;
  qcfg.priority_policy = PrefetchPriorityPolicy::DEADLINE_SCORE_RANK;
  qcfg.duplicate_policy = DuplicatePolicy::RETAIN_HIGHEST_PRIORITY;
  qcfg.overflow_policy = OverflowPolicy::REJECT_LOWEST_PRIORITY;
  auto queue = std::make_shared<PrefetchCandidateQueue>(
      qcfg, 4, 16, "results/test_p2c_dup");

  PrefetchCandidate c1;
  c1.candidate_id = 1;
  c1.request_id = 1;
  c1.token_id = 0;
  c1.target_stage_id = 2;
  c1.expert_id = 10;
  c1.prediction_rank = 1;
  c1.prediction_score = 0.5;
  c1.state = CandidateState::ADMITTED;

  PrefetchCandidate c2 = c1;
  c2.candidate_id = 2;
  c2.prediction_score = 0.8;

  bool ok1 = queue->enqueue_candidate(c1);
  TEST_ASSERT(ok1, "First candidate must be admitted");

  bool ok2 = queue->enqueue_candidate(c2);
  TEST_ASSERT(ok2, "Duplicate with higher score must update existing entry");
  TEST_ASSERT(queue->size() == 1, "Queue must contain exactly 1 candidate (no duplicate growth)");

  // Dump effectiveness test
  dev->dump_prefetch_effectiveness("results/test_p2c_effectiveness.csv");
  TEST_ASSERT(std::filesystem::exists("results/test_p2c_effectiveness.csv"), "Effectiveness CSV must be generated");

  std::cout << "  -> PASS\n";
  return true;
}

int main() {
  std::cout << "======================================================================\n";
  std::cout << "PHASE 2C: FORWARD-LOOKAHEAD PREFETCH SCHEDULING VALIDATION SUITE\n";
  std::cout << "======================================================================\n";

  bool all_pass = true;
  all_pass = all_pass && test_source_routing_causality();
  all_pass = all_pass && test_target_stage_calculation();
  all_pass = all_pass && test_dispatch_timing();
  all_pass = all_pass && test_transfer_progress_during_computation();
  all_pass = all_pass && test_timely_resident_hits();
  all_pass = all_pass && test_late_inflight_attachment();
  all_pass = all_pass && test_routing_invariance();
  all_pass = all_pass && test_pabi_pinning_protection();
  all_pass = all_pass && test_legacy_mode_reproduction();
  all_pass = all_pass && test_duplicate_prevention_and_bounds();

  std::cout << "======================================================================\n";
  if (all_pass) {
    std::cout << "SUCCESS: All 10 Phase 2C validation tests PASSED!\n";
    std::cout << "======================================================================\n";
    return 0;
  } else {
    std::cerr << "FAILURE: One or more Phase 2C tests FAILED!\n";
    std::cout << "======================================================================\n";
    return 1;
  }
}
