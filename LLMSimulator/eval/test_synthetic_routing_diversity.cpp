#include <iostream>
#include <iomanip>
#include <vector>
#include <set>
#include <map>
#include <tuple>
#include <cassert>
#include <memory>
#include <cmath>
#include <cstdlib>

#include "scheduler/scheduler.h"
#include "scheduler/sequence.h"
#include "hardware/hardware_config.h"

using namespace llm_system;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "  [FAIL] " << __FUNCTION__ << " line " << __LINE__         \
                << ": " << (msg) << std::endl;                                 \
      tests_failed++;                                                          \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define RUN_TEST(fn)                                                           \
  do {                                                                         \
    std::cout << "[RUNNING] " << #fn << "...\n";                               \
    if (fn()) {                                                                \
      std::cout << "  [PASS] " << #fn << "\n";                                 \
      tests_passed++;                                                          \
    } else {                                                                   \
      std::cout << "  [FAIL] " << #fn << "\n";                                 \
      tests_failed++;                                                          \
    }                                                                          \
  } while (0)

// Helper to construct a clean test Scheduler
static Scheduler::Ptr create_test_scheduler(int num_experts, int top_k, double skewness) {
  SystemConfig sys_cfg = A100;
  sys_cfg.num_node = 1;
  sys_cfg.num_device = 1;

  // Must persist in memory because Scheduler stores ModelConfig by reference (ModelConfig&)
  static ModelConfig model_cfg;
  model_cfg = deepseekV3;
  model_cfg.num_routed_expert = num_experts;
  model_cfg.top_k = top_k;
  model_cfg.skewness = skewness;
  model_cfg.num_layers = 4;
  model_cfg.first_k_dense = 0;
  model_cfg.expert_freq = 1;
  model_cfg.ne_tp_dg = 1;
  model_cfg.e_tp_dg = 1;

  Scheduler::Ptr sched = Scheduler::Create(sys_cfg, model_cfg, "none", 4096, 4096 * 64, 8192 * 4);
  return sched;
}

// -----------------------------------------------------------------------------
// Test A: Different sequences must not share cache entries (N=256, K=8, B=8)
// -----------------------------------------------------------------------------
bool test_A_different_sequences_not_share_cache() {
  int num_experts = 256;
  int top_k = 8;
  int batch_size = 8;
  int target_layer = 0;
  Scheduler::Ptr scheduler = create_test_scheduler(num_experts, top_k, 0.0);

  BatchedSequence::Ptr batch = BatchedSequence::Create(num_experts, top_k, scheduler);
  batch->cur_layer = target_layer;

  for (int b = 0; b < batch_size; b++) {
    Sequence::Ptr seq = Sequence::Create(b, 128, 128);
    seq->get_expert_from_list = false;
    seq->current_len = 128;
    seq->num_process_token = 1;
    batch->add(seq);
  }

  batch->update_expert(num_experts, top_k, true);

  // The cache must hold exactly 8 distinct entries (one per sequence)
  TEST_ASSERT(batch->cached_routing.size() == (size_t)batch_size,
              "Cache must contain exactly batch_size entries for distinct sequences");

  std::set<int> unique_experts_union;
  std::vector<std::vector<int>> sequence_routes;

  for (auto seq : batch->get_seq()) {
    auto key = std::make_tuple(seq->id, seq->expert_seq_id, seq->current_len, target_layer);
    auto it = batch->cached_routing.find(key);
    TEST_ASSERT(it != batch->cached_routing.end(), "Each sequence must have a cached entry");
    TEST_ASSERT((int)it->second.size() == top_k, "Top-K size must match top_k");

    std::set<int> seq_unique(it->second.begin(), it->second.end());
    TEST_ASSERT((int)seq_unique.size() == top_k, "Top-K selection must have no internal duplicates");

    for (int exp_id : it->second) {
      unique_experts_union.insert(exp_id);
    }
    sequence_routes.push_back(it->second);
  }

  // Under the old bug, unique_experts_union was strictly top_k (8)
  // With independent routing, union across 8 sequences must be strictly greater than 8
  TEST_ASSERT((int)unique_experts_union.size() > top_k,
              "Union of selected experts across sequences must be strictly greater than K");

  // Verify that not all sequences have identical Top-K sets
  bool found_difference = false;
  for (size_t i = 1; i < sequence_routes.size(); i++) {
    if (sequence_routes[i] != sequence_routes[0]) {
      found_difference = true;
      break;
    }
  }
  TEST_ASSERT(found_difference, "Different sequences must not have identical routing selections");

  return true;
}

// -----------------------------------------------------------------------------
// Test B: Repeated access must be deterministic
// -----------------------------------------------------------------------------
bool test_B_repeated_access_deterministic() {
  int num_experts = 256;
  int top_k = 8;
  int target_layer = 1;
  Scheduler::Ptr scheduler = create_test_scheduler(num_experts, top_k, 0.0);

  BatchedSequence::Ptr batch = BatchedSequence::Create(num_experts, top_k, scheduler);
  batch->cur_layer = target_layer;

  Sequence::Ptr seq = Sequence::Create(0, 64, 64);
  seq->get_expert_from_list = false;
  seq->current_len = 64;
  seq->num_process_token = 1;
  batch->add(seq);

  // Pass 1
  batch->update_expert(num_experts, top_k, true);
  auto key = std::make_tuple(seq->id, seq->expert_seq_id, seq->current_len, target_layer);
  TEST_ASSERT(batch->cached_routing.count(key) == 1, "Cache must contain key after Pass 1");
  std::vector<int> pass1_selection = batch->cached_routing[key];

  // Pass 2: Replay exact same state (reset cur_layer cursor)
  batch->cur_layer = target_layer;
  batch->update_expert(num_experts, top_k, true);
  TEST_ASSERT(batch->cached_routing.size() == 1, "Cache size must not grow on re-execution");
  std::vector<int> pass2_selection = batch->cached_routing[key];

  TEST_ASSERT(pass1_selection == pass2_selection, "Pass 2 must return identical expert selections to Pass 1");

  return true;
}

// -----------------------------------------------------------------------------
// Test C: Different token positions must remain distinct
// -----------------------------------------------------------------------------
bool test_C_different_token_positions_distinct() {
  int num_experts = 256;
  int top_k = 8;
  int target_layer = 0;
  Scheduler::Ptr scheduler = create_test_scheduler(num_experts, top_k, 0.0);

  BatchedSequence::Ptr batch = BatchedSequence::Create(num_experts, top_k, scheduler);

  Sequence::Ptr seq = Sequence::Create(0, 10, 10);
  seq->get_expert_from_list = false;
  batch->add(seq);

  // Token position 10
  batch->cur_layer = target_layer;
  seq->current_len = 10;
  seq->num_process_token = 1;
  batch->update_expert(num_experts, top_k, true);

  // Token position 11
  batch->cur_layer = target_layer;
  seq->current_len = 11;
  seq->num_process_token = 1;
  batch->update_expert(num_experts, top_k, true);

  auto key10 = std::make_tuple(seq->id, seq->expert_seq_id, 10, target_layer);
  auto key11 = std::make_tuple(seq->id, seq->expert_seq_id, 11, target_layer);

  TEST_ASSERT(batch->cached_routing.count(key10) == 1, "Key for token 10 must exist");
  TEST_ASSERT(batch->cached_routing.count(key11) == 1, "Key for token 11 must exist");
  TEST_ASSERT(batch->cached_routing.size() == 2, "Cache size must be 2 for two distinct token positions");

  return true;
}

// -----------------------------------------------------------------------------
// Test D: Different layers must remain distinct
// -----------------------------------------------------------------------------
bool test_D_different_layers_distinct() {
  int num_experts = 256;
  int top_k = 8;
  Scheduler::Ptr scheduler = create_test_scheduler(num_experts, top_k, 0.0);

  BatchedSequence::Ptr batch = BatchedSequence::Create(num_experts, top_k, scheduler);

  Sequence::Ptr seq = Sequence::Create(0, 20, 20);
  seq->get_expert_from_list = false;
  seq->current_len = 20;
  seq->num_process_token = 1;
  batch->add(seq);

  // Layer 0
  batch->cur_layer = 0;
  batch->update_expert(num_experts, top_k, true);

  // Layer 1
  batch->cur_layer = 1;
  batch->update_expert(num_experts, top_k, true);

  auto key_l0 = std::make_tuple(seq->id, seq->expert_seq_id, 20, 0);
  auto key_l1 = std::make_tuple(seq->id, seq->expert_seq_id, 20, 1);

  TEST_ASSERT(batch->cached_routing.count(key_l0) == 1, "Key for layer 0 must exist");
  TEST_ASSERT(batch->cached_routing.count(key_l1) == 1, "Key for layer 1 must exist");
  TEST_ASSERT(batch->cached_routing.size() == 2, "Cache size must be 2 for two distinct layers");

  return true;
}

// -----------------------------------------------------------------------------
// Test E: Sequence/request identity lifecycle (reused sequence IDs across requests)
// -----------------------------------------------------------------------------
bool test_E_request_identity_lifecycle() {
  int num_experts = 256;
  int top_k = 8;
  int target_layer = 0;
  Scheduler::Ptr scheduler = create_test_scheduler(num_experts, top_k, 0.0);

  BatchedSequence::Ptr batch = BatchedSequence::Create(num_experts, top_k, scheduler);

  // Request 1: uses slot 0 (expert_seq_id = 0)
  batch->cur_layer = target_layer;
  Sequence::Ptr seq1 = Sequence::Create(0, 50, 50);
  seq1->get_expert_from_list = false;
  seq1->current_len = 50;
  seq1->num_process_token = 1;
  batch->add(seq1);
  batch->update_expert(num_experts, top_k, true);

  int req1_id = seq1->id;
  auto key1 = std::make_tuple(req1_id, 0, 50, target_layer);
  TEST_ASSERT(batch->cached_routing.count(key1) == 1, "Request 1 key must be present in cache");

  // Request 1 completes and is removed from the active batch
  batch->pop(seq1);

  // Request 2 arrives: also assigned slot 0 (expert_seq_id = 0), at same token 50 and layer 0
  batch->cur_layer = target_layer;
  Sequence::Ptr seq2 = Sequence::Create(0, 50, 50);
  seq2->get_expert_from_list = false;
  seq2->current_len = 50;
  seq2->num_process_token = 1;
  batch->add(seq2);
  batch->update_expert(num_experts, top_k, true);

  int req2_id = seq2->id;
  TEST_ASSERT(req2_id != req1_id, "New sequence must have distinct request ID");

  auto key2 = std::make_tuple(req2_id, 0, 50, target_layer);
  TEST_ASSERT(batch->cached_routing.count(key2) == 1, "Request 2 key must be present in cache");
  TEST_ASSERT(batch->cached_routing.size() == 2, "Cache must contain both requests without colliding");

  // Also verify clear_cached_routing() method
  batch->clear_cached_routing();
  TEST_ASSERT(batch->cached_routing.empty(), "clear_cached_routing() must clear cache entries");

  return true;
}

// -----------------------------------------------------------------------------
// Test F: Batch size one regression
// -----------------------------------------------------------------------------
bool test_F_batch_size_one_regression() {
  int num_experts = 256;
  int top_k = 8;
  int target_layer = 2;
  Scheduler::Ptr scheduler = create_test_scheduler(num_experts, top_k, 0.0);

  BatchedSequence::Ptr batch = BatchedSequence::Create(num_experts, top_k, scheduler);
  batch->cur_layer = target_layer;

  Sequence::Ptr seq = Sequence::Create(0, 100, 100);
  seq->get_expert_from_list = false;
  seq->current_len = 100;
  seq->num_process_token = 1;
  batch->add(seq);

  batch->update_expert(num_experts, top_k, true);

  TEST_ASSERT(batch->cached_routing.size() == 1, "Cache size must be exactly 1 for B=1");
  auto key = std::make_tuple(seq->id, seq->expert_seq_id, seq->current_len, target_layer);
  TEST_ASSERT(batch->cached_routing.count(key) == 1, "Key must exist in cache for B=1");

  const auto& route = batch->cached_routing[key];
  TEST_ASSERT((int)route.size() == top_k, "Top-K must be 8 for B=1");

  return true;
}

// -----------------------------------------------------------------------------
// Test G: Real-trace regression (get_expert_from_list = true bypasses cache)
// -----------------------------------------------------------------------------
bool test_G_real_trace_bypasses_cache() {
  int num_experts = 256;
  int top_k = 8;
  int target_layer = 0;
  Scheduler::Ptr scheduler = create_test_scheduler(num_experts, top_k, 0.0);

  // Setup mock sequences_info in scheduler
  scheduler->sequences_info.resize(1);
  scheduler->sequences_info[0] = SequenceInfo::Ptr(new SequenceInfo);
  scheduler->sequences_info[0]->Lin = 10;
  scheduler->sequences_info[0]->Lout = 10;
  scheduler->sequences_info[0]->cur = 0;
  // Size = tokens * layers = 20 * 4 = 80
  scheduler->sequences_info[0]->expert_list.resize(80);

  // Put known experts at table_idx (rel_token = 0, layer = 0 -> table_idx = 0)
  std::vector<int> expected_real_experts = {10, 20, 30, 40, 50, 60, 70, 80};
  scheduler->sequences_info[0]->expert_list[0] = expected_real_experts;

  BatchedSequence::Ptr batch = BatchedSequence::Create(num_experts, top_k, scheduler);
  batch->cur_layer = target_layer;

  Sequence::Ptr seq = Sequence::Create(0, 10, 10);
  seq->get_expert_from_list = true; // REAL TRACE FLAG
  seq->current_len = 10;
  seq->num_process_token = 1;
  batch->add(seq);

  batch->update_expert(num_experts, top_k, true);

  // Real trace replay must NOT touch or populate cached_routing
  TEST_ASSERT(batch->cached_routing.empty(), "Real trace replay must bypass cached_routing completely");

  // Verify that the tokens routed to each expert match expected_real_experts
  for (int exp_id : expected_real_experts) {
    TEST_ASSERT(batch->num_token_in_expert[exp_id] == 1,
                "Real trace expert must have token count incremented");
  }

  return true;
}

// -----------------------------------------------------------------------------
// Test H: Top-K and range invariants (uniform and Zipfian)
// -----------------------------------------------------------------------------
bool test_H_topk_and_range_invariants() {
  int num_experts = 256;
  int top_k = 8;

  // Test uniform (skewness = 0.0) and Zipfian (skewness = 1.2)
  double test_skews[] = {0.0, 1.2};
  for (int s_idx = 0; s_idx < 2; s_idx++) {
    double skew = test_skews[s_idx];
    Scheduler::Ptr scheduler = create_test_scheduler(num_experts, top_k, skew);
    BatchedSequence::Ptr batch = BatchedSequence::Create(num_experts, top_k, scheduler);

    for (int step = 0; step < 5; step++) {
      int target_layer = step % 4;
      batch->cur_layer = target_layer;
      Sequence::Ptr seq = Sequence::Create(step, 100 + step, 100);
      seq->get_expert_from_list = false;
      seq->current_len = 100 + step;
      seq->num_process_token = 1;
      batch->add(seq);
      batch->update_expert(num_experts, top_k, true);

      auto key = std::make_tuple(seq->id, seq->expert_seq_id, seq->current_len, target_layer);
      TEST_ASSERT(batch->cached_routing.count(key) == 1, "Key must exist in cache");
      const auto& route = batch->cached_routing[key];

      TEST_ASSERT((int)route.size() == top_k, "Route size must equal top_k");
      std::set<int> unique_check;
      for (int exp_id : route) {
        TEST_ASSERT(exp_id >= 0 && exp_id < num_experts, "Expert ID must be in valid range [0, N-1]");
        unique_check.insert(exp_id);
      }
      TEST_ASSERT((int)unique_check.size() == top_k, "No duplicate experts within a single Top-K selection");
    }
  }

  return true;
}

// -----------------------------------------------------------------------------
// Section 5: Batch-diversity diagnostic (B = 1, 2, 4, 8, 16, 32)
// -----------------------------------------------------------------------------
void run_batch_diversity_diagnostic() {
  int num_experts = 256;
  int top_k = 8;
  int batch_sizes[] = {1, 2, 4, 8, 16, 32};
  int num_test_sizes = 6;
  int target_layer = 0;

  std::cout << "\n========================================================================================\n";
  std::cout << "BATCH DIVERSITY DIAGNOSTIC (N = 256, K = 8, Uniform Synthetic Routing)\n";
  std::cout << "========================================================================================\n";
  std::cout << std::setw(6)  << "B"
            << std::setw(18) << "Measured Unique"
            << std::setw(22) << "Expected Theoretical"
            << std::setw(22) << "Mean Pairwise Jaccard"
            << std::setw(22) << "Identical Set Pairs\n";
  std::cout << "----------------------------------------------------------------------------------------\n";

  for (int idx = 0; idx < num_test_sizes; idx++) {
    int B = batch_sizes[idx];
    Scheduler::Ptr scheduler = create_test_scheduler(num_experts, top_k, 0.0);
    BatchedSequence::Ptr batch = BatchedSequence::Create(num_experts, top_k, scheduler);
    batch->cur_layer = target_layer;

    for (int b = 0; b < B; b++) {
      Sequence::Ptr seq = Sequence::Create(b, 128, 128);
      seq->get_expert_from_list = false;
      seq->current_len = 128;
      seq->num_process_token = 1;
      batch->add(seq);
    }

    batch->update_expert(num_experts, top_k, true);

    std::set<int> union_experts;
    std::vector<std::set<int>> seq_sets;

    for (auto seq : batch->get_seq()) {
      auto key = std::make_tuple(seq->id, seq->expert_seq_id, seq->current_len, target_layer);
      std::set<int> s(batch->cached_routing[key].begin(), batch->cached_routing[key].end());
      seq_sets.push_back(s);
      for (int exp_id : s) {
        union_experts.insert(exp_id);
      }
    }

    // Expected value: N * (1 - (1 - K/N)^B)
    double p_not = std::pow(1.0 - (double)top_k / (double)num_experts, B);
    double expected_unique = (double)num_experts * (1.0 - p_not);

    // Pairwise overlap and identical set count
    double total_jaccard = 0.0;
    int pair_count = 0;
    int identical_pairs = 0;

    for (int i = 0; i < B; i++) {
      for (int j = i + 1; j < B; j++) {
        std::set<int> intersect_set;
        std::set<int> union_pair = seq_sets[i];
        for (int exp_id : seq_sets[j]) {
          union_pair.insert(exp_id);
          if (seq_sets[i].count(exp_id) > 0) {
            intersect_set.insert(exp_id);
          }
        }
        if (seq_sets[i] == seq_sets[j]) {
          identical_pairs++;
        }
        double jaccard = (double)intersect_set.size() / (double)union_pair.size();
        total_jaccard += jaccard;
        pair_count++;
      }
    }

    double mean_jaccard = 0.0;
    if (pair_count > 0) {
      mean_jaccard = total_jaccard / (double)pair_count;
    }

    std::cout << std::setw(6)  << B
              << std::setw(18) << union_experts.size()
              << std::setw(22) << std::fixed << std::setprecision(2) << expected_unique
              << std::setw(22) << std::fixed << std::setprecision(4) << mean_jaccard
              << std::setw(22) << identical_pairs << "\n";
  }
  std::cout << "========================================================================================\n\n";
}

// -----------------------------------------------------------------------------
// Main Test Runner
// -----------------------------------------------------------------------------
int main() {
  // Direct test output to temporary directory
  setenv("TRACE_DIR", "/tmp", 1);

  std::cout << "======================================================================\n";
  std::cout << "SYNTHETIC MULTI-SEQUENCE ROUTING DIVERSITY REGRESSION TEST SUITE\n";
  std::cout << "======================================================================\n\n";

  RUN_TEST(test_A_different_sequences_not_share_cache);
  RUN_TEST(test_B_repeated_access_deterministic);
  RUN_TEST(test_C_different_token_positions_distinct);
  RUN_TEST(test_D_different_layers_distinct);
  RUN_TEST(test_E_request_identity_lifecycle);
  RUN_TEST(test_F_batch_size_one_regression);
  RUN_TEST(test_G_real_trace_bypasses_cache);
  RUN_TEST(test_H_topk_and_range_invariants);

  run_batch_diversity_diagnostic();

  std::cout << "Test Summary: " << tests_passed << " passed, " << tests_failed << " failed.\n";
  if (tests_failed == 0) {
    std::cout << "[SUCCESS] All synthetic routing diversity regression tests passed!\n";
    return 0;
  } else {
    std::cerr << "[FAILURE] One or more tests failed!\n";
    return 1;
  }
}
