# Implementation Report: Synthetic Multi-Sequence Routing Collision Fix

**Date:** November 2026  
**Auditor/Engineer:** Antigravity Autonomous Coding & Forensic Engine  
**Target Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Deliverable Document:** `mohman/point_2/SYNTHETIC_ROUTING_FIX_REPORT.md`  
**Final Status:** **FIXED**

---

## 1. Original Bug Explanation

In mixture-of-experts (MoE) inference simulation, synthetic routing mode generates Top-$K$ expert selections on the fly when real trace files are not provided. To ensure deterministic multi-pass replay (e.g. running Pass 1 for cold-start memory fetching and Pass 2 for warm-cache hit verification), [`BatchedSequence`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.h#L79-L135) implemented a routing cache named `cached_routing`.

However, the cache key was declared as:
```cpp
std::map<std::pair<int, int>, std::vector<int>> cached_routing;
```
and evaluated in [`BatchedSequence::update_expert()`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L269-L286) as:
```cpp
auto route_key = std::make_pair(token_id, cur_layer);
auto it = cached_routing.find(route_key);
```

When multiple sequences in a batch ($B > 1$) process the same token position (e.g. `token_id = 128`) at stage `cur_layer = 0`:
1. **Sequence 0** evaluates `(128, 0)`, misses in `cached_routing`, draws $K$ random experts (e.g. `{15, 39, 51, 63, 77, 117, 162, 165}`), and stores the result in `cached_routing[(128, 0)]`.
2. **Sequences $1, 2, \dots, B-1$** evaluate `(128, 0)`, hit `cached_routing`, and immediately receive the identical expert vector populated by Sequence 0.

As a result, all $B$ sequences in the batch were forced to share the identical Top-$K$ expert set. The unique expert count across the entire batch was artificially locked at $K$ (e.g. 8 for DeepSeek-V3), underestimating the true multi-sequence working-set size by up to **$20.4\times$ at $B=32$** and artificially creating 100% inter-sequence expert sharing.

---

## 2. Root Cause Analysis

The root cause was the omission of sequence identity from the cache key in both the container declaration ([`src/scheduler/sequence.h`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.h)) and lookup logic ([`src/scheduler/sequence.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp)).

The keying structure assumed that `(token_id, cur_layer)` was globally unique within a batch execution step. Because a `BatchedSequence` aggregates multiple concurrent `Sequence` instances that advance in lockstep through identical decoding token positions and model layers, `(token_id, cur_layer)` is shared across all sequences in the batch.

---

## 3. Exact Files Modified

| File Path | Description of Modification |
| :--- | :--- |
| [`LLMSimulator/src/scheduler/sequence.h`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.h#L13) | Added `#include <tuple>`. Defined `using RouteKey = std::tuple<int, int, int, int>;` (`request_id`, `seq_id`, `token_id`, `cur_layer`). Replaced `std::map<std::pair<int, int>, ...>` with `std::map<RouteKey, ...>`. Added `clear_cached_routing()`. |
| [`LLMSimulator/src/scheduler/sequence.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L270) | Replaced `auto route_key = std::make_pair(token_id, cur_layer);` with `auto route_key = std::make_tuple(request_id, seq_id, token_id, cur_layer);`. |
| [`LLMSimulator/eval/CMakeLists.txt`](file:///home/mohman/ca_project/LLMSimulator/eval/CMakeLists.txt#L106-L111) | Added build target `test_synthetic_routing_diversity`. |
| [`LLMSimulator/eval/test_synthetic_routing_diversity.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test_synthetic_routing_diversity.cpp) | Created automated regression test suite covering Tests A through H and multi-sequence diversity diagnostic ($B \in \{1, 2, 4, 8, 16, 32\}$). |

---

## 4. Before/After Cache-Key Design

### Before (Buggy)
- **Container Type:** `std::map<std::pair<int, int>, std::vector<int>> cached_routing`
- **Key Construction:** `auto route_key = std::make_pair(token_id, cur_layer);`
- **Scope:** Keyed strictly by token offset and layer index. Sequence identity was omitted.

### After (Corrected)
- **Key Alias:** `using RouteKey = std::tuple<int, int, int, int>;`
- **Container Type:** `std::map<RouteKey, std::vector<int>> cached_routing`
- **Key Construction:** `auto route_key = std::make_tuple(request_id, seq_id, token_id, cur_layer);`
- **Scope:** Uniquely scopes each routing decision by:
  1. `request_id` (`seq->id`): Globally unique request/sequence ID from monotonically incrementing `Sequence::count_id`.
  2. `seq_id` (`seq->expert_seq_id`): Sequence slot index within the scheduler queue and batch.
  3. `token_id`: Chronological token position within the sequence.
  4. `cur_layer`: Logical MoE layer / stage index.

---

## 5. Cache-Lifetime and Identity Reasoning

During forensic source code inspection, the lifecycles of `Sequence`, `BatchedSequence`, and `cached_routing` were established:

1. **`Sequence::id` Global Uniqueness:**
   In [`src/scheduler/sequence.cpp:12-13`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L12-L13):
   ```cpp
   id = count_id;
   count_id++;
   ```
   Every `Sequence` object instantiated in the process receives a monotonically incrementing `id`. Even when dummy sequences are created with `expert_seq_id = 0`, their `seq->id` values are distinct ($0, 1, 2, \dots$).
2. **`expert_seq_id` Purpose:**
   `seq->expert_seq_id` represents the sequence slot index within `sequences_info`. In real-trace replay, sequences $0 \dots B-1$ are assigned `expert_seq_id = 0 \dots B-1`.
3. **Cross-Request Slot Reuse (Lifecycle Safety):**
   When Request 1 finishes, it is popped from `BatchedSequence`. When Request 2 arrives later, it may be assigned slot `expert_seq_id = 0`. Because `request_id` is present in the cache key, Request 2 (`request_id = 101, seq_id = 0`) cannot collide with Request 1 (`request_id = 1, seq_id = 0`), even if `cached_routing` remains alive in the batch container.
4. **Replay Determinism:**
   When the exact same sequence is replayed in Pass 2 (e.g. `test_deepseekv3_full_baseline.cpp:157-161`), the sequence retains its original `id`, `expert_seq_id`, `token_id`, and layer. The cache key matches exactly, retrieving the identical expert selection without drawing new random numbers.

---

## 6. RNG Behavior and Handling

1. **RNG Architecture:**
   In [`src/scheduler/scheduler.cpp:181-217`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L181-L217):
   - Engine: `static std::mt19937 generator(seed);` with `static unsigned int seed = 777;`.
   - The RNG generator is function-static, persisting across all calls.
   - It is shared across sequences, ensuring that successive sequences draw successive pseudo-random values.
2. **Reproducibility:**
   Because the generator is not reset per sequence, Sequence 0 consumes random draws $1 \dots D_0$, and Sequence 1 consumes subsequent draws $D_0 + 1 \dots D_1$. Once `cached_routing` was decoupled by sequence identity, different sequences naturally received distinct, independent random draws while maintaining exact determinism across identical runs from process launch.
3. **RNG Scope Preservation:**
   In accordance with scope constraints, the hardcoded seed 777 and shared `generator` were left intact without introducing unnecessary configuration complexity.

---

## 7. Regression Tests Added

A dedicated regression test suite was implemented in [`LLMSimulator/eval/test_synthetic_routing_diversity.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test_synthetic_routing_diversity.cpp) covering all required test specifications:

- **Test A: Different Sequences Must Not Share Cache Entries:**  
  Configured $N=256, K=8, B=8$ with uniform routing under seed 777. Verified all sequences receive exactly 8 unique experts, the cache contains exactly 8 distinct entries, the union of selected experts across sequences is strictly greater than $K$ (measured 54 experts), and no two sequences have identical routing sets.
- **Test B: Repeated Access Must Be Deterministic:**  
  Executed Pass 1 and Pass 2 on the exact same routing identity. Verified cache size does not grow (size remains 1) and Pass 2 returns the identical expert vector as Pass 1.
- **Test C: Different Token Positions Must Remain Distinct:**  
  Processed token 10 and token 11 for the same sequence. Verified cache contains 2 distinct entries with independent routing.
- **Test D: Different Layers Must Remain Distinct:**  
  Executed layer 0 and layer 1 for the same sequence and token. Verified layer identity is preserved across distinct cache entries.
- **Test E: Sequence / Request Identity Lifecycle:**  
  Simulated Request 1 (`id = req1, seq_id = 0`) completion, followed by Request 2 (`id = req2, seq_id = 0`) arriving at the same slot, token position, and layer. Verified Request 2 does not collide with Request 1, producing 2 distinct cache entries. Verified `clear_cached_routing()` flushes cache entries.
- **Test F: Batch Size One Regression ($B=1$):**  
  Verified that single-sequence execution operates identically, creates 1 cache entry, and returns valid Top-8 experts.
- **Test G: Real-Trace Regression:**  
  Configured mock `sequences_info` and set `seq->get_expert_from_list = true`. Verified that real trace execution routes to recorded experts and completely bypasses `cached_routing` (cache remains empty).
- **Test H: Top-K and Range Invariants:**  
  Tested both uniform (`skewness = 0.0`) and Zipfian (`skewness = 1.2`) synthetic routing across multiple tokens and stages. Verified all selections have size exactly $K$, zero internal duplicates, and all IDs satisfy $0 \le \text{expert\_id} < N$.

---

## 8. Build Results

The test suite and modified library compiled cleanly with CMake and Make:

```bash
cd LLMSimulator/build && make -j test_synthetic_routing_diversity
```

**Build Output:**
```
[ 98%] Built target llm_system
[ 98%] Building CXX object eval/CMakeFiles/test_synthetic_routing_diversity.dir/test_synthetic_routing_diversity.cpp.o
[100%] Linking CXX executable ../test_synthetic_routing_diversity
[100%] Built target test_synthetic_routing_diversity
```

---

## 9. Test Results

Execution of [`./build/test_synthetic_routing_diversity`](file:///home/mohman/ca_project/LLMSimulator/build/test_synthetic_routing_diversity):

```
======================================================================
SYNTHETIC MULTI-SEQUENCE ROUTING DIVERSITY REGRESSION TEST SUITE
======================================================================

[RUNNING] test_A_different_sequences_not_share_cache...
Using synthesis trace
  [PASS] test_A_different_sequences_not_share_cache
[RUNNING] test_B_repeated_access_deterministic...
Using synthesis trace
  [PASS] test_B_repeated_access_deterministic
[RUNNING] test_C_different_token_positions_distinct...
Using synthesis trace
  [PASS] test_C_different_token_positions_distinct
[RUNNING] test_D_different_layers_distinct...
Using synthesis trace
  [PASS] test_D_different_layers_distinct
[RUNNING] test_E_request_identity_lifecycle...
Using synthesis trace
  [PASS] test_E_request_identity_lifecycle
[RUNNING] test_F_batch_size_one_regression...
Using synthesis trace
  [PASS] test_F_batch_size_one_regression
[RUNNING] test_G_real_trace_bypasses_cache...
Using synthesis trace
  [PASS] test_G_real_trace_bypasses_cache
[RUNNING] test_H_topk_and_range_invariants...
Using synthesis trace
Using synthesis trace
  [PASS] test_H_topk_and_range_invariants

Test Summary: 8 passed, 0 failed.
[SUCCESS] All synthetic routing diversity regression tests passed!
```

---

## 10. Batch-Diversity Measurements

The automated diagnostic evaluated batches $B \in \{1, 2, 4, 8, 16, 32\}$ under $N=256, K=8$, uniform routing, with identical token positions and layer cursors across sequences:

$$\mathbb{E}[U(B)] = N \cdot \left[1 - \left(1 - \frac{K}{N}\right)^B\right]$$

### Diagnostic Output Table

| Batch Size ($B$) | Buggy Synthetic Working Set | Fixed Measured Unique Experts | Expected Theoretical $\mathbb{E}[U(B)]$ | Mean Pairwise Jaccard Overlap | Identical Top-$K$ Set Pairs |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 8 | **8** | 8.00 | 0.0000 | **0** |
| **2** | 8 | **16** | 15.75 | 0.0000 | **0** |
| **4** | 8 | **31** | 30.53 | 0.0111 | **0** |
| **8** | 8 | **54** | 57.42 | 0.0262 | **0** |
| **16** | 8 | **110** | 101.96 | 0.0130 | **0** |
| **32** | 8 | **160** | 163.31 | 0.0171 | **0** |

### Key Observations
1. **Working Set Growth:** Unique active experts scale smoothly from 8 ($B=1$) to 160 ($B=32$), matching theoretical expectations closely.
2. **Zero Collision Artifacts:** Across all evaluated batch sizes, `Identical Set Pairs` is strictly **0**. Under the buggy implementation, identical set pairs was $\binom{B}{2}$ (e.g. 28 for $B=8$, 496 for $B=32$).
3. **Low Natural Overlap:** Mean pairwise Jaccard overlap remains under $0.03$ across all batch sizes, reflecting genuine independent random sampling from a 256-expert pool.

---

## 11. Confirmation of Real-Trace Replay Integrity

The real-trace ingestion path was verified to ensure zero regression:
1. In [`src/scheduler/sequence.cpp:244-267`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L244-L267), when `seq->get_expert_from_list == true`, execution branches directly into:
   ```cpp
   const auto& routed = scheduler->sequences_info[seq_id]->expert_list[table_idx];
   ```
2. In this branch, `cached_routing` is never accessed or written.
3. Automated Test G explicitly verified that when `get_expert_from_list = true`, `cached_routing.size()` remains 0, and the exact preloaded trace expert IDs are dispatched.
4. Real-trace ingestion scripts and the C++ loader ([`Scheduler::loadRealTrace()`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L650-L735)) were not modified.

---

## 12. Remaining Limitations and Follow-Up Issues

The following non-blocking issues identified during the forensic audit were outside the scope of this bug fix and remain recorded as follow-up tasks:

1. **Static Zipfian Distribution Initialization:**
   In [`src/scheduler/scheduler.cpp:208`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L208), `static std::discrete_distribution<int> distribution(...)` is declared function-static. If skewness weights are modified between runs within the same process, the distribution does not update dynamically.
2. **Sorted Top-$K$ Ranking in Synthetic Traces:**
   In [`src/scheduler/scheduler.cpp:186, 210`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L186), synthetic routing inserts expert IDs into `std::set<int>`, which sorts IDs in ascending numerical order. Consequently, `topk_rank` (1..8) in synthetic traces reflects numerical order rather than router logit ranking.
3. **Hardcoded PRNG Seed:**
   PRNG seed 777 is hardcoded in C++ functions rather than configurable via YAML simulation parameters.

---

## 13. Final Acceptance Verdict

- [x] Different sequences no longer collide on synthetic routing cache entries.
- [x] Repeated access to the same routing identity remains 100% deterministic.
- [x] Reused sequence IDs across requests cannot cause stale cache collisions.
- [x] Batch diversity is no longer artificially fixed at $K$ (empirically scales from 8 to 160 experts).
- [x] Top-$K$ uniqueness and expert ID range invariants ($0 \le e < N$) hold.
- [x] Single-sequence ($B=1$) behavior remains correct.
- [x] Real-trace replay remains completely isolated and unchanged.
- [x] All 8 regression tests and existing prefetch/cache test suites pass.

**Verdict:** **FIXED**

