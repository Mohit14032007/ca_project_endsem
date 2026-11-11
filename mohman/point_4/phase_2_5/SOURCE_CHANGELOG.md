# Point 4 Phase 2.5: Source Code Changelog & Zero-Ternary Audit

## MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference

---

## 1. Executive Summary

This document provides a comprehensive, contiguous audit of all source code modifications made during **Phase 2.5**. 

Every modification strictly adheres to:
1. **Preservation of Existing Behaviors:** Standard LRU and PABI execution paths remain untouched unless explicitly requested.
2. **Deterministic Mechanics:** Deterministic tie-breaking and bounds clamping.
3. **Zero Ternary Operators:** Zero usage of `? :` in C++ and zero usage of `ast.IfExp` conditional expressions in Python.

---

## 2. Detailed File-by-File Code Modifications

### 2.1 `src/hardware/device.cpp`

#### 2.1.1 `materialize_lazy_decay()` — Clamping, Shift Safety, and Underflow Guard
- **Problem in Phase 2:** Unclamped bitwise shifts for $\text{epoch\_diff} \ge 31$ risked undefined integer behavior or zeroing out frequencies ($F \to 0$), and `global_epoch <= last_epoch` had no underflow check.
- **Phase 2.5 Implementation:**
```cpp
void Device::materialize_lazy_decay(ResidentExpertEntry& entry) {
  if (global_epoch <= entry.last_epoch) {
    return;
  }
  uint32_t epoch_diff = global_epoch - entry.last_epoch;
  if (epoch_diff >= 31) {
    entry.frequency = 1;
    entry.last_epoch = global_epoch;
    return;
  }
  uint32_t new_freq = entry.frequency >> epoch_diff;
  if (new_freq < 1) {
    new_freq = 1;
  }
  entry.frequency = new_freq;
  entry.last_epoch = global_epoch;
}
```

#### 2.1.2 `select_fa_lru_victim()` — Pinning-Aware Two-Pass Victim Selection
- **Problem in Phase 2:** Looked only at the bottom-$W$ window without skipping pinned entries; if all candidates in $W$ were pinned, it returned `end()` even if unpinned entries existed outside $W$.
- **Phase 2.5 Implementation:**
```cpp
std::list<std::tuple<int, int, long long>>::iterator Device::select_fa_lru_victim() {
  if (lru_expert_list.empty()) {
    return lru_expert_list.end();
  }

  // PASS 1: Search bottom-W window for unpinned candidate with minimum decayed frequency
  int count = 0;
  auto it = lru_expert_list.begin();
  auto best_it = lru_expert_list.end();
  uint32_t min_freq = UINT32_MAX;

  while (it != lru_expert_list.end() && count < fa_lru_window_size) {
    int l = std::get<0>(*it);
    int e = std::get<1>(*it);
    auto rentry_it = resident_experts.find(std::make_pair(l, e));
    if (rentry_it != resident_experts.end()) {
      if (!rentry_it->second.is_pinned) {
        materialize_lazy_decay(rentry_it->second);
        uint32_t f = rentry_it->second.frequency;
        if (f < min_freq) {
          min_freq = f;
          best_it = it;
        }
      }
    }
    it++;
    count++;
  }

  if (best_it != lru_expert_list.end()) {
    return best_it;
  }

  // PASS 2: Fallback beyond bottom-W window if all window entries are pinned
  it = lru_expert_list.begin();
  while (it != lru_expert_list.end()) {
    int l = std::get<0>(*it);
    int e = std::get<1>(*it);
    auto rentry_it = resident_experts.find(std::make_pair(l, e));
    if (rentry_it != resident_experts.end() && !rentry_it->second.is_pinned) {
      return it;
    }
    it++;
  }

  // ALL entries in cache are pinned: return end()
  return lru_expert_list.end();
}
```

#### 2.1.3 `mark_expert_resident()` — Pinned-Capacity Protection & Prefetch Admission Rejection
- Added safe rejection of prefetch admissions when cache capacity is completely full of pinned experts:
```cpp
if (victim_it == lru_expert_list.end()) {
  if (is_admitting_prefetch) {
    prefetch_metrics.candidate_admission_rejected++;
    return;
  }
}
```

---

### 2.2 `src/hardware/device.h`
- Added method prototypes:
```cpp
void materialize_lazy_decay(ResidentExpertEntry& entry);
std::list<std::tuple<int, int, long long>>::iterator select_fa_lru_victim();
```

---

### 2.3 `src/prefetch/prefetch_metrics_collector.h` & `src/prefetch/prefetch_metrics_collector.cpp`

#### 2.3.1 Decode Step Tracking in Demands
- Extended `CollectorDemandRecord` with `int decode_step = 0;`.
- Added member `int current_decode_step_ = 0;` and accessor `void set_current_decode_step(int step)`.
- In `record_token_start()`, automatically set `current_decode_step_ = gen_step;`.
- Added backward-compatible 13-argument overload and 14-argument `record_expert_demanded(..., int decode_step = -1)`.

#### 2.3.2 Strict Demand Denominator Disentanglement in `finalize()`
- Disentangled the three denominators:
```cpp
// 1. Raw Execution Tensor Events (No synthetic multipliers)
summary.tensor_actual_demands = demands_.size();

// 2. Logical Token-to-Expert Routed Demands (Dedup W1, W3, W2 per token)
struct RoutedKey {
  int req_id;
  int seq_id;
  int token_id;
  int layer_id;
  int expert_id;
  bool operator==(const RoutedKey& o) const {
    return req_id == o.req_id && seq_id == o.seq_id && token_id == o.token_id &&
           layer_id == o.layer_id && expert_id == o.expert_id;
  }
};
std::unordered_set<RoutedKey, RoutedKeyHash> unique_routed_demands;
for (const auto& d : demands_) {
  unique_routed_demands.insert({d.request_id, d.seq_id, d.token_id, d.layer_id, d.expert_id});
}
summary.routed_actual_demands = unique_routed_demands.size();

// 3. Physical Coalesced Demands per Decode Step
struct CoalescedKey {
  int decode_step;
  int layer_id;
  int expert_id;
  bool operator==(const CoalescedKey& o) const {
    return decode_step == o.decode_step && layer_id == o.layer_id && expert_id == o.expert_id;
  }
};
std::unordered_set<CoalescedKey, CoalescedKeyHash> unique_coalesced_demands;
for (const auto& d : demands_) {
  int step = d.decode_step;
  if (step < 0) {
    step = d.token_id;
  }
  unique_coalesced_demands.insert({step, d.layer_id, d.expert_id});
}
summary.coalesced_actual_demands = unique_coalesced_demands.size();
```

---

### 2.4 `eval/test_fa_lru.cpp`
- Re-architected into 13 comprehensive unit tests validating:
  - Zero epoch difference (Test 1).
  - Epoch differences 1 and 2 (Test 2).
  - Clamping for $\Delta e \ge 31$ (Test 3).
  - Idempotence & underflow guard (Test 4).
  - $W=1$ equivalence to LRU (Test 5).
  - Minimum frequency victim selection (Test 6).
  - Window bounding (Test 7).
  - Strict oldest tie-breaking (Test 8).
  - Pinned oldest entry protection (Test 9).
  - Fallback beyond window when all window entries pinned (Test 10).
  - All candidates pinned rejection (Test 11).
  - Demand accounting Scenarios A–E (Test 12).
  - PABI eviction mutual exclusion (Test 13).

---

### 2.5 `eval/test_policy_activation.cpp`
- Implemented deterministic policy activation test demonstrating:
  - Step 11 victim divergence (LRU evicts heavy-hitter `(0, 1)`, FA-LRU protects `(0, 1)` and evicts ephemeral `(0, 5)`).
  - Step 12 resident hit payoff for FA-LRU.
  - Outputs decision trace to `mohman/point_4/phase_2_5/policy_activation_decisions.csv`.

---

### 2.6 `eval/CMakeLists.txt`
- Added target:
```cmake
add_executable(test_policy_activation test_policy_activation.cpp)
target_link_libraries(test_policy_activation PRIVATE llm_system pthread)
```

---

## 3. Zero-Ternary Compliance Audit Results

A strict automated static analysis was conducted across all files modified or added in Phase 2.5:

### 3.1 C++ Source & Header Check
- **Command:** `git diff src/ hardware/ prefetch/ eval/ | grep -E "^\+[^\+].*\?"`
- **Result:** **0 matches found.**
- All branching logic in C++ uses explicit `if (...) { ... } else { ... }` blocks.

### 3.2 Python Script AST Check
- **Script:**
```python
import ast, os
# Walk AST and assert ast.IfExp count == 0
```
- **Files Inspected:**
  - `mohman/point_4/phase_2_5/scripts/run_p2_5_real_trace.py`
- **Result:** **[AST_CHECK] Verified 0 ternary operators (ast.IfExp == 0).**

