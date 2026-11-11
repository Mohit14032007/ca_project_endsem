# Point 4 Phase 2.6: Demand Accounting Correction & Decode-Step Audit Report

## MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference

---

## 1. Executive Summary & Problem Statement

During the Phase 2.5 audit, a discrepancy was noted between the mathematical demand-accounting specification and the console test output:
- **Mathematical Specification:** For two sequences requesting the same expert in the same layer during the same batch decode step (Scenario B), the required accounting is:
  - Raw Tensor Events: $D_{\text{tensor}} = 6$
  - Logical Routed Demands: $D_{\text{routed}} = 2$
  - Coalesced Physical Demands: $D_{\text{coalesced}} = 1$
- **Reported Test Console Output in Phase 2.5:**
  `[DENOMINATORS] Scenario B (B=2 same expert): Raw Events=6, Routed=2, Coalesced=2`

A thorough microarchitectural and source-level audit was initiated to investigate the actual source and test implementation, resolve all decode-step identity ambiguities, and prove that batch coalescing functions with 100% mathematical fidelity.

---

## 2. Forensic Investigation & Root Cause Discovery

### 2.1 The Test-Harness Reference Aliasing Bug
In `eval/test_fa_lru.cpp` (`test_12_demand_accounting_scenarios`), the test harness sequentially exercised Scenarios A, B, C, D, and E using the singleton `PrefetchMetricsCollector::getInstance()`:

```cpp
// In Scenario B:
col.finalize(100.0);
const auto& m_b = col.get_metrics(); // <-- CAPTURED BY CONST REFERENCE!
TEST_ASSERT((m_b.coalesced_actual_demands == 1), "Scenario B: Coalesced demands = 1");

// Later, in Scenario E:
col.finalize(100.0); // <-- Mutates internal metrics_ in singleton!
const auto& m_e = col.get_metrics();
TEST_ASSERT((m_e.coalesced_actual_demands == 2), "Scenario E: Coalesced demands = 2");

// At end of test:
std::cout << "[DENOMINATORS] Scenario B (B=2 same expert): Raw Events=" << m_b.tensor_actual_demands
          << ", Routed=" << m_b.routed_actual_demands
          << ", Coalesced=" << m_b.coalesced_actual_demands << "\n";
```

**The Discovery:**
1. `col.get_metrics()` returns a `const AggregatedPrefetchMetrics&` referencing the internal member `metrics_` of the singleton collector.
2. In Scenario B, `col.finalize()` correctly computed `coalesced_actual_demands = 1`, and `TEST_ASSERT((m_b.coalesced_actual_demands == 1))` **passed cleanly**.
3. However, `m_b` was declared as a reference (`const auto& m_b`), binding it directly to `col.metrics_`.
4. When Scenario E executed at line 403, `col.finalize()` updated `col.metrics_` with Scenario E's state, where `coalesced_actual_demands = 2` (due to two separate decode steps 0 and 1).
5. At the summary print statement, `m_b.coalesced_actual_demands` was evaluated by reading through the reference into Scenario E's mutated memory, printing `Coalesced=2`!

**Remediation:** Capturing metrics **by value** (`AggregatedPrefetchMetrics m_b = col.get_metrics();`) creates an immutable snapshot of Scenario B's state, preserving $D_{\text{coalesced}} = 1$ permanently.

---

### 2.2 Deep Architectural Audit of Decode-Step Identity
In addition to correcting the test harness reference, a comprehensive audit of decode-step propagation across `PrefetchMetricsCollector` and LLMSimulator was performed.

#### Flaw 1: Unsafe Fallback to `token_id`
In the previous implementation of `record_expert_demanded()`:
```cpp
// PREVIOUS FLAWED FALLBACK:
int d_step = decode_step;
if (d_step < 0) {
    if (current_decode_step_ >= 0) {
        d_step = current_decode_step_;
    } else {
        d_step = token_id; // <-- HAZARD: token_id is request-relative!
    }
}
```
In multi-sequence batch serving, tokens across different requests or sequences have different prompt lengths. If Sequence 0 has prompt length 10 and Sequence 1 has prompt length 30, both generate their first token at the same decode step, but their `token_id`s are 10 and 30. Falling back to `token_id` incorrectly split batch-coalesced demands into separate groups.

#### Flaw 2: Defaulting `current_decode_step_` to 0
`current_decode_step_` was previously initialized and reset to `0`. If an event occurred before any token generation began, it was silently assigned to decode step 0.

#### Flaw 3: Silent Collapsing of Missing Decode Steps
When computing coalesced demands in `finalize()`, any demand with an unassigned decode step (`d_step = -1`) was grouped into key `(-1, layer_id, expert_id)`. Unrelated events lacking decode step info were silently collapsed together.

---

## 3. Phase 2.6 Architecture & Code Changes

To resolve these architectural vulnerabilities, `PrefetchMetricsCollector` was updated with the following mechanisms:

### 3.1 Sequence-Aware Decode Step Tracking
Added `token_to_gen_step_` mapping keyed by `(request_id, seq_id, token_id)`:
```cpp
void PrefetchMetricsCollector::record_token_start(
    int request_id, int seq_id, int token_id, int gen_step, double start_ts) {
    if (!enabled_) {
        return;
    }

    current_decode_step_ = gen_step;
    auto key = std::make_tuple(request_id, seq_id, token_id);
    token_start_times_[key] = start_ts;
    token_to_gen_step_[key] = gen_step; // Explicit lookup table
}
```

### 3.2 Robust Resolution in `record_expert_demanded()`
Eliminated the unsafe `token_id` fallback:
```cpp
int d_step = decode_step;
if (d_step < 0) {
    auto t_key = std::make_tuple(request_id, seq_id, token_id);
    auto t_it = token_to_gen_step_.find(t_key);
    if (t_it != token_to_gen_step_.end()) {
        d_step = t_it->second;
    } else if (current_decode_step_ >= 0) {
        d_step = current_decode_step_;
    } else {
        d_step = -1; // Explicitly marked invalid
    }
}
```

### 3.3 Safe Handling of Missing Decode Steps in `finalize()`
Events with missing decode-step information (`d_step < 0`) are explicitly counted in `metrics_.invalid_decode_step_demands` and assigned strictly unique negative keys so they **never** collapse with each other:
```cpp
std::map<std::tuple<int, int, int>, CollectorDemandRecord> stage_coalesced_demands;
uint64_t invalid_decode_step_count = 0;
int invalid_seq_counter = -1;

for (const auto& dem : demands_) {
    int step = dem.decode_step;
    if (step < 0) {
        invalid_decode_step_count = invalid_decode_step_count + 1;
        step = invalid_seq_counter;
        invalid_seq_counter = invalid_seq_counter - 1;
    }
    auto coalesced_key = std::make_tuple(step, dem.layer_id, dem.expert_id);
    if (stage_coalesced_demands.find(coalesced_key) == stage_coalesced_demands.end()) {
        stage_coalesced_demands[coalesced_key] = dem;
    }
}
metrics_.coalesced_actual_demands = stage_coalesced_demands.size();
metrics_.invalid_decode_step_demands = invalid_decode_step_count;
```

### 3.4 Proper Reset State
On `reset()`, `current_decode_step_` is set to `-1` and `token_to_gen_step_` is cleared:
```cpp
void PrefetchMetricsCollector::reset() {
    // ...
    token_start_times_.clear();
    token_to_gen_step_.clear();
    accumulated_exposed_stall_ns_ = 0.0;
    current_decode_step_ = -1;
    // ...
}
```

---

## 4. Empirical Verification of All Scenarios

In `eval/test_fa_lru.cpp`, Test 12 and Test 14 were executed:

```
[DENOMINATORS] Scenario B (B=2 same expert): Raw Events=6, Routed=2, Coalesced=1
[DENOMINATORS] Scenario D (Overlapping token_idx): Raw Events=6, Routed=2, Coalesced=2
[DENOMINATORS] Scenario F (Diff Prompt Lengths): Raw Events=6, Routed=2, Coalesced=1
[PASS] Test 14: Simulator Batch Decode Step Propagation (B=4, 2 steps: Tensor=24, Routed=8, Coalesced=5)
```

| Scenario | Batch Size | Characteristics | Expected Raw | Expected Routed | Expected Coalesced | Verified | Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **A** | $B=1$ | Single token, W1/W3/W2 | 3 | 1 | 1 | 3 / 1 / 1 | `PASS` |
| **B** | $B=2$ | Same expert, same decode step | 6 | 2 | 1 | 6 / 2 / 1 | `PASS` |
| **C** | $B=2$ | Different experts, same step | 6 | 2 | 2 | 6 / 2 / 2 | `PASS` |
| **D** | $B=2$ | Overlapping token_idx, diff steps | 6 | 2 | 2 | 6 / 2 / 2 | `PASS` |
| **E** | $B=1$ | Repeated expert across steps 0 & 1 | 6 | 2 | 2 | 6 / 2 / 2 | `PASS` |
| **F** | $B=2$ | Different prompt lengths (10 vs 30) | 6 | 2 | 1 | 6 / 2 / 1 | `PASS` |
| **G** | $B=2$ | Missing decode step (`d_step = -1`) | 2 | 2 | 2 (no collapse) | 2 / 2 / 2 | `PASS` |
| **H** | - | Inspect individual record IDs | - | - | - | Checked | `PASS` |
| **Test 14**| $B=4$ | 2 batch steps, 4 sequences | 24 | 8 | 5 | 24 / 8 / 5 | `PASS` |

---

## 5. Conclusion

The demand accounting discrepancy is completely resolved. Scenario B unequivocally produces:
- **6 Raw Tensor Execution Events**
- **2 Logical Routed Demands**
- **1 Coalesced Physical Demand**

All accounting invariants are independently computed, verified, and protected from aliasing.

