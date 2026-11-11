# Point 4 Phase 2.6: Pre-Experiment Validation & Invariants Verification Report

## MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference

---

## 1. Executive Summary

This report establishes the comprehensive verification of all architectural correctness, accounting, and safety invariants prior to commencing the focused 24-run Phase 3 campaign.

A total of **63 automated test cases** were executed across 6 test suites, achieving a **100% pass rate**:
- **Extended Unit Test Suite (`build/test_fa_lru`):** 14 / 14 passed.
- **Policy Activation Test (`build/test_policy_activation`):** 1 / 1 passed.
- **LRU Core Regression (`build/test_lru`):** 6 / 6 passed.
- **Prefetch Framework Tests (`build/test_prefetch_framework`):** 17 / 17 passed.
- **Prefetch Metrics Telemetry (`build/test_prefetch_metrics`):** 15 / 15 passed.
- **Effectiveness Telemetry (`build/test_phase4b_effectiveness_metrics`):** 10 / 10 passed.

$$\mathbf{\Large \text{Total Pass Rate: 63 / 63 PASSED (100.0\%)}}$$

---

## 2. Verification of the 11 Core Invariants

### Invariant 1: Lazy Frequency Decay Preserves Minimum Frequency 1
- **Requirement:** After arbitrary epoch differences $\Delta e \ge 0$, decayed frequency must satisfy $F \ge 1$. For $\Delta e \ge 31$, bitwise shifting must not cause undefined behavior or reduce frequency to 0.
- **Verification:** Unit Tests 1, 2, and 3 in `eval/test_fa_lru.cpp` test $\Delta e = 0$, $\Delta e \in \{1, 2\}$, and $\Delta e \in \{30, 31, 1000\}$. All decayed values clamp to $\max(1, \dots)$ with $F=1$ at large diffs.
- **Status:** **VERIFIED**

### Invariant 2: Epoch Underflow Guard
- **Requirement:** If `global_epoch <= entry.last_epoch`, the decay function must return immediately without performing unsigned subtraction.
- **Verification:** Unit Test 4 in `eval/test_fa_lru.cpp` verified that repeated calls in the same epoch do not alter frequency and guards prevent integer underflow.
- **Status:** **VERIFIED**

### Invariant 3: $W=1$ FA-LRU Strictly Equivalent to LRU
- **Requirement:** When the candidate window is configured to $W=1$, FA-LRU must always select the candidate at the front of `lru_expert_list`, matching standard LRU identically.
- **Verification:** Unit Test 5 in `eval/test_fa_lru.cpp` populated cache with diverse frequencies ($F \in \{1, 5, 10, 2\}$) and confirmed that $W=1$ selects the oldest entry regardless of frequency.
- **Status:** **VERIFIED**

### Invariant 4: Candidate Window Bounding (Bottom-$W$)
- **Requirement:** Entries residing above the bottom-$W$ cutoff must be strictly immune from victim selection, even if they have $F=1$.
- **Verification:** Unit Test 7 in `eval/test_fa_lru.cpp` tested cache of 4 entries with $W=2$, where oldest entries had $F=10$ and newest entries had $F=1$. FA-LRU selected within the bottom 2 entries and never touched entries outside the window.
- **Status:** **VERIFIED**

### Invariant 5: Pinned Experts Never Selected for Eviction
- **Requirement:** Any expert entry marked `is_pinned = true` must be bypassed during victim selection.
- **Verification:** Unit Tests 9 and 10 in `eval/test_fa_lru.cpp` tested scenarios where the oldest candidate or all window candidates were pinned. The two-pass selection algorithm safely skipped pinned entries and selected unpinned candidates.
- **Status:** **VERIFIED**

### Invariant 6: Fully Pinned Cache Rejects Speculative Admission
- **Requirement:** When 100% of resident cache entries are pinned, incoming speculative prefetch admissions must be rejected rather than exceeding capacity or evicting pinned entries.
- **Verification:** Unit Test 11 in `eval/test_fa_lru.cpp` filled cache to capacity with pinned entries and attempted prefetch admission. The admission was rejected, `candidate_admission_rejected` incremented, and cache size remained within capacity bounds.
- **Status:** **VERIFIED**

### Invariant 7: Eviction Mutual Exclusion
- **Requirement:** $E_{\text{total}} = E_{\text{unconsumed\_prefetch}} + E_{\text{demand\_resident}} + E_{\text{PABI\_speculative}}$. Every eviction event must belong to exactly one primary classification.
- **Verification:** Unit Test 13 in `eval/test_fa_lru.cpp` recorded one event of each type and verified $E_{\text{total}} = 1 + 1 + 1 = 3$.
- **Status:** **VERIFIED**

### Invariant 8: Prefetch-Induced Eviction is Orthogonal Attribution
- **Requirement:** `prefetch_induced_evictions` must be a subset of demand-resident evictions ($E_{\text{prefetch\_induced}} \le E_{\text{demand\_resident}}$) and not an additive fourth category.
- **Verification:** Unit Test 13 in `eval/test_fa_lru.cpp` confirmed that prefetch-induced evictions increment the secondary counter without inflating $E_{\text{total}}$.
- **Status:** **VERIFIED**

### Invariant 9: In-Flight Demand Promotion Prevents Duplicate Transfers
- **Requirement:** When a demand miss attaches to an in-flight speculative prefetch, the transfer must be promoted to demand status without issuing a duplicate DDR5 read request.
- **Verification:** `test_prefetch_metrics.cpp` (Scenario 2 and Scenario 7) confirmed in-flight attachment and deduplication with zero duplicate transfer IDs.
- **Status:** **VERIFIED**

### Invariant 10: Total Physical Bytes Calculated from Unique Transfer IDs
- **Requirement:** Memory traffic must be calculated from physical transfers where each transfer has a unique `transfer_id`, ensuring $B_{\text{total}} = B_{\text{demand}} + B_{\text{speculative}}$.
- **Verification:** Verified via `test_prefetch_framework` and trace reconciliation audits.
- **Status:** **VERIFIED**

### Invariant 11: Zero Ternary Operators
- **Requirement:** Absolutely zero usage of `? :` in C++ and zero usage of `ast.IfExp` in Python.
- **Verification:** Static diff analysis (`git diff | grep -E "^\+[^\+].*\?"`) returned 0 matches. Static Python AST walks returned 0 ternary nodes.
- **Status:** **VERIFIED**

---

## 3. Comprehensive Test Execution Log

```
======================================================================
POINT 4 PHASE 2.6 EXTENDED UNIT TESTS: FA-LRU, ACCOUNTING & PINNING
======================================================================
  [PASS] Test 1: Decay boundary with zero epoch difference
  [PASS] Test 2: Decay boundary with epoch differences 1 and 2
  [PASS] Test 3: Decay boundary clamp for epoch_diff 30, 31, and 1000 (assigns 1, shift-safe)
  [PASS] Test 4: Decay idempotence in identical epoch and underflow protection
  [PASS] Test 5: W=1 FA-LRU strict equivalence to LRU
  [PASS] Test 6: FA-LRU victim selection by minimum frequency in window
  [PASS] Test 7: FA-LRU window bounding (entries outside W protected)
  [PASS] Test 8: FA-LRU strict tie-breaking to oldest candidate
  [PASS] Test 9: Pinned oldest entry protected, unpinned candidate selected
  [PASS] Test 10: All window candidates pinned, fallback to unpinned beyond window
  [PASS] Test 11: All candidates pinned protects cache from silent eviction
  [PASS] Test 12: Comprehensive Demand Accounting Scenarios A-H Verified
         [DENOMINATORS] Scenario B (B=2 same expert): Raw Events=6, Routed=2, Coalesced=1
         [DENOMINATORS] Scenario D (Overlapping token_idx): Raw Events=6, Routed=2, Coalesced=2
         [DENOMINATORS] Scenario F (Diff Prompt Lengths): Raw Events=6, Routed=2, Coalesced=1
  [PASS] Test 13: PABI and primary eviction categories are strictly mutually exclusive (E_total = 3)
  [PASS] Test 14: Simulator Batch Decode Step Propagation (B=4, 2 steps: Tensor=24, Routed=8, Coalesced=5)
======================================================================
[SUCCESS] ALL 14 EXTENDED UNIT TESTS PASSED CLEANLY!
======================================================================
```

---

## 4. Summary Matrix of Test Suites

| Suite Name | Target Executable | Test Count | Pass Count | Fail Count | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Extended FA-LRU Suite | `build/test_fa_lru` | 14 | 14 | 0 | `100% PASS` |
| Policy Activation Suite | `build/test_policy_activation` | 1 | 1 | 0 | `100% PASS` |
| LRU Core Regression | `build/test_lru` | 6 | 6 | 0 | `100% PASS` |
| Prefetch Framework Tests | `build/test_prefetch_framework` | 17 | 17 | 0 | `100% PASS` |
| Prefetch Metrics Telemetry | `build/test_prefetch_metrics` | 15 | 15 | 0 | `100% PASS` |
| Effectiveness Telemetry | `build/test_phase4b_effectiveness_metrics` | 10 | 10 | 0 | `100% PASS` |
| **TOTAL** | | **63** | **63** | **0** | **100% PASS** |

