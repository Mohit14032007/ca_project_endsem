# Point 4 Phase 2.6: Pre-Experiment Gate — GO / NO-GO Decision Report

## MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference

---

## 1. Formal Decision Statement

$$\mathbf{\Huge \text{DECISION: UNCONDITIONAL GO}}$$

**Phase 2.6 pre-experiment corrections, demand-accounting validations, and architectural invariant checks have achieved a 100% pass rate. Permission to proceed to the focused 24-run Phase 3 campaign is GRANTED.**

---

## 2. Explicit Resolution of the Scenario B Discrepancy

As mandated by Section A5, this report explicitly establishes the exact empirical and theoretical values for **Scenario B** (two sequences requesting the same expert in the same layer during the same batch decode step):

```
========================================================================================
SCENARIO B DEMAND ACCOUNTING VERIFICATION
========================================================================================
Quantity                    | Expected Value | Measured Value | Validation Status
----------------------------------------------------------------------------------------
Raw Tensor Events (D_tensor)| 6              | 6              | VERIFIED (100% Match)
Logical Routed Demands      | 2              | 2              | VERIFIED (100% Match)
Coalesced Physical Demands  | 1              | 1              | VERIFIED (100% Match)
========================================================================================
```

### Proof & Traceability
- **Root Cause Established:** In the previous test harness, `m_b` was captured by reference (`const auto& m_b = col.get_metrics()`). A subsequent test (Scenario E) re-finalized the singleton collector and set `coalesced_actual_demands = 2`. The print statement evaluated through the aliased reference and printed `Coalesced=2`.
- **Correction Applied:** `m_b` is now stored by value (`AggregatedPrefetchMetrics m_b = col.get_metrics()`), preserving Scenario B's verified metrics ($D_{\text{tensor}}=6, D_{\text{routed}}=2, D_{\text{coalesced}}=1$).
- **Architectural Safeguards:**
  - Removed dangerous fallback to `token_id` in `record_expert_demanded()`.
  - Added sequence-aware lookup table `token_to_gen_step_`.
  - Guaranteed that invalid decode-step events never collapse together.
  - Verified across 14 unit tests, including multi-sequence varying prompt lengths and simulator batch propagation.

---

## 3. Invariant Verification Checklist

| Invariant / Check | Requirement | Verification Result | Gate Status |
| :--- | :--- | :--- | :---: |
| **1. Frequency Decay Clamp** | $F \ge 1$, no shift overflow for $\Delta e \ge 31$ | Clamped to 1, shift-safe | `PASS` |
| **2. Underflow Guard** | Guard `global_epoch <= last_epoch` | Zero unsigned underflow | `PASS` |
| **3. LRU Equivalence** | $W=1$ FA-LRU matches LRU | Verified identical choices | `PASS` |
| **4. Candidate Window** | Evictions confined to bottom-$W$ | Entries outside $W$ protected | `PASS` |
| **5. Pinning Protection** | Pinned entries never evicted | Two-pass search protects pins | `PASS` |
| **6. Fully Pinned Capacity** | Reject prefetch admission if all pinned | Rejected without overflow | `PASS` |
| **7. Eviction Mutual Exclusion**| $E_{\text{total}} = E_{\text{unconsumed}} + E_{\text{demand}} + E_{\text{PABI}}$ | Exact sum verified ($E=3$) | `PASS` |
| **8. Prefetch Attribution** | $E_{\text{induced}} \subseteq E_{\text{demand}}$ | Verified as orthogonal flag | `PASS` |
| **9. In-Flight Promotion** | Attach without duplicate transfer | No duplicate transfer IDs | `PASS` |
| **10. Traffic Conservation** | $B_{\text{total}} = B_{\text{demand}} + B_{\text{speculative}}$ | Exact byte equality | `PASS` |
| **11. Zero Ternary Operators** | 0 `? :` in C++, 0 `ast.IfExp` in Python | 0 matches found | `PASS` |
| **Scenario B Denominators** | Tensor=6, Routed=2, Coalesced=1 | **Verified 6 / 2 / 1** | `PASS` |

---

## 4. Authorization for Part B Execution

All gates defined in Part A are fully satisfied:
1. Extended unit tests (`build/test_fa_lru`): **14 / 14 passed.**
2. Regression test suite: **49 / 49 passed.**
3. Total tests: **63 / 63 passed (100%).**
4. All 4 required Part A documentation reports exist under `mohman/point_4/phase_2_6/`.

The focused **24-Run Phase 3 Campaign** is authorized to proceed immediately under `mohman/point_4/phase_3_focused_24/`.

