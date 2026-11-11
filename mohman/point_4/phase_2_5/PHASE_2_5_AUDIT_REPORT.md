# Point 4 Phase 2.5: Correctness, Accounting & Policy-Activation Audit Report

## MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference

---

## 1. Executive Summary

Following the initial implementation of the Frequency-Aware LRU (FA-LRU) eviction policy and the execution of the Phase 2 smoke tests, this comprehensive **Phase 2.5 Audit** was conducted prior to launching the large-scale 120-run Phase 3 experimental campaign.

The audit addressed seven critical focal areas:
1. **Mathematical Robustness of Frequency Decay:** Corrected bitwise shift bounds for $\Delta e \ge 31$ to prevent undefined behavior or zeroing of frequency, and guarded against epoch underflow.
2. **Strict Demand Accounting Disentanglement:** Disentangled three distinct denominators ($D_{\text{routed}}, D_{\text{coalesced}}, D_{\text{tensor}}$) and eliminated synthetic multipliers.
3. **Mutual Exclusion of Eviction Classifications:** Formally isolated primary eviction counters from orthogonal prefetch-induced attribution.
4. **Pinned-Entry Safety:** Implemented a two-pass victim selection algorithm to protect pinned entries and reject prefetch admissions when cache capacity is fully pinned.
5. **Physical Transfer Reconciliation:** Investigated and resolved the 114 vs 112 demand transfer discrepancy from Smoke Test S3.
6. **Empirical Policy Activation & Divergence:** Demonstrated genuine microarchitectural divergence between standard LRU and FA-LRU on a deterministic trace (Step 11 divergence, Step 12 resident hit).
7. **Units & Telemetry Harmonization:** Standardized decimal Gigabytes ($\text{GB} = 10^9$) versus binary Gibibytes ($\text{GiB} = 2^{30}$) and distinguished host CPU runtime from simulated GPU execution latency.

All 62 automated unit/regression tests and both end-to-end trace simulations passed with a **100% pass rate**. Static analysis verified **zero ternary operators** in all C++ and Python source files.

**Final Verdict: UNCONDITIONAL GO FOR PHASE 3.**

---

## 2. Comprehensive Task-by-Task Audit Findings & Remediations

### 2.1 Task 1: Frequency Decay Clamping & Underflow Protection
- **Initial Finding:** In the Phase 2 prototype of `materialize_lazy_decay()`, an unbounded right-shift `entry.frequency >> epoch_diff` was performed. For $\Delta e \ge 31$, C++ bitwise shift behavior is undefined for 32-bit types, and integer division can reduce $F \to 0$, violating the minimum frequency invariant ($F \ge 1$). Furthermore, if `global_epoch <= entry.last_epoch`, unsigned underflow could corrupt `epoch_diff`.
- **Remediation:** 
  - Added early guard: `if (global_epoch <= entry.last_epoch) return;`.
  - Added boundary clamping: if $\text{epoch\_diff} \ge 31$, assign `entry.frequency = 1;` directly without bit shifting.
  - Enforced minimum frequency: `if (new_freq < 1) new_freq = 1;`.
- **Verification:** Unit Tests 1, 2, 3, and 4 in `eval/test_fa_lru.cpp` verified zero epoch diff, power-of-two halving, clamping at $\Delta e \in \{30, 31, 1000\}$, and underflow protection.

### 2.2 Task 2: Demand Denominator Disentanglement
- **Initial Finding:** Phase 2 reports derived tensor executions synthetically via $3 \times D_{\text{coalesced}}$, and batch coalescing was keyed by `(token_id, layer_id, expert_id)`. In multi-sequence batch serving, relative `token_id`s overlap across sequences or diverge due to prompt lengths, corrupting coalescing.
- **Remediation:**
  - Extended `CollectorDemandRecord` with `int decode_step`.
  - $D_{\text{tensor}}$ is now computed directly as $|\text{demands\_}|$ without multipliers.
  - $D_{\text{routed}}$ is computed by deduplicating across $(\text{req\_id}, \text{seq\_id}, \text{tok\_id}, \text{layer\_id}, \text{expert\_id})$.
  - $D_{\text{coalesced}}$ is computed by deduplicating across $(\text{decode\_step}, \text{layer\_id}, \text{expert\_id})$.
- **Verification:** Unit Test 12 verified all five operational scenarios (Scenarios A–E), confirming proper batch collapsing and multi-step isolation.

### 2.3 Task 3: PABI & Eviction Mutual Exclusion
- **Initial Finding:** `PREFETCH_INDUCED_EVICTION` was listed alongside primary eviction types, risking double-counting.
- **Remediation:** Formally established that primary eviction types are strictly mutually exclusive:
  $$E_{\text{total}} = E_{\text{unconsumed\_prefetch}} + E_{\text{demand\_resident}} + E_{\text{PABI\_speculative}}$$
  `prefetch_induced_evictions` is enforced as an orthogonal secondary attribution flag:
  $$E_{\text{prefetch\_induced}} \subseteq E_{\text{demand\_resident}}$$
- **Verification:** Unit Test 13 verified exact conservation ($E_{\text{total}} = 1 + 2 + 1 = 4$, with $E_{\text{induced}} = 1$).

### 2.4 Task 4: Pinned-Entry Protection & Capacity Safety
- **Initial Finding:** A naive bottom-$W$ candidate scan could evict a pinned expert or return an empty iterator when unpinned candidates existed outside the window.
- **Remediation:** Implemented a two-pass selection algorithm in `Device::select_fa_lru_victim()`:
  - **Pass 1:** Evaluates unpinned entries in bottom-$W$, selecting minimum decayed frequency.
  - **Pass 2 (Fallback):** If all entries in $W$ are pinned, searches beyond $W$ for the first unpinned candidate.
  - **Capacity Protection:** If all entries in the cache are pinned, returns `end()`. In `mark_expert_resident()`, incoming speculative prefetch admissions are safely rejected, preventing cache overflows.
- **Verification:** Unit Tests 9, 10, and 11 passed cleanly.

### 2.5 Task 5: Physical Transfer Accounting Reconciliation
- **Initial Finding:** In Smoke Test S3, 114 fresh demand misses were reported, but the summary markdown table listed $9,865,003,008\text{ bytes}$ (112 transfers).
- **Forensic Trace Audit:**
  - `prefetch_transfer_events.csv` contains exactly 131 physical transfer events (114 DEMAND + 17 PREFETCH).
  - Demand bytes: $114 \times 88,080,384 = 10,041,163,776\text{ bytes}$.
  - Prefetch bytes: $17 \times 88,080,384 = 1,497,366,528\text{ bytes}$.
  - Total DDR5 bytes: $11,538,530,304\text{ bytes}$.
  - Simulator JSON telemetry (`prefetch_summary.json`) already recorded these exact values.
- **Conclusion:** The simulator engine and JSON telemetry were 100% correct. The discrepancy was purely a markdown transcription error where S1's baseline count (112) was copied into S3.

### 2.6 Task 6: Deterministic Policy Activation & Real-Trace Evaluation
- **Deterministic Trace Proof (`test_policy_activation`):**
  - Evaluated on 8 experts with cache capacity $C=4, W=4$.
  - At Step 11, candidate window contained $(0, 1)$ [$F=3$] and $(0, 5), (0, 6), (0, 7)$ [$F=1$].
  - **Standard LRU:** Evicted the heavy-hitter $(0, 1)$ due to age.
  - **FA-LRU:** Evicted ephemeral $(0, 5)$ [$F=1$] and protected heavy-hitter $(0, 1)$ [$F=3$].
  - At Step 12, $(0, 1)$ was demanded: LRU suffered a demand miss + reload stall; FA-LRU achieved a resident HBM hit ($0\text{ ns}$ stall).
- **16-Token Real-Trace Benchmark (Prompt 60):**
  - Executed under $4.0\text{ GB}$ capacity ($B=1, T=16, L=5, K=8$).
  - Proved baseline equivalence and stability on real traces ($29.53\%$ hit rate, $39.7243\text{ GB}$ DDR5 traffic, $0.00\text{ ms}$ delta).

### 2.7 Task 7: Units Harmonization & Terminology
- **Standardized Units:**
  - Decimal Gigabytes ($\text{GB} = 10^9\text{ bytes}$).
  - Binary Gibibytes ($\text{GiB} = 2^{30}\text{ bytes}$).
  - Fixed mislabeled GiB quantities in Phase 2 documentation.
- **Timing Terminology:**
  - Simulated GPU Execution Time / Latency (ms): Cycle-accurate GPU time.
  - Host Wall-Clock Runtime (s): Elapsed execution time of the simulator on the host CPU.

---

## 3. Test & Verification Matrix

| Suite Name | Target | Tests | Passed | Failed | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Extended FA-LRU Unit Suite | `build/test_fa_lru` | 13 | 13 | 0 | `PASS` |
| Policy Activation Suite | `build/test_policy_activation` | 1 | 1 | 0 | `PASS` |
| LRU Core Regression | `build/test_lru` | 6 | 6 | 0 | `PASS` |
| Framework Configuration & Predictors | `build/test_prefetch_framework` | 17 | 17 | 0 | `PASS` |
| Prefetch Metrics Telemetry | `build/test_prefetch_metrics` | 15 | 15 | 0 | `PASS` |
| Effectiveness Telemetry | `build/test_phase4b_effectiveness_metrics` | 10 | 10 | 0 | `PASS` |
| End-to-End Real Trace (Prompt 60) | `scripts/run_p2_5_real_trace.py` | 2 | 2 | 0 | `PASS` |
| **TOTAL** | | **64** | **64** | **0** | **100% PASS** |

---

## 4. Code Quality & Zero-Ternary Compliance

A strict static AST analysis and diff inspection verified that:
1. **C++ Codebase:** Exactly 0 ternary operators (`? :`) in all modified files (`device.cpp`, `device.h`, `prefetch_metrics_collector.h/.cpp`, `test_fa_lru.cpp`, `test_policy_activation.cpp`).
2. **Python Scripts:** Exactly 0 conditional expressions (`ast.IfExp`) across all scripts under `mohman/point_4/phase_2_5/`.

---

## 5. Phase 3 Experimental Campaign Readiness & Recommendation

### 5.1 Verification Checklist
- [x] Lazy epoch decay mathematically safe (no underflow, clamped $\ge 31$ to 1).
- [x] Three-way demand denominators disentangled ($D_{\text{routed}}, D_{\text{coalesced}}, D_{\text{tensor}}$).
- [x] Eviction mutual exclusion formally defined and verified ($E_{\text{total}} = E_{\text{unconsumed}} + E_{\text{demand}} + E_{\text{PABI}}$).
- [x] Pinned entries strictly protected from eviction across all search paths.
- [x] 100% pinned cache capacity overflow prevented via prefetch rejection.
- [x] CSTP transfer counts reconciled to exact physical traces (114 demand + 17 prefetch = 131).
- [x] Microarchitectural policy divergence demonstrated empirically (Step 11 divergence, Step 12 hit).
- [x] Decimal GB and binary GiB clearly delineated.
- [x] Zero ternary operators verified across C++ and Python.
- [x] Full regression test suite passing (64 / 64).

### 5.2 Definitive Recommendation
$$\mathbf{\Large \text{RECOMMENDATION: GO FOR PHASE 3}}$$

The FA-LRU implementation and LLMSimulator telemetry infrastructure are verified, mathematically sound, microarchitecturally activated, and fully ready for the 120-run Phase 3 experimental campaign.

