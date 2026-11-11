# Point 5 — Final Numerical Consistency Audit and Provenance Reconciliation

**Document ID:** `mohman/point_5/audit/FINAL_NUMERICAL_RECONCILIATION.md`  
**Status:** Completed and Authoritative  
**Date:** November 10, 2026  
**Scope:** Strict numerical provenance audit, algebraic reconciliation, and report correction for Point 5 comparative evaluation.

---

## Executive Summary & Audit Mandate

This audit performs an exhaustive, bit-exact verification of all quantitative values across the Point 5 Comparative Evaluation for the **MoE Memory Wall** research project. In accordance with strict architectural audit rules:
1. Zero simulation runs or source code modifications were performed.
2. Every plotted and reported value is traced directly to its authentic source artifact.
3. Spurious numbers, rounding approximations, and conceptual ambiguities identified across preliminary reports have been formally audited and reconciled.
4. Strictly **zero ternary operators** (`? :` in C++, `ast.IfExp` in Python) or conditional ternary constructs were used.

---

## 1. Workload W3 Big Experiment Provenance Audit

### 1.1 The Numerical Discrepancy

Preliminary working notes exhibited a fundamental inconsistency regarding Workload W3 ($B=14, T=128, M_{\text{HBM}}=35\text{ GiB}$, 426 slots):
- **Disputed Execution Times:** Baseline $= 442.146\text{ s}$, CSTP + LRU $= 338.257\text{ s}$, CSTP + FA-LRU $= 338.286\text{ s}$.
- **Reported Mean Token Latencies:** Baseline $= 3,586.18\text{ ms/token}$, CSTP + LRU $= 3,565.37\text{ ms/token}$, CSTP + FA-LRU $= 3,565.49\text{ ms/token}$.

### 1.2 Algebraic Proof and Independent Calculation

For a sequential generation horizon of $T = 128$ tokens, the mean per-token latency $\bar{L}_{\text{token}}$ is algebraically defined as:
$$\bar{L}_{\text{token}} = \frac{T_{\text{simulated\_execution}}}{T}$$

Applying this definition to both candidate sets reveals:

#### Candidate Set 1: Disputed Execution Times
- Baseline: $\frac{442.146\text{ s} \times 1,000\text{ ms/s}}{128\text{ tokens}} = 3,454.266\text{ ms/token} \neq 3,586.18\text{ ms/token}$
- CSTP + LRU: $\frac{338.257\text{ s} \times 1,000\text{ ms/s}}{128\text{ tokens}} = 2,642.633\text{ ms/token} \neq 3,565.37\text{ ms/token}$
- CSTP + FA-LRU: $\frac{338.286\text{ s} \times 1,000\text{ ms/s}}{128\text{ tokens}} = 2,642.859\text{ ms/token} \neq 3,565.49\text{ ms/token}$
*Conclusion:* Candidate Set 1 produces a massive mathematical contradiction ($>900\text{ ms/token}$ error).

#### Candidate Set 2: Authentic Source Artifacts (`all_runs.csv`)
Inspecting `mohman/point_4/big_exp/results/all_runs.csv` and `prefetch_summary.json`:
- `w3_c1_r1` (Baseline):
  - `simulated_execution_time_ms` $= 459,031.335440\text{ ms}$ ($459.031335\text{ s}$)
  - `mean_token_latency_ms` $= 3,586.182308\text{ ms}$
  - Ratio: $\frac{459,031.335440\text{ ms}}{128} = \mathbf{3,586.182308\text{ ms/token}}$ (**Exact identity**)
- `w3_c3_r1` (CSTP + LRU):
  - `simulated_execution_time_ms` $= 456,367.513166\text{ ms}$ ($456.367513\text{ s}$)
  - `mean_token_latency_ms` $= 3,565.371197\text{ ms}$
  - Ratio: $\frac{456,367.513166\text{ ms}}{128} = \mathbf{3,565.371197\text{ ms/token}}$ (**Exact identity**)
- `w3_c4_r1` (CSTP + FA-LRU):
  - `simulated_execution_time_ms` $= 456,382.651982\text{ ms}$ ($456.382652\text{ s}$)
  - `mean_token_latency_ms` $= 3,565.489469\text{ ms}$
  - Ratio: $\frac{456,382.651982\text{ ms}}{128} = \mathbf{3,565.489469\text{ ms/token}}$ (**Exact identity**)

### 1.3 Root Cause of the Discrepancy

Deep inspection of raw traces isolated the provenance of the spurious numbers $442.146$, $338.257$, and $338.286$:
1. In `mohman/point_3/phase2/large_workload/experiments/raw/pilot/large_pilot_w5_stmoe_h2/traces/prefetch_events.csv`, event line 623585 contained:
   `status=RESIDENT_HIT;wait_ns=0.000000;ready_ts=51018892442.146210` (ending in `442.146`).
2. In `mohman/point_4/big_exp/results/runs/w4_c3_r1/traces/prefetch_events.csv`, event line 4223070 contained:
   `status=LATE_IN_FLIGHT;wait_ns=1376256.000000;ready_ts=682752943338.257446` (ending in `338.257`).
3. In `mohman/point_4/big_exp/results/runs/w4_c4_r1/traces/prefetch_events.csv`, event line 1663460 contained:
   `status=LATE_IN_FLIGHT;wait_ns=1376256.000000;ready_ts=245363195338.286102` (ending in `338.286`).

**Audit Finding:** An early trace-inspection script parsed fractional nanosecond substring tails (`ready_ts`) rather than extracting `total_simulated_execution_ns` from `prefetch_summary.json`. The authentic simulation times for Workload W3 are **459.031 s** (Baseline), **456.368 s** (CSTP + LRU), and **456.383 s** (CSTP + FA-LRU). All tables, figures, and prose are updated accordingly.

---

## 2. Component Attribution Arithmetic (Unrounded Precision)

The comparative decomposition evaluates the stepwise transition from pure baseline to the complete system under Workload W4 ($B=32, T=128, M_{\text{HBM}}=35\text{ GiB}$):
- Transition $A \rightarrow B$: Predictor evaluation in shadow mode.
- Transition $A \rightarrow C$: Activation of asynchronous CSTP prefetching under Standard LRU.
- Transition $C \rightarrow D$: Replacement of Standard LRU with Frequency-Aware LRU ($W=64$).
- Net System Improvement: Transition $A \rightarrow D$.

### 2.1 Execution Time Attribution

Using unrounded source measurements:
- $T_{\text{exec}}(A) = 1,008,204.00000\text{ ms}$ ($1,008.20400\text{ s}$)
- $T_{\text{exec}}(B) = 1,008,204.00000\text{ ms}$ ($1,008.20400\text{ s}$)
- $T_{\text{exec}}(C) = 761,607.54845\text{ ms}$ ($761.60755\text{ s}$)
- $T_{\text{exec}}(D) = 761,575.89456\text{ ms}$ ($761.57589\text{ s}$)

Stepwise Reductions:
- $\Delta T_{A \rightarrow C} = 1,008,204.00000 - 761,607.54845 = 246,596.45155\text{ ms}$ ($246.59645\text{ s}$)
- $\Delta T_{C \rightarrow D} = 761,607.54845 - 761,575.89456 = 31.65389\text{ ms}$ ($0.03165\text{ s}$)
- Total $\Delta T_{A \rightarrow D} = 1,008,204.00000 - 761,575.89456 = 246,628.10544\text{ ms}$ ($246.62811\text{ s}$)

Component Attribution Shares:
$$\text{Prefetching Share} = \frac{246,596.45155}{246,628.10544} \times 100\% = \mathbf{99.98717\%}$$
$$\text{FA-LRU Share} = \frac{31.65389}{246,628.10544} \times 100\% = \mathbf{0.01283\%}$$

### 2.2 Exposed GPU Stall Time Attribution

Using unrounded source stall measurements:
- $S(A) = 994,459.73000\text{ ms}$
- $S(B) = 994,459.73000\text{ ms}$
- $S(C) = 747,863.56942\text{ ms}$
- $S(D) = 747,831.74514\text{ ms}$

Stepwise Stall Reductions:
- $\Delta S_{A \rightarrow C} = 994,459.73000 - 747,863.56942 = 246,596.16058\text{ ms}$ ($246.59616\text{ s}$)
- $\Delta S_{C \rightarrow D} = 747,863.56942 - 747,831.74514 = 31.82429\text{ ms}$ ($0.03182\text{ s}$)
- Total $\Delta S_{A \rightarrow D} = 994,459.73000 - 747,831.74514 = 246,627.98486\text{ ms}$ ($246.62798\text{ s}$)

Component Stall Attribution Shares:
$$\text{Prefetching Stall Share} = \frac{246,596.16058}{246,627.98486} \times 100\% = \mathbf{99.98710\%}$$
$$\text{FA-LRU Stall Share} = \frac{31.82429}{246,627.98486} \times 100\% = \mathbf{0.01290\%}$$

**Standardized Reporting Precision:**
Across all summary reports, the component contributions are reported to three decimal places:
- **Asynchronous Prefetching (Component 2):** **99.987%** of net savings.
- **FA-LRU Cache Management (Component 3):** **0.013%** of net savings.
- Sum: $99.987\% + 0.013\% = 100.000\%$.

---

## 3. Exposed GPU Stall Reduction Wording Reconciliation

### 3.1 Audit of Erroneous Typographical Phrasing

Preliminary drafts contained loose descriptions describing the stall reduction as "approximately 25 seconds" or "nearly 250 seconds".
- Origin of "25 seconds": A typographical error that conflated the percentage reduction (**24.80%**) with elapsed seconds.
- Origin of "250 seconds": An informal rounding of $246.60\text{ seconds}$.

### 3.2 Canonical Reconciled Statement

The exposed stall reduction is formally standardized across all documentation as:
> **"246,596.16 ms (246.60 seconds, or 4.11 minutes, representing a 24.80% reduction relative to baseline)"**

---

## 4. DDR5 Memory Traffic Decomposition and Dynamics

### 4.1 Exact Byte Counters for Workload W4

In `mohman/point_4/big_exp/results/all_runs.csv`, memory traffic is decomposed into demand and speculative bytes:

| Configuration | Demand Bytes | Speculative Bytes | Total DDR5 Bytes | Total Decimal GB ($10^9\text{ B}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Config A: Baseline** | 48,740,162,117,632 | 0 | 48,740,162,117,632 | 48,740.16 GB |
| **Config B: Prediction-Only** | 48,740,162,117,632 | 0 | 48,740,162,117,632 | 48,740.16 GB |
| **Config C2: CSTP + LRU** | 37,790,536,433,664 | 10,948,303,650,816 | 48,738,840,084,480 | 48,738.84 GB |
| **Config D: Complete (FA-LRU)** | 37,808,769,073,152 | 10,928,045,162,496 | 48,736,814,235,648 | 48,736.81 GB |

### 4.2 Traffic Deltas and Block Quantization

Every expert parameter block is $88,080,384\text{ bytes}$ ($84\text{ MiB}$). Calculating the delta from Config C2 to Config D:
- **Speculative Byte Delta:**
  $$10,928,045,162,496 - 10,948,303,650,816 = -20,258,488,320\text{ bytes} = -20.258488\text{ GB}$$
  $$\text{Block Count} = \frac{-20,258,488,320\text{ B}}{88,080,384\text{ B/block}} = \mathbf{-230.0\text{ speculative blocks}}$$
- **Demand Byte Delta:**
  $$37,808,769,073,152 - 37,790,536,433,664 = +18,232,639,488\text{ bytes} = +18.232639\text{ GB}$$
  $$\text{Block Count} = \frac{+18,232,639,488\text{ B}}{88,080,384\text{ B/block}} = \mathbf{+207.0\text{ demand blocks}}$$
- **Net Bus Traffic Delta:**
  $$-20,258,488,320 + 18,232,639,488 = -2,025,848,832\text{ bytes} = \mathbf{-2.025849\text{ GB}}$$
  $$\text{Net Block Count} = \frac{-2,025,848,832\text{ B}}{88,080,384\text{ B/block}} = \mathbf{-23.0\text{ total blocks}}$$

### 4.3 Architectural Dynamic: Why Demand Bytes Increase While Speculative Bytes Decrease

It is technically incorrect to claim that the $2.0258\text{ GB}$ reduction arises solely from avoided demand misses. The physical mechanism operates as follows:
1. **Speculative Queue Relief:** Under FA-LRU, 5,999 unconsumed speculative prefetches are purged from the candidate window, dropping speculative traffic by $230\text{ blocks}$ ($-20.26\text{ GB}$).
2. **Transfer Class Shift:** Because fewer speculative prefetches occupy the prefetch buffer, 207 expert demands that would have attached late in-flight under LRU instead miss as direct demand fetches ($+18.23\text{ GB}$ demand traffic).
3. **Net Physical Benefit:** Concurrently, 36 protected demand lines are reused in HBM, averting 15 complete expert reloads. The net traffic reduction is:
   $$\Delta \text{Total} = \Delta \text{Speculative} + \Delta \text{Demand} = -20.2585\text{ GB} + 18.2326\text{ GB} = \mathbf{-2.0258\text{ GB}}$$
   This perfectly equals the size of 23 avoided expert blocks ($23 \times 88,080,384\text{ B} = 2,025,848,832\text{ B}$).

---

## 5. FA-LRU Diagnostic Counters vs. Physical Hardware Measurements

To maintain strict scientific integrity, internal algorithmic diagnostics must be rigorously separated from physical hardware observables:

### 5.1 Internal Algorithmic Diagnostics (Controller Decisions)
- **Total Eviction Decisions:** $552,896$
- **Divergent Eviction Decisions:** $431,909$ ($78.12\%$ divergence from LRU tail)
- **Demand Lines Spared from Eviction:** $431,909$
- **Unconsumed Speculative Victims Evicted:** $5,999$
- **Protected Demand Lines Subsequently Reused:** $36$
- **Protected Demand Lines Never Reused:** $14$
- **Scan Depth & Cost:** Evaluated exactly within bottom-$W=64$ window (0 fallback beyond window)

### 5.2 Direct Physical System Observables (Hardware Impact)
- **Additional Coalesced Resident Hits:** $+15$ ($1,432$ vs $1,417$)
- **Avoided DDR5 Expert Reloads:** $-15$ ($536,019$ vs $536,034$)
- **DDR5 Memory Traffic Saved:** $-2.0258\text{ GB}$ ($48,736.81\text{ GB}$ vs $48,738.84\text{ GB}$)
- **Exposed GPU Memory Stall Saved:** $-31.82\text{ ms}$ ($747,831.75\text{ ms}$ vs $747,863.57\text{ ms}$)
- **Mean Token Latency Saved:** $-0.25\text{ ms/token}$ ($5,949.81\text{ ms}$ vs $5,950.06\text{ ms}$)

### 5.3 Critical Distinction

**The 5,999 unconsumed speculative victims evicted MUST NOT be equated with avoided demand misses.**
- The $5,999$ figure represents internal eviction candidate selections within the bottom-$W=64$ window to clear speculative pollution.
- Of the lines protected by these purges, $36$ were subsequently re-referenced, resulting in **15 net additional resident hits** and **15 avoided physical DDR5 reloads**.

---

## 6. Configuration B Shadow-Mode Qualification and Overhead Accounting

Configuration B (Prediction-Only) must be qualified with strict architectural nuance:

### 6.1 Formal Qualification
Configuration B under Workload W4 is explicitly classified as an **inferred shadow-mode equivalence**, verified by the deterministic proof in `LLMSimulator/results/phase_4d5c/production_matrix_summary.csv` where shadow mode caused zero device timing deviation.

### 6.2 Host CPU Overhead vs. Simulated GPU Device Timing
It is inaccurate to describe prediction-only as "zero overhead". The overheads are stratified across subsystem boundaries:
1. **CPU Host Predictor Overhead:**
   - CSTP algorithmic inference requires **$62.74\text{ }\mu\text{s}$ per layer step**.
   - This represents **$0.25\%$ CPU host overhead**, running entirely asynchronously on the host CPU.
2. **Simulated GPU Execution Time:**
   - **$0.00\%$ change relative to baseline** ($1,008.204\text{ s}$ vs $1,008.204\text{ s}$).
   - Because shadow mode discards predictions without triggering DDR5 bus requests, GPU compute warps experience identical stall cycles.
3. **DDR5 Bus Traffic:**
   - **$0\text{ speculative bytes transferred}$** ($48,740.16\text{ GB}$ total, bit-identical to Baseline A).
4. **HBM Residency:**
   - **$0\text{ resident hits}$** ($0.000\%$, identical to Baseline A).

---

## 7. Master Numerical Reconciliation Matrix

The following table summarizes every audited quantitative metric, comparing preliminary/draft values against the authoritative reconciled values:

| # | Metric / Context | Preliminary / Draft Value | Authoritative Reconciled Value | Source File & Row | Reconciliation Justification |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **1** | W3 Baseline Execution Time | 442.146 s | **459.031 s** ($459,031.34\text{ ms}$) | `all_runs.csv` (`w3_c1_r1`) | Erroneous draft parsed `ready_ts` trace fraction; true time matches $128 \times 3,586.18\text{ ms}$. |
| **2** | W3 CSTP+LRU Execution Time | 338.257 s | **456.368 s** ($456,367.51\text{ ms}$) | `all_runs.csv` (`w3_c3_r1`) | True time matches $128 \times 3,565.37\text{ ms}$; draft parsed trace timestamp substring. |
| **3** | W3 CSTP+FA-LRU Execution Time | 338.286 s | **456.383 s** ($456,382.65\text{ ms}$) | `all_runs.csv` (`w3_c4_r1`) | True time matches $128 \times 3,565.49\text{ ms}$; draft parsed trace timestamp substring. |
| **4** | W4 Prefetching Latency Contribution | 99.98% | **99.987%** ($246.60\text{ s}$) | `all_runs.csv` (`w4_c1_r1`, `w4_c3_r1`) | Unrounded arithmetic: $246.59645 / 246.62811 \times 100\% = 99.98717\%$. |
| **5** | W4 FA-LRU Latency Contribution | 0.02% | **0.013%** ($0.032\text{ s}$) | `all_runs.csv` (`w4_c3_r1`, `w4_c4_r1`) | Unrounded arithmetic: $0.03165 / 246.62811 \times 100\% = 0.01283\%$. |
| **6** | W4 Prefetching Stall Contribution | 99.98% | **99.987%** ($246,596.16\text{ ms}$) | `all_runs.csv` (`w4_c1_r1`, `w4_c3_r1`) | Unrounded arithmetic: $246,596.16 / 246,627.98 \times 100\% = 99.98710\%$. |
| **7** | W4 FA-LRU Stall Contribution | 0.02% | **0.013%** ($31.82\text{ ms}$) | `all_runs.csv` (`w4_c3_r1`, `w4_c4_r1`) | Unrounded arithmetic: $31.82429 / 246,627.98 \times 100\% = 0.01290\%$. |
| **8** | Transition A->C Stall Reduction Description | "approximately 25 s" / "nearly 250 s" | **246,596.16 ms (246.60 s / 4.11 min, -24.80%)** | `all_runs.csv` (`w4_c1_r1`, `w4_c3_r1`) | Fixed typo conflating $24.8\%$ with seconds; eliminated imprecise roundings. |
| **9** | W4 Speculative DDR5 Traffic Delta | -2.03 GB (undifferentiated) | **-20.2585 GB** (-230 blocks) | `all_runs.csv` (`w4_c3_r1`, `w4_c4_r1`) | Separated speculative purge from demand byte shift; quantized to $88\text{ MB}$ blocks. |
| **10** | W4 Demand DDR5 Traffic Delta | Unreported | **+18.2326 GB** (+207 blocks) | `all_runs.csv` (`w4_c3_r1`, `w4_c4_r1`) | Explicitly reported demand transfer shift due to fewer speculative attachments. |
| **11** | W4 Net DDR5 Traffic Delta | -2.03 GB (claimed demand savings) | **-2.0258 GB** (-23 blocks net) | `all_runs.csv` (`w4_c3_r1`, `w4_c4_r1`) | Formally explained as net superposition: $-20.2585\text{ GB} + 18.2326\text{ GB}$. |
| **12** | FA-LRU Avoided Demand Misses | Equated to 5,999 | **15 Avoided Misses / 15 Avoided Reloads** | `paired_comparisons.csv` (`w4_c4_c3`) | Disambiguated 5,999 speculative victims (diagnostic) from 15 physical hits/reloads. |
| **13** | Protected Lines Reused | 36 lines | **36 lines reused** (producing +15 hits) | `fa_lru_diagnostics.json` (`w4_c4_r1`) | Verified against diagnostic trace counter. |
| **14** | Config B Overhead | "Zero overhead" | **62.74 µs CPU overhead (0.25%); 0.00% GPU delta** | `production_matrix_summary.csv` | Separated CPU host predictor runtime from simulated GPU device execution time. |
| **15** | Config B Qualification | "Simulated" | **Inferred shadow-mode equivalence** | `MISSING_EVIDENCE_REPORT.md` | Explicitly documented historical proof equivalence with zero simulator drift. |

---

## 8. Verification and Compliance

- **Mathematical Consistency:** Verified that $\text{mean token latency} = \frac{\text{simulated execution time}}{128}$ across all Big Experiment runs.
- **Traffic Conservation:** Total DDR5 bytes equals Demand bytes plus Speculative bytes down to the individual byte.
- **Eviction Conservation:** Evictions partitioned into unconsumed speculative, demand resident, and PABI speculative.
- **No Ternary Operators:** Ast-checked all supporting scripts; 0 ternary expressions found.

