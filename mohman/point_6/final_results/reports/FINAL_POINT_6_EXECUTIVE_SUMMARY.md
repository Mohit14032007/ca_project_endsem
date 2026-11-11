# Point 6 — Final Executive Summary: Comprehensive MoE Sensitivity Analysis

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Sensitivity Analysis Consolidation  
**Author:** Antigravity Engineering  
**Date:** November 10, 2026  
**Status:** Authoritative & Completed  

---

## 1. Executive Summary

This executive summary encapsulates the complete findings, empirical measurements, and architectural conclusions of **Point 6 (Sensitivity Analysis)** for the MoE Memory Wall project.

Using `LLMSimulator` with authentic DeepSeek-V3 routing traces (40 MoE stages, 256 routed experts/stage, Top-8 routing), we conducted an exhaustive, multi-dimensional sensitivity evaluation exploring:
1. **Serving Batch Concurrency ($B \in \{4, 14, 32\}$)**
2. **Initial Context Length ($P \in \{128, 512, 2048, 4096\}$)**
3. **Inter-Layer Lookahead Distance ($H \in \{1, 2, 4\}$)**
4. **Prefetch Candidate Aggressiveness / Budget ($K \in \{2, 4, 8\}$)**
5. **HBM Cache Replacement Policies (Standard LRU vs. Frequency-Aware LRU $W=64$)**

---

## 2. Core Sensitivity Findings Matrix

```
========================================================================================================================
                                     POINT 6 MULTI-DIMENSIONAL SENSITIVITY MATRIX
========================================================================================================================
 Dimension & Sweep     | Parameter Range | Optimal Setting | Observed Sensitivity Dynamics & System Effects
-----------------------+-----------------+-----------------+------------------------------------------------------------
 1. Batch Size ($B$)   | $B \in [4, 32]$ | $B = 32$        | Concurrency increases reuse and working set; transitions from
                       |                 |                 | degradation at B=4 (-0.13%) to acceleration at B=32 (+0.73%).
 2. Context Length ($P$)| $P \in [128,4096]$| $P \ge 2048$  | Attention duration scales with P (13.7s -> 14.3s); expands
                       |                 |                 | overlap slack, increasing saved reloads (1,317 -> 1,528).
 3. Lookahead ($H$)    | $H \in [1, 4]$  | $H = 4$         | Lead time converts late attachments into hits (+49.7% hits);
                       |                 |                 | hides an extra 1,434 ms of stall with CSTP predictor.
 4. Prefetch Budget ($K$)| $K \in [2, 8]$| $K = 2$         | K=2 avoids DDR5 bus congestion, achieving peak speedup
                       |                 |                 | (+1.42%, 10.94s stall hidden); K=8 saturates the bus.
 5. Cache Eviction     | LRU vs FA-LRU64 | FA-LRU ($W=64$) | FA-LRU protects frequently reused demand experts in bottom-64
                       |                 |                 | window, saving 15–22 reloads across all workloads at B=32.
========================================================================================================================
```

---

## 3. The Three Definitive Operating Regimes

### Regime 1: Accelerated Serving (Speedup: $+0.58\%$ to $+1.42\%$)
- **Operating Conditions:** High Concurrency ($B \ge 14$), Conservative Budget ($K = 2\text{ to }4$), Multi-Layer Lookahead ($H \ge 1$).
- **Empirical Evidence:**
  - At $B=14$: Hides $2,693.71\text{ ms}$ of memory stall, saving $1,564$ reloads.
  - At $B=32, K=4$: Hides $5,630.28\text{ ms}$ of memory stall, saving $1,398$ reloads.
  - At $B=32, K=2$: Hides $10,944.92\text{ ms}$ of memory stall, saving $3,287$ reloads and achieving $+1.42\%$ latency speedup.
- **Physical Reason:** High token concurrency generates repeated demand requests across sequences. Prefetching overlaps data movement with layer compute without bus saturation.

### Regime 2: Compute-Bound Plateau (Plateau at $\approx +0.77\%$)
- **Operating Conditions:** Long Context Length ($P \ge 2048$), High Attention Complexity.
- **Empirical Evidence:**
  - Attention stage duration scales from $13,727.88\text{ ms}$ at $P=128$ to $14,355.67\text{ ms}$ at $P=4,096$.
  - Coalesced resident hits increase by $+15.0\%$ (from $1,417$ to $1,630$).
  - However, because decode attention FLOPs dominate total runtime, latency speedup asymptotically plateaus at $+0.769\%$.
- **Physical Reason:** The prefetch overlap slack widens, but total runtime becomes compute-bound rather than memory-bound.

### Regime 3: Performance Inversion / Cache Degradation (Degradation: $-0.026\%$ to $-0.13\%$)
- **Operating Conditions:** Low Concurrency ($B \le 4$) OR Over-Aggressive Prefetch Budget ($K \ge 8$).
- **Empirical Evidence:**
  - At $B=4$: Speculative prefetches pollute the cache, prematurely evicting demand experts. Net reloads increase by $+301$, and latency degrades by $-0.130\%$.
  - At $K=8$: Speculative transfers inject $18.04\text{ TB}$ (204,780 prefetches) onto the 64 GB/s interconnect, causing controller queuing contention that delays demand fetches, increasing stall time by $+178.24\text{ ms}$ and degrading latency by $-0.026\%$ relative to $K=4$.
- **Physical Reason:** When concurrency is too low to sustain reuse, or when speculative bandwidth exceeds channel capacity, prefetching incurs severe net negative utility.

---

## 4. Key Artifact Locations

- **Consolidated Master CSV:** [`consolidated_data/unified_point6_results.csv`](file:///home/mohman/ca_project/mohman/point_6/final_results/consolidated_data/unified_point6_results.csv)
- **Paired Comparisons:** [`consolidated_data/paired_comparisons.csv`](file:///home/mohman/ca_project/mohman/point_6/final_results/consolidated_data/paired_comparisons.csv)
- **Source Manifest:** [`consolidated_data/source_manifest.csv`](file:///home/mohman/ca_project/mohman/point_6/final_results/consolidated_data/source_manifest.csv)
- **Core Technical Report:** [`reports/POINT_6_SENSITIVITY_ANALYSIS.md`](file:///home/mohman/ca_project/mohman/point_6/final_results/reports/POINT_6_SENSITIVITY_ANALYSIS.md)
- **15 Publication Figures:** [`figures/`](file:///home/mohman/ca_project/mohman/point_6/final_results/figures/) (in both 300 DPI PNG and PDF)
