# Point 6 — Historical Experiment Extraction Notes and Methodology

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Sensitivity Analysis Consolidation  
**Author:** Antigravity Engineering  
**Date:** November 10, 2026  
**Status:** Authoritative & Completed  

---

## 1. Executive Summary

To support the Point 6 Sensitivity Analysis without redundant re-simulation, this directory consolidates all relevant historical experiment results collected across Points 3, 4, 5, and 6. 

All extracted datasets originate from verified, version-controlled project artifacts:
1. **`batch_size_historical_results.csv`:** Historical batch size evaluations ($B \in \{14, 32, 48, 70\}$ with $T=16$ from Point 3; and $B \in \{14, 32\}$ with $T=128$ from Point 4 Big Experiment).
2. **`context_length_extraction.csv`:** 74 authentic prompt records from MoE-Anywhere (`e1-sage-a1.5.parquet`) with inferred initial context lengths ($P \in [13, 64]$) and 5-gate structural validation flags.
3. **`predictor_historical_results.csv`:** CSTP and ST-MoE spatial/temporal prediction accuracy, precision, recall, and Top-K coverage across workloads.
4. **`prefetch_historical_results.csv`:** Prefetch timeliness, coverage, unused transfers, and demand-miss avoidance metrics.
5. **`eviction_historical_results.csv`:** Standard LRU versus Frequency-Aware LRU (FA-LRU $W=16$ and $W=64$) eviction diagnostics and hit rates.
6. **`historical_source_manifest.csv`:** Provenance mapping linking each historical run ID to its exact configuration, workload, and source file.

---

## 2. Comparability and Methodological Caveats

To prevent invalid scientific claims, historical results are categorized by their degree of comparability to Point 6's reference configuration:
- **Reference Configuration:** DeepSeek-V3 (40 MoE stages, 256 experts, Top-8 routing), HBM Capacity = 35 GiB ($37,580,963,840$ bytes), DDR5 Bandwidth = 64 GB/s, Generation Horizon $T = 128$, CSTP Predictor ($H=1, K=4$), Forward Lookahead scheduling.

### 2.1 Directly Comparable Runs (Reusable in Point 6)
The following runs from the Point 4 Big Experiment match the Point 6 reference configuration across every architectural dimension at $P=128, T=128$:
- **`w4_c1_r1`, `w4_c1_r2`:** Batch size $B=32$, Context $P=128$, Generation $T=128$, HBM 35 GiB, Baseline LRU.
- **`w4_c3_r1`, `w4_c3_r2`:** Batch size $B=32$, Context $P=128$, Generation $T=128$, HBM 35 GiB, CSTP + LRU ($H=1, K=4$).
- **`w4_c4_r1`, `w4_c4_r2`:** Batch size $B=32$, Context $P=128$, Generation $T=128$, HBM 35 GiB, CSTP + FA-LRU ($W=64, H=1, K=4$).
- **`w3_c1` to `w3_c4`:** Batch size $B=14$, Context $P=128$, Generation $T=128$, HBM 35 GiB (provides $B=14$ reference at $P=128$).

### 2.2 Partially Comparable Runs (Historical Supporting Context Only)
The following runs explore relevant dimensions but cannot be merged into direct paired comparisons without explicitly noting parameter divergence:
1. **Point 3 Phase 2 Large Workloads ($B \in \{14, 32, 48, 70\}$):**
   - **Divergence:** Simulated with generation horizon $T=16$ and unconstrained HBM capacity (80 GB / 74.5 GiB), whereas Point 6 evaluates constrained 35 GiB HBM and $T=128$.
   - **Utility:** Demonstrates cross-sequence expert reuse trends across extreme batch sizes ($B=48, 70$) under cold-cache start.
2. **Point 4 Focused 24 Campaign ($B \in \{1, 2, 4\}$):**
   - **Divergence:** Simulated with short generation horizon $T=20$ across three HBM capacities (10GB, 35GiB, 74.5GiB).
   - **Utility:** Demonstrates low-concurrency cache behavior where batch reuse is absent.
3. **Point 6 Experiment A (Natural Context Lengths $P \in [13, 64]$):**
   - **Divergence:** Disclosed that natural prompt length variation is too narrow ($\Delta P \le 51$) to create measurable memory wall displacement ($\Delta M_{\text{KV}} < 1\text{ expert}$).
   - **Utility:** Provides empirical proof that controlled long-context sweeps ($P \in \{128, 512, 2048, 4096\}$) are required to stress the memory hierarchy.

---

## 3. Data Integrity & Re-simulation Policy

In strict compliance with Part 0 Non-Negotiable Rules:
- Original simulation CSVs and logs have not been modified or overwritten.
- Results from different batch sizes, generation horizons, or HBM capacities are never aggregated without explicit stratification.
- Point 4 Big Experiment runs are reused directly for the $P=128, B=32$ condition, eliminating redundant simulator execution while preserving exact paired comparisons.

