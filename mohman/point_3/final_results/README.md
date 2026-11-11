# Point 3 — Prefetching Effectiveness: Final Consolidated Results and Analysis Directory

**Repository:** [`/home/mohman/ca_project/LLMSimulator`](file:///home/mohman/ca_project/LLMSimulator)  
**Deliverable Path:** [`mohman/point_3/final_results/`](file:///home/mohman/ca_project/mohman/point_3/final_results/)  
**Status:** Complete, Self-Contained, Publication-Ready  
**Evaluation Dates:** November 2026  

---

## 1. Purpose of This Directory

This directory is the single, self-contained, authoritative repository for **Point 3 (Prefetching Effectiveness)** of the project:  
**"MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference"**.

It consolidates all configurations, original source CSVs, consolidated tables, publication-quality visualizations, validation audits, and technical documentation across all executed Point 3 experiment campaigns. Researchers and peer reviewers can directly inspect, reproduce, and verify the performance of:
1. **Demand-Loading Baseline** (no speculative prefetching)
2. **ST-MoE Predictor** (Correlation-based Cross-stage Transition Matrix)
3. **CSTP Predictor** (Contextual Spatio-Temporal Predictor with Dirichlet Smoothing)

---

## 2. Experiment Campaigns Included

A total of **73 simulation runs** across three distinct experimental campaigns are preserved, consolidated, and analyzed in this folder:

```
========================================================================================================================
                                             POINT 3 EXPERIMENTAL CAMPAIGNS
========================================================================================================================
Campaign Name             Runs  MoE Stages  Batch Sizes (B)     Tokens (T)  HBM Cache   Primary Objective
------------------------------------------------------------------------------------------------------------------------
1. Large-Workload         27    40 Stages   14, 32, 48, 70      4, 16, 32,  35 GB       High-concurrency serving, deep
   Phase 2 Campaign             (0 to 39)                       64                      auto-regressive decode, capacity-
                                                                                        constrained cache churn (W1 to W7)
------------------------------------------------------------------------------------------------------------------------
2. Original Sensitivity   20    3 Stages    1, 2, 4, 8          4           80 GB       Initial sensitivity campaign,
   Campaign                     (0 to 2)                                                small batch scaling, baseline
                                                                                        characterization
------------------------------------------------------------------------------------------------------------------------
3. Phase 2 Scaling        26    40 Stages   1, 2, 4, 8          4, 8        30 GB /     Multi-stage depth scaling, cache
   Campaign                     (0 to 39)                                   80 GB       capacity sweeps (25, 35, 50, 80 GB)
========================================================================================================================
Total Runs Included: 73 runs (100% verified exit code 0)
```

---

## 3. Directory Structure

```text
final_results/
│
├── README.md                               <-- Primary entry point and user guide
│
├── experiment_details/
│   ├── EXPERIMENT_INDEX.md                 <-- Full catalog of all 73 simulation runs
│   ├── WORKLOAD_MATRIX.md                  <-- Parametric breakdown across W1-W7, original, and P2
│   ├── CONFIGURATION_DETAILS.md            <-- Architectural, hardware, and predictor hyperparameters
│   ├── METRIC_DEFINITIONS.md               <-- Formal mathematical definitions of all 8 mandatory metrics
│   └── RESULTS_AND_OBSERVATIONS.md         <-- Consolidated empirical findings and detailed analysis
│
├── configs/
│   ├── baseline/                           <-- 17 baseline YAML configurations
│   ├── st_moe/                             <-- 28 ST-MoE YAML configurations (H=1, H=2)
│   └── cstp/                               <-- 28 CSTP YAML configurations (H=1, H=2)
│
├── data/
│   ├── original_csv/                       <-- 18 uncompressed original source CSV files
│   ├── consolidated_csv/                   <-- 4 unified multi-campaign comparison CSV files
│   └── derived_csv/                        <-- Head-to-head, scaling, and trade-off comparison tables
│
├── plots/
│   ├── 01_total_time_vs_stall_reduction/   <-- Mandatory trade-off & per-workload plots (11 PNG/PDF)
│   ├── 02_prefetch_coverage/               <-- Strict timely (0.0%) vs associated coverage (3 PNG/PDF)
│   ├── 03_timeliness/                      <-- Timely vs late rates, bus transit time (2 PNG/PDF)
│   ├── 04_unused_prefetches/               <-- Unused prefetch rates and wasted bytes (2 PNG/PDF)
│   ├── 05_hbm_hit_rate/                    <-- HBM hit rate invariance analysis (2 PNG/PDF)
│   ├── 06_memory_traffic/                  <-- DDR5 traffic breakdowns & net extra bytes (3 PNG/PDF)
│   ├── 07_exposed_memory_stall/            <-- Exposed GPU stalls and reduction scaling (3 PNG/PDF)
│   ├── 08_expert_load_latency/             <-- Expert load service times & perceived savings (2 PNG/PDF)
│   ├── 09_end_to_end_latency/              <-- Mean decode step and tail p95 latency (3 PNG/PDF)
│   ├── 10_throughput/                      <-- Tokens/second aggregate throughput scaling (2 PNG/PDF)
│   └── comparison_dashboards/              <-- Publication multi-panel overview dashboards (3 PNG/PDF)
│
├── analysis/
│   ├── PLOT_CATALOG.md                     <-- Visual index and interpretation guide for all 36 plots
│   ├── DATA_VALIDATION.md                  <-- Forensic reconciliation of experimental evidence
│   ├── METRIC_RECONCILIATION.md            <-- Byte conservation and stall-to-latency math derivations
│   └── KEY_FINDINGS.md                     <-- High-level scientific conclusions and architectural insights
│
└── scripts/
    ├── consolidate_results.py              <-- Script to re-generate all consolidated CSVs
    ├── generate_plots.py                   <-- Script to re-generate all 36 PNG and PDF figures
    └── validate_results.py                 <-- Comprehensive consistency and AST zero-ternary validator
```

---

## 4. Recommended Reading Order

To navigate the experimental evidence efficiently, read the documents in the following order:

1. **[`README.md`](file:///home/mohman/ca_project/mohman/point_3/final_results/README.md)** (This document: Overview and navigation).
2. **[`experiment_details/EXPERIMENT_INDEX.md`](file:///home/mohman/ca_project/mohman/point_3/final_results/experiment_details/EXPERIMENT_INDEX.md)** (Full catalog of all 73 runs).
3. **[`experiment_details/WORKLOAD_MATRIX.md`](file:///home/mohman/ca_project/mohman/point_3/final_results/experiment_details/WORKLOAD_MATRIX.md)** (Workload parameters and serving regimes).
4. **[`experiment_details/METRIC_DEFINITIONS.md`](file:///home/mohman/ca_project/mohman/point_3/final_results/experiment_details/METRIC_DEFINITIONS.md)** (Mathematical formulas and event hooks).
5. **[`experiment_details/RESULTS_AND_OBSERVATIONS.md`](file:///home/mohman/ca_project/mohman/point_3/final_results/experiment_details/RESULTS_AND_OBSERVATIONS.md)** (Detailed quantitative results).
6. **[`analysis/PLOT_CATALOG.md`](file:///home/mohman/ca_project/mohman/point_3/final_results/analysis/PLOT_CATALOG.md)** (Visual gallery and plot interpretations).
7. **[`analysis/DATA_VALIDATION.md`](file:///home/mohman/ca_project/mohman/point_3/final_results/analysis/DATA_VALIDATION.md)** (Independent verification checks).
8. **[`analysis/METRIC_RECONCILIATION.md`](file:///home/mohman/ca_project/mohman/point_3/final_results/analysis/METRIC_RECONCILIATION.md)** (Traffic reconciliation and timing proof).
9. **[`analysis/KEY_FINDINGS.md`](file:///home/mohman/ca_project/mohman/point_3/final_results/analysis/KEY_FINDINGS.md)** (Architectural insights and future directions).

---

## 5. How to Regenerate Artifacts

All consolidated CSVs and publication figures can be re-generated deterministically from original data:

```bash
# 1. Consolidate CSV results from original data
python3 final_results/scripts/consolidate_results.py

# 2. Generate all 36 plots in PNG and vector PDF
python3 final_results/scripts/generate_plots.py

# 3. Run validation suite and AST zero-ternary compliance
python3 final_results/scripts/validate_results.py
```

---

## 6. Known Limitations and Resolved Discrepancies Summary

1. **Strict Timely Coverage = 0.00%:** Across all 73 runs, zero speculative transfers completed before demand. An 88.08 MB transfer over 64 GB/s DDR5 requires **1.376 ms**, whereas inter-stage GPU computation lasts only **20 to 50 microseconds**.
2. **Late In-Flight Attachment Utility:** Prefetching reduced exposed GPU memory stalls (by up to **6.745 seconds** in W7) because in-flight transfers truncate wait times upon demand arrival.
3. **CSTP Superiority via Queue Contention Avoidance:** CSTP achieved **2.94x greater stall reduction** than ST-MoE while dispatching **47.4% fewer transfers** ($54,076$ vs $102,737$). ST-MoE caused queue head-of-line blocking.
4. **Resolution of 110.10 GB vs 122.96 GB Discrepancy:** Confirmed **110.10 GB** as the physically exact DDR5 traffic savings ($1,250$ physical expert loads saved $\times 88.08 \text{ MB}$). The 122.96 GB figure was an uncoalesced nominal figure derived from $1,396$ raw tensor lookups.
5. **Decode Step Latency Reconciliation:** The 6.717s simulation time reduction divided across 64 generation steps yields exactly **104.95 ms per decode step**, matching the mean token latency metric.

