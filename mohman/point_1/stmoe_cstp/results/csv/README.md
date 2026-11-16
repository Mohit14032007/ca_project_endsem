# Comparative Study Results: CSV Data Dictionary & Schemas

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Deliverable:** Point 1 — Comparative Study Results Package  
**Directory:** [`mohman/point_1/stmoe_cstp/results/csv/`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/)  
**Baseline Git Anchor:** `d5bf61ff0c066c06f72173189f9924329f613a88`  

---

## 1. Directory Overview

This directory houses the authoritative comparative datasets evaluating the original CSTP predictor against the paper-faithful ST-MoE baseline under strictly matched experimental conditions.

| Filename | Row Count | File Size | Description |
| :--- | :---: | :---: | :--- |
| [`matched_comparisons.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/matched_comparisons.csv) / [`corrected_st_moe_head_to_head.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/corrected_st_moe_head_to_head.csv) | 48 | 7.3 KB | Direct matched head-to-head comparison between CSTP and ST-MoE across all configurations. |
| [`aggregate_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/aggregate_metrics.csv) / [`corrected_aggregate_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv) | 192 | 33 KB | Master table of all 192 configurations across both predictors, workloads, horizons, seeds, and regimes. |
| [`count_statistics.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/count_statistics.csv) / [`corrected_profiling_count_statistics.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/corrected_profiling_count_statistics.csv) | 48 | 7.6 KB | Spatial and temporal count-mass statistics quantifying saturation dynamics. |
| [`statistical_analysis.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/statistical_analysis.csv) / [`corrected_statistical_analysis.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/corrected_statistical_analysis.csv) | 130 | 35 KB | 95% bootstrap confidence intervals, bootstrap probabilities, and two-sided ASL values. |
| [`per_task_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/per_task_metrics.csv) | 192 | 26 KB | Task-level metrics isolating `gsm8k` vs `alpaca` on MoE Suite. |
| [`per_prompt_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/per_prompt_metrics.csv) | 1,344 | 184 KB | Granular prompt-level metric breakdown for distribution analysis. |

---

## 2. Key Comparative Metrics Explained

- `delta_frozen_recall`: $R_{\text{CSTP, frozen}} - R_{\text{ST-MoE, frozen}}$
- `delta_online_recall`: $R_{\text{CSTP, online}} - R_{\text{ST-MoE, online}}$
- `cstp_online_gain`: $R_{\text{CSTP, online}} - R_{\text{CSTP, frozen}}$
- `stmoe_online_gain`: $R_{\text{ST-MoE, online}} - R_{\text{ST-MoE, frozen}}$
- `delta_online_hit_rate`: $\text{HitRate}_{\text{CSTP, online}} - \text{HitRate}_{\text{ST-MoE, online}}$

