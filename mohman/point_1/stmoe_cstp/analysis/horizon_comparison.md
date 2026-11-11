# Lookahead Horizon Comparison ($H=1$ vs. $H=2$)

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Deliverable:** Point 1 — Comparative Study Analysis  
**Path:** [`mohman/point_1/stmoe_cstp/analysis/horizon_comparison.md`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/analysis/horizon_comparison.md)  
**Authoritative Data Sources:** [`results/csv/matched_comparisons.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/matched_comparisons.csv) & [`results/csv/aggregate_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/aggregate_metrics.csv)  

---

## 1. Horizon Scaling Evaluation

Prefetching $H=2$ layers in advance doubles the available prefetch latency slack (allowing the prefetcher to overlap expert weight transfer across two full transformer layer compute times). However, multi-stage correlation naturally decays as lookahead distance increases.

### Comparative Horizon Retention (Seed 42, 100% Profiling, Online Within-Request):

| Workload | Predictor | Horizon $H=1$ Recall | Horizon $H=2$ Recall | Absolute Loss (pp) | Relative Retention (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MoE Anywhere** | **CSTP** | **45.51%** | **44.24%** | -1.27 pp | **97.21%** |
| **MoE Anywhere** | **ST-MoE** | **47.91%** | **46.20%** | -1.71 pp | **96.43%** |
| **MoE Suite** | **CSTP** | **16.59%** | **16.36%** | -0.23 pp | **98.61%** |
| **MoE Suite** | **ST-MoE** | **26.53%** | **25.90%** | -0.63 pp | **97.63%** |

---

## 2. Key Findings

1. **High Predictive Retention Across Horizons:** Both predictors retain $> 96\%$ of their adjacent-layer ($H=1$) predictive accuracy when tasked with skip-layer ($H=2$) lookahead.
2. **CSTP Superior Horizon Robustness:** On MoE Anywhere, CSTP drops by only 1.27 pp (-2.79% relative), compared to ST-MoE's 1.71 pp drop (-3.57% relative). This is because CSTP's stage marginal frequency prior and temporal recurrence components are independent of spatial distance $H$.
3. **Architectural Feasibility:** The minimal degradation observed at $H=2$ demonstrates that hardware prefetch schedulers can safely prefetch two layers ahead, effectively doubling the PCIe transfer window while preserving prediction accuracy.

