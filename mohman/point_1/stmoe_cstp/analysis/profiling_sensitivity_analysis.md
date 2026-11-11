# Profiling Percentage Sensitivity Analysis

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Deliverable:** Point 1 — Comparative Study Analysis  
**Path:** [`mohman/point_1/stmoe_cstp/analysis/profiling_sensitivity_analysis.md`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/analysis/profiling_sensitivity_analysis.md)  
**Authoritative Data Sources:** [`results/csv/count_statistics.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/count_statistics.csv) & [`results/csv/aggregate_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/aggregate_metrics.csv)  

---

## 1. Offline Sample Efficiency Dynamics

To determine how much offline routing data is necessary to build a reliable predictor, both algorithms were profiled using $25\%, 50\%, 75\%$, and $100\%$ of the offline training split across 3 random seeds.

### Scaling Trajectory on MoE Anywhere ($H=1$, Seed 42):

```
Recall (%)
  48% |                                            * (47.91% ST-MoE Online)
      |                             *              |
  46% |              *                             * (45.51% CSTP Online)
      |              |              * (45.48%)     |
  44% |  *           * (44.85%)     |              * (44.09% ST-MoE Frozen)
      |  |           |              * (44.01%)
  42% |  * (42.79%)  * (43.47%)
      +--------------+--------------+--------------+
            25%            50%            75%            100% Profiling
```

### Key Analytical Takeaways:
1. **Logarithmic Saturation of Frozen Recall:** On MoE Anywhere, frozen recall exhibits steep initial gains (+0.70 pp from 25% to 50%) followed by sharp flattening (+0.10 pp from 75% to 100%). Over 98% of maximum frozen performance is captured using only 50% of profiling data.
2. **Count Mass Growth:** Total spatial count mass grows strictly linearly with profiling percentage ($309,600 \to 619,200 \to 928,800 \to 1,238,400$).
3. **Table Sparsity:** Active non-zero transition entries expand rapidly at low profiling fractions and saturate as new tokens repeatedly exercise the primary routing paths.

