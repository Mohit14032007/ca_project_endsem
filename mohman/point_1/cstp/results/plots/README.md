# CSTP Certified Publication Figures Catalog

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Deliverable:** Point 1 — Standalone CSTP Figures Package  
**Directory:** [`mohman/point_1/cstp/results/plots/`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/plots/)  
**Formats:** Every figure is generated and certified in both 300 DPI PNG (`.png`) and publication-ready vector PDF (`.pdf`).  

---

## 1. Visual Catalog & Scientific Insights

### Figure 1: Frozen Recall@8 vs. Profiling Percentage
- **Files:** [`fig1_certified_frozen_recall_vs_profiling.png`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/plots/fig1_certified_frozen_recall_vs_profiling.png) / [`.pdf`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/plots/fig1_certified_frozen_recall_vs_profiling.pdf)
- **Data Source:** [`corrected_aggregate_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_aggregate_metrics.csv)
- **Visual Description:** Scaling curves showing frozen recall as offline profiling increases from 25% to 100% across MoE Anywhere and MoE Suite.
- **Key Insight:** On MoE Anywhere, frozen recall scales smoothly from 43.83% to 45.37%, demonstrating diminishing returns beyond 50% profiling. On MoE Suite, the 75% configuration peaks at 18.03% due to task-specific alignment.

---

### Figure 2: Online Within-Request Recall@8 vs. Profiling Percentage
- **Files:** [`fig2_certified_online_recall_vs_profiling.png`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/plots/fig2_certified_online_recall_vs_profiling.png) / [`.pdf`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/plots/fig2_certified_online_recall_vs_profiling.pdf)
- **Data Source:** [`corrected_aggregate_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_aggregate_metrics.csv)
- **Visual Description:** Curves of online recall under the primary `online_within_request` standard across profiling fractions.
- **Key Insight:** Online adaptation maintains high absolute recall across both horizons ($H=1$ and $H=2$), reaching 45.51% on MoE Anywhere.

---

### Figure 3: Online Adaptation Gain Attenuation
- **Files:** [`fig3_certified_online_adaptation_gain.png`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/plots/fig3_certified_online_adaptation_gain.png) / [`.pdf`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/plots/fig3_certified_online_adaptation_gain.pdf)
- **Data Source:** [`corrected_aggregate_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_aggregate_metrics.csv)
- **Visual Description:** Bar chart of online adaptation gain ($R_{\text{online}} - R_{\text{frozen}}$) across profiling levels.
- **Key Insight:** Clearly illustrates the count-mass saturation law: online gain drops monotonically from +0.85 pp at 25% down to +0.14 pp at 100% on MoE Anywhere.

---

### Figure 4: Head-to-Head Recall Comparison (CSTP vs. ST-MoE)
- **Files:** [`fig4_certified_stmoe_vs_cstp_head_to_head.png`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/plots/fig4_certified_stmoe_vs_cstp_head_to_head.png) / [`.pdf`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/plots/fig4_certified_stmoe_vs_cstp_head_to_head.pdf)
- **Data Source:** [`corrected_st_moe_head_to_head.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_st_moe_head_to_head.csv)
- **Visual Description:** Grouped bar charts comparing CSTP and ST-MoE under matched conditions for both Frozen and Online regimes.
- **Key Insight:** CSTP holds a consistent frozen recall advantage (+1.04 to +1.28 pp on MoE Anywhere), while ST-MoE achieves higher online gain due to its bounded confidence-counter updates.

---

### Figure 5: Lookahead Horizon Comparison ($H=1$ vs. $H=2$)
- **Files:** [`fig5_certified_horizon1_vs_horizon2.png`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/plots/fig5_certified_horizon1_vs_horizon2.png) / [`.pdf`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/plots/fig5_certified_horizon1_vs_horizon2.pdf)
- **Data Source:** [`corrected_aggregate_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_aggregate_metrics.csv)
- **Visual Description:** Paired bar charts comparing single-stage ($H=1$) against skip-stage ($H=2$) lookahead.
- **Key Insight:** CSTP retains $> 97\%$ of its predictive recall at $H=2$ (dropping only from 45.51% to 44.24% on MoE Anywhere), verifying that two-stage lookahead prefetching is viable.

---

### Figure 6: Profiling Random Seed Robustness
- **Files:** [`fig6_certified_seed_variability.png`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/plots/fig6_certified_seed_variability.png) / [`.pdf`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/plots/fig6_certified_seed_variability.pdf)
- **Data Source:** [`corrected_aggregate_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_aggregate_metrics.csv)
- **Visual Description:** Error bars and dispersion points across random profiling seeds (42, 123, 999).
- **Key Insight:** Seed-induced variance is negligible ($\sigma \le 0.21\text{ pp}$), establishing that CSTP's transition distributions are stable and reproducible.

---

### Figure 7: MoE Suite Task-Level Breakdown
- **Files:** [`fig7_certified_moe_suite_task_level.png`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/plots/fig7_certified_moe_suite_task_level.png) / [`.pdf`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/plots/fig7_certified_moe_suite_task_level.pdf)
- **Data Source:** [`corrected_per_task_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_per_task_metrics.csv)
- **Visual Description:** Decomposed bar charts isolating `gsm8k` vs `alpaca` performance across profiling fractions.
- **Key Insight:** Explains the 75% vs 100% anomaly by showing how `gsm8k` recall drops when the 100% split incorporates discordant out-of-domain training traces.

---

### Figure 8: Transition Count-Mass Growth
- **Files:** [`fig8_certified_transition_count_growth.png`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/plots/fig8_certified_transition_count_growth.png) / [`.pdf`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/plots/fig8_certified_transition_count_growth.pdf)
- **Data Source:** [`corrected_profiling_count_statistics.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_profiling_count_statistics.csv)
- **Visual Description:** Dual-axis chart tracking total accumulated count mass and non-zero matrix sparsity vs profiling percentage.
- **Key Insight:** Count mass grows strictly linearly ($R^2 = 1.0$), demonstrating why Dirichlet smoothing denominators grow and diminish online adaptation.

---

### Figure 9: Certified Hierarchical Bootstrap 95% Confidence Intervals
- **Files:** [`fig9_certified_confidence_intervals.png`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/plots/fig9_certified_confidence_intervals.png) / [`.pdf`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/plots/fig9_certified_confidence_intervals.pdf)
- **Data Source:** [`corrected_statistical_analysis.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_statistical_analysis.csv)
- **Visual Description:** Forest plot displaying 95% hierarchical bootstrap confidence intervals for primary hypothesis contrasts.
- **Key Insight:** Confirms that online adaptation on MoE Anywhere is statistically significant ($p < 0.001$, CI strictly above 0), while MoE Suite contrasts are explicitly annotated as `DESCRIPTIVE_ONLY_N2_TASKS`.

