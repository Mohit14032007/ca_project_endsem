# Comparative Visualizations Catalog

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Deliverable:** Point 1 — Comparative Figures Package  
**Directory:** [`mohman/point_1/stmoe_cstp/results/plots/`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/plots/)  
**Formats:** Every figure is generated and certified in both 300 DPI PNG (`.png`) and publication-ready vector PDF (`.pdf`).  

---

## 1. Figure Overview & Comparative Insights

- **[`fig4_certified_stmoe_vs_cstp_head_to_head.png`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/plots/fig4_certified_stmoe_vs_cstp_head_to_head.png) / [`.pdf`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/plots/fig4_certified_stmoe_vs_cstp_head_to_head.pdf):**
  - **Core Visual:** Side-by-side grouped bar charts of Recall@8 for CSTP and ST-MoE under both Frozen and Online Within-Request regimes across profiling fractions.
  - **Finding:** Highlights CSTP's frozen superiority on MoE Anywhere (+1.04 to +1.28 pp) alongside ST-MoE's higher online adaptation agility (+3.82 pp on MoE Anywhere, +8.03 pp on MoE Suite).

- **[`fig3_certified_online_adaptation_gain.png`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/plots/fig3_certified_online_adaptation_gain.png) / [`.pdf`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/plots/fig3_certified_online_adaptation_gain.pdf):**
  - **Core Visual:** Online adaptation gain ($\Delta_{\text{online}} = R_{\text{online}} - R_{\text{frozen}}$) across profiling levels.
  - **Finding:** Shows the contrast between CSTP's count-mass dilution (gain drops from +0.85 pp to +0.14 pp) and ST-MoE's stable counter agility (+4.12 pp to +3.82 pp).

- **[`fig1_certified_frozen_recall_vs_profiling.png`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/plots/fig1_certified_frozen_recall_vs_profiling.png) / [`.pdf`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/plots/fig1_certified_frozen_recall_vs_profiling.pdf):**
  - **Core Visual:** Frozen recall scaling curves from 25% to 100% profiling.

- **[`fig2_certified_online_recall_vs_profiling.png`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/plots/fig2_certified_online_recall_vs_profiling.png) / [`.pdf`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/plots/fig2_certified_online_recall_vs_profiling.pdf):**
  - **Core Visual:** Online recall scaling curves under the primary `online_within_request` standard.

- **[`fig5_certified_horizon1_vs_horizon2.png`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/plots/fig5_certified_horizon1_vs_horizon2.png) / [`.pdf`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/plots/fig5_certified_horizon1_vs_horizon2.pdf):**
  - **Core Visual:** Adjacent ($H=1$) vs Skip-layer ($H=2$) lookahead recall retention.

- **[`fig6_certified_seed_variability.png`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/plots/fig6_certified_seed_variability.png) / [`.pdf`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/plots/fig6_certified_seed_variability.pdf):**
  - **Core Visual:** Seed dispersion across profiling seeds (42, 123, 999).

- **[`fig7_certified_moe_suite_task_level.png`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/plots/fig7_certified_moe_suite_task_level.png) / [`.pdf`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/plots/fig7_certified_moe_suite_task_level.pdf):**
  - **Core Visual:** Task-decomposed recall for `gsm8k` vs `alpaca`.

- **[`fig8_certified_transition_count_growth.png`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/plots/fig8_certified_transition_count_growth.png) / [`.pdf`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/plots/fig8_certified_transition_count_growth.pdf):**
  - **Core Visual:** Linear scaling of spatial and temporal transition mass.

- **[`fig9_certified_confidence_intervals.png`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/plots/fig9_certified_confidence_intervals.png) / [`.pdf`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/plots/fig9_certified_confidence_intervals.pdf):**
  - **Core Visual:** 95% hierarchical bootstrap confidence intervals.

