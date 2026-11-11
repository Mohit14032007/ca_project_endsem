# Figure and Table Consistency Audit Report

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Verification and Figure Audit  
**Author:** Antigravity Engineering  
**Date:** November 11, 2026  
**Status:** 100% Reconciled and Validated  

---

## 1. Overview

This audit establishes 100% mutual consistency across all 15 publication-quality figures, all consolidated CSV datasets, and all technical reports. Every figure is verified against its raw simulation data source.

---

## 2. Complete Figure Manifest and Verification Matrix

```
=============================================================================================================================================================
                                                     POINT 6 PUBLICATION FIGURE VERIFICATION MATRIX
=============================================================================================================================================================
 Fig # | File Base Name                       | Target Directory   | Underlying CSV Source         | Raw Source Runs Included       | Reconciled Key Metrics
-------+--------------------------------------+--------------------+-------------------------------+--------------------------------+------------------------
 Fig 1 | fig1_batch_size_hit_rate             | figures/batch_size | batch_size_comparison.csv     | b4_c1..c4, b14_c1..c4, anc_c1..| Hit rate: 1.06% -> 0.28%
 Fig 2 | fig2_batch_size_exposed_stall        | figures/batch_size | batch_size_comparison.csv     | b4_c1..c4, b14_c1..c4, anc_c1..| Stall: 183.5s -> 753.2s
 Fig 3 | fig3_batch_size_token_latency        | figures/batch_size | batch_size_comparison.csv     | b4_c1..c4, b14_c1..c4, anc_c1..| Latency: 1466ms -> 5994ms
 Fig 4 | fig4_context_length_kv_cache         | figures/context_len| context_length_comparison.csv | Model Architecture Parameters  | Footprint: 0.18G -> 5.62G
 Fig 5 | fig5_context_length_attention_time   | figures/context_len| context_length_comparison.csv | p128_c1, p512_c1, anc_c1, p4096| Slack: 13.7s -> 14.4s
 Fig 6 | fig6_context_length_expert_cache     | figures/context_len| context_length_comparison.csv | p128..p4096 (all c1, c3, c4)   | Latency: 5992ms -> 5950ms
 Fig 7 | fig7_lookahead_resident_hits         | figures/lookahead  | lookahead_comparison.csv      | anc_c3..c4, h2_c3..c4, h4_c3.. | Hits: 1499 -> 2244 (+49.7%)
 Fig 8 | fig8_lookahead_unused_prefetches     | figures/lookahead  | lookahead_comparison.csv      | anc_c3..c4, h2_c3..c4, h4_c3.. | Unused: 1505 -> 2251 (+49.6%)
 Fig 9 | fig9_lookahead_exposed_stall         | figures/lookahead  | lookahead_comparison.csv      | anc_c3..c4, h2_c3..c4, h4_c3.. | Stall: 747.6s -> 746.2s
 Fig 10| fig10_budget_speculative_ddr5        | figures/prefetch_bg| ddr5_traffic_composition.csv  | k2_c3..c4, anc_c3..c4, k8_c3.. | Spec: 6.08TB -> 18.04TB
 Fig 11| fig11_budget_demand_resident_evict   | figures/prefetch_bg| cache_pollution_analysis.csv  | k2_c3..c4, anc_c3..c4, k8_c3.. | Demand evictions: ~546.9k
 Fig 12| fig12_budget_token_latency           | figures/prefetch_bg| budget_comparison.csv         | k2_c3..c4, anc_c3..c4, k8_c3.. | Latency: 5909ms vs 5952ms
 Fig 13| fig13_prefetch_vs_noprefetch         | figures/combined   | paired_comparisons.csv        | b4, b14, anc (paired deltas)   | Speedup: -0.13% -> +0.74%
 Fig 14| fig14_lru_vs_falru_comparison        | figures/combined   | paired_comparisons.csv        | p128, p512, anc, p4096 (deltas)| Reloads saved: +15 to +22
 Fig 15| fig15_operating_regimes_summary      | figures/combined   | unified_point6_results.csv    | All 26 Evaluated Runs          | 3 Operating Regimes Map
=============================================================================================================================================================
```

---

## 3. Detailed Audit of High-Priority Reconciliations

### 3.1 Figure 10: Prefetch Budget ($K$) DDR5 Traffic Composition
- **Previous Discrepancy:** The report table listed $43,139.11\text{ GB}$ total DDR5 traffic for $K=2$, while Figure 10 plotted $48,395\text{ GB}$. Furthermore, the report table duplicated $K=4$'s traffic values into $K=8$.
- **Audit Verification:**
  - `k2_c3` raw JSON: Demand $= 42,311.44\text{ GB}$, Speculative $= 6,083.98\text{ GB}$, Total $= 48,395.41\text{ GB}$.
  - `anc_c3` ($K=4$) raw JSON: Demand $= 37,792.12\text{ GB}$, Speculative $= 10,949.01\text{ GB}$, Total $= 48,741.13\text{ GB}$.
  - `k8_c3` raw JSON: Demand $= 30,739.09\text{ GB}$, Speculative $= 18,037.10\text{ GB}$, Total $= 48,776.19\text{ GB}$.
- **Resolution:** Figure 10 was correctly representing the raw simulation outputs. The report table has been corrected to display $48,395.41\text{ GB}$ for $K=2$ and $48,776.19\text{ GB}$ for $K=8$.

### 3.2 Figures 7, 8, 9: Lookahead Metric Labeling
- **Previous Discrepancy:** Figure 7 was titled "lookahead timely coverage" in the manifest, but plotted coalesced resident hits.
- **Audit Verification:** The underlying data plotted represents coalesced resident hits ($1,499$ at $H=1$, $1,817$ at $H=2$, and $2,244$ at $H=4$).
- **Resolution:** Retitled Figure 7 as `fig7_lookahead_resident_hits.png` / `.pdf` with the explicit title: *"Figure 7: Lookahead Distance ($H$) vs. HBM Resident Hits"*, completely removing ambiguity.

### 3.3 Figure 5: Compute Duration and Prefetch Overlap Slack
- **Previous Discrepancy:** Figure 5 reported $13,783.4\text{ ms}$ for $P=512$ and $14,355.7\text{ ms}$ for $P=4,096$, while the table displayed $13,782.45\text{ ms}$ and $14,346.52\text{ ms}$.
- **Audit Verification:** Raw JSON nanosecond values are:
  - $P=128$: $13,727.88\text{ ms}$
  - $P=512$: $13,783.40\text{ ms}$
  - $P=2048$: $14,026.23\text{ ms}$
  - $P=4096$: $14,355.67\text{ ms}$
- **Resolution:** The report table has been updated to use the exact nanosecond-derived values, achieving 100% agreement between Figure 5 and the text.

