# Point 5: Comparative Evaluation of MoE Memory Wall Mitigations

This directory contains the complete audit, extracted datasets, publication figures, and technical reports for **Point 5: Comparative Evaluation**.

Point 5 systematically evaluates and compares:
1. **Configuration A: Baseline Without Prefetching** (Demand-only loading under Standard LRU).
2. **Configuration B: Prediction Without Prefetching** (Shadow-mode prediction; zero speculative memory traffic).
3. **Configuration C: Prediction-Driven Prefetching** (Asynchronous prefetching under Standard LRU for ST-MoE and CSTP).
4. **Configuration D: Complete Implementation** (CSTP predictor + asynchronous prefetching + Frequency-Aware LRU $W=64$).

---

## Directory Organization

```text
mohman/point_5/
│
├── audit/
│   ├── FINAL_NUMERICAL_RECONCILIATION.md  <- Authoritative provenance audit and numerical corrections
│   ├── PROJECT_ARTIFACT_INVENTORY.md      <- Complete artifact catalog across Points 1 to 4
│   ├── POINT_WISE_RESULTS_AUDIT.md        <- Empirical findings and progressions from Points 1–4
│   ├── CONFIGURATION_MAPPING.md           <- Conceptual four-configuration mapping
│   ├── METRIC_AND_DENOMINATOR_AUDIT.md    <- Invariant validations and denominator checks
│   └── MISSING_EVIDENCE_REPORT.md         <- Gap analysis, surrogate validation, resolution plan
│
├── extracted_data/
│   ├── baseline_results.csv               <- 16 validated baseline runs across campaigns
│   ├── prediction_only_results.csv        <- Authentic shadow-mode simulation runs
│   ├── prefetch_results.csv               <- 30 prefetch-enabled runs (ST-MoE and CSTP)
│   ├── complete_system_results.csv        <- 8 complete implementation runs (FA-LRU)
│   ├── unified_comparison.csv             <- Master 64-row dataset integrating all campaigns
│   ├── predictor_metrics.csv              <- 12 rows of offline and in-situ predictor metrics
│   ├── source_manifest.csv                <- Exact provenance links for all rows
│   ├── execution_time_comparison.csv      <- Exact data table for Figure 5
│   ├── exposed_stall_comparison.csv       <- Exact data table for Figure 6
│   ├── ddr5_traffic_composition.csv       <- Exact data table for Figure 7
│   ├── hbm_hit_reload_comparison.csv      <- Exact data table for Figure 8
│   ├── predictor_quality_comparison.csv   <- Exact data table for Figure 9
│   ├── prefetch_effectiveness_comparison.csv <- Exact data table for Figure 10
│   ├── falru_incremental_comparison.csv   <- Exact data table for Figure 11
│   ├── batch_scaling_comparison.csv       <- Exact data table for Figure 12
│   ├── extract_point5_data.py             <- Zero-ternary extraction script
│   └── generate_comparative_tables.py     <- Zero-ternary table generation script
│
├── figures/
│   ├── fig1_predictor_effectiveness.png / .pdf       <- Predictor recall, precision, hit rate
│   ├── fig2_prefetching_contribution.png / .pdf      <- Stall reduction vs traffic overhead
│   ├── fig3_falru_contribution.png / .pdf            <- Resident hits, traffic & stall deltas
│   ├── fig4_overall_system_comparison.png / .pdf     <- Normalized execution time and DDR5 traffic
│   ├── fig5_execution_time_comparison.png / .pdf     <- End-to-end execution time across configs
│   ├── fig6_exposed_stall_comparison.png / .pdf      <- Absolute and relative exposed GPU stall
│   ├── fig7_ddr5_traffic_composition.png / .pdf      <- Demand vs speculative DDR5 traffic (GB)
│   ├── fig8_hbm_hits_reloads.png / .pdf              <- Resident HBM hits and DDR5 reloads
│   ├── fig9_predictor_quality_comparison.png / .pdf  <- Macro recall, precision, F1, Top-K hit rate
│   ├── fig10_prefetch_effectiveness_pollution.png / .pdf <- Timely vs late prefetch & unused pollution
│   ├── fig11_falru_incremental_contribution.png / .pdf   <- FA-LRU physical savings & diagnostics
│   ├── fig12_batch_scaling_comparison.png / .pdf     <- Concurrency scaling across campaigns
│   ├── plot_point5_figures.py                        <- Zero-ternary plotting script (Figs 1-4)
│   ├── plot_extended_figures.py                      <- Zero-ternary plotting script (Figs 5-12)
│   ├── png/                                          <- 300 DPI high-resolution figures
│   └── pdf/                                          <- Vector graphic publication figures
│
├── reports/
│   ├── POINT_5_COMPARATIVE_EVALUATION.md             <- Primary 12-section comparative evaluation
│   ├── EXTENDED_COMPARATIVE_VISUAL_ANALYSIS.md       <- Extended visual report with 8 figures & tables
│   ├── COMPONENT_CONTRIBUTION_ANALYSIS.md            <- Isolated analysis of each design component
│   └── COMPARABILITY_AND_LIMITATIONS.md              <- Matched vs supporting methodology & threats
│
└── README.md                                         <- Navigation guide and executive summary
```

---

## Key Experimental Findings

1. **Prediction Alone Provides Zero Hardware Speedup (Configuration B)**:
   - Prediction without prefetching leaves HBM hits and GPU memory stall unchanged ($0.00\%$ latency improvement).
   - In-situ overhead is negligible ($62.74\text{ }\mu\text{s}$, $0.25\%$ CPU host overhead). Prediction functions purely as an informational enabler.
2. **Asynchronous Prefetching Yields Major Latency Reduction (Configuration C)**:
   - Overlapping memory loads with compute eliminates **$246.60\text{ seconds}$ of exposed GPU memory stall** under Workload W4 ($B=32, T=128$), reducing simulated execution time by **$24.46\%$**.
   - CSTP achieves $2\times$ to $5\times$ greater stall reduction than ST-MoE while incurring significantly less DDR5 bus traffic overhead ($+13-22\%$ vs. $+28-42\%$).
3. **FA-LRU Mitigates Speculative Cache Pollution (Configuration D)**:
   - In Workload W4, FA-LRU ($W=64$) selectively evicted **5,999 unconsumed speculative prefetches** from the bottom candidate window to protect mature demand entries.
   - **36 protected lines were subsequently reused**, yielding $+15$ coalesced resident hits, eliminating 15 DDR5 expert reloads, reducing DDR5 traffic by **$2.03\text{ GB}$**, and saving $31.82\text{ ms}$ of stall.
4. **Complete Implementation Balance**:
   - The complete system delivers a **$24.46\%$ net speedup** relative to the baseline while actively safeguarding the HBM cache against speculative pollution.
