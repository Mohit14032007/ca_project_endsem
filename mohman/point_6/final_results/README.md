# Point 6 — Comprehensive Sensitivity Analysis and Experimental Consolidation

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Sensitivity Analysis Consolidation  
**Author:** Antigravity Engineering  
**Date:** November 10, 2026  
**Status:** Authoritative & Completed  

---

## 1. Directory Structure Overview

All Point 6 sensitivity analysis artifacts, configurations, raw execution logs, extracted telemetry, publication figures, and technical reports are organized under:

```text
mohman/point_6/final_results/
│
├── README.md                                  # This master orientation guide
├── execute_point6_campaign.py                 # Master 23-run simulation runner (0 ternaries)
│
├── audit/                                     # Pre-experiment audits and validation gates
│   ├── FINAL_DISCREPANCY_AUDIT.md             # Forensic audit of discrepancies & reconciliation
│   ├── NUMERICAL_RECONCILIATION.md            # Cell-by-cell verification table of all 26 runs
│   ├── CONTEXT_LENGTH_IMPLEMENTATION_EXPLANATION.md # Exact implementation and dataset mechanics
│   ├── MOE_ANYWHERE_TRACE_EQUIVALENCE.md      # Cryptographic SHA-256 trace verification
│   ├── MEMORY_CAPACITY_INTERACTION_VALIDATION.md # Dedicated expert cache architecture audit
│   ├── FIGURE_AND_TABLE_CONSISTENCY_AUDIT.md  # Cross-figure and report numerical audit
│   ├── CORRECTION_CHANGELOG.md                # Itemized changelog of all Point 6 corrections
│   ├── EXISTING_EXPERIMENT_INVENTORY.md       # Complete inventory of 59 historical runs
│   ├── MEMORY_MODEL_AUDIT.md                  # Code audit of KV-cache & attention scaling
│   ├── EXPERIMENT_DESIGN.md                   # Controlled multi-dimensional parameter matrix
│   ├── EXPERIMENT_REUSE_AND_GAP_REPORT.md     # Reuse mapping and execution gap definition
│   └── FINAL_VALIDATION_REPORT.md             # 4-gate pre-flight smoke test clearance
│
├── extracted_existing_results/                # Consolidated historical datasets
│   ├── batch_size_historical_results.csv      # Point 3 and Point 4 batch scaling results
│   ├── context_length_extraction.csv          # Point 6 Exp A 74-prompt inferred P dataset
│   ├── predictor_historical_results.csv       # CSTP and ST-MoE accuracy metrics
│   ├── prefetch_historical_results.csv        # Timeliness and coverage metrics
│   ├── eviction_historical_results.csv        # LRU vs FA-LRU comparative results
│   ├── historical_source_manifest.csv         # Provenance tracking manifest
│   └── HISTORICAL_RESULTS_NOTES.md            # Comparability notes and caveats
│
├── experiments/                               # Controlled experimental suites
│   ├── batch_size/                            # B in {4, 14, 32} at P=2048, T=128
│   ├── context_length/                        # P in {128, 512, 2048, 4096} at B=32, T=128
│   ├── lookahead/                             # H in {1, 2, 4} at B=32, P=2048, T=128
│   └── prefetch_budget/                       # K in {2, 4, 8} at B=32, P=2048, T=128
│
├── consolidated_data/                         # Unified analysis and comparative datasets
│   ├── unified_point6_results.csv             # Master reconciled telemetry for all runs
│   ├── paired_comparisons.csv                 # Paired speedup, stall, and traffic deltas
│   ├── ddr5_traffic_composition.csv           # Reconciled demand vs. speculative DDR5 traffic
│   ├── cache_pollution_analysis.csv           # Demand eviction and victim breakdown
│   ├── prefetch_effectiveness.csv             # Timely hits vs. late attachments
│   ├── source_manifest.csv                    # Full provenance linking runs to raw files
│   ├── batch_size_comparison.csv              # Batch size sweep data table
│   ├── context_length_comparison.csv          # Context length sweep data table
│   ├── lookahead_comparison.csv               # Lookahead sweep data table
│   └── budget_comparison.csv                  # Prefetch budget sweep data table
│
├── figures/                                   # 15 publication figures (PNG 300 DPI & PDF)
│   ├── batch_size/                            # Figures 1, 2, 3
│   ├── context_length/                        # Figures 4, 5, 6
│   ├── lookahead/                             # Figures 7, 8, 9
│   ├── prefetch_budget/                       # Figures 10, 11, 12
│   └── combined/                              # Figures 13, 14, 15
│
└── reports/                                   # Technical analysis and executive reports
    ├── POINT_6_SENSITIVITY_ANALYSIS.md        # Comprehensive technical sensitivity report
    ├── CONTEXT_LENGTH_AND_MOE_ANYWHERE_INTERACTION.md # Trace mechanics & context interaction
    ├── WHEN_PREFETCHING_HELPS_OR_HURTS.md     # Concise interpretation of the 3 regimes
    ├── COMPONENT_AND_PARAMETER_CONTRIBUTION.md# Algorithmic and hardware attribution
    ├── LIMITATIONS_AND_COMPARABILITY.md       # Hardware fidelity & scope disclosures
    └── FINAL_POINT_6_EXECUTIVE_SUMMARY.md     # Executive summary and recommendations
```

---

## 2. Experimental Conventions & Parameters

- **Target Architecture:** DeepSeek-V3 Mixtral-style MoE (40 MoE stages, 256 routed experts, Top-8 routing).
- **Expert Size:** $88,080,384$ bytes ($84.0$ MiB) per expert.
- **HBM Cache Budget:** Dedicated 35 GiB ($37,580,963,840$ bytes).
- **DDR5 Memory Bus:** Event-driven 64 GB/s channel with calibrated request queuing.
- **Authentic Trace Dataset:** `LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet`.
- **Generation Horizon:** $T = 128$ contiguous generation tokens per sequence.
- **Common Anchor Configuration:** $B=32, P=2048, T=128, H=1, K=4$ on 35 GiB HBM.

---

## 3. Strict Verification & Reproducibility

- **Zero Ternary Operators:** All scripts and documents strictly enforce zero ternary operators (`ast.IfExp == 0` / `? :`).
- **Physical Transfer Reconciliation:** Telemetry denominators use stage-coalesced physical demands ($D_{\text{coalesced}}$).
- **Non-Fabrication Guarantee:** Zero synthetic tokens or invented traces used; all metrics are traceable to raw logs and CSVs.

