# Point 7 — Energy Analysis: Complete Master Directory and Index

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 7 — Energy Analysis  
**Location:** `mohman/point_7/`  
**Date:** November 11, 2026  

---

## 1. Directory Structure

This directory contains the complete, self-contained implementation, forensic audit records, extracted datasets, publication figures, and technical reports for **Point 7: Energy Analysis** of the MoE Memory Wall project.

```text
mohman/point_7/
├── README.md                                    # This directory map and master index
├── FINAL_EXECUTIVE_SUMMARY.md                   # Definitive executive summary and GO/NO-GO decision
├── final_audit/                                 # Forensic audit and numerical reconciliation records
│   ├── FINAL_ARTIFACT_INVENTORY.md              # Exhaustive inventory of 28 runs, 8 CSVs, 12 figures
│   ├── ENERGY_MODEL_SOURCE_CODE_AUDIT.md        # Source-level code audit (power.h, device.cpp, module_graph.cpp)
│   ├── FINAL_NUMERICAL_RECONCILIATION.md        # Resolution of 11.45 J discrepancy, K=8 EDP, 32x context fallacy
│   ├── ENERGY_MODEL_VALIDITY_REPORT.md          # Physical validity and analytical model boundaries
│   ├── FIGURE_AND_TABLE_CONSISTENCY_REPORT.md   # Visual and numerical consistency cross-audit
│   └── CORRECTION_CHANGELOG.md                  # Item-by-item changelog of all corrected items
├── audit/                                       # Preliminary Phase A audit deliverables
│   ├── ENERGY_MODEL_AUDIT.md                    # 14-point source audit of LLMSimulator energy modeling
│   ├── ENERGY_COMPONENT_MAPPING.md              # Microarchitectural mapping equations and invariants
│   ├── EXISTING_ENERGY_DATA_INVENTORY.md        # Inventory of completed simulations
│   └── ENERGY_MODEL_VALIDITY_REPORT.md          # Classification: PARTIALLY USABLE
├── extracted_data/                              # Consolidated CSV datasets and extraction scripts
│   ├── extract_point7_energy_data.py            # Extraction and energy accounting script (0 ternaries)
│   ├── validate_point7_energy_data.py           # Automated invariant validation script (0 ternaries)
│   ├── unified_energy_results.csv               # 28-run master dataset with native and system metrics
│   ├── energy_component_breakdown.csv           # Granular breakdown (HBM ACT/RD/WR, MAC, DDR5, Static)
│   ├── energy_traffic_reconciliation.csv        # Byte-level DDR5 demand/speculative energy reconciliation
│   ├── energy_per_token.csv                     # Native and system energy per token (mJ) and request (J)
│   ├── energy_delay_product.csv                 # Native and system EDP (J*s) with deltas
│   ├── paired_energy_comparisons.csv            # 26 rigorous pairwise comparisons against baselines
│   ├── energy_sensitivity_summary.csv           # Sweep summaries across B, P, H, K
│   └── experiment_manifest.csv                  # Provenance tracking back to Point 6 JSON/CSV sources
├── figures/                                     # Publication Figures (PNG 300 DPI and vector PDF)
│   ├── generate_point7_figures.py               # Automated plotting script (0 ternaries)
│   ├── png/                                     # High-resolution raster images (300 DPI)
│   │   ├── fig1_anchor_system_energy.png
│   │   ├── fig2_anchor_energy_per_token.png
│   │   ├── fig3_prefetch_budget_ddr5_breakdown.png
│   │   ├── fig4_ddr5_traffic_vs_energy.png
│   │   ├── fig5_stall_reduction_vs_energy_overhead.png
│   │   ├── fig6_energy_delay_product.png
│   │   ├── fig7_batch_size_energy_scaling.png
│   │   ├── fig8_budget_sweet_spot.png
│   │   ├── fig9_lookahead_energy_dynamics.png
│   │   ├── fig10_context_length_energy_scaling.png
│   │   ├── fig11_energy_performance_pareto.png
│   │   └── fig12_component_pie_breakdown.png
│   └── pdf/                                     # Vector graphics for academic papers
│       ├── fig1_anchor_system_energy.pdf
│       ├── fig2_anchor_energy_per_token.pdf
│       ├── fig3_prefetch_budget_ddr5_breakdown.pdf
│       ├── fig4_ddr5_traffic_vs_energy.pdf
│       ├── fig5_stall_reduction_vs_energy_overhead.pdf
│       ├── fig6_energy_delay_product.pdf
│       ├── fig7_batch_size_energy_scaling.pdf
│       ├── fig8_budget_sweet_spot.pdf
│       ├── fig9_lookahead_energy_dynamics.pdf
│       ├── fig10_context_length_energy_scaling.pdf
│       ├── fig11_energy_performance_pareto.pdf
│       └── fig12_component_pie_breakdown.pdf
└── reports/                                     # In-depth technical analysis deliverables
    ├── ENERGY_ANALYSIS.md                       # Comprehensive core evaluation report
    ├── COMPONENT_WISE_ENERGY_ANALYSIS.md        # Detailed microarchitectural breakdown
    ├── ENERGY_PERFORMANCE_TRADEOFF.md           # Stall reduction vs speculative traffic & Pareto points
    ├── ENERGY_SENSITIVITY_ANALYSIS.md           # Parameter sweeps (B, P, H, K)
    ├── ENERGY_MODEL_LIMITATIONS.md              # Physical model boundaries and assumptions
    └── FINAL_POINT_7_REPORT.md                  # Master synthesis answering the 6 core prompt questions
```

---

## 2. Master Findings at a Glance

| Research Finding | Key Quantitative Evidence | Report Link |
| :--- | :--- | :--- |
| **Prefetch Sweet Spot ($K=2$)** | Saves **-27.68 J (-0.295%)** in net system energy; EDP improves by **-1.715%**. | [ENERGY_PERFORMANCE_TRADEOFF.md](file:///home/mohman/ca_project/mohman/point_7/reports/ENERGY_PERFORMANCE_TRADEOFF.md) |
| **Anchor Trade-off ($K=4$)** | Total energy increases by $+29.04\text{ J}$ ($+0.309\%$); EDP improves by **-0.429%**. | [ENERGY_ANALYSIS.md](file:///home/mohman/ca_project/mohman/point_7/reports/ENERGY_ANALYSIS.md) |
| **Diminishing Returns ($K=8$)** | Total energy increases by $+35.60\text{ J}$ ($+0.379\%$); EDP delta is **-0.325%** vs Baseline A, but **+1.41% worse than $K=2$**. | [ENERGY_PERFORMANCE_TRADEOFF.md](file:///home/mohman/ca_project/mohman/point_7/reports/ENERGY_PERFORMANCE_TRADEOFF.md) |
| **Batch Concurrency Amortization** | Energy per token drops by **50.0%** from $B=4$ ($4,582.7\text{ mJ}$) to $B=32$ ($2,291.9\text{ mJ}$). | [ENERGY_SENSITIVITY_ANALYSIS.md](file:///home/mohman/ca_project/mohman/point_7/reports/ENERGY_SENSITIVITY_ANALYSIS.md) |
| **Low Concurrency Inversion ($B=4$)** | Prefetching increases runtime by $+0.13\%$ and energy by $+1.371\%$ ($+32.17\text{ J}$), worsening EDP by **+1.503%**. | [ENERGY_SENSITIVITY_ANALYSIS.md](file:///home/mohman/ca_project/mohman/point_7/reports/ENERGY_SENSITIVITY_ANALYSIS.md) |
| **Context Length Scaling ($P$)** | Attention compute grows **$6.73\times$** and DRAM grows **$1.86\times$** from $P=128$ to $4096$; MoE energy remains invariant. | [ENERGY_SENSITIVITY_ANALYSIS.md](file:///home/mohman/ca_project/mohman/point_7/reports/ENERGY_SENSITIVITY_ANALYSIS.md) |
| **FA-LRU ($W=64$) Contribution** | Eliminates 22 reloads; saves $2,025,848,832\text{ bytes}$ ($2.026\text{ GB}$) demand traffic and $0.326\text{ J}$ net energy. | [COMPONENT_WISE_ENERGY_ANALYSIS.md](file:///home/mohman/ca_project/mohman/point_7/reports/COMPONENT_WISE_ENERGY_ANALYSIS.md) |
| **System Macro Breakdown** | Complete System D: DDR5 Active $51.8\%$, Static $36.4\%$, HBM $11.1\%$, Compute $0.8\%$. | [COMPONENT_WISE_ENERGY_ANALYSIS.md](file:///home/mohman/ca_project/mohman/point_7/reports/COMPONENT_WISE_ENERGY_ANALYSIS.md) |

---

## 3. Reproduction Commands

To reproduce all extracted CSV datasets, run automated invariant verification, and regenerate all publication figures:

```bash
# 1. Run Data Extraction and Energy Accounting Script
python3 /home/mohman/ca_project/mohman/point_7/extracted_data/extract_point7_energy_data.py

# 2. Run Automated Invariant Validation
python3 /home/mohman/ca_project/mohman/point_7/extracted_data/validate_point7_energy_data.py

# 3. Run Publication Figure Generator (12 figures in PNG and PDF)
python3 /home/mohman/ca_project/mohman/point_7/figures/generate_point7_figures.py
```

### Verification of Code Restrictions:
All Python scripts adhere strictly to **Rule 16: Zero Ternary Operators**. Verified via Python Abstract Syntax Tree (`ast.walk()`):
- `extract_point7_energy_data.py`: `ast.IfExp == 0`
- `validate_point7_energy_data.py`: `ast.IfExp == 0`
- `generate_point7_figures.py`: `ast.IfExp == 0`
