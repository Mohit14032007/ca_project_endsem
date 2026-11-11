# Point 7 Correction Changelog: Audit Resolutions and Reconciliation Ledger

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 7 — Energy Analysis  
**Document:** Authoritative Changelog of Discrepancies, Root Causes, and Corrective Actions  
**Author:** Mohman / Advanced Agentic Coding Assistant  
**Date:** November 11, 2026  

---

## 1. Summary of Corrective Actions

| Issue # | Component / Document | Previous Erroneous Claim / Value | Reconciled Correct Value | Root Cause Analysis | Corrective Action Taken |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **01** | `reports/ENERGY_ANALYSIS.md` & `COMPONENT_WISE_...` | Anchor D sum was $9,428.11\text{ J}$ but reported total was $9,416.66\text{ J}$ (discrepancy: $11.45\text{ J}$) | HBM $1,044.18\text{ J}$, MAC $71.25\text{ J}$, DDR5 $4,873.91\text{ J}$, Static $3,427.32\text{ J}$. Sum = $\mathbf{9,416.66\text{ J}}$ ($0.00\text{ J}$ residual) | Author copied outdated draft numbers from Baseline A and an unverified runtime ($765.56\text{ s}$ vs true $761.63\text{ s}$) into markdown table | Rebuilt all Anchor tables across all reports directly from `unified_energy_results.csv` row 3. |
| **02** | Executive Summary & `ENERGY_PERFORMANCE_...` | Claimed $K=8$ had simulated runtime $770.69\text{ s}$ and worsened EDP by $+0.437\%$ vs Baseline A | True runtime is $\mathbf{761.89\text{ s}}$ ($5.39\text{ s}$ faster than baseline); EDP is $\mathbf{7.179\text{ M J}\cdot\text{s}}$ (**-0.325%** vs Baseline A) | Figure 6 was already plotting true data from CSV. The text author fabricated $770.69\text{ s}$ and $+0.437\%$ in prose | Updated all reports to state that $K=8$ achieves a minor EDP improvement vs Baseline A (-0.33%), but is sub-optimal compared to $K=2$ ($7.079\text{ M}$, -1.72%). |
| **03** | `reports/ENERGY_SENSITIVITY_...` & `ENERGY_ANALYSIS.md` | Claimed attention energy scales $32\times$ between $P=128$ and $P=4096$ ($1.84\text{ J} \to 58.96\text{ J}$, $2.41\text{ J} \to 77.28\text{ J}$) | Actual simulator output: Compute grows **$6.73\times$** ($7.82\text{ J} \to 52.59\text{ J}$); DRAM grows **$1.86\times$** ($27.30\text{ J} \to 50.68\text{ J}$) | Author multiplied $P=2048$ values by $(P/2048)$ without checking decode step CSVs. In decode, average sequence length grows $21.6\times$ | Replaced fabricated numbers with verified simulator outputs across all context length tables and text. |
| **04** | Executive Summary & `COMPONENT_WISE_...` | Mixed Point 4 W4 hit count (+15 hits) with Point 6/7 Anchor hit count; claimed $-2.13\text{ GiB}$ ($2,286,284,800\text{ B}$) | Anchor workload yields **+22 resident hits**, **-22 reloads**, eliminating **$2,025,848,832\text{ bytes}$ ($2.026\text{ GB} = 1.887\text{ GiB}$)** of DDR5 traffic | Point 4 W4 evaluated an isolated synthetic run; Point 6/7 Anchor evaluated authentic trace replay. Traffic was imprecise | Clarified provenance: documented Point 4 W4 vs Anchor, corrected traffic to $2,025,848,832\text{ bytes}$ and net savings to $0.326\text{ J}$. |
| **05** | `figures/generate_point7_figures.py` (Fig 4) | X-axis was labeled "TiB" but plotted as binary while annotations used decimal TB | Standardized to **Decimal Terabytes (TB, $10^{12}$ bytes)** to match the analytical $100\text{ J/TB}$ ($0.1\text{ nJ/byte}$) model | Mismatched binary vs decimal prefix conventions | Corrected X-axis plotting and labels to decimal TB; labeled line as analytical model verification. |
| **06** | Deliverables index | Mentioned 12 figures in one place and 15 figures in another | Exactly **12 publication figures** exist (`fig1` through `fig12`) | Earlier draft brainstormed 15 figures before finalizing on 12 | Standardized all documentation and index files to exactly 12 figures. |
| **07** | Python Code Base | Rule 16 Zero Ternary Operator Compliance | Strictly **ZERO ternary operators** across all Python scripts (`ast.IfExp == 0`) | Continuous strict verification via AST parser | Verified `extract_point7_energy_data.py`, `validate_point7_energy_data.py`, and `generate_point7_figures.py`. |

---

## 2. Verification Protocol

Every correction above was verified using:
1. Automated invariant testing via `validate_point7_energy_data.py` (passed 100%).
2. AST zero-ternary verification via `python3 -c "import ast..."` (passed 100%).
3. Image and PDF rendering verification for Figures 1 to 12.
4. Independent re-summation of raw simulator decode CSV rows.

