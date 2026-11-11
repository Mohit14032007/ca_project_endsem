# Final Artifact and Experiment Inventory: Point 7 Energy Analysis

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 7 — Energy Analysis  
**Document:** Exhaustive Forensic Inventory of Runs, Datasets, Figures, and Reports  
**Author:** Mohman / Advanced Agentic Coding Assistant  
**Date:** November 11, 2026  

---

## 1. Executive Reconciliation of Counts

Before detailing individual runs, we formally establish and reconcile all artifact counts across Point 7 to eliminate historical ambiguities (such as the 12 vs. 15 figures discrepancy):

| Entity Type | Verified Count | Notes / Status |
| :--- | :---: | :--- |
| **Raw Simulation Runs** | **28** | 23 runs from Point 6 execution + 5 verified Anchor/Batch historical runs. All 28 paths verified on disk. |
| **Unique Workload Configurations** | **28** | Spanning sweeps over Batch Size ($B \in \{4, 14, 32\}$), Context Length ($P \in \{128, 512, 2048, 4096\}$), Budget ($K \in \{0, 2, 4, 8\}$), Lookahead ($H \in \{1, 2, 4\}$), and Cache Policy (LRU vs. FA-LRU). |
| **Paired Comparisons** | **26** | Formally captured in `paired_energy_comparisons.csv` evaluating Proposed vs. Baseline deltas. |
| **Publication Figures** | **12** | **Exactly 12 figures** (`fig1` through `fig12`), each available in 300 DPI PNG and vector PDF. (Historical mention of "15 figures" was an artifact of earlier planning; exactly 12 figures are generated and verified). |
| **Extracted Datasets (CSVs)** | **8** | All 8 CSVs located in `extracted_data/`, validated via automated invariant script. |
| **Python Support Scripts** | **3** | `extract_point7_energy_data.py`, `validate_point7_energy_data.py`, `generate_point7_figures.py`. (All verified 0 ternaries). |
| **Technical Analysis Reports** | **6** | Core analysis documents in `reports/`. |
| **Audit & Forensic Documents** | **6** | Deep-dive audit documents in `final_audit/`. |
| **Executive Deliverables** | **2** | `README.md` (directory map) and `FINAL_EXECUTIVE_SUMMARY.md` (master synthesis). |

---

## 2. Exhaustive Run-by-Run Matrix (28 Evaluated Runs)

Every single run in Point 7 is accounted for below with complete provenance, workload parameters, and usage mapping.

| Index | Run ID | Suite | Config Name | $B$ | $P$ | $T$ | $K$ | $H$ | Eviction Policy | Source Raw Path | Decode CSV Present? | Measurement Type | Primary Figures | Primary Reports |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|:---:|:---:|:---|
| 0 | `anc_c1` | anchor | A_Baseline | 32 | 2048 | 128 | 0 | 0 | LRU | `point_6/final_results/experiments/anchor/raw_results/anc_c1` | YES | Measured + Analytical | Fig 1, 2, 3, 4, 5, 6, 7, 8, 11 | All Reports |
| 1 | `anc_c2` | anchor | B_Prediction_Only | 32 | 2048 | 128 | 0 | 0 | LRU | `point_6/final_results/experiments/anchor/raw_results/anc_c2` | YES | Measured + Analytical | Fig 1, 2, 6, 11 | All Reports |
| 2 | `anc_c3` | anchor | C_Prefetch_LRU | 32 | 2048 | 128 | 4 | 1 | LRU | `point_6/final_results/experiments/anchor/raw_results/anc_c3` | YES | Measured + Analytical | Fig 1, 2, 5, 6, 11 | All Reports |
| 3 | `anc_c4` | anchor | D_Complete_FALRU | 32 | 2048 | 128 | 4 | 1 | FA-LRU (64) | `point_6/final_results/experiments/anchor/raw_results/anc_c4` | YES | Measured + Analytical | Fig 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 | All Reports |
| 4 | `b4_c1` | batch_size | A_Baseline | 4 | 2048 | 128 | 0 | 0 | LRU | `point_6/final_results/experiments/batch_size/raw_results/b4_c1` | YES | Measured + Analytical | Fig 4, 7 | Energy Sensitivity |
| 5 | `b4_c3` | batch_size | C_Prefetch_LRU | 4 | 2048 | 128 | 4 | 1 | LRU | `point_6/final_results/experiments/batch_size/raw_results/b4_c3` | YES | Measured + Analytical | Fig 4 | Energy Sensitivity |
| 6 | `b4_c4` | batch_size | D_Complete_FALRU | 4 | 2048 | 128 | 4 | 1 | FA-LRU (64) | `point_6/final_results/experiments/batch_size/raw_results/b4_c4` | YES | Measured + Analytical | Fig 4, 7 | Energy Sensitivity |
| 7 | `b14_c1` | batch_size | A_Baseline | 14 | 2048 | 128 | 0 | 0 | LRU | `point_6/final_results/experiments/batch_size/raw_results/b14_c1` | YES | Measured + Analytical | Fig 4, 7 | Energy Sensitivity |
| 8 | `b14_c3` | batch_size | C_Prefetch_LRU | 14 | 2048 | 128 | 4 | 1 | LRU | `point_6/final_results/experiments/batch_size/raw_results/b14_c3` | YES | Measured + Analytical | Fig 4 | Energy Sensitivity |
| 9 | `b14_c4` | batch_size | D_Complete_FALRU | 14 | 2048 | 128 | 4 | 1 | FA-LRU (64) | `point_6/final_results/experiments/batch_size/raw_results/b14_c4` | YES | Measured + Analytical | Fig 4, 7 | Energy Sensitivity |
| 10 | `p128_c1` | context_length | A_Baseline | 32 | 128 | 128 | 0 | 0 | LRU | `point_6/final_results/experiments/context_length/raw_results/p128_c1` | YES | Measured + Analytical | Fig 4, 10 | Energy Sensitivity |
| 11 | `p128_c2` | context_length | B_Prediction_Only | 32 | 128 | 128 | 0 | 0 | LRU | `point_6/final_results/experiments/context_length/raw_results/p128_c2` | YES | Measured + Analytical | Fig 4 | Energy Sensitivity |
| 12 | `p128_c3` | context_length | C_Prefetch_LRU | 32 | 128 | 128 | 4 | 1 | LRU | `point_6/final_results/experiments/context_length/raw_results/p128_c3` | YES | Measured + Analytical | Fig 4 | Energy Sensitivity |
| 13 | `p128_c4` | context_length | D_Complete_FALRU | 32 | 128 | 128 | 4 | 1 | FA-LRU (64) | `point_6/final_results/experiments/context_length/raw_results/p128_c4` | YES | Measured + Analytical | Fig 4, 10 | Energy Sensitivity |
| 14 | `p512_c1` | context_length | A_Baseline | 32 | 512 | 128 | 0 | 0 | LRU | `point_6/final_results/experiments/context_length/raw_results/p512_c1` | YES | Measured + Analytical | Fig 4, 10 | Energy Sensitivity |
| 15 | `p512_c3` | context_length | C_Prefetch_LRU | 32 | 512 | 128 | 4 | 1 | LRU | `point_6/final_results/experiments/context_length/raw_results/p512_c3` | YES | Measured + Analytical | Fig 4 | Energy Sensitivity |
| 16 | `p512_c4` | context_length | D_Complete_FALRU | 32 | 512 | 128 | 4 | 1 | FA-LRU (64) | `point_6/final_results/experiments/context_length/raw_results/p512_c4` | YES | Measured + Analytical | Fig 4, 10 | Energy Sensitivity |
| 17 | `p4096_c1` | context_length | A_Baseline | 32 | 4096 | 128 | 0 | 0 | LRU | `point_6/final_results/experiments/context_length/raw_results/p4096_c1` | YES | Measured + Analytical | Fig 4, 10 | Energy Sensitivity |
| 18 | `p4096_c3` | context_length | C_Prefetch_LRU | 32 | 4096 | 128 | 4 | 1 | LRU | `point_6/final_results/experiments/context_length/raw_results/p4096_c3` | YES | Measured + Analytical | Fig 4 | Energy Sensitivity |
| 19 | `p4096_c4` | context_length | D_Complete_FALRU | 32 | 4096 | 128 | 4 | 1 | FA-LRU (64) | `point_6/final_results/experiments/context_length/raw_results/p4096_c4` | YES | Measured + Analytical | Fig 4, 10 | Energy Sensitivity |
| 20 | `h2_c3` | lookahead | C_Prefetch_LRU | 32 | 2048 | 128 | 4 | 2 | LRU | `point_6/final_results/experiments/lookahead/raw_results/h2_c3` | YES | Measured + Analytical | Fig 4, 5, 9, 11 | Energy Sensitivity |
| 21 | `h2_c4` | lookahead | D_Complete_FALRU | 32 | 2048 | 128 | 4 | 2 | FA-LRU (64) | `point_6/final_results/experiments/lookahead/raw_results/h2_c4` | YES | Measured + Analytical | Fig 4, 5, 9, 11 | Energy Sensitivity |
| 22 | `h4_c3` | lookahead | C_Prefetch_LRU | 32 | 2048 | 128 | 4 | 4 | LRU | `point_6/final_results/experiments/lookahead/raw_results/h4_c3` | YES | Measured + Analytical | Fig 4, 5, 9, 11 | Energy Sensitivity |
| 23 | `h4_c4` | lookahead | D_Complete_FALRU | 32 | 2048 | 128 | 4 | 4 | FA-LRU (64) | `point_6/final_results/experiments/lookahead/raw_results/h4_c4` | YES | Measured + Analytical | Fig 4, 5, 9, 11 | Energy Sensitivity |
| 24 | `k2_c3` | budget | C_Prefetch_LRU | 32 | 2048 | 128 | 2 | 1 | LRU | `point_6/final_results/experiments/budget/raw_results/k2_c3` | YES | Measured + Analytical | Fig 3, 4, 5, 6, 8, 11 | Trade-off / Synthesis |
| 25 | `k2_c4` | budget | D_Complete_FALRU | 32 | 2048 | 128 | 2 | 1 | FA-LRU (64) | `point_6/final_results/experiments/budget/raw_results/k2_c4` | YES | Measured + Analytical | Fig 3, 4, 5, 6, 8, 11 | Trade-off / Synthesis |
| 26 | `k8_c3` | budget | C_Prefetch_LRU | 32 | 2048 | 128 | 8 | 1 | LRU | `point_6/final_results/experiments/budget/raw_results/k8_c3` | YES | Measured + Analytical | Fig 3, 4, 5, 6, 8, 11 | Trade-off / Synthesis |
| 27 | `k8_c4` | budget | D_Complete_FALRU | 32 | 2048 | 128 | 8 | 1 | FA-LRU (64) | `point_6/final_results/experiments/budget/raw_results/k8_c4` | YES | Measured + Analytical | Fig 3, 4, 5, 6, 8, 11 | Trade-off / Synthesis |

---

## 3. Data Lineage and Reproducibility Verification

1. **Raw Trace Authenticity:** Every run points to a genuine output folder containing `csv/*_decode.csv` (step-by-step 128-token execution metrics) and `traces/prefetch_events.csv` / `traces/expert_cache.csv`.
2. **Deterministic Extraction:** `extract_point7_energy_data.py` reads directly from these raw directories. Re-running the extraction script reproduces the 8 CSV datasets with byte-for-byte fidelity.
3. **Zero Artificial Imputation:** No run in the 28-matrix contains imputed, synthetic, or manufactured numbers. All numbers trace directly to simulation logs and established analytical formulas.

