# MoE Suite Intra-Token Inter-Layer Horizon Sweep (H = 1, 2, 3, 4)
## Controlled Experimental Evaluation, Protocol, Horizon Semantics, and Verification

**Date:** November 1, 2026  
**Dataset:** MoE Suite (`Predicting-Experts-for-MOE-Suite`, token cube representation)  
**Model Architecture:** GLM-5.2 style MoE (75 sequential MoE layers, $E=256$ routed experts, Top-$K=8$)  
**Prediction Budget:** $B = 8$  
**Git Commit:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Execution Script:** `LLMSimulator/scripts/phase_d/run_moe_suite_horizon_sweep.py`  
**Results Directory:** `results/endsem_moe_suite_horizon/`  

---

## 1. Executive Summary

This document reports the controlled experimental horizon sweep conducted exclusively on the **MoE Suite** benchmark dataset. We evaluate sixteen predictor configurations across four intra-token, inter-layer prediction horizons:
- **$H=1$:** Next MoE stage prediction ($l \to l+1$)
- **$H=2$:** Two MoE stages ahead ($l \to l+2$, the primary MoE Suite benchmark track)
- **$H=3$:** Three MoE stages ahead ($l \to l+3$)
- **$H=4$:** Four MoE stages ahead ($l \to l+4$)

All experiments enforce task-disjoint splits (8 training tasks, 2 validation tasks, 2 test tasks), zero test-data leakage, and strict chronological event processing: predictions are generated and recorded strictly before revealing ground-truth routing decisions.

---

## 2. Dataset Representation and Horizon Semantics

### 2.1 Raw Data Schema
The MoE Suite dataset records routing decisions for diverse real-world software engineering workloads. Each corpus file is stored in Apache Parquet format under `data/downloads/moe_suite/corpus-*.parquet`. The verified schema is:
```
route_selected: fixed_size_list<element: fixed_size_list<element: int16>[8]>[75]
```
In NumPy array representation, each task tensor has dimensions:
$$\text{Tensor Shape} = (N_{\text{tokens}}, 75, 8)$$
- **Axis 0 ($N_{\text{tokens}}$):** Sequential tokens in the generation request.
- **Axis 1 ($75$):** Sequential MoE layers (layers 3 through 77 in the host LLM).
- **Axis 2 ($8$):** Top-8 chosen expert IDs ($e \in [0, 255]$).

### 2.2 Intra-Token Inter-Layer Horizon Formulation
For a specific token $t$ and destination MoE layer $l \in [0, 74]$:
$$\text{Target Experts} = \mathbf{arr}[t, l, :]$$
$$\text{Source Experts} = \mathbf{arr}[t, l - H, :]$$

The prediction horizon $H$ explicitly denotes **inter-layer lookahead within the same token $t$**:
- $H=1$: Layer $l-1 \to l$ (next adjacent layer)
- $H=2$: Layer $l-2 \to l$ (skip one intermediate layer)
- $H=3$: Layer $l-3 \to l$ (skip two intermediate layers)
- $H=4$: Layer $l-4 \to l$ (skip three intermediate layers)

The token and layer dimensions are strictly preserved without flattening.

### 2.3 Temporal Prior Semantics
Temporal autocorrelation across consecutive tokens operates at the **same target layer $l$**:
$$\text{Temporal Prior} = \mathbf{arr}[t-1, l, :]$$
When predicting target layer $l$ for token $t$, the predictor accesses the routing history of token $t-1$ at layer $l$, reflecting conversational and positional continuity.

### 2.4 Boundary Event Handling
For the first $H$ layers of each token ($l \in [0, H-1]$), the source stage index $l - H < 0$. These layers have no antecedent MoE stage within the same token:
- **Valid Prediction Events:** Target layers $l \ge H$, where a valid intra-token source stage $l - H \ge 0$ exists.
- **Boundary Skipped Events:** Target layers $l < H$. These events are recorded as boundary events and excluded from lookahead prediction metric calculation.
- **Continuous State Maintenance:** When target layer $l$ executes and reveals ground truth, `update()` is called for all stages $l \in [0, 74]$ so temporal tracking across tokens ($t-1, l \to t, l$) remains active and accurate.

#### Event Accounting per Horizon (100 Test Tokens = 2 Tasks $\times$ 50 Tokens):
| Horizon | Valid Lookahead Events ($l \ge H$) | Skipped Boundary Events ($l < H$) | Total Evaluated Events |
| :---: | :---: | :---: | :---: |
| **$H=1$** | **7,400** ($100 \times 74$) | 100 ($100 \times 1$) | 7,500 |
| **$H=2$** | **7,300** ($100 \times 73$) | 200 ($100 \times 2$) | 7,500 |
| **$H=3$** | **7,200** ($100 \times 72$) | 300 ($100 \times 3$) | 7,500 |
| **$H=4$** | **7,100** ($100 \times 71$) | 400 ($100 \times 4$) | 7,500 |

---

## 3. Evaluated Predictor Configurations

Sixteen predictor and baseline configurations were audited and benchmarked:

1. **`cstp_online`:** Confidence- and Frequency-Aware Spatio-Temporal Predictor with Dirichlet smoothing and causal online profiling updates.
2. **`cstp_frozen`:** CSTP with online updates disabled during test inference.
3. **`st_moe_paper_online`:** Faithful reproduction of Zhao et al. (arXiv:2606.15453v1) with 2-bit saturating counters ($0..3$), CCT replacement, and Equation 1 score aggregation.
4. **`st_moe_paper_frozen`:** ST-MoE paper predictor with frozen CCT counters.
5. **`cstp_st_moe_hybrid`:** Fixed equal-weight fusion ($0.5 \cdot \text{CSTP} + 0.5 \cdot \text{ST-MoE}$).
6. **`acsh_global`:** Adaptive Confidence and Statistical Hybrid with global causal reliability tracking.
7. **`acsh_stage_aware`:** ACSH with layer- and horizon-specific reliability tracking $\gamma_t(l, H)$ and temperature-scaled softmax fusion.
8. **`acsh_fast_recovery`:** ACSH with sudden-shift acceleration ($\beta_{\text{fast}}=0.2$).
9. **`frequency_only`:** Target-stage empirical marginal frequency baseline.
10. **`temporal_only`:** CSTP ablation setting $\alpha = 0.0$ (temporal autocorrelation only).
11. **`spatial_only`:** CSTP ablation setting $\alpha = 1.0$ (spatial transitions only).
12. **`random`:** Uniform pseudo-random baseline ($seed = 42$).
13. **`cstp_cold_start`:** CSTP initialized without offline training profiles, learning entirely online.
14. **`st_moe_paper_cold_start`:** ST-MoE paper initialized without offline profiling.
15. **`acsh_cold_start`:** ACSH initialized from unprofiled components.
16. **`st_moe_legacy`:** Legacy ST-MoE implementation from Phase 1. Evaluated at $H=1$; marked as **unsupported** for $H > 1$ due to hardcoded adjacent layer transitions ($l \to l+1$).

---

## 4. Experimental Results Matrix

All metrics are evaluated at budget $B = 8$ matching the true Top-$K=8$ selection.

### Table 1: Master Horizon Sweep Comparison Table (MoE Suite, $B=8$)

| Predictor | Horizon $H$ | Learning Mode | Status | Valid Events | Macro Recall | Macro F1 | Hit Rate | Query Latency ($\mu\text{s}$) | Update Latency ($\mu\text{s}$) | Memory (KB) | Clustered SE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`acsh_cold_start`** | 1 | cold_start | completed | 7,400 | **0.2855** | **0.2855** | 0.9038 | 1,765.1 | 420.5 | 162.3 | $\pm 0.0210$ |
| **`st_moe_paper_cold_start`**| 1 | cold_start | completed | 7,400 | 0.2772 | 0.2772 | 0.8961 | 104.9 | 60.5 | 22.3 | $\pm 0.0180$ |
| **`cstp_cold_start`** | 1 | cold_start | completed | 7,400 | 0.2746 | 0.2746 | 0.8930 | 834.3 | 358.1 | 25.9 | $\pm 0.0242$ |
| **`st_moe_paper_online`** | 1 | online | completed | 7,400 | **0.2706** | **0.2706** | 0.9076 | **86.0** | **47.7** | **24.5** | $\pm 0.0235$ |
| **`acsh_global`** | 1 | online | completed | 7,400 | 0.2668 | 0.2668 | **0.9077** | 1,923.4 | 520.7 | 173.3 | $\pm 0.0246$ |
| **`acsh_fast_recovery`** | 1 | online | completed | 7,400 | 0.2666 | 0.2666 | 0.9076 | 1,738.7 | 443.8 | 165.2 | $\pm 0.0246$ |
| **`acsh_stage_aware`** | 1 | online | completed | 7,400 | 0.2663 | 0.2663 | 0.9076 | 1,690.7 | 448.6 | 164.4 | $\pm 0.0244$ |
| **`cstp_st_moe_hybrid`** | 1 | online | completed | 7,400 | 0.2550 | 0.2550 | 0.8973 | 859.0 | 404.6 | 43.8 | $\pm 0.0300$ |
| **`st_moe_legacy`** | 1 | online | completed | 7,400 | 0.2406 | 0.2406 | 0.8742 | **28.0** | 37.0 | 18.4 | $\pm 0.0340$ |
| **`spatial_only`** | 1 | online | completed | 7,400 | 0.1874 | 0.1874 | 0.7886 | 698.8 | 338.9 | 25.7 | $\pm 0.0187$ |
| **`st_moe_paper_frozen`** | 1 | frozen | completed | 7,400 | 0.1871 | 0.1871 | 0.7930 | 90.2 | 5.3 | 20.0 | $\pm 0.0179$ |
| **`cstp_online`** | 1 | online | completed | 7,400 | 0.1673 | 0.1673 | 0.7524 | 837.1 | 402.2 | 44.8 | $\pm 0.0198$ |
| **`cstp_frozen`** | 1 | frozen | completed | 7,400 | 0.1657 | 0.1657 | 0.7488 | 844.9 | 6.1 | 25.9 | $\pm 0.0193$ |
| **`temporal_only`** | 1 | online | completed | 7,400 | 0.1202 | 0.1202 | 0.6224 | 729.0 | 339.3 | 25.8 | $\pm 0.0172$ |
| **`frequency_only`** | 1 | frozen | completed | 7,400 | 0.0562 | 0.0562 | 0.3700 | 304.2 | 2.3 | 18.4 | $\pm 0.0089$ |
| **`random`** | 1 | baseline | completed | 7,400 | 0.0323 | 0.0323 | 0.2347 | 46.3 | 2.4 | 12.5 | $\pm 0.0004$ |
| | | | | | | | | | | | |
| **`acsh_cold_start`** | 2 | cold_start | completed | 7,300 | **0.2817** | **0.2817** | 0.8984 | 1,412.0 | 439.9 | 162.3 | $\pm 0.0201$ |
| **`cstp_cold_start`** | 2 | cold_start | completed | 7,300 | 0.2729 | 0.2729 | 0.8889 | 648.7 | 293.6 | 25.9 | $\pm 0.0227$ |
| **`st_moe_paper_cold_start`**| 2 | cold_start | completed | 7,300 | 0.2722 | 0.2722 | 0.8904 | 98.1 | 49.6 | 22.3 | $\pm 0.0169$ |
| **`st_moe_paper_online`** | 2 | online | completed | 7,300 | **0.2670** | **0.2670** | 0.9085 | **95.6** | **38.5** | **24.5** | $\pm 0.0234$ |
| **`acsh_global`** | 2 | online | completed | 7,300 | 0.2632 | 0.2632 | **0.9119** | 1,756.0 | 455.5 | 173.3 | $\pm 0.0256$ |
| **`acsh_fast_recovery`** | 2 | online | completed | 7,300 | 0.2626 | 0.2626 | **0.9119** | 1,449.8 | 460.4 | 165.2 | $\pm 0.0257$ |
| **`acsh_stage_aware`** | 2 | online | completed | 7,300 | 0.2624 | 0.2624 | 0.9118 | 1,828.2 | 393.7 | 164.4 | $\pm 0.0255$ |
| **`cstp_st_moe_hybrid`** | 2 | online | completed | 7,300 | 0.2506 | 0.2506 | 0.8982 | 881.1 | 386.3 | 43.8 | $\pm 0.0311$ |
| **`st_moe_paper_frozen`** | 2 | frozen | completed | 7,300 | 0.1827 | 0.1827 | 0.7947 | 159.7 | 9.7 | 20.0 | $\pm 0.0181$ |
| **`spatial_only`** | 2 | online | completed | 7,300 | 0.1808 | 0.1808 | 0.7755 | 696.3 | 315.3 | 25.7 | $\pm 0.0186$ |
| **`cstp_online`** | 2 | online | completed | 7,300 | 0.1625 | 0.1625 | 0.7445 | 722.8 | 342.3 | 39.7 | $\pm 0.0206$ |
| **`cstp_frozen`** | 2 | frozen | completed | 7,300 | 0.1613 | 0.1613 | 0.7414 | 769.8 | 6.3 | 25.7 | $\pm 0.0199$ |
| **`temporal_only`** | 2 | online | completed | 7,300 | 0.1196 | 0.1196 | 0.6215 | 808.2 | 346.7 | 25.8 | $\pm 0.0173$ |
| **`frequency_only`** | 2 | frozen | completed | 7,300 | 0.0562 | 0.0562 | 0.3693 | 297.2 | 2.3 | 18.4 | $\pm 0.0090$ |
| **`random`** | 2 | baseline | completed | 7,300 | 0.0322 | 0.0322 | 0.2338 | 59.4 | 2.6 | 12.5 | $\pm 0.0004$ |
| **`st_moe_legacy`** | 2 | online | **unsupported**| 7,300 | — | — | — | — | — | — | — |
| | | | | | | | | | | | |
| **`acsh_cold_start`** | 3 | cold_start | completed | 7,200 | **0.2839** | **0.2839** | 0.9079 | 1,501.3 | 419.0 | 162.3 | $\pm 0.0210$ |
| **`st_moe_paper_cold_start`**| 3 | cold_start | completed | 7,200 | 0.2754 | 0.2754 | 0.9026 | 84.0 | 48.9 | 22.3 | $\pm 0.0182$ |
| **`cstp_cold_start`** | 3 | cold_start | completed | 7,200 | 0.2736 | 0.2736 | 0.8958 | 879.0 | 375.3 | 25.9 | $\pm 0.0215$ |
| **`st_moe_paper_online`** | 3 | online | completed | 7,200 | **0.2645** | **0.2645** | 0.9071 | **88.2** | **58.8** | **24.5** | $\pm 0.0248$ |
| **`acsh_global`** | 3 | online | completed | 7,200 | 0.2616 | 0.2616 | **0.9106** | 1,926.6 | 589.0 | 173.3 | $\pm 0.0265$ |
| **`acsh_fast_recovery`** | 3 | online | completed | 7,200 | 0.2612 | 0.2612 | 0.9103 | 1,778.4 | 429.4 | 165.2 | $\pm 0.0268$ |
| **`acsh_stage_aware`** | 3 | online | completed | 7,200 | 0.2607 | 0.2607 | 0.9103 | 1,537.5 | 451.9 | 164.4 | $\pm 0.0262$ |
| **`cstp_st_moe_hybrid`** | 3 | online | completed | 7,200 | 0.2483 | 0.2483 | 0.9015 | 981.9 | 515.6 | 43.8 | $\pm 0.0315$ |
| **`st_moe_paper_frozen`** | 3 | frozen | completed | 7,200 | 0.1787 | 0.1787 | 0.7810 | 97.7 | 6.2 | 20.0 | $\pm 0.0170$ |
| **`spatial_only`** | 3 | online | completed | 7,200 | 0.1756 | 0.1756 | 0.7710 | 816.9 | 323.8 | 25.7 | $\pm 0.0177$ |
| **`cstp_online`** | 3 | online | completed | 7,200 | 0.1595 | 0.1595 | 0.7404 | 880.0 | 350.4 | 31.4 | $\pm 0.0199$ |
| **`cstp_frozen`** | 3 | frozen | completed | 7,200 | 0.1580 | 0.1580 | 0.7374 | 817.2 | 5.4 | 25.7 | $\pm 0.0192$ |
| **`temporal_only`** | 3 | online | completed | 7,200 | 0.1199 | 0.1199 | 0.6229 | 962.2 | 397.6 | 25.8 | $\pm 0.0173$ |
| **`frequency_only`** | 3 | frozen | completed | 7,200 | 0.0560 | 0.0560 | 0.3683 | 328.5 | 2.5 | 18.4 | $\pm 0.0093$ |
| **`random`** | 3 | baseline | completed | 7,200 | 0.0323 | 0.0323 | 0.2343 | 49.8 | 2.5 | 12.5 | $\pm 0.0004$ |
| **`st_moe_legacy`** | 3 | online | **unsupported**| 7,200 | — | — | — | — | — | — | — |
| | | | | | | | | | | | |
| **`acsh_cold_start`** | 4 | cold_start | completed | 7,100 | **0.2828** | **0.2828** | 0.9068 | 1,582.0 | 415.0 | 162.3 | $\pm 0.0204$ |
| **`cstp_cold_start`** | 4 | cold_start | completed | 7,100 | 0.2725 | 0.2725 | 0.8975 | 708.9 | 302.1 | 25.9 | $\pm 0.0212$ |
| **`st_moe_paper_cold_start`**| 4 | cold_start | completed | 7,100 | 0.2724 | 0.2724 | 0.9010 | 91.5 | 36.7 | 22.3 | $\pm 0.0181$ |
| **`st_moe_paper_online`** | 4 | online | completed | 7,100 | **0.2615** | **0.2615** | 0.9048 | **81.9** | **35.6** | **24.5** | $\pm 0.0260$ |
| **`acsh_global`** | 4 | online | completed | 7,100 | 0.2573 | 0.2573 | **0.9038** | 1,529.3 | 476.4 | 173.3 | $\pm 0.0281$ |
| **`acsh_fast_recovery`** | 4 | online | completed | 7,100 | 0.2571 | 0.2571 | 0.9037 | 1,523.0 | 435.3 | 165.2 | $\pm 0.0278$ |
| **`acsh_stage_aware`** | 4 | online | completed | 7,100 | 0.2569 | 0.2569 | 0.9037 | 1,662.2 | 466.7 | 164.4 | $\pm 0.0281$ |
| **`cstp_st_moe_hybrid`** | 4 | online | completed | 7,100 | 0.2438 | 0.2438 | 0.8934 | 796.2 | 345.0 | 43.8 | $\pm 0.0324$ |
| **`st_moe_paper_frozen`** | 4 | frozen | completed | 7,100 | 0.1736 | 0.1736 | 0.7710 | 102.1 | 5.2 | 20.0 | $\pm 0.0171$ |
| **`spatial_only`** | 4 | online | completed | 7,100 | 0.1688 | 0.1688 | 0.7554 | 831.2 | 397.1 | 25.7 | $\pm 0.0174$ |
| **`cstp_online`** | 4 | online | completed | 7,100 | 0.1552 | 0.1552 | 0.7338 | 1,017.6 | 425.3 | 31.4 | $\pm 0.0194$ |
| **`cstp_frozen`** | 4 | frozen | completed | 7,100 | 0.1538 | 0.1538 | 0.7287 | 772.6 | 5.3 | 25.7 | $\pm 0.0189$ |
| **`temporal_only`** | 4 | online | completed | 7,100 | 0.1203 | 0.1203 | 0.6248 | 948.8 | 400.5 | 25.8 | $\pm 0.0175$ |
| **`frequency_only`** | 4 | frozen | completed | 7,100 | 0.0559 | 0.0559 | 0.3672 | 323.1 | 2.3 | 18.4 | $\pm 0.0094$ |
| **`random`** | 4 | baseline | completed | 7,100 | 0.0323 | 0.0323 | 0.2342 | 51.8 | 2.5 | 12.5 | $\pm 0.0004$ |
| **`st_moe_legacy`** | 4 | online | **unsupported**| 7,100 | — | — | — | — | — | — | — |

---

## 5. Architectural Explanation of `st_moe_legacy` Incompatibility

The legacy ST-MoE implementation (`src/predictor/st_moe_predictor.py`) only supports $H=1$:
1. **Hardcoded Transitions:** In its `__init__`, it initializes CCT transitions strictly over `range(num_moe_stages - 1)`, which models transitions only from stage $l$ to stage $l+1$.
2. **Hardcoded Source Stage:** In both `predict()` and `update()`, the source stage is hardcoded as `source_stage = target_stage_id - 1`.
3. **No Horizon Parameter:** The interface lacks a `prediction_horizon` argument.

For $H=1$, `st_moe_legacy` was executed and achieved a Recall of $0.2406$ and Hit Rate of $0.8742$, underperforming `st_moe_paper_online` ($0.2706$ Recall, $0.9076$ Hit Rate) by $-3.00$ percentage points. For $H \in \{2, 3, 4\}$, modifying its algorithm was prohibited by experimental controls; thus it is documented as `unsupported_for_H_gt_1`.

---

## 6. Generated Figures

The following publication figures were generated and saved under `results/endsem_moe_suite_horizon/figures/`:

1. **`recall_vs_horizon.png`:** Macro Recall @ $B=8$ across horizons $H=1, 2, 3, 4$.
2. **`f1_vs_horizon.png`:** Macro F1-Score across horizons $H=1, 2, 3, 4$.
3. **`latency_vs_horizon.png`:** Predictor query latency scaling across horizons.
4. **`pareto_frontier_horizons.png`:** Recall vs Latency Pareto trade-off subplots for each horizon.

---

## 7. Experimental Validity and Reproducibility

- **Task-Disjoint Verification:** Training was conducted exclusively on corpora 2 through 9. Testing was conducted exclusively on corpora 12 and 13.
- **Zero Ternary Operators:** AST parsing confirmed `ast.IfExp == 0` in all runner and plotting scripts.
- **Cluster Bootstrap Uncertainty:** Task-clustered standard errors ($\approx \pm 0.017 - 0.034$) indicate consistent relative rankings across both test tasks.

