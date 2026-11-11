# Phase C: Online vs. Frozen Adaptation Results

## 1. Executive Summary

This report documents the experimental comparison between **Online Causal Adaptation** (Experiment B) and **Frozen Profiling** (Experiment A), alongside **Cold-Start Adaptation** (Experiment C), across all 5 evaluation workloads.

Key Findings:
1. **Online Adaptation Delivers Massive Recall Gains Under Dynamic Workloads**: On synthetic correlated workloads, online adaptation increases CSTP recall from 0.1504 to 0.4775 (+217.5% relative gain) and ST-MoE Paper recall from 0.1955 to 0.4963 (+153.9% relative gain).
2. **ST-MoE Adapts More Aggressively on Disjoint Tasks**: On real production traces (`moe_suite`), ST-MoE Paper online achieves a +45.9% relative gain over frozen (0.2608 vs 0.1788, $\Delta = +0.0820$), whereas CSTP achieves a modest +0.7% gain.
3. **Negative Transfer in Offline Profiling**: When evaluated across disjoint task distributions (`moe_suite` and `moe_anywhere`), cold-start predictors (with 0 offline profiling) outperform frozen predictors. Offline profiling across disparate corpora induces cross-task statistical interference, whereas online-adaptive models rapidly specialize to the current context.

---

## 2. Quantitative Results at Primary Budgets

The following table summarizes Macro Recall at the primary prefetch budget ($B=8$ for $E=256$, $B=2$ for $E=8$):

| Workload ID | Budget ($B$) | CSTP Frozen | CSTP Online | CSTP Delta ($\Delta$) | ST-MoE Frozen | ST-MoE Online | ST-MoE Delta ($\Delta$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`deepseek_random`** | 8 | 0.9974 | 0.9974 | 0.0000 (0.0%) | 0.9829 | 0.9807 | -0.0022 (-0.2%) |
| **`duplex_a100`** | 2 | 0.2500 | 0.2445 | -0.0055 (-2.2%) | 0.2555 | 0.2520 | -0.0035 (-1.4%) |
| **`correlated_synthetic`**| 8 | 0.1504 | 0.4775 | **+0.3271 (+217.5%)** | 0.1955 | 0.4963 | **+0.3008 (+153.9%)** |
| **`moe_anywhere` ($H=1$)** | 8 | 0.5125 | 0.5223 | **+0.0098 (+1.9%)** | 0.4947 | 0.5507 | **+0.0560 (+11.3%)** |
| **`moe_anywhere` ($H=2$)** | 8 | 0.5001 | 0.5115 | **+0.0114 (+2.3%)** | 0.4746 | 0.5340 | **+0.0594 (+12.5%)** |
| **`moe_suite` ($H=2$)** | 8 | 0.1603 | 0.1614 | **+0.0011 (+0.7%)** | 0.1788 | 0.2608 | **+0.0820 (+45.9%)** |

---

## 3. Comprehensive Budget Sweep Analysis

The prefetch budget was swept from $B=4$ to $B=64$ across all large-scale workloads ($E=256$):

### A. Correlated Synthetic Trace (H=1)
| Budget ($B$) | CSTP Frozen | CSTP Online | CSTP Cold | ST-MoE Frozen | ST-MoE Online | ST-MoE Cold | Frequency | Random |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **4** | 0.0963 | 0.3121 | 0.3283 | 0.1197 | 0.3248 | 0.3340 | 0.0910 | 0.0160 |
| **8** | 0.1504 | 0.4775 | 0.4970 | 0.1955 | 0.4963 | 0.5078 | 0.1490 | 0.0339 |
| **16** | 0.2373 | 0.6728 | 0.6974 | 0.3180 | 0.6866 | 0.7029 | 0.2452 | 0.0654 |
| **24** | 0.3086 | 0.7671 | 0.7853 | 0.4079 | 0.7818 | 0.7963 | 0.3175 | 0.0954 |
| **32** | 0.3662 | 0.8228 | 0.8351 | 0.4768 | 0.8357 | 0.8491 | 0.3789 | 0.1257 |
| **48** | 0.4572 | 0.8879 | 0.8936 | 0.5693 | 0.8934 | 0.9015 | 0.4796 | 0.1873 |
| **64** | 0.5283 | 0.9230 | 0.9268 | 0.6384 | 0.9254 | 0.9300 | 0.5594 | 0.2493 |

### B. MoE Anywhere Real Trace (H=1)
| Budget ($B$) | CSTP Frozen | CSTP Online | CSTP Cold | ST-MoE Frozen | ST-MoE Online | ST-MoE Cold | Frequency | Random |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **4** | 0.3168 | 0.3228 | 0.3510 | 0.3134 | 0.3470 | 0.3475 | 0.1485 | 0.0150 |
| **8** | 0.5125 | 0.5223 | 0.5906 | 0.4947 | 0.5507 | 0.5485 | 0.2260 | 0.0309 |
| **16** | 0.6963 | 0.7077 | 0.7898 | 0.6811 | 0.7554 | 0.7490 | 0.3503 | 0.0621 |
| **24** | 0.7820 | 0.7937 | 0.8545 | 0.7681 | 0.8335 | 0.8270 | 0.4254 | 0.0930 |
| **32** | 0.8339 | 0.8443 | 0.8869 | 0.8210 | 0.8753 | 0.8698 | 0.5166 | 0.1244 |
| **48** | 0.8931 | 0.9015 | 0.9177 | 0.8809 | 0.9178 | 0.9157 | 0.6182 | 0.1885 |
| **64** | 0.9242 | 0.9317 | 0.9332 | 0.9135 | 0.9427 | 0.9413 | 0.7017 | 0.2517 |

---

## 4. Cold-Start Dynamics & Negative Transfer Phenomenon

An unexpected and vital theoretical finding emerged from Experiment C (`cold_start`):
- On **`moe_suite`**, `cstp_cold_start` achieves Recall@8 of **0.2705**, compared to **0.1603** for `cstp_frozen` and **0.1614** for `cstp_online`.
- Similarly, `st_moe_paper_cold_start` achieves Recall@8 of **0.2659**, vastly outperforming `st_moe_paper_frozen` (**0.1788**).

### Root Cause Analysis: Cross-Task Disjoint Distribution Shift
In `moe_suite`, the training split consists of 8 disparate NLP tasks (e.g., summarization, code generation, translation), while the test split comprises completely disjoint tasks (`corpus-12` and `corpus-13`).
1. **Frozen CSTP**: Accumulates millions of transition counts during offline profiling across the 8 training tasks. When evaluated on `corpus-12`, these historical transition matrices act as high-inertia priors, heavily penalizing new routing pathways specific to the test task.
2. **Cold-Start CSTP**: Starts with zero counts and a flat Dirichlet prior. It adapts immediately to the token sequence of `corpus-12`, without dragging the legacy baggage of unrelated tasks.
3. **ST-MoE Paper**: Because its CCT uses small 2-bit counters that quickly saturate and replace zero-confidence candidates, online ST-MoE rapidly purges outdated training entries within 2–3 token steps, allowing it to adapt much faster than CSTP's unbounded count matrices.
