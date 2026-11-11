# Comparative Experimental Design

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Deliverable:** Point 1 — Comparative Study Methodology  
**Path:** [`mohman/point_1/stmoe_cstp/methodology/experimental_design.md`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/methodology/experimental_design.md)  

---

## 1. Experimental Objectives and Controlled Variables

The comparative evaluation tests the predictive efficacy, adaptation dynamics, and hardware efficiency of CSTP and ST-MoE under matched experimental controls:

1. **Independent Variables:**
   - **Predictor Framework:** CSTP vs ST-MoE.
   - **Offline Profiling Fraction:** 25%, 50%, 75%, 100% of designated training traces.
   - **Profiling Random Seeds:** Seeds 42, 123, 999.
   - **Lookahead Horizons:** $H=1$ (adjacent stage) and $H=2$ (skip stage).
   - **Evaluation Regimes:** `frozen` vs `online_within_request`.
   - **Workloads:** `moe_anywhere` (Sage-A1.5) and `moe_suite` (GLM-5.2).
2. **Controlled Constants:**
   - **Prediction Budget:** $B = 8$ (fixed to match model routing degree $K=8$).
   - **Causal Evaluation Standard:** Predict $\to$ Score $\to$ Update order.
   - **Test Opportunities:** Identical test sequences and opportunities evaluated across both predictors ($16,800$ for MoE Anywhere, $7,500$ for MoE Suite).
   - **Hardware Interface Model:** PCIe Gen5 x16 / CXL host-to-accelerator interconnect constraints.

---

## 2. Workload Specifications

| Parameter | MoE Anywhere (Sage-A1.5) | MoE Suite (GLM-5.2) |
| :--- | :---: | :---: |
| **Model Family** | DeepSeek/Sage-A1.5 MoE | GLM-5.2 MoE |
| **MoE Stages ($S$)** | 40 | 75 |
| **Experts per Stage ($M$)** | 256 | 256 |
| **Routing Top-$K$** | 8 | 8 |
| **Profiling Prompts (100%)** | 42 | 100 |
| **Evaluation Prompts** | 14 | 100 (30 `gsm8k` + 70 `alpaca`) |
| **Evaluation Token Steps** | 420 | 100 |
| **Total Test Opportunities** | $14 \times 40 \times (420 - 120) = 16,800$ | $100 \times 75 = 7,500$ |
| **Workload Classification** | Authoritative Primary | Authoritative Deep / Multitask |

---

## 3. Subsampling and Profiling Protocol

To isolate sample efficiency:
- At 25%, 50%, and 75%, prompts from the offline training split are uniformly subsampled without replacement using the designated random seed (`42, 123, 999`).
- Subsampled prompt manifests are saved in `profiling_subset_manifest.csv`.
- Each predictor is initialized from scratch and fits its offline tables (`fit_profile()` for CSTP, CCT construction for ST-MoE) on the exact same subsampled prompt subset.
- Evaluation split is strictly held out (100% of test prompts evaluated across all runs).

