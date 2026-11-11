# Point 1: ST-MoE Expert Prediction Baseline Reproduction

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Deliverable:** Point 1 — Paper-Faithful ST-MoE Baseline Reproduction & Evaluation Package  
**Target Repository Path:** [`mohman/point_1/st_moe/`](file:///home/mohman/ca_project/mohman/point_1/st_moe/)  
**Authoritative Baseline Git Anchor:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Reference Paper:** Zhao et al., *"A Spatio-Temporal Expert Prefetching Framework for Efficient MoE-based LLM Inference"*, [arXiv:2606.15453v1](file:///home/mohman/ca_project/mohman/point_1/st_moe/references/Zhao_et_al_STMoE_2606.15453v1.pdf)

---

## 1. Executive Summary

This deliverable provides the authoritative reproduction, experimental isolation, causal hardening, and empirical evaluation of the **ST-MoE** (Spatio-Temporal Mixture-of-Experts) expert prediction framework described by Zhao et al.

ST-MoE is a **training-free predictor** designed to forecast routed experts several layers in advance ($H \in \{1, 2\}$) by combining:
1. **Spatial Expert Correlation:** Captured via an offline-profiled and dynamically updated **Correlation Confidence Table (CCT)** using 2-bit saturating counters ($0 \dots 3$).
2. **Temporal Expert Correlation:** Tracked via a request-local **History Table (HT)** of recently activated experts.
3. **Dynamic Scoring Strategy:** Evaluated via the paper's Equation 1: $\text{Score}(e) = \text{CCT}(e) + \text{HT}(e)$.

### Primary Empirical Standard: `online_within_request`
Following the Phase 2B evaluation semantics audit, all primary benchmarks operate under the **`online_within_request`** protocol. In this protocol, the predictor begins each test request from its offline-profiled CCT, adapts dynamically to intra-prompt token transitions, and causally resets to the profiled baseline at request boundaries. This guarantees strict request-order invariance, prevents synthetic cross-prompt drift, and models real-world multi-tenant inference servers.

---

## 2. Authoritative Headline Results

The table below presents the verified performance of the paper-faithful ST-MoE baseline across our two authoritative open-source MoE routing traces under the primary standard ($B = K = 8$, `online_within_request`):

| Workload / Model | MoE Stages | Routed Experts | Routing Degree | Lookahead Horizon | Test Opportunities | Recall@8 (%) | Precision@8 (%) | Hit Rate (%) [Hits $\ge 1$] | Cold-Start Recall (%) | Steady-State Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MoE Anywhere** (Sage-A1.5) | 40 | 256 | Top-8 | **$H=1$** (Next Layer) | 16,800 | **47.91%** | **47.91%** | **96.94%** | 31.85% | 48.47% |
| **MoE Anywhere** (Sage-A1.5) | 40 | 256 | Top-8 | **$H=2$** (Skip Layer) | 16,800 | **46.20%** | **46.20%** | **95.85%** | 32.01% | 46.69% |
| **MoE Suite** (GLM-5.2) | 75 | 256 | Top-8 | **$H=1$** (Next Layer) | 7,500 | **26.53%** | **26.53%** | **89.84%** | 30.00% | 26.46% |
| **MoE Suite** (GLM-5.2) | 75 | 256 | Top-8 | **$H=2$** (Skip Layer) | 7,500 | **25.90%** | **25.90%** | **88.95%** | 29.67% | 25.82% |

> [!NOTE]
> When budget $B$ equals the model's routing degree $K$ ($B = K = 8$), the number of predicted experts equals the number of ground truth experts, rendering micro-averaged Recall, Precision, and F1 mathematically identical ($47.91\%$).

---

## 3. Core Technical Findings

### 3.1 Intra-Request Adaptation Dominates Online Gain
Comparing `frozen` (static CCT), `online_within_request`, and `online_across_requests`:
- On `moe_anywhere` ($H=1$), `frozen` achieves **42.39%** recall.
- Enabling intra-request adaptation (`online_within_request`) boosts recall to **47.91%** (+5.52 pp).
- Persisting CCT updates across disjoint test prompts (`online_across_requests`) yields **49.45%** (+1.54 pp further).
- **Conclusion:** **78.2% of total online adaptation gain is harvested within the active request.** Intra-prompt context adaptation is the primary driver of correlation learning.

### 3.2 Resolution of the Paper's Claimed "$\sim 85\%$ Accuracy"
Zhao et al. report approximately 85% prediction accuracy without providing a formal mathematical formula. Our reproduction demonstrates:
- Exact Top-8 expert set **Recall@8 is 47.91%** on `moe_anywhere` and **26.53%** on `moe_suite`.
- However, the **Hit Rate** ($\mathbb{P}(\ge 1 \text{ hit})$) is **96.94%** on `moe_anywhere` and **89.84%** on `moe_suite`.
- **Finding:** The paper's colloquial "85% accuracy" matches a relaxed hit-rate metric or a dilated threshold set, not complete expert set recall. Full details are in [`PAPER_COMPARISON.md`](file:///home/mohman/ca_project/mohman/point_1/st_moe/PAPER_COMPARISON.md).

### 3.3 Severe Set-Size Dilation Under Native Thresholding
The paper advocates dynamic thresholding with $\tau = 2$. We evaluated both mathematical interpretations:
1. **Additive Equation 1 ($\text{CCT} + \text{HT} \ge 2$):** Produces severe set-size dilation with an average predicted set size of **20.88 to 32.02 experts** (vs target $K=8$). This inflates recall (72.7%–76.0%) but causes precision collapse to **18.9%–27.8%**, overwhelming memory prefetch bandwidth.
2. **Intersection ($\mathbb{I}(\text{CCT}) + \mathbb{I}(\text{HT}) \ge 2$):** Restricts average set size to **4.07 to 5.36 experts**, achieving 46.0% precision, but recall drops to **27.7%–31.1%**.
- **Architectural Implication:** Hardware prefetchers require bounded, fixed-budget ranking ($B \le K$) to avoid saturating HBM/DRAM channels.

### 3.4 Cold-Start vs. Steady-State Disparity
During initial prompt tokens (tokens 0–4), the History Table is unpopulated, yielding a cold-start recall of **31.85%**. As temporal context accumulates (tokens 5+), steady-state recall rises to **48.47%**.

---

## 4. Deliverable Directory Map

This self-contained package is organized into dedicated technical modules:

```
mohman/point_1/st_moe/
├── README.md                      # Primary overview, headline table, directory guide
├── ST_MOE_METHODOLOGY.md          # Architectural algorithms, CCT/HT logic, Eq 1 interpretations
├── EXPERIMENT_CONFIGURATION.md    # Configurations, isolation protocols, runner flags
├── DATASET_AND_SPLIT_DETAILS.md   # Trace specifications, splits, token counts, opportunity definition
├── EVALUATION_METHODOLOGY.md      # Metric definitions, online policies, causality guarantees
├── RESULTS_AND_FINDINGS.md        # Comprehensive results analysis across all sweeps
├── PAPER_COMPARISON.md            # In-depth audit of Zhao et al., 85% claim investigation
├── REPRODUCIBILITY.md             # Complete step-by-step reproduction guide
├── configs/
│   ├── README.md                  # Configuration manifest guide
│   └── experiment_manifest.json   # Machine-readable parameters, splits, and seeds
├── results/
│   ├── README.md                  # Data dictionary and schema documentation
│   └── csv/
│       ├── fixed_budget_summary.csv        # Budget sweeps (B in 1, 2, 4, 6, 8)
│       ├── per_experiment_results.csv      # Complete 86-experiment catalog
│       ├── online_policy_comparison.csv    # Frozen vs within-req vs across-req
│       ├── native_threshold_results.csv    # Dual scoring mode threshold results
│       ├── profiling_sensitivity.csv       # 10% to 100% offline profiling sweeps
│       └── per_opportunity_results.csv     # Granular opportunity trace schema
├── plots/
│   ├── README.md                  # Figure catalog and visual findings
│   ├── recall_comparison.png      # Primary recall comparison (PNG & PDF)
│   ├── frozen_vs_online.png       # Recall curves vs budget B (PNG & PDF)
│   ├── horizon_comparison.png     # Horizon H=1 vs H=2 degradation (PNG & PDF)
│   ├── hit_rate_comparison.png    # Hit rate scaling and paper anchor (PNG & PDF)
│   ├── cold_vs_warm_recall.png    # Cold-start vs steady-state recall (PNG & PDF)
│   ├── native_threshold_tradeoff.png # Dilation & precision tradeoff (PNG & PDF)
│   └── profiling_sensitivity.png  # Profiling fraction sensitivity (PNG & PDF)
├── scripts/
│   ├── README.md                  # Script instructions and test suite references
│   ├── generate_results.py        # Harmonizes and exports standardized CSVs
│   └── generate_plots.py          # Generates publication-ready PNG & PDF figures
└── references/
    ├── README.md                  # Paper citations and mapping
    └── Zhao_et_al_STMoE_2606.15453v1.pdf # Full local PDF copy of reference paper
```

