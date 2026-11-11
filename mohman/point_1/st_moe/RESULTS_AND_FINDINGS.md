# ST-MoE Empirical Results and Comprehensive Findings

**Deliverable:** Point 1 — Paper-Faithful ST-MoE Baseline Reproduction  
**Result Datasets:** [`mohman/point_1/st_moe/results/csv/`](file:///home/mohman/ca_project/mohman/point_1/st_moe/results/csv/)  
**Visualizations:** [`mohman/point_1/st_moe/plots/`](file:///home/mohman/ca_project/mohman/point_1/st_moe/plots/)

---

## 1. Primary Fixed-Budget Scaling Analysis

In hardware prefetching, prefetch buffers and DRAM channels operate under strict capacity constraints. Table 1 details the scaling of ST-MoE performance across budgets $B \in \{1, 2, 4, 6, 8\}$ for both authoritative workloads under Horizon $H=1$:

### Table 1: ST-MoE Fixed-Budget Scaling ($H=1$, Confidence Additive Scoring)

| Workload | Budget $B$ | Regime | Recall (%) | Precision (%) | F1 (%) | Hit Rate (%) [Hits $\ge 1$] | Cold Recall (%) | Steady Recall (%) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MoE Anywhere** | 1 | Frozen | 7.84% | 62.73% | 13.94% | 62.73% | 5.83% | 7.91% |
| (Sage-A1.5, 40L, 256E) | 1 | Online (Across) | 9.44% | 75.56% | 16.79% | 75.56% | 5.49% | 9.58% |
| | 2 | Frozen | 14.92% | 59.66% | 23.87% | 78.44% | 10.71% | 15.06% |
| | 2 | Online (Across) | 17.55% | 70.20% | 28.08% | 87.40% | 10.96% | 17.78% |
| | 4 | Frozen | 26.77% | 53.55% | 35.70% | 88.85% | 19.31% | 27.03% |
| | 4 | Online (Across) | 30.88% | 61.77% | 41.18% | 93.69% | 20.25% | 31.25% |
| | 6 | Frozen | 35.83% | 47.77% | 40.95% | 92.58% | 26.18% | 36.16% |
| | 6 | Online (Across) | 41.41% | 55.21% | 47.32% | 95.85% | 27.30% | 41.89% |
| | **8** | **Frozen** | **42.39%** | **42.39%** | **42.39%** | **94.33%** | 31.85% | 42.75% |
| | **8** | **Online (Within)** | **47.91%** | **47.91%** | **47.91%** | **96.94%** | **31.85%** | **48.47%** |
| | **8** | **Online (Across)** | **49.45%** | **49.45%** | **49.45%** | **96.96%** | **33.17%** | **50.01%** |
| \midrule | | | | | | | | |
| **MoE Suite** | 1 | Frozen | 2.56% | 20.51% | 4.56% | 20.51% | 5.58% | 2.50% |
| (GLM-5.2, 75L, 256E) | 1 | Online (Across) | 6.60% | 52.84% | 11.74% | 52.84% | 6.50% | 6.61% |
| | 2 | Frozen | 5.04% | 20.16% | 8.06% | 35.17% | 10.33% | 4.93% |
| | 2 | Online (Across) | 11.24% | 44.97% | 17.99% | 67.77% | 12.17% | 11.22% |
| | 4 | Frozen | 9.19% | 18.38% | 12.25% | 53.12% | 18.25% | 9.00% |
| | 4 | Online (Across) | 17.93% | 35.85% | 23.90% | 80.53% | 20.17% | 17.88% |
| | 6 | Frozen | 12.79% | 17.05% | 14.61% | 64.68% | 24.83% | 12.54% |
| | 6 | Online (Across) | 22.94% | 30.59% | 26.22% | 86.61% | 25.58% | 22.89% |
| | **8** | **Frozen** | **15.85%** | **15.85%** | **15.85%** | **72.08%** | 30.00% | 15.56% |
| | **8** | **Online (Within)** | **26.53%** | **26.53%** | **26.53%** | **89.84%** | **30.00%** | **26.46%** |
| | **8** | **Online (Across)** | **26.74%** | **26.74%** | **26.74%** | **89.81%** | **29.75%** | **26.68%** |

*(Visualized in [`plots/frozen_vs_online.png`](file:///home/mohman/ca_project/mohman/point_1/st_moe/plots/frozen_vs_online.png) and [`plots/hit_rate_comparison.png`](file:///home/mohman/ca_project/mohman/point_1/st_moe/plots/hit_rate_comparison.png))*

---

## 2. Online Adaptation Policy Breakdown

Phase 2B quantified the exact source of online adaptation gains by isolating intra-request learning from cumulative cross-request drift:

### Table 2: Online Policy Gain Decomposition ($B = K = 8$)

| Dataset | Horizon | Frozen Recall | Online (Within) Recall | Online (Across) Recall | Intra-Request Gain ($\Delta_{\text{intra}}$) | Cross-Request Gain ($\Delta_{\text{cross}}$) | Total Online Gain ($\Delta_{\text{total}}$) | % Gain from Intra-Request |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MoE Anywhere** | $H=1$ | 42.39% | **47.91%** | 49.45% | **+5.52 pp** | +1.54 pp | +7.06 pp | **78.19%** |
| **MoE Anywhere** | $H=2$ | 39.78% | **46.20%** | 47.83% | **+6.42 pp** | +1.63 pp | +8.05 pp | **79.75%** |
| **MoE Suite** | $H=1$ | 15.85% | **26.53%** | 26.74% | **+10.68 pp** | +0.21 pp | +10.89 pp | **98.07%** |
| **MoE Suite** | $H=2$ | 15.35% | **25.90%** | 26.08% | **+10.55 pp** | +0.18 pp | +10.73 pp | **98.32%** |

### Key Observations:
1. **Intra-Prompt Dynamics Dominate:** Across all workloads, **78% to 98% of total online gain is generated purely by adapting to token transitions within the active request.**
2. **Minimal Cross-Request Bleed in MoE Suite:** On GLM-5.2, carrying CCT updates across tasks adds only +0.21 pp, proving that distinct NLP tasks possess largely disjoint correlation structures.
3. **Multi-Tenant Safety:** Standardizing on `online_within_request` yields >85% of theoretical lifelong adaptation while eliminating the risk of inter-user cache interference.

---

## 3. Lookahead Horizon Degradation ($H=1$ vs. $H=2$)

Table 3 evaluates predictor degradation when the prefetcher is forced to look 2 layers ahead:

### Table 3: Lookahead Horizon Comparison (Primary Standard: `online_within_request`, $B=K=8$)

| Dataset | Metric | Horizon $H=1$ (Adjacent) | Horizon $H=2$ (Skip-Layer) | Absolute Delta ($\Delta$) | Relative Retention |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MoE Anywhere** | Recall@8 | 47.91% | 46.20% | -1.71 pp | **96.43%** |
| (Sage-A1.5) | Hit Rate | 96.94% | 95.85% | -1.09 pp | **98.88%** |
| | Cold Recall | 31.85% | 32.01% | +0.16 pp | 100.50% |
| | Steady Recall | 48.47% | 46.69% | -1.78 pp | 96.33% |
| \midrule | | | | | |
| **MoE Suite** | Recall@8 | 26.53% | 25.90% | -0.63 pp | **97.63%** |
| (GLM-5.2) | Hit Rate | 89.84% | 88.95% | -0.89 pp | **99.01%** |
| | Cold Recall | 30.00% | 29.67% | -0.33 pp | 98.90% |
| | Steady Recall | 26.46% | 25.82% | -0.64 pp | 97.58% |

*(Visualized in [`plots/horizon_comparison.png`](file:///home/mohman/ca_project/mohman/point_1/st_moe/plots/horizon_comparison.png))*

### Key Finding:
Both architectures maintain **>96% of predictive recall** when extending the lookahead horizon from 1 to 2 stages. This confirms that transformer expert correlation persists across multi-layer skips, providing sufficient computation time for asynchronous DRAM-to-HBM prefetching.

---

## 4. Native Thresholding Pathology and Dual Scoring Analysis

The paper proposes admitting all experts with $\text{Score} \ge \tau = 2$. Table 4 contrasts the two mathematical interpretations of Equation 1:

### Table 4: Native Thresholding Tradeoff ($\tau = 2$, $H=1$)

| Workload | Scoring Mode | Regime | Mean Set Size $\overline{\|P\|}$ | Min / Max Set Size | Recall (%) | Precision (%) | Hit Rate (%) | Exact Match (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MoE Anywhere** | `confidence_additive` | Frozen | **32.02** | 0 / 66 | 76.02% | 20.73% | 98.15% | 0.00% |
| (Target $K=8$) | `confidence_additive` | Online (Across) | **20.88** | 0 / 53 | 72.70% | 30.25% | 98.11% | 0.00% |
| | `indicator_intersection`| Frozen | **4.17** | 0 / 8 | 27.69% | 46.02% | 77.76% | 0.04% |
| | `indicator_intersection`| Online (Across) | **5.36** | 0 / 8 | 31.06% | 40.30% | 79.92% | 0.03% |
| \midrule | | | | | | | | |
| **MoE Suite** | `confidence_additive` | Frozen | **44.11** | 0 / 67 | 52.89% | 9.80% | 98.63% | 0.00% |
| (Target $K=8$) | `confidence_additive` | Online (Across) | **27.93** | 0 / 63 | 46.92% | 14.32% | 98.08% | 0.00% |
| | `indicator_intersection`| Frozen | **2.64** | 0 / 8 | 13.20% | 35.42% | 61.81% | 0.00% |
| | `indicator_intersection`| Online (Across) | **4.58** | 0 / 8 | 18.03% | 28.65% | 70.37% | 0.00% |

*(Visualized in [`plots/native_threshold_tradeoff.png`](file:///home/mohman/ca_project/mohman/point_1/st_moe/plots/native_threshold_tradeoff.png))*

### Key Insights:
1. **Bandwidth Explosion:** Under `confidence_additive`, the predictor emits an average of **20.88 to 44.11 experts**, representing a **$2.6\times$ to $5.5\times$ memory bandwidth amplification** over the true target ($K=8$). This collapses precision down to 9.8%–30.2%.
2. **Intersection Precision Recovery:** The `indicator_intersection` rule constrains candidate set sizes ($2.64$ to $5.36$ experts) and improves precision up to 46.02%, but artificially caps recall to 18.0%–31.1% because candidates must exist simultaneously in both tables.

---

## 5. Offline Profiling Sensitivity Campaign

We evaluated the sensitivity of ST-MoE to calibration data volume by sweeping profiling fractions from $10\%$ to $100\%$:

### Table 5: Profiling Sensitivity on MoE Anywhere ($H=1$, $B=K=8$)

| Profiling Fraction | Calibration Prompts | Observed Tokens | Frozen Recall@8 | Online Recall@8 | Threshold Set Size $\overline{\|P\|}$ | Threshold Recall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10%** | 5 | 1,285 | 34.13% | **48.75%** | 20.72 | 71.55% |
| **20%** | 10 | 2,570 | 34.92% | **48.79%** | 20.77 | 71.66% |
| **40%** | 21 | 5,397 | 38.12% | **49.12%** | 20.87 | 72.21% |
| **60%** | 31 | 7,967 | 40.82% | **49.29%** | 20.87 | 72.50% |
| **80%** | 42 | 10,794 | 41.79% | **49.36%** | 20.88 | 72.59% |
| **100%** | 52 | 13,364 | 42.39% | **49.45%** | 20.88 | 72.70% |

*(Visualized in [`plots/profiling_sensitivity.png`](file:///home/mohman/ca_project/mohman/point_1/st_moe/plots/profiling_sensitivity.png))*

### Key Insight:
Under frozen serving, recall scales monotonically from 34.13% to 42.39% as more profiling data is ingested. However, **with online adaptation enabled, even a tiny 10% profiling set achieves 48.75% recall**, within 0.7 pp of full 100% calibration. This proves that ST-MoE's online dynamic update rapidly fills gaps in sparse offline profiling tables.

