# CSTP Empirical Experimental Analysis: Phase 1A Findings

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Deliverable:** Point 1 — Standalone CSTP Experimental Analysis  
**Path:** [`mohman/point_1/cstp/reports/experimental_analysis.md`](file:///home/mohman/ca_project/mohman/point_1/cstp/reports/experimental_analysis.md)  
**Authoritative Data Sources:**  
- [`corrected_aggregate_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_aggregate_metrics.csv)  
- [`corrected_st_moe_head_to_head.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_st_moe_head_to_head.csv)  
- [`corrected_profiling_count_statistics.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_profiling_count_statistics.csv)  
- [`corrected_statistical_analysis.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_statistical_analysis.csv)  

---

## 1. Executive Experimental Overview

Phase 1A conducted an exhaustive profiling-percentage sensitivity sweep on the original CSTP predictor across:
- **Workloads:** `moe_anywhere` (Sage-A1.5, 40 stages, 256 experts, Top-8) and `moe_suite` (GLM-5.2, 75 stages, 256 experts, Top-8).
- **Profiling Percentages:** 25%, 50%, 75%, 100% of offline training traces.
- **Profiling Random Seeds:** 42, 123, 999.
- **Lookahead Horizons:** $H = 1$ (adjacent stage) and $H = 2$ (skip stage).
- **Evaluation Regimes:** `frozen` (static count matrices) and `online_within_request` (causal within-request adaptation with prompt reset).
- **Total Opportunities Evaluated:** 16,800 opportunities on `moe_anywhere` and 7,500 opportunities on `moe_suite` per configuration (totaling 2,332,800 event predictions).

---

## 2. Profiling Sensitivity Analysis

### 2.1 MoE Anywhere Scaling Dynamics ($H=1$, Seed 42)

On `moe_anywhere`, increasing offline profiling data monotonically improves frozen recall while reducing the relative magnitude of online adaptation gain:

| Profiling % | Total Spatial Mass | Total Temporal Mass | Frozen Recall@8 (%) | Online Recall@8 (%) | Online Gain (pp) | Hit Rate (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **25%** | 309,600 | 38,700 | 43.83% | 44.68% | **+0.85 pp** | 96.54% |
| **50%** | 619,200 | 77,400 | 44.53% | 44.85% | **+0.32 pp** | 96.64% |
| **75%** | 928,800 | 116,100 | 45.27% | 45.48% | **+0.21 pp** | 96.81% |
| **100%** | 1,238,400 | 154,800 | **45.37%** | **45.51%** | **+0.14 pp** | 96.67% |

#### Observations:
1. **Diminishing Returns on Offline Profiling:** Frozen recall climbs from 43.83% to 44.53% (+0.70 pp) moving from 25% to 50%, but only gains +0.10 pp moving from 75% to 100% (45.27% to 45.37%).
2. **Online Adaptation Gain Attenuation:** As offline profiling increases, the online gain over the frozen baseline decays sharply from +0.85 pp (at 25%) down to +0.14 pp (at 100%).
3. **High Hit Rate Stability:** Across all profiling percentages, CSTP reliably identifies at least one correct expert in $> 96.5\%$ of all execution opportunities.

---

### 2.2 MoE Suite Scaling Dynamics ($H=1$, Seed 42)

| Profiling % | Total Spatial Mass | Total Temporal Mass | Frozen Recall@8 (%) | Online Recall@8 (%) | Online Gain (pp) | Hit Rate (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **25%** | 202,500 | 27,000 | 15.05% | 15.48% | **+0.43 pp** | 81.33% |
| **50%** | 405,000 | 54,000 | 14.95% | 15.21% | **+0.26 pp** | 81.27% |
| **75%** | 607,500 | 81,000 | **18.03%** | **18.14%** | **+0.11 pp** | 85.57% |
| **100%** | 810,000 | 108,000 | 16.47% | 16.59% | **+0.12 pp** | 82.59% |

#### Observations:
1. **Lower Absolute Performance:** MoE Suite exhibits much lower predictive accuracy (~15%–18% vs ~44%–45% on MoE Anywhere). This reflects GLM-5.2's deeper architecture (75 stages) and higher router dispersion across distinct tasks.
2. **Non-Monotonicity (75% vs 100% Reversal):** On MoE Suite, 75% profiling achieves higher recall (18.03%) than 100% profiling (16.47%). As detailed in Section 5, this stems from task distribution divergence in the training split.

---

## 3. Lookahead Horizon Comparison ($H=1$ vs. $H=2$)

Comparing single-stage lookahead ($H=1$) against two-stage lookahead ($H=2$) at 100% profiling (Seed 42):

| Workload | Metric | Horizon $H=1$ | Horizon $H=2$ | Delta ($H=2 - H=1$) | Relative Retention |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MoE Anywhere** | Frozen Recall@8 | 45.37% | 44.13% | -1.24 pp | **97.27%** |
| **MoE Anywhere** | Online Recall@8 | 45.51% | 44.24% | -1.27 pp | **97.21%** |
| **MoE Anywhere** | Online Hit Rate | 96.67% | 95.84% | -0.83 pp | **99.14%** |
| **MoE Suite** | Frozen Recall@8 | 16.47% | 16.27% | -0.20 pp | **98.79%** |
| **MoE Suite** | Online Recall@8 | 16.59% | 16.36% | -0.23 pp | **98.61%** |
| **MoE Suite** | Online Hit Rate | 82.59% | 82.20% | -0.39 pp | **99.53%** |

### Takeaway:
CSTP exhibits exceptional retention across horizons, maintaining $> 97\%$ of its predictive power at $H=2$. While inter-layer routing correlation diminishes slightly over 2 stages, the stage marginal prior and temporal components remain robustly invariant to $H$.

---

## 4. Profiling Random Seed Variability

To verify that findings are not artifacts of a single profiling sample, we evaluated three independent random subsampling seeds (`42, 123, 999`) across all configurations.

### MoE Anywhere ($H=1$, Online Within-Request)

| Profiling % | Seed 42 Recall (%) | Seed 123 Recall (%) | Seed 999 Recall (%) | 3-Seed Mean (%) | Std Dev (pp) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **25%** | 44.68% | 44.47% | 44.88% | **44.68%** | 0.21 pp |
| **50%** | 44.85% | 44.91% | 44.97% | **44.91%** | 0.06 pp |
| **75%** | 45.48% | 45.42% | 45.56% | **45.49%** | 0.07 pp |
| **100%** | 45.51% | 45.51% | 45.51% | **45.51%** | 0.00 pp |

*Note: At 100% profiling, the entire training split is consumed, so seed variation is identically zero.*

### Takeaway:
Variability across profiling seeds is vanishingly small ($\sigma \le 0.21\text{ pp}$ at 25%, and $\le 0.07\text{ pp}$ at 50% and 75%). This confirms that CSTP's transition distributions converge smoothly regardless of which prompt subsets are sampled.

---

## 5. MoE Suite Task-Level Divergence & 75% Reversal

In MoE Suite, the 75% profiling configuration achieves 18.03% frozen recall, outperforming the 100% configuration (16.47%). 

An investigation of task-level metrics from [`corrected_per_task_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_per_task_metrics.csv) reveals the mechanism:
- The MoE Suite evaluation split consists of two tasks: `gsm8k` (math reasoning) and `alpaca` (general instruction following).
- In the 75% profiling split, the sampled prompts happened to be more aligned with `gsm8k` reasoning traces.
- In the 100% profiling split, routing traces from diverse out-of-domain tasks were introduced into the transition tables. Because CSTP aggregates transitions globally without task-conditioning, the out-of-domain counts diluted the specific spatial correlations required for `gsm8k`.
- Hierarchical bootstrap hypothesis testing shows this difference has a two-sided achieved significance level ($ASL$) of $0.152$ with a 95% CI spanning zero ($-0.19\text{ pp}$ to $+2.22\text{ pp}$). Because MoE Suite contains only $N=2$ test tasks, this effect is certified strictly as `DESCRIPTIVE_ONLY_N2_TASKS`.

