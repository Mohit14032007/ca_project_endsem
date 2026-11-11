# Cross-Workload Comparative Analysis: MoE Anywhere vs. MoE Suite

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Deliverable:** Point 1 — Comparative Study Analysis  
**Path:** [`mohman/point_1/stmoe_cstp/analysis/workload_comparison.md`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/analysis/workload_comparison.md)  
**Authoritative Data Sources:** [`results/csv/matched_comparisons.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/matched_comparisons.csv) & [`results/csv/per_task_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/per_task_metrics.csv)  

---

## 1. Structural Comparison of Workload Traces

| Structural Attribute | MoE Anywhere (Sage-A1.5) | MoE Suite (GLM-5.2) |
| :--- | :---: | :---: |
| **Pipeline Depth ($S$)** | 40 MoE stages | 75 MoE stages ($1.88\times$ deeper) |
| **Total Experts per Layer ($M$)** | 256 | 256 |
| **Routing Degree ($K$)** | 8 | 8 |
| **Evaluation Sequences ($U$)** | 14 prompts | 100 prompts |
| **Evaluation Tasks** | Uniform general traces | 2 distinct tasks (30 `gsm8k`, 70 `alpaca`) |
| **Total Test Opportunities** | 16,800 | 7,500 |
| **Baseline Predictability** | High (~45%–48% recall) | Moderate-Low (~16%–26% recall) |

---

## 2. Key Analytical Differences

### 2.1 Impact of Model Depth on Router Entropy
In GLM-5.2 (75 stages), tokens undergo nearly twice as many sequential routing decisions as in Sage-A1.5 (40 stages). Routing entropy accumulates along the depth of the transformer, leading to higher path branching and lower conditional co-activation probabilities. This fundamental architectural difference explains why both predictors experience an absolute recall drop of $\sim 20$–$30$ percentage points on MoE Suite compared to MoE Anywhere.

### 2.2 Task Heterogeneity and the 75% vs. 100% Profiling Reversal
On MoE Suite, CSTP's recall drops from 18.03% (75% profiling) down to 16.47% (100% profiling). Decomposing the test split in [`results/csv/per_task_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/per_task_metrics.csv):
- `gsm8k` recall drops from 19.82% (75%) down to 16.89% (100%).
- `alpaca` recall remains relatively constant (17.26% vs 16.29%).
- In MoE Suite's training split, the final 25% of prompts contain diverse tasks whose routing transitions diverge from the mathematical reasoning patterns in `gsm8k`. Aggregating these disparate traces into a single global count matrix creates cross-task interference.

### 2.3 Statistical Caveat ($N=2$ Tasks)
Because MoE Suite contains only two test tasks, prompt-level observations are highly clustered. In accordance with certified scientific guidelines, all MoE Suite inferential intervals are formally designated `DESCRIPTIVE_ONLY_N2_TASKS`.

