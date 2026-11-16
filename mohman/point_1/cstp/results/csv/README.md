# CSTP Certified Results: CSV Data Dictionary & Schemas

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Deliverable:** Point 1 — Standalone CSTP Results Package  
**Directory:** [`mohman/point_1/cstp/results/csv/`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/)  
**Baseline Git Anchor:** `d5bf61ff0c066c06f72173189f9924329f613a88`  

---

## 1. Catalog of Certified CSV Datasets

All datasets in this directory are final, certified outputs from Phase 1A. They have been mathematically verified to have zero data truncation, exact dynamic denominators, and multi-seed statistical certification.

| Filename | Row Count | File Size | Primary Purpose |
| :--- | :---: | :---: | :--- |
| [`corrected_aggregate_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_aggregate_metrics.csv) | 192 | 33 KB | Master table of all 192 experimental configurations (CSTP & ST-MoE, 2 workloads, 4 profiling percentages, 3 seeds, 2 horizons, 2 regimes). |
| [`corrected_st_moe_head_to_head.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_st_moe_head_to_head.csv) | 48 | 7.3 KB | Direct matched comparison between CSTP and ST-MoE under identical profiling percentages, seeds, horizons, and workloads. |
| [`corrected_profiling_count_statistics.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_profiling_count_statistics.csv) | 48 | 7.6 KB | Quantifies growth of total spatial count mass, temporal count mass, and non-zero transition entries across profiling percentages. |
| [`corrected_statistical_analysis.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_statistical_analysis.csv) | 130 | 35 KB | Multi-seed hierarchical bootstrap intervals (95% CI), bootstrap probabilities, and two-sided achieved significance levels (ASL). |
| [`corrected_per_task_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_per_task_metrics.csv) | 192 | 26 KB | Per-task metrics isolating `gsm8k` vs `alpaca` performance on MoE Suite and trace partitions on MoE Anywhere. |
| [`corrected_per_prompt_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_per_prompt_metrics.csv) | 1,344 | 184 KB | Granular prompt-level evaluation metrics preserving intra-request variation across all evaluated sequences. |
| [`corrected_per_event_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_per_event_metrics.csv) | 2,332,800 | 222 MB | Complete, untruncated streaming record of all 2,332,800 individual token-level causal prediction events. |

---

## 2. CSV Column Schemas and Operational Definitions

### 2.1 `corrected_aggregate_metrics.csv`
- `dataset_id`: Workload trace (`moe_anywhere`, `moe_suite`).
- `model`: Model name (`Sage-A1.5`, `GLM-5.2`).
- `horizon`: Lookahead horizon $H \in \{1, 2\}$.
- `seed`: Random seed for profiling subsampling (`42, 123, 999`).
- `profiling_pct`: Percentage of offline training data used for profiling (`25, 50, 75, 100`).
- `predictor`: Predictor under test (`CSTP`, `ST-MoE`).
- `regime`: Evaluation protocol (`frozen`, `online_within_request`).
- `budget`: Prediction budget $B = 8$.
- `num_opportunities`: Total causal prediction events in the evaluation split ($16,800$ for `moe_anywhere`, $7,500$ for `moe_suite`).
- `num_units`: Number of independent prompt units evaluated ($14$ for `moe_anywhere`, $100$ for `moe_suite`).
- `macro_recall` / `macro_precision` / `macro_f1`: Macro-averaged metrics computed by averaging across individual prompt units.
- `micro_recall` / `micro_precision` / `micro_f1`: Micro-averaged metrics computed globally over all prediction events:
  $$\text{micro\_recall} = \frac{\sum_{i=1}^N |P_i \cap G_i|}{\sum_{i=1}^N |G_i|}, \quad \text{micro\_precision} = \frac{\sum_{i=1}^N |P_i \cap G_i|}{\sum_{i=1}^N |P_i|}$$
- `micro_hit_rate`: Fraction of opportunities where at least one predicted expert is correct: $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(|P_i \cap G_i| \ge 1)$.
- `exact_set_accuracy`: Fraction of opportunities where all 8 predicted experts match ground truth: $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(|P_i \cap G_i| = 8)$.
- `macro_cold_start_recall` / `macro_warm_token_recall`: Split metrics evaluating cold start (tokens 0–4) vs warm generation (tokens 5+).

### 2.2 `corrected_st_moe_head_to_head.csv`
- `cstp_frozen_recall` / `stmoe_frozen_recall`: Micro recall@8 under static profiling.
- `delta_frozen_recall`: Matched difference $R_{\text{CSTP, frozen}} - R_{\text{ST-MoE, frozen}}$.
- `cstp_online_recall` / `stmoe_online_recall`: Micro recall@8 under `online_within_request`.
- `delta_online_recall`: Matched difference $R_{\text{CSTP, online}} - R_{\text{ST-MoE, online}}$.
- `cstp_online_gain` / `stmoe_online_gain`: Absolute gain in percentage points ($R_{\text{online}} - R_{\text{frozen}}$).
- `cstp_online_hit_rate` / `stmoe_online_hit_rate`: Hit rate under `online_within_request`.
- `comparison_validity`: Scientific validity indicator (`VALID_MATCHED_COMPARISON` or `DESCRIPTIVE_ONLY_N2_TASKS`).

### 2.3 `corrected_statistical_analysis.csv`
- `comparison`: Name of the hypothesis contrast (e.g., `CSTP_Online_minus_CSTP_Frozen`, `CSTP_Frozen_minus_STMoE_Frozen`).
- `point_estimate_diff`: Observed point estimate of the difference in recall.
- `ci_95_lower` / `ci_95_upper`: 95% confidence intervals derived from 1,000 multi-seed hierarchical bootstrap replicates.
- `bootstrap_prob_greater_than_zero`: Empirical fraction of bootstrap samples where the difference exceeds 0.
- `bootstrap_two_sided_asl`: Bootstrap achieved significance level: $2 \times \min(p, 1-p)$.
- `statistical_validity`: Certification flag (`VALID_N14_PROMPT_UNITS` for MoE Anywhere; `DESCRIPTIVE_ONLY_N2_TASKS` for MoE Suite).

