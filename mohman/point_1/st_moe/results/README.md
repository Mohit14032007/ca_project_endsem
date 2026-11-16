# ST-MoE Results Directory & CSV Data Dictionary

**Deliverable:** Point 1 — ST-MoE Baseline Reproduction  
**Directory:** [`mohman/point_1/st_moe/results/`](file:///home/mohman/ca_project/mohman/point_1/st_moe/results/)  
**Primary Engine:** [`LLMSimulator/src/predictor/st_moe_paper.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/st_moe_paper.py)  
**Evaluator:** [`LLMSimulator/src/predictor/evaluation.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/evaluation.py)

---

## 1. Overview of Generated Datasets

All empirical results in this directory originate from isolated, reproducible evaluation runs across authoritative open-source MoE routing traces (`moe_anywhere` and `moe_suite`) as well as diagnostic synthetic/stress traces (`duplex_a100`, `correlated_synthetic`).

The CSV deliverables located in [`results/csv/`](file:///home/mohman/ca_project/mohman/point_1/st_moe/results/csv/) are structured as follows:

| Filename | Description | Row Count | Primary Purpose |
| :--- | :--- | :---: | :--- |
| [`online_policy_comparison.csv`](file:///home/mohman/ca_project/mohman/point_1/st_moe/results/csv/online_policy_comparison.csv) | Head-to-head comparison of `frozen`, `online_within_request`, and `online_across_requests` | 6 | Quantifies intra- vs cross-request adaptation gains |
| [`fixed_budget_summary.csv`](file:///home/mohman/ca_project/mohman/point_1/st_moe/results/csv/fixed_budget_summary.csv) | Fixed prediction budget sweep ($B \in \{1, 2, 4, 6, 8\}$) | 62 | Evaluates recall and precision scaling across budgets |
| [`native_threshold_results.csv`](file:///home/mohman/ca_project/mohman/point_1/st_moe/results/csv/native_threshold_results.csv) | Paper threshold evaluation ($\tau = 2$) across scoring interpretations | 24 | Analyzes set-size dilation and precision collapse |
| [`profiling_sensitivity.csv`](file:///home/mohman/ca_project/mohman/point_1/st_moe/results/csv/profiling_sensitivity.csv) | Profiling fraction sensitivity ($10\%$ to $100\%$) | 48 | Determines sample efficiency of offline CCT construction |
| [`per_experiment_results.csv`](file:///home/mohman/ca_project/mohman/point_1/st_moe/results/csv/per_experiment_results.csv) | Master catalog of every individual experimental configuration | 86 | Complete experimental index for audit and meta-analysis |
| [`per_opportunity_results.csv`](file:///home/mohman/ca_project/mohman/point_1/st_moe/results/csv/per_opportunity_results.csv) | Sample granular per-opportunity trace format | 5 | Documents schema for fine-grained prediction logs |

---

## 2. CSV Schemas and Field Definitions

### 2.1 `online_policy_comparison.csv`
- `dataset_id`: Identifier of trace (`moe_anywhere`, `moe_suite`, `duplex_a100`, `correlated_synthetic`).
- `classification`: Trace tier (`authoritative` vs `diagnostic`).
- `horizon`: Lookahead distance $H \in \{1, 2\}$ stages.
- `top_k`: Model top-$k$ routing degree ($K=8$ for authoritative models, $K=2$ for Duplex).
- `test_requests`: Number of test requests/sequences evaluated.
- `test_opportunities`: Total causal prediction events evaluated ($16,800$ for `moe_anywhere`, $7,500$ for `moe_suite`).
- `frozen_recall` / `frozen_precision` / `frozen_hit_rate`: Performance when CCT is static.
- `within_request_recall` / `within_request_precision` / `within_request_hit_rate`: Performance under `online_within_request` (profiled CCT restored at request boundaries).
- `within_request_cold_recall` / `within_request_steady_recall`: Split metrics (first 5 tokens vs subsequent tokens).
- `across_requests_recall` / `across_requests_precision` / `across_requests_hit_rate`: Performance under `online_across_requests` (CCT updates persist).
- `intra_request_delta_pp`: Gain in percentage points from intra-request adaptation ($R_{\text{within}} - R_{\text{frozen}}$).
- `cross_request_delta_pp`: Additional gain from cross-request persistence ($R_{\text{across}} - R_{\text{within}}$).
- `total_online_delta_pp`: Total online improvement ($R_{\text{across}} - R_{\text{frozen}}$).

### 2.2 `fixed_budget_summary.csv`
- `budget`: Explicit prediction set size constraint $B \in \{1, 2, 4, 6, 8\}$.
- `scoring_mode`: Ranking metric used (authoritative equation 1: `confidence_additive`).
- `recall`: Micro-averaged recall $\frac{\sum |P_i \cap G_i|}{\sum |G_i|}$.
- `precision`: Micro-averaged precision $\frac{\sum |P_i \cap G_i|}{\sum |P_i|}$.
- `hit_rate`: Fraction of opportunities with at least 1 correct expert ($\mathbb{I}(|P_i \cap G_i| \ge 1)$).
- `cold_start_recall`: Recall on the first 5 opportunities of each sequence.
- `steady_state_recall`: Recall on subsequent opportunities ($\text{step} \ge 5$).

### 2.3 `native_threshold_results.csv`
- `threshold`: Score cutoff $\tau$ (paper specifies $\tau = 2$).
- `scoring_mode`: 
  - `confidence_additive`: Mathematical literal of Eq. 1: $\text{Score} = \text{CCT} + \text{HT}$ (where $\text{CCT} \in \{0, 1, 2, 3\}$, $\text{HT} \in \{0, 1, 2, 3\}$).
  - `indicator_intersection`: Paper prose interpretation: $\mathbb{I}(\text{CCT} \ge 1) + \mathbb{I}(\text{HT} \ge 1) \ge 2$.
- `mean_predicted_set_size`: Average cardinality of the dynamically predicted expert set.
- `median_predicted_set_size`, `min_predicted_set_size`, `max_predicted_set_size`: Distribution metrics of predicted set sizes.

### 2.4 `profiling_sensitivity.csv`
- `profiling_fraction`: Fraction of sequence tokens used for offline CCT profiling ($0.1, 0.2, 0.4, 0.6, 0.8, 1.0$).
- `profiling_units`: Number of requests/sequences parsed for profiling.
- `profiling_tokens`: Total token transitions observed during offline calibration.
- `evaluation_tokens`: Token length of test evaluations.
- `fixed_budget_b_k_recall`: Fixed budget $B=K$ recall achieved.
- `threshold_mean_set_size`: Dynamic set size produced by native thresholding.

### 2.5 Note on `per_opportunity_results.csv` and Trace Volume
Authoritative testing on `moe_anywhere` and `moe_suite` evaluates **24,300 distinct opportunities** ($16,800 + 7,500$). Storing full uncompressed per-opportunity logs across all budgets and sweeps requires ~1.2 GB of tabular text. To maintain a lightweight and git-friendly repository structure, [`per_opportunity_results.csv`](file:///home/mohman/ca_project/mohman/point_1/st_moe/results/csv/per_opportunity_results.csv) provides a validated sample schema. Full opportunity traces can be re-emitted on demand by running [`LLMSimulator/scripts/phase2b/compare_online_policies.py`](file:///home/mohman/ca_project/LLMSimulator/scripts/phase2b/compare_online_policies.py) with the `--dump-per-opp` flag.

