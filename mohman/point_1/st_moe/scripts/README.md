# ST-MoE Execution and Processing Scripts

**Directory:** [`mohman/point_1/st_moe/scripts/`](file:///home/mohman/ca_project/mohman/point_1/st_moe/scripts/)

---

## Script Descriptions

### 1. `generate_results.py`
- **Purpose:** Extracts, harmonizes, and standardizes all empirical results from Phase 2, Phase 2A, and Phase 2B experiment runs into official CSV deliverables in `results/csv/`.
- **Outputs:**
  - `online_policy_comparison.csv`
  - `fixed_budget_summary.csv`
  - `native_threshold_results.csv`
  - `profiling_sensitivity.csv`
  - `per_experiment_results.csv`
  - `per_opportunity_results.csv`
- **Usage:**
  ```bash
  python3 mohman/point_1/st_moe/scripts/generate_results.py
  ```

---

### 2. `generate_plots.py`
- **Purpose:** Generates publication-ready figures in both 300 DPI PNG and vector PDF formats.
- **Dependencies:** Matplotlib only (no Seaborn).
- **Outputs:**
  - `recall_comparison.png` / `.pdf`
  - `frozen_vs_online.png` / `.pdf`
  - `horizon_comparison.png` / `.pdf`
  - `hit_rate_comparison.png` / `.pdf`
  - `cold_vs_warm_recall.png` / `.pdf`
  - `native_threshold_tradeoff.png` / `.pdf`
  - `profiling_sensitivity.png` / `.pdf`
- **Usage:**
  ```bash
  python3 mohman/point_1/st_moe/scripts/generate_plots.py
  ```

---

## Core Simulator Scripts and Test Suites

The underlying prediction runs and evaluation semantics are powered by the simulator engines and regression suites located in `LLMSimulator/`:

| Script / Test | Purpose |
| :--- | :--- |
| [`LLMSimulator/src/predictor/st_moe_paper.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/st_moe_paper.py) | Core paper-faithful ST-MoE predictor implementation |
| [`LLMSimulator/src/predictor/evaluation.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/evaluation.py) | Causal evaluation harness supporting `frozen`, `online_within_request`, and `online_across_requests` |
| [`LLMSimulator/scripts/phase2a/run_stmoe_isolated.py`](file:///home/mohman/ca_project/LLMSimulator/scripts/phase2a/run_stmoe_isolated.py) | Isolated experiment runner enforcing fresh predictor instances per test condition |
| [`LLMSimulator/scripts/phase2b/compare_online_policies.py`](file:///home/mohman/ca_project/LLMSimulator/scripts/phase2b/compare_online_policies.py) | Automated comparative runner for online adaptation policies |
| [`LLMSimulator/tests/test_stmoe_isolation.py`](file:///home/mohman/ca_project/LLMSimulator/tests/test_stmoe_isolation.py) | Regression test verifying CCT isolation across budgets and runs (6/6 passing) |
| [`LLMSimulator/tests/test_evaluation_semantics.py`](file:///home/mohman/ca_project/LLMSimulator/tests/test_evaluation_semantics.py) | Regression test verifying policy resetting, None preservation, and API contracts (10/10 passing) |

