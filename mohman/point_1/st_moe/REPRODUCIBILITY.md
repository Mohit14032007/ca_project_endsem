# Reproduction and Verification Guide: ST-MoE Baseline

**Deliverable:** Point 1 — Paper-Faithful ST-MoE Baseline Reproduction  
**Directory:** [`mohman/point_1/st_moe/`](file:///home/mohman/ca_project/mohman/point_1/st_moe/)  
**Authoritative Baseline Git Anchor:** `d5bf61ff0c066c06f72173189f9924329f613a88`

---

## 1. Prerequisites and Environment Setup

### 1.1 Software Dependencies
Reproduction requires a standard Linux environment with Python 3.10+:
- `python >= 3.10`
- `numpy`
- `pyyaml`
- `matplotlib >= 3.8` (for plotting; pure matplotlib, no seaborn required)
- `pyarrow` / `pandas` (for reading parquet routing traces)

Verify your environment:
```bash
python3 -c "import numpy, yaml, matplotlib, pyarrow; print('Environment OK')"
```

---

## 2. Regression and Integrity Verification

Before executing full evaluation sweeps, run the regression suites to verify experimental isolation and evaluation semantics:

### 2.1 Verify Evaluation Semantics (Phase 2B Suite)
Verifies request resetting, causal ordering, `None` sentinel preservation, and API contracts:
```bash
cd /home/mohman/ca_project/LLMSimulator
PYTHONPATH=. python3 tests/test_evaluation_semantics.py
```
*Expected Output:* `ALL 10 PHASE 2B REGRESSION TESTS PASSED SUCCESSFULLY!`

### 2.2 Verify Experimental Isolation (Phase 2A Suite)
Verifies that independent experiments construct fresh predictor instances without CCT counter contamination:
```bash
cd /home/mohman/ca_project/LLMSimulator
PYTHONPATH=. python3 tests/test_stmoe_isolation.py
```
*Expected Output:* `ALL 6 ISOLATION TESTS PASSED SUCCESSFULLY!`

---

## 3. Re-Running Experimental Campaigns

### 3.1 Run Online Policy Comparison (Primary Benchmark)
Executes the isolated comparison between `frozen`, `online_within_request`, and `online_across_requests`:
```bash
cd /home/mohman/ca_project/LLMSimulator
python3 scripts/phase2b/compare_online_policies.py
```
*Output File:* `results/phase2b_evaluation_semantics/policy_comparison_results.csv`

### 3.2 Run Full Isolated Parameter Sweeps
Executes the full matrix of fixed budgets ($B \in \{1, 2, 4, 6, 8\}$), native thresholding ($\tau = 2$), dual scoring modes, and profiling sensitivity ($10\%$ to $100\%$):
```bash
cd /home/mohman/ca_project/LLMSimulator
python3 scripts/phase2a/run_stmoe_isolated.py
```
*Output Files:*
- `results/phase2a_stmoe_isolation/stmoe_fixed_budget_results.csv`
- `results/phase2a_stmoe_isolation/stmoe_native_threshold_results.csv`
- `results/phase2a_stmoe_isolation/stmoe_profiling_sensitivity.csv`

---

## 4. Generating Packaged Deliverables

### 4.1 Harmonize CSV Deliverables
Consolidates and formats raw campaign outputs into the standardized deliverable CSVs:
```bash
cd /home/mohman/ca_project
python3 mohman/point_1/st_moe/scripts/generate_results.py
```
*Generates:*
- `mohman/point_1/st_moe/results/csv/online_policy_comparison.csv`
- `mohman/point_1/st_moe/results/csv/fixed_budget_summary.csv`
- `mohman/point_1/st_moe/results/csv/native_threshold_results.csv`
- `mohman/point_1/st_moe/results/csv/profiling_sensitivity.csv`
- `mohman/point_1/st_moe/results/csv/per_experiment_results.csv`
- `mohman/point_1/st_moe/results/csv/per_opportunity_results.csv`

### 4.2 Generate Publication Plots (PNG & Vector PDF)
Renders all 7 publication-quality figures:
```bash
cd /home/mohman/ca_project
python3 mohman/point_1/st_moe/scripts/generate_plots.py
```
*Generates 7 PNGs and 7 PDFs in:* `mohman/point_1/st_moe/plots/`

---

## 5. Automated Verification Checklist

Run the following sanity check script to verify all deliverable invariants:

```bash
python3 -c '
import csv
from pathlib import Path

csv_dir = Path("/home/mohman/ca_project/mohman/point_1/st_moe/results/csv")
with open(csv_dir / "online_policy_comparison.csv") as f:
    r = {row["dataset_id"] + "_H" + row["horizon"]: row for row in csv.DictReader(f)}

# MoE Anywhere H=1 checks
ma1 = r["moe_anywhere_H1"]
assert abs(float(ma1["within_request_recall"]) - 0.4791) < 1e-4, f"Mismatch: {ma1}"
assert abs(float(ma1["within_request_hit_rate"]) - 0.9694) < 1e-4, f"Mismatch: {ma1}"
assert int(ma1["test_opportunities"]) == 16800, f"Mismatch: {ma1}"

# MoE Suite H=1 checks
ms1 = r["moe_suite_H1"]
assert abs(float(ms1["within_request_recall"]) - 0.2653) < 1e-4, f"Mismatch: {ms1}"
assert abs(float(ms1["within_request_hit_rate"]) - 0.8984) < 1e-4, f"Mismatch: {ms1}"
assert int(ms1["test_opportunities"]) == 7500, f"Mismatch: {ms1}"

print("ALL ST-MOE INVARIANTS AND METRICS VERIFIED PERFECTLY!")
'
```

