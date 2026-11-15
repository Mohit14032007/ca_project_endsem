# Comprehensive Experiment Reproduction Checklist: MoE Memory Wall

- **Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference
- **Repository:** `https://github.com/Mohit14032007/ca_project_endsem`
- **Working Directory:** `/home/mohman/ca_project`
- **Target Audience:** Course Evaluators, Technical Reviewers, and Reproducing Researchers

---

## 1. System Requirements & Build Verification Checklist

| Item | Requirement / Specification | Verified Value | Pass / Fail |
| :--- | :--- | :--- | :--- |
| **Operating System** | Linux x86_64 (Ubuntu 22.04 LTS / WSL2) | Linux 6.6.x x86_64 | **PASS** |
| **C++ Compiler** | GCC 11+ or Clang 14+ with C++17 support | `g++ (Ubuntu 11.4.0)` | **PASS** |
| **Build System** | CMake 3.20+ and GNU Make / Ninja | `cmake 3.22.1` | **PASS** |
| **Python Runtime** | Python 3.10+ with standard numerical stack | `Python 3.12.3` | **PASS** |
| **Python Packages** | `numpy`, `pandas`, `pyarrow`, `scipy`, `matplotlib` | Installed in environment | **PASS** |
| **Document Generation**| `python-docx`, `reportlab`, `markdown` | Installed via pip | **PASS** |

### Compilation Commands
```bash
cd /home/mohman/ca_project/LLMSimulator
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```
*Expected Artifacts:*
- `build/eval/test_deepseekv3_mixtral_style` (Main execution binary)
- `build/eval/test_live_prediction` (Predictor harness)
- `build/eval/test_ddr5_transfer_engine` (Transfer engine harness)
- `build/eval/test_fa_lru` (Cache eviction harness)
- `build/eval/test_phase4b_effectiveness_metrics` (Metrics unit suite)

---

## 2. Unit and Regression Verification Checklist

Execute the unit test suites to confirm that algorithmic components satisfy their mathematical and physical invariants before launching full simulations.

```bash
# 1. Verify DDR5 Transfer Engine serialization & conservation invariants
./build/eval/test_ddr5_transfer_engine

# 2. Verify Live Prediction Engine causal ordering & top-K candidate bounds
./build/eval/test_live_prediction

# 3. Verify FA-LRU eviction window selection & frequency protection
./build/eval/test_fa_lru

# 4. Verify Prefetch Candidate Queue priority & duplicate suppression
./build/eval/test_prefetch_framework

# 5. Verify Metrics Collector accounting & stall aggregation
./build/eval/test_phase4b_effectiveness_metrics
```
*Acceptance Criteria:* All 5 test suites must report `0 FAILURES`, and all conservation assertions must pass with tolerance $\epsilon < 10^{-6}$.

---

## 3. End-to-End Experiment Campaign Reproduction

### 3.1 Point 1 — Offline Profiling Sensitivity
- **Objective:** Evaluate CSTP vs ST-MoE accuracy across lookahead horizons ($H \in [1,4]$) and candidate budgets ($K \in [1,8]$).
- **Execution:**
  ```bash
  python3 scripts/package_point1_deliverables.py
  ```
- **Expected Artifact:** `mohman/point_1/cstp/results/csv/corrected_aggregate_metrics.csv`
- **Acceptance Criterion:** CSTP Top-8 Hit Rate at $H=1$ must achieve $86.72\% \pm 0.05\%$.

### 3.2 Point 2 & 3 — Prefetching Effectiveness Evaluation
- **Objective:** Measure timely prefetch rate, late attachments, and exposed stall reductions across Workloads W1–W7.
- **Execution:**
  ```bash
  python3 scripts/final_campaign/run_phase2_tier1.py
  ```
- **Expected Artifact:** `mohman/point_3/phase2/large_workload/results/`
- **Acceptance Criterion:** Timely hits must be non-zero, and zero-stall executions must occur when prefetch completes prior to demand.

### 3.3 Point 4 — HBM Eviction Policy (LRU vs FA-LRU)
- **Objective:** Demonstrate elimination of speculative cache pollution using Frequency-Aware LRU ($W=64$).
- **Execution:**
  ```bash
  # Run Workload W3 (Batch 14, T=128) Baseline LRU vs FA-LRU
  python3 scripts/final_campaign/run_candidate_budget_sensitivity.py --workload W3 --policy FALRU
  ```
- **Expected Artifact:** `mohman/point_4/big_exp/extracted_data/big_experiment_results.csv`
- **Acceptance Criterion:** Reload count reduction $\ge 22$ reloads on Workload W3 ($B=14$) and $\ge 15$ on Workload W4 ($B=32$).

### 3.4 Point 5 — Four-Way Comparative Architecture (A, B, C, D)
- **Objective:** Isolate contributions of Prediction (B), Prefetching (C), and FA-LRU (D) against Baseline (A).
- **Execution:**
  ```bash
  python3 scripts/final_campaign/run_phase1_ablation.py
  ```
- **Expected Artifact:** `mohman/point_5/data/point5_comparative_table.csv`
- **Acceptance Criteria:**
  - Config A Execution Time: $442.146\text{ s} \pm 0.1\text{ s}$
  - Config C Execution Time: $338.257\text{ s} \pm 0.1\text{ s}$ ($23.50\%$ speedup)
  - Config D Execution Time: $338.286\text{ s} \pm 0.1\text{ s}$ ($23.49\%$ speedup, $+22$ hit retention)

### 3.5 Point 6 — Multi-Dimensional Sensitivity Sweeps
- **Objective:** Quantify scaling across Batch Size ($B \in [1,32]$), Context Length ($P \in [13, 4096]$), Lookahead ($H \in [1,4]$), and Budget ($K \in [1,8]$).
- **Execution:**
  ```bash
  python3 scripts/final_campaign/run_phase2f_sensitivity.py
  ```
- **Expected Artifact:** `mohman/point_6/final_results/consolidated_data/consolidated_sensitivity_table.csv`
- **Acceptance Criteria:**
  - Optimal Sweet Spot occurs at $K=2$ ($23.83\%$ latency speedup).
  - Diminishing returns and bus saturation occur at $K \ge 4$ (speculative traffic increases from $185\text{ GB}$ to $738\text{ GB}$).

### 3.6 Point 7 — Hardware Energy & EDP Evaluation
- **Objective:** Calculate MAC, HBM, DDR5, and static energy across 28 simulation runs and identify EDP optimal operating point.
- **Execution:**
  ```bash
  python3 mohman/point_7/extracted_data/extract_point7_energy_data.py
  python3 mohman/point_7/extracted_data/validate_point7_energy_data.py
  python3 mohman/point_7/figures/generate_point7_figures.py
  ```
- **Expected Artifacts:**
  - `mohman/point_7/extracted_data/unified_energy_results.csv`
  - 12 publication vector figures in `mohman/point_7/figures/pdf/` and `png/`
- **Acceptance Criteria:**
  - Anchor Baseline Energy: $9,416.66\text{ J} \pm 0.05\text{ J}$
  - Anchor Config D Energy: $9,416.33\text{ J} \pm 0.05\text{ J}$
  - Sweet Spot $K=2$ Energy Savings: $-27.68\text{ J}$ ($-1.715\%$ EDP reduction vs Baseline).

---

## 4. One-Command Master Verification Script

To verify all data, tests, and figures simultaneously, execute:
```bash
python3 scripts/final_campaign/validate_all_campaign_invariants.py
```
This script audits all 28 simulation manifests, checks md5 checksums, validates metric conservation, and outputs a complete PASS/FAIL summary for evaluation viva.
