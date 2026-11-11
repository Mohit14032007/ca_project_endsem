# MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference
### End-Semester Master Submission Package & Comprehensive Evaluation

**Repository URL:** [https://github.com/Mohit14032007/CA_Project.git](https://github.com/Mohit14032007/CA_Project.git)  
**Authoritative Git Anchor:** `d5bf61ff0c066c06f72173189f9924329f613a88`

---

## Directory Organization (Structured by Professor's Requirements)

- [`00_MASTER/`](00_MASTER/): Master inventory, git manifest, requirements completion matrix, and limitations.
- [`01_PREDICTION_TECHNIQUE/`](01_PREDICTION_TECHNIQUE/): ST-MoE and CSTP mathematical models, accuracy metrics, and latency.
- [`02_PREFETCHING_MECHANISM/`](02_PREFETCHING_MECHANISM/): Forward-lookahead prefetch engine, DDR5 engine, and queue architecture.
- [`03_PREFETCHING_EFFECTIVENESS/`](03_PREFETCHING_EFFECTIVENESS/): Conservation laws, timely hits, late attachments, and exposed stall reduction.
- [`04_HBM_MANAGEMENT/`](04_HBM_MANAGEMENT/): Standard LRU vs PABI quarantine and Belady-inspired pinning.
- [`05_COMPARATIVE_EVALUATION/`](05_COMPARATIVE_EVALUATION/): 9-configuration component ablation matrix on Workload A1.
- [`06_SENSITIVITY_ANALYSIS/`](06_SENSITIVITY_ANALYSIS/): Sensitivity across Batch Size, Context Length, Lookahead Horizon, Candidate Budget, and Compute Window.
- [`07_ENERGY_ANALYSIS/`](07_ENERGY_ANALYSIS/): Native DRAM/MAC energy counters and calibrated off-chip DDR5 transport energy.
- [`08_DOCUMENTATION_AND_REPRODUCIBILITY/`](08_DOCUMENTATION_AND_REPRODUCIBILITY/): Technical report PDF, presentation slides, team contributions, and ZIP package.
- [`09_BONUS_REAL_TRACE_EVALUATION/`](09_BONUS_REAL_TRACE_EVALUATION/): MoE-Suite Task 12 and MoE-Anywhere recorded routing trace evaluation.

---

## Quick Reproduction Command

```bash
# 1. Build the LLMSimulator
cd LLMSimulator && mkdir -p build && cd build && cmake .. && make -j$(nproc)

# 2. Run unit and integration validation suites
./test_phase2a_predictors
./test_phase2f_multiplier
./test_pabi_quarantine
./test_pabi_pinning

# 3. Execute Candidate Budget Sensitivity Matrix
python3 ../../scripts/final_campaign/run_candidate_budget_sensitivity.py

# 4. Generate All Plots and Final Deliverables
python3 ../../scripts/final_campaign/generate_all_submission_data_and_plots.py
```
