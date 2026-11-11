# Master Reproduction Guide
Complete step-by-step reproduction instructions covering dependencies, build steps, trace verification, experiment execution, and graph generation.

```bash
# 1. Setup build
cd LLMSimulator && mkdir -p build && cd build && cmake .. && make -j$(nproc)

# 2. Run unit tests
./test_phase2a_predictors
./test_phase2f_multiplier
./test_pabi_quarantine
./test_pabi_pinning

# 3. Execute Candidate Budget Matrix
python3 ../../scripts/final_campaign/run_candidate_budget_sensitivity.py

# 4. Generate all CSVs and Plots
python3 ../../scripts/final_campaign/generate_all_comprehensive_plots_and_csvs.py
```
