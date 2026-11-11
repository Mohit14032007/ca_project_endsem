# Command Reference Manual
```bash
# 1. Unit Tests
cd LLMSimulator/build
./test_phase2a_predictors
./test_phase2f_multiplier
./test_pabi_quarantine
./test_pabi_pinning

# 2. Sensitivity Campaigns
python3 scripts/final_campaign/run_candidate_budget_sensitivity.py
python3 scripts/final_campaign/run_phase2f_sensitivity.py

# 3. CSV & Plot Generation
python3 scripts/final_campaign/generate_all_comprehensive_plots_and_csvs.py
```
