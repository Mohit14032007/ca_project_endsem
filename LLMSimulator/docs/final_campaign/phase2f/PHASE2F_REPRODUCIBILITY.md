# Phase 2F — Reproducibility and Verification Protocol

## 1. Overview and Authoritative Baseline Anchor

This document details the complete environment, build instructions, execution steps, and verification procedures to reproduce the **Phase 2F Synthetic Compute-Window Sensitivity Campaign**.

- **Authoritative Git Anchor:** `d5bf61ff0c066c06f72173189f9924329f613a88`
- **Repository Root:** `/home/mohman/ca_project`
- **Simulator Directory:** `/home/mohman/ca_project/LLMSimulator`
- **Calibrated DDR5 Bandwidth:** $22.578\text{ GB/s}$
- **Expert Size:** $88,080,384\text{ bytes}$ ($84.0\text{ MB}$)

---

## 2. Workload Trace Integrity

The campaign evaluates Workload A1 across 4 sequential autoregressive decoding steps:
- **Trace Path:** `/home/mohman/ca_project/LLMSimulator/results/endsem_phase_e/final_baseline/traces/ms_task12_t128_l75.csv`
- **Expected SHA-256 Checksum:** `5a635dece5747e78be7f600082c888a7f9e9d2ee50e1ae829d338f75f235ea1c`

To verify trace integrity:
```bash
sha256sum /home/mohman/ca_project/LLMSimulator/results/endsem_phase_e/final_baseline/traces/ms_task12_t128_l75.csv
```

---

## 3. Simulator Build Instructions

To build the simulator binary and unit test target from clean source:
```bash
cd /home/mohman/ca_project/LLMSimulator
cmake -B build
cmake --build build --target run_deepseekv3_mixtral_style test_phase2f_multiplier -j
```

### 3.1 Executing the Unit Test Suite
```bash
cd /home/mohman/ca_project/LLMSimulator
./build/test_phase2f_multiplier
```
Expected output:
```
============================================================
PHASE 2F UNIT & INTEGRATION TESTS: COMPUTE MULTIPLIER & CLOCK CONTINUITY
============================================================
[RUNNING] Test 1: Config & Environment Multiplier Parsing...
  [PASS] Test 1: Multiplier and clock continuity correctly parsed from environment.
[RUNNING] Test 2: Inter-Token Clock Continuity in restartModuleGraph...
  [PASS] Test 2: Monotonic time preserved across module graph restart when enabled.
[RUNNING] Test 3: Expanded Compute Window Allows DDR5 Transfer Completion...
  [PASS] Test 3: DDR5 transfer incomplete at M=1 window (366 us), successfully completes at M=12 window (4.396 ms)!

ALL PHASE 2F UNIT TESTS PASSED SUCCESSFULLY!
```

---

## 4. Executing the Full 15-Run Campaign

The complete 15-run experiment is orchestrated automatically by `scripts/final_campaign/run_phase2f_sensitivity.py`:

```bash
cd /home/mohman/ca_project
python3 scripts/final_campaign/run_phase2f_sensitivity.py
```

### Execution Flow:
1. Performs AST check verifying zero ternary operators.
2. Runs validation gate `B-M1` ($M=1.0$) and asserts bitwise match against `results/final_comparative_campaign/phase2e3_warm_diagnostic/B`:
   - $T_{\text{exec}} = 8003.70\text{ ms}$
   - $T_{\text{stall}} = 7884.24\text{ ms}$
   - Hits = 5179, Misses = 2021, Hit rate = 71.93%
3. Sequentially executes the remaining 14 runs across $M \in \{1, 4, 8, 12, 16\}$.
4. Outputs raw traces, completion statuses, and manifests to:
   `results/final_comparative_campaign/phase2f/<RUN_ID>/`
5. Generates the 5 required summary CSVs in `docs/final_campaign/phase2f/`.

---

## 5. Generating Publication Figures

To generate high-resolution PNG plots:
```bash
cd /home/mohman/ca_project
python3 scripts/final_campaign/generate_phase2f_plots.py
```
Generated figures:
1. `docs/final_campaign/phase2f/plots/PHASE2F_STALL_REDUCTION_VS_MULTIPLIER.png`
2. `docs/final_campaign/phase2f/plots/PHASE2F_TIMELY_HITS_AND_ATTACHMENTS.png`
3. `docs/final_campaign/phase2f/plots/PHASE2F_QUEUE_WAIT_DYNAMICS.png`
4. `docs/final_campaign/phase2f/plots/PHASE2F_PREDICTOR_QUALITY.png`
5. `docs/final_campaign/phase2f/plots/PHASE2F_ENERGY_BREAKDOWN.png`
