# Phase 4B: Experiment Reproducibility Manifest

**Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Deliverable Path:** [reproducibility_manifest.md](file:///home/mohman/ca_project/mohman/point_3/reproducibility_manifest.md)  
**Parent Report:** [PHASE4B_RESULTS_REPORT.md](file:///home/mohman/ca_project/mohman/point_3/PHASE4B_RESULTS_REPORT.md)  
**Status:** Certified Fully Reproducible Protocol  

---

## 1. Environment & Architectural Configuration

All Phase 4B experiments are strictly executed under the certified Ramulator-backed, event-driven DDR5 memory model with the following environment variables:

```bash
export USE_EVENT_DRIVEN_DRAM=true
export DDR5_BANDWIDTH_GBPS=64.0
```

### Key Hardware & Architecture Parameters

- **Accelerator Target:** NVIDIA A100 Tensor Core GPU (Ampere Architecture)
- **Local HBM Cache Budget:** 80 GB ($85,899,345,920$ bytes)
- **Host Backing Memory:** DDR5-4800 Channel ($64.0$ GB/s peak bandwidth)
- **Off-Chip DDR5 Footprint:** $2.75$ TB ($2.5$ TiB addressable space across $10,485,760$ rows)
- **Model Architecture:** DeepSeek-V3 Mixtral-Style ($5$ layers: $3$ dense, $2$ MoE layers)
- **Expert Sizing:** $256$ routed experts per MoE layer, Top-$8$ active experts per token, $1$ shared expert
- **Single Expert Weight Size:** $88,080,384$ bytes ($\approx 88.08$ MB FP16 across $W_1, W_3, W_2$)
- **Single Expert Transfer Duration:** $88,080,384 \text{ bytes} / (64.0 \times 10^9 \text{ B/s}) = 1,376,256$ ns ($\approx 1.376$ ms)

---

## 2. Compilation and Binary Targets

The simulation binaries and synthetic validation harnesses are built from [LLMSimulator/build](file:///home/mohman/ca_project/LLMSimulator/build):

```bash
cd /home/mohman/ca_project/LLMSimulator/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) run_deepseekv3_mixtral_style test_prefetch_metrics test_phase4b_effectiveness_metrics
```

### Target Binaries

1. **Simulation Runner:** [run_deepseekv3_mixtral_style](file:///home/mohman/ca_project/LLMSimulator/build/run_deepseekv3_mixtral_style)
2. **Regression Test Suite:** [test_prefetch_metrics](file:///home/mohman/ca_project/LLMSimulator/build/test_prefetch_metrics)
3. **Phase 4B Effectiveness Test Suite:** [test_phase4b_effectiveness_metrics](file:///home/mohman/ca_project/LLMSimulator/build/test_phase4b_effectiveness_metrics)

---

## 3. Step-by-Step Reproduction Pipeline

### Step 1: Run Synthetic Effectiveness Unit Tests (Cases A through J)

Execute the 10 unit test cases verifying timely hits, late attachments, unused prefetches, candidate admission, transfer cancellations, projection deduplication, in-flight accounting, stall hidden math, and token latency quantiles:

```bash
cd /home/mohman/ca_project/LLMSimulator/build
./test_phase4b_effectiveness_metrics
```
Expected Output: `Summary: 10 Passed, 0 Failed`

Execute the 15 regression test scenarios:
```bash
./test_prefetch_metrics
```
Expected Output: `Test Summary: 15 Passed, 0 Failed`

---

### Step 2: Execute Primary Core Comparison (5 Configurations)

Run the five core configurations evaluating Baseline (demand loading), ST-MoE ($H=1, 2$), and CSTP ($H=1, 2$) over MoE-Anywhere ($B=1, \text{iter}=4$):

```bash
cd /home/mohman/ca_project/mohman/point_3

# 1. Baseline Demand Loading
USE_EVENT_DRIVEN_DRAM=true DDR5_BANDWIDTH_GBPS=64.0 \
  /home/mohman/ca_project/LLMSimulator/build/run_deepseekv3_mixtral_style configs/core_baseline.yaml

# 2. ST-MoE (Horizon=1, Budget=4)
USE_EVENT_DRIVEN_DRAM=true DDR5_BANDWIDTH_GBPS=64.0 \
  /home/mohman/ca_project/LLMSimulator/build/run_deepseekv3_mixtral_style configs/core_stmoe_h1.yaml

# 3. ST-MoE (Horizon=2, Budget=4)
USE_EVENT_DRIVEN_DRAM=true DDR5_BANDWIDTH_GBPS=64.0 \
  /home/mohman/ca_project/LLMSimulator/build/run_deepseekv3_mixtral_style configs/core_stmoe_h2.yaml

# 4. CSTP (Horizon=1, Budget=4)
USE_EVENT_DRIVEN_DRAM=true DDR5_BANDWIDTH_GBPS=64.0 \
  /home/mohman/ca_project/LLMSimulator/build/run_deepseekv3_mixtral_style configs/core_cstp_h1.yaml

# 5. CSTP (Horizon=2, Budget=4)
USE_EVENT_DRIVEN_DRAM=true DDR5_BANDWIDTH_GBPS=64.0 \
  /home/mohman/ca_project/LLMSimulator/build/run_deepseekv3_mixtral_style configs/core_cstp_h2.yaml
```

Consolidate core metrics into CSV:
```bash
python3 scripts/aggregate_metrics.py
```
Output: [mohman/point_3/reports/consolidated_metrics.csv](file:///home/mohman/ca_project/mohman/point_3/reports/consolidated_metrics.csv)

---

### Step 3: Execute Sensitivity Campaign (31 Experiments)

Run the full sensitivity matrix covering Candidate Budget ($1, 2, 4, 8$), Batch Size ($1, 2, 4$), Cross-Dataset (MoE-Anywhere, MoE-Suite Task 12, MoE-Suite Task 13), Decode Length ($2, 4, 8, 16$), and Constrained Cache ($20, 30$ experts):

```bash
cd /home/mohman/ca_project/mohman/point_3
python3 scripts/run_sensitivity_campaign.py
```
Output: [mohman/point_3/reports/sensitivity_metrics.csv](file:///home/mohman/ca_project/mohman/point_3/reports/sensitivity_metrics.csv)

---

### Step 4: Generate Certified Figures

Generate the 9 publication-grade figures in PNG and PDF formats:

```bash
cd /home/mohman/ca_project/mohman/point_3
python3 scripts/plot_phase4b_figures.py
```
Outputs saved in [mohman/point_3/plots/](file:///home/mohman/ca_project/mohman/point_3/plots/):
- `fig1_core_coverage_and_hit_rate.{png,pdf}`
- `fig2_core_memory_traffic_and_overhead.{png,pdf}`
- `fig3_core_exposed_stall_and_token_latency.{png,pdf}`
- `fig4_expert_load_latency_distributions.{png,pdf}`
- `fig5_budget_sensitivity.{png,pdf}`
- `fig6_batch_size_scaling.{png,pdf}`
- `fig7_cross_dataset_comparison.{png,pdf}`
- `fig8_decode_length_evolution.{png,pdf}`
- `fig9_constrained_cache_behavior.{png,pdf}`

---

### Step 5: Update Experiment Manifest

Generate the catalog of all completed runs:

```bash
cd /home/mohman/ca_project/mohman/point_3
python3 scripts/generate_manifest.py
```
Output: [mohman/point_3/experiment_manifest.csv](file:///home/mohman/ca_project/mohman/point_3/experiment_manifest.csv)

---

## 4. Directory Structure Map

All Phase 4B artifacts reside strictly within `/home/mohman/ca_project/mohman/point_3`:

```
mohman/point_3/
├── configs/
│   ├── core_baseline.yaml
│   ├── core_cstp_h1.yaml
│   ├── core_cstp_h2.yaml
│   ├── core_stmoe_h1.yaml
│   ├── core_stmoe_h2.yaml
│   ├── smoke_baseline.yaml
│   ├── smoke_cstp_h1.yaml
│   ├── smoke_stmoe_h1.yaml
│   └── sensitivity/
│       ├── batch/ (batch1_baseline, batch1_cstp, batch2_*, batch4_*)
│       ├── budget/ (budget_1, budget_2, budget_4, budget_8)
│       ├── cache_size/ (cap20_*, cap30_*)
│       ├── cross_dataset/ (anywhere_*, suite12_*, suite13_*)
│       └── decode_length/ (iter2_*, iter4_*, iter8_*, iter16_*)
├── traces/
│   ├── moe_anywhere_b1_t128.csv
│   ├── moe_anywhere_b2_t128.csv
│   ├── moe_anywhere_b4_t128.csv
│   ├── moe_anywhere_smoke.csv
│   ├── moe_suite_task12_b1_t128.csv
│   └── moe_suite_task13_b1_t128.csv
├── results/
│   ├── core/ (baseline, cstp_h1, cstp_h2, stmoe_h1, stmoe_h2)
│   ├── smoke/ (baseline, cstp_h1, stmoe_h1)
│   └── sensitivity/ (batch, budget, cache_size, cross_dataset, decode_length)
├── reports/
│   ├── consolidated_metrics.csv
│   └── sensitivity_metrics.csv
├── plots/
│   ├── fig1_core_coverage_and_hit_rate.{png,pdf}
│   ├── fig2_core_memory_traffic_and_overhead.{png,pdf}
│   ├── fig3_core_exposed_stall_and_token_latency.{png,pdf}
│   ├── fig4_expert_load_latency_distributions.{png,pdf}
│   ├── fig5_budget_sensitivity.{png,pdf}
│   ├── fig6_batch_size_scaling.{png,pdf}
│   ├── fig7_cross_dataset_comparison.{png,pdf}
│   ├── fig8_decode_length_evolution.{png,pdf}
│   └── fig9_constrained_cache_behavior.{png,pdf}
├── scripts/
│   ├── aggregate_metrics.py
│   ├── generate_manifest.py
│   ├── plot_phase4b_figures.py
│   └── run_sensitivity_campaign.py
├── tests/
│   └── test_phase4b_effectiveness_metrics.cpp
├── experiment_manifest.csv
├── reproducibility_manifest.md
├── PHASE4B_EFFECTIVENESS_AUDIT.md
├── PHASE4B_INSTRUMENTATION_REPORT.md
└── PHASE4B_RESULTS_REPORT.md
```

