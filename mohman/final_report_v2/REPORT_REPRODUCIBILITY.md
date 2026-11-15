# Master Experiment Reproduction and Verification Guide

## Project: MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference
## Course: CSE 511/ECE 511: Computer Architecture (End Semester Project Report)
## Authors: Mohit Gupta (2024354), Mrityunjai Poddar (2024358), Saksham Madan (2024493)
## Repository: https://github.com/Mohit14032007/ca_project_endsem

---

## 1. System Requirements and Environment Setup

### 1.1 Hardware Specifications
The simulations and trace evaluations were performed on a standardized Linux x86_64 host system:
- **Processor:** AMD EPYC / Intel Xeon 8+ cores (multi-threaded simulation acceleration)
- **Host Memory:** Minimum 32 GiB DDR4/DDR5 system memory
- **Storage:** 50 GiB free SSD storage (required for high-volume MoE parquet routing traces)
- **Target Simulated Accelerators:** NVIDIA A100-SXM4-40GB / H100-SXM5-80GB architectural profiles

### 1.2 Software Dependencies and Toolchain
Install the required system toolchain and libraries via Ubuntu package manager:
```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    cmake \
    g++ \
    git \
    libyaml-cpp-dev \
    libomp-dev \
    python3 \
    python3-pip \
    python3-dev
```

Install Python telemetry and evaluation packages:
```bash
pip3 install --user \
    numpy \
    scipy \
    pandas \
    pyarrow \
    pyyaml \
    matplotlib \
    seaborn \
    reportlab \
    python-docx
```

---

## 2. LLMSimulator Build and Verification Instructions

The core cycle-approximate memory simulator is implemented in C++17 under `LLMSimulator/`.

### 2.1 Compiling the Simulator
Execute the following build sequence from the repository root:
```bash
cd LLMSimulator
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```
Upon successful compilation, the primary simulator binary `llm_simulator` is generated in `LLMSimulator/build/`.

### 2.2 Running Unit and Component Tests
Verify correct functionality of predictor hooks, candidate queues, and memory controllers:
```bash
cd LLMSimulator/build
ctest --output-on-failure
# Or run direct component verification binaries:
./test_predictor
./test_prefetch_queue
./test_falru_cache
```

---

## 3. End-to-End Experiment Execution Recipes

### 3.1 Point 1 — Training-Free Expert Prediction Sweeps
To evaluate offline CSTP recall, hit rate, and ST-MoE baseline across profiling percentages:
```bash
# CSTP profiling sweep (10% to 50% frozen recall)
python3 mohman/point_1/cstp/scripts/run_cstp_profiling_sweep.py \
    --trace_dir LLMSimulator/data/downloads/moe_anywhere \
    --output_csv mohman/point_1/cstp/results/cstp_profiling_frozen_sweep.csv

# ST-MoE cold vs warm recall evaluation
python3 mohman/point_1/st_moe/scripts/run_st_moe_eval.py \
    --trace LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet \
    --output_dir mohman/point_1/st_moe/results/

# Generate Point 1 Figures (Fig 3.1 to 3.5)
python3 mohman/point_1/cstp/scripts/plot_frozen_recall.py
python3 mohman/point_1/cstp/scripts/plot_profiling_hit_rate.py
python3 mohman/point_1/st_moe/scripts/plot_cold_warm.py
python3 mohman/point_1/st_moe/scripts/plot_threshold_tradeoff.py
```

### 3.2 Point 2 & 3 — Asynchronous Prefetching Effectiveness Evaluation
To execute end-to-end trace-driven simulation across Workloads W1 through W7:
```bash
# Run Workload W3 (Anchor configuration: B=14, T=128, H=1, K=4)
./LLMSimulator/build/llm_simulator \
    --config LLMSimulator/reproduction/configs/duplex.yaml \
    --workload W3 \
    --prefetch_enabled true \
    --lookahead 1 \
    --budget 4 \
    --cache_policy lru \
    --output_dir mohman/point_3/final_results/raw/w3/

# Batch execution of all workloads W1 through W7
bash mohman/point_3/scripts/run_all_workloads.sh

# Extract telemetry and generate Point 3 Figures (Fig 5.1 to 5.14)
python3 mohman/point_3/scripts/extract_metrics.py
python3 mohman/point_3/scripts/plot_01_time_vs_stall.py
python3 mohman/point_3/scripts/plot_02_coverage.py
python3 mohman/point_3/scripts/plot_04_unused.py
python3 mohman/point_3/scripts/plot_06_traffic.py
python3 mohman/point_3/scripts/plot_07_stall.py
python3 mohman/point_3/scripts/plot_08_load_latency.py
python3 mohman/point_3/scripts/plot_09_decode_latency.py
```

### 3.3 Point 4 — FA-LRU Cache Eviction Scaling
To execute the cache management experiments comparing standard LRU against FA-LRU:
```bash
# Run FA-LRU vs LRU under Workload W3 and W4
./LLMSimulator/build/llm_simulator \
    --config LLMSimulator/reproduction/configs/duplex.yaml \
    --cache_policy falru \
    --hbm_capacity_gb 40 \
    --output_dir mohman/point_4/big_exp/results/falru/

# Generate Point 4 Scaling Figure (Fig 6.1)
python3 mohman/point_4/scripts/plot_falru_scaling.py
```

### 3.4 Point 5 — Four-Way Comparative Architecture Matrix
To evaluate Configurations A, B, C, and D under strictly aligned parameters:
```bash
# Config A: Baseline Demand-Only Offloading
./LLMSimulator/build/llm_simulator --config configs/config_a_baseline.yaml
# Config B: Shadow Prediction (No Prefetch)
./LLMSimulator/build/llm_simulator --config configs/config_b_shadow.yaml
# Config C: Asynchronous Prefetch + LRU
./LLMSimulator/build/llm_simulator --config configs/config_c_prefetch_lru.yaml
# Config D: Asynchronous Prefetch + FA-LRU
./LLMSimulator/build/llm_simulator --config configs/config_d_prefetch_falru.yaml

# Generate Point 5 Comparative Figures (Fig 7.1 to 7.3)
python3 mohman/point_5/scripts/plot_incremental_contribution.py
python3 mohman/point_5/scripts/plot_exposed_stall_comparison.py
python3 mohman/point_5/scripts/plot_traffic_composition.py
```

### 3.5 Point 6 — Multi-Dimensional Sensitivity Sweeps
To execute batch size, context length, lookahead horizon, and prefetch budget sweeps:
```bash
# Run batch size sweep (B = 1, 2, 4, 8, 14, 16, 32)
bash mohman/point_6/scripts/run_batch_sweep.sh
# Run context length sweep (P = 13, 64, 128, 512, 1024, 2048, 4096)
bash mohman/point_6/scripts/run_context_sweep.sh
# Run lookahead sweep (H = 1, 2, 3, 4)
bash mohman/point_6/scripts/run_lookahead_sweep.sh
# Run budget sweep (K = 1, 2, 3, 4, 6, 8)
bash mohman/point_6/scripts/run_budget_sweep.sh

# Generate Point 6 Sensitivity Figures (Fig 8.1 to 8.6)
python3 mohman/point_6/scripts/plot_batch_size.py
python3 mohman/point_6/scripts/plot_context_length.py
python3 mohman/point_6/scripts/plot_lookahead.py
python3 mohman/point_6/scripts/plot_prefetch_budget.py
```

### 3.6 Point 7 — Hardware Energy Model Evaluation
To compute subsystem energy, total energy, and EDP across the 28-run matrix:
```bash
# Execute energy model accounting
python3 mohman/point_7/scripts/compute_energy_model.py \
    --input_matrix mohman/point_6/final_results/csv/consolidated_matrix.csv \
    --output_matrix mohman/point_7/tables/csv/reconciled_energy_matrix.csv

# Generate Point 7 Figures (Fig 9.1 to 9.3)
python3 mohman/point_7/scripts/plot_energy_breakdown.py
python3 mohman/point_7/scripts/plot_pareto_frontier.py
python3 mohman/point_7/scripts/plot_edp.py
```

---

## 4. Telemetry Output Interpretation

Every simulation produces a standardized JSON / CSV summary containing:
- `wall_clock_time_sec`: Total simulated execution time for sequence decode.
- `exposed_memory_stall_sec`: Accumulation of GPU compute idle cycles waiting on expert weights.
- `ddr5_demand_bytes`: Off-chip data transferred due to compulsory cache misses.
- `ddr5_prefetch_bytes`: Speculative off-chip data transferred by the prefetch engine.
- `hbm_cache_hits`: Number of expert activations served directly from on-chip HBM.
- `timely_prefetches`: Number of speculative transfers completed prior to layer GEMM demand.
- `late_prefetches`: Number of speculative transfers attached in-flight.
- `unused_prefetches`: Speculative transfers completed but evicted prior to any demand.
- `total_system_energy_joules`: Granular sum of static DRAM, DDR5 I/O, HBM I/O, and MAC compute.
- `energy_delay_product_j_sec`: Product of total energy (J) and execution time (s).
