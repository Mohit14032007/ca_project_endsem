# MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference
## Complete Experiment Reproduction Guide and Operational Manual

**Project Title:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Harness Executable:** `LLMSimulator/build/run_deepseekv3_mixtral_style`  
**Target Hardware:** NVIDIA A100-SXM4-80GB (Simulated) + Calibrated Off-Chip DDR5 Backing Memory (22.578 GB/s)  
**Target Architecture:** DeepSeek-V3 / Mixtral-Style (43 Layers: 3 Dense Prefix, 40 MoE Stages, Top-8 Routing, 256 Experts)  
**Destination Path:** `mohman/point2/COMPLETE_EXPERIMENT_REPRODUCTION_GUIDE.md`  
**Date:** November 2026  
**Status:** Certified Operational Single Source of Truth  

---

## Reproduction Philosophy & Invariant Rules

This guide is an operational reproduction manual designed to allow any systems or computer architecture researcher to replicate every experimental result across Points 1 through 7 from scratch.

To guarantee scientific reproducibility:
1. **Zero Guesswork:** Every step specifies the exact working directory, command string, configuration file, input trace, and expected output location.
2. **Deterministic Replay:** Trace ingestion utilizes deterministic sequence replay, guaranteeing that identical seeds and configurations produce bit-exact simulation logs.
3. **Conservation Invariants:** All simulation runs must satisfy the five fundamental physical invariants:
   - Capacity Bound: Resident HBM memory $\le 35\text{ GiB}$ ($37,580,963,840\text{ bytes}$) at all timestamps.
   - Eviction Classification: $E_{\text{total}} = E_{\text{unconsumed}} + E_{\text{demand}} + E_{\text{PABI}}$.
   - Attribution Bound: $E_{\text{prefetch-induced}} \le E_{\text{demand}}$.
   - Traffic Conservation: $B_{\text{total}} = B_{\text{demand}} + B_{\text{speculative}}$.
   - Denominator Consistency: $D_{\text{tensor}} = 3 \times D_{\text{routed}}$.
4. **Certified Numerical Reconciliation:** Target metrics must match the authoritative reconciled datasets (`unified_energy_results.csv`, `unified_comparison.csv`, and Point 7 reconciled tables).

---

## Table of Contents

- [1. Environment Setup & System Prerequisites](#1-environment-setup--system-prerequisites)
  - [1.1 Operating System & Hardware Requirements](#11-operating-system--hardware-requirements)
  - [1.2 Compiler & Toolchain Dependencies](#12-compiler--toolchain-dependencies)
  - [1.3 Python Runtime & Package Environment](#13-python-runtime--package-environment)
  - [1.4 System Environment Variables](#14-system-environment-variables)
  - [1.5 Complete Repository Workspace Layout](#15-complete-repository-workspace-layout)
- [2. Dataset Acquisition, Schema & Preprocessing](#2-dataset-acquisition-schema--preprocessing)
  - [2.1 MoE-Anywhere Authentic Trace Acquisition](#21-moe-anywhere-authentic-trace-acquisition)
  - [2.2 Parquet Schema & Inferred Context Lengths](#22-parquet-schema--inferred-context-lengths)
  - [2.3 Extracting Production CSV Traces (W3 & W4)](#23-extracting-production-csv-traces-w3--w4)
  - [2.4 Trace Verification & Invariant Auditing](#24-trace-verification--invariant-auditing)
- [3. Building LLMSimulator](#3-building-llmsimulator)
  - [3.1 CMake Configuration & Compiler Flags](#31-cmake-configuration--compiler-flags)
  - [3.2 Compiling the Core Library and Harness Executables](#32-compiling-the-core-library-and-harness-executables)
  - [3.3 Build Verification & Fast Preflight Smoke Test](#33-build-verification--fast-preflight-smoke-test)
- [4. Point-by-Point Experiment Reproduction Recipes](#4-point-by-point-experiment-reproduction-recipes)
  - [4.1 Point 1 — Offline Predictor Profiling Sensitivity (192 Configurations)](#41-point-1--offline-predictor-profiling-sensitivity-192-configurations)
  - [4.2 Point 2 — Synthetic Multi-Sequence Diversity & Metric Validation](#42-point-2--synthetic-multi-sequence-diversity--metric-validation)
  - [4.3 Point 3 — Prefetching Effectiveness Evaluation (20-Run Matrix)](#43-point-3--prefetching-effectiveness-evaluation-20-run-matrix)
  - [4.4 Point 4 — FA-LRU Eviction Policy & Big Experiment Reproduction](#44-point-4--fa-lru-eviction-policy--big-experiment-reproduction)
  - [4.5 Point 5 — Four-Way Comparative Architecture Evaluation (A, B, C, D)](#45-point-5--four-way-comparative-architecture-evaluation-a-b-c-d)
  - [4.6 Point 6 — Multi-Dimensional Sensitivity Sweeps (B, P, H, K)](#46-point-6--multi-dimensional-sensitivity-sweeps-b-p-h-k)
  - [4.7 Point 7 — Physical & Analytical Energy Analysis (28 Runs, 12 Figures)](#47-point-7--physical--analytical-energy-analysis-28-runs-12-figures)
- [5. Master Configuration Reference Table](#5-master-configuration-reference-table)
- [6. Standalone Algorithm Reproduction & Unit Verification](#6-standalone-algorithm-reproduction--unit-verification)
  - [6.1 CSTP Predictor Standalone Verification](#61-cstp-predictor-standalone-verification)
  - [6.2 ST-MoE Predictor Standalone Verification](#62-st-moe-predictor-standalone-verification)
  - [6.3 FA-LRU Eviction Algorithm Standalone Verification](#63-fa-lru-eviction-algorithm-standalone-verification)
  - [6.4 DDR5 Asynchronous Transfer Engine Standalone Verification](#64-ddr5-asynchronous-transfer-engine-standalone-verification)
- [7. Energy Analysis Pipeline Reproduction](#7-energy-analysis-pipeline-reproduction)
  - [7.1 Executing the Forensic Energy Extraction Script](#71-executing-the-forensic-energy-extraction-script)
  - [7.2 Executing the Automated Invariant Validation Suite](#72-executing-the-automated-invariant-validation-suite)
  - [7.3 Regenerating All 12 Certified Publication Figures](#73-regenerating-all-12-certified-publication-figures)
- [8. Numerical Verification & Acceptable Tolerances](#8-numerical-verification--acceptable-tolerances)
- [9. End-to-End One-Command Reproduction Pipeline](#9-end-to-end-one-command-reproduction-pipeline)
- [10. Comprehensive Troubleshooting & Historical Debugging Manual](#10-comprehensive-troubleshooting--historical-debugging-manual)

---

# 1. Environment Setup & System Prerequisites

### 1.1 Operating System & Hardware Requirements
- **Operating System:** Ubuntu 24.04 LTS (Linux kernel `6.8.0-48-generic` x86_64 or compatible Linux distribution).
- **CPU Architecture:** x86_64 with AVX2 support (Intel Xeon or AMD EPYC recommended, minimum 8 physical cores).
- **System Memory (Host RAM):** Minimum 32 GB RAM (64 GB recommended for parsing large multi-sequence trace parquet files).
- **Storage:** Minimum 50 GB free disk space (to store raw simulation logs, step CSVs, and trace dumps).

### 1.2 Compiler & Toolchain Dependencies
Ensure the system toolchain meets the following verified versions:
- **C++ Compiler:** GCC 13.3.0 (`g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`) supporting C++17.
- **Build System:** CMake 3.28.3 or higher.
- **System Libraries:** `libyaml-cpp-dev` (YAML configuration parser), `libomp-dev` (OpenMP parallel runtime).

Installation command on Ubuntu:
```bash
sudo apt update && sudo apt install -y \
    build-essential \
    cmake \
    git \
    libyaml-cpp-dev \
    libomp-dev \
    python3 \
    python3-pip \
    python3-venv
```

### 1.3 Python Runtime & Package Environment
The analysis, trace extraction, and publication plotting pipeline requires Python 3.12.3 with the following dependencies:
```bash
python3 -m pip install --upgrade pip
python3 -m pip install \
    numpy>=1.26.0 \
    pandas>=2.2.0 \
    pyarrow>=15.0.0 \
    matplotlib>=3.8.0 \
    scipy>=1.12.0 \
    pyyaml>=6.0.1 \
    pytest>=8.0.0
```

Verify the Python installation:
```bash
python3 -c "import numpy, pandas, pyarrow, matplotlib, yaml, scipy; print('All Python dependencies verified successfully.')"
```

### 1.4 System Environment Variables
Configure the following shell environment variables in your active session or `.bashrc`:
```bash
export CA_PROJECT_DIR="/home/mohman/ca_project"
export LLM_SIMULATOR_DIR="${CA_PROJECT_DIR}/LLMSimulator"
export PATH="${LLM_SIMULATOR_DIR}/build:${PATH}"

# Simulator runtime defaults
export OFFCHIP_DRAM_CFG="${LLM_SIMULATOR_DIR}/reproduction_A100/with_off_chip_dram/configs/dram_config_DDR5_retry.yaml"
export FA_LRU_WINDOW_SIZE=64
```

### 1.5 Complete Repository Workspace Layout
Verify your directory structure matches the expected project layout:
```text
/home/mohman/ca_project/
├── LLMSimulator/                           # Core C++ Simulator Repository
│   ├── build/                              # CMake binary output directory
│   ├── eval/                               # Evaluation harnesses & tests
│   ├── src/                                # Engine source code
│   ├── data/downloads/moe_anywhere/        # Raw parquet traces
│   └── reproduction_A100/                  # Hardware configs & baselines
└── mohman/
    ├── point_1/                            # Phase 1A Profiling Campaign
    ├── point_2/                            # Trace & Metrics Fixes
    ├── point_3/                            # Prefetching Effectiveness Matrix
    ├── point_4/                            # FA-LRU & Big Experiment
    ├── point_5/                            # Comparative 4-Way Evaluation
    ├── point_6/                            # Multi-Dimensional Sensitivity Sweeps
    ├── point_7/                            # Energy Analysis & Publication Figures
    └── point2/                             # Master Documentation & Reproduction Manual
```

---

# 2. Dataset Acquisition, Schema & Preprocessing

### 2.1 MoE-Anywhere Authentic Trace Acquisition
The primary source of ground-truth routing decisions is the authentic DeepSeek-V3 / Sage-A1.5 execution trace:
- Target File: `LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet`
- Size: $\approx 16\text{ MB}$ compressed Parquet.

If downloading or relocating:
```bash
mkdir -p /home/mohman/ca_project/LLMSimulator/data/downloads/moe_anywhere
# Confirm file exists and is readable:
ls -lh /home/mohman/ca_project/LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet
```

### 2.2 Parquet Schema & Inferred Context Lengths
Inspect the schema and prompt structure using Python:
```python
import pyarrow.parquet as pq

table = pq.read_table('/home/mohman/ca_project/LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet')
print("Parquet Schema:")
print(table.schema)
print(f"Total Rows: {table.num_rows}") # Must be exactly 760,720
```

Schema Definitions:
- `prompt_id: int32`: Unique prompt index ($0 \le \text{prompt\_id} \le 73$, 74 prompts total).
- `task: string`: Task identifier (e.g. `'code'`).
- `token_pos: int32`: Absolute token position. The initial prompt context length is inferred as:
  $$P_{\text{inferred}} = \min(\text{token\_pos}) \quad \text{for each prompt (range: 13 to 64)}$$
- `layer: int16`: MoE layer index ($0 \le \text{layer} \le 39$, 40 MoE stages).
- `experts: list<int16>[8]`: Top-8 routed expert IDs in range $[0, 255]$.
- `weights: list<halffloat>[8]`: Softmax gating probabilities for the 8 active experts.

### 2.3 Extracting Production CSV Traces (W3 & W4)
To generate the production CSV traces replayed by `LLMSimulator`, run the following trace extraction script:

```bash
python3 - << 'PREPROCESS_PY'
import pyarrow.parquet as pq
import pandas as pd
import os

parquet_path = "/home/mohman/ca_project/LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet"
out_dir = "/home/mohman/ca_project/mohman/point_4/big_exp/traces"
os.makedirs(out_dir, exist_ok=True)

table = pq.read_table(parquet_path)
df = table.to_pandas()

def extract_workload_csv(df, num_seqs, num_tokens, out_csv):
    records = []
    for seq_id in range(num_seqs):
        prompt_id = seq_id % 74
        p_df = df[df["prompt_id"] == prompt_id].sort_values(by=["token_pos", "layer"])
        unique_tokens = p_df["token_pos"].unique()[:num_tokens]
        
        for tok_idx, tok_val in enumerate(unique_tokens):
            tok_df = p_df[p_df["token_pos"] == tok_val]
            for _, row in tok_df.iterrows():
                layer_id = int(row["layer"])
                experts = list(row["experts"])
                for rank, exp_id in enumerate(experts, start=1):
                    records.append({
                        "seq_id": seq_id,
                        "token_id": tok_idx,
                        "layer_id": layer_id,
                        "topk_rank": rank,
                        "expert_id": int(exp_id)
                    })
    res_df = pd.DataFrame(records)
    res_df.to_csv(out_csv, index=False)
    print(f"Generated {out_csv} with {len(res_df)} rows.")

# Generate Workload W3 (B=14, T=128) -> 14 * 128 * 40 * 8 = 573,440 rows
extract_workload_csv(df, 14, 128, os.path.join(out_dir, "w3_b14_t128.csv"))

# Generate Workload W4 (B=32, T=128) -> 32 * 128 * 40 * 8 = 1,310,720 rows
extract_workload_csv(df, 32, 128, os.path.join(out_dir, "w4_b32_t128.csv"))
PREPROCESS_PY
```

### 2.4 Trace Verification & Invariant Auditing
Verify that the generated traces satisfy the row count invariants:
```bash
wc -l /home/mohman/ca_project/mohman/point_4/big_exp/traces/w3_b14_t128.csv
# Expected: 573441 lines (573,440 records + 1 header)

wc -l /home/mohman/ca_project/mohman/point_4/big_exp/traces/w4_b32_t128.csv
# Expected: 1310721 lines (1,310,720 records + 1 header)
```

---

# 3. Building LLMSimulator

### 3.1 CMake Configuration & Compiler Flags
The simulator uses CMake to configure build targets and link external libraries (`yaml-cpp`, `OpenMP`).

Key compiler flags configured in `CMakeLists.txt`:
```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -fopenmp")
```

### 3.2 Compiling the Core Library and Harness Executables
Execute the complete compilation sequence:
```bash
cd /home/mohman/ca_project/LLMSimulator
mkdir -p build && cd build

# Configure CMake in Release mode
cmake -DCMAKE_BUILD_TYPE=Release ..

# Compile using all available CPU threads
make -j$(nproc)
```

Verify that the required executables are successfully produced in `LLMSimulator/build/`:
```bash
ls -lh \
    run_deepseekv3_mixtral_style \
    test_fa_lru \
    test_ddr5_transfer_engine \
    test_live_prediction \
    test_synthetic_routing_diversity \
    test_prefetch_metrics
```

### 3.3 Build Verification & Fast Preflight Smoke Test
Run the harness in preflight validation mode (`--validate-config`). This checks topology, memory geometries, and configuration validity in $0.05\text{ seconds}$ without launching full simulation:

```bash
cd /home/mohman/ca_project/LLMSimulator/build

./run_deepseekv3_mixtral_style \
    --validate-config \
    /home/mohman/ca_project/mohman/point_6/final_results/experiments/anchor/configs/anc_c4.yaml
```

Expected output:
```text
======================================================================
DS-6C-2: DeepSeek-V3 Mixtral-Style Experiment Harness
Executable: run_deepseekv3_mixtral_style
Config Path: .../anc_c4.yaml
Mode: CONFIGURATION VALIDATION ONLY (No simulation will be run)
======================================================================
[TOPOLOGY & GEOMETRY VALIDATION]
  Model: DeepSeek-V3 (43 layers: 3 dense, 40 MoE)
  Experts: 256 routed, top-8, 1 shared
  Expert Footprint: 88,080,384 bytes (~88.08 MB FP16)
  Total Expert Inventory: 14,592 experts = 1285269043200 bytes (~1.285 TB)
  DDR5 Backing Capacity: 3020967936000 bytes (~2.75 TB / 2.5 TiB across 10,485,760 rows)
  HBM Local Cache Budget: 37580963840 bytes (35 GB)
  Prefetching: ENABLED (mode=hardware_async, predictor=CSTP, budget=4, horizon=1)
  Eviction Policy: fa_lru (fa_lru_window_size=64)
[SUCCESS] ALL ARCHITECTURAL, TOPOLOGY, AND GEOMETRY INVARIANTS VALIDATED!
Configuration file is 100% syntactically and structurally sound.
======================================================================
```

---
# 4. Point-by-Point Experiment Reproduction Recipes

---

## 4.1 Point 1 — Offline Predictor Profiling Sensitivity (192 Configurations)

### Objective:
Reproduce the Phase 1A profiling percentage evaluation comparing CSTP against ST-MoE across 192 configurations.

### Prerequisites:
- MoE-Anywhere trace parquet file: `LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet`

### Reproduction Commands:
```bash
cd /home/mohman/ca_project/LLMSimulator

# 1. Execute the automated certified unit and regression test suite:
pytest tests/test_phase1a_certified_correction.py -v
```

Expected test output:
```text
tests/test_phase1a_certified_correction.py::test_profiling_percentage_monotonicity PASSED
tests/test_phase1a_certified_correction.py::test_online_adaptation_gain_attenuation PASSED
tests/test_phase1a_certified_correction.py::test_cstp_vs_stmoe_frozen_recall PASSED
tests/test_phase1a_certified_correction.py::test_precision_recall_equality PASSED
tests/test_phase1a_certified_correction.py::test_moe_suite_dilution_characterization PASSED
...
11 passed in 4.25s (100% verified)
```

To run the full 192-configuration evaluation script:
```bash
python3 src/predictor/evaluation.py \
    --trace-file data/downloads/moe_anywhere/e1-sage-a1.5.parquet \
    --output-dir ../results/phase1a_cstp_profiling/certified/ \
    --profiling-pct 25 50 75 100 \
    --horizons 1 2 \
    --seeds 42 123 999
```

Output locations:
- Extracted metrics CSV: `results/phase1a_cstp_profiling/certified/certified_metrics_summary.csv`
- Certification report: `mohman/point_1/cstp/reports/PHASE1A_CORRECTED_FINAL_REPORT.md`

---

## 4.2 Point 2 — Synthetic Multi-Sequence Diversity & Metric Validation

### Objective:
Verify that multi-sequence serving batches ($B \in [1, 32]$) receive diverse, isolated expert assignments and validate telemetry counters.

### Prerequisites:
- Compiled test executables in `LLMSimulator/build/`

### Reproduction Commands:
```bash
cd /home/mohman/ca_project/LLMSimulator/build

# 1. Run multi-sequence diversity regression suite:
./test_synthetic_routing_diversity

# 2. Run prefetch metrics collector unit tests:
./test_prefetch_metrics
```

Expected output confirms:
- Diversity ratio at $B=32$ reaches **$20.4\times$** ($163.2$ unique experts vs. $8.0$ under the buggy cache).
- Tests A through H report `ALL INVARIANTS SATISFIED [PASS]`.
- Output log: `mohman/point_2/SYNTHETIC_ROUTING_FIX_REPORT.md`

---

## 4.3 Point 3 — Prefetching Effectiveness Evaluation (20-Run Matrix)

### Objective:
Execute the 20-run matrix measuring stall reduction across batch sizes $B \in \{1, 2, 4, 8\}$ and horizons $H \in \{1, 2\}$.

### Prerequisites:
- MoE-Anywhere held-out test split traces.

### Reproduction Commands:
```bash
cd /home/mohman/ca_project/LLMSimulator/build

# Execute CSTP H=1 at Batch Size 8:
./run_deepseekv3_mixtral_style \
    /home/mohman/ca_project/mohman/point_3/configs/moe_anywhere_b8_cstp_h1.yaml

# Execute ST-MoE H=1 at Batch Size 8:
./run_deepseekv3_mixtral_style \
    /home/mohman/ca_project/mohman/point_3/configs/moe_anywhere_b8_stmoe_h1.yaml

# Execute Demand Baseline at Batch Size 8:
./run_deepseekv3_mixtral_style \
    /home/mohman/ca_project/mohman/point_3/configs/moe_anywhere_b8_baseline.yaml
```

Verify completion:
```bash
cat /home/mohman/ca_project/mohman/point_3/results/core/b8_cstp_h1/logs/completion_status.txt
# Expected: STATUS=SUCCESS, EXIT_CODE=0
```

Summary metrics are consolidated in:
`mohman/point_3/POINT_3_PREFETCHING_EFFECTIVENESS_REPORT.md`

---

## 4.4 Point 4 — FA-LRU Eviction Policy & Big Experiment Reproduction

### Objective:
Replicate the Big Experiment evaluating Standard LRU vs. FA-LRU ($W=64$) under Workloads W3 ($B=14$) and W4 ($B=32$) on a 35 GiB cache.

### Prerequisites:
- Extracted traces: `mohman/point_4/big_exp/traces/w3_b14_t128.csv` and `w4_b32_t128.csv`

### Reproduction Commands:
```bash
cd /home/mohman/ca_project/LLMSimulator/build

# 1. Run FA-LRU unit regression suite:
./test_fa_lru

# 2. Execute Workload W3 (B=14) Standard LRU:
./run_deepseekv3_mixtral_style \
    /home/mohman/ca_project/mohman/point_4/big_exp/configs/w3_c1_r1.yaml

# 3. Execute Workload W3 (B=14) FA-LRU W=64:
./run_deepseekv3_mixtral_style \
    /home/mohman/ca_project/mohman/point_4/big_exp/configs/w3_c4_r1.yaml

# 4. Execute Workload W4 (B=32) Standard LRU:
./run_deepseekv3_mixtral_style \
    /home/mohman/ca_project/mohman/point_4/big_exp/configs/w4_c1_r1.yaml

# 5. Execute Workload W4 (B=32) FA-LRU W=64:
./run_deepseekv3_mixtral_style \
    /home/mohman/ca_project/mohman/point_4/big_exp/configs/w4_c4_r1.yaml
```

### Invariant Validation Command:
Verify all 5 physical conservation invariants across all runs:
```bash
python3 /home/mohman/ca_project/mohman/point_4/big_exp/scripts/verify_invariants.py
```

Expected verification summary:
- Run `w4_c1_r1`: Capacity PASS | Eviction PASS | Traffic PASS | Denominator PASS | Status: VALID
- Run `w4_c4_r1`: Capacity PASS | Eviction PASS | Traffic PASS | Denominator PASS | Status: VALID
- Hits Advantage: FA-LRU achieves **+15 resident hits** ($601$ vs $586$).
- Reloads Saved: 15 reloads saved.
- DDR5 Traffic Saved: **$1,321,205,760\text{ bytes}$ ($\approx 1.32\text{ GB}$)**.

---

## 4.5 Point 5 — Four-Way Comparative Architecture Evaluation (A, B, C, D)

### Objective:
Reproduce the four-configuration comparative analysis isolating prediction, prefetching, and cache replacement on the standardized Anchor workload.

### Reproduction Commands:
```bash
cd /home/mohman/ca_project/LLMSimulator/build

# Config A: Baseline (Demand Only, LRU)
./run_deepseekv3_mixtral_style \
    /home/mohman/ca_project/mohman/point_6/final_results/experiments/anchor/configs/anc_c1.yaml

# Config C: CSTP Prefetching + LRU
./run_deepseekv3_mixtral_style \
    /home/mohman/ca_project/mohman/point_6/final_results/experiments/anchor/configs/anc_c3.yaml

# Config D: Complete Implementation (CSTP + Prefetch + FA-LRU W=64)
./run_deepseekv3_mixtral_style \
    /home/mohman/ca_project/mohman/point_6/final_results/experiments/anchor/configs/anc_c4.yaml
```

Consolidate and extract comparative tables:
```bash
python3 /home/mohman/ca_project/mohman/point_5/scripts/extract_point5_comparative.py
```
Output files:
- `mohman/point_5/extracted_data/unified_comparison.csv`
- `mohman/point_5/reports/COMPONENT_CONTRIBUTION_ANALYSIS.md`

---

## 4.6 Point 6 — Multi-Dimensional Sensitivity Sweeps (B, P, H, K)

### Objective:
Reproduce the 26-run sensitivity sweeps across batch size, context length, lookahead, and budget.

### Reproduction Commands:
```bash
cd /home/mohman/ca_project/LLMSimulator/build

# 1. Budget Sweep Sweet Spot (K=2 Prefetching):
./run_deepseekv3_mixtral_style \
    /home/mohman/ca_project/mohman/point_6/final_results/experiments/prefetch_budget/configs/k2_c4.yaml

# 2. Lookahead Sweep (H=4 Lookahead):
./run_deepseekv3_mixtral_style \
    /home/mohman/ca_project/mohman/point_6/final_results/experiments/lookahead/configs/h4_c4.yaml

# 3. Context Length Scaling (P=4096 Context):
./run_deepseekv3_mixtral_style \
    /home/mohman/ca_project/mohman/point_6/final_results/experiments/context_length/configs/p4096_c4.yaml
```

Regenerate all 15 sensitivity publication figures:
```bash
python3 /home/mohman/ca_project/mohman/point_6/final_results/figures/generate_point6_figures.py
```

Output figures stored under:
`mohman/point_6/final_results/figures/` (subfolders: `batch_size/`, `context_length/`, `lookahead/`, `prefetch_budget/`, `combined/`).

---

## 4.7 Point 7 — Physical & Analytical Energy Analysis (28 Runs, 12 Figures)

### Objective:
Extract step telemetry, calculate component-wise energy breakdowns, execute validation tests, and regenerate the 12 publication figures.

### Reproduction Commands:
```bash
cd /home/mohman/ca_project/mohman/point_7/final_audit

# Step 1: Run forensic extraction across all 28 simulation runs:
python3 extract_point7_energy.py

# Step 2: Run automated invariant validation test suite:
python3 validate_point7_energy.py

# Step 3: Regenerate all 12 publication-quality figures:
python3 generate_point7_figures.py
```

Expected validation output:
```text
=== POINT 7 FORENSIC ENERGY VALIDATION SUITE ===
[TEST 1] Invariant Check: Sum of Components == System Total Energy -> PASSED (Max Residual: 1.82e-12 J)
[TEST 2] Invariant Check: Energy per Token == Total / (B * T) -> PASSED (Max Residual: 0.00e+00 J/tok)
[TEST 3] Invariant Check: EDP == Total Energy * Total Time -> PASSED (Max Residual: 0.00e+00 J*s)
[TEST 4] Anchor Value Check: anc_c4 total_energy == 9416.657933 J -> PASSED
[TEST 5] Sweet Spot Check: K=2 EDP improves over Baseline by -1.715% -> PASSED
[TEST 6] K=8 EDP Check: K=8 improves over Baseline by -0.325% -> PASSED
[TEST 7] FA-LRU Energy Check: Saves 0.326 J on Anchor D -> PASSED
ALL 7 FORENSIC TESTS PASSED WITH 100% MATHEMATICAL PRECISION.
```

---
# 5. Master Configuration Reference Table

The following master reference table documents the exact configuration YAML paths, parameters, cache geometries, and raw simulation outputs for primary evaluation runs across the project:

| Run ID | Point / Phase | Configuration YAML Path | Batch ($B$) | Context ($P$) | Horizon ($H$) | Budget ($K$) | Cache Policy | Raw Output Directory |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `anc_c1` | Pt 5, 6, 7 | `point_6/final_results/experiments/anchor/configs/anc_c1.yaml` | 32 | 2048 | - | 0 | Standard LRU | `point_6/final_results/experiments/anchor/raw_results/anc_c1/csv/` |
| `anc_c3` | Pt 5, 6, 7 | `point_6/final_results/experiments/anchor/configs/anc_c3.yaml` | 32 | 2048 | 1 | 4 | Standard LRU | `point_6/final_results/experiments/anchor/raw_results/anc_c3/csv/` |
| `anc_c4` | Pt 5, 6, 7 | `point_6/final_results/experiments/anchor/configs/anc_c4.yaml` | 32 | 2048 | 1 | 4 | FA-LRU ($W=64$) | `point_6/final_results/experiments/anchor/raw_results/anc_c4/csv/` |
| `w3_c1_r1`| Point 4 | `point_4/big_exp/configs/w3_c1_r1.yaml` | 14 | 2048 | - | 0 | Standard LRU | `point_4/big_exp/results/w3_c1_r1/` |
| `w3_c4_r1`| Point 4 | `point_4/big_exp/configs/w3_c4_r1.yaml` | 14 | 2048 | 1 | 4 | FA-LRU ($W=64$) | `point_4/big_exp/results/w3_c4_r1/` |
| `w4_c1_r1`| Point 4 | `point_4/big_exp/configs/w4_c1_r1.yaml` | 32 | 2048 | - | 0 | Standard LRU | `point_4/big_exp/results/w4_c1_r1/` |
| `w4_c4_r1`| Point 4 | `point_4/big_exp/configs/w4_c4_r1.yaml` | 32 | 2048 | 1 | 4 | FA-LRU ($W=64$) | `point_4/big_exp/results/w4_c4_r1/` |
| `k2_c4`   | Pt 6, 7 | `point_6/final_results/experiments/prefetch_budget/configs/k2_c4.yaml` | 32 | 2048 | 1 | 2 | FA-LRU ($W=64$) | `point_6/final_results/experiments/prefetch_budget/raw_results/k2_c4/csv/` |
| `k8_c4`   | Pt 6, 7 | `point_6/final_results/experiments/prefetch_budget/configs/k8_c4.yaml` | 32 | 2048 | 1 | 8 | FA-LRU ($W=64$) | `point_6/final_results/experiments/prefetch_budget/raw_results/k8_c4/csv/` |
| `h2_c4`   | Pt 6, 7 | `point_6/final_results/experiments/lookahead/configs/h2_c4.yaml` | 32 | 2048 | 2 | 4 | FA-LRU ($W=64$) | `point_6/final_results/experiments/lookahead/raw_results/h2_c4/csv/` |
| `h4_c4`   | Pt 6, 7 | `point_6/final_results/experiments/lookahead/configs/h4_c4.yaml` | 32 | 2048 | 4 | 4 | FA-LRU ($W=64$) | `point_6/final_results/experiments/lookahead/raw_results/h4_c4/csv/` |
| `b4_c4`   | Pt 6, 7 | `point_6/final_results/experiments/batch_size/configs/b4_c4.yaml` | 4 | 2048 | 1 | 4 | FA-LRU ($W=64$) | `point_6/final_results/experiments/batch_size/raw_results/b4_c4/csv/` |
| `p128_c4` | Pt 6, 7 | `point_6/final_results/experiments/context_length/configs/p128_c4.yaml` | 32 | 128 | 1 | 4 | FA-LRU ($W=64$) | `point_6/final_results/experiments/context_length/raw_results/p128_c4/csv/` |
| `p4096_c4`| Pt 6, 7 | `point_6/final_results/experiments/context_length/configs/p4096_c4.yaml` | 32 | 4096 | 1 | 4 | FA-LRU ($W=64$) | `point_6/final_results/experiments/context_length/raw_results/p4096_c4/csv/` |

---

# 6. Standalone Algorithm Reproduction & Unit Verification

For unit-level algorithmic verification, standalone C++ test executables in `LLMSimulator/build/` can be executed independently:

### 6.1 CSTP Predictor Standalone Verification
Execute predictor accuracy and Dirichlet smoothing tests:
```bash
/home/mohman/ca_project/LLMSimulator/build/test_live_prediction
```
Verifies:
- Marginal prior computation and Laplace regularization.
- Spatial co-activation probability accumulation across source experts.
- Temporal transition scoring from previous token history.
- Ranking ordering and score normalization.

### 6.2 ST-MoE Predictor Standalone Verification
Execute baseline CCT saturating counter tests:
```bash
/home/mohman/ca_project/LLMSimulator/build/test_phase2a_predictors
```
Verifies:
- 2-bit counter clamping in range $[0, 3]$.
- CCT capacity limits per source expert.
- Confidence threshold filtering.

### 6.3 FA-LRU Eviction Algorithm Standalone Verification
Execute candidate-window and lazy decay regression tests:
```bash
/home/mohman/ca_project/LLMSimulator/build/test_fa_lru
```
Verifies:
- Bottom-$W$ candidate scanning ($W=16$ and $W=64$).
- Frequency materialization on primary demand vs. speculative insertion.
- Lazy epoch bit-shift decay.
- Fallback search beyond window when all candidates are pinned.
- Divergence tracking against standard LRU.

### 6.4 DDR5 Asynchronous Transfer Engine Standalone Verification
Execute discrete-event transfer scheduling tests:
```bash
/home/mohman/ca_project/LLMSimulator/build/test_ddr5_transfer_engine
```
Verifies:
- Priority 1 demand preemption of Priority 0 speculative transfers.
- Byte-level progress preservation across pause/resume cycles.
- Conservation of transfer bytes: $\text{transferred} + \text{remaining} = \text{total}$.
- Dynamic demand promotion and in-flight attachment.

---

# 7. Energy Analysis Pipeline Reproduction

### 7.1 Executing the Forensic Energy Extraction Script
The forensic extraction script reads raw step CSV files from all 28 simulation runs, accumulates genuine hardware event counters, evaluates the analytical DDR5 and DRAM models, and compiles the consolidated CSV datasets:

```bash
cd /home/mohman/ca_project/mohman/point_7/final_audit
python3 extract_point7_energy.py
```

Generated Output CSVs in `mohman/point_7/extracted_data/`:
1. `unified_energy_results.csv`: Master dataset containing all 28 runs and complete metric columns.
2. `table1_component_energy_joules.csv`: Component breakdown in Joules (HBM, MAC, DDR5, Static DRAM).
3. `table2_energy_metrics_summary.csv`: Summary metrics including Energy per Token and EDP.
4. `table3_normalized_energy_percentages.csv`: Percentage distribution across components.
5. `table4_pareto_frontier_edp.csv`: EDP Pareto frontier evaluation.
6. `table5_energy_savings_deltas.csv`: Absolute and percentage deltas relative to baseline.
7. `table6_falru_energy_impact.csv`: Isolated energy impact of FA-LRU vs. LRU.
8. `table7_energy_tradeoff_matrix.csv`: Memory stall reduction vs. speculative energy overhead.

### 7.2 Executing the Automated Invariant Validation Suite
Run the automated test harness to prove mathematical consistency across all extracted data:

```bash
cd /home/mohman/ca_project/mohman/point_7/final_audit
python3 validate_point7_energy.py
```

### 7.3 Regenerating All 12 Certified Publication Figures
Execute the publication-quality plotting generator:

```bash
cd /home/mohman/ca_project/mohman/point_7/final_audit
python3 generate_point7_figures.py
```
This script renders all 12 publication figures in both vector PDF (`mohman/point_7/figures/pdf/`) and high-resolution PNG (`mohman/point_7/figures/png/`) format:
- `fig1_anchor_system_energy.png`
- `fig2_anchor_energy_per_token.png`
- `fig3_prefetch_budget_ddr5_breakdown.png`
- `fig4_ddr5_traffic_vs_energy.png`
- `fig5_stall_reduction_vs_energy_overhead.png`
- `fig6_energy_delay_product.png`
- `fig7_batch_size_energy_scaling.png`
- `fig8_budget_sweet_spot.png`
- `fig9_lookahead_energy_dynamics.png`
- `fig10_context_length_energy_scaling.png`
- `fig11_energy_performance_pareto.png`
- `fig12_component_pie_breakdown.png`

---

# 8. Numerical Verification & Acceptable Tolerances

When replicating experimental results, verify that extracted values match certified reference targets within the following acceptable numerical tolerances:

| Metric Category | Target Reference (Anchor D) | Tolerance | Verification Script / Method |
| :--- | :---: | :---: | :--- |
| **Total System Energy ($E_{\text{total}}$)** | $9,416.657933\text{ J}$ | $\pm 0.001\text{ J}$ | `validate_point7_energy.py` (Test 4) |
| **GPU HBM Operational Energy** | $1,044.180968\text{ J}$ | $\pm 0.0001\text{ J}$ | Sum of step row `act + read + write` |
| **GPU Tensor Core MAC Energy** | $71.251433\text{ J}$ | $\pm 0.0001\text{ J}$ | Sum of step row `mac_energy` |
| **DDR5 Active Transfer Energy** | $4,867.731776\text{ J}$ | $\pm 0.001\text{ J}$ | $\text{DDR5 Bytes} \times 1.288 \times 10^{-9}$ |
| **Static DRAM Background Power** | $3,433.493756\text{ J}$ | $\pm 0.001\text{ J}$ | $12.0\text{ W} \times 765.419\text{ s}$ |
| **Energy per Token ($E_{\text{tok}}$)** | $73.567640\text{ J/token}$ | $\pm 0.001\text{ J/tok}$ | $E_{\text{total}} / (32 \times 128)$ |
| **Energy-Delay Product (EDP)** | $7.20769\times 10^6\text{ J}\cdot\text{s}$ | $\pm 100\text{ J}\cdot\text{s}$ | $E_{\text{total}} \times T_{\text{wallclock}}$ |
| **K=2 EDP Reduction Delta** | **$-1.715\%$** | $\pm 0.01\text{ pp}$ | Relative delta vs. Baseline A |
| **K=8 EDP Reduction Delta** | **$-0.325\%$** | $\pm 0.01\text{ pp}$ | Relative delta vs. Baseline A |
| **FA-LRU Resident Hit Delta** | **+22 hits** | Exact integer ($0$ tolerance) | Anchor D hits ($608$) vs. Anchor C hits ($586$) |
| **FA-LRU Traffic Delta** | **$-2,025,848,832\text{ bytes}$** | Exact integer ($0$ tolerance) | Anchor D bytes vs. Anchor C bytes |
| **Context Attention Compute Scaling** | **$6.73\times$** ($7.82\text{ J} \to 52.59\text{ J}$) | $\pm 0.05\times$ | Compute growth from P=128 to P=4096 |
| **Context Attention DRAM Scaling** | **$1.86\times$** ($27.30\text{ J} \to 50.68\text{ J}$) | $\pm 0.05\times$ | DRAM growth from P=128 to P=4096 |

---

# 9. End-to-End One-Command Reproduction Pipeline

To automate the complete reproduction workflow from compilation to figure generation, a researcher can execute the sequential commands below:

```bash
#!/usr/bin/env bash
# End-to-End Reproduction Pipeline
set -euo pipefail

echo "=== STEP 1: Verifying Environment & Compiling LLMSimulator ==="
cd /home/mohman/ca_project/LLMSimulator
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

echo "=== STEP 2: Running Unit & Regression Test Suites ==="
./test_synthetic_routing_diversity
./test_fa_lru
./test_ddr5_transfer_engine
./test_prefetch_metrics

echo "=== STEP 3: Executing Anchor Workload Replay ==="
./run_deepseekv3_mixtral_style \
    /home/mohman/ca_project/mohman/point_6/final_results/experiments/anchor/configs/anc_c4.yaml

echo "=== STEP 4: Running Energy Extraction and Validation ==="
cd /home/mohman/ca_project/mohman/point_7/final_audit
python3 extract_point7_energy.py
python3 validate_point7_energy.py

echo "=== STEP 5: Regenerating Publication Figures ==="
python3 generate_point7_figures.py
cd /home/mohman/ca_project/mohman/point_6/final_results/figures
python3 generate_point6_figures.py

echo "=========================================================="
echo "ALL EXPERIMENTAL DELIVERABLES REPRODUCED AND CERTIFIED!"
echo "=========================================================="
```

---

# 10. Comprehensive Troubleshooting & Historical Debugging Manual

This section documents real historical defects encountered during project development and their exact verified solutions:

### Issue 1: Synthetic Routing Collision Across Sequences in a Batch
- **Symptom:** In multi-sequence serving ($B > 1$), HBM cache memory usage failed to scale with batch size. Regardless of $B$, resident unique experts remained fixed at 8.
- **Root Cause:** In `src/scheduler/sequence.cpp`, the synthetic routing cache container was keyed only by `(token_id, cur_layer)`. Sequence 0 populated the cache, and all subsequent sequences received identical expert assignments.
- **Solution:** Expand the routing key to a 4-tuple: `using RouteKey = std::tuple<int, int, int, int>;` (`request_id, seq_id, token_id, cur_layer`). Regression tested in `test_synthetic_routing_diversity.cpp`.

### Issue 2: DDR5 Physical-to-Logical Row Index Overflow
- **Symptom:** Simulator crashed with `assert(max_row_gpu < ddr5_cfg.num_row)` during startup when model layers were configured to 43.
- **Root Cause:** DeepSeek-V3 contains $14,592$ total experts ($1.285\text{ TB}$). In early memory YAML configurations, the DDR5 geometry allocated fewer than $10,485,760$ rows, causing expert address translation to exceed physical channel capacity.
- **Solution:** Set `ddr5_32gb_x16_2ch` geometry in `LLMSimulator/src/dram/memory_config.h` with `num_row = 10,485,760`, providing $2.75\text{ TB}$ of backing capacity.

### Issue 3: Triple-Counting Expert Demand from SwiGLU Sister Projections
- **Symptom:** Total demanded expert counts in `summary_csv` were exactly $3\times$ higher than theoretical router activations ($D_{\text{tensor}} = 3 \times D_{\text{routed}}$ recorded as 3 separate loads).
- **Root Cause:** SwiGLU FFN consists of three linear matrices ($W_1, W_3, W_2$). Each projection independently queried `is_expert_resident()`, triggering 3 misses and 3 loads per expert.
- **Solution:** Added `is_primary_expert_weight` to `Tensor`. Mark $W_1$ as primary, $W_3$ and $W_2$ as sisters. Only $W_1$ triggers cache residency checks, fetching all sister weights in one atomic transaction.

### Issue 4: Denominator Hardcoding in Precision and Recall Evaluation
- **Symptom:** Early Point 1 evaluation scripts hardcoded `denominator = 8` for precision and recall calculations.
- **Root Cause:** Hardcoded denominators failed to adapt if candidate list size or actual routing count fluctuated.
- **Solution:** Implemented dynamic denominators: $\text{Precision} = TP / |P|$ and $\text{Recall} = TP / |A|$. Verified that $|P| = |A| = 8$ across all $2,332,800$ events, proving mathematical equivalence.

### Issue 5: Cold-Start DDR5 Bus Saturation and Speculative Prefetch Queueing
- **Symptom:** At Step 0, prefetching produced 0% timely hits, and speculative prefetches completed after target step execution concluded.
- **Root Cause:** During cold start, un-cached demand misses arrive at Priority 1, monopolizing the $22.578\text{ GB/s}$ DDR5 channel for hundreds of milliseconds. Priority 0 speculative prefetches are queued behind cold demands.
- **Solution:** Recognize cold-start bus saturation as a physical constraint. As the cache warms up (by Step 2–3), demand misses collapse, opening channel bandwidth for timely prefetching.

### Issue 6: Discrepancy Between Draft Report Text and Underlying Source CSVs
- **Symptom:** Point 7 draft reports contained a hand-written table where component energies summed to $9,428.11\text{ J}$, but the table total claimed $9,416.66\text{ J}$ (an 11.45 J discrepancy).
- **Root Cause:** A human transcription error in the report markdown mixed stale draft values with the true CSV total. The underlying CSV (`unified_energy_results.csv`) was always 100% exact ($9,416.657933\text{ J}$).
- **Solution:** Always derive report tables directly from authoritative extracted CSVs via automated extraction scripts (`extract_point7_energy.py`). Never edit numerical tables manually.

---
