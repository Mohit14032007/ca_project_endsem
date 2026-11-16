# Master Reproducibility Guide & System Architecture Manual
## Group 13 — Project 2: MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference

**Course:** CSE 511 / ECE 511: Computer Architecture (Monsoon 2024 / Endsem 2026)  
**Project Title:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Group:** Group 13  
**Project Number:** Project 2  
**Authors:**
- Mohit Gupta (Roll: 2024354)
- Mrityunjai Poddar (Roll: 2024358)
- Saksham Madan (Roll: 2024493)

**Repository:** [https://github.com/Mohit14032007/ca_project_endsem](https://github.com/Mohit14032007/ca_project_endsem)  
**Default Branch:** `main`  
**Primary Documentation File:** `MASTER_GROUP13_PROJECT2_README.md`  
**Authoritative Final Report Deliverables:**
- Word Document: `mohman/submission_final/MoE_Memory_Wall_Final_Report.docx`
- Complete 56-Page PDF: `mohman/submission_final/MoE_Memory_Wall_Final_Report.pdf` (also at repository root as `Group13_Project2_Report.pdf`)
- Numerical Reconciliation Matrix: `mohman/submission_final/NUMERICAL_RECONCILIATION_MATRIX.csv` (120 audited claims)
- Consistency Validation Report: `mohman/submission_final/REPORT_CONSISTENCY_VALIDATION.md` (63/63 checks passed)
- Figure Source Data: `mohman/submission_final/FIGURE_DATA/` (32 figures: Figure 3.1 through Figure 9.3)
- Final Submission Archive: `mohman/submission_final/MoE_Memory_Wall_Final_Submission.zip`

---

## Master Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Objectives and Implemented Points](#2-project-objectives-and-implemented-points)
3. [Project Directory Structure and Result Organization](#3-project-directory-structure-and-result-organization)
   - [3.1 Repository Layout](#31-repository-layout)
   - [3.2 Point-Wise Result Mapping Table](#32-point-wise-result-mapping-table)
   - [3.3 Authoritative Result Files and Artifact Types](#33-authoritative-result-files-and-artifact-types)
4. [System Requirements and Hardware Requirements](#4-system-requirements-and-hardware-requirements)
5. [Fresh Laptop Setup — Step by Step](#5-fresh-laptop-setup--step-by-step)
   - [5.1 Repository Cloning and Verification](#51-repository-cloning-and-verification)
   - [5.2 Host Path Compatibility Configuration](#52-host-path-compatibility-configuration)
6. [Dependency Installation](#6-dependency-installation)
   - [6.1 Linux System Packages](#61-linux-system-packages)
   - [6.2 Python Virtual Environment and Packages](#62-python-virtual-environment-and-packages)
7. [Build Instructions](#7-build-instructions)
   - [7.1 Compiling LLMSimulator with CMake](#71-compiling-llmsimulator-with-cmake)
   - [7.2 Generated Binaries and Libraries](#72-generated-binaries-and-libraries)
   - [7.3 Preflight Build Verification](#73-preflight-build-verification)
8. [Dataset and Routing Trace Preparation](#8-dataset-and-routing-trace-preparation)
   - [8.1 Authentic MoE Traces (Sage-A1.5 and MoE-Suite)](#81-authentic-moe-traces-sage-a15-and-moe-suite)
   - [8.2 Trace Verification and Schema Invariants](#82-trace-verification-and-schema-invariants)
9. [Configuration Guide](#9-configuration-guide)
   - [9.1 Architectural Baseline Parameters](#91-architectural-baseline-parameters)
   - [9.2 Configuration Files by Experiment Point](#92-configuration-files-by-experiment-point)
10. [Point-Wise Implementation and Reproduction Commands](#10-point-wise-implementation-and-reproduction-commands)
    - [Point 1: Expert Selection Prediction & Profiling Sensitivity](#point-1-expert-selection-prediction--profiling-sensitivity)
    - [Point 2: Prefetching Mechanism, Latency Hiding & Trace Audit](#point-2-prefetching-mechanism-latency-hiding--trace-audit)
    - [Point 3: Prefetching Effectiveness Evaluation Across Large Workloads (W1–W7)](#point-3-prefetching-effectiveness-evaluation-across-large-workloads-w1w7)
    - [Point 4: HBM Management & Eviction Policy Extension (FA-LRU)](#point-4-hbm-management--eviction-policy-extension-fa-lru)
    - [Point 5: Four-Way Architectural Comparative Evaluation (Configs A, B, C, D)](#point-5-four-way-architectural-comparative-evaluation-configs-a-b-c-d)
    - [Point 6: Multi-Dimensional Sensitivity Analysis (B, P, H, K Sweeps)](#point-6-multi-dimensional-sensitivity-analysis-b-p-h-k-sweeps)
    - [Point 7: Subsystem Energy Modeling and EDP Optimization](#point-7-subsystem-energy-modeling-and-edp-optimization)
11. [ST-MoE Reproduction (Literature Baseline)](#11-st-moe-reproduction-literature-baseline)
12. [CSTP Reproduction (Proposed Spatio-Temporal Predictor)](#12-cstp-reproduction-proposed-spatio-temporal-predictor)
13. [Prefetching Experiments and Timing Dynamics](#13-prefetching-experiments-and-timing-dynamics)
14. [HBM Management Experiments (FA-LRU vs. LRU)](#14-hbm-management-experiments-fa-lru-vs-lru)
15. [Runtime, Energy and EDP Experiments](#15-runtime-energy-and-edp-experiments)
16. [Complete Experiment Execution Sequence](#16-complete-experiment-execution-sequence)
    - [16.1 Fast Verification Suite (< 1 Minute)](#161-fast-verification-suite--1-minute)
    - [16.2 Full Simulation Reproduction Sequence](#162-full-simulation-reproduction-sequence)
17. [Result Processing and Figure Generation](#17-result-processing-and-figure-generation)
18. [Mapping of Results to the Final Report](#18-mapping-of-results-to-the-final-report)
19. [Reproducibility Verification Checklist](#19-reproducibility-verification-checklist)
20. [Troubleshooting Guide](#20-troubleshooting-guide)
21. [Expected Output Directory Structure](#21-expected-output-directory-structure)
22. [Important Notes, Limitations and Scientific Boundaries](#22-important-notes-limitations-and-scientific-boundaries)

---

# 1. Project Overview

Modern Large Language Models (LLMs) increasingly adopt Sparse Mixture-of-Experts (MoE) architectures (such as Mixtral 8x7B, DeepSeek-V2, and DeepSeek-V3) to scale parameter capacity to hundreds of billions of weights while activating only a sparse subset of feed-forward network (FFN) experts per token. However, this architectural paradigm introduces the **MoE Memory Wall**:

1. **Parameter Scale vs. On-Package Memory:** An MoE model such as DeepSeek-V3 houses 256 routed experts per stage across 40 MoE stages, requiring hundreds of gigabytes of storage. High-Bandwidth Memory (HBM) on contemporary accelerator GPUs (e.g., 80 GiB on NVIDIA A100-SXM4) cannot hold the entire expert inventory alongside attention key-value (KV) caches, activation tensors, and dense prefix layers.
2. **Off-Chip Backing Memory Bottleneck:** In tiered memory systems, unmapped expert weights reside in high-capacity host off-chip DRAM (e.g., DDR5-3200 pools). Transferring an 84 MiB expert weight chunk across a 22.578 GB/s effective interconnect imposes a fixed serialized transfer latency:
   $$T_{\text{transfer}} = \frac{84 \times 1024^2 \text{ Bytes}}{22.578 \times 10^9 \text{ Bytes/s}} \approx 3.901\text{ ms}$$
3. **Severe Inference Stalls:** During auto-regressive decode generation, routing decisions are made dynamically at each layer. When a requested expert is absent from HBM, execution stalls completely while the weight chunk is fetched from DDR5 DRAM. Across multi-batch decoding, cumulative memory stall times dominate end-to-end execution, causing tens of seconds of GPU idle time.

### Project 2 Solution Architecture
To conquer the MoE Memory Wall, this project implements and evaluates a four-pillar co-designed architectural system evaluated within `LLMSimulator` (a discrete-event, cycle-approximate hardware simulator integrated with Ramulator 2 DDR5/HBM memory modeling):

```
                                  PROPOSED ARCHITECTURE
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                                                                             │
    │  [1] CSTP Predictor                                                         │
    │      Spatio-temporal correlation predictor forecasts Top-K experts          │
    │      H layers ahead during upstream attention computation.                  │
    │                                │                                            │
    │                                ▼                                            │
    │  [2] Asynchronous DDR5 Prefetch Engine                                      │
    │      Dispatches DMA requests over PCIe/DDR5 to overlap expert transfers      │
    │      with compute; late requests attach directly to in-flight transfers.    │
    │                                │                                            │
    │                                ▼                                            │
    │  [3] Frequency-Aware LRU (FA-LRU) HBM Cache Controller                      │
    │      Sliding-window (W=64) frequency tracking protects reused experts       │
    │      from eviction, preventing cache pollution from mispredicted prefetches.│
    │                                │                                            │
    │                                ▼                                            │
    │  [4] Subsystem Energy & EDP Accounting                                      │
    │      Optimizes Energy-Delay Product by balancing static DRAM energy         │
    │      savings against speculative DDR5 active transfer overheads.            │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
```

---

# 2. Project Objectives and Implemented Points

The project research evolution spans seven distinct points:

| Point | Title | Objective | Key Microarchitectural Contribution |
|---|---|---|---|
| **Point 1** | Expert Selection Prediction & Profiling Sensitivity | Evaluate training-free expert prediction against literature baselines | Evaluates the ST-MoE literature baseline (Zhao et al., 2024) and the proposed Correlation-based Spatio-Temporal Predictor (CSTP). Profiles sensitivity across 192 configurations (5% to 50% profiling, seeds 42, 123, 999). CSTP achieves **82.02% recall** (+13.90 pp absolute, +20.40% relative over ST-MoE's 68.12%). |
| **Point 2** | Prefetching Mechanism, Latency Hiding & Trace Audit | Derive analytical latency hiding model and audit simulation traces | Formulates transfer timing ($T_{\text{transfer}} = 3.901\text{ ms}$ for 84 MiB at 22.578 GB/s). Implements in-flight transfer attachment for late prefetches. Resolves synthetic routing diversity collisions across sequences in a batch. |
| **Point 3** | Prefetching Effectiveness Evaluation Across Large Workloads | Quantify runtime and memory-stall reductions across large workloads | Evaluates 7 large workloads (W1 to W7, batch sizes 14 to 70, context lengths 512 to 8192, 128 decode steps) on DeepSeek-V3 using 73 cycle-level simulations. W7 achieves **6.717 s runtime reduction**, **6,744.86 ms stall reduction**, and **104.95 ms/step decode saving**. |
| **Point 4** | HBM Management & Eviction Policy Extension (FA-LRU) | Eliminate cache pollution caused by speculative prefetching | Implements Frequency-Aware LRU (`LLMSimulator/src/hardware/device.h`) using an epoch-decayed frequency counter ($W=64$). Prevents aggressive speculative prefetches from evicting hot resident experts. Tested across Big Experiments W3 and W4 at 35 GiB HBM capacity. |
| **Point 5** | Four-Way Architectural Comparative Evaluation | Disentangle prefetching gains from caching gains on anchor workload | Evaluates 4 configurations (Config A: Demand+LRU, Config B: Demand+FA-LRU, Config C: CSTP+LRU, Config D: CSTP+FA-LRU) on Anchor Workload (B=32, P=2048, T=128, 22.578 GB/s). Config D achieves **-12.93 s (-14.08%) runtime saving** and **-12,927.68 ms (-19.92%) stall reduction**. |
| **Point 6** | Multi-Dimensional Sensitivity Analysis | Characterize system scaling across architectural dimensions | Sweeps Lookahead Horizon ($H=1, 2, 4$), Prefetch Budget ($K=1, 2, 4, 8$), Batch Size ($B=1 \dots 32$), and Context Length ($P=512 \dots 8192$). Identifies $K=2$ as the prefetch sweet spot and maps bandwidth saturation regimes. |
| **Point 7** | Subsystem Energy Modeling and EDP Optimization | Model hardware energy components and optimize Energy-Delay Product | Analyzes 4 physical energy components (Static DRAM @ 24.0 W, DDR5 Active @ 3.5 pJ/bit, HBM2e @ 7.0 pJ/bit, GPU MAC compute) across 28 configurations. $K=2$ optimum saves -10.93 s runtime, -27.68 J net energy, achieving **-1.72% EDP reduction** ($7,079.32\text{ kJ}\cdot\text{s}$ vs. $7,202.85\text{ kJ}\cdot\text{s}$ baseline). |

---

# 3. Project Directory Structure and Result Organization

## 3.1 Repository Layout

The repository is structured into the simulator core, experiment workspaces under `mohman/`, standardized results, scripts, and submission deliverables:

```text
ca_project_endsem/
├── LLMSimulator/                          # Discrete-event C++ cycle-approximate MoE simulator
│   ├── CMakeLists.txt                     # CMake build configuration (C++17, yaml-cpp, spdlog)
│   ├── configs/                           # Hardware and memory YAML configurations
│   ├── data/downloads/                    # Bundled authentic MoE routing traces (Parquet)
│   │   ├── moe_anywhere/                  # Sage-A1.5 authentic trace (e1-sage-a1.5.parquet, 16.2 MB)
│   │   └── moe_suite/                     # GLM-5.2 authentic traces (12 Parquet files, ~7 MB each)
│   ├── eval/                              # Simulator test harnesses and experiment drivers
│   │   ├── CMakeLists.txt                 # Evaluation binaries build targets
│   │   ├── run_deepseekv3_mixtral_style.cpp # Main DeepSeek-V3 sequential decode simulator
│   │   ├── test_fa_lru.cpp                # FA-LRU cache controller unit and regression test
│   │   ├── test_synthetic_routing_diversity.cpp # Routing diversity collision regression suite
│   │   ├── test_prefetch_metrics.cpp      # Prefetch metrics collector verification test
│   │   └── calibrate_ddr5_transfers.cpp   # DDR5 DMA transfer engine timing calibration
│   └── src/                               # Simulator subsystem implementations
│       ├── prefetch/                      # Predictors, transfer engine and prefetching pipelines
│       │   ├── ddr5_transfer_engine.cpp/h # Calibrated asynchronous DMA transfer engine
│       │   ├── cstp_predictor.cpp/h       # CSTP C++ predictor implementation
│       │   ├── st_moe_predictor.cpp/h     # ST-MoE literature baseline predictor
│       │   └── prefetch_scheduler.cpp/h   # Asynchronous prefetch scheduler and dispatch logic
│       ├── dram/                          # Ramulator 2 DDR5 and memory pool interface
│       │   └── dram_interface.cpp/h       # Multi-tier memory controller interface
│       └── scheduler/                     # Multi-batch sequential decode scheduler
├── mohman/                                # Main research evolution and experiment workspace
│   ├── point_1/                           # Point 1: Expert Prediction & Profiling Sensitivity
│   │   ├── cstp/                          # CSTP implementation, unit tests, and 192-run results
│   │   │   ├── src/                       # Python reference implementation (cstp_predictor.py)
│   │   │   ├── tests/                     # Unit and regression test suites
│   │   │   └── results/csv/               # Certified profiling CSVs (corrected_aggregate_metrics.csv)
│   │   ├── st_moe/                        # ST-MoE literature reproduction workspace
│   │   │   ├── scripts/                   # generate_results.py, generate_plots.py
│   │   │   ├── results/                   # fixed_budget_summary.csv, native_threshold_results.csv
│   │   │   └── plots/                     # Generated 300 DPI PNG and vector PDF figures
│   │   └── stmoe_cstp/                    # Head-to-head comparative analysis and reports
│   ├── point_2/                           # Point 2: Latency Hiding Derivation, Metrics & Audit
│   │   ├── COMPLETE_PROJECT_TECHNICAL_DOCUMENTATION.md # Comprehensive 1400-line technical manual
│   │   └── COMPLETE_EXPERIMENT_REPRODUCTION_GUIDE.md   # Step-by-step reproduction guide
│   ├── point_3/                           # Point 3: Large Workload Campaign (W1–W7, 73 runs)
│   │   ├── configs/                       # Workload simulation YAML configs
│   │   ├── traces/                        # Ground-truth production routing traces
│   │   └── final_results/                 # Authoritative W1-W7 results
│   │       ├── data/consolidated_csv/     # Authoritative metrics repository
│   │       │   ├── all_experiment_results.csv # Complete 73-simulation metric repository
│   │       │   ├── latency_and_stall.csv      # Workload runtime, stall, and decode latency table
│   │       │   ├── memory_traffic.csv         # DDR5 demand vs speculative traffic breakdown
│   │       │   └── prefetch_effectiveness.csv # Coverage, timeliness, and hit rate summaries
│   │       ├── scripts/                   # consolidate_results.py, generate_plots.py, validate_results.py
│   │       └── plots/                     # Publication figures (01_total_time through 10_throughput)
│   ├── point_4/                           # Point 4: HBM Management & FA-LRU Eviction Policy
│   │   ├── big_exp/                       # Big Experiment campaign (W3 & W4, 16 runs)
│   │   │   ├── results/                   # campaign_manifest.csv, all_runs.csv, paired_comparisons.csv
│   │   │   ├── scripts/                   # run_big_experiment.py, plot_big_experiment.py, plot_fig9_scaling.py
│   │   │   └── figures/                   # 8 publication figures in PNG/PDF formats
│   │   └── implementation/                # FA-LRU source files and unit tests
│   ├── point_5/                           # Point 5: Four-Way Architectural Comparative Evaluation
│   │   ├── extracted_data/                # unified_comparison.csv, exposed_stall_comparison.csv
│   │   │   ├── extract_point5_data.py     # Data extraction pipeline
│   │   │   └── generate_comparative_tables.py # Markdown and CSV table generator
│   │   ├── figures/                       # plot_point5_figures.py (Figures 7.1, 7.2, 7.3)
│   │   └── reports/                       # Point 5 comparative evaluation reports
│   ├── point_6/                           # Point 6: Multi-Dimensional Sensitivity Analysis
│   │   └── final_results/                 # Consolidated sensitivity sweep outputs
│   │       ├── consolidated_data/         # unified_point6_results.csv, budget/lookahead/batch CSVs
│   │       ├── execute_point6_campaign.py # Sweep campaign runner
│   │       ├── figures/                   # generate_point6_figures.py (15 publication figures)
│   │       └── reports/                   # Point 6 sensitivity technical reports
│   ├── point_7/                           # Point 7: Subsystem Energy Modeling and EDP
│   │   ├── extracted_data/                # unified_energy_results.csv, energy_component_breakdown.csv
│   │   │   ├── extract_point7_energy_data.py # Physical and analytical energy calculator
│   │   │   └── validate_point7_energy_data.py# 7-gate microarchitectural invariant validator
│   │   ├── figures/                       # generate_point7_figures.py (12 certified energy figures)
│   │   └── final_audit/                   # Mathematical audits and energy model consistency reports
│   └── submission_final/                  # Authoritative End-Semester Submission Deliverables
│       ├── MoE_Memory_Wall_Final_Report.docx # Formatted Word report
│       ├── MoE_Memory_Wall_Final_Report.pdf  # 56-page publication PDF
│       ├── NUMERICAL_RECONCILIATION_MATRIX.csv# 120 audited numerical claims
│       ├── REPORT_CONSISTENCY_VALIDATION.md   # 63/63 checks passed certification report
│       ├── FIGURE_DATA/                   # CSV source data for all 32 figures in the report
│       ├── FINAL_CORRECTION_SUMMARY.md    # Summary of final reconciliations
│       ├── SUBMISSION_MANIFEST.txt        # Full submission inventory manifest
│       └── MoE_Memory_Wall_Final_Submission.zip # Complete standalone archive
├── final_submission/                      # Modular deliverable mirrors (01_PREDICTION through 09_BONUS)
├── results/                               # Global benchmark and phase results repository
├── scripts/                               # Root packaging and campaign verification scripts
└── MASTER_GROUP13_PROJECT2_README.md      # This master reproducibility guide
```

---

## 3.2 Point-Wise Result Mapping Table

The following table maps every implemented project point to its experiment design, configuration file, execution command, raw output location, processed results, and final report figures:

| Point | Main Experiment | Configuration | Execution Script | Raw Results Directory | Authoritative Processed Results | Final Report Figures & Tables |
|---|---|---|---|---|---|---|
| **Point 1** | ST-MoE Literature Reproduction & Profiling Sensitivity (192 configs) | `mohman/point_1/st_moe/configs/experiment_manifest.json` | `python3 mohman/point_1/st_moe/scripts/generate_plots.py` | `mohman/point_1/st_moe/results/csv/` | `mohman/point_1/cstp/results/csv/corrected_aggregate_metrics.csv` & `fixed_budget_summary.csv` | Table 3.1; Figures 3.1, 3.2, 3.3, 3.4, 3.5 |
| **Point 2** | Analytical Latency Hiding Derivation & Diversity Audit | Calibrated DDR5-3200 (22.578 GB/s), Top-8, 84 MiB chunk | `./LLMSimulator/build/test_synthetic_routing_diversity` & `./LLMSimulator/build/test_ddr5_transfer_engine` | Standard terminal test output | `mohman/point2/COMPLETE_PROJECT_TECHNICAL_DOCUMENTATION.md` | Table 4.1; Transfer timeline derivations |
| **Point 3** | Large Workload Campaign (W1–W7, 73 cycle-level runs, 64 GB/s) | `mohman/point_3/configs/*.yaml` | `python3 mohman/point_3/final_results/scripts/generate_plots.py` | `mohman/point_3/results/` | `mohman/point_3/final_results/data/consolidated_csv/latency_and_stall.csv` & `all_experiment_results.csv` | Tables 5.1, 5.2; Figures 5.1 to 5.14 |
| **Point 4** | FA-LRU Eviction Policy & Big Experiment (W3/W4, Cap=35 GiB, 16 runs) | `mohman/point_4/configs/` & `LLMSimulator/build/test_fa_lru` | `python3 mohman/point_4/big_exp/scripts/plot_big_experiment.py` | `mohman/point_4/big_exp/results/` | `mohman/point_4/big_exp/results/all_runs.csv` & `paired_comparisons.csv` | Table 6.1; Figures 6.1, 6.2 |
| **Point 5** | Four-Way Comparative Architecture Evaluation (Configs A, B, C, D) | Anchor Workload: B=32, P=2048, T=128, Cap=35 GiB, 22.578 GB/s | `python3 mohman/point_5/figures/plot_point5_figures.py` | `mohman/point_5/extracted_data/` | `mohman/point_5/extracted_data/unified_comparison.csv` & `exposed_stall_comparison.csv` | Table 7.1; Figures 7.1, 7.2, 7.3 |
| **Point 6** | Multi-Dimensional Sensitivity Sweeps (B, P, H, K sweeps) | Sweeps across $H \in \{1,2,4\}$, $K \in \{1,2,4,8\}$, $B \in \{1\dots 32\}$, $P \in \{512\dots 8192\}$ | `python3 mohman/point_6/final_results/figures/generate_point6_figures.py` | `mohman/point_6/final_results/consolidated_data/` | `mohman/point_6/final_results/consolidated_data/unified_point6_results.csv` | Tables 8.1, 8.2; Figures 8.1 to 8.6 |
| **Point 7** | Subsystem Energy Modeling & EDP Optimization (28 runs) | 4 Subsystems: Static DRAM (24W), DDR5 Active (3.5 pJ/b), HBM (7 pJ/b), GPU MAC | `python3 mohman/point_7/figures/generate_point7_figures.py` | `mohman/point_7/raw_results/` | `mohman/point_7/extracted_data/unified_energy_results.csv` & `paired_energy_comparisons.csv` | Tables 9.1, 9.2; Figures 9.1, 9.2, 9.3 |

---

## 3.3 Authoritative Result Files and Artifact Types

To eliminate any ambiguity between raw traces, intermediate telemetry, and certified report numbers, the repository enforces a strict four-tier hierarchy:

1. **Tier 1 — Raw Simulator Telemetry:** Generated directly during discrete-event C++ execution (e.g., `expert_cache.csv`, `prediction_events.csv`, `request_latency.csv`). These reside inside simulation output subdirectories under `mohman/point_3/results/` and `mohman/point_4/smoke_tests/`.
2. **Tier 2 — Intermediate Aggregations:** Machine-parsed summaries consolidating thousands of simulation cycles into run-level rows (e.g., `all_runs.csv`, `campaign_manifest.csv`).
3. **Tier 3 — Authoritative Processed Result CSVs:** The certified single sources of truth used across all report analyses:
   - Point 1: `mohman/point_1/cstp/results/csv/corrected_aggregate_metrics.csv`
   - Point 3: `mohman/point_3/final_results/data/consolidated_csv/latency_and_stall.csv` & `all_experiment_results.csv`
   - Point 4: `mohman/point_4/big_exp/results/paired_comparisons.csv`
   - Point 5: `mohman/point_5/extracted_data/unified_comparison.csv`
   - Point 6: `mohman/point_6/final_results/consolidated_data/unified_point6_results.csv`
   - Point 7: `mohman/point_7/extracted_data/unified_energy_results.csv`
4. **Tier 4 — Submission Figures and Reconciliation Matrices:**
   - 32 CSV exports directly matching report plots: `mohman/submission_final/FIGURE_DATA/figure_*_data.csv`
   - Complete 120-claim verification ledger: `mohman/submission_final/NUMERICAL_RECONCILIATION_MATRIX.csv`
   - 63-check automated audit pass: `mohman/submission_final/REPORT_CONSISTENCY_VALIDATION.md`

---

# 4. System Requirements and Hardware Requirements

### Host Environment Specifications
- **Operating System:** Linux x86_64. Tested and verified on Ubuntu 24.04.1 LTS and Ubuntu 22.04 LTS (also fully supported on Windows Subsystem for Linux 2 / WSL2).
- **CPU:** 4+ physical cores minimum (8+ recommended for parallel compilation with `make -j$(nproc)`).
- **Host RAM:** 16 GiB minimum (32 GiB recommended for parsing large multi-million token Parquet routing traces).
- **Storage:** Minimum 15 GiB available disk space (covers repository, authentic Parquet traces, C++ build artifacts, and output figure directories).
- **GPU Requirements:** **NONE REQUIRED.** The primary evaluation platform (`LLMSimulator`) is a cycle-approximate, discrete-event C++ software simulator that executes completely on CPU host hardware. Physical NVIDIA A100 or H100 GPUs are **not** needed to compile the simulator, run the regression test suites, execute simulations, or generate publication figures.

### Toolchain and Compiler Requirements
- **C++ Compiler:** GCC / G++ version $\ge 11.0$ supporting C++17 (G++ 13.3.0 tested).
- **Build System:** CMake version $\ge 3.14$ (CMake 3.28.3 tested).
- **Make Tool:** GNU Make $\ge 4.3$.
- **Python Runtime:** Python 3.10, 3.11, or 3.12 (Python 3.12.3 tested).

---

# 5. Fresh Laptop Setup — Step by Step

## 5.1 Repository Cloning and Verification

On a newly provisioned Linux laptop or WSL2 terminal, clone the repository directly from GitHub:

```bash
# Clone the repository
git clone https://github.com/Mohit14032007/ca_project_endsem.git
cd ca_project_endsem

# Verify git branch and commit status
git branch --show-current
git status
git log -1 --oneline
```

Expected output:
```text
main
On branch main
Your branch is up to date with 'origin/main'.
19ec0e7 Update submission manifest with exact commit hash and refresh final submission zip
```

---

## 5.2 Host Path Compatibility Configuration

During development, certain Python plotting and validation scripts were configured with the author's reference workspace path (`/home/mohman/ca_project`). To ensure that **all scripts run seamlessly without modifying a single line of project source code**, use either of the following two portable methods:

### Method A (Recommended — Instant One-Command Symlink)
Create a symbolic link from your local cloned repository directory to `/home/mohman/ca_project`:

```bash
# In the root of your cloned repository:
sudo mkdir -p /home/mohman
sudo ln -s "$(pwd)" /home/mohman/ca_project

# Verify symlink
ls -l /home/mohman/ca_project
```
*Why this works:* Any script referencing `/home/mohman/ca_project` instantly resolves directly to your local clone, giving 100% out-of-the-box execution across all points.

### Method B (Environment Variable / Current Working Directory)
If you do not have root (`sudo`) access, export the repository root and run scripts from the project root:

```bash
export REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/mohman:${REPO_ROOT}/LLMSimulator:${PYTHONPATH}"
```

---

# 6. Dependency Installation

## 6.1 Linux System Packages

Install the necessary build tools and C++ libraries via `apt`:

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-venv \
    python3-pip \
    libyaml-cpp-dev
```

---

## 6.2 Python Virtual Environment and Packages

Set up an isolated Python virtual environment:

```bash
# Create and activate virtual environment in the repository root
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install required Python dependencies
pip install \
    numpy \
    pandas \
    matplotlib \
    pyarrow \
    fastparquet \
    python-docx
```

Verify installed packages:
```bash
python -c "import numpy, pandas, matplotlib, docx; print('Python environment successfully verified!')"
```

---

# 7. Build Instructions

## 7.1 Compiling LLMSimulator with CMake

The core simulator and test harnesses are located in `LLMSimulator/`. Compile using CMake in Release mode:

```bash
# Navigate to LLMSimulator build directory
cd LLMSimulator
mkdir -p build
cd build

# Configure CMake in Release mode
cmake -DCMAKE_BUILD_TYPE=Release ..

# Compile all targets using all CPU cores
make -j"$(nproc)"
```

---

## 7.2 Generated Binaries and Libraries

Upon successful compilation, CMake outputs the following executables and shared libraries directly into `LLMSimulator/build/`:

| Binary / Library | Path | Role in Evaluation |
|---|---|---|
| `run_deepseekv3_mixtral_style` | `LLMSimulator/build/run_deepseekv3_mixtral_style` | Main discrete-event simulation driver for DeepSeek-V3 sequential decoding |
| `libllm_system.so` | `LLMSimulator/build/libllm_system.so` | Shared system library housing GPU device, prefetcher, and DRAM controllers |
| `test_fa_lru` | `LLMSimulator/build/test_fa_lru` | Point 4 FA-LRU cache controller unit and regression test suite (14 tests) |
| `test_synthetic_routing_diversity` | `LLMSimulator/build/test_synthetic_routing_diversity` | Point 2 routing diversity collision regression suite (8 tests) |
| `test_prefetch_metrics` | `LLMSimulator/build/test_prefetch_metrics` | Point 2 prefetch metrics collector unit test suite (15 tests) |
| `test_ddr5_transfer_engine` | `LLMSimulator/build/test_ddr5_transfer_engine` | Point 2 calibrated event-driven DDR5 transfer engine unit tests (12 tests) |
| `calibrate_ddr5_transfers` | `LLMSimulator/build/calibrate_ddr5_transfers` | Timing calibration binary for DDR5-3200 serialized DMA transfers |

---

## 7.3 Preflight Build Verification

Run the core unit test suites to verify that the compiled simulator binaries operate correctly:

```bash
# From LLMSimulator/build/ (or ca_project_endsem root)
./LLMSimulator/build/test_fa_lru
./LLMSimulator/build/test_synthetic_routing_diversity
./LLMSimulator/build/test_ddr5_transfer_engine
./LLMSimulator/build/test_prefetch_metrics
```

Expected output:
```text
[SUCCESS] ALL 14 EXTENDED UNIT TESTS PASSED CLEANLY!
[SUCCESS] All synthetic routing diversity regression tests passed!
TEST SUITE SUMMARY: 12 PASSED, 0 FAILED
Test Summary: 15 Passed, 0 Failed
```

---

# 8. Dataset and Routing Trace Preparation

## 8.1 Authentic MoE Traces (Sage-A1.5 and MoE-Suite)

Unlike projects that rely on synthetic pseudo-random routing, our evaluation is grounded on authentic MoE token routing traces bundled directly in the repository:

1. **MoE-Anywhere Trace (Sage-A1.5):**
   - Path: `LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet`
   - File size: 16.17 MB (16,956,227 bytes)
   - Parameters: 40 MoE stages, 256 routed experts per stage, Top-8 routing ($K=8$)
   - Used in: Point 1 (CSTP vs. ST-MoE profiling sensitivity), Point 3, Point 5, Point 6, Point 7
2. **MoE-Suite Traces (GLM-5.2):**
   - Path: `LLMSimulator/data/downloads/moe_suite/*.parquet` (12 files, ~7 MB each, e.g., `corpus-08-cpp-mmap-index.parquet`)
   - Parameters: 75 layers, 256 experts per layer
   - Used in: Point 1 cross-workload diagnostic evaluation

Confirm the datasets exist and check their sizes:
```bash
ls -lh LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet
ls -lh LLMSimulator/data/downloads/moe_suite/ | head -n 5
```

---

## 8.2 Trace Verification and Schema Invariants

Production routing CSV traces extracted from the Parquet files are organized under `mohman/point_3/traces/` and `mohman/point_3/routing_audit/traces/`:

- Record count per decode step: $B \times T \times L_{\text{MoE}} \times K$
- For Workload W3 ($B=14, T=128, L=40, K=8$): $14 \times 128 \times 40 \times 8 = 573,440$ rows (573,441 lines with header).
- For Workload W4 ($B=32, T=128, L=40, K=8$): $32 \times 128 \times 40 \times 8 = 1,310,720$ rows (1,310,721 lines with header).

Verify line counts:
```bash
# Check W3 trace row count
wc -l mohman/point_3/traces/w3_b14_t128_production.csv 2>/dev/null || true
```

---

# 9. Configuration Guide

## 9.1 Architectural Baseline Parameters

All experiments simulate the following standardized hardware memory subsystem:

| Parameter | Architectural Value | Description |
|---|---|---|
| **Model Target** | DeepSeek-V3 / Mixtral-Style | 43 total layers: 3 dense prefix layers + 40 MoE stages |
| **Expert Topology** | 256 routed experts / stage | Top-8 routing ($K_{\text{routed}} = 8$), softmax gating |
| **Expert Chunk Size** | 84 MiB ($88,080,384\text{ Bytes}$) | Physical weight size per routed expert chunk |
| **GPU Subsystem** | NVIDIA A100-SXM4-80GB (Simulated) | Peak compute throughput, 80 GiB physical package capacity |
| **Off-Chip Backing DRAM** | 2.75 TB DDR5-3200 Memory Pool | Houses the complete 860 GB parameter inventory off-chip |
| **Calibrated Interconnect** | 22.578 GB/s effective bandwidth | Measured DDR5-3200 transfer rate (3.901 ms per 84 MiB expert) |
| **HBM Cache Allocation** | 35 GiB default cache capacity | 416 resident expert slots (allocating remainder to KV-cache) |
| **Predictor Settings** | $\alpha = 0.5$, $\lambda = 1.0$, $\epsilon = 0.1$ | Laplace-smoothed Dirichlet spatio-temporal co-activation |
| **Prefetch Settings** | Lookahead $H=1$, Budget $K=4$ (Anchor) | Lookahead distance in layers; speculative prefetch budget |
| **Eviction Settings** | Sliding window $W=64$ (FA-LRU) | Access frequency window with exponential epoch decay |

---

## 9.2 Configuration Files by Experiment Point

- **LLMSimulator Core:** `LLMSimulator/configs/` (`config.yaml`, `dram_config_HBM3_80GB.yaml`)
- **Point 1 Predictors:** `mohman/point_1/st_moe/configs/experiment_manifest.json`
- **Point 3 Workloads (W1–W7):** `mohman/point_3/configs/` (`w1_config.yaml` to `w7_config.yaml`)
- **Point 4 Big Experiments:** `mohman/point_4/big_exp/results/campaign_manifest.csv`
- **Point 5 Four Configurations (A, B, C, D):** `mohman/point_5/extracted_data/unified_comparison.csv`
- **Point 6 Sensitivity Sweeps:** `mohman/point_6/final_results/consolidated_data/unified_point6_results.csv`
- **Point 7 Energy Configurations (28 runs):** `mohman/point_7/configs/` & `mohman/point_7/extracted_data/experiment_manifest.csv`

---

# 10. Point-Wise Implementation and Reproduction Commands

---

### Point 1: Expert Selection Prediction & Profiling Sensitivity

#### Objective
Evaluate the predictive accuracy of training-free expert prediction algorithms. Quantify the performance of the literature baseline (ST-MoE) versus the proposed Spatio-Temporal Predictor (CSTP) across varying offline profiling budgets (5% to 50%) and multiple random seeds.

#### Implementation
- ST-MoE Baseline: Implemented in `mohman/point_1/st_moe/scripts/generate_results.py` and `LLMSimulator/src/prefetch/st_moe_predictor.cpp`. Tracks Conditional Co-activation Tables (CCT) under frozen and online regimes.
- CSTP Predictor: Implemented in `mohman/point_1/cstp/src/cstp_predictor.py` and `LLMSimulator/src/prefetch/cstp_predictor.cpp`. Employs Laplace smoothing with Dirichlet prior weighting across cross-layer spatial co-activation and temporal history.

#### Important Source Files
- Reference CSTP implementation: `mohman/point_1/cstp/src/cstp_predictor.py`
- ST-MoE results generator: `mohman/point_1/st_moe/scripts/generate_results.py`
- ST-MoE plotting script: `mohman/point_1/st_moe/scripts/generate_plots.py`
- Unit test suite: `mohman/point_1/cstp/tests/test_cstp.py`

#### Input Data
- Authentic Sage-A1.5 routing trace: `LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet`
- MoE Suite GLM-5.2 traces: `LLMSimulator/data/downloads/moe_suite/*.parquet`

#### Configuration
- Profiling percentages: 5%, 10%, 20%, 30%, 50%
- Random seeds: 42, 123, 999
- Lookahead distance: $H \in \{1, 2\}$; Prediction budget: $B \in \{1, 2, 4, 6, 8\}$

#### Step 1 — Prepare inputs
The traces are pre-bundled in `LLMSimulator/data/downloads/`. No external download is needed.

#### Step 2 — Run the experiment
To regenerate the ST-MoE results and sweep across prediction budgets:
```bash
python3 mohman/point_1/st_moe/scripts/generate_results.py
```

#### Step 3 — Process the results
The results are consolidated into CSV files in `mohman/point_1/st_moe/results/csv/` and `mohman/point_1/cstp/results/csv/`.

#### Step 4 — Generate figures and tables
Execute the dedicated plotting script:
```bash
python3 mohman/point_1/st_moe/scripts/generate_plots.py
```

#### Expected outputs
The following publication-quality figures are generated in both PNG (300 DPI) and PDF formats under `mohman/point_1/st_moe/plots/`:
- `recall_comparison.png / .pdf`
- `frozen_vs_online.png / .pdf`
- `horizon_comparison.png / .pdf`
- `hit_rate_comparison.png / .pdf`
- `cold_vs_warm_recall.png / .pdf`
- `native_threshold_tradeoff.png / .pdf`
- `profiling_sensitivity.png / .pdf`

#### Verification
Check that CSTP achieves **82.02% recall** compared to ST-MoE's **68.12% recall** (+13.90 pp absolute, +20.40% relative improvement) in `mohman/point_1/cstp/results/csv/corrected_aggregate_metrics.csv`.

---

### Point 2: Prefetching Mechanism, Latency Hiding & Trace Audit

#### Objective
Derive the physical transfer latency for expert weight chunks over calibrated DDR5 memory. Implement late prefetch in-flight attachment. Resolve synthetic routing diversity collisions across sequences in a batch.

#### Implementation
- Asynchronous DMA transfer engine: `LLMSimulator/src/prefetch/ddr5_transfer_engine.cpp/h`
- Multi-sequence routing fix: `LLMSimulator/eval/test_synthetic_routing_diversity.cpp`
- Prefetch metrics collector: `LLMSimulator/eval/test_prefetch_metrics.cpp`

#### Important Source Files
- Transfer engine: `LLMSimulator/src/prefetch/ddr5_transfer_engine.cpp`
- Transfer engine header: `LLMSimulator/src/prefetch/ddr5_transfer_engine.h`
- Documentation: `mohman/point2/COMPLETE_PROJECT_TECHNICAL_DOCUMENTATION.md`

#### Input Data
- Synthetic multi-sequence batch traces ($B \in \{1, 2, 4, 8, 16, 32\}$) generated during unit test execution.

#### Configuration
- Interconnect rate: 22.578 GB/s (calibrated DDR5-3200)
- Chunk size: 84 MiB ($88,080,384\text{ Bytes}$)
- Transfer duration: $T_{\text{transfer}} = \frac{88,080,384}{22.578 \times 10^9} = 3.901160\text{ ms}$

#### Step 1 — Run the DDR5 transfer engine unit tests
```bash
./LLMSimulator/build/test_ddr5_transfer_engine
```
Expected output:
```text
[PASS] Test A: Single demand (Completed at 3.901160 ms)
[PASS] Test B: Two simultaneous demands (T1=3.901160 ms, T2=7.802319 ms)
[PASS] Test F: Demand preempts active prefetch (Demand completed at 4.901160ms, Prefetch resumed and finished at 7.802319ms)
TEST SUITE SUMMARY: 12 PASSED, 0 FAILED
```

#### Step 2 — Run the synthetic routing diversity regression suite
```bash
./LLMSimulator/build/test_synthetic_routing_diversity
```
Expected output:
```text
[RUNNING] test_A_different_sequences_not_share_cache... [PASS]
[RUNNING] test_G_real_trace_bypasses_cache... [PASS]
Test Summary: 8 passed, 0 failed.
```

#### Step 3 — Run prefetch metrics collector unit tests
```bash
./LLMSimulator/build/test_prefetch_metrics
```
Expected output:
```text
Test Summary: 15 Passed, 0 Failed
```

#### Verification
Confirm all 12 transfer engine tests, 8 routing diversity tests, and 15 prefetch metric tests pass with 0 failures.

---

### Point 3: Prefetching Effectiveness Evaluation Across Large Workloads (W1–W7)

#### Objective
Quantify the simulated execution time and exposed memory-stall reduction of CSTP prefetching across 7 large workloads (W1 to W7) on DeepSeek-V3 using 73 cycle-level simulations.

#### Implementation
- Simulation campaigns executed via `run_deepseekv3_mixtral_style` on a 64 GB/s interconnect.
- Consolidated results, metrics parsing, and plot generation located in `mohman/point_3/final_results/scripts/`.

#### Important Source Files
- Consolidation script: `mohman/point_3/final_results/scripts/consolidate_results.py`
- Plot generator: `mohman/point_3/final_results/scripts/generate_plots.py`
- Invariant validation script: `mohman/point_3/final_results/scripts/validate_results.py`

#### Input Data
- Workload traces: `mohman/point_3/traces/` (W1 to W7)
- Authoritative summary CSV: `mohman/point_3/final_results/data/consolidated_csv/latency_and_stall.csv`

#### Configuration
- Workloads:
  - W1: $B=14, P=512, T=128$
  - W2: $B=14, P=2048, T=128$
  - W3: $B=32, P=2048, T=128$
  - W4: $B=48, P=2048, T=128$
  - W5: $B=70, P=2048, T=128$
  - W6: $B=70, P=4096, T=128$
  - W7: $B=70, P=8192, T=128$

#### Step 1 — Validate existing campaign results
```bash
python3 mohman/point_3/final_results/scripts/validate_results.py
```

#### Step 2 — Regenerate publication plots
```bash
python3 mohman/point_3/final_results/scripts/generate_plots.py
```

#### Expected outputs
All figures generated under `mohman/point_3/final_results/plots/`:
- `01_total_time_vs_stall_reduction/fig_01a_w1_total_time_vs_stall.png / .pdf` through `w7`
- `01_total_time_vs_stall_reduction/fig_01b_all_workloads_time_and_stall_overview.png / .pdf`
- `01_total_time_vs_stall_reduction/fig_01c_time_vs_stall_tradeoff_scatter.png / .pdf`
- `01_total_time_vs_stall_reduction/fig_01d_batch_scaling_total_time.png / .pdf`
- `02_prefetch_coverage/`, `03_timeliness/`, `04_unused_prefetches/`, `05_hbm_hit_rate/`
- `06_memory_traffic/`, `07_exposed_memory_stall/`, `08_expert_load_latency/`
- `09_end_to_end_latency/`, `10_throughput/`, `comparison_dashboards/`

#### Verification
Confirm W7 achieves:
- Total simulated runtime reduction: **6.717 s**
- Total exposed stall reduction: **6,744.86 ms**
- Per-step decode saving: **104.95 ms/step**

---

### Point 4: HBM Management & Eviction Policy Extension (FA-LRU)

#### Objective
Implement and evaluate Frequency-Aware LRU (FA-LRU) caching to eliminate cache pollution caused by speculative prefetching bursts under constrained HBM capacity.

#### Implementation
- C++ implementation in `LLMSimulator/src/hardware/device.h` and `device.cpp`.
- Tracks expert frequency in a sliding window ($W=64$) with epoch-decayed counters. High-frequency entries are protected from eviction during speculative prefetch bursts.
- Evaluated on Big Experiments (W3: B=14 and W4: B=32 at 35 GiB HBM capacity, 16 runs).

#### Important Source Files
- Cache controller header: `LLMSimulator/src/hardware/device.h` (lines 260–350)
- Unit test harness: `LLMSimulator/eval/test_fa_lru.cpp`
- Big experiment plotting script: `mohman/point_4/big_exp/scripts/plot_big_experiment.py`
- Scaling plotting script: `mohman/point_4/big_exp/scripts/plot_fig9_scaling.py`

#### Input Data
- Big experiment results: `mohman/point_4/big_exp/results/all_runs.csv` & `paired_comparisons.csv`

#### Step 1 — Run the FA-LRU C++ unit test suite
```bash
./LLMSimulator/build/test_fa_lru
```

#### Step 2 — Generate Big Experiment figures
```bash
python3 mohman/point_4/big_exp/scripts/plot_big_experiment.py
python3 mohman/point_4/big_exp/scripts/plot_fig9_scaling.py
```

#### Expected outputs
Generated in `mohman/point_4/big_exp/figures/` (in both PNG and PDF):
- `fig1_hit_rate_comparison.png / .pdf`
- `fig2_expert_reloads_comparison.png / .pdf`
- `fig3_ddr5_traffic_comparison.png / .pdf`
- `fig4_exposed_stall_comparison.png / .pdf`
- `fig5_token_latency_comparison.png / .pdf`
- `fig6_cstp_cache_pollution.png / .pdf`
- `fig7_falru_diagnostics.png / .pdf`
- `fig8_repetition_consistency.png / .pdf`
- `fig9_falru_scaling_effect.png / .pdf`

#### Verification
Confirm FA-LRU improves hit rate from **68.97% to 72.10%** under demand execution, and from **71.45% to 74.88%** under prefetching, reducing DDR5 traffic by -54.24 GB and -30.00 GB respectively.

---

### Point 5: Four-Way Architectural Comparative Evaluation (Configs A, B, C, D)

#### Objective
Systematically isolate the individual and synergistic contributions of prefetching and caching on Anchor Workload W4 ($B=32, P=2048, T=128, \text{Cap}=35\text{ GiB}$ at 22.578 GB/s).

#### Implementation
- Evaluates four canonical configurations:
  - **Config A:** Baseline (Demand Loading + Standard LRU)
  - **Config B:** Cache Optimization (Demand Loading + FA-LRU $W=64$)
  - **Config C:** Prefetching Alone (CSTP Prefetching + Standard LRU)
  - **Config D:** Full Proposed Architecture (CSTP Prefetching + FA-LRU $W=64$)

#### Important Source Files
- Extraction script: `mohman/point_5/extracted_data/extract_point5_data.py`
- Table generator: `mohman/point_5/extracted_data/generate_comparative_tables.py`
- Plot generator: `mohman/point_5/figures/plot_point5_figures.py`
- Extended plot generator: `mohman/point_5/figures/plot_extended_figures.py`

#### Input Data
- Unified comparative results: `mohman/point_5/extracted_data/unified_comparison.csv`

#### Step 1 — Regenerate Point 5 comparative figures
```bash
python3 mohman/point_5/figures/plot_point5_figures.py
python3 mohman/point_5/figures/plot_extended_figures.py
```

#### Expected outputs
Generated under `mohman/point_5/figures/`:
- `fig1_predictor_effectiveness.png / .pdf`
- `fig2_prefetching_contribution.png / .pdf`
- `fig3_falru_contribution.png / .pdf`
- `fig4_overall_system_comparison.png / .pdf`

#### Verification
Verify the four-way metrics match Table 7.1 of the final report:
- Config A: Runtime = 91.81 s, Stall = 64,888.63 ms, Hit Rate = 68.97%, Traffic = 538.16 GB
- Config B: Runtime = 87.24 s (-4.57 s), Stall = 60,317.40 ms (-4,571.23 ms), Hit Rate = 72.10%, Traffic = 483.92 GB
- Config C: Runtime = 83.45 s (-8.36 s), Stall = 56,532.18 ms (-8,356.45 ms), Hit Rate = 71.45%, Traffic = 562.40 GB
- Config D: Runtime = 78.88 s (-12.93 s, -14.08%), Stall = 51,960.95 ms (-12,927.68 ms, -19.92%), Hit Rate = 74.88%, Traffic = 508.16 GB

---

### Point 6: Multi-Dimensional Sensitivity Analysis (B, P, H, K Sweeps)

#### Objective
Explore the microarchitectural sensitivity of prefetching and caching across four orthogonal dimensions: Lookahead Horizon ($H$), Prefetch Budget ($K$), Batch Size ($B$), and Context Length ($P$).

#### Implementation
- Automated sensitivity campaign parsing and plotting scripts in `mohman/point_6/final_results/`.

#### Important Source Files
- Consolidated dataset: `mohman/point_6/final_results/consolidated_data/unified_point6_results.csv`
- Figure generator: `mohman/point_6/final_results/figures/generate_point6_figures.py`
- Paired comparison script: `mohman/point_6/final_results/consolidated_data/generate_paired_comparisons.py`

#### Input Data
- Sweep tables in `mohman/point_6/final_results/consolidated_data/` (`lookahead_comparison.csv`, `budget_comparison.csv`, `batch_size_comparison.csv`, `context_length_comparison.csv`).

#### Step 1 — Regenerate all 15 Point 6 publication figures
```bash
python3 mohman/point_6/final_results/figures/generate_point6_figures.py
```

#### Expected outputs
15 figures in both PNG (300 DPI) and PDF formats under `mohman/point_6/final_results/figures/`:
- `batch_size/fig1_batch_size_hit_rate.png / .pdf` through `fig3`
- `context_length/fig4_context_length_kv_cache.png / .pdf` through `fig6`
- `lookahead/fig7_lookahead_resident_hits.png / .pdf` through `fig9`
- `prefetch_budget/fig10_budget_speculative_ddr5.png / .pdf` through `fig12`
- `combined/fig13_prefetch_vs_noprefetch.png / .pdf` through `fig15`

#### Verification
Confirm the lookahead dynamics:
- $H=1$: Hidden stall = 3,568.90 ms
- $H=2$: Hidden stall = 3,873.24 ms
- $H=4$: Hidden stall = 4,005.17 ms
Confirm budget dynamics: $K=2$ is optimal; $K=8$ causes bandwidth saturation (+142.1 GB wasted traffic).

---

### Point 7: Subsystem Energy Modeling and EDP Optimization

#### Objective
Model physical and analytical energy across 4 hardware subsystems (Static DRAM, DDR5 Active, HBM2e, GPU MAC compute) across 28 simulation runs, and identify the Energy-Delay Product (EDP) sweet spot.

#### Implementation
- Subsystem energy model:
  - Static DRAM Power: 24.0 W constant across 2.75 TB DDR5 pool ($E_{\text{static}} = 24.0 \times T_{\text{sim}}$)
  - DDR5 Active Transfer Energy: 3.5 pJ/bit ($28.0\text{ nJ/Byte}$)
  - HBM2e Read/Write Energy: 7.0 pJ/bit ($56.0\text{ nJ/Byte}$)
  - GPU MAC compute energy: Evaluated per token/stage
  - Metric: $\text{EDP} = E_{\text{total}} \times T_{\text{sim}}$

#### Important Source Files
- Extraction script: `mohman/point_7/extracted_data/extract_point7_energy_data.py`
- Invariant validator: `mohman/point_7/extracted_data/validate_point7_energy_data.py`
- Figure generator: `mohman/point_7/figures/generate_point7_figures.py`

#### Input Data
- Unified energy dataset: `mohman/point_7/extracted_data/unified_energy_results.csv` (28 rows)

#### Step 1 — Validate energy invariant accounting
```bash
python3 mohman/point_7/extracted_data/validate_point7_energy_data.py
```
Expected output:
```text
[VALIDATION] unified_energy_results.csv passed all 7 microarchitectural invariant tests!
[VALIDATION] paired_energy_comparisons.csv passed all compatibility and delta checks!
[VALIDATION] All Point 7 dataset validations PASSED with ZERO errors!
```

#### Step 2 — Regenerate all 12 Point 7 publication figures
```bash
python3 mohman/point_7/figures/generate_point7_figures.py
```

#### Expected outputs
12 figures in both PNG and PDF formats under `mohman/point_7/figures/`:
- `fig1_anchor_system_energy.png / .pdf`
- `fig2_anchor_energy_per_token.png / .pdf`
- `fig3_prefetch_budget_ddr5_breakdown.png / .pdf`
- `fig4_ddr5_traffic_vs_energy.png / .pdf`
- `fig5_stall_reduction_vs_energy_overhead.png / .pdf`
- `fig6_energy_delay_product.png / .pdf`
- `fig7_batch_size_energy_scaling.png / .pdf`
- `fig8_budget_sweet_spot.png / .pdf`
- `fig9_lookahead_energy_dynamics.png / .pdf`
- `fig10_context_length_energy_scaling.png / .pdf`
- `fig11_energy_performance_pareto.png / .pdf`
- `fig12_component_pie_breakdown.png / .pdf`

#### Verification
Verify the Energy-Delay Product values match Table 9.1 and Table 9.2:
- Baseline (No Prefetch): Runtime = 91.81 s, Total Energy = 78,452.12 J, EDP = **7,202.85 kJ·s**
- Optimal $K=2$: Runtime = 80.88 s (-10.93 s), Exposed Stall = 53,921.44 ms (-10,967.19 ms), Total Energy = 78,424.44 J (-27.68 J net saving), EDP = **7,079.32 kJ·s** (**-1.72% EDP reduction**)
- Aggressive $K=8$: Runtime = 79.80 s, Total Energy = 92,884.45 J, EDP = **7,412.18 kJ·s** (**+2.91% worse**)

---

# 11. ST-MoE Reproduction (Literature Baseline)

ST-MoE (Zhao et al., 2024) serves as the primary literature baseline. It implements training-free expert prediction using Conditional Co-activation Tables (CCT):

$$P(e_j^{(l+1)} \mid e_i^{(l)}) = \frac{C(e_i^{(l)}, e_j^{(l+1)})}{\sum_k C(e_i^{(l)}, e_k^{(l+1)})}$$

ST-MoE is evaluated under two operational regimes:
1. **Frozen Regime:** CCT is computed exclusively during offline profiling and frozen at inference time.
2. **Online Adaptation Regime:** CCT continues updating dynamically during inference. Evaluated under:
   - *Within-Request:* CCT resets at sequence boundaries (preventing inter-request contamination).
   - *Across-Requests:* Continuous online learning across the entire token trace.

### Execution Commands
```bash
# 1. Regenerate ST-MoE evaluation results across budgets B in {1, 2, 4, 6, 8}
python3 mohman/point_1/st_moe/scripts/generate_results.py

# 2. Regenerate ST-MoE publication figures
python3 mohman/point_1/st_moe/scripts/generate_plots.py
```

### Metrics Produced
- Macro Recall: 47.91% (within-request standard at $H=1$ on Sage-A1.5)
- Macro Precision: 47.91% (at budget $B = K_{\text{top}} = 8$)
- Macro Hit Rate: 96.94% (fraction of token steps with $\ge 1$ correctly predicted expert)
- Profiling Sensitivity: 68.12% Top-8 recall at 20% profiling budget

---

# 12. CSTP Reproduction (Proposed Spatio-Temporal Predictor)

The proposed Correlation-based Spatio-Temporal Predictor (CSTP) addresses the limitations of ST-MoE by fusing:
1. **Cross-Layer Spatial Co-activation:** Measures correlated expert activation between upstream layer $l$ and target layer $l+H$.
2. **Intra-Sequence Temporal History:** Captures token trajectory recurrence within the current request.
3. **Additive Laplace Smoothing:** Prevents zero-probability cold-start rejections:
   $$S(e_j) = \alpha \cdot S_{\text{spatial}}(e_j) + (1-\alpha) \cdot S_{\text{temporal}}(e_j)$$

### Standalone Python Verification
```bash
# Run CSTP unit test suite
python3 -m unittest mohman/point_1/cstp/tests/test_cstp.py
```

### Profiling Sensitivity Sweep (192 Configurations)
CSTP is evaluated across 192 configurations:
- 5 Profiling Percentages: 5%, 10%, 20%, 30%, 50%
- 3 Random Seeds: 42, 123, 999
- 2 Lookahead Horizons: $H=1, 2$
- 5 Prediction Budgets: $B \in \{1, 2, 4, 6, 8\}$

Certified metrics are stored in `mohman/point_1/cstp/results/csv/corrected_aggregate_metrics.csv`.
CSTP achieves **82.02% recall** at 20% profiling (+13.90 pp absolute, +20.40% relative over ST-MoE's 68.12%).

---

# 13. Prefetching Experiments and Timing Dynamics

### Latency Hiding Model
When lookahead $H=1$ issues a prefetch during upstream computation ($T_{\text{compute}}$), the transfer duration $T_{\text{transfer}} = 3.901\text{ ms}$ is overlapped with GPU execution. The exposed memory stall is governed by:

$$\text{Stall}_{\text{exposed}} = \max\left(0, T_{\text{transfer}} - T_{\text{compute}}\right)$$

### Late In-Flight Attachment
If a demand request occurs while an expert chunk is already in transit from DDR5:
1. The demand request attaches to the existing DMA transfer queue entry.
2. No duplicate transfer is issued.
3. The stall is reduced from 3.901 ms to the remaining in-flight transfer time.

### Prefetching Invariant Verification
```bash
./LLMSimulator/build/test_ddr5_transfer_engine
./LLMSimulator/build/test_prefetch_metrics
```

---

# 14. HBM Management Experiments (FA-LRU vs. LRU)

Standard LRU suffers from cache pollution when speculative prefetches bring in unneeded expert chunks, evicting resident hot experts.

### Frequency-Aware LRU (FA-LRU) Mechanism
- Maintains an access frequency counter for each expert in a sliding window ($W=64$).
- Frequency decay: Upon epoch advances, historical counts are scaled:
  $$F_{e} \leftarrow \left\lfloor F_{e} \cdot 2^{-\Delta \text{epoch}} \right\rfloor$$
- Eviction policy: Candidate with minimum frequency in the window is selected for eviction; high-frequency entries are shielded.

### Big Experiment Reproduction (W3 & W4, 16 Runs)
```bash
python3 mohman/point_4/big_exp/scripts/plot_big_experiment.py
python3 mohman/point_4/big_exp/scripts/plot_fig9_scaling.py
```

Results show FA-LRU reduces reloads by 4,210 requests and saves 54.24 GB of DDR5 traffic in demand mode.

---

# 15. Runtime, Energy and EDP Experiments

### Energy Modeling Pipeline
The energy pipeline calculates four physical components across 28 configurations:
1. $\mathbf{E_{\text{static\_dram}}} = P_{\text{static}} \times T_{\text{sim}} = 24.0\text{ W} \times T_{\text{sim}}$
2. $\mathbf{E_{\text{ddr5\_active}}} = \text{Bytes}_{\text{total\_ddr5}} \times 28.0\text{ nJ/Byte}$
3. $\mathbf{E_{\text{hbm}}} = \text{Bytes}_{\text{hbm\_access}} \times 56.0\text{ nJ/Byte}$
4. $\mathbf{E_{\text{mac}}} = \text{FLOPs} \times E_{\text{op}}$
5. $\mathbf{\text{EDP}} = E_{\text{total}} \times T_{\text{sim}}$

### Execution and Validation Commands
```bash
# Step 1: Validate microarchitectural invariants across all 28 runs
python3 mohman/point_7/extracted_data/validate_point7_energy_data.py

# Step 2: Regenerate 12 publication-quality energy figures
python3 mohman/point_7/figures/generate_point7_figures.py
```

---

# 16. Complete Experiment Execution Sequence

## 16.1 Fast Verification Suite (< 1 Minute)

To verify the entire repository on a fresh machine in under 60 seconds without running lengthy simulations, run the following fast verification pipeline:

```bash
#!/usr/bin/env bash
set -e

echo "=== [1/8] Running C++ Simulator Unit Tests ==="
./LLMSimulator/build/test_fa_lru
./LLMSimulator/build/test_synthetic_routing_diversity
./LLMSimulator/build/test_ddr5_transfer_engine
./LLMSimulator/build/test_prefetch_metrics

echo "=== [2/8] Generating Point 1 ST-MoE Figures ==="
python3 mohman/point_1/st_moe/scripts/generate_plots.py

echo "=== [3/8] Generating Point 3 Large Workload Plots ==="
python3 mohman/point_3/final_results/scripts/generate_plots.py

echo "=== [4/8] Generating Point 4 FA-LRU Figures ==="
python3 mohman/point_4/big_exp/scripts/plot_big_experiment.py
python3 mohman/point_4/big_exp/scripts/plot_fig9_scaling.py

echo "=== [5/8] Generating Point 5 Comparative Figures ==="
python3 mohman/point_5/figures/plot_point5_figures.py
python3 mohman/point_5/figures/plot_extended_figures.py

echo "=== [6/8] Generating Point 6 Sensitivity Figures ==="
python3 mohman/point_6/final_results/figures/generate_point6_figures.py

echo "=== [7/8] Validating Point 7 Energy Invariants and Generating Figures ==="
python3 mohman/point_7/extracted_data/validate_point7_energy_data.py
python3 mohman/point_7/figures/generate_point7_figures.py

echo "=== [8/8] Validating Final Report 100% Numerical Consistency ==="
python3 mohman/final_report_v2/audit/validate_report_consistency.py

echo "================================================================="
echo "ALL EXPERIMENT VERIFICATIONS AND PLOT GENERATIONS PASSED CLEANLY!"
echo "================================================================="
```

---

## 16.2 Full Simulation Reproduction Sequence

> [!NOTE]
> Re-executing all 73 cycle-level simulations for Point 3 and the 28 full runs for Point 7 models tens of millions of discrete memory events and requires approximately 12–24 hours of CPU compute time. Pre-simulated raw and intermediate result CSVs are checked into git so full re-simulation is optional.

If a full re-simulation campaign is desired:
```bash
# 1. Execute Point 3 large workload campaign
python3 mohman/point_3/phase2/large_workload/scripts/run_large_campaign.py

# 2. Execute Point 4 Big Experiment campaign
python3 mohman/point_4/big_exp/scripts/run_big_experiment.py

# 3. Execute Point 6 sensitivity campaign
python3 mohman/point_6/final_results/execute_point6_campaign.py
```

---

# 17. Result Processing and Figure Generation

The table below summarizes all figure generation scripts, their input data sources, and output figures corresponding to the final report:

| Script Path | Input Data CSV | Output Figures (PNG & PDF) | Corresponding Report Chapter |
|---|---|---|---|
| `mohman/point_1/st_moe/scripts/generate_plots.py` | `mohman/point_1/st_moe/results/csv/` | `recall_comparison`, `frozen_vs_online`, `horizon_comparison`, `hit_rate_comparison`, `cold_vs_warm_recall`, `native_threshold_tradeoff`, `profiling_sensitivity` | Chapter 3 (Figures 3.1–3.5) |
| `mohman/point_3/final_results/scripts/generate_plots.py` | `mohman/point_3/final_results/data/consolidated_csv/latency_and_stall.csv` | `01_total_time_vs_stall_reduction/` (10 plots), `02_prefetch_coverage/`, `03_timeliness/`, `05_hbm_hit_rate/`, `06_memory_traffic/`, `07_exposed_memory_stall/`, `dashboards` | Chapter 5 (Figures 5.1–5.14) |
| `mohman/point_4/big_exp/scripts/plot_big_experiment.py` | `mohman/point_4/big_exp/results/all_runs.csv` | `fig1_hit_rate_comparison` through `fig8_repetition_consistency` | Chapter 6 (Figure 6.1) |
| `mohman/point_4/big_exp/scripts/plot_fig9_scaling.py` | `mohman/point_4/big_exp/results/paired_comparisons.csv` | `fig9_falru_scaling_effect` | Chapter 6 (Figure 6.2) |
| `mohman/point_5/figures/plot_point5_figures.py` | `mohman/point_5/extracted_data/unified_comparison.csv` | `fig1_predictor_effectiveness`, `fig2_prefetching_contribution`, `fig3_falru_contribution`, `fig4_overall_system_comparison` | Chapter 7 (Figures 7.1–7.3) |
| `mohman/point_6/final_results/figures/generate_point6_figures.py` | `mohman/point_6/final_results/consolidated_data/` | `fig1_batch_size_hit_rate` through `fig15_operating_regimes_summary` | Chapter 8 (Figures 8.1–8.6) |
| `mohman/point_7/figures/generate_point7_figures.py` | `mohman/point_7/extracted_data/unified_energy_results.csv` | `fig1_anchor_system_energy` through `fig12_component_pie_breakdown` | Chapter 9 (Figures 9.1–9.3) |
| `mohman/final_report_v2/audit/validate_report_consistency.py` | Authoritative result CSVs across Points 1–7 | `REPORT_CONSISTENCY_VALIDATION.md` (63/63 checks passed) | Global consistency audit |

---

# 18. Mapping of Results to the Final Report

The following master mapping connects every result file directly to its corresponding section, table, and figure in `MoE_Memory_Wall_Final_Report.pdf`:

| Point | Experiment & Context | Key Verified Metrics | Authoritative Result File | Report Section | Report Tables & Figures |
|---|---|---|---|---|---|
| **Point 1** | Predictor Profiling Sensitivity (Top-8, 20% profiling) | CSTP Recall: **82.02%** vs. ST-MoE: **68.12%** (+13.90 pp absolute, +20.40% relative) | `mohman/point_1/cstp/results/csv/corrected_aggregate_metrics.csv` | Section 3.2 | Table 3.1; Figures 3.1, 3.2, 3.3, 3.4, 3.5 |
| **Point 2** | Calibrated DDR5 Transfer Timing (84 MiB chunk @ 22.578 GB/s) | Serialized transfer latency = **3.901 ms**; In-flight attachment verified | `LLMSimulator/build/test_ddr5_transfer_engine` | Section 4.2 | Table 4.1; Transfer timeline derivations |
| **Point 3** | Workload W7 (B=70, P=8192, T=128, 64 GB/s interconnect) | Runtime saving: **6.717 s**; Stall reduction: **6,744.86 ms**; Per-step decode saving: **104.95 ms/step** | `mohman/point_3/final_results/data/consolidated_csv/latency_and_stall.csv` | Section 5.3 | Tables 5.1, 5.2; Figures 5.1–5.14 |
| **Point 4** | FA-LRU vs. LRU on Anchor Workload ($W=64$, Cap=35 GiB) | Hit rate: **74.88%** (FA-LRU) vs. **71.45%** (LRU); Traffic: **508.16 GB** vs. **562.40 GB** (-30 GB) | `mohman/point_4/big_exp/results/paired_comparisons.csv` | Section 6.2 | Table 6.1; Figures 6.1, 6.2 |
| **Point 5** | Four-Way Comparative Architecture (Configs A, B, C, D) | Config D saving: **-12.93 s (-14.08%)** runtime; **-12,927.68 ms (-19.92%)** stall | `mohman/point_5/extracted_data/unified_comparison.csv` | Section 7.2 | Table 7.1; Figures 7.1, 7.2, 7.3 |
| **Point 6** | Sensitivity Sweeps ($H \in \{1,2,4\}$, $K \in \{1,2,4,8\}$) | Hidden stall: **3,568.90 ms** ($H=1$) to **4,005.17 ms** ($H=4$); $K=2$ sweet spot | `mohman/point_6/final_results/consolidated_data/unified_point6_results.csv` | Section 8.2 | Tables 8.1, 8.2; Figures 8.1–8.6 |
| **Point 7** | Subsystem Energy & EDP Optimization (28 runs) | $K=2$ saves **-10.93 s** runtime, **-27.68 J** net energy; EDP = **7,079.32 kJ·s** (**-1.72%**) | `mohman/point_7/extracted_data/unified_energy_results.csv` | Section 9.2 | Tables 9.1, 9.2; Figures 9.1, 9.2, 9.3 |

---

# 19. Reproducibility Verification Checklist

Use this checklist during TA evaluation:

- [ ] **System Dependencies:** `build-essential`, `cmake`, `python3-venv`, `libyaml-cpp-dev` installed.
- [ ] **Python Packages:** `numpy`, `pandas`, `matplotlib`, `pyarrow`, `fastparquet`, `python-docx` installed.
- [ ] **Simulator Compilation:** `cd LLMSimulator && mkdir -p build && cd build && cmake .. && make -j$(nproc)` succeeds with 100% build completion and 0 errors.
- [ ] **Harness Verification:** `./LLMSimulator/build/test_fa_lru` reports 14/14 tests PASSED.
- [ ] **Diversity Verification:** `./LLMSimulator/build/test_synthetic_routing_diversity` reports 8/8 tests PASSED.
- [ ] **Transfer Calibration:** `./LLMSimulator/build/test_ddr5_transfer_engine` reports 12/12 tests PASSED.
- [ ] **Metric Harness:** `./LLMSimulator/build/test_prefetch_metrics` reports 15/15 tests PASSED.
- [ ] **Trace Availability:** `LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet` is present (16.2 MB).
- [ ] **Point 1 Figures:** `python3 mohman/point_1/st_moe/scripts/generate_plots.py` generates 7 PNG/PDF figures.
- [ ] **Point 3 Plots:** `python3 mohman/point_3/final_results/scripts/generate_plots.py` generates all workload plots.
- [ ] **Point 4 Figures:** `python3 mohman/point_4/big_exp/scripts/plot_big_experiment.py` generates 8 figures.
- [ ] **Point 5 Figures:** `python3 mohman/point_5/figures/plot_point5_figures.py` generates 4 figures.
- [ ] **Point 6 Figures:** `python3 mohman/point_6/final_results/figures/generate_point6_figures.py` generates 15 figures.
- [ ] **Point 7 Energy:** `python3 mohman/point_7/extracted_data/validate_point7_energy_data.py` passes all 7 invariant gates.
- [ ] **Point 7 Figures:** `python3 mohman/point_7/figures/generate_point7_figures.py` generates 12 figures.
- [ ] **Report Consistency:** `python3 mohman/final_report_v2/audit/validate_report_consistency.py` passes 8/8 test gates (63/63 checks).

---

# 20. Troubleshooting Guide

### Issue 1: Script cannot locate `/home/mohman/ca_project`
- **Symptom:** `FileNotFoundError: [Errno 2] No such file or directory: '/home/mohman/ca_project/...'`
- **Root Cause:** A python script contains the default path used by the author during development.
- **Resolution:** Create a symlink from your repo root to `/home/mohman/ca_project`:
  ```bash
  sudo mkdir -p /home/mohman
  sudo ln -s "$(pwd)" /home/mohman/ca_project
  ```
  Or export the path and run from the root:
  ```bash
  export REPO_ROOT="$(pwd)"
  ```

### Issue 2: `libyaml-cpp.a` or YAML build error during CMake
- **Symptom:** `fatal error: yaml-cpp/yaml.h: No such file or directory`
- **Root Cause:** Missing `libyaml-cpp-dev` system header package.
- **Resolution:**
  ```bash
  sudo apt install -y libyaml-cpp-dev
  ```

### Issue 3: Python `ModuleNotFoundError: No module named 'pandas'` (or `matplotlib`, `pyarrow`)
- **Symptom:** `ModuleNotFoundError: No module named 'pandas'`
- **Root Cause:** Virtual environment not activated.
- **Resolution:**
  ```bash
  source .venv/bin/activate
  pip install -r requirements.txt || pip install numpy pandas matplotlib pyarrow fastparquet python-docx
  ```

### Issue 4: CMake cache corruption after branch switching
- **Symptom:** CMake reports compiler change or stale cache variables.
- **Resolution:**
  ```bash
  rm -rf LLMSimulator/build
  mkdir -p LLMSimulator/build
  cd LLMSimulator/build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make -j"$(nproc)"
  ```

### Issue 5: Simulation execution appears slow
- **Explanation:** Discrete-event C++ simulation of DeepSeek-V3 evaluates tens of millions of memory access cycles. For quick grading and evaluation, **use the fast verification suite (Section 16.1)** which operates directly on certified intermediate CSVs and completes in < 60 seconds.

---

# 21. Expected Output Directory Structure

When all figure generation scripts are executed, the resulting artifact tree will look as follows:

```text
mohman/
├── point_1/st_moe/plots/
│   ├── recall_comparison.png / .pdf
│   ├── frozen_vs_online.png / .pdf
│   ├── horizon_comparison.png / .pdf
│   ├── hit_rate_comparison.png / .pdf
│   ├── cold_vs_warm_recall.png / .pdf
│   ├── native_threshold_tradeoff.png / .pdf
│   └── profiling_sensitivity.png / .pdf
├── point_3/final_results/plots/
│   ├── 01_total_time_vs_stall_reduction/ (fig_01a_w1 to w7, fig_01b, fig_01c, fig_01d)
│   ├── 02_prefetch_coverage/
│   ├── 03_timeliness/
│   ├── 04_unused_prefetches/
│   ├── 05_hbm_hit_rate/
│   ├── 06_memory_traffic/
│   ├── 07_exposed_memory_stall/
│   ├── 08_expert_load_latency/
│   ├── 09_end_to_end_latency/
│   ├── 10_throughput/
│   └── comparison_dashboards/
├── point_4/big_exp/figures/
│   ├── fig1_hit_rate_comparison.png / .pdf
│   ├── fig2_expert_reloads_comparison.png / .pdf
│   ├── fig3_ddr5_traffic_comparison.png / .pdf
│   ├── fig4_exposed_stall_comparison.png / .pdf
│   ├── fig5_token_latency_comparison.png / .pdf
│   ├── fig6_cstp_cache_pollution.png / .pdf
│   ├── fig7_falru_diagnostics.png / .pdf
│   ├── fig8_repetition_consistency.png / .pdf
│   └── fig9_falru_scaling_effect.png / .pdf
├── point_5/figures/
│   ├── fig1_predictor_effectiveness.png / .pdf
│   ├── fig2_prefetching_contribution.png / .pdf
│   ├── fig3_falru_contribution.png / .pdf
│   └── fig4_overall_system_comparison.png / .pdf
├── point_6/final_results/figures/
│   ├── batch_size/ (fig1 to fig3)
│   ├── context_length/ (fig4 to fig6)
│   ├── lookahead/ (fig7 to fig9)
│   ├── prefetch_budget/ (fig10 to fig12)
│   └── combined/ (fig13 to fig15)
└── point_7/figures/
    ├── fig1_anchor_system_energy.png / .pdf
    ├── fig2_anchor_energy_per_token.png / .pdf
    ├── fig3_prefetch_budget_ddr5_breakdown.png / .pdf
    ├── fig4_ddr5_traffic_vs_energy.png / .pdf
    ├── fig5_stall_reduction_vs_energy_overhead.png / .pdf
    ├── fig6_energy_delay_product.png / .pdf
    ├── fig7_batch_size_energy_scaling.png / .pdf
    ├── fig8_budget_sweet_spot.png / .pdf
    ├── fig9_lookahead_energy_dynamics.png / .pdf
    ├── fig10_context_length_energy_scaling.png / .pdf
    ├── fig11_energy_performance_pareto.png / .pdf
    └── fig12_component_pie_breakdown.png / .pdf
```

---

# 22. Important Notes, Limitations and Scientific Boundaries

1. **Simulated Time vs. Wall-Clock Execution Time:** All performance reductions (e.g., W7 runtime reduction of 6.717 s) represent **simulated GPU model execution time**, not host PC wall-clock execution time.
2. **Bandwidth Saturation Operating Boundary:** While prefetching with $K=2$ achieves substantial latency and energy reductions, aggressive speculative prefetching ($K \ge 8$) saturates the DDR5 interconnect, increasing traffic by +142.1 GB and degrading EDP by +2.91%. Prefetch budgets must be tuned to available bus bandwidth.
3. **KV Cache Footprint Scaling:** As prompt context length grows ($P \ge 4096$), attention KV-cache memory expands, reducing the effective HBM capacity available for expert caching from 35 GiB to $< 20\text{ GiB}$. Under these long-context conditions, FA-LRU cache protection becomes even more crucial.
4. **Strict Zero-Ternary Policy:** All newly authored Python processing scripts enforce `ast.IfExp == 0` (zero ternary operators) to ensure code readability and prevent subtle syntax bugs during academic audits.
5. **Authoritative Submission Package:** The final submission package is consolidated in `mohman/submission_final/`, containing the verified Word document (`MoE_Memory_Wall_Final_Report.docx`), the 56-page PDF (`MoE_Memory_Wall_Final_Report.pdf`), 32 figure data CSVs, the numerical reconciliation matrix, and the submission zip archive.

---
*Group 13 — Computer Architecture End-Semester Project (November 2026)*  
*Mohit Gupta (2024354), Mrityunjai Poddar (2024358), Saksham Madan (2024493)*
