# Research File Index: Navigable Inventory of Code, Data & Artifacts

**Document:** `docs/RESEARCH_FILE_INDEX.md`  
**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Repository:** `LLMSimulator` (`/home/mohman/ca_project/LLMSimulator`)  
**Scope:** Complete Map of Python/C++ Source, Datasets, Result Artifacts & Documentation  
**Auditor / Author:** ML Systems Research & Reproducibility Team  

---

## 1. Executive Repository Layout

```
/home/mohman/ca_project/LLMSimulator/
├── data/
│   └── downloads/
│       ├── moe_anywhere/          # Real production MoE trace (e1-sage-a1.5.parquet)
│       └── moe_suite/             # Multi-corpus coding benchmark Parquet files
├── docs/
│   ├── phase_d/                   # Detailed Phase D architectural & ablation reports
│   ├── RESEARCH_MASTER_SUMMARY.md # Master handover research summary
│   ├── PYTHON_PIPELINE_EXPLAINED.md # Deep-dive into Python ingestion & prediction
│   ├── RESULTS_AND_DATASET_INVENTORY.md # Audit of datasets and benchmark results
│   ├── NEXT_IMPLEMENTATION_ROADMAP.md # Engineering plan for Phase E (E0–E6)
│   └── RESEARCH_FILE_INDEX.md     # This comprehensive file index
├── reproduction_A100/             # A100 cluster reproduction traces & logs
│   └── results/a100_duplex/       # Duplex A100 trace & logs (negative control)
├── results/
│   └── endsem_phase_d/            # Official Phase D experimental results
│       ├── aggregate/             # Cross-workload summary CSVs
│       ├── per_dataset/           # Workload-specific evaluation records
│       ├── latency/               # Microsecond query & update latency CSVs
│       ├── memory/                # Tracemalloc heap allocation CSVs
│       ├── distribution_shift/    # Cluster A -> B transition CSVs
│       ├── ablations/             # Normalization & hyperparameter sweep CSVs
│       ├── figures/               # 22 publication-grade figures (PNG)
│       └── audit/                 # Critical routing selection audit CSVs
├── scripts/
│   ├── generate_correlated_benchmark_trace.py # Synthetic benchmark generator
│   └── phase_d/                   # Experiment runners, benchmarkers & figure generators
├── src/
│   ├── predictor/                 # Core prediction algorithms (ACSH, CSTP, ST-MoE)
│   │   └── adapters/              # Dataset loaders implementing base_adapter.py
│   ├── scheduler/                 # Request scheduling & sequence management (C++)
│   ├── module/                    # Module graph & route execution (C++)
│   ├── hardware/                  # Device, cluster & layer execution timing (C++)
│   └── dram/                      # Ramulator2 DDR5/HBM memory interface (C++)
└── tests/                         # Pytest test suite (62 unit tests)
```

---

## 2. Python Source Code Inventory

### 2.1 Core Predictor Algorithms (`src/predictor/`)

| File Path | Primary Classes / Functions | Responsibility |
| :--- | :--- | :--- |
| [`src/predictor/base_predictor.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/base_predictor.py) | `BasePredictor` | Common abstract interface defining `predict_next_stage()` and `update()`. |
| [`src/predictor/cstp_predictor.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py) | `CSTPPredictor` | Correlation-based Spatio-Temporal Predictor using Dirichlet-smoothed count matrices. |
| [`src/predictor/st_moe_paper.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/st_moe_paper.py) | `STMoEPaperPredictor` | Paper-faithful reproduction of Zhao et al. (CCT candidate sets, 2-bit counters). |
| [`src/predictor/acsh_predictor.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/acsh_predictor.py) | `ACSHPredictor` | Adaptive Confidence and Statistical Hybrid (Softmax fusion, EMA reliability, Fast Recovery). |
| [`src/predictor/trace_parser.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/trace_parser.py) | `RoutingTraceParser` | Parses C++ CSV routing traces into normalized records. |

### 2.2 Dataset Adapters (`src/predictor/adapters/`)

| File Path | Class Name | Input Format | Workload Ingested |
| :--- | :--- | :--- | :--- |
| [`src/predictor/adapters/base_adapter.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/adapters/base_adapter.py) | `RoutingDatasetAdapter` | Abstract base class | Standard `PredictionRecord` generator contract. |
| [`src/predictor/adapters/simulator_adapter.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/adapters/simulator_adapter.py) | `SimulatorDatasetAdapter` | CSV | `duplex_a100` (Negative control trace). |
| [`src/predictor/adapters/synthetic_adapter.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/adapters/synthetic_adapter.py) | `SyntheticCorrelatedDatasetAdapter` | CSV | `correlated_synthetic` (Controlled benchmark). |
| [`src/predictor/adapters/moe_anywhere_adapter.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/adapters/moe_anywhere_adapter.py) | `MoeAnywhereDatasetAdapter` | Parquet | `moe_anywhere` (InternScience Sage-A1.5 production trace). |
| [`src/predictor/adapters/moe_suite_adapter.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/adapters/moe_suite_adapter.py) | `MoeSuiteDatasetAdapter` | Parquet | `moe_suite` (MistrJirka coding benchmark). |

### 2.3 Experiment Scripts (`scripts/phase_d/`)

| Script Path | Purpose | Key Artifacts Output |
| :--- | :--- | :--- |
| [`scripts/phase_d/run_phase_d.py`](file:///home/mohman/ca_project/LLMSimulator/scripts/phase_d/run_phase_d.py) | Master experiment runner across all datasets and predictors. | `results/endsem_phase_d/per_dataset/*/per_prediction_results.csv` |
| [`scripts/phase_d/benchmark_predictors_d.py`](file:///home/mohman/ca_project/LLMSimulator/scripts/phase_d/benchmark_predictors_d.py) | Microsecond latency benchmarking and heap memory profiling. | `results/endsem_phase_d/latency/*.csv`, `memory/*.csv` |
| [`scripts/phase_d/run_distribution_shift_d.py`](file:///home/mohman/ca_project/LLMSimulator/scripts/phase_d/run_distribution_shift_d.py) | Evaluates abrupt context switch at token 50. | `results/endsem_phase_d/distribution_shift/*.csv` |
| [`scripts/phase_d/run_ablations_d.py`](file:///home/mohman/ca_project/LLMSimulator/scripts/phase_d/run_ablations_d.py) | Sweeps normalization strategies, temperatures, $\beta$, and $\tau$. | `results/endsem_phase_d/ablations/*.csv` |
| [`scripts/phase_d/aggregate_phase_d.py`](file:///home/mohman/ca_project/LLMSimulator/scripts/phase_d/aggregate_phase_d.py) | Aggregates per-dataset metrics into master tables. | `results/endsem_phase_d/aggregate/*.csv` |
| [`scripts/phase_d/generate_figures_d.py`](file:///home/mohman/ca_project/LLMSimulator/scripts/phase_d/generate_figures_d.py) | Renders 22 publication-grade figures. | `results/endsem_phase_d/figures/fig*.png` |
| [`scripts/phase_d/audit_routing_mechanisms.py`](file:///home/mohman/ca_project/LLMSimulator/scripts/phase_d/audit_routing_mechanisms.py) | Evaluates C++ PRNG randomness and dataset Jaccard overlaps. | `results/endsem_phase_d/audit/routing_selection_audit.csv` |

---

## 3. C++ Simulator Subsystems

| Subsystem | Key Files | Responsibility |
| :--- | :--- | :--- |
| **Scheduler** | [`src/scheduler/scheduler.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp), [`src/scheduler/sequence.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp) | Implements `getRandomExpert(top_k)` PRNG generator, `update_expert`, and sequence scheduling. |
| **Module Graph** | [`src/module/route.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/module/route.cpp), [`src/module/expert.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/module/expert.cpp) | Models activation routing, token aggregation, and MoE compute module execution. |
| **Hardware Timing** | [`src/hardware/layer_impl.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/layer_impl.cpp) | Computes layer execution time; contains execution cache bypass toggle. |
| **DRAM Interface** | [`src/dram/dram_interface.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/dram/dram_interface.cpp) | Dispatches memory transactions to Ramulator2 (currently blocking). |
| **Execution Harness** | [`eval/test_deepseekv3_mixtral_style.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp) | Program entry point for simulated MoE decoding loops. |

---

## 4. Dataset Registry

| Workload ID | Local File Path | Size | Schema / Records |
| :--- | :--- | :--- | :--- |
| **`moe_anywhere`** | `data/downloads/moe_anywhere/e1-sage-a1.5.parquet` | ~45 MB | Parquet table with 760,720 rows ($K=8$, $S=40$, 74 prompts). |
| **`moe_suite`** | `data/downloads/moe_suite/corpus-*.parquet` (12 files) | ~120 MB | Parquet tables storing $(N, 75, 8)$ route selection matrices. |
| **`correlated_synthetic`** | `results/endsem_phase1/benchmark_correlated_trace.csv` | 1.2 MB | CSV with 4,560 rows ($E=256, K=8, S=57, N=16$). |
| **`duplex_a100`** | `reproduction_A100/results/a100_duplex/traces/expert_routing.csv` | 145 KB | CSV with 5,120 rows ($E=8, K=2, S=32$). |

---

## 5. Results & Benchmark Inventory (`results/endsem_phase_d/`)

| Relative File Path | Contents & Generating Script |
| :--- | :--- |
| `aggregate/all_datasets_aggregate.csv` | Master evaluation metrics across all predictors and retained workloads (`aggregate_phase_d.py`). |
| `aggregate/all_datasets_budget_sweep.csv` | Parametric sweep of prefetch budgets $B \in [2, 64]$ (`aggregate_phase_d.py`). |
| `aggregate/online_learning_gains.csv` | Measured delta between online adaptation and frozen profiling (`aggregate_phase_d.py`). |
| `latency/prediction_latency.csv` | Host CPU microsecond query latency across budgets (`benchmark_predictors_d.py`). |
| `latency/update_latency.csv` | Host CPU microsecond online state update latency (`benchmark_predictors_d.py`). |
| `memory/memory_footprint.csv` | Heap memory allocation and peak bytes via `tracemalloc` (`benchmark_predictors_d.py`). |
| `distribution_shift/distribution_shift_recovery.csv` | Drop magnitude and token recovery metrics for Cluster A $\to$ B shift (`run_distribution_shift_d.py`). |
| `ablations/calibration_ablation.csv` | Evaluation of Softmax, Max, Rank, and Raw normalization (`run_ablations_d.py`). |
| `ablations/beta_sensitivity.csv` | Sensitivity curve for EMA smoothing factor $\beta \in [0.01, 0.5]$ (`run_ablations_d.py`). |
| `ablations/tau_sensitivity.csv` | Sensitivity curve for fusion temperature $\tau \in [0.1, 2.0]$ (`run_ablations_d.py`). |
| `audit/routing_selection_audit.csv` | Complete provenance and classification matrix for all workloads (`audit_routing_mechanisms.py`). |
| `figures/fig1_overall_recall_comparison.png` through `fig22_concurrency_memory_scaling.png` | 22 publication-grade figures illustrating all experimental dimensions (`generate_figures_d.py`). |

---

## 6. Documentation Registry

| Document Path | Purpose & Topic |
| :--- | :--- |
| [`docs/RESEARCH_MASTER_SUMMARY.md`](file:///home/mohman/ca_project/LLMSimulator/docs/RESEARCH_MASTER_SUMMARY.md) | Executive summary of research problem, proposed architecture, datasets, and results. |
| [`docs/PYTHON_PIPELINE_EXPLAINED.md`](file:///home/mohman/ca_project/LLMSimulator/docs/PYTHON_PIPELINE_EXPLAINED.md) | Technical guide to dataset adapters, statistical profiling, and inference evaluation. |
| [`docs/RESULTS_AND_DATASET_INVENTORY.md`](file:///home/mohman/ca_project/LLMSimulator/docs/RESULTS_AND_DATASET_INVENTORY.md) | Comprehensive audit of all datasets, splits, macro metrics, and ablations. |
| [`docs/NEXT_IMPLEMENTATION_ROADMAP.md`](file:///home/mohman/ca_project/LLMSimulator/docs/NEXT_IMPLEMENTATION_ROADMAP.md) | Detailed engineering roadmap for Phase E hardware integration (E0–E6). |
| [`docs/RESEARCH_FILE_INDEX.md`](file:///home/mohman/ca_project/LLMSimulator/docs/RESEARCH_FILE_INDEX.md) | This navigable file index. |
| [`docs/phase_d/ROUTING_SELECTION_AUDIT.md`](file:///home/mohman/ca_project/LLMSimulator/docs/phase_d/ROUTING_SELECTION_AUDIT.md) | In-depth audit report distinguishing C++ PRNG random routing from real model routing. |
