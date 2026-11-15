# Master Source Code, Data Provenance, and Section Mapping Audit

## Project: MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference
## Course: CSE 511/ECE 511: Computer Architecture (End Semester Project Report)
## Authors: Mohit Gupta (2024354), Mrityunjai Poddar (2024358), Saksham Madan (2024493)
## Repository: https://github.com/Mohit14032007/ca_project_endsem

---

### 1. Architectural Section & Source Code Mapping

| Report Section | Primary Component | Source Code Implementation | Key Functions & Data Structures | Verification Status |
|:---|:---|:---|:---|:---|
| Chapter 3 (Point 1) | CSTP Predictor | `LLMSimulator/src/predictor/cstp_predictor.cpp` | `predict_top_k()`, `update_correlation_matrix()`, `compute_temporal_decay()` | Verified Source & Tests |
| Chapter 3 (Point 1) | ST-MoE Baseline | `LLMSimulator/src/predictor/st_moe_predictor.cpp` | `predict()`, `update_history()`, `confidence_counter` | Verified Source & Tests |
| Chapter 3 (Point 1) | Predictor Registry | `LLMSimulator/src/predictor/predictor_registry.cpp` | `PredictorFactory::create()`, `register_predictor()` | Verified Source |
| Chapter 4 (Point 2) | LivePredictionEngine | `LLMSimulator/src/prefetch/live_prediction_engine.cpp` | `on_layer_routing_event()`, `dispatch_prefetches()`, `prune_candidates()` | Verified Source |
| Chapter 4 (Point 2) | PrefetchCandidateQueue | `LLMSimulator/src/prefetch/prefetch_candidate_queue.cpp` | `push_candidate()`, `pop_highest_priority()`, `deduplicate()` | Verified Source |
| Chapter 4 (Point 2) | DDR5TransferEngine | `LLMSimulator/src/transfer/ddr5_transfer_engine.cpp` | `submit_transfer()`, `advance_time()`, `check_completion()` | Verified Source |
| Chapter 4 (Point 2) | LinearExecutionGPU | `LLMSimulator/src/gpu/linear_execution_gpu.cpp` | `execute_expert_gemm()`, `reconcile_demand()`, `record_stall()` | Verified Source |
| Chapter 4 (Point 2) | Causal Hook Integration | `LLMSimulator/src/simulator/sequence_scheduler.cpp` | `step_sequence()`, `pre_layer_hook()`, `post_layer_hook()` | Verified Source |
| Chapter 6 (Point 4) | HBM Cache Management | `LLMSimulator/src/cache/hbm_expert_cache.cpp` | `evict_victim()`, `insert_expert()`, `touch_expert()` | Verified Source |
| Chapter 6 (Point 4) | FA-LRU Eviction Policy | `LLMSimulator/src/cache/fa_lru_policy.cpp` | `select_victim()`, `calculate_score()`, `speculative_penalty()` | Verified Source |
| Chapter 9 (Point 7) | Energy Model Accounting | `LLMSimulator/src/energy/energy_model.cpp` | `compute_total_energy()`, `calculate_edp()`, `dram_static_energy()` | Verified Source |

---

### 2. Experimental Data Provenance and Raw Telemetry Files

| Point / Phase | Raw CSV / Parquet Path | Generation / Plot Script | Workload / Sweep Dimension | Telemetry Counters Recorded |
|:---|:---|:---|:---|:---|
| Point 1 (CSTP) | `mohman/point_1/cstp/results/cstp_profiling_frozen_sweep.csv` | `plot_frozen_recall.py` | Profiling budget (10%-50%), H=1,2,4 | Frozen recall, Top-K recall, hit rate |
| Point 1 (ST-MoE) | `mohman/point_1/st_moe/results/cold_warm_layer_recall.csv` | `plot_cold_warm.py` | Cold-start vs warm across layers | Layer-wise recall, confidence threshold |
| Point 1 (Comparison)| `LLMSimulator/results/endsem_phase_c/aggregated_metrics.csv` | `plot_phase_c_comparison.py` | Comprehensive MoE-Anywhere suite | Precision, recall, hit rate, candidate size |
| Point 3 (Effectiveness) | `mohman/point_3/final_results/csv/01_execution_time_stall_breakdown.csv` | `plot_01_time_vs_stall.py` | Workloads W1 to W7 (B=1 to 128) | Wall time, compute time, exposed stall time |
| Point 3 (Coverage) | `mohman/point_3/final_results/csv/02_prefetch_coverage_summary.csv` | `plot_02_coverage.py` | Batch scaling & sequence length | Timely coverage %, late attachment %, miss % |
| Point 3 (Unused & Wasted)| `mohman/point_3/final_results/csv/04_unused_prefetches.csv` | `plot_04_unused.py` | Workloads W1 to W7 | Unused prefetches, wasted bytes (GiB) |
| Point 3 (Traffic) | `mohman/point_3/final_results/csv/06_memory_traffic.csv` | `plot_06_traffic.py` | Workloads W1 to W7 | Demand traffic, speculative traffic (GiB) |
| Point 3 (Stall) | `mohman/point_3/final_results/csv/07_stall_reduction.csv` | `plot_07_stall.py` | Workloads W1 to W7 | Stall reduction %, stall cycles hidden |
| Point 3 (Latencies) | `mohman/point_3/final_results/csv/08_expert_load_latency.csv` | `plot_08_load_latency.py` | Workloads W1 to W7 | Expert load latency (ms), decode step latency |
| Point 4 (FA-LRU) | `mohman/point_4/big_exp/results/falru_scaling_data.csv` | `plot_falru_scaling.py` | W3 & W4 (C_HBM=40 GiB) | Cache hit %, victim evictions, speculative drop |
| Point 5 (Comparative)| `mohman/point_5/results/comparative_four_way_matrix.csv` | `plot_incremental_contribution.py` | Configs A, B, C, D (W3) | Total time, exposed stall, DDR5 traffic, speedup % |
| Point 6 (Batch Size) | `mohman/point_6/final_results/csv/batch_size_sweep.csv` | `plot_batch_size.py` | Batch size B in [1, 32] | Exposed stall, token latency, speedup |
| Point 6 (Context) | `mohman/point_6/final_results/csv/context_length_sweep.csv` | `plot_context_length.py` | Context P in [13, 4096] | KV cache memory, attention execution time |
| Point 6 (Lookahead) | `mohman/point_6/final_results/csv/lookahead_sweep.csv` | `plot_lookahead.py` | Lookahead H in [1, 4] | Timely coverage %, accuracy decay, traffic |
| Point 6 (Budget) | `mohman/point_6/final_results/csv/prefetch_budget_sweep.csv` | `plot_prefetch_budget.py` | Budget K in [1, 8] | Speculative traffic, stall reduction, U-curve |
| Point 7 (Component) | `mohman/point_7/tables/csv/energy_breakdown_by_component.csv` | `plot_energy_breakdown.py` | Subsystem components | Static DRAM (J), HBM (J), DDR5 (J), Compute (J) |
| Point 7 (28-Run Matrix)| `mohman/point_7/tables/csv/reconciled_energy_matrix.csv` | `plot_pareto_frontier.py` | 28 controlled simulations | Energy (kJ), EDP (kJ·s), delay (s), savings % |

---

### 3. Verification of Physical Figures and Integrity

All 32 figures listed in `REPORT_FIGURE_CHECKLIST.md` were independently inspected for file existence, image integrity, and axis label legibility. Both high-resolution raster (.png) and scalable vector (.pdf) formats are preserved across all phase subdirectories.

