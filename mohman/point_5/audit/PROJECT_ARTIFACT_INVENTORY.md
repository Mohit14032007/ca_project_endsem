# Project Artifact Inventory: Point 1 to Point 4

## 1. Executive Summary

This inventory catalogs all relevant research, simulation, implementation, and experimental artifacts across the **MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference** project.

The inventory covers:
1. **Point 1: Expert Prediction** (Offline predictor modeling, head-to-head comparisons of ST-MoE vs. CSTP on the Sage-A1.5 MoE-Anywhere trace).
2. **Point 2: Simulator Integration & Metrics Foundation** (Ramulator-2/DDR5 event-driven transfer engine calibration, prefetch metrics instrumentation, and trace pipeline validation).
3. **Point 3: Prefetching Effectiveness** (Large-workload 27-run evaluation of asynchronous prefetching under LRU comparing Baseline, ST-MoE, and CSTP across batches $B \in [14, 70]$ and horizons $h \in [1, 2]$).
4. **Point 4: HBM Management & Eviction** (FA-LRU implementation, smoke validation, Phase 2.5/2.6 accounting correction, Phase 3 focused 24-run capacity study, and Big Experiment 16-run scaling evaluation).
5. **Historical In-Situ Evaluations** (`phase_4d5c` 9-configuration matrix providing authentic in-simulation prediction-only shadow-mode data).

All artifacts are maintained with complete traceability to exact file paths, commit states, and raw logs.

---

## 2. Point 1 Artifact Inventory: Expert Prediction

- **Primary Directory**: `/home/mohman/ca_project/mohman/point_1/`
- **Objective**: Design, implement, and evaluate predictive routing mechanisms (ST-MoE vs. CSTP) for next-token expert demands.
- **Trace Dataset**: MoE-Anywhere `e1-sage-a1.5.parquet` (74 prompts $\times$ 257 contiguous token steps, DeepSeek-V3 routing style with Top-$K=8$ out of 256 routed experts + 1 shared expert).

### Key Files and Datasets

| Artifact Path | Artifact Type | Description | Key Contents / Parameters |
| :--- | :--- | :--- | :--- |
| `mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv` | CSV Dataset | Aggregated prediction metrics across models, regimes, and horizons | Macro/micro precision, recall, F1, hit rate, cold-start recall for ST-MoE and CSTP |
| `mohman/point_1/stmoe_cstp/results/csv/corrected_st_moe_head_to_head.csv` | CSV Dataset | Paired head-to-head comparison of CSTP vs. ST-MoE | 14 test units, 16,800 opportunities per test condition across seeds 42, 123, 999 |
| `mohman/point_1/stmoe_cstp/results/csv/corrected_statistical_analysis.csv` | CSV Dataset | Formal statistical tests of prediction delta | Paired t-tests, Wilcoxon signed-rank tests, 95% confidence intervals |
| `mohman/point_1/stmoe_cstp/COMPARATIVE_INTEGRITY_REPORT.md` | Markdown Report | Comprehensive validation report for predictor evaluation | Profiling percentage sensitivities, online gain quantification, leakage prevention proofs |
| `mohman/point_1/cstp/src/cstp_predictor.py` | Source Code | Standalone CSTP predictor implementation | Cross-sequence history table, frequency tracking, confidence scoring |
| `mohman/point_1/st_moe/ST_MOE_METHODOLOGY.md` | Markdown Report | Mathematical formulation of ST-MoE predictor | Prior-token transition matrix, online discount decay, top-K selection |

---

## 3. Point 2 Artifact Inventory: Memory Engine & Metrics

- **Primary Directory**: `/home/mohman/ca_project/mohman/point_2/`
- **Objective**: Establish cycle-accurate DDR5 transfer latency modeling, integrate event-driven queuing, and instrument prefetch effectiveness metrics.

### Key Files and Datasets

| Artifact Path | Artifact Type | Description | Key Contents / Parameters |
| :--- | :--- | :--- | :--- |
| `mohman/point_2/PHASE4A_CONFIGURATION_AND_RAMULATOR_RUNTIME_AUDIT.md` | Markdown Audit | DDR5 bus arbitration and Ramulator runtime audit | Event-driven DDR5 calibration at 64.0 GB/s, 88,080,384 bytes/expert footprint |
| `mohman/point_2/PREFETCH_METRICS_IMPLEMENTATION_REPORT.md` | Markdown Report | Specification of prefetch effectiveness counters | Strict timely hits, late attachments, unused prefetches, exposed GPU memory stall |
| `mohman/point_2/PREFETCH_METRICS_VALIDATION_REPORT.md` | Markdown Report | Unit test verification of metric accounting | Proves conservation: total prefetches = timely + late + unused completed + discarded |
| `LLMSimulator/eval/test_prefetch_metrics.cpp` | C++ Test Suite | 15 unit tests validating prefetch counters | Pass rate: 15/15 tests passing cleanly |

---

## 4. Point 3 Artifact Inventory: Prefetching Effectiveness

- **Primary Directory**: `/home/mohman/ca_project/mohman/point_3/`
- **Objective**: Evaluate prefetching under standard LRU across varying batch sizes, horizons, and sequence lengths.

### Key Files and Datasets

| Artifact Path | Artifact Type | Description | Key Contents / Parameters |
| :--- | :--- | :--- | :--- |
| `mohman/point_3/phase2/large_workload/results/per_run_metrics.csv` | CSV Dataset | Primary 27-run evaluation dataset | 27 simulation runs across 7 workloads (W1–W7), batch sizes 14 to 70, tokens 4 to 64 |
| `mohman/point_3/phase2/large_workload/results/predictor_comparison.csv` | CSV Dataset | Direct comparison table of Baseline vs. ST-MoE vs. CSTP | HBM hit rate, DDR5 traffic overhead, stall reduction, mean token latency |
| `mohman/point_3/phase2/large_workload/LARGE_WORKLOAD_PREFETCH_EFFECTIVENESS_REPORT.md` | Markdown Report | Primary evaluation report for Point 3 | Coverage analysis, prefetch timeliness, DDR5 traffic inflation, exposed memory stall |
| `mohman/point_3/point_3_explanation.md` | Markdown Report | Academic consolidation of Point 3 findings | Complete multi-workload synthesis, hardware transfer analysis, and prefetch limits |
| `mohman/point_3/final_results/` | Consolidated Folder | Self-contained reproducibility bundle for Point 3 | Contains configs, scripts, CSV data, and vector plots |

---

## 5. Point 4 Artifact Inventory: HBM Management (FA-LRU)

- **Primary Directory**: `/home/mohman/ca_project/mohman/point_4/`
- **Objective**: Design, implement, correct, and evaluate Frequency-Aware LRU (FA-LRU) against Standard LRU.

### Key Files and Datasets

| Artifact Path | Artifact Type | Description | Key Contents / Parameters |
| :--- | :--- | :--- | :--- |
| `LLMSimulator/src/hardware/device.cpp` & `device.h` | Simulator Source | FA-LRU implementation in C++ | Bottom-$W$ candidate scan, lazy epoch discounting, demand frequency updates, C.4 diagnostics |
| `mohman/point_4/phase_2_6/ACCOUNTING_CORRECTION_REPORT.md` | Markdown Report | Demand-accounting correction audit | Solves token-vs-coalesced demand discrepancy: $D_{\text{tensor}} = 3 \times D_{\text{routed}}$ |
| `mohman/point_4/phase_3_focused_24/consolidated/all_24_runs.csv` | CSV Dataset | 24-run capacity & workload campaign | 4 configurations $\times$ 5 capacities (10, 15, 20, 25, 35 GiB) $\times$ 4 workloads ($T=20$) |
| `mohman/point_4/phase_3_focused_24/reports/FINAL_POINT_4_REPORT.md` | Markdown Report | Phase 3 comprehensive analysis | Invariance proofs for pure demand workloads, PABI comparisons, capacity thresholds |
| `mohman/point_4/big_exp/results/all_runs.csv` | CSV Dataset | 16-run large-workload campaign ($T=128$) | Workloads W3 ($B=14$) and W4 ($B=32$), 4 configs $\times$ 2 bit-exact repetitions |
| `mohman/point_4/big_exp/results/paired_comparisons.csv` | CSV Dataset | Paired delta analysis for Big Experiment | Direct deltas for C1 vs C2 (No prefetch) and C3 vs C4 (CSTP enabled) |
| `mohman/point_4/big_exp/reports/FINAL_BIG_EXPERIMENT_REPORT.md` | Markdown Report | Final evaluation report for Big Experiment | Proof of FA-LRU speculative pollution mitigation (+15 hits, -15 reloads, -2.03 GB traffic) |
| `mohman/point_4/big_exp/reports/FA_LRU_SCALING_ANALYSIS.md` | Markdown Report | Detailed concurrency scaling analysis | Explains sign inversion from $B=14$ (-0.00337%) to $B=32$ (+0.00420%) |
| `mohman/point_4/big_exp/figures/fig9_falru_scaling_effect.png` | Publication Plot | Concurrency scaling plot with table | Magnified Y-axis line plot, zero reference line, supporting metric table |

---

## 6. Historical In-Situ Prediction-Only Artifacts (`phase_4d5c`)

- **Primary Directory**: `/home/mohman/ca_project/LLMSimulator/results/phase_4d5c/`
- **Objective**: In-situ simulator validation of prediction-only shadow mode compared directly against baseline and hardware prefetching under identical execution parameters.

### Key Files and Datasets

| Artifact Path | Artifact Type | Description | Key Contents / Parameters |
| :--- | :--- | :--- | :--- |
| `LLMSimulator/results/phase_4d5c/production_matrix_summary.csv` | CSV Dataset | 9-run matrix summary | Evaluates Baseline (`c01`), ST-MoE Pred-Only (`c02`), CSTP Pred-Only (`c03`), ST-MoE Prefetch (`c04`), CSTP Prefetch (`c05`) |
| `LLMSimulator/configs/phase_4d5c/c01_baseline_lru.yaml` | YAML Config | Baseline execution configuration | $L=78, B=1, T=10$, trace `ms_task12_t128_l75.csv`, cache 80 GB |
| `LLMSimulator/configs/phase_4d5c/c02_stmoe_predonly.yaml` | YAML Config | ST-MoE shadow-mode configuration | `mode: prediction_only`, prefetch transfers suppressed, predictor latency measured |
| `LLMSimulator/configs/phase_4d5c/c03_cstp_predonly.yaml` | YAML Config | CSTP shadow-mode configuration | `mode: prediction_only`, prefetch transfers suppressed, predictor latency measured |

---

## 7. Artifact Integrity and Provenance Summary

1. **Deterministic Reproducibility**: Across all simulation runs in Point 3, Point 4 Phase 3, and Point 4 Big Experiment, all repeated runs produced bit-exact identical cycle, byte, and event counts.
2. **Strict Directory Isolation**: Point 5 artifacts reside exclusively in `mohman/point_5/`. No existing files in Point 1, Point 2, Point 3, Point 4, or `LLMSimulator` have been modified or overwritten.
3. **Trace Provenance**: Traces for large workloads originate from the authenticated `moe_anywhere/e1-sage-a1.5.parquet` dataset without artificial prompt repetition or extrapolation.

