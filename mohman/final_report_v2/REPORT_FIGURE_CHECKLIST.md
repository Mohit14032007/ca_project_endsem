# Complete Experimental Figure Verification Checklist

## Master Audit of All 32 Requested Figures

This checklist documents the forensic verification of every single figure requested for the final End-Semester Computer Architecture Project Report on **MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference**.

| Figure ID | Key Filename | Project Point | Descriptive Title | Format & Dimensions | Provenance Data Source | Audit Status |
|:---|:---|:---|:---|:---|:---|:---|
| **Fig 3.1** | `fig1_certified_frozen_recall_vs_profiling` | Point 1 | Certified Frozen Recall vs. Profiling Percentage for CSTP Predictor | PNG (403,088 B), PDF | `mohman/point_1/cstp/results/cstp_profiling_frozen_sweep.csv` | **Verified Present** |
| **Fig 3.2** | `fig6_profiling_pct_vs_hit_rate` | Point 1 | Offline Hit Rate vs. Profiling Percentage across Prediction Horizons (H=1, 2, 4) | PNG (441,402 B), PDF | `mohman/point_1/cstp/results/cstp_multistep_horizon_sweep.csv` | **Verified Present** |
| **Fig 3.3** | `cold_vs_warm_recall` | Point 1 (ST-MoE) | ST-MoE Cold-Start vs. Warm-State Prediction Recall across Layers | PNG (146,639 B), PDF | `mohman/point_1/st_moe/results/cold_warm_layer_recall.csv` | **Verified Present** |
| **Fig 3.4** | `hit_rate_comparison` | Point 1 (ST-MoE vs CSTP) | End-to-End Prediction Hit Rate Comparison: CSTP vs. ST-MoE Baseline | PNG (164,362 B), PDF | `LLMSimulator/results/endsem_phase_c/aggregated_metrics.csv` | **Verified Present** |
| **Fig 3.5** | `native_threshold_tradeoff` | Point 1 (ST-MoE) | ST-MoE Native Confidence Threshold vs. Candidate Recall Trade-Off | PNG (201,727 B), PDF | `mohman/point_1/st_moe/results/threshold_sweep.csv` | **Verified Present** |
| **Fig 5.1** | `fig_01a_w1_total_time_vs_stall` | Point 3 | Total Execution Time vs. Exposed Memory Stall for Workload W1 (B=1) | PNG (148,758 B), PDF | `mohman/point_3/final_results/csv/01_execution_time_stall_breakdown.csv` | **Verified Present** |
| **Fig 5.2** | `fig_01a_w3_total_time_vs_stall` | Point 3 | Total Execution Time vs. Exposed Memory Stall for Workload W3 (B=14) | PNG (164,294 B), PDF | `mohman/point_3/final_results/csv/01_execution_time_stall_breakdown.csv` | **Verified Present** |
| **Fig 5.3** | `fig_01a_w4_total_time_vs_stall` | Point 3 | Total Execution Time vs. Exposed Memory Stall for Workload W4 (B=32) | PNG (158,186 B), PDF | `mohman/point_3/final_results/csv/01_execution_time_stall_breakdown.csv` | **Verified Present** |
| **Fig 5.4** | `fig_01a_w5_total_time_vs_stall` | Point 3 | Total Execution Time vs. Exposed Memory Stall for Workload W5 (B=64) | PNG (161,819 B), PDF | `mohman/point_3/final_results/csv/01_execution_time_stall_breakdown.csv` | **Verified Present** |
| **Fig 5.5** | `fig_01a_w7_total_time_vs_stall` | Point 3 | Total Execution Time vs. Exposed Memory Stall for Workload W7 (B=128) | PNG (159,285 B), PDF | `mohman/point_3/final_results/csv/01_execution_time_stall_breakdown.csv` | **Verified Present** |
| **Fig 5.6** | `fig_01b_all_workloads_time_and_stall_overview` | Point 3 | Comprehensive Overview of Execution Time and Stall Reduction across All Workloads (W1–W7) | PNG (212,067 B), PDF | `mohman/point_3/final_results/csv/01_execution_time_stall_breakdown.csv` | **Verified Present** |
| **Fig 5.7** | `fig_02b_coverage_vs_batch_size` | Point 3 | Prefetch Coverage Scaling vs. Batch Size (B=1 to 128) | PNG (135,927 B), PDF | `mohman/point_3/final_results/csv/02_prefetch_coverage_summary.csv` | **Verified Present** |
| **Fig 5.8** | `fig_02c_coverage_vs_sequence_length` | Point 3 | Prefetch Coverage vs. Sequence Generation Length (T=32 to 512 Tokens) | PNG (123,405 B), PDF | `mohman/point_3/final_results/csv/02_prefetch_coverage_summary.csv` | **Verified Present** |
| **Fig 5.9** | `fig_04a_unused_prefetch_rate` | Point 3 | Unused Prefetch Rate (%) across Workloads W1–W7 | PNG (122,911 B), PDF | `mohman/point_3/final_results/csv/04_unused_prefetches.csv` | **Verified Present** |
| **Fig 5.10** | `fig_04b_wasted_speculative_bytes` | Point 3 | Wasted Speculative DDR5 Memory Traffic Volume (GiB) | PNG (119,071 B), PDF | `mohman/point_3/final_results/csv/04_unused_prefetches.csv` | **Verified Present** |
| **Fig 5.11** | `fig_06a_ddr5_traffic_breakdown` | Point 3 | DDR5 Memory Traffic Composition: Demand Misses vs. Speculative Prefetches | PNG (135,153 B), PDF | `mohman/point_3/final_results/csv/06_memory_traffic.csv` | **Verified Present** |
| **Fig 5.12** | `fig_07c_stall_reduction_percentage` | Point 3 | Percentage Stall Time Reduction under Predictor-Driven Prefetching | PNG (121,170 B), PDF | `mohman/point_3/final_results/csv/07_stall_reduction.csv` | **Verified Present** |
| **Fig 5.13** | `fig_08b_expert_load_latency_reduction` | Point 3 | Average Expert Load Latency Reduction across Workloads | PNG (106,808 B), PDF | `mohman/point_3/final_results/csv/08_expert_load_latency.csv` | **Verified Present** |
| **Fig 5.14** | `fig_09b_decode_step_latency_reduction` | Point 3 | End-to-End Decode Step Latency Reduction (%) | PNG (101,132 B), PDF | `mohman/point_3/final_results/csv/09_end_to_end_latency.csv` | **Verified Present** |
| **Fig 6.1** | `fig9_falru_scaling_effect` | Point 4 | FA-LRU Scaling Effect: Cache Hit Rate and Pollution Resistance vs. Standard LRU | PNG (781,153 B), PDF | `mohman/point_4/big_exp/results/falru_scaling_data.csv` | **Verified Present** |
| **Fig 7.1** | `fig11_falru_incremental_contribution` | Point 5 | FA-LRU Incremental Contribution: Isolating Prefetching from Cache Management Gains | PNG (333,784 B), PDF | `mohman/point_5/results/comparative_four_way_matrix.csv` | **Verified Present** |
| **Fig 7.2** | `fig6_exposed_stall_comparison` | Point 5 | Four-Way Architectural Exposed Memory Stall Comparison (Configs A, B, C, D) | PNG (331,288 B), PDF | `mohman/point_5/results/comparative_four_way_matrix.csv` | **Verified Present** |
| **Fig 7.3** | `fig7_ddr5_traffic_composition` | Point 5 | DDR5 Off-Chip Traffic Composition across Four Architectural Configurations | PNG (347,401 B), PDF | `mohman/point_5/results/comparative_four_way_matrix.csv` | **Verified Present** |
| **Fig 8.1** | `fig2_batch_size_exposed_stall` | Point 6 | Exposed Memory Stall vs. Inference Batch Size Scaling (B=1 to 32) | PNG (170,075 B), PDF | `mohman/point_6/final_results/csv/batch_size_sweep.csv` | **Verified Present** |
| **Fig 8.2** | `fig3_batch_size_token_latency` | Point 6 | Per-Token Generation Latency vs. Batch Size Scaling | PNG (160,074 B), PDF | `mohman/point_6/final_results/csv/batch_size_sweep.csv` | **Verified Present** |
| **Fig 8.3** | `fig4_context_length_kv_cache` | Point 6 | KV Cache Memory Capacity Footprint vs. Context Sequence Length | PNG (143,978 B), PDF | `mohman/point_6/final_results/csv/context_length_sweep.csv` | **Verified Present** |
| **Fig 8.4** | `fig5_context_length_attention_time` | Point 6 | Attention Computation Time Scaling vs. Context Length and MoE Weight Hiding | PNG (187,173 B), PDF | `mohman/point_6/final_results/csv/context_length_sweep.csv` | **Verified Present** |
| **Fig 8.5** | `fig7_lookahead_timely_coverage` | Point 6 | Timely Prefetch Coverage Dynamics vs. Lookahead Horizon (H=1 to 4) | PNG (100,780 B), PDF | `mohman/point_6/final_results/csv/lookahead_sweep.csv` | **Verified Present** |
| **Fig 8.6** | `fig10_budget_speculative_ddr5` | Point 6 | Speculative DDR5 Bandwidth Inflation vs. Prefetch Budget (K=1 to 8) | PNG (114,641 B), PDF | `mohman/point_6/final_results/csv/prefetch_budget_sweep.csv` | **Verified Present** |
| **Fig 9.1** | `fig12_component_pie_breakdown` | Point 7 | Subsystem Energy Consumption Breakdown (Static DRAM, HBM, DDR5, Compute) | PNG (278,777 B), PDF | `mohman/point_7/tables/csv/energy_breakdown_by_component.csv` | **Verified Present** |
| **Fig 9.2** | `fig11_energy_performance_pareto` | Point 7 | Energy-Performance Pareto Frontier across Configurations and Prefetch Budgets | PNG (296,113 B), PDF | `mohman/point_7/tables/csv/reconciled_energy_matrix.csv` | **Verified Present** |
| **Fig 9.3** | `fig6_energy_delay_product` | Point 7 | Energy-Delay Product (EDP) Scaling across Prefetch Aggressiveness Levels | PNG (197,071 B), PDF | `mohman/point_7/tables/csv/reconciled_energy_matrix.csv` | **Verified Present** |

## Summary Statistics

- **Total Figures Audited:** 32
- **Total Figures Verified Present on Disk:** 32
- **Total Missing Figures:** 0 (100% coverage)
- **Point 1 Figures:** 5 (CSTP recall & hit rate, ST-MoE cold vs warm, hit rate comparison, native threshold)
- **Point 2 Figures:** 0 (Grounded in source code, architecture diagrams, sequence diagrams, and worked execution traces)
- **Point 3 Figures:** 14 (Total time vs stall W1, W3, W4, W5, W7, overview, coverage sweeps, unused prefetches, traffic composition, stall reduction, expert load latency, decode step latency)
- **Point 4 Figures:** 1 (FA-LRU scaling effect and cache pollution resilience)
- **Point 5 Figures:** 3 (FA-LRU incremental contribution, exposed stall comparison, DDR5 traffic composition)
- **Point 6 Figures:** 6 (Batch size stall & token latency, context length KV cache & attention time, lookahead timely coverage, budget speculative traffic)
- **Point 7 Figures:** 3 (Subsystem energy breakdown pie, energy-performance Pareto frontier, energy-delay product EDP)
