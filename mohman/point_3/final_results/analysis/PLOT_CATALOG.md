# Point 3 — Publication Plot Catalog and Interpretation Guide

**Repository:** [`/home/mohman/ca_project/LLMSimulator`](file:///home/mohman/ca_project/LLMSimulator)  
**Deliverable Path:** [`mohman/point_3/final_results/analysis/PLOT_CATALOG.md`](file:///home/mohman/ca_project/mohman/point_3/final_results/analysis/PLOT_CATALOG.md)  
**Status:** Complete Visual Index (36 Figures in PNG and PDF Formats)  

---

## Overview

This catalog indexes all 36 publication-quality figures generated across 11 thematic categories within [`plots/`](file:///home/mohman/ca_project/mohman/point_3/final_results/plots/). Every figure is available in both 300 DPI PNG raster format and vector PDF format.

```
========================================================================================================
                                         PLOT SUITE DIRECTORY INDEX
========================================================================================================
Category Directory                       Figures  Formats    Primary Subject
--------------------------------------------------------------------------------------------------------
01_total_time_vs_stall_reduction/        11       PNG + PDF  Simulated time vs exposed stall reduction
02_prefetch_coverage/                    3        PNG + PDF  Strict timely (0.0%) vs associated coverage
03_timeliness/                           2        PNG + PDF  Timely vs late rates, bus transit lead time
04_unused_prefetches/                    2        PNG + PDF  Unused prefetch rates and wasted DDR5 bytes
05_hbm_hit_rate/                         2        PNG + PDF  HBM hit rate invariance analysis
06_memory_traffic/                       3        PNG + PDF  DDR5 traffic breakdown & net overhead
07_exposed_memory_stall/                 3        PNG + PDF  Exposed GPU stall time & reduction scaling
08_expert_load_latency/                  2        PNG + PDF  Expert load service time & perceived savings
09_end_to_end_latency/                   3        PNG + PDF  Decode step latency & tail p95 latency
10_throughput/                           2        PNG + PDF  Aggregate inference throughput (tokens/s)
comparison_dashboards/                   3        PNG + PDF  Multi-panel overview & head-to-head dashboards
========================================================================================================
Total Generated Figures: 36 unique visualizations (72 files total: 36 PNG + 36 PDF)
```

---

## Category 1: Total Simulated Time vs Exposed Memory-Stall Reduction

### Figures 01a (W1 to W7) — Per-Workload Comparison Plots
- **Files:** `plots/01_total_time_vs_stall_reduction/fig_01a_w{1..7}_total_time_vs_stall.png` and `.pdf`
- **Source CSV:** [`data/consolidated_csv/all_experiment_results.csv`](file:///home/mohman/ca_project/mohman/point_3/final_results/data/consolidated_csv/all_experiment_results.csv)
- **Description:** Dual-axis grouped bar charts showing Total Simulated Execution Time (seconds, left Y-axis) and Exposed Memory-Stall Reduction (milliseconds, right Y-axis) across Baseline, ST-MoE, and CSTP for each individual workload.
- **Key Trend:** In Workload 7, CSTP achieves a **$6,744.86 \text{ ms}$** stall reduction and a **$6.717 \text{ second}$** simulation time reduction, compared to **$2,289.03 \text{ ms}$** and **$2.273 \text{ seconds}$** for ST-MoE.

### Figure 01b — All-Workloads Overview Comparison
- **File:** [`plots/01_total_time_vs_stall_reduction/fig_01b_all_workloads_time_and_stall_overview.png`](file:///home/mohman/ca_project/mohman/point_3/final_results/plots/01_total_time_vs_stall_reduction/fig_01b_all_workloads_time_and_stall_overview.png)
- **Description:** Two-panel overview comparing Total Simulated Time (Panel 1) and Exposed Memory-Stall Reduction (Panel 2) across all seven workloads side-by-side.
- **Architectural Insight:** Stall reductions grow super-linearly with sequence length ($T$), scaling from $+2.75 \text{ ms}$ at $T=4$ to $+6,744.86 \text{ ms}$ at $T=64$.

### Figure 01c — Trade-Off Scatter Plot
- **File:** [`plots/01_total_time_vs_stall_reduction/fig_01c_time_vs_stall_tradeoff_scatter.png`](file:///home/mohman/ca_project/mohman/point_3/final_results/plots/01_total_time_vs_stall_reduction/fig_01c_time_vs_stall_tradeoff_scatter.png)
- **Description:** Scatter plot mapping Total Simulated Execution Time (X-axis, lower is better) against Exposed Memory-Stall Reduction (Y-axis, higher is better).
- **Architectural Insight:** CSTP configurations consistently occupy the optimal upper-left quadrant relative to ST-MoE, demonstrating superior stall reduction per second of execution.

### Figure 01d — Batch Scaling Curves
- **Files:** `fig_01d_batch_scaling_total_time.png` and `fig_01d_batch_scaling_stall_reduction.png`
- **Description:** Concurrency scaling curves across batch sizes $B \in \{14, 32, 48, 70\}$ at fixed sequence length $T=16$.
- **Architectural Insight:** Stall reduction increases monotonically with batch size as working set oversubscription intensifies cache churn.

---

## Category 2: Prefetch Coverage

### Figure 02a — Strict Timely vs Prefetch-Associated Coverage
- **File:** [`plots/02_prefetch_coverage/fig_02a_strict_vs_associated_coverage.png`](file:///home/mohman/ca_project/mohman/point_3/final_results/plots/02_prefetch_coverage/fig_02a_strict_vs_associated_coverage.png)
- **Description:** Grouped bar chart comparing Strict Timely Coverage (identically 0.00%) against Prefetch-Associated Demand Coverage (up to 9.23% for ST-MoE and 4.81% for CSTP).
- **Architectural Insight:** Visually demonstrates why strict timely coverage cannot be achieved under 64 GB/s DDR5, while confirming that up to 9.23% of demands successfully attached to in-flight transfers.

### Figures 02b & 02c — Coverage Scaling with Batch Size and Sequence Length
- **Files:** `fig_02b_coverage_vs_batch_size.png` and `fig_02c_coverage_vs_sequence_length.png`
- **Description:** Trend lines showing how associated coverage evolves with batch size and generation depth.

---

## Category 3: Timeliness and In-Flight Transit Lead Time

### Figure 03a — Timely vs Late Prefetch Rates
- **File:** [`plots/03_timeliness/fig_03a_timely_and_late_rates.png`](file:///home/mohman/ca_project/mohman/point_3/final_results/plots/03_timeliness/fig_03a_timely_and_late_rates.png)
- **Description:** Stacked bar chart showing that 100.0% of matched speculative transfers were in-flight at demand arrival across all evaluated runs.

### Figure 03b — In-Flight Bus Transit Lead Time at Demand Arrival
- **File:** [`plots/03_timeliness/fig_03b_in_flight_transit_time_comparison.png`](file:///home/mohman/ca_project/mohman/point_3/final_results/plots/03_timeliness/fig_03b_in_flight_transit_time_comparison.png)
- **Description:** Head-to-head comparison of average physical bus transit elapsed time when demand arrived in Workload 7: **$66.97 \mu\text{s}$ for CSTP** vs **$3.81 \mu\text{s}$ for ST-MoE**.
- **Root Cause Explanation:** ST-MoE flooded the transfer queue, causing head-of-line blocking. CSTP maintained clear queues, allowing transfers to begin transmission immediately and saving 17.6x more stall per demand.

---

## Category 4: Unused Prefetches and Wasted Memory Traffic

### Figures 04a & 04b — Unused Prefetch Rates and Wasted Bytes
- **Files:** `plots/04_unused_prefetches/fig_04a_unused_prefetch_rate.png` and `fig_04b_wasted_speculative_bytes.png`
- **Description:** Unused prefetch rate (%) and wasted speculative DDR5 traffic (GB) across workloads.
- **Architectural Insight:** Unused prefetch rates decline from ~16% at $T=4$ to 5.87% (ST-MoE) and 7.05% (CSTP) at $T=64$, as longer sequences provide more opportunities for future reuse.

---

## Category 5: HBM Hit Rate Invariance

### Figures 05a & 05b — HBM Cache Hit Rate Comparisons
- **Files:** `plots/05_hbm_hit_rate/fig_05a_hbm_hit_rate_comparison.png` and `fig_05b_hbm_hit_rate_scaling.png`
- **Description:** Demonstrates the invariance of the architectural HBM hit rate (constant 67.0% across all large-workload runs).
- **Architectural Insight:** In-flight attachments are classified as cache misses that attach to active DMA transfers; they do not establish prior cache residency, leaving the HBM hit rate unchanged.

---

## Category 6: Memory Traffic Breakdown & Net Additional Traffic

### Figures 06a, 06b & 06c — DDR5 Memory Traffic Analysis
- **Files:** `fig_06a_ddr5_traffic_breakdown.png`, `fig_06b_net_additional_traffic_comparison.png`, `fig_06c_traffic_overhead_percentage.png`
- **Description:** Total DDR5 bytes transferred, net additional memory traffic (wasted speculative minus prevented demand), and traffic overhead percentage.
- **Architectural Insight:** In Workload 7, CSTP introduces only **$+119.53 \text{ GB}$ ($+0.39\%$)** of net additional traffic, compared to **$+420.32 \text{ GB}$ ($+1.37\%$)** for ST-MoE.

---

## Category 7: Exposed Memory Stall

### Figures 07a, 07b & 07c — Exposed Memory Stall Analysis
- **Files:** `fig_07a_exposed_stall_vs_batch.png`, `fig_07b_exposed_stall_reduction_scaling.png`, `fig_07c_stall_reduction_percentage.png`
- **Description:** Exposed GPU stall time (seconds), stall reduction scaling, and percentage stall reduction.
- **Architectural Insight:** CSTP reduces exposed stall by **$6,744.86 \text{ ms}$** in W7, saving 2.94x more stall than ST-MoE.

---

## Category 8: Expert Load Latency

### Figures 08a & 08b — Expert Load Latency Comparisons
- **Files:** `plots/08_expert_load_latency/fig_08a_mean_expert_load_latency.png` and `fig_08b_expert_load_latency_reduction.png`
- **Description:** Mean perceived expert load latency across all demand accesses, showing perceived latency reductions of up to **$6.45 \mu\text{s}$** per demand under CSTP.

---

## Category 9: End-to-End and Token Latency

### Figures 09a, 09b & 09c — Decode Step and Tail Latency Analysis
- **Files:** `fig_09a_mean_decode_step_latency.png`, `fig_09b_decode_step_latency_reduction.png`, `fig_09c_p95_tail_token_latency.png`
- **Description:** Mean decode step latency, decode step latency reduction, and P95 tail latency scaling.
- **Architectural Insight:** In W7, CSTP saves **$104.95 \text{ ms}$ per decode step** across all 64 generation steps.

---

## Category 10: Aggregate Throughput

### Figures 10a & 10b — Inference Throughput Scaling
- **Files:** `plots/10_throughput/fig_10a_throughput_tokens_per_sec.png` and `fig_10b_throughput_scaling_with_batch.png`
- **Description:** Aggregate inference throughput (tokens/second) scaling with serving concurrency. Throughput increases from $12.07 \text{ tok/s}$ at $B=14$ to $17.64 \text{ tok/s}$ at $B=70$.

---

## Category 11: Comparison Dashboards

### Dashboard 1 — Large Workload Comprehensive Overview (6 Panels)
- **File:** [`plots/comparison_dashboards/dashboard_large_workloads_overall.png`](file:///home/mohman/ca_project/mohman/point_3/final_results/plots/comparison_dashboards/dashboard_large_workloads_overall.png)
- **Description:** Executive 6-panel dashboard summarizing: (A) Total Simulated Time, (B) Exposed Stall Reduction, (C) Associated Coverage, (D) Net Additional DDR5 Traffic, (E) Unused Prefetch Rate, and (F) Inference Throughput.

### Dashboard 2 — Workload 7 Head-to-Head Comparison (4 Panels)
- **File:** [`plots/comparison_dashboards/dashboard_w7_deepseek_v3_head_to_head.png`](file:///home/mohman/ca_project/mohman/point_3/final_results/plots/comparison_dashboards/dashboard_w7_deepseek_v3_head_to_head.png)
- **Description:** High-impact comparison of ST-MoE vs CSTP in the largest serving regime ($B=70, T=64, M=40$).

### Dashboard 3 — Original Campaign Summary Dashboard (3 Panels)
- **File:** [`plots/comparison_dashboards/dashboard_original_campaign.png`](file:///home/mohman/ca_project/mohman/point_3/final_results/plots/comparison_dashboards/dashboard_original_campaign.png)
- **Description:** Summarizes the original 20-run campaign ($M=3, T=4, \text{Cache}=80\text{ GB}$), illustrating identical curves across baseline and prefetchers due to cache unconstrained residency.

