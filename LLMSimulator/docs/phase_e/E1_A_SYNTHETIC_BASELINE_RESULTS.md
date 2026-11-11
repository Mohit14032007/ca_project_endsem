# Phase E1-A: Synthetic DeepSeek Baseline Characterization Results

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** E1-A Hardware Demand-Loading Baseline  
**Date:** November 2026  
**Status:** COMPLETE  
**Repository Path:** `/home/mohman/ca_project/LLMSimulator`  
**Execution Binary:** `build/run_deepseekv3_mixtral_style`  
**Data Artifacts:** `results/endsem_phase_e/e1_a_synthetic_baseline/`  

---

## 1. Executive Summary

This document presents the authoritative hardware-performance baseline for `LLMSimulator` using its synthetic DeepSeek-style expert routing workload. This evaluation characterizes the existing **demand-loading** memory subsystem under strictly controlled **NO-PREFETCH** execution (`bypass_offchip_execution_cache = true`).

Key findings include:

1. **The 97% MoE Memory Wall:** Under cold demand loading, DDR5 expert weight transfers consume **$96.8\%$ to $97.5\%$ of total decoding step latency**. For a 2-MoE-layer block ($K=8$), loading 16 cold experts ($1.34\text{ GB}$) from DDR5 incurs **$62.42\text{ ms}$ of synchronous GPU stall time** compared to just **$2.05\text{ ms}$ of GPU compute execution**.
2. **Linear Batch Throughput Scaling:** Because DDR5 expert loads dominate execution time and batch sequences reuse the same routed expert working set, increasing batch size from $B=1$ to $B=8$ produces an **$8.0\times$ linear increase in decoding throughput** (from $15.51\text{ tokens/s}$ to $124.01\text{ tokens/s}$) with negligible step latency increase ($+30\,\mu\text{s}$).
3. **Invariance to Prompt Context Length:** Scaling prompt context length by $8\times$ (from $L=128$ to $L=1024$ tokens) increases decoding step latency by only **$0.005\%$ ($+3.2\,\mu\text{s}$)**, showing that decoding step duration is overwhelmingly bounded by DDR5 expert memory bandwidth, completely overshadowing attention computation.
4. **$31.5\times$ Speedup Potential with Expert Residency:** Under Warm HBM execution where active experts are resident in HBM, decoding latency plummets from **$64.48\text{ ms}$ to $2.05\text{ ms}$**, yielding an instantaneous **$31.5\times$ speedup** and driving throughput from $15.51\text{ tokens/s}$ to **$487.80\text{ tokens/s}$**.
5. **DDR5 Bus Utilization:** Cycle-accurate Ramulator2 DDR5 simulation across 2 channels achieves an average sustained bandwidth of **$22.58\text{ GB/s}$**, representing **$44.10\%$ sustained utilization** of the $51.2\text{ GB/s}$ theoretical peak bandwidth.

---

## 2. Experimental Setup and Execution Invariants

All baseline experiments were conducted on a dedicated host running Linux on a 12-core `x86_64` system. In strict adherence to project constraints:

1. **Prefetching Disabled:** No asynchronous staging queue, prefetch thread, or proactive transfer logic is enabled.
2. **Zero Predictor Connection:** Neither CSTP, ST-MoE, ACSH, nor TraceReplayPredictor is connected.
3. **Execution Cache Bypassed:** Both YAML (`system.optimization.bypass_offchip_execution_cache: true`) and environment override (`BYPASS_OFFCHIP_EXECUTION_CACHE=1`) were enforced, ensuring every DRAM request enters Ramulator2.
4. **Deterministic Reproducibility:** RNG seed is hardcoded to `777` in `Scheduler::getRandomExpert`, ensuring identical routing across identical configurations.
5. **Zero Ternary Operators:** All automation scripts and processing pipelines enforce `ast.IfExp == 0`.

### Summary of System Hardware Parameters

| Hierarchy Component | Specification | Operational Parameter |
|---|---|---|
| **GPU Compute** | NVIDIA A100-SXM4-80GB Preset | 312 TFLOPS FP16 Tensor Core Peak |
| **On-Chip HBM3** | 80 GB Capacity (5 Cubes, 32 Pseudo-Channels) | 2,039 GB/s Theoretical Peak Bandwidth |
| **Off-Chip DDR5** | 2.75 TB Capacity (1 Cube, 2 Channels, 2 Ranks) | 51.2 GB/s Theoretical Peak Bandwidth (DDR5-3200BN) |
| **MoE Geometry** | 256 Routed Experts / Layer, Top-8 Selection | 88.08 MB FP16 Footprint per Expert ($3 \times 7168 \times 2048 \times 2\text{ B}$) |
| **Interconnect** | PCIe / CXL Channel Mapping | `RoBaRaCoCh` Address Mapping, 64-Byte Burst Granularity |

---

## 3. Workload Scaling Characterization

### 3.1 Batch-Size Sensitivity ($B \in \{1, 2, 4, 8\}$, Fixed $L=128$)

The batch size sweep evaluates decoding performance as concurrent sequences scale from 1 to 8 tokens per step:

| Batch Size ($B$) | Total Tokens | Sim Step Time ($T_{\text{step}}$) | Average Token Latency | Simulated Throughput | Host Wall-Clock | DDR5 Traffic | Stall Fraction |
|---|---|---|---|---|---|---|---|
| **$B = 1$** | 1 | 64.48 ms | 64.48 ms | 15.51 tokens/s | 181.94 s | 1344.0 MB | 96.80% |
| **$B = 2$** | 2 | 64.48 ms | 32.24 ms | 31.02 tokens/s | 179.31 s | 1344.0 MB | 96.80% |
| **$B = 4$** | 4 | 64.49 ms | 16.12 ms | 62.03 tokens/s | 189.04 s | 1344.0 MB | 96.79% |
| **$B = 8$** | 8 | 64.51 ms | 8.06 ms | 124.01 tokens/s | 188.63 s | 1344.0 MB | 96.76% |

```
Execution Time vs Batch Size:
  B=1: [==================================================] 64.48 ms (Throughput: 15.51 tok/s)
  B=2: [==================================================] 64.48 ms (Throughput: 31.02 tok/s)
  B=4: [==================================================] 64.49 ms (Throughput: 62.03 tok/s)
  B=8: [==================================================] 64.51 ms (Throughput: 124.01 tok/s)
```

**Key Takeaways:**
- Step latency is practically invariant to batch size ($64.48\text{ ms} \to 64.51\text{ ms}$, a mere $+0.04\%$ increase).
- Throughput scales linearly ($15.51 \to 124.01\text{ tokens/s}$, an $8.0\times$ speedup) because the $62.42\text{ ms}$ DDR5 expert loading latency is shared across all tokens in the batch.
- Per-token latency drops from $64.48\text{ ms}$ to $8.06\text{ ms}$.

### 3.2 Sequence-Length Sensitivity ($L \in \{128, 256, 512, 1024\}$, Fixed $B=1$)

The sequence length sweep tests sensitivity to context length:

| Context Length ($L$) | Sim Step Time | Attention Gen Time | RoPE Time | DDR5 Traffic | Throughput | Stall Fraction |
|---|---|---|---|---|---|---|
| **$L = 128$** | 64.4784 ms | 1.03 µs | 0.24 µs | 1344.0 MB | 15.51 tokens/s | 96.80% |
| **$L = 256$** | 64.4786 ms | 1.20 µs | 0.40 µs | 1344.0 MB | 15.51 tokens/s | 96.80% |
| **$L = 512$** | 64.4795 ms | 2.29 µs | 0.73 µs | 1344.0 MB | 15.51 tokens/s | 96.80% |
| **$L = 1024$** | 64.4816 ms | 4.58 µs | 1.37 µs | 1344.0 MB | 15.51 tokens/s | 96.80% |

**Key Takeaways:**
- Attention generation latency scales quadratically with context length ($1.03\,\mu\text{s} \to 4.58\,\mu\text{s}$), and RoPE latency scales linearly ($0.24\,\mu\text{s} \to 1.37\,\mu\text{s}$).
- However, because DDR5 expert weight transfer requires $62,416.45\,\mu\text{s}$, the total step latency changes by only $+3.2\,\mu\text{s}$ ($+0.005\%$).
- Context length scaling in decode mode has virtually no impact on token latency when off-chip expert loading is synchronous.

### 3.3 Combined Workload Scaling

Evaluating simultaneous batch size and sequence length scaling:

| Configuration ($B, L$) | Sim Step Time | Per-Token Latency | Throughput | DDR5 Traffic | DRAM Requests |
|---|---|---|---|---|---|
| **$B=1, L=128$** | 64.48 ms | 64.48 ms | 15.51 tokens/s | 1344.0 MB | 22,020,096 |
| **$B=2, L=256$** | 64.48 ms | 32.24 ms | 31.02 tokens/s | 1344.0 MB | 22,020,096 |
| **$B=4, L=512$** | 64.50 ms | 16.13 ms | 62.01 tokens/s | 1344.0 MB | 22,020,096 |
| **$B=8, L=512$** | 64.52 ms | 8.07 ms | 123.99 tokens/s | 1344.0 MB | 22,020,096 |
| **$B=8, L=1024$** | 64.54 ms | 8.07 ms | 123.95 tokens/s | 1344.0 MB | 22,020,096 |

**Key Takeaways:**
- Scaling across the full combined matrix from $(B=1, L=128)$ to $(B=8, L=1024)$ yields an **$8.0\times$ throughput expansion** while keeping total step latency bounded under $64.55\text{ ms}$.
- Memory stalls account for $>96.7\%$ of execution time in all evaluated operating regimes.

---

## 4. HBM Initial-State and Cache Residency Effects

To isolate the impact of on-chip memory residency, three HBM cache states were evaluated under identical routing seeds:

| HBM Initial State | Step Latency | Exposed Memory Stall | DDR5 Traffic | HBM Hit Rate | Resident Memory | Throughput |
|---|---|---|---|---|---|---|
| **Cold HBM** | 64.48 ms | 62.42 ms | 1344.0 MB | 66.7% | 1.31 GB (16 experts) | 15.51 tokens/s |
| **Natural LRU (2-Step)** | 128.96 ms | 124.83 ms | 2688.0 MB | 66.7% | 2.62 GB (32 experts) | 15.51 tokens/s |
| **Warm Replay** | **2.05 ms** | **0.00 ms** | **0.0 MB** | **100.0%** | 2.62 GB (32 experts) | **487.80 tokens/s** |

### Detailed Residency Progression

```
HBM Cache Hit Rate Progression:
  Cold Demand Start: [===================               ] 66.7% (1 Miss on Gate, 2 Hits on Up/Down)
  Natural LRU (2-Step): [===================            ] 66.7% (New random experts loaded in Step 1)
  Warm HBM Replay:   [==================================] 100.0% (Zero DDR5 accesses, pure compute)

Execution Time Breakdown:
  Cold HBM:    [Compute: 2.06 ms | DDR5 Memory Stall: 62.42 ms (96.8%)] -> Total: 64.48 ms
  Warm Replay: [Compute: 2.05 ms | Zero Stall]                        -> Total: 2.05 ms (31.5x speedup)
```

**Key Takeaways:**
1. **Intra-Expert Cache Behavior:** When an expert is first accessed, $W_{\text{gate}}$ encounters an HBM cache MISS, triggering DDR5 demand fetch. As $W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$ are loaded and registered via `mark_expert_resident`, subsequent accesses to $W_{\text{up}}$ and $W_{\text{down}}$ within that same decoding step hit the HBM cache ($2\text{ HITs} : 1\text{ MISS} \implies 66.7\%$ hit rate).
2. **Warm Cache Speedup:** When all expert weights are resident in HBM prior to execution, DDR5 traffic drops to **zero**, eliminating all $62.42\text{ ms}$ of stall time. Step latency drops to **$2.05\text{ ms}$**, unlocking a **$31.5\times$ speedup** and raising throughput to **$487.80\text{ tokens/s}$**.
3. **Capacity Thresholds:** In this bounded 2-MoE-layer setup, 32 resident experts consume $2.82\text{ GB}$ ($3.5\%$ of the $80\text{ GB}$ HBM budget). In the full 57-MoE-layer DeepSeek-V3 architecture, 456 experts per step consume $40.16\text{ GB}$ ($50.2\%$), reaching the $908$-expert ($80\text{ GB}$) cache limit by Step 2, where Natural LRU eviction begins.

---

## 5. DDR5 Memory Subsystem Metrics

Cycle-accurate Ramulator2 DDR5 telemetry reveals the exact physical dynamics of off-chip expert transfers:

- **Total Bytes per Expert:** $88,080,384\text{ bytes}$ ($88.08\text{ MB}$).
- **Burst Granularity:** $64\text{ bytes}$ per DRAM request $\implies 1,376,256\text{ requests}$ per expert.
- **Simulated Transfer Duration:**
  $$T_{\text{load}} = 3,901,028\text{ ns} \approx 3.901\text{ ms per expert}$$
- **Average DRAM Request Latency:**
  $$t_{\text{req}} = \frac{3,901,028\text{ ns}}{1,376,256\text{ requests}} \approx 2.83\text{ ns per 64-byte burst}$$
- **Sustained DDR5 Bandwidth:**
  $$\text{BW}_{\text{sustained}} = \frac{88,080,384\text{ bytes}}{0.003901\text{ seconds}} \approx 22.58\text{ GB/s}$$
- **Bandwidth Utilization:**
  $$\eta_{\text{bw}} = \frac{22.58\text{ GB/s}}{51.20\text{ GB/s}} \times 100\% = 44.10\%$$
- **Host Simulation Overhead:**
  Simulating 16 cold experts ($1.34\text{ GB}$) in Ramulator2 consumes **$\sim 176 - 189$ seconds of host CPU wall-clock time** ($\sim 11.2\text{ s}$ per cold expert).

---

## 6. Artifact Inventory

All generated artifacts are saved under `results/endsem_phase_e/e1_a_synthetic_baseline/` and mirrored in the repository:

### 6.1 CSV Tables and Manifests
1. [`baseline_summary.csv`](file:///home/mohman/ca_project/LLMSimulator/results/endsem_phase_e/e1_a_synthetic_baseline/baseline_summary.csv): Full metrics across all 14 evaluated configurations.
2. [`batch_size_sweep.csv`](file:///home/mohman/ca_project/LLMSimulator/results/endsem_phase_e/e1_a_synthetic_baseline/batch_size_sweep.csv): Sensitivity across batch sizes $B \in \{1, 2, 4, 8\}$.
3. [`sequence_length_sweep.csv`](file:///home/mohman/ca_project/LLMSimulator/results/endsem_phase_e/e1_a_synthetic_baseline/sequence_length_sweep.csv): Sensitivity across context lengths $L \in \{128, 256, 512, 1024\}$.
4. [`combined_scaling.csv`](file:///home/mohman/ca_project/LLMSimulator/results/endsem_phase_e/e1_a_synthetic_baseline/combined_scaling.csv): Combined workload matrix scaling.
5. [`hbm_initial_state.csv`](file:///home/mohman/ca_project/LLMSimulator/results/endsem_phase_e/e1_a_synthetic_baseline/hbm_initial_state.csv): Comparison of Cold, Natural LRU, and Warm Replay.
6. [`per_run_metrics.csv`](file:///home/mohman/ca_project/LLMSimulator/results/endsem_phase_e/e1_a_synthetic_baseline/per_run_metrics.csv): Detailed per-run performance counters.
7. [`experiment_manifest.json`](file:///home/mohman/ca_project/LLMSimulator/results/endsem_phase_e/e1_a_synthetic_baseline/experiment_manifest.json): Machine-readable execution manifest.
8. [`execution_log.txt`](file:///home/mohman/ca_project/LLMSimulator/results/endsem_phase_e/e1_a_synthetic_baseline/execution_log.txt): Comprehensive stdout/stderr execution log.

### 6.2 Publication Figures
- **Figure 1 (Execution Time vs Batch Size):** [`exec_time_vs_batch_size.png`](file:///home/mohman/ca_project/LLMSimulator/results/endsem_phase_e/e1_a_synthetic_baseline/figures/exec_time_vs_batch_size.png)
- **Figure 2 (Throughput vs Batch Size):** [`throughput_vs_batch_size.png`](file:///home/mohman/ca_project/LLMSimulator/results/endsem_phase_e/e1_a_synthetic_baseline/figures/throughput_vs_batch_size.png)
- **Figure 3 (Execution Time vs Sequence Length):** [`exec_time_vs_seq_length.png`](file:///home/mohman/ca_project/LLMSimulator/results/endsem_phase_e/e1_a_synthetic_baseline/figures/exec_time_vs_seq_length.png)
- **Figure 4 (DDR5 Traffic vs Workload Size):** [`ddr5_traffic_vs_workload_size.png`](file:///home/mohman/ca_project/LLMSimulator/results/endsem_phase_e/e1_a_synthetic_baseline/figures/ddr5_traffic_vs_workload_size.png)
- **Figure 5 (HBM Hit Rate vs Workload Size):** [`hbm_hit_rate_vs_workload_size.png`](file:///home/mohman/ca_project/LLMSimulator/results/endsem_phase_e/e1_a_synthetic_baseline/figures/hbm_hit_rate_vs_workload_size.png)
- **Figure 6 (Memory Stall Time vs Workload Size):** [`memory_stall_time_vs_workload_size.png`](file:///home/mohman/ca_project/LLMSimulator/results/endsem_phase_e/e1_a_synthetic_baseline/figures/memory_stall_time_vs_workload_size.png)

---

## 7. Limitations and Unsupported Metrics

1. **Synthetic Routing Artifacts:** Because routing uses pseudo-random selection with seed 777, inter-token expert reuse is lower than in real NLP traces where semantic locality produces temporal correlation.
2. **Channel-Level DDR5 Bus Queuing:** The simulator logs cumulative DDR5 interface time and aggregate traffic; per-channel cycle-by-cycle command traces are not broken out per expert in standard CSV summaries.
3. **Synchronous Paging Stalls:** The current demand paging path blocks the GPU pipeline immediately upon a cold miss. There is no background staging buffer or non-blocking issue queue.

---

## 8. Implications for Predictor-Driven Asynchronous Prefetching

The quantitative baseline establishes crucial design principles for the upcoming Phase E prefetch implementation:

1. **Target Latency Window:** Loading an 88.08 MB expert from DDR5 requires **$3.90\text{ ms}$**. Because single-layer GPU compute takes only $\sim 30\,\mu\text{s}$, an intra-token horizon of $H=1$ layer does **not** provide enough compute time to fully hide DDR5 transfer latency ($30\,\mu\text{s} \ll 3900\,\mu\text{s}$). Prefetching must operate across multiple layers ahead ($H \ge 2$) or across future tokens ($H_{\text{token}} \ge 1$) to achieve meaningful overlap.
2. **Bandwidth Headroom:** Achieving $22.58\text{ GB/s}$ sustained out of $51.20\text{ GB/s}$ ($44.1\%$ utilization) indicates substantial DDR5 bus headroom ($\approx 28.6\text{ GB/s}$). Prefetching can utilize this idle bandwidth concurrently with compute without saturating the DDR5 bus.
3. **The Value of Precision:** With cold loading imposing a $3.90\text{ ms}$ penalty per expert, fetching an unneeded expert (false positive) wastes $88\text{ MB}$ of DDR5 bandwidth and risks evicting an active expert from HBM. ACSH's confidence gating ($\tau$-thresholding) will be essential to prevent cache pollution.
4. **Speedup Upper Bound:** The Warm Replay baseline demonstrates that eliminating memory stalls unlocks up to **$31.5\times$ acceleration** in decoding throughput ($15.5 \to 487.8\text{ tok/s}$). Even a partial prefetch hit rate of $60 - 80\%$ can yield massive end-to-end inference speedups ($3\times - 5\times$).

