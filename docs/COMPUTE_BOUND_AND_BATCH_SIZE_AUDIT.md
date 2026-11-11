# Forensic Audit: Batch-Size Experiments, Compute-Time Accounting, and Compute-Bound MoE Configurations

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Git Baseline Anchor:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Audit Date:** November 9, 2026  
**Document Locations:**
- `docs/COMPUTE_BOUND_AND_BATCH_SIZE_AUDIT.md`
- `LLMSimulator/docs/COMPUTE_BOUND_AND_BATCH_SIZE_AUDIT.md`

---

## 1. Phase 1: Audit of All Existing Batch-Size and Compute-Scaling Experiments

### 1.1 Scope and Inventory of Historical Experiments
Forensic inspection of the repository identified four distinct historical experiment campaigns evaluating batch sizes, sequence lengths, or compute duration scaling:

1. **Phase E1-A Synthetic Baseline Campaign (`batch_size_sweep.csv`):**
   - **Evaluated Batch Sizes:** $B \in \{1, 2, 4, 8\}$.
   - **Mechanism:** Evaluated actual batch sizes in `Scheduler::running_queue` with synthetic uniform routing (`seed 777`).
   - **Model & Configuration:** DeepSeek-style model ($K=8$, $E=256$, 5 total layers, 2 MoE stages), single-step decode ($iter=1$), input length 128, cold HBM cache.
   - **Critical Caveat:** Synthetic routing was cached using `cached_routing` keyed on `(token_id, cur_layer)`. Because all sequences in a batch shared the same `token_id = 128`, every sequence in the batch selected the **identical 8 experts**. As a result, the number of unique experts remained fixed at exactly 16 across all batch sizes, causing DDR5 traffic and memory stall time to remain completely flat ($62.42\text{ ms}$).

2. **Phase E Final Hardware Baseline Campaign (`FINAL_BASELINE_RESULTS.csv`):**
   - **Evaluated Batch Sizes:** Table A evaluated $B=1$; Table B evaluated $B=2$ and $B=4$.
   - **Mechanism:** Evaluated actual multi-sequence batching using real routing traces from **MoE Anywhere** (`prompts_60_61`, 40 MoE stages, Top-8, 256 routed experts).
   - **Workload:** 128 and 256 decoding steps ($T=128, 256$).
   - **Critical Finding:** Under real multi-sequence traces, sequences exhibited genuine routing diversity. Increasing batch size from $B=2$ to $B=4$ tripled off-chip DDR5 traffic from $3.76\text{ TB}$ to $10.81\text{ TB}$ ($T=128$) and from $8.08\text{ TB}$ to $23.34\text{ TB}$ ($T=256$), increasing memory stall from $325\text{ s}$ to $717\text{ s}$ ($T=128$) and driving HBM hit rate down from $79.34\%$ to $67.99\%$.

3. **Phase 2F Synthetic Compute-Multiplier Campaign (`PHASE2F_SUMMARY.csv`):**
   - **Evaluated Batch Sizes:** $B=1$ only.
   - **Mechanism:** Artificial compute scaling via `COMPUTE_MULTIPLIER` ($M \in \{1.0\times, 4.0\times, 8.0\times, 12.0\times, 16.0\times\}$).
   - **Model & Configuration:** DeepSeek-V3 Mixtral-style, Workload A1 (MoE-Suite Task 12, 75 MoE stages, Top-8, 256 routed experts), 4 decoding tokens.
   - **Outcome:** Proved that when the compute window artificially crosses the physical DDR5 transfer threshold ($3.90116\text{ ms}$), timely resident hits emerge ($H_{\text{timely}} = 1 \text{ to } 2$) and exposed memory stall decreases by up to $12.86\text{ ms}$.

4. **Duplex A100 Baseline Reproduction (`reproduction_A100/results/a100_duplex/`):**
   - **Evaluated Batch Sizes:** $B=32$.
   - **Mechanism:** Actual batch size 32 on Mixtral 8x7B (32 tokens per step, 5 decoding iterations), evaluating pure on-chip HBM and PIM logic execution without off-chip DDR5 transfers.

---

### 1.2 Consolidated Experiment Table

| Experiment Run / Config | Actual Batch Size | Tokens Processed | Compute Multiplier | Unique Experts per Batch | Compute Time | DDR5 Transfer Time | Exposed Memory Stall | End-to-End Latency | Source Artifact Path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **E1-A: `exp_b1_l128`** | 1 | 1 | 1.0x (None) | 16 | 2.06 ms | 62.42 ms | 62.42 ms | 64.48 ms | `LLMSimulator/results/endsem_phase_e/e1_a_synthetic_baseline/batch_size_sweep.csv` |
| **E1-A: `exp_b2_l128`** | 2 | 2 | 1.0x (None) | 16 (identical) | 2.07 ms | 62.42 ms | 62.42 ms | 64.48 ms | `LLMSimulator/results/endsem_phase_e/e1_a_synthetic_baseline/batch_size_sweep.csv` |
| **E1-A: `exp_b4_l128`** | 4 | 4 | 1.0x (None) | 16 (identical) | 2.07 ms | 62.42 ms | 62.42 ms | 64.49 ms | `LLMSimulator/results/endsem_phase_e/e1_a_synthetic_baseline/batch_size_sweep.csv` |
| **E1-A: `exp_b8_l128`** | 8 | 8 | 1.0x (None) | 16 (identical) | 2.09 ms | 62.42 ms | 62.42 ms | 64.51 ms | `LLMSimulator/results/endsem_phase_e/e1_a_synthetic_baseline/batch_size_sweep.csv` |
| **Phase E: `B1` (MoE Anywhere)** | 2 | 256 | 1.0x (None) | NOT RECORDED | NOT RECORDED | NOT RECORDED | 325,048.13 ms | 177,851.00 ms | `LLMSimulator/docs/phase_e/FINAL_BASELINE_RESULTS.csv` |
| **Phase E: `B2` (MoE Anywhere)** | 4 | 512 | 1.0x (None) | NOT RECORDED | NOT RECORDED | NOT RECORDED | 717,441.46 ms | 506,134.00 ms | `LLMSimulator/docs/phase_e/FINAL_BASELINE_RESULTS.csv` |
| **Phase E: `B3` (MoE Anywhere)** | 2 | 512 | 1.0x (None) | NOT RECORDED | NOT RECORDED | NOT RECORDED | 685,875.53 ms | 381,603.00 ms | `LLMSimulator/docs/phase_e/FINAL_BASELINE_RESULTS.csv` |
| **Phase E: `B4` (MoE Anywhere)** | 4 | 1024 | 1.0x (None) | NOT RECORDED | NOT RECORDED | NOT RECORDED | 1,531,237.30 ms | 1,092,650.00 ms | `LLMSimulator/docs/phase_e/FINAL_BASELINE_RESULTS.csv` |
| **Phase 2F: `B-M1` (Baseline)** | 1 | 4 | 1.0x (None) | 2,021 demands | 119.46 ms | 7,884.24 ms | 7,884.24 ms | 8,003.70 ms | `LLMSimulator/docs/final_campaign/phase2f/PHASE2F_SUMMARY.csv` |
| **Phase 2F: `FWD-S-H2-M1`** | 1 | 4 | 1.0x (None) | 2,021 demands | 119.46 ms | 7,884.24 ms | 7,884.24 ms | 8,003.70 ms | `LLMSimulator/docs/final_campaign/phase2f/PHASE2F_SUMMARY.csv` |
| **Phase 2F: `FWD-S-H2-M4`** | 1 | 4 | 4.0x (Artificial) | 2,021 demands | 439.67 ms | 7,884.24 ms | 7,880.34 ms | 8,320.01 ms | `LLMSimulator/docs/final_campaign/phase2f/PHASE2F_SUMMARY.csv` |
| **Phase 2F: `FWD-S-H2-M12`** | 1 | 4 | 12.0x (Artificial) | 2,021 demands | 1,310.20 ms | 7,884.24 ms | 7,876.44 ms | 9,186.64 ms | `LLMSimulator/docs/final_campaign/phase2f/PHASE2F_SUMMARY.csv` |
| **Phase 2F: `FWD-S-H2-M16`** | 1 | 4 | 16.0x (Artificial) | 2,021 demands | 1,754.57 ms | 7,884.24 ms | 7,871.39 ms | 9,625.96 ms | `LLMSimulator/docs/final_campaign/phase2f/PHASE2F_SUMMARY.csv` |
| **Duplex A100: Mixtral Decode** | 32 | 160 (32/step) | 1.0x (None) | NOT RECORDED | NOT RECORDED | NOT RECORDED | NOT RECORDED | 5.51 ms / step | `LLMSimulator/reproduction_A100/results/a100_duplex/csv/...maxbatch32...csv` |

---

## 2. Phase 2: Trace of Compute-Time Calculation in LLMSimulator

Forensic tracing across `src/hardware/linear_impl.cpp`, `src/module/module_graph.cpp`, `src/module/linear.cpp`, and `src/hardware/cluster.cpp` answers the 10 core questions with exact file and line references:

### 1. Where is the execution time of a GEMM calculated?
- **File & Function:** `src/hardware/linear_impl.cpp:16-40` in `LinearExecutionGPU()`, and `src/hardware/linear_impl.cpp:379-420` in `BatchedLinearExecutionGPU()`.
- **Exact Code Lines (23-33):**
  ```cpp
  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;
  hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;
  ```

### 2. Which quantities determine it?
The execution time of a GEMM is determined strictly by:
- $M$: Number of tokens processed by this expert in the batch (`input->shape[0]`).
- $K$: Input hidden dimension (`input->shape[1]`, e.g. 7,168 in DeepSeek-V3, 4,096 in Mixtral).
- $N$: Output projection dimension (`weight->shape[1]`, e.g. 2,048 for DeepSeek-V3 routed intermediate, 14,336 for Mixtral).
- $	ext{total\_flops} = 2.0 	imes M 	imes K 	imes N$.
- `compute_peak_flops`: A static hardware constant from `device->config.compute_peak_flops` (e.g. $312 	imes 10^{12}	ext{ FLOPS}$ for NVIDIA A100 FP16 Tensor Core, or $989.4 	imes 10^{12}	ext{ FLOPS}$).

### 3. Is computation time calculated analytically, from a lookup table, or measured?
It is calculated **purely analytically**:
$$	ext{compute\_duration} = rac{2.0 	imes M 	imes K 	imes N}{	ext{compute\_peak\_flops}} 	imes 10^9	ext{ ns}$$
There are no kernel execution measurements, no lookup tables, and no CUDA runtime profiling.

### 4. How does increasing batch size change the dimensions of the GEMM?
In `src/module/route.cpp:230-236`:
```cpp
for (int expert_idx = 0; expert_idx < num_expert; expert_idx++) {
  std::vector<int> shape = {
      sequences_metadata->num_token_in_expert[expert_idx],
      input[0]->shape[1]};
  Tensor::Ptr output =
      get_activation("expert_input_" + std::to_string(expert_idx), shape);
  output_vec.push_back(output);
}
```
Increasing the batch size increases `sequences_metadata->num_token_in_expert[expert_idx]`, which directly sets the row dimension $M$ of the input tensor passed to `Linear::forward`. If 4 tokens route to Expert 5, $M=4$. If 0 tokens route to Expert 5, $M=0$, `input->getSize() == 0`, and execution returns early with zero duration (`src/hardware/linear_impl.cpp:36-38`).

### 5. Does the simulator model improved GPU utilization or efficiency at larger batch sizes?
**NO.** The simulator assumes a constant peak compute rate (`compute_peak_flops`) regardless of $M$.
- For $M=1$, it assumes $100\%$ Tensor Core peak throughput.
- For $M=64$, it assumes $100\%$ Tensor Core peak throughput.
The variable `exec_status.compute_util` (`linear_impl.cpp:239-240`) is simply the calculated ratio of compute duration to total duration ($	ext{compute\_duration} / 	ext{total\_duration}$). It does **not** model lower Tensor Core efficiency or memory latency bubbles at small batch sizes.

### 6. Where is `exec_status.total_duration` generated?
In `src/hardware/linear_impl.cpp:234-237`:
```cpp
exec_status.total_duration =
    std::max(exec_status.compute_duration, exec_status.memory_duration);
```
When `compute_time_multiplier` is active (lines 216-233), `onchip_duration = std::max(exec_status.compute_duration, onchip_mem) * multiplier`, and `exec_status.total_duration = onchip_duration + offchip_stall`.

### 7. How does `status.device_time` advance?
In `src/module/module_graph.cpp:249`:
```cpp
status.device_time += exec_status.total_duration;
```
As each module in the computational graph completes, `status.device_time` advances monotonically by `exec_status.total_duration`.

### 8. Is compute time calculated as `total execution time - transfer time` anywhere?
**NO.** Compute time is never computed as a subtraction from total time.

### 9. What does the simulator report as compute time?
The simulator computes `compute_duration` directly from FLOPs and peak FLOPS.
In the output CSV headers (`cluster.cpp:448`), columns `fc_comp`, `attn_comp`, and `moe_comp` report **energy in nanoJoules (`energy_nJ`)**, not time. Compute duration is accumulated internally in `exec_status.compute_duration` and on the `TimeBoard`.

### 10. Are transfer time and compute time independently accounted for?
**YES, but they interact via `std::max()`.**
- `compute_duration` is strictly independent: calculated solely from FLOPs.
- `memory_duration` is strictly independent: calculated from HBM access latency plus off-chip DDR5 transfer stalls.
- However, `total_duration = std::max(compute_duration, memory_duration)` means that when DDR5 transfer time ($3.901	ext{ ms}$) dwarfs single-token compute duration ($0.022	ext{ ms}$), memory stall completely dictates `total_duration`.

---

## 3. Phase 3: Relationship Between Batch Size, Routing Diversity, and Expert Transfers

### 3.1 Illustrative Comparison (32 Tokens, Top-K = 2, DeepSeek-V3 / Mixtral Geometry)
- **Token Count:** 32 tokens.
- **Top-K:** 2 experts per token.
- **Total Routing Assignments:** $32 	imes 2 = 64$ expert evaluations.
- **Expert Size:** $88.08	ext{ MB}$ (DeepSeek-V3 FP16) or $58.7	ext{ MB}$ (Mixtral 8x7B).
- **DDR5 Calibrated Bandwidth:** $22.578	ext{ GB/s}$.
- **Isolated Transfer Duration:** $3.90116	ext{ ms}$ ($88.08	ext{ MB}$) or $2.599	ext{ ms}$ ($58.7	ext{ MB}$).

```
Case A: Maximum Expert Clustering (2 Shared Experts)
  Tokens: 32 | Unique Experts: 2 | M per Expert: 32
  [Compute: 32x FLOPs (0.71 ms) | DDR5 Transfer: 2 Experts (7.80 ms)]
  Compute-to-Transfer Ratio: High (0.71 ms / 7.80 ms = 0.091)

Case B: Disjoint Routing Spread (64 Unique Experts)
  Tokens: 32 | Unique Experts: 64 | M per Expert: 1
  [Compute: Identical FLOPs (0.71 ms) | DDR5 Transfer: 64 Experts (249.67 ms)]
  Compute-to-Transfer Ratio: Severely Memory-Bound (0.71 ms / 249.67 ms = 0.0028)
```

### 3.2 Scientific Answers to the 6 Core Mechanics

1. **Why increasing tokens increases computation:**
   Every token routed to an expert requires three GEMMs (`gate_proj`, `up_proj`, `down_proj`). Because FLOPs scale as $2.0 	imes M 	imes K 	imes N$, processing 32 tokens requires exactly $32	imes$ more matrix multiply-accumulate operations than processing 1 token.

2. **Why increasing tokens does not necessarily increase unique expert transfers:**
   Expert weights are transferred at the granularity of the entire expert parameter matrix ($88.08	ext{ MB}$). Once loaded into HBM, that matrix serves all tokens routed to it in the batch. If 32 tokens route to 2 experts, only 2 expert weights ($176.16	ext{ MB}$) are transferred, identical to the transfer volume of 1 token routing to 2 experts.

3. **Under what conditions expert weights are reused from HBM:**
   - **Intra-Batch Reuse:** Multiple tokens within the same decoding step route to the same expert. The first token triggers the demand or prefetch; subsequent tokens in the batch hit HBM resident state with zero memory stall.
   - **Inter-Batch Reuse:** Consecutive decoding steps demand previously loaded experts that remain resident in HBM because the cache has not evicted them.

4. **Whether the current simulator models this reuse correctly:**
   **YES.** In `src/hardware/linear_impl.cpp:64-106`, `LinearExecutionGPU` queries `device->is_expert_resident()`. If an expert was loaded by an earlier token in the batch, `is_demand_miss` evaluates to `false`, `weight_memory_cost = 0`, and the layer executes with zero memory stall.

5. **Whether routing diversity can offset the benefit of larger batches:**
   **YES (Critically).** If 32 tokens activate 64 distinct experts, the off-chip DDR5 transfer engine must serialize 64 expert transfers ($64 	imes 3.901	ext{ ms} = 249.67	ext{ ms}$). The compute work ($0.71	ext{ ms}$) is completely overwhelmed by the $249.67	ext{ ms}$ transfer burden. High routing diversity transforms batched inference into an extreme memory-bound regime unless HBM capacity is large enough to retain the expanded working set.

6. **Joint Interaction on Compute-to-Memory Ratio:**
   $$	ext{Compute-to-Memory Ratio} = rac{\sum_{e=1}^{U} 	ext{FLOPs}(M_e)}{\sum_{e \in 	ext{Misses}} 	ext{Bytes}(e)} \propto rac{B 	imes K}{U_{	ext{miss}}}$$
   - When batch size $B$ increases with high expert locality ($U_{	ext{miss}} \ll B 	imes K$), the workload rapidly becomes compute-bound.
   - When batch size $B$ increases with high routing diversity ($U_{	ext{miss}} pprox B 	imes K$), the workload remains memory-bound.

---

## 4. Phase 4: How to Configure a Genuinely Compute-Bound MoE Workload

### 4.1 Theoretical Compute-Bound Condition
Prefetching achieves full latency hiding when the available compute window across lookahead horizon $H$ matches or exceeds the physical DDR5 transfer latency of the prefetch candidate budget $B_{	ext{cand}}$:
$$W_{	ext{comp}}(H) \ge B_{	ext{cand}} 	imes t_{	ext{transfer}}$$
For $H=2$ and $B_{	ext{cand}}=1$ candidate ($t_{	ext{transfer}} = 3.90116	ext{ ms}$):
$$W_{	ext{comp}}(H=2) = 2 	imes t_{	ext{stage\_comp}} \ge 3.90116	ext{ ms} \implies t_{	ext{stage\_comp}} \ge 1.9506	ext{ ms}$$
At batch size $B=1$, single-token decode stage compute duration is only $\sim 0.18317	ext{ ms}$ ($10.65	imes$ deficit).

---

### 4.2 Distinguishing Realistic vs. Artificial Compute Scaling

| Characteristic | Option A: Realistic Compute Scaling (Actual Batch Size) | Option B: Artificial Compute Scaling (`COMPUTE_MULTIPLIER`) |
|---|---|---|
| **Mechanism** | Scales `max_batch_size` ($B \in \{16, 32, 64\}$) | Sets `COMPUTE_MULTIPLIER = 12.0` or `16.0` |
| **GEMM Dimensions** | Increases $M$ row dimension ($M = B$) | Leaves $M=1$; divides `compute_peak_flops` by $M$ |
| **Real FLOP Count** | Scales linearly ($B 	imes 88	ext{ MFLOPs}$) | Constant ($88	ext{ MFLOPs}$) |
| **Activation Memory** | Scales linearly ($B 	imes (m \cdot k + m \cdot n)$) | Constant |
| **Routing Realism** | Models real multi-token expert dispersion | Keeps single-token routing trace |
| **Scientific Defensibility** | **High:** Reflects production serving dynamics | **Exploratory Only:** Simulates a slow GPU, not a batched GPU |

### Limitations of Option B as Evidence of Real Batch Performance:
1. It artificially stretches time without increasing memory traffic for activations or KV cache.
2. It fails to test the candidate queue and memory bus under real multi-sequence routing contention.
3. It does not stress HBM capacity with a multi-sequence working set.

---

### 4.3 Concrete Proposed Configurations to Achieve Compute-Bound Execution

#### Configuration 1: Batched Decode ($B = 32$, Real MoE Anywhere Trace)
- **Batch Size:** $B = 32$ concurrent decoding sequences.
- **Model:** DeepSeek-V3 Mixtral-style ($K=8$, $E=256$, 75 MoE stages).
- **GEMM Dimension:** $M pprox 32 	imes 8 / U pprox 4 	ext{ to } 8$ tokens per expert.
- **Stage Compute Duration:** $32 	imes 0.183	ext{ ms} pprox 5.86	ext{ ms}$ per stage.
- **Available Overlap Window ($H=2$):** $2 	imes 5.86	ext{ ms} = 11.72	ext{ ms}$.
- **Feasibility:** $11.72	ext{ ms} > 3.901	ext{ ms}$ ($3.0	imes$ margin). Allows prefetching up to $B_{	ext{cand}} = 3$ serialized candidates with zero stall.

#### Configuration 2: Batched Decode ($B = 16$, Hot-Expert Clustering)
- **Batch Size:** $B = 16$.
- **Stage Compute Duration:** $16 	imes 0.183	ext{ ms} pprox 2.93	ext{ ms}$ per stage.
- **Available Overlap Window ($H=2$):** $2 	imes 2.93	ext{ ms} = 5.86	ext{ ms}$.
- **Feasibility:** $5.86	ext{ ms} > 3.901	ext{ ms}$ ($1.5	imes$ margin). Yields 1 timely resident hit per stage with complete latency hiding.

#### Configuration 3: Prefill Phase Workload ($L_{	ext{prompt}} = 512	ext{ tokens}$, $B=1$)
- **Prompt Length:** 512 tokens processed simultaneously in prefill mode.
- **GEMM Dimension:** $M = 512 	imes 8 / 256 pprox 16$ tokens per expert.
- **Stage Compute Duration:** $pprox 93.7	ext{ ms}$ per stage.
- **Feasibility:** $93.7	ext{ ms} \gg 3.901	ext{ ms}$ ($24	imes$ margin). Strongly compute-bound; 100% of prefetch transfers complete timely.

---

## 5. Phase 5: Measurement Framework for Compute-Bound MoE Validation

A rigorous evaluation must independently log the following 12 metrics:

| Metric Category | Specific Measurement | Unit | Source Code Provider |
|---|---|---|---|
| **Computation** | Pure GPU Compute Duration | ms | Sum of `exec_status.compute_duration` |
| **Transfer** | Total DDR5 Transfer Time | ms | `DDR5TransferEngine::metrics.demand_service_time_ns + prefetch_service_time_ns` |
| **Traffic** | Total DDR5 Bytes Transferred | MB | `DDR5TransferEngine::metrics.total_bytes_transferred` |
| **Working Set** | Unique Experts Demanded | Count | Distinct expert IDs in `expert_routing.csv` per stage |
| **Residency** | Natural HBM Cache Hits | Count | `resident_experts` hits with `is_prefetched = false` |
| **Timeliness** | Timely Prefetch Hits ($H_{	ext{timely}}$) | Count | `DemandOutcome::TIMELY_RESIDENT` in `demand_outcomes` |
| **Attachment** | Late In-Flight Attachments ($A_{	ext{late}}$) | Count | `DemandOutcome::LATE_IN_FLIGHT` in `demand_outcomes` |
| **Misses** | Unprefetched Demand Misses | Count | `DemandOutcome::NOT_PREFETCHED` in `demand_outcomes` |
| **Stall** | Exposed Memory Stall Time | ms | `tr->completion_time_ns - start_time` in `LinearExecutionGPU` |
| **Overlap** | Hidden Transfer Overlap Duration | ms | $\text{Transfer Duration} - \text{Exposed Stall}$ |
| **Latency** | End-to-End Decoding Step Latency | ms | `status.device_time` in `Cluster::runIterationMixed` |
| **Throughput** | Effective Token Generation Rate | tokens/s | $B / (\text{Step Latency in seconds})$ |

### Operating Regime Classification Criteria:
1. **Strongly Memory-Bound:**
   $$rac{	ext{Exposed Memory Stall}}{	ext{End-to-End Step Latency}} > 0.50$$
   Compute resources are idle for more than half the step waiting for DDR5 expert loads.
2. **Partially Overlapped:**
   $$0.05 < rac{	ext{Exposed Memory Stall}}{	ext{End-to-End Step Latency}} \le 0.50$$
   Transfers make progress concurrently with compute, but exposed stall remains due to queue serialization or insufficient compute window.
3. **Genuinely Compute-Bound:**
   $$rac{	ext{Exposed Memory Stall}}{	ext{End-to-End Step Latency}} \le 0.05 \quad 	ext{and} \quad 	ext{End-to-End Latency} pprox 	ext{Compute Duration}$$
   Memory stalls are virtually eliminated ($\le 5\%$); throughput is governed strictly by GPU TFLOPS.

---

## 6. Phase 6: Final Recommendations and Implementation Roadmap

### 6.1 Core Audit Conclusions
1. **Existing Experiments Do Not Demonstrate Compute-Bound Execution:**
   - Single-token decode runs ($B=1$) are **$97\%$ memory-bound** ($0.183\text{ ms}$ compute vs $3.901\text{ ms}$ transfer).
   - Historical $B \in \{2, 4, 8\}$ synthetic runs in E1-A suffered from a caching artifact that forced all sequences to select identical experts, disguising routing diversity.
   - Phase 2F demonstrated latency hiding using an **artificial multiplier ($M=12\times, 16\times$)**, proving architectural feasibility but not real batch-size performance.
2. **LLMSimulator IS Capable of Representing Real Batched Execution:**
   The multi-sequence scheduling pipeline (`Scheduler::running_queue`), dynamic tensor shaping (`Route::forward`), and per-expert FLOP scaling in `LinearExecutionGPU` natively support real batch sizes $B \in \{8, 16, 32\}$.

### 6.2 Simulator Limitations That Must Be Addressed
1. **Synthetic Routing Cache Decoupling:**
   Modify `cached_routing` in `src/scheduler/sequence.cpp:270` to key on `(seq_id, token_id, cur_layer)` instead of `(token_id, cur_layer)` so that synthetic batches exhibit realistic routing diversity across concurrent sequences.
2. **Multi-Sequence Real Trace Generation:**
   Construct authentic multi-sequence trace files for DeepSeek-V3 for $B=8, 16, 32$ by merging independent sequence traces with distinct `seq_id` values.
3. **Dynamic Channel-Aware Budget Throttling:**
   Incorporate dynamic budget capping in `PrefetchCandidateQueue` based on the measured compute window to prevent $B_{\text{cand}}=4$ candidates from congesting the single DDR5 bus.

### 6.3 Proposed Follow-Up Experiment Matrix

| Experiment ID | Batch Size ($B$) | Horizon ($H$) | Predictor | Budget ($B_{\text{cand}}$) | Trace Source | Expected Regime | Primary Objective |
|---|---:|---:|---|---:|---|---|---|
| **EXP-REAL-B1** | 1 | 2 | none | 0 | MoE Suite Task 12 | Strongly Memory-Bound | Hardware baseline at batch size 1 |
| **EXP-REAL-B8** | 8 | 2 | none | 0 | Multi-seq MoE Suite | Partially Overlapped | Baseline throughput scaling |
| **EXP-REAL-B16** | 16 | 2 | none | 0 | Multi-seq MoE Suite | Approaching Compute-Bound | Baseline natural reuse inflection |
| **EXP-REAL-B32** | 32 | 2 | none | 0 | Multi-seq MoE Suite | Genuinely Compute-Bound | Compute-bound baseline |
| **EXP-PREF-B8** | 8 | 2 | CSTP | 2 | Multi-seq MoE Suite | Partially Overlapped | Evaluate prefetch stall reduction at $B=8$ |
| **EXP-PREF-B16** | 16 | 2 | CSTP | 2 | Multi-seq MoE Suite | Genuinely Compute-Bound | Confirm $H_{\text{timely}} \ge 1$ with real batching |
| **EXP-PREF-B32** | 32 | 2 | CSTP | 4 | Multi-seq MoE Suite | Genuinely Compute-Bound | Confirm complete latency hiding ($H_{\text{timely}} = 2$) |
