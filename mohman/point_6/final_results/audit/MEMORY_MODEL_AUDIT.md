# Point 6 — Memory Model Implementation Audit and Context Scaling Mechanics

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Sensitivity Analysis Consolidation  
**Author:** Antigravity Engineering  
**Date:** November 10, 2026  
**Status:** Authoritative & Completed  

---

## 1. Executive Summary

Before executing the context-length sensitivity sweep ($P \in \{128, 512, 2048, 4096\}$), this read-only audit establishes the exact physical and software mechanisms implemented in `LLMSimulator` governing memory allocation, KV-cache sizing, expert cache capacity, and attention execution timing.

### Core Discoveries
1. **Dedicated Expert Cache Budget:**
   In [LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp line 346](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L346), `device->cache_capacity_bytes` is assigned directly from the YAML parameter `system.optimization.expert_cache_capacity_bytes` (35 GiB = $37,580,963,840$ bytes). 
2. **Independent Expert Residency Management:**
   In [LLMSimulator/src/hardware/device.cpp lines 958–1054](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L958-L1054), `Device::mark_expert_resident()` enforces capacity limits by comparing `resident_bytes + expert_size > cache_capacity_bytes`. The tracking variable `resident_bytes` accounts solely for expert parameter weights. **The simulator does not dynamically subtract KV-cache memory from `cache_capacity_bytes` during decode execution.**
3. **Genuine Attention Compute & DRAM Scaling:**
   In [LLMSimulator/src/hardware/attention_gen_impl.cpp lines 60–80](file:///home/mohman/ca_project/LLMSimulator/src/hardware/attention_gen_impl.cpp#L60-L80), decode attention scoring FLOPs and KV memory traffic scale linearly with active sequence length $n = P + t + 1$. Consequently, increasing $P$ increases per-step attention execution time, widening the temporal window available to overlap asynchronous expert prefetching before the MoE stage commences.
4. **Invariant Trace Replay:**
   In [LLMSimulator/src/scheduler/sequence.cpp line 248](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L248), relative token indexing evaluates $\text{rel\_token\_id} = \text{token\_id} - \text{input\_len} = (P + t) - P = t$. Authentic empirical routing decisions from `e1-sage-a1.5.parquet` are replayed identically across all context lengths.

---

## 2. Forensic Investigation of the Nine Memory-Model Questions

### Q1: How `simulation.input_len` is Consumed
In [`test_deepseekv3_mixtral_style.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp):
- Loaded on Line 125: `int input_len = config["simulation"]["input_len"].as<int>();`
- Assigned to sequence metadata on Lines 439–441:
  ```cpp
  Sequence::Ptr new_seq = Sequence::Create(b, input_len, seq_out_len);
  if (system_config.decode_mode) {
    new_seq->current_len = input_len;
  }
  ```
- Serves as the base sequence offset during decode steps (Lines 502, 520):
  `token_id = input_len + step`

### Q2: How KV-Cache Memory is Calculated
In [`cluster.cpp lines 255–270`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/cluster.cpp#L255-L270), when Multi-Head Latent Attention (MLA) is active:
```cpp
if ((device->model_config.qk_rope_head_dim != 0) && (device->model_config.compressed_kv == true)) {
  kv_cache_size_per_seq = 1.0 *
    (device->model_config.input_len + device->model_config.output_len) *
    (device->model_config.kv_lora_rank + device->model_config.qk_rope_head_dim) *
    device->model_config.num_layers * device->model_config.precision_byte;
}
```
For DeepSeek-V3:
- $d_{\text{kv\_lora\_rank}} = 512$
- $d_{\text{qk\_rope\_head\_dim}} = 64$
- $L = 40$ layers, $\text{precision\_byte} = 2$ (FP16)
$$\text{KV Bytes per Token per Sequence} = (512 + 64) \times 40 \times 2 = 46,080 \text{ bytes} \approx 45.0 \text{ KiB}$$

### Q3: Where KV-Cache Allocations are Recorded
KV-cache allocations are evaluated analytically inside `Cluster::checkMemorySize` at the start of simulation. They are not instantiated as physical byte arrays in host memory, but their size is verified against total device DRAM capacity.

### Q4: Whether KV-Cache Memory Reduces Capacity Available to Expert Residency
**NO.** In the current `LLMSimulator` C++ codebase:
- `device->cache_capacity_bytes` is a static dedicated threshold ($35\text{ GiB}$).
- The eviction loop in `Device::mark_expert_resident()` checks only `resident_bytes + expert_size > cache_capacity_bytes`.
- The variable `resident_bytes` is updated strictly by `+expert_size` on insertion and `-ev_size` on eviction.
- **Therefore, KV-cache expansion does NOT dynamically shrink `cache_capacity_bytes` in the native simulation engine.**

### Q5: Whether `Device::mark_expert_resident()` Uses the Same Memory Budget
**NO.** `mark_expert_resident()` manages an isolated expert cache pool. It does not share a unified dynamic allocator with tensor activations or KV buffers.

### Q6: Whether Capacity Enforcement Accounts for Other Device Allocations
When `simulation.mem_cap_limit: false` (the validated setting in all project configurations), `Cluster::checkMemorySize` issues diagnostic printouts but does not curtail batch size or truncate memory pools.

### Q7: Whether Attention Computation and Memory Traffic Change with $P$
**YES.** In [`attention_gen_impl.cpp lines 60–80`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/attention_gen_impl.cpp#L60-L80):
- $m = \text{num\_process\_token} = 1$
- $k = \text{head\_dim} = 128$
- $n = \text{current\_len} + m = P + t + 1$
- Scoring FLOPs: $\text{FLOPs} = 2 \times m \times k \times n \times \text{attention\_group\_size} \propto P + t$
- DRAM accessed: $\text{Memory} \approx (k \cdot n + m \cdot n) \times \text{precision\_byte} \propto P + t$
Both compute FLOPs and DRAM access volumes grow linearly with $P + t$.

### Q8: Whether Reported Attention Timing Changes with $P$
**YES.** As $P$ increases, `compute_duration` and `memory_duration` in `attention_gen_impl.cpp` increase. The execution status is accumulated into `stat.atten_gen` and output directly to the summary CSV, directly increasing simulated execution time.

### Q9: Whether the Same Routing Trace is Replayed for Every $P$
**YES.** In [`sequence.cpp line 248`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L248):
$$\text{rel\_token\_id} = \text{token\_id} - \text{input\_len} = (P + t) - P = t$$
Because `rel_token_id` evaluates identically to $t \in [0, T-1]$, the exact same token routing decisions are replayed regardless of whether $P = 128, 512, 2048,$ or $4096$.

---

## 3. Scientific Reporting Mandate for Context-Length Sensitivity

In strict adherence to Non-Negotiable Rule 10 ("Do not introduce an artificial expert-cache capacity reduction and present it as native simulator behavior"):
1. **Context-Length Sensitivity Measures Attention Duration & Prefetch Overlap:**
   Increasing $P$ increases the non-MoE attention stage execution time. This widens the available time window for asynchronous prefetch transfers to complete before the MoE stage demands the experts.
2. **Analytical vs. Simulated Capacity Separation:**
   We report the analytical KV-cache memory footprint ($M_{\text{KV}} \in [180\text{ MiB}, 5.625\text{ GiB}]$ for $B=32$) as a documented architectural reality of DeepSeek-V3 MLA.
3. **Transparent Eviction Disclosure:**
   We explicitly state that because `LLMSimulator` uses a dedicated expert cache budget ($35\text{ GiB}$), expert hit rates and reload counts remain invariant across $P$ sweeps, isolating the pure effect of attention compute latency and prefetch overlap on end-to-end token latency.

