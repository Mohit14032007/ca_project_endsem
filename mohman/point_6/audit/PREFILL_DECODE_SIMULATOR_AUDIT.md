# Point 6 — Prefill and Decode Simulator Implementation Audit

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Phase 0: Feasibility & Trace Audit  
**Author:** Antigravity Engineering  
**Date:** November 10, 2026  
**Status:** Authoritative & Completed  

---

## 1. Executive Summary

This report delivers a comprehensive audit of `LLMSimulator`'s source code to establish how prompt prefill and token decode are modeled, how sequence length influences execution time and memory allocation, and whether variations in initial context length ($P$) can be simulated cleanly without modifying the core simulation engine.

### Key Architectural Discoveries
1. **Hardcoded Decode Mode in Evaluation Harness:**
   In [LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L208-L209), the execution flags are explicitly configured:
   ```cpp
   system_config.decode_mode = true;
   system_config.prefill_mode = false;
   ```
   The trace replay engine is architected specifically for autoregressive token generation.

2. **Decoupled Trace Replay Indexing:**
   In [LLMSimulator/src/scheduler/sequence.cpp](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L248-L250), sequence indexing calculates:
   ```cpp
   int rel_token_id = token_id - seq->input_len;
   if (rel_token_id < 0) rel_token_id = 0;
   size_t table_idx = (size_t)rel_token_id * num_layer + cur_layer;
   ```
   Because `token_id` is initialized to `input_len` and increments at each decode step ($t = 0, 1, \dots, T-1$), `rel_token_id` simplifies identically to $t$. **This decouples the routing trace index from the configured context length $P$.** Changing `input_len` in YAML shifts memory and attention scaling without perturbing expert routing replay.

3. **Dynamic KV-Cache Scaling and HBM Displacement:**
   In [LLMSimulator/src/hardware/cluster.cpp](file:///home/mohman/ca_project/LLMSimulator/src/hardware/cluster.cpp#L255-L270), KV-cache allocation is directly proportional to $(P + T)$. For DeepSeek-V3 with Multi-Head Latent Attention (MLA), each token requires $46,080$ bytes ($45.0$ KiB) per sequence across 40 layers. At batch size $B=32$, increasing $P$ from 128 to 4,096 expands the KV cache from $0.18$ GiB to $5.62$ GiB, directly squeezing available HBM cache space for expert weights.

4. **Attention Latency Scaling:**
   In [LLMSimulator/src/hardware/attention_gen_impl.cpp](file:///home/mohman/ca_project/LLMSimulator/src/hardware/attention_gen_impl.cpp#L60-L82), decode attention FLOPs and KV memory traffic scale linearly with $n = P + t + 1$. Thus, higher initial context lengths increase per-step attention compute and DRAM traffic.

---

## 2. Codebase Implementation Audit

### 2.1 Evaluation Driver: `test_deepseekv3_mixtral_style.cpp`
The primary driver manages the simulation cycle:
- **Parameter Ingestion (Lines 125–126):**
  ```cpp
  int input_len = config["simulation"]["input_len"].as<int>();
  int output_len = config["simulation"]["output_len"].as<int>();
  ```
- **Operational Mode Enforcement (Lines 208–209):**
  ```cpp
  system_config.decode_mode = true;
  system_config.prefill_mode = false;
  ```
- **Sequence Instantiation (Lines 439–444):**
  ```cpp
  Sequence::Ptr new_seq = Sequence::Create(b, input_len, seq_out_len);
  if (system_config.decode_mode) {
    new_seq->current_len = input_len;
  }
  new_seq->get_expert_from_list = true;
  scheduler->sequence_queue.push_back(new_seq);
  ```
- **Per-Token Generation Loop (Lines 492–522):**
  ```cpp
  for (int step = 0; step < iter; step++) {
    auto metadata = scheduler->setMetadata();
    // Record token start with absolute position: input_len + step
    for (int s = 0; s < max_batch_size; s++) {
      PrefetchMetricsCollector::getInstance().record_token_start(
          0, s, input_len + step, step, static_cast<double>(step_start_dev_time));
    }
    cluster->run(metadata);
    // ...
  }
  ```

### 2.2 Trace Loader & Routing Mapping: `scheduler.cpp` & `sequence.cpp`
- **Trace Ingestion ([scheduler.cpp lines 662–717](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L662-L717)):**
  `loadRealTrace` reads records with format `(seq_id, token_id, layer_id, topk_rank, expert_id)`:
  ```cpp
  size_t table_idx = (size_t)token_id * num_layer + layer_id;
  auto& exp_vec = sequences_info[seq_id]->expert_list[table_idx];
  exp_vec[topk_rank - 1] = expert_id;
  ```
  Here `token_id` is a relative generation index in $[0, \text{iter\_tokens}-1]$.
- **Execution Replay ([sequence.cpp lines 248–254](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L248-L254)):**
  When executing layer `cur_layer` at token `token_id`:
  ```cpp
  int rel_token_id = token_id - seq->input_len;
  if (rel_token_id < 0) rel_token_id = 0;
  size_t table_idx = (size_t)rel_token_id * num_layer + cur_layer;
  const auto& routed = scheduler->sequences_info[seq_id]->expert_list[table_idx];
  ```
  Because `token_id = seq->input_len + step`, we have:
  $$\text{rel\_token\_id} = (\text{input\_len} + \text{step}) - \text{input\_len} = \text{step}$$
  The trace index is completely invariant to `input_len`.

### 2.3 Memory Sizing & KV-Cache Footprint: `cluster.cpp`
In `Cluster::checkMemorySize` ([cluster.cpp lines 255–270](file:///home/mohman/ca_project/LLMSimulator/src/hardware/cluster.cpp#L255-L270)), when memory capacity limits are enforced:
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
- Compressed dimension per layer = $512 + 64 = 576$ components.
- With $L = 40$ layers and $\text{precision\_byte} = 2$ (FP16):
  $$\text{Bytes per token per sequence} = 576 \times 40 \times 2 = 46,080 \text{ bytes} \approx 45.0 \text{ KiB}$$

#### Total KV Cache Footprint Matrix ($B \times P$)

| Context Length ($P$) | $B = 1$ | $B = 14$ | $B = 32$ | Impact on 35 GiB HBM Budget |
| :---: | :---: | :---: | :---: | :---: |
| **$P = 128$** | $5.76$ MiB | $80.64$ MiB ($0.079$ GiB) | $184.32$ MiB ($0.180$ GiB) | Negligible displacement ($< 0.5\%$) |
| **$P = 512$** | $23.04$ MiB | $322.56$ MiB ($0.315$ GiB) | $737.28$ MiB ($0.720$ GiB) | Minor displacement ($~2.0\%$) |
| **$P = 2,048$** | $92.16$ MiB | $1.26$ GiB | $2.88$ GiB | Moderate displacement ($~8.2\%$) |
| **$P = 4,096$** | $184.32$ MiB | $2.52$ GiB | $5.76$ GiB | Severe displacement ($~16.5\%$) |
| **$P = 8,192$** | $368.64$ MiB | $5.04$ GiB | $11.52$ GiB | Extreme displacement ($~32.9\%$) |

*Note: In non-MLA architectures (e.g., standard MHA with 128 heads and 128 head dim), KV cache size per token would be $2 \times 128 \times 128 \times 40 \times 2 = 2,621,440$ bytes ($2.50$ MiB/token), consuming $80$ GiB at $P=1024, B=32$ and completely exhausting memory.*

### 2.4 Attention Latency Modeling: `attention_gen_impl.cpp`
In [LLMSimulator/src/hardware/attention_gen_impl.cpp lines 60–80](file:///home/mohman/ca_project/LLMSimulator/src/hardware/attention_gen_impl.cpp#L60-L80):
- Sequence active length:
  $$n = \text{seq->current\_len} + \text{seq->num\_process\_token} = P + t + 1$$
- Attention scoring FLOPs per KV head:
  $$\text{FLOPs} = m \cdot k \cdot n \cdot 2.0 \cdot \text{attention\_group\_size} = 1 \times 128 \times (P + t + 1) \times 2 \times 1$$
- Attention memory traffic:
  $$\text{Memory} = \left(m \cdot k \cdot \frac{N_{\text{heads}}}{N_{\text{kv}}} + k \cdot n + m \cdot n \cdot \frac{N_{\text{heads}}}{N_{\text{kv}}}\right) \times \text{precision\_byte}$$
Both FLOPs and DRAM traffic scale as $\mathcal{O}(P + t)$. Consequently, as initial context length $P$ grows, the non-MoE attention stage duration increases linearly, altering the available time window for overlapping asynchronous expert transfers!

---

## 3. Comparison: Prefill vs. Decode Simulation

| Architectural Feature | Summary / Prefill Mode (`attention_sum_impl.cpp`) | Generation / Decode Mode (`attention_gen_impl.cpp`) |
| :--- | :--- | :--- |
| **Active Token Count ($m$)** | $m = P$ (entire prompt processed in parallel) | $m = 1$ (single token generated per sequence) |
| **Attention Complexity** | Quadratic $\mathcal{O}(P^2)$ | Linear $\mathcal{O}(P + t)$ per step |
| **MoE Layer Activation** | Evaluates all $P$ tokens simultaneously | Evaluates $B$ tokens simultaneously ($1$ per sequence) |
| **Trace Replay Support** | **None.** No multi-token prompt routing traces exist. | **Complete.** Contiguous step-by-step routing traces exist. |
| **Driver Implementation** | Stubbed in `test_deepseekv3_mixtral_style.cpp` (`prefill_mode = false`) | Fully functional and validated across Points 3, 4, and 5 |

---

## 4. Simulator Constraints and Context Limits

1. **Max Sequence Length Boundary:**
   In [LLMSimulator/src/scheduler/scheduler.cpp line 137](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L137):
   ```cpp
   assertTrue(model_config.input_len < model_config.max_seq_len, "Invalid input_len");
   if (model_config.input_len + model_config.output_len > model_config.max_seq_len) {
     model_config.output_len = model_config.max_seq_len - model_config.input_len;
   }
   ```
   In [model_config.h line 108](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h#L108), `deepseekV3.max_seq_len` is configured as **131,072** (128K).
   Therefore, sequence lengths up to $P = 8,192$ or $16,384$ are well within architectural limits.

2. **Memory Limit Toggle (`mem_cap_limit`):**
   When `simulation.mem_cap_limit: false`, the simulator simulates an unconstrained HBM address space for KV cache without dynamic batch eviction. When `mem_cap_limit: true`, the simulator dynamically verifies whether $M_{\text{KV}} + M_{\text{weights}} + M_{\text{activations}} \le M_{\text{capacity}}$.

3. **Prefetch Lookahead and Budget Independence:**
   Neither the prefetch prediction engine (`LivePredictionEngine`) nor the candidate queue (`PrefetchCandidateQueue`) depends on `input_len`. They operate on stage indices ($l \in [0, L-1]$) and decode step indices ($t \in [0, T-1]$). This guarantees that lookahead $H$ and budget $K$ can be evaluated orthogonally to context length $P$.

