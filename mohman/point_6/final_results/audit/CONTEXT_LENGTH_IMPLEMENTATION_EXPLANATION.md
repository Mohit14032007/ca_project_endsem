# Context Length in LLMSimulator and the MoE-Anywhere Trace Dataset

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Technical Implementation & Validation  
**Author:** Antigravity Engineering  
**Date:** November 11, 2026  
**Status:** Authoritative & Completed  

---

## 1. The Authentic MoE-Anywhere Dataset (`e1-sage-a1.5.parquet`)

The primary workload dataset utilized in this project is:
`LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet`

### Dataset Contents & Structure:
- **Total Prompt Sequences:** 74 distinct prompts (IDs 0 through 73).
- **Recorded Layers:** 40 MoE stages per token position.
- **Expert Routing:** Top-8 routed expert selections per token per layer (selected from 256 total routed experts).
- **Positions per Prompt:** 257 contiguous recorded token positions.
- **Position Tracking:** Absolute token positions are recorded in `token_pos`.
- **What is NOT in the dataset:**
  - The dataset does **not** contain original prompt text.
  - It does **not** contain raw tokenizer input IDs.
  - It does **not** contain router logits or prefill routing decisions.
  - It records solely the **decode-generation routing decisions** emitted by an authentic DeepSeek-V3 serving system.

### Inferred Initial Prompt Lengths:
The first recorded position for each prompt in the parquet file indicates when generation recording commenced:
$$\text{inferred\_initial\_P} = \min(\text{token\_pos})$$
Across the 74 prompts, $\text{inferred\_initial\_P}$ ranges from $13$ to $64$ tokens (mean $= 26.6$ tokens, median $= 24$ tokens). 

Because the natural prompt lengths ($P \in [13, 64]$) represent a narrow range that produces negligible differences in memory footprint and compute duration, evaluating context lengths up to $P = 4,096$ requires a controlled, synthetic initial-context experiment.

---

## 2. Implementation Execution Path in LLMSimulator

In LLMSimulator, changing `input_len` in the simulation configuration (e.g., $P \in \{128, 512, 2048, 4096\}$) executes through the following exact path:

### 2.1 Initialization (`test_deepseekv3_mixtral_style.cpp`)
When decode mode is enabled (`decode_mode: true`), sequence initialization sets:
```cpp
Sequence::Ptr new_seq = Sequence::Create(b, input_len, seq_out_len);
new_seq->current_len = input_len;
```
For sequence $b$ and generation step $t \in [0, T-1]$, the simulated token identifier is:
$$\text{token\_id} = \text{input\_len} + t$$

### 2.2 Trace Ingestion & Relative Indexing (`sequence.cpp` and `linear.cpp`)
When the sequence executes an MoE stage, expert selections are looked up from the pre-loaded trace using relative indexing:
```cpp
// In LLMSimulator/src/scheduler/sequence.cpp (lines 248-251)
int rel_token_id = token_id - seq->input_len;
if (rel_token_id < 0) rel_token_id = 0;
size_t table_idx = (size_t)rel_token_id * num_layer + cur_layer;
const auto& routed = scheduler->sequences_info[seq_id]->expert_list[table_idx];
```
Because $\text{token\_id} = \text{input\_len} + t$:
$$\text{rel\_token\_id} = (\text{input\_len} + t) - \text{input\_len} = t$$
$$\text{table\_idx} = t \times 40 + \text{cur\_layer}$$

### 2.3 Mathematical Trace Equivalence Guarantee
As a direct consequence of relative indexing:
$$\forall P \in \{128, 512, 2048, 4096\}, \quad \text{table\_idx}(t, \text{cur\_layer}, P) = t \times 40 + \text{cur\_layer}$$
The exact same routing record is retrieved from `expert_list` at step $t$ regardless of the value of $P$.

**Scientific Distinction:**
> **The controlled context sweep changes the assumed initial context length in LLMSimulator while replaying the same authentic MoE-Anywhere generated-token routing sequence. It does not reconstruct the original prompt or simulate its original prefill routing.**

---

## 3. Disentangling the Three Distinct Effects of Context Length

Varying $P$ in an MoE inference system influences hardware performance along three independent dimensions:

### Effect 1: Attention Compute Scaling and Prefetch Overlap Slack
In `LLMSimulator/src/hardware/attention_gen_impl.cpp`:
- Scoring and context projections in Multi-Head / Multi-Latent Attention scale proportionally to the cumulative sequence length:
  $$n = P + t + 1$$
- Attention FLOPs per token per layer scale as:
  $$\text{FLOPs} = 2 \times m \times k \times n \times \text{attention\_group\_size}$$
- Decode attention execution duration scales linearly with $P + t$.
- Because attention computation runs **immediately prior** to the MoE stage in each layer, scaling $P$ from $128$ to $4,096$ expands non-stall compute duration from $13,727.88\text{ ms}$ to $14,355.67\text{ ms}$ (+627.79 ms).
- This extra compute duration acts as **asynchronous prefetch overlap slack**, allowing background DDR5 transfers more time to arrive before the expert demand occurs.

### Effect 2: MLA KV-Cache Memory Footprint
DeepSeek-V3 utilizes Multi-Latent Attention (MLA), which compresses Key-Value representations into a low-rank latent vector:
- `kv_lora_rank` $= 512$
- `qk_rope_head_dim` $= 64$
- Latent elements per token per layer $= 512 + 64 = 576$ FP16 values
- Bytes per token per layer $= 576 \times 2 = 1,152\text{ bytes}$
- For 40 layers, the KV cache footprint per token is:
  $$40 \times 1,152 = 46,080\text{ bytes/token}$$

Across batch size $B=32$ and cumulative length $S$:
$$M_{\text{KV}} = 32 \times S \times 46,080\text{ bytes} = S \times 1,474,560\text{ bytes}$$

| Context Length Condition | Sequence Length ($S$) | Exact Bytes | Decimal (GB) | Binary (GiB) |
|---|---|---|---|---|
| Natural Minimum ($P=13, T=0$) | 13 | 19,169,280 | 0.0192 GB | 0.0179 GiB (18.28 MiB) |
| Natural Maximum ($P=64, T=0$) | 64 | 94,371,840 | 0.0944 GB | 0.0879 GiB (90.00 MiB) |
| Natural Delta ($\Delta P = 51$) | 51 | 75,202,560 | 0.0752 GB | 0.0700 GiB (71.72 MiB) |
| Controlled $P=128$, Decode Start | 128 | 188,743,680 | 0.1887 GB | 0.1758 GiB (180.0 MiB) |
| Controlled $P=128$, Horizon ($P+128$) | 256 | 377,487,360 | 0.3775 GB | 0.3516 GiB (360.0 MiB) |
| Controlled $P=512$, Decode Start | 512 | 754,974,720 | 0.7550 GB | 0.7031 GiB (720.0 MiB) |
| Controlled $P=512$, Horizon ($P+128$) | 640 | 943,718,400 | 0.9437 GB | 0.8789 GiB (900.0 MiB) |
| Controlled $P=2048$, Decode Start | 2048 | 3,019,898,880 | 3.0199 GB | 2.8125 GiB (2,880.0 MiB) |
| Controlled $P=2048$, Horizon ($P+128$) | 2176 | 3,208,642,560 | 3.2086 GB | 2.9883 GiB (3,060.0 MiB) |
| Controlled $P=4096$, Decode Start | 4096 | 6,039,797,760 | 6.0398 GB | 5.6250 GiB (5,760.0 MiB) |
| Controlled $P=4096$, Horizon ($P+128$) | 4224 | 6,228,541,440 | 6.2285 GB | 5.8008 GiB (5,940.0 MiB) |

### Effect 3: Interaction with Dedicated Expert HBM Capacity
In physical GPUs with unified memory, KV cache growth dynamically consumes HBM, reducing the space remaining for model weights and expert caching.

In `LLMSimulator`, however:
- The expert cache has a dedicated, fixed partition:
  ```cpp
  // LLMSimulator/src/hardware/device.h:301
  long long cache_capacity_bytes = 37580963840LL; // 35 GiB
  ```
- Cache admission in `device.cpp` checks:
  ```cpp
  while (resident_bytes + expert_size > cache_capacity_bytes && !lru_expert_list.empty())
  ```
- `resident_bytes` tracks solely active expert tensors ($88,080,384$ bytes each).
- KV cache tensors (`k_cache` and `v_cache`) are allocated independently and **do not decrement `cache_capacity_bytes`**.

**Conclusion:**
> **In this simulator configuration, increasing $P$ does not directly evict expert-cache entries. The context-length sensitivity experiment isolates the pure computational overlap slack effect enabled by attention scaling, unconfounded by memory-budget shrinkage.**

