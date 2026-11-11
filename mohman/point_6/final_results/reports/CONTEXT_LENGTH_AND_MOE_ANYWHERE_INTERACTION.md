# Context Length and MoE-Anywhere Trace Interaction: Architectural Attribution and Validity

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Technical Analysis Report  
**Author:** Antigravity Engineering  
**Date:** November 11, 2026  
**Status:** Authoritative & Completed  

---

## 1. Executive Summary

This report provides the foundational architectural and empirical explanation of how context length ($P$) is modeled in our system, how it interacts with the authentic MoE-Anywhere trace dataset, and how the experimental results must be interpreted scientifically.

### Core Scientific Conclusions:
1. **Decode Routing Invariance:** Across all tested context lengths ($P \in \{128, 512, 2048, 4096\}$), the sequence of generated-token expert routing decisions is **100% invariant**. The exact same expert selections are executed step-for-step at every layer across all 32 sequences (cryptographically proven via SHA-256 hash match over all $1,310,720$ expert selections).
2. **Computational Slack Mechanism:** In `LLMSimulator`, decode attention FLOPs and KV-cache read volume scale proportionally to cumulative sequence length ($n = P + t + 1$). Consequently, increasing $P$ from $128$ to $4,096$ widens the attention compute execution duration by $+627.79\text{ ms}$, providing an extended temporal window during which asynchronous DDR5 prefetch transfers can progress.
3. **Dedicated Cache Budget:** The simulator dedicates a static $35\text{ GiB}$ partition exclusively for expert weights. Dynamic KV-cache expansion does not contract this budget. Context length sensitivity therefore isolates the pure effect of attention compute overlap slack, free from capacity shrinkage.
4. **Scope of the Evaluation:** This experiment answers: *“How does the amount of pre-existing context affect the usefulness of asynchronous expert prefetching when the generated-token routing sequence is held constant?”* It does not evaluate how differing prompt lengths alter router logits or prefill routing decisions.

---

## 2. Dataset Mechanics: The MoE-Anywhere Trace

### 2.1 The Authentic Trace (`e1-sage-a1.5.parquet`)
The MoE-Anywhere dataset is a production routing trace captured from authentic DeepSeek-V3 execution:
- 74 prompts (IDs 0–73).
- 40 MoE layers per token.
- Top-8 expert selections per layer (out of 256 routed experts).
- 257 contiguous generated token positions per prompt.
- Absolute position recorded in `token_pos`.

### 2.2 Natural Inferred Context vs. Controlled Sweep
In the raw trace, the minimum recorded token position represents when decode logging started:
$$\text{inferred\_initial\_P} = \min(\text{token\_pos})$$
Across the 74 prompts, this natural inferred prompt length ranges from $13$ to $64$ tokens.
- For batch size $B=32$, the natural context length variation ($P=13$ vs. $P=64$) represents a KV-cache footprint difference of only $71.72\text{ MiB}$ ($75.2\text{ MB}$), which produces sub-millisecond differences in attention execution duration.
- To rigorously test modern long-context regimes (where $P$ reaches thousands of tokens), we conducted a controlled synthetic initial-context sweep:
  $$P \in \{128, 512, 2048, 4096\}$$
  This sweep systematically modulates attention compute duration across authentic decode routing steps.

---

## 3. Simulator Replay Engine & Relative Indexing

### 3.1 Trace Indexing Math
In LLMSimulator, each sequence is initialized in decode mode with `input_len = P`:
```cpp
// Sequence initialization
new_seq->current_len = input_len;
```
At generation step $t \in [0, 127]$, the simulator sets:
$$\text{token\_id} = P + t$$
When the sequence reaches an MoE stage, expert selections are retrieved via:
```cpp
// Relative token index calculation
int rel_token_id = token_id - seq->input_len;
size_t table_idx = (size_t)rel_token_id * num_layer + cur_layer;
const auto& routed = scheduler->sequences_info[seq_id]->expert_list[table_idx];
```
Substituting $\text{token\_id} = P + t$ into the relative index formula:
$$\text{rel\_token\_id} = (P + t) - P = t$$
$$\text{table\_idx} = t \times 40 + \text{cur\_layer}$$

### 3.2 Implication: Identical Trace Replay
Because $\text{table\_idx}$ depends solely on generation step $t$ and layer index $\text{cur\_layer}$, changing $P$ has **zero effect on expert routing selection**:
$$\text{SelectedExperts}(t, \text{layer}, P_1) \equiv \text{SelectedExperts}(t, \text{layer}, P_2)$$
This guarantees that expert reuse patterns, demand skewness, and routing entropy remain perfectly controlled.

---

## 4. Physical Mechanisms of Context-Length Sensitivity

```
+---------------------------------------------------------------------------------------------------+
|                            CONTEXT LENGTH SENSITIVITY ATTRIBUTION                                  |
+-------------------+-----------------------+-----------------------+-------------------------------+
| Context Length P  | Compute Slack (ms)    | Coalesced Hits (LRU)  | Reloads Avoided vs Baseline   |
+-------------------+-----------------------+-----------------------+-------------------------------+
| P = 128           | 13,727.88 ms          | 1,417 hits (0.2589%)  | 1,317 reloads saved           |
| P = 512           | 13,783.40 ms          | 1,436 hits (0.2624%)  | 1,336 reloads saved           |
| P = 2048 (Anchor) | 14,026.23 ms          | 1,499 hits (0.2739%)  | 1,398 reloads saved           |
| P = 4096          | 14,355.67 ms          | 1,630 hits (0.2978%)  | 1,528 reloads saved           |
+-------------------+-----------------------+-----------------------+-------------------------------+
```

### 4.1 Temporal Overlap Slack Expansion
- As $P$ increases from $128$ to $4,096$, non-stall decode compute duration expands monotonically from $13,727.88\text{ ms}$ to $14,355.67\text{ ms}$ (+627.79 ms).
- This extra compute duration acts as an expanded asynchronous prefetch window.
- In-flight transfers from DDR5 that arrived slightly late at $P=128$ complete before the MoE stage commences at $P=4,096$, raising resident hits by **+15.0%** (from $1,417$ to $1,630$) and avoiding **211 additional expert reloads**.

### 4.2 The Compute-Bound Latency Plateau (Regime 2)
- Although prefetching hides more memory stalls in absolute terms at $P=4,096$ ($5,919.29\text{ ms}$ hidden vs. $5,381.98\text{ ms}$ at $P=128$), total token latency is increasingly dominated by decode attention FLOPs.
- Consequently, the end-to-end latency speedup asymptotically plateaus at **+0.769%**.

---

## 5. Architectural Disclosures and Limitations

1. **Synthetic Initial Context Length:** The $P$ sweep tests simulated context length in decode mode replaying authentic routing traces; it does not simulate the original prompt prefill phase.
2. **Dedicated Expert Cache Capacity:** LLMSimulator allocates a dedicated $35\text{ GiB}$ partition for expert weights. In systems with unified HBM pools, KV-cache growth at $P=4,096$ would consume $5.63\text{ GiB}$, reducing available weight caching capacity.
3. **Event-Driven Interconnect:** The 64 GB/s DDR5 channel is modeled using calibrated event-driven bandwidth and queue latency, rather than cycle-accurate pin-level simulation.

