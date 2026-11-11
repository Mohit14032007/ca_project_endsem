# Point 6 — Context-Length Feasibility and Methodological Classification Report

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Phase 0: Feasibility & Trace Audit  
**Author:** Antigravity Engineering  
**Date:** November 10, 2026  
**Status:** Authoritative & Completed  

---

## 1. Executive Summary and Problem Formulation

Point 6 requires evaluating system sensitivity across multiple batch sizes ($B$), initial context lengths ($P$), decode horizons ($T$), lookaheads ($H$), and prefetch budgets ($K$).

In LLM serving systems, "context length" physically manifests in two distinct operational regimes:
1. **The Prefill Phase (Prompt Processing):** The model processes all prompt tokens $0 \le t < P$ in parallel ($m = P$). This phase features quadratic attention compute $\mathcal{O}(P^2)$, dense activation footprints, and simultaneous multi-token routing through the MoE layers.
2. **The Decode Phase (Autoregressive Token Generation):** The model generates one token at a time ($m = 1$). However, at every step $t$, the token attends to all prior $P + t$ tokens via the KV cache. Consequently, the initial context length $P$ physically manifests as:
   - **KV-Cache Memory Footprint:** The size of the KV cache scales as $\mathcal{O}(P + T)$, directly consuming HBM capacity and displacing the space available for caching MoE expert weights.
   - **Decode Attention Latency:** Decode attention compute FLOPs and DRAM memory accesses scale as $\mathcal{O}(P + t)$, lengthening the attention phase and widening the temporal window available to overlap asynchronous expert prefetching.

To ensure absolute scientific integrity, this report classifies three distinct experimental approaches for Point 6.

---

## 2. Formal Classification Matrix

| Dimension | Approach A: Genuine Prefill Sensitivity | Approach B: Decode-Only Context Sensitivity | Approach C: Generation Horizon Sensitivity ($T$) |
| :--- | :--- | :--- | :--- |
| **Definition** | Simulating prompt prefill execution ($0 \le t < P$) followed by token generation | Simulating token generation ($P \le t < P+T$) while scaling $P$ to modulate KV memory & attention | Simulating token generation across varying sequence generation steps ($T \in [16, 256]$) |
| **Trace Data Availability** | **Unsupported** (0 prefill records in `e1-sage` or `moe_suite`) | **Fully Supported** (Replays existing contiguous 257-token decode traces) | **Fully Supported** (Replays up to 257 contiguous decode tokens) |
| **Simulator Support** | **Unsupported** (`test_deepseekv3_mixtral_style.cpp` hardcodes `prefill_mode = false`) | **Fully Supported** (Fully functional via `simulation.input_len` YAML parameter) | **Fully Supported** (Fully functional via `simulation.iter` YAML parameter) |
| **Required Engineering** | Major external dataset collection & simulator C++ rewrite | **Zero code changes**; standard configuration changes | **Zero code changes**; standard configuration changes |
| **Physical Soundness** | High if empirical data existed; currently zero data | **High**: Accurately models KV-cache pressure and attention timing on decode | **High**: Accurately models temporal routing dynamics and cache pollution |
| **Feasibility Status** | **UNSUPPORTED / BLOCKED** | **FEASIBLE / RECOMMENDED** | **FEASIBLE / RECOMMENDED** |

---

## 3. Deep Dive: Approach A (Genuine Prefill Sensitivity)

### 3.1 Why Approach A is Unsupported
1. **Absence of Empirical Prefill Routing Data:**
   As verified in `DATASET_CONTEXT_LENGTH_AUDIT.md`, `e1-sage-a1.5.parquet` contains zero records for tokens $0 \le t < \text{min\_token\_pos}$. Upstream trace generation only recorded tokens during the decode loop. MoE-Suite similarly has 0.0% prompt tokens (`token_flag_prompt == False`).
2. **Absence of Token Text:**
   The dataset contains no prompt text or vocabulary IDs. It is impossible to pass the prompts through a forward pass of DeepSeek-V3 to extract routing decisions without the original text.
3. **Driver Limitations:**
   The evaluation driver `test_deepseekv3_mixtral_style.cpp` hardcodes `system_config.prefill_mode = false;`. Enabling prefill would require implementing a multi-token prompt routing loader and prefill scheduler.

### 3.2 Scientific Boundaries for Approach A
> [!CAUTION]
> It is strictly forbidden to synthesize prompt routing traces or claim that prefill routing was simulated. Any paper or report claiming to evaluate prefill expert prefetching without genuine prefill routing data is scientifically invalid.

---

## 4. Deep Dive: Approach B (Decode-Only Context Length Sensitivity)

### 4.1 Physical & System-Level Validity
In real-world LLM serving, long-context inference frequently bottlenecks on the decode phase because:
1. Each sequence maintains an extensive KV cache in HBM for thousands of previous prompt tokens.
2. In unified-memory architectures where expert weights and the KV cache share the same HBM capacity (e.g., A100-80GB or H100-80GB), **every gigabyte consumed by the KV cache directly reduces the HBM capacity available for expert caching**.
3. Decode attention duration increases linearly with $P$, shifting the computation-to-communication ratio of the transformer layer.

### 4.2 Mathematical Modeling in LLMSimulator
In `LLMSimulator`, setting `input_len: P` in the configuration file achieves the following:
1. **KV Cache Footprint Scaling:**
   $$M_{\text{KV}}(P, B) = B \times (P + T) \times (d_{\text{kv\_lora\_rank}} + d_{\text{qk\_rope\_head\_dim}}) \times L \times \text{precision\_byte}$$
   For DeepSeek-V3 ($L=40, d_{\text{kv\_lora\_rank}}=512, d_{\text{qk\_rope\_head\_dim}}=64, \text{precision}=2$):
   $$M_{\text{KV}}(P, B) = B \times (P + T) \times 46,080 \text{ bytes}$$
   - At $B=32, P=128$: $M_{\text{KV}} \approx 0.18$ GiB.
   - At $B=32, P=4,096$: $M_{\text{KV}} \approx 5.76$ GiB.
   This cleanly displaces expert cache capacity, creating heightened eviction pressure that rigorously stresses LRU vs. FA-LRU!

2. **Attention Latency Scaling:**
   Scoring and context attention FLOPs and DRAM access volumes scale proportionally to $n = P + t + 1$. This increases attention duration from a few microseconds at $P=128$ to hundreds of microseconds at $P=4,096$, testing whether asynchronous prefetching can overlap with larger attention compute windows.

3. **Trace Replay Invariance:**
   In [sequence.cpp line 248](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L248):
   $$\text{rel\_token\_id} = \text{token\_id} - \text{input\_len} = (P + t) - P = t$$
   Because `rel_token_id` is invariant to $P$, the authentic empirical routing traces from `e1-sage-a1.5.parquet` are replayed identically across all context lengths!

### 4.3 Legitimate vs. Illegitimate Scientific Claims for Approach B

| Permitted Claims | Prohibited Claims |
| :--- | :--- |
| "We evaluate decode-phase sensitivity to initial context length by varying KV-cache memory pressure and decode attention durations from $P=128$ to $P=4,096$." | "We evaluate the expert prefetching speedup during the prompt prefill phase." |
| "As initial context length grows, KV-cache expansion reduces available HBM expert cache capacity, increasing expert thrashing and demonstrating FA-LRU's eviction stability." | "Prompt tokens were routed through MoE experts according to empirical prefill traces." |
| "Longer initial context lengths lengthen decode attention execution, providing wider temporal slack for asynchronous prefetch overlap." | "Prefill FLOPs were measured using end-to-end prompt inference." |

---

## 5. Deep Dive: Approach C (Generation Horizon Sensitivity $T$)

### 5.1 System-Level Validity
Varying the decode generation horizon $T$ (e.g., $T \in \{16, 64, 128, 256\}$) evaluates:
1. **Long-Horizon Cache Stability:** Does speculative prefetching gradually pollute the HBM cache over hundreds of tokens?
2. **Predictor Warm-Up vs. Saturation:** How quickly do spatial transition matrices (CSTP) or temporal counters (ST-MoE) converge to steady-state accuracy?
3. **Cumulative Memory Stall Exposure:** How do total exposed stalls scale over long generation sequences?

### 5.2 Implementation Status
- `e1-sage-a1.5.parquet` provides exactly 257 contiguous tokens per prompt.
- Horizons of $T = 16, 64, 128, 256$ are 100% covered by existing traces.
- Supported directly via `simulation.iter: T` in YAML.

---

## 6. Strategic Methodological Recommendation for Point 6

To deliver an exhaustive, rigorous, and completely honest sensitivity analysis for Point 6:
1. **Declare Approach A as Unsupported:** Transparently document in all project reports that empirical prefill traces do not exist in the MoE-Anywhere or MoE-Suite datasets.
2. **Execute Approach B (Decode-Only Context Length Sensitivity):** Vary initial context length $P \in \{128, 512, 2048, 4096\}$ under batch sizes $B \in \{14, 32\}$ to evaluate the effect of KV-cache memory displacement and attention duration scaling.
3. **Execute Approach C (Decode Generation Horizon Sensitivity):** Vary decode horizon $T \in \{16, 128, 256\}$ to evaluate temporal prefetch stability and cache pollution resistance.
4. **Evaluate Prefetch Hyperparameters:** Vary lookahead $H \in \{1, 2, 4\}$ and budget $K \in \{2, 4, 8\}$ across these configurations to establish the exact boundaries where prefetching accelerates, provides diminishing returns, or degrades performance.

