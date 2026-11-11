# Point 6 — Sensitivity Analysis Experiment Design

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Phase 0: Feasibility & Trace Audit  
**Author:** Antigravity Engineering  
**Date:** November 10, 2026  
**Status:** Authoritative & Completed  

---

## 1. Executive Summary & Objective

The primary objective of Point 6 is conducting an authoritative **Sensitivity Analysis** across the key operational parameters governing MoE expert prefetching and cache management:
1. **Batch Size ($B$):** Concurrency degree ($B \in \{14, 32\}$).
2. **Initial Context Length ($P$):** KV-cache memory pressure and decode attention duration ($P \in \{128, 512, 2048, 4096\}$).
3. **Decode Generation Horizon ($T$):** Temporal duration of autoregressive generation ($T \in \{16, 128, 256\}$).
4. **Inter-Layer Lookahead ($H$):** Forward prediction distance ($H \in \{1, 2, 4\}$ layers ahead).
5. **Prefetch Candidate Budget ($K$):** Number of speculative experts prefetched per token ($K \in \{2, 4, 8\}$).
6. **HBM Eviction Policy:** Standard LRU vs. Frequency-Aware LRU (FA-LRU with $W=64$).

This design defines a disciplined, non-combinatorial experimental matrix that isolates each parameter's individual impact, identifies performance boundaries, and explains when prefetching succeeds, delivers diminishing returns, or degrades performance.

---

## 2. Controlled Parameter Matrix

To prevent combinatorial explosion ($2 \times 4 \times 3 \times 3 \times 3 \times 2 = 432$ possible configurations), the campaign is structured into five focused single-variable and two-variable exploration sweeps around a standardized anchor configuration.

### 2.1 Standardized Anchor Configuration
- **Model:** DeepSeek-V3 (40 MoE layers, 256 routed experts, Top-8 routing).
- **HBM Cache Capacity:** 35 GiB (constrained HBM regime where memory wall is exposed).
- **Predictor:** CSTP (Confidence- and Frequency-Aware Spatio-Temporal Predictor).
- **Scheduling Mode:** Forward Lookahead (`FORWARD_LOOKAHEAD`).
- **Anchor Parameters:** $B = 32$, $P = 2,048$, $T = 128$, $H = 1$, $K = 4$.

---

### 2.2 Sweep 1: Initial Context Length Sensitivity ($P$)
- **Objective:** Evaluate how KV-cache memory expansion and decode attention scaling affect prefetch timeliness and eviction policy efficacy.
- **Fixed Parameters:** $B = 32$, $T = 128$, $H = 1$, $K = 4$, HBM = 35 GiB.
- **Varied Parameter:** $P \in \{128, 512, 2048, 4096\}$.
- **Evaluated Systems:**
  1. Baseline (No Prefetching, LRU).
  2. CSTP + LRU.
  3. CSTP + FA-LRU ($W=64$).
- **Total Configurations:** $4 \text{ context lengths} \times 3 \text{ systems} = 12 \text{ runs}$.

---

### 2.3 Sweep 2: Inter-Layer Lookahead Sensitivity ($H$)
- **Objective:** Determine the trade-off between transfer timeliness (earlier dispatch) and prediction precision (accuracy degrades as lookahead distance increases).
- **Fixed Parameters:** $B = 32$, $P = 2,048$, $T = 128$, $K = 4$, HBM = 35 GiB.
- **Varied Parameter:** $H \in \{1, 2, 4\}$.
- **Evaluated Systems:**
  1. CSTP + LRU.
  2. CSTP + FA-LRU ($W=64$).
- **Total Configurations:** $3 \text{ lookaheads} \times 2 \text{ systems} = 6 \text{ runs}$.

---

### 2.4 Sweep 3: Prefetch Candidate Budget Sensitivity ($K$)
- **Objective:** Determine the threshold where additional speculative transfers saturate DDR5 bus bandwidth and pollute the HBM cache.
- **Fixed Parameters:** $B = 32$, $P = 2,048$, $T = 128$, $H = 1$, HBM = 35 GiB.
- **Varied Parameter:** $K \in \{2, 4, 8\}$.
- **Evaluated Systems:**
  1. CSTP + LRU.
  2. CSTP + FA-LRU ($W=64$).
- **Total Configurations:** $3 \text{ budgets} \times 2 \text{ systems} = 6 \text{ runs}$ (includes anchor).

---

### 2.5 Sweep 4: Batch Size Scaling ($B$)
- **Objective:** Investigate how sequence parallelism modulates expert reuse and cache contention across context lengths.
- **Fixed Parameters:** $T = 128$, $H = 1$, $K = 4$, HBM = 35 GiB.
- **Varied Parameters:** $B \in \{14, 32\}$, $P \in \{512, 2048\}$.
- **Evaluated Systems:**
  1. Baseline (No Prefetching, LRU).
  2. CSTP + LRU.
  3. CSTP + FA-LRU ($W=64$).
- **Total Configurations:** $2 \text{ batches} \times 2 \text{ lengths} \times 3 \text{ systems} = 12 \text{ runs}$ (partially overlapping with Sweep 1).

---

### 2.6 Sweep 5: Generation Horizon Sensitivity ($T$)
- **Objective:** Evaluate long-horizon temporal prefetch stability and cache pollution resistance over extended generation horizons.
- **Fixed Parameters:** $B = 32$, $P = 2,048$, $H = 1$, $K = 4$, HBM = 35 GiB.
- **Varied Parameter:** $T \in \{16, 128, 256\}$.
- **Evaluated Systems:**
  1. CSTP + LRU.
  2. CSTP + FA-LRU ($W=64$).
- **Total Configurations:** $3 \text{ horizons} \times 2 \text{ systems} = 6 \text{ runs}$ (includes anchor).

---

## 3. Analytical Memory Capacity & KV-Cache Sizing

### 3.1 Mathematical Formulation
For DeepSeek-V3 with Multi-Head Latent Attention (MLA):
- Layers: $L = 40$
- Latent rank: $d_{\text{kv\_lora\_rank}} = 512$
- RoPE rank: $d_{\text{qk\_rope\_head\_dim}} = 64$
- Precision: $\text{precision\_byte} = 2$ bytes (FP16)
- Expert size: $88,080,384$ bytes ($84.0$ MiB)

$$\text{KV Cache Size per Token} = (512 + 64) \times 40 \times 2 = 46,080 \text{ bytes} \approx 45.0 \text{ KiB}$$
$$M_{\text{KV}}(B, P, T) = B \times (P + T) \times 46,080 \text{ bytes}$$

### 3.2 HBM Capacity Displacement Table (35 GiB HBM Budget, $T=128$)

| Batch Size ($B$) | Context Length ($P$) | KV Cache Size ($M_{\text{KV}}$) | Effective Expert Cache Capacity | Max Resident Experts | Cache Squeeze Ratio |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$B = 14$** | $P = 128$ | $0.16$ GiB | $34.84$ GiB | $414$ experts | Baseline (0.5% squeeze) |
| **$B = 14$** | $P = 512$ | $0.40$ GiB | $34.60$ GiB | $411$ experts | 1.1% squeeze |
| **$B = 14$** | $P = 2,048$ | $1.37$ GiB | $33.63$ GiB | $400$ experts | 3.9% squeeze |
| **$B = 14$** | $P = 4,096$ | $2.66$ GiB | $32.34$ GiB | $385$ experts | 7.6% squeeze |
| **$B = 32$** | $P = 128$ | $0.36$ GiB | $34.64$ GiB | $412$ experts | 1.0% squeeze |
| **$B = 32$** | $P = 512$ | $0.91$ GiB | $34.09$ GiB | $405$ experts | 2.6% squeeze |
| **$B = 32$** | $P = 2,048$ | $3.13$ GiB | $31.87$ GiB | $379$ experts | 8.9% squeeze |
| **$B = 32$** | $P = 4,096$ | $6.08$ GiB | $28.92$ GiB | $344$ experts | **17.4% squeeze** |

*Key Insight:* At $B=32$, increasing initial context length from $P=128$ to $P=4,096$ displaces **68 expert slots** from HBM! This creates acute eviction pressure, providing the exact operational regime where FA-LRU's frequency-aware victim selection can protect hot experts from being prematurely evicted by cold prefetch candidates.

---

## 4. Evaluation Metrics & Decoupling Strategy

To isolate the multiple physical effects interacting during inference, metrics are decoupled into five distinct telemetry groups:

```
+-----------------------------------------------------------------------------------+
|                            POINT 6 TELEMETRY DECOUPLING                           |
+---------------------+-------------------+-------------------+---------------------+
| 1. Latency & Stalls | 2. Cache Pressure | 3. Prefetch Stats | 4. Memory Traffic   |
+---------------------+-------------------+-------------------+---------------------+
| - Mean Token Latency| - Resident Hits   | - Useful Prefetch | - DDR5 Volume (GB)  |
| - Exposed Stalls    | - Demand Reloads  | - Waste Prefetch  | - DDR5 Bandwidth    |
| - Attn Gen Time     | - FA-LRU Evictions| - Timeliness Frac | - Bus Utilization   |
| - FFN Compute Time  | - LRU Evictions   | - Cache Pollution | - Queue Saturation  |
+---------------------+-------------------+-------------------+---------------------+
```

### Decoupling Rules:
1. **Separating Attention Scaling from Memory Stalls:**
   Because attention duration grows with $P$, overall latency increases even with zero memory stalls. To measure prefetching efficacy cleanly, we compute:
   $$\text{Prefetch Gain} = \frac{\text{Exposed Memory Stall}_{\text{Baseline}} - \text{Exposed Memory Stall}_{\text{Prefetch}}}{\text{Exposed Memory Stall}_{\text{Baseline}}} \times 100\%$$
2. **Measuring Cache Pollution:**
   Tracked by counting demand misses induced by prefetch evictions:
   $$\text{Pollution Events} = \text{Demand Reloads}_{\text{Prefetch}} - \text{Demand Reloads}_{\text{Baseline}}$$
3. **Evaluating Lookahead Timeliness:**
   A prefetch is classified as:
   - **Timely:** Completed before target stage requires it (0 stall cycles).
   - **Late / Partially Timely:** In-flight when target stage starts, reducing stall duration but not eliminating it.
   - **Useless:** Completed, but expert was never demanded.

---

## 5. Theoretical Regime Boundaries (When Prefetching Helps vs. Degrades)

| Operating Regime | Parameter Conditions | Expected Behavior | Physical Mechanism |
| :--- | :--- | :--- | :--- |
| **Accelerated Zone** | Moderate $K \in \{2, 4\}$, Lookahead $H \in \{1, 2\}$, Context $P \le 2048$ | **Maximum Speedup:** Latency drops, stalls collapse by 60–80%. | High predictor precision; attention execution provides sufficient slack to hide transfer latency; DDR5 bus is unconstrained. |
| **Diminishing Returns** | High Context $P \ge 4096$, Large Budget $K=8$ | **Plateau:** Incremental gain drops to $< 2\%$. | KV cache displacement squeezes expert cache capacity; prefetch requests compete with demand misses for DDR5 bandwidth. |
| **Degraded Performance** | High Lookahead $H=4$, Over-aggressive Budget $K=8$ | **Performance Inversion:** Latency increases, DDR5 traffic explodes by $> 150\%$. | Predictor precision degrades with distance; speculative prefetches evict resident hot experts; cache pollution creates self-inflicted demand stalls. |

---

## 6. Output Artifact & Directory Structure

All Point 6 experimental configurations, traces, tables, and visualization artifacts will be saved strictly under:

```text
mohman/point_6/
├── audit/
│   ├── DATASET_CONTEXT_LENGTH_AUDIT.md
│   ├── PREFILL_DECODE_SIMULATOR_AUDIT.md
│   ├── CONTEXT_LENGTH_FEASIBILITY_REPORT.md
│   ├── POINT_6_EXPERIMENT_DESIGN.md
│   └── POINT_6_GO_NO_GO_REPORT.md
├── configs/
│   ├── sweep1_context_length/
│   ├── sweep2_lookahead/
│   ├── sweep3_budget/
│   ├── sweep4_batch/
│   └── sweep5_horizon/
├── results/
│   └── raw/
├── extracted_data/
│   └── sensitivity_matrix.csv
└── figures/
    ├── png/
    └── pdf/
```

