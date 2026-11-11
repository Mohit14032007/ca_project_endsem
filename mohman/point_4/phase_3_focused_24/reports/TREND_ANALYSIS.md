# Point 4 Phase 3: Trend Analysis — Capacity, Workload, and Prefetch Dynamics

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Harness Executable:** `run_deepseekv3_mixtral_style`  
**Dataset:** MoE-Anywhere (`e1-sage-a1.5.parquet`), 40 MoE stages, Top-$K=8$  
**Scope:** 24 Experimental Runs Across 3 Capacities and 2 Workload Batch Sizes  

---

## 1. Executive Summary

This report analyzes the behavioral trends observed across HBM cache capacities (10 GB, 35 GiB, and 80 GB) and batch sizes ($B=4$ in W2 and $B=14$ in W3). The experimental findings reveal clear working-set threshold transitions, explain why HBM hit rates scale non-linearly with memory capacity, and uncover the exact conditions under which speculative prefetching either provides modest latency hiding or induces severe cache pollution.

---

## 2. Working-Set Capacity Threshold Dynamics

The fundamental driver of cache behavior in a 40-stage MoE model is the relationship between the **single-step token generation working set ($S_{\text{ws}}$)** and the **available HBM expert capacity ($C_{\text{HBM}}$)**.

Each expert in the DeepSeek-V3 Mixtral-style architecture occupies:

$$S_{\text{expert}} = 88,080,384\text{ Bytes} \approx 83.999\text{ MiB} \approx 88.08\text{ MB}$$

The available capacity slots for the three evaluated memory sizes are:
- **CAP1 (10 GB = $10^{10}\text{ B}$):** $113\text{ expert slots}$ ($9.953\text{ GB}$ usable)
- **CAP2 (35 GiB = $37,580,963,840\text{ B}$):** $426\text{ expert slots}$ ($37.522\text{ GB}$ usable)
- **CAP3 (80 GB = $8 \times 10^{10}\text{ B}$):** $908\text{ expert slots}$ ($79.977\text{ GB}$ usable)

### Table 1: Working Set vs. HBM Capacity Regime Analysis

| Workload | Batch ($B$) | Sequences | Coalesced Demands / Step | Unique Working Set ($S_{\text{ws}}$) | 10 GB ($113$ slots) | 35 GiB ($426$ slots) | 80 GB ($908$ slots) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **W2** | 4 | 4 | $\approx 517\text{ demands}$ | $\approx 312\text{ unique experts}$ | $C < S_{\text{ws}}$ (Thrashing) | $C > S_{\text{ws}}$ (**Residency Onset**) | $C \gg S_{\text{ws}}$ (**High Residency**) |
| **W3** | 14 | 14 | $\approx 823\text{ demands}$ | $\approx 520\text{ unique experts}$ | $C \ll S_{\text{ws}}$ (Thrashing) | $C < S_{\text{ws}}$ (**Thrashing**) | $C > S_{\text{ws}}$ (**Residency Onset**) |

---

## 3. Capacity Scaling Trends

### 3.1 HBM Hit Rate Scaling

```text
Coalesced HBM Hit Rate vs. Capacity
0.50 ┤                                        [W2 C1/C2: 41.96%]
0.40 ┤                                              ▲
0.30 ┤                                              │   [W3 C1/C2: 24.14%]
0.20 ┤                      [W2 C1/C2: 14.31%]      │         ▲
0.10 ┤                             ▲                │         │
0.00 ┼─────────■───────────────────┴────────────────┼─────────┴──────
           10 GB (CAP1)       35 GiB (CAP2)             80 GB (CAP3)
```

1. **At 10 GB (CAP1, 113 slots):**
   - For both W2 ($S_{\text{ws}} \approx 312$) and W3 ($S_{\text{ws}} \approx 520$), cache capacity is strictly smaller than the working set of a single decode pass across 40 stages.
   - Resident lines are evicted sequentially within 10 to 14 stages. By the time the pipeline wraps back to Stage 0 for the next token, all Stage 0 experts have been purged.
   - Consequently, baseline coalesced hit rate is **$0.000000$** across both workloads.
2. **At 35 GiB (CAP2, 426 slots):**
   - **Workload W2 ($B=4$):** Capacity ($426$ slots) exceeds the single-step working set ($312$ slots). Experts referenced in early stages can remain resident across the entire 40-stage traversal. Coalesced hit rate jumps dramatically to **$14.3082\%$** (1,184 hits).
   - **Workload W3 ($B=14$):** Capacity ($426$ slots) is smaller than the larger batch working set ($520$ slots). Cache thrashing continues across decode steps. Hit rate remains at **$0.000000$**.
3. **At 80 GB (CAP3, 908 slots):**
   - Capacity ($908$ slots) accommodates the working sets of both workloads with substantial margin.
   - For W2, hit rate rises to **$41.9577\%$** (3,472 hits).
   - For W3, hit rate rises to **$24.1416\%$** (3,178 hits).

### 3.2 Total Evictions and Expert Reloads
As capacity expands from 10 GB to 80 GB:
- In W2, total evictions drop by **$52.28\%$** ($8,162 \to 3,895$), and expert reloads drop by **$63.80\%$** ($5,442 \to 1,970$).
- In W3, total evictions drop by **$30.44\%$** ($13,051 \to 9,078$), and expert reloads drop by **$35.55\%$** ($8,940 \to 5,762$).

---

## 4. Workload Batch-Size Sensitivity

Scaling the serving batch size from $B=4$ (W2) to $B=14$ (W3) increases the number of concurrent token routing decisions per stage from 32 ($4 \times 8$) to 112 ($14 \times 8$).

### 4.1 Demand Coalescing Efficiency
- **W2 ($B=4$):** Generated 24,825 tensor projections, 8,275 routed demands, and 8,275 coalesced demands.
- **W3 ($B=14$):** Generated 39,492 tensor projections, 13,164 routed demands, and 13,164 coalesced demands.
- Because prompt sequences route to diverse expert subsets across the 256 available experts per stage, routing collisions per stage remain rare at $B=4$ and $B=14$. Consequently, coalesced demands closely match routed demands.

### 4.2 Simulated Throughput Scaling
- At 80 GB, increasing batch size from $B=4$ to $B=14$ improves simulated generation throughput from **$9.17\text{ tokens/s}$** to **$15.74\text{ tokens/s}$** ($+71.65\%$).
- Compute execution scales sub-linearly with batch size due to batched tensor core efficiency, amortizing the fixed memory transfer stalls across more generated tokens.

---

## 5. CSTP Prefetch Dynamics and Cache Pollution

Evaluating CSTP prefetching across capacities reveals a critical dichotomy in speculative memory systems:

### 5.1 Thrashing Regimes (10 GB and W3 at 35 GiB)
In heavily constrained caches where resident hit rate is zero ($C_{\text{HBM}} < S_{\text{ws}}$), the cache is already fully evicted every decode step.
- Speculative prefetches cannot displace valuable long-term resident lines because no lines would survive anyway.
- When a speculative transfer is dispatched with horizon $H=1$, it begins loading while the previous stage computes.
- Even though prefetch coverage is modest ($7.77\%$ in W2, $5.56\%$ in W3), early transfers allow **late in-flight attachment**, trimming remaining wait time and reducing exposed stall:
  - W2 10 GB: Exposed stall decreases by **$-9.63\text{ ms}$** ($-0.08\%$).
  - W3 10 GB: Exposed stall decreases by **$-15.14\text{ ms}$** ($-0.08\%$).
  - W3 35 GiB: Exposed stall decreases by **$-34.03\text{ ms}$** ($-0.19\%$).

### 5.2 Residency Regimes (35 GiB in W2 and 80 GB in W2/W3)
Once HBM capacity is large enough to retain expert working sets ($C_{\text{HBM}} > S_{\text{ws}}$), speculative prefetching causes **cache pollution**:
- In W2 at 35 GiB, CSTP dispatches 1,204 speculative transfers.
- These speculative lines enter the cache and evict demand-resident lines that were destined to be hit in subsequent tokens.
- Coalesced resident hit rate drops sharply from **$14.31\%$** to **$9.62\%$** ($\Delta = -4.69\%$).
- Total evictions increase by $+631$, expert reloads increase by $+412$, and total DDR5 traffic expands by $+59.81\text{ GB}$ ($+9.58\%$).
- Most critically, exposed GPU stall increases by **$+533.73\text{ ms}$** ($+5.47\%$), and mean token latency degrades by **$+33.08\text{ ms}$** ($+5.24\%$).

---

## 6. Summary of Architectural Takeaways

1. **Threshold Effect:** Deep MoE inference exhibits an abrupt working-set capacity threshold ($S_{\text{ws}} \approx 312$ for $B=4$, $\approx 520$ for $B=14$). Below this threshold, cache hit rate is zero; above it, hit rate scales linearly with capacity.
2. **Prefetching Vulnerability:** Speculative prefetching without strict admission control or cache partitioning harms performance precisely when HBM capacity begins to deliver resident hits.
3. **Eviction Invariance:** Across non-prefetch runs, the sequential stage-by-stage traversal creates uniform candidate frequencies, rendering FA-LRU identical to Standard LRU. Under CSTP, FA-LRU achieves micro-eviction divergence by targeting unconsumed speculative lines, but macro performance remains governed by overall capacity constraints.

