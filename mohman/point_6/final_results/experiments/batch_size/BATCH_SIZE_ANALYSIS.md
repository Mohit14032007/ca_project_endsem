# Batch Size Sensitivity Analysis: Scaling Concurrency and Regimes of Acceleration vs. Cache Degradation

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Dimension 1: Batch Size Sweeps ($B \in \{4, 14, 32\}$)  
**Author:** Antigravity Engineering  
**Date:** November 10, 2026  
**Status:** Authoritative Empirical Report  

---

## 1. Executive Summary

We evaluated the sensitivity of asynchronous prediction-driven prefetching (CSTP) and cache eviction policies (Standard LRU vs. Frequency-Aware LRU $W=64$) across three serving concurrency levels:
- Low Concurrency ($B=4$)
- Medium Concurrency ($B=14$)
- High Concurrency ($B=32$, Common Anchor)

All simulations execute on DeepSeek-V3 (40 MoE layers, Top-8 routing per token), 35 GiB dedicated HBM expert cache, DDR5 64 GB/s off-chip interconnect, fixed context length $P=2,048$, generation horizon $T=128$, lookahead $H=1$, and prefetch budget $K=4$.

### Primary Findings:
1. **Regime 3 (Performance Inversion at $B=4$):** At low concurrency, speculative prefetching **degrades** performance. Mean token latency increases from $1,466.88\text{ ms}$ to $1,468.79\text{ ms}$ (+0.13% slower), resident hits drop from $1,431$ to $1,237$ (-13.6%), and expert reloads increase from $124,975$ to $125,276$ (+301 reloads). This confirms that speculative transfers pollute the cache when concurrency is too low to sustain reuse.
2. **Regime 1 (Accelerated Serving at $B=14$ and $B=32$):** At medium and high concurrency, prefetching consistently improves performance:
   - At $B=14$: Latency improves from $3,587.20\text{ ms}$ to $3,566.31\text{ ms}$ (+0.58% speedup), hiding $2,693.71\text{ ms}$ of memory stall and avoiding $1,564$ expert reloads.
   - At $B=32$: Latency improves from $5,994.31\text{ ms}$ to $5,950.46\text{ ms}$ (+0.73% speedup), hiding $5,630.28\text{ ms}$ of memory stall and avoiding $1,398$ expert reloads.
3. **FA-LRU Scaling Inversion:** At $B=4$ and $B=14$, FA-LRU ($W=64$) behaves nearly identically to LRU (-2 and -6 reloads). However, at $B=32$, FA-LRU protects frequently reused experts, gaining $+22$ resident hits, reducing reloads by $-22$, and trimming exposed stalls by an additional $31.90\text{ ms}$.

---

## 2. Experimental Data Table

The table below presents the verified empirical results across the three batch sizes:

| Batch Size ($B$) | Configuration | Execution Time (s) | Mean Latency (ms) | Exposed Stall (s) | Coalesced Demands | Resident Hits | Hit Rate (%) | Reloads | DDR5 Traffic (GB) |
|---|---|---|---|---|---|---|---|---|---|
| **$B = 4$** | Baseline (No Prefetch) | 187.761 | 1,466.88 | 183.514 | 126,406 | 1,431 | 1.1321% | 124,975 | 10,990.23 |
| **$B = 4$** | CSTP + LRU | 188.005 | 1,468.79 | 183.761 | 126,406 | 1,237 | 0.9786% | 125,276 | 13,732.18 |
| **$B = 4$** | CSTP + FA-LRU ($W=64$) | 188.007 | 1,468.80 | 183.762 | 126,406 | 1,235 | 0.9770% | 125,278 | 13,732.09 |
| **$B = 14$** | Baseline (No Prefetch) | 459.162 | 3,587.20 | 450.381 | 317,775 | 0 | 0.0000% | 317,775 | 27,964.20 |
| **$B = 14$** | CSTP + LRU | 456.489 | 3,566.31 | 447.687 | 317,775 | 1,673 | 0.5265% | 316,211 | 34,923.75 |
| **$B = 14$** | CSTP + FA-LRU ($W=64$) | 456.489 | 3,566.31 | 447.688 | 317,775 | 1,667 | 0.5246% | 316,217 | 34,923.23 |
| **$B = 32$** | Baseline (No Prefetch) | 767.272 | 5,994.31 | 753.246 | 547,315 | 0 | 0.0000% | 537,351 | 48,207.72 |
| **$B = 32$** | CSTP + LRU | 761.659 | 5,950.46 | 747.615 | 547,315 | 1,499 | 0.2739% | 535,953 | 48,741.13 |
| **$B = 32$** | CSTP + FA-LRU ($W=64$) | 761.627 | 5,950.21 | 747.583 | 547,315 | 1,521 | 0.2779% | 535,931 | 48,739.10 |

---

## 3. Detailed Mechanism Analysis

### 3.1 Why Prefetching Degrades at Low Concurrency ($B=4$)
At $B=4$, the aggregate demand rate is low, and consecutive tokens from the 4 sequences do not create immediate overlapping reuse in HBM. However, speculative prefetching dispatches candidates that have not yet been requested. These candidates displace currently resident expert weights. Because the cache capacity is constrained to 35 GiB, speculative arrivals prematurely evict valid demand experts before they are reused, resulting in **301 additional reloads** and a **24.9% increase in DDR5 memory traffic** (from 10.99 TB to 13.73 TB). Consequently, GPU execution experiences higher stalls and latency increases by $+1.91\text{ ms}$ (+0.13%).

### 3.2 Transition to Acceleration ($B=14$ and $B=32$)
When batch size increases to $B=14$ and $B=32$, two phenomena take place:
1. **Inter-Sequence Overlap:** Multiple concurrent sequences share hot experts across decoding steps, raising the demand reuse frequency.
2. **Batch Computation Slack:** Computing attention and GEMM across 14 to 32 tokens extends the layer compute phase, providing more wall-clock time for asynchronous transfers from DDR5 to complete in the background.

At $B=14$, prefetching successfully saves $1,564$ reloads and reduces stall time by $2,693.71\text{ ms}$. At $B=32$, prefetching hides $5,630.28\text{ ms}$ of memory stall and reduces simulated execution time by $5.613\text{ seconds}$.

### 3.3 Eviction Policy Behavior: LRU vs. FA-LRU ($W=64$)
Under low-to-medium concurrency ($B=4, 14$), the frequency window of $W=64$ rarely sees repeated hits within the active window because the working set rotates too quickly across the 40 layers. At $B=32$, however, high token concurrency creates dense clusters of repeated expert requests. FA-LRU protects these high-frequency experts from eviction by low-frequency speculative prefetches, resulting in $+22$ higher resident hits, $22$ fewer expert reloads, and $31.90\text{ ms}$ lower memory stall.

---

## 4. Associated Visualizations

- [Figure 1: Batch Size vs. HBM Hit Rate](file:///home/mohman/ca_project/mohman/point_6/final_results/figures/batch_size/fig1_batch_size_hit_rate.png)
- [Figure 2: Batch Size vs. Exposed Memory Stall](file:///home/mohman/ca_project/mohman/point_6/final_results/figures/batch_size/fig2_batch_size_exposed_stall.png)
- [Figure 3: Batch Size vs. Mean Token Latency](file:///home/mohman/ca_project/mohman/point_6/final_results/figures/batch_size/fig3_batch_size_token_latency.png)
- [Figure 13: End-to-End Latency Speedup Across Batch Sizes](file:///home/mohman/ca_project/mohman/point_6/final_results/figures/combined/fig13_prefetch_vs_noprefetch.png)

