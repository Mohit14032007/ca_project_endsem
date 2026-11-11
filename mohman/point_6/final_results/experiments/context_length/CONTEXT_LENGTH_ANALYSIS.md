# Context Length Sensitivity Analysis: Attention Scaling and Prefetch Overlap Slack

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Dimension 2: Context Length Sweeps ($P \in \{128, 512, 2048, 4096\}$)  
**Author:** Antigravity Engineering  
**Date:** November 10, 2026  
**Status:** Authoritative Empirical Report  

---

## 1. Executive Summary

We evaluated the effect of varying initial context length $P \in \{128, 512, 2048, 4096\}$ on asynchronous expert prefetching (CSTP) and cache eviction policies (Standard LRU vs. Frequency-Aware LRU $W=64$).

All simulations evaluate DeepSeek-V3 with batch size $B=32$, generation horizon $T=128$, lookahead $H=1$, prefetch budget $K=4$, and dedicated 35 GiB HBM expert cache over a 64 GB/s DDR5 interconnect. The $P=128$ condition is reused directly from the verified Point 4 Big Experiment, while $P=512, 2048, 4096$ were executed as newly required simulations.

### Primary Findings:
1. **Monotonic Overlap Slack Expansion:** Non-stall compute execution time expands monotonically from $13,727.88\text{ ms}$ at $P=128$ to $14,346.52\text{ ms}$ at $P=4,096$ (+618.64 ms). This expansion arises from the $\mathcal{O}(P+t)$ computational complexity of decode attention evaluated in `attention_gen_impl.cpp`.
2. **Monotonic Gain in Prefetch Effectiveness:** As context length scales and attention execution duration widens, the background prefetch window expands. Consequently:
   - Coalesced resident hits increase from $1,417$ at $P=128$ to $1,630$ at $P=4,096$ (+213 hits, a 15.0% increase).
   - Net expert reloads saved increase from $1,317$ at $P=128$ to $1,528$ at $P=4,096$ (+211 reloads saved).
   - Exposed memory stalls hidden increase from $5,381.98\text{ ms}$ to $5,919.29\text{ ms}$.
   - End-to-end latency speedup increases from $+0.700\%$ at $P=128$ to $+0.769\%$ at $P=4,096$.
3. **Regime 2 (Compute-Bound Plateau):** While prefetching hides more stall duration as $P$ increases, total token latency is increasingly governed by attention FLOPs. Thus, latency gains plateau around 0.77%, illustrating the compute-bound transition at long context lengths.
4. **FA-LRU Consistency:** Across all context lengths, FA-LRU consistently outperforms LRU:
   - At $P=128$: +15 resident hits, 15 fewer reloads, $31.82\text{ ms}$ lower stall.
   - At $P=512$: +15 resident hits, 15 fewer reloads, $29.07\text{ ms}$ lower stall.
   - At $P=2048$: +22 resident hits, 22 fewer reloads, $31.90\text{ ms}$ lower stall.
   - At $P=4096$: +15 resident hits, 15 fewer reloads, $35.95\text{ ms}$ lower stall.

---

## 2. Experimental Data Table

| Context Length ($P$) | Configuration | Execution Time (s) | Mean Latency (ms) | Exposed Stall (s) | Compute Slack (ms) | Resident Hits | Hit Rate (%) | Reloads | Saved Reloads |
|---|---|---|---|---|---|---|---|---|---|
| **$P = 128$** | Baseline (No Prefetch) | 766.973 | 5,991.98 | 753.246 | 13,727.88 | 0 | 0.0000% | 537,351 | — |
| **$P = 128$** | CSTP + LRU | 761.642 | 5,950.06 | 747.864 | 13,778.43 | 1,417 | 0.2589% | 536,034 | 1,317 |
| **$P = 128$** | CSTP + FA-LRU ($W=64$) | 761.610 | 5,949.81 | 747.832 | 13,778.25 | 1,432 | 0.2616% | 536,019 | 1,332 |
| **$P = 512$** | Baseline (No Prefetch) | 767.028 | 5,992.41 | 753.246 | 13,782.45 | 0 | 0.0000% | 537,351 | — |
| **$P = 512$** | CSTP + LRU | 761.595 | 5,950.09 | 747.812 | 13,782.86 | 1,436 | 0.2624% | 536,015 | 1,336 |
| **$P = 512$** | CSTP + FA-LRU ($W=64$) | 761.565 | 5,949.87 | 747.783 | 13,782.93 | 1,451 | 0.2651% | 536,000 | 1,351 |
| **$P = 2048$** | Baseline (No Prefetch) | 767.272 | 5,994.31 | 753.246 | 14,026.23 | 0 | 0.0000% | 537,351 | — |
| **$P = 2048$** | CSTP + LRU | 761.659 | 5,950.46 | 747.615 | 14,043.73 | 1,499 | 0.2739% | 535,953 | 1,398 |
| **$P = 2048$** | CSTP + FA-LRU ($W=64$) | 761.627 | 5,950.21 | 747.583 | 14,043.63 | 1,521 | 0.2779% | 535,931 | 1,420 |
| **$P = 4096$** | Baseline (No Prefetch) | 767.592 | 5,996.88 | 753.246 | 14,346.52 | 0 | 0.0000% | 537,351 | — |
| **$P = 4096$** | CSTP + LRU | 761.673 | 5,950.78 | 747.326 | 14,346.74 | 1,630 | 0.2978% | 535,823 | 1,528 |
| **$P = 4096$** | CSTP + FA-LRU ($W=64$) | 761.637 | 5,950.51 | 747.290 | 14,347.69 | 1,645 | 0.3006% | 535,808 | 1,543 |

---

## 3. Memory Architecture Disclosure

As established in our comprehensive memory model audit:
1. **Dedicated Expert Cache Capacity:** LLMSimulator allocates a dedicated cache partition (`device->cache_capacity_bytes = 35 GiB`) strictly for expert weights.
2. **KV-Cache Independence:** Scaling $P$ from 128 to 4,096 increases the simulated compute FLOPs and DRAM access latency of Multi-Head / Multi-Latent Attention ($\mathcal{O}(P+t)$), but does **not** dynamically squeeze the 35 GiB expert cache budget.
3. **Physical KV-Cache Footprint:** For DeepSeek-V3 MLA (FP16, 40 layers, rank 512, rope head dim 64), the KV-cache consumes:
   - At $P=128$: $0.176\text{ GiB}$
   - At $P=512$: $0.703\text{ GiB}$
   - At $P=2048$: $2.812\text{ GiB}$
   - At $P=4096$: $5.625\text{ GiB}$  
   In systems without dedicated caching, this growth would compress available expert capacity. In our architecture, the 35 GiB expert cache is preserved, isolating the pure compute-slack expansion mechanism.

---

## 4. Associated Visualizations

- [Figure 4: MLA KV-Cache Memory Footprint](file:///home/mohman/ca_project/mohman/point_6/final_results/figures/context_length/fig4_context_length_kv_cache.png)
- [Figure 5: Compute Duration & Overlap Slack vs. Context Length](file:///home/mohman/ca_project/mohman/point_6/final_results/figures/context_length/fig5_context_length_attention_time.png)
- [Figure 6: Context Length Sensitivity: Mean Token Latency Across P](file:///home/mohman/ca_project/mohman/point_6/final_results/figures/context_length/fig6_context_length_expert_cache.png)
- [Figure 14: FA-LRU Reloads Saved Across Context Lengths](file:///home/mohman/ca_project/mohman/point_6/final_results/figures/combined/fig14_lru_vs_falru_comparison.png)

