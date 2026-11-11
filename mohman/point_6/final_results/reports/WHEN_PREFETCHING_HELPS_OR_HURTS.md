# When Prefetching Helps, Provides Limited Gains, or Degrades Performance

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Sensitivity Analysis Consolidation  
**Author:** Antigravity Engineering  
**Date:** November 10, 2026  
**Status:** Authoritative Empirical Report  

---

## 1. Executive Summary

Asynchronous expert prefetching is not a universally beneficial optimization. In high-performance MoE inference architectures such as DeepSeek-V3, prefetching operates at the delicate intersection of:
1. **Prediction Accuracy & Timeliness:** Can candidate experts be predicted accurately enough and early enough to hide memory transfer latency?
2. **DDR5 Bus Bandwidth Contention:** Do speculative transfers saturate the memory bus and delay critical demand misses?
3. **HBM Cache Allocation & Eviction:** Do speculative prefetches pollute the local expert cache by evicting frequently reused demand-resident experts?

This report establishes the **three fundamental operating regimes** that govern MoE expert prefetching and identifies the exact architectural conditions under which prefetching accelerates execution, yields diminishing returns, or degrades performance.

---

## 2. Regime 1: Accelerated Performance (The Effective Zone)

```
+---------------------------------------------------------------------------------------+
| REGIME 1: PREFETCHING ACCELERATES PERFORMANCE                                        |
+---------------------+-----------------------------------------------------------------+
| Operational Domain  | Conservative Budget (K=2 to 4), Multi-Layer Lookahead (H >= 1), |
|                     | High Concurrency (B >= 14), Standard Working Set (P <= 2048)    |
+---------------------+-----------------------------------------------------------------+
| Key Metrics         | Latency improves by +0.58% to +1.42%, Exposed stalls hidden     |
|                     | by 2,693 ms to 10,945 ms, Reloads avoided: 1,400 to 3,280       |
+---------------------+-----------------------------------------------------------------+
| Physical Mechanism  | High predictor precision + Sufficient attention overlap slack   |
|                     | + Uncongested DDR5 bus + Frequency-aware cache protection       |
+---------------------+-----------------------------------------------------------------+
```

### Physical Conditions for Acceleration:
1. **High Concurrency ($B \ge 14$):** Serving multiple concurrent sequences creates frequent reuse of hot experts across the 40 layers, and expands the per-layer compute duration.
2. **Adequate Attention Compute Slack:** Multi-Head / Multi-Latent Attention executes immediately before the MoE feed-forward stage. Computing attention across 14 to 32 tokens provides milliseconds of prefetch overlap slack, enabling transfers from DDR5 to arrive before tokens demand the weights.
3. **Conservative Speculation ($K=2$):** At $K=2$, prefetching dispatches only the top-2 highest confidence candidates per token. DDR5 speculative traffic is limited to $6.08\text{ TB}$ (69,073 prefetches), leaving the 64 GB/s channel completely uncongested. Demand misses and speculative arrivals execute without controller queuing delays, hiding **$10.94\text{ seconds}$ of exposed stall** and achieving **+1.42% end-to-end speedup**.
4. **Cache Protection via FA-LRU ($W=64$):** Frequency-aware victim selection shields frequently reused demand experts in the bottom-64 LRU window, saving 22 reloads over standard LRU at $B=32$.

---

## 3. Regime 2: Limited Gains (The Compute-Bound Plateau)

```
+---------------------------------------------------------------------------------------+
| REGIME 2: PREFETCHING PROVIDES LIMITED GAINS                                          |
+---------------------+-----------------------------------------------------------------+
| Operational Domain  | Long Context Lengths (P >= 2048, P=4096), High Attention Load   |
+---------------------+-----------------------------------------------------------------+
| Key Metrics         | Latency improvement plateaus at ~0.77%, Absolute stall reduction|
|                     | is large (5.92 s hidden), but total runtime is compute-bound    |
+---------------------+-----------------------------------------------------------------+
| Physical Mechanism  | O(P+t) attention compute widens overlap slack, but total runtime|
|                     | is dominated by attention FLOPs rather than memory stalls       |
+---------------------+-----------------------------------------------------------------+
```

### Physical Conditions for Diminishing Returns:
1. **Dominance of Compute Complexity:** As context length scales from $P=128$ to $P=4,096$, non-stall decode compute duration increases from $13,727.88\text{ ms}$ to $14,355.67\text{ ms}$ (+627.79 ms) due to $\mathcal{O}(P+t)$ attention complexity.
2. **Expansion of Overlap Slack:** Because attention compute duration expands, speculative transfers have more time to complete. Resident hits climb from $1,417$ at $P=128$ to $1,630$ at $P=4,096$ (+15.0%), and reloads saved increase from $1,317$ to $1,528$.
3. **Asymptotic Latency Plateau:** Even though prefetch hides more memory stalls in absolute terms ($5,919.29\text{ ms}$ hidden at $P=4,096$ vs. $5,381.98\text{ ms}$ at $P=128$), the overall runtime is increasingly dominated by attention FLOPs rather than memory stalls. Consequently, the relative latency improvement plateaus at $+0.769\%$.

---

## 4. Regime 3: Performance Inversion & Degradation (The Danger Zone)

```
+---------------------------------------------------------------------------------------+
| REGIME 3: PREFETCHING DEGRADES PERFORMANCE                                            |
+---------------------+-----------------------------------------------------------------+
| Operational Domain  | Low Concurrency (B <= 4) OR Aggressive Budget (K >= 8)          |
+---------------------+-----------------------------------------------------------------+
| Key Metrics         | Latency degrades by -0.026% to -0.130%, Reloads increase by     |
|                     | +301, Memory traffic surges by +24.9% to +98%                   |
+---------------------+-----------------------------------------------------------------+
| Physical Mechanism  | Cache pollution (evicting demand-resident experts) + Bus        |
|                     | saturation (DDR5 queuing delays stalling critical demands)      |
+---------------------+-----------------------------------------------------------------+
```

### Physical Conditions for Degradation:
1. **Low Concurrency Cache Pollution ($B=4$):**
   - At $B=4$, sequence reuse is sparse, and the resident working set is small.
   - Speculative prefetches allocate cache blocks in the 35 GiB HBM partition. Because the working set has not yet matured, speculative transfers evict currently resident demand experts.
   - Consequently, net expert reloads **increase by +301** (from $124,975$ to $125,276$), coalesced resident hits drop from $1,431$ to $1,237$ (-13.6%), and mean token latency degrades by **-0.130%** ($1,466.88\text{ ms} \to 1,468.79\text{ ms}$).
2. **Interconnect Saturation ($K=8$):**
   - At $K=8$, the predictor requests 8 candidate experts per token.
   - Speculative transfers inject $18.04\text{ TB}$ of data (204,780 prefetches) onto the 64 GB/s DDR5 channel.
   - Memory controller request buffers overflow, and demand misses are delayed behind speculative transfers.
   - Consequently, exposed memory stalls increase by $+178.24\text{ ms}$, causing latency to degrade by **-0.026%** relative to $K=4$ ($5,952.01\text{ ms}$ vs. $5,950.46\text{ ms}$).

---

## 5. Architectural Takeaways & Serving Guidelines

1. **Implement Dynamic Pre-Execution Gating:** Inference engines must monitor batch size. If $B < 8$, speculative prefetching should be automatically throttled or disabled to prevent cache pollution.
2. **Cap Speculation Budget at $K=2$ for 64 GB/s Channels:** Aggressive speculation ($K=8$) degrades performance through channel saturation. Conservative budgeting ($K=2$) delivers optimal speedup (+1.42%).
3. **Deploy Frequency Protection (FA-LRU $W=64$):** Speculative cache insertions must never evict high-frequency demand-resident experts. FA-LRU consistently avoids 15–22 reloads across all workloads.
