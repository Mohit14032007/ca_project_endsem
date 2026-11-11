# Point 6 — Sensitivity Analysis Controlled Experiment Design

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Sensitivity Analysis Consolidation  
**Author:** Antigravity Engineering  
**Date:** November 10, 2026  
**Status:** Authoritative & Completed  

---

## 1. Executive Summary & Objective

Point 6 requires an exhaustive, empirical **Sensitivity Analysis** evaluating how system performance, expert reuse, cache residency, and prefetch effectiveness respond across four core dimensions:
1. **Batch Size ($B \in \{4, 14, 32\}$):** Cross-sequence concurrency, spatial reuse, and working-set expansion.
2. **Initial Context Length ($P \in \{128, 512, 2048, 4096\}$):** Attention duration scaling and temporal transfer overlap slack.
3. **Inter-Layer Lookahead ($H \in \{1, 2, 4\}$):** Prediction horizon trade-off between transfer lead time and accuracy decay.
4. **Prefetch Candidate Budget ($K \in \{2, 4, 8\}$):** Speculative transfer aggressiveness, bandwidth saturation, and cache pollution.

To maintain direct, rigorous scientific comparability, all experiments use a standardized, non-combinatorial experimental matrix anchored at a common intersection point.

---

## 2. Standardized Reference Architecture & Anchor Point

| Parameter | Specification | Physical Meaning |
| :--- | :--- | :--- |
| **Model** | DeepSeek-V3 Mixtral-style | 40 MoE stages, 256 routed experts/stage, Top-8 routing |
| **Expert Footprint** | $88,080,384$ bytes ($84.0$ MiB) | $W_1 + W_3 + W_2$ linear projection weights |
| **HBM Cache Budget** | 35 GiB ($37,580,963,840$ bytes) | Constrained HBM regime exposing memory stalls |
| **DDR5 Memory Bus** | 64 GB/s event-driven DRAM | Models host-to-device offchip transfer latency |
| **Scheduling Engine** | `FORWARD_LOOKAHEAD` | Asynchronous forward-layer prefetch dispatch |
| **Predictor Engine** | CSTP | Spatial-temporal transition matrix + EMA confidence |
| **Generation Horizon** | $T = 128$ tokens | Authentic contiguous generation steps |
| **Trace Source** | `e1-sage-a1.5.parquet` | Authentic MoE-Anywhere routing traces |

### The Common Anchor Configuration:
- **Batch Size:** $B = 32$
- **Initial Context Length:** $P = 2,048$ tokens
- **Generation Horizon:** $T = 128$ tokens
- **Inter-Layer Lookahead:** $H = 1$ layer ahead
- **Prefetch Budget:** $K = 4$ candidate experts per token
- **HBM Budget:** $35\text{ GiB}$

The anchor configuration serves as the common hub across all four experimental dimensions:
- In **Experiment 1 (Batch Size)**, it represents the $B = 32$ point.
- In **Experiment 2 (Context Length)**, it represents the $P = 2,048$ point.
- In **Experiment 3 (Lookahead)**, it represents the $H = 1$ point.
- In **Experiment 4 (Prefetch Budget)**, it represents the $K = 4$ point.

---

## 3. The Four Evaluated Systems

Across every experimental dimension, systems are evaluated under standard notation:
- **Configuration A — Baseline:** No prefetching, standard LRU eviction.
- **Configuration B — Prediction-Only:** CSTP executes in shadow mode; zero speculative transfers issued.
- **Configuration C — Prediction-Driven Prefetching:** CSTP asynchronous prefetching + standard LRU eviction.
- **Configuration D — Complete System:** CSTP asynchronous prefetching + Frequency-Aware LRU (FA-LRU $W=64$).

---

## 4. Controlled Parameter Grid

```
+---------------------------------------------------------------------------------------+
|                             POINT 6 EXPERIMENTAL MATRIX                               |
+---------------------+-------------------+---------------------+-----------------------+
| Experiment          | Varied Parameter  | Fixed Parameters    | Systems Evaluated     |
+---------------------+-------------------+---------------------+-----------------------+
| Exp 1: Batch Size   | B in {4, 14, 32}  | P=2048, T=128, H=1, | Config A (Baseline)   |
|                     |                   | K=4, HBM=35GiB      | Config C (CSTP+LRU)   |
|                     |                   |                     | Config D (FA-LRU W=64)|
+---------------------+-------------------+---------------------+-----------------------+
| Exp 2: Context Len  | P in {128, 512,   | B=32, T=128, H=1,   | Config A (Baseline)   |
|                     |       2048, 4096} | K=4, HBM=35GiB      | Config C (CSTP+LRU)   |
|                     |                   |                     | Config D (FA-LRU W=64)|
+---------------------+-------------------+---------------------+-----------------------+
| Exp 3: Lookahead    | H in {1, 2, 4}    | B=32, P=2048, T=128,| Config C (CSTP+LRU)   |
|                     |                   | K=4, HBM=35GiB      | Config D (FA-LRU W=64)|
+---------------------+-------------------+---------------------+-----------------------+
| Exp 4: Prefetch K   | K in {2, 4, 8}    | B=32, P=2048, T=128,| Config C (CSTP+LRU)   |
|                     |                   | H=1, HBM=35GiB      | Config D (FA-LRU W=64)|
+---------------------+-------------------+---------------------+-----------------------+
```

### Trace Nesting Guarantee
To eliminate trace variance:
- $B = 4$ uses sequences 0–3 (Prompts 60, 61, 62, 63).
- $B = 14$ uses sequences 0–13 (Prompts 60–73).
- $B = 32$ uses sequences 0–31 (Prompts 60–73, 52–59, 0–9).
Thus: $\text{Trace}(B=4) \subset \text{Trace}(B=14) \subset \text{Trace}(B=32)$.
Paired comparisons evaluate identical sequence histories.

