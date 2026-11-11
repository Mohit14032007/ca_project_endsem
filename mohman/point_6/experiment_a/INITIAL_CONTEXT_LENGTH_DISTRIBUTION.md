# Point 6 — Authentic Initial Context Length Distribution Analysis

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Experiment A: Inferred Initial Context Length Audit  
**Author:** Antigravity Engineering  
**Date:** November 10, 2026  
**Status:** Authoritative & Completed  

---

## 1. Executive Summary

This report delivers a rigorous empirical analysis of the inferred initial context lengths ($P$) extracted from the authentic MoE-Anywhere trace dataset:

`LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet`

The dataset records token-level expert routing for **74 prompt sequences** (IDs 0–73) across **40 MoE layers** with Top-8 routing for DeepSeek-V3. In every sequence, exactly **257 contiguous generation steps** are logged. Absolute token positions are recorded in `token_pos`.

Because autoregressive generation begins at `token_pos = min_token_pos`, we derive:

$$\text{inferred\_initial\_P} = \min(\text{token\_pos})$$

This inference represents the length of the prompt context immediately preceding the first generated token.

> [!IMPORTANT]
> **Data Provenance Caveat:** This inferred $P$ is derived strictly from the first recorded decode token position. The parquet file contains neither the original prompt text nor prefill routing decisions. It must not be presented as independently tokenized prompt text.

---

## 2. Overall Statistical Summary

All 74 prompts passed 100% of validation gates with zero missing, dropped, or duplicated records ($74 \times 257 \times 40 = 760,720$ total records).

| Statistical Metric | Inferred Prompt Length ($P$) | Notes & Architectural Meaning |
| :--- | :---: | :--- |
| **Total Prompt Sequences ($N$)** | **74** | 100% authentic MoE-Anywhere benchmark prompts |
| **Minimum ($P_{\min}$)** | **13 tokens** | Prompt ID 18 (`chat_en`, range 13–269) |
| **Maximum ($P_{\max}$)** | **64 tokens** | Prompt ID 62 (`summarize`, range 64–320) |
| **Full Dynamic Range / Span** | **51 tokens** | $P_{\max} - P_{\min} = 64 - 13 = 51$ |
| **Mean ($\mu$)** | **24.62 tokens** | Modest conversational prompt length |
| **Median ($P_{50}$)** | **22.00 tokens** | Half of all prompts have $\le 22$ prompt tokens |
| **Standard Deviation ($\sigma$)** | **8.61 tokens** | Narrow dispersion across the benchmark suite |
| **25th Percentile ($P_{25}$)** | **19.00 tokens** | First quartile |
| **75th Percentile ($P_{75}$)** | **27.75 tokens** | Third quartile |
| **Interquartile Range (IQR)** | **8.75 tokens** | 50% of all prompts cluster between 19 and 28 tokens |

---

## 3. Frequency Distribution of Inferred Initial Context Lengths

The 74 prompts exhibit 24 distinct values of inferred $P$. Prompts heavily cluster between 16 and 28 tokens ($75.7\%$ of all prompts fall in this interval).

| Inferred $P$ (Tokens) | Prompt Count ($n$) | Percentage (%) | Cumulative Count | Cumulative (%) | Task Types Represented |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **13** | 1 | 1.35% | 1 | 1.35% | `chat_en` |
| **16** | 4 | 5.41% | 5 | 6.76% | `chat_en`, `chat_es`, `chat_pt` |
| **17** | 5 | 6.76% | 10 | 13.51% | `chat_en`, `chat_pt`, `code`, `creative` |
| **18** | 6 | 8.11% | 16 | 21.62% | `chat_en`, `chat_es`, `chat_pt`, `creative` |
| **19** | 6 | 8.11% | 22 | 29.73% | `chat_en`, `chat_pt`, `code`, `math` |
| **20** | 8 | 10.81% | 30 | 40.54% | `chat_en`, `chat_es`, `code`, `summarize` |
| **21** | 6 | 8.11% | 36 | 48.65% | `chat_en`, `creative` |
| **22** | 5 | 6.76% | 41 | 55.41% | `chat_pt`, `reasoning`, `translate` |
| **23** | 1 | 1.35% | 42 | 56.76% | `translate` |
| **24** | 4 | 5.41% | 46 | 62.16% | `code`, `translate` |
| **25** | 3 | 4.05% | 49 | 66.22% | `code`, `math` |
| **26** | 4 | 5.41% | 53 | 71.62% | `code`, `math` |
| **27** | 2 | 2.70% | 55 | 74.32% | `math`, `reasoning` |
| **28** | 3 | 4.05% | 58 | 78.38% | `code` |
| **30** | 2 | 2.70% | 60 | 81.08% | `math`, `tools` |
| **31** | 2 | 2.70% | 62 | 83.78% | `math` |
| **34** | 1 | 1.35% | 63 | 85.14% | `tools` |
| **35** | 5 | 6.76% | 68 | 91.89% | `reasoning`, `tools` |
| **39** | 1 | 1.35% | 69 | 93.24% | `reasoning` |
| **40** | 1 | 1.35% | 70 | 94.59% | `tools` |
| **43** | 1 | 1.35% | 71 | 95.95% | `tools` |
| **44** | 1 | 1.35% | 72 | 97.30% | `reasoning` |
| **45** | 1 | 1.35% | 73 | 98.65% | `reasoning` |
| **64** | 1 | 1.35% | 74 | 100.00% | `summarize` |
| **Total** | **74** | **100.00%** | **74** | **100.00%** | **10 Task Domains** |

---

## 4. Task-Wise Context Length Breakdown

The prompt lengths exhibit task-dependent clustering reflecting standard user interaction styles:
- **Chat and conversational tasks** (`chat_en`, `chat_es`, `chat_pt`, `creative`) have the shortest prompts (means $17.5$–$19.3$ tokens).
- **Code and Math tasks** have intermediate lengths (means $23.0$–$24.8$ tokens).
- **Reasoning, Tools, and Summarization** have the longest prompts (means $32.1$–$38.8$ tokens), with the maximum observed length ($P=64$) occurring in `summarize`.

| Task Classification | Prompt Count ($n$) | Min $P$ | Max $P$ | Mean $P$ | Median $P$ | Std Dev ($\sigma$) | 25th Pct ($P_{25}$) | 75th Pct ($P_{75}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `chat_es` | 4 | 16 | 20 | 17.50 | 17.0 | 1.91 | 16.00 | 18.50 |
| `chat_en` | 10 | 13 | 21 | 17.90 | 18.5 | 2.28 | 17.00 | 19.00 |
| `creative` | 4 | 17 | 21 | 18.50 | 18.0 | 1.73 | 17.75 | 18.75 |
| `chat_pt` | 10 | 16 | 22 | 19.30 | 19.5 | 2.00 | 18.00 | 21.00 |
| `code` | 12 | 17 | 28 | 23.00 | 24.0 | 3.67 | 20.00 | 25.25 |
| `translate` | 4 | 22 | 24 | 23.00 | 23.0 | 1.15 | 22.00 | 24.00 |
| `math` | 10 | 19 | 31 | 24.80 | 25.5 | 4.05 | 21.50 | 26.75 |
| `reasoning` | 8 | 22 | 44 | 32.13 | 31.5 | 8.51 | 25.75 | 39.25 |
| `tools` | 8 | 30 | 43 | 34.75 | 35.0 | 3.88 | 33.25 | 35.00 |
| `summarize` | 4 | 20 | 64 | 38.75 | 35.5 | 19.92 | 24.50 | 49.75 |
| **All Tasks Combined** | **74** | **13** | **64** | **24.62** | **22.0** | **8.61** | **19.00** | **27.75** |

---

## 5. Visual Analysis of Context Length Distribution

The distribution of inferred initial context lengths is visualized below (histogram of prompt counts stacked by task, alongside task-wise boxplots and individual prompt scatter points):

![Authentic Initial Context Length Distribution](file:///home/mohman/.gemini/antigravity/brain/b7db0340-f5d9-4df3-86d8-be407d44242d/figures/point_6/INITIAL_CONTEXT_LENGTH_HISTOGRAM.png)

*Figure 1: Authentic MoE-Anywhere initial context length distribution. Top: Histogram of prompt counts stacked by task category, indicating central tendencies (Mean = 24.62, Median = 22.0, IQR = [19.0, 27.75]). Bottom: Task-wise distributions showing the separation between conversational tasks ($\le 22$ tokens) and tool-use / reasoning tasks ($30$–$45$ tokens).*

---

## 6. Experimental Implications for Sensitivity Analysis

### 6.1 Insufficiency of Natural Variation (13–64 Tokens) for Sensitivity Analysis
Can the natural variation in inferred prompt lengths ($P \in [13, 64]$) support a scientifically meaningful context-length sensitivity experiment?

To answer this quantitatively, consider the physical effects of context length on the inference engine:

#### 1. KV-Cache Memory Impact Across Natural Range ($P=13$ vs. $P=64$)
Using the DeepSeek-V3 Multi-Head Latent Attention (MLA) memory formulation:
$$\text{KV Bytes per Token per Sequence} = (d_{\text{kv\_lora\_rank}} + d_{\text{qk\_rope\_head\_dim}}) \times L \times \text{precision\_byte} = (512 + 64) \times 40 \times 2 = 46,080 \text{ bytes}$$

- At $P = 13$ (Minimum):
  - $M_{\text{KV}}(B=1) = 13 \times 46,080 = 599.0 \text{ KiB}$ ($0.58$ MiB)
  - $M_{\text{KV}}(B=32) = 32 \times 13 \times 46,080 = 18.72 \text{ MiB}$ ($0.018$ GiB)
- At $P = 64$ (Maximum):
  - $M_{\text{KV}}(B=1) = 64 \times 46,080 = 2.88 \text{ MiB}$
  - $M_{\text{KV}}(B=32) = 32 \times 64 \times 46,080 = 92.16 \text{ MiB}$ ($0.090$ GiB)
- **Net Memory Differential:**
  $$\Delta M_{\text{KV}} = 92.16 \text{ MiB} - 18.72 \text{ MiB} = 73.44 \text{ MiB} \approx 0.072 \text{ GiB}$$
- **Displacement of Expert Weights:**
  A single DeepSeek-V3 routed expert requires $88.08$ MiB.
  $$\text{Displaced Experts} = \frac{73.44 \text{ MiB}}{88.08 \text{ MiB}} \approx 0.83 \text{ experts}$$
  Expanding $P$ from 13 to 64 displaces **less than one expert** out of hundreds in the 35 GiB HBM cache!

#### 2. Decode Attention Compute Impact Across Natural Range
In `attention_gen_impl.cpp`, scoring FLOPs per step scale as:
$$\text{FLOPs}(n) = 2 \times \text{head\_dim} \times n \times \text{attention\_group\_size}$$
At $P=13$, $n = 14$. At $P=64$, $n = 65$. On an A100 GPU capable of 312 TFLOP/s, this compute differential corresponds to less than **$0.02$ microseconds per step**, which is completely negligible compared to the ~200 microsecond expert transfer latency.

### 6.2 Comparison: Natural Range vs. Proposed Long-Context Sweep

| Context Length Regime | $P$ Range (Tokens) | KV Cache Size ($B=32$) | Displaced HBM Fraction (35 GiB) | Displaced Experts | Attention Compute Slack | Eviction Pressure |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Natural Trace Inferred** | **13 – 64** | **18.7 – 92.2 MiB** | **0.05% – 0.26%** | **< 1 expert** | Negligible ($\Delta < 0.02\mu\text{s}$) | Undetectable |
| **Proposed Sweep Level 1** | $P = 128$ | $188.7$ MiB | $0.53\%$ | $2$ experts | Minimal | Nominal |
| **Proposed Sweep Level 2** | $P = 512$ | $754.9$ MiB | $2.11\%$ | $8$ experts | Modest | Low |
| **Proposed Sweep Level 3** | $P = 2,048$ | $3.02$ GiB | $8.63\%$ | $34$ experts | Noticeable | Moderate |
| **Proposed Sweep Level 4** | $P = 4,096$ | $6.04$ GiB | **17.26%** | **68 experts** | Substantial | **Severe (Thrashing)** |

---

## 7. Authoritative Conclusions & Recommendation

### 7.1 Can 74 Real Prompts Support Natural Initial-Context-Length Sensitivity?
**NO.** The natural variation in inferred prompt lengths ($13 \le P \le 64$) is too small by two orders of magnitude to induce measurable memory wall stress, cache eviction differences, or prefetch timeliness variations. The entire span ($51$ tokens) represents less than one displaced expert in HBM and less than 0.05% of the attention latency envelope.

### 7.2 Can 74 Real Prompts Support Controlled Long-Context Sensitivity?
**YES.** As proven in the Phase 0 simulator audit, `LLMSimulator`'s trace replay indexing calculates $\text{rel\_token\_id} = \text{token\_id} - \text{input\_len} = t$, which decouples the authentic decode routing trace from the configured context length $P$. The 74 authentic prompts can be cleanly simulated under controlled initial context lengths of $P \in \{128, 512, 2048, 4096\}$ to evaluate genuine KV-cache memory pressure and decode attention duration without altering empirical routing decisions.

---

## 8. Executive Go / No-Go Decision

```
========================================================================================
             POINT 6 EXPERIMENT A: PROMPT CONTEXT LENGTH DECISION GATE
========================================================================================
 Proposed Experimental Direction              | Status       | Action / Recommendation
---------------------------------------------+--------------+---------------------------
 Natural Variation Sensitivity ($P \in [13,64]$)| NO-GO        | Reject (Physically imperceptible)
 Controlled Long-Context Sweep ($P \in [128,4096]$)| IMMEDIATE GO | Approve (Approach B execution)
 Replay of Authentic Decode Traces ($T \le 257$)| IMMEDIATE GO | Approve (100% trace validity)
========================================================================================
```

- **Natural Prompt Length Sensitivity:** **NO-GO**. The variation from 13 to 64 tokens does not create meaningful systems-level pressure.
- **Controlled Long-Context Sensitivity (Approach B):** **GO**. Scale $P \in \{128, 512, 2048, 4096\}$ to evaluate true KV cache displacement and attention slack using authentic decode traces.

