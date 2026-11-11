# Point 6 — Limitations, Architectural Boundaries, and Comparability Analysis

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Sensitivity Analysis Consolidation  
**Author:** Antigravity Engineering  
**Date:** November 10, 2026  
**Status:** Authoritative & Completed  

---

## 1. Executive Summary

Scientific integrity demands transparent disclosure of simulator fidelity, memory modeling boundaries, dataset characteristics, and cross-campaign comparability. This report outlines the explicit limitations, assumptions, and valid interpretation scopes for the Point 6 experimental campaign.

---

## 2. Hardware and Simulator Modeling Fidelity

### 2.1 Simulator-Based Evaluation (Non-Silicon)
All reported results are produced by `LLMSimulator`—an event-driven architectural performance simulator for large language models.
- **What is Modeled:** Kernel-level compute durations (FLOPs/compute peak), DRAM request latency (read/write bandwidth, queuing, request serialization), HBM expert cache eviction dynamics (LRU and FA-LRU), and interconnect routing.
- **What is Not Claimed:** These are **not physical measurements on real GPU hardware**. Latency numbers reflect simulated architectural nanoseconds derived from validated hardware parameters (NVIDIA A100-80GB SXM4, 312 TFLOP/s FP16 Tensor Core compute, 2,039 GB/s HBM2e bandwidth, and 64 GB/s offchip DDR5 host-to-device bandwidth).

### 2.2 Memory Model Fidelity (Event-Driven DDR5 vs. Ramulator)
- In Point 6 configurations, DDR5 transfers are modeled using an **event-driven, bandwidth-calibrated queuing model** (`USE_EVENT_DRIVEN_DRAM=true`, `DDR5_BANDWIDTH_GBPS=64.0`).
- While this model reproduces channel bandwidth saturation, transfer concurrency limits, and serialization delays, it is **not a cycle-accurate DDR5 DRAM timing simulation** (such as full Ramulator2 cycle-by-cycle pin-level simulation).
- This distinction must be maintained in all technical presentations.

### 2.3 Memory Budget Accounting: Dedicated Expert Cache
- In `LLMSimulator`, `device->cache_capacity_bytes` is a static dedicated capacity allocated specifically for expert weights ($35\text{ GiB} = 37,580,963,840$ bytes).
- As established in the `MEMORY_MODEL_AUDIT.md`, the native simulator does not implement a dynamic unified memory allocator that subtracts KV-cache allocations from `cache_capacity_bytes` at runtime.
- **Context-Length Sensitivity Scope:** Consequently, variations in initial context length $P \in \{128, 512, 2048, 4096\}$ directly scale:
  1. Attention scoring and context FLOPs and DRAM access volumes ($\mathcal{O}(P + t)$).
  2. Per-layer attention execution duration (logged in `atten_gen`).
  3. The temporal compute slack window available to overlap asynchronous expert prefetching.
  4. The analytical KV-cache memory footprint ($M_{\text{KV}} \in [180\text{ MiB}, 5.625\text{ GiB}]$ for $B=32$).
- However, context scaling does **not** dynamically trigger expert cache capacity reductions in native `LLMSimulator`. This separation is rigorously maintained in all analyses.

---

## 3. Dataset Boundaries & Trace Provenance

### 3.1 Trace Provenance
- All routing traces are extracted directly from the authentic MoE-Anywhere profiling dataset:  
  `LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet`.
- Traces capture real activation decisions for DeepSeek-V3 (40 MoE stages, 256 routed experts, Top-8 routing).
- Exactly 257 contiguous autoregressive decode steps are present for every prompt.

### 3.2 Inferred Initial Context Lengths vs. Independent Tokenization
- The upstream parquet file logs tokens starting from $\text{min\_token\_pos} \in [13, 64]$.
- The prompt text strings, vocabulary token IDs, and prefill routing decisions are completely absent from the dataset.
- The parameter $\text{inferred\_initial\_P}$ is inferred from the sequence offset at which trace logging began. It must **not** be presented as independently verified prompt text tokenization.
- Point 6 context-length sensitivity experiments evaluate **controlled assumed initial context lengths ($P$) during decode replay (Approach B)**, not genuine prompt prefill execution (Approach A).

---

## 4. Cross-Campaign Comparability Guardrails

To prevent misleading cross-experiment claims, comparisons between historical campaigns are restricted as follows:

| Historical Campaign | Key Parameters | Direct Comparability to Point 6 | Appropriate Usage Scope |
| :--- | :--- | :---: | :--- |
| **Point 4 Big Experiment** | $B \in \{14, 32\}, P=128, T=128, \text{HBM}=35\text{GiB}, W=64$ | **DIRECTLY COMPARABLE** | Reused directly as the $P=128$ reference point |
| **Point 3 Large Workloads** | $B \in \{14, 32, 48, 70\}, T=16, \text{HBM}=80\text{GB}, W=0$ | **NON-COMPARABLE** | Supporting qualitative evidence of high-concurrency reuse only |
| **Point 4 Focused 24** | $B \in \{1, 2, 4\}, T=20, \text{HBM} \in \{10, 35, 74.5\}\text{GiB}, W=16$ | **NON-COMPARABLE** | Low-concurrency scaling trends only; cannot be merged into Point 6 curves |
| **Point 6 Experiment A** | Natural $P \in [13, 64]$, 74 prompts | **NON-COMPARABLE** | Proves the necessity of controlled long-context sweeps |

---

## 5. Statistical Interpretation of Repetitions

- When two simulation repetitions are executed (`rep1` and `rep2`), identical routing traces and deterministic pseudorandom seeds are used.
- In a cycle-calibrated deterministic simulator, identical inputs produce identical outputs with zero variance ($\Delta = 0.00\%$).
- Host wall-clock runtimes may fluctuate slightly ($1$–$3\%$) due to operating system thread scheduling, but all simulated metrics (stalls, hits, traffic, latencies) are deterministic.
- Repetitions confirm simulation stability and reproducibility; they must not be treated as independent stochastic samples.

