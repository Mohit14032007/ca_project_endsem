# Point 6 — Executive Go / No-Go Decision and Feasibility Report

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Phase 0: Feasibility & Trace Audit  
**Author:** Antigravity Engineering  
**Date:** November 10, 2026  
**Status:** Authoritative & Completed  

---

## 1. Executive Summary & Authoritative Decisions

Following an exhaustive read-only audit of trace datasets (`LLMSimulator/data/downloads/moe_anywhere/` and `moe_suite/`) and the `LLMSimulator` C++ simulation engine, this report delivers the formal **Go / No-Go recommendations** for the Point 6 Sensitivity Analysis campaign.

```
========================================================================================
                          POINT 6 FEASIBILITY DECISION GATE
========================================================================================
 Capability / Exploration Area                | Status       | Action / Recommendation
---------------------------------------------+--------------+---------------------------
 Approach A: Genuine Prefill Routing          | NO-GO        | Blocked (Zero prefill data)
 Approach B: Decode-Only Context Scaling      | IMMEDIATE GO | Approved (KV & Attn sweep)
 Approach C: Generation Horizon Scaling ($T$) | IMMEDIATE GO | Approved ($T \in [16, 256]$)
 Sweep: Inter-Layer Lookahead ($H \in \{1,2,4\}$)| IMMEDIATE GO | Approved (Timeliness sweep)
 Sweep: Prefetch Budget ($K \in \{2,4,8\}$)   | IMMEDIATE GO | Approved (Bandwidth sweep)
 Sweep: Batch Size Scaling ($B \in \{14,32\}$)| IMMEDIATE GO | Approved (Reuse sweep)
 Sweep: Eviction Policy (LRU vs FA-LRU W=64)  | IMMEDIATE GO | Approved (Cache protection)
========================================================================================
```

---

## 2. Detailed Rationale for Decisions

### 2.1 Decision 1: Approach A (Genuine Prefill Sensitivity) — NO-GO
- **Primary Blocker:** Empirical data absence.
  - The MoE-Anywhere parquet file `e1-sage-a1.5.parquet` contains **zero records for prompt tokens** ($0 \le t < \text{min\_token\_pos}$). Only autoregressive decode steps ($1 \dots 257$) were recorded during upstream trace profiling.
  - The MoE-Suite dataset (`LLMSimulator/data/downloads/moe_suite/*.parquet`) records **0.0% prompt tokens** across all 12 parquet files (`token_flag_prompt == False` for 100% of rows), and models GLM-5.2 (75 layers) rather than DeepSeek-V3 (40 layers).
  - Neither dataset provides input prompt text, tokenized vocabulary sequences, or intermediate router logits.
- **Simulator Blocker:** The evaluation harness `eval/test_deepseekv3_mixtral_style.cpp` hardcodes `system_config.prefill_mode = false;` and lacks a prefill routing ingestion module.
- **Scientific Integrity Mandate:** Synthesizing prompt routing traces without underlying text or claiming to have measured prefill expert prefetching would be scientifically invalid. Therefore, Approach A is classified as **NO-GO**.

---

### 2.2 Decision 2: Approach B (Decode-Only Context-Length Sensitivity) — IMMEDIATE GO
- **Physical & System Soundness:**
  - Real-world long-context inference bottlenecks during the decode phase due to massive KV-cache footprints and longer attention durations.
  - DeepSeek-V3's Multi-Head Latent Attention (MLA) requires $46,080$ bytes per token per sequence. At $B=32$, increasing initial context length from $P=128$ to $P=4,096$ expands the KV cache from $0.18$ GiB to $5.76$ GiB.
  - In a constrained 35 GiB HBM budget, this displaces **68 expert slots**, creating intense eviction competition that rigorously stresses LRU vs. FA-LRU.
- **Simulator Capability:**
  - Configurable directly via `simulation.input_len: P` in YAML.
  - Trace replay indexing in `sequence.cpp` calculates $\text{rel\_token\_id} = \text{token\_id} - \text{input\_len} = t$, completely preserving authentic empirical expert routing decisions regardless of $P$.
  - Scoring and context attention durations scale linearly with $P + t$, providing accurate attention execution slack for prefetch overlap.
- **Status:** **IMMEDIATE GO**.

---

### 2.3 Decision 3: Approach C (Decode Generation Horizon Sensitivity $T$) — IMMEDIATE GO
- **Physical & System Soundness:**
  - Evaluating generation horizons $T \in \{16, 128, 256\}$ tests long-horizon temporal prefetch stability, predictor saturation, and cumulative cache pollution.
- **Trace Support:**
  - `e1-sage-a1.5.parquet` contains exactly 257 contiguous tokens per prompt for all 74 prompts, supporting horizons up to $T=257$ with zero extrapolation.
- **Status:** **IMMEDIATE GO**.

---

### 2.4 Decision 4: Prefetch Hyperparameter Sweeps ($H, K, B$, Eviction Policy) — IMMEDIATE GO
- **Inter-Layer Lookahead ($H \in \{1, 2, 4\}$):** Fully supported via `LivePredictionEngine` and YAML `prediction_horizon`. Evaluates earlier dispatch against prediction accuracy decay.
- **Prefetch Budget ($K \in \{2, 4, 8\}$):** Fully supported via YAML `budget`. Evaluates bus saturation and cache pollution.
- **Batch Size ($B \in \{14, 32\}$):** Fully supported via YAML `max_batch_size`.
- **Eviction Policy (LRU vs. FA-LRU $W=64$):** Validated in Point 4 and Point 5.
- **Status:** **IMMEDIATE GO**.

---

## 3. Transition Roadmap to Point 6 Execution

With Phase 0 complete and verified, the project proceeds along the following structured execution gates:

```
+-----------------------------------------------------------------------------------+
|                            POINT 6 EXECUTION ROADMAP                              |
+---------------------+-------------------+-------------------+---------------------+
| Phase 1: Setup      | Phase 2: Run      | Phase 3: Telemetry| Phase 4: Visuals    |
+---------------------+-------------------+-------------------+---------------------+
| - Generate YAML     | - Run Sweep 1 (P) | - Parse raw CSVs  | - Generate 6-8 pub  |
|   configs for       | - Run Sweep 2 (H) | - Extract stalls, |   quality figures   |
|   Sweeps 1-5 under  | - Run Sweep 3 (K) |   hits, pollution | - Plot sensitivity  |
|   point_6/configs/  | - Run Sweep 4 (B) | - Build unified   |   heatmaps & lines  |
| - Validate schemas  | - Run Sweep 5 (T) |   Point 6 dataset | - Final report      |
+---------------------+-------------------+-------------------+---------------------+
```

### Execution Gates:
1. **Gate 1 (Zero Ternary Operators):** All configuration generators, analysis scripts, and plotting routines must strictly contain zero ternary operators (`? :` in C++, or `ast.IfExp` in Python).
2. **Gate 2 (Strict Isolation):** All Point 6 configurations, logs, extracted data, and figures must reside under `mohman/point_6/`.
3. **Gate 3 (Scientific Claim Guardrails):** All publications and summary documents must explicitly designate context-length sensitivity as **decode-phase KV-cache and attention scaling (Approach B)**, completely disclaiming any claim to prefill routing evaluation.

