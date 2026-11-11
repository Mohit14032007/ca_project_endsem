# Point 6 — Dataset Context-Length and Trace Availability Audit

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Phase 0: Feasibility & Trace Audit  
**Author:** Antigravity Engineering  
**Date:** November 10, 2026  
**Status:** Authoritative & Completed  

---

## 1. Executive Summary

A core requirement of Point 6 is evaluating system sensitivity across multiple batch sizes ($B$), initial context lengths ($P$), generation decode horizons ($T$), prediction lookaheads ($H$), and prefetch budgets ($K$). Prior to designing or executing simulations, this audit rigorously verifies whether existing trace datasets provide empirical routing traces for:
1. Initial prompt prefill phase (tokens $0 \le t < P$).
2. Contiguous autoregressive decode phase (tokens $P \le t < P + T$).

### Key Findings
1. **MoE-Anywhere Dataset (`e1-sage-a1.5.parquet`):**
   - Contains **760,720 rows** across **74 prompts** and **40 MoE layers** with Top-$K=8$ routing.
   - For every single prompt, exactly **257 contiguous autoregressive decode token steps** are recorded.
   - **Zero prefill routing records exist.** The prompt tokens ($0 \le t < \text{min\_token\_pos}$) were discarded or omitted during the upstream profiling run.
   - The starting position `min_token_pos` ranges from **13 to 64** (mean 24.62, median 22), indicating that the original prompts had lengths between 13 and 64 tokens, but their routing decisions are completely missing.
   - No prompt raw text, tokenized vocabulary IDs, or prefill attention traces are preserved in this parquet file.

2. **MoE-Suite Dataset (`LLMSimulator/data/downloads/moe_suite/*.parquet`):**
   - 12 parquet files spanning diverse code domains (8,191 tokens per file).
   - Evaluates the **GLM-5.2** architecture with **75 layers**, making it architecturally incompatible with our 40-layer DeepSeek-V3 execution engine.
   - Crucially, the explicit boolean column `token_flag_prompt` is `False` for **100.0% of all rows across all 12 files** (`token_flag_generated == True` for 100.0%). Zero prefill tokens are captured.

3. **Definitive Trace Conclusion:**
   - **Approach A (Genuine Prefill Context-Length Sensitivity):** **Unsupported** by existing trace repositories. Neither prompt text nor prefill expert routing records exist in any project dataset.
   - **Approach B (Decode-Only Context-Length Sensitivity via KV-Cache / Attention Scaling):** **Fully Supported**. Existing 257-token decode traces can be replayed while varying initial context length $P$ to modulate KV-cache memory pressure and decode attention duration.
   - **Approach C (Decode Generation Horizon Sensitivity $T$):** **Fully Supported**. Traces support any decode horizon up to $T = 257$ steps across all 74 prompts.

---

## 2. MoE-Anywhere Trace Audit (`e1-sage-a1.5.parquet`)

### 2.1 File Characteristics & Schema
- **File Path:** `LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet`
- **File Size:** 16,166,925 bytes (~15.42 MiB)
- **Total Records:** 760,720 rows
- **Columns & Data Types:**

| Column Name | Arrow Type | Physical Interpretation |
| :--- | :--- | :--- |
| `prompt_id` | `int32` | Unique prompt sequence index ($0 \le \text{prompt\_id} \le 73$) |
| `task` | `string` | Benchmark task classification (10 distinct categories) |
| `token_pos` | `int32` | Absolute token index within the sequence ($13 \le \text{token\_pos} \le 320$) |
| `layer` | `int16` | MoE transformer layer index ($0 \le \text{layer} \le 39$) |
| `experts` | `list<element: int64>` | Array of exactly 8 routed expert IDs ($0 \le \text{expert\_id} \le 255$) |
| `weights` | `list<element: float>` | Array of 8 softmax routing probabilities |

### 2.2 Coverage and Granularity
- **Prompts:** Exactly 74 prompts (IDs 0 through 73).
- **Layers:** Exactly 40 MoE layers (0 through 39).
- **Top-K:** Exactly 8 routed experts per token per layer.
- **Tokens per Prompt:** Exactly 257 contiguous tokens per prompt:
  $$\frac{760,720 \text{ rows}}{74 \text{ prompts} \times 40 \text{ layers}} = 257 \text{ token steps per prompt}$$
- **Contiguity Check:** For every prompt $p \in [0, 73]$:
  $$\text{max\_token\_pos}(p) - \text{min\_token\_pos}(p) + 1 = 257 \equiv \text{nunique}(p)$$
  Every sequence is contiguous with zero gaps and zero dropped tokens.

### 2.3 Prompt Length Distribution
The `token_pos` column records the absolute position of each token in the autoregressive generation sequence. Because generation began at `token_pos = min_token_pos`, the preceding tokens $0 \dots \text{min\_token\_pos} - 1$ correspond to the unrecorded prompt prefill phase.

The distribution of inferred prompt lengths across the 74 prompts is:

| Metric | Inferred Prompt Length ($P_{\text{original}}$) |
| :--- | :--- |
| Minimum | 13 tokens |
| 25th Percentile | 19 tokens |
| Median (50th) | 22 tokens |
| Mean | 24.62 tokens |
| 75th Percentile | 27.75 tokens |
| Maximum | 64 tokens |
| Standard Deviation | 8.61 tokens |

#### Task Breakdown
The 74 prompts represent 10 distinct task domains. Inferred prompt lengths vary across tasks:

| Task Name | Prompt Count | Total Rows | Min $P$ | Max $P$ | Mean $P$ | Median $P$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `chat_en` | 10 | 102,800 | 13 | 21 | 17.90 | 18.5 |
| `chat_es` | 4 | 41,120 | 16 | 20 | 17.50 | 17.0 |
| `chat_pt` | 10 | 102,800 | 16 | 22 | 19.30 | 19.5 |
| `code` | 12 | 123,360 | 17 | 28 | 23.00 | 24.0 |
| `creative` | 4 | 41,120 | 17 | 21 | 18.50 | 18.0 |
| `math` | 10 | 102,800 | 19 | 31 | 24.80 | 25.5 |
| `reasoning` | 8 | 82,240 | 22 | 44 | 32.13 | 31.5 |
| `summarize` | 4 | 41,120 | 20 | 64 | 38.75 | 35.5 |
| `tools` | 8 | 82,240 | 30 | 43 | 34.75 | 35.0 |
| `translate` | 4 | 41,120 | 22 | 24 | 23.00 | 23.0 |
| **Total / Summary** | **74** | **760,720** | **13** | **64** | **24.62** | **22.0** |

### 2.4 Evidence of Missing Prefill Routing
An exhaustive query across all 760,720 rows in `e1-sage-a1.5.parquet` confirms:
1. **Zero records with `token_pos < 13`:** There are no records for tokens $0, 1, \dots, 12$.
2. **Zero prompt text or token IDs:** The parquet file does not store input prompt text, tokenized vocabulary indices, embeddings, or intermediate router logits.
3. **Pure Autoregressive Decode Traces:** The data exclusively captures token-by-token generation steps from step 1 to step 257.

---

## 3. MoE-Suite Trace Audit (`LLMSimulator/data/downloads/moe_suite/*.parquet`)

### 3.1 Overview
The `moe_suite` directory contains 12 parquet files capturing execution traces across code repositories:
- `corpus-02-cpp-framed-rpc-parser.parquet`
- `corpus-03-python-incremental-etl.parquet`
- `corpus-04-rust-priority-worker.parquet`
- `corpus-05-typescript-collab-sync.parquet`
- `corpus-06-go-crawler-limits.parquet`
- `corpus-07-java-outbox-dispatcher.parquet`
- `corpus-08-cpp-mmap-index.parquet`
- `corpus-09-python-import-analyzer.parquet`
- `corpus-10-cuda-segmented-reduction.parquet`
- `corpus-11-cpp-bounded-channel.parquet`
- `corpus-12-typescript-migration-planner.parquet`
- `corpus-13-go-fenced-leases.parquet`

### 3.2 Architectural Mismatch
- **Target Model in LLMSimulator:** DeepSeek-V3 (40 MoE layers, 256 routed experts, Top-8 routing).
- **MoE-Suite Model:** `zai-org/GLM-5.2` with **75 layers** (`route_selected` is typed as `fixed_size_list<element: fixed_size_list<int16>[8]>[75]`).
- Replaying 75-layer traces on a 40-layer DeepSeek-V3 simulator requires ad-hoc layer truncation or projection, which distorts the spatial correlation structure of routing.

### 3.3 Verification of Prefill Columns in MoE-Suite
The MoE-Suite schema includes explicit flags:
- `token_flag_prompt` (bool)
- `token_flag_generated` (bool)
- `context_position` (int32)
- `generated_position` (int32)

Auditing all 12 parquet files (8,191 rows each, total 98,292 rows) reveals:
- `token_flag_prompt`: **0 rows** (0.00%) are `True`.
- `token_flag_generated`: **98,292 rows** (100.00%) are `True`.
- `context_position`: Ranges from 831 up to 9,449. This indicates that the generation occurred in an extended context window, but **only generated tokens were logged**. Zero prompt routing decisions were recorded.

---

## 4. Synthesis and Implications for Context-Length Evaluation

| Dimension | MoE-Anywhere (`e1-sage-a1.5`) | MoE-Suite (`corpus-*.parquet`) |
| :--- | :--- | :--- |
| **Model Compatibility** | DeepSeek-V3 (40 layers, 256 experts) — **100% Match** | GLM-5.2 (75 layers) — **Architecture Mismatch** |
| **Top-K Routing** | Top-8 routed experts | Top-8 routed experts |
| **Prefill Routing Coverage** | 0 tokens (0.0%) | 0 tokens (0.0%) |
| **Decode Routing Coverage** | 257 tokens per prompt (74 prompts) | 8,191 tokens per trace (12 traces) |
| **Contiguity** | 100% contiguous generation | 100% contiguous generation |
| **Prompt Length Range** | Inferred $P \in [13, 64]$ | Inferred $P \in [831, 1259]$ |

### Definitive Conclusion
Neither dataset contains prefill routing data. Consequently:
1. **It is physically impossible to execute a genuine prefill routing sensitivity study (Approach A)** without collecting new routing traces from an active inference harness.
2. **Decode-only context length scaling (Approach B)** is scientifically sound and fully supported: the existing 257-token decode traces can be replayed under varying initial context lengths ($P \in \{128, 512, 2048, 4096\}$), allowing LLMSimulator to scale the KV-cache memory footprint and decode attention duration while preserving authentic expert routing decisions.
3. **Generation horizon scaling (Approach C)** is fully supported for horizons $T \in [1, 257]$.

