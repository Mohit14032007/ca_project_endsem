# Point 6 — Authentic Initial Context Length Extraction and Validation Audit

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Experiment A: Inferred Initial Context Length Audit  
**Author:** Antigravity Engineering  
**Date:** November 10, 2026  
**Status:** Authoritative & Completed  

---

## 1. Executive Summary

This report delivers the complete extraction methodology, validation protocol, and audit results for the inferred initial context lengths ($P$) extracted from the authentic MoE-Anywhere trace dataset:

`LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet`

The dataset records expert routing decisions for DeepSeek-V3 across **760,720 rows**, covering **74 prompt sequences** (IDs 0–73), **40 MoE layers** (0–39), and **Top-8 expert routing** ($k=8$).

Every prompt sequence was subjected to five rigorous validation gates to determine whether `min_token_pos` provides an unbroken, trustworthy inference of the prompt length preceding autoregressive generation.

### Key Audit Result
**100% of all 74 prompts passed all five validation gates.**
- Zero dropped tokens.
- Zero incomplete layers.
- Zero out-of-range expert IDs.
- Zero duplicate routing records.
- All 74 traces are contiguous and valid for generation horizons up to $T = 257$ steps.

---

## 2. Validation Gates & Audit Criteria

To ensure that inferred prompt lengths are not artifacts of corrupted, truncated, or interleaved data, each prompt was evaluated against five strict validation gates:

```
+---------------------------------------------------------------------------------------+
|                               FIVE-GATE VALIDATION PROTOCOL                           |
+-------------------+----------------------------------------------------+--------------+
| Gate Identifier   | Required Invariant                                 | Result       |
+-------------------+----------------------------------------------------+--------------+
| Gate 1: Sequence  | Exactly 257 unique positions; max - min + 1 == 257 | 74/74 PASSED |
| Gate 2: Layers    | Exactly 40 unique layers; 40 rows per token pos    | 74/74 PASSED |
| Gate 3: Routing   | Exactly 8 expert IDs per row; all IDs in [0, 255]  | 74/74 PASSED |
| Gate 4: Alignment | Min token_pos identical across all 40 layers       | 74/74 PASSED |
| Gate 5: Integrity | Row count == 10,280 per prompt (760,720 total)     | 74/74 PASSED |
+-------------------+----------------------------------------------------+--------------+
```

### Detailed Gate Definitions:
1. **Gate 1 — Sequence Contiguity & Horizon ($T = 257$):**
   $$\text{unique\_token\_positions} = 257 \quad \land \quad \max(\text{token\_pos}) - \min(\text{token\_pos}) + 1 = 257$$
   Verifies that no token steps are missing or skipped between the starting position and the final generation step.
2. **Gate 2 — Complete Layer Coverage ($L = 40$):**
   $$\text{unique\_layers} = 40 \quad \land \quad \forall t \in [\min, \max], \, \text{count}(\text{token\_pos} = t) = 40$$
   Verifies that every recorded token step contains routing records for all 40 MoE layers.
3. **Gate 3 — Top-8 Routing Integrity:**
   $$\forall \text{ record}, \, \text{len}(\text{experts}) = 8 \quad \land \quad \forall e \in \text{experts}, \, 0 \le e \le 255$$
   Verifies that exactly 8 experts are selected per layer and all expert indices fall within the valid DeepSeek-V3 routed expert range.
4. **Gate 4 — Multi-Layer Starting Alignment:**
   $$\forall l \in [0, 39], \, \min(\text{token\_pos} \mid \text{layer} = l) = \min(\text{token\_pos})$$
   Verifies that the prompt length boundary is synchronous across all transformer layers.
5. **Gate 5 — Record Integrity & Completeness:**
   $$\text{row\_count} = 257 \times 40 = 10,280 \text{ rows per prompt}$$
   Verifies that there are no duplicate records, null entries, or orphaned token steps.

---

## 3. Complete Per-Prompt Extraction and Validation Table

The table below presents the verified extraction metrics for all 74 prompts in `e1-sage-a1.5.parquet`, sorted by `inferred_initial_P` ascending, then `prompt_id`.

| Prompt ID | Task Category | Min `token_pos` | Max `token_pos` | Inferred Initial $P$ | Unique Token Steps | Row Count | Unique Layers | Contiguous Trace | Layer Coverage | Top-K Valid | Trace Valid |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **18** | `chat_en` | 13 | 269 | **13** | 257 | 10,280 | 40 | True | True | True | **True** |
| **20** | `chat_en` | 16 | 272 | **16** | 257 | 10,280 | 40 | True | True | True | **True** |
| **25** | `chat_pt` | 16 | 272 | **16** | 257 | 10,280 | 40 | True | True | True | **True** |
| **34** | `chat_es` | 16 | 272 | **16** | 257 | 10,280 | 40 | True | True | True | **True** |
| **35** | `chat_es` | 16 | 272 | **16** | 257 | 10,280 | 40 | True | True | True | **True** |
| **7** | `code` | 17 | 273 | **17** | 257 | 10,280 | 40 | True | True | True | **True** |
| **12** | `chat_en` | 17 | 273 | **17** | 257 | 10,280 | 40 | True | True | True | **True** |
| **16** | `chat_en` | 17 | 273 | **17** | 257 | 10,280 | 40 | True | True | True | **True** |
| **24** | `chat_pt` | 17 | 273 | **17** | 257 | 10,280 | 40 | True | True | True | **True** |
| **73** | `creative` | 17 | 273 | **17** | 257 | 10,280 | 40 | True | True | True | **True** |
| **21** | `chat_en` | 18 | 274 | **18** | 257 | 10,280 | 40 | True | True | True | **True** |
| **27** | `chat_pt` | 18 | 274 | **18** | 257 | 10,280 | 40 | True | True | True | **True** |
| **30** | `chat_pt` | 18 | 274 | **18** | 257 | 10,280 | 40 | True | True | True | **True** |
| **32** | `chat_es` | 18 | 274 | **18** | 257 | 10,280 | 40 | True | True | True | **True** |
| **70** | `creative` | 18 | 274 | **18** | 257 | 10,280 | 40 | True | True | True | **True** |
| **71** | `creative` | 18 | 274 | **18** | 257 | 10,280 | 40 | True | True | True | **True** |
| **6** | `code` | 19 | 275 | **19** | 257 | 10,280 | 40 | True | True | True | **True** |
| **14** | `chat_en` | 19 | 275 | **19** | 257 | 10,280 | 40 | True | True | True | **True** |
| **17** | `chat_en` | 19 | 275 | **19** | 257 | 10,280 | 40 | True | True | True | **True** |
| **19** | `chat_en` | 19 | 275 | **19** | 257 | 10,280 | 40 | True | True | True | **True** |
| **23** | `chat_pt` | 19 | 275 | **19** | 257 | 10,280 | 40 | True | True | True | **True** |
| **37** | `math` | 19 | 275 | **19** | 257 | 10,280 | 40 | True | True | True | **True** |
| **0** | `code` | 20 | 276 | **20** | 257 | 10,280 | 40 | True | True | True | **True** |
| **2** | `code` | 20 | 276 | **20** | 257 | 10,280 | 40 | True | True | True | **True** |
| **10** | `code` | 20 | 276 | **20** | 257 | 10,280 | 40 | True | True | True | **True** |
| **15** | `chat_en` | 20 | 276 | **20** | 257 | 10,280 | 40 | True | True | True | **True** |
| **33** | `chat_es` | 20 | 276 | **20** | 257 | 10,280 | 40 | True | True | True | **True** |
| **44** | `math` | 20 | 276 | **20** | 257 | 10,280 | 40 | True | True | True | **True** |
| **45** | `math` | 20 | 276 | **20** | 257 | 10,280 | 40 | True | True | True | **True** |
| **60** | `summarize` | 20 | 276 | **20** | 257 | 10,280 | 40 | True | True | True | **True** |
| **13** | `chat_en` | 21 | 277 | **21** | 257 | 10,280 | 40 | True | True | True | **True** |
| **26** | `chat_pt` | 21 | 277 | **21** | 257 | 10,280 | 40 | True | True | True | **True** |
| **28** | `chat_pt` | 21 | 277 | **21** | 257 | 10,280 | 40 | True | True | True | **True** |
| **29** | `chat_pt` | 21 | 277 | **21** | 257 | 10,280 | 40 | True | True | True | **True** |
| **31** | `chat_pt` | 21 | 277 | **21** | 257 | 10,280 | 40 | True | True | True | **True** |
| **72** | `creative` | 21 | 277 | **21** | 257 | 10,280 | 40 | True | True | True | **True** |
| **22** | `chat_pt` | 22 | 278 | **22** | 257 | 10,280 | 40 | True | True | True | **True** |
| **41** | `math` | 22 | 278 | **22** | 257 | 10,280 | 40 | True | True | True | **True** |
| **46** | `reasoning` | 22 | 278 | **22** | 257 | 10,280 | 40 | True | True | True | **True** |
| **67** | `translate` | 22 | 278 | **22** | 257 | 10,280 | 40 | True | True | True | **True** |
| **69** | `translate` | 22 | 278 | **22** | 257 | 10,280 | 40 | True | True | True | **True** |
| **66** | `translate` | 23 | 279 | **23** | 257 | 10,280 | 40 | True | True | True | **True** |
| **1** | `code` | 24 | 280 | **24** | 257 | 10,280 | 40 | True | True | True | **True** |
| **9** | `code` | 24 | 280 | **24** | 257 | 10,280 | 40 | True | True | True | **True** |
| **40** | `math` | 24 | 280 | **24** | 257 | 10,280 | 40 | True | True | True | **True** |
| **68** | `translate` | 24 | 280 | **24** | 257 | 10,280 | 40 | True | True | True | **True** |
| **4** | `code` | 25 | 281 | **25** | 257 | 10,280 | 40 | True | True | True | **True** |
| **11** | `code` | 25 | 281 | **25** | 257 | 10,280 | 40 | True | True | True | **True** |
| **42** | `math` | 25 | 281 | **25** | 257 | 10,280 | 40 | True | True | True | **True** |
| **5** | `code` | 26 | 282 | **26** | 257 | 10,280 | 40 | True | True | True | **True** |
| **38** | `math` | 26 | 282 | **26** | 257 | 10,280 | 40 | True | True | True | **True** |
| **39** | `math` | 26 | 282 | **26** | 257 | 10,280 | 40 | True | True | True | **True** |
| **47** | `reasoning` | 26 | 282 | **26** | 257 | 10,280 | 40 | True | True | True | **True** |
| **43** | `math` | 27 | 283 | **27** | 257 | 10,280 | 40 | True | True | True | **True** |
| **48** | `reasoning` | 27 | 283 | **27** | 257 | 10,280 | 40 | True | True | True | **True** |
| **3** | `code` | 28 | 284 | **28** | 257 | 10,280 | 40 | True | True | True | **True** |
| **8** | `code` | 28 | 284 | **28** | 257 | 10,280 | 40 | True | True | True | **True** |
| **63** | `summarize` | 28 | 284 | **28** | 257 | 10,280 | 40 | True | True | True | **True** |
| **36** | `math` | 30 | 286 | **30** | 257 | 10,280 | 40 | True | True | True | **True** |
| **56** | `tools` | 30 | 286 | **30** | 257 | 10,280 | 40 | True | True | True | **True** |
| **49** | `reasoning` | 31 | 287 | **31** | 257 | 10,280 | 40 | True | True | True | **True** |
| **58** | `tools` | 32 | 288 | **32** | 257 | 10,280 | 40 | True | True | True | **True** |
| **55** | `tools` | 34 | 290 | **34** | 257 | 10,280 | 40 | True | True | True | **True** |
| **50** | `reasoning` | 35 | 291 | **35** | 257 | 10,280 | 40 | True | True | True | **True** |
| **52** | `tools` | 35 | 291 | **35** | 257 | 10,280 | 40 | True | True | True | **True** |
| **54** | `tools` | 35 | 291 | **35** | 257 | 10,280 | 40 | True | True | True | **True** |
| **57** | `tools` | 35 | 291 | **35** | 257 | 10,280 | 40 | True | True | True | **True** |
| **59** | `tools` | 35 | 291 | **35** | 257 | 10,280 | 40 | True | True | True | **True** |
| **51** | `reasoning` | 39 | 295 | **39** | 257 | 10,280 | 40 | True | True | True | **True** |
| **53** | `tools` | 40 | 296 | **40** | 257 | 10,280 | 40 | True | True | True | **True** |
| **61** | `summarize` | 43 | 299 | **43** | 257 | 10,280 | 40 | True | True | True | **True** |
| **64** | `reasoning` | 44 | 300 | **44** | 257 | 10,280 | 40 | True | True | True | **True** |
| **65** | `reasoning` | 45 | 301 | **45** | 257 | 10,280 | 40 | True | True | True | **True** |
| **62** | `summarize` | 64 | 320 | **64** | 257 | 10,280 | 40 | True | True | True | **True** |

*Note: Table is sorted ascending by `inferred_initial_P`, then `prompt_id`.*

---

## 4. Methodological Guardrails and Scientific Boundaries

1. **Inference Basis:**
   The parameter `inferred_initial_P` is defined strictly as $\min(\text{token\_pos})$. It reflects the sequence offset at which the autoregressive trace logger commenced logging.
2. **Absence of Tokenized Prompt Text:**
   Neither the original prompt strings, BPE tokens, embeddings, nor prefill attention matrices are present in `e1-sage-a1.5.parquet`. It is physically impossible to verify whether the original text had exactly $P$ tokens without external prompt files.
3. **No Fabrication of Extended Traces:**
   All 74 prompts contain exactly 257 generation tokens. No traces have been lengthened, shortened, interpolated, or extrapolated.
4. **Decoupled Simulation Support:**
   In `LLMSimulator`, setting `simulation.input_len: P` scales KV-cache memory and attention duration without altering trace replay. The trace index is given by $\text{rel\_token\_id} = \text{token\_id} - \text{input\_len} = t$. Thus, empirical routing fidelity is preserved across all context length evaluations.

---

## 5. Artifact Verification & Deliverables Manifest

| Artifact File | Path | Size | Description |
| :--- | :--- | :---: | :--- |
| **CSV Dataset** | [`PROMPT_INITIAL_CONTEXT_LENGTHS.csv`](file:///home/mohman/ca_project/mohman/point_6/experiment_a/PROMPT_INITIAL_CONTEXT_LENGTHS.csv) | 7,654 B | Complete 74-prompt verified extraction dataset with all validation flags |
| **Distribution Report** | [`INITIAL_CONTEXT_LENGTH_DISTRIBUTION.md`](file:///home/mohman/ca_project/mohman/point_6/experiment_a/INITIAL_CONTEXT_LENGTH_DISTRIBUTION.md) | 11,480 B | Statistical analysis, KV memory formulas, task breakdown, and Go/No-Go decision |
| **Histogram Plot** | [`INITIAL_CONTEXT_LENGTH_HISTOGRAM.png`](file:///home/mohman/ca_project/mohman/point_6/experiment_a/INITIAL_CONTEXT_LENGTH_HISTOGRAM.png) | 266,742 B | Publication-quality 300 DPI visualization of prompt distribution and task breakdown |
| **Validation Report** | [`INITIAL_CONTEXT_LENGTH_EXTRACTION_VALIDATION.md`](file:///home/mohman/ca_project/mohman/point_6/experiment_a/INITIAL_CONTEXT_LENGTH_EXTRACTION_VALIDATION.md) | ~14,000 B | This authoritative report containing the 5-gate audit and complete 74-prompt table |

