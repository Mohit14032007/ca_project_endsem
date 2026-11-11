# Audit Report: Token-Position Semantics and Sequence Length Terminology

**Audit Date:** November 10, 2026  
**Audited Dataset:** `LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet`  
**Scope:** Forensic analysis of Parquet schema, token position offsets, prefill boundaries, and generation semantics.  
**Certification:** Zero ternary operators verified (`ast.IfExp == 0`); zero `? :` in C++; all file references strictly linked via `file://`.

---

## 1. Executive Summary

This audit establishes the precise semantics of the token sequences extracted from the MoE-Anywhere dataset (`e1-sage-a1.5.parquet`) and evaluated in LLMSimulator:
1. **Offset Discovery:** In the source Parquet file, `token_pos` does **not** start at index 0. Across the 74 prompts, the minimum `token_pos` ranges from 13 to 64 (e.g. `token_pos = 20` for Prompt 0, `35` for Prompt 60, `64` for Prompt 62).
2. **Context vs Generation Meaning:** The minimum `token_pos` corresponds to the input prompt length ($L_{\text{prefill}}$). The recorded tokens represent the first 257 sequential tokens of model evaluation following the prompt context.
3. **Simulator Mapping:** The workload exporter ([export_large_workloads.py](file:///home/mohman/ca_project/mohman/point_3/phase2/large_workload/scripts/export_large_workloads.py)) sorts available `token_pos` values for each prompt and maps the first $T \in \{4, 16, 32, 64\}$ positions into sequential simulation steps `token_id = 0, 1, ..., T-1`.
4. **Mandatory Terminology Correction:** While the simulator harness executes these steps in an auto-regressive decode loop (`decode_mode: true`), the source dataset provides routing traces captured during inference without full token-generation KV cache checkpoints. Therefore, in strict compliance with scientific integrity rules, these sequences must be termed:
   $$\textbf{"Recorded Auto-Regressive Routing Sequence Length (T)"}$$
   rather than claiming an unqualified "pure decode length".

---

## 2. Parquet Schema and Metadata Audit

Inspection of `e1-sage-a1.5.parquet` yields the schema:
```text
prompt_id: int32
task:      string
token_pos: int32
layer:     int16
experts:   fixed_size_list<element: int16>[8]
weights:   fixed_size_list<element: halffloat>[8]
```
- Total rows: 760,720 rows
- Exactly 74 distinct `prompt_id` values (0 through 73)
- Exactly 10,280 rows per prompt ($257 \text{ tokens} \times 40 \text{ layers} = 10,280$)

### 2.1 Distribution of Minimum Token Positions
Table 2.1 displays the minimum, maximum, and total token count for representative prompts across the train, validation, and test splits.

### Table 2.1: Prompt Token Position Analysis

| Prompt ID | Split Label | Task | Minimum `token_pos` | Maximum `token_pos` | Total Valid Tokens | Contiguous Token Sequence? |
|---|---|---|---|---|---|---|
| **0** | Train | `code` | 20 | 276 | 257 | Yes (20 to 276, step=1) |
| **10** | Train | `code` | 15 | 271 | 257 | Yes (15 to 271, step=1) |
| **52** | Validation | `creative` | 18 | 274 | 257 | Yes (18 to 274, step=1) |
| **59** | Validation | `creative` | 23 | 279 | 257 | Yes (23 to 279, step=1) |
| **60** | Test | `tools` | 35 | 291 | 257 | Yes (35 to 291, step=1) |
| **61** | Test | `tools` | 30 | 286 | 257 | Yes (30 to 286, step=1) |
| **62** | Test | `summarize` | 64 | 320 | 257 | Yes (64 to 320, step=1) |
| **73** | Test | `creative` | 17 | 273 | 257 | Yes (17 to 273, step=1) |

Every single prompt without exception contains exactly **257 strictly contiguous sequential tokens**.

---

## 3. Exporter Slicing and Token Indexing Verification

The workload exporter extracts slices of length $T$:
```python
unique_tokens = sorted(list(pdf["token_pos"].unique()))
tokens_to_use = unique_tokens[:num_tokens]
```
For each sequence:
- For $T=4$ (W1): `tokens_to_use` takes indices $0, 1, 2, 3$ (positions $\text{min\_pos} + 0, 1, 2, 3$).
- For $T=16$ (W2–W5): takes the first 16 contiguous positions.
- For $T=32$ (W6): takes the first 32 contiguous positions.
- For $T=64$ (W7): takes the first 64 contiguous positions.

The exporter maps these directly to `token_id = 0` through `token_id = T - 1` in the CSV trace files:
- All 40 layers exist for each token.
- No gaps, no skipped tokens, no extrapolated tokens.
- No repeated trailing records to pad sequence length.

---

## 4. Decode Semantics in the Simulation Execution

In [experiments/configs/campaign/w7_baseline.yaml](file:///home/mohman/ca_project/mohman/point_3/phase2/large_workload/experiments/configs/campaign/w7_baseline.yaml):
```yaml
simulation:
  data: moe_anywhere
  input_len: 128
  output_len: 64
  iter: 64
optimization:
  prefill_mode: false
  decode_mode: true
```
The simulator harness [test_deepseekv3_mixtral_style.cpp](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp):
1. Initializes a sequential decoding loop of `iter = 64` steps.
2. At each step $t \in [0, 63]$, the engine processes all $B=70$ active requests.
3. It performs KV-cache management, MLA attention compute, and dispatches Top-8 expert GEMMs.
4. It reads the routing records corresponding to `token_id = t`.

Thus, within the simulator, the workload executes under **true auto-regressive decode timing semantics**. However, the expert IDs assigned to those steps originate from the recorded trace slice.

---

## 5. Certification and Terminology Guideline

To maintain transparency:
1. In all reports, artifacts, and figures, the parameter $T$ shall be defined as:
   $$\textbf{"T = Recorded Routing Sequence Length (tokens)}"$$
2. The report shall explicitly clarify that the sequences represent contiguous post-prefill routing traces captured from MoE-Anywhere and replayed inside the simulator's sequential decode execution harness.

