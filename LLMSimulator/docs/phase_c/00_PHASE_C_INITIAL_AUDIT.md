# Phase C Initial Repository, Implementation and Dataset Audit

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Phase C — Comprehensive Multi-Dataset Benchmarking with Online Adaptation  
**Date:** 2026-11-01  
**Author:** Antigravity  

---

## 1. Executive Summary

This audit establishes the pre-experiment status of the LLMSimulator repository, the algorithmic readiness of our predictors (CSTP, ST-MoE Paper, ST-MoE Legacy, and baselines), the availability and provenance of all five target datasets, the causal execution guarantees, and the exact commands to execute Phase C.

All 49 unit tests from Phase B and earlier phases pass cleanly (`python3 -m unittest discover tests -v`). The repository contains the necessary baseline implementations, offline profiling frameworks, and latency measurement harnesses. However, several critical adaptations and extensions are required to support multi-dataset execution with online adaptation across multiple prediction horizons ($L \to L+1$ and $L \to L+2$).

---

## 2. Existing Implementation Status

### 2.1 Predictor Implementations

| Component | File | Current Status | Readiness for Phase C |
| :--- | :--- | :--- | :--- |
| **CSTP Predictor** | [`src/predictor/cstp_predictor.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py) | Mathematical formulation verified. Supports `online_adaptation: bool = False` flag. Default mode freezes statistical tensors (`stage_marginal_counts`, `spatial_counts`, `temporal_counts`) while updating per-request temporal `history_table`. | Ready. Needs generalized source-stage indexing for multi-horizon ($L \to L+2$) support. |
| **ST-MoE Paper Predictor** | [`src/predictor/st_moe_paper.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/st_moe_paper.py) | Faithful implementation of Zhao et al. (arXiv:2606.15453v1), Algorithms 1–3, 2-bit saturating counters, replacement policy, and initial confidence 2. | Ready for online mode. Requires an explicit `online_adaptation` toggle to support `st_moe_paper_frozen` (where CCT counter increments/decrements and replacements are suppressed). |
| **ST-MoE Legacy Predictor** | [`src/predictor/st_moe_predictor.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/st_moe_predictor.py) | Legacy implementation retained untouched for historical reference. | Ready as reference baseline. |
| **Common Predictor Interface** | [`src/predictor/base_predictor.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/base_predictor.py) | Defines `BasePredictor` ABC with `reset()`, `predict()`, `update()`, `get_candidate_scores()`. Provides `RandomPredictor`, `FrequencyPredictor`, `CSTPWrapper`, `STMoELegacyWrapper`. | Needs `TemporalOnlyPredictor`, `SpatialOnlyPredictor`, and optional `HybridPredictor`. |
| **Evaluation Harness** | [`src/predictor/evaluation.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/evaluation.py) | Strict split isolation, `copy.deepcopy` per budget, request-level history reset, macro/micro metric aggregation. | Ready. Formats prediction opportunities into standard structures. |
| **Trace Parser** | [`src/predictor/trace_parser.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/trace_parser.py) | Normalizes simulator traces, validates ranges, detects modulo-60 stage index drift. | Ready for simulator traces. |

---

## 3. Required Code Changes

1. **ST-MoE Paper Frozen Mode Support:**
   - In [`src/predictor/st_moe_paper.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/st_moe_paper.py), add an `online_adaptation: bool = True` argument to `STMoEPaperPredictor.__init__()`.
   - In `update()`, when `self.online_adaptation` is `False`, bypass CCT counter increment/decrement and zero-confidence replacement, while continuing to update `self.history_table` for temporal tracking. This isolates the spatial CCT adaptation from temporal sequence tracking.
2. **Generalized Source-Stage Indexing for Prediction Horizons:**
   - For adjacent-stage prediction ($L \to L+1$), source stage is $L$ and target stage is $L+1$.
   - For two-stage-ahead prediction ($L \to L+2$, e.g., in MoE Suite $J \to J+2$), source stage is $L$ and target stage is $L+2$.
   - Predictors must accept `source_stage_id: Optional[int] = None` in both `predict()` and `update()`. If omitted, default to `target_stage_id - 1` for 100% backward compatibility.
3. **Dedicated Baseline Predictors:**
   - Add `TemporalOnlyPredictor` (equivalent to CSTP with $\alpha=0.0$).
   - Add `SpatialOnlyPredictor` (equivalent to CSTP with $\alpha=1.0$).
   - Add `HybridPredictor` (combines CSTP probability scores with ST-MoE discrete confidence counters).
4. **Dataset Adapters (`src/predictor/adapters/`):**
   - Implement `base_adapter.py`: Abstract `RoutingDatasetAdapter` defining `load()`, `validate()`, `get_splits()`, `iter_predictions()`, and `get_metadata()`.
   - Implement `simulator_adapter.py`: Ingests DeepSeek-V3 simulator random trace and Duplex A100 trace.
   - Implement `synthetic_adapter.py`: Ingests and regenerates the correlated synthetic benchmark trace.
   - Implement `moe_suite_adapter.py`: Ingests Hugging Face `mistrjirka/Predicting-Experts-for-MOE-Suite` (`token_cube` configuration, 75 layers, $E=256$, Top-$K=8$, 12 tasks, task-disjoint splits, horizon $J \to J+2$).
   - Implement `moe_anywhere_adapter.py`: Ingests Hugging Face `carlosfelipe/moe-anywhere-routing-traces` (`e1-sage-a1.5.parquet`, 40 layers, $E=256$, Top-$K=8$, 74 prompts, prompt-disjoint splits, horizons $L \to L+1$ and $L \to L+2$).

---

## 4. Existing Bugs, Ambiguities, and Mitigations

1. **Modulo-60 Stage Index Drift in Simulator Traces:**
   - *Issue:* The DeepSeek-V3 C++ simulation trace outputs dense stage IDs 0..2 alongside raw layer indices that wrap modulo-60.
   - *Mitigation:* The trace parser's `reconstruct_stages_from_sequence=True` maps sequential MoE execution events to contiguous stage IDs $0..56$.
2. **Budget Independence:**
   - *Issue:* In some evaluation loops, evaluating budget $B=1$ before $B=8$ could mutate predictor internal state if the same instance is reused.
   - *Mitigation:* `EvaluationHarness` uses `copy.deepcopy(predictor_instance)` at the start of each budget loop, ensuring strict isolation between budgets.
3. **Duplex A100 Expert Universe:**
   - *Issue:* Duplex A100 uses $E=8$ and Top-$K=2$. At budget $B=8$, recall is mathematically guaranteed to be 1.0 (100%), which is a trivial universe coverage effect rather than predictive intelligence.
   - *Mitigation:* Clearly mark $B=8$ on Duplex as trivial full-universe coverage.
4. **ST-MoE Paper Ties in Alg 2:**
   - *Issue:* When multiple experts share identical confidence scores, sorting must be deterministic.
   - *Mitigation:* Enforce primary key by descending score, secondary key by ascending expert ID.
5. **No Ternary Operators:**
   - *Issue:* Explicit project instruction prohibits `ast.IfExp` ternary expressions in Python code.
   - *Mitigation:* All new code uses standard `if/else` blocks and is validated with an automated AST scanner.

---

## 5. Dataset Availability and Provenance

| Dataset | Local / Remote Source | Dimensions | Top-$K$ | Records / Tokens | Availability Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dataset 1: DeepSeek Random** | `LLMSimulator/reproduction_A100/.../expert_routing.csv` | 57 stages, $E=256$ | 8 | 1 request, 5 tokens, 2,280 routing events | Locally available (35 KB). |
| **Dataset 2: Duplex A100** | `LLMSimulator/reproduction_A100/results/a100_duplex/traces/expert_routing.csv` | 32 stages, $E=8$ | 2 | 32 requests, 5 tokens, 10,240 routing events | Locally available (167 KB). |
| **Dataset 3: Correlated Synthetic** | `LLMSimulator/results/endsem_phase1/benchmark_correlated_trace.csv` | 57 stages, $E=256$ | 8 | 16 requests, 5 tokens, 36,480 routing events | Locally available (551 KB), generated via `seed=42`. |
| **Dataset 4: MoE Suite** | `mistrjirka/Predicting-Experts-for-MOE-Suite` (`token_cube`) | 75 stages, $E=256$ | 8 | 12 tasks, 8 train / 2 val / 2 test tasks, ~98K tokens | Remote HF repository verified; sample downloaded (6.9 MB). Streaming/batch ingestion ready. |
| **Dataset 5: MoE Anywhere** | `carlosfelipe/moe-anywhere-routing-traces` (`e1-sage-a1.5.parquet`) | 40 stages, $E=256$ | 8 | 74 prompts, 10 tasks, 19,018 tokens, 760,720 rows | Downloaded locally (16.1 MB). Ingestion verified. |

---

## 6. Split Requirements

1. **DeepSeek-V3 Random:** Single request (5 tokens). Evaluated as negative control for predictable routing.
2. **Duplex A100:** 32 requests. 75% train (24 requests), 25% test (8 requests). Request-disjoint.
3. **Correlated Synthetic:** 16 requests. 75% train (12 requests), 25% test (4 requests). Request-disjoint.
4. **MoE Suite:** 12 tasks. Official task-disjoint splits:
   - Train: `corpus-02` through `corpus-09` (8 tasks).
   - Validation: `corpus-10` and `corpus-11` (2 tasks).
   - Test: `corpus-12` and `corpus-13` (2 tasks).
   - Never randomly shuffle tokens across tasks.
5. **MoE Anywhere:** 74 prompts. Deterministic prompt-disjoint split:
   - Train: 52 prompts (~70%).
   - Validation: 8 prompts (~10%).
   - Test: 14 prompts (~20%).
   - Grouped strictly by `prompt_id`.

---

## 7. Estimated Experiment Sizes

| Dataset | Split | Requests / Prompts | Opportunities ($L \to L+1$) | Opportunities ($L \to L+2$) | Budgets Swept |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepSeek Random** | Test | 1 | 285 | N/A | 7 ($B \in \{4, 8, 16, 24, 32, 48, 64\}$) |
| **Duplex A100** | Test | 8 | 1,280 | N/A | 5 ($B \in \{1, 2, 4, 6, 8\}$) |
| **Correlated Synthetic** | Test | 4 | 1,140 | N/A | 7 ($B \in \{4, 8, 16, 24, 32, 48, 64\}$) |
| **MoE Suite** | Test | 2 tasks | N/A | ~1.19M | 7 ($B \in \{4, 8, 16, 24, 32, 48, 64\}$) |
| **MoE Anywhere** | Test | 14 prompts | ~140K | ~136K | 7 ($B \in \{4, 8, 16, 24, 32, 48, 64\}$) |

---

## 8. Exact Reproduction Commands

```bash
# 1. Run all unit tests
cd /home/mohman/ca_project/LLMSimulator && python3 -m unittest discover tests -v

# 2. Validate all datasets and generate quality report
python3 scripts/phase_c/validate_datasets.py

# 3. Run small-scale correctness verification
python3 scripts/phase_c/run_online_vs_frozen.py --small_scale

# 4. Run Phase C full benchmark
python3 scripts/phase_c/run_phase_c.py

# 5. Run distribution-shift experiment
python3 scripts/phase_c/run_distribution_shift.py

# 6. Benchmark predictor latency and memory
python3 scripts/phase_c/benchmark_predictors.py

# 7. Aggregate results and generate all 18 figures
python3 scripts/phase_c/aggregate_results.py
python3 scripts/phase_c/generate_figures.py

# 8. Run automated Phase C validation checks
python3 -m unittest discover tests/phase_c -v
```

---

## 9. Algorithmic and Causal Audit Verdict

**Verdict:** PROCEED TO PHASE C IMPLEMENTATION.  
No fundamental blocker exists. The existing codebase guarantees strict causal separation between prediction and update, and split isolation is fully supported. Implementation of the adapters and the online/frozen toggles will proceed according to the Phase C execution order.

