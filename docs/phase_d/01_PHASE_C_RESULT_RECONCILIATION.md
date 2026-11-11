# Phase C Result Reconciliation & Data Accounting Report

**Author:** Antigravity Senior ML Systems Researcher & Reproducibility Auditor  
**Date:** November 2026  
**Document ID:** `docs/phase_d/01_PHASE_C_RESULT_RECONCILIATION.md`  
**Reference Data:** `results/endsem_phase_d/audit/phase_c_reconciliation.csv`  

---

## 1. Overview

This document provides complete, byte-level data accounting and event reconciliation for all 5 Phase C datasets. We independently verify the relationship between raw trace rows, routing events, prediction events, split partitions, and evaluated prediction records.

---

## 2. Comprehensive Reconciliation Table

| Metric / Dimension | `deepseek_random` | `duplex_a100` | `correlated_synthetic` | `moe_anywhere` | `moe_suite` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Raw Trace File** | `reproduction_A100/.../expert_routing.csv` | `reproduction_A100/results/a100_duplex/traces/expert_routing.csv` | `results/endsem_phase1/benchmark_correlated_trace.csv` | `data/downloads/moe_anywhere/e1-sage-a1.5.parquet` | `data/downloads/moe_suite/*.parquet` (12 files) |
| **File Format** | CSV | CSV | CSV | Parquet | Parquet |
| **Raw Rows** | 2,280 | 10,240 | 36,480 | 760,720 | 98,292 tokens |
| **Experts ($E$)** | 256 | 8 | 256 | 256 | 256 |
| **Top-$K$** | 8 | 2 | 8 | 8 | 8 |
| **MoE Stages ($S$)**| 57 (dense 0–2 omitted) | 32 | 57 | 40 | 75 |
| **Total Routing Events** | 285 | 5,120 | 4,560 | 760,720 | 7,371,900 |
| **Unit Partition Type** | Request ID (Fallback) | Request ID | Request ID | Prompt ID | Task Corpus ID |
| **Total Units** | 1 request | 32 requests | 16 requests | 74 prompts | 12 tasks |
| **Train Units** | 1 (Leakage) | 24 requests | 12 requests | 44 prompts | 8 tasks |
| **Validation Units**| 0 | 0 | 0 | 15 prompts | 2 tasks (`corpus-10`, `11`) |
| **Test Units** | 1 request | 8 requests | 4 requests | 15 prompts (14 evaluated) | 2 tasks (`corpus-12`, `13`) |
| **Evaluated Test Tokens** | 5 | 5 | 5 | 20 | 50 |
| **Prediction Events / Eval** | 285 | 1,280 | 1,140 | 11,200 ($H=1$), 11,200 ($H=2$) | 7,500 ($H=2$) |
| **Evaluated Horizons** | $H=1$ | $H=1$ | $H=1$ | $H \in \{1, 2\}$ | $H=2$ |
| **Evaluated Budgets** | 4, 8, 16, 24, 32, 48, 64 | 1, 2, 4, 6, 8 | 4, 8, 16, 24, 32, 48, 64 | 4, 8, 16, 24, 32, 48, 64 | 4, 8, 16, 24, 32, 48, 64 |
| **Evaluated Predictors** | 12 | 12 | 12 | 12 | 12 |
| **Total Prediction Records** | **23,940** | **76,800** | **95,760** | **1,881,600** | **630,000** |

---

## 3. Workload Accounting Notes

### A. DeepSeek Random Trace
- **Raw Row Derivation**: 1 request $\times$ 5 tokens $\times$ 57 MoE stages $\times$ 8 experts = 2,280 rows.
- **Stage Exclusion**: The original simulator trace contains 60 layers (layers 0–59). In DeepSeek-V3 architecture, the first 3 layers (0, 1, 2) are dense feedforward layers; MoE routing begins at layer 3 (stage 0) through layer 59 (stage 56). The adapter reconstructed 57 MoE stages.
- **Split Leakage**: Because `num_requests == 1`, both train and test splits pointed to request 1. Evaluated test records (285 events $\times$ 84 configurations = 23,940) represent 100% memorization for profiled models.

### B. Duplex A100 Trace
- **Raw Row Derivation**: 32 requests $\times$ 5 tokens $\times$ 32 layers $\times$ 2 experts = 10,240 rows.
- **Split**: 24 train requests (75%), 8 test requests (25%).
- **Prediction Events**: 8 test requests $\times$ 5 tokens $\times$ 32 stages = 1,280 events per evaluation.
- Across 12 predictors $\times$ 5 budgets: $1,280 \times 60 =$ 76,800 prediction records.

### C. Correlated Synthetic Benchmark
- **Raw Row Derivation**: 16 requests $\times$ 5 tokens $\times$ 57 stages $\times$ 8 experts = 36,480 rows.
- **Split**: 12 train requests (75%), 4 test requests (25%).
- **Prediction Events**: 4 test requests $\times$ 5 tokens $\times$ 57 stages = 1,140 events per evaluation.
- Across 12 predictors $\times$ 7 budgets: $1,140 \times 84 =$ 95,760 prediction records.

### D. MoE Anywhere
- **Raw Format**: Parquet table with 760,720 rows. Each row represents `(prompt_id, token_id, layer)` and stores `selected_experts` as an array of 8 integers. Total expert activations = $760,720 \times 8 = 6,085,760$.
- **Split**: 44 train prompts, 15 validation prompts, 15 test prompts.
- **Evaluation Subset**: To maintain practical benchmarking runtime, the test evaluation was bounded to 14 prompts and 20 tokens per prompt:
  $$14 \text{ prompts} \times 20 \text{ tokens} \times 40 \text{ layers} = 11,200 \text{ prediction events per horizon}$$
- Two horizons evaluated ($H=1$ and $H=2$): $11,200 \times 2 = 22,400$ events per predictor-budget.
- Across 12 predictors $\times$ 7 budgets: $22,400 \times 84 =$ 1,881,600 prediction records.

### E. MoE Suite
- **Raw Format**: 12 Parquet files from `token_cube` track. Each file contains 8,191 token rows for 75 layers. Total tokens = $8,191 \times 12 = 98,292$. Total routing events = $98,292 \times 75 = 7,371,900$.
- **Split**: Official benchmark disjoint split: 8 train tasks, 2 validation tasks (`corpus-10`, `corpus-11`), 2 test tasks (`corpus-12-typescript-migration-planner`, `corpus-13-go-fenced-leases`).
- **Evaluation Subset**: 2 test tasks $\times$ 50 tokens $\times$ 75 layers = 7,500 prediction events per evaluation at horizon $H=2$.
- Across 12 predictors $\times$ 7 budgets: $7,500 \times 84 =$ 630,000 prediction records.

---

## 4. Total Phase C Evaluation Scope

$$\sum_{\text{datasets}} \text{Prediction Records} = 23,940 + 76,800 + 95,760 + 1,881,600 + 630,000 = \mathbf{2,708,100}$$

Every single prediction record was logged with its exact request ID, token ID, source stage, target stage, budget, predicted set, ground truth set, hits, precision, recall, and hit rate, guaranteeing 100% reproducible auditing.

