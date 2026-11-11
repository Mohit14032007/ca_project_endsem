# Phase D: Dataset Provenance and Workload Characterization

**Document:** `docs/phase_d/05_DATASET_PROVENANCE.md`  
**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Repository:** `LLMSimulator` (`/home/mohman/ca_project/LLMSimulator`)  
**Commit:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Author:** ML Systems Research & Reproducibility Team  

---

## 1. Workload Suite Overview

Phase D evaluates prediction algorithms across five heterogeneous routing workloads spanning synthetic negative controls, hardware-accurate A100 execution traces, multi-prompt cluster traces, and multi-task MoE benchmarks.

| Dataset Identifier | Domain / Architecture | Layers ($L$) | Experts ($E$) | Top-$K$ | Total Tokens | Split Scheme | Primary $B$ | Primary $H$ |
|---|---|---|---|---|---|---|---|---|
| `deepseek_random` | DeepSeek-V3 (Negative Control) | 57 | 256 | 8 | 5 | Single Request | 8 | 1 |
| `duplex_a100` | Duplex A100 Simulator Trace | 32 | 8 | 2 | 128 | 75% / 25% Requests | 2 | 1 |
| `correlated_synthetic` | Markov Synthetic Correlated | 57 | 256 | 8 | 80 | 75% / 25% Requests | 8 | 1 |
| `moe_anywhere` | Production Inference Trace | 40 | 256 | 8 | Multi-prompt | Disjoint Prompts | 8 | 1, 2 |
| `moe_suite` | Multi-Task MoE Benchmark | 75 | 256 | 8 | Multi-task | Disjoint Tasks | 8 | 2 |

---

## 2. Detailed Workload Provenance

### 2.1. `deepseek_random` (Negative Control)
- **Source File:** `reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/traces/expert_routing.csv`
- **Adapter Class:** `SimulatorDatasetAdapter`
- **Architecture:** 57 routed MoE stages (first 3 stages dense), 256 routed experts per stage, Top-8 routing ($K=8$).
- **Trace Characteristics:** Single-request generation sequence spanning 5 generated tokens.
- **Role in Research:** Functions strictly as a **Negative Control**. Routing decisions are generated pseudo-randomly. In Phase C, single-request train-test leakage caused profiled models to memorize the sequence (Recall ~0.9974). In Phase D, cold-start unprofiled models verify true expected baseline performance:
  $$\mathbb{E}[\text{Recall}] = \frac{B}{E} = \frac{8}{256} = 0.03125$$
  Empirical Phase D cold-start recall matches this theoretical expectation: **0.0307 – 0.0320**.

### 2.2. `duplex_a100` (Hardware Simulator Trace)
- **Source File:** `reproduction_A100/results/a100_duplex/traces/expert_routing.csv`
- **Adapter Class:** `SimulatorDatasetAdapter`
- **Architecture:** 32 MoE stages, 8 routed experts per stage, Top-2 routing ($K=2$).
- **Trace Characteristics:** 32 concurrent requests with 4 generated tokens per request.
- **Split Configuration:** 75% train (requests 2624–2647, 24 requests), 25% test (requests 2648–2655, 8 requests).
- **Role in Research:** Validates behavior on small-expert topologies ($E=8$) under high hardware contention and memory pressure.

### 2.3. `correlated_synthetic` (Markov Correlated Benchmark)
- **Source File:** `results/endsem_phase1/benchmark_correlated_trace.csv`
- **Adapter Class:** `SyntheticCorrelatedDatasetAdapter`
- **Architecture:** 57 MoE stages, 256 routed experts per stage, Top-8 routing ($K=8$).
- **Trace Characteristics:** 16 requests, 5 tokens each (80 tokens total). Generated with controlled spatial Markov affinity ($\alpha=0.6$) and temporal affinity ($\beta=0.5$).
- **Split Configuration:** 12 train requests, 4 test requests.
- **Role in Research:** Controlled ground truth with known spatio-temporal structure to test predictor feature extraction.

### 2.4. `moe_anywhere` (Multi-Tenant Inference Trace)
- **Source File:** `data/downloads/moe_anywhere/e1-sage-a1.5.parquet`
- **Adapter Class:** `MoeAnywhereDatasetAdapter`
- **Architecture:** 40 MoE stages, 256 routed experts per stage, Top-8 routing ($K=8$).
- **Trace Characteristics:** Production multi-tenant inference trace from real-world MoE deployment.
- **Split Configuration:** Grouped by prompt ID; test split comprises disjoint prompt sequences.
- **Role in Research:** Tests predictor adaptability on non-stationary, multi-user routing distributions.

### 2.5. `moe_suite` (Multi-Task MoE Benchmark)
- **Source File:** `data/downloads/moe_suite/`
- **Adapter Class:** `MoeSuiteDatasetAdapter`
- **Architecture:** 75 MoE stages, 256 routed experts per stage, Top-8 routing ($K=8$).
- **Trace Characteristics:** Multi-task routing traces across distinct cognitive tasks (coding, mathematics, creative writing, scientific reasoning).
- **Split Configuration:** Grouped by task ID; test split comprises held-out evaluation tasks.
- **Role in Research:** Reveals negative transfer from offline profiling across disjoint tasks and highlights cold-start vs online adaptation dynamics.

---

## 3. Data Integrity and Verification

All traces have been verified for integrity:
- Zero NaN or null entries in routing records.
- Expert IDs strictly bounded within $[0, E-1]$.
- Stage IDs strictly bounded within $[0, L-1]$.
- Validated causality: all test prediction sequences are causally sorted by `(request_id, token_id, moe_stage_id)`.

