# Point 6 — Experiment Reuse Analysis and Execution Gap Report

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Sensitivity Analysis Consolidation  
**Author:** Antigravity Engineering  
**Date:** November 10, 2026  
**Status:** Authoritative & Completed  

---

## 1. Executive Summary

This report establishes the exact boundary between **reusable historical simulations** and **newly required executions** to complete Point 6.

### Primary Reusability Determination:
1. **Point 4 Big Experiment ($B=32, P=128, T=128$):**
   Runs `w4_c1_r1/r2` (Baseline LRU), `w4_c3_r1/r2` (CSTP + LRU), and `w4_c4_r1/r2` (CSTP + FA-LRU $W=64$) match the reference architecture across all dimensions (DeepSeek-V3, 35 GiB HBM, DDR5 64 GB/s, $H=1, K=4$). They are **reused directly as the $P=128$ condition in Experiment 2 (Context-Length Sensitivity)**, avoiding redundant re-simulation.
2. **Point 4 Big Experiment ($B=14, P=128, T=128$):**
   Runs `w3_c1` through `w3_c4` are preserved as historical references for $B=14$ at $P=128$.
3. **Execution Gaps:**
   To establish a consistent anchor at $P = 2,048$ tokens across batch size, lookahead, and prefetch budget sweeps, **exactly 23 new simulation runs** must be executed.

---

## 2. Inventory of the 23 Newly Required Simulations

The 23 runs are structured across four experiment suites around the common anchor:

```
+-----------------------------------------------------------------------------------------------------------------+
|                               POINT 6 EXECUTION MATRIX (23 NEW RUNS)                                           |
+-------------------+-------------+----+------+-----+---+---+---------+--------+---+------------------------------+
| Experiment Suite  | Run ID      | B  | P    | T   | H | K | Predict | Policy | W | Primary Research Purpose     |
+-------------------+-------------+----+------+-----+---+---+---------+--------+---+------------------------------+
| Common Anchor     | `anc_c1`    | 32 | 2048 | 128 | 0 | 0 | None    | LRU    | 64| Baseline reference           |
| Common Anchor     | `anc_c3`    | 32 | 2048 | 128 | 1 | 4 | CSTP    | LRU    | 64| Prefetch + LRU reference     |
| Common Anchor     | `anc_c4`    | 32 | 2048 | 128 | 1 | 4 | CSTP    | FA-LRU | 64| Complete system reference    |
+-------------------+-------------+----+------+-----+---+---+---------+--------+---+------------------------------+
| Exp 1: Batch Size | `b4_c1`     |  4 | 2048 | 128 | 0 | 0 | None    | LRU    | 64| Low concurrency baseline     |
| Exp 1: Batch Size | `b4_c3`     |  4 | 2048 | 128 | 1 | 4 | CSTP    | LRU    | 64| Low concurrency prefetch     |
| Exp 1: Batch Size | `b4_c4`     |  4 | 2048 | 128 | 1 | 4 | CSTP    | FA-LRU | 64| Low concurrency FA-LRU       |
| Exp 1: Batch Size | `b14_c1`    | 14 | 2048 | 128 | 0 | 0 | None    | LRU    | 64| Medium concurrency baseline  |
| Exp 1: Batch Size | `b14_c3`    | 14 | 2048 | 128 | 1 | 4 | CSTP    | LRU    | 64| Medium concurrency prefetch  |
| Exp 1: Batch Size | `b14_c4`    | 14 | 2048 | 128 | 1 | 4 | CSTP    | FA-LRU | 64| Medium concurrency FA-LRU    |
+-------------------+-------------+----+------+-----+---+---+---------+--------+---+------------------------------+
| Exp 2: Context P  | `p512_c1`   | 32 |  512 | 128 | 0 | 0 | None    | LRU    | 64| Short context baseline       |
| Exp 2: Context P  | `p512_c3`   | 32 |  512 | 128 | 1 | 4 | CSTP    | LRU    | 64| Short context prefetch       |
| Exp 2: Context P  | `p512_c4`   | 32 |  512 | 128 | 1 | 4 | CSTP    | FA-LRU | 64| Short context FA-LRU         |
| Exp 2: Context P  | `p4096_c1`  | 32 | 4096 | 128 | 0 | 0 | None    | LRU    | 64| Long context baseline        |
| Exp 2: Context P  | `p4096_c3`  | 32 | 4096 | 128 | 1 | 4 | CSTP    | LRU    | 64| Long context prefetch        |
| Exp 2: Context P  | `p4096_c4`  | 32 | 4096 | 128 | 1 | 4 | CSTP    | FA-LRU | 64| Long context FA-LRU          |
+-------------------+-------------+----+------+-----+---+---+---------+--------+---+------------------------------+
| Exp 3: Lookahead H| `h2_c3`     | 32 | 2048 | 128 | 2 | 4 | CSTP    | LRU    | 64| Lookahead H=2 prefetch       |
| Exp 3: Lookahead H| `h2_c4`     | 32 | 2048 | 128 | 2 | 4 | CSTP    | FA-LRU | 64| Lookahead H=2 complete sys   |
| Exp 3: Lookahead H| `h4_c3`     | 32 | 2048 | 128 | 4 | 4 | CSTP    | LRU    | 64| Far lookahead H=4 prefetch   |
| Exp 3: Lookahead H| `h4_c4`     | 32 | 2048 | 128 | 4 | 4 | CSTP    | FA-LRU | 64| Far lookahead H=4 complete sys|
+-------------------+-------------+----+------+-----+---+---+---------+--------+---+------------------------------+
| Exp 4: Budget K   | `k2_c3`     | 32 | 2048 | 128 | 1 | 2 | CSTP    | LRU    | 64| Conservative budget K=2      |
| Exp 4: Budget K   | `k2_c4`     | 32 | 2048 | 128 | 1 | 2 | CSTP    | FA-LRU | 64| Conservative budget FA-LRU   |
| Exp 4: Budget K   | `k8_c3`     | 32 | 2048 | 128 | 1 | 8 | CSTP    | LRU    | 64| Aggressive budget K=8        |
| Exp 4: Budget K`  | `k8_c4`     | 32 | 2048 | 128 | 1 | 8 | CSTP    | FA-LRU | 64| Aggressive budget FA-LRU     |
+-------------------+-------------+----+------+-----+---+---+---------+--------+---+------------------------------+
```

---

## 3. Workload Trace Provenance & Generation

All simulations ingest authentic routing traces derived from `LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet`:
- **For $B = 32$ Runs:** Ingest `/home/mohman/ca_project/mohman/point_4/big_exp/traces/w4_b32_t128.csv` (32 sequences, 128 tokens, $1,310,720$ records).
- **For $B = 14$ Runs:** Ingest `/home/mohman/ca_project/mohman/point_4/big_exp/traces/w3_b14_t128.csv` (14 sequences, 128 tokens, $573,440$ records).
- **For $B = 4$ Runs:** Ingest `mohman/point_6/final_results/experiments/batch_size/traces/b4_t128.csv` (sequences 0–3 of `w3_b14_t128.csv`, $163,840$ records).

This strictly guarantees nested sequence evaluation:
$$\text{Trace}(B=4) \subset \text{Trace}(B=14) \subset \text{Trace}(B=32)$$

---

## 4. Execution Strategy & Smoke Test Protocol

1. **Pre-Flight Smoke Test:** Execute a 2-step verification run for each newly introduced dimension ($B=4$, $P=4096$, $H=2$, $K=8$) to confirm zero assertion errors, clean DRAM allocation, and metric logging.
2. **Parallel Dispatch:** Execute simulations with 3 concurrent workers across the 12 CPU cores.
3. **Telemetry Validation:** Extract raw telemetry, parse `prefetch_summary.json` and `fa_lru_diagnostics.json`, and reconcile transfer bytes.

