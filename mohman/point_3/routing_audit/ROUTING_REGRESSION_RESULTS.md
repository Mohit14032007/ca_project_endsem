# Routing Pipeline Regression and Validation Results

**Audit & Validation Directory:** `/home/mohman/ca_project/mohman/point_3/routing_audit`  
**Repository:** [`/home/mohman/ca_project/LLMSimulator`](file:///home/mohman/ca_project/LLMSimulator)  
**Execution Timestamp:** 2026-11-10  
**Status:** ALL TESTS AND SIMULATIONS PASSED (10/10 Unit Tests, 33/33 Prior Regression Tests, 12/12 Validation Runs)

---

## 1. Unit Test Suite: `test_routing_ground_truth` (10 / 10 Passed)

The dedicated unit test suite in [`eval/test_routing_ground_truth.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test_routing_ground_truth.cpp) exercises every component of the routing and prefetch pipeline in strict isolation.

| Test Case | Description | Verification Criterion | Status |
|---|---|---|---|
| **Test 1** | Synthetic Routing Source | Invocations populate `cached_routing` with 4-tuple key; exact Top-K generated within valid bounds $[0, N)$. | **PASS** |
| **Test 2** | Real-Trace Routing Source | `loadRealTrace()` consumes exact trace records; `SequenceInfo` holds exact expert IDs. | **PASS** |
| **Test 3** | No Random Fallback | Real-trace mode asserts 0 random fallback across all tokens; dummy sequences forbidden. | **PASS** |
| **Test 4** | Mapping Isolation | Multi-sequence batch maintains independent routing; distinct requests do not collide. | **PASS** |
| **Test 5** | Scheduler-Dispatch Consistency | Top-K routing decisions identically match dispatch vectors in `expert_routing.csv`. | **PASS** |
| **Test 6** | GEMM Aggregation Consistency | `num_token_in_expert` sum across all experts equals $B \times K$ ($2 \times 8 = 16$). | **PASS** |
| **Test 7** | Predictor Isolation | Predictor receives only past/current context; zero access to future ground-truth trace. | **PASS** |
| **Test 8** | Observational Scoring | Prediction engine evaluates hits without mutating candidate queue or simulator state. | **PASS** |
| **Test 9** | Prefetch Matching Semantics | Exact 5-tuple matching `(req, seq, tok, layer, eid)` distinguishes timely hits from late attachments. | **PASS** |
| **Test 10** | Regression Independence | Real-trace execution and synthetic generator can run consecutively without cross-contamination. | **PASS** |

**Unit Suite Summary:** 10 Passed, 0 Failed, 0 Flaky.

---

## 2. Compatibility with Prior Regression Test Suites (33 / 33 Passed)

To ensure that the forensic fixes in `sequence.cpp`, `linear.cpp`, and test harnesses introduced zero side-effects into existing instrumentation, all prior test suites were compiled and executed:

1. **[`test_phase4b_effectiveness_metrics`](file:///home/mohman/ca_project/LLMSimulator/eval/test_phase4b_effectiveness_metrics.cpp) (Cases A through J):**
   - 10 passed, 0 failed.
   - Verified: Timely hits, late in-flight attachments, unused prefetches, admission rejections, cancellations, multi-projection deduplication, tail latency distributions.

2. **[`test_prefetch_metrics`](file:///home/mohman/ca_project/LLMSimulator/eval/test_prefetch_metrics.cpp) (Scenarios 1 through 15):**
   - 15 passed, 0 failed.
   - Verified: Centralized `PrefetchMetricsCollector` invariants, equivalence between enabled and disabled instrumentation, schema compliance.

3. **[`test_synthetic_routing_diversity`](file:///home/mohman/ca_project/LLMSimulator/eval/test_synthetic_routing_diversity.cpp) (Tests A through H):**
   - 8 passed, 0 failed.
   - Verified: Four-component synthetic cache key `(seq_id, expert_seq_id, token_pos, layer_id)`, independent sequence routing across batch sizes $B \in \{1, 2, 4, 8, 16, 32\}$, mean pairwise Jaccard similarity bounds.

---

## 3. Comprehensive Validation Experiment Matrix (12 Runs)

Twelve controlled simulation runs were conducted spanning 2-stage and 3-stage topologies across MoE-Anywhere, MoE-Suite Task 12, and Synthetic DeepSeek workloads. All runs used event-driven DDR5 DRAM modeling at 64.0 GB/s bandwidth with Ramulator enabled.

Summary data from [`CORE_SMOKE_VALIDATION.csv`](file:///home/mohman/ca_project/mohman/point_3/routing_audit/CORE_SMOKE_VALIDATION.csv):

| Experiment Name | Topology | Dataset | Predictor | Horizon | Total Demands | Demand Misses | HBM Hit Rate | Dispatched Prefetches | Unused Prefetches | Speculative Bytes | Total DDR5 Bytes | Exposed Stall (ms) | Mean Token Latency (ms) | Routing Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `smoke_2stg_baseline_anywhere` | 2-stage | MoE-Anywhere | Baseline | - | 192 | 42 | 78.00% | 0 | 0 | 0 B | 3.70 GB | 57.80 | 16.58 | 100% MATCH |
| `smoke_2stg_stmoe_h1_anywhere` | 2-stage | MoE-Anywhere | ST-MoE | 1 | 192 | 42 | 78.00% | 4 | 4 | 352.32 MB | 4.05 GB | 57.80 | 16.58 | 100% MATCH |
| `smoke_2stg_cstp_h1_anywhere` | 2-stage | MoE-Anywhere | CSTP | 1 | 192 | 42 | 78.00% | 4 | 4 | 352.32 MB | 4.05 GB | 57.80 | 16.58 | 100% MATCH |
| `smoke_2stg_baseline_suite12` | 2-stage | MoE-Suite 12 | Baseline | - | 192 | 60 | 69.00% | 0 | 0 | 0 B | 5.28 GB | 82.58 | 22.72 | 100% MATCH |
| `smoke_2stg_stmoe_h1_suite12` | 2-stage | MoE-Suite 12 | ST-MoE | 1 | 192 | 60 | 69.00% | 4 | 4 | 352.32 MB | 5.64 GB | 82.58 | 22.72 | 100% MATCH |
| `smoke_2stg_cstp_h1_suite12` | 2-stage | MoE-Suite 12 | CSTP | 1 | 192 | 60 | 69.00% | 4 | 4 | 352.32 MB | 5.64 GB | 82.58 | 22.72 | 100% MATCH |
| `smoke_2stg_synthetic` | 2-stage | Synthetic | Baseline | - | 192 | 61 | 68.00% | 0 | 0 | 0 B | 5.37 GB | 83.95 | 23.06 | 100% MATCH |
| `smoke_3stg_baseline_anywhere` | 3-stage | MoE-Anywhere | Baseline | - | 288 | 59 | 80.00% | 0 | 0 | 0 B | 5.20 GB | 81.20 | 22.83 | 100% MATCH |
| `smoke_3stg_stmoe_h1_anywhere` | 3-stage | MoE-Anywhere | ST-MoE | 1 | 288 | 59 | 80.00% | 8 | 6 | 704.64 MB | 5.90 GB | 81.20 | 22.83 | 100% MATCH |
| `smoke_3stg_stmoe_h2_anywhere` | 3-stage | MoE-Anywhere | ST-MoE | 2 | 288 | 59 | 80.00% | 4 | 4 | 352.32 MB | 5.55 GB | 81.20 | 22.83 | 100% MATCH |
| `smoke_3stg_cstp_h1_anywhere` | 3-stage | MoE-Anywhere | CSTP | 1 | 288 | 59 | 80.00% | 8 | 6 | 704.64 MB | 5.90 GB | 81.20 | 22.83 | 100% MATCH |
| `smoke_3stg_cstp_h2_anywhere` | 3-stage | MoE-Anywhere | CSTP | 2 | 288 | 59 | 80.00% | 4 | 4 | 352.32 MB | 5.55 GB | 81.20 | 22.83 | 100% MATCH |

---

## 4. In-Depth Forensic Analysis of the Results

### 4.1 Verification of Ground Truth Ingestion
Across all 12 validation runs:
1. `moe_anywhere` runs consumed the exact sequence of expert IDs exported from `e1-sage-a1.5.parquet`.
2. `moe_suite_corpus12` runs consumed the exact expert IDs from `corpus-12-typescript-migration-planner.parquet`.
3. `synthetic` runs generated reproducible, sequence-isolated Top-K sets using the PRNG generator.
4. In every run, Step 3 (Token 131) executed its intended sequence without premature termination, producing **0 random fallbacks** and a **100% exact match** rate against ground truth trace records.

### 4.2 Why Horizon $H=2$ Succeeded in 3-Stage Topology
In the original 5-layer model (2 MoE stages), lookahead with $H=2$ attempted to predict stage index $0 + 2 = 2 \ge N_{\text{stages}}$, which was immediately dropped by the engine.

In the 3-stage topology (`num_layers: 6`, 3 dense prefix + 3 MoE stages: Stage 0, Stage 1, Stage 2):
- Lookahead with $H=2$ evaluates `target_stage = 0 + 2 = 2`.
- The engine admits candidates for Stage 2.
- In both ST-MoE and CSTP runs, exactly 4 transfers were admitted and dispatched for Stage 2.
- The diagnostic logs ([`diagnostic_smoke_3stg_cstp_h2_anywhere.md`](file:///home/mohman/ca_project/mohman/point_3/routing_audit/routing_diagnostics/diagnostic_smoke_3stg_cstp_h2_anywhere.md)) demonstrate that the predictor achieved up to **75% hit rate** on target Stage 2 (predicting 3 out of 4 correct experts in Token 130).

### 4.3 Architectural Mechanism: Why Exposed Stall and Latency Remained Identical
The simulation results reveal why exposed GPU stall time and token latency did not decrease in these 4-token smoke runs, despite accurate predictions:

1. **DDR5 Channel Contention by Priority-1 Demands:**
   - In Step 0 (cold start), 16 to 24 expert demands arrive for non-resident experts.
   - Demands execute at Priority 1, while speculative prefetches execute at Priority 0.
   - Loading 24 experts across a 64 GB/s DDR5 bus consumes:
     $$24 \times 88.08\text{ MB} \div 64\text{ GB/s} = 33.03\text{ ms}$$
   - The speculative prefetches submitted at $t = 1.48$ ms are queued behind Priority 1 transfers and cannot complete until $t = 35.88$ ms.
   - Device compute reaches Stage 1 and Stage 2 between $t = 14$ ms and $t = 34$ ms, before the speculative transfers complete. Therefore, all Step 0 demands remain fresh demand misses.

2. **Large HBM Cache Budget (80 GB):**
   - The 80 GB local HBM cache holds over 900 experts.
   - Once Step 0 loads its demand misses, those experts remain permanently resident in HBM.
   - In subsequent tokens (Tokens 129–131), when the online predictor predicts experts that repeat from earlier tokens, the prefetch queue finds `has_resident_expert == true` and skips dispatching redundant transfers.
   - Unseen expert demands in subsequent tokens arrive as fresh demand misses.

3. **Memory Traffic Overhead of Speculative Prefetches:**
   - Dispatched prefetches that do not complete in time or are not demanded consume DDR5 bandwidth.
   - For $H=1$, 4 speculative prefetches add $4 \times 88.08\text{ MB} = 352.32\text{ MB}$ of memory traffic (+9.5% overhead).
   - Because prefetches run at Priority 0 during idle channel cycles or inter-demand gaps, they do not increase demand stall, but they cannot hide cold-start demand latency.

---

## 5. Diagnostic Artifacts

All individual run diagnostics are archived under:
[`/home/mohman/ca_project/mohman/point_3/routing_audit/routing_diagnostics/`](file:///home/mohman/ca_project/mohman/point_3/routing_audit/routing_diagnostics/)

Key diagnostic files:
- [`diagnostic_smoke_2stg_baseline_anywhere.md`](file:///home/mohman/ca_project/mohman/point_3/routing_audit/routing_diagnostics/diagnostic_2stg_baseline_anywhere.md)
- [`diagnostic_smoke_2stg_stmoe_h1_anywhere.md`](file:///home/mohman/ca_project/mohman/point_3/routing_audit/routing_diagnostics/diagnostic_smoke_2stg_stmoe_h1_anywhere.md)
- [`diagnostic_smoke_2stg_cstp_h1_anywhere.md`](file:///home/mohman/ca_project/mohman/point_3/routing_audit/routing_diagnostics/diagnostic_smoke_2stg_cstp_h1_anywhere.md)
- [`diagnostic_smoke_2stg_stmoe_h1_suite12.md`](file:///home/mohman/ca_project/mohman/point_3/routing_audit/routing_diagnostics/diagnostic_smoke_2stg_stmoe_h1_suite12.md)
- [`diagnostic_smoke_2stg_synthetic.md`](file:///home/mohman/ca_project/mohman/point_3/routing_audit/routing_diagnostics/diagnostic_smoke_2stg_synthetic.md)
- [`diagnostic_smoke_3stg_cstp_h2_anywhere.md`](file:///home/mohman/ca_project/mohman/point_3/routing_audit/routing_diagnostics/diagnostic_smoke_3stg_cstp_h2_anywhere.md)
- [`diagnostic_smoke_3stg_stmoe_h2_anywhere.md`](file:///home/mohman/ca_project/mohman/point_3/routing_audit/routing_diagnostics/diagnostic_smoke_3stg_stmoe_h2_anywhere.md)

