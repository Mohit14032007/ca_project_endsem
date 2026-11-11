# Phase 4B: Prefetching Effectiveness Instrumentation & Verification Report

**Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Deliverable Path:** [PHASE4B_INSTRUMENTATION_REPORT.md](file:///home/mohman/ca_project/mohman/point_3/PHASE4B_INSTRUMENTATION_REPORT.md)  
**Audit Reference:** [PHASE4B_EFFECTIVENESS_AUDIT.md](file:///home/mohman/ca_project/mohman/point_3/PHASE4B_EFFECTIVENESS_AUDIT.md)  
**Status:** Certified Implementation & Validation  
**Phase:** Phase 4B.2 & Phase 4B.3 Completed  

---

## 1. Executive Summary

In Phase 4B.2 and 4B.3, we addressed the instrumentation gaps identified during the read-only forensic audit and established full observability across all eight prefetch-effectiveness metrics:

1. **Prefetch Coverage:** Implemented dual-granularity accounting separating unique-expert invocations from projection-level operations ($W_1, W_3, W_2$).
2. **Timely and Late Prefetch Rates:** Captured timely resident hits, late in-flight attachments, remaining wait durations, and byte volumes.
3. **Unused Prefetches:** Quantified completed unused prefetches, evicted-before-demand prefetches, in-flight-at-end prefetches, and wasted speculative memory traffic.
4. **HBM Hit Rates:** Certified dual-granularity hit rates, resolving the $66.67\%$ cold-start artifact caused by projection conflation.
5. **Additional Memory Traffic:** Quantified speculative vs demand DDR5 bytes, baseline comparisons, and traffic overhead ratios.
6. **Exposed Memory Stall:** Measured unhidden GPU stall durations, wait per token, wait per unique demand, and prefetch-hidden stall savings.
7. **Expert-Load Latency Distributions:** Built category-separated quantile distributions (Count, Mean, Median/P50, P90, P95, P99, Max) across resident hits, timely prefetch hits, late in-flight attachments, and fresh demand misses.
8. **End-to-End & Token Latency:** Connected token and request lifecycle hooks in the simulation loop, enabling measurement of token latencies and simulated throughput (tokens/sec).

All changes are strictly observational. Zero changes were made to execution order, cache eviction policy, or memory transfer timing. All 15 existing prefetch regression tests and 10 newly authored unit tests (Cases A–J) passed with a 100% success rate.

---

## 2. Source Modifications and Call Paths

### 2.1 Simulation Loop Hook Integration

In [eval/test_deepseekv3_mixtral_style.cpp](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp):

1. **Unconditional Metric Observation:**  
   In [eval/test_deepseekv3_mixtral_style.cpp:350-362](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L350-L362), `PrefetchMetricsCollector::getInstance().set_enabled(true)` is now called unconditionally. In baseline simulations where `prefetching.enabled` is false, `predictor_type` is set to `"none"`, `horizon` to 0, and `budget` to 0, allowing complete baseline demand traffic, exposed stall, and token latency profiling without enabling speculation.

2. **Request Lifecycle Registration:**  
   Before decoding starts (around line 455), `record_request_start()` is called for each sequence in the batch:
   ```cpp
   time_ns req_start_time = device->status.device_time;
   for (int s = 0; s < max_batch_size; s++) {
     PrefetchMetricsCollector::getInstance().record_request_start(
         0, s, input_len, 0.0, static_cast<double>(req_start_time));
   }
   ```
   After decoding finishes (around line 615), `record_request_end()` is called:
   ```cpp
   time_ns req_end_time = device->status.device_time;
   for (int s = 0; s < max_batch_size; s++) {
     PrefetchMetricsCollector::getInstance().record_request_end(
         0, s, iter, static_cast<double>(req_end_time));
   }
   ```

3. **Per-Token Boundary Registration:**  
   Inside the decoding loop, before `cluster->run(metadata)`:
   ```cpp
   time_ns step_start_dev_time = device->status.device_time;
   for (int s = 0; s < max_batch_size; s++) {
     PrefetchMetricsCollector::getInstance().record_token_start(
         0, s, input_len + step, step, static_cast<double>(step_start_dev_time));
   }
   ```
   Immediately following `cluster->run(metadata)`:
   ```cpp
   time_ns step_end_dev_time = device->status.device_time;
   for (int s = 0; s < max_batch_size; s++) {
     PrefetchMetricsCollector::getInstance().record_token_end(
         0, s, input_len + step, step, static_cast<double>(step_end_dev_time));
   }
   ```

---

### 2.2 Metrics Aggregation and Quantile Processing

In [src/prefetch/prefetch_metrics_collector.h](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.h) and [src/prefetch/prefetch_metrics_collector.cpp](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.cpp):

1. **Quantile Data Structure:**  
   Defined struct `LatencyQuantiles` containing `count`, `mean_ns`, `p50_ns`, `p90_ns`, `p95_ns`, `p99_ns`, and `max_ns`.

2. **Unique Expert Invocations Deduplication:**  
   In `PrefetchMetricsCollector::finalize()`, demand records are deduplicated across projections by grouping on the 5-tuple:
   $$\langle \text{request\_id}, \text{seq\_id}, \text{token\_id}, \text{layer\_id}, \text{expert\_id} \rangle$$
   The first projection ($W_1$) determines the arrival and cold-start classification. Subsequent hits from $W_3$ and $W_2$ for the same expert within that token are recorded at the projection level but do not inflate unique expert counts.

3. **Category-Separated Latency Distributions:**  
   Quantiles are independently computed for:
   - `overall`: All expert demands combined.
   - `resident_hits`: Immediate HBM hits ($0$ ns DDR5 wait).
   - `timely_prefetch`: Prefetch completed before demand ($0$ ns DDR5 wait).
   - `late_attachment`: Demand attached to in-flight prefetch (remaining transfer wait).
   - `demand_miss`: Fresh demand miss (full DDR5 transfer wait $\approx 1.376$ ms).

4. **Stall Hidden Calculation:**  
   Stall savings are evaluated based on actual transfer times:
   $$\text{Stall Hidden} = (N_{\text{timely}} \times T_{\text{xfer}}) + \sum_{i \in \text{late}} \max(0, T_{\text{xfer}} - T_{\text{remaining}, i})$$

5. **Unified Structured Event Logging:**  
   Implemented `dump_unified_event_log(output_dir + "/prefetch_events.csv")`. This merges and sorts predictions, candidate admissions/rejections, transfers, demands, and token boundaries into a single chronological trace.

---

### 2.3 YAML Parser Robustness

In [src/prefetch/prefetch_config.cpp](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_config.cpp):
Extended `PrefetchConfig::parse_yaml` to parse `budget` and `prediction_horizon` from under `predictor:` as well as under `candidate_selection:` and `scheduling:`. This guarantees that experiment configurations with different horizons ($H=1, 2$) and budgets ($B=1, 2, 4, 8$) are parsed correctly regardless of section nesting.

---

## 3. Verification: Synthetic Unit Tests (Cases A through J)

A comprehensive unit test suite was implemented in [eval/test_phase4b_effectiveness_metrics.cpp](file:///home/mohman/ca_project/LLMSimulator/eval/test_phase4b_effectiveness_metrics.cpp) (and mirrored to [mohman/point_3/tests/test_phase4b_effectiveness_metrics.cpp](file:///home/mohman/ca_project/mohman/point_3/tests/test_phase4b_effectiveness_metrics.cpp)).

All 10 cases passed:

| Test Case | Scenario Description | Key Assertion | Status |
| :--- | :--- | :--- | :---: |
| **Case A** | Timely Prefetch Hit | `timely_resident_hits == 1`, $0$ ns wait, `overall_coverage == 1.0` | **PASS** |
| **Case B** | Late In-Flight Attachment | Demand attaches to in-flight transfer, wait = remaining ns, 0 duplicate bytes | **PASS** |
| **Case C** | Unused Speculative Prefetch | Completed but unreferenced prefetch accounts for wasted speculative bytes | **PASS** |
| **Case D** | Candidate Admission Rejection | Queue budget rejection increments `candidates_rejected_admission` | **PASS** |
| **Case E** | Transfer Cancellation | In-flight cancellation increments `transfers_cancelled` and partial bytes | **PASS** |
| **Case F** | Evicted Before Demand | Eviction before demand increments `prefetches_evicted_before_demand` | **PASS** |
| **Case G** | Projection Deduplication | $W_1, W_3, W_2$ deduplicated to 1 unique demand; cold hit rate = 0% | **PASS** |
| **Case H** | End-of-Sim In-Flight Accounting | In-flight transfer at sim end increments `prefetches_still_inflight_at_end` | **PASS** |
| **Case I** | Stall Hidden Calculation | Hidden stall equals timely savings + late attachment savings | **PASS** |
| **Case J** | Token Latency Quantiles | Token duration, mean/p50/p90/p95/p99/max and tokens/sec computed | **PASS** |

Regression suite check:
`./test_prefetch_metrics`: **15 Passed, 0 Failed**.

---

## 4. Smoke Test Results (Baseline, ST-MoE, CSTP)

Three smoke test simulations were executed using the real MoE-Anywhere trace with $B=1$, `iter=2`, `input_len=128`, `output_len=4`:

```
Baseline: /home/mohman/ca_project/mohman/point_3/configs/smoke_baseline.yaml
ST-MoE:   /home/mohman/ca_project/mohman/point_3/configs/smoke_stmoe_h1.yaml
CSTP:     /home/mohman/ca_project/mohman/point_3/configs/smoke_cstp_h1.yaml
```

### Measured Smoke Results Summary

| Metric Group | Metric | Baseline (Demand) | ST-MoE ($H=1$) | CSTP ($H=1$) |
| :--- | :--- | :---: | :---: | :---: |
| **Simulation Host Time** | Host Runtime | 2.65 s | 2.77 s | 2.73 s |
| **Demand Counts** | Total Projection Demands | 96 | 96 | 96 |
| | Unique Expert Demands | 32 | 32 | 32 |
| **HBM Hit Rates** | Projection HBM Hit Rate | 69.79% | 69.79% | 69.79% |
| | Unique Expert Hit Rate | 9.38% | 9.38% | 9.38% |
| | Fresh Demand Misses | 29 | 29 | 29 |
| **DDR5 Memory Traffic** | Demand Bytes | 2.554 GB | 2.554 GB | 2.554 GB |
| | Speculative Bytes | 0.000 GB | 0.088 GB | 0.088 GB |
| | Total DDR5 Bytes | 2.554 GB | 2.642 GB | 2.642 GB |
| **Token Timing** | Mean Token Latency | 22.035 ms | 22.035 ms | 22.035 ms |
| | P50 Token Latency | 22.035 ms | 22.035 ms | 22.035 ms |
| | Simulated Throughput | 45.38 tok/s | 45.38 tok/s | 45.38 tok/s |
| **Exposed Stall** | Aggregate Demand Wait | 39.911 ms | 39.911 ms | 39.911 ms |
| | Wait per Token | 19.956 ms | 19.956 ms | 19.956 ms |
| | Wait per Unique Demand | 1.247 ms | 1.247 ms | 1.247 ms |

### Key Observations from Smoke Tests

1. **Resolution of Cold-Start Hit Rate:**  
   In the 2-step simulation, out of 32 unique expert demands, 29 were cold first-time demands and 3 were repeated across steps (hit rate $= 3 / 32 = 9.38\%$). At the projection level, each cold demand misses on $W_1$ and hits on $W_3$ and $W_2$, yielding $67$ projection hits out of $96$ demands ($69.79\%$). Reporting both resolves the discrepancy.
2. **Token Latency Captured:**  
   `mean_token_latency_ns` evaluated to $22.035$ ms and simulated tokens/sec evaluated to $45.38$.
3. **Trace and Artifact Integrity:**  
   All 12 expected artifacts, including the unified event log `prefetch_events.csv`, were generated and verified.

The platform is certified and ready for the Primary Five-Configuration Core Comparison (Phase 4B.5).

