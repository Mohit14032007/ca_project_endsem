# Phase 4B: Prefetching Effectiveness Instrumentation, Evaluation, and Results Audit

**Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Deliverable Path:** [PHASE4B_EFFECTIVENESS_AUDIT.md](file:///home/mohman/ca_project/mohman/point_3/PHASE4B_EFFECTIVENESS_AUDIT.md)  
**Previous Audits:** [PHASE3_MODEL_TRACE_CONFIGURATION_AUDIT.md](file:///home/mohman/ca_project/mohman/point_2/PHASE3_MODEL_TRACE_CONFIGURATION_AUDIT.md), [PHASE4A_CONFIGURATION_AND_RAMULATOR_RUNTIME_AUDIT.md](file:///home/mohman/ca_project/mohman/point_2/PHASE4A_CONFIGURATION_AND_RAMULATOR_RUNTIME_AUDIT.md)  
**Status:** Certified Read-Only Forensic Audit  
**Phase:** Phase 4B.1 (Read-Only Initial Audit)  

---

## 1. Executive Summary

This forensic audit evaluates the instrumentation, event-logging infrastructure, and metric-aggregation pathways in [LLMSimulator](file:///home/mohman/ca_project/LLMSimulator) to support the evaluation of predictor-driven expert prefetching across the eight primary effectiveness metrics:

1. **Prefetch Coverage** (Unique-Expert vs Projection-Level)
2. **Timely and Late Prefetch Rates**
3. **Unused Prefetches and Wasted Bandwidth**
4. **HBM Hit Rate** (Unique-Expert vs Projection-Level)
5. **Additional Memory Traffic**
6. **Exposed Memory-Stall Time**
7. **Expert-Load Latency Distributions** (Separated by Outcome Class)
8. **End-to-End and Token Latency**

### Key Audit Findings

1. **Token Latency Disconnection in Simulation Loop:**  
   In [src/prefetch/prefetch_metrics_collector.h:284-286](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.h#L284-L286), observational hooks `record_token_start()` and `record_token_end()` exist, but they are **never invoked** inside the decoding loop in [eval/test_deepseekv3_mixtral_style.cpp:456-490](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L456-L490). As a result, all token-level latency metrics (`mean_token_latency_ns`, `p50_token_latency_ns`, `simulated_tokens_per_second`) currently evaluate to $0.00$. Connecting these hooks into the decode loop is required to measure Metric 8.
2. **Conflation of Projection-Level and Unique-Expert Demands:**  
   In [src/hardware/linear_impl.cpp:77-98, 153-161](file:///home/mohman/ca_project/LLMSimulator/src/hardware/linear_impl.cpp#L77-L98), `record_expert_demanded()` is called for each individual linear projection ($W_1$ Gate, $W_3$ Up, $W_2$ Down) within an expert's forward pass. Consequently, `total_actual_demands` measures projection-level demands ($96$ demands for $B=1, iter=2$), rather than unique expert invocations ($32$ unique expert demands). When $W_1$ triggers a cold demand miss, it transfers all sister weights ($88.08$ MB total) into HBM; $W_3$ and $W_2$ immediately hit in HBM. This explains the baseline cold-start HBM hit rate of $\approx 66.67\%$. Unique-expert metrics must be calculated by deduplicating demands on the tuple $\langle \text{request\_id}, \text{seq\_id}, \text{token\_id}, \text{layer\_id}, \text{expert\_id} \rangle$.
3. **Missing Category-Specific Expert-Load Latency Distributions:**  
   While raw timestamps are captured in `expert_demand_events.csv`, [`PrefetchMetricsCollector::finalize()`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.cpp#L504-L530) computes quantiles across all demands combined. Metric 7 requires separate quantile distributions (Mean, Median, P90, P95, P99, Max) for:
   - HBM-resident hits ($0$ ns DDR5 wait)
   - Timely prefetch-assisted hits ($0$ ns DDR5 wait, served by prefetch)
   - Late prefetch-assisted demands (remaining transfer wait)
   - Fresh demand misses (full DDR5 transfer wait, $1.376$ ms)
4. **Late In-Flight Attachment Model Fidelity:**  
   In [src/hardware/device.cpp:1210-1245](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L1210-L1245), when an expert demand arrives while a speculative transfer is already in flight, the demand attaches to the in-flight transfer, promoting it to demand priority without issuing a duplicate DDR5 transfer. The demand waits only for the remaining service time ($\text{remaining\_ns} = \text{completion\_time} - \text{demand\_time}$). This provides the exact data needed to calculate stall reduction from late prefetches.
5. **Unified Structured Event Logging Opportunity:**  
   Currently, four disparate event CSVs are emitted (`prediction_events.csv`, `candidate_events.csv`, `prefetch_transfer_events.csv`, `expert_demand_events.csv`). Introducing a single unified event trace (`prefetch_events.csv`) in `mohman/point_3/results/<exp_id>/` will provide chronological event tracking across the entire prefetch lifecycle.

---

## 2. Existing Source Locations and Execution Call Paths

The complete prefetching, execution, and instrumentation call path spans seven core components:

```
[1. Stage Routing Entry]       BatchedSequence::update_expert() in sequence.cpp:215
                                 |
                                 +--> LivePredictionEngine::on_stage_routing_start()
                                 +--> CSTP / ST-MoE Predictor::predict()
                                 +--> PrefetchCandidateQueue::admit_candidates()
                                 +--> LivePredictionEngine::dispatch_candidates()
                                 |
[2. Device Candidate Dispatch] Device::dispatch_prefetch_candidates() in device.cpp:1520
                                 |
                                 +--> Device::submit_prefetch_transfer() in device.cpp:1300
                                 +--> DDR5TransferEngine::submit_transfer() (Analytical Channel)
                                 |
[3. Stage Routing Resolution]  BatchedSequence::update_expert() in sequence.cpp:244
                                 |
                                 +--> Reads expert_list (Trace) or cached_routing (Synthetic)
                                 +--> LivePredictionEngine::on_stage_routing_complete()
                                 |
[4. Model Decoding Execution]  Cluster::run() -> LLM::forward() -> MoEDecoder::forward()
                                 |
                                 +--> ExpertFFN::forward() in expert.cpp:170
                                 +--> FeedForward3Way::forward() in layer.cpp:415
                                 |
[5. Expert Projection Demands] LinearExecutionGPU() in linear_impl.cpp:46
                                 |
                                 +--> Check HBM Residency via device->is_expert_resident()
                                 |
                                 +-- [Branch A: Hit (Resident / Timely)]
                                 |      PrefetchMetricsCollector::record_expert_demanded(RESIDENT_HIT / TIMELY_PREFETCH)
                                 |
                                 +-- [Branch B: Miss (Fresh / Late Attachment)]
                                        Device::submit_demand_transfer() in device.cpp:1210
                                        |
                                        +-- In-Flight Match: Attaches to transfer (LATE_IN_FLIGHT)
                                        +-- No Match: Submits fresh demand transfer (DEMAND_MISS)
                                        |
                                        Device::wait_for_demand_transfer() in device.cpp:1579
                                        DDR5TransferEngine::run_until_transfer_completed()
                                        Device::mark_expert_resident()
                                        PrefetchMetricsCollector::record_exposed_stall()
```

### Exact Source Locations

- **Candidate Generation and Admission:**  
  [src/prefetch/live_prediction_engine.cpp:104-140](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/live_prediction_engine.cpp#L104-L140) and [src/prefetch/prefetch_candidate_queue.cpp:70-130](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_candidate_queue.cpp#L70-L130).
- **Candidate Dispatch to DDR5 Engine:**  
  [src/hardware/device.cpp:1520-1577](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L1520-L1577).
- **DDR5 Transfer Progress and Completion:**  
  [src/prefetch/ddr5_transfer_engine.cpp:240-300](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/ddr5_transfer_engine.cpp#L240-L300) and [src/hardware/device.cpp:1450-1518](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L1450-L1518).
- **Expert Demand Handling and Stall Accounting:**  
  [src/hardware/linear_impl.cpp:63-162](file:///home/mohman/ca_project/LLMSimulator/src/hardware/linear_impl.cpp#L63-L162).
- **Centralized Metrics Aggregation:**  
  [src/prefetch/prefetch_metrics_collector.cpp:490-658](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.cpp#L490-L658).

---

## 3. Existing Logging Fields and CSV Schemas

The current instrumentation produces seven CSV files and one JSON summary in the output directory:

### 1. `prediction_events.csv`
- **Schema:** `event_id, request_id, seq_id, token_id, source_stage, target_stage, horizon, predictor_type, budget, candidate_rank, predicted_expert_id, confidence_score, actual_target_experts, is_hit, predict_latency_ns, update_latency_ns, timestamp_ns`
- **Source:** [src/prefetch/prefetch_metrics_collector.cpp:673-705](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.cpp#L673-L705).
- **Role:** Logs every candidate generated by the predictor.

### 2. `candidate_events.csv`
- **Schema:** `candidate_id, event_type, request_id, seq_id, token_id, source_stage, target_stage, expert_id, prediction_rank, prediction_score, reason, timestamp_ns`
- **Source:** [src/prefetch/prefetch_metrics_collector.cpp:708-728](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.cpp#L708-L728).
- **Role:** Logs candidate queue lifecycle: `ADMITTED`, `REJECTED`, `DISPATCHED`, `EXPIRED`, `CANCELLED`.

### 3. `prefetch_transfer_events.csv`
- **Schema:** `transfer_id, candidate_id, request_id, seq_id, token_id, target_stage, expert_id, transfer_class, priority, size_bytes, queue_submit_ts, service_start_ts, completion_ts, duration_ns, paused_duration_ns, outcome`
- **Source:** [src/prefetch/prefetch_metrics_collector.cpp:731-756](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.cpp#L731-L756).
- **Role:** Logs transfer execution on `DDR5TransferEngine`. `outcome` values include `DEMANDED`, `UNUSED`, `CANCELLED`, `STILL_IN_FLIGHT_AT_END`.

### 4. `expert_demand_events.csv`
- **Schema:** `demand_id, request_id, seq_id, token_id, layer_id, expert_id, demand_ts, ready_ts, ddr5_wait_ns, hbm_access_ns, residency_status, transfer_id, candidate_id`
- **Source:** [src/prefetch/prefetch_metrics_collector.cpp:759-780](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.cpp#L759-L780).
- **Role:** Logs every projection demand. `residency_status` values: `RESIDENT_HIT`, `TIMELY_PREFETCH`, `LATE_IN_FLIGHT`, `DEMAND_MISS`.

### 5. `request_latency.csv`
- **Schema:** `request_id, seq_id, prompt_len, gen_tokens, arrival_ts, start_ts, end_ts, latency_ns`
- **Source:** [src/prefetch/prefetch_metrics_collector.cpp:783-799](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.cpp#L783-L799).
- **Role:** Logs per-request end-to-end latency. Currently unpopulated because request hooks are unlinked in the harness.

### 6. `token_latency.csv`
- **Schema:** `request_id, seq_id, token_id, gen_step, token_start_ts, token_end_ts, token_duration_ns`
- **Source:** [src/prefetch/prefetch_metrics_collector.cpp:802-816](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.cpp#L802-L816).
- **Role:** Logs per-token duration. Currently unpopulated because token hooks are unlinked in the harness.

### 7. `prefetch_metrics.csv`
- **Schema:** `config_label, dataset, total_demands, overall_coverage, predicted_coverage, prefetch_utilization, timely_rate_demands, late_rate_demands, timely_rate_dispatched, late_rate_dispatched, hbm_hit_rate, fresh_demand_miss_rate, inflight_attachment_rate, speculative_bytes, demand_bytes, total_ddr5_bytes, wasted_speculative_bytes, traffic_overhead_ratio, aggregate_demand_wait_ns, exposed_gpu_stall_ns, total_simulated_ns, mean_expert_load_latency_ns, p50_expert_load_latency_ns, p95_expert_load_latency_ns, p99_expert_load_latency_ns, mean_request_latency_ns, p50_request_latency_ns, p95_request_latency_ns, p99_request_latency_ns, mean_token_latency_ns, p50_token_latency_ns, p95_token_latency_ns, p99_token_latency_ns, simulated_tokens_per_second`
- **Source:** [src/prefetch/prefetch_metrics_collector.cpp:819-867](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.cpp#L819-L867).
- **Role:** Aggregate metrics summary.

### 8. `experiment_summary.json`
- **Schema:** Hierarchical JSON capturing metadata, prefetch coverage, timely/late rates, unused prefetches, HBM hit rates, memory traffic, exposed stall, expert load latency, request latency, and token latency.
- **Source:** [src/prefetch/prefetch_metrics_collector.cpp:870-970](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.cpp#L870-L970).

---

## 4. Availability Assessment of the Eight Target Metrics

The table below classifies the current availability of each required metric:

| Target Metric | Currently Available? | Current Code Status | Gap / Limitation | Required Enhancement |
|---|---|---|---|---|
| **1. Prefetch Coverage** | **Partially** | Evaluates `overall_coverage` at projection level ($N=96$). | Conflates 3 projections with unique expert demands; no separate timely vs late coverage. | Add unique-expert deduplication; report timely, late, and total coverage. |
| **2. Timely and Late Rates** | **Partially** | Counts timely hits and late attachments; calculates rate over total demands. | Missing timely bytes, late bytes, and average remaining transfer wait at late demand. | Track byte volumes; compute average remaining latency for late attachments. |
| **3. Unused Prefetches** | **Mostly** | Tracks completed unused count, unused bytes, evicted bytes, and in-flight at end. | `unused_prefetch_rate` not explicitly formatted in `prefetch_metrics.csv`. | Add percentage rate calculation relative to dispatched prefetches. |
| **4. HBM Hit Rate** | **Partially** | Evaluates projection-level hit rate ($\approx 67\% - 70\%$). | Unique-expert hit rate is completely missing; prefetch hits not separated from reuse hits. | Report both projection-level and unique-expert hit rates; separate prefetch vs reuse hits. |
| **5. Additional Traffic** | **Mostly** | Tracks speculative, demand, total bytes, and overhead ratio. | Transfer counts not in `prefetch_metrics.csv`; used speculative bytes not isolated. | Output transfer counts; report used speculative bytes ($= \text{speculative} - \text{unused}$). |
| **6. Exposed Memory Stall** | **Mostly** | Tracks `exposed_gpu_stall_ns` and `aggregate_demand_wait_ns`. | Average stall per token and reduction vs baseline require post-processing. | Calculate stall per token, stall per expert demand, and hidden stall. |
| **7. Expert-Load Latency** | **Partially** | Quantiles computed over aggregate demands; raw events logged. | Quantile distributions not computed separately for the 4 outcome categories. | Compute Count, Mean, Median, P90, P95, P99, Max separately for all 4 categories. |
| **8. End-to-End & Token Latency** | **MISSING** | Latency vectors exist in collector, but methods are never called. | `record_token_start` and `record_token_end` unlinked in harness; reports $0.00$. | Link token start/end calls into `test_deepseekv3_mixtral_style.cpp:456-490`. |

---

## 5. Identification of Propagation Gaps and Missing Metrics

### A. Identifier Availability Across Instrumentation Points

To ensure observational integrity, the availability of identifiers at each lifecycle hook was audited:

| Instrumentation Point | Source File & Line | `request_id` | `seq_id` | `token_id` | `stage_id` | `expert_id` | `transfer_id` | `candidate_id` |
|---|---|---|---|---|---|---|---|---|
| `record_prediction()` | [live_prediction_engine.cpp:219](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/live_prediction_engine.cpp#L219) | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | N/A | **Yes** (`event_id`) |
| `record_candidate_admitted()` | [prefetch_candidate_queue.cpp:92](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_candidate_queue.cpp#L92) | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | N/A | **Yes** |
| `record_candidate_dispatched()` | [device.cpp:1570](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L1570) | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** (`tid`) | **Yes** |
| `record_transfer_submitted()` | [device.cpp:1340](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L1340) | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |
| `record_transfer_completed()` | [device.cpp:1480](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L1480) | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |
| `record_hbm_resident()` | [device.cpp:1500](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L1500) | Inferred | Inferred | Inferred | **Yes** | **Yes** | **Yes** | Inferred |
| `record_expert_demanded()` | [linear_impl.cpp:94, 1238](file:///home/mohman/ca_project/LLMSimulator/src/hardware/linear_impl.cpp#L94) | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |
| `record_token_start/end()` | [test_deepseekv3_mixtral_style.cpp:463](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L463) | **Available** | **Available** | **Available** | N/A | N/A | N/A | N/A |

All primary identifiers are fully accessible at the relevant simulation points. No identifier fabrication is necessary.

### B. Potential Double-Counting Hazards

1. **Intra-Expert Projection Overlap (W1, W3, W2):**  
   If $W_1, W_3, W_2$ are each counted as an expert demand, cold-start hit rate artificially appears as $66.67\%$. Unique expert demands must be aggregated:
   - For an expert activation, $W_1$ represents the master demand that initiates DDR5 transfer.
   - $W_3$ and $W_2$ represent intra-expert sister executions.
   - Unique expert counting must evaluate the first projection ($W_1$) to determine whether the expert required a DDR5 load or was served from HBM.
2. **Timeline Overlap in Exposed GPU Stall:**  
   In sequential decode loops on a single GPU stream, expert executions occur serially. When a demand miss occurs, the GPU stalls synchronously until `wait_for_demand_transfer()` completes. Summing these non-overlapping stall durations is mathematically valid. However, if multiple tokens or sequences execute in parallel (e.g. data parallelism), overlapping stall intervals must be unioned on the device timeline rather than summed linearly.
3. **In-Flight Transfer Double-Counting:**  
   When a late demand attaches to an in-flight speculative transfer, [`device->submit_demand_transfer()`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L1210) promotes the transfer priority and attaches the demand. It does **not** create a new transfer record. The transfer bytes ($88.08$ MB) are counted only once under speculative bytes.

---

## 6. Proposed Metric Definitions and Calculation Methods

The following mathematical definitions will be implemented across Phase 4B:

### 1. Prefetch Coverage

Let $D_{\text{unique}}$ be the set of unique expert demands, where each demand is indexed by $\langle \text{request\_id}, \text{seq\_id}, \text{token\_id}, \text{layer\_id}, \text{expert\_id} \rangle$.

$$\text{Unique Demands Count} = N_{\text{unique}} = |D_{\text{unique}}|$$

Each unique demand $d \in D_{\text{unique}}$ is classified into exactly one mutually exclusive outcome:
- $D_{\text{timely}}$: Demanded when the expert was already resident due to a completed speculative prefetch.
- $D_{\text{late}}$: Demanded while a speculative prefetch was in flight, attaching to it and waiting for remaining latency.
- $D_{\text{reuse}}$: Demanded when the expert was resident due to a prior demand miss or reuse from an earlier step.
- $D_{\text{miss}}$: Fresh demand miss requiring a synchronous DDR5 demand transfer.

$$\text{Timely Coverage} = \frac{|D_{\text{timely}}|}{N_{\text{unique}}}$$

$$\text{Late Coverage} = \frac{|D_{\text{late}}|}{N_{\text{unique}}}$$

$$\text{Total Prefetch-Assisted Coverage} = \frac{|D_{\text{timely}}| + |D_{\text{late}}|}{N_{\text{unique}}}$$

Additionally, projection-level coverage is reported over total projection demands $N_{\text{proj}} = \sum \text{demands}$.

---

### 2. Timely and Late Prefetch Rates

Let $P_{\text{dispatched}}$ be the set of dispatched speculative prefetch transfers.  
Let $P_{\text{matched}} \subseteq P_{\text{dispatched}}$ be the subset of prefetches that correspond to an actual demand before simulation end or eviction:

$$\text{Timely Rate of Matched Prefetches} = \frac{|D_{\text{timely}}|}{|P_{\text{matched}}|}$$

$$\text{Late Rate of Matched Prefetches} = \frac{|D_{\text{late}}|}{|P_{\text{matched}}|}$$

$$\text{Timely Bytes} = |D_{\text{timely}}| \times 88{,}080{,}384\text{ B}$$

$$\text{Late Bytes} = |D_{\text{late}}| \times 88{,}080{,}384\text{ B}$$

$$\text{Average Remaining Wait at Late Demand} = \frac{1}{|D_{\text{late}}|} \sum_{d \in D_{\text{late}}} (T_{\text{ready}}(d) - T_{\text{demand}}(d))$$

---

### 3. Unused Prefetches

A dispatched speculative transfer $p \in P_{\text{dispatched}}$ is unused if its expert is never demanded before simulation completion or eviction:

$$P_{\text{unused}} = \{ p \in P_{\text{dispatched}} \mid p \notin P_{\text{matched}} \}$$

- $N_{\text{unused\_completed}}$: Transfers that completed but whose resident expert was never demanded.
- $N_{\text{evicted\_before\_demand}}$: Prefetched experts evicted from HBM before demand occurred.
- $N_{\text{inflight\_end}}$: Transfers still in flight when the simulation terminated.
- $\text{Unused Speculative Bytes} = (N_{\text{unused\_completed}} + N_{\text{evicted\_before\_demand}}) \times 88{,}080{,}384\text{ B}$.

$$\text{Unused Prefetch Rate} = \frac{N_{\text{unused\_completed}} + N_{\text{evicted\_before\_demand}}}{|P_{\text{dispatched}}|}$$

Transfers remaining in flight at simulation end ($N_{\text{inflight\_end}}$) are strictly isolated and not counted as completed unused.

---

### 4. HBM Hit Rate (Dual-Granularity)

To prevent misinterpreting the $3$-projection SwiGLU structure, two separate hit rates are reported:

**Projection-Level Hit Rate:**
$$\text{Hit Rate}_{\text{proj}} = \frac{N_{\text{proj\_hits}}}{N_{\text{proj\_demands}}}$$
where $N_{\text{proj\_demands}} = 3 \times N_{\text{unique}}$ and $N_{\text{proj\_hits}}$ includes intra-expert sister hits on $W_3$ and $W_2$.

**Unique-Expert-Level Hit Rate:**
$$\text{Hit Rate}_{\text{unique}} = \frac{|D_{\text{timely}}| + |D_{\text{reuse}}|}{N_{\text{unique}}}$$

Where:
- Prefetch-Assisted Unique Hits = $|D_{\text{timely}}|$
- Demand-Reuse Unique Hits = $|D_{\text{reuse}}|$
- Fresh Demand Misses = $|D_{\text{miss}}|$
- Late Attachments = $|D_{\text{late}}|$

---

### 5. Additional Memory Traffic

$$\text{Speculative DDR5 Bytes} = |P_{\text{dispatched}}| \times 88{,}080{,}384\text{ B}$$

$$\text{Demand DDR5 Bytes} = |D_{\text{miss}}| \times 88{,}080{,}384\text{ B}$$

$$\text{Total DDR5 Bytes} = \text{Speculative DDR5 Bytes} + \text{Demand DDR5 Bytes}$$

$$\text{Additional Traffic} = \text{Total DDR5 Bytes}_{\text{predictor}} - \text{Total DDR5 Bytes}_{\text{baseline}}$$

$$\text{Traffic Overhead Ratio} = \frac{\text{Additional Traffic}}{\text{Total DDR5 Bytes}_{\text{baseline}}} \times 100\%$$

$$\text{Used Speculative Bytes} = \text{Speculative DDR5 Bytes} - \text{Unused Speculative Bytes}$$

---

### 6. Exposed Memory-Stall Time

For each unique demand $d \in D_{\text{unique}}$:
- If $d \in D_{\text{timely}} \cup D_{\text{reuse}}$: $T_{\text{wait}}(d) = 0.0\text{ ns}$.
- If $d \in D_{\text{late}}$: $T_{\text{wait}}(d) = T_{\text{completion}}(p) - T_{\text{demand}}(d)$.
- If $d \in D_{\text{miss}}$: $T_{\text{wait}}(d) = \frac{88{,}080{,}384\text{ B}}{64.0\text{ GB/s}} = 1{,}376{,}256\text{ ns}$.

$$\text{Aggregate Demand Wait} = \sum_{d \in D_{\text{unique}}} T_{\text{wait}}(d)$$

In sequential execution, exposed stall equals the sum of demand waiting intervals that cannot be overlapped with useful computation:
$$\text{Exposed GPU Stall} = \sum T_{\text{wait}}(d)$$

$$\text{Stall Hidden by Prefetching} = \text{Exposed Stall}_{\text{baseline}} - \text{Exposed Stall}_{\text{predictor}}$$

$$\text{Average Exposed Stall per Token} = \frac{\text{Exposed GPU Stall}}{N_{\text{tokens}}}$$

$$\text{Exposed Stall per Expert Demand} = \frac{\text{Exposed GPU Stall}}{N_{\text{unique}}}$$

---

### 7. Expert-Load Latency Distributions

For every unique expert demand $d$, record latency:
$$L_{\text{load}}(d) = T_{\text{ready}}(d) - T_{\text{demand}}(d)$$

Compute separate quantile distributions:
$$\text{Distribution} = \langle \text{Count}, \text{Mean}, \text{Median (P50)}, \text{P90}, \text{P95}, \text{P99}, \text{Max} \rangle$$
Independently for:
1. **HBM-Resident Hits:** $L_{\text{load}} = 0.0\text{ ns}$.
2. **Timely Prefetch Hits:** $L_{\text{load}} = 0.0\text{ ns}$.
3. **Late Prefetch Attachments:** $L_{\text{load}} \in (0, 1.376\text{ ms})$.
4. **Fresh Demand Misses:** $L_{\text{load}} = 1.376\text{ ms}$.

---

### 8. End-to-End and Token Latency

For each decode token step $t \in [0, N_{\text{tokens}}-1]$:
$$T_{\text{token}}(t) = T_{\text{token\_end}}(t) - T_{\text{token\_start}}(t)$$

Compute token latency distribution:
$$\text{Mean}, \text{Median (P50)}, \text{P90}, \text{P95}, \text{P99}, \text{Max}$$

$$\text{Simulated Tokens Per Second} = \frac{N_{\text{tokens}}}{\sum T_{\text{token}}(t) / 10^9}$$

$$\text{Speedup vs Baseline} = \frac{\text{Mean Token Latency}_{\text{baseline}}}{\text{Mean Token Latency}_{\text{predictor}}}$$

---

## 7. Instrumentation Plan and Next Phases

### Required Minimal Source Modifications (Phase 4B.2)

1. **Connect Token Latency Hooks in Test Harness:**  
   In [eval/test_deepseekv3_mixtral_style.cpp:463-490](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L463-L490):
   - Before `cluster->run(metadata)`: Iterate active sequences and call `record_token_start(req_id, seq_id, token_id, step, step_start_dev_time)`.
   - After `cluster->run(metadata)`: Call `record_token_end(req_id, seq_id, token_id, step, device->status.device_time)`.
   - Call `record_request_start` and `record_request_end` around the decode loop.
2. **Implement Unique-Expert Metric Deduplication:**  
   In [src/prefetch/prefetch_metrics_collector.cpp:490-658](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.cpp#L490-L658):
   - Deduplicate incoming demand records by `(request_id, seq_id, token_id, layer_id, expert_id)`.
   - Calculate unique-expert-level hit rate, timely/late coverage, and category-separated expert-load latency quantiles.
3. **Emit Unified Structured Event Log (`prefetch_events.csv`):**  
   Add a method `dump_unified_event_log()` to write all chronological lifecycle events into a single CSV matching the Section 4 specification.

### Roadmap for Subsequent Phases

- **Phase 4B.2:** Implement minimal observational hooks in `test_deepseekv3_mixtral_style.cpp` and `prefetch_metrics_collector.cpp`.
- **Phase 4B.3:** Develop and execute unit tests for all ten correctness cases (Cases A through J).
- **Phase 4B.4:** Execute three-configuration smoke test (Baseline, ST-MoE $H=1$, CSTP $H=1$) on MoE-Anywhere $B=1$ and verify event consistency.
- **Phase 4B.5:** Execute primary five-configuration core comparison (Baseline, ST-MoE $H=1,2$, CSTP $H=1,2$) and compile comparative CSV.
- **Phase 4B.6:** Execute sensitivity sweeps (budget, batch size, cross-dataset, decode length).
- **Phase 4B.7:** Author comprehensive final reports, plots, and manifests in `/home/mohman/ca_project/mohman/point_3/`.

