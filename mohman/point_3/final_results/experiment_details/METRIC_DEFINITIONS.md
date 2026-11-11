# Point 3 — Metric Definitions and Mathematical Validation

**Repository:** [`/home/mohman/ca_project/LLMSimulator`](file:///home/mohman/ca_project/LLMSimulator)  
**Deliverable Path:** [`mohman/point_3/final_results/experiment_details/METRIC_DEFINITIONS.md`](file:///home/mohman/ca_project/mohman/point_3/final_results/experiment_details/METRIC_DEFINITIONS.md)  
**Status:** Authoritative Mathematical Specification  

---

## 1. Lifecycle Event Terminology

To eliminate metric ambiguity, the simulator enforces strict boundaries between seven discrete events in an expert prefetch lifecycle:

```
+----------------------------------------------------------------------------------------------------+
|                                  PREFETCH EVENT LIFECYCLE BOUNDARIES                               |
+-------------------+--------------------------------------------------------------------------------+
| Lifecycle Stage   | Operational Definition & Simulator Boundary Hook                               |
+-------------------+--------------------------------------------------------------------------------+
| 1. Prediction     | Software generation of candidate expert ID e with confidence score in          |
|                   | LivePredictionEngine::on_stage_routing_start(). Does not move bytes.          |
+-------------------+--------------------------------------------------------------------------------+
| 2. Candidate      | Filtering against resident/inflight registry and budget clamping (top C) in    |
|    Admission      | PrefetchCandidateQueue::admit_candidates().                                     |
+-------------------+--------------------------------------------------------------------------------+
| 3. Dispatch       | Host issues an asynchronous DMA transfer command to Device::submit_prefetch(). |
+-------------------+--------------------------------------------------------------------------------+
| 4. Transfer Start | The transfer reaches head of DDR5 queue and begins physical bus transmission.  |
+-------------------+--------------------------------------------------------------------------------+
| 5. Transfer       | All 88,080,384 bytes finish crossing the interconnect into GPU HBM buffers.    |
|    Completion     | Logged in DDR5TransferEngine::complete_active_transfer().                       |
+-------------------+--------------------------------------------------------------------------------+
| 6. HBM Residency  | Expert weights are registered in resident_experts and entered into LRU list.   |
+-------------------+--------------------------------------------------------------------------------+
| 7. Demand         | A compute kernel (LinearImpl) accesses the expert weights for GEMM computation.|
+-------------------+--------------------------------------------------------------------------------+
```

---

## 2. Mathematical Formulations of the Eight Mandatory Metrics

### Metric 1: Prefetch Coverage

#### Strict Timely Coverage (%)
Measures the proportion of demand events where speculative data was completely resident in HBM *prior* to demand arrival:
$$\text{Coverage}_{\text{strict}} = \frac{N_{\text{timely\_resident\_hits}}}{N_{\text{demands\_with\_prediction}}} \times 100\%$$
- **Simulator Event Hook:** Incremented in `Device::is_expert_resident()` when `res_it->second.is_prefetched == true` and `res_it->second.was_demanded == false`.
- **Observed Value:** **0.00% across all 73 runs**. Transfer time ($1.376 \text{ ms}$) exceeds compute window ($20$–$50 \mu\text{s}$).

#### Prefetch-Associated Demand Coverage (%)
Measures the proportion of demand events that attached to a completed or in-flight speculative transfer:
$$\text{Coverage}_{\text{associated}} = \frac{N_{\text{timely\_hits}} + N_{\text{late\_hits}}}{N_{\text{demands\_with\_prediction}}} \times 100\%$$
- **Simulator Event Hook:** Incremented when demand attaches to `inflight_registry` with `transfer_class == PREFETCH`.
- **Observed Value:** Scales up to **9.23% (ST-MoE)** and **4.81% (CSTP)** in Workload 7.

---

### Metric 2: Timely Prefetch Rate (%)
The fraction of matched speculative transfers that completed prior to demand:
$$\text{Rate}_{\text{timely}} = \frac{N_{\text{timely\_hits}}}{N_{\text{timely\_hits}} + N_{\text{late\_hits}}} \times 100\%$$
- **Observed Value:** **0.00% across all runs**.

---

### Metric 3: Late Prefetch Rate (%)
The fraction of matched speculative transfers that were in-flight at demand arrival:
$$\text{Rate}_{\text{late}} = \frac{N_{\text{late\_hits}}}{N_{\text{timely\_hits}} + N_{\text{late\_hits}}} \times 100\%$$
- **Observed Value:** **100.00% across all matched runs**. Late prefetches truncate stall by their elapsed bus transit time.

---

### Metric 4: Unused Prefetches
The count and percentage of dispatched transfers that were never referenced before eviction:
$$\text{Unused}_{\text{rate}} = \frac{N_{\text{completed\_unused\_resident}} + N_{\text{completed\_unused\_evicted}}}{N_{\text{dispatched\_prefetches}}} \times 100\%$$
- **Wasted Speculative Traffic:**
  $$\text{Bytes}_{\text{wasted}} = N_{\text{unused\_prefetches}} \times 88,080,384 \text{ bytes}$$
- **Observed Values (W7):** ST-MoE wasted **530.42 GB** ($5.87\%$ unused); CSTP wasted **335.76 GB** ($7.05\%$ unused).

---

### Metric 5: HBM Expert Cache Hit Rate (%)
The percentage of expert tensor accesses satisfied immediately from GPU HBM:
$$\text{HitRate}_{\text{HBM}} = \frac{N_{\text{ordinary\_hbm\_hits}} + N_{\text{timely\_resident\_hits}}}{N_{\text{total\_demands}}} \times 100\%$$
- **Observed Value:** **Identically 67.0% across all large-workload runs**. Late in-flight attachments do not count as HBM hits because weights are still in transit.

---

### Metric 6: Memory Traffic Accounting

#### Total DDR5 Traffic
$$T_{\text{total}} = T_{\text{demand}} + T_{\text{speculative}}$$

#### Prevented Demand Traffic
Demand loads avoided because an expert was already transferred speculatively:
$$T_{\text{prevented}} = N_{\text{avoided\_physical\_loads}} \times 88,080,384 \text{ bytes}$$

#### Net Additional Memory Traffic
$$\Delta T = T_{\text{total}}(\text{Predictor}) - T_{\text{total}}(\text{Baseline})$$
$$\Delta T = T_{\text{wasted\_speculative}} - T_{\text{prevented\_demand}}$$

#### Traffic Overhead Percentage (%)
$$\text{Overhead}_{\text{traffic}} = \frac{\Delta T}{T_{\text{total}}(\text{Baseline})} \times 100\%$$

---

### Metric 7: Exposed Memory-Stall Time

#### Absolute Exposed Memory Stall (ms)
Cumulative idle time on GPU tensor cores waiting for off-chip transfers:
$$S_{\text{exposed}} = \sum_{i=1}^{N_{\text{demands}}} \max\left(0, T_{\text{ready}}(i) - T_{\text{demand}}(i)\right)$$

#### Exposed Memory-Stall Reduction ($\Delta S$)
$$\Delta S = S_{\text{baseline}} - S_{\text{predictor}}$$

#### Percentage Stall Reduction ($R_S$)
$$R_S = \frac{S_{\text{baseline}} - S_{\text{predictor}}}{S_{\text{baseline}}} \times 100\%$$
- **Important Distinction:** Do not confuse $R_S$ with overall execution speedup. A $-1.39\%$ stall reduction in W7 saved **6.745 seconds** of tensor core idle time.

---

### Metric 8: Latency and Throughput Metrics

#### Mean Decode Step Latency (ms)
Average simulated duration across all concurrent sequences to complete one decode generation step:
$$\text{Latency}_{\text{step}} = \frac{T_{\text{simulated\_total}}}{T_{\text{tokens}}}$$

#### Total Simulated Time Reduction ($\Delta T_{\text{sim}}$)
$$\Delta T_{\text{sim}} = T_{\text{sim}}(\text{Baseline}) - T_{\text{sim}}(\text{Predictor})$$

#### Decode Step Latency Reduction ($\Delta \text{Latency}_{\text{step}}$)
$$\Delta \text{Latency}_{\text{step}} = \frac{\Delta T_{\text{sim}} \times 1000}{T_{\text{tokens}}}$$
- **Verification Proof in W7:** $\frac{6,716.95 \text{ ms}}{64 \text{ steps}} = \mathbf{104.952 \text{ ms per decode step}}$, matching mean token latency.

#### Inference Throughput (tokens/s)
$$\text{Throughput} = \frac{B \times T_{\text{tokens}}}{T_{\text{simulated\_total}}}$$

