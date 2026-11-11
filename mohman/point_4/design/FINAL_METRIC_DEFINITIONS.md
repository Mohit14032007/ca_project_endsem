# Final Metric Definitions and Instrumentation Specification

## Point 4: HBM Management Beyond Standard LRU

---

## 1. Executive Summary

This document establishes the definitive, mathematically rigorous definitions, denominators, event counters, and instrumentation hooks for Point 4. It explicitly resolves the sub-projection counting inflation ($3\times$ vs $1\times$) and ensures that every reported metric is free from double-counting.

---

## 2. Event Definitions and Denominators

### 2.1 The Two Levels of Demand Accounting
In DeepSeek-V3 MoE execution, each routed expert consists of three sister projection tensors: $W_1$ (gate), $W_3$ (up), and $W_2$ (down).
Because `PrefetchMetricsCollector` logs an event for each projection tensor, we formally distinguish:
1. **Coalesced Demand Events ($D_{\text{coalesced}}$):** An MoE stage activation where expert $\{l, e\}$ is demanded by one or more sequences in the batch. This represents the actual weight loading unit ($88.08 \text{ MB}$).
2. **Tensor Demand Events ($D_{\text{tensor}}$):** Individual projection tensor executions ($D_{\text{tensor}} = 3 \times D_{\text{coalesced}}$).

> [!IMPORTANT]
> All primary architectural evaluations for Point 4 will use **Coalesced Demand Events** as the standard denominator, ensuring physical correspondence to expert loading and cache residency.

---

## 3. Mathematical Metric Definitions

### 3.1 Cache Performance & Thrashing Metrics

#### 1. Coalesced HBM Hit Rate ($H_{\text{coalesced}}$)
Fraction of coalesced expert demands satisfied directly from GPU HBM without incurring off-chip DDR5 transfer wait:
$$H_{\text{coalesced}} = \frac{N_{\text{coalesced\_resident\_hits}} + N_{\text{coalesced\_timely\_hits}}}{N_{\text{coalesced\_actual\_demands}}}$$
- **Numerator:** Sum of demands hitting already-resident experts plus demands hitting timely prefetched experts.
- **Denominator:** Total unique expert activations across all stages and tokens.

#### 2. Coalesced Demand Miss Rate ($M_{\text{coalesced}}$)
$$M_{\text{coalesced}} = \frac{N_{\text{coalesced\_demand\_misses}}}{N_{\text{coalesced\_actual\_demands}}} = 1.0 - H_{\text{coalesced}}$$

#### 3. Total Eviction Count ($E_{\text{total}}$)
Total number of expert eviction events from HBM:
$$E_{\text{total}} = E_{\text{capacity}} + E_{\text{speculative}} + E_{\text{unconsumed\_prefetch}}$$
- $E_{\text{capacity}}$: Eviction of a demand-verified resident expert to make space for a new expert.
- $E_{\text{speculative}}$: Eviction of a quarantined speculative expert under PABI.
- $E_{\text{unconsumed\_prefetch}}$: Eviction of a prefetched expert that completed transfer but was never demanded before eviction.

#### 4. Expert Reload Count & Churn Rate ($R_{\text{reload}}$)
Measures cache thrashing—how often the cache evicts an expert and subsequently reloads it:
$$R_{\text{reload}} = \sum_{k=(l, e)} \mathbb{I}\Big( \text{demand\_miss}(k) \;\land\; k \in \text{previously\_evicted\_experts} \Big)$$
$$\text{Churn Rate} = \frac{R_{\text{reload}}}{N_{\text{coalesced\_actual\_demands}}}$$

---

### 3.2 Memory Traffic Metrics

#### 5. Demand DDR5 Traffic ($M_{\text{demand}}$)
Total bytes transferred across the DDR5 interface to service demand misses:
$$M_{\text{demand}} = N_{\text{demand\_transfers\_completed}} \times 88,080,384 \text{ bytes}$$

#### 6. Speculative DDR5 Traffic ($M_{\text{speculative}}$)
Total bytes transferred across DDR5 for speculative prefetching:
$$M_{\text{speculative}} = N_{\text{prefetch\_transfers\_completed}} \times 88,080,384 \text{ bytes}$$

#### 7. Total Off-Chip DDR5 Traffic ($M_{\text{total}}$)
$$M_{\text{total}} = M_{\text{demand}} + M_{\text{speculative}} \quad [\text{Bytes}]$$

#### 8. Traffic Overhead Ratio ($O_{\text{traffic}}$)
Ratio of total traffic to the demand-only baseline traffic ($M_{\text{baseline}}$):
$$O_{\text{traffic}} = \frac{M_{\text{total}}}{M_{\text{baseline}}}$$

---

### 3.3 Latency & Stall Metrics

#### 9. Exposed GPU Memory Stall Time ($S_{\text{exposed}}$)
Simulated time (ns) where the GPU execution pipeline is fully stalled waiting for off-chip expert transfers:
$$S_{\text{exposed}} = \sum_{i \in \text{demand\_misses}} \max\big(0.0, \; \text{transfer\_complete\_ts}_i - \text{demand\_start\_ts}_i\big)$$

#### 10. Aggregate Demand Wait Time ($W_{\text{demand}}$)
Sum of latency experienced by all individual expert demands:
$$W_{\text{demand}} = \sum_{i} (\text{ready\_ts}_i - \text{demand\_ts}_i) \quad [\text{ns}]$$

#### 11. End-to-End Execution Time ($T_{\text{exec}}$)
Total simulated wall-clock time required to generate all tokens in the workload:
$$T_{\text{exec}} = \text{device\_time}_{\text{end}} - \text{device\_time}_{\text{start}} \quad [\text{ns}]$$

#### 12. Mean Token Latency ($L_{\text{token}}$)
$$L_{\text{token}} = \frac{T_{\text{exec}}}{N_{\text{output\_tokens}}} \quad [\text{ms/tok}]$$

---

### 3.4 Prefetch Effectiveness Metrics

#### 13. Prefetch Coverage ($C_{\text{cov}}$)
Fraction of expert demands assisted by prefetching:
$$C_{\text{cov}} = \frac{N_{\text{timely\_resident\_hits}} + N_{\text{late\_inflight\_attachments}}}{N_{\text{coalesced\_actual\_demands}}}$$

#### 14. Timely Prefetch Rate ($R_{\text{timely}}$)
Fraction of dispatched prefetches that completed in time to eliminate demand stall:
$$R_{\text{timely}} = \frac{N_{\text{timely\_resident\_hits}}}{N_{\text{dispatched\_prefetches}}}$$

#### 15. Unused Completed Prefetch Rate ($U_{\text{unused}}$)
Fraction of completed prefetches that were never demanded:
$$U_{\text{unused}} = \frac{N_{\text{prefetches\_evicted\_before\_demand}} + N_{\text{prefetches\_unused\_at\_end}}}{N_{\text{completed\_prefetches}}}$$

---

## 4. Instrumentation Implementation Locations

| Metric Counter | Existing Status | Implementation Location | Implementation Action |
| :--- | :--- | :--- | :--- |
| **`coalesced_resident_hits`** | Tracked as `unique_resident_hits` | [`linear_impl.cpp:78-98`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/linear_impl.cpp#L78-L98) | Verified active in `PrefetchMetricsCollector` |
| **`coalesced_demand_misses`** | Tracked as `unique_fresh_demand_misses` | [`device.cpp:1293`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L1293) | Verified active in `PrefetchMetricsCollector` |
| **`total_evictions`** | Written to CSV, stubbed in collector | [`prefetch_metrics_collector.cpp:342`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.cpp#L342) | Un-stub `record_hbm_evicted()`; maintain counter |
| **`evictions_by_reason`** | CSV action only | [`prefetch_metrics_collector.cpp:342`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.cpp#L342) | Record breakdown in `prefetch_summary.json` |
| **`reloaded_experts_count`** | **Missing** | [`device.h`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.h), [`device.cpp:761`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L761) | Add `previously_evicted_experts` set; increment on miss |
| **`access_frequency`** | **Missing** | [`device.h:134`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.h#L134) | Add `uint32_t access_frequency = 1;` to `ResidentExpertEntry` |
| **`last_epoch`** | **Missing** | [`device.h:134`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.h#L134) | Add `uint32_t last_epoch = 0;` for lazy decay |
| **`effective_capacity_logged`** | Existing stdout only | [`test_deepseekv3_mixtral_style.cpp:284`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L284) | Export effective cache bytes to `prefetch_summary.json` |

---

## 5. Double-Counting Prevention Invariants

To ensure complete numerical integrity:
1. **Single Increment Per Stage:** When `device->is_expert_resident()` is called for sub-projections ($W_3, W_2$), frequency and hit counters must **not** increment. Only the primary access ($W_1$) triggers state updates.
2. **Disjoint Miss Classifications:** A demand miss must be classified as exactly one of:
   - `DEMAND_MISS` (fresh DDR5 transfer initiated).
   - `LATE_IN_FLIGHT` (attached to an already running prefetch transfer).
   Under no circumstances can an access increment both counters.
3. **Conservation of Cache Residency:**
   $$\text{demand\_insertions} + \text{prefetch\_insertions} - \text{evictions} = |\text{resident\_experts}|$$
   This conservation identity must be asserted at simulation completion.

