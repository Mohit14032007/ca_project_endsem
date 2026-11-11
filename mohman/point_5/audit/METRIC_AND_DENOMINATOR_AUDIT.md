# Metric and Denominator Audit: Formula Validation and Reconciliation

## 1. Executive Summary

This audit validates all metric formulas, denominators, unit conversions, and accounting reconciliations applied across the project.

Following the Phase 2.6 accounting correction, all metrics adhere to physical hardware invariants:
1. Demand coalescing is strictly separated from token-level routing events.
2. Tensor projections adhere to the DeepSeek-V3 3-linear layer architecture ($D_{\text{tensor}} = 3 \times D_{\text{routed}}$).
3. Memory traffic balances exactly: $\text{Total DDR5 Bytes} = \text{Demand Bytes} + \text{Speculative Bytes}$.
4. Capacity definitions are strictly disambiguated between decimal $\text{GB}$ ($10^9\text{ B}$) and binary $\text{GiB}$ ($2^{30}\text{ B}$).

---

## 2. Demand Hierarchy and Denominators

In LLMSimulator, expert access requests occur across three distinct structural levels:

```
                          Demand Event Hierarchy in DeepSeek-V3
                          
  1. Routed Token Demands (D_routed):
     Batch Size (B) × Sequence Length (T) × Layers (L) × Top-K (8)
     Example (W4, B=32, T=128, L=40): 32 × 128 × 40 × 8 = 1,310,720 routing selections
                           │
                           ▼
  2. Coalesced Actual Demands (D_coalesced):
     Unique expert activations evaluated per layer step after cross-batch coalescing.
     Example (W4): 537,451 coalesced invocations.
                           │
                           ▼
  3. Tensor Projection Events (D_tensor):
     Each routed expert invokes 3 linear transformations: Gate, Up, and Down projections.
     Invariant: D_tensor = 3 × D_routed = 3 × 1,310,720 = 3,932,160 tensor evaluations.
```

### Denominator Audit Table

| Metric | Validated Numerator | Validated Denominator | Notes & Clarifications |
| :--- | :--- | :--- | :--- |
| **Coalesced HBM Hit Rate** | `coalesced_resident_hits` | `coalesced_actual_demands` | Measures cache efficacy for actual unique memory requests. |
| **Routed Demand Hit Rate** | `routed_resident_hits` | `routed_actual_demands` | Reflects per-token hit experience across batch sequences. |
| **Predictor Coverage** | `demands_with_prediction` | `coalesced_actual_demands` | Fraction of demand experts correctly predicted in advance. |
| **Strict Timely Coverage** | `timely_resident_hits` | `coalesced_actual_demands` | Prefetch completed into HBM strictly before demand arrived. |
| **Associated Coverage** | `timely_hits + late_hits` | `coalesced_actual_demands` | Prefetch was either resident or in-flight upon demand arrival. |
| **Unused Prefetch Rate** | `unused_completed_prefetches` | `prefetches_completed` | Prefetches loaded into HBM but evicted before being demanded. |
| **Relative Latency Impr (%)** | `Latency_LRU - Latency_FALRU` | `Latency_LRU` | Positive value indicates FA-LRU reduced latency. |

---

## 3. Memory Traffic and Capacity Unit Disambiguation

### Exact Mathematical Distinctions
- **Decimal Gigabyte ($\text{GB}$)**:
  $$1\text{ GB} = 10^9\text{ bytes} = 1,000,000,000\text{ bytes}$$
- **Binary Gibibyte ($\text{GiB}$)**:
  $$1\text{ GiB} = 2^{30}\text{ bytes} = 1,073,741,824\text{ bytes}$$

### Expert Footprint Invariant
DeepSeek-V3 expert parameters:
$$\text{Parameters per Expert} = 44,040,192$$
$$\text{Bytes per Expert (FP16/BF16)} = 44,040,192 \times 2 = 88,080,384\text{ bytes} \approx 88.08\text{ MB}$$

### Cache Capacity Invariants
Under the standard $35\text{ GiB}$ evaluation capacity:
$$\text{Capacity in Bytes} = 35 \times 1,073,741,824 = 37,580,963,840\text{ bytes}$$
$$\text{Max Resident Expert Slots} = \left\lfloor \frac{37,580,963,840}{88,080,384} \right\rfloor = 426\text{ slots}$$
$$\text{Max Resident Expert Bytes} = 426 \times 88,080,384 = 37,522,243,584\text{ bytes} \approx 34.945\text{ GiB}$$
$$\text{Unusable Remainder} = 37,580,963,840 - 37,522,243,584 = 58,720,256\text{ bytes} \approx 56.0\text{ MiB}$$

---

## 4. Traffic Conservation and Reconciliation Audit

In the Big Experiment, traffic conservation was audited across all 16 runs:

### Workload W3 ($B=14, T=128$) Conservation Proof
- In Baseline (C1 / C2):
  $$\text{Total DDR5 Bytes} = 327,251\text{ demands} \times 88,080,384\text{ bytes} = 28,824,393,744,384\text{ bytes} = 28,824.39\text{ GB}$$
  $$\text{Speculative Bytes} = 0$$
- In CSTP Prefetching (C3 / C4):
  $$\text{Demand Bytes} = 28,829,414,326,272\text{ bytes} = 28,829.41\text{ GB}$$
  $$\text{Speculative Bytes} = 387,906,011,136\text{ bytes} = 387.91\text{ GB}$$
  $$\text{Total DDR5 Bytes} = 28,829,414,326,272 + 387,906,011,136 = 29,217,320,337,408\text{ bytes} = 29,217.32\text{ GB}$$
  Difference between C3 and C4:
  $$\Delta \text{DDR5 Bytes} = +880,803,840\text{ bytes} = +10 \times 88,080,384\text{ bytes} = +0.88\text{ GB}$$

### Workload W4 ($B=32, T=128$) Conservation Proof
- In Baseline (C1 / C2):
  $$\text{Total DDR5 Bytes} = 553,361\text{ demands} \times 88,080,384\text{ bytes} = 48,740,157,001,664\text{ bytes} = 48,740.16\text{ GB}$$
- In CSTP Prefetching (C3 vs C4):
  $$\text{Total DDR5 Bytes (C3)} = 48,738,840,084,480\text{ bytes} = 48,738.84\text{ GB}$$
  $$\text{Total DDR5 Bytes (C4)} = 48,736,814,235,648\text{ bytes} = 48,736.81\text{ GB}$$
  $$\Delta \text{DDR5 Bytes} = -2,025,848,832\text{ bytes} = -23 \times 88,080,384\text{ bytes} = -2.0258\text{ GB}$$
  Traffic reduction matches the 23 fewer evictions and 15 fewer expert reloads precisely.

---

## 5. Metric Ambiguity Reconciliations

1. **Host Wall-Clock Runtime vs. Simulated GPU Latency**:
   - Host wall-clock runtime (seconds) measures the CPU execution duration of the simulator process.
   - Simulated GPU latency (milliseconds) measures the simulated time required for the GPU cluster to execute the workload.
   - All performance evaluations strictly use simulated GPU latency and tokens per second.
2. **Deterministic Repetitions vs. Statistical Samples**:
   - Repetition 1 and Repetition 2 across all Point 4 experiments produced identical bit-exact values.
   - These are deterministic verification repetitions, not independent statistical samples. No p-values or standard errors are reported.

