# Comparability and Limitations: Methodological Rigor in Comparative Evaluation

## 1. Executive Summary

This report establishes the methodological boundaries for comparative evaluation in Point 5.

Cross-campaign comparisons frequently suffer from confounding variables when hardware parameters, cache capacities, or sequence lengths vary. To ensure scientific rigor, we partition all evaluations into:
1. **Category 1: Strictly Matched Comparisons** (Identical workloads, model topology, cache capacity, DDR5 memory bandwidth, and generation horizon).
2. **Category 2: Supporting Comparisons** (Evaluations isolating specific scaling trends or predictor dynamics that do not share identical baseline conditions).

---

## 2. Category 1: Strictly Matched Primary Comparisons

The primary strictly matched comparisons are derived exclusively from the **Point 4 Big Experiment** and the historical **phase_4d5c** campaign.

### Experimental Controls for Primary Primary Matched Baseline
- **Model Topology**: DeepSeek-V3 40 MoE layers ($L=40$, Top-$K=8$, 256 routed experts + 1 shared expert).
- **HBM Expert Cache Capacity**: $35\text{ GiB}$ ($37,580,963,840\text{ bytes}$, 426 expert slots).
- **DDR5 Memory Model**: Event-driven queuing at $64.0\text{ GB/s}$ peak bandwidth.
- **Trace Source**: MoE-Anywhere `e1-sage-a1.5.parquet` authentic generation traces.
- **Generation Horizon**: $T=128$ contiguous decoding tokens.
- **Workloads**:
  - Workload W3: Batch Size $B=14$ (573,440 token routing records).
  - Workload W4: Batch Size $B=32$ (1,310,720 token routing records).

### Strictly Matched Comparison Pairs

```
                      Strictly Matched Comparison Pathway
                      
  ┌────────────────────────────────────────────────────────┐
  │  Configuration A: Baseline (No Prefetch, Standard LRU)  │
  └───────────────────────────┬────────────────────────────┘
                              │
                              │ [A vs C2]
                              │ Measures: Pure effect of CSTP asynchronous prefetching
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │  Configuration C2: CSTP Prefetch + Standard LRU        │
  └───────────────────────────┬────────────────────────────┘
                              │
                              │ [C2 vs D]
                              │ Measures: Incremental effect of FA-LRU (W=64)
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │  Configuration D: CSTP Prefetch + FA-LRU (W=64)        │
  └────────────────────────────────────────────────────────┘
```

1. **Comparison A vs. C2 (Prefetching under Standard LRU)**:
   - Evaluates the effect of adding asynchronous CSTP prefetching to the baseline LRU cache.
   - Metrics: Exposed GPU stall reduction, DDR5 bus traffic inflation, and coalesced resident hit generation.
2. **Comparison C2 vs. D (Incremental Contribution of FA-LRU)**:
   - Isolates the exact effect of replacing Standard LRU with FA-LRU ($W=64$) under identical prefetching traffic.
   - Metrics: Elimination of DDR5 expert reloads, reduction in DDR5 memory traffic, avoidance of GPU stall, and eviction divergence.
3. **Comparison A vs. D (Complete System-Level Effect)**:
   - Measures the net system performance gained by deploying predictive prefetching and frequency-aware eviction together relative to the unmodified baseline.
4. **Comparison A vs. B (Baseline vs. Prediction-Only Shadow Mode)**:
   - Historical in-situ evaluation (`phase_4d5c`) establishes that running the predictor in shadow mode introduces zero speculative transfers and leaves HBM contents unchanged.
   - Evaluates CPU computational overhead and prediction latency without claiming memory-level acceleration.

---

## 3. Category 2: Supporting Comparisons

Supporting comparisons illustrate specialized architectural relationships across campaigns where one or more parameters vary:

1. **ST-MoE vs. CSTP Predictor Head-to-Head (Point 1 & Point 3)**:
   - Point 1 evaluated ST-MoE vs. CSTP on offline trace accuracy across horizons $h \in [1, 3]$ and seeds 42, 123, 999.
   - Point 3 evaluated ST-MoE vs. CSTP prefetching across 7 workloads ($B \in [14, 70]$), but utilized an 80 GB cache capacity ($74.5\text{ GiB}$, 908 slots).
   - This comparison establishes CSTP's superior bandwidth efficiency and stall reduction, but must not be directly averaged with the 35 GiB Big Experiment runs.
2. **Capacity Sweeps (Point 4 Phase 3)**:
   - Phase 3 evaluated HBM capacities from 10 GB to 35 GiB under $T=20$ tokens.
   - Demonstrates that when working sets fit comfortably in HBM, eviction policy divergence vanishes.
3. **Horizon Sensitivity (Point 3 vs. Big Experiment)**:
   - Point 3 evaluated horizons from $T=4$ to $T=64$, whereas the Big Experiment evaluated $T=128$.
   - Shows that longer generation horizons amplify both prefetch benefits and speculative cache pollution.

---

## 4. Threats to Validity and Confounding Variables

1. **Cache Capacity Disparity**:
   - Point 3 experiments operated with an 80 GB ($74.5\text{ GiB}$) cache, providing 908 expert slots.
   - Point 4 Big Experiment operated with a constrained 35 GiB cache, providing 426 expert slots.
   - Under an 80 GB cache, capacity pressure is reduced, allowing speculative prefetches to remain resident longer. Under 35 GiB, eviction pressure is severe, making eviction policy quality critical.
2. **Sequence Horizon Differences**:
   - Short runs ($T=20$) fail to establish recurring expert demand frequencies, causing FA-LRU to decay to LRU.
   - Long runs ($T=128$) provide sufficient operational time for cross-sequence frequency tracking to function.
3. **Simulation vs. Physical Hardware**:
   - All performance numbers are simulated GPU latencies derived from cycle-accurate compute and memory models.
   - They reflect architectural behavior under idealized scheduling and bus models, not physical silicon measurements.
4. **Deterministic Repetitions**:
   - Repetitions across Point 4 produced identical cycle and byte counts.
   - These are deterministic proofs of simulator correctness; they are not independent random samples, and no statistical significance is claimed.

