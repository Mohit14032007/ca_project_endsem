# Extended Comparative Visual Analysis: Deconstructing MoE Inference Mitigations

## 1. Experimental Configurations

This extended visual analysis systematically examines the experimental evidence across the four conceptual configurations of our MoE memory wall mitigation architecture:

1. **Configuration A: Baseline (Demand-Only LRU)**:
   - No prediction, no speculative prefetching.
   - Experts are loaded from DDR5 across the 64.0 GB/s bus only upon an execution demand miss.
   - Cache replacement follows true Least Recently Used (LRU) timestamp ordering.
2. **Configuration B: Prediction-Only (Shadow Mode)**:
   - Predictor generates next-token expert routing forecasts.
   - Speculative DDR5 memory transfers are completely suppressed.
   - HBM cache residency and DDR5 memory traffic remain mathematically identical to Configuration A.
3. **Configuration C: Prediction-Driven Prefetching (CSTP + LRU)**:
   - CSTP predictor issues speculative prefetch requests to an asynchronous DDR5 queue.
   - Standard LRU manages HBM cache replacement.
   - Unconsumed speculative prefetches enter HBM and induce cache pollution.
4. **Configuration D: Complete Implementation (CSTP + FA-LRU $W=64$)**:
   - CSTP predictor issues speculative prefetch requests.
   - Frequency-Aware LRU (FA-LRU) evaluates a candidate window of $W=64$ entries at the LRU tail.
   - Demand-only frequency tracking selectively evicts cold speculative prefetches ($f=1$) while shielding mature demand lines ($f \ge 2$).

The primary strictly matched evaluation is conducted on **Workload W4** ($B=32, T=128, L=40, \text{Top-}K=8, M_{\text{HBM}}=35\text{ GiB}$, 426 expert slots) using authentic MoE-Anywhere `e1-sage-a1.5.parquet` traces. Workload W3 ($B=14$) and supporting Point 3 datasets are evaluated to analyze concurrency scaling.

---

## 2. Predictor Effectiveness (Figure 9)

Figure 9 evaluates the offline predictive accuracy of ST-MoE versus CSTP on the full MoE-Anywhere Sage-A1.5 dataset across prediction horizons $h=1$ and $h=2$.

![Figure 9: Predictor Quality Comparison](file:///home/mohman/ca_project/mohman/point_5/figures/fig9_predictor_quality_comparison.png)

### Data Table: Predictor Quality Metrics (Point 1 Dataset)

| Predictor Configuration | Horizon | Macro Precision (%) | Macro Recall (%) | Macro F1 (%) | Macro Top-K Hit Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ST-MoE** | $h=1$ | 47.91% | 47.91% | 47.91% | **96.94%** |
| **CSTP** | $h=1$ | 45.51% | 45.51% | 45.51% | 96.67% |
| **ST-MoE** | $h=2$ | 46.20% | 46.20% | 46.20% | 95.85% |
| **CSTP** | $h=2$ | 44.18% | 44.18% | 44.18% | 96.14% |

*Source: `mohman/point_5/extracted_data/predictor_quality_comparison.csv` (16 validated records).*

### Six-Point Architectural Explanation
1. **What the graph measures**: Exact-set macro precision, recall, and F1 score against ground-truth routing decisions, alongside Top-$K$ hit rate (fraction of tokens where at least one predicted expert is activated).
2. **The observed result**: Both algorithms achieve near-universal Top-$K$ hit rates ($> 95\%$), while exact set recall remains between $44\%$ and $48\%$. ST-MoE exhibits a $+2.40\%$ advantage in exact recall at $h=1$.
3. **Why the result occurs architecturally**: ST-MoE tracks token-to-token transition probabilities within individual sequence contexts, capturing immediate Markovian dependencies. CSTP tracks cross-sequence co-occurrence, slightly trading single-token precision for global popular cluster coverage.
4. **What component contributes to the result**: Component 1 (Predictor Algorithms).
5. **Important limitations**: A $96\%$ Top-$K$ hit rate does not mean all 8 expert predictions are correct; it means at least one prediction succeeded. Exact-set recall reflects the true fraction of demanded experts identified.
6. **Presentation takeaway**: *MoE expert routing is highly structured; lightweight predictors reliably forecast at least one active expert for over 96% of tokens.*

---

## 3. Prediction-Only Contribution (Figure 5 & Historical Evidence)

Figure 5 compares end-to-end execution time across the four configurations on Workload W4 ($B=32, T=128$).

![Figure 5: End-to-End Execution Time Comparison](file:///home/mohman/ca_project/mohman/point_5/figures/fig5_execution_time_comparison.png)

### Data Table: Execution Time Comparison (Workloads W4 and W3, T=128, M=35 GiB)

| Workload | Configuration | Simulated Latency (s) | Normalized Time (%) | Improvement vs. Baseline (%) | Empirical Classification |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **W4 ($B=32$)** | **Config A: Baseline** | 1,008.204 s | 100.000% | 0.000% | Measured (`w4_c1_r1`) |
| **W4 ($B=32$)** | **Config B: Prediction-Only** | 1,008.204 s | 100.000% | 0.000% | Inferred shadow-mode equivalence |
| **W4 ($B=32$)** | **Config C2: CSTP + LRU** | 761.608 s | 75.541% | **+24.459%** | Measured (`w4_c3_r1`) |
| **W4 ($B=32$)** | **Config D: Complete System** | **761.576 s** | **75.538%** | **+24.462%** | Measured (`w4_c4_r1`) |
| **W3 ($B=14$)** | **Config A: Baseline** | 459.031 s | 100.000% | 0.000% | Measured (`w3_c1_r1`) |
| **W3 ($B=14$)** | **Config B: Prediction-Only** | 459.031 s | 100.000% | 0.000% | Inferred shadow-mode equivalence |
| **W3 ($B=14$)** | **Config C2: CSTP + LRU** | 456.368 s | 99.420% | **+0.580%** | Measured (`w3_c3_r1`) |
| **W3 ($B=14$)** | **Config D: Complete System** | **456.383 s** | **99.423%** | **+0.577%** | Measured (`w3_c4_r1`) |

*Source: `mohman/point_5/extracted_data/execution_time_comparison.csv` (16 validated records).*

### Six-Point Architectural Explanation
1. **What the graph measures**: Total simulated GPU execution time in seconds, normalized percentage relative to baseline, and percentage speedup.
2. **The observed result**: In W4 ($B=32$), Configuration B achieves identical simulated GPU latency as Baseline A (1,008.204 s), while Configuration C2 reduces runtime to 761.608 s and Configuration D reduces runtime to 761.576 s. In W3 ($B=14$), runtime decreases from 459.031 s to 456.368 s (C2) and 456.383 s (D).
3. **Why the result occurs architecturally**: In shadow mode (Config B), predictions are computed by CSTP incurring $62.74\text{ }\mu\text{s}$ CPU host overhead ($0.25\%$), but no speculative DDR5 transfers are dispatched to the GPU memory system. Because DDR5 bus transactions and HBM residency are bit-identical to Baseline A, simulated GPU execution time is unchanged ($0.00\%$ delta). Prefetching (Config C2 and D) initiates asynchronous transfers, successfully overlapping transfer latency with GPU compute.
4. **What component contributes to the result**: Component 1 (enabler) vs. Component 2 (asynchronous prefetching execution).
5. **Important limitations**: Configuration B's performance is an inferred shadow-mode equivalence based on historical `phase_4d5c` verification where shadow mode demonstrated zero device timing deviation.
6. **Presentation takeaway**: *Prediction without memory data movement provides zero hardware acceleration; execution speedup requires asynchronous prefetching.*

---

## 4. Asynchronous Prefetching Contribution (Figure 6)

Figure 6 analyzes exposed GPU memory stall across the configurations under Workload W4.

![Figure 6: Exposed Memory Stall Comparison](file:///home/mohman/ca_project/mohman/point_5/figures/fig6_exposed_stall_comparison.png)

### Data Table: Exposed GPU Memory Stall (Workload W4, B=32, T=128)

| Configuration | Exposed Memory Stall (ms) | Stall Reduction (ms) | Stall Reduction (%) | Transition Contribution |
| :--- | :---: | :---: | :---: | :--- |
| **Config A: Baseline** | 994,459.73 ms | 0.00 ms | 0.00% | Reference point |
| **Config B: Prediction-Only** | 994,459.73 ms | 0.00 ms | 0.00% | Zero stall impact |
| **Config C2: CSTP + LRU** | 747,863.57 ms | **-246,596.16 ms** | **-24.797%** | Transition A -> C (Prefetching) |
| **Config D: Complete System** | **747,831.75 ms** | **-246,627.98 ms** | **-24.800%** | Transition C -> D (FA-LRU: -31.82 ms) |

*Source: `mohman/point_5/extracted_data/exposed_stall_comparison.csv` (9 validated records).*

### Six-Point Architectural Explanation
1. **What the graph measures**: Total exposed GPU memory stall (time GPU compute warps spend stalled awaiting DDR5 expert arrival) and percentage stall reduction.
2. **The observed result**: Prefetching eliminates **246,596.16 ms (246.60 seconds, or 4.11 minutes, representing a 24.80% reduction relative to baseline)** of GPU stall. FA-LRU provides an additional 31.82 ms stall savings.
3. **Why the result occurs architecturally**: Moving expert parameters asynchronously while the GPU computes prior layers allows transfers to complete before demand execution occurs. When demand arrives, the expert is either resident (timely hit) or already traversing the bus (late attachment), drastically reducing stall.
4. **What component contributes to the result**: Component 2 (Asynchronous Prefetch Engine) accounts for **99.987%** of stall reduction; Component 3 (FA-LRU) contributes **0.013%**.
5. **Important limitations**: Total DDR5 transfer time is much larger than exposed stall; stall measures only the exposed critical-path wait time.
6. **Presentation takeaway**: *Asynchronous prefetching eliminates 246.60 seconds (4.11 minutes) of exposed memory stall, collapsing GPU memory idle time by 24.80%.*

---

## 5. Memory Traffic Trade-Off (Figure 7)

Figure 7 examines DDR5 memory traffic composition, decomposing total traffic into demand transfers and speculative prefetch transfers for Workloads W3 and W4.

![Figure 7: DDR5 Memory Traffic Composition](file:///home/mohman/ca_project/mohman/point_5/figures/fig7_ddr5_traffic_composition.png)

### Data Table: DDR5 Memory Traffic Decomposition

| Workload | Configuration | Demand Traffic (GB) | Speculative Traffic (GB) | Total DDR5 Traffic (GB) | Net Traffic Delta vs. Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **W3 ($B=14$)** | Config A: Baseline | 28,824.39 GB | 0.00 GB | 28,824.39 GB | Reference |
| **W3 ($B=14$)** | Config C2: CSTP + LRU | 21,972.27 GB | 7,245.05 GB | 29,217.32 GB | +392.93 GB (+1.36%) |
| **W3 ($B=14$)** | Config D: Complete | 21,978.17 GB | 7,240.03 GB | 29,218.20 GB | +393.81 GB (+1.37%) |
| **W4 ($B=32$)** | Config A: Baseline | 48,740.16 GB | 0.00 GB | 48,740.16 GB | Reference |
| **W4 ($B=32$)** | Config C2: CSTP + LRU | 37,790.54 GB | 10,948.30 GB | 48,738.84 GB | **-1.32 GB (-0.003%)** |
| **W4 ($B=32$)** | Config D: Complete | 37,808.77 GB | 10,928.05 GB | **48,736.81 GB** | **-3.35 GB (-0.007%)** |

*Source: `mohman/point_5/extracted_data/ddr5_traffic_composition.csv` (21 validated records).*

### Six-Point Architectural Explanation
1. **What the graph measures**: Total DDR5 memory traffic partitioned into demand transfer bytes (miss servicing) and speculative prefetch transfer bytes.
2. **The observed result**: In W3, speculative prefetching increases total bus traffic by $+1.36\%$. In W4, speculative prefetching decreases total bus traffic by $-1.32\text{ GB}$, and FA-LRU reduces it further by $-2.0258\text{ GB}$ ($-3.35\text{ GB}$ net savings relative to baseline).
3. **Why the result occurs architecturally**: Speculative prefetches consume bus bandwidth. Under FA-LRU, 5,999 unconsumed speculative prefetches are purged from the bottom-$W=64$ candidate window, reducing speculative traffic by $-20.2585\text{ GB}$ ($-230\text{ blocks}$). Concurrently, fewer speculative lines occupying the cache shifts 207 expert transfers to direct demand transfers ($+18.2326\text{ GB}$). Because 36 protected demand lines are reused in HBM, 15 complete expert reloads are avoided. The net bus traffic savings is the superposition of both effects: $-20.2585\text{ GB} + 18.2326\text{ GB} = \mathbf{-2.0258\text{ GB}}$ (exactly $-23\text{ blocks}$ of $88\text{ MB}$).
4. **What component contributes to the result**: Component 2 (Prefetching) introduces speculative bytes; Component 3 (FA-LRU) actively purges speculative pollution and protects high-frequency demand lines.
5. **Important limitations**: At lower concurrency ($B=14$), cross-prompt expert re-use is sparse, so speculative traffic is not fully offset by avoided reloads (+1.36% traffic).
6. **Presentation takeaway**: *FA-LRU achieves traffic neutrality by pruning speculative lines (-20.26 GB), fully paying for the net traffic balance under batch concurrency B=32.*

---

## 6. HBM Cache Behavior (Figure 8)

Figure 8 evaluates resident HBM hits and DDR5 expert reloads across Workloads W3 and W4 under the corrected Phase 2.6 demand accounting.

![Figure 8: HBM Hits and Expert Reloads](file:///home/mohman/ca_project/mohman/point_5/figures/fig8_hbm_hits_reloads.png)

### Data Table: HBM Hits and Expert Reloads Comparison

| Workload | Policy Configuration | Coalesced Demands | Resident Hits | Hit Rate (%) | Expert Reloads | Reload Delta vs. LRU |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **W3 ($B=14$)** | Config A: Baseline | 327,251 | 0 | 0.000% | 317,775 | Reference |
| **W3 ($B=14$)** | Config C2: CSTP + LRU | 327,251 | 1,657 | 0.5063% | 316,227 | -1,548 vs Baseline |
| **W3 ($B=14$)** | Config D: Complete | 327,251 | 1,643 | 0.5021% | 316,241 | +14 vs LRU |
| **W4 ($B=32$)** | Config A: Baseline | 536,361 | 0 | 0.000% | 537,137 | Reference |
| **W4 ($B=32$)** | Config C2: CSTP + LRU | 536,361 | 1,417 | 0.2642% | 536,034 | -1,103 vs Baseline |
| **W4 ($B=32$)** | Config D: Complete | 536,361 | **1,432** | **0.2670%** | **536,019** | **-15 vs LRU** |

*Source: `mohman/point_5/extracted_data/hbm_hit_reload_comparison.csv` (18 validated records).*

### Six-Point Architectural Explanation
1. **What the graph measures**: Coalesced resident HBM hits (demands satisfied from cache without DDR5 request) and expert reloads from DDR5. Denominator: `coalesced_actual_demands`.
2. **The observed result**: Pure demand baseline achieves 0 hits due to capacity overflow. Prefetching produces 1,657 hits in W3 and 1,417 hits in W4. In W4, FA-LRU adds $+15$ hits and eliminates 15 expert reloads. In W3, FA-LRU loses 14 hits.
3. **Why the result occurs architecturally**: The 40-layer working set exceeds the 426-slot cache, causing sequential sweeps to flush lines before re-use. Prefetching deposits lines just in time for access. Under high concurrency (W4), cross-prompt overlap allows FA-LRU to protect reused demand lines, avoiding reloads.
4. **What component contributes to the result**: Component 2 (Prefetching) establishes base hits; Component 3 (FA-LRU) modulates hit retention based on concurrency.
5. **Important limitations**: Hit rates appear low ($< 1\%$) because the denominator represents hundreds of thousands of coalesced layer requests across 128 decode steps.
6. **Presentation takeaway**: *Prefetching creates resident hits where demand LRU experiences 100% turnover; FA-LRU improves hit retention when batch concurrency reaches B=32.*

---

## 7. Prefetch Effectiveness and Cache Pollution (Figure 10)

Figure 10 analyzes prefetch dispatch accuracy, timely resident hits, late in-flight attachments, and speculative cache pollution using Point 3 large-workload data.

![Figure 10: Prefetch Effectiveness and Pollution](file:///home/mohman/ca_project/mohman/point_5/figures/fig10_prefetch_effectiveness_pollution.png)

### Data Table: Prefetch Accounting and Cache Pollution (Point 3 Workloads)

| Workload | Predictor | Dispatched Prefetches | Late In-Flight Attachments | Unused Completed Prefetches | Unused Prefetch Rate (%) | Total Evictions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **W6 ($B=70, T=32$)** | ST-MoE ($h=2$) | 35,272 | 32,817 | 2,388 | 6.78% | 136,898 |
| **W6 ($B=70, T=32$)** | CSTP ($h=2$) | 17,659 | 16,002 | 1,656 | 9.38% | 135,713 |
| **W7 ($B=70, T=64$)** | ST-MoE ($h=2$) | 102,737 | 96,569 | 6,022 | 5.87% | 352,857 |
| **W7 ($B=70, T=64$)** | CSTP ($h=2$) | 54,076 | 50,264 | 3,812 | 7.05% | 349,588 |

*Source: `mohman/point_5/extracted_data/prefetch_effectiveness_comparison.csv` (14 validated records).*

### Six-Point Architectural Explanation
1. **What the graph measures**: Dispatched prefetches, late in-flight attachments, unused completed prefetches (prefetches loaded but evicted before demand), and total evictions.
2. **The observed result**: Over $90\%$ of prefetches in heavy workloads operate as late in-flight attachments rather than strict timely hits. CSTP generates $47\%$ fewer dispatches and produces $2,210$ fewer unused prefetches than ST-MoE in W7.
3. **Why the result occurs architecturally**: Under constrained DDR5 bus bandwidth, prefetch transfers queue behind prior transfers. When demand arrives, the transfer is already underway, allowing the execution thread to attach and avoid a duplicate miss. Unused prefetches represent speculative errors that occupy HBM and force premature eviction of useful experts.
4. **What component contributes to the result**: Component 2 (Prefetch Transfer Engine) and Component 1 (Predictor Dispatch Discipline).
5. **Important limitations**: Late attachments do not hide 100% of transfer latency, but they eliminate bus re-request overhead and drastically reduce wait time.
6. **Presentation takeaway**: *CSTP cuts speculative cache pollution by 36.7% relative to ST-MoE while achieving greater stall reduction.*

---

## 8. FA-LRU Contribution Across Serving Concurrency (Figure 11)

Figure 11 isolates the incremental contribution of FA-LRU ($W=64$) relative to Standard LRU under CSTP prefetching across Workloads W3 and W4.

![Figure 11: FA-LRU Incremental Contribution](file:///home/mohman/ca_project/mohman/point_5/figures/fig11_falru_incremental_contribution.png)

### Data Table: FA-LRU Incremental Physical Savings (Workload W4, B=32, T=128)

| Metric Category | Metric | CSTP + Standard LRU | CSTP + FA-LRU (W=64) | Incremental Delta (Savings) | Physical Unit |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Physical Hardware** | **Coalesced Resident Hits** | 1,417 | 1,432 | **+15 hits** | Count ($+1.06\%$) |
| **Physical Hardware** | **Expert Reloads from DDR5** | 536,034 | 536,019 | **-15 reloads** | Count (avoided) |
| **Physical Hardware** | **Total DDR5 Memory Traffic** | 48,738.84 GB | 48,736.81 GB | **-2.0258 GB** | decimal GB (saved) |
| **Physical Hardware** | **Exposed GPU Memory Stall** | 747,863.57 ms | 747,831.75 ms | **-31.82 ms** | milliseconds (saved) |
| **Physical Hardware** | **Simulated Execution Time** | 761,607.55 ms | 761,575.89 ms | **-31.65 ms** | milliseconds (saved) |
| **Physical Hardware** | **Mean Token Latency** | 5,950.06 ms | 5,949.81 ms | **-0.25 ms** | ms/token ($+0.0042\%$) |
| **Algorithmic Diagnostic** | **Divergent Evictions** | — | 431,909 | **78.12% of total** | Eviction decisions |
| **Algorithmic Diagnostic** | **Speculative Prefetches Evicted** | — | 5,999 | Pruned cold entries | Cache lines |
| **Algorithmic Diagnostic** | **Shielded Demand Lines Reused** | — | 36 | **Direct reuse hits** | Converted misses |

*Source: `mohman/point_5/extracted_data/falru_incremental_comparison.csv` (13 validated records).*

### Six-Point Architectural Explanation
1. **What the graph measures**: Coalesced resident hits for W3 and W4, and incremental physical savings (traffic, stall, reloads, and protected reuse) for W4.
2. **The observed result**: In W4 ($B=32$), FA-LRU saves $2.0258\text{ GB}$ DDR5 traffic, avoids 15 expert reloads, and reduces stall by $31.82\text{ ms}$. In W3 ($B=14$), FA-LRU loses 14 resident hits.
3. **Why the result occurs architecturally**: FA-LRU scans the bottom $W=64$ entries at the LRU tail. In W4, cross-sequence re-use is dense; FA-LRU evicts unconsumed speculative prefetches ($f=1$) to protect demand lines ($f=2$). 36 protected lines are reused, averting DDR5 reloads. In W3, inter-prompt re-use is sparse, so protecting stale lines causes minor cache churn.
4. **What component contributes to the result**: Component 3 (Frequency-Aware HBM Management).
5. **Important limitations**: Diagnostic counters must not be confused with physical hardware metrics. Evicting 5,999 unconsumed speculative prefetches is an internal candidate purging metric, not 5,999 avoided demand misses. The physical hardware observables are exactly +15 resident hits, 15 avoided reloads, -2.0258 GB traffic, and -31.82 ms stall.
6. **Presentation takeaway**: *FA-LRU functions as a speculative pollution filter that actively purges cold prefetches, saving 2.03 GB of DDR5 traffic and 15 reloads under batch concurrency B=32.*

---

## 9. Batch-Size Scaling Comparison (Figure 12)

Figure 12 presents concurrency scaling across Point 3 ($B \in [14, 70]$ at $T=16$) and the Big Experiment ($B \in \{14, 32\}$ at $T=128$).

![Figure 12: Batch-Size Scaling Comparison](file:///home/mohman/ca_project/mohman/point_5/figures/fig12_batch_scaling_comparison.png)

### Data Table: Concurrency Scaling Across Campaigns

| Campaign | Workload | Batch Size | Horizon ($T$) | Baseline Execution / Latency | CSTP + LRU Execution / Latency | Complete (FA-LRU) Execution / Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Point 3** | W2 | 14 | 16 | 18,117.03 ms stall | 18,055.42 ms stall | — |
| **Point 3** | W3 | 32 | 16 | 33,639.29 ms stall | 33,529.28 ms stall | — |
| **Point 3** | W4 | 48 | 16 | 50,768.71 ms stall | 50,566.14 ms stall | — |
| **Point 3** | W5 | 70 | 16 | 62,304.49 ms stall | 61,794.55 ms stall | — |
| **Big Exp** | W3 | 14 | 128 | **459.031 s** (3,586.18 ms/tok) | **456.368 s** (3,565.37 ms/tok) | **456.383 s** (3,565.49 ms/tok) |
| **Big Exp** | W4 | 32 | 128 | **1,008.204 s** (7,876.59 ms/tok) | **761.608 s** (5,950.06 ms/tok) | **761.576 s** (**5,949.81 ms/tok**) |

*Source: `mohman/point_5/extracted_data/batch_scaling_comparison.csv` (24 validated records).*

### Six-Point Architectural Explanation
1. **What the graph measures**: Scaling of exposed GPU memory stall across batch sizes 14 to 70 (Point 3) and mean token latency across batch sizes 14 and 32 (Big Experiment). Full simulated execution times for Big Experiment are also presented, obeying the algebraic identity $\bar{L}_{\text{token}} = \frac{T_{\text{exec}}}{128}$.
2. **The observed result**: In Point 3, stall reduction expands from $61.6\text{ ms}$ at $B=14$ to $509.9\text{ ms}$ at $B=70$. In the Big Experiment, prefetch token latency savings expand from $-20.8\text{ ms}$ at $B=14$ to **$-1,926.8\text{ ms}$ at $B=32$** ($24.5\%$ speedup, saving $246.60\text{ s}$).
3. **Why the result occurs architecturally**: Higher batch concurrency increases the total volume of computation per layer step, providing longer execution windows to overlap DDR5 prefetch transfers. Furthermore, higher concurrency increases cross-sequence expert overlap, magnifying the effectiveness of both CSTP and FA-LRU.
4. **What component contributes to the result**: Component 2 (Prefetching) and Component 3 (FA-LRU).
5. **Important limitations**: Point 3 and Big Experiment utilize different cache sizes (80 GB vs. 35 GiB) and sequence horizons ($T=16$ vs. $T=128$); they represent separate panels and must not be connected continuously.
6. **Presentation takeaway**: *Prefetching benefits scale super-linearly with batch concurrency, expanding from minor gains at B=14 to a 24.5% latency reduction at B=32.*

---

## 10. Complete System Performance (Figure 4 Revisited)

Figure 4 unifies the four conceptual configurations on Workload W4, showing normalized execution time and total memory traffic.

![Figure 4: Overall System Comparison](file:///home/mohman/ca_project/mohman/point_5/figures/fig4_overall_system_comparison.png)

### Master System Performance Summary

```
========================================================================================
MASTER COMPARATIVE EVALUATION MATRIX (Workload W4: B=32, T=128, M_HBM=35 GiB, 426 slots)
========================================================================================
Configuration                    Simulated Time (s)   Exposed Stall (ms)   DDR5 GB   Hits
----------------------------------------------------------------------------------------
Config A: Baseline                         1,008.20           994,459.73  48,740.16     0
Config B: Prediction-Only                  1,008.20           994,459.73  48,740.16     0
Config C2: CSTP + LRU                        761.61           747,863.57  48,738.84 1,417
Config D: Complete (FA-LRU W=64)             761.58           747,831.75  48,736.81 1,432
----------------------------------------------------------------------------------------
Net System Improvement (A -> D)            -246.62 s       -246,627.98 ms  -3.35 GB +1,432
Percentage Improvement                       -24.46%              -24.80%   -0.007%   +∞
========================================================================================
```

---

## 11. Component Attribution Breakdown

1. **Prediction (Component 1)**:
   - Contributes **0.00%** direct hardware speedup on its own.
   - Provides **100% of the foresight** enabling asynchronous prefetch scheduling.
   - Overhead: $62.74\text{ }\mu\text{s}$ latency ($0.25\%$ CPU host overhead, zero simulated GPU device impact).
2. **Asynchronous Prefetching (Component 2)**:
   - Contributes **99.987%** of the net execution latency reduction ($-246.60\text{ seconds}$).
   - Contributes **99.987%** of exposed memory stall reduction ($-246,596.16\text{ ms}$).
   - Generates speculative cache pollution (5,999 unconsumed prefetch entries in W4).
3. **Frequency-Aware Cache Management (Component 3)**:
   - Contributes **0.013%** of net execution latency reduction ($-0.032\text{ seconds}$ / $-31.82\text{ ms}$ stall).
   - Contributes **100% of speculative cache pollution mitigation** (evicts 5,999 unconsumed prefetches to protect demand lines).
   - Saves **$2.0258\text{ GB}$ of DDR5 memory traffic** (net superposition of $-20.26\text{ GB}$ speculative relief and $+18.23\text{ GB}$ demand shift) and eliminates 15 expert reloads.
4. **Complete Implementation (Component 4)**:
   - Achieves a **$24.46\%$ net reduction in execution latency** while maintaining DDR5 memory traffic neutrality.

---

## 12. Methodological Limitations and Threats to Validity

1. **Deterministic Repetitions**: Repetition 1 and 2 across all Point 4 experiments produced identical bit-exact cycle, byte, and event counts. These repetitions confirm simulation determinism; they are not independent random samples, and no statistical significance is claimed.
2. **Simulation Model**: All reported latencies reflect simulated GPU compute and memory cycle times generated by LLMSimulator, not physical A100 GPU measurements.
3. **Cache Capacity Sensitivity**: FA-LRU benefits are observed under constrained cache capacity ($35\text{ GiB}$, 426 slots). In oversized caches ($80\text{ GB}$), capacity pressure is insufficient to trigger frequency-aware victim selection.
4. **Concurrency Thresholding**: FA-LRU requires sufficient batch concurrency ($B \ge 32$) to establish cross-prompt expert re-use. At low concurrency ($B \le 14$), standard LRU is recommended.

---

## 13. Final Conclusions

1. **Prediction alone does not alleviate the memory wall**: Without asynchronous data movement, forecasting future expert routes leaves execution stalls and HBM hit rates unchanged.
2. **Asynchronous prefetching is the primary driver of execution speedup**: By overlapping DDR5 memory loads with GPU compute execution, prefetching reduces exposed memory stall by $246.60\text{ seconds}$ / $4.11\text{ minutes}$ ($24.80\%$).
3. **CSTP is superior to ST-MoE for prefetching**: CSTP achieves $2\times$ to $5\times$ greater stall reduction while generating $56.9\%$ less memory traffic overhead.
4. **FA-LRU ($W=64$) effectively mitigates speculative cache pollution**: Under batch concurrency $B=32$, FA-LRU selectively purges cold speculative prefetches, recovering 15 resident hits, saving $2.0258\text{ GB}$ of DDR5 traffic, and reducing exposed stall by $31.82\text{ ms}$.
5. **The complete implementation achieves balanced memory wall mitigation**: Pairing CSTP prefetching with FA-LRU cache management delivers a $24.46\%$ net speedup while actively protecting the HBM cache against speculative pollution.

