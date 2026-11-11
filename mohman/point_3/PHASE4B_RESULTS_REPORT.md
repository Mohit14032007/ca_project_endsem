# Phase 4B Final Report: Prefetching Effectiveness Evaluation & Architectural Analysis

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Deliverable Path:** [PHASE4B_RESULTS_REPORT.md](file:///home/mohman/ca_project/mohman/point_3/PHASE4B_RESULTS_REPORT.md)  
**Supporting Artifacts:**
- Initial Effectiveness Audit: [PHASE4B_EFFECTIVENESS_AUDIT.md](file:///home/mohman/ca_project/mohman/point_3/PHASE4B_EFFECTIVENESS_AUDIT.md)
- Instrumentation Report: [PHASE4B_INSTRUMENTATION_REPORT.md](file:///home/mohman/ca_project/mohman/point_3/PHASE4B_INSTRUMENTATION_REPORT.md)
- Experiment Manifest: [experiment_manifest.csv](file:///home/mohman/ca_project/mohman/point_3/experiment_manifest.csv)
- Reproducibility Guide: [reproducibility_manifest.md](file:///home/mohman/ca_project/mohman/point_3/reproducibility_manifest.md)
- Core Consolidated Metrics: [consolidated_metrics.csv](file:///home/mohman/ca_project/mohman/point_3/reports/consolidated_metrics.csv)
- Sensitivity Metrics: [sensitivity_metrics.csv](file:///home/mohman/ca_project/mohman/point_3/reports/sensitivity_metrics.csv)

---

## 1. Executive Summary

This report delivers the empirical and architectural evaluation of predictor-driven expert prefetching for Mixture-of-Experts (MoE) inference in [LLMSimulator](file:///home/mohman/ca_project/LLMSimulator). We evaluated three operating paradigms across three routing trace corpuses under a cycle-level, event-driven DDR5 memory model with active Ramulator timing:

1. **No-Prefetch Baseline:** Demand-driven off-chip loading over a $64.0$ GB/s DDR5 bus into an $80$ GB HBM cache.
2. **ST-MoE Predictor:** Sequence-to-expert Markov transition predictor ($H=1$ and $H=2$, Budget $B=4$).
3. **CSTP Predictor:** Contextual and Staged Transition Predictor with temporal smoothing and online frequency adaptation ($H=1$ and $H=2$, Budget $B=4$).

### Core Findings & Headline Metrics

- **Resolution of Projection-Level vs Unique-Expert Hit Rates:**  
  In previous evaluations, baseline cold-start simulations exhibited an apparent HBM hit rate of $\approx 66.67\%$ to $70\%$. Our dual-granularity instrumentation proves that this was an artifact of linear projection conflation: each expert forward pass evaluates three distinct projections ($W_1$ Gate, $W_3$ Up, $W_2$ Down). When $W_1$ experiences a cold demand miss, the device loads the full $88.08$ MB expert parameter bundle into HBM, causing $W_3$ and $W_2$ to register as immediate HBM hits. In our certified core evaluation ($B=1, \text{iter}=4$), projection-level hit rate is $75.52\%$, whereas the true unique-expert HBM hit rate is $26.56\%$ ($17$ unique hits out of $64$ unique expert invocations).
- **Bus Contention and Speculative Transfer Priority:**  
  Because DDR5 transfer queues assign demand transfers priority $1$ and speculative prefetch transfers priority $0$, cold-start demand bursts monopolize the DDR5 bus. At $64.0$ GB/s, transferring $16$ cold demand experts ($8$ per layer $\times 2$ layers) requires $22.02$ ms of contiguous DDR5 service time. Consequently, speculative transfers dispatched during Step 0 cannot begin service until the bus clears, pushing prefetch completion past the demand deadline of Step 0.
- **Intra-Token Horizon Boundary Invariant:**  
  In a 5-layer model with $2$ MoE layers, intra-token lookahead with $H=1$ bridges Stage 0 to Stage 1. However, $H=2$ attempts to predict Stage 2 ($0 + 2 = 2$). Because no third MoE stage exists within the token, $H=2$ correctly emits $0$ speculative candidates, automatically conserving DDR5 bandwidth.
- **Batch Scaling Throughput Gains:**  
  As batch size increases from $B=1$ to $B=4$, sequence-level parallelism amortizes GPU kernel launch overhead, increasing simulated decoding throughput from $54.70$ tokens/sec ($B=1$) to $112.79$ tokens/sec ($B=4$), while memory demand traffic scales from $4.14$ GB to $8.46$ GB.
- **Cache Constrained Behavior:**  
  When HBM cache capacity is restricted from unconstrained ($80$ GB) to constrained ($20$ experts $\approx 1.76$ GB), cache thrashing forces $7$ additional demand misses ($54$ vs $47$), increasing demand traffic from $4.14$ GB to $4.76$ GB and exposed stall time from $64.68$ ms to $74.32$ ms.

---

## 2. Experimental Methodology & Architectural Invariants

### 2.1 Hardware and Simulation Platform

All experiments were executed on the certified simulator binary [run_deepseekv3_mixtral_style](file:///home/mohman/ca_project/LLMSimulator/build/run_deepseekv3_mixtral_style) with the following hardware topology:

| Parameter | Configuration Value | Physical Significance |
| :--- | :--- | :--- |
| **GPU Architecture** | NVIDIA A100-SXM4-80GB | Standard datacenter inference accelerator |
| **HBM Local Cache** | 80 GB ($85,899,345,920$ bytes) | High-bandwidth on-chip expert storage |
| **Backing Store** | DDR5-4800 ($64.0$ GB/s) | Host memory channel backing offloaded experts |
| **Addressable Rows** | $10,485,760$ rows ($2.75$ TB) | Total capacity for DeepSeek-V3 inventory |
| **Model Architecture** | DeepSeek-V3 Mixtral-Style | 5 layers total (3 dense, 2 MoE stages) |
| **Routed Experts** | 256 per layer, Top-8 routed | High sparsity factor ($8 / 256 = 3.125\%$) |
| **Expert Footprint** | $88,080,384$ bytes ($\approx 88.08$ MB) | FP16 weights for $W_1$ (Gate), $W_3$ (Up), $W_2$ (Down) |
| **Single Expert Transfer** | $1,376,256$ ns ($\approx 1.376$ ms) | Raw transfer time at $64.0$ GB/s peak bus rate |
| **Simulation Timing** | Event-Driven DRAM + Ramulator | Hardware queueing, bus arbitration, and stall accounting |

### 2.2 Trace Datasets Evaluated

Three distinct trace corpuses were ingested and replayed:
1. **MoE-Anywhere Real Trace:** Traced from production MoE workloads, characterized by moderate temporal locality and skewed expert popularity.
2. **MoE-Suite Task 12:** Real multi-task routing trace with high routing dispersion across $256$ experts.
3. **MoE-Suite Task 13:** Real reasoning routing trace exhibiting distinct token-to-token transition dynamics.

---

## 3. Primary Five-Configuration Core Comparison

The primary evaluation compared five configurations executing on the real MoE-Anywhere trace with batch size $B=1$ across $4$ sequential decoding iterations ($64$ total unique expert invocations):

1. **Baseline:** No prefetching; all un-cached experts demand-loaded on miss.
2. **ST-MoE ($H=1$):** Intra-token 1-step lookahead, candidate budget $B=4$.
3. **ST-MoE ($H=2$):** Intra-token 2-step lookahead, candidate budget $B=4$.
4. **CSTP ($H=1$):** Intra-token 1-step lookahead, candidate budget $B=4$.
5. **CSTP ($H=2$):** Intra-token 2-step lookahead, candidate budget $B=4$.

### 3.1 Consolidated Core Metrics Table

The table below presents the certified measurements extracted from [consolidated_metrics.csv](file:///home/mohman/ca_project/mohman/point_3/reports/consolidated_metrics.csv):

| Metric # | Effectiveness Metric | Baseline (Demand) | ST-MoE ($H=1$) | ST-MoE ($H=2$) | CSTP ($H=1$) | CSTP ($H=2$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Projection Demands ($D_{\text{proj}}$)** | 192 | 192 | 192 | 192 | 192 |
| | **Unique Expert Demands ($D_{\text{unique}}$)** | 64 | 64 | 64 | 64 | 64 |
| | **Projection Coverage** | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| | **Unique Expert Coverage** | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **2** | **Dispatched Prefetches** | 0 | 4 | 0 | 4 | 0 |
| | **Timely Hits (Unique)** | 0 | 0 | 0 | 0 | 0 |
| | **Late Attachments (Unique)** | 0 | 0 | 0 | 0 | 0 |
| **3** | **Completed Unused Prefetches** | 0 | 4 | 0 | 4 | 0 |
| | **Wasted Speculative Traffic** | 0.00 MB | 336.00 MB | 0.00 MB | 336.00 MB | 0.00 MB |
| **4** | **Projection HBM Hit Rate** | **75.52%** | **75.52%** | **75.52%** | **75.52%** | **75.52%** |
| | **Unique Expert HBM Hit Rate** | **26.56%** | **26.56%** | **26.56%** | **26.56%** | **26.56%** |
| | **Fresh Demand Misses (Unique)** | 47 | 47 | 47 | 47 | 47 |
| **5** | **Demand DDR5 Traffic** | 4.140 GB | 4.140 GB | 4.140 GB | 4.140 GB | 4.140 GB |
| | **Speculative DDR5 Traffic** | 0.000 GB | 0.352 GB | 0.000 GB | 0.352 GB | 0.000 GB |
| | **Total DDR5 Traffic** | 4.140 GB | 4.492 GB | 4.140 GB | 4.492 GB | 4.140 GB |
| | **Traffic Overhead vs Baseline** | **0.00%** | **+8.51%** | **0.00%** | **+8.51%** | **0.00%** |
| **6** | **Aggregate Demand Wait** | 64.684 ms | 64.684 ms | 64.684 ms | 64.684 ms | 64.684 ms |
| | **Exposed GPU Stall Time** | 64.684 ms | 64.684 ms | 64.684 ms | 64.684 ms | 64.684 ms |
| | **Wait per Generated Token** | 16.171 ms | 16.171 ms | 16.171 ms | 16.171 ms | 16.171 ms |
| | **Wait per Unique Demand** | 1.011 ms | 1.011 ms | 1.011 ms | 1.011 ms | 1.011 ms |
| **7** | **Mean Demand Miss Latency** | 1.376 ms | 1.376 ms | 1.376 ms | 1.376 ms | 1.376 ms |
| | **P50 Demand Miss Latency** | 1.376 ms | 1.376 ms | 1.376 ms | 1.376 ms | 1.376 ms |
| | **Overall Mean Demand Latency** | 0.337 ms | 0.337 ms | 0.337 ms | 0.337 ms | 0.337 ms |
| **8** | **Mean Per-Token Latency** | 18.282 ms | 18.282 ms | 18.282 ms | 18.282 ms | 18.282 ms |
| | **P50 Per-Token Latency** | 20.670 ms | 20.670 ms | 20.670 ms | 20.670 ms | 20.670 ms |
| | **P95 Per-Token Latency** | 23.877 ms | 23.877 ms | 23.877 ms | 23.877 ms | 23.877 ms |
| | **P99 Per-Token Latency** | 24.041 ms | 24.041 ms | 24.041 ms | 24.041 ms | 24.041 ms |
| | **Throughput (Tokens / Sec)** | 54.70 | 54.70 | 54.70 | 54.70 | 54.70 |
| | **Speedup vs Baseline** | **1.0000x** | **1.0000x** | **1.0000x** | **1.0000x** | **1.0000x** |

---

### 3.2 Visual Comparisons (Figures 1 to 4)

#### Figure 1: Projection-Level vs Unique-Expert HBM Hit Rate
![Figure 1: Core Comparison - Projection-Level vs Unique-Expert HBM Hit Rate](file:///home/mohman/ca_project/mohman/point_3/plots/fig1_core_coverage_and_hit_rate.png)

*Figure 1 demonstrates the stark disparity between projection-level accounting ($75.5\%$) and unique-expert hit rate ($26.6\%$). Out of $64$ unique expert invocations across the $4$ decoding steps, $17$ were resident hits from previous tokens, while $47$ were cold demand misses. For each of the $47$ misses, $W_1$ missed while $W_3$ and $W_2$ hit, producing $145$ projection hits out of $192$ projection demands.*

#### Figure 2: Memory Traffic Volume and Speculative Overhead
![Figure 2: Memory Traffic Volume and Speculative Overhead](file:///home/mohman/ca_project/mohman/point_3/plots/fig2_core_memory_traffic_and_overhead.png)

*Figure 2 shows DDR5 traffic breakdown. The demand baseline transfers $4.14$ GB ($47$ misses $\times 88.08$ MB). For ST-MoE ($H=1$) and CSTP ($H=1$), $4$ speculative transfers ($352.3$ MB) were dispatched during Step 0, increasing total DDR5 traffic to $4.49$ GB ($+8.51\%$ overhead). For $H=2$, because intra-token stage lookahead was bounded by $2$ MoE layers, zero speculative bytes were issued.*

#### Figure 3: Exposed Memory Stall Duration and Per-Token Latency
![Figure 3: Core Timing Comparison - Exposed Memory Stall & Token Latency](file:///home/mohman/ca_project/mohman/point_3/plots/fig3_core_exposed_stall_and_token_latency.png)

*Figure 3 displays total exposed memory stall time ($64.68$ ms across the $4$ tokens) alongside per-token latency ($18.28$ ms mean, $20.67$ ms median). Step 0 took $24.08$ ms due to $16$ cold misses, while subsequent steps dropped to $18.62$ ms as expert reuse in HBM increased.*

#### Figure 4: Category-Separated Expert Load Latency Distributions
![Figure 4: Category-Separated Expert Load Latency Distributions](file:///home/mohman/ca_project/mohman/point_3/plots/fig4_expert_load_latency_distributions.png)

*Figure 4 illustrates the bimodal nature of expert loading latencies in this configuration: resident hits incur strictly $0.00$ ms DDR5 latency, while fresh demand misses incur exactly $1.376$ ms ($1,376,256$ ns).*

---

## 4. Sensitivity Analyses

To thoroughly characterize prefetching effectiveness under varied operational conditions, $31$ independent experiments were executed and aggregated in [sensitivity_metrics.csv](file:///home/mohman/ca_project/mohman/point_3/reports/sensitivity_metrics.csv).

### 4.1 Candidate Budget Sensitivity ($B \in \{1, 2, 4, 8\}$)

We evaluated CSTP ($H=1$) on MoE-Anywhere ($B=1, \text{iter}=4$) while varying prefetch candidate budget $B$ from $1$ to $8$:

| Budget | Dispatched Prefetches | Wasted Traffic (MB) | Traffic Overhead (%) | Unique Hit Rate (%) | Mean Token Latency (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$B=1$** | 1 | 84.00 MB | +2.13% | 26.56% | 18.28 ms |
| **$B=2$** | 2 | 168.00 MB | +4.26% | 26.56% | 18.28 ms |
| **$B=4$** | 4 | 336.00 MB | +8.51% | 26.56% | 18.28 ms |
| **$B=8$** | 8 | 420.00 MB | +17.02% | 26.56% | 18.28 ms |

#### Figure 5: Candidate Budget Sensitivity Analysis
![Figure 5: Candidate Budget Sensitivity Analysis](file:///home/mohman/ca_project/mohman/point_3/plots/fig5_budget_sensitivity.png)

*Key Takeaway:* Under bus saturation where prefetch transfers complete after demand deadlines, increasing candidate budget directly scales wasted speculative traffic (from $84$ MB at $B=1$ to $420$ MB at $B=8$) without improving hit rate. Budget pruning is essential to avoid bandwidth degradation under contention.

---

### 4.2 Batch Size Scaling ($B \in \{1, 2, 4\}$)

We evaluated multi-sequence routing traces with independent routing decisions:

| Batch Size | Unique Demands | Demand Traffic (GB) | CSTP Traffic (GB) | Mean Token Latency (ms) | Throughput (tok/s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$B=1$** | 64 | 4.140 GB | 4.492 GB | 18.28 ms | 54.70 tok/s |
| **$B=2$** | 82 | 5.725 GB | 6.078 GB | 24.58 ms | 81.36 tok/s |
| **$B=4$** | 117 | 8.456 GB | 8.808 GB | 35.46 ms | 112.79 tok/s |

#### Figure 6: Batch Size Scaling and Memory Traffic Demand
![Figure 6: Batch Size Scaling and Memory Traffic Demand](file:///home/mohman/ca_project/mohman/point_3/plots/fig6_batch_size_scaling.png)

*Key Takeaway:* Larger batch sizes increase cross-sequence expert sharing: unique demands scale sub-linearly ($64 \to 82 \to 117$ rather than $64 \to 128 \to 256$). This increases overall decoding throughput by $+106.2\%$ ($112.79$ vs $54.70$ tok/s), proving that multi-sequence batching is highly effective at amortizing off-chip weight transfers.

---

### 4.3 Cross-Dataset Validation (MoE-Anywhere vs MoE-Suite Tasks 12 & 13)

We compared routing predictability across three datasets ($B=1, \text{iter}=4$):

| Dataset | Unique Demands | Fresh Misses | Unique Hit Rate | Projection Hit Rate | Demand Traffic |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MoE-Anywhere** | 64 | 47 | **26.56%** | **75.52%** | 4.140 GB |
| **MoE-Suite Task 12** | 64 | 59 | **7.81%** | **69.27%** | 5.197 GB |
| **MoE-Suite Task 13** | 64 | 60 | **6.25%** | **68.75%** | 5.285 GB |

#### Figure 7: Cross-Dataset Hit Rate Evaluation
![Figure 7: Cross-Dataset Hit Rate Evaluation](file:///home/mohman/ca_project/mohman/point_3/plots/fig7_cross_dataset_comparison.png)

*Key Takeaway:* MoE-Anywhere displays significantly higher temporal locality and expert reuse ($26.56\%$ unique hit rate) than MoE-Suite tasks ($6.25\% - 7.81\%$). In MoE-Suite, routing decisions disperse across a wider expert inventory, resulting in $12$ to $13$ additional cold misses and $\approx 1.1$ GB more DDR5 traffic.

---

### 4.4 Decode Length Evolution ($\text{Iter} \in \{2, 4, 8, 16\}$)

We observed the progression of cache residency and throughput across decoding iterations:

| Iterations | Unique Demands | Fresh Misses | Unique Hit Rate | Projection Hit Rate | Throughput (tok/s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Iter 2** | 32 | 28 | 12.50% | 70.83% | 46.83 tok/s |
| **Iter 4** | 64 | 47 | 26.56% | 75.52% | 54.70 tok/s |
| **Iter 8** | 128 | 103 | 19.53% | 73.18% | 50.46 tok/s |
| **Iter 16** | 256 | 197 | 23.05% | 74.35% | 52.50 tok/s |

#### Figure 8: Decoding Evolution and Steady-State Convergence
![Figure 8: Decoding Evolution and Steady-State Convergence](file:///home/mohman/ca_project/mohman/point_3/plots/fig8_decode_length_evolution.png)

*Key Takeaway:* As generation progresses beyond the cold-start phase ($T \ge 4$), the HBM cache stabilizes at $\approx 73\% - 75\%$ projection hit rate and throughput converges to $\approx 50 - 54$ tokens/second.

---

### 4.5 Constrained Cache Capacity Behavior ($20$ vs $30$ Experts vs $80$ GB)

To evaluate prefetching behavior under memory pressure, we restricted HBM cache capacity:

| Cache Budget | Capacity (GB) | Unique Demands | Fresh Misses | Demand Traffic | Exposed Stall |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **20 Experts** | 1.76 GB | 64 | **54** | **4.756 GB** | **74.32 ms** |
| **30 Experts** | 2.64 GB | 64 | **47** | **4.140 GB** | **64.68 ms** |
| **Unconstrained (80 GB)** | 80.00 GB | 64 | **47** | **4.140 GB** | **64.68 ms** |

#### Figure 9: Impact of Constrained Cache Capacity on Memory Stall
![Figure 9: Impact of Constrained Cache Capacity on Memory Stall](file:///home/mohman/ca_project/mohman/point_3/plots/fig9_constrained_cache_behavior.png)

*Key Takeaway:* When cache capacity falls below the active working set ($20$ experts $< 30$ resident experts required across $4$ steps), capacity evictions occur. This causes $7$ re-demand misses, forcing $616.6$ MB of redundant DDR5 traffic and adding $9.64$ ms of unhidden memory stall.

---

## 5. Critical Architectural Insights & Recommendations

### 5.1 The DDR5 Bandwidth Contention Bottleneck

A central revelation of this study is the interaction between priority arbitration and speculative prefetching:
1. When prefetch transfers have lower priority than demand misses, they are queued behind all concurrent demand transfers.
2. In cold-start or low-reuse phases, a single token generates $16$ demand misses, requiring $22.02$ ms of contiguous transfer time.
3. Because the GPU computation of a layer completes in $11.0$ ms, demand for the next layer arrives *before* speculative prefetch transfers can traverse the queue.
4. **Architectural Recommendation:** Future prefetch engines should implement *hybrid priority boosting* or *bandwidth partitioning* (e.g., reserving $25\%$ of DDR5 channel slots for speculative transfers) so that prefetches are not completely starved by demand cascades.

### 5.2 Multi-Stage Lookahead Alignment

In shallow MoE models ($2$ MoE layers), intra-token horizon $H \ge 2$ cannot find valid target stages within the same token. To enable effective lookahead in shallow architectures:
- The prefetch scheduler must support *inter-token lookahead* (predicting Stage 0 of Token $T+1$ while executing Stage 1 of Token $T$).
- In deeper models (e.g. DeepSeek-V3 full with $60$ MoE layers), intra-token $H=1$ and $H=2$ operate with ample horizon depth.

### 5.3 Deduplicated vs Projection-Level Accounting Standard

We strongly recommend that future MoE literature report dual-granularity metrics:
- **Projection-Level Metrics:** Essential for physical DRAM bus accounting (bytes transferred, burst requests).
- **Unique-Expert Metrics:** Essential for architectural evaluation (algorithmic hit rate, routing coverage, working set modeling).
Conflating the two conceals true cache miss rates behind sister-projection hits.

---

## 6. Verification and Certification

- **Unit Test Suite (Cases A–J):** [test_phase4b_effectiveness_metrics.cpp](file:///home/mohman/ca_project/mohman/point_3/tests/test_phase4b_effectiveness_metrics.cpp) — **10 Passed, 0 Failed**.
- **Regression Test Suite:** [test_prefetch_metrics.cpp](file:///home/mohman/ca_project/LLMSimulator/eval/test_prefetch_metrics.cpp) — **15 Passed, 0 Failed**.
- **Experiment Manifest:** [experiment_manifest.csv](file:///home/mohman/ca_project/mohman/point_3/experiment_manifest.csv) — **39 / 39 Experiments Completed**.
- **Certified Reproducibility:** All simulation commands, YAML files, and analysis scripts operate deterministically under the instructions in [reproducibility_manifest.md](file:///home/mohman/ca_project/mohman/point_3/reproducibility_manifest.md).

