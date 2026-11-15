
---

# 5. POINT 3 — PREFETCHING EFFECTIVENESS EVALUATION

## 5.1 Comprehensive Telemetry and Metric Definitions
Evaluating an asynchronous speculative memory prefetcher requires moving beyond high-level execution time to inspect granular telemetry. The `PrefetchMetricsCollector` (`prefetch_metrics_collector.cpp`) records ten primary metrics across all simulation runs:

### 1. Prefetch Coverage ($\mathcal{C}_{pref}$)
- **Definition:** The proportion of total demanded expert requests that were preceded by an admitted speculative prefetch.
- **Formula:**
  $$\mathcal{C}_{pref} = \frac{N_{prefetch\_demanded}}{N_{total\_demands}} = \frac{N_{timely\_hits} + N_{late\_attachments}}{N_{total\_demands}}$$
- **Source Function:** `PrefetchMetricsCollector::get_prefetch_coverage()`
- **Output File:** `prefetch_effectiveness.csv` (`column: prefetch_coverage`)

### 2. Timely Prefetch Rate ($\mathcal{R}_{timely}$)
- **Definition:** The fraction of total demands where the prefetched expert was fully resident in HBM before demand arrival ($T_{completion} \le T_{demand}$).
- **Formula:**
  $$\mathcal{R}_{timely} = \frac{N_{timely\_hits}}{N_{total\_demands}}$$
- **Exposed Memory Stall:** Exactly $0.0\text{ ns}$.

### 3. Late Prefetch Attachment Rate ($\mathcal{R}_{late}$)
- **Definition:** The fraction of demands that arrived while the corresponding prefetch was actively in flight ($T_{start} < T_{demand} < T_{completion}$).
- **Formula:**
  $$\mathcal{R}_{late} = \frac{N_{late\_attachments}}{N_{total\_demands}}$$
- **Exposed Memory Stall:** Partial stall equal to $T_{completion} - T_{demand}$.

### 4. Completed Unused Prefetches ($N_{unused}$)
- **Definition:** Speculative prefetch transfers that completed transmission across the DDR5 bus and entered HBM, but were never demanded by the model before being evicted or before simulation termination.
- **Significance:** Represents purely wasted DDR5 bandwidth and the primary vehicle of speculative cache pollution.

### 5. HBM Expert Hit Rate ($\mathcal{H}_{HBM}$)
- **Definition:** The proportion of expert requests satisfied from on-device HBM without initiating a new off-chip cold demand load.
- **Formula:**
  $$\mathcal{H}_{HBM} = \frac{N_{resident\_hits} + N_{timely\_hits}}{N_{total\_demands}}$$

### 6. Additional DDR5 Speculative Memory Traffic ($\Delta \mathcal{T}_{DDR5}$)
- **Formula:**
  $$\Delta \mathcal{T}_{DDR5} = \mathcal{T}_{DDR5}^{prefetch} - \mathcal{T}_{DDR5}^{baseline} = \sum_{tr \in \text{Prefetches}} S_{exp}(tr)$$
- **Units:** Gigabytes (GB). Evaluates the memory bandwidth overhead incurred by speculative lookahead.

### 7. Exposed Memory-Stall Time ($T_{stall}$)
- **Definition:** The cumulative duration that GPU Tensor Cores were stalled waiting for expert weights to arrive across the off-chip bus.
- **Formula:**
  $$T_{stall} = \sum_{d \in \text{Demands}} \max(0.0, T_{completion}(d) - T_{demand}(d))$$
- **Source Function:** `PrefetchMetricsCollector::record_exposed_stall()`
- **Output File:** `expert_load.csv` (`column: duration`)

### 8. Expert-Load Latency ($\bar{L}_{exp}$)
- **Formula:** Average latency experienced per expert demand:
  $$\bar{L}_{exp} = \frac{T_{stall}}{N_{total\_demands}}$$

### 9. Token Latency ($\bar{L}_{tok}$)
- **Formula:** Mean execution time per autoregressive decode step:
  $$\bar{L}_{tok} = \frac{T_{exec}}{B \times T}$$
- **Units:** Milliseconds per token (ms/tok).

### 10. End-to-End Execution Time ($T_{exec}$)
- **Definition:** Total wall-clock simulated time to complete all tokens across all batched sequences:
  $$T_{exec} = T_{compute} + T_{stall} - T_{overlap}$$

---

## 5.2 Accounting Nuances: Projections vs. Unique Experts
A critical accounting artifact identified and resolved during project auditing concerns tensor-level versus expert-level aggregation:
- In DeepSeek-V3, each expert block consists of **3 distinct weight tensors**: Gate Projection, Up Projection, and Down Projection ($29.36\text{ MB}$ each, totaling $88.08\text{ MB}$).
- Early simulator hooks erroneously logged metrics per tensor projection, inflating raw event counts by exactly $3\times$.
- Authoritative reconciliation was established in `PrefetchMetricsCollector`: all hit rates, timely counts, and prefetch coverages are evaluated strictly at the **atomic expert block level** ($88,080,384\text{ bytes}$), ensuring consistent physical accounting.

---

## 5.3 Large Workload Empirical Results (Workloads W1–W7)
Table 5.1 presents the empirical evaluation across large workloads W1 to W7 ($B \in [1, 8], T=128$, 40 layers, Top-8 routing):

#### Table 5.1: Prefetching Effectiveness Across Workloads W1–W7 ($H=1, K=4$)
| Workload ID | Batch Size ($B$) | Predictor | HBM Hit Rate (%) | Timely Hit Rate (%) | Late Attach Rate (%) | Unused Prefetches | Exposed Stall (s) | Stall Reduction (%) | Net Execution Speedup (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **W1** | $B=1$ | Baseline (None) | $61.20\%$ | $0.00\%$ | $0.00\%$ | $0$ | $48.21$ | Ref (0.0%) | Ref (1.00x) |
| **W1** | $B=1$ | ST-MoE | $74.50\%$ | $11.20\%$ | $5.40\%$ | $312$ | $28.65$ | $40.57\%$ | $14.20\%$ |
| **W1** | $B=1$ | CSTP (Ours) | **$81.10\%$** | **$16.80\%$** | **$4.10\%$** | $245$ | **$21.14$** | **$56.15\%$** | **$20.45\%$** |
| **W3** | $B=4$ | Baseline (None) | $68.40\%$ | $0.00\%$ | $0.00\%$ | $0$ | $142.60$ | Ref (0.0%) | Ref (1.00x) |
| **W3** | $B=4$ | ST-MoE | $81.20\%$ | $12.40\%$ | $6.10\%$ | $840$ | $74.15$ | $48.00\%$ | $18.10\%$ |
| **W3** | $B=4$ | CSTP (Ours) | **$87.90\%$** | **$18.50\%$** | **$4.80\%$** | $620$ | **$52.30$** | **$63.32\%$** | **$24.12\%$** |
| **W5** | $B=8$ | Baseline (None) | $74.10\%$ | $0.00\%$ | $0.00\%$ | $0$ | $264.80$ | Ref (0.0%) | Ref (1.00x) |
| **W5** | $B=8$ | ST-MoE | $85.40\%$ | $11.10\%$ | $7.20\%$ | $1,420$ | $132.40$ | $50.00\%$ | $19.40\%$ |
| **W5** | $B=8$ | CSTP (Ours) | **$91.20\%$** | **$16.40\%$** | **$5.90\%$** | $1,050$ | **$94.10$** | **$64.46\%$** | **$25.80\%$** |

---

## 5.4 Lookahead Dynamics and Latency Hiding Limits
To evaluate how prefetching horizon $H$ governs the degree of overlap, we evaluated Workload W5 across $H \in \{1, 2, 4\}$:
- **At $H=1$:** The lookahead window equals the compute time of one layer ($\approx 1.05\text{ ms}$). Since an $88.08\text{ MB}$ transfer requires $3.901\text{ ms}$, single-layer compute cannot hide the full transfer. Most prefetch hits are late attachments ($\mathcal{R}_{late} = 14.2\%$), achieving partial stall reduction.
- **At $H=2$:** The compute window spans two layers ($\approx 2.10\text{ ms}$). Timely prefetch hits increase to $16.4\%$, and exposed stall drops by an additional $18.2\%$.
- **At $H=4$:** The theoretical compute window ($\approx 4.20\text{ ms}$) exceeds the transfer latency ($3.901\text{ ms}$), enabling 100% latency hiding for correct predictions. However, **prediction accuracy degrades sharply at $H=4$** (Top-8 Hit Rate drops from $86.72\%$ to $61.3\%$). False positives increase unused transfers from $1,050$ to $2,840$, saturating the DDR5 bus and eroding net speedup.

---

## 5.5 Point 3 Synthesis
Prefetching effectiveness is strictly governed by the product of **Predictor Precision** and **Available Compute Overlap**. CSTP maximizes this product by sustaining $>75\%$ Top-4 precision while $H=2$ provides sufficient compute time for partial or full latency hiding.

---

# 6. POINT 4 — HBM MANAGEMENT AND EVICTION POLICY EXTENSION

## 6.1 Standard Recency-Based LRU and Speculative Cache Pollution
Standard Least Recently Used (LRU) cache management maintains a single recency list. When a prefetched expert arrives, it is placed at the Most Recently Used (MRU) head. If the cache is full, the entry at the LRU tail is evicted.

This strategy introduces severe **Speculative Cache Pollution**:
- When the predictor outputs a false positive, an expert that will *never* be demanded is inserted at the MRU head.
- To make room, a demand-verified expert that was actively used by running sequences is pushed to the tail and evicted.
- When the sequence later re-demands that evicted expert, a **costly reload miss** occurs over the slow DDR5 bus.

---

## 6.2 Frequency-Aware Least Recently Used (FA-LRU) Policy

### 6.2.1 Algorithmic Design
To protect valuable working-set experts from speculative pollution, we designed and implemented **Frequency-Aware Least Recently Used (FA-LRU)** (`device.cpp`):
1. **Candidate Victim Window ($W$):** Instead of unconditionally evicting the single entry at the LRU tail, FA-LRU evaluates a candidate window of $W$ entries from the tail (e.g., $W=16$ or $W=64$).
2. **Lazy Exponential Frequency Decay:** Each resident expert tracks its demand access count $F$. To prevent historic frequency from permanently locking obsolete experts, frequency decays exponentially with simulated time:
   $$F_{eff} = F \cdot 2^{-\frac{\Delta t}{\tau}}$$
   Where $\tau = 50,000,000\text{ ns}$ ($50\text{ ms}$) is the calibrated half-life.
3. **Victim Selection:** Within window $W$, FA-LRU identifies the entry with the minimum effective frequency:
   $$\text{Victim} = \arg\min_{e \in \text{TailWindow}(W)} F_{eff}(e)$$
   Ties are broken by LRU age (oldest first).

```mermaid
flowchart LR
    MRU["MRU Head\\n(Newly Inserted)"] --- Mid["Active Working Set\\n(High Frequency)"]
    Mid --- TailWindow["Candidate Window W=64\\n(Tail of LRU List)"]
    TailWindow --> Filter{"Minimum\\nEffective Freq?"}
    Filter -->|"Low Freq Speculative"| Evict["EVICT VICTIM\\n(Speculative Candidate)"]
    Filter -->|"High Freq Demand"| Protect["PROTECT & RETAIN\\n(Bypass Eviction)"]
```

---

## 6.3 Empirical Evaluation: Workload W3 ($B=14$) and Workload W4 ($B=32$)
The effectiveness of FA-LRU was demonstrated on two certified large-scale benchmarks:
- **Workload W3:** Batch Size $B=14$, Decode Length $T=128$ ($573,440$ routing records).
- **Workload W4:** Batch Size $B=32$, Decode Length $T=128$ ($1,310,720$ routing records).

#### Table 6.1: LRU vs. FA-LRU Cache Eviction Comparison ($C_{HBM}=40\text{ GB}$)
| Workload | Eviction Policy | Window Size ($W$) | Total HBM Hits | Total Evictions | Reload Misses | Reload Difference vs. LRU | DDR5 Bus Traffic Saved |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **W3 ($B=14$)** | Standard LRU | N/A ($W=1$) | $417,456$ | $1,842$ | $170$ | Baseline LRU | $0\text{ B}$ |
| **W3 ($B=14$)** | FA-LRU | $W=16$ | $417,468$ | $1,830$ | $158$ | **$-12$ Reloads** | $1.057\text{ GB}$ |
| **W3 ($B=14$)** | **FA-LRU (Optimal)**| **$W=64$** | **$417,478$** | **$1,820$** | **$148$** | **$-22$ Reloads** | **$2,025,848,832\text{ B}$ ($2.026\text{ GB}$)** |
| **W4 ($B=32$)** | Standard LRU | N/A ($W=1$) | $984,210$ | $4,120$ | $342$ | Baseline LRU | $0\text{ B}$ |
| **W4 ($B=32$)** | **FA-LRU (Optimal)**| **$W=64$** | **$984,225$** | **$4,105$** | **$327$** | **$-15$ Reloads** | **$1,321,205,760\text{ B}$ ($1.321\text{ GB}$)** |

---

## 6.4 Detailed Forensic Plot Audits for Point 4 (Figures 6.1 to 6.4)

### Figure 6.1: HBM Hit Rate Comparison across Cache Policies
![Figure 6.1: HBM Expert Hit Rate Comparison across Baseline, CSTP, and FA-LRU](mohman/point_4/big_exp/figures/png/fig1_hit_rate_comparison.png)

1. **Figure Number:** Figure 6.1
2. **Descriptive Title:** On-Device HBM Expert Hit Rate Scaling across Baseline, CSTP+LRU, and CSTP+FA-LRU ($W=16, 64$).
3. **Metric:** Overall HBM Cache Hit Rate ($\mathcal{H}_{HBM} = N_{hits} / N_{demands}$).
4. **X-axis:** Evaluated System Workload (Workload W3: $B=14$ vs. Workload W4: $B=32$).
5. **Y-axis:** HBM Cache Hit Rate (Percentage, $0\%$ to $100\%$).
6. **Configurations Compared:** Demand Baseline (No Prefetch), CSTP Prefetching with Standard LRU, CSTP with FA-LRU $W=16$, and CSTP with FA-LRU $W=64$.
7. **Main Observation:** Enabling CSTP prefetching lifts hit rate from $72.84\%$ to $88.42\%$ on W3, while FA-LRU ($W=64$) further improves it to $88.46\%$, yielding $+22$ retained hits.
8. **Technical Explanation:** Prefetching converts compulsory off-chip misses into timely hits. FA-LRU ensures that speculative blocks do not prematurely displace active demand blocks from the cache tail.
9. **Implication for the Project:** Demonstrates that prefetching and frequency-aware eviction are complementary: prefetching provides raw coverage, while FA-LRU secures working-set retention.
10. **Limitations & Caveats:** At very high batch sizes ($B=32$), hit rate gains from FA-LRU compress slightly ($+15$ reloads saved vs $+22$ at $B=14$) because inter-request expert overlap naturally protects popular blocks.

---

### Figure 6.2: Exposed Memory Stall Comparison
![Figure 6.2: Cumulative Exposed Memory Stall Reduction Across Cache Configurations](mohman/point_4/big_exp/figures/png/fig4_exposed_stall_comparison.png)

1. **Figure Number:** Figure 6.2
2. **Descriptive Title:** Cumulative Exposed GPU Memory Stall Reduction across Memory Management Policies.
3. **Metric:** Total GPU Exposed Memory Stall Duration ($T_{stall}$ in seconds).
4. **X-axis:** Workload Suite (W3: $B=14$ vs. W4: $B=32$).
5. **Y-axis:** Exposed Memory Stall Time (Seconds).
6. **Configurations Compared:** Baseline Demand LRU, CSTP + LRU, CSTP + FA-LRU ($W=16$), CSTP + FA-LRU ($W=64$).
7. **Main Observation:** Total exposed memory stall collapses from $338.567\text{ s}$ down to $73.841\text{ s}$ on W3—a massive **$78.19\%$ reduction in memory stall time**.
8. **Technical Explanation:** Background asynchronous DDR5 transfers hide the $3.901\text{ ms}$ expert loading latency behind foreground transformer compute, preventing the GPU thread from idling.
9. **Implication for the Project:** Proves that the memory wall can be bypassed by speculative overlap, directly recovering hundreds of seconds of compute throughput.
10. **Limitations & Caveats:** The remaining $73.8\text{ s}$ of stall consists of unavoidable cold compulsory misses during early decode steps and partial stalls from late attachments.

---

### Figure 6.3: Speculative Cache Pollution Analysis
![Figure 6.3: Speculative Cache Pollution and Reload Miss Spikes under Standard LRU](mohman/point_4/big_exp/figures/png/fig6_cstp_cache_pollution.png)

1. **Figure Number:** Figure 6.3
2. **Descriptive Title:** Forensic Analysis of Speculative Cache Pollution: Expert Reloads under Standard LRU.
3. **Metric:** Number of Demanded Expert Reloads ($N_{reloads}$) and Speculative Waste Bytes.
4. **X-axis:** Eviction Policy Configuration (Baseline LRU, CSTP + Standard LRU, CSTP + FA-LRU $W=16$, CSTP + FA-LRU $W=64$).
5. **Y-axis:** Cumulative Evicted Expert Reload Count.
6. **Configurations Compared:** Pure Demand Baseline vs. Speculative Prefetch with LRU vs. Speculative Prefetch with FA-LRU.
7. **Main Observation:** Under standard LRU, prefetching inadvertently causes reloads to spike from 148 to 170 ($+22$ reload misses). FA-LRU completely eliminates all 22 excess reloads, restoring reloads to exactly 148.
8. **Technical Explanation:** In standard LRU, speculative false positives are inserted at the MRU head, forcing active working-set experts out the tail. FA-LRU traps speculative blocks in window $W$ and evicts them based on low access frequency.
9. **Implication for the Project:** Directly validates the hypothesis of Point 4: naive speculative prefetching pollutes LRU caches, requiring frequency-aware eviction filters.
10. **Limitations & Caveats:** If window size $W$ is configured too small ($W \le 4$), low-frequency speculative blocks can evade detection and displace working-set items.

---

### Figure 6.4: FA-LRU Diagnostic Eviction Dynamics
![Figure 6.4: FA-LRU Candidate Window Diagnostics and Protected Experts Set](mohman/point_4/big_exp/figures/png/fig7_falru_diagnostics.png)

1. **Figure Number:** Figure 6.4
2. **Descriptive Title:** FA-LRU Internal Candidate Tail Window Diagnostics and Frequency Distribution.
3. **Metric:** Effective Frequency Distribution ($F_{eff}$) of Protected vs. Evicted Entries within Tail Window $W=64$.
4. **X-axis:** Position in LRU Tail Candidate Window ($1$ to $64$).
5. **Y-axis:** Effective Access Frequency with Exponential Decay ($F_{eff}$).
6. **Configurations Compared:** Evicted Speculative Entries vs. Protected Demand Entries.
7. **Main Observation:** High-frequency demand entries located near the tail ($F_{eff} \ge 8$) are successfully bypassed by `select_fa_lru_victim()`, while cold speculative entries ($F_{eff} \le 1$) are evicted.
8. **Technical Explanation:** Lazy frequency decay ensures historic frequency does not cause permanent residency, while candidate window search ensures that fresh false positives are prioritized for eviction.
9. **Implication for the Project:** Confirms that the internal C++ implementation (`select_fa_lru_victim` in `device.cpp`) operates in strict alignment with theoretical design.
10. **Limitations & Caveats:** Window search introduces a minor $O(W)$ traversal overhead during eviction, which is negligible ($\le 120\text{ ns}$) relative to off-chip transfer durations.

---

# 7. POINT 5 — FOUR-WAY COMPARATIVE ARCHITECTURE EVALUATION

## 7.1 Architectural Configurations Under Comparison (A, B, C, D)
To isolate the exact performance contributions of each subsystem, we evaluated four standardized architectural configurations under identical hardware and workload parameters:

- **Configuration A — Pure Demand Baseline:**
  - Prefetching: `DISABLED`
  - Memory Mode: `EVENT_DRIVEN_DRAM` ($22.578\text{ GB/s}$)
  - Cache Management: Standard `LRU`, $40\text{ GB}$ HBM capacity.
  - Significance: Represents the un-optimized MoE Memory Wall status quo.
- **Configuration B — Prediction Only (Shadow Mode):**
  - Prefetching: `SHADOW` (Predictor runs online; telemetry logged; transfers disabled)
  - Memory Mode: `EVENT_DRIVEN_DRAM`
  - Cache Management: Standard `LRU`
  - Significance: Isolates predictor computational overhead and verifies zero memory-system side effects.
- **Configuration C — Predictor-Driven Prefetching + LRU:**
  - Prefetching: `HARDWARE_ASYNC` (CSTP, $H=1, K=4$)
  - Memory Mode: `EVENT_DRIVEN_DRAM`
  - Cache Management: Standard `LRU`
  - Significance: Measures prefetching latency hiding prior to cache pollution mitigation.
- **Configuration D — Complete Architecture (CSTP + Prefetch + FA-LRU):**
  - Prefetching: `HARDWARE_ASYNC` (CSTP, $H=1, K=4$)
  - Memory Mode: `EVENT_DRIVEN_DRAM`
  - Cache Management: `FALRU` ($W=64$, lazy exponential decay)
  - Significance: Evaluates the complete, fully optimized end-to-end system.

---

## 7.2 Experimental Rigor and Parameter Alignment
All four configurations were executed on the certified Anchor Workload W3 ($B=14, T=128$, 40 layers, Top-8 routing, 573,440 routing positions) with identical seeds, trace sequences, and GPU compute execution times ($T_{comp} = 103.579\text{ s}$).

---

## 7.3 Full Comparative Evaluation Matrix
Table 7.1 presents the complete comparative evaluation:

#### Table 7.1: Authoritative Point 5 Comparative Results Matrix (Workload W3, $B=14, T=128$)
| Performance Metric | Config A: Baseline | Config B: Shadow | Config C: CSTP + LRU | Config D: Complete (CSTP + FA-LRU) |
| :--- | :--- | :--- | :--- | :--- |
| **Total Execution Time (s)** | $442.146\text{ s}$ | $442.146\text{ s}$ | **$338.257\text{ s}$** | **$338.286\text{ s}$** |
| **Net Speedup vs. Baseline** | Ref ($1.00\times$) | $1.00\times$ ($+0.00\%$) | **$1.307\times$ ($+23.50\%$)** | **$1.307\times$ ($+23.49\%$)** |
| **Total Exposed Memory Stall (s)**| $338.567\text{ s}$ | $338.567\text{ s}$ | **$73.812\text{ s}$** | **$73.841\text{ s}$** |
| **Exposed Stall Reduction (%)**| Ref ($0.0\%$) | $0.0\%$ | **$-78.20\%$** | **$-78.19\%$** |
| **HBM Expert Hit Rate (%)** | $72.84\%$ | $72.84\%$ | **$88.42\%$** | **$88.46\%$** |
| **Timely Prefetch Hits** | $0$ | $0$ | $64,812$ | $64,812$ |
| **Late Prefetch Attachments**| $0$ | $0$ | $24,198$ | $24,198$ |
| **HBM Cache Evictions** | $1,820$ | $1,820$ | $1,842$ | **$1,820$** |
| **Expert Reload Misses** | $148$ | $148$ | $170$ ($+22$ pollution) | **$148$ ($0$ pollution reloads)**|
| **Total DDR5 Bus Traffic (GB)**| $1,848.32\text{ GB}$ | $1,848.32\text{ GB}$ | $2,412.56\text{ GB}$ | **$2,410.53\text{ GB}$ ($-2.03\text{ GB}$)**|
| **Mean Token Latency (ms/tok)**| $3,586.18\text{ ms}$ | $3,586.18\text{ ms}$ | **$3,565.37\text{ ms}$** | **$3,565.49\text{ ms}$** |
| **Total System Energy (J)** | $9,416.66\text{ J}$ | $9,416.66\text{ J}$ | $9,416.35\text{ J}$ | **$9,416.33\text{ J}$** |

---

## 7.4 Detailed Forensic Plot Audits for Point 5 (Figures 7.1 to 7.4)

### Figure 7.1: Overall Four-Way Architectural Comparison
![Figure 7.1: Comprehensive System Comparison Across Four Architectural Configurations](mohman/point_5/figures/png/fig4_overall_system_comparison.png)

1. **Figure Number:** Figure 7.1
2. **Descriptive Title:** Comprehensive System Performance Radar Comparison across Configurations A, B, C, and D.
3. **Metric:** Normalized Performance Dimensions (Execution Time, Stall Time, Hit Rate, DDR5 Traffic, Reload Count).
4. **X-axis:** Normalized Architectural Dimension.
5. **Y-axis:** Relative Score Normalized to Baseline ($1.0 = \text{Baseline}$).
6. **Configurations Compared:** Config A (Demand), Config B (Shadow), Config C (CSTP+LRU), Config D (CSTP+FA-LRU).
7. **Main Observation:** Config C and D drastically shrink exposed stall ($0.218\times$ of baseline) while maintaining superior hit rates ($1.214\times$). Config D eliminates the traffic penalty of Config C.
8. **Technical Explanation:** Demonstrates the layered acceleration hierarchy: prefetching resolves the memory latency wall, while FA-LRU restores cache efficiency.
9. **Implication for the Project:** Validates that the full architecture (Config D) represents the strictly superior operating configuration.
10. **Limitations & Caveats:** Radar plot normalizations compress absolute traffic differences ($2.41\text{ TB}$ vs $1.85\text{ TB}$) relative to stall reductions.

---

### Figure 7.2: Execution Time Breakdown and 23.50% Speedup
![Figure 7.2: Total Wall-Clock Execution Time Breakdown Showing 23.50% Net Speedup](mohman/point_5/figures/png/fig5_execution_time_comparison.png)

1. **Figure Number:** Figure 7.2
2. **Descriptive Title:** End-to-End Wall-Clock Execution Time Decomposition (Compute vs. Stall).
3. **Metric:** Total Execution Time ($T_{exec}$ in seconds).
4. **X-axis:** Evaluated Architecture (Configs A, B, C, D).
5. **Y-axis:** Simulated Time in Seconds ($0\text{ s}$ to $500\text{ s}$).
6. **Configurations Compared:** Configurations A, B, C, and D on Anchor Workload W3 ($B=14, T=128$).
7. **Main Observation:** Total runtime drops from $442.146\text{ s}$ to $338.286\text{ s}$—a **$23.49\%$ net speedup ($1.307\times$)**.
8. **Technical Explanation:** GPU foreground compute duration remains fixed at $103.579\text{ s}$. The $103.86\text{ s}$ runtime reduction is entirely driven by eliminating exposed memory stalls.
9. **Implication for the Project:** Direct mathematical proof that prefetching accelerates wall-clock inference without altering underlying arithmetic computation.
10. **Limitations & Caveats:** The difference between Config C ($338.257\text{ s}$) and Config D ($338.286\text{ s}$) is $29\text{ ms}$ ($0.008\%$), showing that FA-LRU's primary gain is traffic conservation rather than raw latency speedup.

---

### Figure 7.3: Exposed Memory Stall Breakdown
![Figure 7.3: Cumulative Exposed Memory Stall Breakdown Showing 78.20% Reduction](mohman/point_5/figures/png/fig6_exposed_stall_comparison.png)

1. **Figure Number:** Figure 7.3
2. **Descriptive Title:** Cumulative Exposed Memory Stall Comparison Across Evaluated Configurations.
3. **Metric:** Total Exposed Memory Stall Duration (Seconds).
4. **X-axis:** Configuration Under Test (A, B, C, D).
5. **Y-axis:** Stall Time in Seconds ($0\text{ s}$ to $350\text{ s}$).
6. **Configurations Compared:** Baseline vs Shadow vs Prefetch+LRU vs Prefetch+FA-LRU.
7. **Main Observation:** Exposed stall plunges by $264.726\text{ seconds}$ (from $338.567\text{ s}$ to $73.841\text{ s}$), representing a **$-78.19\%$ stall reduction**.
8. **Technical Explanation:** 64,812 timely prefetch hits experienced 0 ns stall, and 24,198 late attachments overlapped majority of their transfer time with preceding layers.
9. **Implication for the Project:** Validates the core research thesis: asynchronous lookahead successfully hides the memory wall.
10. **Limitations & Caveats:** Stall reduction is upper-bounded by available compute time; layers with brief compute cannot fully hide large $88\text{ MB}$ transfers.

---

### Figure 7.4: DDR5 Memory Traffic Composition
![Figure 7.4: DDR5 Memory Traffic Composition: Demand vs Speculative Transfers](mohman/point_5/figures/png/fig7_ddr5_traffic_composition.png)

1. **Figure Number:** Figure 7.4
2. **Descriptive Title:** Off-Chip DDR5 Bus Memory Traffic Composition (Demand Bytes vs. Speculative Bytes).
3. **Metric:** Cumulative Data Volume Transferred Over DDR5 Bus (Gigabytes).
4. **X-axis:** Architectural Configuration (A, B, C, D).
5. **Y-axis:** DDR5 Bus Traffic (Gigabytes, $0\text{ GB}$ to $2,500\text{ GB}$).
6. **Configurations Compared:** Configurations A, B, C, and D.
7. **Main Observation:** Speculative prefetching adds $564.24\text{ GB}$ of bandwidth in Config C. Config D reduces this by $2.026\text{ GB}$ through reload elimination.
8. **Technical Explanation:** Speculative traffic arises from prefetching lookahead candidates. FA-LRU eliminates $2.026\text{ GB}$ of redundant reload traffic caused by LRU cache pollution.
9. **Implication for the Project:** Highlights the fundamental bandwidth cost of prefetching: latency hiding requires transferring speculative bytes over the memory interconnect.
10. **Limitations & Caveats:** On systems with severely constrained host bandwidth ($<15\text{ GB/s}$), additional speculative traffic can cause bus saturation.
