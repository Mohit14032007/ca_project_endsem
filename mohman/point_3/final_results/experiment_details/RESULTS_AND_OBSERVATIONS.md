# Point 3 — Consolidated Results and Experimental Observations

**Repository:** [`/home/mohman/ca_project/LLMSimulator`](file:///home/mohman/ca_project/LLMSimulator)  
**Deliverable Path:** [`mohman/point_3/final_results/experiment_details/RESULTS_AND_OBSERVATIONS.md`](file:///home/mohman/ca_project/mohman/point_3/final_results/experiment_details/RESULTS_AND_OBSERVATIONS.md)  
**Status:** Authoritative Empirical Reference  

---

## 1. Per-Workload Quantitative Findings (Large-Workload Campaign)

The extended large-workload campaign executed 27 runs across seven serving regimes (Workloads W1 through W7) modeling DeepSeek-V3 across all 40 physical MoE stages under a constrained 35 GB HBM cache budget.

### Workload W1 ($B=14, T=4, M=40, \text{Cache}=35\text{ GB}$)
- **Operational Regime:** Low decode depth ($T=4$), medium batch concurrency ($B=14$). Total tokens: 56.
- **Baseline Metrics:** Total Sim Time = $4.855 \text{ s}$, Exposed Stall = $4,619.64 \text{ ms}$, HBM Hit Rate = $67.0\%$, Total DDR5 = $385.10 \text{ GB}$, Throughput = $11.53 \text{ tok/s}$.
- **ST-MoE ($H=2$):** Sim Time = $4.853 \text{ s}$ ($\Delta = -2.73 \text{ ms}$), Stall Reduction = $+2.75 \text{ ms}$, Assoc Coverage = $4.81\%$, Late Rate = $100.0\%$, Unused Rate = $14.16\%$, Net Extra Traffic = $+38.76 \text{ GB}$ ($+10.06\%$).
- **CSTP ($H=2$):** Sim Time = $4.853 \text{ s}$ ($\Delta = -2.73 \text{ ms}$), Stall Reduction = $+2.75 \text{ ms}$, Assoc Coverage = $4.15\%$, Late Rate = $100.0\%$, Unused Rate = $16.05\%$, Net Extra Traffic = $+36.64 \text{ GB}$ ($+9.51\%$).
- **Observation:** At $T=4$, online adaptation is in cold-start warmup. Both predictors achieve identical minor stall reductions ($+2.75 \text{ ms}$) with elevated unused transfer rates (~14–16%).

### Workload W2 ($B=14, T=16, M=40, \text{Cache}=35\text{ GB}$)
- **Operational Regime:** Medium decode depth ($T=16$), medium batch concurrency ($B=14$). Total tokens: 224.
- **Baseline Metrics:** Total Sim Time = $18.565 \text{ s}$, Exposed Stall = $18,467.58 \text{ ms}$, HBM Hit Rate = $67.0\%$, Total DDR5 = $1,536.08 \text{ GB}$, Throughput = $12.07 \text{ tok/s}$.
- **ST-MoE ($H=2$):** Sim Time = $18.526 \text{ s}$ ($\Delta = -39.56 \text{ ms}$), Stall Reduction = $+39.88 \text{ ms}$, Assoc Coverage = $8.27\%$, Late Rate = $100.0\%$, Unused Rate = $9.00\%$, Net Extra Traffic = $+80.68 \text{ GB}$ ($+5.25\%$).
- **CSTP ($H=2$):** Sim Time = $18.504 \text{ s}$ ($\Delta = -61.14 \text{ ms}$), Stall Reduction = $+61.61 \text{ ms}$, Assoc Coverage = $5.31\%$, Late Rate = $100.0\%$, Unused Rate = $13.31\%$, Net Extra Traffic = $+45.89 \text{ GB}$ ($+2.99\%$).
- **Observation:** Extending sequence length from 4 to 16 allows online adaptation to stabilize. CSTP achieves a $1.54\times$ greater stall reduction than ST-MoE while generating 43.1% less additional memory traffic.

### Workload W3 ($B=32, T=16, M=40, \text{Cache}=35\text{ GB}$)
- **Operational Regime:** Medium-high concurrency ($B=32$), medium decode depth ($T=16$). Total tokens: 512.
- **Baseline Metrics:** Total Sim Time = $35.732 \text{ s}$, Exposed Stall = $35,532.74 \text{ ms}$, HBM Hit Rate = $67.0\%$, Total DDR5 = $3,511.97 \text{ GB}$, Throughput = $14.33 \text{ tok/s}$.
- **ST-MoE ($H=2$):** Sim Time = $35.699 \text{ s}$ ($\Delta = -33.23 \text{ ms}$), Stall Reduction = $+33.47 \text{ ms}$, Assoc Coverage = $7.26\%$, Late Rate = $100.0\%$, Unused Rate = $8.70\%$, Net Extra Traffic = $+152.03 \text{ GB}$ ($+4.33\%$).
- **CSTP ($H=2$):** Sim Time = $35.567 \text{ s}$ ($\Delta = -165.08 \text{ ms}$), Stall Reduction = $+165.72 \text{ ms}$, Assoc Coverage = $4.42\%$, Late Rate = $100.0\%$, Unused Rate = $12.15\%$, Net Extra Traffic = $+70.99 \text{ GB}$ ($+2.02\%$).
- **Observation:** At $B=32$, CSTP's stall reduction ($+165.72 \text{ ms}$) expands to **4.95x that of ST-MoE** ($+33.47 \text{ ms}$). ST-MoE transfer queues begin experiencing head-of-line blocking.

### Workload W4 ($B=48, T=16, M=40, \text{Cache}=35\text{ GB}$)
- **Operational Regime:** High concurrency ($B=48$), medium decode depth ($T=16$). Total tokens: 768.
- **Baseline Metrics:** Total Sim Time = $51.772 \text{ s}$, Exposed Stall = $51,475.29 \text{ ms}$, HBM Hit Rate = $67.0\%$, Total DDR5 = $5,267.83 \text{ GB}$, Throughput = $14.83 \text{ tok/s}$.
- **ST-MoE ($H=2$):** Sim Time = $51.672 \text{ s}$ ($\Delta = -99.73 \text{ ms}$), Stall Reduction = $+100.55 \text{ ms}$, Assoc Coverage = $7.57\%$, Late Rate = $100.0\%$, Unused Rate = $7.99\%$, Net Extra Traffic = $+198.87 \text{ GB}$ ($+3.78\%$).
- **CSTP ($H=2$):** Sim Time = $51.570 \text{ s}$ ($\Delta = -201.98 \text{ ms}$), Stall Reduction = $+202.57 \text{ ms}$, Assoc Coverage = $4.09\%$, Late Rate = $100.0\%$, Unused Rate = $12.25\%$, Net Extra Traffic = $+107.55 \text{ GB}$ ($+2.04\%$).
- **Observation:** CSTP achieves double the stall reduction of ST-MoE ($202.57 \text{ ms}$ vs $100.55 \text{ ms}$) with approximately half the net additional traffic ($+107.55 \text{ GB}$ vs $+198.87 \text{ GB}$).

### Workload W5 ($B=70, T=16, M=40, \text{Cache}=35\text{ GB}$)
- **Operational Regime:** Maximum batch concurrency ($B=70$), medium decode depth ($T=16$). Total tokens: 1,120.
- **Baseline Metrics:** Total Sim Time = $63.507 \text{ s}$, Exposed Stall = $63,068.53 \text{ ms}$, HBM Hit Rate = $67.0\%$, Total DDR5 = $7,681.33 \text{ GB}$, Throughput = $17.64 \text{ tok/s}$.
- **ST-MoE ($H=2$):** Sim Time = $63.248 \text{ s}$ ($\Delta = -258.75 \text{ ms}$), Stall Reduction = $+260.57 \text{ ms}$, Assoc Coverage = $6.82\%$, Late Rate = $100.0\%$, Unused Rate = $8.35\%$, Net Extra Traffic = $+227.42 \text{ GB}$ ($+2.96\%$).
- **CSTP ($H=2$):** Sim Time = $62.999 \text{ s}$ ($\Delta = -508.16 \text{ ms}$), Stall Reduction = $+509.94 \text{ ms}$, Assoc Coverage = $3.50\%$, Late Rate = $100.0\%$, Unused Rate = $12.16\%$, Net Extra Traffic = $+125.69 \text{ GB}$ ($+1.64\%$).
- **Observation:** Concurrency scaling dramatically increases cache churn. CSTP reduces stall by half a second ($509.94 \text{ ms}$), outperforming ST-MoE by $1.96\times$.

### Workload W6 ($B=70, T=32, M=40, \text{Cache}=35\text{ GB}$)
- **Operational Regime:** Maximum batch concurrency ($B=70$), extended decode depth ($T=32$). Total tokens: 2,240.
- **Baseline Metrics:** Total Sim Time = $189.535 \text{ s}$, Exposed Stall = $188,198.24 \text{ ms}$, HBM Hit Rate = $67.0\%$, Total DDR5 = $15,359.39 \text{ GB}$, Throughput = $11.82 \text{ tok/s}$.
- **ST-MoE ($H=2$):** Sim Time = $188.995 \text{ s}$ ($\Delta = -539.92 \text{ ms}$), Stall Reduction = $+543.23 \text{ ms}$, Assoc Coverage = $8.09\%$, Late Rate = $100.0\%$, Unused Rate = $6.78\%$, Net Extra Traffic = $+332.17 \text{ GB}$ ($+2.16\%$).
- **CSTP ($H=2$):** Sim Time = $187.382 \text{ s}$ ($\Delta = -2.153 \text{ s}$), Stall Reduction = $+2,160.97 \text{ ms}$, Assoc Coverage = $3.94\%$, Late Rate = $100.0\%$, Unused Rate = $9.38\%$, Net Extra Traffic = $+118.82 \text{ GB}$ ($+0.77\%$).
- **Observation:** Doubling sequence length to 32 quadruples CSTP's stall reduction from $509.94 \text{ ms}$ to **$2,160.97 \text{ ms}$** ($3.98\times$ ST-MoE). Traffic overhead drops below 1% for CSTP ($+0.77\%$).

### Workload W7 ($B=70, T=64, M=40, \text{Cache}=35\text{ GB}$)
- **Operational Regime:** Maximum batch concurrency ($B=70$), deep generation sequence ($T=64$). Total tokens: 4,480.
- **Baseline Metrics:** Total Sim Time = $488.483 \text{ s}$, Exposed Stall = $484,792.17 \text{ ms}$, HBM Hit Rate = $67.0\%$, Total DDR5 = $30,709.84 \text{ GB}$, Throughput = $9.17 \text{ tok/s}$.
- **ST-MoE ($H=2$):** Sim Time = $486.210 \text{ s}$ ($\Delta = -2.273 \text{ s}$), Stall Reduction = $+2,289.03 \text{ ms}$, Assoc Coverage = $9.23\%$, Late Rate = $100.0\%$, Unused Rate = $5.87\%$, Net Extra Traffic = $+420.32 \text{ GB}$ ($+1.37\%$), Mean Decode Latency Delta = $-35.52 \text{ ms}$.
- **CSTP ($H=2$):** Sim Time = $481.766 \text{ s}$ ($\Delta = -6.717 \text{ s}$), Stall Reduction = $+6,744.86 \text{ ms}$, Assoc Coverage = $4.81\%$, Late Rate = $100.0\%$, Unused Rate = $7.05\%$, Net Extra Traffic = $+119.53 \text{ GB}$ ($+0.39\%$), Mean Decode Latency Delta = $-104.95 \text{ ms}$.
- **Observation:** W7 represents the definitive benchmark result. CSTP achieves a **$6.745 \text{ second}$** stall reduction and a **$6.717 \text{ second}$** simulation time reduction ($104.95 \text{ ms}$ per decode step), outperforming ST-MoE by **2.94x** while consuming **71.6% less additional bandwidth**.

---

## 2. Cross-Workload Comparative Analysis

```
========================================================================================================================
                                     CROSS-WORKLOAD CORE RESEARCH QUESTIONS
========================================================================================================================
Research Question                      ST-MoE Behavior                     CSTP Behavior                       Superior Method
------------------------------------------------------------------------------------------------------------------------
1. Stall Reduction Magnitude           Scales from +2.75ms (W1) to         Scales from +2.75ms (W1) to         CSTP (2.94x higher
                                       +2,289.03ms (W7)                    +6,744.86ms (W7)                    in W7)
------------------------------------------------------------------------------------------------------------------------
2. Net Additional DDR5 Traffic         +38.76 GB (W1) to                   +36.64 GB (W1) to                   CSTP (71.6% less
                                       +420.32 GB (W7)                     +119.53 GB (W7)                     traffic in W7)
------------------------------------------------------------------------------------------------------------------------
3. Dispatched Prefetch Volume          High: 102,737 transfers (W7)        Selective: 54,076 transfers (W7)    CSTP (47.4% fewer
                                       Floods DMA queues                   Keeps bus clear                     dispatches)
------------------------------------------------------------------------------------------------------------------------
4. HBM Cache Hit Rate Impact           Invariable: Exactly 67.0%           Invariable: Exactly 67.0%           Neither (Late hits
                                       across all runs                     across all runs                     do not alter hit rate)
------------------------------------------------------------------------------------------------------------------------
5. Stall Red to Sim Time Conversion    1:1 conversion                      1:1 conversion                      Both (Synchronous
                                       2.289s stall -> 2.273s sim          6.745s stall -> 6.717s sim          batch critical path)
------------------------------------------------------------------------------------------------------------------------
6. Sensitivity to Batch Size (B)       Stall red grows from 39.88ms        Stall red grows from 61.61ms        CSTP (8.3x growth
                                       (B=14) to 260.57ms (B=70)           (B=14) to 509.94ms (B=70)           vs 6.5x for ST-MoE)
------------------------------------------------------------------------------------------------------------------------
7. Sensitivity to Seq Length (T)       Stall red grows from 260.57ms       Stall red grows from 509.94ms       CSTP (13.2x growth
                                       (T=16) to 2,289.03ms (T=64)         (T=16) to 6,744.86ms (T=64)         vs 8.8x for ST-MoE)
------------------------------------------------------------------------------------------------------------------------
8. Timeliness vs Performance           0.00% strict timely;                0.00% strict timely;                CSTP (17.6x deeper
                                       3.81 us in transit at demand        66.97 us in transit at demand       bus transit progress)
========================================================================================================================
```

### Synthesis of Core Observations

1. **The In-Flight Progress Mechanism:**  
   Because transferring an 88.08 MB expert across a 64 GB/s bus requires 1.376 ms, whereas inter-stage execution takes only 20–50 microseconds, prefetches cannot finish before demand arrives within a 1- or 2-stage horizon. However, prefetches that are actively in-flight on the bus truncate stall times. The GPU waits only for the remaining bus transfer time.
2. **The Queue Congestion Phenomenon:**  
   ST-MoE's integer counter saturation emitted 102,737 prefetches in W7, congesting the transfer engine. Transfers sat in FIFO queues and had only been transmitting for an average of **$3.81 \mu\text{s}$** when demand arrived. CSTP's Dirichlet-smoothed contextual scoring dispatched 54,076 prefetches, maintaining clear queues. CSTP transfers had been transmitting across the physical wires for an average of **$66.97 \mu\text{s}$** at demand arrival, saving **17.6x more stall per matched demand**.
3. **Cache Invariance:**  
   Neither predictor altered the architectural HBM hit rate (67.0%). In-flight attachments are classified as cache misses that attach to existing DMA buffers, leaving cache residency metrics unchanged.

