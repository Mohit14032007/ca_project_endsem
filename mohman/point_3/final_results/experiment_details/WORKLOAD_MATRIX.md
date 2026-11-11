# Point 3 — Workload and Parameter Matrix

**Repository:** [`/home/mohman/ca_project/LLMSimulator`](file:///home/mohman/ca_project/LLMSimulator)  
**Deliverable Path:** [`mohman/point_3/final_results/experiment_details/WORKLOAD_MATRIX.md`](file:///home/mohman/ca_project/mohman/point_3/final_results/experiment_details/WORKLOAD_MATRIX.md)  
**Status:** Complete Architectural Reference  

---

## 1. Overview of Experimental Regimes

Point 3 evaluates mixture-of-experts prefetching across three distinct serving regimes to explore memory capacity constraints, lookahead distances, queueing contention, and temporal sequence depth.

```
========================================================================================================================
                                          WORKLOAD REGIME ARCHITECTURAL MATRIX
========================================================================================================================
Parameter / Dimension        Original Sensitivity Campaign       Phase 2 Depth Scaling       Large-Workload Serving
------------------------------------------------------------------------------------------------------------------------
Model Architecture           DeepSeek-V3 (Truncated)             DeepSeek-V3 (Full Depth)    DeepSeek-V3 (Full Depth)
Physical MoE Stages          3 Stages (Stages 0–2)               40 Stages (Stages 0–39)     40 Stages (Stages 0–39)
Experts per Stage            256 Routed + 1 Shared               256 Routed + 1 Shared       256 Routed + 1 Shared
Top-K Routing Selection      8 Experts per Token                 8 Experts per Token         8 Experts per Token
Serving Batch Concurrency    B in {1, 2, 4, 8}                   B in {1, 2, 4, 8}           B in {14, 32, 48, 70}
Decode Sequence Length (T)   4 Decode Tokens                     4 and 8 Decode Tokens       4, 16, 32, 64 Decode Tokens
HBM Cache Capacity           80 GB (~908 expert slots)           30 GB (~340 expert slots)   35 GB (~397 expert slots)
Active Working Set Size      ~2.11 GB (Fits 100% in HBM)         ~11.3 GB to ~28.2 GB        ~24.4 GB to ~30.7 GB
Cache Pressure & Evictions   Zero churn (Permanent residency)    Moderate churn at B=8       Intense continuous churn
DDR5 Interconnect Bandwidth  64 GB/s (0.064 bytes/ns)            64 GB/s (0.064 bytes/ns)    64 GB/s (0.064 bytes/ns)
Transfer Time per Expert     1,376,256 ns (1.376 ms)             1,376,256 ns (1.376 ms)     1,376,256 ns (1.376 ms)
Inter-Stage Compute Horizon  20 to 50 microseconds               20 to 50 microseconds       20 to 50 microseconds
Evaluated Predictors         Baseline, ST-MoE, CSTP              Baseline, ST-MoE, CSTP      Baseline, ST-MoE, CSTP
Prediction Horizons          H = 1, H = 2                        H = 1, H = 2                H = 1, H = 2
Total Evaluated Runs         20 Runs                             26 Runs                     27 Runs
========================================================================================================================
```

---

## 2. Large-Workload Campaign Matrix (Workloads W1 through W7)

The large-workload campaign constitutes the primary evaluation benchmark. It models high-concurrency production serving on an A100 accelerator where the 35 GB HBM cache is heavily oversubscribed by the 40-stage model.

| Workload ID | Batch Size ($B$) | Sequence Length ($T$) | MoE Stages ($M$) | Total Tokens | Raw Selections | Coalesced Demands | Cache Budget | Tested Configurations |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **W1** | 14 | 4 | 40 | 56 | 17,920 | 4,372 | 35 GB | Baseline, ST-MoE ($H=1, 2$), CSTP ($H=1, 2$) |
| **W2** | 14 | 16 | 40 | 224 | 71,680 | 17,440 | 35 GB | Baseline, ST-MoE ($H=1, 2$), CSTP ($H=1, 2$) |
| **W3** | 32 | 16 | 40 | 512 | 163,840 | 39,872 | 35 GB | Baseline, ST-MoE ($H=2$), CSTP ($H=2$) |
| **W4** | 48 | 16 | 40 | 768 | 245,760 | 59,808 | 35 GB | Baseline, ST-MoE ($H=2$), CSTP ($H=2$) |
| **W5** | 70 | 16 | 40 | 1,120 | 358,400 | 87,208 | 35 GB | Baseline, ST-MoE ($H=1, 2$), CSTP ($H=1, 2$) |
| **W6** | 70 | 32 | 40 | 2,240 | 716,800 | 174,380 | 35 GB | Baseline, ST-MoE ($H=2$), CSTP ($H=2$) |
| **W7** | 70 | 64 | 40 | 4,480 | 1,433,600 | 348,657 | 35 GB | Baseline, ST-MoE ($H=2$), CSTP ($H=2$) |

### Concurrency and Sequence Sub-Studies
- **Batch Scaling Sub-Study (Fixed $T=16$, $H=2$):** Workloads W2 ($B=14$) $\to$ W3 ($B=32$) $\to$ W4 ($B=48$) $\to$ W5 ($B=70$). Evaluates memory reuse across parallel prompt streams and queue contention.
- **Sequence Scaling Sub-Study (Fixed $B=70$, $H=2$):** Workloads W5 ($T=16$) $\to$ W6 ($T=32$) $\to$ W7 ($T=64$). Evaluates online adaptation warmup, temporal prefetch reuse, and accumulated stall reduction.
- **Horizon Sensitivity Sub-Study ($H=1$ vs $H=2$):** Tested on W1 ($B=14, T=4$), W2 ($B=14, T=16$), and W5 ($B=70, T=16$).

---

## 3. Original 20-Run Sensitivity Campaign Matrix

The original campaign benchmarked small batch sizes ($B \in \{1, 2, 4, 8\}$) on a truncated 3-stage model under an unconstrained 80 GB cache budget.

| Workload ID | Batch Size ($B$) | Sequence Length ($T$) | MoE Stages ($M$) | Total Tokens | Raw Selections | Coalesced Demands | Cache Budget | Tested Configurations |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ORIG_B1** | 1 | 4 | 3 | 4 | 96 | 56 | 80 GB | Baseline, ST-MoE ($H=1, 2$), CSTP ($H=1, 2$) |
| **ORIG_B2** | 2 | 4 | 3 | 8 | 192 | 60 | 80 GB | Baseline, ST-MoE ($H=1, 2$), CSTP ($H=1, 2$) |
| **ORIG_B4** | 4 | 4 | 3 | 16 | 384 | 66 | 80 GB | Baseline, ST-MoE ($H=1, 2$), CSTP ($H=1, 2$) |
| **ORIG_B8** | 8 | 4 | 3 | 32 | 768 | 74 | 80 GB | Baseline, ST-MoE ($H=1, 2$), CSTP ($H=1, 2$) |

**Key Takeaway:** Because the active working set was only ~2.11 GB, all demand-loaded experts permanently remained in the 80 GB cache after token 0. Zero cache evictions occurred, and prefetching could not provide measurable latency improvements.

---

## 4. Phase 2 Multi-Stage Scaling Campaign Matrix

Phase 2 investigated the transition from 3 stages to the full 40-stage topology under small batch sizes ($B \in \{1, 2, 4, 8\}$) and varied cache allocations.

| Workload ID | Batch Size ($B$) | Sequence Length ($T$) | MoE Stages ($M$) | Cache Budget | Tested Configurations | Primary Focus |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **P2_B1** | 1 | 4 | 40 | 30 GB | Baseline, ST-MoE ($H=1, 2$), CSTP ($H=1, 2$) | Single sequence depth scaling |
| **P2_B2** | 2 | 4 | 40 | 30 GB | Baseline, ST-MoE ($H=1, 2$), CSTP ($H=1, 2$) | Concurrency transition |
| **P2_B4** | 4 | 4 | 40 | 30 GB | Baseline, ST-MoE ($H=1, 2$), CSTP ($H=1, 2$) | Medium concurrency |
| **P2_B8** | 8 | 4 | 40 | 30 GB | Baseline, ST-MoE ($H=1, 2$), CSTP ($H=1, 2$) | Working set saturation |
| **P2_REF80G** | 1 | 4 | 40 | 80 GB | Baseline, ST-MoE ($H=1$), CSTP ($H=1$) | Unconstrained cache reference |
| **P2_T8** | 1 | 8 | 40 | 30 GB | Baseline, ST-MoE ($H=1$), CSTP ($H=1$) | Sequence length extension |
| **CACHE_SWEEP** | 8 | 4 | 40 | 25, 35, 50, 80 GB | Baseline, CSTP | Cache capacity threshold curve |

