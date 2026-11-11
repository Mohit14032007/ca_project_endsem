# Audit Report: Final Metric Correctness and Performance Verification

**Audit Date:** November 10, 2026  
**Audited Directory:** `mohman/point_3/phase2/large_workload/`  
**Scope:** Recomputation and verification of the Eight Mandatory Prefetching Metrics, latency distributions, and throughput speedups from raw event traces.  
**Certification:** Zero ternary operators verified (`ast.IfExp == 0`); zero `? :` in C++; all file references strictly linked via `file://`.

---

## 1. Executive Summary

This audit performs an independent, ground-up verification of all reported metrics directly from the raw event files ([expert_demand_events.csv](file:///home/mohman/ca_project/mohman/point_3/phase2/large_workload/experiments/raw/campaign/w7_cstp_h2/traces/expert_demand_events.csv), [prefetch_transfer_events.csv](file:///home/mohman/ca_project/mohman/point_3/phase2/large_workload/experiments/raw/campaign/w7_cstp_h2/traces/prefetch_transfer_events.csv), and [token_latency.csv](file:///home/mohman/ca_project/mohman/point_3/phase2/large_workload/experiments/raw/campaign/w7_cstp_h2/traces/token_latency.csv)).

### Key Audit Confirmations:
1. **Strict Timely Coverage is 0.00%:** Verified. Zero speculative transfers completed transit before demand arrival.
2. **Late In-Flight Attachment Coverage is 9.23% (ST-MoE) and 4.81% (CSTP) at W7:** Verified. Every assisted demand attached to an already transmitting speculative transfer.
3. **Exposed GPU Stall Reductions are Authenticated:**
   - At W5 ($B=70, T=16$): CSTP saves **509.94 ms**, ST-MoE saves **260.57 ms**.
   - At W6 ($B=70, T=32$): CSTP saves **2,160.97 ms (2.16 s)**, ST-MoE saves **543.23 ms**.
   - At W7 ($B=70, T=64$): CSTP saves **6,744.86 ms (6.74 s)**, ST-MoE saves **2,289.03 ms (2.29 s)**.
4. **End-to-End Simulation Speedup is Authenticated:** In Workload W7, total simulated execution time drops from **488.483 s** (baseline) to **481.766 s** (CSTP), saving **6.717 seconds** of wall-clock simulation time. The demand wait reduction (6.745 s) maps almost 1:1 onto end-to-end execution reduction (6.717 s).

---

## 2. Recomputation of the Eight Mandatory Metrics

### 2.1 Metric 8.1: Prefetch Coverage
Audited from `expert_demand_events.csv`:
- `strict_timely_coverage_pct`:
  $$\frac{\text{Count of RESIDENCY\_STATUS == 'TIMELY\_PREFETCH'}}{\text{Total Demand Events}} = \frac{0}{1,045,971} = \mathbf{0.00\%}$$
- `prefetch_associated_coverage_pct`:
  $$\frac{\text{Count of RESIDENCY\_STATUS == 'LATE\_IN\_FLIGHT'}}{\text{Total Demand Events}} = \frac{50,264}{1,045,971} = \mathbf{4.8055\% \text{ (CSTP)}} \quad \Big[\frac{96,569}{1,045,971} = \mathbf{9.2325\% \text{ (ST-MoE)}}\Big]$$

### 2.2 Metrics 8.2 & 8.3: Timely Rate and Late In-Flight Rate of Matched
Audited from matched transfers:
- Timely Rate: $\frac{0}{50,264} = \mathbf{0.00\%}$
- Late Rate: $\frac{50,264}{50,264} = \mathbf{100.00\%}$

### 2.3 Metric 8.4: Unused Prefetches and Wasted Bytes
Audited from `prefetch_transfer_events.csv` for W7:
- CSTP completed unused transfers: 2,456 (resident at end) + 1,356 (evicted before demand) = **3,812 transfers**.
- CSTP wasted bytes: $3,812 \times 88,080,384 = \mathbf{335,762,423,808 \text{ bytes (335.76 GB)}}$.
- ST-MoE completed unused transfers: 1,402 (resident at end) + 4,620 (evicted before demand) = **6,022 transfers**.
- ST-MoE wasted bytes: $6,022 \times 88,080,384 = \mathbf{530,420,072,448 \text{ bytes (530.42 GB)}}$.

### 2.4 Metric 8.5: HBM Demand Hit Rates
Audited across all 1,045,971 demand rows in W7:
- Baseline: 697,314 `RESIDENT_HIT` ($66.67\%$).
- ST-MoE: 698,710 `RESIDENT_HIT` ($66.80\%$) + 96,569 `LATE_IN_FLIGHT` ($9.23\%$).
- CSTP: 699,769 `RESIDENT_HIT` ($66.90\%$) + 50,264 `LATE_IN_FLIGHT` ($4.81\%$).

### 2.5 Metric 8.6: Additional Memory Traffic
Audited from physical transfer events in W7:
- Baseline: $348,657 \times 88,080,384 = \mathbf{30.7098 \text{ TB}}$
- ST-MoE: Demand $22.0810 \text{ TB} + \text{Speculative } 9.0491 \text{ TB} = \mathbf{31.1302 \text{ TB}}$ (+420.32 GB overhead, +1.37%)
- CSTP: Demand $26.0663 \text{ TB} + \text{Speculative } 4.7630 \text{ TB} = \mathbf{30.8294 \text{ TB}}$ (+119.53 GB overhead, +0.39%)

### 2.6 Metric 8.7: Exposed GPU Memory Stall
Audited by summing `ddr5_wait_ns` in `expert_demand_events.csv` for W7:
- Baseline: $348,657 \times 1,376,256 \text{ ns} = \mathbf{479,841,288,192 \text{ ns} = 479,841.29 \text{ ms}}$
- ST-MoE: $\mathbf{477,552,261,854 \text{ ns} = 477,552.26 \text{ ms}}$ (Delta: **+2,289.03 ms stall reduction**)
- CSTP: $\mathbf{473,096,431,024 \text{ ns} = 473,096.43 \text{ ms}}$ (Delta: **+6,744.86 ms stall reduction**)

### 2.7 Metric 8.8: Expert-Load Latency Distributions
Table 2.1 verifies the latency percentiles across categories in Workload W7.

### Table 2.1: Expert-Load Latency Breakdown (W7, 40 Stages)

| Residency Status | Event Count (Baseline) | Event Count (CSTP H2) | Mean Latency ($\mu$s) | P50 Latency ($\mu$s) | P95 Latency ($\mu$s) | Max Latency ($\mu$s) |
|---|---|---|---|---|---|---|
| `RESIDENT_HIT` | 697,314 | 699,769 | 0.00 $\mu$s | 0.00 $\mu$s | 0.00 $\mu$s | 0.00 $\mu$s |
| `DEMAND_MISS` | 348,657 | 295,938 | 1,376.26 $\mu$s | 1,376.26 $\mu$s | 1,376.26 $\mu$s | 1,376.26 $\mu$s |
| `LATE_IN_FLIGHT` | 0 | 50,264 | **1,309.29 $\mu$s** | 1,376.26 $\mu$s | 1,376.26 $\mu$s | 1,376.26 $\mu$s |
| **Overall Demands** | **1,045,971** | **1,045,971** | **452.30 $\mu$s** (vs 458.75) | 0.00 $\mu$s | 1,376.26 $\mu$s | 1,376.26 $\mu$s |

Late attachments reduce mean miss wait from 1,376.26 $\mu$s to **1,309.29 $\mu$s** (saving $66.97 \mu$s per attached event in CSTP).

---

## 3. End-to-End Simulation Speedup and Throughput Verification

Table 3.1 details the verified end-to-end execution metrics for Workload W7 from [token_latency.csv](file:///home/mohman/ca_project/mohman/point_3/phase2/large_workload/experiments/raw/campaign/w7_cstp_h2/traces/token_latency.csv).

### Table 3.1: Workload W7 Performance and Throughput Reconciliation

| Metric | W7 Baseline | W7 ST-MoE ($H=2$) | W7 CSTP ($H=2$) | CSTP Delta vs Baseline |
|---|---|---|---|---|
| **Total Generated Tokens** | 4,480 | 4,480 | 4,480 | 0 |
| **Simulation End Time (s)** | 488.483 s | 486.210 s | **481.766 s** | **-6.717 s speedup** |
| **Sum of Demand Waits (s)** | 479.841 s | 477.552 s | **473.096 s** | **-6.745 s stall reduction** |
| **Mean Token Latency (ms)** | 7,632.55 ms | 7,597.03 ms | **7,527.60 ms** | **-104.95 ms/token** |
| **Median Token Latency (ms)** | 9,081.67 ms | 9,033.12 ms | **8,943.99 ms** | **-137.68 ms/token** |
| **P95 Token Latency (ms)** | 9,678.41 ms | 9,640.59 ms | **9,522.71 ms** | **-155.70 ms/token** |
| **Serving Throughput (tok/s)** | 9.17 tok/s | 9.21 tok/s | **9.30 tok/s** | **+0.13 tok/s (+1.42%)** |

### 3.1 Overlap and Critical-Path Timing Verification
The audit examined why the stall reduction (6,744.86 ms) corresponds so closely to the end-to-end simulation time reduction (6,717.00 ms):
1. In auto-regressive decode generation, MoE expert GEMMs lie directly on the sequential critical path.
2. When the memory subsystem stalls on an expert miss, GPU tensor cores sit completely idle waiting for DDR5 data.
3. Every millisecond saved by attaching to an in-flight transfer advances the dispatch of subsequent layers.
4. Hence, memory stall savings translate almost 100% directly into wall-clock serving latency reductions.

---

## 4. Conclusion

All eight mandatory prefetching metrics and performance speedups have been certified against raw event traces. The claimed 6.74-second stall reduction and 104.95-ms mean token latency reduction are verified by timestamp arithmetic.

