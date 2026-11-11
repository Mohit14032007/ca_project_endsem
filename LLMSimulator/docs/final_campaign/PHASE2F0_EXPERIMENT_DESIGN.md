# Phase 2F.0 — Synthetic Compute-Window Sensitivity Experiment Design

## 1. Executive Summary

This document specifies the rigorous experimental design for the **Phase 2F Synthetic Compute-Window Sensitivity Experiment**. The objective of Phase 2F is to isolate and evaluate the latent latency-hiding capability of forward-lookahead prefetching by synthetically expanding the available computation window across a sweep of controlled compute-time multipliers:
$$M \in \{1\times, 4\times, 8\times, 12\times, 16\times\}$$

By comparing each prefetch configuration against a strictly matched demand-only baseline at every multiplier ($B(M)$), this experiment establishes whether forward-lookahead prefetching can achieve timely resident hits and expose measurable memory stall reductions once the compute window approaches and exceeds physical transfer thresholds.

---

## 2. Theoretical Timing Thresholds and Multiplier Bounds

### 2.1 Physical Parameters
- **Calibrated Expert Footprint:** $88,080,384\text{ bytes}$ ($84.0\text{ MB}$, three projection tensors of $29,360,128\text{ bytes}$).
- **Calibrated DDR5 Bandwidth:** $22.578\text{ GB/s}$ ($22.578\text{ bytes/ns}$).
- **Isolated Transfer Duration:**
  $$t_{\text{transfer}} = \frac{88,080,384\text{ bytes}}{22.578 \times 10^9\text{ bytes/s}} = 3.90116\text{ ms}$$
- **Original Stage Compute Duration:**
  $$t_{\text{stage}} \approx 0.18317\text{ ms} \quad (54.95\text{ ms} \text{ total compute across } 300\text{ MoE stages in Workload A1})$$

### 2.2 Idealized Lower-Bound Minimum Multipliers
The compute window available over a forward lookahead horizon $H$ with compute multiplier $M$ is:
$$W(H, M) = H \times (t_{\text{stage}} \times M)$$
For prefetch transfers to complete strictly before demand arrives, the available window must satisfy $W(H, M) \ge t_{\text{transfer\_total}}$.

#### Case A: Single Expert Transfer ($t_{\text{transfer}} = 3.90116\text{ ms}$)
$$M_{\text{min}}(1 \text{ expert}) = \frac{3.90116\text{ ms}}{H \times 0.18317\text{ ms}}$$
- **Horizon $H=1$:** $M_{\text{min}} = \frac{3.90116}{0.18317} = \mathbf{21.30\times}$
- **Horizon $H=2$:** $M_{\text{min}} = \frac{3.90116}{2 \times 0.18317} = \mathbf{10.65\times}$
- **Horizon $H=3$:** $M_{\text{min}} = \frac{3.90116}{3 \times 0.18317} = \mathbf{7.10\times}$

#### Case B: Four Serialized Expert Transfers ($B=4$, $t_{\text{4-transfers}} = 4 \times 3.90116\text{ ms} = 15.60464\text{ ms}$)
$$M_{\text{min}}(4 \text{ experts}) = \frac{15.60464\text{ ms}}{H \times 0.18317\text{ ms}}$$
- **Horizon $H=1$:** $M_{\text{min}} = \frac{15.60464}{0.18317} = \mathbf{85.19\times}$
- **Horizon $H=2$:** $M_{\text{min}} = \frac{15.60464}{2 \times 0.18317} = \mathbf{42.60\times}$
- **Horizon $H=3$:** $M_{\text{min}} = \frac{15.60464}{3 \times 0.18317} = \mathbf{28.40\times}$

> [!WARNING]
> **Strict Lower Bounds Only:** These calculations represent idealized theoretical lower bounds under zero queue contention, zero channel serialization behind competing demand traffic, and line-rate bus efficiency. In practice, competing demand transfers preempt speculative transfers, and FIFO queuing behind uncancelled requests delays service start.

---

## 3. Experiment Matrix and Control Specification

### 3.1 Sweep Parameters
- **Compute Multipliers ($5$ levels):** $1\times, 4\times, 8\times, 12\times, 16\times$.
- **Predictor Configurations ($2$ families):** ST-MoE (`FWD-S-H2`) and CSTP (`FWD-C-H2`).
- **Baseline Configurations ($1$ family):** Matched Demand-only baseline (`B(M)`) for every multiplier.
- **Fixed Horizon:** $H = 2$.
- **Fixed Candidate Budget:** $B = 4$.
- **Fixed Memory Capacity:** $80\text{ GB}$ ($975$ experts), LRU replacement, PABI quarantine + Belády pinning.
- **Fixed Workload:** Workload A1 (`ms_task12_t128_l75.csv`, 4 decoding tokens, batch size 1).
- **Total Experiment Matrix:** $5 \text{ multipliers} \times 3 \text{ configurations} = \mathbf{15\text{ runs}}$.

### 3.2 Full 15-Run Execution Matrix

| Run ID | Multiplier | Predictor | Horizon ($H$) | Budget ($B$) | Config Designation | Expected Compute Window ($H=2$) | Single-Transfer Feasibility ($3.90\text{ ms}$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | $1\times$ | None (Demand) | N/A | N/A | `B-M1` | $0.366\text{ ms}$ | Baseline Reference |
| **02** | $1\times$ | ST-MoE | $2$ | $4$ | `FWD-S-H2-M1` | $0.366\text{ ms}$ | Impossible ($0.366 \ll 3.901$) |
| **03** | $1\times$ | CSTP | $2$ | $4$ | `FWD-C-H2-M1` | $0.366\text{ ms}$ | Impossible ($0.366 \ll 3.901$) |
| **04** | $4\times$ | None (Demand) | N/A | N/A | `B-M4` | $1.465\text{ ms}$ | Baseline Reference |
| **05** | $4\times$ | ST-MoE | $2$ | $4$ | `FWD-S-H2-M4` | $1.465\text{ ms}$ | Deficit ($1.465 < 3.901$) |
| **06** | $4\times$ | CSTP | $2$ | $4$ | `FWD-C-H2-M4` | $1.465\text{ ms}$ | Deficit ($1.465 < 3.901$) |
| **07** | $8\times$ | None (Demand) | N/A | N/A | `B-M8` | $2.931\text{ ms}$ | Baseline Reference |
| **08** | $8\times$ | ST-MoE | $2$ | $4$ | `FWD-S-H2-M8` | $2.931\text{ ms}$ | Deficit ($2.931 < 3.901$) |
| **09** | $8\times$ | CSTP | $2$ | $4$ | `FWD-C-H2-M8` | $2.931\text{ ms}$ | Deficit ($2.931 < 3.901$) |
| **10** | $12\times$ | None (Demand) | N/A | N/A | `B-M12` | $4.396\text{ ms}$ | Baseline Reference |
| **11** | $12\times$ | ST-MoE | $2$ | $4$ | `FWD-S-H2-M12` | $4.396\text{ ms}$ | **FEASIBLE ($4.396 > 3.901$)** |
| **12** | $12\times$ | CSTP | $2$ | $4$ | `FWD-C-H2-M12` | $4.396\text{ ms}$ | **FEASIBLE ($4.396 > 3.901$)** |
| **13** | $16\times$ | None (Demand) | N/A | N/A | `B-M16` | $5.861\text{ ms}$ | Baseline Reference |
| **14** | $16\times$ | ST-MoE | $2$ | $4$ | `FWD-S-H2-M16` | $5.861\text{ ms}$ | **FEASIBLE ($5.861 > 3.901$)** |
| **15** | $16\times$ | CSTP | $2$ | $4$ | `FWD-C-H2-M16` | $5.861\text{ ms}$ | **FEASIBLE ($5.861 > 3.901$)** |

---

## 4. Evaluation Metrics and Conservation Invariants

### 4.1 Primary Operational Metrics
For each of the 15 runs, the following metrics must be extracted and reported:
1. **Execution Latency:**
   - Simulated Execution Time ($T_{\text{exec}}$, $ms$).
   - Exposed Memory Stall ($T_{\text{stall}}$, $ms$).
   - Stall Reduction vs Matched Baseline: $\Delta T_{\text{stall}} = T_{\text{stall}}(B(M)) - T_{\text{stall}}(\text{FWD}(M))$.
   - Execution Speedup vs Matched Baseline: $\frac{T_{\text{exec}}(B(M)) - T_{\text{exec}}(\text{FWD}(M))}{T_{\text{exec}}(B(M))} \times 100\%$.
2. **Prefetch Effectiveness:**
   - Timely Resident Hits ($H_{\text{timely}}$).
   - Late In-Flight Attachments ($A_{\text{late}}$).
   - Natural Demand Hits ($H_{\text{natural}}$).
   - Unprefetched Demand Misses ($M_{\text{unprefetched}}$).
   - Prefetch Conversion Rate: $\frac{H_{\text{timely}}}{H_{\text{timely}} + A_{\text{late}} + \text{Unused}}$.
3. **Queue and Channel Dynamics:**
   - Average DDR5 Queue Waiting Time ($ms$).
   - Average Remaining Transfer Service Time at Demand ($ms$).
   - Total DDR5 Demand Traffic ($MB$).
   - Total DDR5 Speculative Traffic ($MB$).
4. **Predictor Quality:**
   - Prediction Precision ($\%$), Recall ($\%$), and F1 Score ($\%$).

### 4.2 Mandatory Conservation Checks
Both strict conservation laws established in Phase 2E.3 must hold for every run:
1. **Demand-Outcome Conservation:**
   $$\text{Total Lookups } (7,200) = H_{\text{natural}} + H_{\text{timely}} + A_{\text{late}} + M_{\text{unprefetched}}$$
2. **Transfer-Lifecycle Conservation:**
   $$\text{Speculative Submitted} = \text{Demanded} + \text{Unused} + \text{Evicted} + \text{Still In-Flight} + \text{Cancelled}$$

---

## 5. Architectural Diagnostics to Monitor During Experiment

Because $B=4$ candidates are dispatched per stage ($15.60\text{ ms}$ total transfer time), the experiment will serve as a definitive diagnostic on two competing hypotheses:

1. **Hypothesis 1 (Compute Window Sufficiency):**
   At $M=12\times$ and $M=16\times$, the available window ($4.40\text{ ms}$ and $5.86\text{ ms}$) exceeds $3.901\text{ ms}$. If queue waiting time is low, timely resident hits will emerge ($H_{\text{timely}} > 0$) and exposed memory stall will measurably decrease.
2. **Hypothesis 2 (Queue Congestion Dominance):**
   Because $B=4$ injects $15.60\text{ ms}$ of work every $t_{\text{stage}} \times M$ ms, if FIFO queue delays exceed the compute window ($t_{\text{queue}} > W(H, M)$), speculative candidates will still arrive late despite the expanded compute window. This will conclusively isolate queue congestion as the primary architectural bottleneck rather than pure compute window deficit.
