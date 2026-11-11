# Phase 2F — Experiment Protocol: Synthetic Compute-Window Sensitivity Campaign

## 1. Objective and Scope

The objective of the **Phase 2F Synthetic Compute-Window Sensitivity Campaign** is to isolate and evaluate the latent latency-hiding capability of training-free predictor-driven forward-lookahead prefetching by synthetically expanding the available computation window across a controlled multiplier sweep:
$$M \in \{1\times, 4\times, 8\times, 12\times, 16\times\}$$

The campaign determines whether forward-lookahead prefetching can achieve timely resident hits and expose measurable memory stall reductions once the inter-stage computation window approaches and exceeds the isolated DDR5 expert transfer latency ($3.90116\text{ ms}$).

---

## 2. Workload and Environment Specifications

### 2.1 Model and Workload Architecture (Workload A1)
- **Workload Designation:** Workload A1
- **Model Architecture:** DeepSeek-V3 Mixtral-style MoE
- **Task Source:** MoE-Suite Task 12 (`ms_task12_t128_l75.csv`)
- **Total Layers:** 78 layers (3 dense attention/MLP layers, 75 MoE stages)
- **Decoding Tokens:** 4 sequential autoregressive decoding iterations
- **Batch Size:** 1 sequence
- **Input Sequence Length:** 128 tokens
- **Output Sequence Length:** 128 tokens
- **Precision:** FP16 (2 bytes per parameter)
- **Routed Experts per Stage ($top\_k$):** 8 experts
- **Total Expert Routing Decisions:** $4 \text{ tokens} \times 75 \text{ stages} \times 8 \text{ experts} = 2,400 \text{ routings}$
- **Total Linear Expert Lookups:** $2,400 \times 3 \text{ tensors} = 7,200 \text{ lookups}$
- **Trace Path:** `/home/mohman/ca_project/LLMSimulator/results/endsem_phase_e/final_baseline/traces/ms_task12_t128_l75.csv`
- **Trace SHA-256 Checksum:** `5a635dece5747e78be7f600082c888a7f9e9d2ee50e1ae829d338f75f235ea1c`

### 2.2 Physical Memory and Transfer Subsystem
- **HBM Capacity:** $80.0\text{ GB}$ ($80,000,000,000\text{ bytes}$)
- **HBM Bandwidth:** $2.039\text{ TB/s}$ (NVIDIA A100 SXM4)
- **Off-Chip Host DRAM:** DDR5-4800 (Calibrated Bandwidth: $22.578\text{ GB/s}$)
- **Expert Size:** $88,080,384\text{ bytes}$ ($84.0\text{ MB}$, three projection tensors of $29,360,128\text{ bytes}$)
- **Isolated Transfer Duration:**
  $$t_{\text{transfer}} = \frac{88,080,384\text{ bytes}}{22.578 \times 10^9\text{ bytes/s}} = 3.90116\text{ ms}$$
- **Original Stage Compute Duration:**
  $$t_{\text{stage}} \approx 0.18317\text{ ms} \quad (54.95\text{ ms total across 300 MoE stages})$$

---

## 3. Experiment Matrix and Control Protocol

### 3.1 Sweep Parameters
- **5 Compute Multipliers:** $1.0\times, 4.0\times, 8.0\times, 12.0\times, 16.0\times$
- **3 Predictor Configurations:**
  1. `B`: Demand-only baseline (prefetching disabled, LRU replacement).
  2. `FWD-S-H2`: ST-MoE predictor, forward-lookahead scheduling, horizon $H=2$, budget $B=4$, confidence threshold 2, PABI quarantine + Belády pinning (max 4 pinned).
  3. `FWD-C-H2`: CSTP predictor, forward-lookahead scheduling, horizon $H=2$, budget $B=4$, $\alpha=0.5, \lambda=1.0$, PABI quarantine + Belády pinning (max 4 pinned).
- **Total Configurations:** $5 \times 3 = \mathbf{15\text{ runs}}$.

### 3.2 Theoretical Timing Feasibility Thresholds ($H=2$)

| Multiplier ($M$) | Compute Window $W(H=2, M)$ | Isolated Transfer ($3.901\text{ ms}$) | Feasibility Status |
| :---: | :---: | :---: | :--- |
| **$1.0\times$** | $0.366\text{ ms}$ | $3.901\text{ ms}$ | **Impossible** ($0.366 \ll 3.901$) |
| **$4.0\times$** | $1.465\text{ ms}$ | $3.901\text{ ms}$ | **Deficit** ($1.465 < 3.901$) |
| **$8.0\times$** | $2.931\text{ ms}$ | $3.901\text{ ms}$ | **Deficit** ($2.931 < 3.901$) |
| **$12.0\times$** | $4.396\text{ ms}$ | $3.901\text{ ms}$ | **FEASIBLE** ($4.396 > 3.901$, $+0.495\text{ ms}$ margin) |
| **$16.0\times$** | $5.861\text{ ms}$ | $3.901\text{ ms}$ | **FEASIBLE** ($5.861 > 3.901$, $+1.960\text{ ms}$ margin) |

---

## 4. Matched Baseline Pairing and Conservation Rules

### 4.1 Strict Matched Baseline Pairing Rule
Under no circumstances may a prefetching configuration running at multiplier $M$ be evaluated against the $M=1$ baseline. Every prefetching configuration $\text{FWD}(M)$ must be evaluated strictly against its corresponding demand-only baseline $B(M)$ executed at the identical multiplier:
$$\Delta T_{\text{stall}}(M) = T_{\text{stall}}(B(M)) - T_{\text{stall}}(\text{FWD}(M))$$
$$\text{Speedup}(M) = \frac{T_{\text{exec}}(B(M)) - T_{\text{exec}}(\text{FWD}(M))}{T_{\text{exec}}(B(M))} \times 100\%$$

### 4.2 Mandatory Conservation Invariant Checks
1. **Demand-Outcome Conservation:**
   $$7,200 = H_{\text{natural}} + H_{\text{timely}} + A_{\text{late}} + M_{\text{unprefetched}}$$
2. **Transfer-Lifecycle Conservation:**
   $$\text{Transfers Submitted} = \text{Demanded} + \text{Unused} + \text{Evicted} + \text{Still In-Flight} + \text{Cancelled}$$
