# Component-Wise Energy Analysis: Macro Subsystems and Microarchitectural Breakdown

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 7 — Energy Analysis  
**Document:** Granular Component-Wise Energy Accounting Report  
**Author:** Mohman / Advanced Agentic Coding Assistant  
**Date:** November 11, 2026  

---

## 1. Executive Summary

This report delivers a granular, verified component-wise breakdown of energy consumption across the entire LLM MoE inference system. By decomposing energy into its microarchitectural constituents—GPU HBM2e active memory, GPU Tensor Core compute, off-chip DDR5 physical bus data movement, and static background DRAM power—we expose the fundamental drivers of energy expenditure under the MoE Memory Wall.

### Key Microarchitectural Findings:
1. **Off-Chip DDR5 Dominance:** Data movement across the off-chip DDR5 bus is the single largest energy sink in the system, consuming **51.8%** ($4,873.91\text{ J}$) of total modeled system energy in the complete system (Anchor D).
2. **Static Background DRAM Power:** Continuous DRAM standby and auto-refresh power represents the second largest component, accounting for **36.4%** ($3,427.32\text{ J}$) of total system energy.
3. **On-Chip Memory & Compute Subordination:** Local GPU HBM2e memory consumes **11.1%** ($1,044.18\text{ J}$), while active Tensor Core MAC arithmetic consumes only **0.8%** ($71.25\text{ J}$) of system energy.
4. **The MoE Memory Wall Imbalance:** The system expends **131.2 Joules of memory and bus energy for every single Joule of arithmetic compute energy** ($E_{\text{mem}} / E_{\text{comp}} \approx 131.2$).
5. **Component Contribution of FA-LRU:** Upgrading the cache replacement policy from standard LRU (Config C) to FA-LRU $W=64$ (Config D) eliminates 22 expert reloads, avoiding $2,025,848,832\text{ bytes}$ ($2.026\text{ GB} = 1.887\text{ GiB}$) of DDR5 traffic and saving **$0.326\text{ Joules}$** of net system energy.

---

## 2. Macro Subsystem Breakdown across Anchor Configurations

The Anchor workload represents our primary experimental reference point: Batch Size $B=32$, Context Length $P=2048$, Tokens Generated $T=128$, Prefetch Budget $K=4$, and Lookahead $H=1$ (Total Tokens = $4,096$).

### Table 1: Macro Subsystem Energy Accounting (Joules)

| Macro Subsystem Component | Baseline A (`anc_c1`) | Pred-Only B (`anc_c2`) | Prefetch + LRU C (`anc_c3`) | Complete System D (`anc_c4`) | Share of System D (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GPU HBM2e Active DRAM** | $1,042.87\text{ J}$ | $1,042.87\text{ J}$ | $1,044.16\text{ J}$ | $1,044.18\text{ J}$ | $11.09\%$ |
| — *HBM Row Activations ($E_{\text{ACT}}$)* | $50.36\text{ J}$ | $50.36\text{ J}$ | $50.42\text{ J}$ | $50.42\text{ J}$ | $0.54\%$ |
| — *HBM Column Reads ($E_{\text{READ}}$)* | $989.49\text{ J}$ | $989.49\text{ J}$ | $990.72\text{ J}$ | $990.73\text{ J}$ | $10.52\%$ |
| — *HBM Column Writes ($E_{\text{WRITE}}$)* | $3.03\text{ J}$ | $3.03\text{ J}$ | $3.03\text{ J}$ | $3.03\text{ J}$ | $0.03\%$ |
| **GPU Tensor Core MAC Compute** | $71.25\text{ J}$ | $71.25\text{ J}$ | $71.25\text{ J}$ | $71.25\text{ J}$ | $0.76\%$ |
| **Subtotal: Native GPU Energy** | $\mathbf{1,114.12\ J}$ | $\mathbf{1,114.12\ J}$ | $\mathbf{1,115.41\ J}$ | $\mathbf{1,115.43\ J}$ | $\mathbf{11.85\%}$ |
| **Off-Chip DDR5 Bus Active Energy** | $4,820.77\text{ J}$ | $4,820.77\text{ J}$ | $4,874.11\text{ J}$ | $4,873.91\text{ J}$ | $51.76\%$ |
| — *Demand DDR5 Transfer* | $4,820.77\text{ J}$ | $4,820.77\text{ J}$ | $3,779.21\text{ J}$ | $3,781.09\text{ J}$ | $40.15\%$ |
| — *Speculative Prefetch Transfer* | $0.00\text{ J}$ | $0.00\text{ J}$ | $1,094.90\text{ J}$ | $1,092.82\text{ J}$ | $11.61\%$ |
| **Static Background DRAM Power** | $3,452.72\text{ J}$ | $3,452.72\text{ J}$ | $3,427.46\text{ J}$ | $3,427.32\text{ J}$ | $36.40\%$ |
| **Host CPU Predictor Overhead** | $0.00\text{ J}$ | $<0.05\text{ J}$ | $<0.05\text{ J}$ | $<0.05\text{ J}$ | $<0.001\%$ |
| **Total Modeled System Energy** | $\mathbf{9,387.61\ J}$ | $\mathbf{9,387.61\ J}$ | $\mathbf{9,416.98\ J}$ | $\mathbf{9,416.66\ J}$ | $\mathbf{100.00\%}$ |
| **Relative $\Delta$ vs. Baseline A** | **Ref (100.0%)** | **+0.000%** | **+0.313%** | **+0.309%** | — |

*Arithmetic Verification:* In Complete System D, $1044.180968 + 71.245508 + 4873.910433 + 3427.321050 = 9416.657933\text{ J}$ ($0.00\text{ J}$ residual).  
*Visual References:* See [Figure 1](file:///home/mohman/ca_project/mohman/point_7/figures/png/fig1_anchor_system_energy.png) for macro subsystem comparisons and [Figure 12](file:///home/mohman/ca_project/mohman/point_7/figures/png/fig12_component_pie_breakdown.png) for the pie chart distribution.

---

## 3. Deep Dive into Subsystem Behaviors

### 3.1. Local GPU HBM2e Memory System

The local GPU HBM2e memory subsystem consumes $1,044.18\text{ J}$ in Anchor D:
- **Read Dominance:** Column reads ($990.73\text{ J}$) represent $94.9\%$ of HBM energy, driven by continuous weight streaming and KV cache fetches during token generation.
- **Row Activations:** HBM row activations ($E_{\text{ACT}} = 50.42\text{ J}$) consume $4.8\%$ of HBM energy, reflecting efficient bank-level locality and high row-buffer hit rates within the HBM memory controller.
- **Writes:** Memory writes ($3.03\text{ J}$) consume less than $0.3\%$ of HBM energy, corresponding exclusively to KV cache appends for each newly generated token.
- **Dynamic Responding to Prefetching:** In Configurations C and D, native HBM energy increases slightly by $+1.31\text{ J}$ (+0.12%) relative to Baseline A ($1,044.18\text{ J}$ vs. $1,042.87\text{ J}$) due to row activations and line reads associated with speculative expert cache occupancy.

### 3.2. Tensor Core MAC Compute Subsystem

The native arithmetic compute energy is $71.25\text{ J}$, calculated as:
$$E_{\text{MAC}} = \text{FLOPs} \times 0.23\text{ pJ/FLOP}$$
Despite performing billions of matrix multiplications across 4,096 tokens, arithmetic compute accounts for only **0.76%** of total system energy. This empirical result is the quantitative embodiment of the MoE Memory Wall: the compute engines are energy-efficient, but starved for parameters trapped behind the memory interface.

### 3.3. Layer-Wise Microarchitectural Breakdown (Native Energy)

The native simulator profiles energy across three primary module categories: Fully Connected (`fc`), Multi-Head Attention (`attn`), and Mixture-of-Experts (`moe`).

### Table 2: Layer-Wise Module Energy Breakdown (Complete System D, Anchor)

| Module Type | DRAM Energy (HBM) | Compute Energy (MAC) | Total Module Energy | Share of On-Chip Energy |
| :--- | :--- | :--- | :--- | :--- |
| **Mixture-of-Experts (`moe`)** | $959.06\text{ J}$ | $30.01\text{ J}$ | $989.07\text{ J}$ | **88.67%** |
| **Self-Attention (`attn`)** | $38.64\text{ J}$ | $29.48\text{ J}$ | $68.12\text{ J}$ | **6.11%** |
| **Dense Projection (`fc`)** | $46.48\text{ J}$ | $11.76\text{ J}$ | $58.24\text{ J}$ | **5.22%** |
| **Total On-Chip** | $\mathbf{1,044.18\ J}$ | $\mathbf{71.25\ J}$ | $\mathbf{1,115.43\ J}$ | **100.00%** |

- **MoE Dominance:** The MoE layers account for **88.67%** of all on-chip energy, with MoE DRAM reads ($959.06\text{ J}$) overwhelmingly dominating.
- **Attention Compute vs. DRAM:** In the attention module, compute ($29.48\text{ J}$) and DRAM ($38.64\text{ J}$) are relatively balanced at context length $P=2048$.

### 3.4. Off-Chip DDR5 Bus Active Energy

Off-chip DDR5 bus active energy is governed by:
$$E_{\text{DDR5}} = \text{Bytes}_{\text{DDR5}} \times 0.1\text{ nJ/byte} = \text{Bytes} \times 1.0 \times 10^{-10}\text{ J/byte}$$

1. **Baseline A:** Consumes $4,820.77\text{ J}$ across $48.21\text{ TB}$ of demand-miss traffic. There is zero speculative traffic.
2. **Complete System D ($K=4$):**
   - Demand traffic decreases from $48.21\text{ TB}$ to $37.81\text{ TB}$, saving $1,039.68\text{ J}$ in demand bus energy.
   - Speculative prefetch traffic injects $10.93\text{ TB}$, consuming $1,092.82\text{ J}$.
   - **Net DDR5 Bus Delta:** $+53.14\text{ J}$ ($+1.10\%$).
   - Speculative prefetch energy represents **22.4%** of total DDR5 transfer energy.
3. **Linearity:** As demonstrated in [Figure 4](file:///home/mohman/ca_project/mohman/point_7/figures/png/fig4_ddr5_traffic_vs_energy.png), the correlation between off-chip traffic and bus energy is strictly linear ($100\text{ J/TB}$ decimal).

### 3.5. Static DRAM Background Power

Static power is modeled as $P_{\text{static}} = 4.5\text{ W}$ over the simulated execution duration:
- **Baseline A:** Runtime $t_{\text{sim}} = 767.27\text{ s} \implies E_{\text{static}} = 3,452.72\text{ J}$.
- **Complete System D:** Runtime $t_{\text{sim}} = 761.63\text{ s} \implies E_{\text{static}} = 3,427.32\text{ J}$.
- **Static Energy Savings:** Prefetching shortens the simulation by $5.64\text{ seconds}$, saving **$25.40\text{ J}$ of static energy**.
- At $K=2$ (the sweet spot), runtime is shortened by $10.93\text{ seconds}$, saving **$49.18\text{ J}$ of static energy**.

---

## 4. Component Contribution: LRU (Config C) vs. FA-LRU (Config D)

Comparing Prefetch + LRU (Config C, `anc_c3`) against Complete System D (FA-LRU $W=64$, `anc_c4`) directly isolates the physical impact of Frequency-Aware eviction under identical prefetching.

### Table 3: Isolated Impact of FA-LRU ($W=64$) at Anchor ($B=32$)

| Metric | Prefetch + LRU (Config C) | Complete FA-LRU (Config D) | Delta (D minus C) |
| :--- | :--- | :--- | :--- |
| **Resident Hits** | 1,499 | 1,521 | **+22 hits** |
| **Expert Reloads** | 535,953 | 535,931 | **-22 reloads** |
| **Total DDR5 Traffic** | 48,741,130,174,464 B | 48,739,104,325,632 B | **-2,025,848,832 B (-2.026 GB = -1.887 GiB)** |
| — *Demand DDR5 Traffic* | 37,792,121,880,576 B | 37,810,883,002,368 B | $+18,761,121,792\text{ B}$ |
| — *Speculative DDR5 Traffic*| 10,949,008,293,888 B | 10,928,221,323,264 B | $-20,786,970,624\text{ B}$ |
| **DDR5 Active Energy** | $4,874.11\text{ J}$ | $4,873.91\text{ J}$ | **-0.203 J** |
| **Simulated Runtime** | $761.66\text{ s}$ | $761.63\text{ s}$ | **-0.032 s (-31.6 ms)** |
| **Static DRAM Energy** | $3,427.46\text{ J}$ | $3,427.32\text{ J}$ | **-0.142 J** |
| **Native GPU Energy** | $1,115.41\text{ J}$ | $1,115.43\text{ J}$ | **+0.019 J** |
| **Total System Energy** | $\mathbf{9,416.98\ J}$ | $\mathbf{9,416.66\ J}$ | **-0.326 J (-0.0035%)** |
| **System EDP** | $7,172,526\text{ J}\cdot\text{s}$ | $7,171,980\text{ J}\cdot\text{s}$ | **-546 J}\cdot\text{s (-0.0076%)** |

### Mechanism of FA-LRU Energy Benefit:
1. By protecting high-frequency experts in the frequency window ($W=64$), FA-LRU prevents the CSTP prefetcher from polluting the cache with low-utility speculative candidates.
2. This preserves 22 resident hits, eliminating 22 redundant expert reloads across the DDR5 bus.
3. This saves $2.026\text{ GB}$ ($1.887\text{ GiB}$) of physical bus traffic, translating directly to a **$0.326\text{ Joule}$ net system energy saving** and a **$546\text{ J}\cdot\text{s}$ EDP reduction**.
