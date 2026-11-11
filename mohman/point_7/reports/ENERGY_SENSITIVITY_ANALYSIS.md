# Energy Sensitivity Analysis: Concurrency, Sequence Length, Lookahead, and Prefetch Budget

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 7 — Energy Analysis  
**Document:** Sensitivity Analysis of Energy Metrics across Architectural Sweeps (Reconciled & Audited)  
**Author:** Mohman / Advanced Agentic Coding Assistant  
**Date:** November 11, 2026  

---

## 1. Executive Summary

This report systematically analyzes how system energy, energy per token, and energy-delay product (EDP) respond to four key architectural and workload dimensions:
1. **Batch Size Concurrency ($B \in \{4, 14, 32\}$)**
2. **Initial Context Length ($P \in \{128, 512, 2048, 4096\}$)**
3. **Prefetch Lookahead Horizon ($H \in \{1, 2, 4\}$)**
4. **Prefetch Budget Aggressiveness ($K \in \{0, 2, 4, 8\}$)**

### Core Sensitivity Insights:
- **Massive Concurrency Amortization:** Increasing batch size from $B=4$ to $B=32$ amortizes fixed system overheads and model parameter transfers across more concurrent tokens, cutting system energy per token by **50.0%** (from $4,582.70\text{ mJ/tok}$ to $2,291.90\text{ mJ/tok}$).
- **The Low-Concurrency Inversion ($B=4$):** At low concurrency ($B=4$), prefetching experiences an inversion: it increases execution time by $+0.13\%$ and increases energy by $+1.371\%$ ($+32.17\text{ J}$), degrading EDP by **+1.503%**. Cache capacity is underutilized, and speculative traffic introduces bus contention.
- **Context Length KV Cache Energy Scaling:** Extending context length from $P=128$ to $P=4096$ increases attention compute energy by **$6.73\times$** ($7.82\text{ J} \to 52.59\text{ J}$) and attention HBM DRAM access energy by **$1.86\times$** ($27.30\text{ J} \to 50.68\text{ J}$), strictly matching the $O(P+t)$ decoding formulation. However, off-chip MoE DDR5 traffic and prefetching deltas remain invariant across prompt lengths.
- **Lookahead Diminishing Returns ($H$):** Increasing lookahead from $H=1$ to $H=4$ lowers exposed stall time from $747.58\text{ s}$ to $746.15\text{ s}$ and saves $14.88\text{ J}$ of system energy, but yields diminishing returns as multi-step prediction uncertainty increases.

---

## 2. Batch Size Sensitivity ($B \in \{4, 14, 32\}$)

The batch size determines concurrency, memory reuse in the expert cache, and the amortization of memory transfers across tokens. All runs evaluate $P=2048, T=128, K=4, H=1$.

### Table 1: Batch Size Scaling Ledger

| Parameter | $B=4$ Baseline (`b4_c1`) | $B=4$ Complete (`b4_c4`) | $B=14$ Baseline (`b14_c1`) | $B=14$ Complete (`b14_c4`) | $B=32$ Baseline (`anc_c1`) | $B=32$ Complete (`anc_c4`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Tokens** | 512 | 512 | 1,792 | 1,792 | 4,096 | 4,096 |
| **Simulated Runtime ($t_{\text{sim}}$)** | $187.76\text{ s}$ | $188.01\text{ s}$ | $459.16\text{ s}$ | $456.49\text{ s}$ | $767.27\text{ s}$ | $761.63\text{ s}$ |
| **Runtime Delta** | Ref | **+0.130%** | Ref | **-0.582%** | Ref | **-0.736%** |
| **Exposed Stall ($t_{\text{stall}}$)** | $183.51\text{ s}$ | $183.76\text{ s}$ | $450.38\text{ s}$ | $447.69\text{ s}$ | $753.25\text{ s}$ | $747.58\text{ s}$ |
| **DDR5 Total Traffic** | $11.74\text{ TB}$ | $12.06\text{ TB}$ | $28.82\text{ TB}$ | $29.22\text{ TB}$ | $48.21\text{ TB}$ | $48.74\text{ TB}$ |
| **DDR5 Active Energy** | $1,174.49\text{ J}$ | $1,205.72\text{ J}$ | $2,882.44\text{ J}$ | $2,922.49\text{ J}$ | $4,820.77\text{ J}$ | $4,873.91\text{ J}$ |
| — *DDR5 Active Delta* | Ref | **+31.23 J (+2.66%)** | Ref | **+40.05 J (+1.39%)** | Ref | **+53.14 J (+1.10%)** |
| **Static DRAM Energy** | $844.92\text{ J}$ | $846.03\text{ J}$ | $2,066.23\text{ J}$ | $2,054.19\text{ J}$ | $3,452.72\text{ J}$ | $3,427.32\text{ J}$ |
| **Native GPU Energy** | $318.03\text{ J}$ | $317.86\text{ J}$ | $654.11\text{ J}$ | $655.54\text{ J}$ | $1,114.12\text{ J}$ | $1,115.43\text{ J}$ |
| **Total System Energy** | $\mathbf{2,346.34\ J}$ | $\mathbf{2,378.51\ J}$ | $\mathbf{5,633.94\ J}$ | $\mathbf{5,663.39\ J}$ | $\mathbf{9,387.61\ J}$ | $\mathbf{9,416.66\ J}$ |
| — *System Energy Delta* | **Ref** | $\mathbf{+32.17\ J\ (+1.371\%)}$ | **Ref** | $\mathbf{+29.45\ J\ (+0.523\%)}$ | **Ref** | $\mathbf{+29.04\ J\ (+0.309\%)}$ |
| **System Energy/Token** | $\mathbf{4,582.70\ mJ}$ | $\mathbf{4,645.53\ mJ}$ | $\mathbf{3,143.94\ mJ}$ | $\mathbf{3,160.37\ mJ}$ | $\mathbf{2,291.90\ mJ}$ | $\mathbf{2,298.99\ mJ}$ |
| **Energy-Delay Product** | $440,550\text{ J}\cdot\text{s}$ | $447,173\text{ J}\cdot\text{s}$ | $2,586,881\text{ J}\cdot\text{s}$ | $2,585,310\text{ J}\cdot\text{s}$ | $7,202,852\text{ J}\cdot\text{s}$ | $7,171,980\text{ J}\cdot\text{s}$ |
| — *Relative EDP Delta* | **Ref** | $\mathbf{+1.503\%}$ | **Ref** | $\mathbf{-0.061\%}$ | **Ref** | $\mathbf{-0.429\%}$ |

*Visual Reference:* See [Figure 7](file:///home/mohman/ca_project/mohman/point_7/figures/png/fig7_batch_size_energy_scaling.png) for batch size energy and per-token scaling.

---

## 3. Initial Context Length Sensitivity ($P \in \{128, 512, 2048, 4096\}$)

Context length determines the size of the attention Key-Value (KV) cache stored in GPU HBM2e memory. In all runs: $B=32, T=128, K=4, H=1$, Complete System D.

### Table 2: Context Length Energy Breakdown (Complete System D)

| Initial Context Length ($P$) | Run ID | Attn Compute Energy | Attn DRAM (HBM) Energy | MoE Compute Energy | MoE DRAM (HBM) Energy | Total On-Chip GPU Energy | Modeled System Total Energy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$P=128$ tokens** | `p128_c4` | $7.82\text{ J}$ | $27.30\text{ J}$ | $30.01\text{ J}$ | $958.99\text{ J}$ | $1,082.35\text{ J}$ | $9,383.12\text{ J}$ |
| **$P=512$ tokens** | `p512_c4` | $12.15\text{ J}$ | $29.57\text{ J}$ | $30.01\text{ J}$ | $959.01\text{ J}$ | $1,089.02\text{ J}$ | $9,389.86\text{ J}$ |
| **$P=2048$ tokens** | `anc_c4` | $29.48\text{ J}$ | $38.64\text{ J}$ | $30.01\text{ J}$ | $959.06\text{ J}$ | $1,115.43\text{ J}$ | $9,416.66\text{ J}$ |
| **$P=4096$ tokens** | `p4096_c4` | $52.59\text{ J}$ | $50.68\text{ J}$ | $30.01\text{ J}$ | $959.18\text{ J}$ | $1,150.90\text{ J}$ | $9,452.43\text{ J}$ |

*Visual Reference:* See [Figure 10](file:///home/mohman/ca_project/mohman/point_7/figures/png/fig10_context_length_energy_scaling.png) for attention vs MoE energy scaling.

### Analysis of Context Length Dynamics:
1. **True Attention Energy Scaling:**
   - As $P$ increases from $128$ to $4096$, attention compute energy increases from $7.82\text{ J} \to 52.59\text{ J}$ ($6.73\times$), and attention DRAM access energy scales from $27.30\text{ J} \to 50.68\text{ J}$ ($1.86\times$).
   - This sub-linear growth reflects that during decoding with fixed $T=128$, each decode step $t$ processes sequence length $n = P + t + 1$. Average sequence length increases from $192.5$ to $4160.5$ ($21.6\times$). Because attention layers also perform fixed per-token projection and RoPE operations that do not scale with $P$, net compute energy grows by $6.73\times$.
2. **MoE Independence from Context Length:**
   - MoE DRAM energy ($959.0\text{ J}$) and compute energy ($30.01\text{ J}$) remain **strictly invariant** across all context lengths. During generation, each token routes to Top-8 experts regardless of prompt history length.
3. **Decoupled Memory Partitions:**
   - The GPU HBM expert cache (35 GiB) operates as a dedicated partition, decoupled from the KV cache. Increasing $P$ does not evict expert parameters from HBM.

---

## 4. Prefetch Lookahead Horizon Sensitivity ($H \in \{1, 2, 4\}$)

Lookahead horizon $H$ governs how many future token steps ahead the predictor forecasts expert routing targets ($B=32, P=2048, T=128, K=4$, Complete System D).

### Table 3: Lookahead Horizon Ledger

| Lookahead Horizon ($H$) | Run ID | Exposed Stall Time ($t_{\text{stall}}$) | Simulated Runtime ($t_{\text{sim}}$) | Speculative DDR5 Traffic | Static DRAM Energy | Total Modeled System Energy | System EDP ($10^6\text{ J}\cdot\text{s}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$H=1$ step** | `anc_c4` | $747.58\text{ s}$ | $761.63\text{ s}$ | $10.93\text{ TB}$ | $3,427.32\text{ J}$ | $9,416.66\text{ J}$ | $7.172\text{ M}$ |
| **$H=2$ steps** | `h2_c4` | $746.86\text{ s}$ | $760.91\text{ s}$ | $10.94\text{ TB}$ | $3,424.10\text{ J}$ | $9,409.14\text{ J}$ | $7.159\text{ M}$ |
| **$H=4$ steps** | `h4_c4` | $746.15\text{ s}$ | $760.20\text{ s}$ | $10.94\text{ TB}$ | $3,420.92\text{ J}$ | $9,401.78\text{ J}$ | $7.147\text{ M}$ |

*Visual Reference:* See [Figure 9](file:///home/mohman/ca_project/mohman/point_7/figures/png/fig9_lookahead_energy_dynamics.png) for lookahead energy and stall trends.

### Analysis of Lookahead Dynamics:
- Increasing lookahead from $H=1$ to $H=4$ shortens runtime by $1.43\text{ seconds}$ ($761.63\text{ s} \to 760.20\text{ s}$), saving $6.40\text{ J}$ of static DRAM standby energy.
- Off-chip DDR5 traffic remains virtually identical ($10.93\text{ TB} \to 10.94\text{ TB}$).
- Diminishing returns: moving from $H=1$ to $H=2$ provides $0.72\text{ s}$ speedup; moving from $H=2$ to $H=4$ provides $0.71\text{ s}$ speedup.

---

## 5. Prefetch Budget Aggressiveness Sensitivity ($K \in \{0, 2, 4, 8\}$)

Prefetch budget $K$ controls the number of speculative candidate experts dispatched per token per layer ($B=32, P=2048, T=128, H=1$, Complete System D).

### Table 4: Budget Aggressiveness Summary

| Budget ($K$) | Run ID | Net Energy Delta ($\Delta E_{\text{sys}}$) | Delta System Energy (%) | Delta EDP (%) vs Baseline | Operating Regime Classification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$K=0$ (Baseline)** | `anc_c1` | $0.00\text{ J}$ | Ref (0.000%) | Ref (0.000%) | Reference (Zero Speculation) |
| **$K=2$ (Conservative)** | `k2_c4` | $\mathbf{-27.68\ J}$ | $\mathbf{-0.295\%}$ | $\mathbf{-1.715\%}$ | **Global Energy & EDP Minimum (Sweet Spot)** |
| **$K=4$ (Anchor)** | `anc_c4` | $+29.04\text{ J}$ | $+0.309\%$ | $-0.429\%$ | Balanced Latency / Throughput Trade-off |
| **$K=8$ (Aggressive)** | `k8_c4` | $+35.60\text{ J}$ | $+0.379\%$ | $-0.325\%$ | Diminishing Returns (+1.41% worse EDP than K=2) |

*Visual Reference:* See [Figure 8](file:///home/mohman/ca_project/mohman/point_7/figures/png/fig8_budget_sweet_spot.png) for the sweet spot curve and [Figure 6](file:///home/mohman/ca_project/mohman/point_7/figures/png/fig6_energy_delay_product.png) for EDP comparisons.

### Conclusion on Architectural Sensitivity:
The globally optimal configuration for energy efficiency is:
$$\mathbf{Batch\ Size\ B=32,\ Prefetch\ Budget\ K=2,\ Lookahead\ H=1\ or\ 2,\ Policy\ FA-LRU\ (W=64)}$$
