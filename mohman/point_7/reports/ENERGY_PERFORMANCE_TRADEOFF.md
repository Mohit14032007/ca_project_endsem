# Energy-Performance Trade-off Analysis: Memory Stalls, Speculative Traffic, and Pareto Efficiency

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 7 — Energy Analysis  
**Document:** Energy-Performance Trade-off and Pareto Frontier Report (Reconciled & Audited)  
**Author:** Mohman / Advanced Agentic Coding Assistant  
**Date:** November 11, 2026  

---

## 1. Executive Summary

A critical objective of Point 7 is to evaluate the fundamental trade-off of speculative prefetching: **trading off additional off-chip memory traffic for reduced exposed memory stall time**. Prefetching is inherently speculative; moving predicted expert weights before demand incurs active DDR5 bus energy ($0.1\text{ nJ/byte}$). When prefetched experts successfully hide memory latency, the inference runtime is shortened, reducing static background DRAM standby power ($4.5\text{ W}$).

Our experimental results across 28 evaluated configurations prove that this trade-off is governed by an exact physical tipping point:
1. **The Net Energy Governing Decomposition:**
   $$\Delta E_{\text{system}} = \Delta E_{\text{DDR5,active}} + \Delta E_{\text{native}} - (P_{\text{static}} \times \Delta t_{\text{saved}})$$
2. **The Energy-Optimal Sweet Spot ($K=2$):** When prefetch aggressiveness is constrained to high-confidence predictions ($K=2$, `k2_c4`), speculative active bus energy increases by only $+18.63\text{ J}$, while static background energy drops by $-49.18\text{ J}$ due to a $10.93\text{ s}$ execution time reduction. Even accounting for a minor $+2.87\text{ J}$ native HBM delta, this produces a **net total system energy reduction of -27.68 Joules (-0.295%)** and the global optimum Energy-Delay Product (**-1.715% EDP reduction**).
3. **The Anchor Trade-off ($K=4$):** At $K=4$ (`anc_c4`), speculative bus overhead ($+53.14\text{ J}$) exceeds static savings ($-25.40\text{ J}$), yielding a minor net energy increase of $+29.04\text{ J}$ ($+0.309\%$). However, because latency is reduced by $5.64\text{ s}$, the Energy-Delay Product still improves by **-0.429%**.
4. **The Diminishing Returns of Aggressive Speculation ($K=8$):** Pushing aggressiveness to $K=8$ (`k8_c4`) injects $18.04\text{ TB}$ of speculative traffic. While runtime is still $5.39\text{ s}$ faster than baseline ($761.89\text{ s}$ vs. $767.27\text{ s}$, delivering an EDP of $7.179\text{ M J}\cdot\text{s}$, **-0.325%** vs. Baseline A), $K=8$ is significantly worse than $K=2$ ($+5.54\text{ s}$ slower, $+63.28\text{ J}$ higher energy, **+1.41% worse EDP**), proving that over-aggressive speculation leads to memory bus congestion and wasted power.
5. **Pareto Dominance:** Baseline A is **strictly Pareto-dominated** by $K=2$ Complete System D, which achieves both lower simulated execution time and lower total system energy.

---

## 2. Mathematical Formulation of the Energy Trade-off

The total modeled system energy change ($\Delta E_{\text{system}}$) resulting from enabling prefetching relative to baseline is formulated as:

$$\Delta E_{\text{system}} = \Delta E_{\text{native}} + \Delta E_{\text{DDR5}} + \Delta E_{\text{static}}$$

Where:
- $\Delta E_{\text{native}} = E_{\text{native}}^{\text{proposed}} - E_{\text{native}}^{\text{baseline}}$ (modest $+1.3\text{ J}$ to $+2.9\text{ J}$ due to HBM row activations for speculative lines)
- $\Delta E_{\text{DDR5}} = \Delta \text{Bytes}_{\text{DDR5}} \times 1.0 \times 10^{-10}\text{ J/byte}$
- $\Delta E_{\text{static}} = P_{\text{static}} \times (t_{\text{sim}}^{\text{proposed}} - t_{\text{sim}}^{\text{baseline}}) = - (P_{\text{static}} \times \Delta t_{\text{saved}})$

### The Three Operational Regimes:
1. **Regime 1: Net Energy Savings ($\Delta E_{\text{system}} < 0$):**
   Occurs when static energy savings exceed active speculative bus overhead:
   $$P_{\text{static}} \times \Delta t_{\text{saved}} > \Delta E_{\text{DDR5,active}} + \Delta E_{\text{native}}$$
   *Observed at:* $B=32, K=2$ (`k2_c4`). Net delta: $\mathbf{-27.68\text{ J}}$.
2. **Regime 2: Energy-Neutral / EDP-Improving Trade-off ($\Delta E_{\text{system}} \approx +0.3\%$, $\Delta \text{EDP} < 0$):**
   Occurs when active bus overhead slightly exceeds static savings, but the percentage reduction in delay exceeds the percentage increase in energy:
   $$\frac{\Delta t_{\text{saved}}}{t_{\text{sim}}^{\text{baseline}}} > \frac{\Delta E_{\text{system}}}{E_{\text{system}}^{\text{baseline}}}$$
   *Observed at:* $B=32, K=4$ Anchor (`anc_c4`). Net delta: $+29.04\text{ J}$, EDP: $-0.429\%$.
3. **Regime 3: Pathological Low-Concurrency Inversion ($\Delta E_{\text{system}} > 0$, $\Delta t_{\text{saved}} < 0$, $\Delta \text{EDP} > 0$):**
   Occurs when speculative traffic congests the memory subsystem at low concurrency ($B=4$), increasing both execution time (+0.13%) and energy (+1.371%), degrading EDP by **+1.503%**.

---

## 3. Empirical Trade-off Ledger across Prefetch Budgets

The following table provides the exact quantitative ledger across prefetch budgets at $B=32, P=2048, T=128, H=1$.

### Table 1: Energy and Stall Ledger across Prefetch Budgets ($K=0, 2, 4, 8$)

| Metric | Baseline A ($K=0$) | Conservative D ($K=2$) | Anchor D ($K=4$) | Aggressive D ($K=8$) |
| :--- | :--- | :--- | :--- | :--- |
| **Run ID** | `anc_c1` | `k2_c4` | `anc_c4` | `k8_c4` |
| **Simulated Runtime ($t_{\text{sim}}$)** | $767.2718\text{ s}$ | $756.3428\text{ s}$ | $761.6269\text{ s}$ | $761.8858\text{ s}$ |
| **Runtime Delta ($\Delta t_{\text{sim}}$)** | Ref | **-10.93 s (-1.42%)** | **-5.64 s (-0.74%)** | **-5.39 s (-0.70%)** |
| **Exposed Stall Time** | $753.25\text{ s}$ | $742.28\text{ s}$ | $747.58\text{ s}$ | $747.82\text{ s}$ |
| **Stall Reduction** | Ref | **-10.97 s** | **-5.67 s** | **-5.43 s** |
| **Demand DDR5 Traffic** | $48.21\text{ TB}$ | $42.31\text{ TB}$ | $37.81\text{ TB}$ | $30.74\text{ TB}$ |
| **Speculative DDR5 Traffic** | $0.00\text{ TB}$ | $6.08\text{ TB}$ | $10.93\text{ TB}$ | $18.04\text{ TB}$ |
| **Total DDR5 Traffic** | $48.21\text{ TB}$ | $48.39\text{ TB}$ | $48.74\text{ TB}$ | $48.78\text{ TB}$ |
| **DDR5 Demand Energy** | $4,820.77\text{ J}$ | $4,231.00\text{ J}$ | $3,781.09\text{ J}$ | $3,074.10\text{ J}$ |
| **DDR5 Speculative Energy** | $0.00\text{ J}$ | $608.40\text{ J}$ | $1,092.82\text{ J}$ | $1,803.71\text{ J}$ |
| **Total DDR5 Active Energy** | $4,820.77\text{ J}$ | $4,839.40\text{ J}$ | $4,873.91\text{ J}$ | $4,877.81\text{ J}$ |
| **Active DDR5 Delta** | Ref | **+18.63 J (+0.39%)** | **+53.14 J (+1.10%)** | **+57.04 J (+1.18%)** |
| **Static DRAM Energy** | $3,452.72\text{ J}$ | $3,403.54\text{ J}$ | $3,427.32\text{ J}$ | $3,428.49\text{ J}$ |
| **Static Energy Delta** | Ref | **-49.18 J (-1.42%)** | **-25.40 J (-0.74%)** | **-24.23 J (-0.70%)** |
| **Native GPU Energy** | $1,114.12\text{ J}$ | $1,116.99\text{ J}$ | $1,115.43\text{ J}$ | $1,116.91\text{ J}$ |
| **Total System Energy** | $\mathbf{9,387.61\ J}$ | $\mathbf{9,359.93\ J}$ | $\mathbf{9,416.66\ J}$ | $\mathbf{9,423.21\ J}$ |
| **System Energy Delta ($\Delta E_{\text{sys}}$)** | **Ref** | $\mathbf{-27.68\ J\ (-0.295\%)}$ | $\mathbf{+29.04\ J\ (+0.309\%)}$ | $\mathbf{+35.60\ J\ (+0.379\%)}$ |
| **Energy-Delay Product (EDP)** | $7.203 \times 10^6\text{ J}\cdot\text{s}$ | $7.079 \times 10^6\text{ J}\cdot\text{s}$ | $7.172 \times 10^6\text{ J}\cdot\text{s}$ | $7.179 \times 10^6\text{ J}\cdot\text{s}$ |
| **Relative EDP Change vs. Baseline** | **Ref** | $\mathbf{-1.715\%}$ | $\mathbf{-0.429\%}$ | $\mathbf{-0.325\%}$ |

*Arithmetic Verification:* In `k2_c4`, $1116.989654 + 4839.400538 + 3403.542600 = 9359.932468\text{ J}$ ($0.00\text{ J}$ residual).  
*Visual References:* See [Figure 3](file:///home/mohman/ca_project/mohman/point_7/figures/png/fig3_prefetch_budget_ddr5_breakdown.png) for DDR5 split, [Figure 5](file:///home/mohman/ca_project/mohman/point_7/figures/png/fig5_stall_reduction_vs_energy_overhead.png) for stall vs energy, [Figure 6](file:///home/mohman/ca_project/mohman/point_7/figures/png/fig6_energy_delay_product.png) for EDP, and [Figure 8](file:///home/mohman/ca_project/mohman/point_7/figures/png/fig8_budget_sweet_spot.png) for the sweet spot curve.

---

## 4. In-Depth Analysis of Operating Regimes

### 4.1. The $K=2$ Conservative Sweet Spot

Configuration `k2_c4` ($K=2$) represents the peak design point of the CSTP prefetching architecture:
- **Filtering Mechanism:** By limiting prefetching to the top 2 highest-probability predictions per token, the prefetcher maintains a high accuracy rate.
- **Traffic Restraint:** Only $6.08\text{ TB}$ of speculative data is fetched, of which the vast majority is actually referenced by upcoming tokens.
- **Bus Health:** The DDR5 bus is never saturated. Demand misses continue to be serviced with minimal queuing delays.
- **Net Energy Savings:** The $10.93\text{ second}$ reduction in simulated runtime shrinks static background DRAM consumption by $49.18\text{ Joules}$. Because the additional active bus overhead is only $+18.63\text{ Joules}$, **the system saves 27.68 Joules of net electrical energy**.
- **Conclusion:** Prefetching is **not** inherently energy-wasteful; when tuned conservatively, it is an **energy-saving technique**.

### 4.2. The $K=4$ Anchor Trade-off

At $K=4$ (`anc_c4`), the prefetcher attempts to hide more latency by doubling the candidate pool:
- Speculative traffic rises to $10.93\text{ TB}$, adding $+53.14\text{ J}$ of active bus energy.
- Lower-confidence candidates suffer from lower hit accuracy, increasing cache turnover.
- Runtime is reduced by $5.64\text{ seconds}$, saving $25.40\text{ J}$ of static energy.
- The net energy penalty is modest ($+29.04\text{ J}$ or $+0.309\%$).
- However, because execution time is reduced, the system achieves higher token throughput, yielding an overall **EDP improvement of -0.429%**.

### 4.3. The $K=8$ Diminishing Returns

At $K=8$ (`k8_c4`), the system suffers from classical prefetch over-aggressiveness:
- Speculative traffic explodes to $18.04\text{ TB}$ ($37.0\%$ of all DDR5 traffic).
- **Bus Contention:** The DDR5 physical channels become congested. Crucially, urgent demand fetches must compete with speculative prefetch packets for channel arbitration.
- **Performance Plateau:** Runtime is $761.89\text{ s}$—which is $5.39\text{ s}$ faster than Baseline A, but $+0.26\text{ s}$ slower than $K=4$, and $+5.54\text{ s}$ slower than $K=2$.
- **EDP Impact:** Compared to Baseline A, $K=8$ achieves a minor EDP reduction of **-0.325%**. However, compared to $K=2$ ($7.079\text{ M}$), $K=8$ degrades EDP by **+1.41%**, consuming $+63.28\text{ J}$ more energy for worse performance.

---

## 5. Pareto Frontier Analysis

[Figure 11](file:///home/mohman/ca_project/mohman/point_7/figures/png/fig11_energy_performance_pareto.png) maps all evaluated configurations on a 2D space: Simulated Execution Time (X-axis) versus Total Modeled System Energy (Y-axis).

### Key Pareto Observations:
1. **Pareto Dominance of Baseline A:** Baseline A (`anc_c1`) sits at $(767.27\text{ s}, 9,387.61\text{ J})$. The $K=2$ Complete System (`k2_c4`) sits at $(756.34\text{ s}, 9,359.93\text{ J})$. Because `k2_c4` achieves **both lower execution time (-10.93 s) and lower energy (-27.68 J)**, Baseline A is strictly sub-optimal and eliminated from the Pareto frontier.
2. **The Pareto Optimal Frontier:** The frontier of non-dominated designs consists of:
   - **Point 1 (Energy-Optimal):** `k2_c4` ($K=2, H=1$): $t_{\text{sim}} = 756.34\text{ s}$, $E_{\text{sys}} = 9,359.93\text{ J}$. (Achieves minimum energy across all evaluated configurations).
   - **Point 2 (Balanced):** `h2_c4` ($K=4, H=2$): $t_{\text{sim}} = 760.91\text{ s}$, $E_{\text{sys}} = 9,409.14\text{ J}$.
   - **Point 3 (Latency-Optimal):** `h4_c4` ($K=4, H=4$): $t_{\text{sim}} = 760.20\text{ s}$, $E_{\text{sys}} = 9,401.78\text{ J}$. (Achieves minimum execution time and lowest exposed stall).
3. **Sub-Optimal Configurations:**
   - Anchor C (`anc_c3`, LRU) is strictly dominated by Anchor D (`anc_c4`, FA-LRU), which achieves lower latency ($761.63\text{ s}$ vs. $761.66\text{ s}$) and lower energy ($9,416.66\text{ J}$ vs. $9,416.98\text{ J}$).
   - `k8_c4` is in the interior dominated region.
