# Final Numerical Reconciliation: Forensic Resolution of Discrepancies and Mathematical Consistency

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 7 — Energy Analysis  
**Document:** Forensic Audit of Calculations, Invariant Reconciliations, and Discrepancy Resolutions  
**Author:** Mohman / Advanced Agentic Coding Assistant  
**Date:** November 11, 2026  

---

## 1. Executive Summary

This document provides the definitive forensic resolution of every numerical inconsistency identified in earlier drafts of Point 7. Through source-code tracing, raw telemetry inspection, and automated invariant verification across all 28 evaluation runs, we establish the single source of mathematical truth.

### Key Forensic Findings:
1. **The 11.45 J Arithmetic Discrepancy Was Pure Text Transcription Error:** The underlying extracted CSV (`unified_energy_results.csv`) has always been 100% mathematically exact down to floating-point precision ($1.8 \times 10^{-12}\text{ J}$ residual). The discrepancy arose because the previous report author hand-wrote a table mixing outdated draft values (`hbm_total_j = 1042.87`, `ddr5_total_j = 4868.96`, `static_dram_j = 3445.03`) with the CSV's true total (`system_total_j = 9416.66`).
2. **The K=8 EDP Contradiction Resolved:** Figure 6 was already plotting the correct computed EDP from the CSV ($7.179\text{ M J}\cdot\text{s}$, **-0.325%** vs. Baseline A). The executive summary's claim of "$+0.437\%$ degradation" was an erroneous narrative based on an unverified runtime ($770.69\text{ s}$). In reality, $K=8$ runtime is $761.89\text{ s}$ ($5.39\text{ s}$ faster than baseline), yielding a minor EDP improvement over baseline, though significantly worse than $K=2$ ($7.079\text{ M}$).
3. **The 32x Context-Length Scaling Fallacy Corrected:** The previous report's claim that attention energy scales $32\times$ between $P=128$ and $P=4096$ was an artificial extrapolation (`29.48 * P / 2048`). The actual simulator output reveals that attention compute grows by **$6.73\times$** ($7.82\text{ J} \to 52.59\text{ J}$) and attention DRAM access grows by **$1.86\times$** ($27.30\text{ J} \to 50.68\text{ J}$), strictly matching the $O(P+t)$ decoding formulation.
4. **FA-LRU Hit Count Provenance Clarified:** The $+15$ resident hit figure belonged to the Point 4 W4 Big Experiment. For the Point 6 / Point 7 Anchor workload ($B=32, P=2048, T=128, K=4, H=1$), FA-LRU ($W=64$) delivers **exactly +22 resident hits**, avoids **22 expert reloads**, eliminates **$2,025,848,832\text{ bytes}$ ($2.026\text{ GB} = 1.887\text{ GiB}$)** of DDR5 traffic, and saves **$0.326\text{ Joules}$** of net system energy.

---

## 2. Forensic Resolution of the 11.45 J Arithmetic Discrepancy

### 2.1 The Stale Report Table vs. The Source CSV

In earlier report drafts, the Anchor comparison table for Complete System D reported:
- GPU HBM2e: $1,042.87\text{ J}$
- GPU MAC Compute: $71.25\text{ J}$
- DDR5 Active Transfer: $4,868.96\text{ J}$
- Static DRAM Power: $3,445.03\text{ J}$
- **Stated Total: $9,416.66\text{ J}$**
- **Actual Sum:** $1042.87 + 71.25 + 4868.96 + 3445.03 = \mathbf{9,428.11\text{ J}}$
- **Discrepancy:** $9428.11 - 9416.66 = \mathbf{11.45\text{ Joules}}$

### 2.2 Forensic Trace of the True Source Values

We independently summed the 128 decode step rows in the raw simulator output file:
`point_6/final_results/experiments/anchor/raw_results/anc_c4/csv/deepseekV3_moe_anywhere_2048_128_GPU_N1_D1_TP1_DP1_maxbatch32_maxprocess524288_iter128_skew0_precision_byte2_parallel_execution0_ramul_decode.csv`

The true values extracted from the raw simulator output and recorded in `unified_energy_results.csv` row 3 (`anc_c4`) are:

| Subsystem Component | Raw Value from Simulator / Extraction | Exact Formula | Reconciled Value (Joules) |
| :--- | :--- | :--- | :--- |
| **GPU HBM2e DRAM** | $\sum \text{act\_energy} + \sum \text{read\_energy} + \sum \text{write\_energy}$ | $50.421413 + 990.733440 + 3.026115$ | **$1,044.180968\text{ J}$** |
| **GPU Tensor Core MAC** | $\sum \text{mac\_energy}$ | $71.245508\text{ nJ} / 10^9$ | **$71.245508\text{ J}$** |
| **Subtotal: Native GPU** | $\sum \text{total\_energy}$ | $1,044.180968 + 71.245508$ | **$1,115.426450\text{ J}$** |
| **DDR5 Active Demand** | `demand_ddr5_bytes` $\times 1.0 \times 10^{-10}\text{ J/byte}$ | $37,810,883,002,368 \times 10^{-10}$ | **$3,781.088300\text{ J}$** |
| **DDR5 Active Speculative** | `speculative_ddr5_bytes` $\times 1.0 \times 10^{-10}\text{ J/byte}$ | $10,928,221,323,264 \times 10^{-10}$ | **$1,092.822132\text{ J}$** |
| **Subtotal: DDR5 Active** | `total_ddr5_bytes` $\times 1.0 \times 10^{-10}\text{ J/byte}$ | $48,739,104,325,632 \times 10^{-10}$ | **$4,873.910433\text{ J}$** |
| **Static Background DRAM** | $P_{\text{static}} \times t_{\text{sim}}$ | $4.5\text{ W} \times 761.6269\text{ s}$ | **$3,427.321050\text{ J}$** |
| **Total Modeled System Energy** | $E_{\text{native}} + E_{\text{DDR5}} + E_{\text{static}}$ | $1115.426450 + 4873.910433 + 3427.321050$ | $\mathbf{9,416.657933\text{ J}}$ |
| **Arithmetic Residual** | Sum of Components minus Total | $9416.6579325632 - 9416.6579325632$ | $\mathbf{0.000000000000\text{ J}}$ |

### 2.3 Why Did the Stale Report Table Have Wrong Numbers?
1. **The Native Invariance Myth:** The report author erroneously assumed native GPU energy was identical across all configurations ($1,042.87\text{ J}$), copying Baseline A's HBM energy into Config D. In reality, Config D experiences slightly higher HBM activations and reads ($+1.31\text{ J}$) due to speculative expert cache lines.
2. **Runtime Inconsistency:** The report author used $t = 765.56\text{ s} \implies E_{\text{static}} = 3,445.03\text{ J}$. In reality, the true simulated runtime for `anc_c4` is $761.6269\text{ s} \implies E_{\text{static}} = 3,427.32\text{ J}$.
3. **DDR5 Traffic Transcription:** The author hand-typed $4,868.96\text{ J}$ instead of the true value $4,873.91\text{ J}$.

All markdown tables and text have now been updated to reflect the exact, reconciled values.

---

## 3. Forensic Resolution of the K=8 EDP Contradiction

### 3.1 The Contradiction
- **Report Text Claim:** $K=8$ increases execution time by $+3.42\text{ s}$ (to $770.69\text{ s}$), increasing total energy to $9,423.21\text{ J}$, thereby **worsening EDP by +0.437%**.
- **Figure 6 Annotation:** Plotted Baseline EDP as $7.203\text{ M J}\cdot\text{s}$ and $K=8$ EDP as $7.179\text{ M J}\cdot\text{s}$ (**-0.33% improvement**).

### 3.2 Raw Data Verification
Inspecting `unified_energy_results.csv` row 27 (`k8_c4`) vs. row 0 (`anc_c1`):
- `anc_c1` (Baseline A):
  - $t_{\text{sim}} = 767.2718\text{ s}$
  - $E_{\text{sys}} = 9,387.614137\text{ J}$
  - $\text{EDP} = 9387.614137 \times 767.2718 = \mathbf{7,202,851.60\ J\cdot s}$ ($7.203\text{ M}$)
- `k8_c4` (Aggressive Budget $K=8$):
  - $t_{\text{sim}} = \mathbf{761.8858\text{ s}}$ (NOT $770.69\text{ s}$!)
  - $E_{\text{sys}} = 9,423.210644\text{ J}$
  - $\text{EDP} = 9423.210644 \times 761.8858 = \mathbf{7,179,410.38\ J\cdot s}$ ($7.179\text{ M}$)
- **True Relative EDP Delta vs. Baseline A:**
  $$\frac{7,179,410.38 - 7,202,851.60}{7,202,851.60} \times 100\% = \mathbf{-0.3254\%} \approx \mathbf{-0.33\%}$$

### 3.3 The Correct Scientific Interpretation
1. **Figure 6 was 100% correct.** It plotted the true unrounded values from the CSV.
2. Compared to Baseline A ($K=0$), $K=8$ is still faster ($761.89\text{ s}$ vs. $767.27\text{ s}$, a $5.39\text{ s}$ reduction). Because the speedup ($-0.70\%$) exceeds the energy overhead ($+0.38\%$), $K=8$ achieves a **net EDP reduction of -0.325%**.
3. **However, $K=8$ is deeply sub-optimal compared to $K=2$ and $K=4$:**
   - $K=2$ (`k2_c4`): $t = 756.34\text{ s}$, $E = 9,359.93\text{ J}$, $\text{EDP} = 7.079\text{ M}$ (**-1.715%**).
   - $K=4$ (`anc_c4`): $t = 761.63\text{ s}$, $E = 9,416.66\text{ J}$, $\text{EDP} = 7.172\text{ M}$ (**-0.429%**).
   - $K=8$ injects $18.04\text{ TB}$ of speculative traffic, causing bus contention that adds $+5.54\text{ s}$ of delay relative to $K=2$.
   - **Its EDP is +1.41% WORSE than $K=2$.**
4. The text has been corrected: $K=8$ is an example of diminishing returns and bus congestion, but it does not degrade EDP below the un-prefetched baseline.

---

## 4. Forensic Resolution of Context-Length Energy Scaling

### 4.1 The 32x Growth Fallacy
The previous report asserted that moving from $P=128$ to $P=4096$ caused a $32\times$ increase in attention compute ($1.84\text{ J} \to 58.96\text{ J}$) and attention DRAM energy ($2.41\text{ J} \to 77.28\text{ J}$).
- **Why this was wrong:** The author simply took $P=2048$ values ($29.48\text{ J}$ and $38.64\text{ J}$) and multiplied them by $(128 / 2048)$ and $(4096 / 2048)$.
- In generation decoding with fixed $T=128$, each decode step $t \in [0, 127]$ processes a sequence of length $n = P + t + 1$. The ratio of average sequence length between $P=4096$ and $P=128$ is:
  $$\frac{4096 + 64.5}{128 + 64.5} = \frac{4160.5}{192.5} \approx 21.6$$
  Furthermore, attention layers include fixed per-token projection operations that do not scale with sequence length.

### 4.2 Verified Simulator Outputs across Context Lengths (Complete System D)

| Initial Context ($P$) | Run ID | Attn Compute Energy | Attn DRAM (HBM) Energy | MoE Compute Energy | MoE DRAM (HBM) Energy | Total On-Chip Energy | Modeled System Energy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$P=128$** | `p128_c4` | $7.82\text{ J}$ | $27.30\text{ J}$ | $30.01\text{ J}$ | $958.99\text{ J}$ | $1,082.35\text{ J}$ | $9,383.12\text{ J}$ |
| **$P=512$** | `p512_c4` | $12.15\text{ J}$ | $29.57\text{ J}$ | $30.01\text{ J}$ | $959.01\text{ J}$ | $1,089.02\text{ J}$ | $9,389.86\text{ J}$ |
| **$P=2048$** | `anc_c4` | $29.48\text{ J}$ | $38.64\text{ J}$ | $30.01\text{ J}$ | $959.06\text{ J}$ | $1,115.43\text{ J}$ | $9,416.66\text{ J}$ |
| **$P=4096$** | `p4096_c4` | $52.59\text{ J}$ | $50.68\text{ J}$ | $30.01\text{ J}$ | $959.18\text{ J}$ | $1,150.90\text{ J}$ | $9,452.43\text{ J}$ |

- **Actual Compute Scaling ($P=128 \to 4096$):** $52.59 / 7.82 = \mathbf{6.73\times}$ (NOT $32\times$).
- **Actual DRAM Scaling ($P=128 \to 4096$):** $50.68 / 27.30 = \mathbf{1.86\times}$ (NOT $32\times$).
- **MoE Invariance:** MoE DRAM ($959.0\text{ J}$) and MoE Compute ($30.01\text{ J}$) are **invariant** across context lengths.

---

## 5. Forensic Reconciliation of the FA-LRU Energy Comparison

### 5.1 Provenance of the Hit Differential
- **Point 4 W4 Big Experiment:** Evaluated an isolated $B=32$ configuration on a specific synthetic trace, reporting $+15$ resident hits.
- **Point 6 / Point 7 Anchor Workload:** Evaluates full authentic trace replay ($B=32, P=2048, T=128, K=4, H=1$):
  - Prefetch + LRU (`anc_c3`): **1,499 resident hits**, **535,953 reloads**.
  - Complete System FA-LRU (`anc_c4`): **1,521 resident hits**, **535,931 reloads**.
  - **Hit Differential:** $1521 - 1499 = \mathbf{+22\text{ resident hits}}$.
  - **Reload Differential:** $535953 - 535931 = \mathbf{22\text{ fewer expert reloads}}$.

### 5.2 Exact Microarchitectural Energy Ledger (Config D minus Config C)

| Metric | Prefetch + LRU (`anc_c3`) | Complete FA-LRU (`anc_c4`) | Exact Delta (D minus C) |
| :--- | :--- | :--- | :--- |
| **Simulated Runtime** | $761.6585\text{ s}$ | $761.6269\text{ s}$ | **-0.0316 s (-31.6 ms)** |
| **Exposed Stall Time** | $747,615.27\text{ ms}$ | $747,583.37\text{ ms}$ | **-31.90 ms** |
| **Total DDR5 Bytes** | $48,741,130,174,464\text{ B}$ | $48,739,104,325,632\text{ B}$ | **-2,025,848,832 B (-2.026 GB = -1.887 GiB)** |
| — *Demand DDR5 Bytes* | $37,792,121,880,576\text{ B}$ | $37,810,883,002,368\text{ B}$ | $+18,761,121,792\text{ B}$ |
| — *Speculative DDR5 Bytes*| $10,949,008,293,888\text{ B}$ | $10,928,221,323,264\text{ B}$ | $-20,786,970,624\text{ B}$ |
| **Active DDR5 Energy** | $4,874.113017\text{ J}$ | $4,873.910433\text{ J}$ | **-0.202585 J** |
| **Static DRAM Energy** | $3,427.463250\text{ J}$ | $3,427.321050\text{ J}$ | **-0.142200 J** |
| **Native GPU Energy** | $1,115.407640\text{ J}$ | $1,115.426450\text{ J}$ | **+0.018810 J** |
| **Total Modeled System Energy**| $\mathbf{9,416.983907\ J}$ | $\mathbf{9,416.657933\ J}$ | $\mathbf{-0.325974\ J\ (-0.0035\%)}$ |
| **System EDP** | $7,172,525.68\text{ J}\cdot\text{s}$ | $7,171,979.99\text{ J}\cdot\text{s}$ | **-545.69 J}\cdot\text{s (-0.0076%)** |

All reports have been updated to report $-2,025,848,832\text{ bytes}$ ($-2.026\text{ GB} = -1.887\text{ GiB}$) and $-0.326\text{ Joules}$ net savings.

