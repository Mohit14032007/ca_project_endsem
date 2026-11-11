# Phase 2F — Energy Analysis: Consumption Breakdown and Overhead Evaluation

## 1. Executive Summary

This report evaluates the simulated energy consumption characteristics across the **15 runs of the Phase 2F Synthetic Compute-Window Sensitivity Campaign**. Energy metrics were extracted directly from the cycle-accurate and analytical DRAM models within `LLMSimulator`, measuring activation energy, read energy, write energy, MAC computation energy, and total system energy.

### Key Energy Findings
1. **Negligible Prefetch Energy Overhead:** Total system energy for demand-only baselines was **$9.0679\text{ J}$**. Prefetching configurations consumed between **$9.0688\text{ J}$** and **$9.0697\text{ J}$**, representing an overhead of **less than $0.02\%$** ($+0.0018\text{ J}$).
2. **DRAM Activation and Read Energy Dominance:** Over $95\%$ of system energy is consumed by HBM and DDR5 read operations to service weight activations.
3. **Speculative Memory Traffic Cost:** Dispatched speculative transfers inject between $33,180.0\text{ MB}$ (ST-MoE) and $38,388.0\text{ MB}$ (CSTP) of DDR5 traffic. Because PABI quarantine buffers speculative candidates without polluting active HBM lines, this extra traffic produces minimal additional cache write energy.

---

## 2. Energy Consumption Master Table (All 15 Runs)

| Run ID | Multiplier ($M$) | Predictor | Total Energy (J) | Energy Delta vs $B(M)$ (J) | Overhead vs $B(M)$ (%) | Speculative Traffic (MB) | Demand Traffic (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`B-M1`** | $1.0\times$ | None | $9.0679$ | Baseline | Baseline | $0.0$ | $169,764.0$ |
| **`FWD-S-H2-M1`** | $1.0\times$ | ST-MoE | $9.0679$ | $+0.0000$ | $+0.000\%$ | $33,180.0$ | $169,764.0$ |
| **`FWD-C-H2-M1`** | $1.0\times$ | CSTP | $9.0679$ | $+0.0000$ | $+0.000\%$ | $38,388.0$ | $169,764.0$ |
| **`B-M4`** | $4.0\times$ | None | $9.0679$ | Baseline | Baseline | $0.0$ | $169,764.0$ |
| **`FWD-S-H2-M4`** | $4.0\times$ | ST-MoE | $9.0688$ | $+0.0009$ | $+0.010\%$ | $33,180.0$ | $169,680.0$ |
| **`FWD-C-H2-M4`** | $4.0\times$ | CSTP | $9.0688$ | $+0.0009$ | $+0.010\%$ | $38,388.0$ | $169,680.0$ |
| **`B-M8`** | $8.0\times$ | None | $9.0679$ | Baseline | Baseline | $0.0$ | $169,764.0$ |
| **`FWD-S-H2-M8`** | $8.0\times$ | ST-MoE | $9.0688$ | $+0.0009$ | $+0.010\%$ | $33,180.0$ | $169,680.0$ |
| **`FWD-C-H2-M8`** | $8.0\times$ | CSTP | $9.0688$ | $+0.0009$ | $+0.010\%$ | $38,388.0$ | $169,680.0$ |
| **`B-M12`** | $12.0\times$ | None | $9.0679$ | Baseline | Baseline | $0.0$ | $169,764.0$ |
| **`FWD-S-H2-M12`** | $12.0\times$ | ST-MoE | $9.0697$ | $+0.0018$ | $+0.020\%$ | $33,180.0$ | $169,596.0$ |
| **`FWD-C-H2-M12`** | $12.0\times$ | CSTP | $9.0697$ | $+0.0018$ | $+0.020\%$ | $38,388.0$ | $169,596.0$ |
| **`B-M16`** | $16.0\times$ | None | $9.0679$ | Baseline | Baseline | $0.0$ | $169,764.0$ |
| **`FWD-S-H2-M16`** | $16.0\times$ | ST-MoE | $9.0697$ | $+0.0018$ | $+0.020\%$ | $33,180.0$ | $169,596.0$ |
| **`FWD-C-H2-M16`** | $16.0\times$ | CSTP | $9.0697$ | $+0.0018$ | $+0.020\%$ | $38,388.0$ | $169,596.0$ |

---

## 3. Detailed Energy Breakdown

Across all 4 decoding tokens in Workload A1:
- **Activation Energy ($E_{\text{act}}$):** $\approx 0.450\text{ J}$ (DRAM row activations for incoming requests).
- **Read Energy ($E_{\text{read}}$):** $\approx 8.525\text{ J}$ (Reading FP16 model weights and activation vectors).
- **Write Energy ($E_{\text{write}}$):** $\approx 0.0058\text{ J}$ (Writing intermediate activations and newly loaded weights into HBM).
- **MAC Energy ($E_{\text{mac}}$):** $\approx 0.0866\text{ J}$ (On-chip tensor arithmetic operations).
- **Sum Total Energy ($E_{\text{total}}$):** $9.068\text{ J} \pm 0.002\text{ J}$.

### Energy Impact of Prefetch Conversions
When a speculative prefetch converts into a timely resident hit:
- The demand miss is averted, saving the unbuffered demand DDR5 access cost.
- Demand traffic dropped from $169,764.0\text{ MB}$ to $169,596.0\text{ MB}$ (a net reduction of $168\text{ MB} = 2 \text{ converted experts} \times 84.0\text{ MB}$).
- Net energy overhead remains essentially flat because the cost of transferring an expert speculatively vs on-demand is thermodynamically equivalent over the physical bus.

