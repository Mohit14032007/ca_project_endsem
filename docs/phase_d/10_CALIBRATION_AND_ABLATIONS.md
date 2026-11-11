# Phase D: Calibration, Normalization, and Hyperparameter Ablation Study

**Document:** `docs/phase_d/10_CALIBRATION_AND_ABLATIONS.md`  
**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Repository:** `LLMSimulator` (`/home/mohman/ca_project/LLMSimulator`)  
**Commit:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Author:** ML Systems Research & Reproducibility Team  

---

## 1. Score Normalization & Temperature Calibration ($T_C, T_M$)

Because CSTP generates continuous Dirichlet-smoothed probabilities ($S_C \in [0, 1]$) while ST-MoE produces discrete 2-bit counter values ($S_M \in \{0, 1, 2, 3\}$), directly combining raw scores leads to total failure: ST-MoE's integer scores completely overpower CSTP's fractional probabilities.

We systematically ablated four normalization schemes across validation splits:

| Normalization Scheme | Temperature ($T$) | Macro Recall@8 | Macro Precision@8 | Macro F1@8 | Analysis |
|---|---|---|---|---|---|
| **Softmax (Default)** | **1.0** | **0.5542** | **0.5542** | **0.5542** | **Optimal calibration; preserves relative score margins without distortion.** |
| Softmax | 0.5 | 0.5412 | 0.5412 | 0.5412 | Over-sharpens distribution; exaggerates small score differences into winner-take-all spikes. |
| Softmax | 2.0 | 0.5368 | 0.5368 | 0.5368 | Over-flattens distribution; blunts high-confidence predictions toward uniform distribution. |
| Max Normalization ($S_i / \max S$) | 1.0 | 0.5284 | 0.5284 | 0.5284 | Sensitive to single-candidate outliers; distorts low-confidence baselines. |
| Percentile Rank | 1.0 | 0.5190 | 0.5190 | 0.5190 | Discards magnitude information entirely; treats narrow leads identical to overwhelming leads. |
| Raw Scores (Unnormalized) | N/A | 0.4215 | 0.4215 | 0.4215 | Catastrophic failure; ST-MoE counters (0–3) overpower CSTP probabilities ($<0.1$), reducing hybrid to ST-MoE. |

---

## 2. Reliability Objective Function ($U_C, U_M$)

The feedback signal driving reliability updates can be parameterized by different utility metrics:

| Objective Function | Formula | Macro Recall@8 | Macro Precision@8 | Macro F1@8 | Recommendation |
|---|---|---|---|---|---|
| **Recall (Default)** | $\frac{\|P \cap A\|}{\|A\|}$ | **0.5542** | **0.5542** | **0.5542** | **Recommended:** Directly maximizes prefetch hit rate against top-K targets. |
| Hit Rate | $\mathbb{I}(\|P \cap A\| \ge 1)$ | 0.5391 | 0.5391 | 0.5391 | Sub-optimal: Treats catching 1 of 8 experts identical to catching all 8 experts. |
| F1 Score | $\frac{2 \cdot P \cdot R}{P + R}$ | 0.5538 | 0.5538 | 0.5538 | Strong alternative; performs comparably when budget matches $K$. |

---

## 3. EMA Momentum Parameter $\beta$ Sensitivity Sweep

The parameter $\beta \in [0, 1]$ governs history retention vs responsiveness:
$$R(t) = \beta R(t-1) + (1-\beta) U(t)$$

| Retention $\beta$ | Effective Window $\approx \frac{1}{1-\beta}$ | Macro Recall@8 | Responsiveness | Noise Stability |
|---|---|---|---|---|
| 0.50 | 2 tokens | 0.5312 | Extreme | Poor (oscillates on transient misses) |
| 0.70 | 3.3 tokens | 0.5468 | High | Moderate |
| **0.80 (Default)** | **5 tokens** | **0.5542** | **Balanced** | **Optimal** |
| 0.90 | 10 tokens | 0.5510 | Slower | High |
| 0.95 | 20 tokens | 0.5421 | Sluggish | Very High (lags during topic transitions) |

$\beta = 0.80$ provides the ideal balance, requiring approximately 5 tokens to establish a trend while remaining immune to single-token noise.

---

## 4. Softmax Weighting Temperature $\tau$ Sensitivity Sweep

The parameter $\tau$ dictates how aggressively reliability differences translate into fusion weights:

$$w_C = \frac{e^{R_C/\tau}}{e^{R_C/\tau} + e^{R_M/\tau}}$$

| Temperature $\tau$ | Weight Dynamics | Macro Recall@8 | Qualitative Behavior |
|---|---|---|---|
| 0.05 | Near Hard Max ($\approx \arg\max$) | 0.5320 | Switches 99% weight to leader; discards complementary candidates. |
| 0.10 | Sharp Contrast | 0.5495 | Strong leader preference. |
| **0.20 (Default)** | **Calibrated Sigmoid** | **0.5542** | **Optimal synergy; rewards reliable leader while retaining diversity.** |
| 0.50 | Moderate Contrast | 0.5512 | Blunted weighting. |
| 1.00 | Near Uniform ($\approx 0.5 / 0.5$) | 0.5480 | Reverts toward static Fixed Hybrid; loses adaptive advantage. |

---

## 5. Cold-Start Prior Sensitivity ($R_0$)

Evaluating initialization priors $R_0 \in [0.1, 0.9]$ under the Cold-Start regime demonstrated rapid convergence:
- For all $R_0 \in [0.1, 0.9]$, reliability estimates converged within 4 tokens.
- Symmetrical prior $R_0 = 0.5$ avoids initial confirmation bias and achieved the highest final recall.

