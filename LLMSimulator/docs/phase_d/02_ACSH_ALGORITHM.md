# Phase D: Mathematical Formulation of ACSH (Adaptive Confidence and Statistical Hybrid)

**Author:** Antigravity Senior ML Systems Researcher & Algorithm Engineer  
**Date:** November 2026  
**Document ID:** `docs/phase_d/02_ACSH_ALGORITHM.md`  

---

## 1. Mathematical Objective & Motivation

Phase C demonstrated that **CSTP** and **ST-MoE** possess fundamentally distinct and complementary strengths:
- **CSTP** models continuous probability distributions via Bayesian Dirichlet-multinomial conjugate updates over spatial transitions and temporal transitions. It provides fine-grained probabilistic density, but suffers from statistical inertia when trained over diverse corpora.
- **ST-MoE** models bounded spatial correlations using discrete 2-bit saturating counters (0–3) in a Capacity-Constrained Table (CCT), plus an integer History Table (HT). It provides rapid, zero-inertia adaptation during distribution shifts, but produces discrete integer scores with many ties.

The existing fixed hybrid (`cstp_st_moe_hybrid`) achieved high performance, but applied an arbitrary, static 50/50 weighting regardless of whether CSTP or ST-MoE was performing well in the current context.

**ACSH (Adaptive Confidence and Statistical Hybrid)** replaces static fusion with **online causal reliability estimation**. It dynamically adjusts its trust weights based on each predictor's empirically observed accuracy on recent tokens.

---

## 2. Component Scoring & Normalization

For prediction event $t$ at target stage $L_{\text{tgt}} = L + H$ given source experts $U_t \subseteq \{0, \dots, E-1\}$:

### 2.1 Component Scores
1. **CSTP Score**: $S_C(v, t) \in [0, 1]$ for each expert $v \in \{0, \dots, E-1\}$.
   Computed via Dirichlet-smoothed spatial and temporal probabilities:
   $$S_C(v, t) = \alpha \cdot P_{\text{spatial}}(v \mid U_t, L, H) + (1-\alpha) \cdot P_{\text{temporal}}(v \mid E_{t-1}, L_{\text{tgt}})$$
2. **ST-MoE Confidence Score**: $S_M(v, t) \in \{0, 1, \dots, 5\}$ for each expert $v \in \{0, \dots, E-1\}$.
   Computed via Algorithm 2 (spatial CCT counter accumulation + temporal HT bonus):
   $$S_M(v, t) = \sum_{u \in U_t} \text{CCT}[L, u, v] + 2 \cdot \mathbb{I}(v \in \text{HT}[L_{\text{tgt}}])$$

### 2.2 Score Normalization via Calibrated Softmax
Because $S_C$ is a continuous probability density while $S_M$ is a discrete integer confidence counter, raw summation introduces scale mismatch. We apply temperature-scaled softmax normalization:

$$P_C(v, t) = \frac{\exp\left(\frac{S_C(v, t) - \max_j S_C(j, t)}{T_C}\right)}{\sum_{k=0}^{E-1} \exp\left(\frac{S_C(k, t) - \max_j S_C(j, t)}{T_C}\right)}$$

$$P_M(v, t) = \frac{\exp\left(\frac{S_M(v, t) - \max_j S_M(j, t)}{T_M}\right)}{\sum_{k=0}^{E-1} \exp\left(\frac{S_M(k, t) - \max_j S_M(j, t)}{T_M}\right)}$$

Where:
- $T_C > 0$ is the CSTP calibration temperature (default 1.0, tuned on validation split).
- $T_M > 0$ is the ST-MoE calibration temperature (default 1.0, tuned on validation split).
- Subtracting the maximum ensures complete numerical stability without overflow.

---

## 3. Causal Online Reliability Tracking

### 3.1 Instantaneous Event Utility
At prediction event $t$, both predictors independently generate their Top-$B$ candidate sets:
- $P_C^{(B)}(t) = \operatorname{TopB}_v P_C(v, t)$
- $P_M^{(B)}(t) = \operatorname{TopB}_v P_M(v, t)$

After prediction logging, the actual ground-truth routing $A_t$ ($|A_t| = K$) is revealed. The instantaneous utility of each predictor is:
$$r_C(t) = \frac{|P_C^{(B)}(t) \cap A_t|}{|A_t|} \in [0, 1]$$
$$r_M(t) = \frac{|P_M^{(B)}(t) \cap A_t|}{|A_t|} \in [0, 1]$$

### 3.2 Exponential Moving Average (EMA) Reliability
To track recent performance while filtering noise, ACSH maintains an exponential moving average:
$$R_C(t) = \beta R_C(t-1) + (1-\beta) r_C(t-1)$$
$$R_M(t) = \beta R_M(t-1) + (1-\beta) r_M(t-1)$$

Where $\beta \in [0, 1)$ is the momentum decay factor (default $\beta = 0.8$).

> [!IMPORTANT]
> **Strict Causal Indexing**: The reliability state used to compute weights for event $t$ depends strictly on $r(t-1)$ and earlier observations. Target label $A_t$ is never accessed prior to generating event $t$'s prediction.

### 3.3 Initialization
At the beginning of a sequence ($t=0$):
$$R_C(0) = 0.5, \quad R_M(0) = 0.5$$

---

## 4. Adaptive Fusion Weighting

The fusion weights $w_C(t)$ and $w_M(t)$ are computed via a softmax over the reliability estimates scaled by fusion sensitivity temperature $\tau > 0$:

$$w_C(t) = \frac{\exp(R_C(t) / \tau)}{\exp(R_C(t) / \tau) + \exp(R_M(t) / \tau)}$$
$$w_M(t) = 1 - w_C(t)$$

- When $\tau \to \infty$, $w_C(t) \to 0.5$ (reverting to fixed uniform hybrid).
- When $\tau \to 0$, ACSH behaves as a winner-take-all selector.
- When $R_C(t) = R_M(t)$, $w_C(t) = w_M(t) = 0.5$ regardless of $\tau$.

### 4.1 Combined ACSH Score
The final ensemble probability for expert $v$ is:
$$P_{\text{ACSH}}(v, t) = w_C(t) \cdot P_C(v, t) + w_M(t) \cdot P_M(v, t)$$

The prefetch set is returned via deterministic tie-breaking:
$$\hat{A}_t = \operatorname{TopB}_v \left( P_{\text{ACSH}}(v, t) \right)$$
Ties are broken first by descending probability, then by ascending expert ID.

---

## 5. Architectural Variants

### Variant 1: ACSH Global
Maintains a single reliability pair $(R_C(t), R_M(t))$ per request stream.
- Shared across all stages within a request.
- Captures macro-level workload shifts (e.g., code vs. text prompt).

### Variant 2: ACSH Stage-Aware
Maintains independent reliability pairs per stage and horizon:
$$R_C(L, H, t), \quad R_M(L, H, t)$$
- Allows early layers to trust CSTP (if temporal locality is high) and late layers to trust ST-MoE (if spatial co-activation dominates).

### Variant 3: ACSH Stage-Aware Fast Recovery
Augments Stage-Aware ACSH with a causal shift detector:
- Computes recent rolling prediction error over window $W=3$:
  $$\bar{e}(t) = 1 - \frac{1}{W} \sum_{i=1}^W r_{\text{ACSH}}(t-i)$$
- If $\bar{e}(t) > \theta_{\text{shift}}$ (e.g., error $> 0.85$ over 3 tokens), a distribution shift is flagged.
- Upon detection, the momentum parameter temporarily drops to $\beta_{\text{fast}} = 0.2$ for 3 tokens, enabling near-instantaneous weight recalibration without legacy baggage.

---

## 6. Worked Numerical Example

Consider an MoE layer with $E=4$ experts, Top-$K=2$, budget $B=2$, $\beta = 0.8$, $\tau = 0.2$, $T_C = 1.0$, $T_M = 1.0$.

### Event $t=0$:
- Prior reliability: $R_C(0) = 0.5, R_M(0) = 0.5$.
- Weights: $w_C(0) = 0.5, w_M(0) = 0.5$.
- CSTP raw scores: $S_C = [0.4, 0.3, 0.2, 0.1]$.
  Normalized $P_C = [0.311, 0.281, 0.222, 0.186]$.
- ST-MoE confidence: $S_M = [3, 0, 2, 0]$.
  Normalized $P_M = [0.659, 0.033, 0.242, 0.033]$.
- ACSH ensemble score:
  $$P_{\text{ACSH}} = 0.5 \times [0.311, 0.281, 0.222, 0.186] + 0.5 \times [0.659, 0.033, 0.242, 0.033] = [0.485, 0.157, 0.232, 0.110]$$
- Prefetch Set: Top-2 candidates are **Expert 0** (0.485) and **Expert 2** (0.232).

### Evaluation & Causal Update:
- Actual ground-truth revealed: $A_0 = \{0, 1\}$.
- CSTP Top-2 $\{0, 1\}$ hits $A_0$ with 2 hits $\implies r_C(0) = 2/2 = 1.0$.
- ST-MoE Top-2 $\{0, 2\}$ hits $A_0$ with 1 hit $\implies r_M(0) = 1/2 = 0.5$.
- ACSH Top-2 $\{0, 2\}$ hits $A_0$ with 1 hit $\implies r_{\text{ACSH}}(0) = 0.5$.

### State Update for Event $t=1$:
- New reliability:
  $$R_C(1) = 0.8 \times 0.5 + 0.2 \times 1.0 = 0.60$$
  $$R_M(1) = 0.8 \times 0.5 + 0.2 \times 0.5 = 0.50$$
- New fusion weights:
  $$\frac{R_C(1)}{\tau} = \frac{0.60}{0.2} = 3.0, \quad \frac{R_M(1)}{\tau} = \frac{0.50}{0.2} = 2.5$$
  $$w_C(1) = \frac{\exp(3.0)}{\exp(3.0) + \exp(2.5)} = \frac{20.086}{20.086 + 12.182} = \mathbf{0.622}$$
  $$w_M(1) = 1 - 0.622 = \mathbf{0.378}$$
- Result: Because CSTP achieved higher recall on event 0, ACSH autonomously increased CSTP's voting share from $50.0\%$ to **$62.2\%$** for event 1!

