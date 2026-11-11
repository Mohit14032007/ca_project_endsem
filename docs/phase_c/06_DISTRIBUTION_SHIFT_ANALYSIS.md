# Phase C: Distribution Shift Resilience & Online Recovery Analysis

## 1. Objective and Experimental Design

In real-world LLM inference systems, request streams experience frequent distribution shifts due to prompt topic changes, conversation mode transitions, or system multi-tenancy. A static (frozen) predictor risks catastrophic failure when routing patterns deviate from offline training corpora.

To rigorously evaluate how quickly predictors detect and recover from distribution changes, we conducted a controlled 100-token distribution shift benchmark:
- **Tokens 0–49 (Phase 1)**: Routing generated according to **Cluster A** (experts 0–127 active).
- **Tokens 50–99 (Phase 2)**: Routing shifts abruptly at token 50 to **Cluster B** (experts 128–255 active).
- **Architecture**: DeepSeek-V3 scale ($E=256$, Top-$K=8$, $S=57$ stages).
- **Evaluated Predictors**: `cstp_online`, `cstp_frozen`, `st_moe_paper_online`, `st_moe_paper_frozen`, `frequency_only`, `temporal_only`, `random`.
- **Primary Budget**: $B=8$.

---

## 2. Quantitative Recovery Metrics

The table below summarizes predictor behavior before the shift, immediately following the shift (tokens 50–54), and after convergence (tokens 75–99):

| Predictor Configuration | Pre-Shift Recall (Tokens 0–49) | Immediate Post-Shift Recall (Tokens 50–54) | Post-Shift Drop ($\Delta_{\text{drop}}$) | Late Post-Shift Recall (Tokens 75–99) | Convergence Speed (Tokens to 90% Stable) | Resilience Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`cstp_online`** | 0.7316 | 0.3867 | -0.3448 | **0.4341** | **3 tokens** | **Recovered** |
| **`cstp_frozen`** | 0.8037 | 0.1805 | -0.6233 | **0.0847** | Never | **Catastrophic Collapse** |
| **`st_moe_paper_online`**| 0.5919 | 0.4688 | -0.1231 | **0.4778** | **Immediate** | **Highly Resilient** |
| **`st_moe_paper_frozen`**| 0.6566 | 0.2313 | -0.4253 | **0.1766** | Never | **Catastrophic Collapse** |
| **`temporal_only`** | 0.7178 | 0.3727 | -0.3452 | **0.4406** | **3 tokens** | **Recovered** |
| **`frequency_only`** | 0.3364 | 0.1234 | -0.2130 | **0.0773** | Never | **Permanent Failure** |
| **`random`** | 0.0280 | 0.0258 | -0.0022 | **0.0336** | N/A | **Uninformative** |

---

## 3. Dynamics and Failure Mode Analysis

### A. Catastrophic Collapse of Frozen Predictors
Both `cstp_frozen` and `st_moe_paper_frozen` suffer total collapse:
- Prior to token 50, `cstp_frozen` achieves high recall (0.8037) by exploiting Cluster A patterns.
- Immediately upon shift to Cluster B, `cstp_frozen` recall plummets to 0.1805, and continues degrading to **0.0847** (barely above uniform random).
- Because `cstp_frozen` cannot update its transition tables, it persistently prefetches Cluster A experts for the remainder of the session, squandering prefetch bandwidth on unused weights.
- `frequency_only` similarly collapses to 0.0773, as its static marginal distribution has zero relevance to Cluster B.

### B. Rapid Online Adaptation & Counter Replacement
Both online models demonstrate strong self-healing capabilities:
- **`cstp_online`**: Absorbs the shock at token 50 and rebounds from 0.3867 to 0.4341 within **3 tokens**. Its Bayesian count updates steadily increment transitions into Cluster B experts, adjusting marginal and joint distributions in real time.
- **`st_moe_paper_online`**: Demonstrates the highest resilience, dropping only 0.1231 and stabilizing at **0.4778**. This superior resilience stems directly from Algorithm 3 Lines 11–14:
  - When an expert mismatch occurs, zero-confidence CCT entries are immediately replaced with the actual active experts from Cluster B.
  - Within 2 tokens, the 2-bit counters saturate to 2 and 3, fully repopulating the CCT with Cluster B pathways.

---

## 4. Key Takeaways for MoE System Architecture
1. **Static Predictors Are Unusable in Production**: Any MoE prefetcher deployed without online adaptation will fail when request distributions drift, leading to severe prefetch cache pollution.
2. **2-Bit CCT vs. Count Matrices**: ST-MoE's bounded 2-bit saturating counters provide faster adaptation during sudden regime shifts because they replace obsolete state with $O(1)$ turnover, whereas CSTP's unbounded float64 count matrices exhibit higher inertia and require dampening or sliding windows under extreme distribution drift.
