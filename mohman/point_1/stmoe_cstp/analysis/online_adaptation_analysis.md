# Online Adaptation Dynamics & Attenuation Analysis

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Deliverable:** Point 1 — Comparative Study Analysis  
**Path:** [`mohman/point_1/stmoe_cstp/analysis/online_adaptation_analysis.md`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/analysis/online_adaptation_analysis.md)  
**Authoritative Data Sources:** [`results/csv/matched_comparisons.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/matched_comparisons.csv) & [`results/csv/aggregate_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/aggregate_metrics.csv)  

---

## 1. Quantification of Online Adaptation Gains

Online adaptation gain is defined as the absolute recall improvement obtained by updating internal predictor state during request execution relative to the static frozen baseline:

$$\Delta_{\text{online}} = R_{\text{online\_within\_request}} - R_{\text{frozen}}$$

### Matched Comparison (Seed 42, $H=1$):

| Workload | Profiling % | CSTP Online Gain (pp) | ST-MoE Online Gain (pp) | Gain Ratio (ST-MoE / CSTP) |
| :--- | :---: | :---: | :---: | :---: |
| **MoE Anywhere** | 25% | **+0.85 pp** | **+4.12 pp** | $4.85\times$ |
| **MoE Anywhere** | 50% | **+0.32 pp** | **+4.05 pp** | $12.66\times$ |
| **MoE Anywhere** | 75% | **+0.21 pp** | **+3.84 pp** | $18.29\times$ |
| **MoE Anywhere** | 100% | **+0.14 pp** | **+3.82 pp** | $27.29\times$ |
| **MoE Suite** | 25% | **+0.43 pp** | **+8.24 pp** | $19.16\times$ |
| **MoE Suite** | 50% | **+0.26 pp** | **+8.73 pp** | $33.58\times$ |
| **MoE Suite** | 75% | **+0.11 pp** | **+7.44 pp** | $67.64\times$ |
| **MoE Suite** | 100% | **+0.12 pp** | **+8.03 pp** | $66.92\times$ |

---

## 2. Divergent Scaling Mechanisms

### 2.1 Why CSTP Online Gain Decays with Profiling
In CSTP, the online update for an observed transition $(e_i, e_j)$ is an integer count increment:

$$P_{\text{new}}(e_j \mid e_i) = \frac{C_{\text{prof}}(e_i, e_j) + 1 + \lambda P_{\text{marginal}}(e_j)}{\sum_{e'} C_{\text{prof}}(e_i, e') + 1 + \lambda}$$

When profiling is low ($25\%$), the total count denominator for expert $e_i$ is small ($N \approx 1,200$). Adding $+1$ alters the transition probability visibly (+0.85 pp gain). When profiling reaches $100\%$, the denominator reaches $N \approx 4,800$, reducing the probability shift by $75\%$.

### 2.2 Why ST-MoE Online Gain Remains Robust
ST-MoE's Correlation Confidence Table (CCT) stores 2-bit counters saturated at 3. The CCT does not sum counts; it tracks confidence state. Regardless of how many training prompts were processed, a counter that was at 1 or 2 can be promoted to 3 with only 1–2 online observations. Furthermore, ST-MoE's History Table dynamically promotes recently seen experts into the scoring equation ($\text{Score} = \text{CCT} + \text{HT}$) with substantial additive weight ($+1$ or $+2$). This grants ST-MoE immediate, high-amplitude agility during generation.

