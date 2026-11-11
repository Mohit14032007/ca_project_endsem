# Detailed Comparative Analysis: CSTP vs. ST-MoE

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Deliverable:** Point 1 — Comparative Study Analysis  
**Path:** [`mohman/point_1/stmoe_cstp/analysis/cstp_vs_st_moe.md`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/analysis/cstp_vs_st_moe.md)  
**Authoritative Data Sources:** [`results/csv/corrected_st_moe_head_to_head.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/corrected_st_moe_head_to_head.csv) & [`results/csv/corrected_statistical_analysis.csv`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/results/csv/corrected_statistical_analysis.csv)  

---

## 1. Head-to-Head Performance Matrix

The matched head-to-head comparison evaluates CSTP against ST-MoE across identical configurations (Seed 42, $H=1$, $B=K=8$):

```
+----------------------------------------------------------------------------------------------------------------------+
| Workload      | Profiling % | CSTP Frozen | ST-MoE Frozen | Frozen Delta | CSTP Online | ST-MoE Online | Online Delta    |
+---------------+-------------+-------------+---------------+--------------+-------------+---------------+-----------------+
| MoE Anywhere  | 25%         | 43.83%      | 42.79%        | +1.04 pp     | 44.68%      | 46.92%        | -2.24 pp        |
| MoE Anywhere  | 50%         | 44.53%      | 43.47%        | +1.06 pp     | 44.85%      | 47.52%        | -2.67 pp        |
| MoE Anywhere  | 75%         | 45.27%      | 44.01%        | +1.26 pp     | 45.48%      | 47.84%        | -2.37 pp        |
| MoE Anywhere  | 100%        | 45.37%      | 44.09%        | +1.28 pp     | 45.51%      | 47.91%        | -2.40 pp        |
| MoE Suite     | 25%         | 15.05%      | 18.12%        | -3.07 pp     | 15.48%      | 26.36%        | -10.87 pp       |
| MoE Suite     | 50%         | 14.95%      | 17.34%        | -2.39 pp     | 15.21%      | 26.07%        | -10.86 pp       |
| MoE Suite     | 75%         | 18.03%      | 19.36%        | -1.33 pp     | 18.14%      | 26.80%        | -8.66 pp        |
| MoE Suite     | 100%        | 16.47%      | 18.50%        | -2.03 pp     | 16.59%      | 26.53%        | -9.94 pp        |
+----------------------------------------------------------------------------------------------------------------------+
```

---

## 2. Theoretical Breakdown of Trade-Offs

### 2.1 The Frozen Advantage of CSTP
On MoE Anywhere, CSTP consistently achieves higher frozen recall than ST-MoE (+1.04 pp to +1.28 pp). 
- **Cause:** ST-MoE's CCT stores discrete integer confidence values ($0 \dots 3$) thresholded at $\tau=2$. Rare or tail expert co-occurrences that occurred only once or twice during profiling fail to cross the threshold or get flattened into identical confidence scores.
- CSTP uses continuous Dirichlet smoothing with stage-specific marginal priors. Even when a specific transition is infrequent, the marginal prior provides a calibrated continuous tie-breaker, yielding superior ranking fidelity.

### 2.2 The Online Adaptation Advantage of ST-MoE
Under `online_within_request`, ST-MoE consistently achieves higher online recall (+2.24 pp to +2.67 pp on MoE Anywhere; +8.66 pp to +10.87 pp on MoE Suite).
- **Cause:** ST-MoE's 2-bit counters have a maximum capacity of 3 and start from 1. Just 2 successive activations promote a candidate to maximum confidence ($3$). Furthermore, the History Table assigns a score of 1 or 2 based directly on token recency.
- In contrast, CSTP's online updates add $+1$ count to an accumulated denominator of $N \approx 10^5 \dots 10^6$, causing online probability shifts to be substantially more conservative.
- **Architectural Caveat:** While ST-MoE demonstrates higher empirical online gain in these traces, we avoid stating that the 2-bit counter alone caused the gain without a dedicated counter-depth ablation.

