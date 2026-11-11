# Evaluation Protocol & Causal Invariants

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Deliverable:** Point 1 — Comparative Study Methodology  
**Path:** [`mohman/point_1/stmoe_cstp/methodology/evaluation_protocol.md`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/methodology/evaluation_protocol.md)  

---

## 1. Causal Execution Lifecycle

Expert prediction operates in an online inference pipeline where token execution proceeds sequentially. The evaluation harness enforces strict temporal causality at every step:

```
[Token t, Stage L_src]
       |
       v
1. Generate Candidate Prediction Set P_tgt (size B = 8)
       |
       v
2. [Causal Freeze] Record P_tgt BEFORE observing actual ground truth G_tgt
       |
       v
3. Receive Router Output G_tgt = TopK(Softmax(W_g x), K)
       |
       v
4. Compute Metrics:
   - Hits = |P_tgt ∩ G_tgt|
   - Precision = Hits / |P_tgt|
   - Recall = Hits / |G_tgt|
   - Hit_Rate = I(Hits >= 1)
       |
       v
5. [Post-Scoring Mutation] If regime == online_within_request:
   - ST-MoE: Update CCT counters (+1 for hits, -1 for misses), update HT
   - CSTP: Increment spatial_counts, temporal_counts, stage_marginal_counts
```

---

## 2. Evaluation Regimes Defined

### 2.1 `online_within_request` (Primary Causal Benchmark)
- **Initialization:** CCT (ST-MoE) or count tables (CSTP) are initialized from offline profiling.
- **Intra-Request Adaptation:** As tokens are generated within request $R$, online updates occur causally after scoring each opportunity.
- **Request Boundary Reset:** At the completion of request $R$, the entire predictor internal state is restored to the offline-profiled baseline before evaluating request $R+1$.
- **Scientific Significance:**
  1. Guarantees strict **request-order invariance**: evaluating requests in order $[R_1, R_2, R_3]$ yields numerically identical results to $[R_3, R_1, R_2]$.
  2. Accurately simulates multi-tenant production inference servers where subsequent requests originate from unrelated users and domains.

### 2.2 `frozen` (Static Baseline)
- Zero state mutations occur during the entire evaluation phase.
- Isolates the pure quality of the offline-profiled representation without online adaptation assistance.

