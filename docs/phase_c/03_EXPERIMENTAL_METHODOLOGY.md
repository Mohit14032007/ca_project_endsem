# Phase C: Experimental Methodology & Evaluation Protocol

## 1. Overview and Research Objective

Phase C establishes a rigorous, reproducible, multi-dataset benchmarking framework to evaluate MoE expert prediction and prefetch utility. The central investigation evaluates whether our proposed **Confidence- and Frequency-Aware Spatio-Temporal Predictor (CSTP)** outperforms the published **ST-MoE** baseline (Zhao et al., arXiv:2606.15453v1) under identical routing traces, identical causal adaptation regimes, and identical prefetch budgets.

This phase evaluates expert prediction quality and state overheads offline. It does **not** implement asynchronous C++ DRAM prefetching or claim end-to-end wall-clock latency speedup.

---

## 2. Workload Suite Specifications

The evaluation encompasses 5 distinct MoE routing workloads spanning synthetic benchmarks, hardware simulation traces, and real-world multi-task LLM serving workloads:

| Workload ID | Source / Origin | Num Experts ($E$) | Top-$K$ | Stages / Layers ($S$) | Evaluated Units | Primary Horizon ($H$) | Primary Budget ($B$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`deepseek_random`** | LLMSimulator synthetic trace | 256 | 8 | 57 | 1 unit (5 tokens) | $H=1$ | $B=8$ |
| **`duplex_a100`** | LLMSimulator A100 trace | 8 | 2 | 32 | 1 unit (5 tokens) | $H=1$ | $B=2$ |
| **`correlated_synthetic`** | Markov cluster synthetic | 256 | 8 | 57 | 1 unit (5 tokens) | $H=1$ | $B=8$ |
| **`moe_anywhere`** | Real prompt traces (Qwen/DeepSeek) | 256 | 8 | 40 | 14 prompts (20 tokens) | $H \in \{1, 2\}$ | $B=8$ |
| **`moe_suite`** | `Predicting-Experts-for-MOE-Suite` | 256 | 8 | 75 | 2 tasks (50 tokens) | $H=2$ | $B=8$ |

### Data Splitting and Unit Isolation
All workloads follow strict unit-level partitioning without cross-contamination:
- **`deepseek_random`**: Partitioned by token index. Profiling on tokens 0–3, test evaluation on token 4.
- **`duplex_a100`**: Partitioned by request index. 32 requests partitioned into 22 train, 5 validation, and 5 test requests (0 request overlap).
- **`correlated_synthetic`**: 16 requests partitioned into 10 train, 3 validation, and 3 test requests.
- **`moe_anywhere`**: 74 prompts partitioned into 44 train, 15 validation, and 15 test prompts.
- **`moe_suite`**: 12 tasks partitioned according to official benchmark disjoint task splits: 8 train tasks, 2 validation tasks (`corpus-10`, `corpus-11`), and 2 test tasks (`corpus-12`, `corpus-13`).

---

## 3. Evaluated Predictor Configurations

A total of 12 distinct predictor configurations were evaluated across all workloads:

1. **`cstp_online`**: Full CSTP with online Bayesian Dirichlet-multinomial transition count updates ($\alpha = 0.5$).
2. **`cstp_frozen`**: CSTP initialized from offline profiling; transition counts are held fixed during inference, while the request temporal history table remains online.
3. **`st_moe_paper_online`**: Paper-faithful ST-MoE implementation (Algorithms 1, 2, and 3). 2-bit saturating counters in CCT ($C=2$) update online upon expert match/mismatch; zero-confidence candidates are replaced. History table updates post-routing.
4. **`st_moe_paper_frozen`**: Paper-faithful ST-MoE initialized from profiling; CCT counter updates and candidate replacements are suppressed during inference. History table remains active.
5. **`st_moe_legacy`**: Pre-existing reference ST-MoE implementation for backwards validation.
6. **`frequency_only`**: Ranks experts strictly by global marginal routing frequency from profiling.
7. **`temporal_only`**: CSTP variant with spatial weight set to 0 ($\alpha = 0.0$); relies solely on token-to-token temporal transitions and marginal priors.
8. **`spatial_only`**: CSTP variant with temporal weight set to 0 ($\alpha = 1.0$); relies solely on stage-to-stage spatial co-activations and marginal priors.
9. **`random`**: Uniformly samples $B$ distinct experts from $\{0, \dots, E-1\}$.
10. **`cstp_st_moe_hybrid`**: Ensembled combination of normalized CSTP probability distributions and ST-MoE integer confidence scores.
11. **`cstp_cold_start`**: Online CSTP started with completely empty transition counts and uniform marginal priors (zero offline profiling).
12. **`st_moe_paper_cold_start`**: Online ST-MoE started with an empty CCT (zero offline profiling).

---

## 4. Evaluation Protocol and Causality Guarantees

The evaluation follows a strict causal step-by-step pipeline ensuring zero target leakage:

```
For each request r:
    Reset predictor request-level state (clear History Table)
    For each token t:
        For each MoE stage L:
            Target stage L_tgt = L + H
            
            1. PREDICT:
               Predictor queries internal state using current source experts E_src(t, L).
               Outputs Top-B prefetch candidates P = {p_1, ..., p_B}.
               
            2. EVALUATE:
               Reveal actual target routing A = {a_1, ..., a_K}.
               Compute Hits = |P ∩ A|, Recall = Hits / K, Precision = Hits / B, F1, Hit Rate.
               Record evaluation row to CSV.
               
            3. CAUSAL STATE UPDATE:
               Pass actual target routing A to predictor.update().
               - If online: update transition counts / CCT saturating counters.
               - Update history table with A for subsequent tokens.
```

### Invariant Checks
1. **Target Invariance**: Predictions depend only on history prior to time $(t, L_{\text{tgt}})$.
2. **Frozen Invariance**: In frozen regime, transition matrices and CCT counters undergo 0 bit changes across all evaluation steps.
3. **Budget Independence**: Each budget $B \in \{4, 8, 16, 24, 32, 48, 64\}$ is evaluated with an independently cloned predictor instance, guaranteeing identical evaluation trajectories.

---

## 5. Metric Formulations

For each prediction event $i$ where target size is $K = \text{Top-}K$, budget is $B$, predicted set is $P_i$, and ground-truth set is $A_i$:

- **Hits**: $H_i = |P_i \cap A_i|$
- **Recall@B**: $R_i = \frac{H_i}{K}$
- **Precision@B**: $P_i = \frac{H_i}{B}$
- **F1@B**: $F1_i = \frac{2 \cdot P_i \cdot R_i}{P_i + R_i}$ (0 if $P_i + R_i = 0$)
- **Hit Rate**: $HR_i = \mathbb{I}(H_i \ge 1)$
- **Macro Recall**: $\frac{1}{N} \sum_{i=1}^N R_i$
- **Micro Recall**: $\frac{\sum_{i=1}^N H_i}{N \cdot K}$
