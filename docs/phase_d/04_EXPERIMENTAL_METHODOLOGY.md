# Phase D: Experimental Methodology

**Document:** `docs/phase_d/04_EXPERIMENTAL_METHODOLOGY.md`  
**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Repository:** `LLMSimulator` (`/home/mohman/ca_project/LLMSimulator`)  
**Commit:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Author:** ML Systems Research & Reproducibility Team  

---

## 1. Objective and Research Questions

The primary objective of Phase D is to design, implement, and rigorously benchmark **ACSH (Adaptive Confidence and Statistical Hybrid)** against our proposed frequency-aware **CSTP** and the published **paper-faithful ST-MoE** baseline (Zhao et al., arXiv:2606.15453v1).

The investigation addresses four primary scientific questions:
1. **Can ACSH outperform both component predictors across diverse MoE routing regimes?**  
   Does dynamic reliability-weighted fusion achieve higher prediction recall, precision, and F1 than CSTP or ST-MoE alone across disjoint multi-task routing, synthetic Markov correlation, and multi-tenant traces?
2. **Does stage- and horizon-specific reliability tracking outperform global reliability tracking?**  
   Does partitioning reliability tracking per layer $(L, H)$ provide superior adaptability compared to a single global pair $(R_C, R_M)$?
3. **Can fast error detection enable rapid recovery during distribution shift?**  
   Does monitoring the rolling prediction error $\bar{e}(t) > \theta$ allow ACSH Fast Recovery to regain pre-shift accuracy faster than fixed-rate adaptation?
4. **What is the host-side computational and memory overhead of ACSH?**  
   Does ACSH maintain sub-microsecond query latency and minimal memory footprint suitable for real-time CPU prefetch orchestration?

---

## 2. Predictor Configurations (15 Variants)

Phase D evaluates an exhaustive suite of 15 predictor configurations under identical experimental conditions:

| ID | Predictor Name | Learning Mode | Description |
|---|---|---|---|
| 1 | `cstp_online` | Online | CSTP with online count accumulation |
| 2 | `cstp_frozen` | Frozen | CSTP with offline profiled counts, updates disabled |
| 3 | `st_moe_paper_online` | Online | Paper-faithful ST-MoE with 2-bit saturating counters |
| 4 | `st_moe_paper_frozen` | Frozen | ST-MoE Paper with profiled CCT, counter updates disabled |
| 5 | `cstp_st_moe_hybrid` | Online | Fixed static hybrid ($w_C = 0.5, w_M = 0.5$) |
| 6 | `acsh_global` | Online | ACSH with single global reliability pair $(R_C(t), R_M(t))$ |
| 7 | `acsh_stage_aware` | Online | ACSH with stage- and horizon-specific pairs $(R_C(L,H,t), R_M(L,H,t))$ |
| 8 | `acsh_fast_recovery` | Online | ACSH Stage-Aware with rolling miss-burst fast adaptation |
| 9 | `frequency_only` | Frozen | Offline marginal frequency baseline |
| 10 | `temporal_only` | Online | CSTP with only temporal co-occurrence ($\alpha=1.0$) |
| 11 | `spatial_only` | Online | CSTP with only spatial Markov transitions ($\alpha=0.0$) |
| 12 | `random` | Baseline | Uniform random selection from expert pool |
| 13 | `cstp_cold_start` | Cold-Start | CSTP initialized with zero counts (unprofiled) |
| 14 | `st_moe_paper_cold_start` | Cold-Start | ST-MoE initialized with empty CCT (unprofiled) |
| 15 | `acsh_cold_start` | Cold-Start | ACSH initialized with unprofiled components |

---

## 3. Causal Evaluation Protocol & Zero-Leakage Guarantee

To prevent temporal leakage and optimistic evaluation artifacts:
1. **Strict Causal Sequencing:** For every token $t$ and stage $L$, the predictor's `predict()` method is invoked strictly *before* the ground-truth target experts $A_{t, L+H}$ are revealed.
2. **Post-Reveal Update:** The ground truth $A_{t, L+H}$ is revealed only during the subsequent `update()` call, ensuring pending predictions are evaluated strictly causally.
3. **Split Isolation:** Profiling tensors are constructed exclusively from `split="train"`. All reported evaluation metrics derive strictly from `split="test"`. Validation tuning and ablation sweeps operate on internal holdout slices of `split="train"` without ever exposing `split="test"`.
4. **Independent Budget Instantiation:** For each prefetch budget $B$, a fresh predictor instance is constructed. Predictor internal state is never shared or carried over across different budget evaluations.

---

## 4. Evaluation Metrics

For prediction set $P$ ($|P| = B$) and actual routed target set $A$ ($|A| = K$):

- **Instance Hits:** $H = |P \cap A|$
- **Instance Recall:** $\text{Rec} = \frac{|P \cap A|}{|A|}$
- **Instance Precision:** $\text{Prec} = \frac{|P \cap A|}{|P|}$
- **Instance F1 Score:** $\text{F1} = \frac{2 \cdot \text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$ (defined as $0.0$ when $\text{Prec} + \text{Rec} = 0$)
- **Instance Hit Rate:** $\text{HR} = 1.0$ if $|P \cap A| \ge 1$ else $0.0$
- **Macro Metrics:** Arithmetic mean of instance metrics across all evaluated tokens and stages:
  $$\text{MacroRecall} = \frac{1}{N} \sum_{i=1}^N \text{Rec}_i, \quad \text{MacroPrecision} = \frac{1}{N} \sum_{i=1}^N \text{Prec}_i, \quad \text{MacroF1} = \frac{1}{N} \sum_{i=1}^N \text{F1}_i$$
- **Micro Metrics (Coverage):** Global aggregated ratio:
  $$\text{Coverage} = \frac{\sum_{i=1}^N H_i}{N \cdot K}, \quad \text{MicroPrecision} = \frac{\sum_{i=1}^N H_i}{N \cdot B}$$

---

## 5. Statistical Rigor and Reproducibility

- **Fixed Random Seed:** All stochastic components (e.g., Random baseline, tie-breaking fallbacks) use fixed seed `42`.
- **Paired Statistical Testing:** Paired two-tailed Student's t-tests and Wilcoxon signed-rank tests assess significance between ACSH and baselines on matching sequence tokens.
- **Confidence Intervals:** 95% confidence intervals ($\pm 1.96 \cdot \text{SEM}$) accompany all reported differences.
- **Zero Ternary Operators:** All newly written Python code conforms strictly to the zero-ternary rule (`ast.IfExp == 0`), verified via automated AST parsing.
- **Hardware Neutrality:** Phase D benchmarks host-side algorithmic utility only. No DRAM simulation cycles or hardware speedups are asserted.

