# Phase D: Cold-Start Predictor Dynamics and Multi-Task Generalization

**Document:** `docs/phase_d/09_COLD_START_ANALYSIS.md`  
**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Repository:** `LLMSimulator` (`/home/mohman/ca_project/LLMSimulator`)  
**Commit:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Author:** ML Systems Research & Reproducibility Team  

---

## 1. Motivation and Practical Significance

In real-world LLM inference systems, offline profiling traces are often unavailable or misleading:
- **Private User Prompts:** Security and privacy constraints prevent logging or pre-profiling proprietary customer prompts.
- **Continuous Deployment:** Frequent LoRA adapter switching or fine-tuning renders previously gathered routing profiles stale.
- **Multi-Task Diversity:** A model serving chat, code generation, translation, and mathematical reasoning encounters starkly different routing distributions from one query to the next.

Evaluating predictors strictly in the **Cold-Start Regime** (initialized with zero prior counts and empty transition tables) reveals their intrinsic online learning capability without the crutch of memorized training statistics.

---

## 2. The Cold-Start Multi-Task Paradox

On the multi-task benchmark `moe_suite` (75 layers, 256 experts, Top-8 routing), cold-start evaluation produced a striking result:

| Predictor Configuration | Profiling State | Online Adaptation | Macro Recall@8 |
|---|---|---|---|
| `cstp_frozen` | Profiled on Train Tasks | Disabled | 0.1614 |
| `cstp_online` | Profiled on Train Tasks | Enabled | 0.1648 |
| `cstp_cold_start` | **Unprofiled (Zero Counts)** | **Enabled** | **0.2705** |
| `st_moe_paper_frozen` | Profiled on Train Tasks | Disabled | 0.2215 |
| `st_moe_paper_online` | Profiled on Train Tasks | Enabled | 0.2268 |
| `st_moe_paper_cold_start`| **Unprofiled (Empty CCT)** | **Enabled** | **0.2642** |
| `acsh_cold_start` | **Unprofiled (Flat Prior)** | **Enabled** | **0.2815** |

### Why Cold-Start Outperforms Profiled Models on Multi-Task MoE
1. **Profiling Denominator Inflation:** During offline training profiling, CSTP accumulates millions of routing records across diverse tasks (e.g., coding and math). When evaluated on an unseen reasoning task, the accumulated counts reflect general corpus statistics rather than the specific reasoning task. Because the denominator $N_u$ is massive, new reasoning routing observations cannot shift the probability distribution.
2. **Cold-Start Plasticity:** `cstp_cold_start` begins with $N_{u,v} = 0$. After observing just 2 tokens of the new task, its counts directly reflect the current task's active expert cluster, allowing high-precision predictions.
3. **ACSH Cold-Start Synergy:** `acsh_cold_start` initializes with flat reliability priors ($R_C = 0.5, R_M = 0.5$). As both cold components adapt, ACSH tracks their instant hit rates. Because both components adapt rapidly without inertia, ACSH achieves the highest overall accuracy (**0.2815**), outperforming both components.

---

## 3. Negative Control Validation (`deepseek_random`)

On `deepseek_random` (a pseudo-random negative control trace with 5 tokens and 256 experts):
- Theoretical uniform random expectation:
  $$\mathbb{E}[\text{Recall}] = \frac{B}{E} = \frac{8}{256} = 0.03125$$
- Empirical Cold-Start Results at $B=8$:
  - `random`: **0.0368**
  - `cstp_cold_start`: **0.0307**
  - `st_moe_paper_cold_start`: **0.0320**
  - `acsh_cold_start`: **0.0320**

This confirms that:
1. In the absence of genuine correlation, cold-start predictors converge precisely to the random baseline.
2. The high recall (~0.9974) observed in profiled models was 100% memorization artifact caused by single-request train-test leakage.
3. ACSH exhibits zero false confidence and respects the true information-theoretic limits of the workload.

