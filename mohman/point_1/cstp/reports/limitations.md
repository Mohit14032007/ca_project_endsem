# CSTP Architectural Limitations & Analytical Discussion

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Deliverable:** Point 1 — Standalone CSTP Limitations  
**Path:** [`mohman/point_1/cstp/reports/limitations.md`](file:///home/mohman/ca_project/mohman/point_1/cstp/reports/limitations.md)  

---

## 1. Count-Mass Accumulation and Sample Efficiency Saturation

The primary theoretical bottleneck identified during Phase 1A experimentation is **count-mass accumulation** in CSTP's transition tables.

### Mathematical Mechanism
In CSTP, the smoothed conditional probability of an expert transition is computed as:

$$P(e_j \mid e_i) = \frac{C(e_i, e_j) + \lambda P_{\text{marginal}}(e_j)}{\sum_{e'} C(e_i, e') + \lambda}$$

Let $N = \sum_{e'} C(e_i, e')$ represent the total accumulated count mass for source condition $e_i$. During offline profiling on 100% of the training split:
- On `moe_anywhere`, $N$ reaches an average of $\sim 4,800$ counts per expert condition (total spatial mass = $1,238,400$).
- When online adaptation is active, each observed online transition increments $C(e_i, e_j)$ by $+1$, and increments the denominator by $+1$.
- The marginal sensitivity of the probability update with respect to a single observed online event is:

$$\frac{\partial P(e_j \mid e_i)}{\partial C_{\text{online}}} \approx \frac{1}{N + \lambda}$$

As $N$ scales from $3 \times 10^5$ (at 25% profiling) to $1.2 \times 10^6$ (at 100% profiling), the effective weight of each online token update is diluted by a factor of 4. Consequently:
- At 25% profiling, online adaptation provides **+0.85 pp** gain.
- At 100% profiling, online adaptation provides only **+0.14 pp** gain.

### Architectural Takeaway
In a heavily profiled system, static transition mass overwhelms dynamic adaptation. Without a decay factor, exponential moving average, or effective sample size (ESS) bound, CSTP's online adaptation capability naturally saturates as profiling duration increases.

---

## 2. Global Aggregation and Cross-Task Distribution Interference

CSTP aggregates transition counts globally across all profiling sequences into a single monolithic transition matrix:

$$C_{\text{spatial}}(L_{\text{src}}, e_i, L_{\text{tgt}}, e_j) = \sum_{\text{all prompts } p} C_p(L_{\text{src}}, e_i, L_{\text{tgt}}, e_j)$$

### Failure Mode Under Multitask Traces
When a model processes distinct tasks (e.g., Python code generation vs conversational QA vs mathematical reasoning):
- In MoE models, routing sub-networks specialize by semantic domain.
- Aggregating routing traces from diverse tasks sums discordant spatial co-activation patterns into the same table.
- Highly predictive transitions for Task A become smoothed out or diluted by uncorrelated transitions from Task B.
- This explains the descriptive drop from 18.03% to 16.47% recall on MoE Suite when profiling expands from 75% to 100%: the final 25% of training prompts introduced divergent domain distributions that reduced alignment with the test split (`gsm8k` and `alpaca`).

---

## 3. Statistical Scope Limitations of MoE Suite ($N=2$ Tasks)

A fundamental empirical caveat of the MoE Suite benchmark is the composition of its designated evaluation split:
- The evaluation split consists of only **two distinct test tasks**: `gsm8k` (30 prompts, 2,250 opportunities) and `alpaca` (70 prompts, 5,250 opportunities).
- Although the evaluation contains 7,500 token-level prediction opportunities, these opportunities are highly clustered within only two task distributions.
- Hierarchical bootstrap resampling treating prompts or tasks as independent clusters produces wide confidence intervals.
- Therefore, all MoE Suite aggregate figures and hypothesis comparisons are certified strictly as **`DESCRIPTIVE_ONLY_N2_TASKS`**. They must not be interpreted as strong generalizable population claims.

---

## 4. Cold-Start Performance Gap

During the initial 0–4 tokens of an incoming request:
- `recent_experts` history buffer is empty or partially populated.
- The temporal component cannot leverage intra-request repetition and defaults to the marginal prior: $S_{\text{temporal}} \to P_{\text{marginal}}$.
- On `moe_anywhere`, cold-start recall drops to **~32%**, compared to steady-state recall of **~48.5%**.
- Prefetch engines relying on CSTP must anticipate lower hit rates during prompt prefill and initial token decoding.

