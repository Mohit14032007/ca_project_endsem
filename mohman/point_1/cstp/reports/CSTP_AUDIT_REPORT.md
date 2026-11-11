# Phase 0: CSTP Read-Only Comprehensive Audit Report

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Focus Area:** Point 1 — Proposed Expert Prediction Technique: CSTP (Correlation-based Spatio-Temporal Predictor)  
**Evaluator Reference:** [`LLMSimulator/src/predictor/evaluation.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/evaluation.py)  
**Python Source:** [`LLMSimulator/src/predictor/cstp_predictor.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py)  
**C++ Source:** [`LLMSimulator/src/prefetch/cstp_predictor.h`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/cstp_predictor.h), [`cstp_predictor.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/cstp_predictor.cpp)  
**Authoritative Baseline Reference (ST-MoE):** Recall@8 = **47.91%** (`moe_anywhere`, $H=1$), **26.53%** (`moe_suite`, $H=1$) under `online_within_request`.

---

## 1. Executive Summary

This audit provides a comprehensive, read-only architectural, mathematical, and empirical evaluation of our proposed expert prediction technique, **CSTP (Correlation-based Spatio-Temporal Predictor)**, before initiating code modifications or experimental campaigns.

### Key Audit Findings:
1. **Mathematical Soundness:** The core Dirichlet-smoothed formulations for target-stage frequency priors $P_{\text{prior}}(v \mid S)$, spatial conditional probabilities $S_{\text{spatial}}(v)$, temporal cross-token transition probabilities $S_{\text{temporal}}(v)$, and convex composite weighting $S_{\text{CSTP}}(v)$ are mathematically sound and produce strictly normalized categorical distributions ($\sum_v S(v) = 1.0$).
2. **Promising Offline Correlation Modeling:** Historical frozen evaluations on `moe_anywhere` ($H=1, B=8$) reveal that frozen CSTP achieved **51.25%** recall, outperforming frozen ST-MoE (**49.47%**) by **+1.78 percentage points**. This confirms the fundamental validity of probability-based correlation modeling.
3. **The Cumulative-Count Inertia Pathology:** In online serving, CSTP's update rule accumulates unscaled observation counts directly into massive offline count matrices ($N_{\text{offline}} > 100,000$). Consequently, an online update contributes only $\approx 0.01\%$ weight to conditional probabilities, completely paralyzing online adaptation on newly observed transitions. On `moe_suite` ($H=2$), CSTP online recall remained virtually flat (+0.11 pp over frozen: 16.03% $\to$ 16.14%), whereas ST-MoE's 2-bit saturating counters surged (+8.20 pp: 17.88% $\to$ 26.08%).
4. **Temporal Modeling Discrepancy:** ST-MoE uses a request-local recency buffer (an LRU cache of recently active experts), whereas CSTP models temporal correlation as a 1-step token-to-token transition matrix $P(v^{(t)} \mid w^{(t-1)})$. When task tokens route to different experts within a cluster, sparse transition matrices cause temporal scores to collapse to the global frequency prior.
5. **Python / C++ Parity Gap:** Python `CSTPPredictor` implements trace profiling (`fit_profile()`), whereas C++ `CSTPPredictor` only exposes raw setters (`set_marginal_count`, `set_spatial_count`, `set_temporal_count`). In standalone C++ tools, CSTP ran with uninitialized zero counts (flat prior).
6. **No Authoritative Hardened Results:** Existing CSTP evaluations in `results/endsem_phase_c/` and `results/endsem_phase1/` are diagnostic. None have been evaluated under the hardened Phase 2B `online_within_request` standard on the authoritative 16,800 (`moe_anywhere`) and 7,500 (`moe_suite`) opportunity sets.

---

## 2. Mathematical Implementation Audit

The intended CSTP equations and their concrete implementation in Python and C++ were systematically audited:

### 2.1 Empirical Target-Stage Marginal Frequency Prior
The intended formulation:

$$P_{\text{prior}}(v \mid S) = \frac{Count(S, v) + \epsilon}{\sum_{v'} Count(S, v') + E\epsilon}$$

- **Implementation:** [`cstp_predictor.py:124-135`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L124-L135), [`cstp_predictor.cpp:80-110`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/cstp_predictor.cpp#L80-L110).
- **Verification:**
  $$\sum_{v=0}^{E-1} \left(Count(S, v) + \epsilon\right) = \sum_v Count(S, v) + E\epsilon = \text{denom} \implies \sum_v P_{\text{prior}}(v \mid S) = 1.0$$
- **Smoothing:** When $Count(S, v) = 0$, $P_{\text{prior}}(v \mid S) = \frac{\epsilon}{\sum Count + E\epsilon} > 0$. Zero division is impossible because $\epsilon = 0.1 > 0$.
- **Audit Verdict:** **Fully verified and mathematically exact.**

---

### 2.2 Spatial Conditional Probability
The intended formulation:

$$P_{\text{spatial}}(v \mid u, S_{\text{src}}) = \frac{C_{\text{spatial}}(S_{\text{src}}, u, v) + \lambda P_{\text{prior}}(v \mid S)}{\sum_{v'} C_{\text{spatial}}(S_{\text{src}}, u, v') + \lambda}$$

Aggregated across active source experts $U = \{u_1, \dots, u_{|U|}\}$:

$$S_{\text{spatial}}(v) = \frac{1}{|U|} \sum_{u \in U} P_{\text{spatial}}(v \mid u, S_{\text{src}})$$

- **Implementation:** [`cstp_predictor.py:137-176`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L137-L176), [`cstp_predictor.cpp:112-161`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/cstp_predictor.cpp#L112-L161).
- **Verification:**
  $$\sum_v \left(C(u, v) + \lambda P_{\text{prior}}(v \mid S)\right) = \sum_v C(u, v) + \lambda \sum_v P_{\text{prior}}(v \mid S) = \text{total\_co} + \lambda(1.0) = \text{denom}$$
  $$\sum_v P_{\text{spatial}}(v \mid u, S_{\text{src}}) = 1.0 \implies \sum_v S_{\text{spatial}}(v) = \frac{1}{|U|} \sum_{u \in U} (1.0) = 1.0$$
- **Cold-Start / Boundary Fallbacks:**
  - If $U = \emptyset$ (no active source experts): returns $P_{\text{prior}}(v \mid S)$.
  - If $S_{\text{tgt}} < H$ (no valid predecessor stage): returns $P_{\text{prior}}(v \mid S)$.
  - If source expert $u$ was never observed during training ($\sum_{v'} C(u, v') = 0$): $P_{\text{spatial}}(v \mid u) = \frac{\lambda P_{\text{prior}}(v)}{\lambda} = P_{\text{prior}}(v \mid S)$.
- **Audit Verdict:** **Fully verified, smooth fallback, properly normalized.**

---

### 2.3 Temporal Autocorrelation Probability
The intended formulation:

$$P_{\text{temporal}}(v \mid w, S) = \frac{C_{\text{temporal}}(S, w, v) + \lambda P_{\text{prior}}(v \mid S)}{\sum_{v'} C_{\text{temporal}}(S, w, v') + \lambda}$$

Aggregated across previous token's active experts $W = \{w_1, \dots, w_{|W|}\}$ at stage $S$:

$$S_{\text{temporal}}(v) = \frac{1}{|W|} \sum_{w \in W} P_{\text{temporal}}(v \mid w, S)$$

- **Implementation:** [`cstp_predictor.py:177-210`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L177-L210), [`cstp_predictor.cpp:163-221`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/cstp_predictor.cpp#L163-L221).
- **Verification:**
  $$\sum_v P_{\text{temporal}}(v \mid w, S) = 1.0 \implies \sum_v S_{\text{temporal}}(v) = 1.0$$
- **Cold-Start Fallback:** When $W = \emptyset$ (token position $t = 0$), falls back cleanly to $P_{\text{prior}}(v \mid S)$.
- **Structural Nuance Identified:** This models 1-step token transition Markov dynamics at the same layer ($w^{(t-1)} \to v^{(t)}$). If transition counts $C_{\text{temporal}}$ are sparse or unobserved, the score falls back completely to $P_{\text{prior}}(v \mid S)$. It does not give an explicit bonus for persistence ($w = v$) unless $C_{\text{temporal}}(S, w, w)$ was heavily populated during profiling.
- **Audit Verdict:** **Mathematically valid categorical distribution.**

---

### 2.4 Composite Score and Top-$B$ Prediction Selection
The intended formulation:

$$S_{\text{CSTP}}(v) = \alpha S_{\text{spatial}}(v) + (1 - \alpha) S_{\text{temporal}}(v)$$

- **Implementation:** [`cstp_predictor.py:231-240`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L231-L240), [`cstp_predictor.cpp:250-258`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/cstp_predictor.cpp#L250-L258).
- **Properties:** Because $\alpha \in [0.0, 1.0]$ and both components sum to 1.0, the composite score is a convex combination of two probability distributions:
  $$\sum_v S_{\text{CSTP}}(v) = \alpha(1.0) + (1 - \alpha)(1.0) = 1.0$$
- **Top-$B$ Selection & Tie-Breaking:** Candidate expert indices $v \in \{0, \dots, E-1\}$ are sorted by:
  1. Primary key: $S_{\text{CSTP}}(v)$ descending.
  2. Secondary key: Expert index $v$ ascending (deterministic tie-breaker).
  Top-$B$ candidates are emitted ($B=K=8$).
- **Audit Verdict:** **Fully verified; Python and C++ tie-breaking contracts match.**

---

## 3. Data Structures and Memory Footprint Audit

### 3.1 Data Structures Representation

| Component | Mathematical Meaning | Python Representation | C++ Representation | Shape ($S=75, E=256$) | Memory in RAM |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `stage_marginal_counts` | $\text{Count}(S, v)$ | `ndarray[float64]` | `vector<vector<double>>` | $75 \times 256$ | 150 KB |
| `spatial_counts` | $C_{\text{spatial}}(S_{\text{src}}, u, v)$ | `ndarray[float64]` | `vector<vector<vector<double>>>` | $(75-H) \times 256 \times 256$ | 38.3 MB |
| `temporal_counts` | $C_{\text{temporal}}(S, w, v)$ | `ndarray[float64]` | `vector<vector<vector<double>>>` | $75 \times 256 \times 256$ | 39.3 MB |
| `history_table` | $W = \{w\}$ at $t-1$ | `Dict[req, Dict[stage, Set]]`| `unordered_map<int, unordered_map<int, vector<int>>>`| Dynamic | < 1 MB |
| **Total Memory** | | | | | **~78 MB** |

*Finding:* Unlike ST-MoE's compact CCT (which caps candidates at $C=8$ per source expert, requiring only ~150 KB), CSTP allocates full dense 3D matrices ($S \times E \times E$). While 78 MB is trivial for CPU host RAM, it is ~500$\times$ larger than ST-MoE's memory footprint.

---

## 4. Root Cause Analysis: Cumulative-Count Inertia in Online Adaptation

The most critical defect in the original CSTP design is **Cumulative-Count Inertia**.

### 4.1 The Mathematical Mechanism
In [`cstp_predictor.py:255-284`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L255-L284) and [`cstp_predictor.cpp:271-314`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/cstp_predictor.cpp#L271-L314), online updates directly add integer increments (`+= 1.0`):

```python
self.stage_marginal_counts[target_stage_id, v] += 1.0
self.spatial_counts[source_stage, u, v] += 1.0
self.temporal_counts[target_stage_id, w, v] += 1.0
```

### 4.2 Magnitude Imbalance on Real Workloads
- **Offline Training Phase:**
  - On `moe_anywhere` (52 prompts, 13,364 tokens): Total marginal observations per stage $\approx 106,912$. Total spatial transitions per stage pair $\approx 855,296$. A typical frequent source expert $u$ accumulates $\sum_{v'} C_{\text{spatial}}(u, v') \approx 8,000$ counts.
  - On `moe_suite` (8 tasks, 65,528 tokens): A frequent source expert accumulates $\sum_{v'} C_{\text{spatial}}(u, v') \approx 16,000$ counts.
- **Online Test Phase:**
  - A test prompt has 30 or 50 tokens.
  - When an expert transition $u \to v$ occurs online, its count increments by $+1.0$.
  - The relative impact on conditional probability is:
    $$\Delta P = \frac{C(u, v) + 1.0 + \lambda P_{\text{prior}}}{N_{\text{offline}} + 1.0 + \lambda} - \frac{C(u, v) + \lambda P_{\text{prior}}}{N_{\text{offline}} + \lambda} \approx \frac{1.0}{N_{\text{offline}}} \approx \frac{1}{8,000} \approx \mathbf{0.0125\%}$$
  - If a new, highly relevant transition was unseen during offline training ($C_{\text{offline}} = 0$):
    After 5 consecutive online occurrences, its conditional probability is only $\frac{5}{8,005} \approx 0.00062$.
    Meanwhile, an irrelevant expert seen 300 times during offline training retains a probability of $\frac{300}{8,005} \approx 0.0375$.
    The offline expert is **60$\times$ more probable**, completely preventing the new transition from ever entering the Top-$B$ prediction set!

### 4.3 Empirical Proof from Existing Historical Data

| Dataset | Predictor | Evaluation Regime | Recall@8 | Hit Rate | Online Adaptation Delta |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **`moe_anywhere`** | CSTP | Frozen | 51.25% | 97.87% | Baseline |
| ($H=1$, Sage-A1.5) | CSTP | Online | **52.23%** | 97.91% | **+0.98 pp** |
| | ST-MoE Paper | Frozen | 49.47% | 96.85% | Baseline |
| | ST-MoE Paper | Online | **55.07%** | 97.64% | **+5.60 pp** |
| \midrule | | | | | |
| **`moe_suite`** | CSTP | Frozen | 16.03% | 73.73% | Baseline |
| ($H=2$, GLM-5.2) | CSTP | Online | **16.14%** | 74.04% | **+0.11 pp (Paralyzed)** |
| | ST-MoE Paper | Frozen | 17.88% | 78.01% | Baseline |
| | ST-MoE Paper | Online | **26.08%** | 89.09% | **+8.20 pp (Active)** |

### 4.4 Why ST-MoE Succeeds Online While CSTP Stalls
ST-MoE bounds its correlation counters to **2 bits** ($0 \dots 3$):
- Stored correlations saturate at 2 or 3.
- When an obsolete correlation is absent, its counter decrements: $3 \to 2 \to 1 \to 0$.
- A newly observed online correlation is inserted with confidence 1 or 2, and reaches maximum confidence (3) in just 1–2 tokens!
- Because ST-MoE has an effective sample memory of at most 3 observations, it adapts immediately to domain shifts.

---

## 5. Other Identified Weaknesses in Original CSTP

1. **Temporal Mixing Degradation:**
   In `results/endsem_phase_c/aggregate/ablation_summary.csv`:
   - `spatial_only` ($\alpha = 1.0$) achieved **51.92%** on `moe_anywhere` and **17.63%** on `moe_suite`.
   - `temporal_only` ($\alpha = 0.0$) achieved **43.11%** on `moe_anywhere` and **11.81%** on `moe_suite`.
   - Combined CSTP ($\alpha = 0.5$) achieved **51.25%** and **16.03%** — **lower than spatial alone!**
   - *Cause:* The equal 50/50 weighting ($\alpha = 0.5$) dilutes strong spatial predictions with weak 1-step temporal transition probabilities. Optimal $\alpha$ is likely $\ge 0.75$.
2. **Global Frequency Prior Bias:**
   In GLM-5.2, certain heavy-hitter experts have massive global frequencies. When $\lambda = 1.0$ and conditional counts are modest, $P_{\text{prior}}(v \mid S)$ dominates the ranking, repeatedly predicting the same heavy-hitter experts regardless of context.
3. **Inverted Default Parameter in Python vs. C++:**
   - In Python `CSTPPredictor.__init__`, `online_adaptation: bool = False` by default.
   - In C++ `CSTPPredictor::CSTPPredictor`, `online_adaptation: bool = true` by default.
   - If a Python script instantiates `CSTPPredictor` without explicitly passing `online_adaptation=True`, calling `update()` silently bypasses count updates.

---

## 6. Python / C++ Parity Status

| Feature | Python Implementation | C++ Implementation | Parity Status |
| :--- | :--- | :--- | :---: |
| **Probability Formulation** | Dirichlet smoothing, prior, spatial, temporal | Dirichlet smoothing, prior, spatial, temporal | **Identical** |
| **Tie-Breaking** | Score descending, index ascending | Score descending, index ascending | **Identical** |
| **Offline Profiling** | `fit_profile(records)` | None (relies on manual count setters) | **Divergent** |
| **Array Allocations** | Spatial allocated as $(S-H) \times E \times E$ | Spatial allocated as $S \times E \times E$ | Functional match |
| **Default `online_adaptation`**| `False` | `true` | **Inverted** |
| **Unit Test Coverage** | `test_cstp.py` (7/7 passing) | `test_phase2a_predictors.cpp` (35/35 passing) | Passing |

---

## 7. Audit of Existing Historical CSTP Results

| Historical Directory | Traces Used | Evaluation Harness | Status & Validity | Action Required |
| :--- | :--- | :--- | :--- | :--- |
| `results/endsem_phase1/` | `benchmark_correlated`, `deepseekv3_5step`, `duplex_a100_32req` | Pre-Phase 1 legacy | **Diagnostic / Obsolete** | Exclude from authoritative comparison. |
| `results/endsem_phase_c/` | `moe_anywhere`, `moe_suite` | Phase C harness (20 tokens/prompt; across-req drift) | **Diagnostic Only** | Useful historical reference; cannot mix with Phase 2B. |
| `results/phase_2b_replay/` | Replay traces | C++ simulator replay | **Diagnostic** | C++ CSTP ran with unprofiled zero counts. |
| **Authoritative Status** | **None** | Hardened Phase 2B (`online_within_request`) | **NO RUNS EXIST YET** | Must execute full matrix freshly under Phase 2B protocol. |

---

## 8. Proposed Solutions for CSTP Implementation (Phase 1 Preview)

To preserve scientific rigor, we will:
1. **Preserve `CSTP_original` Unchanged:** Maintain the exact existing implementation as the baseline reference.
2. **Implement `CSTP_decay` (Bounded Effective Sample Size / Normalization):**
   To resolve count inertia without abandoning Bayesian probability, scale the offline profiling counts to a bounded effective sample size $N_{\text{eff}}$ (e.g., $N_{\text{eff}} \in [10, 50]$):
   $$C_{\text{scaled}}(u, v) = \frac{C_{\text{profiling}}(u, v)}{\sum_{v'} C_{\text{profiling}}(u, v')} \times N_{\text{eff}}$$
   - **Why this works:** The relative conditional probability distribution learned offline is perfectly preserved.
   - However, the total count mass is normalized to $N_{\text{eff}}$.
   - When a new online transition occurs, its $+1.0$ update carries a meaningful weight of $\frac{1.0}{N_{\text{eff}} + 1} \approx 2\%–10\%$, allowing the predictor to adapt dynamically within 3–5 tokens!
3. **Hyperparameter Selection on Validation Split:**
   Tune $\alpha \in [0.0, 1.0]$, $\lambda \in [0.01, 10.0]$, and $\epsilon \in [0.01, 1.0]$ exclusively on the **validation split** (8 prompts on `moe_anywhere`, 2 tasks on `moe_suite`) to eliminate test leakage.

---

## 9. Next Steps and Required Authoritative Experiments

Once this audit is reviewed, the authorized sequence will proceed:
1. **Phase 1:** Add regression tests verifying normalization, tie-breaking, and Python/C++ parity. Implement `CSTP_decay` alongside `CSTP_original`.
2. **Phase 2:** Verify identical prediction opportunities against the finalized ST-MoE manifest (16,800 for `moe_anywhere`, 7,500 for `moe_suite`).
3. **Phase 3:** Execute primary authoritative matrix under `online_within_request` ($B=K=8, H \in \{1, 2\}$) for both `moe_anywhere` and `moe_suite`.
4. **Phase 4–12:** Run ablations, hyperparameter sweeps, ST-MoE head-to-head comparison, CSV exports, publication plots, and final packaging readiness report.

*All outputs will remain strictly outside `mohman/`.*

