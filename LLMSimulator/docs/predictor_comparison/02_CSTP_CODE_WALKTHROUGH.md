# CSTP Code Walkthrough and Internal Data-Structure Audit

**Implementation Source File**: [`src/predictor/cstp_predictor.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py)  
**Standardized Wrapper**: [`CSTPWrapper` in `src/predictor/base_predictor.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/base_predictor.py#L140-L190)

---

## 1. Executive Summary & Design Philosophy

The **Confidence- and Frequency-Aware Spatio-Temporal Predictor (CSTP)** addresses the fundamental limitation of heuristic table lookup methods by formulating expert prefetching as a **continuous probability estimation problem**.

Instead of storing a small, hard-bounded set of expert IDs with integer counters, CSTP models the entire expert vocabulary ($E$ candidate experts) using:
1. **Dirichlet-Smoothed Spatial Conditional Transition Probabilities**: $P(v \mid u)$ across adjacent layers $(L \to L+1)$.
2. **Dirichlet-Smoothed Temporal Autocorrelation Probabilities**: $P(v \mid w)$ across consecutive tokens $(t-1 \to t)$ within the same layer.
3. **Target-Stage Empirical Marginal Frequency Priors**: $P_{\text{prior}}(v \mid \text{stage})$ with smoothing factor $\epsilon$.
4. **Parameterized Composite Weighting**: Balance parameter $\alpha \in [0.0, 1.0]$.

CSTP's mathematical operations evaluate probability mass over all $E$ candidates, ensuring that every expert has a well-calibrated score even under sparse observations or cold starts.

---

## 2. Exhaustive Data-Structure Audit Table

The following audit documents every internal data structure owned and operated by `CSTPPredictor`:

| Property | `self.stage_marginal_counts` | `self.spatial_counts` | `self.temporal_counts` | `self.history_table` |
| :--- | :--- | :--- | :--- | :--- |
| **Actual Variable Name** | `self.stage_marginal_counts` | `self.spatial_counts` | `self.temporal_counts` | `self.history_table` |
| **Data Type** | `numpy.ndarray` (`dtype=np.float64`) | `numpy.ndarray` (`dtype=np.float64`) | `numpy.ndarray` (`dtype=np.float64`) | `Dict[int, Dict[int, Set[int]]]` |
| **Shape / Index Structure** | Shape: `(num_moe_stages, num_routed_experts)` | Shape: `(num_moe_stages - 1, E, E)` | Shape: `(num_moe_stages, E, E)` | `[request_id][stage_id] -> Set[int]` |
| **Key Semantics** | `[stage, expert_id]` | `[source_stage, source_exp u, target_exp v]` | `[stage, prev_exp w, curr_exp v]` | Sequence ID $\to$ Stage ID $\to$ Expert ID set |
| **Stored Value** | Cumulative number of times expert $v$ was activated at $stage$ | Joint co-activation count of token selecting $u$ at $L$ and $v$ at $L+1$ | Transition count of sequence selecting $w$ at $t-1$ and $v$ at $t$ | Set of actual expert IDs activated during decoding step $t-1$ |
| **Initialization** | `np.zeros(...)` in `__init__` ([L40-L42](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L40-L42)) | `np.zeros(...)` in `__init__` ([L47-L53](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L47-L53)) | `np.zeros(...)` in `__init__` ([L57-L60](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L57-L60)) | `{}` empty dict in `__init__` ([L63](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L63)) |
| **Offline Update** | `fit_profile()` ([L94-L96](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L94-L96)) | `fit_profile()` ([L102-L106](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L102-L106)) | `fit_profile()` ([L117-L121](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L117-L121)) | N/A (Profiling groups sequences) |
| **Online Update** | `update()` if `online_adaptation=True` ([L242-L244](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L242-L244)) | `update()` if `online_adaptation=True` ([L247-L254](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L247-L254)) | `update()` if `online_adaptation=True` ([L256-L261](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L256-L261)) | `update()` always overwrites ([L264-L266](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L264-L266)) |
| **Read Method** | `get_marginal_prior()` ([L127-L134](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L127-L134)) | `compute_spatial_score()` ([L157](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L157)) | `compute_temporal_score()` ([L192](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L192)) | `compute_temporal_score()` ([L178-L180](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L178-L180)) |
| **Scope** | Instance-level (shared across all requests) | Instance-level (shared across all requests) | Instance-level (shared across all requests) | Request-level (partitioned by `request_id`) |
| **Lifetime** | Preserved across entire evaluation run | Preserved across entire evaluation run | Preserved across entire evaluation run | Cleared per request via `reset_request_state()` |

### 2.2 Hyperparameters & Configuration Variables

| Variable Name | Type | Default | Role & Mathematical Meaning |
| :--- | :--- | :---: | :--- |
| `self.alpha` | `float` | `0.5` | Spatial vs. temporal trade-off weight: $\text{Score} = \alpha S_{\text{spatial}} + (1-\alpha) S_{\text{temporal}}$. |
| `self.smoothing_lambda` | `float` | `1.0` | Dirichlet pseudocount scaling prior distribution mass when denominator count is low. |
| `self.prior_epsilon` | `float` | `0.1` | Laplace/Dirichlet additive smoothing ensuring non-zero marginal probability for all $E$ candidates. |
| `self.online_adaptation` | `bool` | `False` | When True, updates tensor transition counts dynamically during test inference. Default False ensures clean offline separation. |

---

## 3. Mathematical Formulations & Exact Source Alignment

### 3.1 Target-Stage Marginal Frequency Prior
Calculated in `get_marginal_prior(stage_id)` ([lines 123-134](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L123-L134)):
$$P_{\text{prior}}(v \mid \text{stage}) = \frac{\text{self.stage\_marginal\_counts}[\text{stage}, v] + \epsilon}{\sum_{k=0}^{E-1} \text{self.stage\_marginal\_counts}[\text{stage}, k] + (E \times \epsilon)}$$

*Code excerpt*:
```python
counts = self.stage_marginal_counts[stage_id]
total_counts = np.sum(counts)
denom = total_counts + (self.num_routed_experts * self.prior_epsilon)
if denom > 0.0:
    prior = (counts + self.prior_epsilon) / denom
else:
    prior = np.ones(self.num_routed_experts, dtype=np.float64) / float(self.num_routed_experts)
```

### 3.2 Smoothed Spatial Conditional Score
Calculated in `compute_spatial_score(target_stage_id, source_experts)` ([lines 136-167](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L136-L167)):
For a set of active source experts $U = \{u_1, u_2, \dots, u_{|U|}\}$ at stage $L-1$:
$$S_{\text{spatial}}(v) = \frac{1}{|U|} \sum_{u \in U} \frac{\text{self.spatial\_counts}[L-1, u, v] + \lambda \cdot P_{\text{prior}}(v \mid L)}{\sum_{k=0}^{E-1} \text{self.spatial\_counts}[L-1, u, k] + \lambda}$$

*Cold-start fallback*: If $L = 0$ (first transformer layer) or $U = \emptyset$, `compute_spatial_score` returns $P_{\text{prior}}(v \mid L)$ directly.

### 3.3 Smoothed Temporal Transition Score
Calculated in `compute_temporal_score(request_id, target_stage_id)` ([lines 169-202](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L169-L202)):
For actual previous-token experts $W = \text{self.history\_table}[request\_id][stage]$ from token $t-1$:
$$S_{\text{temporal}}(v) = \frac{1}{|W|} \sum_{w \in W} \frac{\text{self.temporal\_counts}[\text{stage}, w, v] + \lambda \cdot P_{\text{prior}}(v \mid \text{stage})}{\sum_{k=0}^{E-1} \text{self.temporal\_counts}[\text{stage}, w, k] + \lambda}$$

*Cold-start fallback*: If $W = \emptyset$ (decoding token $t=0$), `compute_temporal_score` returns $P_{\text{prior}}(v \mid \text{stage})$ directly.

### 3.4 Composite Score & Ranking
Calculated in `predict(request_id, target_stage_id, current_source_experts, budget)` ([lines 204-225](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py#L204-L225)):
$$\text{CompositeScore}(v) = \alpha \cdot S_{\text{spatial}}(v) + (1 - \alpha) \cdot S_{\text{temporal}}(v)$$

*Deterministic Ranking*:
Candidates are sorted by $(-\text{CompositeScore}[v], v)$ and truncated to budget $B$:
```python
composite_score = (self.alpha * s_spatial) + ((1.0 - self.alpha) * s_temporal)
expert_indices = list(range(self.num_routed_experts))
sorted_candidates = sorted(
    expert_indices,
    key=lambda idx: (-composite_score[idx], idx)
)
return sorted_candidates[:budget]
```

---

## 4. Complete CSTP Lifecycle (Steps 1 through 10)

1. **Predictor Initialization (`__init__`)**: Allocates dense float64 NumPy matrices for marginal counts, spatial transitions, and temporal transitions. Sets default smoothing parameters $\alpha=0.5, \lambda=1.0, \epsilon=0.1$.
2. **Offline Profiling (`fit_profile`)**: Ingests training records, counts co-activations across stages ($L \to L+1$) and consecutive tokens ($t-1 \to t$), and populates marginal stage frequencies.
3. **Request Sequence Reset (`reset_request_state`)**: Called when a new request begins. Deletes `self.history_table[request_id]`, clearing previous temporal history.
4. **First Prediction at Token 0 (Cold-Start)**: Because `self.history_table[request_id]` is empty, `compute_temporal_score` falls back to $P_{\text{prior}}(v \mid \text{stage})$.
5. **Spatial Score Computation**: Gathers active source experts $U$ at layer $L-1$, retrieves transition slices `self.spatial_counts[L-1, u, :]`, blends with prior via Dirichlet smoothing, and averages across $u \in U$.
6. **Temporal Score Computation**: Gathers previous token actual experts $W$ from `self.history_table[request_id][stage]`, retrieves transition slices `self.temporal_counts[stage, w, :]`, blends with prior, and averages across $w \in W$.
7. **Score Combination**: Linearly interpolates spatial and temporal scores using weight $\alpha$.
8. **Candidate Ranking & Top-B Selection**: Ranks candidate expert IDs $0 \dots E-1$ by composite score descending (ties broken by ID ascending) and returns the top $B$ candidates.
9. **Observation of Actual Ground-Truth Routing**: The hardware gating network reveals actual selected experts for target stage $L$. Prediction metrics are calculated and logged.
10. **State Update (`update`)**: `self.history_table[request_id][stage]` is overwritten with the revealed actual experts. If `online_adaptation=True`, transition counts and marginal frequencies are incremented.

---

## 5. Wrapper Architecture (`CSTPWrapper`)

Located in [`src/predictor/base_predictor.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/base_predictor.py#L140-L190):
- **Encapsulation**: Stores reference to the core predictor instance: `self.inner = cstp_predictor`.
- **Zero Mathematical Modification**: Delegates `predict()` directly to `self.inner.predict()`.
- **Score Transparency**: Implements `get_candidate_scores()` to expose full probability distributions without modifying core ranking logic.
- **Budget Guard**: Enforces `if budget <= 0: return []` before calling internal slice operations.

