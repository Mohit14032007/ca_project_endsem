# CSTP Profiling Percentage Sensitivity Audit

**Date:** November 9, 2026  
**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Phase 1A — Profiling Percentage Sensitivity Experiment  
**Target Architecture & Implementation:** `src/predictor/cstp_predictor.py`, `src/prefetch/cstp_predictor.h`, `src/prefetch/cstp_predictor.cpp`

---

## 1. Executive Summary & Purpose

This audit establishes the technical foundation for **Phase 1A: CSTP Profiling Percentage Sensitivity Experiment**. The central research question of this phase is:
> *How does the quantity of offline profiling data ($25\%, 50\%, 75\%, 100\%$ of training prompts/tasks) govern the frozen prediction accuracy and online adaptation efficacy of the Correlation-based Spatio-Temporal Predictor (CSTP)?*

Before running sensitivity experiments, this audit verifies the exact APIs, internal data structures, profiling loops, transition boundaries, and architectural discrepancies between the Python research harness and the C++ simulator engine.

---

## 2. API Signatures and Data Structures

### 2.1 Python Reference Implementation (`src/predictor/cstp_predictor.py`)

The primary research class is `CSTPPredictor`. Its core interface and internal structures are:

```python
class CSTPPredictor(BasePredictor):
    def __init__(
        self,
        num_moe_stages: int,
        num_routed_experts: int,
        top_k: int = 8,
        alpha: float = 0.5,
        smoothing_lambda: float = 1.0,
        prior_epsilon: float = 0.1,
        online_adaptation: bool = True,
        prediction_horizon: int = 1
    )
```

#### Core Data Structures:
1. **Marginal Counts (`self.stage_marginal_counts`):**
   - Type: `np.ndarray` of shape `(num_moe_stages, num_routed_experts)`, `dtype=np.float64`.
   - Content: Frequency counts of each expert activated at stage $s$.
2. **Spatial Counts (`self.spatial_counts`):**
   - Type: `np.ndarray` of shape `(num_moe_stages - prediction_horizon, num_routed_experts, num_routed_experts)`, `dtype=np.float64`.
   - Content: Co-activation frequency of source expert $u$ at stage $s$ and target expert $v$ at stage $s + H$.
3. **Temporal Counts (`self.temporal_counts`):**
   - Type: `np.ndarray` of shape `(num_moe_stages, num_routed_experts, num_routed_experts)`, `dtype=np.float64`.
   - Content: Transition frequency of expert $w$ at token $t-1$ to expert $v$ at token $t$ within the same stage $s$.
4. **History Table (`self.history_table`):**
   - Type: `Dict[int, Dict[int, Set[int]]]`: `request_id -> stage_id -> Set[actual_routed_experts]`.
   - Content: Caches the previous token's actual routed experts for the active request.

#### Python Method Signatures & Mechanics:
- `fit_profile(training_records: List[Any]) -> None`:
  Aggregates offline trace records into `stage_marginal_counts`, `spatial_counts`, and `temporal_counts`.
- `get_marginal_prior(stage_id: int) -> np.ndarray`:
  Computes Dirichlet-smoothed target-stage prior vector $P_{\text{prior}}(v \mid s) = \frac{\text{Count}(s, v) + \epsilon}{\sum_{v'} \text{Count}(s, v') + E \cdot \epsilon}$.
- `compute_spatial_score(target_stage_id: int, source_experts: List[int], source_stage_id: Optional[int] = None) -> np.ndarray`:
  Calculates Bayesian-smoothed spatial conditional transition score averaged over all active source experts:
  $$S_{\text{spatial}}(v) = \frac{1}{|U|} \sum_{u \in U} \frac{\text{Spatial}(s_{\text{src}}, u, v) + \lambda \cdot P_{\text{prior}}(v)}{\sum_{v'} \text{Spatial}(s_{\text{src}}, u, v') + \lambda}$$
- `compute_temporal_score(request_id: int, target_stage_id: int) -> np.ndarray`:
  Extracts previously active experts $W$ from `history_table[request_id][target_stage_id]`. If cold-start ($|W| = 0$), falls back cleanly to $P_{\text{prior}}$. Otherwise computes:
  $$S_{\text{temporal}}(v) = \frac{1}{|W|} \sum_{w \in W} \frac{\text{Temporal}(s, w, v) + \lambda \cdot P_{\text{prior}}(v)}{\sum_{v'} \text{Temporal}(s, w, v') + \lambda}$$
- `predict(request_id: int, target_stage_id: int, current_source_experts: List[int], budget: int, token_id: Optional[int] = None, source_stage_id: Optional[int] = None) -> List[int]`:
  Forms composite score $S_{\text{comp}}(v) = \alpha \cdot S_{\text{spatial}}(v) + (1 - \alpha) \cdot S_{\text{temporal}}(v)$, breaks ties deterministically by lowest expert ID ascending, and selects the Top-$B$ candidates.
- `update(request_id: int, target_stage_id: int, source_experts: List[int], actual_target_experts: List[int], token_id: Optional[int] = None, source_stage_id: Optional[int] = None) -> None`:
  If `self.online_adaptation` is True, increments `stage_marginal_counts`, `spatial_counts`, and `temporal_counts` by $+1.0$. Regardless of `online_adaptation`, updates `history_table[request_id][target_stage_id] = set(actual_target_experts)`.
- `reset_request_state(request_id: int) -> None`:
  Purges `history_table[request_id]` at request boundaries.

---

### 2.2 C++ Simulator Implementation (`src/prefetch/cstp_predictor.h`, `cstp_predictor.cpp`)

The C++ simulator engine provides native prefetching prediction during hardware cycle simulation:

```cpp
class CSTPPredictor : public IPredictor {
public:
    CSTPPredictor(
        int num_moe_stages,
        int num_routed_experts,
        int top_k = 8,
        double alpha = 0.5,
        double smoothing_lambda = 1.0,
        double prior_epsilon = 0.1,
        bool online_adaptation = true,
        int prediction_horizon = 1);

    void reset(int request_id) override;
    void update(...) override;
    PredictionResult predict(...) override;

    // Manual count manipulation helpers:
    void set_marginal_count(int stage_id, int expert_id, double count);
    void set_spatial_count(int source_stage_id, int source_expert, int target_expert, double count);
    void set_temporal_count(int stage_id, int prev_expert, int curr_expert, double count);
    void set_history_entry(int request_id, int stage_id, const std::vector<int>& experts);
};
```

---

## 3. Critical Python versus C++ Discrepancy

| Capability | Python (`cstp_predictor.py`) | C++ (`cstp_predictor.cpp`) | Impact & Resolution |
|---|---|---|---|
| **Profiling Engine** | Full native `fit_profile(training_records)` | **None** (Exposes only point count setters `set_*_count`) | C++ simulator cannot parse raw routing traces directly; offline profiling must be executed via Python, and count tensors transferred or configured before simulation runs. |
| **Data Types** | NumPy 64-bit float arrays | Nested `std::vector<double>` | Identical mathematical precision. |
| **Scoring Formulae** | Vectorized Dirichlet smoothing | Loop-based Dirichlet smoothing | Identical mathematical outputs confirmed in unit tests. |
| **History Lifecycle** | `reset_request_state(req)` | `reset(req)` | Handled symmetrically by `EvaluationHarness`. |

---

## 4. Audit of Profiling and Sequence Mechanics

### 4.1 How Profiling Populates Count Structures
Inside `CSTPPredictor.fit_profile(training_records)`:
1. Records are grouped into a lookup map: `grouped[(req, tok, stage)] = [expert_1, ..., expert_k]`.
2. Requests are sorted deterministically: `requests = sorted(list(set(k[0] for k in grouped.keys())))`.
3. For each request, tokens are sorted deterministically: `req_tokens = sorted(list(set(k[1] for k in req_keys)))`.
4. **Marginal:** For every stage and expert $v \in \text{curr\_experts}$, `stage_marginal_counts[stage, v] += 1.0`.
5. **Spatial:** For stage $< S - H$, next key is $(req, tok, stage + H)$. For all $u \in \text{curr\_experts}, v \in \text{next\_experts}$, `spatial_counts[stage, u, v] += 1.0`.
6. **Temporal:** For `tok_idx > 0`, previous token is $t-1 = \text{req\_tokens}[\text{tok\_idx} - 1]$. For all $w \in \text{prev\_experts}, v \in \text{curr\_experts}$, `temporal_counts[stage, w, v] += 1.0`.

### 4.2 Sequence Invariants & Verification Checklist
- **Request ID Representation:**
  - `moe_anywhere`: Integer prompt index ($0 \dots 73$).
  - `moe_suite`: Synthetic request index assigned sequentially ($9000 + \text{task\_idx}$).
- **Prompt / Task Isolation:**
  - Training and testing partitions are strictly disjoint across prompt/task identifiers.
  - In `fit_profile`, grouping by `(req, tok, stage)` ensures that transitions **never cross request or prompt boundaries**.
- **Token ID Representation & Consecutive Continuity:**
  - In `moe_anywhere`, tokens per prompt are contiguous integers ($[20 \dots 276]$), confirmed with zero missing sequence gaps.
  - In `moe_suite`, tokens per task are contiguous indices ($0 \dots 8190$), verified with zero gaps.
  - Because tokens are strictly contiguous, `req_tokens[tok_idx - 1]` never bridges across missing sequence gaps.
- **Spatial Stage Distance $H$:**
  - Strictly parameterized as $H \in \{1, 2\}$.
  - Target stage is strictly $s + H$, and source stage is strictly $s$.
  - Opportunities where $s < H$ (stages $0 \dots H-1$) lack an MoE predecessor and fall back to empty source experts, yielding marginal prior.

---

## 5. Train / Test Isolation and Test Opportunities

### 5.1 Dataset Specifications

| Metric / Dimension | MoE Anywhere | MoE Suite |
|---|---|---|
| **Model** | Sage-A1.5 | GLM-5.2 |
| **MoE Stages ($S$)** | 40 | 75 |
| **Routed Experts ($E$)** | 256 | 256 |
| **Top-$K$ Routing** | 8 | 8 |
| **Total Prompts / Tasks** | 74 prompts | 12 tasks |
| **Train Split** | 52 prompts | 8 tasks |
| **Validation Split** | 8 prompts | 2 tasks |
| **Test Split** | 14 prompts | 2 tasks |
| **Test Evaluation Tokens** | 30 tokens / prompt | 50 tokens / task |
| **Total Test Opportunities** | $14 \times 30 \times 40 = \mathbf{16,800}$ | $2 \times 50 \times 75 = \mathbf{7,500}$ |

### 5.2 Verification of Test Opportunity Invariance
Every experiment across all profiling percentages ($25\%, 50\%, 75\%, 100\%$), horizons ($H=1, 2$), seeds, and evaluation regimes (`frozen`, `online_within_request`) is evaluated against the **exact identical set of test opportunities**:
- **MoE Anywhere:** 16,800 opportunities.
- **MoE Suite:** 7,500 opportunities.

---

## 6. Evaluation Protocol Semantics: Frozen vs Online

### 6.1 `frozen` Protocol
- `predictor.online_adaptation = False`.
- Evaluated under `online_within_request` sequence isolation.
- At runtime, `predictor.update()` is called post-scoring:
  - `self.stage_marginal_counts`, `spatial_counts`, and `temporal_counts` are **completely frozen** (bypassed).
  - `self.history_table[request_id][stage_id]` **is updated** with actual selections.
- Result: Token $0$ evaluates marginal+spatial cold-start; Tokens $1 \dots T-1$ leverage temporal transitions from the profiled matrix without mutating transition counts.

### 6.2 `online_within_request` Protocol
- `predictor.online_adaptation = True`.
- Evaluated under `online_within_request` sequence isolation.
- Each test prompt/task starts from a clean, isolated copy of the offline-profiled counts.
- At runtime, `predictor.update()` is called post-scoring:
  - Marginal, spatial, and temporal count structures receive $+1.0$ online increments.
  - History Table is updated with actual routed experts.
- At request boundaries, the predictor is restored to the clean profiling-derived state.

---

## 7. Audit Conclusion & Readiness Certification

The mathematical equations, API signatures, dataset adapters, and evaluation harnesses are verified. We are certified to proceed directly to the implementation and execution of `scripts/phase1a/run_cstp_profiling_sensitivity.py`.

