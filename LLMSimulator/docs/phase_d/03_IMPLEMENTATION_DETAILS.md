# Phase D: ACSH Implementation Details & Architecture

**Author:** Antigravity Senior ML Systems Researcher & Algorithm Engineer  
**Date:** November 2026  
**Document ID:** `docs/phase_d/03_IMPLEMENTATION_DETAILS.md`  
**Source Module:** `src/predictor/acsh_predictor.py`  

---

## 1. Class Architecture and Interface Compliance

The `ACSHPredictor` class directly inherits from `BasePredictor` ([`src/predictor/base_predictor.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/base_predictor.py)), ensuring drop-in compatibility with the evaluation harness and dataset adapters.

### Constructor Signature
```python
class ACSHPredictor(BasePredictor):
    def __init__(
        self,
        cstp_predictor: Any,
        st_moe_predictor: Any,
        variant: str = "stage_aware",
        beta: float = 0.8,
        tau: float = 0.2,
        temp_c: float = 1.0,
        temp_m: float = 1.0,
        normalization: str = "softmax",
        reliability_objective: str = "recall",
        initial_reliability: float = 0.5,
        shift_window: int = 3,
        shift_threshold: float = 0.75,
        fast_beta: float = 0.2,
        fast_cooldown_steps: int = 3,
        name: str = "acsh"
    ):
```

---

## 2. Internal State Representation & Isolation

To prevent state contamination across independent requests, budgets, and stages:
1. **Global Reliability State**:
   - `rel_c_global: Dict[Any, float]`: Maps `request_id -> float`.
   - `rel_m_global: Dict[Any, float]`: Maps `request_id -> float`.
2. **Stage-Aware Reliability State**:
   - `rel_c_stage: Dict[Any, Dict[Tuple[int, int], float]]`: Maps `request_id -> (target_stage_id, horizon) -> float`.
   - `rel_m_stage: Dict[Any, Dict[Tuple[int, int], float]]`: Maps `request_id -> (target_stage_id, horizon) -> float`.
3. **Pending Predictions Buffer**:
   - `pending_predictions: Dict[Tuple[Any, int, Optional[int]], Dict[str, Any]]`:
     Maps `(request_id, target_stage_id, token_id)` to the exact component predictions and weights generated *before* target routing is revealed.
   - When `update()` is called, the recorded predictions are retrieved, evaluated against actual ground-truth targets, and immediately removed from the pending buffer.
4. **Fast Recovery Detection**:
   - `recent_errors: Dict[Any, Dict[Tuple[int, int], List[float]]]`: Stores rolling prediction errors over window $W=3$.
   - `fast_cooldown: Dict[Any, Dict[Tuple[int, int], int]]`: Tracks remaining fast-adaptation steps.
   - `shift_detections: List[Dict[str, Any]]`: Logs shift detection events with timestamps, stages, and error values.

---

## 3. Causal Workflow & Step-by-Step Execution

```
[At Prediction Event t]
1. CSTP extracts composite score S_C(v, t) from Dirichlet spatial & temporal tables.
2. ST-MoE extracts integer confidence S_M(v, t) from 2-bit CCT counters & History Table.
3. Component Top-B candidate sets are computed for independent reliability tracking.
4. Softmax normalization applies temperature scaling:
   P_C = softmax((S_C - max S_C) / T_C)
   P_M = softmax((S_M - max S_M) / T_M)
5. Current reliability estimates (R_C, R_M) are retrieved.
6. Softmax reliability weighting computes:
   w_C = exp(R_C / tau) / (exp(R_C / tau) + exp(R_M / tau)), w_M = 1 - w_C
7. Ensemble score is computed: P_ACSH = w_C * P_C + w_M * P_M
8. Final Top-B prefetch set is returned via deterministic tie-breaking.
9. Prediction details are buffered in pending_predictions.

[Post-Routing Gating Reveal]
10. Actual target routing A_t is revealed.
11. Component predictors (CSTP and ST-MoE) update internal tables.
12. ACSH retrieves pending prediction record and computes instantaneous utilities:
    r_C(t) = |P_C ∩ A_t| / K,  r_M(t) = |P_M ∩ A_t| / K
13. Fast recovery checks recent rolling error: if error > threshold, activate fast beta.
14. Exponential moving average updates reliability:
    R_C(t+1) = beta * R_C(t) + (1 - beta) * r_C(t)
    R_M(t+1) = beta * R_M(t) + (1 - beta) * r_M(t)
15. Diagnostic entry logged to weight_history.
```

---

## 4. Zero Ternary Enforcement & Test Verification

All branching logic in `acsh_predictor.py` utilizes explicit multi-line `if/else` control flow. AST verification confirms exactly 0 ternary operator instances (`ast.IfExp`). Full unit test suite (`tests/phase_d/test_acsh_validation.py`) verifies all 8 algorithmic properties with 100% pass rate.

