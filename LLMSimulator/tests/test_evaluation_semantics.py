"""
Phase 2B: Regression Test Suite for Evaluation Semantics and Integrity.

Verifies the 10 required invariants:
1. Within-request adaptation does not carry into the next request.
2. Across-request adaptation does carry into the next request.
3. Both modes maintain predict-before-update ordering.
4. History resets at request boundaries.
5. CCT profiling state is restored correctly in within-request mode.
6. Invalid source stages remain None (for H=1 and H=2).
7. Invalid evaluation modes raise exceptions.
8. Internal TypeErrors are not silently swallowed.
9. Request order does not affect within-request results when requests are evaluated independently.
10. Existing macro/micro metric formulas remain unchanged.
"""

import os
import sys
import copy
import unittest
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predictor.evaluation import (
    EvaluationHarness,
    EvaluationModeError,
    PredictionOpportunity,
    PredictionResultRecord
)
from src.predictor.st_moe_paper import STMoEPaperPredictor


class TrackingSpyPredictor:
    """Spy predictor that tracks method call order and mutations."""
    calls: List[Tuple[str, int, int]] = []
    reset_calls: List[int] = []

    def __init__(self, num_stages=4, num_experts=8, top_k=2):
        self.num_moe_stages = num_stages
        self.num_routed_experts = num_experts
        self.top_k = top_k
        self.cct: Dict[int, int] = {0: 100} # Mock state
        TrackingSpyPredictor.calls = []
        TrackingSpyPredictor.reset_calls = []

    def reset(self, request_id: int):
        TrackingSpyPredictor.reset_calls.append(request_id)

    def predict(self, request_id: int, target_stage_id: int, current_source_experts: List[int], budget: int, token_id: Optional[int] = None, source_stage_id: Optional[int] = None):
        TrackingSpyPredictor.calls.append(("predict", request_id, target_stage_id))
        return [0]

    def update(self, request_id: int, target_stage_id: int, source_experts: List[int], actual_target_experts: List[int], token_id: Optional[int] = None, source_stage_id: Optional[int] = None):
        TrackingSpyPredictor.calls.append(("update", request_id, target_stage_id))
        self.cct[0] += 1  # State mutation


class BuggyPredictor:
    """Predictor that intentionally raises a TypeError internally."""
    def predict_threshold(self, request_id, target_stage_id, current_source_experts, source_stage_id=None):
        raise TypeError("Internal logic error in predictor math")

    def update(self, *args, **kwargs):
        pass


def make_test_opportunities(num_requests=2, tokens_per_req=3, num_stages=3, horizon=1):
    opps = []
    for req in range(num_requests):
        for tok in range(tokens_per_req):
            for stage in range(num_stages):
                src_stage = stage - horizon if stage >= horizon else None
                opp = PredictionOpportunity(
                    request_id=req,
                    token_id=tok,
                    moe_stage_id=stage,
                    source_experts=[0] if src_stage is not None else [],
                    actual_experts=[1],
                    source_stage_id=src_stage,
                    prediction_horizon=horizon
                )
                opps.append(opp)
    return opps


def test_1_within_request_adaptation_does_not_carry():
    """1. Within-request adaptation does not carry into the next request."""
    harness = EvaluationHarness(num_moe_stages=3)
    initial_cct_val = 100

    class StateObserverPredictor(TrackingSpyPredictor):
        observed_cct_at_predict = []

        def __init__(self):
            super().__init__()
            StateObserverPredictor.observed_cct_at_predict = []

        def predict(self, request_id, target_stage_id, current_source_experts, budget, token_id=None, source_stage_id=None):
            super().predict(request_id, target_stage_id, current_source_experts, budget, token_id, source_stage_id)
            StateObserverPredictor.observed_cct_at_predict.append((request_id, target_stage_id, self.cct[0]))
            return [0]

    obs = StateObserverPredictor()
    opps = make_test_opportunities(num_requests=2, tokens_per_req=2, num_stages=2)
    harness.evaluate_predictor(obs, "obs", opps, budgets=[1], evaluation_mode="online_within_request")

    req0_initial = [val for req, stg, val in StateObserverPredictor.observed_cct_at_predict if req == 0][0]
    req1_initial = [val for req, stg, val in StateObserverPredictor.observed_cct_at_predict if req == 1][0]

    assert req0_initial == initial_cct_val, f"Req 0 should see initial state {initial_cct_val}, got {req0_initial}"
    assert req1_initial == initial_cct_val, f"Req 1 should see initial state {initial_cct_val}, got {req1_initial}"


def test_2_across_request_adaptation_carries():
    """2. Across-request adaptation does carry into the next request."""
    harness = EvaluationHarness(num_moe_stages=3)
    initial_cct_val = 100

    class StateObserverPredictor2(TrackingSpyPredictor):
        observed_cct_at_predict = []

        def __init__(self):
            super().__init__()
            StateObserverPredictor2.observed_cct_at_predict = []

        def predict(self, request_id, target_stage_id, current_source_experts, budget, token_id=None, source_stage_id=None):
            super().predict(request_id, target_stage_id, current_source_experts, budget, token_id, source_stage_id)
            StateObserverPredictor2.observed_cct_at_predict.append((request_id, target_stage_id, self.cct[0]))
            return [0]

    obs = StateObserverPredictor2()
    opps = make_test_opportunities(num_requests=2, tokens_per_req=2, num_stages=2)
    harness.evaluate_predictor(obs, "obs", opps, budgets=[1], evaluation_mode="online_across_requests")

    req0_initial = [val for req, stg, val in StateObserverPredictor2.observed_cct_at_predict if req == 0][0]
    req1_initial = [val for req, stg, val in StateObserverPredictor2.observed_cct_at_predict if req == 1][0]

    assert req0_initial == initial_cct_val
    assert req1_initial > initial_cct_val, f"Req 1 should see mutated state > {initial_cct_val}, got {req1_initial}"


def test_3_predict_before_update_ordering():
    """3. Both modes maintain predict-before-update ordering."""
    harness = EvaluationHarness(num_moe_stages=3)
    for mode in ["online_within_request", "online_across_requests"]:
        spy = TrackingSpyPredictor()
        opps = make_test_opportunities(num_requests=1, tokens_per_req=2, num_stages=2)
        harness.evaluate_predictor(spy, "spy", opps, budgets=[1], evaluation_mode=mode)

        call_types = [c[0] for c in TrackingSpyPredictor.calls]
        assert call_types[0] == "predict"
        assert call_types[1] == "update"
        assert call_types[2] == "predict"
        assert call_types[3] == "update"


def test_4_history_resets_at_request_boundaries():
    """4. History resets at request boundaries."""
    harness = EvaluationHarness(num_moe_stages=3)
    spy = TrackingSpyPredictor()
    opps = make_test_opportunities(num_requests=2, tokens_per_req=2, num_stages=2)
    harness.evaluate_predictor(spy, "spy", opps, budgets=[1], evaluation_mode="online_within_request")

    assert 0 in TrackingSpyPredictor.reset_calls, "Request 0 must trigger reset"
    assert 1 in TrackingSpyPredictor.reset_calls, "Request 1 must trigger reset"


def test_5_cct_profiling_state_restored_in_within_request():
    """5. CCT profiling state is restored correctly in within-request mode."""
    pred = STMoEPaperPredictor(num_moe_stages=3, num_routed_experts=4, top_k=1, online_adaptation=True)
    pred.cct[0][0][1] = 2
    initial_cct = copy.deepcopy(pred.cct)

    harness = EvaluationHarness(num_moe_stages=3)
    opps = make_test_opportunities(num_requests=2, tokens_per_req=2, num_stages=3)
    harness.evaluate_predictor(pred, "st_moe", opps, budgets=[1], evaluation_mode="online_within_request")

    assert pred.cct == initial_cct, "Parent predictor must remain untouched and restored"


def test_6_invalid_source_stages_remain_none():
    """6. Invalid source stages remain None (for H=1 and H=2)."""
    # H = 1: Stage 0 has no source
    opp_h1_s0 = PredictionOpportunity(0, 0, 0, [], [1], source_stage_id=None, prediction_horizon=1)
    assert opp_h1_s0.source_stage_id is None, "H=1 Stage 0 must have source_stage_id=None, not 0!"

    opp_h1_s1 = PredictionOpportunity(0, 0, 1, [0], [1], source_stage_id=0, prediction_horizon=1)
    assert opp_h1_s1.source_stage_id == 0

    # H = 2: Stage 0 and Stage 1 have no source
    opp_h2_s0 = PredictionOpportunity(0, 0, 0, [], [1], source_stage_id=None, prediction_horizon=2)
    assert opp_h2_s0.source_stage_id is None, "H=2 Stage 0 must have source_stage_id=None!"

    opp_h2_s1 = PredictionOpportunity(0, 0, 1, [], [1], source_stage_id=None, prediction_horizon=2)
    assert opp_h2_s1.source_stage_id is None, "H=2 Stage 1 must have source_stage_id=None!"

    opp_h2_s2 = PredictionOpportunity(0, 0, 2, [0], [1], source_stage_id=0, prediction_horizon=2)
    assert opp_h2_s2.source_stage_id == 0

    # Test auto-computation when source_stage_id is omitted
    opp_auto_0 = PredictionOpportunity(0, 0, 0, [], [1], prediction_horizon=2)
    assert opp_auto_0.source_stage_id is None

    opp_auto_1 = PredictionOpportunity(0, 0, 1, [], [1], prediction_horizon=2)
    assert opp_auto_1.source_stage_id is None

    opp_auto_2 = PredictionOpportunity(0, 0, 2, [], [1], prediction_horizon=2)
    assert opp_auto_2.source_stage_id == 0


def test_7_invalid_evaluation_modes_raise_exceptions():
    """7. Invalid evaluation modes raise EvaluationModeError."""
    harness = EvaluationHarness(num_moe_stages=3)
    spy = TrackingSpyPredictor()
    opps = make_test_opportunities(num_requests=1, tokens_per_req=1, num_stages=2)

    invalid_modes = ["unknown_mode", "arbitrary_string", "offline_mode", "magic"]
    for bad_mode in invalid_modes:
        try:
            harness.evaluate_predictor(spy, "spy", opps, budgets=[1], evaluation_mode=bad_mode)
            assert False, f"Should have raised EvaluationModeError for {bad_mode}"
        except EvaluationModeError:
            pass  # Expected


def test_8_internal_type_errors_not_swallowed():
    """8. Internal TypeErrors in predictor methods are NOT swallowed."""
    harness = EvaluationHarness(num_moe_stages=3)
    buggy = BuggyPredictor()
    opps = make_test_opportunities(num_requests=1, tokens_per_req=1, num_stages=2)

    try:
        harness.evaluate_predictor(buggy, "buggy", opps, evaluation_mode="threshold")
        assert False, "Should have propagated internal TypeError!"
    except TypeError as e:
        assert "Internal logic error" in str(e), f"Expected original error message, got {e}"


def test_9_request_order_independence_in_within_request():
    """9. Request order does not affect within-request results."""
    harness = EvaluationHarness(num_moe_stages=3)

    pred1 = STMoEPaperPredictor(num_moe_stages=3, num_routed_experts=4, top_k=1, online_adaptation=True)
    pred2 = STMoEPaperPredictor(num_moe_stages=3, num_routed_experts=4, top_k=1, online_adaptation=True)
    pred1.cct[0][0][1] = 2
    pred2.cct[0][0][1] = 2

    opps_req0 = [op for op in make_test_opportunities(num_requests=1, tokens_per_req=3, num_stages=3) if op.request_id == 0]
    # Create request 1 with slightly different targets
    opps_req1 = []
    for op in opps_req0:
        opp_copy = copy.deepcopy(op)
        opp_copy.request_id = 1
        opp_copy.actual_experts = [2]
        opps_req1.append(opp_copy)

    # Order A: [0, 1]
    res_A = harness.evaluate_predictor(pred1, "st_moe", opps_req0 + opps_req1, budgets=[1], evaluation_mode="online_within_request")
    # Order B: [1, 0]
    res_B = harness.evaluate_predictor(pred2, "st_moe", opps_req1 + opps_req0, budgets=[1], evaluation_mode="online_within_request")

    res_A_by_req = {0: [r for r in res_A if r.request_id == 0], 1: [r for r in res_A if r.request_id == 1]}
    res_B_by_req = {0: [r for r in res_B if r.request_id == 0], 1: [r for r in res_B if r.request_id == 1]}

    # For request 0, predictions in Order A and Order B must be strictly identical
    preds_A_0 = [r.predicted_experts for r in res_A_by_req[0]]
    preds_B_0 = [r.predicted_experts for r in res_B_by_req[0]]
    assert preds_A_0 == preds_B_0, "Request 0 predictions must be independent of whether request 1 was evaluated first!"

    preds_A_1 = [r.predicted_experts for r in res_A_by_req[1]]
    preds_B_1 = [r.predicted_experts for r in res_B_by_req[1]]
    assert preds_A_1 == preds_B_1, "Request 1 predictions must be independent of whether request 0 was evaluated first!"


def test_10_metric_formulas_unchanged():
    """10. Existing macro/micro metric formulas remain unchanged."""
    harness = EvaluationHarness(num_moe_stages=2)

    # Opportunity 1: target [0, 1], predicted [0, 2] -> hits=1, prec=0.5, rec=0.5
    # Opportunity 2: target [0, 1], predicted [0, 1] -> hits=2, prec=1.0, rec=1.0
    r1 = PredictionResultRecord(0, 0, 0, 0, 2, "test", [0, 2], [0, 1])
    r2 = PredictionResultRecord(0, 1, 1, 0, 2, "test", [0, 1], [0, 1])

    summary = harness.compute_summary_metrics([r1, r2])

    assert summary["total_instances"] == 2
    # Macro metrics
    assert np.isclose(summary["macro_precision"], (0.5 + 1.0) / 2)
    assert np.isclose(summary["macro_recall"], (0.5 + 1.0) / 2)
    assert np.isclose(summary["macro_f1"], (0.5 + 1.0) / 2)
    assert np.isclose(summary["hit_rate"], 1.0)
    assert np.isclose(summary["exact_set_accuracy"], 0.5)

    # Micro metrics: total_hits=3, total_pred=4, total_act=4
    assert np.isclose(summary["micro_precision"], 3.0 / 4.0)
    assert np.isclose(summary["micro_recall"], 3.0 / 4.0)
    assert np.isclose(summary["micro_f1"], 3.0 / 4.0)


if __name__ == "__main__":
    print("Running Phase 2B Evaluation Semantics Regression Suite...")
    test_1_within_request_adaptation_does_not_carry()
    print("  [PASS] Test 1: Within-request adaptation does not carry into next request.")
    test_2_across_request_adaptation_carries()
    print("  [PASS] Test 2: Across-request adaptation does carry into next request.")
    test_3_predict_before_update_ordering()
    print("  [PASS] Test 3: Predict-before-update ordering maintained.")
    test_4_history_resets_at_request_boundaries()
    print("  [PASS] Test 4: History resets at request boundaries.")
    test_5_cct_profiling_state_restored_in_within_request()
    print("  [PASS] Test 5: CCT profiling state restored correctly in within-request mode.")
    test_6_invalid_source_stages_remain_none()
    print("  [PASS] Test 6: Invalid source stages remain None (H=1 and H=2).")
    test_7_invalid_evaluation_modes_raise_exceptions()
    print("  [PASS] Test 7: Invalid evaluation modes raise exceptions.")
    test_8_internal_type_errors_not_swallowed()
    print("  [PASS] Test 8: Internal TypeErrors are not silently swallowed.")
    test_9_request_order_independence_in_within_request()
    print("  [PASS] Test 9: Request order does not affect within-request results.")
    test_10_metric_formulas_unchanged()
    print("  [PASS] Test 10: Existing macro/micro metric formulas remain unchanged.")
    print("ALL 10 PHASE 2B REGRESSION TESTS PASSED SUCCESSFULLY!")
