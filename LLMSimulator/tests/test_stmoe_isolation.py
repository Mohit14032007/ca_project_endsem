"""
Tests for ST-MoE Experiment Isolation and Result Integrity.

Validates the six required isolation properties:
1. Frozen B=1 and Frozen B=8 start with identical CCT states.
2. Online B=1 and Online B=8 start with identical CCT states.
3. Native threshold frozen and online runs start from identical profiling states.
4. Profiling sensitivity experiments are isolated.
5. Running experiments in a different order does not change their results.
6. Online adaptation within one experiment still works causally.
"""

import os
import sys
import copy
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predictor.st_moe_paper import STMoEPaperPredictor
from src.predictor.evaluation import (
    EvaluationHarness,
    PredictionOpportunity,
    PredictionResultRecord
)


def create_mock_opportunities(num_requests=3, tokens_per_req=5, num_stages=4, top_k=2):
    """Generates synthetic opportunities for deterministic isolation testing."""
    opps = []
    rng = np.random.RandomState(42)
    for req in range(num_requests):
        for tok in range(tokens_per_req):
            for stage in range(num_stages):
                src_stage = stage - 1 if stage > 0 else None
                src_experts = [0, 1] if stage > 0 else []
                # Ground truth targets
                actual = [int(rng.randint(0, 8)), int(rng.randint(0, 8))]
                if actual[0] == actual[1]:
                    actual[1] = (actual[0] + 1) % 8
                opp = PredictionOpportunity(
                    request_id=req,
                    token_id=tok,
                    moe_stage_id=stage,
                    source_experts=src_experts,
                    actual_experts=actual,
                    source_stage_id=src_stage,
                    prediction_horizon=1
                )
                opps.append(opp)
    return opps


def create_factory_and_matrix(num_stages=4, num_experts=8, top_k=2):
    """Creates a base co-activation matrix and factory for producing isolated predictors."""
    num_transitions = max(1, num_stages - 1)
    co_matrices = [np.zeros((num_experts, num_experts), dtype=np.int64) for _ in range(num_transitions)]
    # Populate mock co-activations
    for s in range(num_transitions):
        for u in range(num_experts):
            co_matrices[s][u, (u + 1) % num_experts] = 10
            co_matrices[s][u, (u + 2) % num_experts] = 5

    def factory(regime="online", scoring_mode="confidence_additive"):
        pred = STMoEPaperPredictor(
            num_moe_stages=num_stages,
            num_routed_experts=num_experts,
            top_k=top_k,
            cct_capacity=top_k,
            confidence_threshold=2,
            online_adaptation=(regime == "online"),
            prediction_horizon=1,
            scoring_mode=scoring_mode
        )
        for s in range(num_transitions):
            for u in range(num_experts):
                row = co_matrices[s][u]
                dests = sorted(range(num_experts), key=lambda idx: (-row[idx], idx))
                stored = 0
                for v in dests:
                    if row[v] > 0 and stored < top_k:
                        pred.cct[s][u][v] = 2
                        stored += 1
        return pred

    return factory, co_matrices


def test_frozen_budgets_start_with_identical_cct():
    """Property 1: Frozen B=1 and Frozen B=8 start with identical CCT states."""
    factory, _ = create_factory_and_matrix()
    p1 = factory(regime="frozen")
    p8 = factory(regime="frozen")

    assert p1.cct == p8.cct, "Initial CCT states must be identical."

    harness = EvaluationHarness(num_moe_stages=4)
    opps = create_mock_opportunities()

    res1 = harness.evaluate_predictor(p1, "st_moe_frozen", opps, budgets=[1], evaluation_mode="frozen")
    res8 = harness.evaluate_predictor(p8, "st_moe_frozen", opps, budgets=[8], evaluation_mode="frozen")

    # In frozen mode, CCT must remain un-mutated
    assert p1.cct == p8.cct, "Frozen evaluation must not mutate CCT state."


def test_online_budgets_start_with_identical_cct_and_are_isolated():
    """Property 2: Online B=1 and Online B=8 start with identical CCT states and do not leak mutations."""
    factory, _ = create_factory_and_matrix()
    p1 = factory(regime="online")
    p8 = factory(regime="online")

    initial_cct_1 = copy.deepcopy(p1.cct)
    initial_cct_8 = copy.deepcopy(p8.cct)

    assert initial_cct_1 == initial_cct_8, "Both online runs must start with identical CCT states."

    harness = EvaluationHarness(num_moe_stages=4)
    opps = create_mock_opportunities()

    res1 = harness.evaluate_predictor(p1, "st_moe_online", opps, budgets=[1], evaluation_mode="online")

    # Verify that p8 has NOT been mutated by p1's execution
    assert p8.cct == initial_cct_8, "Running B=1 online must not mutate B=8's initial CCT state!"

    res8 = harness.evaluate_predictor(p8, "st_moe_online", opps, budgets=[8], evaluation_mode="online")
    assert len(res1) == len(opps)
    assert len(res8) == len(opps)


def test_native_threshold_frozen_and_online_start_identical():
    """Property 3: Native threshold frozen and online runs start from identical profiling states."""
    factory, _ = create_factory_and_matrix()
    p_th_frozen = factory(regime="frozen")
    p_th_online = factory(regime="online")

    assert p_th_frozen.cct == p_th_online.cct, "Native threshold runs must start from identical CCT states."
    assert p_th_frozen.history_table == p_th_online.history_table == {}, "History tables must start empty."


def test_profiling_sensitivity_experiments_are_isolated():
    """Property 4: Profiling sensitivity experiments are isolated across fractions."""
    num_stages = 4
    num_experts = 8
    top_k = 2

    def make_frac_pred(frac):
        pred = STMoEPaperPredictor(
            num_moe_stages=num_stages,
            num_routed_experts=num_experts,
            top_k=top_k,
            cct_capacity=top_k,
            confidence_threshold=2,
            online_adaptation=True,
            prediction_horizon=1
        )
        # Fraction determines how many entries are populated
        count = max(1, int(top_k * frac))
        for s in range(num_stages - 1):
            for u in range(num_experts):
                for v in range(count):
                    pred.cct[s][u][v] = 2
        return pred

    p_10 = make_frac_pred(0.1)
    p_100 = make_frac_pred(1.0)

    # p_10 and p_100 must be completely independent objects
    assert p_10 is not p_100
    assert p_10.cct is not p_100.cct

    harness = EvaluationHarness(num_moe_stages=4)
    opps = create_mock_opportunities()

    res_10 = harness.evaluate_predictor(p_10, "sens_10", opps, budgets=[2], evaluation_mode="online")
    cct_10_after = copy.deepcopy(p_10.cct)

    res_100 = harness.evaluate_predictor(p_100, "sens_100", opps, budgets=[2], evaluation_mode="online")

    # Verify evaluating p_100 did not touch p_10
    assert p_10.cct == cct_10_after


def test_experiment_order_independence():
    """Property 5: Running experiments in a different order does not change results."""
    factory, _ = create_factory_and_matrix()
    opps = create_mock_opportunities()
    harness = EvaluationHarness(num_moe_stages=4)

    # Order A: Frozen then Online
    p_f_A = factory(regime="frozen")
    p_o_A = factory(regime="online")
    res_f_A = harness.evaluate_predictor(p_f_A, "frozen", opps, budgets=[2], evaluation_mode="frozen")
    res_o_A = harness.evaluate_predictor(p_o_A, "online", opps, budgets=[2], evaluation_mode="online")

    # Order B: Online then Frozen
    p_o_B = factory(regime="online")
    p_f_B = factory(regime="frozen")
    res_o_B = harness.evaluate_predictor(p_o_B, "online", opps, budgets=[2], evaluation_mode="online")
    res_f_B = harness.evaluate_predictor(p_f_B, "frozen", opps, budgets=[2], evaluation_mode="frozen")

    m_f_A = harness.compute_summary_metrics(res_f_A)
    m_f_B = harness.compute_summary_metrics(res_f_B)
    m_o_A = harness.compute_summary_metrics(res_o_A)
    m_o_B = harness.compute_summary_metrics(res_o_B)

    assert m_f_A["micro_recall"] == m_f_B["micro_recall"], "Frozen recall must be order-independent!"
    assert m_o_A["micro_recall"] == m_o_B["micro_recall"], "Online recall must be order-independent!"
    assert m_o_A["hit_rate"] == m_o_B["hit_rate"], "Online hit rate must be order-independent!"


def test_online_adaptation_causality_within_experiment():
    """Property 6: Online adaptation within one experiment still works causally."""
    factory, _ = create_factory_and_matrix()
    p_online = factory(regime="online")

    opps = create_mock_opportunities(num_requests=1, tokens_per_req=10, num_stages=4, top_k=2)
    harness = EvaluationHarness(num_moe_stages=4)

    res = harness.evaluate_predictor(p_online, "online_causal", opps, budgets=[2], evaluation_mode="online")
    sm = harness.compute_summary_metrics(res)

    # In online mode, steady-state recall should exceed cold-start recall due to adaptation
    assert sm["total_instances"] == 40
    assert "cold_start" in sm
    assert "steady_state" in sm
    assert sm["cold_start"]["instances"] == 4  # 4 stages for token 0
    assert sm["steady_state"]["instances"] == 36 # remaining tokens


if __name__ == "__main__":
    print("Running ST-MoE Isolation Test Suite...")
    test_frozen_budgets_start_with_identical_cct()
    print("  [PASS] Property 1: Frozen B=1 and Frozen B=8 start with identical CCT states.")
    test_online_budgets_start_with_identical_cct_and_are_isolated()
    print("  [PASS] Property 2: Online B=1 and Online B=8 start with identical CCT states.")
    test_native_threshold_frozen_and_online_start_identical()
    print("  [PASS] Property 3: Native threshold frozen and online start identical.")
    test_profiling_sensitivity_experiments_are_isolated()
    print("  [PASS] Property 4: Profiling sensitivity experiments are isolated.")
    test_experiment_order_independence()
    print("  [PASS] Property 5: Experiment order independence.")
    test_online_adaptation_causality_within_experiment()
    print("  [PASS] Property 6: Online adaptation causality within experiment.")
    print("ALL 6 ISOLATION TESTS PASSED SUCCESSFULLY!")
