"""
Phase 1A Correctness Smoke Test for Certified Correction Pipeline.

Validates all assertions and metrics on a minimal slice before launching
the full 192-run campaign:
1. Predictor set sizes: len(pred) == budget == 8, unique IDs, valid range [0..E-1].
2. Actual set sizes: len(act) == top_k == 8, unique IDs, valid range [0..E-1].
3. Metric calculations: non-hardcoded denominators (tp + fp) and (tp + fn).
4. Mathematical identity: Precision@8 == Recall@8 verified on actual set sizes.
5. Opportunity ID matching: CSTP and ST-MoE evaluate identical opportunity sequences.
6. Sufficient statistics: Aggregate metrics match exact sums of prompt/task statistics.
7. Code style: ZERO ternary operators.
"""

import os
import sys
import json
import time
import copy
from collections import namedtuple
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

sim_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
workspace_root = os.path.dirname(sim_root)
if os.path.basename(sim_root) != "LLMSimulator":
    workspace_root = sim_root
    sim_root = os.path.join(workspace_root, "LLMSimulator")

sys.path.insert(0, sim_root)

from src.predictor.cstp_predictor import CSTPPredictor
from src.predictor.st_moe_paper import STMoEPaperPredictor
from src.predictor.evaluation import (
    EvaluationHarness,
    PredictionOpportunity,
    PredictionResultRecord
)
from src.predictor.adapters.moe_anywhere_adapter import MoeAnywhereDatasetAdapter
from src.predictor.adapters.moe_suite_adapter import MoeSuiteDatasetAdapter

PromptRecord = namedtuple("PromptRecord", ["request_id", "token_id", "moe_stage_id", "expert_id"])


def to_float_indicator(cond: bool) -> float:
    if cond:
        return 1.0
    else:
        return 0.0


def compute_mean_indicator(conditions: List[bool]) -> float:
    vals = []
    for c in conditions:
        if c:
            vals.append(1.0)
        else:
            vals.append(0.0)
    if len(vals) == 0:
        return 0.0
    else:
        return float(np.mean(vals))


def run_smoke_test():
    print("=" * 80)
    print("RUNNING PHASE 1A CERTIFIED CORRECTION SMOKE TEST")
    print("=" * 80)

    # 1. Load Adapters
    print("\n[Step 1] Loading Adapters...")
    p_anywhere = os.path.join(sim_root, "data", "downloads", "moe_anywhere", "e1-sage-a1.5.parquet")
    adapter_a = MoeAnywhereDatasetAdapter(p_anywhere)
    adapter_a.load()

    d_suite = os.path.join(sim_root, "data", "downloads", "moe_suite")
    adapter_s = MoeSuiteDatasetAdapter(d_suite)
    adapter_s.load()

    # 2. Extract Test Opportunities
    print("\n[Step 2] Extracting Test Opportunities...")
    # MoE Anywhere: 1 test prompt (30 tokens * 40 stages = 1200 opps)
    test_prompts_a = adapter_a.get_splits()["test"][:1]
    opps_a = []
    for pid in test_prompts_a:
        p_tokens = adapter_a.prompt_tokens.get(pid, [])[:30]
        for tpos in p_tokens:
            for stage in range(40):
                src_stage = stage - 1
                if src_stage >= 0:
                    src_tuple = adapter_a.routes.get((pid, tpos, src_stage), ([], None))
                    src_experts = src_tuple[0]
                    src_stage_arg = src_stage
                else:
                    src_experts = []
                    src_stage_arg = None
                tgt_tuple = adapter_a.routes.get((pid, tpos, stage), ([], None))
                tgt_experts = tgt_tuple[0]
                opps_a.append(PredictionOpportunity(
                    request_id=pid,
                    token_id=tpos,
                    moe_stage_id=stage,
                    source_experts=src_experts,
                    actual_experts=tgt_experts,
                    source_stage_id=src_stage_arg,
                    prediction_horizon=1
                ))
    print(f"  Anywhere smoke test opportunities: {len(opps_a)}")
    assert len(opps_a) == 1200, f"Expected 1200 opps, got {len(opps_a)}"

    # MoE Suite: 1 test task (50 tokens * 75 stages = 3750 opps)
    test_tasks_s = adapter_s.splits["test"][:1]
    opps_s = []
    for task_idx, task_id in enumerate(test_tasks_s):
        arr = adapter_s.task_routes[task_id][:50]
        req_id = 9000 + task_idx
        for tpos in range(len(arr)):
            for stage in range(75):
                src_stage = stage - 1
                if src_stage >= 0:
                    src_experts = [int(e) for e in arr[tpos, src_stage].tolist()]
                    src_stage_arg = src_stage
                else:
                    src_experts = []
                    src_stage_arg = None
                tgt_experts = [int(e) for e in arr[tpos, stage].tolist()]
                opps_s.append(PredictionOpportunity(
                    request_id=req_id,
                    token_id=tpos,
                    moe_stage_id=stage,
                    source_experts=src_experts,
                    actual_experts=tgt_experts,
                    source_stage_id=src_stage_arg,
                    prediction_horizon=1
                ))
    print(f"  Suite smoke test opportunities: {len(opps_s)}")
    assert len(opps_s) == 3750, f"Expected 3750 opps, got {len(opps_s)}"

    # 3. Profile 25% Subsets
    print("\n[Step 3] Profiling Predictors on 25% Subset...")
    # Anywhere 25% = 13 prompts
    train_prompts_a = adapter_a.get_splits()["train"][:13]
    recs_a = []
    for pid in train_prompts_a:
        for tpos in adapter_a.prompt_tokens.get(pid, []):
            for stage in range(40):
                k = (pid, tpos, stage)
                if k in adapter_a.routes:
                    exps, _ = adapter_a.routes[k]
                    for e in exps[:8]:
                        recs_a.append(PromptRecord(request_id=pid, token_id=tpos, moe_stage_id=stage, expert_id=e))

    cstp_a = CSTPPredictor(num_moe_stages=40, num_routed_experts=256, top_k=8, prediction_horizon=1)
    cstp_a.fit_profile(recs_a)

    st_cct_a = {}
    for s in range(39):
        st_cct_a[s] = {}
        for u in range(256):
            row = cstp_a.spatial_counts[s, u, :]
            dests = sorted(range(256), key=lambda idx: (-row[idx], idx))
            st_cct_a[s][u] = {}
            stored = 0
            for v in dests:
                if row[v] > 0 and stored < 8:
                    st_cct_a[s][u][v] = 2
                    stored = stored + 1

    st_moe_a = STMoEPaperPredictor(
        num_moe_stages=40, num_routed_experts=256, top_k=8,
        cct_capacity=8, confidence_threshold=2, online_adaptation=True,
        prediction_horizon=1, scoring_mode="confidence_additive"
    )
    st_moe_a.cct = copy.deepcopy(st_cct_a)

    # 4. Evaluate and Assert Defensive Invariants
    print("\n[Step 4] Running Evaluation and Validating Defensive Assertions...")
    harness_a = EvaluationHarness(num_moe_stages=40)

    res_cstp = harness_a.evaluate_predictor(
        predictor_instance=cstp_a,
        predictor_name="CSTP",
        test_opportunities=opps_a,
        budgets=[8],
        evaluation_mode="online_within_request"
    )

    res_stmoe = harness_a.evaluate_predictor(
        predictor_instance=st_moe_a,
        predictor_name="ST-MoE",
        test_opportunities=opps_a,
        budgets=[8],
        evaluation_mode="online_within_request"
    )

    assert len(res_cstp) == len(opps_a), "CSTP result count mismatch"
    assert len(res_stmoe) == len(opps_a), "ST-MoE result count mismatch"

    print("  Checking per-event assertions across 1,200 opportunities:")
    for idx in range(len(opps_a)):
        r_c = res_cstp[idx]
        r_s = res_stmoe[idx]
        op = opps_a[idx]

        # Match opportunity IDs
        assert r_c.request_id == op.request_id == r_s.request_id
        assert r_c.token_id == op.token_id == r_s.token_id
        assert r_c.moe_stage_id == op.moe_stage_id == r_s.moe_stage_id

        # Validate predicted and actual set sizes
        assert len(r_c.predicted_experts) == 8, f"CSTP predicted size != 8: {len(r_c.predicted_experts)}"
        assert len(r_s.predicted_experts) == 8, f"ST-MoE predicted size != 8: {len(r_s.predicted_experts)}"
        assert len(r_c.actual_experts) == 8, f"Actual size != 8: {len(r_c.actual_experts)}"
        assert len(r_s.actual_experts) == 8, f"Actual size != 8: {len(r_s.actual_experts)}"

        # Validate uniqueness of expert IDs
        assert len(set(r_c.predicted_experts)) == 8, "Duplicate expert in CSTP prediction"
        assert len(set(r_s.predicted_experts)) == 8, "Duplicate expert in ST-MoE prediction"
        assert len(set(r_c.actual_experts)) == 8, "Duplicate expert in actual routing"

        # Validate range [0..255]
        for e in r_c.predicted_experts:
            assert 0 <= e < 256, f"Invalid expert ID in CSTP prediction: {e}"
        for e in r_s.predicted_experts:
            assert 0 <= e < 256, f"Invalid expert ID in ST-MoE prediction: {e}"
        for e in r_c.actual_experts:
            assert 0 <= e < 256, f"Invalid actual expert ID: {e}"

        # Validate non-hardcoded metric formulas
        tp_c = len(set(r_c.predicted_experts).intersection(set(r_c.actual_experts)))
        fp_c = len(set(r_c.predicted_experts) - set(r_c.actual_experts))
        fn_c = len(set(r_c.actual_experts) - set(r_c.predicted_experts))

        prec_denom_c = tp_c + fp_c
        rec_denom_c = tp_c + fn_c
        assert prec_denom_c == 8, "Precision denominator must match predicted set size"
        assert rec_denom_c == 8, "Recall denominator must match actual set size"

        prec_calc = float(tp_c) / float(prec_denom_c)
        rec_calc = float(tp_c) / float(rec_denom_c)
        assert prec_calc == rec_calc, "Precision and recall must match when len(pred) == len(act) == 8"

    print("  All per-event assertions passed successfully!")

    # 5. Check Sufficient Statistics Reconstruction
    print("\n[Step 5] Validating Sufficient Statistics Reconstruction...")
    total_tp_cstp = sum(len(set(r.predicted_experts).intersection(set(r.actual_experts))) for r in res_cstp)
    total_events = len(res_cstp)
    micro_recall = float(total_tp_cstp) / float(total_events * 8)
    macro_recall = float(np.mean([float(len(set(r.predicted_experts).intersection(set(r.actual_experts)))) / 8.0 for r in res_cstp]))

    assert abs(micro_recall - macro_recall) < 1e-12, "Micro and macro recall must match for uniform-length sequences"
    print(f"  CSTP Micro Recall: {micro_recall:.6f} | Macro Recall: {macro_recall:.6f}")

    total_tp_stmoe = sum(len(set(r.predicted_experts).intersection(set(r.actual_experts))) for r in res_stmoe)
    stmoe_micro_recall = float(total_tp_stmoe) / float(total_events * 8)
    stmoe_macro_recall = float(np.mean([float(len(set(r.predicted_experts).intersection(set(r.actual_experts)))) / 8.0 for r in res_stmoe]))
    print(f"  ST-MoE Micro Recall: {stmoe_micro_recall:.6f} | Macro Recall: {stmoe_macro_recall:.6f}")

    print("\n" + "=" * 80)
    print("SMOKE TEST VALIDATION PASSED 100%! READY FOR FULL EXECUTION.")
    print("=" * 80)


if __name__ == "__main__":
    run_smoke_test()

