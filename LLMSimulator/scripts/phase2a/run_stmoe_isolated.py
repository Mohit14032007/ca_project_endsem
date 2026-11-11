"""
Phase 2A: ST-MoE Experiment Isolation and Result Integrity Execution.

Guarantees 100% Experiment Isolation:
1. Every independent experiment receives a newly constructed STMoEPaperPredictor instance.
2. Freshly populated CCT dictionaries from base co-activation matrices with zero shared references.
3. Clean, empty initial History Table for each experiment.
4. Identical profiling data and evaluation opportunities across comparable runs.
5. Strict dataset integrity: excludes leaked deepseek_random; labels duplex_a100 and correlated_synthetic as diagnostic workloads.
6. Evaluates both Equation 1 interpretations:
   - st_moe_confidence_additive (Equation 1 literal: CCT confidence + HT confidence)
   - st_moe_indicator_intersection (Interpretation B: I(e in CCT) + I(e in HT))
"""

import os
import sys
import json
import time
import copy
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd

sim_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
workspace_root = os.path.dirname(sim_root) if os.path.basename(sim_root) == "LLMSimulator" else sim_root
if os.path.basename(sim_root) != "LLMSimulator":
    sim_root = os.path.join(workspace_root, "LLMSimulator")

sys.path.insert(0, sim_root)

from src.predictor.st_moe_paper import STMoEPaperPredictor
from src.predictor.evaluation import (
    EvaluationHarness,
    PredictionOpportunity,
    PredictionResultRecord
)
from src.predictor.adapters.moe_anywhere_adapter import MoeAnywhereDatasetAdapter
from src.predictor.adapters.moe_suite_adapter import MoeSuiteDatasetAdapter
from src.predictor.adapters.simulator_adapter import SimulatorDatasetAdapter
from src.predictor.adapters.synthetic_adapter import SyntheticCorrelatedDatasetAdapter


def populate_cct_from_co_matrix(st_moe: STMoEPaperPredictor, s: int, co_matrix: np.ndarray) -> None:
    """Populates CCT for stage s from co-activation matrix adhering to Algorithm 1."""
    num_routed_experts = st_moe.num_routed_experts
    cct_capacity = st_moe.cct_capacity
    st_moe.cct[s] = {}
    for u in range(num_routed_experts):
        row = co_matrix[u]
        dests = sorted(range(num_routed_experts), key=lambda idx: (-row[idx], idx))
        st_moe.cct[s][u] = {}
        stored = 0
        for v in dests:
            if row[v] > 0 and stored < cct_capacity:
                st_moe.cct[s][u][v] = 2  # Binary 10: initial weakly preferred
                stored += 1


def compute_co_matrices_moe_anywhere(
    adapter: MoeAnywhereDatasetAdapter,
    allowed_prompts: Set[Any],
    horizon: int
) -> Tuple[List[np.ndarray], int]:
    """Computes co-activation matrices for moe_anywhere using vectorized NumPy arrays."""
    num_stages = adapter.num_moe_stages
    num_experts = adapter.num_routed_experts
    num_transitions = max(1, num_stages - horizon)
    co_matrices = [
        np.zeros((num_experts, num_experts), dtype=np.int64)
        for _ in range(num_transitions)
    ]

    total_train_tokens = 0
    for pid in allowed_prompts:
        p_tokens = adapter.prompt_tokens.get(pid, [])
        total_train_tokens += len(p_tokens)
        if not p_tokens:
            continue

        arr = np.full((len(p_tokens), num_stages, adapter.top_k), -1, dtype=np.int32)
        for t_idx, tpos in enumerate(p_tokens):
            for stage in range(num_stages):
                k = (pid, tpos, stage)
                if k in adapter.routes:
                    exps, _ = adapter.routes[k]
                    for r_idx, e in enumerate(exps[:adapter.top_k]):
                        arr[t_idx, stage, r_idx] = e

        for s in range(num_transitions):
            target_stage = s + horizon
            src_slice = arr[:, s, :]
            tgt_slice = arr[:, target_stage, :]
            valid = (src_slice[:, :, None] >= 0) & (tgt_slice[:, None, :] >= 0)
            u_indices = np.broadcast_to(src_slice[:, :, None], valid.shape)[valid]
            v_indices = np.broadcast_to(tgt_slice[:, None, :], valid.shape)[valid]
            np.add.at(co_matrices[s], (u_indices, v_indices), 1)

    return co_matrices, total_train_tokens


def compute_co_matrices_moe_suite(
    adapter: MoeSuiteDatasetAdapter,
    allowed_tasks: Set[Any],
    horizon: int
) -> Tuple[List[np.ndarray], int]:
    """Computes co-activation matrices for moe_suite using vectorized NumPy slices."""
    num_stages = adapter.num_moe_stages
    num_experts = adapter.num_routed_experts
    num_transitions = max(1, num_stages - horizon)
    co_matrices = [
        np.zeros((num_experts, num_experts), dtype=np.int64)
        for _ in range(num_transitions)
    ]

    total_train_tokens = 0
    for task_id in allowed_tasks:
        if task_id not in adapter.task_routes:
            continue
        arr = adapter.task_routes[task_id]
        total_train_tokens += len(arr)

        for s in range(num_transitions):
            target_stage = s + horizon
            src_slice = arr[:, s, :]
            tgt_slice = arr[:, target_stage, :]
            valid = (src_slice[:, :, None] >= 0) & (tgt_slice[:, None, :] >= 0)
            u_indices = np.broadcast_to(src_slice[:, :, None], valid.shape)[valid]
            v_indices = np.broadcast_to(tgt_slice[:, None, :], valid.shape)[valid]
            np.add.at(co_matrices[s], (u_indices, v_indices), 1)

    return co_matrices, total_train_tokens


def compute_co_matrices_generic(
    adapter: Any,
    allowed_reqs: Set[Any],
    horizon: int
) -> Tuple[List[np.ndarray], int]:
    num_stages = getattr(adapter, "num_moe_stages", getattr(adapter, "num_stages", 57))
    num_experts = getattr(adapter, "num_routed_experts", getattr(adapter, "num_experts", 256))
    num_transitions = max(1, num_stages - horizon)
    co_matrices = [
        np.zeros((num_experts, num_experts), dtype=np.int64)
        for _ in range(num_transitions)
    ]
    seen_tokens = set()

    for rec in adapter.iter_predictions(horizon=horizon, split="train"):
        if rec.request_id in allowed_reqs:
            seen_tokens.add((rec.request_id, rec.token_id))
            s = rec.source_stage_id
            if s is not None and 0 <= s < num_transitions:
                for u in rec.source_experts:
                    for v in rec.actual_target_experts:
                        if 0 <= u < num_experts and 0 <= v < num_experts:
                            co_matrices[s][u, v] += 1

    return co_matrices, len(seen_tokens)


def extract_test_opportunities_anywhere(
    adapter: MoeAnywhereDatasetAdapter,
    horizon: int,
    max_tokens_per_prompt: int = 30
) -> Tuple[List[PredictionOpportunity], int]:
    test_prompts = adapter.get_splits()["test"]
    test_opps = []
    total_eval_tokens = 0

    for pid in test_prompts:
        p_tokens = adapter.prompt_tokens.get(pid, [])[:max_tokens_per_prompt]
        total_eval_tokens += len(p_tokens)
        for tpos in p_tokens:
            for target_stage in range(adapter.num_moe_stages):
                source_stage = target_stage - horizon
                if source_stage >= 0:
                    src_key = (pid, tpos, source_stage)
                    source_experts = adapter.routes.get(src_key, ([], None))[0]
                else:
                    source_experts = []

                tgt_key = (pid, tpos, target_stage)
                if tgt_key not in adapter.routes:
                    continue
                actual_target, _ = adapter.routes[tgt_key]

                opp = PredictionOpportunity(
                    request_id=pid,
                    token_id=tpos,
                    moe_stage_id=target_stage,
                    source_experts=source_experts,
                    actual_experts=actual_target,
                    source_stage_id=source_stage if source_stage >= 0 else None,
                    prediction_horizon=horizon
                )
                test_opps.append(opp)

    return test_opps, total_eval_tokens


def extract_test_opportunities_suite(
    adapter: MoeSuiteDatasetAdapter,
    horizon: int,
    max_tokens_per_task: int = 50
) -> Tuple[List[PredictionOpportunity], int]:
    test_tasks = adapter.splits["test"]
    test_opps = []
    total_eval_tokens = 0

    for task_idx, task_id in enumerate(test_tasks):
        arr = adapter.task_routes[task_id]
        num_eval_tokens = min(len(arr), max_tokens_per_task)
        total_eval_tokens += num_eval_tokens
        req_id = 9000 + task_idx

        for tpos in range(num_eval_tokens):
            for target_stage in range(adapter.num_moe_stages):
                source_stage = target_stage - horizon
                if source_stage >= 0:
                    source_experts = [int(e) for e in arr[tpos, source_stage].tolist()]
                else:
                    source_experts = []

                actual_target = [int(e) for e in arr[tpos, target_stage].tolist()]

                opp = PredictionOpportunity(
                    request_id=req_id,
                    token_id=tpos,
                    moe_stage_id=target_stage,
                    source_experts=source_experts,
                    actual_experts=actual_target,
                    source_stage_id=source_stage if source_stage >= 0 else None,
                    prediction_horizon=horizon
                )
                test_opps.append(opp)

    return test_opps, total_eval_tokens


def extract_test_opportunities_generic(
    adapter: Any,
    horizon: int
) -> Tuple[List[PredictionOpportunity], int]:
    test_opps = []
    test_tokens = set()

    for rec in adapter.iter_predictions(horizon=horizon, split="test"):
        test_tokens.add((rec.request_id, rec.token_id))
        opp = PredictionOpportunity(
            request_id=rec.request_id,
            token_id=rec.token_id,
            moe_stage_id=rec.target_stage_id,
            source_experts=rec.source_experts,
            actual_experts=rec.actual_target_experts,
            source_stage_id=rec.source_stage_id,
            prediction_horizon=horizon
        )
        test_opps.append(opp)

    return test_opps, len(test_tokens)


def run_isolated_reproduction():
    out_dir = os.path.join(workspace_root, "results", "phase2a_stmoe_isolation")
    cfg_dir = os.path.join(out_dir, "experiment_configurations")
    os.makedirs(cfg_dir, exist_ok=True)

    print("=" * 80)
    print("STARTING PHASE 2A: ST-MoE EXPERIMENT ISOLATION & INTEGRITY VERIFICATION")
    print(f"Results Directory: {out_dir}")
    print("=" * 80)

    # 1. Instantiate Dataset Adapters
    print("\n[1/5] Loading Dataset Adapters...")
    p_anywhere = os.path.join(sim_root, "data", "downloads", "moe_anywhere", "e1-sage-a1.5.parquet")
    adapter_anywhere = MoeAnywhereDatasetAdapter(p_anywhere)
    adapter_anywhere.load()

    d_suite = os.path.join(sim_root, "data", "downloads", "moe_suite")
    adapter_suite = MoeSuiteDatasetAdapter(d_suite)
    adapter_suite.load()

    p_duplex = os.path.join(sim_root, "reproduction_A100", "results", "a100_duplex", "traces", "expert_routing.csv")
    adapter_duplex = SimulatorDatasetAdapter(
        dataset_id="duplex_a100",
        trace_path=p_duplex,
        num_moe_stages=32,
        num_routed_experts=8,
        top_k=2,
        first_k_dense=0,
        train_ratio=0.75
    )
    adapter_duplex.load()

    p_synth = os.path.join(sim_root, "results", "endsem_phase1", "benchmark_correlated_trace.csv")
    adapter_synth = SyntheticCorrelatedDatasetAdapter(p_synth)
    adapter_synth.load()

    # Authoritative & Diagnostic Workload Matrix
    # Excludes deepseek_random (flagged as leaked single request)
    workloads = [
        ("moe_anywhere", adapter_anywhere, 1, [1, 2, 4, 6, 8], "authoritative"),
        ("moe_anywhere", adapter_anywhere, 2, [1, 2, 4, 6, 8], "authoritative"),
        ("moe_suite", adapter_suite, 2, [1, 2, 4, 6, 8], "authoritative"),
        ("moe_suite", adapter_suite, 1, [1, 2, 4, 6, 8], "authoritative"),
        ("duplex_a100", adapter_duplex, 1, [1, 2, 4], "diagnostic"),
        ("correlated_synthetic", adapter_synth, 1, [1, 2, 4, 6, 8], "diagnostic")
    ]

    fixed_budget_rows = []
    native_thresh_rows = []
    profiling_rows = []

    for name, adapter, horizon, budgets, classification in workloads:
        print(f"\n--- Workload: {name} (H={horizon}, Class={classification}) ---")
        t0 = time.time()

        if name == "moe_anywhere":
            train_units = set(adapter.get_splits()["train"])
            test_opps, eval_tokens = extract_test_opportunities_anywhere(adapter, horizon, max_tokens_per_prompt=30)
            co_matrices, train_tokens = compute_co_matrices_moe_anywhere(adapter, train_units, horizon)
            num_stages = adapter.num_moe_stages
            top_k = adapter.top_k
            num_experts = adapter.num_routed_experts
        elif name == "moe_suite":
            train_units = set(adapter.splits["train"])
            test_opps, eval_tokens = extract_test_opportunities_suite(adapter, horizon, max_tokens_per_task=50)
            co_matrices, train_tokens = compute_co_matrices_moe_suite(adapter, train_units, horizon)
            num_stages = adapter.num_moe_stages
            top_k = adapter.top_k
            num_experts = adapter.num_routed_experts
        else:
            num_stages = getattr(adapter, "num_moe_stages", getattr(adapter, "num_stages", 57))
            top_k = getattr(adapter, "top_k", 8)
            num_experts = getattr(adapter, "num_routed_experts", getattr(adapter, "num_experts", 256))
            train_units = set(adapter.splits["train"])
            test_opps, eval_tokens = extract_test_opportunities_generic(adapter, horizon)
            co_matrices, train_tokens = compute_co_matrices_generic(adapter, train_units, horizon)

        meta = {
            "dataset_id": name,
            "classification": classification,
            "num_moe_stages": num_stages,
            "num_routed_experts": num_experts,
            "top_k": top_k,
            "horizon": horizon,
            "train_units": len(train_units),
            "test_units": len(set(op.request_id for op in test_opps)),
            "total_train_tokens": train_tokens,
            "total_eval_tokens": eval_tokens,
            "total_test_opportunities": len(test_opps),
            "experiment_isolation_policy": "fresh_instance_per_experiment"
        }

        print(f"  Profiling: {meta['train_units']} units, {meta['total_train_tokens']} tokens.")
        print(f"  Evaluation: {meta['test_units']} units, {meta['total_eval_tokens']} tokens, {meta['total_test_opportunities']} opps.")

        cfg_filename = f"config_{name}_h{horizon}.json"
        with open(os.path.join(cfg_dir, cfg_filename), "w") as f:
            json.dump(meta, f, indent=2)

        harness = EvaluationHarness(num_moe_stages=num_stages)
        num_transitions = max(1, num_stages - horizon)

        # Factory function guaranteeing 100% Predictor Instance Isolation
        def create_isolated_predictor(regime: str, scoring_mode: str = "confidence_additive", co_mats: List[np.ndarray] = co_matrices) -> STMoEPaperPredictor:
            pred = STMoEPaperPredictor(
                num_moe_stages=num_stages,
                num_routed_experts=num_experts,
                top_k=top_k,
                cct_capacity=top_k,
                confidence_threshold=2,
                online_adaptation=(regime == "online"),
                prediction_horizon=horizon,
                scoring_mode=scoring_mode
            )
            for s in range(num_transitions):
                populate_cct_from_co_matrix(pred, s, co_mats[s])
            return pred

        # -------------------------------------------------------------
        # PART A: Fixed-Budget Sweeps (Fully Isolated Predictor per B)
        # -------------------------------------------------------------
        for regime in ["frozen", "online"]:
            for b in budgets:
                # ISOLATION GUARANTEE: Fresh instance per (regime, budget) run
                fresh_predictor = create_isolated_predictor(regime=regime, scoring_mode="confidence_additive")

                res = harness.evaluate_predictor(
                    predictor_instance=fresh_predictor,
                    predictor_name=f"st_moe_{regime}",
                    test_opportunities=test_opps,
                    budgets=[b],
                    evaluation_mode=regime
                )
                summary = harness.compute_summary_metrics(res)

                row = {
                    "dataset_id": name,
                    "classification": classification,
                    "horizon": horizon,
                    "regime": regime,
                    "budget": b,
                    "scoring_mode": "confidence_additive",
                    "total_instances": summary["total_instances"],
                    "micro_recall": round(summary["micro_recall"], 4),
                    "macro_recall": round(summary["macro_recall"], 4),
                    "micro_precision": round(summary["micro_precision"], 4),
                    "macro_precision": round(summary["macro_precision"], 4),
                    "micro_f1": round(summary["micro_f1"], 4),
                    "macro_f1": round(summary["macro_f1"], 4),
                    "hit_rate": round(summary["hit_rate"], 4),
                    "exact_set_accuracy": round(summary["exact_set_accuracy"], 4),
                    "cold_start_instances": summary["cold_start"]["instances"],
                    "cold_start_recall": round(summary["cold_start"]["recall"], 4),
                    "cold_start_precision": round(summary["cold_start"]["precision"], 4),
                    "cold_start_f1": round(summary["cold_start"]["f1"], 4),
                    "cold_start_hit_rate": round(summary["cold_start"]["hit_rate"], 4),
                    "steady_state_instances": summary["steady_state"]["instances"],
                    "steady_state_recall": round(summary["steady_state"]["recall"], 4),
                    "steady_state_precision": round(summary["steady_state"]["precision"], 4),
                    "steady_state_f1": round(summary["steady_state"]["f1"], 4),
                    "steady_state_hit_rate": round(summary["steady_state"]["hit_rate"], 4),
                }
                fixed_budget_rows.append(row)
                if b == top_k:
                    print(f"  [Isolated] Fixed B={b} ({regime.upper()}): Recall={row['macro_recall']:.4f}, HitRate={row['hit_rate']:.4f}, ColdRecall={row['cold_start_recall']:.4f}, WarmRecall={row['steady_state_recall']:.4f}")

        # -------------------------------------------------------------
        # PART B: Native Threshold Mode (Both Eq 1 Interpretations Isolated)
        # -------------------------------------------------------------
        scoring_modes = ["confidence_additive", "indicator_intersection"]
        for s_mode in scoring_modes:
            for regime in ["frozen", "online"]:
                # ISOLATION GUARANTEE: Fresh instance per threshold run
                fresh_th_pred = create_isolated_predictor(regime=regime, scoring_mode=s_mode)

                res_th = harness.evaluate_predictor(
                    predictor_instance=fresh_th_pred,
                    predictor_name=f"st_moe_thresh_{regime}_{s_mode}",
                    test_opportunities=test_opps,
                    evaluation_mode="threshold",
                    threshold=2
                )
                summary_th = harness.compute_summary_metrics(res_th)

                set_sizes = [len(r.predicted_experts) for r in res_th]
                th_row = {
                    "dataset_id": name,
                    "classification": classification,
                    "horizon": horizon,
                    "regime": regime,
                    "threshold": 2,
                    "scoring_mode": s_mode,
                    "total_instances": summary_th["total_instances"],
                    "mean_predicted_set_size": round(float(np.mean(set_sizes)), 2),
                    "median_predicted_set_size": round(float(np.median(set_sizes)), 1),
                    "min_predicted_set_size": int(np.min(set_sizes)),
                    "max_predicted_set_size": int(np.max(set_sizes)),
                    "micro_recall": round(summary_th["micro_recall"], 4),
                    "macro_recall": round(summary_th["macro_recall"], 4),
                    "micro_precision": round(summary_th["micro_precision"], 4),
                    "macro_precision": round(summary_th["macro_precision"], 4),
                    "micro_f1": round(summary_th["micro_f1"], 4),
                    "macro_f1": round(summary_th["macro_f1"], 4),
                    "hit_rate": round(summary_th["hit_rate"], 4),
                    "exact_set_accuracy": round(summary_th["exact_set_accuracy"], 4),
                    "cold_start_recall": round(summary_th["cold_start"]["recall"], 4),
                    "cold_start_precision": round(summary_th["cold_start"]["precision"], 4),
                    "steady_state_recall": round(summary_th["steady_state"]["recall"], 4),
                    "steady_state_precision": round(summary_th["steady_state"]["precision"], 4),
                }
                native_thresh_rows.append(th_row)
                print(f"  [Isolated] Thresh ({regime.upper()}, {s_mode}): SetSize={th_row['mean_predicted_set_size']}, Recall={th_row['macro_recall']:.4f}, Prec={th_row['macro_precision']:.4f}, WarmRecall={th_row['steady_state_recall']:.4f}")

        # -------------------------------------------------------------
        # PART C: Profiling Data Sensitivity (Isolated Instances per Fraction)
        # -------------------------------------------------------------
        if name in ["moe_anywhere", "moe_suite"]:
            print(f"  Conducting Isolated Profiling Sensitivity Sweep for {name}...")
            fractions = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
            ordered_units = list(train_units)

            for frac in fractions:
                num_take = max(1, int(round(len(ordered_units) * frac)))
                sub_units = set(ordered_units[:num_take])

                if name == "moe_anywhere":
                    sub_co_mats, sub_tokens = compute_co_matrices_moe_anywhere(adapter, sub_units, horizon)
                else:
                    sub_co_mats, sub_tokens = compute_co_matrices_moe_suite(adapter, sub_units, horizon)

                for regime in ["frozen", "online"]:
                    # ISOLATION GUARANTEE: Fresh instance for budget sweep
                    fresh_sens_b = create_isolated_predictor(regime=regime, scoring_mode="confidence_additive", co_mats=sub_co_mats)
                    res_sens_b = harness.evaluate_predictor(
                        predictor_instance=fresh_sens_b,
                        predictor_name=f"st_moe_sens_b_{regime}",
                        test_opportunities=test_opps,
                        budgets=[top_k],
                        evaluation_mode=regime
                    )
                    sm_b = harness.compute_summary_metrics(res_sens_b)

                    # ISOLATION GUARANTEE: Brand new fresh instance for threshold sweep
                    fresh_sens_th = create_isolated_predictor(regime=regime, scoring_mode="confidence_additive", co_mats=sub_co_mats)
                    res_sens_th = harness.evaluate_predictor(
                        predictor_instance=fresh_sens_th,
                        predictor_name=f"st_moe_sens_th_{regime}",
                        test_opportunities=test_opps,
                        evaluation_mode="threshold",
                        threshold=2
                    )
                    sm_th = harness.compute_summary_metrics(res_sens_th)

                    prof_row = {
                        "dataset_id": name,
                        "classification": classification,
                        "horizon": horizon,
                        "profiling_fraction": frac,
                        "regime": regime,
                        "profiling_units": num_take,
                        "profiling_tokens": sub_tokens,
                        "evaluation_tokens": eval_tokens,
                        "b_k_recall": round(sm_b["macro_recall"], 4),
                        "b_k_precision": round(sm_b["macro_precision"], 4),
                        "b_k_hit_rate": round(sm_b["hit_rate"], 4),
                        "b_k_cold_start_recall": round(sm_b["cold_start"]["recall"], 4),
                        "b_k_steady_state_recall": round(sm_b["steady_state"]["recall"], 4),
                        "thresh_mean_set_size": round(float(np.mean([len(r.predicted_experts) for r in res_sens_th])), 2),
                        "thresh_recall": round(sm_th["macro_recall"], 4),
                        "thresh_precision": round(sm_th["macro_precision"], 4),
                        "thresh_cold_start_recall": round(sm_th["cold_start"]["recall"], 4),
                        "thresh_steady_state_recall": round(sm_th["steady_state"]["recall"], 4),
                    }
                    profiling_rows.append(prof_row)

        print(f"Completed {name} (H={horizon}) in {time.time() - t0:.2f}s")

    # Export Isolated Results
    print("\n[5/5] Exporting Phase 2A Isolated Result Artifacts...")
    df_fixed = pd.DataFrame(fixed_budget_rows)
    p_fixed = os.path.join(out_dir, "stmoe_fixed_budget_results.csv")
    df_fixed.to_csv(p_fixed, index=False)
    print(f"Exported Fixed-Budget Results: {len(df_fixed)} rows -> {p_fixed}")

    df_th = pd.DataFrame(native_thresh_rows)
    p_th = os.path.join(out_dir, "stmoe_native_threshold_results.csv")
    df_th.to_csv(p_th, index=False)
    print(f"Exported Native Threshold Results: {len(df_th)} rows -> {p_th}")

    df_prof = pd.DataFrame(profiling_rows)
    p_prof = os.path.join(out_dir, "stmoe_profiling_sensitivity.csv")
    df_prof.to_csv(p_prof, index=False)
    print(f"Exported Profiling Sensitivity Results: {len(df_prof)} rows -> {p_prof}")

    print("=" * 80)
    print("PHASE 2A ISOLATED REPRODUCTION COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_isolated_reproduction()

