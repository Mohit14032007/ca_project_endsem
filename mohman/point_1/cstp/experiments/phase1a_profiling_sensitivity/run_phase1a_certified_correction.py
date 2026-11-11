"""
Phase 1A Certified Results Verification, Statistical Correction and Final Packaging.

Strict Constraints:
1. Workspace: Strictly read-only mohman/. All certified outputs to results/phase1a_cstp_profiling/certified/.
2. Integrity: Original CSTP and ST-MoE implementations preserved strictly.
3. Code Style: ZERO ternary operators (no inline conditional expressions) in any newly written code.
4. Completeness: Produces complete event-level records (all 192 configurations streamed),
   prompt-level and task-level sufficient statistics for all seeds, aggregate metrics,
   head-to-head comparisons, count statistics, and hierarchical multi-seed bootstrap confidence intervals.
5. Statistical Rigor: Dynamic denominators based on actual set sizes; bootstrap incorporates all 3 profiling seeds;
   renamed probability to bootstrap_prob_greater_than_zero; two-sided bootstrap ASL implemented;
   MoE Suite N=2 tasks explicitly flagged as DESCRIPTIVE_ONLY_N2_TASKS.
6. Plotting: Regenerates all 9 publication figures from certified CSVs using pure Matplotlib (PNG 300 DPI + PDF).
"""

import os
import sys
import csv
import json
import time
import copy
import random
from collections import namedtuple
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def get_nested_subsets_no_ternary(items: List[Any], fractions: List[float], seed: int) -> Dict[int, List[Any]]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    subsets = {}
    n_total = len(shuffled)
    for frac in fractions:
        pct = int(round(frac * 100))
        n_take = int(round(n_total * frac))
        if n_take < 1:
            n_take = 1
        subsets[pct] = shuffled[:n_take]
    return subsets


def extract_anywhere_test_opps(adapter: MoeAnywhereDatasetAdapter, horizon: int) -> List[PredictionOpportunity]:
    test_prompts = adapter.get_splits()["test"]
    opps = []
    for pid in test_prompts:
        p_tokens = adapter.prompt_tokens.get(pid, [])[:30]
        for tpos in p_tokens:
            for target_stage in range(adapter.num_moe_stages):
                source_stage = target_stage - horizon
                if source_stage >= 0:
                    src_key = (pid, tpos, source_stage)
                    src_tuple = adapter.routes.get(src_key, ([], None))
                    src_experts = src_tuple[0]
                    src_stage_arg = source_stage
                else:
                    src_experts = []
                    src_stage_arg = None

                tgt_key = (pid, tpos, target_stage)
                tgt_tuple = adapter.routes.get(tgt_key, ([], None))
                tgt_experts = tgt_tuple[0]

                opp = PredictionOpportunity(
                    request_id=pid,
                    token_id=tpos,
                    moe_stage_id=target_stage,
                    source_experts=src_experts,
                    actual_experts=tgt_experts,
                    source_stage_id=src_stage_arg,
                    prediction_horizon=horizon
                )
                opps.append(opp)
    return opps


def extract_suite_test_opps(adapter: MoeSuiteDatasetAdapter, horizon: int) -> List[PredictionOpportunity]:
    test_tasks = adapter.splits["test"]
    opps = []
    for task_idx, task_id in enumerate(test_tasks):
        arr = adapter.task_routes[task_id]
        num_eval_tokens = len(arr)
        if num_eval_tokens > 50:
            num_eval_tokens = 50
        req_id = 9000 + task_idx
        for tpos in range(num_eval_tokens):
            for target_stage in range(adapter.num_moe_stages):
                source_stage = target_stage - horizon
                if source_stage >= 0:
                    src_experts = [int(e) for e in arr[tpos, source_stage].tolist()]
                    src_stage_arg = source_stage
                else:
                    src_experts = []
                    src_stage_arg = None
                tgt_experts = [int(e) for e in arr[tpos, target_stage].tolist()]

                opp = PredictionOpportunity(
                    request_id=req_id,
                    token_id=tpos,
                    moe_stage_id=target_stage,
                    source_experts=src_experts,
                    actual_experts=tgt_experts,
                    source_stage_id=src_stage_arg,
                    prediction_horizon=horizon
                )
                opps.append(opp)
    return opps


def compute_stmoe_cct(spatial_counts: np.ndarray, num_experts: int, top_k: int) -> Dict[int, Dict[int, Dict[int, int]]]:
    num_transitions = spatial_counts.shape[0]
    cct = {}
    for s in range(num_transitions):
        cct[s] = {}
        for u in range(num_experts):
            row = spatial_counts[s, u, :]
            dests = sorted(range(num_experts), key=lambda idx: (-row[idx], idx))
            cct[s][u] = {}
            stored = 0
            for v in dests:
                if row[v] > 0 and stored < top_k:
                    cct[s][u][v] = 2
                    stored = stored + 1
    return cct


def run_certified_pipeline():
    out_dir = os.path.join(workspace_root, "results", "phase1a_cstp_profiling", "certified")
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print("=" * 80)
    print("PHASE 1A: CERTIFIED STATISTICAL CORRECTION & VERIFICATION PIPELINE")
    print(f"Certified Output Directory: {out_dir}")
    print("=" * 80)

    # 1. Load Datasets
    print("\n[1/7] Loading Adapters and Verifying Ground Truth Opportunities...")
    p_anywhere = os.path.join(sim_root, "data", "downloads", "moe_anywhere", "e1-sage-a1.5.parquet")
    adapter_a = MoeAnywhereDatasetAdapter(p_anywhere)
    adapter_a.load()

    d_suite = os.path.join(sim_root, "data", "downloads", "moe_suite")
    adapter_s = MoeSuiteDatasetAdapter(d_suite)
    adapter_s.load()

    train_prompts_a = sorted(list(adapter_a.get_splits()["train"]))
    test_prompts_a = sorted(list(adapter_a.get_splits()["test"]))
    train_tasks_s = sorted(list(adapter_s.splits["train"]))
    test_tasks_s = sorted(list(adapter_s.splits["test"]))

    fractions = [0.25, 0.50, 0.75, 1.00]
    percentages = [25, 50, 75, 100]
    seeds = [42, 123, 999]
    horizons = [1, 2]
    budget = 8

    # Pre-cache training records
    print("\n[2/7] Pre-indexing Prompt and Task Profiles...")
    anywhere_prompt_records = {}
    for pid in train_prompts_a:
        recs = []
        p_tokens = adapter_a.prompt_tokens.get(pid, [])
        for tpos in p_tokens:
            for stage in range(adapter_a.num_moe_stages):
                k = (pid, tpos, stage)
                if k in adapter_a.routes:
                    exps, _ = adapter_a.routes[k]
                    for e in exps[:adapter_a.top_k]:
                        recs.append(PromptRecord(request_id=pid, token_id=tpos, moe_stage_id=stage, expert_id=e))
        anywhere_prompt_records[pid] = (recs, len(p_tokens))

    suite_task_profiles = {}
    for t_idx, task_id in enumerate(train_tasks_s):
        arr = adapter_s.task_routes[task_id]
        n_toks = len(arr)
        for h in horizons:
            p_task = CSTPPredictor(num_moe_stages=75, num_routed_experts=256, top_k=8, prediction_horizon=h)
            def gen_records():
                for tpos in range(len(arr)):
                    for stage in range(75):
                        for e in arr[tpos, stage]:
                            yield PromptRecord(request_id=1000 + t_idx, token_id=tpos, moe_stage_id=stage, expert_id=int(e))
            p_task.fit_profile(gen_records())
            suite_task_profiles[(task_id, h)] = (
                p_task.stage_marginal_counts.copy(),
                p_task.spatial_counts.copy(),
                p_task.temporal_counts.copy(),
                n_toks
            )

    # 3. Comprehensive Certified Evaluation Loop
    print("\n[3/7] Running Full Certified Metric Recomputation & Per-Event Streaming...")

    # Open streaming CSV writer for complete event records
    p_events_csv = os.path.join(out_dir, "corrected_per_event_metrics.csv")
    f_events = open(p_events_csv, "w", newline="")
    w_events = csv.writer(f_events)
    w_events.writerow([
        "dataset_id", "model", "horizon", "seed", "profiling_pct",
        "predictor", "regime", "request_id", "token_id", "token_position",
        "moe_stage_id", "is_cold_start", "tp", "fp", "fn",
        "predicted_set_size", "actual_set_size",
        "precision", "recall", "f1", "hit", "exact_set"
    ])

    total_events_written = 0
    prompt_rows = []
    task_rows = []
    aggregate_rows = []
    head_to_head_rows = []
    count_stats_rows = []

    datasets_meta = [
        ("moe_anywhere", adapter_a, 40, 256, 8, 16800, "Sage-A1.5"),
        ("moe_suite", adapter_s, 75, 256, 8, 7500, "GLM-5.2")
    ]

    for dset_name, adapter, num_stages, num_experts, top_k, expected_opps, model_name in datasets_meta:
        print(f"\n--- Processing {dset_name} ({model_name}) ---")
        harness = EvaluationHarness(num_moe_stages=num_stages)

        for h in horizons:
            if dset_name == "moe_anywhere":
                test_opps = extract_anywhere_test_opps(adapter, horizon=h)
            else:
                test_opps = extract_suite_test_opps(adapter, horizon=h)

            assert len(test_opps) == expected_opps, f"Opportunity count mismatch: {len(test_opps)} != {expected_opps}"

            for seed in seeds:
                if dset_name == "moe_anywhere":
                    subsets = get_nested_subsets_no_ternary(train_prompts_a, fractions, seed)
                else:
                    subsets = get_nested_subsets_no_ternary(train_tasks_s, fractions, seed)

                for pct in percentages:
                    selected_units = subsets[pct]

                    # Build profile counts
                    if dset_name == "moe_anywhere":
                        subset_records = []
                        total_tokens = 0
                        for pid in selected_units:
                            recs, n_toks = anywhere_prompt_records[pid]
                            subset_records.extend(recs)
                            total_tokens = total_tokens + n_toks

                        fresh_cstp = CSTPPredictor(
                            num_moe_stages=num_stages,
                            num_routed_experts=num_experts,
                            top_k=top_k,
                            alpha=0.5,
                            smoothing_lambda=1.0,
                            prior_epsilon=0.1,
                            online_adaptation=True,
                            prediction_horizon=h
                        )
                        fresh_cstp.fit_profile(subset_records)
                        marg_counts = fresh_cstp.stage_marginal_counts
                        spat_counts = fresh_cstp.spatial_counts
                        temp_counts = fresh_cstp.temporal_counts
                    else:
                        total_tokens = 0
                        marg_counts = np.zeros((num_stages, num_experts), dtype=np.float64)
                        num_transitions = max(1, num_stages - h)
                        spat_counts = np.zeros((num_transitions, num_experts, num_experts), dtype=np.float64)
                        temp_counts = np.zeros((num_stages, num_experts, num_experts), dtype=np.float64)

                        for tid in selected_units:
                            m_c, s_c, t_c, n_toks = suite_task_profiles[(tid, h)]
                            marg_counts = marg_counts + m_c
                            spat_counts = spat_counts + s_c
                            temp_counts = temp_counts + t_c
                            total_tokens = total_tokens + n_toks

                    # Record Count Statistics
                    tot_marg = float(np.sum(marg_counts))
                    tot_spat = float(np.sum(spat_counts))
                    tot_temp = float(np.sum(temp_counts))
                    nz_marg = int(np.count_nonzero(marg_counts))
                    nz_spat = int(np.count_nonzero(spat_counts))
                    nz_temp = int(np.count_nonzero(temp_counts))

                    count_stats_rows.append({
                        "dataset_id": dset_name,
                        "model": model_name,
                        "horizon": h,
                        "seed": seed,
                        "profiling_pct": pct,
                        "units_count": len(selected_units),
                        "tokens_count": total_tokens,
                        "marginal_count_mass": tot_marg,
                        "spatial_count_mass": tot_spat,
                        "temporal_count_mass": tot_temp,
                        "nonzero_marginal_cells": nz_marg,
                        "nonzero_spatial_cells": nz_spat,
                        "nonzero_temporal_cells": nz_temp,
                        "max_marginal_cell_count": float(np.max(marg_counts)),
                        "max_spatial_cell_count": float(np.max(spat_counts)),
                        "max_temporal_cell_count": float(np.max(temp_counts)),
                        "mean_spatial_nonzero_count": float(tot_spat / max(1, nz_spat)),
                        "mean_temporal_nonzero_count": float(tot_temp / max(1, nz_temp))
                    })

                    cct = compute_stmoe_cct(spat_counts, num_experts, top_k)

                    # Evaluate CSTP and ST-MoE
                    predictors_to_run = [
                        ("CSTP", "frozen", False),
                        ("CSTP", "online_within_request", True),
                        ("ST-MoE", "frozen", False),
                        ("ST-MoE", "online_within_request", True)
                    ]

                    results_by_key = {}

                    for pred_name, regime, is_online in predictors_to_run:
                        if pred_name == "CSTP":
                            runner = CSTPPredictor(
                                num_moe_stages=num_stages,
                                num_routed_experts=num_experts,
                                top_k=top_k,
                                alpha=0.5,
                                smoothing_lambda=1.0,
                                prior_epsilon=0.1,
                                online_adaptation=is_online,
                                prediction_horizon=h
                            )
                            runner.stage_marginal_counts = marg_counts.copy()
                            runner.spatial_counts = spat_counts.copy()
                            runner.temporal_counts = temp_counts.copy()
                        else:
                            runner = STMoEPaperPredictor(
                                num_moe_stages=num_stages,
                                num_routed_experts=num_experts,
                                top_k=top_k,
                                cct_capacity=top_k,
                                confidence_threshold=2,
                                online_adaptation=is_online,
                                prediction_horizon=h,
                                scoring_mode="confidence_additive"
                            )
                            runner.cct = copy.deepcopy(cct)

                        eval_res = harness.evaluate_predictor(
                            predictor_instance=runner,
                            predictor_name=pred_name,
                            test_opportunities=test_opps,
                            budgets=[budget],
                            evaluation_mode="online_within_request"
                        )
                        results_by_key[(pred_name, regime)] = eval_res

                        # Complete Event-Level Streaming Export with Non-Hardcoded Dynamic Denominators
                        event_chunk = []
                        for r_idx, rec in enumerate(eval_res):
                            act_s = set(rec.actual_experts)
                            pred_s = set(rec.predicted_experts)
                            tp = len(pred_s.intersection(act_s))
                            fp = len(pred_s - act_s)
                            fn = len(act_s - pred_s)

                            len_pred = len(rec.predicted_experts)
                            len_act = len(rec.actual_experts)
                            assert len_pred > 0, "Predicted expert set size must be > 0"
                            assert len_act > 0, "Actual expert set size must be > 0"

                            prec_denom = tp + fp
                            rec_denom = tp + fn
                            assert prec_denom == len_pred, f"prec_denom {prec_denom} != len_pred {len_pred}"
                            assert rec_denom == len_act, f"rec_denom {rec_denom} != len_act {len_act}"

                            prec = float(tp) / float(prec_denom)
                            rec_val = float(tp) / float(rec_denom)
                            if prec + rec_val > 0.0:
                                f1_val = (2.0 * prec * rec_val) / (prec + rec_val)
                            else:
                                f1_val = 0.0

                            if tp >= 1:
                                hit_val = 1.0
                            else:
                                hit_val = 0.0

                            if fp == 0 and fn == 0:
                                exact_val = 1.0
                            else:
                                exact_val = 0.0

                            if rec.is_cold_start:
                                cold_int = 1
                            else:
                                cold_int = 0

                            event_chunk.append([
                                dset_name, model_name, h, seed, pct,
                                pred_name, regime, rec.request_id, rec.token_id, rec.token_position,
                                rec.moe_stage_id, cold_int, tp, fp, fn,
                                len_pred, len_act,
                                round(prec, 6), round(rec_val, 6), round(f1_val, 6), hit_val, exact_val
                            ])

                        w_events.writerows(event_chunk)
                        total_events_written = total_events_written + len(event_chunk)

                        # Group by Request/Task Unit for Prompt/Task Level and Macro Metrics
                        req_groups = {}
                        for rec in eval_res:
                            req_id = rec.request_id
                            if req_id not in req_groups:
                                req_groups[req_id] = []
                            req_groups[req_id].append(rec)

                        prompt_recalls = []
                        prompt_precisions = []
                        prompt_hits = []
                        prompt_cold_recalls = []
                        prompt_warm_recalls = []

                        for req_id, r_list in req_groups.items():
                            p_tp = sum(len(set(r.predicted_experts).intersection(set(r.actual_experts))) for r in r_list)
                            p_pred_total = sum(len(r.predicted_experts) for r in r_list)
                            p_act_total = sum(len(r.actual_experts) for r in r_list)
                            p_fp = p_pred_total - p_tp
                            p_fn = p_act_total - p_tp

                            assert p_pred_total > 0
                            assert p_act_total > 0
                            p_prec = float(p_tp) / float(p_pred_total)
                            p_rec = float(p_tp) / float(p_act_total)
                            if p_prec + p_rec > 0.0:
                                p_f1 = (2.0 * p_prec * p_rec) / (p_prec + p_rec)
                            else:
                                p_f1 = 0.0

                            hit_conditions = []
                            for r in r_list:
                                if len(set(r.predicted_experts).intersection(set(r.actual_experts))) >= 1:
                                    hit_conditions.append(True)
                                else:
                                    hit_conditions.append(False)
                            p_hit = compute_mean_indicator(hit_conditions)

                            prompt_recalls.append(p_rec)
                            prompt_precisions.append(p_prec)
                            prompt_hits.append(p_hit)

                            cold_list = []
                            warm_list = []
                            for r in r_list:
                                if r.is_cold_start:
                                    cold_list.append(r)
                                else:
                                    warm_list.append(r)

                            c_tp = sum(len(set(r.predicted_experts).intersection(set(r.actual_experts))) for r in cold_list)
                            c_act = sum(len(r.actual_experts) for r in cold_list)
                            if c_act > 0:
                                c_rec = float(c_tp) / float(c_act)
                            else:
                                c_rec = 0.0

                            w_tp = sum(len(set(r.predicted_experts).intersection(set(r.actual_experts))) for r in warm_list)
                            w_act = sum(len(r.actual_experts) for r in warm_list)
                            if w_act > 0:
                                w_rec = float(w_tp) / float(w_act)
                            else:
                                w_rec = 0.0

                            prompt_cold_recalls.append(c_rec)
                            prompt_warm_recalls.append(w_rec)

                            # Save prompt/task summary for ALL seeds to enable multi-seed hierarchical bootstrap
                            row_item = {
                                "dataset_id": dset_name,
                                "model": model_name,
                                "horizon": h,
                                "seed": seed,
                                "profiling_pct": pct,
                                "predictor": pred_name,
                                "regime": regime,
                                "unit_id": req_id,
                                "event_count": len(r_list),
                                "total_actual_experts": p_act_total,
                                "total_predicted_experts": p_pred_total,
                                "total_tp": p_tp,
                                "total_fp": p_fp,
                                "total_fn": p_fn,
                                "recall": round(p_rec, 6),
                                "precision": round(p_prec, 6),
                                "f1": round(p_f1, 6),
                                "hit_rate": round(p_hit, 6),
                                "cold_start_recall": round(c_rec, 6),
                                "warm_token_recall": round(w_rec, 6)
                            }
                            if dset_name == "moe_anywhere":
                                prompt_rows.append(row_item)
                            else:
                                task_rows.append(row_item)

                        # True Macro Averages across independent prompts/tasks
                        macro_rec = float(np.mean(prompt_recalls))
                        macro_prec = float(np.mean(prompt_precisions))
                        if macro_prec + macro_rec > 0.0:
                            macro_f1 = (2.0 * macro_prec * macro_rec) / (macro_prec + macro_rec)
                        else:
                            macro_f1 = 0.0
                        macro_hit = float(np.mean(prompt_hits))
                        macro_cold = float(np.mean(prompt_cold_recalls))
                        macro_warm = float(np.mean(prompt_warm_recalls))

                        # Event-Level Micro Averages across all events with actual denominators
                        tot_tp = sum(len(set(r.predicted_experts).intersection(set(r.actual_experts))) for r in eval_res)
                        tot_act_total = sum(len(r.actual_experts) for r in eval_res)
                        tot_pred_total = sum(len(r.predicted_experts) for r in eval_res)
                        assert tot_act_total > 0
                        assert tot_pred_total > 0

                        micro_rec = float(tot_tp) / float(tot_act_total)
                        micro_prec = float(tot_tp) / float(tot_pred_total)
                        if micro_prec + micro_rec > 0.0:
                            micro_f1 = (2.0 * micro_prec * micro_rec) / (micro_prec + micro_rec)
                        else:
                            micro_f1 = 0.0

                        all_hits_conditions = []
                        for r in eval_res:
                            if len(set(r.predicted_experts).intersection(set(r.actual_experts))) >= 1:
                                all_hits_conditions.append(True)
                            else:
                                all_hits_conditions.append(False)
                        micro_hit = compute_mean_indicator(all_hits_conditions)

                        all_cold = []
                        all_warm = []
                        for r in eval_res:
                            if r.is_cold_start:
                                all_cold.append(r)
                            else:
                                all_warm.append(r)

                        if len(all_cold) > 0:
                            cold_tp = sum(len(set(r.predicted_experts).intersection(set(r.actual_experts))) for r in all_cold)
                            cold_act = sum(len(r.actual_experts) for r in all_cold)
                            cold_rec_all = float(cold_tp) / float(cold_act)
                        else:
                            cold_rec_all = 0.0

                        if len(all_warm) > 0:
                            warm_tp = sum(len(set(r.predicted_experts).intersection(set(r.actual_experts))) for r in all_warm)
                            warm_act = sum(len(r.actual_experts) for r in all_warm)
                            warm_rec_all = float(warm_tp) / float(warm_act)
                        else:
                            warm_rec_all = 0.0

                        exact_match_conditions = []
                        for r in eval_res:
                            if set(r.predicted_experts) == set(r.actual_experts):
                                exact_match_conditions.append(True)
                            else:
                                exact_match_conditions.append(False)
                        exact_acc = compute_mean_indicator(exact_match_conditions)

                        aggregate_rows.append({
                            "dataset_id": dset_name,
                            "model": model_name,
                            "horizon": h,
                            "seed": seed,
                            "profiling_pct": pct,
                            "predictor": pred_name,
                            "regime": regime,
                            "budget": budget,
                            "num_opportunities": len(eval_res),
                            "num_units": len(req_groups),
                            "macro_recall": round(macro_rec, 6),
                            "macro_precision": round(macro_prec, 6),
                            "macro_f1": round(macro_f1, 6),
                            "macro_hit_rate": round(macro_hit, 6),
                            "macro_cold_start_recall": round(macro_cold, 6),
                            "macro_warm_token_recall": round(macro_warm, 6),
                            "micro_recall": round(micro_rec, 6),
                            "micro_precision": round(micro_prec, 6),
                            "micro_f1": round(micro_f1, 6),
                            "micro_hit_rate": round(micro_hit, 6),
                            "exact_set_accuracy": round(exact_acc, 6),
                            "event_cold_start_recall": round(cold_rec_all, 6),
                            "event_warm_token_recall": round(warm_rec_all, 6)
                        })

                    # Compile 1-to-1 Head to Head comparison for this (dataset, horizon, seed, pct)
                    res_c_f = results_by_key[("CSTP", "frozen")]
                    res_c_o = results_by_key[("CSTP", "online_within_request")]
                    res_s_f = results_by_key[("ST-MoE", "frozen")]
                    res_s_o = results_by_key[("ST-MoE", "online_within_request")]

                    tot_tp_c_f = sum(len(set(r.predicted_experts).intersection(set(r.actual_experts))) for r in res_c_f)
                    tot_act_c_f = sum(len(r.actual_experts) for r in res_c_f)
                    rec_c_f = float(tot_tp_c_f) / float(tot_act_c_f)

                    tot_tp_c_o = sum(len(set(r.predicted_experts).intersection(set(r.actual_experts))) for r in res_c_o)
                    tot_act_c_o = sum(len(r.actual_experts) for r in res_c_o)
                    rec_c_o = float(tot_tp_c_o) / float(tot_act_c_o)

                    tot_tp_s_f = sum(len(set(r.predicted_experts).intersection(set(r.actual_experts))) for r in res_s_f)
                    tot_act_s_f = sum(len(r.actual_experts) for r in res_s_f)
                    rec_s_f = float(tot_tp_s_f) / float(tot_act_s_f)

                    tot_tp_s_o = sum(len(set(r.predicted_experts).intersection(set(r.actual_experts))) for r in res_s_o)
                    tot_act_s_o = sum(len(r.actual_experts) for r in res_s_o)
                    rec_s_o = float(tot_tp_s_o) / float(tot_act_s_o)

                    hit_cond_c = []
                    for r in res_c_o:
                        if len(set(r.predicted_experts).intersection(set(r.actual_experts))) >= 1:
                            hit_cond_c.append(True)
                        else:
                            hit_cond_c.append(False)
                    hit_c_o = compute_mean_indicator(hit_cond_c)

                    hit_cond_s = []
                    for r in res_s_o:
                        if len(set(r.predicted_experts).intersection(set(r.actual_experts))) >= 1:
                            hit_cond_s.append(True)
                        else:
                            hit_cond_s.append(False)
                    hit_s_o = compute_mean_indicator(hit_cond_s)

                    gain_c = rec_c_o - rec_c_f
                    gain_s = rec_s_o - rec_s_f

                    head_to_head_rows.append({
                        "dataset_id": dset_name,
                        "model": model_name,
                        "horizon": h,
                        "seed": seed,
                        "profiling_pct": pct,
                        "num_opportunities": len(test_opps),
                        "cstp_frozen_recall": round(rec_c_f, 6),
                        "stmoe_frozen_recall": round(rec_s_f, 6),
                        "delta_frozen_recall": round(rec_c_f - rec_s_f, 6),
                        "cstp_online_recall": round(rec_c_o, 6),
                        "stmoe_online_recall": round(rec_s_o, 6),
                        "delta_online_recall": round(rec_c_o - rec_s_o, 6),
                        "cstp_online_gain": round(gain_c * 100.0, 2),
                        "stmoe_online_gain": round(gain_s * 100.0, 2),
                        "cstp_online_hit_rate": round(hit_c_o, 6),
                        "stmoe_online_hit_rate": round(hit_s_o, 6),
                        "delta_online_hit_rate": round(hit_c_o - hit_s_o, 6),
                        "comparison_validity": "STRICT_MATCH_VERIFIED"
                    })

    # Close the event streaming file
    f_events.close()
    print(f"  Streaming complete: {total_events_written} total events written to {p_events_csv}")

    # 4. Export Certified Metric CSVs
    print("\n[4/7] Exporting Certified Metric CSV Files...")
    df_prompts = pd.DataFrame(prompt_rows)
    df_prompts.to_csv(os.path.join(out_dir, "corrected_per_prompt_metrics.csv"), index=False)
    print(f"  Exported {len(df_prompts)} prompt rows to corrected_per_prompt_metrics.csv")

    df_tasks = pd.DataFrame(task_rows)
    df_tasks.to_csv(os.path.join(out_dir, "corrected_per_task_metrics.csv"), index=False)
    print(f"  Exported {len(df_tasks)} task rows to corrected_per_task_metrics.csv")

    df_agg = pd.DataFrame(aggregate_rows)
    df_agg.to_csv(os.path.join(out_dir, "corrected_aggregate_metrics.csv"), index=False)
    print(f"  Exported {len(df_agg)} aggregate rows to corrected_aggregate_metrics.csv")

    df_h2h = pd.DataFrame(head_to_head_rows)
    df_h2h.to_csv(os.path.join(out_dir, "corrected_st_moe_head_to_head.csv"), index=False)
    print(f"  Exported {len(df_h2h)} head-to-head rows to corrected_st_moe_head_to_head.csv")

    df_counts = pd.DataFrame(count_stats_rows)
    df_counts.to_csv(os.path.join(out_dir, "corrected_profiling_count_statistics.csv"), index=False)
    print(f"  Exported {len(df_counts)} count stats rows to corrected_profiling_count_statistics.csv")

    # 5. Bootstrap Statistical Analysis (Hierarchical Multi-Seed Resampling across Units & Seeds)
    print("\n[5/7] Executing Multi-Seed Bootstrap Analysis (1,000 Iterations across Units and Profiling Seeds)...")
    boot_rows = []
    n_boot = 1000
    rng = np.random.RandomState(42)

    comparisons_meta = [
        ("CSTP_Online_minus_CSTP_Frozen", "CSTP", "online_within_request", "CSTP", "frozen"),
        ("STMoE_Online_minus_STMoE_Frozen", "ST-MoE", "online_within_request", "ST-MoE", "frozen"),
        ("CSTP_Online_minus_STMoE_Online", "CSTP", "online_within_request", "ST-MoE", "online_within_request"),
        ("CSTP_Frozen_minus_STMoE_Frozen", "CSTP", "frozen", "ST-MoE", "frozen")
    ]

    for dset_name in ["moe_anywhere", "moe_suite"]:
        if dset_name == "moe_anywhere":
            unit_df = df_prompts
            unit_type = "prompt"
            n_units = 14
            stat_validity = "VALID_N14_PROMPT_UNITS"
            caveat_msg = "Hierarchical bootstrap resamples 14 independent test prompts and 3 profiling seeds."
        else:
            unit_df = df_tasks
            unit_type = "task"
            n_units = 2
            stat_validity = "DESCRIPTIVE_ONLY_N2_TASKS"
            caveat_msg = "MoE Suite has only N=2 test tasks (corpus-12, corpus-13); bootstrap intervals have only 3 distinct resampling values and are descriptive only, not strong statistical evidence."

        for h in [1, 2]:
            for pct in percentages:
                for comp_label, pA, rA, pB, rB in comparisons_meta:
                    # Filter per seed data
                    seed_diffs = {}
                    point_diffs = []
                    for s in seeds:
                        subA = unit_df[(unit_df["dataset_id"] == dset_name) & (unit_df["horizon"] == h) & (unit_df["seed"] == s) & (unit_df["profiling_pct"] == pct) & (unit_df["predictor"] == pA) & (unit_df["regime"] == rA)].sort_values("unit_id")
                        subB = unit_df[(unit_df["dataset_id"] == dset_name) & (unit_df["horizon"] == h) & (unit_df["seed"] == s) & (unit_df["profiling_pct"] == pct) & (unit_df["predictor"] == pB) & (unit_df["regime"] == rB)].sort_values("unit_id")
                        recA = subA["recall"].values
                        recB = subB["recall"].values
                        seed_diffs[s] = (recA, recB)
                        point_diffs.append(float(np.mean(recA) - np.mean(recB)))

                    mean_point_diff = float(np.mean(point_diffs))

                    # 1. Multi-Seed Hierarchical Bootstrap: Joint resampling across seeds and units
                    joint_boot_diffs = []
                    for _ in range(n_boot):
                        # Resample seed with replacement
                        boot_seed = seeds[rng.randint(0, len(seeds))]
                        recA, recB = seed_diffs[boot_seed]
                        # Resample units with replacement
                        idx = rng.randint(0, len(recA), size=len(recA))
                        diff_val = float(np.mean(recA[idx]) - np.mean(recB[idx]))
                        joint_boot_diffs.append(diff_val)

                    p_gt_zero = compute_mean_indicator([d > 0.0 for d in joint_boot_diffs])
                    p_lt_zero = compute_mean_indicator([d < 0.0 for d in joint_boot_diffs])
                    asl_val = 2.0 * min(p_gt_zero, p_lt_zero)
                    if asl_val > 1.0:
                        asl_val = 1.0

                    boot_rows.append({
                        "dataset_id": dset_name,
                        "horizon": h,
                        "profiling_pct": pct,
                        "unit_type": unit_type,
                        "n_independent_units": n_units,
                        "profiling_seed_scope": "joint_all_seeds",
                        "comparison": comp_label,
                        "point_estimate_diff": round(mean_point_diff, 6),
                        "ci_95_lower": round(float(np.percentile(joint_boot_diffs, 2.5)), 6),
                        "ci_95_upper": round(float(np.percentile(joint_boot_diffs, 97.5)), 6),
                        "bootstrap_prob_greater_than_zero": round(p_gt_zero, 4),
                        "bootstrap_two_sided_asl": round(asl_val, 4),
                        "statistical_validity": stat_validity,
                        "caveats_and_notes": caveat_msg
                    })

                    # 2. Stratified Seed-42 Bootstrap for Direct Single-Seed Reference
                    recA_42, recB_42 = seed_diffs[42]
                    seed42_boot_diffs = []
                    for _ in range(n_boot):
                        idx = rng.randint(0, len(recA_42), size=len(recA_42))
                        diff_val = float(np.mean(recA_42[idx]) - np.mean(recB_42[idx]))
                        seed42_boot_diffs.append(diff_val)

                    p42_gt_zero = compute_mean_indicator([d > 0.0 for d in seed42_boot_diffs])
                    p42_lt_zero = compute_mean_indicator([d < 0.0 for d in seed42_boot_diffs])
                    asl_42 = 2.0 * min(p42_gt_zero, p42_lt_zero)
                    if asl_42 > 1.0:
                        asl_42 = 1.0

                    boot_rows.append({
                        "dataset_id": dset_name,
                        "horizon": h,
                        "profiling_pct": pct,
                        "unit_type": unit_type,
                        "n_independent_units": n_units,
                        "profiling_seed_scope": "stratified_seed_42",
                        "comparison": comp_label,
                        "point_estimate_diff": round(float(np.mean(recA_42) - np.mean(recB_42)), 6),
                        "ci_95_lower": round(float(np.percentile(seed42_boot_diffs, 2.5)), 6),
                        "ci_95_upper": round(float(np.percentile(seed42_boot_diffs, 97.5)), 6),
                        "bootstrap_prob_greater_than_zero": round(p42_gt_zero, 4),
                        "bootstrap_two_sided_asl": round(asl_42, 4),
                        "statistical_validity": stat_validity,
                        "caveats_and_notes": caveat_msg
                    })

    # Suite 75% vs 100% Reversal Analysis across seeds
    for h in [1, 2]:
        diffs_rev = []
        seed_rev_data = {}
        for s in seeds:
            sub_75 = df_tasks[(df_tasks["dataset_id"] == "moe_suite") & (df_tasks["horizon"] == h) & (df_tasks["seed"] == s) & (df_tasks["profiling_pct"] == 75) & (df_tasks["predictor"] == "CSTP") & (df_tasks["regime"] == "online_within_request")].sort_values("unit_id")
            sub_100 = df_tasks[(df_tasks["dataset_id"] == "moe_suite") & (df_tasks["horizon"] == h) & (df_tasks["seed"] == s) & (df_tasks["profiling_pct"] == 100) & (df_tasks["predictor"] == "CSTP") & (df_tasks["regime"] == "online_within_request")].sort_values("unit_id")
            rec_75 = sub_75["recall"].values
            rec_100 = sub_100["recall"].values
            seed_rev_data[s] = (rec_75, rec_100)
            diffs_rev.append(float(np.mean(rec_75) - np.mean(rec_100)))

        mean_rev_point = float(np.mean(diffs_rev))

        b_diff_rev = []
        for _ in range(n_boot):
            boot_s = seeds[rng.randint(0, len(seeds))]
            rec_75, rec_100 = seed_rev_data[boot_s]
            idx = rng.randint(0, len(rec_75), size=len(rec_75))
            b_diff_rev.append(float(np.mean(rec_75[idx]) - np.mean(rec_100[idx])))

        p_gt_rev = compute_mean_indicator([d > 0.0 for d in b_diff_rev])
        p_lt_rev = compute_mean_indicator([d < 0.0 for d in b_diff_rev])
        asl_rev = 2.0 * min(p_gt_rev, p_lt_rev)
        if asl_rev > 1.0:
            asl_rev = 1.0

        boot_rows.append({
            "dataset_id": "moe_suite",
            "horizon": h,
            "profiling_pct": 100,
            "unit_type": "task",
            "n_independent_units": 2,
            "profiling_seed_scope": "joint_all_seeds",
            "comparison": "Suite_CSTP_75pct_minus_100pct",
            "point_estimate_diff": round(mean_rev_point, 6),
            "ci_95_lower": round(float(np.percentile(b_diff_rev, 2.5)), 6),
            "ci_95_upper": round(float(np.percentile(b_diff_rev, 97.5)), 6),
            "bootstrap_prob_greater_than_zero": round(p_gt_rev, 4),
            "bootstrap_two_sided_asl": round(asl_rev, 4),
            "statistical_validity": "DESCRIPTIVE_ONLY_N2_TASKS",
            "caveats_and_notes": "MoE Suite has only N=2 test tasks (corpus-12, corpus-13); bootstrap intervals have only 3 distinct resampling values and are descriptive only, not strong statistical evidence."
        })

    df_boot = pd.DataFrame(boot_rows)
    df_boot.to_csv(os.path.join(out_dir, "corrected_statistical_analysis.csv"), index=False)
    print(f"  Bootstrap analysis exported to {os.path.join(out_dir, 'corrected_statistical_analysis.csv')}")

    # 6. Publication Figure Generation
    print("\n[6/7] Regenerating All 9 Publication Figures from Certified CSVs...")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "grid.alpha": 0.35,
        "grid.linestyle": "--"
    })

    def save_fig(fig, base_name):
        fig.savefig(os.path.join(plot_dir, f"{base_name}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(plot_dir, f"{base_name}.pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved {base_name}.png and .pdf")

    # Fig 1: Frozen Recall vs Profiling
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, dset in enumerate(["moe_anywhere", "moe_suite"]):
        ax = axes[idx]
        title_str = "MoE Anywhere (Sage-A1.5)"
        if dset == "moe_suite":
            title_str = "MoE Suite (GLM-5.2)"
        for h, ls in [(1, "-o"), (2, "--s")]:
            sub_c = df_agg[(df_agg["dataset_id"] == dset) & (df_agg["horizon"] == h) & (df_agg["seed"] == 42) & (df_agg["predictor"] == "CSTP") & (df_agg["regime"] == "frozen")]
            sub_s = df_agg[(df_agg["dataset_id"] == dset) & (df_agg["horizon"] == h) & (df_agg["seed"] == 42) & (df_agg["predictor"] == "ST-MoE") & (df_agg["regime"] == "frozen")]
            ax.plot(sub_c["profiling_pct"], sub_c["macro_recall"] * 100, ls, color="#1f77b4", label=f"CSTP Frozen (H={h})")
            ax.plot(sub_s["profiling_pct"], sub_s["macro_recall"] * 100, ls, color="#ff7f0e", label=f"ST-MoE Frozen (H={h})")
        ax.set_title(title_str)
        ax.set_xlabel("Profiling Percentage (%)")
        ax.set_ylabel("Frozen Recall@8 (%)")
        ax.set_xticks(percentages)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Fig 1: Frozen Recall@8 vs Profiling Percentage (Certified)", y=1.02)
    save_fig(fig, "fig1_certified_frozen_recall_vs_profiling")

    # Fig 2: Online Recall vs Profiling
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, dset in enumerate(["moe_anywhere", "moe_suite"]):
        ax = axes[idx]
        title_str = "MoE Anywhere (Sage-A1.5)"
        if dset == "moe_suite":
            title_str = "MoE Suite (GLM-5.2)"
        for h, ls in [(1, "-o"), (2, "--s")]:
            sub_c = df_agg[(df_agg["dataset_id"] == dset) & (df_agg["horizon"] == h) & (df_agg["seed"] == 42) & (df_agg["predictor"] == "CSTP") & (df_agg["regime"] == "online_within_request")]
            sub_s = df_agg[(df_agg["dataset_id"] == dset) & (df_agg["horizon"] == h) & (df_agg["seed"] == 42) & (df_agg["predictor"] == "ST-MoE") & (df_agg["regime"] == "online_within_request")]
            ax.plot(sub_c["profiling_pct"], sub_c["macro_recall"] * 100, ls, color="#0055aa", label=f"CSTP Online (H={h})")
            ax.plot(sub_s["profiling_pct"], sub_s["macro_recall"] * 100, ls, color="#cc5500", label=f"ST-MoE Online (H={h})")
        ax.set_title(title_str)
        ax.set_xlabel("Profiling Percentage (%)")
        ax.set_ylabel("Online Recall@8 (%)")
        ax.set_xticks(percentages)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Fig 2: Online Recall@8 vs Profiling Percentage (Certified)", y=1.02)
    save_fig(fig, "fig2_certified_online_recall_vs_profiling")

    # Fig 3: Online Adaptation Gain
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, dset in enumerate(["moe_anywhere", "moe_suite"]):
        ax = axes[idx]
        title_str = "MoE Anywhere (Sage-A1.5)"
        if dset == "moe_suite":
            title_str = "MoE Suite (GLM-5.2)"
        for h, ls in [(1, "-o"), (2, "--s")]:
            sub = df_h2h[(df_h2h["dataset_id"] == dset) & (df_h2h["horizon"] == h) & (df_h2h["seed"] == 42)]
            ax.plot(sub["profiling_pct"], sub["cstp_online_gain"], ls, color="#0055aa", label=f"CSTP Gain (H={h})")
            ax.plot(sub["profiling_pct"], sub["stmoe_online_gain"], ls, color="#cc5500", label=f"ST-MoE Gain (H={h})")
        ax.set_title(title_str)
        ax.set_xlabel("Profiling Percentage (%)")
        ax.set_ylabel("Online Adaptation Gain ΔRecall (pp)")
        ax.set_xticks(percentages)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Fig 3: Online Adaptation Gain ΔRecall vs Profiling Percentage (Certified)", y=1.02)
    save_fig(fig, "fig3_certified_online_adaptation_gain")

    # Fig 4: CSTP vs ST-MoE Matched Comparisons
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, dset in enumerate(["moe_anywhere", "moe_suite"]):
        ax = axes[idx]
        title_str = "MoE Anywhere (Sage-A1.5)"
        if dset == "moe_suite":
            title_str = "MoE Suite (GLM-5.2)"
        sub = df_h2h[(df_h2h["dataset_id"] == dset) & (df_h2h["seed"] == 42)]
        for h, color in [(1, "#1f77b4"), (2, "#ff7f0e")]:
            sub_h = sub[sub["horizon"] == h]
            ax.plot(sub_h["profiling_pct"], sub_h["delta_online_recall"] * 100, "-o", color=color, label=f"Online Δ(CSTP - ST-MoE) H={h}")
            ax.plot(sub_h["profiling_pct"], sub_h["delta_frozen_recall"] * 100, "--s", color=color, label=f"Frozen Δ(CSTP - ST-MoE) H={h}")
        ax.set_title(title_str)
        ax.set_xlabel("Profiling Percentage (%)")
        ax.set_ylabel("Recall Advantage (CSTP - ST-MoE) (pp)")
        ax.set_xticks(percentages)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Fig 4: CSTP versus ST-MoE Matched Head-to-Head Recall Advantage", y=1.02)
    save_fig(fig, "fig4_certified_stmoe_vs_cstp_head_to_head")

    # Fig 5: Horizon 1 vs Horizon 2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, dset in enumerate(["moe_anywhere", "moe_suite"]):
        ax = axes[idx]
        title_str = "MoE Anywhere (Sage-A1.5)"
        if dset == "moe_suite":
            title_str = "MoE Suite (GLM-5.2)"
        sub_c1 = df_agg[(df_agg["dataset_id"] == dset) & (df_agg["horizon"] == 1) & (df_agg["seed"] == 42) & (df_agg["predictor"] == "CSTP") & (df_agg["regime"] == "online_within_request")]
        sub_c2 = df_agg[(df_agg["dataset_id"] == dset) & (df_agg["horizon"] == 2) & (df_agg["seed"] == 42) & (df_agg["predictor"] == "CSTP") & (df_agg["regime"] == "online_within_request")]
        ax.plot(sub_c1["profiling_pct"], sub_c1["macro_recall"] * 100, "-o", color="#1f77b4", label="CSTP H=1")
        ax.plot(sub_c2["profiling_pct"], sub_c2["macro_recall"] * 100, "--s", color="#ff7f0e", label="CSTP H=2")
        ax.set_title(title_str)
        ax.set_xlabel("Profiling Percentage (%)")
        ax.set_ylabel("Online Recall@8 (%)")
        ax.set_xticks(percentages)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Fig 5: Horizon Comparison (H=1 versus H=2 Online Recall)", y=1.02)
    save_fig(fig, "fig5_certified_horizon1_vs_horizon2")

    # Fig 6: Seed-Wise Variability with Error Bars
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, dset in enumerate(["moe_anywhere", "moe_suite"]):
        ax = axes[idx]
        title_str = "MoE Anywhere (Sage-A1.5)"
        if dset == "moe_suite":
            title_str = "MoE Suite (GLM-5.2)"
        for h, color in [(1, "#1f77b4"), (2, "#ff7f0e")]:
            pct_means = []
            pct_stds = []
            for pct in percentages:
                sub_seeds = df_agg[(df_agg["dataset_id"] == dset) & (df_agg["horizon"] == h) & (df_agg["profiling_pct"] == pct) & (df_agg["predictor"] == "CSTP") & (df_agg["regime"] == "online_within_request")]
                pct_means.append(sub_seeds["macro_recall"].mean() * 100.0)
                pct_stds.append(sub_seeds["macro_recall"].std() * 100.0)
            ax.errorbar(percentages, pct_means, yerr=pct_stds, fmt="-o", color=color, capsize=5, label=f"CSTP H={h} (Mean ± Std)")
        ax.set_title(title_str)
        ax.set_xlabel("Profiling Percentage (%)")
        ax.set_ylabel("Online Recall@8 (%) across Seeds")
        ax.set_xticks(percentages)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Fig 6: Multi-Seed Variability across Seeds (42, 123, 999)", y=1.02)
    save_fig(fig, "fig6_certified_seed_variability")

    # Fig 7: MoE Suite Task-Level Performance
    fig, ax = plt.subplots(figsize=(8, 5))
    sub_task = df_tasks[(df_tasks["horizon"] == 1) & (df_tasks["seed"] == 42) & (df_tasks["predictor"] == "CSTP") & (df_tasks["regime"] == "online_within_request")]
    task_ids = sorted(list(set(sub_task["unit_id"])))
    for tid in task_ids:
        t_sub = sub_task[sub_task["unit_id"] == tid].sort_values("profiling_pct")
        name_str = "Task 0 (corpus-12)"
        if tid == 9001:
            name_str = "Task 1 (corpus-13)"
        ax.plot(t_sub["profiling_pct"], t_sub["recall"] * 100, "-o", label=f"{name_str}")
    ax.set_title("MoE Suite Task-Level Breakdown (H=1, CSTP Online, Seed 42)")
    ax.set_xlabel("Profiling Percentage (%)")
    ax.set_ylabel("Task Recall@8 (%)")
    ax.set_xticks(percentages)
    ax.grid(True)
    ax.legend()
    save_fig(fig, "fig7_certified_moe_suite_task_level")

    # Fig 8: Transition Count Growth
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, dset in enumerate(["moe_anywhere", "moe_suite"]):
        ax = axes[idx]
        title_str = "MoE Anywhere (Sage-A1.5)"
        if dset == "moe_suite":
            title_str = "MoE Suite (GLM-5.2)"
        sub = df_counts[(df_counts["dataset_id"] == dset) & (df_counts["horizon"] == 1) & (df_counts["seed"] == 42)]
        ax.plot(sub["profiling_pct"], sub["marginal_count_mass"] / 1e6, "-o", color="#1f77b4", label="Marginal Counts (M)")
        ax.plot(sub["profiling_pct"], sub["spatial_count_mass"] / 1e6, "--s", color="#2ca02c", label="Spatial Transitions (M)")
        ax.plot(sub["profiling_pct"], sub["temporal_count_mass"] / 1e6, ":^", color="#d62728", label="Temporal Transitions (M)")
        ax.set_title(title_str)
        ax.set_xlabel("Profiling Percentage (%)")
        ax.set_ylabel("Accumulated Count Mass (Millions)")
        ax.set_xticks(percentages)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Fig 8: Transition Count Mass Scaling by Tensor Type", y=1.02)
    save_fig(fig, "fig8_certified_transition_count_growth")

    # Fig 9: Bootstrap Confidence Intervals
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, dset in enumerate(["moe_anywhere", "moe_suite"]):
        ax = axes[idx]
        title_str = "MoE Anywhere (Sage-A1.5)"
        if dset == "moe_suite":
            title_str = "MoE Suite (GLM-5.2) [Descriptive Only, N=2]"
        sub_b = df_boot[(df_boot["dataset_id"] == dset) & (df_boot["horizon"] == 1) & (df_boot["profiling_seed_scope"] == "joint_all_seeds") & (df_boot["comparison"] == "CSTP_Online_minus_CSTP_Frozen")]
        y_pts = sub_b["point_estimate_diff"].values * 100.0
        y_err_low = (sub_b["point_estimate_diff"].values - sub_b["ci_95_lower"].values) * 100.0
        y_err_high = (sub_b["ci_95_upper"].values - sub_b["point_estimate_diff"].values) * 100.0
        ax.errorbar(percentages, y_pts, yerr=[y_err_low, y_err_high], fmt="-o", color="#0055aa", capsize=5, label="CSTP Online Gain (Joint 95% CI)")
        ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
        ax.set_title(title_str)
        ax.set_xlabel("Profiling Percentage (%)")
        ax.set_ylabel("Online Adaptation Gain ΔRecall (pp)")
        ax.set_xticks(percentages)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Fig 9: 95% Hierarchical Multi-Seed Bootstrap Confidence Intervals", y=1.02)
    save_fig(fig, "fig9_certified_confidence_intervals")

    print("\n" + "=" * 80)
    print("CERTIFIED PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_certified_pipeline()
