"""
Phase 1A: CSTP Profiling Percentage Sensitivity Experiment Runner.

Investigates how the quantity of offline profiling data (25%, 50%, 75%, 100%)
governs frozen prediction accuracy and online adaptation efficacy.

Strict Constraints:
1. Workspace: All outputs saved to results/phase1a_cstp_profiling (outside mohman/).
2. Predictor: Preserves CSTP original mathematical formulation strictly.
3. Isolation: Fresh predictor instances per experiment; zero state accumulation.
4. Determinism: Strictly nested subsets (25% subset ⊂ 50% subset ⊂ 75% subset ⊂ 100% subset).
5. Robustness: Primary seed 42, robustness seeds 42, 123, 999.
6. Fair Comparison: ST-MoE evaluated on identical training subsets and test opportunities.
7. Regimes: Primary regime is 'online_within_request'; secondary baseline is 'frozen'.
"""

import os
import sys
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
workspace_root = os.path.dirname(sim_root) if os.path.basename(sim_root) == "LLMSimulator" else sim_root
if os.path.basename(sim_root) != "LLMSimulator":
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


def get_nested_subsets(items: List[Any], fractions: List[float], seed: int) -> Dict[int, List[Any]]:
    """
    Returns deterministically ordered nested subsets:
    items_25% ⊂ items_50% ⊂ items_75% ⊂ items_100%
    """
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    subsets = {}
    n_total = len(shuffled)
    for frac in fractions:
        pct = int(round(frac * 100))
        n_take = max(1, int(round(n_total * frac)))
        subsets[pct] = shuffled[:n_take]
    return subsets


def extract_moe_anywhere_test_opportunities(
    adapter: MoeAnywhereDatasetAdapter,
    horizon: int,
    max_tokens_per_prompt: int = 30
) -> List[PredictionOpportunity]:
    """Extracts strictly causal test opportunities for MoE Anywhere (Sage-A1.5)."""
    test_prompts = adapter.get_splits()["test"]
    opps = []
    for pid in test_prompts:
        p_tokens = adapter.prompt_tokens.get(pid, [])[:max_tokens_per_prompt]
        for tpos in p_tokens:
            for target_stage in range(adapter.num_moe_stages):
                source_stage = target_stage - horizon
                if source_stage >= 0:
                    src_key = (pid, tpos, source_stage)
                    src_experts = adapter.routes.get(src_key, ([], None))[0]
                else:
                    src_experts = []

                tgt_key = (pid, tpos, target_stage)
                tgt_experts = adapter.routes.get(tgt_key, ([], None))[0]

                opps.append(PredictionOpportunity(
                    request_id=pid,
                    token_id=tpos,
                    moe_stage_id=target_stage,
                    source_experts=src_experts,
                    actual_experts=tgt_experts,
                    source_stage_id=source_stage if source_stage >= 0 else None,
                    prediction_horizon=horizon
                ))
    return opps


def extract_moe_suite_test_opportunities(
    adapter: MoeSuiteDatasetAdapter,
    horizon: int,
    max_tokens_per_task: int = 50
) -> List[PredictionOpportunity]:
    """Extracts strictly causal test opportunities for MoE Suite (GLM-5.2)."""
    test_tasks = adapter.splits["test"]
    opps = []
    for task_idx, task_id in enumerate(test_tasks):
        arr = adapter.task_routes[task_id]
        num_eval_tokens = min(len(arr), max_tokens_per_task)
        req_id = 9000 + task_idx
        for tpos in range(num_eval_tokens):
            for target_stage in range(adapter.num_moe_stages):
                source_stage = target_stage - horizon
                if source_stage >= 0:
                    src_experts = [int(e) for e in arr[tpos, source_stage].tolist()]
                else:
                    src_experts = []
                tgt_experts = [int(e) for e in arr[tpos, target_stage].tolist()]

                opps.append(PredictionOpportunity(
                    request_id=req_id,
                    token_id=tpos,
                    moe_stage_id=target_stage,
                    source_experts=src_experts,
                    actual_experts=tgt_experts,
                    source_stage_id=source_stage if source_stage >= 0 else None,
                    prediction_horizon=horizon
                ))
    return opps


def extract_anywhere_prompt_records(adapter: MoeAnywhereDatasetAdapter, pid: int) -> List[PromptRecord]:
    """Extracts all training routing records for one prompt in MoE Anywhere."""
    records = []
    p_tokens = adapter.prompt_tokens.get(pid, [])
    for tpos in p_tokens:
        for stage in range(adapter.num_moe_stages):
            k = (pid, tpos, stage)
            if k in adapter.routes:
                exps, _ = adapter.routes[k]
                for e in exps[:adapter.top_k]:
                    records.append(PromptRecord(request_id=pid, token_id=tpos, moe_stage_id=stage, expert_id=e))
    return records


def extract_suite_task_records(adapter: MoeSuiteDatasetAdapter, task_id: str, req_id: int):
    """Generates all training routing records for one task in MoE Suite."""
    arr = adapter.task_routes[task_id]
    num_stages = adapter.num_moe_stages
    for tpos in range(len(arr)):
        for stage in range(num_stages):
            for e in arr[tpos, stage]:
                yield PromptRecord(request_id=req_id, token_id=tpos, moe_stage_id=stage, expert_id=int(e))


def compute_stmoe_cct_from_spatial(spatial_counts: np.ndarray, num_routed_experts: int, top_k: int) -> Dict[int, Dict[int, Dict[int, int]]]:
    """
    Constructs ST-MoE CCT strictly adhering to Algorithm 1 from the exact
    spatial co-activation counts of the profiling subset.
    """
    num_transitions = spatial_counts.shape[0]
    cct: Dict[int, Dict[int, Dict[int, int]]] = {}
    for s in range(num_transitions):
        cct[s] = {}
        for u in range(num_routed_experts):
            row = spatial_counts[s, u, :]
            dests = sorted(range(num_routed_experts), key=lambda idx: (-row[idx], idx))
            cct[s][u] = {}
            stored = 0
            for v in dests:
                if row[v] > 0 and stored < top_k:
                    cct[s][u][v] = 2  # Binary 10: initial weakly preferred
                    stored += 1
    return cct


def run_phase1a_campaign():
    out_dir = os.path.join(workspace_root, "results", "phase1a_cstp_profiling")
    csv_dir = os.path.join(out_dir, "csv")
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print("=" * 80)
    print("PHASE 1A: CSTP PROFILING PERCENTAGE SENSITIVITY CAMPAIGN")
    print(f"Output directory: {out_dir}")
    print("=" * 80)

    # 1. Load Adapters
    print("\n[1/6] Loading Dataset Adapters...")
    p_anywhere = os.path.join(sim_root, "data", "downloads", "moe_anywhere", "e1-sage-a1.5.parquet")
    adapter_anywhere = MoeAnywhereDatasetAdapter(p_anywhere)
    adapter_anywhere.load()

    d_suite = os.path.join(sim_root, "data", "downloads", "moe_suite")
    adapter_suite = MoeSuiteDatasetAdapter(d_suite)
    adapter_suite.load()

    # Verify Splits
    train_prompts_anywhere = sorted(list(adapter_anywhere.get_splits()["train"]))
    test_prompts_anywhere = sorted(list(adapter_anywhere.get_splits()["test"]))
    train_tasks_suite = sorted(list(adapter_suite.splits["train"]))
    test_tasks_suite = sorted(list(adapter_suite.splits["test"]))

    print(f"MoE Anywhere: {len(train_prompts_anywhere)} train prompts, {len(test_prompts_anywhere)} test prompts.")
    print(f"MoE Suite: {len(train_tasks_suite)} train tasks, {len(test_tasks_suite)} test tasks.")

    assert len(train_prompts_anywhere) == 52, f"Expected 52 train prompts, found {len(train_prompts_anywhere)}"
    assert len(test_prompts_anywhere) == 14, f"Expected 14 test prompts, found {len(test_prompts_anywhere)}"
    assert len(train_tasks_suite) == 8, f"Expected 8 train tasks, found {len(train_tasks_suite)}"
    assert len(test_tasks_suite) == 2, f"Expected 2 test tasks, found {len(test_tasks_suite)}"

    # Check zero train/test overlap
    assert len(set(train_prompts_anywhere).intersection(set(test_prompts_anywhere))) == 0
    assert len(set(train_tasks_suite).intersection(set(test_tasks_suite))) == 0

    fractions = [0.25, 0.50, 0.75, 1.00]
    percentages = [25, 50, 75, 100]
    seeds = [42, 123, 999]
    horizons = [1, 2]
    budget = 8

    # Pre-cache per-task / per-prompt count contributions to guarantee speed and numerical determinism
    print("\n[2/6] Pre-indexing Training Units for Fast Bitwise Deterministic Profiling...")

    # Anywhere records per prompt
    anywhere_prompt_data = {}
    for pid in train_prompts_anywhere:
        recs = extract_anywhere_prompt_records(adapter_anywhere, pid)
        n_toks = len(adapter_anywhere.prompt_tokens.get(pid, []))
        anywhere_prompt_data[pid] = (recs, n_toks)

    # Suite tasks profiling counts (per task, per horizon)
    # We fit each task individually once using CSTPPredictor.fit_profile to strictly use the reference logic
    suite_task_profiles = {}  # (task_id, horizon) -> (marginal_counts, spatial_counts, temporal_counts, tokens_count)
    print("  Profiling individual MoE Suite training tasks (8 tasks)...")
    t0_suite = time.time()
    for t_idx, task_id in enumerate(train_tasks_suite):
        t_start = time.time()
        arr = adapter_suite.task_routes[task_id]
        n_toks = len(arr)
        for h in horizons:
            p_task = CSTPPredictor(num_moe_stages=75, num_routed_experts=256, top_k=8, prediction_horizon=h)
            p_task.fit_profile(extract_suite_task_records(adapter_suite, task_id, req_id=1000 + t_idx))
            suite_task_profiles[(task_id, h)] = (
                p_task.stage_marginal_counts.copy(),
                p_task.spatial_counts.copy(),
                p_task.temporal_counts.copy(),
                n_toks
            )
        print(f"    Task {task_id} ({n_toks} tokens) profiled for H=1,2 in {time.time() - t_start:.2f}s")
    print(f"  All MoE Suite training tasks indexed in {time.time() - t0_suite:.2f}s.")

    # Manifest records
    manifest_rows = []
    count_stats_rows = []
    detailed_results_rows = []

    print("\n[3/6] Generating Deterministic Nested Subsets and Manifests...")
    subset_definitions = {}  # (dataset_id, seed, pct) -> list of prompt/task IDs
    for seed in seeds:
        # Anywhere
        subs_anywhere = get_nested_subsets(train_prompts_anywhere, fractions, seed)
        for pct in percentages:
            p_list = subs_anywhere[pct]
            subset_definitions[("moe_anywhere", seed, pct)] = p_list
            manifest_rows.append({
                "dataset_id": "moe_anywhere",
                "seed": seed,
                "profiling_pct": pct,
                "num_selected_units": len(p_list),
                "total_available_units": len(train_prompts_anywhere),
                "selected_unit_ids": json.dumps(p_list)
            })

        # Suite
        subs_suite = get_nested_subsets(train_tasks_suite, fractions, seed)
        for pct in percentages:
            t_list = subs_suite[pct]
            subset_definitions[("moe_suite", seed, pct)] = t_list
            manifest_rows.append({
                "dataset_id": "moe_suite",
                "seed": seed,
                "profiling_pct": pct,
                "num_selected_units": len(t_list),
                "total_available_units": len(train_tasks_suite),
                "selected_unit_ids": json.dumps(t_list)
            })

    # Verify Nesting Property
    for dset in ["moe_anywhere", "moe_suite"]:
        for seed in seeds:
            s25 = set(subset_definitions[(dset, seed, 25)])
            s50 = set(subset_definitions[(dset, seed, 50)])
            s75 = set(subset_definitions[(dset, seed, 75)])
            s100 = set(subset_definitions[(dset, seed, 100)])
            assert s25.issubset(s50), f"Nesting violation {dset} seed {seed}: 25% not subset of 50%"
            assert s50.issubset(s75), f"Nesting violation {dset} seed {seed}: 50% not subset of 75%"
            assert s75.issubset(s100), f"Nesting violation {dset} seed {seed}: 75% not subset of 100%"
    print("  Nesting property verified: 25% ⊂ 50% ⊂ 75% ⊂ 100% across all seeds and datasets.")

    # 4. Run Evaluation Sweep
    print("\n[4/6] Executing Profiling Sensitivity Experiments...")

    datasets_meta = [
        ("moe_anywhere", adapter_anywhere, 40, 256, 8, 16800),
        ("moe_suite", adapter_suite, 75, 256, 8, 7500)
    ]

    for dset_name, adapter, num_stages, num_experts, top_k, expected_opps in datasets_meta:
        print(f"\n=======================================================")
        print(f"Dataset: {dset_name} ({num_stages} stages, {num_experts} experts, top-{top_k})")
        print(f"=======================================================")

        harness = EvaluationHarness(num_moe_stages=num_stages)

        for h in horizons:
            print(f"\n--- Horizon H={h} ---")

            # Extract test opportunities
            if dset_name == "moe_anywhere":
                test_opps = extract_moe_anywhere_test_opportunities(adapter, horizon=h, max_tokens_per_prompt=30)
            else:
                test_opps = extract_moe_suite_test_opportunities(adapter, horizon=h, max_tokens_per_task=50)

            print(f"  Test Opportunities extracted: {len(test_opps)} (Expected: {expected_opps})")
            assert len(test_opps) == expected_opps, f"Opportunity mismatch: {len(test_opps)} vs {expected_opps}"

            for seed in seeds:
                is_primary_seed = (seed == 42)
                for pct in percentages:
                    selected_units = subset_definitions[(dset_name, seed, pct)]

                    # Build profile
                    if dset_name == "moe_anywhere":
                        # Collect records for selected prompts
                        subset_records = []
                        total_tokens = 0
                        for pid in selected_units:
                            recs, n_toks = anywhere_prompt_data[pid]
                            subset_records.extend(recs)
                            total_tokens += n_toks

                        # Fresh CSTP predictor
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
                        # MoE Suite: Sum the pre-indexed task count profiles
                        total_tokens = 0
                        marg_counts = np.zeros((num_stages, num_experts), dtype=np.float64)
                        num_transitions = max(1, num_stages - h)
                        spat_counts = np.zeros((num_transitions, num_experts, num_experts), dtype=np.float64)
                        temp_counts = np.zeros((num_stages, num_experts, num_experts), dtype=np.float64)

                        for tid in selected_units:
                            m_c, s_c, t_c, n_toks = suite_task_profiles[(tid, h)]
                            marg_counts += m_c
                            spat_counts += s_c
                            temp_counts += t_c
                            total_tokens += n_toks

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
                        fresh_cstp.stage_marginal_counts = marg_counts.copy()
                        fresh_cstp.spatial_counts = spat_counts.copy()
                        fresh_cstp.temporal_counts = temp_counts.copy()

                    # Record Count Statistics
                    total_marg_mass = float(np.sum(marg_counts))
                    total_spat_mass = float(np.sum(spat_counts))
                    total_temp_mass = float(np.sum(temp_counts))
                    nonzero_marg_cells = int(np.count_nonzero(marg_counts))
                    nonzero_spat_cells = int(np.count_nonzero(spat_counts))
                    nonzero_temp_cells = int(np.count_nonzero(temp_counts))

                    count_stats_rows.append({
                        "dataset_id": dset_name,
                        "horizon": h,
                        "seed": seed,
                        "profiling_pct": pct,
                        "units_count": len(selected_units),
                        "tokens_count": total_tokens,
                        "marginal_count_mass": total_marg_mass,
                        "spatial_count_mass": total_spat_mass,
                        "temporal_count_mass": total_temp_mass,
                        "nonzero_marginal_cells": nonzero_marg_cells,
                        "nonzero_spatial_cells": nonzero_spat_cells,
                        "nonzero_temporal_cells": nonzero_temp_cells
                    })

                    # Construct ST-MoE Predictor using identical spatial counts
                    cct = compute_stmoe_cct_from_spatial(spat_counts, num_experts, top_k)

                    # Evaluate Regimes: Frozen and Online-within-request
                    # For CSTP:
                    for regime in ["frozen", "online_within_request"]:
                        cstp_runner = CSTPPredictor(
                            num_moe_stages=num_stages,
                            num_routed_experts=num_experts,
                            top_k=top_k,
                            alpha=0.5,
                            smoothing_lambda=1.0,
                            prior_epsilon=0.1,
                            online_adaptation=(regime == "online_within_request"),
                            prediction_horizon=h
                        )
                        cstp_runner.stage_marginal_counts = marg_counts.copy()
                        cstp_runner.spatial_counts = spat_counts.copy()
                        cstp_runner.temporal_counts = temp_counts.copy()

                        res_cstp = harness.evaluate_predictor(
                            predictor_instance=cstp_runner,
                            predictor_name="CSTP",
                            test_opportunities=test_opps,
                            budgets=[budget],
                            evaluation_mode="online_within_request"
                        )
                        sm_cstp = harness.compute_summary_metrics(res_cstp)

                        detailed_results_rows.append({
                            "dataset_id": dset_name,
                            "model": "Sage-A1.5" if dset_name == "moe_anywhere" else "GLM-5.2",
                            "horizon": h,
                            "seed": seed,
                            "profiling_pct": pct,
                            "predictor": "CSTP",
                            "regime": regime,
                            "budget": budget,
                            "num_opportunities": len(test_opps),
                            "units_count": len(selected_units),
                            "tokens_count": total_tokens,
                            "macro_recall": sm_cstp["macro_recall"],
                            "macro_precision": sm_cstp["macro_precision"],
                            "macro_f1": sm_cstp["macro_f1"],
                            "micro_recall": sm_cstp["micro_recall"],
                            "micro_precision": sm_cstp["micro_precision"],
                            "micro_f1": sm_cstp["micro_f1"],
                            "hit_rate": sm_cstp["hit_rate"],
                            "exact_set_accuracy": sm_cstp["exact_set_accuracy"],
                            "cold_start_recall": sm_cstp["cold_start"]["recall"],
                            "cold_start_precision": sm_cstp["cold_start"]["precision"],
                            "warm_token_recall": sm_cstp["steady_state"]["recall"],
                            "warm_token_precision": sm_cstp["steady_state"]["precision"]
                        })

                    # For ST-MoE:
                    for regime in ["frozen", "online_within_request"]:
                        st_runner = STMoEPaperPredictor(
                            num_moe_stages=num_stages,
                            num_routed_experts=num_experts,
                            top_k=top_k,
                            cct_capacity=top_k,
                            confidence_threshold=2,
                            online_adaptation=(regime == "online_within_request"),
                            prediction_horizon=h,
                            scoring_mode="confidence_additive"
                        )
                        st_runner.cct = copy.deepcopy(cct)

                        res_st = harness.evaluate_predictor(
                            predictor_instance=st_runner,
                            predictor_name="ST-MoE",
                            test_opportunities=test_opps,
                            budgets=[budget],
                            evaluation_mode="online_within_request"
                        )
                        sm_st = harness.compute_summary_metrics(res_st)

                        detailed_results_rows.append({
                            "dataset_id": dset_name,
                            "model": "Sage-A1.5" if dset_name == "moe_anywhere" else "GLM-5.2",
                            "horizon": h,
                            "seed": seed,
                            "profiling_pct": pct,
                            "predictor": "ST-MoE",
                            "regime": regime,
                            "budget": budget,
                            "num_opportunities": len(test_opps),
                            "units_count": len(selected_units),
                            "tokens_count": total_tokens,
                            "macro_recall": sm_st["macro_recall"],
                            "macro_precision": sm_st["macro_precision"],
                            "macro_f1": sm_st["macro_f1"],
                            "micro_recall": sm_st["micro_recall"],
                            "micro_precision": sm_st["micro_precision"],
                            "micro_f1": sm_st["micro_f1"],
                            "hit_rate": sm_st["hit_rate"],
                            "exact_set_accuracy": sm_st["exact_set_accuracy"],
                            "cold_start_recall": sm_st["cold_start"]["recall"],
                            "cold_start_precision": sm_st["cold_start"]["precision"],
                            "warm_token_recall": sm_st["steady_state"]["recall"],
                            "warm_token_precision": sm_st["steady_state"]["precision"]
                        })

                    # Print progress for primary seed
                    if is_primary_seed:
                        cstp_f = [r for r in detailed_results_rows if r["dataset_id"] == dset_name and r["horizon"] == h and r["seed"] == seed and r["profiling_pct"] == pct and r["predictor"] == "CSTP" and r["regime"] == "frozen"][0]
                        cstp_o = [r for r in detailed_results_rows if r["dataset_id"] == dset_name and r["horizon"] == h and r["seed"] == seed and r["profiling_pct"] == pct and r["predictor"] == "CSTP" and r["regime"] == "online_within_request"][0]
                        st_f = [r for r in detailed_results_rows if r["dataset_id"] == dset_name and r["horizon"] == h and r["seed"] == seed and r["profiling_pct"] == pct and r["predictor"] == "ST-MoE" and r["regime"] == "frozen"][0]
                        st_o = [r for r in detailed_results_rows if r["dataset_id"] == dset_name and r["horizon"] == h and r["seed"] == seed and r["profiling_pct"] == pct and r["predictor"] == "ST-MoE" and r["regime"] == "online_within_request"][0]
                        gain_cstp = (cstp_o["macro_recall"] - cstp_f["macro_recall"]) * 100.0
                        gain_st = (st_o["macro_recall"] - st_f["macro_recall"]) * 100.0
                        print(f"  [Seed {seed}] {pct}% profiling ({len(selected_units)} units): "
                              f"CSTP F={cstp_f['macro_recall']:.4f} O={cstp_o['macro_recall']:.4f} (Gain={gain_cstp:+.2f}%) | "
                              f"ST-MoE F={st_f['macro_recall']:.4f} O={st_o['macro_recall']:.4f} (Gain={gain_st:+.2f}%)")

    # 5. Export All Required CSV Files
    print("\n[5/6] Exporting Required CSV Artifacts...")

    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(os.path.join(csv_dir, "profiling_subset_manifest.csv"), index=False)

    df_counts = pd.DataFrame(count_stats_rows)
    df_counts.to_csv(os.path.join(csv_dir, "profiling_count_statistics.csv"), index=False)

    df_detailed = pd.DataFrame(detailed_results_rows)
    df_detailed.to_csv(os.path.join(csv_dir, "profiling_percentage_detailed_results.csv"), index=False)

    # Profiling percentage summary (Primary seed 42)
    df_primary = df_detailed[df_detailed["seed"] == 42].copy()
    df_primary.to_csv(os.path.join(csv_dir, "profiling_percentage_summary.csv"), index=False)

    # Frozen vs Online Table
    frozen_vs_online_rows = []
    configs = df_primary[["dataset_id", "model", "horizon", "profiling_pct", "predictor", "units_count", "tokens_count"]].drop_duplicates().to_dict("records")
    for cfg in configs:
        sub = df_primary[
            (df_primary["dataset_id"] == cfg["dataset_id"]) &
            (df_primary["horizon"] == cfg["horizon"]) &
            (df_primary["profiling_pct"] == cfg["profiling_pct"]) &
            (df_primary["predictor"] == cfg["predictor"])
        ]
        f_row = sub[sub["regime"] == "frozen"].iloc[0]
        o_row = sub[sub["regime"] == "online_within_request"].iloc[0]

        delta_recall = o_row["macro_recall"] - f_row["macro_recall"]
        pct_gain = (delta_recall / f_row["macro_recall"] * 100.0) if f_row["macro_recall"] > 0 else 0.0

        frozen_vs_online_rows.append({
            "dataset_id": cfg["dataset_id"],
            "model": cfg["model"],
            "horizon": cfg["horizon"],
            "profiling_pct": cfg["profiling_pct"],
            "predictor": cfg["predictor"],
            "units_count": cfg["units_count"],
            "tokens_count": cfg["tokens_count"],
            "frozen_recall": f_row["macro_recall"],
            "online_recall": o_row["macro_recall"],
            "delta_recall": round(delta_recall, 6),
            "pct_gain": round(pct_gain, 2),
            "frozen_hit_rate": f_row["hit_rate"],
            "online_hit_rate": o_row["hit_rate"],
            "frozen_cold_start_recall": f_row["cold_start_recall"],
            "online_cold_start_recall": o_row["cold_start_recall"],
            "frozen_warm_recall": f_row["warm_token_recall"],
            "online_warm_recall": o_row["warm_token_recall"]
        })
    df_fvo = pd.DataFrame(frozen_vs_online_rows)
    df_fvo.to_csv(os.path.join(csv_dir, "frozen_vs_online.csv"), index=False)

    # Profiling Robustness Seeds Table (Aggregates across seeds 42, 123, 999)
    robustness_rows = []
    group_cols = ["dataset_id", "model", "horizon", "profiling_pct", "predictor", "regime"]
    for keys, grp in df_detailed.groupby(group_cols):
        robustness_rows.append({
            "dataset_id": keys[0],
            "model": keys[1],
            "horizon": keys[2],
            "profiling_pct": keys[3],
            "predictor": keys[4],
            "regime": keys[5],
            "num_seeds": len(grp),
            "macro_recall_mean": round(float(grp["macro_recall"].mean()), 6),
            "macro_recall_std": round(float(grp["macro_recall"].std()), 6),
            "hit_rate_mean": round(float(grp["hit_rate"].mean()), 6),
            "hit_rate_std": round(float(grp["hit_rate"].std()), 6),
            "cold_start_recall_mean": round(float(grp["cold_start_recall"].mean()), 6),
            "cold_start_recall_std": round(float(grp["cold_start_recall"].std()), 6),
            "warm_recall_mean": round(float(grp["warm_token_recall"].mean()), 6),
            "warm_recall_std": round(float(grp["warm_token_recall"].std()), 6),
        })
    df_robust = pd.DataFrame(robustness_rows)
    df_robust.to_csv(os.path.join(csv_dir, "profiling_robustness_seeds.csv"), index=False)

    # ST-MoE vs CSTP Profiling Comparison Table
    st_vs_cstp_rows = []
    for (dset, horiz, pct), grp in df_primary.groupby(["dataset_id", "horizon", "profiling_pct"]):
        c_f = grp[(grp["predictor"] == "CSTP") & (grp["regime"] == "frozen")].iloc[0]
        c_o = grp[(grp["predictor"] == "CSTP") & (grp["regime"] == "online_within_request")].iloc[0]
        s_f = grp[(grp["predictor"] == "ST-MoE") & (grp["regime"] == "frozen")].iloc[0]
        s_o = grp[(grp["predictor"] == "ST-MoE") & (grp["regime"] == "online_within_request")].iloc[0]

        st_vs_cstp_rows.append({
            "dataset_id": dset,
            "horizon": horiz,
            "profiling_pct": pct,
            "cstp_frozen_recall": c_f["macro_recall"],
            "stmoe_frozen_recall": s_f["macro_recall"],
            "delta_frozen_recall": round(c_f["macro_recall"] - s_f["macro_recall"], 6),
            "cstp_online_recall": c_o["macro_recall"],
            "stmoe_online_recall": s_o["macro_recall"],
            "delta_online_recall": round(c_o["macro_recall"] - s_o["macro_recall"], 6),
            "cstp_online_gain": round((c_o["macro_recall"] - c_f["macro_recall"]) * 100.0, 2),
            "stmoe_online_gain": round((s_o["macro_recall"] - s_f["macro_recall"]) * 100.0, 2),
            "cstp_online_hit_rate": c_o["hit_rate"],
            "stmoe_online_hit_rate": s_o["hit_rate"],
            "delta_online_hit_rate": round(c_o["hit_rate"] - s_o["hit_rate"], 6)
        })
    df_st_vs_cstp = pd.DataFrame(st_vs_cstp_rows)
    df_st_vs_cstp.to_csv(os.path.join(csv_dir, "st_moe_vs_cstp_profiling.csv"), index=False)

    print(f"  Exported 7 CSV files to {csv_dir}")

    # 6. Generate Publication-Quality Plots
    print("\n[6/6] Generating Publication-Quality Plots...")

    # Plot styling
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

    # Colors
    c_cstp_f = "#1f77b4"  # Blue
    c_cstp_o = "#0055aa"  # Dark Blue
    c_st_f = "#ff7f0e"    # Orange
    c_st_o = "#cc5500"    # Dark Orange

    # Helper for saving both png and pdf
    def save_figure(fig, name):
        p_png = os.path.join(plot_dir, f"{name}.png")
        p_pdf = os.path.join(plot_dir, f"{name}.pdf")
        fig.savefig(p_png, dpi=300, bbox_inches="tight")
        fig.savefig(p_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved {name}.png and .pdf")

    # Plot 1: Profiling % vs Frozen Recall@8
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for idx, (dset, title) in enumerate([("moe_anywhere", "MoE Anywhere (Sage-A1.5)"), ("moe_suite", "MoE Suite (GLM-5.2)")]):
        ax = axes[idx]
        for h, ls in [(1, "-o"), (2, "--s")]:
            sub_c = df_primary[(df_primary["dataset_id"] == dset) & (df_primary["horizon"] == h) & (df_primary["predictor"] == "CSTP") & (df_primary["regime"] == "frozen")]
            sub_s = df_primary[(df_primary["dataset_id"] == dset) & (df_primary["horizon"] == h) & (df_primary["predictor"] == "ST-MoE") & (df_primary["regime"] == "frozen")]
            ax.plot(sub_c["profiling_pct"], sub_c["macro_recall"] * 100, ls, color=c_cstp_f, label=f"CSTP Frozen (H={h})")
            ax.plot(sub_s["profiling_pct"], sub_s["macro_recall"] * 100, ls, color=c_st_f, label=f"ST-MoE Frozen (H={h})")
        ax.set_title(title)
        ax.set_xlabel("Profiling Training Split Percentage (%)")
        ax.set_ylabel("Frozen Recall@8 (%)")
        ax.set_xticks(percentages)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Fig 1: Profiling Percentage vs Frozen Recall@8", y=1.02)
    save_figure(fig, "fig1_profiling_pct_vs_frozen_recall")

    # Plot 2: Profiling % vs Online Recall@8
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for idx, (dset, title) in enumerate([("moe_anywhere", "MoE Anywhere (Sage-A1.5)"), ("moe_suite", "MoE Suite (GLM-5.2)")]):
        ax = axes[idx]
        for h, ls in [(1, "-o"), (2, "--s")]:
            sub_c = df_primary[(df_primary["dataset_id"] == dset) & (df_primary["horizon"] == h) & (df_primary["predictor"] == "CSTP") & (df_primary["regime"] == "online_within_request")]
            sub_s = df_primary[(df_primary["dataset_id"] == dset) & (df_primary["horizon"] == h) & (df_primary["predictor"] == "ST-MoE") & (df_primary["regime"] == "online_within_request")]
            ax.plot(sub_c["profiling_pct"], sub_c["macro_recall"] * 100, ls, color=c_cstp_o, label=f"CSTP Online (H={h})")
            ax.plot(sub_s["profiling_pct"], sub_s["macro_recall"] * 100, ls, color=c_st_o, label=f"ST-MoE Online (H={h})")
        ax.set_title(title)
        ax.set_xlabel("Profiling Training Split Percentage (%)")
        ax.set_ylabel("Online Recall@8 (%)")
        ax.set_xticks(percentages)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Fig 2: Profiling Percentage vs Online-Within-Request Recall@8", y=1.02)
    save_figure(fig, "fig2_profiling_pct_vs_online_recall")

    # Plot 3: Profiling % vs Online Adaptation Gain
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for idx, (dset, title) in enumerate([("moe_anywhere", "MoE Anywhere (Sage-A1.5)"), ("moe_suite", "MoE Suite (GLM-5.2)")]):
        ax = axes[idx]
        for h, ls in [(1, "-o"), (2, "--s")]:
            sub = df_fvo[(df_fvo["dataset_id"] == dset) & (df_fvo["horizon"] == h)]
            sub_c = sub[sub["predictor"] == "CSTP"]
            sub_s = sub[sub["predictor"] == "ST-MoE"]
            ax.plot(sub_c["profiling_pct"], sub_c["delta_recall"] * 100, ls, color=c_cstp_o, label=f"CSTP ΔRecall (H={h})")
            ax.plot(sub_s["profiling_pct"], sub_s["delta_recall"] * 100, ls, color=c_st_o, label=f"ST-MoE ΔRecall (H={h})")
        ax.set_title(title)
        ax.set_xlabel("Profiling Training Split Percentage (%)")
        ax.set_ylabel("Online Adaptation Gain ΔRecall (percentage points)")
        ax.set_xticks(percentages)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Fig 3: Profiling Percentage vs Online Adaptation Gain (ΔRecall = Online - Frozen)", y=1.02)
    save_figure(fig, "fig3_profiling_pct_vs_online_adaptation_gain")

    # Plot 4: Profiling % vs Cold-Start Recall
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for idx, (dset, title) in enumerate([("moe_anywhere", "MoE Anywhere (Sage-A1.5)"), ("moe_suite", "MoE Suite (GLM-5.2)")]):
        ax = axes[idx]
        for h, ls in [(1, "-o"), (2, "--s")]:
            sub_c = df_primary[(df_primary["dataset_id"] == dset) & (df_primary["horizon"] == h) & (df_primary["predictor"] == "CSTP") & (df_primary["regime"] == "online_within_request")]
            sub_s = df_primary[(df_primary["dataset_id"] == dset) & (df_primary["horizon"] == h) & (df_primary["predictor"] == "ST-MoE") & (df_primary["regime"] == "online_within_request")]
            ax.plot(sub_c["profiling_pct"], sub_c["cold_start_recall"] * 100, ls, color=c_cstp_o, label=f"CSTP Cold-Start (H={h})")
            ax.plot(sub_s["profiling_pct"], sub_s["cold_start_recall"] * 100, ls, color=c_st_o, label=f"ST-MoE Cold-Start (H={h})")
        ax.set_title(title)
        ax.set_xlabel("Profiling Training Split Percentage (%)")
        ax.set_ylabel("Cold-Start Recall@8 (%) (Token t=0)")
        ax.set_xticks(percentages)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Fig 4: Profiling Percentage vs Cold-Start Recall@8 (Token t=0)", y=1.02)
    save_figure(fig, "fig4_profiling_pct_vs_cold_start_recall")

    # Plot 5: Profiling % vs Warm-Token (Steady-State) Recall
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for idx, (dset, title) in enumerate([("moe_anywhere", "MoE Anywhere (Sage-A1.5)"), ("moe_suite", "MoE Suite (GLM-5.2)")]):
        ax = axes[idx]
        for h, ls in [(1, "-o"), (2, "--s")]:
            sub_c = df_primary[(df_primary["dataset_id"] == dset) & (df_primary["horizon"] == h) & (df_primary["predictor"] == "CSTP") & (df_primary["regime"] == "online_within_request")]
            sub_s = df_primary[(df_primary["dataset_id"] == dset) & (df_primary["horizon"] == h) & (df_primary["predictor"] == "ST-MoE") & (df_primary["regime"] == "online_within_request")]
            ax.plot(sub_c["profiling_pct"], sub_c["warm_token_recall"] * 100, ls, color=c_cstp_o, label=f"CSTP Warm-Token (H={h})")
            ax.plot(sub_s["profiling_pct"], sub_s["warm_token_recall"] * 100, ls, color=c_st_o, label=f"ST-MoE Warm-Token (H={h})")
        ax.set_title(title)
        ax.set_xlabel("Profiling Training Split Percentage (%)")
        ax.set_ylabel("Warm-Token Recall@8 (%) (Tokens t > 0)")
        ax.set_xticks(percentages)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Fig 5: Profiling Percentage vs Warm-Token Recall@8 (Steady-State Tokens t > 0)", y=1.02)
    save_figure(fig, "fig5_profiling_pct_vs_warm_token_recall")

    # Plot 6: Profiling % vs Hit Rate
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for idx, (dset, title) in enumerate([("moe_anywhere", "MoE Anywhere (Sage-A1.5)"), ("moe_suite", "MoE Suite (GLM-5.2)")]):
        ax = axes[idx]
        for h, ls in [(1, "-o"), (2, "--s")]:
            sub_c = df_primary[(df_primary["dataset_id"] == dset) & (df_primary["horizon"] == h) & (df_primary["predictor"] == "CSTP") & (df_primary["regime"] == "online_within_request")]
            sub_s = df_primary[(df_primary["dataset_id"] == dset) & (df_primary["horizon"] == h) & (df_primary["predictor"] == "ST-MoE") & (df_primary["regime"] == "online_within_request")]
            ax.plot(sub_c["profiling_pct"], sub_c["hit_rate"] * 100, ls, color=c_cstp_o, label=f"CSTP Hit Rate (H={h})")
            ax.plot(sub_s["profiling_pct"], sub_s["hit_rate"] * 100, ls, color=c_st_o, label=f"ST-MoE Hit Rate (H={h})")
        ax.set_title(title)
        ax.set_xlabel("Profiling Training Split Percentage (%)")
        ax.set_ylabel("Hit Rate (%) (At least 1 correct expert)")
        ax.set_xticks(percentages)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Fig 6: Profiling Percentage vs Hit Rate (B=8)", y=1.02)
    save_figure(fig, "fig6_profiling_pct_vs_hit_rate")

    # Plot 7: Profiling % vs Total Spatial Count Mass
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for idx, (dset, title) in enumerate([("moe_anywhere", "MoE Anywhere (Sage-A1.5)"), ("moe_suite", "MoE Suite (GLM-5.2)")]):
        ax = axes[idx]
        for h, ls in [(1, "-o"), (2, "--s")]:
            sub = df_counts[(df_counts["dataset_id"] == dset) & (df_counts["horizon"] == h) & (df_counts["seed"] == 42)]
            ax.plot(sub["profiling_pct"], sub["spatial_count_mass"] / 1e6, ls, color="#2ca02c", label=f"Spatial Mass H={h}")
        ax.set_title(title)
        ax.set_xlabel("Profiling Training Split Percentage (%)")
        ax.set_ylabel("Total Spatial Count Mass (Millions)")
        ax.set_xticks(percentages)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Fig 7: Profiling Percentage vs Total Spatial Co-Activation Count Mass", y=1.02)
    save_figure(fig, "fig7_profiling_pct_vs_total_spatial_count_mass")

    # Plot 8: Profiling % vs Total Temporal Count Mass
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for idx, (dset, title) in enumerate([("moe_anywhere", "MoE Anywhere (Sage-A1.5)"), ("moe_suite", "MoE Suite (GLM-5.2)")]):
        ax = axes[idx]
        for h, ls in [(1, "-o"), (2, "--s")]:
            sub = df_counts[(df_counts["dataset_id"] == dset) & (df_counts["horizon"] == h) & (df_counts["seed"] == 42)]
            ax.plot(sub["profiling_pct"], sub["temporal_count_mass"] / 1e6, ls, color="#d62728", label=f"Temporal Mass H={h}")
        ax.set_title(title)
        ax.set_xlabel("Profiling Training Split Percentage (%)")
        ax.set_ylabel("Total Temporal Transition Count Mass (Millions)")
        ax.set_xticks(percentages)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Fig 8: Profiling Percentage vs Total Temporal Transition Count Mass", y=1.02)
    save_figure(fig, "fig8_profiling_pct_vs_total_temporal_count_mass")

    # Plot 9: Profiling % vs Online Adaptation Gain by Horizon
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for idx, (dset, title) in enumerate([("moe_anywhere", "MoE Anywhere (Sage-A1.5)"), ("moe_suite", "MoE Suite (GLM-5.2)")]):
        ax = axes[idx]
        for h, color in [(1, "#1f77b4"), (2, "#ff7f0e")]:
            sub = df_fvo[(df_fvo["dataset_id"] == dset) & (df_fvo["horizon"] == h) & (df_fvo["predictor"] == "CSTP")]
            ax.plot(sub["profiling_pct"], sub["delta_recall"] * 100, "-o", color=color, label=f"CSTP H={h}")
            sub_st = df_fvo[(df_fvo["dataset_id"] == dset) & (df_fvo["horizon"] == h) & (df_fvo["predictor"] == "ST-MoE")]
            ax.plot(sub_st["profiling_pct"], sub_st["delta_recall"] * 100, ":s", color=color, label=f"ST-MoE H={h}")
        ax.set_title(title)
        ax.set_xlabel("Profiling Training Split Percentage (%)")
        ax.set_ylabel("Adaptation Gain ΔRecall (pp)")
        ax.set_xticks(percentages)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Fig 9: Profiling Percentage vs Online Adaptation Gain Stratified by Horizon H", y=1.02)
    save_figure(fig, "fig9_profiling_pct_vs_online_adaptation_gain_by_horizon")

    print("\n" + "=" * 80)
    print("PHASE 1A CAMPAIGN EXECUTION AND EXPORT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_phase1a_campaign()

