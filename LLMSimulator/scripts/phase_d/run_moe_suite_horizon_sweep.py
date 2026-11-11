"""
MoE Suite Intra-Token Inter-Layer Horizon Sweep Runner (H = 1, 2, 3, 4).
Evaluates all implemented predictors and baselines strictly on the MoE Suite dataset.
Preserves official task-disjoint splits and strictly avoids ternary operators (ast.IfExp == 0).
"""

import os
import sys
import copy
import time
import json
import tracemalloc
import subprocess
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.predictor.cstp_predictor import CSTPPredictor
from src.predictor.st_moe_paper import STMoEPaperPredictor
from src.predictor.st_moe_predictor import STMoEPredictor
from src.predictor.acsh_predictor import ACSHPredictor
from src.predictor.base_predictor import (
    BasePredictor,
    CSTPWrapper,
    STMoELegacyWrapper,
    RandomPredictor,
    FrequencyPredictor,
    TemporalOnlyPredictor,
    SpatialOnlyPredictor,
    CSTPSTMoEHybridPredictor
)
from src.predictor.adapters.moe_suite_adapter import MoeSuiteDatasetAdapter
from src.predictor.adapters.base_adapter import PredictionRecord


def get_git_commit() -> str:
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        commit = subprocess.check_output(cmd, cwd=project_root).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown_commit"


def compute_timing_stats(latencies_ns: List[int]) -> Dict[str, float]:
    arr_us = np.array(latencies_ns, dtype=np.float64) / 1000.0
    return {
        "mean_us": round(float(np.mean(arr_us)), 3),
        "median_us": round(float(np.median(arr_us)), 3),
        "std_us": round(float(np.std(arr_us)), 3),
        "p95_us": round(float(np.percentile(arr_us, 95)), 3),
        "p99_us": round(float(np.percentile(arr_us, 99)), 3),
        "min_us": round(float(np.min(arr_us)), 3),
        "max_us": round(float(np.max(arr_us)), 3)
    }


def profile_moe_suite_predictors(
    adapter: MoeSuiteDatasetAdapter,
    horizon: int
) -> Dict[str, Any]:
    profile_data = adapter.profile_tensors(
        horizon=horizon,
        split="train",
        num_moe_stages=adapter.num_moe_stages,
        num_routed_experts=adapter.num_routed_experts
    )

    marginal_counts, spatial_counts, temporal_counts = profile_data

    # CSTP
    cstp = CSTPPredictor(
        num_moe_stages=adapter.num_moe_stages,
        num_routed_experts=adapter.num_routed_experts,
        top_k=adapter.top_k,
        alpha=0.5,
        online_adaptation=True,
        prediction_horizon=horizon
    )
    cstp.stage_marginal_counts = np.copy(marginal_counts)
    cstp.spatial_counts = np.copy(spatial_counts)
    cstp.temporal_counts = np.copy(temporal_counts)

    # ST-MoE Paper
    st_moe_paper = STMoEPaperPredictor(
        num_moe_stages=adapter.num_moe_stages,
        num_routed_experts=adapter.num_routed_experts,
        top_k=adapter.top_k,
        cct_capacity=adapter.top_k,
        confidence_threshold=2,
        online_adaptation=True,
        prediction_horizon=horizon
    )
    n_transitions = max(1, adapter.num_moe_stages - horizon)
    for stage in range(n_transitions):
        for u in range(adapter.num_routed_experts):
            row = spatial_counts[stage, u, :]
            dest_candidates = list(range(adapter.num_routed_experts))
            dest_candidates.sort(key=lambda idx: (-row[idx], idx))
            stored = 0
            for v in dest_candidates:
                if row[v] > 0 and stored < adapter.top_k:
                    st_moe_paper.cct[stage][u][v] = 2
                    stored += 1

    # Frequency baseline
    freq = FrequencyPredictor(
        num_moe_stages=adapter.num_moe_stages,
        num_routed_experts=adapter.num_routed_experts
    )
    freq.frequency_counts = np.copy(marginal_counts.astype(np.int64))

    # ST-MoE Legacy (only valid when horizon == 1)
    st_legacy = None
    if horizon == 1:
        st_legacy = STMoEPredictor(
            num_moe_stages=adapter.num_moe_stages,
            num_routed_experts=adapter.num_routed_experts,
            top_k=adapter.top_k,
            cct_capacity=adapter.top_k,
            confidence_threshold=2
        )
        for stage in range(adapter.num_moe_stages - 1):
            for u in range(adapter.num_routed_experts):
                row = spatial_counts[stage, u, :]
                dest_candidates = list(range(adapter.num_routed_experts))
                dest_candidates.sort(key=lambda idx: (-row[idx], idx))
                stored = 0
                for v in dest_candidates:
                    if row[v] > 0 and stored < adapter.top_k:
                        st_legacy.cct[stage][u][v] = 2
                        stored += 1

        priors_by_stage = {}
        for stage in range(adapter.num_moe_stages):
            row = marginal_counts[stage, :]
            dest_candidates = list(range(adapter.num_routed_experts))
            dest_candidates.sort(key=lambda idx: (-row[idx], idx))
            priors_by_stage[stage] = dest_candidates[:adapter.top_k]
        st_legacy.set_profiling_prior(priors_by_stage)

    return {
        "cstp": cstp,
        "st_moe_paper": st_moe_paper,
        "frequency": freq,
        "st_legacy": st_legacy,
        "marginal_counts": marginal_counts,
        "spatial_counts": spatial_counts,
        "temporal_counts": temporal_counts
    }


def create_moe_suite_predictor(
    pred_key: str,
    horizon: int,
    profiled_state: Dict[str, Any],
    adapter: MoeSuiteDatasetAdapter,
    seed: int = 42
) -> Optional[BasePredictor]:
    num_stages = adapter.num_moe_stages
    num_experts = adapter.num_routed_experts
    top_k = adapter.top_k

    if pred_key == "cstp_online":
        p = copy.deepcopy(profiled_state["cstp"])
        p.online_adaptation = True
        return CSTPWrapper(p, name="cstp_online")

    if pred_key == "cstp_frozen":
        p = copy.deepcopy(profiled_state["cstp"])
        p.online_adaptation = False
        return CSTPWrapper(p, name="cstp_frozen")

    if pred_key == "st_moe_paper_online":
        p = copy.deepcopy(profiled_state["st_moe_paper"])
        p.online_adaptation = True
        p.name = "st_moe_paper_online"
        return p

    if pred_key == "st_moe_paper_frozen":
        p = copy.deepcopy(profiled_state["st_moe_paper"])
        p.online_adaptation = False
        p.name = "st_moe_paper_frozen"
        return p

    if pred_key == "cstp_st_moe_hybrid":
        c_copy = copy.deepcopy(profiled_state["cstp"])
        s_copy = copy.deepcopy(profiled_state["st_moe_paper"])
        c_copy.online_adaptation = True
        s_copy.online_adaptation = True
        return CSTPSTMoEHybridPredictor(c_copy, s_copy, cstp_weight=0.5)

    if pred_key == "acsh_global":
        c_copy = copy.deepcopy(profiled_state["cstp"])
        s_copy = copy.deepcopy(profiled_state["st_moe_paper"])
        c_copy.online_adaptation = True
        s_copy.online_adaptation = True
        return ACSHPredictor(
            cstp_predictor=c_copy,
            st_moe_predictor=s_copy,
            variant="global",
            name="acsh_global"
        )

    if pred_key == "acsh_stage_aware":
        c_copy = copy.deepcopy(profiled_state["cstp"])
        s_copy = copy.deepcopy(profiled_state["st_moe_paper"])
        c_copy.online_adaptation = True
        s_copy.online_adaptation = True
        return ACSHPredictor(
            cstp_predictor=c_copy,
            st_moe_predictor=s_copy,
            variant="stage_aware",
            name="acsh_stage_aware"
        )

    if pred_key == "acsh_fast_recovery":
        c_copy = copy.deepcopy(profiled_state["cstp"])
        s_copy = copy.deepcopy(profiled_state["st_moe_paper"])
        c_copy.online_adaptation = True
        s_copy.online_adaptation = True
        return ACSHPredictor(
            cstp_predictor=c_copy,
            st_moe_predictor=s_copy,
            variant="fast_recovery",
            name="acsh_fast_recovery"
        )

    if pred_key == "frequency_only":
        return copy.deepcopy(profiled_state["frequency"])

    if pred_key == "temporal_only":
        c_copy = copy.deepcopy(profiled_state["cstp"])
        c_copy.online_adaptation = True
        return TemporalOnlyPredictor(c_copy)

    if pred_key == "spatial_only":
        c_copy = copy.deepcopy(profiled_state["cstp"])
        c_copy.online_adaptation = True
        return SpatialOnlyPredictor(c_copy)

    if pred_key == "random":
        return RandomPredictor(num_routed_experts=num_experts, seed=seed)

    if pred_key == "cstp_cold_start":
        c_cold = CSTPPredictor(
            num_moe_stages=num_stages,
            num_routed_experts=num_experts,
            top_k=top_k,
            alpha=0.5,
            online_adaptation=True,
            prediction_horizon=horizon
        )
        return CSTPWrapper(c_cold, name="cstp_cold_start")

    if pred_key == "st_moe_paper_cold_start":
        s_cold = STMoEPaperPredictor(
            num_moe_stages=num_stages,
            num_routed_experts=num_experts,
            top_k=top_k,
            cct_capacity=top_k,
            confidence_threshold=2,
            online_adaptation=True,
            prediction_horizon=horizon
        )
        s_cold.name = "st_moe_paper_cold_start"
        return s_cold

    if pred_key == "acsh_cold_start":
        c_cold = CSTPPredictor(
            num_moe_stages=num_stages,
            num_routed_experts=num_experts,
            top_k=top_k,
            alpha=0.5,
            online_adaptation=True,
            prediction_horizon=horizon
        )
        s_cold = STMoEPaperPredictor(
            num_moe_stages=num_stages,
            num_routed_experts=num_experts,
            top_k=top_k,
            cct_capacity=top_k,
            confidence_threshold=2,
            online_adaptation=True,
            prediction_horizon=horizon
        )
        return ACSHPredictor(
            cstp_predictor=c_cold,
            st_moe_predictor=s_cold,
            variant="stage_aware",
            name="acsh_cold_start"
        )

    if pred_key == "st_moe_legacy":
        if horizon == 1:
            st_leg = copy.deepcopy(profiled_state["st_legacy"])
            return STMoELegacyWrapper(st_leg)
        else:
            return None

    raise ValueError(f"Unknown predictor key: {pred_key}")


def benchmark_predictor_overhead(
    pred_runner: BasePredictor,
    adapter: MoeSuiteDatasetAdapter,
    horizon: int,
    num_warmup: int = 50,
    num_timed: int = 200
) -> Dict[str, Any]:
    tracemalloc.start()
    mem_before, _ = tracemalloc.get_traced_memory()

    # Create dummy queries
    src_sample = [0, 1, 2, 3, 4, 5, 6, 7]
    tgt_sample = [8, 9, 10, 11, 12, 13, 14, 15]

    # Warmup
    for i in range(num_warmup):
        _ = pred_runner.predict(
            request_id=9999,
            target_stage_id=min(adapter.num_moe_stages - 1, horizon + 5),
            current_source_experts=src_sample,
            budget=8,
            token_id=i,
            source_stage_id=5
        )
        pred_runner.update(
            request_id=9999,
            target_stage_id=min(adapter.num_moe_stages - 1, horizon + 5),
            source_experts=src_sample,
            actual_target_experts=tgt_sample,
            token_id=i,
            source_stage_id=5
        )

    # Timed queries
    q_times: List[int] = []
    for i in range(num_timed):
        t0 = time.perf_counter_ns()
        _ = pred_runner.predict(
            request_id=9999,
            target_stage_id=min(adapter.num_moe_stages - 1, horizon + 5),
            current_source_experts=src_sample,
            budget=8,
            token_id=num_warmup + i,
            source_stage_id=5
        )
        t1 = time.perf_counter_ns()
        q_times.append(t1 - t0)

    # Timed updates
    u_times: List[int] = []
    for i in range(num_timed):
        t0 = time.perf_counter_ns()
        pred_runner.update(
            request_id=9999,
            target_stage_id=min(adapter.num_moe_stages - 1, horizon + 5),
            source_experts=src_sample,
            actual_target_experts=tgt_sample,
            token_id=num_warmup + i,
            source_stage_id=5
        )
        t1 = time.perf_counter_ns()
        u_times.append(t1 - t0)

    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    q_stats = compute_timing_stats(q_times)
    u_stats = compute_timing_stats(u_times)

    mem_kb = round(float(mem_peak - mem_before) / 1024.0, 2)
    if mem_kb <= 0.0:
        mem_kb = round(float(mem_peak) / 1024.0, 2)

    return {
        "query_stats": q_stats,
        "update_stats": u_stats,
        "memory_kb": mem_kb
    }


def run_horizon_sweep():
    start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start_time_sec = time.time()

    print("=" * 85)
    print("MOE SUITE INTRA-TOKEN INTER-LAYER HORIZON SWEEP (H = 1, 2, 3, 4)")
    print(f"Timestamp: {start_time_iso}")
    print(f"Git Commit: {get_git_commit()}")
    print("=" * 85)

    data_dir = os.path.join(project_root, "data", "downloads", "moe_suite")
    adapter = MoeSuiteDatasetAdapter(data_dir=data_dir)
    adapter.load()

    out_dir = os.path.join(project_root, "results", "endsem_moe_suite_horizon")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    validation_report = adapter.validate()
    print("Adapter validation report:")
    print(json.dumps(validation_report, indent=2))

    horizons = [1, 2, 3, 4]
    budget = 8
    max_tokens_per_task = 50

    all_predictor_keys = [
        "cstp_online",
        "cstp_frozen",
        "st_moe_paper_online",
        "st_moe_paper_frozen",
        "cstp_st_moe_hybrid",
        "acsh_global",
        "acsh_stage_aware",
        "acsh_fast_recovery",
        "frequency_only",
        "temporal_only",
        "spatial_only",
        "random",
        "cstp_cold_start",
        "st_moe_paper_cold_start",
        "acsh_cold_start",
        "st_moe_legacy"
    ]

    learning_mode_map = {
        "cstp_online": "online",
        "cstp_frozen": "frozen",
        "st_moe_paper_online": "online",
        "st_moe_paper_frozen": "frozen",
        "cstp_st_moe_hybrid": "online",
        "acsh_global": "online",
        "acsh_stage_aware": "online",
        "acsh_fast_recovery": "online",
        "frequency_only": "frozen",
        "temporal_only": "online",
        "spatial_only": "online",
        "random": "baseline",
        "cstp_cold_start": "cold_start",
        "st_moe_paper_cold_start": "cold_start",
        "acsh_cold_start": "cold_start",
        "st_moe_legacy": "online"
    }

    comparison_rows: List[Dict[str, Any]] = []
    all_per_pred_rows: List[Dict[str, Any]] = []
    per_task_rows: List[Dict[str, Any]] = []
    overhead_rows: List[Dict[str, Any]] = []

    test_tasks = sorted(adapter.splits["test"])
    print(f"\nOfficial Test Tasks ({len(test_tasks)}): {test_tasks}")
    print(f"Evaluation budget: B={budget}, Tokens per task: {max_tokens_per_task}")

    for h in horizons:
        print("\n" + "#" * 80)
        print(f"EVALUATING PREDICTION HORIZON H = {h} (l -> l + {h})")
        print("#" * 80)

        # Profile state on 8 training tasks
        profiled_state = profile_moe_suite_predictors(adapter, horizon=h)

        # Load test records for horizon h
        test_records_by_task: Dict[str, List[PredictionRecord]] = {}
        for task_id in test_tasks:
            test_records_by_task[task_id] = []

        for rec in adapter.iter_predictions(horizon=h, split="test"):
            if rec.request_id in test_records_by_task:
                test_records_by_task[rec.request_id].append(rec)

        # Count tokens and events
        total_eval_tokens = 0
        total_valid_events = 0
        total_boundary_events = 0

        # Subsample to max_tokens_per_task
        eval_recs_by_task: Dict[str, List[PredictionRecord]] = {}
        for task_id in test_tasks:
            raw_recs = test_records_by_task[task_id]
            unit_tokens = sorted(list(set(r.token_id for r in raw_recs)))[:max_tokens_per_task]
            tok_set = set(unit_tokens)
            task_eval_recs = [r for r in raw_recs if r.token_id in tok_set]
            eval_recs_by_task[task_id] = task_eval_recs
            total_eval_tokens += len(tok_set)

            for r in task_eval_recs:
                if r.source_stage_id >= 0:
                    total_valid_events += 1
                else:
                    total_boundary_events += 1

        print(f"Horizon H={h}: {len(test_tasks)} tasks, {total_eval_tokens} tokens total.")
        print(f"  Valid prediction events (l >= {h}): {total_valid_events}")
        print(f"  Boundary skipped events (l < {h}): {total_boundary_events}")
        print(f"  Total events processed: {total_valid_events + total_boundary_events}")

        # Evaluate each predictor
        for pred_key in all_predictor_keys:
            mode = learning_mode_map[pred_key]

            # Check support for this predictor at horizon h
            if pred_key == "st_moe_legacy" and h > 1:
                print(f"  [{pred_key} | H={h}] SKIPPED (unsupported for H > 1, hardcoded to H=1)")
                comp_row = {
                    "dataset_id": "moe_suite",
                    "predictor": pred_key,
                    "learning_mode": mode,
                    "prediction_horizon": h,
                    "budget": budget,
                    "status": "unsupported_for_H_gt_1",
                    "num_tasks": len(test_tasks),
                    "total_tokens": total_eval_tokens,
                    "valid_events": total_valid_events,
                    "boundary_events": total_boundary_events,
                    "macro_recall": np.nan,
                    "macro_precision": np.nan,
                    "macro_f1": np.nan,
                    "hit_rate": np.nan,
                    "query_latency_mean_us": np.nan,
                    "query_latency_p95_us": np.nan,
                    "update_latency_mean_us": np.nan,
                    "memory_kb": np.nan,
                    "task_12_recall": np.nan,
                    "task_13_recall": np.nan,
                    "task_clustered_std_err": np.nan
                }
                comparison_rows.append(comp_row)
                continue

            # Instantiate predictor
            pred_runner = create_moe_suite_predictor(
                pred_key=pred_key,
                horizon=h,
                profiled_state=profiled_state,
                adapter=adapter,
                seed=42
            )

            # Benchmark overhead
            ovh = benchmark_predictor_overhead(
                pred_runner=pred_runner,
                adapter=adapter,
                horizon=h,
                num_warmup=30,
                num_timed=150
            )

            overhead_rows.append({
                "dataset_id": "moe_suite",
                "predictor": pred_key,
                "learning_mode": mode,
                "prediction_horizon": h,
                "budget": budget,
                "query_mean_us": ovh["query_stats"]["mean_us"],
                "query_median_us": ovh["query_stats"]["median_us"],
                "query_p95_us": ovh["query_stats"]["p95_us"],
                "update_mean_us": ovh["update_stats"]["mean_us"],
                "update_p95_us": ovh["update_stats"]["p95_us"],
                "memory_kb": ovh["memory_kb"]
            })

            # Re-instantiate clean predictor for actual evaluation sequence
            pred_runner = create_moe_suite_predictor(
                pred_key=pred_key,
                horizon=h,
                profiled_state=profiled_state,
                adapter=adapter,
                seed=42
            )

            task_recalls: Dict[str, List[float]] = {}
            task_precs: Dict[str, List[float]] = {}
            task_f1s: Dict[str, List[float]] = {}
            task_hrs: Dict[str, List[float]] = {}

            all_valid_hits = 0
            all_valid_actual = 0
            all_valid_preds = 0

            for task_id in test_tasks:
                task_eval_recs = eval_recs_by_task[task_id]
                pred_runner.reset(task_id)

                task_recalls[task_id] = []
                task_precs[task_id] = []
                task_f1s[task_id] = []
                task_hrs[task_id] = []

                for rec in task_eval_recs:
                    is_boundary = False
                    if rec.source_stage_id < 0:
                        is_boundary = True

                    # 1. PREDICT strictly before revealing actual target
                    predicted_experts = pred_runner.predict(
                        request_id=rec.request_id,
                        target_stage_id=rec.target_stage_id,
                        current_source_experts=rec.source_experts,
                        budget=budget,
                        token_id=rec.token_id,
                        source_stage_id=rec.source_stage_id
                    )

                    p_set = set(predicted_experts)
                    a_set = set(rec.actual_target_experts)
                    h_count = len(p_set.intersection(a_set))

                    recall = float(h_count) / float(len(a_set))
                    if len(p_set) > 0:
                        prec = float(h_count) / float(len(p_set))
                    else:
                        prec = 0.0

                    denom = prec + recall
                    if denom > 0.0:
                        f1 = float(2.0 * prec * recall / denom)
                    else:
                        f1 = 0.0

                    if h_count > 0:
                        hr = 1.0
                    else:
                        hr = 0.0

                    # Record per prediction row
                    all_per_pred_rows.append({
                        "dataset_id": "moe_suite",
                        "task_id": str(rec.request_id),
                        "token_id": rec.token_id,
                        "source_stage": rec.source_stage_id,
                        "target_stage": rec.target_stage_id,
                        "prediction_horizon": h,
                        "budget": budget,
                        "predictor": pred_key,
                        "learning_mode": mode,
                        "is_boundary": is_boundary,
                        "hits": h_count,
                        "recall": round(recall, 4),
                        "precision": round(prec, 4),
                        "f1": round(f1, 4),
                        "hit_rate": hr
                    })

                    # Metrics computed on valid lookahead events (l >= h)
                    if not is_boundary:
                        task_recalls[task_id].append(recall)
                        task_precs[task_id].append(prec)
                        task_f1s[task_id].append(f1)
                        task_hrs[task_id].append(hr)

                        all_valid_hits += h_count
                        all_valid_actual += len(a_set)
                        all_valid_preds += len(p_set)

                    # 2. UPDATE predictor state post-reveal
                    pred_runner.update(
                        request_id=rec.request_id,
                        target_stage_id=rec.target_stage_id,
                        source_experts=rec.source_experts,
                        actual_target_experts=rec.actual_target_experts,
                        token_id=rec.token_id,
                        source_stage_id=rec.source_stage_id
                    )

            # Task level summaries
            t12_rec = float(np.mean(task_recalls[test_tasks[0]]))
            t13_rec = float(np.mean(task_recalls[test_tasks[1]]))
            t12_f1 = float(np.mean(task_f1s[test_tasks[0]]))
            t13_f1 = float(np.mean(task_f1s[test_tasks[1]]))

            per_task_rows.append({
                "dataset_id": "moe_suite",
                "predictor": pred_key,
                "learning_mode": mode,
                "prediction_horizon": h,
                "budget": budget,
                "task_id": test_tasks[0],
                "valid_events": len(task_recalls[test_tasks[0]]),
                "recall": round(t12_rec, 4),
                "precision": round(float(np.mean(task_precs[test_tasks[0]])), 4),
                "f1": round(t12_f1, 4),
                "hit_rate": round(float(np.mean(task_hrs[test_tasks[0]])), 4)
            })

            per_task_rows.append({
                "dataset_id": "moe_suite",
                "predictor": pred_key,
                "learning_mode": mode,
                "prediction_horizon": h,
                "budget": budget,
                "task_id": test_tasks[1],
                "valid_events": len(task_recalls[test_tasks[1]]),
                "recall": round(t13_rec, 4),
                "precision": round(float(np.mean(task_precs[test_tasks[1]])), 4),
                "f1": round(t13_f1, 4),
                "hit_rate": round(float(np.mean(task_hrs[test_tasks[1]])), 4)
            })

            # Task-clustered standard error (std of task means / sqrt(num_tasks))
            task_means = [t12_rec, t13_rec]
            task_std = float(np.std(task_means, ddof=1))
            task_se = float(task_std / np.sqrt(len(test_tasks)))

            # Overall macro across all valid events
            all_recs: List[float] = []
            all_precs: List[float] = []
            all_f1s: List[float] = []
            all_hrs: List[float] = []
            for tid in test_tasks:
                all_recs.extend(task_recalls[tid])
                all_precs.extend(task_precs[tid])
                all_f1s.extend(task_f1s[tid])
                all_hrs.extend(task_hrs[tid])

            macro_rec = float(np.mean(all_recs))
            macro_prec = float(np.mean(all_precs))
            macro_f1 = float(np.mean(all_f1s))
            macro_hr = float(np.mean(all_hrs))

            comp_row = {
                "dataset_id": "moe_suite",
                "predictor": pred_key,
                "learning_mode": mode,
                "prediction_horizon": h,
                "budget": budget,
                "status": "completed",
                "num_tasks": len(test_tasks),
                "total_tokens": total_eval_tokens,
                "valid_events": len(all_recs),
                "boundary_events": total_boundary_events,
                "macro_recall": round(macro_rec, 4),
                "macro_precision": round(macro_prec, 4),
                "macro_f1": round(macro_f1, 4),
                "hit_rate": round(macro_hr, 4),
                "query_latency_mean_us": ovh["query_stats"]["mean_us"],
                "query_latency_p95_us": ovh["query_stats"]["p95_us"],
                "update_latency_mean_us": ovh["update_stats"]["mean_us"],
                "memory_kb": ovh["memory_kb"],
                "task_12_recall": round(t12_rec, 4),
                "task_13_recall": round(t13_rec, 4),
                "task_clustered_std_err": round(task_se, 4)
            }
            comparison_rows.append(comp_row)

            print(f"  [{pred_key:22s} | H={h}] Recall: {macro_rec:.4f}, Prec: {macro_prec:.4f}, F1: {macro_f1:.4f}, HitRate: {macro_hr:.4f} | Q_lat: {ovh['query_stats']['mean_us']:.1f}us")

    # Save output artifacts
    print("\nWriting experiment CSV artifacts...")
    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(os.path.join(out_dir, "horizon_comparison.csv"), index=False)
    df_comp.to_csv(os.path.join(out_dir, "aggregate_metrics.csv"), index=False)

    df_task = pd.DataFrame(per_task_rows)
    df_task.to_csv(os.path.join(out_dir, "per_task_metrics.csv"), index=False)

    df_ovh = pd.DataFrame(overhead_rows)
    df_ovh.to_csv(os.path.join(out_dir, "latency_memory_metrics.csv"), index=False)

    df_per_pred = pd.DataFrame(all_per_pred_rows)
    df_per_pred.to_csv(os.path.join(out_dir, "per_prediction_results.csv"), index=False)

    # Manifest
    elapsed_sec = time.time() - start_time_sec
    manifest = {
        "experiment": "MoE Suite Intra-Token Inter-Layer Horizon Sweep",
        "dataset_id": "moe_suite",
        "horizons": horizons,
        "prediction_budget": budget,
        "max_tokens_per_task": max_tokens_per_task,
        "num_test_tasks": len(test_tasks),
        "test_tasks": test_tasks,
        "start_time": start_time_iso,
        "elapsed_seconds": round(elapsed_sec, 2),
        "git_commit": get_git_commit(),
        "predictors_evaluated": all_predictor_keys,
        "supported_counts": len([r for r in comparison_rows if r["status"] == "completed"]),
        "unsupported_counts": len([r for r in comparison_rows if r["status"] != "completed"])
    }
    with open(os.path.join(out_dir, "experiment_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(out_dir, "execution_log.txt"), "w") as f:
        f.write(f"MoE Suite Horizon Sweep Completed Successfully in {elapsed_sec:.2f} seconds.\n")
        f.write(f"Git commit: {get_git_commit()}\n")
        f.write(f"Output directory: {out_dir}\n")

    print(f"\nAll results saved to {out_dir}/ successfully in {elapsed_sec:.2f}s!")


if __name__ == "__main__":
    run_horizon_sweep()

