"""
Phase D Master Experiment Orchestrator.
Executes the comprehensive benchmarking of ACSH (Global, Stage-Aware, Fast Recovery)
against CSTP, Paper-Faithful ST-MoE, Fixed Hybrid, and Baselines across all workloads.
"""

import os
import sys
import copy
import time
import json
import argparse
import subprocess
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.predictor.cstp_predictor import CSTPPredictor
from src.predictor.st_moe_paper import STMoEPaperPredictor
from src.predictor.acsh_predictor import ACSHPredictor
from src.predictor.base_predictor import (
    BasePredictor,
    CSTPWrapper,
    RandomPredictor,
    FrequencyPredictor,
    TemporalOnlyPredictor,
    SpatialOnlyPredictor,
    CSTPSTMoEHybridPredictor
)
from src.predictor.adapters.base_adapter import PredictionRecord, RoutingDatasetAdapter
from src.predictor.adapters.simulator_adapter import SimulatorDatasetAdapter
from src.predictor.adapters.synthetic_adapter import SyntheticCorrelatedDatasetAdapter
from src.predictor.adapters.moe_suite_adapter import MoeSuiteDatasetAdapter
from src.predictor.adapters.moe_anywhere_adapter import MoeAnywhereDatasetAdapter


def get_git_commit() -> str:
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        commit = subprocess.check_output(cmd, cwd=project_root).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown_commit"


def get_dataset_adapter(dataset_id: str) -> RoutingDatasetAdapter:
    if dataset_id == "deepseek_random":
        trace_path = os.path.join(
            project_root,
            "reproduction_A100",
            "with_off_chip_dram",
            "deepseekv3",
            "experiments",
            "deepseekv3_mixtral_style_20261011",
            "full_run_5step_20261011",
            "traces",
            "expert_routing.csv"
        )
        return SimulatorDatasetAdapter(
            dataset_id="deepseek_random",
            trace_path=trace_path,
            num_moe_stages=57,
            num_routed_experts=256,
            top_k=8,
            first_k_dense=3,
            reconstruct_stages=True
        )
    else:
        if dataset_id == "duplex_a100":
            trace_path = os.path.join(
                project_root,
                "reproduction_A100",
                "results",
                "a100_duplex",
                "traces",
                "expert_routing.csv"
            )
            return SimulatorDatasetAdapter(
                dataset_id="duplex_a100",
                trace_path=trace_path,
                num_moe_stages=32,
                num_routed_experts=8,
                top_k=2,
                first_k_dense=0,
                reconstruct_stages=False,
                train_ratio=0.75
            )
        else:
            if dataset_id == "correlated_synthetic":
                trace_path = os.path.join(
                    project_root,
                    "results",
                    "endsem_phase1",
                    "benchmark_correlated_trace.csv"
                )
                return SyntheticCorrelatedDatasetAdapter(trace_path=trace_path)
            else:
                if dataset_id == "moe_suite":
                    data_dir = os.path.join(
                        project_root,
                        "data",
                        "downloads",
                        "moe_suite"
                    )
                    return MoeSuiteDatasetAdapter(data_dir=data_dir)
                else:
                    if dataset_id == "moe_anywhere":
                        trace_path = os.path.join(
                            project_root,
                            "data",
                            "downloads",
                            "moe_anywhere",
                            "e1-sage-a1.5.parquet"
                        )
                        return MoeAnywhereDatasetAdapter(parquet_path=trace_path)
                    else:
                        raise ValueError(f"Unknown dataset_id: {dataset_id}")


def get_default_budgets(num_routed_experts: int, small_scale: bool = False) -> List[int]:
    if small_scale:
        return [2, 4]
    if num_routed_experts == 8:
        return [1, 2, 4, 6, 8]
    return [4, 8, 16, 24, 32, 48, 64]


def profile_predictors(
    adapter: RoutingDatasetAdapter,
    num_moe_stages: int,
    num_routed_experts: int,
    top_k: int,
    horizon: int = 1
) -> Dict[str, Any]:
    profile_data = adapter.profile_tensors(
        horizon=horizon,
        split="train",
        num_moe_stages=num_moe_stages,
        num_routed_experts=num_routed_experts
    )

    if isinstance(profile_data, tuple):
        marginal_counts = profile_data[0]
        spatial_counts = profile_data[1]
        temporal_counts = profile_data[2]
    else:
        marginal_counts = profile_data["marginal_counts"]
        spatial_counts = profile_data["spatial_counts"]
        temporal_counts = profile_data["temporal_counts"]

    cstp = CSTPPredictor(
        num_moe_stages=num_moe_stages,
        num_routed_experts=num_routed_experts,
        top_k=top_k,
        alpha=0.5,
        online_adaptation=True,
        prediction_horizon=horizon
    )
    cstp.stage_marginal_counts = np.copy(marginal_counts)
    cstp.spatial_counts = np.copy(spatial_counts)
    cstp.temporal_counts = np.copy(temporal_counts)

    st_moe_paper = STMoEPaperPredictor(
        num_moe_stages=num_moe_stages,
        num_routed_experts=num_routed_experts,
        top_k=top_k,
        cct_capacity=top_k,
        confidence_threshold=2,
        online_adaptation=True,
        prediction_horizon=horizon
    )
    n_transitions = max(1, num_moe_stages - horizon)
    for stage in range(n_transitions):
        for u in range(num_routed_experts):
            row = spatial_counts[stage, u, :]
            dest_candidates = list(range(num_routed_experts))
            dest_candidates.sort(key=lambda idx: (-row[idx], idx))
            stored = 0
            for v in dest_candidates:
                if row[v] > 0 and stored < top_k:
                    st_moe_paper.cct[stage][u][v] = 2
                    stored += 1

    freq = FrequencyPredictor(
        num_moe_stages=num_moe_stages,
        num_routed_experts=num_routed_experts
    )
    freq.frequency_counts = np.copy(marginal_counts.astype(np.int64))

    return {
        "cstp": cstp,
        "st_moe_paper": st_moe_paper,
        "frequency": freq
    }


def create_predictor_instance(
    name: str,
    regime: str,
    profiled_state: Dict[str, Any],
    num_moe_stages: int,
    num_routed_experts: int,
    top_k: int,
    horizon: int = 1,
    seed: int = 42
) -> BasePredictor:
    if name == "cstp":
        if regime == "cold_start":
            pred = CSTPPredictor(
                num_moe_stages=num_moe_stages,
                num_routed_experts=num_routed_experts,
                top_k=top_k,
                alpha=0.5,
                online_adaptation=True,
                prediction_horizon=horizon
            )
        else:
            base_p = copy.deepcopy(profiled_state["cstp"])
            if regime == "online":
                base_p.online_adaptation = True
            else:
                base_p.online_adaptation = False
            pred = base_p
        wrapper = CSTPWrapper(pred)
        if regime == "online":
            wrapper.name = "cstp_online"
        else:
            if regime == "cold_start":
                wrapper.name = "cstp_cold_start"
            else:
                wrapper.name = "cstp_frozen"
        return wrapper

    if name == "st_moe_paper":
        if regime == "cold_start":
            pred = STMoEPaperPredictor(
                num_moe_stages=num_moe_stages,
                num_routed_experts=num_routed_experts,
                top_k=top_k,
                cct_capacity=top_k,
                confidence_threshold=2,
                online_adaptation=True,
                prediction_horizon=horizon
            )
        else:
            base_p = copy.deepcopy(profiled_state["st_moe_paper"])
            if regime == "online":
                base_p.online_adaptation = True
            else:
                base_p.online_adaptation = False
            pred = base_p

        if regime == "online":
            pred.name = "st_moe_paper_online"
        else:
            if regime == "cold_start":
                pred.name = "st_moe_paper_cold_start"
            else:
                pred.name = "st_moe_paper_frozen"
        return pred

    if name == "cstp_st_moe_hybrid":
        cstp_copy = copy.deepcopy(profiled_state["cstp"])
        st_copy = copy.deepcopy(profiled_state["st_moe_paper"])
        if regime == "online":
            cstp_copy.online_adaptation = True
            st_copy.online_adaptation = True
        else:
            cstp_copy.online_adaptation = False
            st_copy.online_adaptation = False
        return CSTPSTMoEHybridPredictor(cstp_copy, st_copy, cstp_weight=0.5)

    if name == "acsh_global":
        cstp_copy = copy.deepcopy(profiled_state["cstp"])
        st_copy = copy.deepcopy(profiled_state["st_moe_paper"])
        cstp_copy.online_adaptation = True
        st_copy.online_adaptation = True
        return ACSHPredictor(
            cstp_predictor=cstp_copy,
            st_moe_predictor=st_copy,
            variant="global",
            name="acsh_global"
        )

    if name == "acsh_stage_aware":
        cstp_copy = copy.deepcopy(profiled_state["cstp"])
        st_copy = copy.deepcopy(profiled_state["st_moe_paper"])
        cstp_copy.online_adaptation = True
        st_copy.online_adaptation = True
        return ACSHPredictor(
            cstp_predictor=cstp_copy,
            st_moe_predictor=st_copy,
            variant="stage_aware",
            name="acsh_stage_aware"
        )

    if name == "acsh_fast_recovery":
        cstp_copy = copy.deepcopy(profiled_state["cstp"])
        st_copy = copy.deepcopy(profiled_state["st_moe_paper"])
        cstp_copy.online_adaptation = True
        st_copy.online_adaptation = True
        return ACSHPredictor(
            cstp_predictor=cstp_copy,
            st_moe_predictor=st_copy,
            variant="fast_recovery",
            name="acsh_fast_recovery"
        )

    if name == "acsh_cold_start":
        cstp_cold = CSTPPredictor(
            num_moe_stages=num_moe_stages,
            num_routed_experts=num_routed_experts,
            top_k=top_k,
            alpha=0.5,
            online_adaptation=True,
            prediction_horizon=horizon
        )
        st_cold = STMoEPaperPredictor(
            num_moe_stages=num_moe_stages,
            num_routed_experts=num_routed_experts,
            top_k=top_k,
            cct_capacity=top_k,
            confidence_threshold=2,
            online_adaptation=True,
            prediction_horizon=horizon
        )
        return ACSHPredictor(
            cstp_predictor=cstp_cold,
            st_moe_predictor=st_cold,
            variant="stage_aware",
            name="acsh_cold_start"
        )

    if name == "frequency_only":
        return copy.deepcopy(profiled_state["frequency"])

    if name == "temporal_only":
        cstp_copy = copy.deepcopy(profiled_state["cstp"])
        cstp_copy.online_adaptation = True
        return TemporalOnlyPredictor(cstp_copy)

    if name == "spatial_only":
        cstp_copy = copy.deepcopy(profiled_state["cstp"])
        cstp_copy.online_adaptation = True
        return SpatialOnlyPredictor(cstp_copy)

    if name == "random":
        return RandomPredictor(num_routed_experts=num_routed_experts, seed=seed)

    raise ValueError(f"Unknown predictor name: {name}")


def evaluate_dataset(
    dataset_id: str,
    small_scale: bool = False,
    horizons: Optional[List[int]] = None,
    regimes: Optional[List[str]] = None,
    max_test_units: Optional[int] = None,
    max_tokens_per_unit: Optional[int] = None
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"PHASE D EVALUATION: {dataset_id}")
    print("=" * 80)

    adapter = get_dataset_adapter(dataset_id)
    adapter.load()

    out_base = os.path.join(project_root, "results", "endsem_phase_d", "per_dataset", dataset_id)
    os.makedirs(out_base, exist_ok=True)

    meta = adapter.get_metadata()
    num_stages = meta.get("num_moe_stages", None)
    if num_stages is None:
        num_stages = getattr(adapter, "num_stages", 57)

    num_experts = meta.get("num_routed_experts", None)
    if num_experts is None:
        num_experts = getattr(adapter, "num_experts", 256)

    top_k = meta.get("top_k", None)
    if top_k is None:
        top_k = getattr(adapter, "top_k", 8)

    if horizons is None:
        eval_horizons = [1]
    else:
        eval_horizons = horizons

    budgets = get_default_budgets(num_experts, small_scale=small_scale)

    # Full list of 15 predictor configurations
    predictor_configs = [
        ("cstp", "online", "cstp_online"),
        ("cstp", "frozen", "cstp_frozen"),
        ("st_moe_paper", "online", "st_moe_paper_online"),
        ("st_moe_paper", "frozen", "st_moe_paper_frozen"),
        ("cstp_st_moe_hybrid", "online", "cstp_st_moe_hybrid"),
        ("acsh_global", "online", "acsh_global"),
        ("acsh_stage_aware", "online", "acsh_stage_aware"),
        ("acsh_fast_recovery", "online", "acsh_fast_recovery"),
        ("frequency_only", "frozen", "frequency_only"),
        ("temporal_only", "online", "temporal_only"),
        ("spatial_only", "online", "spatial_only"),
        ("random", "baseline", "random"),
        ("cstp", "cold_start", "cstp_cold_start"),
        ("st_moe_paper", "cold_start", "st_moe_paper_cold_start"),
        ("acsh_cold_start", "cold_start", "acsh_cold_start")
    ]

    all_per_pred_rows: List[Dict[str, Any]] = []
    all_aggregate_rows: List[Dict[str, Any]] = []
    all_weights_rows: List[Dict[str, Any]] = []

    for h in eval_horizons:
        print(f"\n--- Profiling predictors for Horizon H={h} ---")
        profiled_state = profile_predictors(
            adapter=adapter,
            num_moe_stages=num_stages,
            num_routed_experts=num_experts,
            top_k=top_k,
            horizon=h
        )

        print(f"Collecting test prediction sequence for Horizon H={h}...")
        test_records: List[PredictionRecord] = []
        for rec in adapter.iter_predictions(horizon=h, split="test"):
            test_records.append(rec)

        units_map: Dict[Any, List[PredictionRecord]] = {}
        for r in test_records:
            if r.request_id not in units_map:
                units_map[r.request_id] = []
            units_map[r.request_id].append(r)

        sorted_units = sorted(list(units_map.keys()), key=lambda x: str(x))
        if max_test_units is not None:
            sorted_units = sorted_units[:max_test_units]

        print(f"Evaluating {len(sorted_units)} test units across {len(predictor_configs)} predictors and {len(budgets)} budgets.")

        for p_name, regime, pred_label in predictor_configs:
            for b in budgets:
                pred_runner = create_predictor_instance(
                    name=p_name,
                    regime=regime,
                    profiled_state=profiled_state,
                    num_moe_stages=num_stages,
                    num_routed_experts=num_experts,
                    top_k=top_k,
                    horizon=h,
                    seed=42
                )

                hits_list: List[int] = []
                rec_list: List[float] = []
                prec_list: List[float] = []
                f1_list: List[float] = []
                hit_rate_list: List[float] = []

                for uid in sorted_units:
                    pred_runner.reset(uid)
                    unit_records = units_map[uid]
                    if max_tokens_per_unit is not None:
                        unit_tokens = sorted(list(set(r.token_id for r in unit_records)))[:max_tokens_per_unit]
                        token_set = set(unit_tokens)
                        eval_recs = [r for r in unit_records if r.token_id in token_set]
                    else:
                        eval_recs = unit_records

                    for rec in eval_recs:
                        # 1. PREDICT strictly before revealing actual target
                        pred_experts = pred_runner.predict(
                            request_id=rec.request_id,
                            target_stage_id=rec.target_stage_id,
                            current_source_experts=rec.source_experts,
                            budget=b,
                            token_id=rec.token_id,
                            source_stage_id=rec.source_stage_id
                        )

                        # 2. EVALUATE prediction against ground truth
                        p_set = set(pred_experts)
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

                        hits_list.append(h_count)
                        rec_list.append(recall)
                        prec_list.append(prec)
                        f1_list.append(f1)
                        hit_rate_list.append(hr)

                        all_per_pred_rows.append({
                            "dataset_id": dataset_id,
                            "split": "test",
                            "predictor": pred_label,
                            "learning_mode": regime,
                            "prediction_horizon": h,
                            "budget": b,
                            "seed": 42,
                            "request_id": str(rec.request_id),
                            "token_id": rec.token_id,
                            "source_stage": rec.source_stage_id,
                            "target_stage": rec.target_stage_id,
                            "hits": h_count,
                            "precision": round(prec, 4),
                            "recall": round(recall, 4),
                            "f1": round(f1, 4),
                            "hit_rate": hr,
                            "missed": len(a_set.difference(p_set)),
                            "unused": len(p_set.difference(a_set))
                        })

                        # 3. UPDATE predictor state post-reveal
                        pred_runner.update(
                            request_id=rec.request_id,
                            target_stage_id=rec.target_stage_id,
                            source_experts=rec.source_experts,
                            actual_target_experts=rec.actual_target_experts,
                            token_id=rec.token_id,
                            source_stage_id=rec.source_stage_id
                        )

                # Capture weights history for ACSH at primary budget
                record_budget = 8
                if 8 not in budgets:
                    record_budget = budgets[-1]
                if "acsh" in pred_label and b == record_budget:
                    if hasattr(pred_runner, "weight_history"):
                        for wh in pred_runner.weight_history:
                            wh_copy = dict(wh)
                            wh_copy["dataset_id"] = dataset_id
                            wh_copy["predictor"] = pred_label
                            all_weights_rows.append(wh_copy)

                # Aggregate metrics for this predictor-budget
                n_eval = len(hits_list)
                if n_eval > 0:
                    tot_hits = sum(hits_list)
                    tot_act = n_eval * top_k
                    tot_pred = n_eval * b
                    agg_row = {
                        "dataset_id": dataset_id,
                        "predictor": pred_label,
                        "learning_mode": regime,
                        "prediction_horizon": h,
                        "budget": b,
                        "num_instances": n_eval,
                        "macro_recall": round(float(np.mean(rec_list)), 4),
                        "macro_precision": round(float(np.mean(prec_list)), 4),
                        "macro_f1": round(float(np.mean(f1_list)), 4),
                        "hit_rate": round(float(np.mean(hit_rate_list)), 4),
                        "micro_recall": round(float(tot_hits) / float(tot_act), 4),
                        "micro_precision": round(float(tot_hits) / float(tot_pred), 4),
                        "coverage": round(float(tot_hits) / float(tot_act), 4)
                    }
                    all_aggregate_rows.append(agg_row)
                    print(f"  [{pred_label} | H={h} | B={b}] Recall: {agg_row['macro_recall']:.4f}, Prec: {agg_row['macro_precision']:.4f}, F1: {agg_row['macro_f1']:.4f}, HitRate: {agg_row['hit_rate']:.4f}")

    # Save CSV outputs
    df_agg = pd.DataFrame(all_aggregate_rows)
    df_agg.to_csv(os.path.join(out_base, "aggregate_metrics.csv"), index=False)
    df_agg.to_csv(os.path.join(out_base, "budget_sweep.csv"), index=False)

    df_per_pred = pd.DataFrame(all_per_pred_rows)
    df_per_pred.to_csv(os.path.join(out_base, "per_prediction_results.csv"), index=False)

    if len(all_weights_rows) > 0:
        df_wh = pd.DataFrame(all_weights_rows)
        df_wh.to_csv(os.path.join(out_base, "acsh_weights.csv"), index=False)

    print(f"\nSaved Phase D results for {dataset_id} under {out_base}/")
    return {
        "aggregate": all_aggregate_rows,
        "total_predictions": len(all_per_pred_rows)
    }


def run_full_phase_d() -> None:
    start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start_time_sec = time.time()

    print("=" * 80)
    print("STARTING PHASE D MASTER BENCHMARK (ACSH vs BASELINES)")
    print(f"Timestamp: {start_time_iso}")
    print(f"Git Commit: {get_git_commit()}")
    print("=" * 80)

    workload_configs = [
        {
            "dataset_id": "deepseek_random",
            "horizons": [1],
            "max_units": None,
            "max_tokens": None
        },
        {
            "dataset_id": "duplex_a100",
            "horizons": [1],
            "max_units": None,
            "max_tokens": None
        },
        {
            "dataset_id": "correlated_synthetic",
            "horizons": [1],
            "max_units": None,
            "max_tokens": None
        },
        {
            "dataset_id": "moe_anywhere",
            "horizons": [1, 2],
            "max_units": 14,
            "max_tokens": 20
        },
        {
            "dataset_id": "moe_suite",
            "horizons": [2],
            "max_units": 2,
            "max_tokens": 50
        }
    ]

    manifest: Dict[str, Any] = {
        "phase": "Phase D",
        "start_time": start_time_iso,
        "git_commit": get_git_commit(),
        "workloads": []
    }

    for cfg in workload_configs:
        d_id = cfg["dataset_id"]
        t0 = time.time()
        res = evaluate_dataset(
            dataset_id=d_id,
            small_scale=False,
            horizons=cfg["horizons"],
            max_test_units=cfg["max_units"],
            max_tokens_per_unit=cfg["max_tokens"]
        )
        duration_s = time.time() - t0
        manifest["workloads"].append({
            "dataset_id": d_id,
            "duration_seconds": round(duration_s, 2),
            "total_predictions": res["total_predictions"]
        })

    end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    total_elapsed = time.time() - start_time_sec
    manifest["end_time"] = end_time_iso
    manifest["total_elapsed_seconds"] = round(total_elapsed, 2)
    manifest["status"] = "SUCCESS"

    meta_dir = os.path.join(project_root, "results", "endsem_phase_d", "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    manifest_path = os.path.join(meta_dir, "run_manifest.json")
    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp, indent=2)

    print("\n" + "=" * 80)
    print("PHASE D FULL BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Total Elapsed Time: {total_elapsed:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    run_full_phase_d()
