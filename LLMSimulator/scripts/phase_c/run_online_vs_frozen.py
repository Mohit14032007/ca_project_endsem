"""
Core Evaluation Engine for Online vs Frozen Expert Predictors.
Evaluates CSTP, ST-MoE Paper, ST-MoE Legacy, and all baselines across
Frozen (Exp A), Online (Exp B), and Cold-Start (Exp C) regimes.
"""

import os
import sys
import copy
import argparse
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.predictor.cstp_predictor import CSTPPredictor
from src.predictor.st_moe_paper import STMoEPaperPredictor
from src.predictor.st_moe_predictor import STMoEPredictor
from src.predictor.base_predictor import (
    BasePredictor,
    STMoELegacyWrapper,
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
                        parquet_path = os.path.join(
                            project_root,
                            "data",
                            "downloads",
                            "moe_anywhere",
                            "e1-sage-a1.5.parquet"
                        )
                        return MoeAnywhereDatasetAdapter(parquet_path=parquet_path)
                    else:
                        raise ValueError(f"Unknown dataset_id: {dataset_id}")


def get_default_budgets(num_routed_experts: int, small_scale: bool = False) -> List[int]:
    if num_routed_experts == 8:
        if small_scale:
            return [2, 4]
        else:
            return [1, 2, 4, 6, 8]
    else:
        if small_scale:
            return [8, 16]
        else:
            return [4, 8, 16, 24, 32, 48, 64]


def profile_predictors(
    adapter: RoutingDatasetAdapter,
    num_moe_stages: int,
    num_routed_experts: int,
    top_k: int,
    horizon: int
) -> Dict[str, Any]:
    """
    Profiles training records to initialize CSTP and ST-MoE offline state.
    Uses vectorized tensor accumulation for maximum efficiency.
    """
    marginal_counts, spatial_counts, temporal_counts = adapter.profile_tensors(
        horizon=horizon,
        split="train",
        num_moe_stages=num_moe_stages,
        num_routed_experts=num_routed_experts
    )

    # Initialize CSTP
    cstp_prof = CSTPPredictor(
        num_moe_stages=num_moe_stages,
        num_routed_experts=num_routed_experts,
        top_k=top_k,
        alpha=0.5,
        smoothing_lambda=1.0,
        prior_epsilon=0.1,
        online_adaptation=False,
        prediction_horizon=horizon
    )
    cstp_prof.stage_marginal_counts = marginal_counts
    cstp_prof.spatial_counts = spatial_counts
    cstp_prof.temporal_counts = temporal_counts

    # Initialize ST-MoE Paper
    st_moe_prof = STMoEPaperPredictor(
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
                    st_moe_prof.cct[stage][u][v] = 2
                    stored += 1

    # Initialize ST-MoE Legacy
    st_moe_legacy = STMoEPredictor(
        num_moe_stages=num_moe_stages,
        num_routed_experts=num_routed_experts,
        top_k=top_k,
        cct_capacity=top_k
    )
    st_moe_legacy.cct = copy.deepcopy(st_moe_prof.cct)

    # Initialize Frequency Baseline
    freq_prof = FrequencyPredictor(
        num_moe_stages=num_moe_stages,
        num_routed_experts=num_routed_experts
    )
    freq_prof.frequency_counts = marginal_counts.astype(np.int64)

    return {
        "cstp": cstp_prof,
        "st_moe_paper": st_moe_prof,
        "st_moe_legacy": st_moe_legacy,
        "frequency": freq_prof
    }


def create_predictor_instance(
    name: str,
    regime: str,
    profiled_state: Dict[str, Any],
    num_moe_stages: int,
    num_routed_experts: int,
    top_k: int,
    horizon: int,
    seed: int = 42
) -> BasePredictor:
    """
    Creates a fresh predictor instance for the specified configuration and regime.
    """
    if name == "cstp":
        if regime == "cold_start":
            pred = CSTPPredictor(
                num_moe_stages=num_moe_stages,
                num_routed_experts=num_routed_experts,
                top_k=top_k,
                alpha=0.5,
                smoothing_lambda=1.0,
                prior_epsilon=0.1,
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

    if name == "st_moe_legacy":
        base_p = copy.deepcopy(profiled_state["st_moe_legacy"])
        return STMoELegacyWrapper(base_p)

    if name == "frequency_only":
        return copy.deepcopy(profiled_state["frequency"])

    if name == "temporal_only":
        cstp_copy = copy.deepcopy(profiled_state["cstp"])
        if regime == "online":
            cstp_copy.online_adaptation = True
        else:
            cstp_copy.online_adaptation = False
        return TemporalOnlyPredictor(cstp_copy)

    if name == "spatial_only":
        cstp_copy = copy.deepcopy(profiled_state["cstp"])
        if regime == "online":
            cstp_copy.online_adaptation = True
        else:
            cstp_copy.online_adaptation = False
        return SpatialOnlyPredictor(cstp_copy)

    if name == "random":
        return RandomPredictor(num_routed_experts=num_routed_experts, seed=seed)

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

    raise ValueError(f"Unknown predictor name: {name}")


def evaluate_dataset(
    dataset_id: str,
    small_scale: bool = False,
    horizons: Optional[List[int]] = None,
    regimes: Optional[List[str]] = None,
    max_test_units: Optional[int] = None,
    max_tokens_per_unit: Optional[int] = None
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"EVALUATING DATASET: {dataset_id}")
    print(f"=======================================================")

    adapter = get_dataset_adapter(dataset_id)
    adapter.load()
    meta = adapter.get_metadata()

    num_stages = meta.get("num_moe_stages", 57)
    num_experts = meta.get("num_routed_experts", 256)
    top_k = meta.get("top_k", 8)

    if horizons is None:
        if dataset_id == "moe_suite":
            eval_horizons = [2]
        else:
            if dataset_id == "moe_anywhere":
                if small_scale:
                    eval_horizons = [1]
                else:
                    eval_horizons = [1, 2]
            else:
                eval_horizons = [1]
    else:
        eval_horizons = horizons

    if regimes is None:
        if small_scale:
            eval_regimes = ["frozen", "online"]
        else:
            eval_regimes = ["frozen", "online", "cold_start"]
    else:
        eval_regimes = regimes

    budgets = get_default_budgets(num_experts, small_scale=small_scale)

    # Predictor configurations to test
    if small_scale:
        predictor_names = [
            ("cstp", "online"),
            ("cstp", "frozen"),
            ("st_moe_paper", "online"),
            ("st_moe_paper", "frozen"),
            ("random", "frozen")
        ]
    else:
        predictor_names = [
            ("cstp", "online"),
            ("cstp", "frozen"),
            ("st_moe_paper", "online"),
            ("st_moe_paper", "frozen"),
            ("st_moe_legacy", "online"),
            ("frequency_only", "frozen"),
            ("temporal_only", "frozen"),
            ("spatial_only", "frozen"),
            ("random", "frozen"),
            ("cstp_st_moe_hybrid", "online")
        ]
        # Add cold-start variants
        if "cold_start" in eval_regimes:
            predictor_names.append(("cstp", "cold_start"))
            predictor_names.append(("st_moe_paper", "cold_start"))

    out_base = os.path.join(project_root, "results", "endsem_phase_c", dataset_id)
    os.makedirs(out_base, exist_ok=True)

    all_per_pred_rows = []
    all_aggregate_rows = []
    all_learning_curve_rows = []

    for h in eval_horizons:
        print(f"\n--- Profiling predictors for Horizon H={h} ---")
        profiled_state = profile_predictors(
            adapter=adapter,
            num_moe_stages=num_stages,
            num_routed_experts=num_experts,
            top_k=top_k,
            horizon=h
        )

        # Collect test records
        print(f"Collecting test prediction sequence for Horizon H={h}...")
        test_records = []
        for rec in adapter.iter_predictions(horizon=h, split="test"):
            test_records.append(rec)

        # Group test records by unit (request_id or prompt_id)
        units_map: Dict[Any, List[PredictionRecord]] = {}
        for r in test_records:
            if r.request_id not in units_map:
                units_map[r.request_id] = []
            units_map[r.request_id].append(r)

        sorted_units = sorted(list(units_map.keys()), key=lambda x: str(x))
        if max_test_units is not None:
            sorted_units = sorted_units[:max_test_units]

        print(f"Evaluating {len(sorted_units)} test units across {len(predictor_names)} predictors and {len(budgets)} budgets.")

        for pred_type, regime in predictor_names:
            if regime not in eval_regimes:
                continue

            for b in budgets:
                # Instantiate fresh predictor for budget independence
                pred_runner = create_predictor_instance(
                    name=pred_type,
                    regime=regime,
                    profiled_state=profiled_state,
                    num_moe_stages=num_stages,
                    num_routed_experts=num_experts,
                    top_k=top_k,
                    horizon=h,
                    seed=42
                )
                pred_label = getattr(pred_runner, "name", f"{pred_type}_{regime}")

                hits_list = []
                prec_list = []
                rec_list = []
                f1_list = []
                hit_rate_list = []

                # Group evaluation by unit
                for unit_id in sorted_units:
                    pred_runner.reset(unit_id)
                    unit_recs = units_map[unit_id]

                    # Filter by max tokens per unit if configured
                    if max_tokens_per_unit is not None:
                        unit_tokens = sorted(list(set(r.token_id for r in unit_recs)))[:max_tokens_per_unit]
                        token_set = set(unit_tokens)
                        eval_recs = [r for r in unit_recs if r.token_id in token_set]
                    else:
                        eval_recs = unit_recs

                    for rec in eval_recs:
                        # 1. Predict strictly before revealing target
                        preds = pred_runner.predict(
                            request_id=rec.request_id,
                            target_stage_id=rec.target_stage_id,
                            current_source_experts=rec.source_experts,
                            budget=b,
                            token_id=rec.token_id,
                            source_stage_id=rec.source_stage_id
                        )

                        # 2. Reveal target and evaluate
                        p_set = set(preds)
                        a_set = set(rec.actual_target_experts)
                        h_count = len(p_set.intersection(a_set))

                        p_len = len(p_set)
                        if p_len > 0:
                            prec = float(h_count) / float(p_len)
                        else:
                            prec = 0.0

                        a_len = len(a_set)
                        if a_len > 0:
                            recall = float(h_count) / float(a_len)
                        else:
                            recall = 0.0

                        if (prec + recall) > 0.0:
                            f1 = (2.0 * prec * recall) / (prec + recall)
                        else:
                            f1 = 0.0

                        if h_count > 0:
                            hr = 1.0
                        else:
                            hr = 0.0

                        hits_list.append(h_count)
                        prec_list.append(prec)
                        rec_list.append(recall)
                        f1_list.append(f1)
                        hit_rate_list.append(hr)

                        # Record per-prediction result
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

                        # 3. Update predictor state strictly after prediction is recorded
                        pred_runner.update(
                            request_id=rec.request_id,
                            target_stage_id=rec.target_stage_id,
                            source_experts=rec.source_experts,
                            actual_target_experts=rec.actual_target_experts,
                            token_id=rec.token_id,
                            source_stage_id=rec.source_stage_id
                        )

                # Compute aggregate metrics for this predictor and budget
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

    # Save CSV tables
    df_agg = pd.DataFrame(all_aggregate_rows)
    df_agg.to_csv(os.path.join(out_base, "aggregate_metrics.csv"), index=False)
    df_agg.to_csv(os.path.join(out_base, "budget_sweep.csv"), index=False)

    df_per_pred = pd.DataFrame(all_per_pred_rows)
    df_per_pred.to_csv(os.path.join(out_base, "per_prediction_results.csv"), index=False)

    # Save learning curve by token position (for budget = top_k)
    df_topk_budget = df_per_pred[df_per_pred["budget"] == top_k]
    if len(df_topk_budget) > 0:
        grp = df_topk_budget.groupby(["predictor", "learning_mode", "token_id"])["recall"].mean().reset_index()
        grp.to_csv(os.path.join(out_base, "online_learning_curve.csv"), index=False)

    print(f"\nSaved results for {dataset_id} under {out_base}/")
    return {
        "aggregate": all_aggregate_rows,
        "total_predictions": len(all_per_pred_rows)
    }


def main():
    parser = argparse.ArgumentParser(description="Run Phase C Online vs Frozen Benchmarking.")
    parser.add_argument("--dataset", type=str, default="all", choices=[
        "all", "deepseek_random", "duplex_a100", "correlated_synthetic", "moe_suite", "moe_anywhere"
    ])
    parser.add_argument("--small_scale", action="store_true", help="Run small-scale test for fast validation")
    args = parser.parse_args()

    if args.dataset == "all":
        datasets = ["deepseek_random", "duplex_a100", "correlated_synthetic", "moe_anywhere", "moe_suite"]
    else:
        datasets = [args.dataset]

    for d in datasets:
        max_u = None
        max_tok = None
        if args.small_scale:
            max_u = 2
            max_tok = 5
        evaluate_dataset(
            dataset_id=d,
            small_scale=args.small_scale,
            max_test_units=max_u,
            max_tokens_per_unit=max_tok
        )


if __name__ == "__main__":
    main()
