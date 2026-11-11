"""
Phase D Ablation and Sensitivity Study Orchestrator.
Systematically evaluates calibration temperature, score normalization schemes,
reliability objectives, EMA momentum beta, softmax weighting tau,
and cold-start initialization sensitivity on validation splits.
"""

import os
import sys
import copy
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.predictor.cstp_predictor import CSTPPredictor
from src.predictor.st_moe_paper import STMoEPaperPredictor
from src.predictor.acsh_predictor import ACSHPredictor
from src.predictor.adapters.base_adapter import PredictionRecord
from scripts.phase_d.run_phase_d import get_dataset_adapter, profile_predictors


def run_eval_loop(
    pred_instance: ACSHPredictor,
    eval_records: List[PredictionRecord],
    top_k: int,
    budget: int
) -> Dict[str, float]:
    units: List[Any] = []
    seen: Set[Any] = set()
    for r in eval_records:
        if r.request_id not in seen:
            seen.add(r.request_id)
            units.append(r.request_id)

    hits_list: List[int] = []
    rec_list: List[float] = []
    prec_list: List[float] = []
    f1_list: List[float] = []
    hit_rate_list: List[float] = []

    for uid in units:
        pred_instance.reset(uid)
        u_records = [r for r in eval_records if r.request_id == uid]

        for rec in u_records:
            preds = pred_instance.predict(
                request_id=rec.request_id,
                target_stage_id=rec.target_stage_id,
                current_source_experts=rec.source_experts,
                budget=budget,
                token_id=rec.token_id,
                source_stage_id=rec.source_stage_id
            )

            p_set = set(preds)
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

            pred_instance.update(
                request_id=rec.request_id,
                target_stage_id=rec.target_stage_id,
                source_experts=rec.source_experts,
                actual_target_experts=rec.actual_target_experts,
                token_id=rec.token_id,
                source_stage_id=rec.source_stage_id
            )

    return {
        "macro_recall": round(float(np.mean(rec_list)), 4),
        "macro_precision": round(float(np.mean(prec_list)), 4),
        "macro_f1": round(float(np.mean(f1_list)), 4),
        "hit_rate": round(float(np.mean(hit_rate_list)), 4),
        "total_evals": len(hits_list)
    }


def run_all_ablations():
    print("=" * 80)
    print("STARTING PHASE D ABLATION AND SENSITIVITY SUITE")
    print("=" * 80)

    out_dir = os.path.join(project_root, "results", "endsem_phase_d", "ablations")
    os.makedirs(out_dir, exist_ok=True)

    # Use correlated_synthetic validation split
    adapter = get_dataset_adapter("correlated_synthetic")
    adapter.load()
    meta = adapter.get_metadata()
    num_stages = 57
    num_experts = 256
    top_k = 8
    budget = 8

    # Profile on training split
    prof_state = profile_predictors(adapter, num_stages, num_experts, top_k, horizon=1)

    # Use validation slice of train (e.g. requests 10, 11, 12)
    val_records = [
        r for r in adapter.iter_predictions(horizon=1, split="train")
        if int(r.request_id) in [10, 11, 12]
    ]
    print(f"Loaded {len(val_records)} validation records from correlated_synthetic.")

    # -------------------------------------------------------------
    # 1. Calibration and Normalization Ablation
    # -------------------------------------------------------------
    print("\n1. Running Calibration and Score Normalization Ablation...")
    norm_configs = [
        ("softmax", 0.5),
        ("softmax", 1.0),
        ("softmax", 2.0),
        ("rank", 1.0),
        ("max", 1.0),
        ("raw", 1.0)
    ]
    norm_rows = []
    for norm_name, temp in norm_configs:
        c_copy = copy.deepcopy(prof_state["cstp"])
        s_copy = copy.deepcopy(prof_state["st_moe_paper"])
        acsh = ACSHPredictor(
            c_copy, s_copy,
            variant="stage_aware",
            normalization=norm_name,
            temp_c=temp,
            temp_m=temp
        )
        res = run_eval_loop(acsh, val_records, top_k, budget)
        res["normalization"] = norm_name
        res["temperature"] = temp
        norm_rows.append(res)
        print(f"  [{norm_name} (T={temp})] Recall: {res['macro_recall']:.4f}, Prec: {res['macro_precision']:.4f}, F1: {res['macro_f1']:.4f}")

    df_norm = pd.DataFrame(norm_rows)
    df_norm.to_csv(os.path.join(out_dir, "calibration_ablation.csv"), index=False)

    # -------------------------------------------------------------
    # 2. Reliability Objective Function Ablation
    # -------------------------------------------------------------
    print("\n2. Running Reliability Objective Ablation...")
    obj_configs = ["recall", "hit_rate", "f1"]
    obj_rows = []
    for obj in obj_configs:
        c_copy = copy.deepcopy(prof_state["cstp"])
        s_copy = copy.deepcopy(prof_state["st_moe_paper"])
        acsh = ACSHPredictor(
            c_copy, s_copy,
            variant="stage_aware",
            reliability_objective=obj
        )
        res = run_eval_loop(acsh, val_records, top_k, budget)
        res["objective"] = obj
        obj_rows.append(res)
        print(f"  [Objective: {obj}] Recall: {res['macro_recall']:.4f}, Prec: {res['macro_precision']:.4f}, F1: {res['macro_f1']:.4f}")

    df_obj = pd.DataFrame(obj_rows)
    df_obj.to_csv(os.path.join(out_dir, "objective_ablation.csv"), index=False)

    # -------------------------------------------------------------
    # 3. EMA Momentum Beta Sensitivity Sweep
    # -------------------------------------------------------------
    print("\n3. Running Beta Sensitivity Sweep...")
    beta_configs = [0.5, 0.7, 0.8, 0.9, 0.95]
    beta_rows = []
    for b_val in beta_configs:
        c_copy = copy.deepcopy(prof_state["cstp"])
        s_copy = copy.deepcopy(prof_state["st_moe_paper"])
        acsh = ACSHPredictor(
            c_copy, s_copy,
            variant="stage_aware",
            beta=b_val
        )
        res = run_eval_loop(acsh, val_records, top_k, budget)
        res["beta"] = b_val
        res["update_weight_alpha"] = round(1.0 - b_val, 4)
        beta_rows.append(res)
        print(f"  [Beta: {b_val}] Recall: {res['macro_recall']:.4f}, Prec: {res['macro_precision']:.4f}, F1: {res['macro_f1']:.4f}")

    df_beta = pd.DataFrame(beta_rows)
    df_beta.to_csv(os.path.join(out_dir, "beta_sensitivity.csv"), index=False)

    # -------------------------------------------------------------
    # 4. Softmax Weighting Tau Sensitivity Sweep
    # -------------------------------------------------------------
    print("\n4. Running Tau Sensitivity Sweep...")
    tau_configs = [0.05, 0.1, 0.2, 0.5, 1.0]
    tau_rows = []
    for t_val in tau_configs:
        c_copy = copy.deepcopy(prof_state["cstp"])
        s_copy = copy.deepcopy(prof_state["st_moe_paper"])
        acsh = ACSHPredictor(
            c_copy, s_copy,
            variant="stage_aware",
            tau=t_val
        )
        res = run_eval_loop(acsh, val_records, top_k, budget)
        res["tau"] = t_val
        tau_rows.append(res)
        print(f"  [Tau: {t_val}] Recall: {res['macro_recall']:.4f}, Prec: {res['macro_precision']:.4f}, F1: {res['macro_f1']:.4f}")

    df_tau = pd.DataFrame(tau_rows)
    df_tau.to_csv(os.path.join(out_dir, "tau_sensitivity.csv"), index=False)

    # -------------------------------------------------------------
    # 5. Cold-Start Initial Reliability Sensitivity
    # -------------------------------------------------------------
    print("\n5. Running Cold-Start Initial Reliability Sensitivity...")
    init_configs = [0.1, 0.3, 0.5, 0.7, 0.9]
    init_rows = []
    for init_val in init_configs:
        c_cold = CSTPPredictor(
            num_moe_stages=num_stages,
            num_routed_experts=num_experts,
            top_k=top_k,
            alpha=0.5,
            online_adaptation=True,
            prediction_horizon=1
        )
        s_cold = STMoEPaperPredictor(
            num_moe_stages=num_stages,
            num_routed_experts=num_experts,
            top_k=top_k,
            cct_capacity=top_k,
            confidence_threshold=2,
            online_adaptation=True,
            prediction_horizon=1
        )
        acsh = ACSHPredictor(
            c_cold, s_cold,
            variant="stage_aware",
            initial_reliability=init_val
        )
        res = run_eval_loop(acsh, val_records, top_k, budget)
        res["initial_reliability"] = init_val
        init_rows.append(res)
        print(f"  [Initial R: {init_val}] Recall: {res['macro_recall']:.4f}, Prec: {res['macro_precision']:.4f}, F1: {res['macro_f1']:.4f}")

    df_init_r = pd.DataFrame(init_rows)
    df_init_r.to_csv(os.path.join(out_dir, "cold_start_sensitivity.csv"), index=False)

    print("\n" + "=" * 80)
    print("PHASE D ABLATION AND SENSITIVITY SUITE COMPLETED!")
    print(f"Saved all results under {out_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    run_all_ablations()

