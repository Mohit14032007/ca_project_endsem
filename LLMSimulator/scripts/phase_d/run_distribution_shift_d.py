"""
Phase D Distribution Shift Experiment.
Evaluates predictor adaptation and recovery across an abrupt regime change:
Cluster A (experts 0-127) for tokens 0-49 -> Cluster B (experts 128-255) for tokens 50-99.
Compares CSTP, ST-MoE Paper, Fixed Hybrid, and ACSH variants (Global, Stage-Aware, Fast Recovery).
"""

import os
import sys
import copy
import json
import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple

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
    CSTPSTMoEHybridPredictor
)


def generate_controlled_shift_trace(
    num_tokens: int = 100,
    shift_token: int = 50,
    num_stages: int = 57,
    num_experts: int = 256,
    top_k: int = 8,
    seed: int = 42
) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    trace_events: List[Dict[str, Any]] = []

    cluster_a = list(range(0, 128))
    cluster_b = list(range(128, 256))

    for tok in range(num_tokens):
        if tok < shift_token:
            active_pool = cluster_a
            phase = "pre_shift_cluster_A"
        else:
            active_pool = cluster_b
            phase = "post_shift_cluster_B"

        for stage in range(num_stages):
            chosen = rng.choice(active_pool, size=top_k, replace=False).tolist()
            chosen.sort()
            trace_events.append({
                "request_id": 1,
                "token_id": tok,
                "moe_stage_id": stage,
                "phase": phase,
                "experts": chosen
            })

    return trace_events


def run_distribution_shift_experiment():
    print("=" * 80)
    print("STARTING PHASE D CONTROLLED DISTRIBUTION SHIFT EXPERIMENT")
    print("=" * 80)

    out_dir = os.path.join(project_root, "results", "endsem_phase_d", "distribution_shift")
    os.makedirs(out_dir, exist_ok=True)

    num_stages = 57
    num_experts = 256
    top_k = 8
    budget = 8
    horizon = 1

    trace_events = generate_controlled_shift_trace(
        num_tokens=100,
        shift_token=50,
        num_stages=num_stages,
        num_experts=num_experts,
        top_k=top_k,
        seed=42
    )

    # Offline profile on pre-shift Cluster A
    pre_shift_events = [e for e in trace_events if e["token_id"] < 25]
    cstp_profile = CSTPPredictor(num_stages, num_experts, top_k, alpha=0.5, online_adaptation=True)
    st_profile = STMoEPaperPredictor(num_stages, num_experts, top_k, cct_capacity=top_k, confidence_threshold=2)

    for tok_idx in range(25):
        tok_events = [e for e in pre_shift_events if e["token_id"] == tok_idx]
        for s in range(num_stages - 1):
            src_e = tok_events[s]["experts"]
            tgt_e = tok_events[s + 1]["experts"]
            for u in src_e:
                for v in tgt_e:
                    cstp_profile.spatial_counts[s, u, v] += 1.0
            st_profile.update(1, s + 1, src_e, tgt_e, token_id=tok_idx, source_stage_id=s)

    freq_profile = FrequencyPredictor(num_stages, num_experts)
    for e in pre_shift_events:
        for exp in e["experts"]:
            freq_profile.frequency_counts[e["moe_stage_id"], exp] += 1

    predictors_to_test = [
        ("cstp_frozen", CSTPWrapper(copy.deepcopy(cstp_profile))),
        ("cstp_online", CSTPWrapper(copy.deepcopy(cstp_profile))),
        ("st_moe_paper_frozen", copy.deepcopy(st_profile)),
        ("st_moe_paper_online", copy.deepcopy(st_profile)),
        ("cstp_st_moe_hybrid", CSTPSTMoEHybridPredictor(copy.deepcopy(cstp_profile), copy.deepcopy(st_profile))),
        ("acsh_global", ACSHPredictor(copy.deepcopy(cstp_profile), copy.deepcopy(st_profile), variant="global", name="acsh_global")),
        ("acsh_stage_aware", ACSHPredictor(copy.deepcopy(cstp_profile), copy.deepcopy(st_profile), variant="stage_aware", name="acsh_stage_aware")),
        ("acsh_fast_recovery", ACSHPredictor(copy.deepcopy(cstp_profile), copy.deepcopy(st_profile), variant="fast_recovery", name="acsh_fast_recovery")),
        ("frequency_only", copy.deepcopy(freq_profile)),
        ("temporal_only", TemporalOnlyPredictor(copy.deepcopy(cstp_profile))),
        ("random", RandomPredictor(num_experts, seed=42))
    ]

    # Set adaptation flags
    for p_name, p_inst in predictors_to_test:
        if p_name == "cstp_frozen":
            p_inst.inner.online_adaptation = False
        if p_name == "st_moe_paper_frozen":
            p_inst.online_adaptation = False

    per_token_metrics: List[Dict[str, Any]] = []
    detection_rows: List[Dict[str, Any]] = []

    for p_name, p_inst in predictors_to_test:
        p_inst.reset(1)

        token_recalls: Dict[int, List[float]] = {}
        for tok in range(100):
            token_recalls[tok] = []

        for tok in range(100):
            tok_events = [e for e in trace_events if e["token_id"] == tok]
            phase_label = tok_events[0]["phase"]

            for s in range(num_stages - 1):
                src_e = tok_events[s]["experts"]
                tgt_e = tok_events[s + 1]["experts"]

                preds = p_inst.predict(
                    request_id=1,
                    target_stage_id=s + 1,
                    current_source_experts=src_e,
                    budget=budget,
                    token_id=tok,
                    source_stage_id=s
                )

                hits = len(set(preds).intersection(set(tgt_e)))
                rec = float(hits) / float(top_k)
                token_recalls[tok].append(rec)

                p_inst.update(
                    request_id=1,
                    target_stage_id=s + 1,
                    source_experts=src_e,
                    actual_target_experts=tgt_e,
                    token_id=tok,
                    source_stage_id=s
                )

            mean_rec = float(np.mean(token_recalls[tok]))
            per_token_metrics.append({
                "predictor": p_name,
                "token_id": tok,
                "phase": phase_label,
                "budget": budget,
                "mean_recall": round(mean_rec, 4)
            })

        # Capture shift detections if fast recovery
        if hasattr(p_inst, "shift_detections"):
            for sd in p_inst.shift_detections:
                sd_row = dict(sd)
                sd_row["predictor"] = p_name
                detection_rows.append(sd_row)

        print(f"Finished shift test for: {p_name}")

    df_metrics = pd.DataFrame(per_token_metrics)
    metrics_csv = os.path.join(out_dir, "distribution_shift_metrics.csv")
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"Saved shift metrics to: {metrics_csv}")

    # Calculate recovery metrics
    recovery_rows = []
    for p_name, grp in df_metrics.groupby("predictor"):
        grp_sorted = grp.sort_values("token_id")
        pre_shift = grp_sorted[grp_sorted["token_id"] < 50]["mean_recall"].mean()
        drop_val = grp_sorted[grp_sorted["token_id"].isin([50, 51, 52])]["mean_recall"].mean()
        late_val = grp_sorted[grp_sorted["token_id"] >= 75]["mean_recall"].mean()
        drop_mag = pre_shift - drop_val

        # Recovery time: tokens after 50 to reach 90% of late stable performance
        post_shift = grp_sorted[grp_sorted["token_id"] >= 50]
        target_val = 0.90 * late_val
        recovery_tokens = -1

        for idx, row in post_shift.iterrows():
            if row["mean_recall"] >= target_val:
                recovery_tokens = int(row["token_id"]) - 50
                break

        recovery_rows.append({
            "predictor": p_name,
            "budget": budget,
            "pre_shift_recall": round(pre_shift, 4),
            "immediate_drop_recall": round(drop_val, 4),
            "drop_magnitude": round(drop_mag, 4),
            "post_shift_late_recall": round(late_val, 4),
            "recovery_tokens_to_90pct": recovery_tokens,
            "adaptation_delta": round(late_val - drop_val, 4)
        })

    df_rec = pd.DataFrame(recovery_rows)
    rec_csv = os.path.join(out_dir, "distribution_shift_recovery.csv")
    df_rec.to_csv(rec_csv, index=False)
    print(f"Saved recovery metrics to: {rec_csv}")

    # Save detections and false alarms
    df_det = pd.DataFrame(detection_rows)
    det_csv = os.path.join(out_dir, "distribution_shift_detection.csv")
    df_det.to_csv(det_csv, index=False)

    # False alarms: detections that occurred before token 50
    if len(df_det) > 0:
        df_fa = df_det[df_det["token_id"] < 50]
    else:
        df_fa = pd.DataFrame(columns=["predictor", "token_id", "target_stage_id", "mean_error"])
    fa_csv = os.path.join(out_dir, "distribution_shift_false_alarms.csv")
    df_fa.to_csv(fa_csv, index=False)
    print(f"Saved detections to: {det_csv} (False alarms: {len(df_fa)})")


if __name__ == "__main__":
    run_distribution_shift_experiment()

