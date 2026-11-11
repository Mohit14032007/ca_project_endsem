"""
Controlled Distribution Shift Experiment for Online vs Frozen Expert Predictors.
Evaluates adaptation speed when routing distribution abruptly shifts from Cluster A to Cluster B.
"""

import os
import sys
import copy
import math
from typing import Any, Dict, List, Set, Tuple
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.predictor.cstp_predictor import CSTPPredictor
from src.predictor.st_moe_paper import STMoEPaperPredictor
from src.predictor.base_predictor import (
    BasePredictor,
    CSTPWrapper,
    RandomPredictor,
    FrequencyPredictor,
    TemporalOnlyPredictor
)


def generate_distribution_shift_trace(
    num_tokens_p1: int = 50,
    num_tokens_p2: int = 50,
    num_stages: int = 32,
    num_experts: int = 256,
    top_k: int = 8,
    cluster_size: int = 16,
    cluster_a: int = 2,
    cluster_b: int = 9,
    affinity: float = 0.8,
    temporal_affinity: float = 0.4,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generates a controlled 100-token trace with an abrupt distribution shift at token 50.
    Tokens 0..49: Cluster A (experts 32..47) is dominant.
    Tokens 50..99: Cluster B (experts 144..159) is dominant.
    """
    np.random.seed(seed)
    zipf_weights = 1.0 / np.power(np.arange(1, num_experts + 1), 0.8)
    base_probs = zipf_weights / np.sum(zipf_weights)

    total_tokens = num_tokens_p1 + num_tokens_p2
    events = []
    prev_routed: Dict[int, List[int]] = {}

    for tok_idx in range(total_tokens):
        if tok_idx < num_tokens_p1:
            active_cluster = cluster_a
            phase_name = "pre_shift_cluster_A"
        else:
            active_cluster = cluster_b
            phase_name = "post_shift_cluster_B"

        c_start = active_cluster * cluster_size
        c_end = c_start + cluster_size

        for stage_idx in range(num_stages):
            prob = base_probs.copy()
            # Boost active cluster
            prob[c_start:c_end] += affinity / float(cluster_size)

            # Temporal persistence
            if stage_idx in prev_routed:
                for pe in prev_routed[stage_idx]:
                    prob[pe] += temporal_affinity

            prob = prob / np.sum(prob)

            chosen = np.random.choice(
                num_experts,
                size=top_k,
                replace=False,
                p=prob
            )
            chosen_list = [int(x) for x in chosen]
            prev_routed[stage_idx] = chosen_list

            if stage_idx > 0:
                source_exps = prev_routed.get(stage_idx - 1, [])
            else:
                source_exps = []

            events.append({
                "request_id": 1,
                "token_id": tok_idx,
                "token_phase": phase_name,
                "source_stage_id": stage_idx - 1,
                "target_stage_id": stage_idx,
                "source_experts": source_exps,
                "actual_target_experts": chosen_list
            })

    return events


def run_distribution_shift_experiment() -> None:
    print("=" * 70)
    print("PHASE C: CONTROLLED DISTRIBUTION SHIFT EXPERIMENT")
    print("=" * 70)

    out_dir = os.path.join(project_root, "results", "endsem_phase_c", "distribution_shift")
    os.makedirs(out_dir, exist_ok=True)

    num_stages = 32
    num_experts = 256
    top_k = 8
    budget = 8
    num_p1 = 50
    num_p2 = 50

    # 1. Generate shift trace
    print("Generating distribution shift synthetic sequence (100 tokens, shift at token 50)...")
    events = generate_distribution_shift_trace(
        num_tokens_p1=num_p1,
        num_tokens_p2=num_p2,
        num_stages=num_stages,
        num_experts=num_experts,
        top_k=top_k,
        affinity=0.8,
        temporal_affinity=0.4,
        seed=42
    )

    # 2. Profile offline state strictly on Phase 1 (Cluster A)
    class DummyRecord:
        def __init__(self, req, tok, stage, exp):
            self.request_id = req
            self.token_id = tok
            self.moe_stage_id = stage
            self.expert_id = exp

    phase1_events = [e for e in events if e["token_id"] < num_p1]
    profile_objs = []
    for e in phase1_events:
        for exp in e["actual_target_experts"]:
            profile_objs.append(DummyRecord(e["request_id"], e["token_id"], e["target_stage_id"], exp))

    # Base profiled CSTP
    cstp_base = CSTPPredictor(
        num_moe_stages=num_stages,
        num_routed_experts=num_experts,
        top_k=top_k,
        alpha=0.5,
        smoothing_lambda=1.0,
        prior_epsilon=0.1,
        online_adaptation=False
    )
    cstp_base.fit_profile(profile_objs)

    # Base profiled ST-MoE
    st_moe_base = STMoEPaperPredictor(
        num_moe_stages=num_stages,
        num_routed_experts=num_experts,
        top_k=top_k,
        cct_capacity=top_k,
        confidence_threshold=2,
        online_adaptation=True
    )
    st_moe_base.build_cct_from_profile(profile_objs)

    # Base profiled Frequency
    freq_base = FrequencyPredictor(
        num_moe_stages=num_stages,
        num_routed_experts=num_experts
    )
    freq_base.fit_profile(profile_objs)

    # 3. Instantiate evaluated predictors
    predictors: Dict[str, BasePredictor] = {}

    # CSTP Online
    p_cstp_on = copy.deepcopy(cstp_base)
    p_cstp_on.online_adaptation = True
    predictors["cstp_online"] = CSTPWrapper(p_cstp_on)
    predictors["cstp_online"].name = "cstp_online"

    # CSTP Frozen
    p_cstp_frz = copy.deepcopy(cstp_base)
    p_cstp_frz.online_adaptation = False
    predictors["cstp_frozen"] = CSTPWrapper(p_cstp_frz)
    predictors["cstp_frozen"].name = "cstp_frozen"

    # ST-MoE Online
    p_st_on = copy.deepcopy(st_moe_base)
    p_st_on.online_adaptation = True
    p_st_on.name = "st_moe_paper_online"
    predictors["st_moe_paper_online"] = p_st_on

    # ST-MoE Frozen
    p_st_frz = copy.deepcopy(st_moe_base)
    p_st_frz.online_adaptation = False
    p_st_frz.name = "st_moe_paper_frozen"
    predictors["st_moe_paper_frozen"] = p_st_frz

    # Baselines
    predictors["frequency_only"] = copy.deepcopy(freq_base)
    p_temp = copy.deepcopy(cstp_base)
    p_temp.online_adaptation = True
    predictors["temporal_only"] = TemporalOnlyPredictor(p_temp)
    predictors["random"] = RandomPredictor(num_routed_experts=num_experts, seed=42)

    per_token_rows = []
    shift_summary_rows = []

    print("\nRunning online evaluation through distribution shift...")
    for pred_name, runner in predictors.items():
        runner.reset(1)
        token_recalls: Dict[int, List[float]] = {}
        for t in range(num_p1 + num_p2):
            token_recalls[t] = []

        for ev in events:
            tok = ev["token_id"]
            preds = runner.predict(
                request_id=ev["request_id"],
                target_stage_id=ev["target_stage_id"],
                current_source_experts=ev["source_experts"],
                budget=budget,
                token_id=tok,
                source_stage_id=ev["source_stage_id"]
            )
            act = ev["actual_target_experts"]
            hits = len(set(preds).intersection(set(act)))
            rec = float(hits) / float(len(act))
            token_recalls[tok].append(rec)

            runner.update(
                request_id=ev["request_id"],
                target_stage_id=ev["target_stage_id"],
                source_experts=ev["source_experts"],
                actual_target_experts=act,
                token_id=tok,
                source_stage_id=ev["source_stage_id"]
            )

        # Average recall per token
        avg_recalls = [float(np.mean(token_recalls[t])) for t in range(num_p1 + num_p2)]

        for t, r in enumerate(avg_recalls):
            if t < num_p1:
                ph = "pre_shift_cluster_A"
            else:
                ph = "post_shift_cluster_B"
            per_token_rows.append({
                "predictor": pred_name,
                "token_id": t,
                "phase": ph,
                "budget": budget,
                "mean_recall": round(r, 4)
            })

        # Calculate adaptation metrics
        pre_shift_recall = float(np.mean(avg_recalls[25:50]))
        immediate_drop_recall = float(np.mean(avg_recalls[50:55]))
        post_shift_late_recall = float(np.mean(avg_recalls[75:100]))
        drop_magnitude = pre_shift_recall - immediate_drop_recall

        # Recovery tokens: first token after shift where recall >= 0.90 * post_shift_late_recall
        target_rec = 0.90 * post_shift_late_recall
        recovery_tokens = -1
        for post_idx in range(50, 100):
            if avg_recalls[post_idx] >= target_rec:
                recovery_tokens = post_idx - 50
                break

        shift_summary_rows.append({
            "predictor": pred_name,
            "budget": budget,
            "pre_shift_recall": round(pre_shift_recall, 4),
            "immediate_drop_recall": round(immediate_drop_recall, 4),
            "drop_magnitude": round(drop_magnitude, 4),
            "post_shift_late_recall": round(post_shift_late_recall, 4),
            "recovery_tokens_to_90pct": recovery_tokens,
            "adaptation_delta": round(post_shift_late_recall - immediate_drop_recall, 4)
        })

        print(f"  [{pred_name:20s}] Pre-shift: {pre_shift_recall:.4f} -> Drop: {immediate_drop_recall:.4f} -> Recovered: {post_shift_late_recall:.4f} (Tokens: {recovery_tokens})")

    # Save CSV outputs
    df_metrics = pd.DataFrame(per_token_rows)
    df_metrics.to_csv(os.path.join(out_dir, "distribution_shift_metrics.csv"), index=False)

    df_recov = pd.DataFrame(shift_summary_rows)
    df_recov.to_csv(os.path.join(out_dir, "distribution_shift_recovery.csv"), index=False)

    print(f"\nSaved distribution shift results under {out_dir}/")


if __name__ == "__main__":
    run_distribution_shift_experiment()

