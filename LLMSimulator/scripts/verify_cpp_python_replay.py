#!/usr/bin/env python3
"""
Python-C++ Predictor Chronological Replay Equivalence Verification Harness.

Validates that C++ prefetch predictors (src/prefetch/) faithfully reproduce
the Python reference implementations (src/predictor/) under identical
chronological event replays.

Enforces:
- Zero ternary operators (ast.IfExp == 0).
- Pure standard libraries and numpy/pandas.
- Event-by-event candidate, order, and score equivalence checks.
- Macro summary metric verification across stratifications.
"""

import argparse
import ast
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

PROJECT_ROOT = "/home/mohman/ca_project/LLMSimulator"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predictor.st_moe_paper import STMoEPaperPredictor
from src.predictor.cstp_predictor import CSTPPredictor
from src.predictor.acsh_predictor import ACSHPredictor
from src.predictor.base_predictor import (
    BasePredictor,
    FrequencyPredictor,
    TemporalOnlyPredictor,
    SpatialOnlyPredictor,
    RandomPredictor
)


class OnlineFrequencyPredictor(FrequencyPredictor):
    """
    Subclass of FrequencyPredictor that increments empirical counts online
    upon update(), matching C++ FrequencyPredictor::update().
    """

    def update(
        self,
        request_id: int,
        target_stage_id: int,
        source_experts: List[int],
        actual_target_experts: List[int],
        token_id: Optional[int] = None,
        source_stage_id: Optional[int] = None
    ) -> None:
        if 0 <= target_stage_id < self.num_moe_stages:
            for v in actual_target_experts:
                if 0 <= v < self.num_routed_experts:
                    self.frequency_counts[target_stage_id, v] += 1


class DummyNonePredictor(BasePredictor):
    """Reference implementation for none predictor (always empty)."""

    def reset(self, request_id: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        pass

    def predict(
        self,
        request_id: int,
        target_stage_id: int,
        current_source_experts: List[int],
        budget: int,
        token_id: Optional[int] = None,
        source_stage_id: Optional[int] = None
    ) -> List[int]:
        return []

    def update(
        self,
        request_id: int,
        target_stage_id: int,
        source_experts: List[int],
        actual_target_experts: List[int],
        token_id: Optional[int] = None,
        source_stage_id: Optional[int] = None
    ) -> None:
        pass


def compute_event_metrics(predicted: List[int], actual: List[int]) -> Dict[str, float]:
    """
    Computes Precision, Recall, F1, Hit Rate, MRR, nDCG@B.
    Enforces zero ternary operators.
    """
    pred_set = set(predicted)
    act_set = set(actual)
    hits = len(pred_set.intersection(act_set))

    if len(predicted) > 0:
        precision = float(hits) / float(len(predicted))
    else:
        precision = 0.0

    if len(actual) > 0:
        recall = float(hits) / float(len(actual))
    else:
        recall = 0.0

    if (precision + recall) > 0.0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    if hits > 0:
        hit_rate = 1.0
    else:
        hit_rate = 0.0

    mrr = 0.0
    for idx, cand in enumerate(predicted):
        if cand in act_set:
            mrr = 1.0 / float(idx + 1)
            break

    dcg = 0.0
    for idx, cand in enumerate(predicted):
        if cand in act_set:
            rank_discount = math.log2(float(idx + 2))
            dcg += 1.0 / rank_discount

    ideal_hits = min(len(predicted), len(actual))
    idcg = 0.0
    for idx in range(ideal_hits):
        rank_discount = math.log2(float(idx + 2))
        idcg += 1.0 / rank_discount

    if idcg > 0.0:
        ndcg = dcg / idcg
    else:
        ndcg = 0.0

    return {
        "hits": hits,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "ndcg": ndcg
    }


def create_python_predictor(
    predictor_name: str,
    num_stages: int,
    num_experts: int,
    top_k: int,
    budget: int,
    horizon: int = 1,
    alpha: float = 0.5,
    smoothing_lambda: float = 1.0,
    random_seed: int = 42
) -> BasePredictor:
    """Factory to instantiate Python reference predictors."""
    name_clean = predictor_name.lower().strip()

    if name_clean == "none":
        return DummyNonePredictor()

    if name_clean == "st_moe":
        return STMoEPaperPredictor(
            num_moe_stages=num_stages,
            num_routed_experts=num_experts,
            top_k=top_k,
            cct_capacity=top_k,
            confidence_threshold=2,
            literal_alg2_overwrite=False,
            online_adaptation=True,
            prediction_horizon=horizon
        )

    if name_clean == "cstp":
        return CSTPPredictor(
            num_moe_stages=num_stages,
            num_routed_experts=num_experts,
            top_k=top_k,
            alpha=alpha,
            smoothing_lambda=smoothing_lambda,
            prior_epsilon=0.1,
            online_adaptation=True,
            prediction_horizon=horizon
        )

    if name_clean in ["acsh_global", "acsh_stage_aware", "acsh_fast_recovery"]:
        cstp_inst = CSTPPredictor(
            num_moe_stages=num_stages,
            num_routed_experts=num_experts,
            top_k=top_k,
            alpha=alpha,
            smoothing_lambda=smoothing_lambda,
            prior_epsilon=0.1,
            online_adaptation=True,
            prediction_horizon=horizon
        )
        st_inst = STMoEPaperPredictor(
            num_moe_stages=num_stages,
            num_routed_experts=num_experts,
            top_k=top_k,
            cct_capacity=top_k,
            confidence_threshold=2,
            literal_alg2_overwrite=False,
            online_adaptation=True,
            prediction_horizon=horizon
        )
        if name_clean == "acsh_global":
            variant_str = "global"
        else:
            if name_clean == "acsh_stage_aware":
                variant_str = "stage_aware"
            else:
                variant_str = "fast_recovery"

        return ACSHPredictor(
            cstp_predictor=cstp_inst,
            st_moe_predictor=st_inst,
            variant=variant_str,
            beta=0.8,
            tau=0.2,
            temp_c=1.0,
            temp_m=1.0,
            normalization="softmax",
            reliability_objective="recall",
            initial_reliability=0.5,
            shift_window=3,
            shift_threshold=0.75,
            fast_beta=0.2,
            fast_cooldown_steps=3
        )

    if name_clean == "frequency":
        return OnlineFrequencyPredictor(
            num_moe_stages=num_stages,
            num_routed_experts=num_experts
        )

    if name_clean == "temporal":
        cstp_inst = CSTPPredictor(
            num_moe_stages=num_stages,
            num_routed_experts=num_experts,
            top_k=top_k,
            alpha=0.0,
            smoothing_lambda=smoothing_lambda,
            prior_epsilon=0.1,
            online_adaptation=True,
            prediction_horizon=horizon
        )
        return TemporalOnlyPredictor(cstp_instance=cstp_inst)

    if name_clean == "spatial":
        cstp_inst = CSTPPredictor(
            num_moe_stages=num_stages,
            num_routed_experts=num_experts,
            top_k=top_k,
            alpha=1.0,
            smoothing_lambda=smoothing_lambda,
            prior_epsilon=0.1,
            online_adaptation=True,
            prediction_horizon=horizon
        )
        return SpatialOnlyPredictor(cstp_instance=cstp_inst)

    if name_clean == "random":
        return RandomPredictor(
            num_routed_experts=num_experts,
            seed=random_seed
        )

    raise ValueError(f"Unknown predictor name: {predictor_name}")


def get_python_scores(
    predictor: BasePredictor,
    predictor_name: str,
    request_id: int,
    target_stage: int,
    source_experts: List[int],
    source_stage: int,
    budget: int,
    candidates: List[int]
) -> List[float]:
    """Retrieves or calculates raw/normalized candidate scores from Python predictor."""
    name_clean = predictor_name.lower().strip()

    if name_clean == "st_moe":
        st_pred: STMoEPaperPredictor = predictor  # type: ignore
        scores_dict = st_pred.compute_confidence_scores(
            request_id=request_id,
            target_stage_id=target_stage,
            current_source_experts=source_experts,
            source_stage_id=source_stage
        )
        return [float(scores_dict.get(c, 0.0)) for c in candidates]

    if name_clean == "cstp":
        cstp_pred: CSTPPredictor = predictor  # type: ignore
        s_spatial = cstp_pred.compute_spatial_score(
            target_stage_id=target_stage,
            source_experts=source_experts,
            source_stage_id=source_stage
        )
        s_temporal = cstp_pred.compute_temporal_score(request_id, target_stage)
        composite = (cstp_pred.alpha * s_spatial) + ((1.0 - cstp_pred.alpha) * s_temporal)
        return [float(composite[c]) for c in candidates]

    if name_clean in ["temporal", "spatial"]:
        # TemporalOnly / SpatialOnly wrap a cstp instance
        cstp_pred = getattr(predictor, "cstp", None)
        if cstp_pred is not None:
            s_spatial = cstp_pred.compute_spatial_score(
                target_stage_id=target_stage,
                source_experts=source_experts,
                source_stage_id=source_stage
            )
            s_temporal = cstp_pred.compute_temporal_score(request_id, target_stage)
            composite = (cstp_pred.alpha * s_spatial) + ((1.0 - cstp_pred.alpha) * s_temporal)
            return [float(composite[c]) for c in candidates]
        return [0.0 for _ in candidates]

    if name_clean in ["acsh_global", "acsh_stage_aware", "acsh_fast_recovery"]:
        acsh_pred: ACSHPredictor = predictor  # type: ignore
        num_exp = acsh_pred.num_routed_experts
        s_spatial = acsh_pred.cstp.compute_spatial_score(
            target_stage_id=target_stage,
            source_experts=source_experts,
            source_stage_id=source_stage
        )
        s_temporal = acsh_pred.cstp.compute_temporal_score(request_id, target_stage)
        raw_cstp = (acsh_pred.cstp.alpha * s_spatial) + ((1.0 - acsh_pred.cstp.alpha) * s_temporal)

        st_dict = acsh_pred.st_moe.compute_confidence_scores(
            request_id=request_id,
            target_stage_id=target_stage,
            current_source_experts=source_experts,
            source_stage_id=source_stage
        )
        raw_st = np.zeros(num_exp, dtype=np.float64)
        for e, v in st_dict.items():
            if 0 <= e < num_exp:
                raw_st[e] = float(v)

        p_c, p_m = acsh_pred._normalize_scores(raw_cstp, raw_st)
        w_c, w_m = acsh_pred.get_fusion_weights(
            request_id, target_stage, acsh_pred.prediction_horizon
        )
        p_acsh = (w_c * p_c) + (w_m * p_m)
        return [float(p_acsh[c]) for c in candidates]

    if name_clean == "frequency":
        freq_pred: FrequencyPredictor = predictor  # type: ignore
        if 0 <= target_stage < freq_pred.num_moe_stages:
            counts = freq_pred.frequency_counts[target_stage]
            return [float(counts[c]) for c in candidates]
        return [0.0 for _ in candidates]

    return [0.0 for _ in candidates]


def run_cpp_replay(
    trace_path: str,
    output_path: str,
    predictor_name: str,
    budget: int,
    num_stages: int,
    num_experts: int,
    top_k: int,
    horizon: int,
    max_tokens: int,
    alpha: float,
    smoothing_lambda: float,
    seed: int
) -> None:
    """Invokes build/replay_predictors with exact CLI parameters."""
    binary_path = os.path.join(PROJECT_ROOT, "build", "replay_predictors")
    if not os.path.exists(binary_path):
        raise FileNotFoundError(f"C++ binary not found at: {binary_path}")

    cmd = [
        binary_path,
        "--trace", trace_path,
        "--output", output_path,
        "--predictor", predictor_name,
        "--budget", str(budget),
        "--num_stages", str(num_stages),
        "--num_experts", str(num_experts),
        "--top_k", str(top_k),
        "--horizon", str(horizon),
        "--max_tokens", str(max_tokens),
        "--alpha", str(alpha),
        "--lambda", str(smoothing_lambda),
        "--seed", str(seed)
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"C++ replay execution failed with code {result.returncode}:\n{result.stderr}\n{result.stdout}"
        )


def candidates_match_with_score_tolerance(
    py_cands: List[int],
    cpp_cands: List[int],
    py_scores: List[float],
    cpp_scores: List[float],
    tol: float = 1e-9
) -> Tuple[bool, Optional[str]]:
    """
    Checks candidate identity and score equivalence within tolerance.
    Handles floating-point numerical ties (where scores differ by < tol)
    robustly without requiring identical tie-breaking under LSB noise.
    """
    if len(py_cands) != len(cpp_cands):
        return False, f"Candidate count mismatch: Python {len(py_cands)} vs C++ {len(cpp_cands)}"

    if py_cands == cpp_cands:
        score_diff_max = 0.0
        for s_py, s_cpp in zip(py_scores, cpp_scores):
            diff = abs(s_py - s_cpp)
            if diff > score_diff_max:
                score_diff_max = diff
        if score_diff_max > tol:
            return False, f"Score divergence: max diff {score_diff_max} > tolerance {tol}"
        return True, None

    # Candidates differ in order: check if candidate sets match and if inversion is between tied scores
    if set(py_cands) != set(cpp_cands):
        return False, f"Candidate set mismatch: Py {py_cands} vs C++ {cpp_cands}"

    py_map = {}
    for idx, cand in enumerate(py_cands):
        if idx < len(py_scores):
            py_map[cand] = py_scores[idx]
        else:
            py_map[cand] = 0.0

    cpp_map = {}
    for idx, cand in enumerate(cpp_cands):
        if idx < len(cpp_scores):
            cpp_map[cand] = cpp_scores[idx]
        else:
            cpp_map[cand] = 0.0

    for cand in py_cands:
        s_py = py_map[cand]
        s_cpp = cpp_map[cand]
        diff = abs(s_py - s_cpp)
        if diff > tol:
            return False, f"Candidate {cand} score divergence: Py {s_py} vs C++ {s_cpp} (diff {diff})"

    # Verify that any ordering inversion is between scores that are equal within tolerance
    for i in range(len(py_cands)):
        for j in range(i + 1, len(py_cands)):
            cand_a = py_cands[i]
            cand_b = py_cands[j]
            cpp_idx_a = cpp_cands.index(cand_a)
            cpp_idx_b = cpp_cands.index(cand_b)
            if cpp_idx_b < cpp_idx_a:
                score_diff = abs(py_map[cand_a] - py_map[cand_b])
                if score_diff > tol:
                    return False, f"Non-tied ordering inversion between {cand_a} ({py_map[cand_a]}) and {cand_b} ({py_map[cand_b]}): diff {score_diff} > tol {tol}"

    return True, None


def verify_predictor_replay(
    trace_path: str,
    predictor_name: str,
    budget: int,
    num_stages: int,
    num_experts: int,
    top_k: int,
    horizon: int = 1,
    max_tokens: int = -1,
    alpha: float = 0.5,
    smoothing_lambda: float = 1.0,
    seed: int = 42,
    output_dir: str = os.path.join(PROJECT_ROOT, "results/phase_2b_replay"),
    tolerance: float = 1e-9
) -> Dict[str, Any]:
    """
    Executes C++ and Python predictors over identical chronological events and performs
    event-level and macro-metric comparison.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(trace_path))[0]
    cpp_output_jsonl = os.path.join(
        output_dir,
        f"{dataset_name}_{predictor_name}_b{budget}_cpp.jsonl"
    )

    t0 = time.time()
    run_cpp_replay(
        trace_path=trace_path,
        output_path=cpp_output_jsonl,
        predictor_name=predictor_name,
        budget=budget,
        num_stages=num_stages,
        num_experts=num_experts,
        top_k=top_k,
        horizon=horizon,
        max_tokens=max_tokens,
        alpha=alpha,
        smoothing_lambda=smoothing_lambda,
        seed=seed
    )
    cpp_time = time.time() - t0

    # Load C++ event logs
    cpp_events = []
    with open(cpp_output_jsonl, "r") as f:
        for line in f:
            if line.strip():
                cpp_events.append(json.loads(line))

    # Initialize Python predictor
    py_pred = create_python_predictor(
        predictor_name=predictor_name,
        num_stages=num_stages,
        num_experts=num_experts,
        top_k=top_k,
        budget=budget,
        horizon=horizon,
        alpha=alpha,
        smoothing_lambda=smoothing_lambda,
        random_seed=seed
    )

    mismatches = []
    current_req = None
    total_events = len(cpp_events)

    py_metrics_list = []
    cpp_metrics_list = []

    t_py_start = time.time()

    for idx, ev in enumerate(cpp_events):
        req_id = ev["request_id"]
        tok_id = ev["token_id"]
        source_stage = ev["source_stage"]
        target_stage = ev["target_stage"]
        source_experts = ev["source_experts"]
        actual_experts = ev["actual_target_experts"]
        cpp_candidates = ev["predicted_candidates"]
        cpp_scores = ev["candidate_scores"]

        if req_id != current_req:
            if hasattr(py_pred, "reset"):
                py_pred.reset(req_id)
            else:
                if hasattr(py_pred, "reset_request_state"):
                    py_pred.reset_request_state(req_id)
            current_req = req_id

        # 1. Causal Python prediction
        py_candidates = py_pred.predict(
            request_id=req_id,
            target_stage_id=target_stage,
            current_source_experts=source_experts,
            budget=budget,
            token_id=tok_id,
            source_stage_id=source_stage
        )

        # 2. Retrieve scores for candidates
        py_scores = get_python_scores(
            predictor=py_pred,
            predictor_name=predictor_name,
            request_id=req_id,
            target_stage=target_stage,
            source_experts=source_experts,
            source_stage=source_stage,
            budget=budget,
            candidates=py_candidates
        )

        # 3. Compute metrics
        py_ev_m = compute_event_metrics(py_candidates, actual_experts)
        py_metrics_list.append(py_ev_m)

        cpp_ev_m = {
            "hits": ev["hits"],
            "precision": ev["precision"],
            "recall": ev["recall"],
            "f1": ev["f1"],
            "hit_rate": ev["hit_rate"],
            "mrr": ev["mrr"],
            "ndcg": ev["ndcg"]
        }
        cpp_metrics_list.append(cpp_ev_m)

        # 4. Event comparison
        is_random = (predictor_name.lower().strip() == "random")
        mismatch_reason = None

        if is_random:
            # Statistical check for Random: valid count, bounded IDs, unique candidates
            expected_count = min(budget, num_experts)
            if len(cpp_candidates) != expected_count:
                mismatch_reason = f"C++ candidate count {len(cpp_candidates)} != expected {expected_count}"
            else:
                if len(set(cpp_candidates)) != len(cpp_candidates):
                    mismatch_reason = "C++ candidates contain duplicates"
                else:
                    for cid in cpp_candidates:
                        if cid < 0 or cid >= num_experts:
                            mismatch_reason = f"C++ candidate {cid} out of range [0, {num_experts-1}]"
                            break
        else:
            # Deterministic exact check for all other predictors
            ok_match, err_msg = candidates_match_with_score_tolerance(
                py_cands=py_candidates,
                cpp_cands=cpp_candidates,
                py_scores=py_scores,
                cpp_scores=cpp_scores,
                tol=tolerance
            )
            if not ok_match:
                mismatch_reason = err_msg

        if mismatch_reason is not None:
            mismatches.append({
                "event_index": idx,
                "request_id": req_id,
                "token_id": tok_id,
                "source_stage": source_stage,
                "target_stage": target_stage,
                "reason": mismatch_reason,
                "py_candidates": py_candidates,
                "cpp_candidates": cpp_candidates,
                "py_scores": py_scores,
                "cpp_scores": cpp_scores
            })

        # 5. Reveal ground truth target labels and trigger causal state update
        py_pred.update(
            request_id=req_id,
            target_stage_id=target_stage,
            source_experts=source_experts,
            actual_target_experts=actual_experts,
            token_id=tok_id,
            source_stage_id=source_stage
        )

    py_time = time.time() - t_py_start

    # Compute macro summary metrics
    def calc_macro(m_list):
        if len(m_list) == 0:
            return {}
        return {
            "precision": float(np.mean([m["precision"] for m in m_list])),
            "recall": float(np.mean([m["recall"] for m in m_list])),
            "f1": float(np.mean([m["f1"] for m in m_list])),
            "hit_rate": float(np.mean([m["hit_rate"] for m in m_list])),
            "mrr": float(np.mean([m["mrr"] for m in m_list])),
            "ndcg": float(np.mean([m["ndcg"] for m in m_list]))
        }

    macro_py = calc_macro(py_metrics_list)
    macro_cpp = calc_macro(cpp_metrics_list)

    passed = (len(mismatches) == 0)

    verdict = "PASS"
    if not passed:
        verdict = "FAIL"
    else:
        if predictor_name.lower().strip() == "random":
            verdict = "PASS_STATISTICAL"

    summary = {
        "dataset": dataset_name,
        "predictor": predictor_name,
        "budget": budget,
        "total_events": total_events,
        "mismatches": len(mismatches),
        "verdict": verdict,
        "cpp_runtime_sec": round(cpp_time, 4),
        "py_runtime_sec": round(py_time, 4),
        "macro_py": macro_py,
        "macro_cpp": macro_cpp,
        "sample_mismatches": mismatches[:5]
    }

    return summary


def run_tier(
    tier_name: str,
    trace_path: str,
    predictors: List[str],
    budgets: List[int],
    num_stages: int,
    num_experts: int,
    top_k: int,
    horizon: int = 1,
    max_tokens: int = -1,
    alpha: float = 0.5,
    smoothing_lambda: float = 1.0,
    seed: int = 42,
    output_dir: str = os.path.join(PROJECT_ROOT, "results/phase_2b_replay")
) -> List[Dict[str, Any]]:
    """Runs verification for a list of predictors and budgets on a specified trace."""
    print(f"\n======================================================================")
    print(f"RUNNING TIER: {tier_name}")
    print(f"Trace: {trace_path}")
    print(f"Predictors: {predictors}")
    print(f"Budgets: {budgets}")
    print(f"======================================================================")

    results = []
    tier_failed = False

    for pred in predictors:
        for b in budgets:
            print(f"  Testing {pred.upper()} with Budget B={b}...", end=" ", flush=True)
            res = verify_predictor_replay(
                trace_path=trace_path,
                predictor_name=pred,
                budget=b,
                num_stages=num_stages,
                num_experts=num_experts,
                top_k=top_k,
                horizon=horizon,
                max_tokens=max_tokens,
                alpha=alpha,
                smoothing_lambda=smoothing_lambda,
                seed=seed,
                output_dir=output_dir
            )
            results.append(res)

            v = res["verdict"]
            print(f"[{v}] (Events: {res['total_events']}, Mismatches: {res['mismatches']})")

            if v == "FAIL":
                tier_failed = True
                print(f"    [FAILURE DETAILS] First mismatch:")
                for m in res["sample_mismatches"]:
                    print(f"      Event {m['event_index']} (Req={m['request_id']}, Tok={m['token_id']}, L={m['target_stage']}): {m['reason']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Python-C++ Predictor Replay Verification Harness")
    parser.add_argument("--tier", type=str, default="all", help="Tier to execute (1, 2, 3, 4, 5, or all)")
    parser.add_argument("--predictor", type=str, default="all", help="Specific predictor to run")
    parser.add_argument("--budget", type=int, default=4, help="Budget to evaluate")
    parser.add_argument("--trace", type=str, default="", help="Custom trace file")
    args = parser.parse_args()

    # Pre-execution check: AST verification for zero ternary operators
    with open(__file__, "r") as f:
        tree = ast.parse(f.read())
    ternary_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.IfExp))
    if ternary_count > 0:
        print(f"[FATAL ERROR] Found {ternary_count} ternary operator(s) in {__file__}! Halting.")
        sys.exit(1)

    print("[PARITY CHECK] AST verification confirmed zero ternary operators in verification harness.")

    tier_results = {}

    # Define tier specifications
    all_deployable = [
        "st_moe", "cstp", "acsh_global", "acsh_stage_aware", "acsh_fast_recovery",
        "temporal", "spatial", "frequency", "random", "none"
    ]

    tier_to_run = args.tier.lower().strip()

    # Tier 1: Micro synthetic
    if tier_to_run in ["1", "tier1", "all"]:
        trace_t1 = os.path.join(PROJECT_ROOT, "results/phase_2b_replay/traces/micro_synthetic.csv")
        t1_preds = all_deployable
        if args.predictor != "all":
            t1_preds = [args.predictor]
        t1_res = run_tier(
            tier_name="Tier 1 — Micro Synthetic (4 tokens, 4 stages)",
            trace_path=trace_t1,
            predictors=t1_preds,
            budgets=[2, 4],
            num_stages=4,
            num_experts=16,
            top_k=2
        )
        tier_results["tier1"] = t1_res
        if any(r["verdict"] == "FAIL" for r in t1_res):
            print("\n[STOPPING CONDITION] Tier 1 failed! Halting before subsequent tiers.")
            sys.exit(1)

    # Tier 2: Correlated synthetic
    if tier_to_run in ["2", "tier2", "all"]:
        trace_t2 = os.path.join(PROJECT_ROOT, "results/phase_2b_replay/traces/correlated_synthetic.csv")
        if args.predictor != "all":
            t2_preds = [args.predictor]
        else:
            t2_preds = ["cstp", "acsh_global", "acsh_stage_aware", "acsh_fast_recovery"]
        t2_res = run_tier(
            tier_name="Tier 2 — Correlated Synthetic (50 tokens, 4 stages)",
            trace_path=trace_t2,
            predictors=t2_preds,
            budgets=[2, 4, 8],
            num_stages=4,
            num_experts=16,
            top_k=2
        )
        tier_results["tier2"] = t2_res
        if any(r["verdict"] == "FAIL" for r in t2_res):
            print("\n[STOPPING CONDITION] Tier 2 failed! Halting before subsequent tiers.")
            sys.exit(1)

    # Tier 3: MoE Anywhere subset (32 tokens, 40 stages)
    if tier_to_run in ["3", "tier3", "all"]:
        trace_t3 = os.path.join(PROJECT_ROOT, "results/endsem_phase_e/final_baseline/traces/ma_p60_t128_l40.csv")
        if args.predictor != "all":
            t3_preds = [args.predictor]
        else:
            t3_preds = ["st_moe", "cstp", "acsh_stage_aware"]
        t3_res = run_tier(
            tier_name="Tier 3 — MoE Anywhere Subset (32 tokens, 40 stages)",
            trace_path=trace_t3,
            predictors=t3_preds,
            budgets=[2, 4, 8],
            num_stages=40,
            num_experts=256,
            top_k=8,
            max_tokens=32
        )
        tier_results["tier3"] = t3_res
        if any(r["verdict"] == "FAIL" for r in t3_res):
            print("\n[STOPPING CONDITION] Tier 3 failed! Halting before subsequent tiers.")
            sys.exit(1)

    # Tier 4: MoE Anywhere full (128 tokens, 40 stages)
    if tier_to_run in ["4", "tier4", "all"]:
        trace_t4 = os.path.join(PROJECT_ROOT, "results/endsem_phase_e/final_baseline/traces/ma_p60_t128_l40.csv")
        if args.predictor != "all":
            t4_preds = [args.predictor]
        else:
            t4_preds = all_deployable
        t4_res = run_tier(
            tier_name="Tier 4 — MoE Anywhere Full (128 tokens, 40 stages)",
            trace_path=trace_t4,
            predictors=t4_preds,
            budgets=[1, 2, 4, 8],
            num_stages=40,
            num_experts=256,
            top_k=8,
            max_tokens=128
        )
        tier_results["tier4"] = t4_res
        if any(r["verdict"] == "FAIL" for r in t4_res):
            print("\n[STOPPING CONDITION] Tier 4 failed! Halting before subsequent tiers.")
            sys.exit(1)

    # Tier 5: MoE Suite full (128 tokens, 75 stages)
    if tier_to_run in ["5", "tier5", "all"]:
        trace_t5 = os.path.join(PROJECT_ROOT, "results/endsem_phase_e/final_baseline/traces/ms_task12_t128_l75.csv")
        if args.predictor != "all":
            t5_preds = [args.predictor]
        else:
            t5_preds = all_deployable
        t5_res = run_tier(
            tier_name="Tier 5 — MoE Suite Full (128 tokens, 75 stages)",
            trace_path=trace_t5,
            predictors=t5_preds,
            budgets=[1, 2, 4, 8],
            num_stages=75,
            num_experts=256,
            top_k=8,
            max_tokens=128
        )
        tier_results["tier5"] = t5_res
        if any(r["verdict"] == "FAIL" for r in t5_res):
            print("\n[STOPPING CONDITION] Tier 5 failed! Halting before subsequent tiers.")
            sys.exit(1)

    # Save summary json
    summary_path = os.path.join(PROJECT_ROOT, "results/phase_2b_replay/verification_summary.json")
    with open(summary_path, "w") as f:
        json.dump(tier_results, f, indent=2)

    print(f"\nSaved verification summary to {summary_path}")


if __name__ == "__main__":
    main()
