"""
Unified Evaluation Framework for Offline MoE Expert Predictors.

Implements:
1. Request-level and chronological token-level train/validation/test split isolation.
2. Elimination of single-request training/test leakage.
3. Standardized Precision@B, Recall@B, F1@B, Hit Rate, Exact-Set Accuracy, and Coverage metrics.
4. Stratified reporting across budgets, MoE stages, and token positions (cold-start vs steady-state).
5. Explicit separation of fixed-budget and variable-threshold ST-MoE evaluation regimes.
6. Causal verification and strict sequence boundary isolation.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import copy
import inspect
import pandas as pd
import numpy as np

from src.predictor.trace_parser import NormalizedRoutingRecord


class EvaluationSplitError(Exception):
    """Raised when records cannot be causally and safely partitioned into train and test splits."""
    pass


class EvaluationModeError(ValueError):
    """Raised when an unsupported or invalid evaluation mode is passed."""
    pass


_SENTINEL_SOURCE_STAGE = object()

VALID_EVALUATION_MODES = {
    "online_within_request",
    "online_across_requests",
    "online",  # Backward-compatible alias for online_across_requests
    "frozen",
    "threshold",
    "threshold_within_request",
    "threshold_across_requests",
    "threshold_frozen",
}


class PredictionOpportunity:
    """Represents a single prediction instance (request, token, stage)."""
    def __init__(
        self,
        request_id: int,
        token_id: int,
        moe_stage_id: int,
        source_experts: List[int],
        actual_experts: List[int],
        source_stage_id: Any = _SENTINEL_SOURCE_STAGE,
        prediction_horizon: int = 1
    ):
        self.request_id = request_id
        self.token_id = token_id
        self.moe_stage_id = moe_stage_id
        self.source_experts = list(source_experts)
        self.actual_experts = list(actual_experts)
        self.prediction_horizon = prediction_horizon

        # Strictly preserve None when no valid source stage exists (do not clamp to 0)
        if source_stage_id is not _SENTINEL_SOURCE_STAGE:
            self.source_stage_id = source_stage_id
        else:
            if moe_stage_id >= prediction_horizon:
                self.source_stage_id = moe_stage_id - prediction_horizon
            else:
                self.source_stage_id = None


class PredictionResultRecord:
    """Contains outcome metrics for a single prediction opportunity at budget B or threshold."""
    def __init__(
        self,
        request_id: int,
        token_id: int,
        token_position: int,
        moe_stage_id: int,
        budget: Optional[int],
        predictor_name: str,
        predicted_experts: List[int],
        actual_experts: List[int],
        evaluation_mode: str = "online",
        is_cold_start: Optional[bool] = None
    ):
        self.request_id = request_id
        self.token_id = token_id
        self.token_position = token_position
        self.moe_stage_id = moe_stage_id
        self.budget = budget
        self.predictor_name = predictor_name
        self.predicted_experts = list(predicted_experts)
        self.actual_experts = list(actual_experts)
        self.evaluation_mode = evaluation_mode
        self.is_cold_start = is_cold_start if is_cold_start is not None else (token_position == 0)

        pred_set = set(predicted_experts)
        act_set = set(actual_experts)
        hits = len(pred_set.intersection(act_set))
        self.hits = hits

        # Precision@B (|A ∩ P| / |P|)
        if len(pred_set) > 0:
            self.precision = float(hits) / float(len(pred_set))
        else:
            self.precision = 0.0

        # Recall@B (|A ∩ P| / |A|)
        if len(act_set) > 0:
            self.recall = float(hits) / float(len(act_set))
        else:
            self.recall = 0.0

        # F1@B (2 * P * R / (P + R))
        if (self.precision + self.recall) > 0.0:
            self.f1 = (2.0 * self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1 = 0.0

        # Hit rate (1 if |A ∩ P| > 0 else 0)
        if hits > 0:
            self.hit_rate = 1.0
        else:
            self.hit_rate = 0.0

        # Exact-set accuracy (1 if A == P else 0)
        self.exact_set_accuracy = 1.0 if pred_set == act_set else 0.0

        # Prefetch accounting
        self.predicted_count = len(pred_set)
        self.actual_count = len(act_set)
        self.missed_count = len(act_set.difference(pred_set))
        self.unused_count = len(pred_set.difference(act_set))


class EvaluationHarness:
    """Evaluates multiple predictors under identical conditions."""

    def __init__(self, num_moe_stages: int = 57):
        self.num_moe_stages = num_moe_stages

    def prepare_opportunities(
        self,
        records: List[NormalizedRoutingRecord],
        prediction_horizon: int = 1
    ) -> List[PredictionOpportunity]:
        """
        Groups routing records into sequentially ordered prediction opportunities.
        """
        if not records:
            return []

        # Group by (request_id, token_id, moe_stage_id)
        grouped: Dict[Tuple[int, int, int], List[int]] = {}
        for rec in records:
            key = (rec.request_id, rec.token_id, rec.moe_stage_id)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(rec.expert_id)

        opportunities: List[PredictionOpportunity] = []
        unique_requests = sorted(list(set(k[0] for k in grouped.keys())))

        for req in unique_requests:
            req_keys = [k for k in grouped.keys() if k[0] == req]
            req_tokens = sorted(list(set(k[1] for k in req_keys)))

            for tok_idx, tok in enumerate(req_tokens):
                for stage in range(self.num_moe_stages):
                    curr_key = (req, tok, stage)
                    actual_experts = grouped.get(curr_key, [])
                    if len(actual_experts) == 0:
                        continue

                    if stage >= prediction_horizon:
                        source_stage = stage - prediction_horizon
                        prev_stage_key = (req, tok, source_stage)
                        source_experts = grouped.get(prev_stage_key, [])
                    else:
                        source_stage = None
                        source_experts = []

                    opp = PredictionOpportunity(
                        request_id=req,
                        token_id=tok,
                        moe_stage_id=stage,
                        source_experts=source_experts,
                        actual_experts=actual_experts,
                        source_stage_id=source_stage,
                        prediction_horizon=prediction_horizon
                    )
                    opportunities.append(opp)

        return opportunities

    def split_by_request(
        self,
        records: List[NormalizedRoutingRecord],
        train_ratio: float = 0.75,
        allow_token_split: bool = True,
        min_tokens_for_split: int = 4
    ) -> Tuple[List[NormalizedRoutingRecord], List[NormalizedRoutingRecord], Dict]:
        """
        Splits records by request ID to prevent data leakage across train and test.
        If only a single request exists:
          - Strategy A: If it contains >= min_tokens_for_split chronological tokens and allow_token_split is True,
            performs a strict chronological prefix/suffix split.
          - Strategy B: If tokens are insufficient or allow_token_split is False, raises an informative EvaluationSplitError.
        Never silently returns identical training and testing records.
        """
        if not records:
            raise EvaluationSplitError("Cannot split empty records: records list is empty.")

        if not (0.0 < train_ratio < 1.0):
            raise ValueError(f"train_ratio must be strictly between 0.0 and 1.0, got {train_ratio}")

        unique_requests = sorted(list(set(r.request_id for r in records)))
        num_requests = len(unique_requests)

        # Multi-request case: Request-level disjoint splitting
        if num_requests > 1:
            num_train = max(1, min(num_requests - 1, int(round(num_requests * train_ratio))))
            train_req_set = set(unique_requests[:num_train])
            test_req_set = set(unique_requests[num_train:])

            train_records = [r for r in records if r.request_id in train_req_set]
            test_records = [r for r in records if r.request_id in test_req_set]

            split_meta = {
                "split_strategy": "request_disjoint",
                "total_requests": num_requests,
                "train_requests": sorted(list(train_req_set)),
                "test_requests": sorted(list(test_req_set)),
                "train_record_count": len(train_records),
                "test_record_count": len(test_records),
                "is_single_request": False
            }
            return train_records, test_records, split_meta

        # Single-request case: num_requests == 1
        req_id = unique_requests[0]
        if not allow_token_split:
            raise EvaluationSplitError(
                f"Trace contains only a single request (request_id={req_id}), which cannot be split "
                f"by request. Set allow_token_split=True to permit chronological token-level splitting."
            )

        unique_tokens = sorted(list(set(r.token_id for r in records)))
        num_tokens = len(unique_tokens)

        if num_tokens < min_tokens_for_split:
            raise EvaluationSplitError(
                f"Single-request trace (request_id={req_id}) contains only {num_tokens} chronological token(s), "
                f"which is insufficient for a valid chronological train/test split (minimum required: {min_tokens_for_split}). "
                f"Returning identical training and testing records is strictly prohibited to eliminate data leakage."
            )

        # Strategy A: Strict chronological token split
        num_train_tokens = max(1, min(num_tokens - 1, int(round(num_tokens * train_ratio))))
        train_tokens = unique_tokens[:num_train_tokens]
        test_tokens = unique_tokens[num_train_tokens:]

        train_token_set = set(train_tokens)
        test_token_set = set(test_tokens)

        # Invariant: chronological prefix strictly precedes evaluation suffix
        assert max(train_tokens) < min(test_tokens), "Chronological ordering violated in token split!"

        train_records = [r for r in records if r.token_id in train_token_set]
        test_records = [r for r in records if r.token_id in test_token_set]

        split_meta = {
            "split_strategy": "chronological_token_split",
            "total_requests": 1,
            "train_requests": [req_id],
            "test_requests": [req_id],
            "total_tokens": num_tokens,
            "train_tokens": train_tokens,
            "test_tokens": test_tokens,
            "train_token_range": (train_tokens[0], train_tokens[-1]),
            "test_token_range": (test_tokens[0], test_tokens[-1]),
            "train_record_count": len(train_records),
            "test_record_count": len(test_records),
            "is_single_request": True
        }
        return train_records, test_records, split_meta

    def evaluate_predictor(
        self,
        predictor_instance: Any,
        predictor_name: str,
        test_opportunities: List[PredictionOpportunity],
        budgets: Optional[List[int]] = None,
        evaluation_mode: str = "online",
        threshold: int = 2
    ) -> List[PredictionResultRecord]:
        """
        Runs strictly causal evaluation of a predictor across budgets or variable threshold.

        Modes:
        - "online_within_request": Starts every test request from the original profiling-derived CCT.
          Allows online updates within that request. Restores the original profiling state before the next request.
        - "online_across_requests" (or "online"): Preserves persistent learned statistics (CCT) across requests.
          Resets request-specific temporal history at sequence boundaries.
        - "frozen": Predicts using fixed profiling state. Does NOT update predictor on test observations.
        - "threshold": Variable-set ST-MoE evaluation (Score >= threshold).
          Supports threshold_within_request, threshold_across_requests, and threshold_frozen.

        Causal ordering:
        Predict -> Score -> Update. Current target ground truth NEVER influences its own prediction.
        """
        results: List[PredictionResultRecord] = []

        if not test_opportunities:
            return results

        if evaluation_mode not in VALID_EVALUATION_MODES:
            raise EvaluationModeError(
                f"Unsupported evaluation mode: '{evaluation_mode}'. "
                f"Valid evaluation modes are: {sorted(list(VALID_EVALUATION_MODES))}."
            )

        is_threshold = evaluation_mode.startswith("threshold")
        is_within_request = evaluation_mode in ("online_within_request", "threshold_within_request")
        is_frozen = evaluation_mode in ("frozen", "threshold_frozen")

        if is_threshold:
            eval_iterations = [None]
        else:
            if budgets is None or len(budgets) == 0:
                eval_iterations = [8]  # Default to Top-8
            else:
                eval_iterations = list(budgets)

        for budget in eval_iterations:
            if predictor_name == "NoPrediction" or predictor_instance is None:
                base_predictor = None
            else:
                base_predictor = copy.deepcopy(predictor_instance)

            runner = copy.deepcopy(base_predictor) if base_predictor is not None else None

            # Group opportunities by request to preserve token and stage sequence
            unique_requests = sorted(list(set(op.request_id for op in test_opportunities)))

            for req in unique_requests:
                # In within-request mode: start each test request from the original profiling state!
                if is_within_request and base_predictor is not None:
                    runner = copy.deepcopy(base_predictor)

                # Sequence boundary: reset temporal history for this sequence
                if runner is not None:
                    if hasattr(runner, "reset_request_state"):
                        runner.reset_request_state(req)
                    elif hasattr(runner, "reset"):
                        runner.reset(req)

                req_ops = [op for op in test_opportunities if op.request_id == req]
                unique_tokens = sorted(list(set(op.token_id for op in req_ops)))
                tok_to_pos = {tok: idx for idx, tok in enumerate(unique_tokens)}

                for op in req_ops:
                    tok_pos = tok_to_pos[op.token_id]

                    # 1. Predict strictly before revealing target
                    if predictor_name == "NoPrediction" or runner is None:
                        predicted: List[int] = []
                    elif is_threshold:
                        if hasattr(runner, "predict_threshold"):
                            sig = inspect.signature(runner.predict_threshold)
                            if "source_stage_id" in sig.parameters:
                                predicted = runner.predict_threshold(
                                    request_id=op.request_id,
                                    target_stage_id=op.moe_stage_id,
                                    current_source_experts=op.source_experts,
                                    source_stage_id=op.source_stage_id
                                )
                            else:
                                predicted = runner.predict_threshold(
                                    request_id=op.request_id,
                                    target_stage_id=op.moe_stage_id,
                                    current_source_experts=op.source_experts
                                )
                        elif hasattr(runner, "get_candidate_scores"):
                            scores = runner.get_candidate_scores(
                                request_id=op.request_id,
                                target_stage_id=op.moe_stage_id,
                                current_source_experts=op.source_experts
                            )
                            predicted = [
                                exp for exp, s in sorted(scores.items(), key=lambda x: (-x[1], x[0]))
                                if s >= threshold
                            ]
                        else:
                            predicted = runner.predict(
                                request_id=op.request_id,
                                target_stage_id=op.moe_stage_id,
                                current_source_experts=op.source_experts,
                                budget=8
                            )
                    else:
                        sig = inspect.signature(runner.predict)
                        call_kwargs = {
                            "request_id": op.request_id,
                            "target_stage_id": op.moe_stage_id,
                            "current_source_experts": op.source_experts,
                            "budget": budget,
                        }
                        if "token_id" in sig.parameters:
                            call_kwargs["token_id"] = op.token_id
                        if "source_stage_id" in sig.parameters:
                            call_kwargs["source_stage_id"] = op.source_stage_id
                        predicted = runner.predict(**call_kwargs)

                    # 2. Record prediction and evaluate metrics against actual target
                    rec = PredictionResultRecord(
                        request_id=op.request_id,
                        token_id=op.token_id,
                        token_position=tok_pos,
                        moe_stage_id=op.moe_stage_id,
                        budget=budget,
                        predictor_name=predictor_name,
                        predicted_experts=predicted,
                        actual_experts=op.actual_experts,
                        evaluation_mode=evaluation_mode,
                        is_cold_start=(tok_pos == 0)
                    )
                    results.append(rec)

                    # 3. Post-scoring update (strictly NOT in frozen modes)
                    if runner is not None and not is_frozen:
                        runner.update(
                            request_id=op.request_id,
                            target_stage_id=op.moe_stage_id,
                            source_experts=op.source_experts,
                            actual_target_experts=op.actual_experts,
                            token_id=op.token_id,
                            source_stage_id=op.source_stage_id
                        )

        return results

    def compute_summary_metrics(
        self,
        results: List[PredictionResultRecord]
    ) -> Dict[str, Any]:
        """
        Computes macro and micro summary metrics across evaluation results,
        including cold-start vs steady-state stratifications and exact-set accuracy.
        """
        if len(results) == 0:
            return {}

        total_instances = len(results)
        macro_precision = float(np.mean([r.precision for r in results]))
        macro_recall = float(np.mean([r.recall for r in results]))
        macro_f1 = float(np.mean([r.f1 for r in results]))
        hit_rate = float(np.mean([r.hit_rate for r in results]))
        exact_set_accuracy = float(np.mean([r.exact_set_accuracy for r in results]))

        total_hits = sum(r.hits for r in results)
        total_predicted = sum(r.predicted_count for r in results)
        total_actual = sum(r.actual_count for r in results)
        total_missed = sum(r.missed_count for r in results)
        total_unused = sum(r.unused_count for r in results)

        pred_counts = [r.predicted_count for r in results]
        mean_predicted_count = float(np.mean(pred_counts))
        median_predicted_count = float(np.median(pred_counts))
        min_predicted_count = int(np.min(pred_counts))
        max_predicted_count = int(np.max(pred_counts))

        if total_predicted > 0:
            micro_precision = float(total_hits) / float(total_predicted)
        else:
            micro_precision = 0.0

        if total_actual > 0:
            micro_recall = float(total_hits) / float(total_actual)
            coverage = float(total_hits) / float(total_actual)
        else:
            micro_recall = 0.0
            coverage = 0.0

        if (micro_precision + micro_recall) > 0.0:
            micro_f1 = (2.0 * micro_precision * micro_recall) / (micro_precision + micro_recall)
        else:
            micro_f1 = 0.0

        # Cold-start vs steady-state breakdown
        cold_results = [r for r in results if r.is_cold_start]
        steady_results = [r for r in results if not r.is_cold_start]

        def _sub_metrics(sub_list: List[PredictionResultRecord]) -> Dict[str, Any]:
            if not sub_list:
                return {
                    "instances": 0,
                    "recall": 0.0,
                    "precision": 0.0,
                    "f1": 0.0,
                    "hit_rate": 0.0,
                    "exact_set_accuracy": 0.0
                }
            return {
                "instances": len(sub_list),
                "recall": float(np.mean([r.recall for r in sub_list])),
                "precision": float(np.mean([r.precision for r in sub_list])),
                "f1": float(np.mean([r.f1 for r in sub_list])),
                "hit_rate": float(np.mean([r.hit_rate for r in sub_list])),
                "exact_set_accuracy": float(np.mean([r.exact_set_accuracy for r in sub_list]))
            }

        cold_start_metrics = _sub_metrics(cold_results)
        steady_state_metrics = _sub_metrics(steady_results)

        evaluation_modes = sorted(list(set(r.evaluation_mode for r in results)))

        summary = {
            "total_instances": total_instances,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "hit_rate": hit_rate,
            "exact_set_accuracy": exact_set_accuracy,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "coverage": coverage,
            "total_hits": total_hits,
            "total_predicted": total_predicted,
            "total_actual": total_actual,
            "total_missed": total_missed,
            "total_unused": total_unused,
            "predicted_set_size": {
                "mean": mean_predicted_count,
                "median": median_predicted_count,
                "min": min_predicted_count,
                "max": max_predicted_count
            },
            "cold_start": cold_start_metrics,
            "steady_state": steady_state_metrics,
            "evaluation_mode": evaluation_modes[0] if len(evaluation_modes) == 1 else "mixed"
        }
        return summary
