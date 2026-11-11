"""
Adaptive Confidence and Statistical Hybrid (ACSH) Predictor.
Combines Bayesian probabilistic state from CSTP with bounded saturating confidence
counters from ST-MoE using causal online reliability estimation and adaptive softmax fusion.
"""

import copy
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Set

from src.predictor.base_predictor import BasePredictor


class ACSHPredictor(BasePredictor):
    """
    Adaptive Confidence and Statistical Hybrid (ACSH) Predictor.
    Dynamically balances CSTP and ST-MoE via causal online reliability tracking.
    Supports Global, Stage-Aware, and Fast Recovery variants.
    """

    def __init__(
        self,
        cstp_predictor: Any,
        st_moe_predictor: Any,
        variant: str = "stage_aware",
        beta: float = 0.8,
        tau: float = 0.2,
        temp_c: float = 1.0,
        temp_m: float = 1.0,
        normalization: str = "softmax",
        reliability_objective: str = "recall",
        initial_reliability: float = 0.5,
        shift_window: int = 3,
        shift_threshold: float = 0.75,
        fast_beta: float = 0.2,
        fast_cooldown_steps: int = 3,
        name: str = "acsh"
    ):
        self.cstp = cstp_predictor
        self.st_moe = st_moe_predictor
        self.variant = variant.lower()
        self.beta = beta
        self.tau = max(1e-4, tau)
        self.temp_c = max(1e-4, temp_c)
        self.temp_m = max(1e-4, temp_m)
        self.normalization = normalization.lower()
        self.reliability_objective = reliability_objective.lower()
        self.initial_reliability = initial_reliability
        self.shift_window = shift_window
        self.shift_threshold = shift_threshold
        self.fast_beta = fast_beta
        self.fast_cooldown_steps = fast_cooldown_steps
        self.name = name

        # Extract architectural properties from component predictors
        if hasattr(self.cstp, "num_routed_experts"):
            self.num_routed_experts = self.cstp.num_routed_experts
        else:
            self.num_routed_experts = self.st_moe.num_routed_experts

        if hasattr(self.cstp, "num_moe_stages"):
            self.num_moe_stages = self.cstp.num_moe_stages
        else:
            self.num_moe_stages = self.st_moe.num_moe_stages

        if hasattr(self.cstp, "prediction_horizon"):
            self.prediction_horizon = self.cstp.prediction_horizon
        else:
            self.prediction_horizon = 1

        # Reliability state tracking:
        # Global variant: request_id -> float
        self.rel_c_global: Dict[Any, float] = {}
        self.rel_m_global: Dict[Any, float] = {}

        # Stage-aware variant: request_id -> (stage_id, horizon) -> float
        self.rel_c_stage: Dict[Any, Dict[Tuple[int, int], float]] = {}
        self.rel_m_stage: Dict[Any, Dict[Tuple[int, int], float]] = {}

        # Fast recovery tracking: request_id -> (stage_id, horizon) -> state
        self.recent_errors: Dict[Any, Dict[Tuple[int, int], List[float]]] = {}
        self.fast_cooldown: Dict[Any, Dict[Tuple[int, int], int]] = {}
        self.shift_detections: List[Dict[str, Any]] = []

        # Store pending prediction outcomes for causal post-reveal updates
        # key: (request_id, target_stage_id, token_id) -> record dict
        self.pending_predictions: Dict[Tuple[Any, int, Optional[int]], Dict[str, Any]] = {}

        # Diagnostics history
        self.weight_history: List[Dict[str, Any]] = []

    def reset(self, request_id: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Resets request-level reliability tracking and component predictor states.
        """
        # Reset component predictors
        if hasattr(self.cstp, "reset"):
            self.cstp.reset(request_id, metadata=metadata)
        else:
            if hasattr(self.cstp, "reset_request_state"):
                self.cstp.reset_request_state(request_id)

        if hasattr(self.st_moe, "reset"):
            self.st_moe.reset(request_id, metadata=metadata)

        # Clear request reliability tracking
        if request_id in self.rel_c_global:
            del self.rel_c_global[request_id]
        if request_id in self.rel_m_global:
            del self.rel_m_global[request_id]

        if request_id in self.rel_c_stage:
            del self.rel_c_stage[request_id]
        if request_id in self.rel_m_stage:
            del self.rel_m_stage[request_id]

        if request_id in self.recent_errors:
            del self.recent_errors[request_id]
        if request_id in self.fast_cooldown:
            del self.fast_cooldown[request_id]

        # Purge pending predictions for this request
        keys_to_remove = [k for k in self.pending_predictions.keys() if k[0] == request_id]
        for k in keys_to_remove:
            del self.pending_predictions[k]

    def _normalize_scores(
        self,
        raw_cstp: np.ndarray,
        raw_st_moe: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies calibrated normalization to component scores.
        """
        num_exp = self.num_routed_experts

        if self.normalization == "softmax":
            # CSTP softmax with temp_c
            max_c = np.max(raw_cstp)
            shift_c = (raw_cstp - max_c) / self.temp_c
            exp_c = np.exp(shift_c)
            sum_c = np.sum(exp_c)
            if sum_c > 0.0:
                p_c = exp_c / sum_c
            else:
                p_c = np.ones(num_exp, dtype=np.float64) / float(num_exp)

            # ST-MoE softmax with temp_m
            max_m = np.max(raw_st_moe)
            shift_m = (raw_st_moe - max_m) / self.temp_m
            exp_m = np.exp(shift_m)
            sum_m = np.sum(exp_m)
            if sum_m > 0.0:
                p_m = exp_m / sum_m
            else:
                p_m = np.ones(num_exp, dtype=np.float64) / float(num_exp)

            return p_c, p_m

        else:
            if self.normalization == "rank":
                # Rank-based normalization: higher score gets higher rank
                order_c = np.argsort(np.argsort(raw_cstp))
                p_c = (order_c + 1.0) / np.sum(order_c + 1.0)

                order_m = np.argsort(np.argsort(raw_st_moe))
                p_m = (order_m + 1.0) / np.sum(order_m + 1.0)
                return p_c, p_m
            else:
                if self.normalization == "max":
                    # Simple max scaling (legacy hybrid style)
                    max_c = np.max(raw_cstp)
                    if max_c > 0.0:
                        p_c = raw_cstp / max_c
                    else:
                        p_c = raw_cstp

                    max_m = np.max(raw_st_moe)
                    if max_m > 0.0:
                        p_m = raw_st_moe / max_m
                    else:
                        p_m = raw_st_moe
                    return p_c, p_m
                else:
                    # Raw normalization: sum to 1
                    sum_c = np.sum(raw_cstp)
                    if sum_c > 0.0:
                        p_c = raw_cstp / sum_c
                    else:
                        p_c = np.ones(num_exp, dtype=np.float64) / float(num_exp)

                    sum_m = np.sum(raw_st_moe)
                    if sum_m > 0.0:
                        p_m = raw_st_moe / sum_m
                    else:
                        p_m = np.ones(num_exp, dtype=np.float64) / float(num_exp)
                    return p_c, p_m

    def get_current_reliability(
        self,
        request_id: Any,
        target_stage_id: int,
        horizon: int
    ) -> Tuple[float, float]:
        """
        Returns current reliability estimates (R_C, R_M).
        """
        if self.variant == "global":
            if request_id not in self.rel_c_global:
                self.rel_c_global[request_id] = self.initial_reliability
                self.rel_m_global[request_id] = self.initial_reliability
            return self.rel_c_global[request_id], self.rel_m_global[request_id]
        else:
            # Stage-aware or Fast-recovery
            if request_id not in self.rel_c_stage:
                self.rel_c_stage[request_id] = {}
                self.rel_m_stage[request_id] = {}

            key = (target_stage_id, horizon)
            if key not in self.rel_c_stage[request_id]:
                self.rel_c_stage[request_id][key] = self.initial_reliability
                self.rel_m_stage[request_id][key] = self.initial_reliability

            return self.rel_c_stage[request_id][key], self.rel_m_stage[request_id][key]

    def get_fusion_weights(
        self,
        request_id: Any,
        target_stage_id: int,
        horizon: int
    ) -> Tuple[float, float]:
        """
        Computes normalized fusion weights (w_C, w_M) via softmax reliability weighting.
        """
        r_c, r_m = self.get_current_reliability(request_id, target_stage_id, horizon)

        # Softmax over reliability scores scaled by tau
        max_r = max(r_c, r_m)
        e_c = np.exp((r_c - max_r) / self.tau)
        e_m = np.exp((r_m - max_r) / self.tau)
        denom = e_c + e_m

        if denom > 0.0:
            w_c = float(e_c / denom)
            w_m = float(e_m / denom)
        else:
            w_c = 0.5
            w_m = 0.5

        return w_c, w_m

    def predict(
        self,
        request_id: int,
        target_stage_id: int,
        current_source_experts: List[int],
        budget: int,
        token_id: Optional[int] = None,
        source_stage_id: Optional[int] = None
    ) -> List[int]:
        """
        Generates Top-B prefetch prediction using adaptive reliability-weighted fusion.
        """
        if budget <= 0:
            return []

        horizon = self.prediction_horizon
        num_exp = self.num_routed_experts

        # 1. Obtain CSTP component scores
        s_spatial = self.cstp.compute_spatial_score(
            target_stage_id=target_stage_id,
            source_experts=current_source_experts,
            source_stage_id=source_stage_id
        )
        s_temporal = self.cstp.compute_temporal_score(request_id, target_stage_id)
        raw_cstp = (self.cstp.alpha * s_spatial) + ((1.0 - self.cstp.alpha) * s_temporal)

        # 2. Obtain ST-MoE component scores
        st_dict = self.st_moe.compute_confidence_scores(
            request_id=request_id,
            target_stage_id=target_stage_id,
            current_source_experts=current_source_experts,
            source_stage_id=source_stage_id
        )
        raw_st_moe = np.zeros(num_exp, dtype=np.float64)
        for exp_id, val in st_dict.items():
            if exp_id >= 0 and exp_id < num_exp:
                raw_st_moe[exp_id] = float(val)

        # 3. Component Top-B predictions (for causal reliability evaluation)
        all_experts = list(range(num_exp))
        cstp_sorted = sorted(all_experts, key=lambda idx: (-raw_cstp[idx], idx))
        cstp_top_b = cstp_sorted[:min(budget, num_exp)]

        st_sorted = sorted(all_experts, key=lambda idx: (-raw_st_moe[idx], idx))
        st_top_b = st_sorted[:min(budget, num_exp)]

        # 4. Score normalization
        p_c, p_m = self._normalize_scores(raw_cstp, raw_st_moe)

        # 5. Fusion weights
        w_c, w_m = self.get_fusion_weights(request_id, target_stage_id, horizon)

        # 6. Adaptive combination
        p_acsh = (w_c * p_c) + (w_m * p_m)

        # 7. Select final Top-B with deterministic tie-breaking
        acsh_sorted = sorted(all_experts, key=lambda idx: (-p_acsh[idx], idx))
        acsh_top_b = acsh_sorted[:min(budget, num_exp)]

        # 8. Record pending prediction outcome for update()
        pred_key = (request_id, target_stage_id, token_id)
        self.pending_predictions[pred_key] = {
            "cstp_top_b": cstp_top_b,
            "st_top_b": st_top_b,
            "acsh_top_b": acsh_top_b,
            "w_c": w_c,
            "w_m": w_m,
            "horizon": horizon,
            "budget": budget
        }

        return acsh_top_b

    def _calculate_utility(self, predicted: List[int], actual: List[int], budget: int) -> float:
        """
        Calculates instantaneous utility metric (Recall, HitRate, or F1).
        """
        p_set = set(predicted)
        a_set = set(actual)
        hits = len(p_set.intersection(a_set))
        num_act = len(a_set)

        if self.reliability_objective == "recall":
            if num_act > 0:
                return float(hits) / float(num_act)
            else:
                return 0.0

        else:
            if self.reliability_objective == "hit_rate":
                if hits > 0:
                    return 1.0
                else:
                    return 0.0
            else:
                # F1 objective
                if num_act == 0 or len(p_set) == 0:
                    return 0.0
                prec = float(hits) / float(len(p_set))
                rec = float(hits) / float(num_act)
                denom = prec + rec
                if denom > 0.0:
                    return float(2.0 * prec * rec / denom)
                else:
                    return 0.0

    def update(
        self,
        request_id: int,
        target_stage_id: int,
        source_experts: List[int],
        actual_target_experts: List[int],
        token_id: Optional[int] = None,
        source_stage_id: Optional[int] = None
    ) -> None:
        """
        Updates component predictors and adapts ACSH reliability weights post-routing.
        """
        # 1. Update component predictors
        self.cstp.update(
            request_id=request_id,
            target_stage_id=target_stage_id,
            source_experts=source_experts,
            actual_target_experts=actual_target_experts,
            token_id=token_id,
            source_stage_id=source_stage_id
        )
        self.st_moe.update(
            request_id=request_id,
            target_stage_id=target_stage_id,
            source_experts=source_experts,
            actual_target_experts=actual_target_experts,
            token_id=token_id,
            source_stage_id=source_stage_id
        )

        # 2. Retrieve pending prediction record
        pred_key = (request_id, target_stage_id, token_id)
        if pred_key not in self.pending_predictions:
            # Fallback if prediction wasn't logged for this exact key
            return

        rec = self.pending_predictions[pred_key]
        del self.pending_predictions[pred_key]

        cstp_top_b = rec["cstp_top_b"]
        st_top_b = rec["st_top_b"]
        acsh_top_b = rec["acsh_top_b"]
        w_c = rec["w_c"]
        w_m = rec["w_m"]
        horizon = rec["horizon"]
        budget = rec["budget"]

        # 3. Calculate instantaneous event utilities
        r_c_event = self._calculate_utility(cstp_top_b, actual_target_experts, budget)
        r_m_event = self._calculate_utility(st_top_b, actual_target_experts, budget)
        r_acsh_event = self._calculate_utility(acsh_top_b, actual_target_experts, budget)

        # 4. Determine adaptive momentum beta (handling Fast Recovery)
        stage_key = (target_stage_id, horizon)
        active_beta = self.beta

        if self.variant == "fast_recovery":
            if request_id not in self.recent_errors:
                self.recent_errors[request_id] = {}
                self.fast_cooldown[request_id] = {}

            if stage_key not in self.recent_errors[request_id]:
                self.recent_errors[request_id][stage_key] = []
                self.fast_cooldown[request_id][stage_key] = 0

            # Instantaneous error = 1.0 - recall
            err = 1.0 - r_acsh_event
            err_list = self.recent_errors[request_id][stage_key]
            err_list.append(err)
            if len(err_list) > self.shift_window:
                err_list.pop(0)

            # Check if cooldown active
            cd = self.fast_cooldown[request_id][stage_key]
            if cd > 0:
                active_beta = self.fast_beta
                self.fast_cooldown[request_id][stage_key] = cd - 1
            else:
                # Check for shift detection
                if len(err_list) >= self.shift_window:
                    mean_err = float(np.mean(err_list))
                    if mean_err >= self.shift_threshold:
                        # Shift detected! Activate fast beta
                        active_beta = self.fast_beta
                        self.fast_cooldown[request_id][stage_key] = self.fast_cooldown_steps
                        self.shift_detections.append({
                            "request_id": request_id,
                            "target_stage_id": target_stage_id,
                            "token_id": token_id,
                            "mean_error": round(mean_err, 4),
                            "active_beta": active_beta
                        })

        # 5. Causal EMA update of reliability estimates
        if self.variant == "global":
            old_rc = self.rel_c_global[request_id]
            old_rm = self.rel_m_global[request_id]

            new_rc = (active_beta * old_rc) + ((1.0 - active_beta) * r_c_event)
            new_rm = (active_beta * old_rm) + ((1.0 - active_beta) * r_m_event)

            self.rel_c_global[request_id] = float(new_rc)
            self.rel_m_global[request_id] = float(new_rm)
        else:
            # Stage-aware or Fast-recovery
            old_rc = self.rel_c_stage[request_id][stage_key]
            old_rm = self.rel_m_stage[request_id][stage_key]

            new_rc = (active_beta * old_rc) + ((1.0 - active_beta) * r_c_event)
            new_rm = (active_beta * old_rm) + ((1.0 - active_beta) * r_m_event)

            self.rel_c_stage[request_id][stage_key] = float(new_rc)
            self.rel_m_stage[request_id][stage_key] = float(new_rm)

        # 6. Record diagnostics history
        self.weight_history.append({
            "request_id": request_id,
            "token_id": token_id,
            "target_stage_id": target_stage_id,
            "horizon": horizon,
            "budget": budget,
            "w_c": round(w_c, 4),
            "w_m": round(w_m, 4),
            "r_c_event": round(r_c_event, 4),
            "r_m_event": round(r_m_event, 4),
            "r_acsh_event": round(r_acsh_event, 4),
            "new_r_c": round(new_rc, 4),
            "new_r_m": round(new_rm, 4),
            "active_beta": active_beta
        })

