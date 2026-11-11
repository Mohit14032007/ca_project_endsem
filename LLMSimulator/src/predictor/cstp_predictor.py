"""
Proposed CSTP (Confidence- and Frequency-Aware Spatio-Temporal Predictor).

Combines:
1. Dirichlet-smoothed spatial conditional transition probabilities.
2. Dirichlet-smoothed temporal autocorrelation probabilities.
3. Target-layer empirical marginal frequency priors.
4. Composite weighting with parameter alpha in [0.0, 1.0].
5. Configurable smoothing constants lambda and epsilon.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np


class CSTPPredictor:
    """
    Proposed Confidence- and Frequency-Aware Spatio-Temporal Predictor (CSTP).
    """

    def __init__(
        self,
        num_moe_stages: int = 57,
        num_routed_experts: int = 256,
        top_k: int = 8,
        alpha: float = 0.5,
        smoothing_lambda: float = 1.0,
        prior_epsilon: float = 0.1,
        online_adaptation: bool = False,
        prediction_horizon: int = 1
    ):
        self.num_moe_stages = num_moe_stages
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.alpha = alpha
        self.smoothing_lambda = smoothing_lambda
        self.prior_epsilon = prior_epsilon
        self.online_adaptation = online_adaptation
        self.prediction_horizon = prediction_horizon

        # Target-stage marginal frequency counts: stage -> array of shape (E,)
        self.stage_marginal_counts = np.zeros(
            (num_moe_stages, num_routed_experts), dtype=np.float64
        )

        # Spatial co-activation counts:
        # transition (L -> L+H): shape (num_moe_stages - prediction_horizon, E, E)
        if num_moe_stages > prediction_horizon:
            self.spatial_counts = np.zeros(
                (num_moe_stages - prediction_horizon, num_routed_experts, num_routed_experts),
                dtype=np.float64
            )
        else:
            self.spatial_counts = np.zeros((1, num_routed_experts, num_routed_experts), dtype=np.float64)

        # Temporal transition counts:
        # temporal_counts[L, w, v] = count of token transition at stage L from w at t-1 to v at t
        self.temporal_counts = np.zeros(
            (num_moe_stages, num_routed_experts, num_routed_experts),
            dtype=np.float64
        )

        # History table: request_id -> stage_id -> set of expert IDs from previous token
        self.history_table: Dict[int, Dict[int, Set[int]]] = {}

    def fit_profile(
        self,
        training_records: List
    ) -> None:
        """
        Fits marginal frequency priors and initial spatial/temporal transition counts
        strictly on permitted training/profiling records.
        """
        # Group records by (request_id, token_id, moe_stage_id)
        grouped: Dict[Tuple[int, int, int], List[int]] = {}
        for rec in training_records:
            key = (rec.request_id, rec.token_id, rec.moe_stage_id)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(rec.expert_id)

        # Populate marginal counts and spatial transitions
        requests = sorted(list(set(k[0] for k in grouped.keys())))

        for req in requests:
            req_keys = [k for k in grouped.keys() if k[0] == req]
            req_tokens = sorted(list(set(k[1] for k in req_keys)))

            for tok_idx, tok in enumerate(req_tokens):
                # Process stages within token
                for stage in range(self.num_moe_stages):
                    curr_key = (req, tok, stage)
                    curr_experts = grouped.get(curr_key, [])

                    for v in curr_experts:
                        if v >= 0 and v < self.num_routed_experts:
                            self.stage_marginal_counts[stage, v] += 1.0

                    # Spatial co-activations (stage -> stage + horizon)
                    if stage < self.num_moe_stages - self.prediction_horizon:
                        next_key = (req, tok, stage + self.prediction_horizon)
                        next_experts = grouped.get(next_key, [])
                        for u in curr_experts:
                            for v in next_experts:
                                if u >= 0 and u < self.num_routed_experts:
                                    if v >= 0 and v < self.num_routed_experts:
                                        self.spatial_counts[stage, u, v] += 1.0

                # Temporal transitions (token tok_idx-1 -> tok_idx at each stage)
                if tok_idx > 0:
                    prev_tok = req_tokens[tok_idx - 1]
                    for stage in range(self.num_moe_stages):
                        prev_key = (req, prev_tok, stage)
                        curr_key = (req, tok, stage)
                        prev_experts = grouped.get(prev_key, [])
                        curr_experts = grouped.get(curr_key, [])

                        for w in prev_experts:
                            for v in curr_experts:
                                if w >= 0 and w < self.num_routed_experts:
                                    if v >= 0 and v < self.num_routed_experts:
                                        self.temporal_counts[stage, w, v] += 1.0

    def get_marginal_prior(self, stage_id: int) -> np.ndarray:
        """
        Calculates Dirichlet-smoothed target-stage prior distribution P_prior(v | stage_id).
        """
        counts = self.stage_marginal_counts[stage_id]
        total_counts = np.sum(counts)
        denom = total_counts + (self.num_routed_experts * self.prior_epsilon)
        if denom > 0.0:
            prior = (counts + self.prior_epsilon) / denom
        else:
            prior = np.ones(self.num_routed_experts, dtype=np.float64) / float(self.num_routed_experts)
        return prior

    def compute_spatial_score(
        self,
        target_stage_id: int,
        source_experts: List[int],
        source_stage_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculates smoothed spatial conditional score S_spatial(v) for all candidate experts v.
        """
        prior = self.get_marginal_prior(target_stage_id)

        if target_stage_id < self.prediction_horizon or len(source_experts) == 0:
            return prior.copy()

        if source_stage_id is not None:
            source_stage = source_stage_id
        else:
            source_stage = target_stage_id - self.prediction_horizon

        if source_stage < 0 or source_stage >= self.spatial_counts.shape[0]:
            return prior.copy()

        valid_sources = [u for u in source_experts if u >= 0 and u < self.num_routed_experts]
        if len(valid_sources) == 0:
            return prior.copy()

        spatial_accum = np.zeros(self.num_routed_experts, dtype=np.float64)

        for u in valid_sources:
            co_counts = self.spatial_counts[source_stage, u, :]
            total_co = np.sum(co_counts)
            denom = total_co + self.smoothing_lambda
            if denom > 0.0:
                p_cond = (co_counts + (self.smoothing_lambda * prior)) / denom
            else:
                p_cond = prior.copy()
            spatial_accum += p_cond

        return spatial_accum / float(len(valid_sources))

    def compute_temporal_score(
        self,
        request_id: int,
        target_stage_id: int
    ) -> np.ndarray:
        """
        Calculates smoothed temporal transition score S_temporal(v) for all candidate experts v.
        """
        prior = self.get_marginal_prior(target_stage_id)

        req_hist = self.history_table.get(request_id, {})
        prev_experts = req_hist.get(target_stage_id, set())

        if len(prev_experts) == 0:
            # Cold-start: fallback to target-stage marginal prior
            return prior.copy()

        valid_prev = [w for w in prev_experts if w >= 0 and w < self.num_routed_experts]
        if len(valid_prev) == 0:
            return prior.copy()

        temporal_accum = np.zeros(self.num_routed_experts, dtype=np.float64)

        for w in valid_prev:
            trans_counts = self.temporal_counts[target_stage_id, w, :]
            total_trans = np.sum(trans_counts)
            denom = total_trans + self.smoothing_lambda
            if denom > 0.0:
                p_trans = (trans_counts + (self.smoothing_lambda * prior)) / denom
            else:
                p_trans = prior.copy()
            temporal_accum += p_trans

        return temporal_accum / float(len(valid_prev))

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
        Predicts top-B expert candidates for target_stage_id using composite score.
        """
        s_spatial = self.compute_spatial_score(
            target_stage_id=target_stage_id,
            source_experts=current_source_experts,
            source_stage_id=source_stage_id
        )
        s_temporal = self.compute_temporal_score(request_id, target_stage_id)

        composite_score = (self.alpha * s_spatial) + ((1.0 - self.alpha) * s_temporal)

        # Sort candidate indices by composite score descending, breaking ties by index ascending
        expert_indices = list(range(self.num_routed_experts))
        sorted_candidates = sorted(
            expert_indices,
            key=lambda idx: (-composite_score[idx], idx)
        )

        return sorted_candidates[:budget]

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
        Updates History Table and (if online_adaptation is True) updates transition counts.
        """
        # 1. Update temporal transition counts if online adaptation is enabled
        if self.online_adaptation:
            req_hist = self.history_table.get(request_id, {})
            prev_experts = req_hist.get(target_stage_id, set())

            for v in actual_target_experts:
                if v >= 0 and v < self.num_routed_experts:
                    self.stage_marginal_counts[target_stage_id, v] += 1.0

            # Online spatial transition
            if target_stage_id >= self.prediction_horizon and len(source_experts) > 0:
                if source_stage_id is not None:
                    source_stage = source_stage_id
                else:
                    source_stage = target_stage_id - self.prediction_horizon

                if 0 <= source_stage < self.spatial_counts.shape[0]:
                    for u in source_experts:
                        for v in actual_target_experts:
                            if u >= 0 and u < self.num_routed_experts:
                                if v >= 0 and v < self.num_routed_experts:
                                    self.spatial_counts[source_stage, u, v] += 1.0

            # Online temporal transition
            if len(prev_experts) > 0:
                for w in prev_experts:
                    for v in actual_target_experts:
                        if w >= 0 and w < self.num_routed_experts:
                            if v >= 0 and v < self.num_routed_experts:
                                self.temporal_counts[target_stage_id, w, v] += 1.0

        # 2. Update History Table for this request and stage
        if request_id not in self.history_table:
            self.history_table[request_id] = {}
        self.history_table[request_id][target_stage_id] = set(actual_target_experts)

    def reset_request_state(self, request_id: int) -> None:
        """Clears temporal history for a completed request."""
        if request_id in self.history_table:
            del self.history_table[request_id]

