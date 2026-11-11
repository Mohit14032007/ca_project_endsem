"""
Faithful ST-MoE Predictor Implementation.

Reference:
"A Spatio-Temporal Expert Prefetching Framework for Efficient MoE-based LLM Inference"
Zhao et al., 2026 (arXiv:2606.15453)

Implements:
1. Cross-Layer Correlation Table (CCT) with 2-bit saturating counters.
2. History Table (HT) tracking previous-token routing per request and stage.
3. Spatial and temporal candidate selection under budget B.
4. Online causal counter updates and candidate replacement.
"""

from typing import Dict, List, Optional, Set, Tuple


class STMoEPredictor:
    """
    Faithful implementation of ST-MoE spatio-temporal expert prediction.
    """

    def __init__(
        self,
        num_moe_stages: int = 57,
        num_routed_experts: int = 256,
        top_k: int = 8,
        cct_capacity: int = 8,
        confidence_threshold: int = 2
    ):
        self.num_moe_stages = num_moe_stages
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.cct_capacity = cct_capacity
        self.confidence_threshold = confidence_threshold

        # CCT: stage -> source_expert -> dict of {dest_expert: counter (0..3)}
        # Stages 0..num_moe_stages-2 transitions to stage+1
        self.cct: Dict[int, Dict[int, Dict[int, int]]] = {}
        for stage in range(num_moe_stages - 1):
            self.cct[stage] = {}
            for u in range(num_routed_experts):
                self.cct[stage][u] = {}

        # History Table: request_id -> stage_id -> set of expert_ids from previous token
        self.history_table: Dict[int, Dict[int, Set[int]]] = {}

        # Cold-start fallback frequency prior per stage (optional offline profiling prior)
        self.profiling_priors: Dict[int, List[int]] = {}

    def set_profiling_prior(self, priors_by_stage: Dict[int, List[int]]) -> None:
        """Sets fallback marginal candidate ordering from profiling data."""
        self.profiling_priors = priors_by_stage

    def reset_request_state(self, request_id: int) -> None:
        """Clears temporal history for a completed request."""
        if request_id in self.history_table:
            del self.history_table[request_id]

    def predict(
        self,
        request_id: int,
        target_stage_id: int,
        current_source_experts: List[int],
        budget: int
    ) -> List[int]:
        """
        Predicts top-B expert candidates for target_stage_id.

        Args:
            request_id: Unique sequence request identifier.
            target_stage_id: Destination MoE stage (0..num_moe_stages-1).
            current_source_experts: Actual selected experts at source stage (target_stage_id - 1).
            budget: Maximum number of experts to prefetch (B).

        Returns:
            List of predicted expert IDs of length at most budget.
        """
        predicted_candidates: List[int] = []
        seen_candidates: Set[int] = set()

        # 1. Spatial Candidates from CCT (if predecessor stage exists)
        spatial_scores: Dict[int, int] = {}
        if target_stage_id > 0 and len(current_source_experts) > 0:
            source_stage = target_stage_id - 1
            stage_cct = self.cct.get(source_stage, {})

            for u in current_source_experts:
                dest_map = stage_cct.get(u, {})
                for v, counter in dest_map.items():
                    if counter >= self.confidence_threshold:
                        if v in spatial_scores:
                            if counter > spatial_scores[v]:
                                spatial_scores[v] = counter
                        else:
                            spatial_scores[v] = counter

        # Sort spatial candidates by confidence descending, then expert_id ascending
        sorted_spatial = sorted(
            spatial_scores.keys(),
            key=lambda exp: (-spatial_scores[exp], exp)
        )

        for cand in sorted_spatial:
            if cand not in seen_candidates:
                seen_candidates.add(cand)
                predicted_candidates.append(cand)
                if len(predicted_candidates) >= budget:
                    return predicted_candidates

        # 2. Temporal Candidates from History Table (HT)
        req_history = self.history_table.get(request_id, {})
        temporal_experts = req_history.get(target_stage_id, set())

        sorted_temporal = sorted(list(temporal_experts))
        for cand in sorted_temporal:
            if cand not in seen_candidates:
                seen_candidates.add(cand)
                predicted_candidates.append(cand)
                if len(predicted_candidates) >= budget:
                    return predicted_candidates

        # 3. Fallback: Lower-confidence spatial candidates (c == 1)
        if target_stage_id > 0 and len(current_source_experts) > 0:
            source_stage = target_stage_id - 1
            stage_cct = self.cct.get(source_stage, {})
            low_spatial: Dict[int, int] = {}
            for u in current_source_experts:
                dest_map = stage_cct.get(u, {})
                for v, counter in dest_map.items():
                    if counter < self.confidence_threshold:
                        if v in low_spatial:
                            if counter > low_spatial[v]:
                                low_spatial[v] = counter
                        else:
                            low_spatial[v] = counter

            sorted_low = sorted(
                low_spatial.keys(),
                key=lambda exp: (-low_spatial[exp], exp)
            )
            for cand in sorted_low:
                if cand not in seen_candidates:
                    seen_candidates.add(cand)
                    predicted_candidates.append(cand)
                    if len(predicted_candidates) >= budget:
                        return predicted_candidates

        # 4. Fallback: Profiling marginal prior
        fallback_list = self.profiling_priors.get(target_stage_id, [])
        for cand in fallback_list:
            if cand not in seen_candidates:
                seen_candidates.add(cand)
                predicted_candidates.append(cand)
                if len(predicted_candidates) >= budget:
                    return predicted_candidates

        return predicted_candidates

    def update(
        self,
        request_id: int,
        target_stage_id: int,
        source_experts: List[int],
        actual_target_experts: List[int]
    ) -> None:
        """
        Updates CCT and History Table strictly after target_stage_id routing is revealed.

        Args:
            request_id: Unique sequence request identifier.
            target_stage_id: The MoE stage that just executed.
            source_experts: Selected experts at stage target_stage_id - 1.
            actual_target_experts: Actual selected experts at target_stage_id.
        """
        actual_set = set(actual_target_experts)

        # 1. Update Spatial CCT for transition (target_stage_id - 1 -> target_stage_id)
        if target_stage_id > 0 and len(source_experts) > 0:
            source_stage = target_stage_id - 1
            stage_cct = self.cct.get(source_stage, {})

            for u in source_experts:
                dest_map = stage_cct.get(u, {})

                # Update existing entries
                for v in list(dest_map.keys()):
                    current_c = dest_map[v]
                    if v in actual_set:
                        if current_c < 3:
                            dest_map[v] = current_c + 1
                        else:
                            dest_map[v] = 3
                    else:
                        if current_c > 0:
                            dest_map[v] = current_c - 1
                        else:
                            dest_map[v] = 0

                # Insert newly observed actual experts not yet in dest_map
                for v_star in actual_target_experts:
                    if v_star not in dest_map:
                        if len(dest_map) < self.cct_capacity:
                            dest_map[v_star] = 1
                        else:
                            # Search for an entry with counter == 0 for replacement
                            zero_keys = [k for k, c in dest_map.items() if c == 0]
                            if len(zero_keys) > 0:
                                victim = zero_keys[0]
                                del dest_map[victim]
                                dest_map[v_star] = 1

        # 2. Update History Table (HT) for temporal tracking
        if request_id not in self.history_table:
            self.history_table[request_id] = {}
        self.history_table[request_id][target_stage_id] = set(actual_target_experts)

