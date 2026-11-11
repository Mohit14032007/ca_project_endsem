"""
Paper-Faithful ST-MoE Expert Predictor Implementation.

Reference:
"A Spatio-Temporal Expert Prefetching Framework for Efficient MoE-based LLM Inference"
Yingnan Zhao, Razvan Bunescu, Ahmed Louri, Avinash Karanth, and Ke Wang.
arXiv:2606.15453v1 [cs.AR], 13 Jun 2026.

Implements:
- Algorithm 1: Build Cross-Layer Correlation Table (CCT) with initial confidence 10 (decimal 2).
- Algorithm 2: Expert Predict Function with Equation 1 spatio-temporal score aggregation.
- Algorithm 3: Table Update Function with 2-bit saturating counters (0..3) and replacement.
- History Table (HT): Request-isolated temporal tracking initialized with confidence 10 (decimal 2).
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from src.predictor.base_predictor import BasePredictor


class STMoEPaperPredictor(BasePredictor):
    """
    Faithful reproduction of Zhao et al. (arXiv:2606.15453v1).
    """

    def __init__(
        self,
        num_moe_stages: int = 57,
        num_routed_experts: int = 256,
        top_k: int = 8,
        cct_capacity: int = 8,
        confidence_threshold: int = 2,
        literal_alg2_overwrite: bool = False,
        online_adaptation: bool = True,
        prediction_horizon: int = 1,
        scoring_mode: str = "confidence_additive"
    ):
        """
        Args:
            num_moe_stages: Total number of sequential MoE stages in model.
            num_routed_experts: Total pool of routed experts (E).
            top_k: Number of experts selected per token (K).
            cct_capacity: Maximum destination candidates stored per source expert (C = Top-K).
            confidence_threshold: Threshold for prefetch set (default 2, binary 10).
            literal_alg2_overwrite: If True, executes literal Alg 2 lines 7-9 (ConfScore[h]=1+1=2).
                                    If False (default), executes Equation 1 (CCT[e] + HT[e]).
            online_adaptation: If True (default), updates CCT 2-bit counters and performs replacement.
                               If False (frozen), CCT counters and entries remain unchanged.
            prediction_horizon: Step distance between source and target stage (default 1 for L->L+1).
            scoring_mode: "confidence_additive" (default, Equation 1 literal: CCT confidence + HT confidence)
                          or "indicator_intersection" (Interpretation B: I(e in CCT) + I(e in HT)).
        """
        self.num_moe_stages = num_moe_stages
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.cct_capacity = cct_capacity
        self.confidence_threshold = confidence_threshold
        self.literal_alg2_overwrite = literal_alg2_overwrite
        self.online_adaptation = online_adaptation
        self.prediction_horizon = prediction_horizon
        self.scoring_mode = scoring_mode
        self.name = f"st_moe_paper_{scoring_mode}"

        # CCT: source_stage -> source_expert -> dict of {dest_expert: counter (0..3)}
        # Transitions are (stage -> stage + prediction_horizon)
        num_transitions = max(1, num_moe_stages - prediction_horizon)
        self.cct: Dict[int, Dict[int, Dict[int, int]]] = {}
        for stage in range(num_transitions):
            self.cct[stage] = {}
            for u in range(num_routed_experts):
                self.cct[stage][u] = {}

        # History Table: request_id -> stage_id -> list of actual expert IDs from previous token
        # Each stored expert is associated with confidence 2 (binary 10)
        self.history_table: Dict[int, Dict[int, List[int]]] = {}

        # Optional offline profiling priors for cold-start tie-breaking
        self.profiling_priors: Dict[int, List[int]] = {}

    def reset(self, request_id: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Clears temporal history for a completed request to guarantee request isolation.
        """
        if request_id in self.history_table:
            del self.history_table[request_id]

    def set_cct_entry(
        self,
        source_stage_id: int,
        source_expert: int,
        dest_expert: int,
        confidence: int
    ) -> None:
        """
        Explicitly sets a CCT entry for deterministic testing and manual verification.
        """
        if source_stage_id not in self.cct:
            self.cct[source_stage_id] = {}
        if source_expert not in self.cct[source_stage_id]:
            self.cct[source_stage_id][source_expert] = {}
        self.cct[source_stage_id][source_expert][dest_expert] = confidence

    def set_history_entry(
        self,
        request_id: int,
        stage_id: int,
        expert_ids: List[int]
    ) -> None:
        """
        Explicitly sets History Table entries for deterministic testing.
        """
        if request_id not in self.history_table:
            self.history_table[request_id] = {}
        self.history_table[request_id][stage_id] = list(expert_ids)

    def set_profiling_prior(self, priors_by_stage: Dict[int, List[int]]) -> None:
        """
        Sets fallback marginal candidate ordering from profiling data.
        """
        self.profiling_priors = priors_by_stage

    def build_cct_from_profile(self, training_records: List[Any]) -> None:
        """
        Implements Algorithm 1: Build Cross-Layer Correlation Table (CCT).

        Counts co-activations across adjacent layers (stage L and stage L+1)
        for identical tokens, selects the Top-K destination candidates for
        each source expert, and initializes their confidence to binary 10 (decimal 2).
        """
        # Group records by (request_id, token_id, moe_stage_id)
        grouped: Dict[Tuple[int, int, int], List[int]] = {}
        for rec in training_records:
            key = (rec.request_id, rec.token_id, rec.moe_stage_id)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(rec.expert_id)

        # Build co-activation matrices for each stage transition
        num_transitions = max(1, self.num_moe_stages - self.prediction_horizon)
        co_matrices: Dict[int, np.ndarray] = {}
        for stage in range(num_transitions):
            co_matrices[stage] = np.zeros(
                (self.num_routed_experts, self.num_routed_experts),
                dtype=np.int64
            )

        unique_keys = sorted(list(grouped.keys()))
        for req, tok, stage in unique_keys:
            if stage < self.num_moe_stages - self.prediction_horizon:
                next_key = (req, tok, stage + self.prediction_horizon)
                if next_key in grouped:
                    src_experts = grouped[(req, tok, stage)]
                    dst_experts = grouped[next_key]
                    for u in src_experts:
                        for v in dst_experts:
                            if u >= 0 and u < self.num_routed_experts:
                                if v >= 0 and v < self.num_routed_experts:
                                    co_matrices[stage][u, v] += 1

        # Populate CCT with Top-K candidates per source expert, initialized to decimal 2 (10)
        for stage in range(num_transitions):
            for u in range(self.num_routed_experts):
                row = co_matrices[stage][u]
                dest_candidates = []
                for v in range(self.num_routed_experts):
                    dest_candidates.append(v)

                # Sort by co-activation count descending, break ties by destination ID ascending
                dest_candidates.sort(key=lambda idx: (-row[idx], idx))

                # Store Top-K candidates with count > 0, up to cct_capacity
                self.cct[stage][u] = {}
                stored_count = 0
                for v in dest_candidates:
                    if row[v] > 0 and stored_count < self.cct_capacity:
                        self.cct[stage][u][v] = 2  # Binary 10: weakly preferred
                        stored_count += 1

    def compute_confidence_scores(
        self,
        request_id: int,
        target_stage_id: int,
        current_source_experts: List[int],
        source_stage_id: Optional[int] = None
    ) -> Dict[int, int]:
        """
        Computes the combined spatio-temporal confidence scores according to
        Algorithm 2 and Equation 1.

        Returns:
            Dictionary mapping candidate expert ID to its total confidence score.
        """
        conf_score: Dict[int, int] = {}
        for exp_id in range(self.num_routed_experts):
            conf_score[exp_id] = 0

        if self.scoring_mode == "indicator_intersection":
            # Interpretation B: Score(e) = I(e in CCT) + I(e in HT) in {0, 1, 2}
            in_cct = set()
            if target_stage_id >= self.prediction_horizon and len(current_source_experts) > 0:
                if source_stage_id is not None:
                    source_stage = source_stage_id
                else:
                    source_stage = target_stage_id - self.prediction_horizon
                stage_cct = self.cct.get(source_stage, {})
                for j in current_source_experts:
                    dest_map = stage_cct.get(j, {})
                    for k in dest_map.keys():
                        if 0 <= k < self.num_routed_experts:
                            in_cct.add(k)

            req_hist = self.history_table.get(request_id, {})
            ht_experts = set(req_hist.get(target_stage_id, []))

            for exp_id in range(self.num_routed_experts):
                s = 0
                if exp_id in in_cct:
                    s += 1
                if exp_id in ht_experts:
                    s += 1
                conf_score[exp_id] = s

            return conf_score

        # Default scoring_mode: "confidence_additive" (Equation 1 literal)
        # Spatial accumulation (Algorithm 2, Lines 3-5)
        if target_stage_id >= self.prediction_horizon and len(current_source_experts) > 0:
            if source_stage_id is not None:
                source_stage = source_stage_id
            else:
                source_stage = target_stage_id - self.prediction_horizon

            stage_cct = self.cct.get(source_stage, {})

            for j in current_source_experts:
                dest_map = stage_cct.get(j, {})
                for k, counter in dest_map.items():
                    if k >= 0 and k < self.num_routed_experts:
                        conf_score[k] = conf_score[k] + counter

        # Temporal contribution from History Table (HT)
        req_hist = self.history_table.get(request_id, {})
        ht_experts = req_hist.get(target_stage_id, [])

        for h in ht_experts:
            if h >= 0 and h < self.num_routed_experts:
                if self.literal_alg2_overwrite:
                    # Literal Algorithm 2 Lines 7-9: ConfScore[h] <- 1; ConfScore[h] <- ConfScore[h] + 1
                    conf_score[h] = 2
                else:
                    # Equation 1: Score(e) = CCT(e) + HT(e), where HT(e) = 2
                    conf_score[h] = conf_score[h] + 2

        return conf_score

    def predict_threshold(
        self,
        request_id: int,
        target_stage_id: int,
        current_source_experts: List[int],
        source_stage_id: Optional[int] = None
    ) -> List[int]:
        """
        Returns the variable-sized PrefetchSet defined by Algorithm 2 Line 10:
        { k | ConfScore[k] >= confidence_threshold }
        """
        scores = self.compute_confidence_scores(
            request_id=request_id,
            target_stage_id=target_stage_id,
            current_source_experts=current_source_experts,
            source_stage_id=source_stage_id
        )
        prefetch_set = []
        for exp_id in range(self.num_routed_experts):
            if scores[exp_id] >= self.confidence_threshold:
                prefetch_set.append(exp_id)
        prefetch_set.sort(key=lambda idx: (-scores[idx], idx))
        return prefetch_set

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
        Implements budget-constrained prediction for standardized prefetching evaluation.

        Ranks candidates by ConfScore descending, breaking ties by expert ID ascending.
        Returns the top B candidates.
        """
        if budget <= 0:
            return []

        scores = self.compute_confidence_scores(
            request_id=request_id,
            target_stage_id=target_stage_id,
            current_source_experts=current_source_experts,
            source_stage_id=source_stage_id
        )

        all_experts = list(range(self.num_routed_experts))
        all_experts.sort(key=lambda idx: (-scores[idx], idx))

        effective_budget = min(budget, self.num_routed_experts)
        return all_experts[:effective_budget]

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
        Implements Algorithm 3: Table Update Function.

        Updates CCT 2-bit saturating counters and replaces zero-confidence candidates
        strictly after the gating network reveals actual target routing.
        If self.online_adaptation is False, CCT updates are bypassed.
        Updates History Table with the latest Top-K actual selections.
        """
        actual_set = set(actual_target_experts)

        # 1. Update CCT only if online adaptation is enabled
        if self.online_adaptation:
            if target_stage_id >= self.prediction_horizon and len(source_experts) > 0:
                if source_stage_id is not None:
                    source_stage = source_stage_id
                else:
                    source_stage = target_stage_id - self.prediction_horizon

                if source_stage not in self.cct:
                    self.cct[source_stage] = {}

                for j in source_experts:
                    if j not in self.cct[source_stage]:
                        self.cct[source_stage][j] = {}
                    dest_map = self.cct[source_stage][j]

                    stored_candidates = list(dest_map.keys())
                    for k in stored_candidates:
                        conf = dest_map[k]
                        if k in actual_set:
                            # Match: increment counter up to 3 (Algorithm 3, Lines 6-7)
                            if conf < 3:
                                dest_map[k] = conf + 1
                            else:
                                dest_map[k] = 3
                        else:
                            # Mismatch: decrement counter (Algorithm 3, Lines 8-10)
                            if conf > 0:
                                dest_map[k] = conf - 1
                            else:
                                # ConfScore == 0: eligible for replacement (Algorithm 3, Lines 11-14)
                                prefetch_set = set(dest_map.keys())
                                eligible_replacements = [
                                    cand for cand in actual_target_experts
                                    if cand not in prefetch_set
                                ]
                                if len(eligible_replacements) > 0:
                                    # Deterministic replacement: select lowest eligible expert ID
                                    eligible_replacements.sort()
                                    g = eligible_replacements[0]
                                    del dest_map[k]
                                    dest_map[g] = 2  # Binary 10 (decimal 2)

                    # If CCT entry has capacity remaining, insert un-stored actual target experts
                    if len(dest_map) < self.cct_capacity:
                        prefetch_set = set(dest_map.keys())
                        for cand in actual_target_experts:
                            if cand not in prefetch_set and len(dest_map) < self.cct_capacity:
                                dest_map[cand] = 2  # Binary 10 (decimal 2)
                                prefetch_set.add(cand)

        # 2. Update History Table (HT) for temporal tracking
        if request_id not in self.history_table:
            self.history_table[request_id] = {}
        self.history_table[request_id][target_stage_id] = list(actual_target_experts)

    def get_candidate_scores(
        self,
        request_id: int,
        target_stage_id: int,
        current_source_experts: List[int],
        token_id: Optional[int] = None
    ) -> Dict[int, float]:
        """
        Returns full score dictionary for candidate experts.
        """
        scores = self.compute_confidence_scores(
            request_id=request_id,
            target_stage_id=target_stage_id,
            current_source_experts=current_source_experts
        )
        result = {}
        for exp_id, val in scores.items():
            result[exp_id] = float(val)
        return result

