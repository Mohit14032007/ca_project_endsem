"""
Common Predictor Interface and Standard Adapters for MoE Expert Prediction.

This module provides the abstract base class and standardized wrappers
allowing all predictors (ST-MoE paper, ST-MoE legacy, CSTP, and baselines)
to be evaluated interchangeably under identical conditions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class BasePredictor(ABC):
    """
    Abstract base class defining the common interface for all MoE expert predictors.
    """

    @abstractmethod
    def reset(self, request_id: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Resets all sequence-specific / request-specific temporal tracking state.

        Args:
            request_id: Unique identifier for the generation request.
            metadata: Optional dictionary with sequence parameters.
        """
        pass

    def reset_request_state(self, request_id: int) -> None:
        """
        Backwards-compatible alias for reset().
        """
        self.reset(request_id=request_id)

    @abstractmethod
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
        Predicts top-B expert candidates for the target MoE stage.

        Args:
            request_id: Unique sequence request identifier.
            target_stage_id: Destination MoE stage index (e.g., 0..num_moe_stages-1).
            current_source_experts: Experts selected at the source stage.
            budget: Maximum number of experts to prefetch (B).
            token_id: Optional token sequence index.
            source_stage_id: Optional source stage index (defaults to target_stage_id - 1).

        Returns:
            List of predicted expert IDs of length at most budget.
        """
        pass

    @abstractmethod
    def update(
        self,
        request_id: int,
        target_stage_id: int,
        source_experts: List[int],
        actual_target_experts: List[int],
        token_id: Optional[int] = None
    ) -> None:
        """
        Updates internal prediction state strictly after actual target routing is revealed.

        Args:
            request_id: Unique sequence request identifier.
            target_stage_id: MoE stage that just executed.
            source_experts: Experts selected at the predecessor stage.
            actual_target_experts: Actual experts selected at target_stage_id.
            token_id: Optional token sequence index.
        """
        pass

    def get_candidate_scores(
        self,
        request_id: int,
        target_stage_id: int,
        current_source_experts: List[int],
        token_id: Optional[int] = None
    ) -> Dict[int, float]:
        """
        Returns full score dictionary for candidate experts if supported by predictor.
        """
        return {}


class STMoELegacyWrapper(BasePredictor):
    """
    Wrapper around the legacy STMoEPredictor to conform to BasePredictor interface.
    """

    def __init__(self, legacy_predictor: Any):
        self.inner = legacy_predictor
        self.name = "st_moe_legacy"

    def reset(self, request_id: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.inner.reset_request_state(request_id)

    def predict(
        self,
        request_id: int,
        target_stage_id: int,
        current_source_experts: List[int],
        budget: int,
        token_id: Optional[int] = None,
        source_stage_id: Optional[int] = None
    ) -> List[int]:
        if budget <= 0:
            return []
        return self.inner.predict(
            request_id=request_id,
            target_stage_id=target_stage_id,
            current_source_experts=current_source_experts,
            budget=budget
        )

    def update(
        self,
        request_id: int,
        target_stage_id: int,
        source_experts: List[int],
        actual_target_experts: List[int],
        token_id: Optional[int] = None,
        source_stage_id: Optional[int] = None
    ) -> None:
        self.inner.update(
            request_id=request_id,
            target_stage_id=target_stage_id,
            source_experts=source_experts,
            actual_target_experts=actual_target_experts
        )


class CSTPWrapper(BasePredictor):
    """
    Wrapper around CSTPPredictor to conform to BasePredictor interface
    without modifying its underlying mathematical formulation.
    """

    def __init__(self, cstp_predictor: Any, name: str = "cstp"):
        self.inner = cstp_predictor
        self.name = name

    def reset(self, request_id: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.inner.reset_request_state(request_id)

    def predict(
        self,
        request_id: int,
        target_stage_id: int,
        current_source_experts: List[int],
        budget: int,
        token_id: Optional[int] = None,
        source_stage_id: Optional[int] = None
    ) -> List[int]:
        if budget <= 0:
            return []
        return self.inner.predict(
            request_id=request_id,
            target_stage_id=target_stage_id,
            current_source_experts=current_source_experts,
            budget=budget,
            token_id=token_id,
            source_stage_id=source_stage_id
        )

    def update(
        self,
        request_id: int,
        target_stage_id: int,
        source_experts: List[int],
        actual_target_experts: List[int],
        token_id: Optional[int] = None,
        source_stage_id: Optional[int] = None
    ) -> None:
        self.inner.update(
            request_id=request_id,
            target_stage_id=target_stage_id,
            source_experts=source_experts,
            actual_target_experts=actual_target_experts,
            token_id=token_id,
            source_stage_id=source_stage_id
        )

    def get_candidate_scores(
        self,
        request_id: int,
        target_stage_id: int,
        current_source_experts: List[int],
        token_id: Optional[int] = None
    ) -> Dict[int, float]:
        s_spatial = self.inner.compute_spatial_score(target_stage_id, current_source_experts)
        s_temporal = self.inner.compute_temporal_score(request_id, target_stage_id)
        composite = (self.inner.alpha * s_spatial) + ((1.0 - self.inner.alpha) * s_temporal)
        scores_dict = {}
        for idx in range(len(composite)):
            scores_dict[idx] = float(composite[idx])
        return scores_dict


class RandomPredictor(BasePredictor):
    """
    Baseline predictor selecting candidate experts uniformly at random.
    Deterministic with seed for reproducibility.
    """

    def __init__(self, num_routed_experts: int = 256, seed: int = 42):
        self.num_routed_experts = num_routed_experts
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.name = "random"

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
        if budget <= 0:
            return []
        effective_budget = min(budget, self.num_routed_experts)
        candidates = list(range(self.num_routed_experts))
        chosen = self.rng.choice(candidates, size=effective_budget, replace=False)
        return [int(x) for x in chosen]

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


class TemporalOnlyPredictor(BasePredictor):
    """
    Ablation baseline predicting solely via temporal autocorrelation and target-stage marginal priors.
    Mathematically equivalent to CSTP with alpha=0.0.
    """

    def __init__(self, cstp_instance: Any):
        self.cstp = cstp_instance
        self.cstp.alpha = 0.0
        self.name = "temporal_only"

    def reset(self, request_id: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.cstp.reset_request_state(request_id)

    def predict(
        self,
        request_id: int,
        target_stage_id: int,
        current_source_experts: List[int],
        budget: int,
        token_id: Optional[int] = None,
        source_stage_id: Optional[int] = None
    ) -> List[int]:
        if budget <= 0:
            return []
        return self.cstp.predict(
            request_id=request_id,
            target_stage_id=target_stage_id,
            current_source_experts=current_source_experts,
            budget=budget,
            token_id=token_id,
            source_stage_id=source_stage_id
        )

    def update(
        self,
        request_id: int,
        target_stage_id: int,
        source_experts: List[int],
        actual_target_experts: List[int],
        token_id: Optional[int] = None,
        source_stage_id: Optional[int] = None
    ) -> None:
        self.cstp.update(
            request_id=request_id,
            target_stage_id=target_stage_id,
            source_experts=source_experts,
            actual_target_experts=actual_target_experts,
            token_id=token_id,
            source_stage_id=source_stage_id
        )


class SpatialOnlyPredictor(BasePredictor):
    """
    Ablation baseline predicting solely via spatial conditional transitions and target-stage marginal priors.
    Mathematically equivalent to CSTP with alpha=1.0.
    """

    def __init__(self, cstp_instance: Any):
        self.cstp = cstp_instance
        self.cstp.alpha = 1.0
        self.name = "spatial_only"

    def reset(self, request_id: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.cstp.reset_request_state(request_id)

    def predict(
        self,
        request_id: int,
        target_stage_id: int,
        current_source_experts: List[int],
        budget: int,
        token_id: Optional[int] = None,
        source_stage_id: Optional[int] = None
    ) -> List[int]:
        if budget <= 0:
            return []
        return self.cstp.predict(
            request_id=request_id,
            target_stage_id=target_stage_id,
            current_source_experts=current_source_experts,
            budget=budget,
            token_id=token_id,
            source_stage_id=source_stage_id
        )

    def update(
        self,
        request_id: int,
        target_stage_id: int,
        source_experts: List[int],
        actual_target_experts: List[int],
        token_id: Optional[int] = None,
        source_stage_id: Optional[int] = None
    ) -> None:
        self.cstp.update(
            request_id=request_id,
            target_stage_id=target_stage_id,
            source_experts=source_experts,
            actual_target_experts=actual_target_experts,
            token_id=token_id,
            source_stage_id=source_stage_id
        )


class CSTPSTMoEHybridPredictor(BasePredictor):
    """
    Exploratory hybrid predictor combining CSTP composite probability scores
    and ST-MoE saturating confidence counters.
    """

    def __init__(self, cstp_predictor: Any, st_moe_predictor: Any, cstp_weight: float = 0.5):
        self.cstp = cstp_predictor
        self.st_moe = st_moe_predictor
        self.cstp_weight = cstp_weight
        self.name = "cstp_st_moe_hybrid"

    def reset(self, request_id: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.cstp.reset_request_state(request_id)
        self.st_moe.reset(request_id, metadata)

    def predict(
        self,
        request_id: int,
        target_stage_id: int,
        current_source_experts: List[int],
        budget: int,
        token_id: Optional[int] = None,
        source_stage_id: Optional[int] = None
    ) -> List[int]:
        if budget <= 0:
            return []

        # CSTP scores (probabilities summing to 1.0)
        s_spatial = self.cstp.compute_spatial_score(
            target_stage_id=target_stage_id,
            source_experts=current_source_experts,
            source_stage_id=source_stage_id
        )
        s_temporal = self.cstp.compute_temporal_score(request_id, target_stage_id)
        cstp_scores = (self.cstp.alpha * s_spatial) + ((1.0 - self.cstp.alpha) * s_temporal)

        # ST-MoE scores (integer counters, max 3 per source + 2 temporal)
        st_scores = self.st_moe.compute_confidence_scores(
            request_id=request_id,
            target_stage_id=target_stage_id,
            current_source_experts=current_source_experts,
            source_stage_id=source_stage_id
        )

        num_experts = self.cstp.num_routed_experts
        st_arr = np.zeros(num_experts, dtype=np.float64)
        for e, v in st_scores.items():
            if 0 <= e < num_experts:
                st_arr[e] = float(v)

        # Normalize ST-MoE scores
        max_st = np.max(st_arr)
        if max_st > 0.0:
            st_arr = st_arr / max_st

        # Normalize CSTP scores
        max_cstp = np.max(cstp_scores)
        if max_cstp > 0.0:
            cstp_arr = cstp_scores / max_cstp
        else:
            cstp_arr = cstp_scores

        combined_scores = (self.cstp_weight * cstp_arr) + ((1.0 - self.cstp_weight) * st_arr)

        all_experts = list(range(num_experts))
        all_experts.sort(key=lambda idx: (-combined_scores[idx], idx))

        effective_budget = min(budget, num_experts)
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


class FrequencyPredictor(BasePredictor):
    """
    Baseline predictor selecting experts based strictly on target-stage marginal frequency.
    """

    def __init__(self, num_moe_stages: int = 57, num_routed_experts: int = 256):
        self.num_moe_stages = num_moe_stages
        self.num_routed_experts = num_routed_experts
        self.frequency_counts = np.zeros((num_moe_stages, num_routed_experts), dtype=np.int64)
        self.name = "frequency_only"

    def fit_profile(self, training_records: List[Any]) -> None:
        for rec in training_records:
            s = rec.moe_stage_id
            e = rec.expert_id
            if s >= 0 and s < self.num_moe_stages:
                if e >= 0 and e < self.num_routed_experts:
                    self.frequency_counts[s, e] += 1

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
        if budget <= 0:
            return []
        if target_stage_id < 0 or target_stage_id >= self.num_moe_stages:
            return list(range(min(budget, self.num_routed_experts)))
        counts = self.frequency_counts[target_stage_id]
        sorted_indices = sorted(
            list(range(self.num_routed_experts)),
            key=lambda idx: (-counts[idx], idx)
        )
        return sorted_indices[:budget]

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


class TemporalOnlyPredictor(BasePredictor):
    """
    Ablation baseline predicting solely via temporal autocorrelation and target-stage marginal priors.
    Mathematically equivalent to CSTP with alpha=0.0.
    """

    def __init__(self, cstp_instance: Any):
        self.cstp = cstp_instance
        self.cstp.alpha = 0.0
        self.name = "temporal_only"

    def reset(self, request_id: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.cstp.reset_request_state(request_id)

    def predict(
        self,
        request_id: int,
        target_stage_id: int,
        current_source_experts: List[int],
        budget: int,
        token_id: Optional[int] = None,
        source_stage_id: Optional[int] = None
    ) -> List[int]:
        if budget <= 0:
            return []
        return self.cstp.predict(
            request_id=request_id,
            target_stage_id=target_stage_id,
            current_source_experts=current_source_experts,
            budget=budget,
            token_id=token_id,
            source_stage_id=source_stage_id
        )

    def update(
        self,
        request_id: int,
        target_stage_id: int,
        source_experts: List[int],
        actual_target_experts: List[int],
        token_id: Optional[int] = None,
        source_stage_id: Optional[int] = None
    ) -> None:
        self.cstp.update(
            request_id=request_id,
            target_stage_id=target_stage_id,
            source_experts=source_experts,
            actual_target_experts=actual_target_experts,
            token_id=token_id,
            source_stage_id=source_stage_id
        )


class SpatialOnlyPredictor(BasePredictor):
    """
    Ablation baseline predicting solely via spatial conditional transitions and target-stage marginal priors.
    Mathematically equivalent to CSTP with alpha=1.0.
    """

    def __init__(self, cstp_instance: Any):
        self.cstp = cstp_instance
        self.cstp.alpha = 1.0
        self.name = "spatial_only"

    def reset(self, request_id: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.cstp.reset_request_state(request_id)

    def predict(
        self,
        request_id: int,
        target_stage_id: int,
        current_source_experts: List[int],
        budget: int,
        token_id: Optional[int] = None,
        source_stage_id: Optional[int] = None
    ) -> List[int]:
        if budget <= 0:
            return []
        return self.cstp.predict(
            request_id=request_id,
            target_stage_id=target_stage_id,
            current_source_experts=current_source_experts,
            budget=budget,
            token_id=token_id,
            source_stage_id=source_stage_id
        )

    def update(
        self,
        request_id: int,
        target_stage_id: int,
        source_experts: List[int],
        actual_target_experts: List[int],
        token_id: Optional[int] = None,
        source_stage_id: Optional[int] = None
    ) -> None:
        self.cstp.update(
            request_id=request_id,
            target_stage_id=target_stage_id,
            source_experts=source_experts,
            actual_target_experts=actual_target_experts,
            token_id=token_id,
            source_stage_id=source_stage_id
        )


class CSTPSTMoEHybridPredictor(BasePredictor):
    """
    Exploratory hybrid predictor combining CSTP composite probability scores
    and ST-MoE saturating confidence counters.
    """

    def __init__(self, cstp_predictor: Any, st_moe_predictor: Any, cstp_weight: float = 0.5):
        self.cstp = cstp_predictor
        self.st_moe = st_moe_predictor
        self.cstp_weight = cstp_weight
        self.name = "cstp_st_moe_hybrid"

    def reset(self, request_id: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.cstp.reset_request_state(request_id)
        self.st_moe.reset(request_id, metadata)

    def predict(
        self,
        request_id: int,
        target_stage_id: int,
        current_source_experts: List[int],
        budget: int,
        token_id: Optional[int] = None,
        source_stage_id: Optional[int] = None
    ) -> List[int]:
        if budget <= 0:
            return []

        # CSTP scores (probabilities summing to 1.0)
        s_spatial = self.cstp.compute_spatial_score(
            target_stage_id=target_stage_id,
            source_experts=current_source_experts,
            source_stage_id=source_stage_id
        )
        s_temporal = self.cstp.compute_temporal_score(request_id, target_stage_id)
        cstp_scores = (self.cstp.alpha * s_spatial) + ((1.0 - self.cstp.alpha) * s_temporal)

        # ST-MoE scores (integer counters, max 3 per source + 2 temporal)
        st_scores = self.st_moe.compute_confidence_scores(
            request_id=request_id,
            target_stage_id=target_stage_id,
            current_source_experts=current_source_experts,
            source_stage_id=source_stage_id
        )

        num_experts = self.cstp.num_routed_experts
        st_arr = np.zeros(num_experts, dtype=np.float64)
        for e, v in st_scores.items():
            if 0 <= e < num_experts:
                st_arr[e] = float(v)

        # Normalize ST-MoE scores
        max_st = np.max(st_arr)
        if max_st > 0.0:
            st_arr = st_arr / max_st

        # Normalize CSTP scores
        max_cstp = np.max(cstp_scores)
        if max_cstp > 0.0:
            cstp_arr = cstp_scores / max_cstp
        else:
            cstp_arr = cstp_scores

        combined_scores = (self.cstp_weight * cstp_arr) + ((1.0 - self.cstp_weight) * st_arr)

        all_experts = list(range(num_experts))
        all_experts.sort(key=lambda idx: (-combined_scores[idx], idx))

        effective_budget = min(budget, num_experts)
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
