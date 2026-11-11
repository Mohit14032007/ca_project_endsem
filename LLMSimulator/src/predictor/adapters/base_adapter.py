"""
Base Dataset Adapter Interface and Standard Routing Record Schema.

Every dataset adapter maps disparate raw routing formats (CSV, Parquet, TSV)
into a unified, strictly-causal sequence of PredictionRecord instances.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
import numpy as np


class PredictionRecord:
    """
    Standardized, causal routing record representation.
    Fields that are unavailable in the source data must remain None (missing), not fabricated.
    """

    def __init__(
        self,
        dataset_id: str,
        split: str,
        request_id: Any,
        token_id: int,
        source_stage_id: int,
        target_stage_id: int,
        source_experts: List[int],
        actual_target_experts: List[int],
        source_routing_weights: Optional[List[float]] = None,
        target_routing_weights: Optional[List[float]] = None,
        prediction_horizon: int = 1
    ):
        self.dataset_id = str(dataset_id)
        self.split = str(split)
        self.request_id = request_id
        self.token_id = int(token_id)
        self.source_stage_id = int(source_stage_id)
        self.target_stage_id = int(target_stage_id)
        self.source_experts = list(source_experts)
        self.actual_target_experts = list(actual_target_experts)

        if source_routing_weights is not None:
            self.source_routing_weights = [float(w) for w in source_routing_weights]
        else:
            self.source_routing_weights = None

        if target_routing_weights is not None:
            self.target_routing_weights = [float(w) for w in target_routing_weights]
        else:
            self.target_routing_weights = None

        self.prediction_horizon = int(prediction_horizon)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "split": self.split,
            "request_id": self.request_id,
            "token_id": self.token_id,
            "source_stage_id": self.source_stage_id,
            "target_stage_id": self.target_stage_id,
            "source_experts": self.source_experts,
            "actual_target_experts": self.actual_target_experts,
            "source_routing_weights": self.source_routing_weights,
            "target_routing_weights": self.target_routing_weights,
            "prediction_horizon": self.prediction_horizon
        }


class RoutingDatasetAdapter(ABC):
    """
    Abstract base class for all routing trace dataset adapters.
    """

    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.is_loaded = False
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def load(self) -> None:
        """
        Loads and parses the raw dataset from disk or memory into internal structures.
        """
        pass

    @abstractmethod
    def validate(self) -> Dict[str, Any]:
        """
        Runs comprehensive data quality and integrity checks on the loaded dataset.
        Returns a dictionary summarizing validation outcomes, record counts, and anomalies.
        """
        pass

    @abstractmethod
    def get_splits(self) -> Dict[str, List[Any]]:
        """
        Returns a dictionary mapping split names (e.g., 'train', 'validation', 'test')
        to lists of request_ids or prompt_ids.
        """
        pass

    @abstractmethod
    def iter_predictions(
        self,
        horizon: int = 1,
        split: Optional[str] = None
    ) -> Iterator[PredictionRecord]:
        """
        Yields standardized PredictionRecord instances in causal temporal sequence.
        """
        pass

    def profile_tensors(
        self,
        horizon: int = 1,
        split: str = "train",
        num_moe_stages: Optional[int] = None,
        num_routed_experts: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Profiles the specified split to compute:
        1. Marginal counts: shape (num_moe_stages, num_routed_experts)
        2. Spatial co-activation counts: shape (max(1, num_moe_stages - horizon), num_routed_experts, num_routed_experts)
        3. Temporal transition counts: shape (num_moe_stages, num_routed_experts, num_routed_experts)
        """
        if num_moe_stages is None:
            n_stages = self.metadata.get("num_moe_stages", 57)
        else:
            n_stages = num_moe_stages

        if num_routed_experts is None:
            n_experts = self.metadata.get("num_routed_experts", 256)
        else:
            n_experts = num_routed_experts

        n_transitions = max(1, n_stages - horizon)
        import numpy as np
        marginal_counts = np.zeros((n_stages, n_experts), dtype=np.float64)
        spatial_counts = np.zeros((n_transitions, n_experts, n_experts), dtype=np.float64)
        temporal_counts = np.zeros((n_stages, n_experts, n_experts), dtype=np.float64)

        prev_by_unit: Dict[Any, Dict[int, List[int]]] = {}

        for rec in self.iter_predictions(horizon=horizon, split=split):
            tgt_stage = rec.target_stage_id
            src_stage = rec.source_stage_id

            # Marginal counts
            for v in rec.actual_target_experts:
                if 0 <= v < n_experts:
                    marginal_counts[tgt_stage, v] += 1.0

            # Spatial co-activations
            if src_stage >= 0 and src_stage < n_transitions:
                for u in rec.source_experts:
                    for v in rec.actual_target_experts:
                        if 0 <= u < n_experts and 0 <= v < n_experts:
                            spatial_counts[src_stage, u, v] += 1.0

            # Temporal transitions
            u_id = rec.request_id
            if u_id not in prev_by_unit:
                prev_by_unit[u_id] = {}

            if tgt_stage in prev_by_unit[u_id]:
                prev_exps = prev_by_unit[u_id][tgt_stage]
                for w in prev_exps:
                    for v in rec.actual_target_experts:
                        if 0 <= w < n_experts and 0 <= v < n_experts:
                            temporal_counts[tgt_stage, w, v] += 1.0

            prev_by_unit[u_id][tgt_stage] = list(rec.actual_target_experts)

        return marginal_counts, spatial_counts, temporal_counts

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns summary metadata describing dataset dimensions, provenance, and configuration.
        """
        return self.metadata
