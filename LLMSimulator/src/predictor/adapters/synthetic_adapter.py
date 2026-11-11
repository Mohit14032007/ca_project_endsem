"""
Synthetic Correlated Benchmark Dataset Adapter.
"""

from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
import os
import pandas as pd
import numpy as np

from src.predictor.adapters.base_adapter import (
    PredictionRecord,
    RoutingDatasetAdapter
)
from scripts.generate_correlated_benchmark_trace import generate_benchmark_trace


class SyntheticCorrelatedDatasetAdapter(RoutingDatasetAdapter):
    """
    Adapter for the correlated synthetic benchmark trace.
    Supports loading an existing trace or deterministic regeneration.
    """

    def __init__(
        self,
        trace_path: str,
        num_requests: int = 16,
        num_tokens: int = 5,
        num_stages: int = 57,
        num_experts: int = 256,
        top_k: int = 8,
        first_k_dense: int = 3,
        spatial_affinity: float = 0.6,
        temporal_affinity: float = 0.5,
        seed: int = 42,
        train_ratio: float = 0.75
    ):
        super().__init__(dataset_id="correlated_synthetic")
        self.trace_path = trace_path
        self.num_requests = num_requests
        self.num_tokens = num_tokens
        self.num_stages = num_stages
        self.num_experts = num_experts
        self.top_k = top_k
        self.first_k_dense = first_k_dense
        self.spatial_affinity = spatial_affinity
        self.temporal_affinity = temporal_affinity
        self.seed = seed
        self.train_ratio = train_ratio

        self.df: Optional[pd.DataFrame] = None
        self.splits: Dict[str, List[int]] = {"train": [], "test": []}
        self.grouped_routes: Dict[Tuple[int, int, int], List[int]] = {}

    def load(self) -> None:
        if not os.path.exists(self.trace_path):
            generate_benchmark_trace(
                output_path=self.trace_path,
                num_requests=self.num_requests,
                num_tokens=self.num_tokens,
                num_stages=self.num_stages,
                num_experts=self.num_experts,
                top_k=self.top_k,
                first_k_dense=self.first_k_dense,
                spatial_affinity=self.spatial_affinity,
                temporal_affinity=self.temporal_affinity,
                seed=self.seed
            )

        self.df = pd.read_csv(self.trace_path)

        self.grouped_routes = {}
        for row in self.df.itertuples(index=False):
            req_id = int(row.request_id)
            tok_id = int(row.token_id)
            stage_id = int(row.layer_id)
            exp_id = int(row.expert_id)
            key = (req_id, tok_id, stage_id)
            if key not in self.grouped_routes:
                self.grouped_routes[key] = []
            self.grouped_routes[key].append(exp_id)

        unique_requests = sorted(list(set(self.df["request_id"].unique())))
        num_req = len(unique_requests)
        num_train = max(1, int(round(num_req * self.train_ratio)))
        self.splits["train"] = unique_requests[:num_train]
        self.splits["test"] = unique_requests[num_train:]

        self.is_loaded = True
        self.metadata = {
            "dataset_id": self.dataset_id,
            "trace_path": self.trace_path,
            "num_requests": self.num_requests,
            "num_tokens": self.num_tokens,
            "num_stages": self.num_stages,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "spatial_affinity": self.spatial_affinity,
            "temporal_affinity": self.temporal_affinity,
            "seed": self.seed,
            "total_records": len(self.df),
            "train_requests": len(self.splits["train"]),
            "test_requests": len(self.splits["test"])
        }

    def validate(self) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load()

        assert self.df is not None
        missing_records = int(self.df.isnull().sum().sum())
        total_records = len(self.df)

        invalid_experts = int(((self.df["expert_id"] < 0) | (self.df["expert_id"] >= self.num_experts)).sum())

        duplicate_records = int(self.df.duplicated(subset=["request_id", "token_id", "layer_id", "expert_id"]).sum())

        inconsistent_topk = 0
        for key, exp_list in self.grouped_routes.items():
            if len(exp_list) != self.top_k:
                inconsistent_topk += 1

        unique_requests = sorted(list(set(self.df["request_id"].unique())))
        unique_tokens = sorted(list(set(self.df["token_id"].unique())))

        report = {
            "dataset_id": self.dataset_id,
            "total_records": total_records,
            "num_requests": len(unique_requests),
            "num_tokens_total": len(unique_tokens),
            "num_moe_stages": self.num_stages,
            "num_routed_experts": self.num_experts,
            "top_k": self.top_k,
            "missing_records": missing_records,
            "duplicate_records": duplicate_records,
            "invalid_expert_ids": invalid_experts,
            "inconsistent_topk_count": inconsistent_topk,
            "train_requests_count": len(self.splits["train"]),
            "test_requests_count": len(self.splits["test"]),
            "zipf_exponent": 0.8,
            "cluster_count": 16,
            "cluster_size": 16,
            "spatial_affinity": self.spatial_affinity,
            "temporal_affinity": self.temporal_affinity,
            "seed": self.seed
        }
        return report

    def get_splits(self) -> Dict[str, List[Any]]:
        if not self.is_loaded:
            self.load()
        return self.splits

    def iter_predictions(
        self,
        horizon: int = 1,
        split: Optional[str] = None
    ) -> Iterator[PredictionRecord]:
        if not self.is_loaded:
            self.load()

        if split is not None:
            target_requests = set(self.splits.get(split, []))
        else:
            target_requests = set(self.splits.get("train", []) + self.splits.get("test", []))

        unique_requests = sorted(list(target_requests))
        for req in unique_requests:
            req_keys = [k for k in self.grouped_routes.keys() if k[0] == req]
            req_tokens = sorted(list(set(k[1] for k in req_keys)))

            if req in self.splits.get("test", []):
                req_split = "test"
            else:
                req_split = "train"

            for tok in req_tokens:
                for target_stage in range(self.num_stages):
                    source_stage = target_stage - horizon
                    if source_stage < 0:
                        source_experts = []
                    else:
                        source_key = (req, tok, source_stage)
                        source_experts = self.grouped_routes.get(source_key, [])

                    target_key = (req, tok, target_stage)
                    actual_target = self.grouped_routes.get(target_key, [])
                    if len(actual_target) == 0:
                        continue

                    rec = PredictionRecord(
                        dataset_id=self.dataset_id,
                        split=req_split,
                        request_id=req,
                        token_id=tok,
                        source_stage_id=source_stage,
                        target_stage_id=target_stage,
                        source_experts=source_experts,
                        actual_target_experts=actual_target,
                        source_routing_weights=None,
                        target_routing_weights=None,
                        prediction_horizon=horizon
                    )
                    yield rec

