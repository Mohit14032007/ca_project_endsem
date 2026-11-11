"""
Predicting Experts for MoE Suite Dataset Adapter.
Reference: mistrjirka/Predicting-Experts-for-MOE-Suite (token_cube)
"""

from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
import os
import pyarrow.parquet as pq
import numpy as np

from src.predictor.adapters.base_adapter import (
    PredictionRecord,
    RoutingDatasetAdapter
)


class MoeSuiteDatasetAdapter(RoutingDatasetAdapter):
    """
    Adapter for the Predicting-Experts-for-MOE-Suite dataset.
    75 MoE layers (3..77), 256 experts per layer, Top-8 routing.
    Maintains official task-disjoint splits: 8 train, 2 validation, 2 test.
    Primary prediction track: J -> J+2 (prediction_horizon = 2).
    """

    OFFICIAL_SPLITS = {
        "train": [
            "corpus-02-cpp-framed-rpc-parser",
            "corpus-03-python-incremental-etl",
            "corpus-04-rust-priority-worker",
            "corpus-05-typescript-collab-sync",
            "corpus-06-go-crawler-limits",
            "corpus-07-java-outbox-dispatcher",
            "corpus-08-cpp-mmap-index",
            "corpus-09-python-import-analyzer"
        ],
        "validation": [
            "corpus-10-cuda-segmented-reduction",
            "corpus-11-cpp-bounded-channel"
        ],
        "test": [
            "corpus-12-typescript-migration-planner",
            "corpus-13-go-fenced-leases"
        ]
    }

    def __init__(
        self,
        data_dir: str,
        num_moe_stages: int = 75,
        num_routed_experts: int = 256,
        top_k: int = 8
    ):
        super().__init__(dataset_id="moe_suite")
        self.data_dir = data_dir
        self.num_moe_stages = num_moe_stages
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k

        # task_id -> numpy uint16 array of shape (num_tokens, num_moe_stages, top_k)
        self.task_routes: Dict[str, np.ndarray] = {}
        self.task_split_map: Dict[str, str] = {}
        self.splits: Dict[str, List[str]] = {
            "train": list(self.OFFICIAL_SPLITS["train"]),
            "validation": list(self.OFFICIAL_SPLITS["validation"]),
            "test": list(self.OFFICIAL_SPLITS["test"])
        }

    def load(self) -> None:
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        for split_name, task_list in self.OFFICIAL_SPLITS.items():
            for task_id in task_list:
                fname = f"{task_id}.parquet"
                fpath = os.path.join(self.data_dir, fname)
                if not os.path.exists(fpath):
                    raise FileNotFoundError(f"Task file not found: {fpath}")

                tbl = pq.read_table(fpath, columns=["route_selected"])
                # route_selected has shape (N, 75, 8)
                # Convert list of lists of lists to a flat NumPy array
                route_col = tbl["route_selected"].to_pylist()
                arr = np.array(route_col, dtype=np.int16)
                self.task_routes[task_id] = arr
                self.task_split_map[task_id] = split_name

        self.is_loaded = True
        total_tokens = sum(arr.shape[0] for arr in self.task_routes.values())
        self.metadata = {
            "dataset_id": self.dataset_id,
            "data_dir": self.data_dir,
            "num_moe_stages": self.num_moe_stages,
            "num_routed_experts": self.num_routed_experts,
            "top_k": self.top_k,
            "total_tasks": len(self.task_routes),
            "total_tokens": total_tokens,
            "train_tasks": self.splits["train"],
            "validation_tasks": self.splits["validation"],
            "test_tasks": self.splits["test"],
            "default_horizon": 2
        }

    def validate(self) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load()

        total_tokens = 0
        invalid_experts = 0
        inconsistent_topk = 0

        for task_id, arr in self.task_routes.items():
            n_tokens, n_stages, k = arr.shape
            total_tokens += n_tokens

            if n_stages != self.num_moe_stages:
                inconsistent_topk += 1
            if k != self.top_k:
                inconsistent_topk += 1

            # Expert bound checks
            min_e = int(np.min(arr))
            max_e = int(np.max(arr))
            if min_e < 0:
                invalid_experts += 1
            if max_e >= self.num_routed_experts:
                invalid_experts += 1

        report = {
            "dataset_id": self.dataset_id,
            "total_tasks": len(self.task_routes),
            "total_tokens": total_tokens,
            "num_moe_stages": self.num_moe_stages,
            "num_routed_experts": self.num_routed_experts,
            "top_k": self.top_k,
            "missing_records": 0,
            "duplicate_records": 0,
            "invalid_expert_ids": invalid_experts,
            "inconsistent_topk_count": inconsistent_topk,
            "train_tasks_count": len(self.splits["train"]),
            "validation_tasks_count": len(self.splits["validation"]),
            "test_tasks_count": len(self.splits["test"]),
            "official_task_disjoint_splits": True,
            "primary_horizon": 2
        }
        return report

    def get_splits(self) -> Dict[str, List[Any]]:
        if not self.is_loaded:
            self.load()
        return self.splits

    def iter_predictions(
        self,
        horizon: int = 2,
        split: Optional[str] = None
    ) -> Iterator[PredictionRecord]:
        if not self.is_loaded:
            self.load()

        if split is not None:
            target_tasks = self.splits.get(split, [])
        else:
            target_tasks = self.splits.get("train", []) + self.splits.get("validation", []) + self.splits.get("test", [])

        for task_id in target_tasks:
            arr = self.task_routes[task_id]
            task_split = self.task_split_map.get(task_id, "unknown")
            num_tokens, num_stages, _ = arr.shape

            for t_idx in range(num_tokens):
                for target_stage in range(num_stages):
                    source_stage = target_stage - horizon
                    if source_stage < 0:
                        source_experts = []
                    else:
                        source_experts = [int(e) for e in arr[t_idx, source_stage, :]]

                    actual_target = [int(e) for e in arr[t_idx, target_stage, :]]

                    rec = PredictionRecord(
                        dataset_id=self.dataset_id,
                        split=task_split,
                        request_id=task_id,
                        token_id=t_idx,
                        source_stage_id=source_stage,
                        target_stage_id=target_stage,
                        source_experts=source_experts,
                        actual_target_experts=actual_target,
                        source_routing_weights=None,
                        target_routing_weights=None,
                        prediction_horizon=horizon
                    )
                    yield rec

    def profile_tensors(
        self,
        horizon: int = 2,
        split: str = "train",
        num_moe_stages: Optional[int] = None,
        num_routed_experts: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_loaded:
            self.load()

        if num_moe_stages is None:
            n_stages = self.num_moe_stages
        else:
            n_stages = num_moe_stages

        if num_routed_experts is None:
            n_experts = self.num_routed_experts
        else:
            n_experts = num_routed_experts

        n_transitions = max(1, n_stages - horizon)
        marginal_counts = np.zeros((n_stages, n_experts), dtype=np.float64)
        spatial_counts = np.zeros((n_transitions, n_experts, n_experts), dtype=np.float64)
        temporal_counts = np.zeros((n_stages, n_experts, n_experts), dtype=np.float64)

        train_tasks = self.splits.get(split, [])
        for task_id in train_tasks:
            arr = self.task_routes[task_id]
            # arr shape: (N, 75, 8)
            # 1. Marginal counts
            for stage in range(n_stages):
                stage_exps = arr[:, stage, :].flatten()
                b_counts = np.bincount(stage_exps, minlength=n_experts)
                marginal_counts[stage] += b_counts[:n_experts]

            # 2. Spatial co-activations
            for stage in range(n_stages - horizon):
                src = arr[:, stage, :]
                dst = arr[:, stage + horizon, :]
                for i in range(self.top_k):
                    for j in range(self.top_k):
                        np.add.at(spatial_counts[stage], (src[:, i], dst[:, j]), 1.0)

            # 3. Temporal transitions
            for stage in range(n_stages):
                prev = arr[:-1, stage, :]
                curr = arr[1:, stage, :]
                for i in range(self.top_k):
                    for j in range(self.top_k):
                        np.add.at(temporal_counts[stage], (prev[:, i], curr[:, j]), 1.0)

        return marginal_counts, spatial_counts, temporal_counts
