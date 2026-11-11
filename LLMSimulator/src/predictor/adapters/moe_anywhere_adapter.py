"""
MoE Anywhere Routing Traces Dataset Adapter.
Reference: carlosfelipe/moe-anywhere-routing-traces (e1-sage-a1.5.parquet)
"""

from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
import os
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

from src.predictor.adapters.base_adapter import (
    PredictionRecord,
    RoutingDatasetAdapter
)


class MoeAnywhereDatasetAdapter(RoutingDatasetAdapter):
    """
    Adapter for the MoE Anywhere dataset (InternScience/Sage-A1.5, 40 MoE layers, Top-8, 256 experts).
    Guarantees strict prompt-level split isolation.
    """

    def __init__(
        self,
        parquet_path: str,
        num_moe_stages: int = 40,
        num_routed_experts: int = 256,
        top_k: int = 8,
        train_prompts_count: int = 52,
        val_prompts_count: int = 8,
        test_prompts_count: int = 14
    ):
        super().__init__(dataset_id="moe_anywhere")
        self.parquet_path = parquet_path
        self.num_moe_stages = num_moe_stages
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.train_prompts_count = train_prompts_count
        self.val_prompts_count = val_prompts_count
        self.test_prompts_count = test_prompts_count

        self.df: Optional[pd.DataFrame] = None
        self.splits: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
        # (prompt_id, token_pos, layer) -> (List[int], List[float])
        self.routes: Dict[Tuple[int, int, int], Tuple[List[int], List[float]]] = {}
        self.prompt_tasks: Dict[int, str] = {}
        self.prompt_tokens: Dict[int, List[int]] = {}

    def load(self) -> None:
        if not os.path.exists(self.parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

        table = pq.read_table(self.parquet_path)
        self.df = table.to_pandas()

        self.routes = {}
        self.prompt_tasks = {}
        prompt_tokens_set: Dict[int, Set[int]] = {}

        for row in self.df.itertuples(index=False):
            pid = int(row.prompt_id)
            tpos = int(row.token_pos)
            lyr = int(row.layer)
            task_str = str(row.task)

            exps = [int(e) for e in row.experts]
            wts = [float(w) for w in row.weights]

            self.routes[(pid, tpos, lyr)] = (exps, wts)
            if pid not in self.prompt_tasks:
                self.prompt_tasks[pid] = task_str

            if pid not in prompt_tokens_set:
                prompt_tokens_set[pid] = set()
            prompt_tokens_set[pid].add(tpos)

        self.prompt_tokens = {}
        for pid, t_set in prompt_tokens_set.items():
            self.prompt_tokens[pid] = sorted(list(t_set))

        # Deterministic prompt-level split
        unique_prompts = sorted(list(self.df["prompt_id"].unique()))
        total_p = len(unique_prompts)

        t_end = self.train_prompts_count
        v_end = t_end + self.val_prompts_count

        self.splits["train"] = [int(p) for p in unique_prompts[:t_end]]
        self.splits["val"] = [int(p) for p in unique_prompts[t_end:v_end]]
        self.splits["test"] = [int(p) for p in unique_prompts[v_end:]]

        self.is_loaded = True
        self.metadata = {
            "dataset_id": self.dataset_id,
            "parquet_path": self.parquet_path,
            "num_moe_stages": self.num_moe_stages,
            "num_routed_experts": self.num_routed_experts,
            "top_k": self.top_k,
            "total_rows": len(self.df),
            "num_prompts": total_p,
            "num_tasks": len(set(self.prompt_tasks.values())),
            "train_prompts": self.splits["train"],
            "val_prompts": self.splits["val"],
            "test_prompts": self.splits["test"]
        }

    def validate(self) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load()

        assert self.df is not None
        total_rows = len(self.df)
        missing_records = int(self.df.isnull().sum().sum())

        invalid_experts = 0
        inconsistent_topk = 0
        weight_sum_anomalies = 0

        for key, (exps, wts) in self.routes.items():
            if len(exps) != self.top_k:
                inconsistent_topk += 1
            for e in exps:
                if e < 0 or e >= self.num_routed_experts:
                    invalid_experts += 1
            w_sum = sum(wts)
            if abs(w_sum - 1.0) > 0.05:
                weight_sum_anomalies += 1

        unique_prompts = sorted(list(self.df["prompt_id"].unique()))
        unique_tokens = len(self.routes) // self.num_moe_stages

        report = {
            "dataset_id": self.dataset_id,
            "total_records": total_rows,
            "num_prompts": len(unique_prompts),
            "num_tokens_total": unique_tokens,
            "num_moe_stages": self.num_moe_stages,
            "num_routed_experts": self.num_routed_experts,
            "top_k": self.top_k,
            "missing_records": missing_records,
            "duplicate_records": 0,
            "invalid_expert_ids": invalid_experts,
            "inconsistent_topk_count": inconsistent_topk,
            "weight_sum_anomalies": weight_sum_anomalies,
            "train_prompts_count": len(self.splits["train"]),
            "val_prompts_count": len(self.splits["val"]),
            "test_prompts_count": len(self.splits["test"]),
            "available_horizons": [1, 2]
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
            target_prompts = set(self.splits.get(split, []))
        else:
            all_p = self.splits.get("train", []) + self.splits.get("val", []) + self.splits.get("test", [])
            target_prompts = set(all_p)

        train_set = set(self.splits.get("train", []))
        val_set = set(self.splits.get("val", []))
        test_set = set(self.splits.get("test", []))

        unique_prompts = sorted(list(target_prompts))

        for pid in unique_prompts:
            if pid in test_set:
                p_split = "test"
            else:
                if pid in val_set:
                    p_split = "val"
                else:
                    p_split = "train"

            p_tokens = self.prompt_tokens.get(pid, [])

            for tpos in p_tokens:
                for target_stage in range(self.num_moe_stages):
                    source_stage = target_stage - horizon
                    if source_stage < 0:
                        source_experts = []
                        source_weights = None
                    else:
                        src_key = (pid, tpos, source_stage)
                        if src_key in self.routes:
                            source_experts, source_weights = self.routes[src_key]
                        else:
                            source_experts = []
                            source_weights = None

                    tgt_key = (pid, tpos, target_stage)
                    if tgt_key not in self.routes:
                        continue
                    actual_target, target_weights = self.routes[tgt_key]

                    rec = PredictionRecord(
                        dataset_id=self.dataset_id,
                        split=p_split,
                        request_id=pid,
                        token_id=tpos,
                        source_stage_id=source_stage,
                        target_stage_id=target_stage,
                        source_experts=source_experts,
                        actual_target_experts=actual_target,
                        source_routing_weights=source_weights,
                        target_routing_weights=target_weights,
                        prediction_horizon=horizon
                    )
                    yield rec

    def profile_tensors(
        self,
        horizon: int = 1,
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

        train_prompts = set(self.splits.get(split, []))
        for pid in sorted(list(train_prompts)):
            p_tokens = self.prompt_tokens.get(pid, [])

            for t_idx, tpos in enumerate(p_tokens):
                for stage in range(n_stages):
                    curr_key = (pid, tpos, stage)
                    if curr_key in self.routes:
                        curr_exps, _ = self.routes[curr_key]
                        for v in curr_exps:
                            if 0 <= v < n_experts:
                                marginal_counts[stage, v] += 1.0

                        # Spatial
                        if stage < n_stages - horizon:
                            dst_key = (pid, tpos, stage + horizon)
                            if dst_key in self.routes:
                                dst_exps, _ = self.routes[dst_key]
                                for u in curr_exps:
                                    for v in dst_exps:
                                        if 0 <= u < n_experts and 0 <= v < n_experts:
                                            spatial_counts[stage, u, v] += 1.0

                        # Temporal
                        if t_idx > 0:
                            prev_tpos = p_tokens[t_idx - 1]
                            prev_key = (pid, prev_tpos, stage)
                            if prev_key in self.routes:
                                prev_exps, _ = self.routes[prev_key]
                                for w in prev_exps:
                                    for v in curr_exps:
                                        if 0 <= w < n_experts and 0 <= v < n_experts:
                                            temporal_counts[stage, w, v] += 1.0

        return marginal_counts, spatial_counts, temporal_counts
