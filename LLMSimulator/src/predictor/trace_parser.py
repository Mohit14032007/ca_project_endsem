"""
Unified MoE Routing Trace Parser and Validator.

Supports normalized routing records with explicit distinction between:
- raw_transformer_layer_id (e.g. 3..59 for DeepSeek-V3)
- moe_stage_id (0..56 for DeepSeek-V3)

Performs schema verification, data integrity validation, anomaly detection,
and reconstructs execution stages from sequential traces where applicable.
"""

from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np


class TraceValidationError(Exception):
    """Raised when a trace violates fundamental integrity invariants."""
    pass


class ModelArchitectureConfig:
    """Configuration defining model layer layout and expert topology."""
    def __init__(
        self,
        num_layers: int = 60,
        first_k_dense: int = 3,
        num_routed_experts: int = 256,
        top_k: int = 8,
        name: str = "deepseekV3"
    ):
        self.num_layers = num_layers
        self.first_k_dense = first_k_dense
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.name = name
        self.num_moe_stages = num_layers - first_k_dense


DEEPSEEK_V3_CONFIG = ModelArchitectureConfig(
    num_layers=60,
    first_k_dense=3,
    num_routed_experts=256,
    top_k=8,
    name="deepseekV3"
)

MIXTRAL_8X7B_CONFIG = ModelArchitectureConfig(
    num_layers=32,
    first_k_dense=0,
    num_routed_experts=8,
    top_k=2,
    name="mixtral"
)


class NormalizedRoutingRecord:
    """Represents a single expert routing decision."""
    def __init__(
        self,
        request_id: int,
        token_id: int,
        moe_stage_id: int,
        raw_transformer_layer_id: int,
        topk_rank: int,
        expert_id: int
    ):
        self.request_id = request_id
        self.token_id = token_id
        self.moe_stage_id = moe_stage_id
        self.raw_transformer_layer_id = raw_transformer_layer_id
        self.topk_rank = topk_rank
        self.expert_id = expert_id

    def to_dict(self) -> Dict[str, int]:
        record_dict = {
            "request_id": self.request_id,
            "token_id": self.token_id,
            "moe_stage_id": self.moe_stage_id,
            "raw_transformer_layer_id": self.raw_transformer_layer_id,
            "topk_rank": self.topk_rank,
            "expert_id": self.expert_id
        }
        return record_dict


class RoutingTraceParser:
    """Common parser and validator for offline expert prediction traces."""

    REQUIRED_RAW_COLUMNS = [
        "request_id",
        "token_id",
        "layer_id",
        "topk_rank",
        "expert_id"
    ]

    def __init__(self, config: Optional[ModelArchitectureConfig] = None):
        if config is not None:
            self.config = config
        else:
            self.config = DEEPSEEK_V3_CONFIG
        self.validation_report = {}

    def parse_file(
        self,
        csv_path: str,
        reconstruct_stages_from_sequence: bool = False
    ) -> List[NormalizedRoutingRecord]:
        """
        Loads and validates a CSV routing trace file.
        Returns a list of NormalizedRoutingRecord instances.
        """
        raw_df = pd.read_csv(csv_path)
        return self.parse_dataframe(raw_df, reconstruct_stages_from_sequence)

    def parse_dataframe(
        self,
        df: pd.DataFrame,
        reconstruct_stages_from_sequence: bool = False
    ) -> List[NormalizedRoutingRecord]:
        """
        Validates and parses a pandas DataFrame into normalized routing records.
        """
        self.validation_report = {
            "total_rows": len(df),
            "missing_values": 0,
            "duplicate_records": 0,
            "invalid_expert_ids": 0,
            "invalid_ranks": 0,
            "requests_found": 0,
            "tokens_found": 0,
            "modulo_60_drift_detected": False,
            "is_single_request": False,
            "warnings": [],
            "reconstructed_stages": False
        }

        # 1. Validate required columns
        for col in self.REQUIRED_RAW_COLUMNS:
            if col not in df.columns:
                raise TraceValidationError(
                    f"Missing required trace column: '{col}'. Available: {list(df.columns)}"
                )

        # 2. Detect missing values
        null_count = int(df[self.REQUIRED_RAW_COLUMNS].isnull().sum().sum())
        self.validation_report["missing_values"] = null_count
        if null_count > 0:
            raise TraceValidationError(
                f"Trace contains {null_count} null or missing values."
            )

        # 3. Detect duplicate routing records
        duplicate_mask = df.duplicated(
            subset=["request_id", "token_id", "layer_id", "topk_rank"],
            keep=False
        )
        duplicate_count = int(duplicate_mask.sum())
        self.validation_report["duplicate_records"] = duplicate_count
        if duplicate_count > 0:
            raise TraceValidationError(
                f"Trace contains {duplicate_count} duplicate routing records."
            )

        # 4. Validate expert IDs
        invalid_experts = df[
            (df["expert_id"] < 0) | (df["expert_id"] >= self.config.num_routed_experts)
        ]
        invalid_exp_count = len(invalid_experts)
        self.validation_report["invalid_expert_ids"] = invalid_exp_count
        if invalid_exp_count > 0:
            raise TraceValidationError(
                f"Trace contains {invalid_exp_count} expert IDs outside valid range [0, {self.config.num_routed_experts - 1}]."
            )

        # 5. Validate Top-K ranks
        invalid_ranks = df[
            (df["topk_rank"] < 1) | (df["topk_rank"] > self.config.top_k)
        ]
        invalid_rank_count = len(invalid_ranks)
        self.validation_report["invalid_ranks"] = invalid_rank_count
        if invalid_rank_count > 0:
            self.validation_report["warnings"].append(
                f"Trace contains {invalid_rank_count} ranks outside expected [1, {self.config.top_k}]."
            )

        unique_requests = df["request_id"].unique()
        self.validation_report["requests_found"] = len(unique_requests)
        if len(unique_requests) == 1:
            self.validation_report["is_single_request"] = True
            self.validation_report["warnings"].append(
                "Trace contains only 1 request ID. Request-level train/test splitting is impossible on this trace."
            )

        unique_tokens = df["token_id"].unique()
        self.validation_report["tokens_found"] = len(unique_tokens)

        # 6. Check layer sequencing and detect modulo-60 bug
        normalized_records: List[NormalizedRoutingRecord] = []

        for req_id in unique_requests:
            req_df = df[df["request_id"] == req_id]
            req_tokens = sorted(req_df["token_id"].unique())

            for tok_id in req_tokens:
                tok_df = req_df[req_df["token_id"] == tok_id]
                unique_layers_in_tok = tok_df["layer_id"].unique()

                # Check if layer sequence shows drift across tokens
                first_layer_in_tok = unique_layers_in_tok[0]
                if self.config.num_layers == 60:
                    if first_layer_in_tok != 0:
                        self.validation_report["modulo_60_drift_detected"] = True

                if reconstruct_stages_from_sequence:
                    # In sequential simulation, rows for a token occur in order of MoE layer execution.
                    # Each stage has self.config.top_k consecutive entries.
                    self.validation_report["reconstructed_stages"] = True
                    num_tok_rows = len(tok_df)
                    expected_rows = self.config.num_moe_stages * self.config.top_k
                    if num_tok_rows != expected_rows:
                        self.validation_report["warnings"].append(
                            f"Token {tok_id} has {num_tok_rows} rows, expected {expected_rows}."
                        )

                    # Group rows in order of appearance
                    tok_df_sorted = tok_df.sort_index()
                    row_idx = 0
                    for stage_idx in range(self.config.num_moe_stages):
                        stage_slice = tok_df_sorted.iloc[row_idx : row_idx + self.config.top_k]
                        row_idx += len(stage_slice)
                        raw_layer_id = stage_idx + self.config.first_k_dense

                        for _, row in stage_slice.iterrows():
                            rec = NormalizedRoutingRecord(
                                request_id=int(row["request_id"]),
                                token_id=int(row["token_id"]),
                                moe_stage_id=int(stage_idx),
                                raw_transformer_layer_id=int(raw_layer_id),
                                topk_rank=int(row["topk_rank"]),
                                expert_id=int(row["expert_id"])
                            )
                            normalized_records.append(rec)
                else:
                    # Direct parsing without automatic reconstruction
                    for _, row in tok_df.iterrows():
                        raw_layer_val = int(row["layer_id"])
                        if self.config.first_k_dense == 0:
                            moe_stage_val = raw_layer_val
                            raw_layer_id = raw_layer_val
                        else:
                            # If layer_id was recorded as 0..56
                            if raw_layer_val < self.config.num_moe_stages:
                                moe_stage_val = raw_layer_val
                                raw_layer_id = raw_layer_val + self.config.first_k_dense
                            else:
                                # Suspicious layer value
                                self.validation_report["modulo_60_drift_detected"] = True
                                moe_stage_val = raw_layer_val % self.config.num_moe_stages
                                raw_layer_id = (raw_layer_val % self.config.num_layers)

                        rec = NormalizedRoutingRecord(
                            request_id=int(row["request_id"]),
                            token_id=int(row["token_id"]),
                            moe_stage_id=int(moe_stage_val),
                            raw_transformer_layer_id=int(raw_layer_id),
                            topk_rank=int(row["topk_rank"]),
                            expert_id=int(row["expert_id"])
                        )
                        normalized_records.append(rec)

        return normalized_records

    def get_validation_report(self) -> Dict:
        """Returns the dictionary containing validation statistics and warnings."""
        return self.validation_report

