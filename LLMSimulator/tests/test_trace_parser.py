"""
Unit tests for RoutingTraceParser and Normalization.
Verifies schema validation, expert bounds, stage reconstruction, and error handling.
"""

import unittest
import pandas as pd
from src.predictor.trace_parser import (
    RoutingTraceParser,
    TraceValidationError,
    ModelArchitectureConfig,
    DEEPSEEK_V3_CONFIG
)


class TestTraceParser(unittest.TestCase):

    def setUp(self):
        self.parser = RoutingTraceParser(DEEPSEEK_V3_CONFIG)

    def test_valid_trace_parsing(self):
        # 1 request, 1 token, stage 0 (raw layer 3), top-2
        data = {
            "request_id": [1, 1],
            "token_id": [100, 100],
            "layer_id": [0, 0],
            "topk_rank": [1, 2],
            "expert_id": [42, 88]
        }
        df = pd.DataFrame(data)
        records = self.parser.parse_dataframe(df)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].request_id, 1)
        self.assertEqual(records[0].token_id, 100)
        self.assertEqual(records[0].moe_stage_id, 0)
        self.assertEqual(records[0].raw_transformer_layer_id, 3)
        self.assertEqual(records[0].expert_id, 42)
        self.assertEqual(records[1].expert_id, 88)

    def test_missing_column_raises_error(self):
        data = {
            "request_id": [1],
            "token_id": [100],
            "layer_id": [0],
            "topk_rank": [1]
            # missing expert_id
        }
        df = pd.DataFrame(data)
        with self.assertRaises(TraceValidationError):
            self.parser.parse_dataframe(df)

    def test_null_value_raises_error(self):
        data = {
            "request_id": [1],
            "token_id": [100],
            "layer_id": [None],
            "topk_rank": [1],
            "expert_id": [5]
        }
        df = pd.DataFrame(data)
        with self.assertRaises(TraceValidationError):
            self.parser.parse_dataframe(df)

    def test_invalid_expert_id_raises_error(self):
        data = {
            "request_id": [1],
            "token_id": [100],
            "layer_id": [0],
            "topk_rank": [1],
            "expert_id": [256] # DeepSeek-V3 has 256 experts (0..255)
        }
        df = pd.DataFrame(data)
        with self.assertRaises(TraceValidationError):
            self.parser.parse_dataframe(df)

    def test_duplicate_record_raises_error(self):
        data = {
            "request_id": [1, 1],
            "token_id": [100, 100],
            "layer_id": [0, 0],
            "topk_rank": [1, 1], # duplicate rank
            "expert_id": [10, 20]
        }
        df = pd.DataFrame(data)
        with self.assertRaises(TraceValidationError):
            self.parser.parse_dataframe(df)

    def test_modulo_60_drift_detection(self):
        # Starts with layer 57
        data = {
            "request_id": [1],
            "token_id": [101],
            "layer_id": [57],
            "topk_rank": [1],
            "expert_id": [15]
        }
        df = pd.DataFrame(data)
        self.parser.parse_dataframe(df)
        report = self.parser.get_validation_report()
        self.assertTrue(report["modulo_60_drift_detected"])

    def test_sequential_stage_reconstruction(self):
        # 2 stages with top-2
        data = {
            "request_id": [1, 1, 1, 1],
            "token_id": [200, 200, 200, 200],
            "layer_id": [57, 57, 58, 58], # Drifted recorded layers
            "topk_rank": [1, 2, 1, 2],
            "expert_id": [10, 20, 30, 40]
        }
        df = pd.DataFrame(data)
        cfg = ModelArchitectureConfig(num_layers=5, first_k_dense=3, num_routed_experts=256, top_k=2)
        parser = RoutingTraceParser(cfg)
        records = parser.parse_dataframe(df, reconstruct_stages_from_sequence=True)

        self.assertEqual(len(records), 4)
        # First 2 rows reconstructed as stage 0
        self.assertEqual(records[0].moe_stage_id, 0)
        self.assertEqual(records[0].raw_transformer_layer_id, 3)
        self.assertEqual(records[1].moe_stage_id, 0)
        # Next 2 rows reconstructed as stage 1
        self.assertEqual(records[2].moe_stage_id, 1)
        self.assertEqual(records[2].raw_transformer_layer_id, 4)
        self.assertEqual(records[3].moe_stage_id, 1)


if __name__ == "__main__":
    unittest.main()

