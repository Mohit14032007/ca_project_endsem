"""
Unit tests for EvaluationHarness and Prediction Metrics.
Verifies Precision@B, Recall@B, F1@B, Hit Rate, prefetch metrics,
macro/micro aggregations, and request-level train/test isolation.
"""

import unittest
from src.predictor.evaluation import (
    EvaluationHarness,
    EvaluationSplitError,
    PredictionOpportunity,
    PredictionResultRecord
)
from src.predictor.trace_parser import NormalizedRoutingRecord


class TestEvaluationFramework(unittest.TestCase):

    def setUp(self):
        self.harness = EvaluationHarness(num_moe_stages=3)

    def test_single_prediction_metrics(self):
        rec = PredictionResultRecord(
            request_id=1,
            token_id=10,
            token_position=0,
            moe_stage_id=1,
            budget=2,
            predictor_name="MockPredictor",
            predicted_experts=[2, 5],
            actual_experts=[5, 7]
        )
        self.assertEqual(rec.hits, 1)
        self.assertEqual(rec.precision, 0.5)
        self.assertEqual(rec.recall, 0.5)
        self.assertEqual(rec.f1, 0.5)
        self.assertEqual(rec.hit_rate, 1.0)
        self.assertEqual(rec.missed_count, 1) # Expert 7 was missed
        self.assertEqual(rec.unused_count, 1) # Expert 2 was unused

    def test_zero_predicted_metrics(self):
        rec = PredictionResultRecord(
            request_id=1,
            token_id=10,
            token_position=0,
            moe_stage_id=1,
            budget=2,
            predictor_name="NoPrediction",
            predicted_experts=[],
            actual_experts=[5, 7]
        )
        self.assertEqual(rec.hits, 0)
        self.assertEqual(rec.precision, 0.0)
        self.assertEqual(rec.recall, 0.0)
        self.assertEqual(rec.f1, 0.0)
        self.assertEqual(rec.hit_rate, 0.0)
        self.assertEqual(rec.missed_count, 2)
        self.assertEqual(rec.unused_count, 0)

    def test_summary_aggregations(self):
        rec1 = PredictionResultRecord(
            request_id=1, token_id=10, token_position=0, moe_stage_id=1, budget=2,
            predictor_name="Test", predicted_experts=[1, 2], actual_experts=[1, 2] # 2/2 hits: P=1.0, R=1.0
        )
        rec2 = PredictionResultRecord(
            request_id=1, token_id=11, token_position=1, moe_stage_id=1, budget=2,
            predictor_name="Test", predicted_experts=[3, 4], actual_experts=[5, 6] # 0/2 hits: P=0.0, R=0.0
        )

        summary = self.harness.compute_summary_metrics([rec1, rec2])
        self.assertEqual(summary["total_instances"], 2)
        self.assertAlmostEqual(summary["macro_precision"], 0.5)
        self.assertAlmostEqual(summary["macro_recall"], 0.5)
        self.assertAlmostEqual(summary["hit_rate"], 0.5)

        # Micro precision = 2 hits / 4 predicted = 0.5
        self.assertAlmostEqual(summary["micro_precision"], 0.5)
        self.assertAlmostEqual(summary["micro_recall"], 0.5)
        self.assertAlmostEqual(summary["coverage"], 0.5)

    def test_request_level_split_isolation(self):
        records = [
            NormalizedRoutingRecord(request_id=1, token_id=10, moe_stage_id=0, raw_transformer_layer_id=3, topk_rank=1, expert_id=1),
            NormalizedRoutingRecord(request_id=2, token_id=10, moe_stage_id=0, raw_transformer_layer_id=3, topk_rank=1, expert_id=2),
            NormalizedRoutingRecord(request_id=3, token_id=10, moe_stage_id=0, raw_transformer_layer_id=3, topk_rank=1, expert_id=3),
            NormalizedRoutingRecord(request_id=4, token_id=10, moe_stage_id=0, raw_transformer_layer_id=3, topk_rank=1, expert_id=4)
        ]
        train, test, meta = self.harness.split_by_request(records, train_ratio=0.5)

        train_reqs = set(meta["train_requests"])
        test_reqs = set(meta["test_requests"])

        # Invariant: Train and Test request sets must be completely disjoint
        self.assertEqual(len(train_reqs.intersection(test_reqs)), 0)
        self.assertEqual(len(train_reqs) + len(test_reqs), 4)

    def test_single_request_limitation_reported(self):
        records = [
            NormalizedRoutingRecord(request_id=100, token_id=1, moe_stage_id=0, raw_transformer_layer_id=3, topk_rank=1, expert_id=5)
        ]
        # A single request with only 1 token must raise EvaluationSplitError to eliminate leakage
        with self.assertRaises(EvaluationSplitError):
            self.harness.split_by_request(records, train_ratio=0.75)


if __name__ == "__main__":
    unittest.main()

