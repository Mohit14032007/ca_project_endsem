"""
Unit tests validating the Common Predictor Interface (BasePredictor).
Ensures polymorphic substitution across all model adapters and baseline predictors.
"""

import unittest
from typing import List

from src.predictor.base_predictor import (
    BasePredictor,
    STMoELegacyWrapper,
    CSTPWrapper,
    RandomPredictor,
    FrequencyPredictor
)
from src.predictor.st_moe_paper import STMoEPaperPredictor
from src.predictor.st_moe_predictor import STMoEPredictor
from src.predictor.cstp_predictor import CSTPPredictor
from src.predictor.trace_parser import NormalizedRoutingRecord


class TestPredictorInterface(unittest.TestCase):
    """
    Validates adherence to BasePredictor abstract interface.
    """

    def setUp(self):
        self.paper_pred = STMoEPaperPredictor(num_moe_stages=4, num_routed_experts=16, top_k=2)
        self.legacy_inner = STMoEPredictor(num_moe_stages=4, num_routed_experts=16, top_k=2)
        self.legacy_pred = STMoELegacyWrapper(self.legacy_inner)
        self.cstp_inner = CSTPPredictor(num_moe_stages=4, num_routed_experts=16, top_k=2)
        self.cstp_pred = CSTPWrapper(self.cstp_inner)
        self.random_pred = RandomPredictor(num_routed_experts=16, seed=123)
        self.freq_pred = FrequencyPredictor(num_moe_stages=4, num_routed_experts=16)

        self.all_predictors: List[BasePredictor] = [
            self.paper_pred,
            self.legacy_pred,
            self.cstp_pred,
            self.random_pred,
            self.freq_pred
        ]

    def test_all_classes_inherit_from_base_predictor(self):
        for pred in self.all_predictors:
            self.assertIsInstance(pred, BasePredictor)

    def test_predict_returns_valid_expert_ids_and_respects_budget(self):
        budget = 3
        for pred in self.all_predictors:
            preds = pred.predict(
                request_id=1,
                target_stage_id=1,
                current_source_experts=[0, 1],
                budget=budget
            )
            self.assertIsInstance(preds, list)
            self.assertLessEqual(len(preds), budget)
            for exp_id in preds:
                self.assertIsInstance(exp_id, int)
                self.assertGreaterEqual(exp_id, 0)
                self.assertLess(exp_id, 16)
            # Distinct candidate verification
            self.assertEqual(len(preds), len(set(preds)))

    def test_zero_and_negative_budgets(self):
        for pred in self.all_predictors:
            preds_zero = pred.predict(
                request_id=1,
                target_stage_id=1,
                current_source_experts=[0, 1],
                budget=0
            )
            self.assertEqual(preds_zero, [])

            preds_neg = pred.predict(
                request_id=1,
                target_stage_id=1,
                current_source_experts=[0, 1],
                budget=-5
            )
            self.assertEqual(preds_neg, [])

    def test_cstp_mathematical_preservation_in_wrapper(self):
        """Verifies that CSTPWrapper returns exact same output as inner CSTPPredictor."""
        direct_output = self.cstp_inner.predict(
            request_id=10,
            target_stage_id=2,
            current_source_experts=[3, 4],
            budget=4
        )
        wrapper_output = self.cstp_pred.predict(
            request_id=10,
            target_stage_id=2,
            current_source_experts=[3, 4],
            budget=4
        )
        self.assertEqual(direct_output, wrapper_output)

    def test_frequency_predictor_profiling_and_ranking(self):
        records = [
            NormalizedRoutingRecord(0, 0, 1, 0, 0, 7),
            NormalizedRoutingRecord(0, 0, 1, 1, 0, 7),
            NormalizedRoutingRecord(0, 1, 1, 0, 0, 7),
            NormalizedRoutingRecord(0, 1, 1, 1, 0, 3)
        ]
        self.freq_pred.fit_profile(records)
        # Stage 1 has expert 7 with count 3, expert 3 with count 1
        preds = self.freq_pred.predict(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[0],
            budget=2
        )
        self.assertEqual(preds[0], 7)
        self.assertEqual(preds[1], 3)


if __name__ == "__main__":
    unittest.main()

