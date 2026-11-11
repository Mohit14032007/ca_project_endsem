"""
Unit tests for STMoEPredictor.
Verifies 2-bit saturating counter mechanics, CCT updates, History Table temporal tracking,
candidate replacement, and cold-start handling.
"""

import unittest
from src.predictor.st_moe_predictor import STMoEPredictor


class TestSTMoEPredictor(unittest.TestCase):

    def setUp(self):
        # Small 4-stage model with 16 experts, top-2, CCT capacity 4, threshold 2
        self.predictor = STMoEPredictor(
            num_moe_stages=4,
            num_routed_experts=16,
            top_k=2,
            cct_capacity=4,
            confidence_threshold=2
        )

    def test_cct_counter_increment_and_saturation(self):
        # Update stage 0 -> 1: source expert 1, target expert 5
        self.predictor.update(request_id=1, target_stage_id=1, source_experts=[1], actual_target_experts=[5])
        # Initial counter should be 1
        c = self.predictor.cct[0][1][5]
        self.assertEqual(c, 1)

        # Hit 1 -> counter becomes 2
        self.predictor.update(request_id=1, target_stage_id=1, source_experts=[1], actual_target_experts=[5])
        self.assertEqual(self.predictor.cct[0][1][5], 2)

        # Hit 2 -> counter becomes 3
        self.predictor.update(request_id=1, target_stage_id=1, source_experts=[1], actual_target_experts=[5])
        self.assertEqual(self.predictor.cct[0][1][5], 3)

        # Hit 3 -> saturates at 3
        self.predictor.update(request_id=1, target_stage_id=1, source_experts=[1], actual_target_experts=[5])
        self.assertEqual(self.predictor.cct[0][1][5], 3)

    def test_cct_counter_decrement_and_saturation(self):
        # Initialize at 3
        for _ in range(3):
            self.predictor.update(request_id=1, target_stage_id=1, source_experts=[1], actual_target_experts=[5])
        self.assertEqual(self.predictor.cct[0][1][5], 3)

        # Miss 1 (actual target is 7, not 5)
        self.predictor.update(request_id=1, target_stage_id=1, source_experts=[1], actual_target_experts=[7])
        self.assertEqual(self.predictor.cct[0][1][5], 2)

        # Miss 2
        self.predictor.update(request_id=1, target_stage_id=1, source_experts=[1], actual_target_experts=[7])
        self.assertEqual(self.predictor.cct[0][1][5], 1)

        # Miss 3 -> reaches 0
        self.predictor.update(request_id=1, target_stage_id=1, source_experts=[1], actual_target_experts=[7])
        self.assertEqual(self.predictor.cct[0][1][5], 0)

        # Miss 4 -> saturates at 0
        self.predictor.update(request_id=1, target_stage_id=1, source_experts=[1], actual_target_experts=[7])
        self.assertEqual(self.predictor.cct[0][1][5], 0)

    def test_candidate_replacement_policy(self):
        pred = STMoEPredictor(num_moe_stages=3, num_routed_experts=10, cct_capacity=2, confidence_threshold=2)
        # Fill capacity with experts 1 and 2
        pred.update(request_id=1, target_stage_id=1, source_experts=[0], actual_target_experts=[1, 2])
        self.assertEqual(len(pred.cct[0][0]), 2)
        self.assertIn(1, pred.cct[0][0])
        self.assertIn(2, pred.cct[0][0])

        # Miss expert 1 until counter is 0
        pred.update(request_id=1, target_stage_id=1, source_experts=[0], actual_target_experts=[2])
        self.assertEqual(pred.cct[0][0][1], 0)

        # New expert 3 observed -> should replace expert 1 which has counter 0
        pred.update(request_id=1, target_stage_id=1, source_experts=[0], actual_target_experts=[2, 3])
        self.assertIn(3, pred.cct[0][0])
        self.assertNotIn(1, pred.cct[0][0])
        self.assertEqual(pred.cct[0][0][3], 1)

    def test_confidence_threshold_prediction(self):
        # Boost expert 5 to confidence 2
        self.predictor.update(request_id=1, target_stage_id=1, source_experts=[2], actual_target_experts=[5])
        self.predictor.update(request_id=1, target_stage_id=1, source_experts=[2], actual_target_experts=[5])
        # Expert 6 has confidence 1
        self.predictor.update(request_id=1, target_stage_id=1, source_experts=[2], actual_target_experts=[5, 6])

        # Predict for stage 1 with budget 1 -> should select expert 5 (confidence 3) over expert 6 (confidence 1)
        predictions = self.predictor.predict(request_id=1, target_stage_id=1, current_source_experts=[2], budget=1)
        self.assertEqual(predictions, [5])

    def test_history_table_temporal_prediction(self):
        # Token 0 routes experts [8, 9] at stage 2
        self.predictor.update(request_id=42, target_stage_id=2, source_experts=[0], actual_target_experts=[8, 9])

        # Token 1 at stage 1 predicts stage 2 with empty CCT -> should fallback to HT [8, 9]
        predictions = self.predictor.predict(request_id=42, target_stage_id=2, current_source_experts=[0], budget=2)
        self.assertEqual(set(predictions), {8, 9})

    def test_cold_start_and_budget_bound(self):
        # Empty CCT, empty HT
        predictions = self.predictor.predict(request_id=99, target_stage_id=1, current_source_experts=[0], budget=3)
        self.assertLessEqual(len(predictions), 3)


if __name__ == "__main__":
    unittest.main()

