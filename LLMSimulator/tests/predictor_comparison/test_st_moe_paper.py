"""
Unit tests for Paper-Faithful ST-MoE Expert Predictor.
Reference: Zhao et al., arXiv:2606.15453v1.
"""

import unittest
from typing import List

from src.predictor.st_moe_paper import STMoEPaperPredictor
from src.predictor.trace_parser import NormalizedRoutingRecord


class TestSTMoEPaperPredictor(unittest.TestCase):
    """
    Validation suite for CCT, HT, Algorithm 2 prediction, and Algorithm 3 updates.
    """

    def setUp(self):
        self.predictor = STMoEPaperPredictor(
            num_moe_stages=4,
            num_routed_experts=16,
            top_k=2,
            cct_capacity=2,
            confidence_threshold=2
        )

    def test_cct_initialization_to_decimal_two(self):
        """Paper requirement: CCT entries initialize to binary 10 (decimal 2)."""
        # Create small profiling records
        records = [
            NormalizedRoutingRecord(0, 0, 0, 0, 0, 1),
            NormalizedRoutingRecord(0, 0, 0, 1, 0, 2),
            NormalizedRoutingRecord(0, 0, 1, 0, 0, 5),
            NormalizedRoutingRecord(0, 0, 1, 1, 0, 6)
        ]
        self.predictor.build_cct_from_profile(records)

        # Expert 1 at stage 0 co-activated with 5 and 6 at stage 1
        entry = self.predictor.cct[0][1]
        self.assertIn(5, entry)
        self.assertIn(6, entry)
        self.assertEqual(entry[5], 2)
        self.assertEqual(entry[6], 2)

    def test_cct_saturating_counter_increment_and_ceiling(self):
        """Test increment transitions: 0->1, 1->2, 2->3, 3->3."""
        self.predictor.set_cct_entry(0, 1, 5, 2)
        
        # Match with actual target expert 5
        self.predictor.update(
            request_id=1,
            target_stage_id=1,
            source_experts=[1],
            actual_target_experts=[5]
        )
        self.assertEqual(self.predictor.cct[0][1][5], 3)

        # Match again: must saturate at 3
        self.predictor.update(
            request_id=1,
            target_stage_id=1,
            source_experts=[1],
            actual_target_experts=[5]
        )
        self.assertEqual(self.predictor.cct[0][1][5], 3)

    def test_cct_saturating_counter_decrement_and_floor(self):
        """Test decrement transitions: 3->2, 2->1, 1->0."""
        self.predictor.set_cct_entry(0, 1, 5, 3)

        # Mismatch (actual target experts do not contain 5)
        self.predictor.update(
            request_id=1,
            target_stage_id=1,
            source_experts=[1],
            actual_target_experts=[9]
        )
        self.assertEqual(self.predictor.cct[0][1][5], 2)

        self.predictor.update(
            request_id=1,
            target_stage_id=1,
            source_experts=[1],
            actual_target_experts=[9]
        )
        self.assertEqual(self.predictor.cct[0][1][5], 1)

        self.predictor.update(
            request_id=1,
            target_stage_id=1,
            source_experts=[1],
            actual_target_experts=[9]
        )
        self.assertEqual(self.predictor.cct[0][1][5], 0)

    def test_cct_zero_confidence_replacement_eligibility(self):
        """
        Algorithm 3 Lines 11-14: When ConfScore == 0 upon mismatch,
        candidate is replaced by an un-stored actual target expert and initialized to 2.
        """
        self.predictor.set_cct_entry(0, 1, 5, 0)
        self.predictor.set_cct_entry(0, 1, 6, 2)

        # Actual target experts: 7 and 8 (5 and 6 are mismatches)
        self.predictor.update(
            request_id=1,
            target_stage_id=1,
            source_experts=[1],
            actual_target_experts=[7, 8]
        )

        dest_map = self.predictor.cct[0][1]
        # Candidate 5 had score 0, should be replaced by lowest un-stored actual expert (7)
        self.assertNotIn(5, dest_map)
        self.assertIn(7, dest_map)
        self.assertEqual(dest_map[7], 2)
        # Candidate 6 had score 2, decremented to 1
        self.assertEqual(dest_map[6], 1)

    def test_history_table_isolation_and_timing(self):
        """History Table stores actual selections per request and stage, updated post-routing."""
        # Initial cold start: HT empty
        self.assertEqual(self.predictor.history_table.get(1, {}).get(1, []), [])

        # Update request 1 at stage 1
        self.predictor.update(
            request_id=1,
            target_stage_id=1,
            source_experts=[0],
            actual_target_experts=[4, 8]
        )
        self.assertEqual(self.predictor.history_table[1][1], [4, 8])

        # Request 2 must remain isolated and unaffected
        self.assertEqual(self.predictor.history_table.get(2, {}).get(1, []), [])

        # Resetting request 1 clears its history
        self.predictor.reset(1)
        self.assertNotIn(1, self.predictor.history_table)

    def test_multi_source_spatial_score_accumulation(self):
        """Algorithm 2 Lines 3-5: Multiple source experts accumulate confidence for same candidate."""
        # Both source expert 1 and source expert 2 point to candidate 10
        self.predictor.set_cct_entry(0, 1, 10, 3)
        self.predictor.set_cct_entry(0, 2, 10, 2)

        scores = self.predictor.compute_confidence_scores(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[1, 2]
        )
        # 3 + 2 = 5
        self.assertEqual(scores[10], 5)

    def test_equation1_spatio_temporal_combination(self):
        """Equation 1: Candidate appearing in both CCT and HT receives sum of scores."""
        self.predictor.set_cct_entry(0, 1, 7, 3)
        self.predictor.set_history_entry(request_id=1, stage_id=1, expert_ids=[7])

        scores = self.predictor.compute_confidence_scores(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[1]
        )
        # Spatial 3 + Temporal 2 = 5
        self.assertEqual(scores[7], 5)

    def test_predict_threshold_filtering(self):
        """Algorithm 2 Line 10: PrefetchSet contains candidates with ConfScore >= 2."""
        self.predictor.set_cct_entry(0, 1, 4, 1)  # Below threshold
        self.predictor.set_cct_entry(0, 1, 5, 2)  # Meets threshold
        self.predictor.set_cct_entry(0, 1, 6, 3)  # Exceeds threshold

        threshold_set = self.predictor.predict_threshold(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[1]
        )
        self.assertIn(5, threshold_set)
        self.assertIn(6, threshold_set)
        self.assertNotIn(4, threshold_set)

    def test_budget_limited_prediction(self):
        """Fixed budget selection takes top-B scored candidates with deterministic tie-breaking."""
        self.predictor.set_cct_entry(0, 1, 4, 3)
        self.predictor.set_cct_entry(0, 1, 8, 3)  # Tied with 4
        self.predictor.set_cct_entry(0, 1, 2, 2)

        preds_b1 = self.predictor.predict(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[1],
            budget=1
        )
        # Between 4 and 8 (both score 3), 4 has lower ID
        self.assertEqual(preds_b1, [4])

        preds_b2 = self.predictor.predict(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[1],
            budget=2
        )
        self.assertEqual(preds_b2, [4, 8])

    def test_cold_start_prediction_without_history(self):
        """Cold start prediction relies solely on spatial CCT without crashing."""
        self.predictor.set_cct_entry(0, 2, 9, 3)
        preds = self.predictor.predict(
            request_id=99,
            target_stage_id=1,
            current_source_experts=[2],
            budget=1
        )
        self.assertEqual(preds, [9])


if __name__ == "__main__":
    unittest.main()

