"""
Hand-Calculated Verification Example Test.

Strictly verifies the deterministic numerical example documented in
docs/predictor_comparison/01_ST_MOE_PAPER_FIDELITY.md Section 7.
"""

import unittest
from src.predictor.st_moe_paper import STMoEPaperPredictor


class TestPredictorHandExample(unittest.TestCase):
    """
    Validates exact numerical predictions, confidence score combinations,
    threshold filtering, deterministic tie-breaking, and table updates.
    """

    def setUp(self):
        # 6 candidate experts (0..5), 2 MoE stages (0 and 1), Top-K = 2
        self.predictor = STMoEPaperPredictor(
            num_moe_stages=2,
            num_routed_experts=6,
            top_k=2,
            cct_capacity=2,
            confidence_threshold=2
        )

        # Setup explicit CCT entries for transition 0 -> 1
        # Source 0 points to 1 (conf 3) and 2 (conf 2)
        self.predictor.set_cct_entry(0, 0, 1, 3)
        self.predictor.set_cct_entry(0, 0, 2, 2)

        # Source 1 points to 2 (conf 3) and 4 (conf 2)
        self.predictor.set_cct_entry(0, 1, 2, 3)
        self.predictor.set_cct_entry(0, 1, 4, 2)

        # Setup History Table for request 1 at target stage 1 with previous token experts {2, 3}
        self.predictor.set_history_entry(request_id=1, stage_id=1, expert_ids=[2, 3])

    def test_hand_calculated_confidence_scores(self):
        """
        Verify exact confidence scores for all 6 candidate experts:
        Exp 0: 0
        Exp 1: 3 (spatial 3 + temporal 0)
        Exp 2: 7 (spatial 2 + 3 = 5, plus temporal 2)
        Exp 3: 2 (spatial 0 + temporal 2)
        Exp 4: 2 (spatial 2 + temporal 0)
        Exp 5: 0
        """
        scores = self.predictor.compute_confidence_scores(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[0, 1]
        )

        self.assertEqual(scores[0], 0)
        self.assertEqual(scores[1], 3)
        self.assertEqual(scores[2], 7)
        self.assertEqual(scores[3], 2)
        self.assertEqual(scores[4], 2)
        self.assertEqual(scores[5], 0)

    def test_hand_calculated_threshold_set(self):
        """
        Candidates with ConfScore >= 2: {1, 2, 3, 4}.
        """
        threshold_candidates = self.predictor.predict_threshold(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[0, 1]
        )
        self.assertEqual(set(threshold_candidates), {1, 2, 3, 4})

    def test_hand_calculated_budget_selections(self):
        """
        Ranks:
        1. Exp 2 (score 7)
        2. Exp 1 (score 3)
        3. Exp 3 (score 2, ID 3 < 4)
        4. Exp 4 (score 2, ID 4)
        """
        preds_b1 = self.predictor.predict(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[0, 1],
            budget=1
        )
        self.assertEqual(preds_b1, [2])

        preds_b2 = self.predictor.predict(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[0, 1],
            budget=2
        )
        self.assertEqual(preds_b2, [2, 1])

        preds_b3 = self.predictor.predict(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[0, 1],
            budget=3
        )
        self.assertEqual(preds_b3, [2, 1, 3])

        preds_b4 = self.predictor.predict(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[0, 1],
            budget=4
        )
        self.assertEqual(preds_b4, [2, 1, 3, 4])

    def test_hand_calculated_algorithm3_update(self):
        """
        Reveal actual target experts: F_j = [2, 5].
        Updating source expert 0:
        - Dest 1: 1 not in [2, 5] -> decrements 3 to 2
        - Dest 2: 2 in [2, 5] -> increments 2 to 3
        Updating source expert 1:
        - Dest 2: 2 in [2, 5] -> increments 3 to 3 (saturates)
        - Dest 4: 4 not in [2, 5] -> decrements 2 to 1
        History Table updated to [2, 5].
        """
        self.predictor.update(
            request_id=1,
            target_stage_id=1,
            source_experts=[0, 1],
            actual_target_experts=[2, 5]
        )

        cct_0 = self.predictor.cct[0][0]
        self.assertEqual(cct_0[1], 2)
        self.assertEqual(cct_0[2], 3)

        cct_1 = self.predictor.cct[0][1]
        self.assertEqual(cct_1[2], 3)
        self.assertEqual(cct_1[4], 1)

        ht_entry = self.predictor.history_table[1][1]
        self.assertEqual(ht_entry, [2, 5])


if __name__ == "__main__":
    unittest.main()

