"""
Unit tests validating Strict Request Isolation across all predictors.
Ensures no cross-contamination between concurrent or interleaved sequences.
"""

import unittest
from src.predictor.st_moe_paper import STMoEPaperPredictor
from src.predictor.st_moe_predictor import STMoEPredictor
from src.predictor.cstp_predictor import CSTPPredictor
from src.predictor.base_predictor import STMoELegacyWrapper, CSTPWrapper


class TestPredictorRequestIsolation(unittest.TestCase):
    """
    Verifies that temporal state (History Table) is strictly partitioned by request_id.
    """

    def setUp(self):
        self.paper_pred = STMoEPaperPredictor(num_moe_stages=4, num_routed_experts=32, top_k=3)
        self.legacy_pred = STMoELegacyWrapper(STMoEPredictor(num_moe_stages=4, num_routed_experts=32, top_k=3))
        self.cstp_inner = CSTPPredictor(num_moe_stages=4, num_routed_experts=32, top_k=3)
        self.cstp_pred = CSTPWrapper(self.cstp_inner)

    def test_st_moe_paper_and_legacy_request_isolation(self):
        """
        Tests Section 4.3 request isolation for ST-MoE:
        - Request A has previous-token experts {1, 5, 8}.
        - Request B has previous-token experts {12, 18, 21}.
        - Predicting for B must not use A's HT.
        """
        st_moe_predictors = [self.paper_pred, self.legacy_pred]
        req_a = 101
        req_b = 202

        for pred in st_moe_predictors:
            pred.reset(req_a)
            pred.reset(req_b)

            # Baseline prediction for B before any history
            baseline_b = pred.predict(
                request_id=req_b,
                target_stage_id=1,
                current_source_experts=[],
                budget=3
            )

            # Update Req A with actual experts {1, 5, 8}
            pred.update(
                request_id=req_a,
                target_stage_id=1,
                source_experts=[],
                actual_target_experts=[1, 5, 8]
            )

            # Predict for Req B: must remain unaffected by Req A
            after_a_update_b = pred.predict(
                request_id=req_b,
                target_stage_id=1,
                current_source_experts=[],
                budget=3
            )
            self.assertEqual(baseline_b, after_a_update_b)

            # Update Req B with actual experts {12, 18, 21}
            pred.update(
                request_id=req_b,
                target_stage_id=1,
                source_experts=[],
                actual_target_experts=[12, 18, 21]
            )

            # Req B predicts {12, 18, 21}
            preds_b = pred.predict(
                request_id=req_b,
                target_stage_id=1,
                current_source_experts=[],
                budget=3
            )
            self.assertEqual(set(preds_b), {12, 18, 21})

            # Req A predicts {1, 5, 8}
            preds_a = pred.predict(
                request_id=req_a,
                target_stage_id=1,
                current_source_experts=[],
                budget=3
            )
            self.assertEqual(set(preds_a), {1, 5, 8})

            # Resetting Req A clears A's history without affecting Req B
            pred.reset(req_a)
            preds_b_post_reset = pred.predict(
                request_id=req_b,
                target_stage_id=1,
                current_source_experts=[],
                budget=3
            )
            self.assertEqual(set(preds_b_post_reset), {12, 18, 21})

            preds_a_post_reset = pred.predict(
                request_id=req_a,
                target_stage_id=1,
                current_source_experts=[],
                budget=3
            )
            self.assertNotEqual(set(preds_a_post_reset), {1, 5, 8})

    def test_cstp_request_isolation(self):
        """
        Tests request isolation for CSTP:
        Temporal counts map previous expert 1 -> 9, and previous expert 12 -> 7.
        Req A with previous expert 1 predicts 9.
        Req B with previous expert 12 predicts 7.
        """
        # Configure transition counts
        self.cstp_inner.temporal_counts[1, 1, 9] = 100.0
        self.cstp_inner.temporal_counts[1, 12, 7] = 100.0

        req_a = 501
        req_b = 602

        self.cstp_pred.reset(req_a)
        self.cstp_pred.reset(req_b)

        # Update Req A with expert 1
        self.cstp_pred.update(
            request_id=req_a,
            target_stage_id=1,
            source_experts=[],
            actual_target_experts=[1]
        )

        # Update Req B with expert 12
        self.cstp_pred.update(
            request_id=req_b,
            target_stage_id=1,
            source_experts=[],
            actual_target_experts=[12]
        )

        # Req A should top-rank candidate 9
        preds_a = self.cstp_pred.predict(
            request_id=req_a,
            target_stage_id=1,
            current_source_experts=[],
            budget=1
        )
        self.assertEqual(preds_a, [9])

        # Req B should top-rank candidate 7
        preds_b = self.cstp_pred.predict(
            request_id=req_b,
            target_stage_id=1,
            current_source_experts=[],
            budget=1
        )
        self.assertEqual(preds_b, [7])

        # Resetting Req A clears A's history without affecting Req B
        self.cstp_pred.reset(req_a)
        preds_b_post_reset = self.cstp_pred.predict(
            request_id=req_b,
            target_stage_id=1,
            current_source_experts=[],
            budget=1
        )
        self.assertEqual(preds_b_post_reset, [7])

        preds_a_post_reset = self.cstp_pred.predict(
            request_id=req_a,
            target_stage_id=1,
            current_source_experts=[],
            budget=1
        )
        self.assertNotEqual(preds_a_post_reset, [9])


if __name__ == "__main__":
    unittest.main()
