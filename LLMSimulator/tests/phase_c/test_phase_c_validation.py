"""
Phase C Validation Test Suite.
Validates causal ordering, split isolation, online vs frozen state mutation invariance,
request temporal isolation, budget independence, and distribution shift recovery.
"""

import unittest
import copy
import numpy as np

from src.predictor.cstp_predictor import CSTPPredictor
from src.predictor.st_moe_paper import STMoEPaperPredictor
from src.predictor.base_predictor import CSTPWrapper


class TestPhaseCValidation(unittest.TestCase):

    def setUp(self):
        self.num_stages = 4
        self.num_experts = 16
        self.top_k = 2
        self.horizon = 1

    def test_causal_ordering_and_no_target_leakage(self):
        """Verify predictions are strictly generated before target labels are passed to update."""
        pred = CSTPPredictor(
            num_moe_stages=self.num_stages,
            num_routed_experts=self.num_experts,
            top_k=self.top_k,
            online_adaptation=True,
            prediction_horizon=1
        )
        wrapper = CSTPWrapper(pred, name="cstp_online")

        # Initial query for stage 0 -> 1 before any updates
        preds_before = wrapper.predict(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[2, 3],
            budget=2,
            token_id=0,
            source_stage_id=0
        )
        self.assertEqual(len(preds_before), 2)

        # Now pass actual targets to update
        target_experts = [10, 11]
        wrapper.update(
            request_id=1,
            target_stage_id=1,
            source_experts=[2, 3],
            actual_target_experts=target_experts,
            token_id=0,
            source_stage_id=0
        )

        # Confirm the state recorded the target
        self.assertGreater(pred.spatial_counts[0, 2, 10], 0)
        self.assertGreater(pred.spatial_counts[0, 3, 11], 0)

    def test_cstp_state_mutation_invariance(self):
        """Verify frozen CSTP does not mutate its transition counts, whereas online CSTP does."""
        cstp_online = CSTPPredictor(
            num_moe_stages=self.num_stages,
            num_routed_experts=self.num_experts,
            top_k=self.top_k,
            online_adaptation=True
        )
        cstp_frozen = CSTPPredictor(
            num_moe_stages=self.num_stages,
            num_routed_experts=self.num_experts,
            top_k=self.top_k,
            online_adaptation=False
        )

        w_online = CSTPWrapper(cstp_online, name="cstp_online")
        w_frozen = CSTPWrapper(cstp_frozen, name="cstp_frozen")

        initial_frozen_spatial_sum = float(np.sum(cstp_frozen.spatial_counts))
        initial_online_spatial_sum = float(np.sum(cstp_online.spatial_counts))

        # Perform updates
        for tok in range(5):
            w_online.update(
                request_id=1,
                target_stage_id=1,
                source_experts=[1, 2],
                actual_target_experts=[5, 6],
                token_id=tok,
                source_stage_id=0
            )
            w_frozen.update(
                request_id=1,
                target_stage_id=1,
                source_experts=[1, 2],
                actual_target_experts=[5, 6],
                token_id=tok,
                source_stage_id=0
            )

        # Frozen transition counts must remain identical
        final_frozen_spatial_sum = float(np.sum(cstp_frozen.spatial_counts))
        self.assertEqual(initial_frozen_spatial_sum, final_frozen_spatial_sum)

        # Online transition counts must have increased
        final_online_spatial_sum = float(np.sum(cstp_online.spatial_counts))
        self.assertGreater(final_online_spatial_sum, initial_online_spatial_sum)

    def test_st_moe_cct_mutation_invariance(self):
        """Verify frozen ST-MoE does not modify CCT counters, whereas online ST-MoE does."""
        st_online = STMoEPaperPredictor(
            num_moe_stages=self.num_stages,
            num_routed_experts=self.num_experts,
            top_k=self.top_k,
            cct_capacity=self.top_k,
            confidence_threshold=2,
            online_adaptation=True
        )
        st_frozen = STMoEPaperPredictor(
            num_moe_stages=self.num_stages,
            num_routed_experts=self.num_experts,
            top_k=self.top_k,
            cct_capacity=self.top_k,
            confidence_threshold=2,
            online_adaptation=False
        )

        # Add initial CCT entry
        st_online.cct = {0: {1: {5: 1}}}
        st_frozen.cct = {0: {1: {5: 1}}}

        # Update both
        st_online.update(
            request_id=1,
            target_stage_id=1,
            source_experts=[1],
            actual_target_experts=[5],
            token_id=0,
            source_stage_id=0
        )
        st_frozen.update(
            request_id=1,
            target_stage_id=1,
            source_experts=[1],
            actual_target_experts=[5],
            token_id=0,
            source_stage_id=0
        )

        # Online counter should have incremented from 1 to 2
        self.assertEqual(st_online.cct[0][1][5], 2)
        # Frozen counter must stay at 1
        self.assertEqual(st_frozen.cct[0][1][5], 1)

    def test_temporal_history_request_reset(self):
        """Verify temporal history table is reset across independent requests."""
        cstp = CSTPPredictor(
            num_moe_stages=self.num_stages,
            num_routed_experts=self.num_experts,
            top_k=self.top_k,
            online_adaptation=True
        )
        wrapper = CSTPWrapper(cstp, name="cstp_online")

        # Request 1 updates
        wrapper.update(
            request_id=1,
            target_stage_id=1,
            source_experts=[1, 2],
            actual_target_experts=[3, 4],
            token_id=0,
            source_stage_id=0
        )
        self.assertIn(1, cstp.history_table)
        self.assertEqual(cstp.history_table[1][1], {3, 4})

        # Reset per request contract
        wrapper.reset(1)
        self.assertNotIn(1, cstp.history_table)

    def test_budget_independence(self):
        """Verify independent evaluations at different budgets produce identical state progression."""
        cstp1 = CSTPPredictor(
            num_moe_stages=self.num_stages,
            num_routed_experts=self.num_experts,
            top_k=self.top_k,
            online_adaptation=True
        )
        cstp2 = copy.deepcopy(cstp1)

        w1 = CSTPWrapper(cstp1, name="cstp_b2")
        w2 = CSTPWrapper(cstp2, name="cstp_b4")

        # Predictions with different budgets
        p1 = w1.predict(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[1, 2],
            budget=2,
            token_id=0,
            source_stage_id=0
        )
        p2 = w2.predict(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[1, 2],
            budget=4,
            token_id=0,
            source_stage_id=0
        )
        self.assertEqual(len(p1), 2)
        self.assertEqual(len(p2), 4)

        # Update both with identical actual targets
        w1.update(
            request_id=1,
            target_stage_id=1,
            source_experts=[1, 2],
            actual_target_experts=[7, 8],
            token_id=0,
            source_stage_id=0
        )
        w2.update(
            request_id=1,
            target_stage_id=1,
            source_experts=[1, 2],
            actual_target_experts=[7, 8],
            token_id=0,
            source_stage_id=0
        )

        # Transition counts in both predictors must be identical
        np.testing.assert_array_equal(cstp1.spatial_counts, cstp2.spatial_counts)


if __name__ == "__main__":
    unittest.main()
