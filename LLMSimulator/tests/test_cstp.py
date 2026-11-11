"""
Unit tests for CSTPPredictor.
Verifies Dirichlet prior smoothing, spatial scores, temporal scores, composite weighting,
cold-start behavior, and Top-B deterministic selection.
"""

import unittest
import numpy as np
from src.predictor.cstp_predictor import CSTPPredictor
from src.predictor.trace_parser import NormalizedRoutingRecord


class TestCSTPPredictor(unittest.TestCase):

    def setUp(self):
        # Small 4-stage model with 8 experts, top-2
        self.predictor = CSTPPredictor(
            num_moe_stages=4,
            num_routed_experts=8,
            top_k=2,
            alpha=0.5,
            smoothing_lambda=1.0,
            prior_epsilon=0.1
        )

    def test_marginal_prior_uniform_on_zero_counts(self):
        prior = self.predictor.get_marginal_prior(stage_id=1)
        self.assertEqual(len(prior), 8)
        expected_uniform = 1.0 / 8.0
        for p in prior:
            self.assertAlmostEqual(p, expected_uniform, places=5)

    def test_marginal_prior_with_observations(self):
        # Observe expert 3 twice at stage 2
        self.predictor.stage_marginal_counts[2, 3] = 2.0
        prior = self.predictor.get_marginal_prior(stage_id=2)
        # Prior for expert 3 should be strictly higher than all other experts
        self.assertGreater(prior[3], prior[0])
        self.assertGreater(prior[3], prior[1])
        self.assertAlmostEqual(np.sum(prior), 1.0, places=5)

    def test_spatial_score_with_coactivations(self):
        # Stage 0 expert 1 strongly co-activates with stage 1 expert 5
        self.predictor.spatial_counts[0, 1, 5] = 10.0
        score = self.predictor.compute_spatial_score(target_stage_id=1, source_experts=[1])
        # Score for expert 5 should be highest
        best_expert = int(np.argmax(score))
        self.assertEqual(best_expert, 5)

    def test_temporal_score_cold_start_equals_prior(self):
        # No history for request 1 at stage 2
        prior = self.predictor.get_marginal_prior(stage_id=2)
        score = self.predictor.compute_temporal_score(request_id=1, target_stage_id=2)
        np.testing.assert_allclose(score, prior)

    def test_composite_score_alpha_weighting(self):
        # Pure spatial mode (alpha = 1.0)
        self.predictor.alpha = 1.0
        self.predictor.spatial_counts[0, 2, 7] = 20.0
        preds_spatial = self.predictor.predict(request_id=1, target_stage_id=1, current_source_experts=[2], budget=1)
        self.assertEqual(preds_spatial, [7])

        # Pure temporal mode (alpha = 0.0)
        self.predictor.alpha = 0.0
        self.predictor.temporal_counts[1, 4, 0] = 20.0
        self.predictor.history_table[1] = {1: {4}}
        preds_temporal = self.predictor.predict(request_id=1, target_stage_id=1, current_source_experts=[2], budget=1)
        self.assertEqual(preds_temporal, [0])

    def test_deterministic_tie_breaking(self):
        # With zero counts, prior is completely flat
        # Ties should be broken by expert_id ascending: [0, 1, 2]
        preds = self.predictor.predict(request_id=1, target_stage_id=1, current_source_experts=[], budget=3)
        self.assertEqual(preds, [0, 1, 2])

    def test_fit_profile_records(self):
        # Create a small sequence of training records
        # Request 1, Token 1, Stage 0 -> expert 2; Stage 1 -> expert 6
        rec1 = NormalizedRoutingRecord(request_id=1, token_id=1, moe_stage_id=0, raw_transformer_layer_id=3, topk_rank=1, expert_id=2)
        rec2 = NormalizedRoutingRecord(request_id=1, token_id=1, moe_stage_id=1, raw_transformer_layer_id=4, topk_rank=1, expert_id=6)

        pred = CSTPPredictor(num_moe_stages=3, num_routed_experts=8, top_k=1)
        pred.fit_profile([rec1, rec2])

        # Verify marginal counts
        self.assertEqual(pred.stage_marginal_counts[0, 2], 1.0)
        self.assertEqual(pred.stage_marginal_counts[1, 6], 1.0)
        # Verify spatial transition
        self.assertEqual(pred.spatial_counts[0, 2, 6], 1.0)


if __name__ == "__main__":
    unittest.main()

