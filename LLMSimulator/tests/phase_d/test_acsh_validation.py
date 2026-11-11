"""
Phase D ACSH Validation Test Suite.
Validates mathematical correctness, causality, reliability updates, weight adaptation,
stage isolation, numerical stability, and fast recovery detection.
"""

import unittest
import copy
import numpy as np

from src.predictor.cstp_predictor import CSTPPredictor
from src.predictor.st_moe_paper import STMoEPaperPredictor
from src.predictor.acsh_predictor import ACSHPredictor


class TestACSHValidation(unittest.TestCase):

    def setUp(self):
        self.num_stages = 4
        self.num_experts = 16
        self.top_k = 2

        self.cstp = CSTPPredictor(
            num_moe_stages=self.num_stages,
            num_routed_experts=self.num_experts,
            top_k=self.top_k,
            online_adaptation=True,
            prediction_horizon=1
        )
        self.st_moe = STMoEPaperPredictor(
            num_moe_stages=self.num_stages,
            num_routed_experts=self.num_experts,
            top_k=self.top_k,
            cct_capacity=self.top_k,
            confidence_threshold=2,
            online_adaptation=True,
            prediction_horizon=1
        )

    def test_first_event_weights_are_equal(self):
        """Verify initial weights are exactly 50/50 when reliabilities are initialized equally."""
        acsh = ACSHPredictor(
            cstp_predictor=copy.deepcopy(self.cstp),
            st_moe_predictor=copy.deepcopy(self.st_moe),
            variant="stage_aware",
            tau=0.2
        )
        w_c, w_m = acsh.get_fusion_weights(request_id=1, target_stage_id=1, horizon=1)
        self.assertAlmostEqual(w_c, 0.5, places=5)
        self.assertAlmostEqual(w_m, 0.5, places=5)
        self.assertAlmostEqual(w_c + w_m, 1.0, places=5)

    def test_higher_cstp_reliability_increases_cstp_weight(self):
        """Verify that when CSTP achieves higher utility than ST-MoE, CSTP's weight increases."""
        cstp_copy = copy.deepcopy(self.cstp)
        st_copy = copy.deepcopy(self.st_moe)

        # Give CSTP and ST-MoE distinct preferences
        cstp_copy.spatial_counts[0, 0, 5] = 10.0
        cstp_copy.spatial_counts[0, 1, 6] = 10.0
        st_copy.set_cct_entry(0, 0, 10, 3)
        st_copy.set_cct_entry(0, 1, 11, 3)

        acsh = ACSHPredictor(
            cstp_predictor=cstp_copy,
            st_moe_predictor=st_copy,
            variant="stage_aware",
            beta=0.8,
            tau=0.2
        )

        preds = acsh.predict(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[0, 1],
            budget=2,
            token_id=0,
            source_stage_id=0
        )
        self.assertEqual(len(preds), 2)

        # Actual targets match CSTP [5, 6], so CSTP hits 2/2 and ST-MoE hits 0/2
        acsh.update(
            request_id=1,
            target_stage_id=1,
            source_experts=[0, 1],
            actual_target_experts=[5, 6],
            token_id=0,
            source_stage_id=0
        )

        w_c, w_m = acsh.get_fusion_weights(request_id=1, target_stage_id=1, horizon=1)
        r_c, r_m = acsh.get_current_reliability(request_id=1, target_stage_id=1, horizon=1)

        self.assertGreater(r_c, r_m)
        self.assertGreater(w_c, 0.5)
        self.assertLess(w_m, 0.5)
        self.assertAlmostEqual(w_c + w_m, 1.0, places=5)

    def test_higher_st_moe_reliability_increases_st_moe_weight(self):
        """Verify that when ST-MoE achieves higher utility than CSTP, ST-MoE's weight increases."""
        cstp_copy = copy.deepcopy(self.cstp)
        st_copy = copy.deepcopy(self.st_moe)

        cstp_copy.spatial_counts[0, 0, 5] = 10.0
        cstp_copy.spatial_counts[0, 1, 6] = 10.0
        st_copy.set_cct_entry(0, 0, 10, 3)
        st_copy.set_cct_entry(0, 1, 11, 3)

        acsh = ACSHPredictor(
            cstp_predictor=cstp_copy,
            st_moe_predictor=st_copy,
            variant="stage_aware",
            beta=0.8,
            tau=0.2
        )

        preds = acsh.predict(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[0, 1],
            budget=2,
            token_id=0,
            source_stage_id=0
        )
        self.assertEqual(len(preds), 2)

        # Actual targets match ST-MoE [10, 11], so ST-MoE hits 2/2 and CSTP hits 0/2
        acsh.update(
            request_id=1,
            target_stage_id=1,
            source_experts=[0, 1],
            actual_target_experts=[10, 11],
            token_id=0,
            source_stage_id=0
        )

        w_c, w_m = acsh.get_fusion_weights(request_id=1, target_stage_id=1, horizon=1)
        r_c, r_m = acsh.get_current_reliability(request_id=1, target_stage_id=1, horizon=1)

        self.assertGreater(r_m, r_c)
        self.assertGreater(w_m, 0.5)
        self.assertLess(w_c, 0.5)
        self.assertAlmostEqual(w_c + w_m, 1.0, places=5)

    def test_stage_aware_isolation(self):
        """Verify reliability updates at stage 1 do not contaminate stage 2 in Stage-Aware variant."""
        cstp_copy = copy.deepcopy(self.cstp)
        st_copy = copy.deepcopy(self.st_moe)

        cstp_copy.spatial_counts[0, 0, 5] = 10.0
        st_copy.set_cct_entry(0, 0, 10, 3)

        acsh = ACSHPredictor(
            cstp_predictor=cstp_copy,
            st_moe_predictor=st_copy,
            variant="stage_aware",
            beta=0.8,
            tau=0.2
        )

        # Query and update stage 1
        acsh.predict(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[0, 1],
            budget=2,
            token_id=0,
            source_stage_id=0
        )
        acsh.update(
            request_id=1,
            target_stage_id=1,
            source_experts=[0, 1],
            actual_target_experts=[5, 6],
            token_id=0,
            source_stage_id=0
        )

        # Stage 1 weights should have adapted to favor CSTP
        w_c_1, _ = acsh.get_fusion_weights(request_id=1, target_stage_id=1, horizon=1)
        self.assertGreater(w_c_1, 0.5)

        # Stage 2 weights must still be at initial 50/50
        w_c_2, w_m_2 = acsh.get_fusion_weights(request_id=1, target_stage_id=2, horizon=1)
        self.assertAlmostEqual(w_c_2, 0.5, places=5)
        self.assertAlmostEqual(w_m_2, 0.5, places=5)

    def test_global_variant_shares_reliability_across_stages(self):
        """Verify Global variant updates a single reliability pair across stages."""
        cstp_copy = copy.deepcopy(self.cstp)
        st_copy = copy.deepcopy(self.st_moe)

        cstp_copy.spatial_counts[0, 0, 5] = 10.0
        st_copy.set_cct_entry(0, 0, 10, 3)

        acsh = ACSHPredictor(
            cstp_predictor=cstp_copy,
            st_moe_predictor=st_copy,
            variant="global",
            beta=0.8,
            tau=0.2
        )

        acsh.predict(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[0, 1],
            budget=2,
            token_id=0,
            source_stage_id=0
        )
        acsh.update(
            request_id=1,
            target_stage_id=1,
            source_experts=[0, 1],
            actual_target_experts=[5, 6],
            token_id=0,
            source_stage_id=0
        )

        # In global variant, stage 2 sees the adapted weights immediately
        w_c_2, _ = acsh.get_fusion_weights(request_id=1, target_stage_id=2, horizon=1)
        self.assertGreater(w_c_2, 0.5)

    def test_request_reset_clears_reliability(self):
        """Verify calling reset() clears reliability estimates for that request."""
        acsh = ACSHPredictor(
            cstp_predictor=copy.deepcopy(self.cstp),
            st_moe_predictor=copy.deepcopy(self.st_moe),
            variant="stage_aware"
        )

        acsh.predict(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[0, 1],
            budget=2,
            token_id=0,
            source_stage_id=0
        )
        acsh.update(
            request_id=1,
            target_stage_id=1,
            source_experts=[0, 1],
            actual_target_experts=[5, 6],
            token_id=0,
            source_stage_id=0
        )

        self.assertIn(1, acsh.rel_c_stage)

        acsh.reset(1)
        self.assertNotIn(1, acsh.rel_c_stage)

    def test_fast_recovery_triggers_on_persistent_misses(self):
        """Verify Fast Recovery variant detects high error and activates fast beta."""
        acsh = ACSHPredictor(
            cstp_predictor=copy.deepcopy(self.cstp),
            st_moe_predictor=copy.deepcopy(self.st_moe),
            variant="fast_recovery",
            shift_window=2,
            shift_threshold=0.8,
            fast_beta=0.1
        )

        # Feed 2 complete misses with shifting targets so online model cannot adapt immediately
        targets = [[14, 15], [12, 13]]
        for tok in range(2):
            acsh.predict(
                request_id=1,
                target_stage_id=1,
                current_source_experts=[0, 1],
                budget=2,
                token_id=tok,
                source_stage_id=0
            )
            acsh.update(
                request_id=1,
                target_stage_id=1,
                source_experts=[0, 1],
                actual_target_experts=targets[tok],
                token_id=tok,
                source_stage_id=0
            )

        # Fast recovery should have triggered
        self.assertGreater(len(acsh.shift_detections), 0)
        self.assertEqual(acsh.shift_detections[0]["active_beta"], 0.1)

    def test_prediction_output_properties(self):
        """Verify Top-B contains no duplicates, respects budget, and all indices are valid."""
        acsh = ACSHPredictor(
            cstp_predictor=copy.deepcopy(self.cstp),
            st_moe_predictor=copy.deepcopy(self.st_moe),
            variant="stage_aware"
        )

        budget = 4
        preds = acsh.predict(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[0, 1],
            budget=budget,
            token_id=0,
            source_stage_id=0
        )

        self.assertEqual(len(preds), budget)
        self.assertEqual(len(set(preds)), budget)
        for exp in preds:
            self.assertGreaterEqual(exp, 0)
            self.assertLess(exp, self.num_experts)


if __name__ == "__main__":
    unittest.main()
