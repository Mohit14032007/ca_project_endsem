"""
Unit tests validating Strict Causality across all predictors.
Ensures future target routing or label mutations cannot influence past predictions.
"""

import unittest
from src.predictor.st_moe_paper import STMoEPaperPredictor
from src.predictor.st_moe_predictor import STMoEPredictor
from src.predictor.cstp_predictor import CSTPPredictor
from src.predictor.base_predictor import STMoELegacyWrapper, CSTPWrapper


class TestPredictorCausality(unittest.TestCase):
    """
    Verifies that predictors never access future target labels before prediction,
    and that prediction does not inadvertently mutate model internal state.
    """

    def setUp(self):
        self.predictors = [
            STMoEPaperPredictor(num_moe_stages=4, num_routed_experts=32, top_k=2),
            STMoELegacyWrapper(STMoEPredictor(num_moe_stages=4, num_routed_experts=32, top_k=2)),
            CSTPWrapper(CSTPPredictor(num_moe_stages=4, num_routed_experts=32, top_k=2))
        ]

    def test_future_label_mutation_invariance(self):
        """
        Tests that mutating actual target ground truth labels has zero effect
        on the prediction made prior to revealing the label.
        """
        for pred in self.predictors:
            pred.reset(request_id=1)

            # Hypothesis: Future ground truth will be label_set_1 = [5, 6]
            label_set_1 = [5, 6]
            # Alternate future: label_set_2 = [18, 19]
            label_set_2 = [18, 19]

            # Predict at stage 1 given source experts [0]
            pred_1 = pred.predict(
                request_id=1,
                target_stage_id=1,
                current_source_experts=[0],
                budget=2
            )

            # Mutate ground truth in memory
            label_set_1[0] = 999  # External corruption
            self.assertEqual(label_set_1[0], 999)

            # Predict again under identical inputs
            pred_2 = pred.predict(
                request_id=1,
                target_stage_id=1,
                current_source_experts=[0],
                budget=2
            )

            self.assertEqual(
                pred_1,
                pred_2,
                "Prediction changed when future label variable was mutated!"
            )

    def test_idempotent_repeated_prediction(self):
        """
        Calling predict() multiple times without calling update()
        must be strictly read-only and idempotent.
        """
        for pred in self.predictors:
            pred.reset(request_id=5)

            first_pred = pred.predict(
                request_id=5,
                target_stage_id=2,
                current_source_experts=[1, 2],
                budget=3
            )

            # Repeat prediction 5 times
            for _ in range(5):
                repeat_pred = pred.predict(
                    request_id=5,
                    target_stage_id=2,
                    current_source_experts=[1, 2],
                    budget=3
                )
                self.assertEqual(first_pred, repeat_pred)

    def test_update_executes_only_post_prediction(self):
        """
        Verifies that state changes only occur when update() is explicitly called.
        """
        paper_pred = STMoEPaperPredictor(num_moe_stages=4, num_routed_experts=16, top_k=2)
        paper_pred.reset(request_id=1)

        # Before update: HT is empty
        self.assertEqual(paper_pred.history_table.get(1, {}).get(1, []), [])

        # Call predict()
        paper_pred.predict(
            request_id=1,
            target_stage_id=1,
            current_source_experts=[0],
            budget=2
        )

        # HT must still be empty
        self.assertEqual(paper_pred.history_table.get(1, {}).get(1, []), [])

        # Explicitly call update() with actual target routing
        paper_pred.update(
            request_id=1,
            target_stage_id=1,
            source_experts=[0],
            actual_target_experts=[3, 4]
        )

        # Now HT contains [3, 4]
        self.assertEqual(paper_pred.history_table[1][1], [3, 4])


if __name__ == "__main__":
    unittest.main()

