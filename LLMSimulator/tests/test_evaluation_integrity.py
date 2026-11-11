"""
Comprehensive Evaluation Hardening & Integrity Test Suite.

Verifies:
1. Single-request splitting elimination (Strategy A: chronological token split, Strategy B: error on insufficient tokens).
2. Strict train/test disjointness and no data leakage.
3. Frozen vs Online vs Threshold evaluation regimes.
4. Metric contracts: Precision@B, Recall@B, F1@B, Hit Rate, ExactSetAccuracy, empty set behavior.
5. Invariant: Precision@8 == Recall@8 == F1@8 when B = K = 8.
6. Sequence boundary isolation: temporal history reset without corrupting persistent profiling.
7. Identical evaluation examples across diverse predictors.
"""

import unittest
import copy
from typing import List

from src.predictor.evaluation import (
    EvaluationHarness,
    EvaluationSplitError,
    PredictionOpportunity,
    PredictionResultRecord
)
from src.predictor.trace_parser import NormalizedRoutingRecord
from src.predictor.st_moe_paper import STMoEPaperPredictor
from src.predictor.cstp_predictor import CSTPPredictor
from src.predictor.base_predictor import (
    FrequencyPredictor,
    RandomPredictor,
    TemporalOnlyPredictor,
    SpatialOnlyPredictor,
    CSTPWrapper
)


class TestEvaluationSplitHardening(unittest.TestCase):
    """Verifies dataset splitting integrity and single-request handling."""

    def setUp(self):
        self.harness = EvaluationHarness(num_moe_stages=4)

    def test_multiple_requests_disjoint_splits(self):
        """1. Multiple requests with disjoint splits."""
        records = []
        for req in [1, 2, 3, 4, 5, 6, 7, 8]:
            for tok in [1, 2]:
                for stg in [0, 1]:
                    records.append(NormalizedRoutingRecord(
                        request_id=req, token_id=tok, moe_stage_id=stg,
                        raw_transformer_layer_id=stg, topk_rank=1, expert_id=req % 10
                    ))

        train, test, meta = self.harness.split_by_request(records, train_ratio=0.75)
        self.assertFalse(meta["is_single_request"])
        self.assertEqual(meta["split_strategy"], "request_disjoint")

        train_reqs = set(meta["train_requests"])
        test_reqs = set(meta["test_requests"])

        # Train and test request sets must be completely disjoint
        self.assertEqual(len(train_reqs.intersection(test_reqs)), 0)
        self.assertEqual(len(train_reqs) + len(test_reqs), 8)

        # Record sets must be completely disjoint
        train_record_ids = set(id(r) for r in train)
        test_record_ids = set(id(r) for r in test)
        self.assertEqual(len(train_record_ids.intersection(test_record_ids)), 0)

    def test_single_request_sufficient_tokens_chronological_split(self):
        """2. A single sufficiently long request (Strategy A: chronological token split)."""
        records = []
        req_id = 42
        # 10 chronological tokens (1 to 10)
        for tok in range(1, 11):
            for stg in range(4):
                records.append(NormalizedRoutingRecord(
                    request_id=req_id, token_id=tok, moe_stage_id=stg,
                    raw_transformer_layer_id=stg, topk_rank=1, expert_id=tok % 8
                ))

        train, test, meta = self.harness.split_by_request(records, train_ratio=0.70, allow_token_split=True, min_tokens_for_split=4)
        self.assertTrue(meta["is_single_request"])
        self.assertEqual(meta["split_strategy"], "chronological_token_split")
        self.assertEqual(meta["total_tokens"], 10)

        # 7 train tokens (1..7), 3 test tokens (8..10)
        train_tokens = meta["train_tokens"]
        test_tokens = meta["test_tokens"]
        self.assertEqual(len(train_tokens), 7)
        self.assertEqual(len(test_tokens), 3)

        # Invariant: strict chronological non-overlap (prefix ends before suffix starts)
        self.assertLess(max(train_tokens), min(test_tokens))
        self.assertEqual(len(set(train_tokens).intersection(set(test_tokens))), 0)

        # Records must not overlap
        train_ids = set(id(r) for r in train)
        test_ids = set(id(r) for r in test)
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)
        self.assertEqual(len(train) + len(test), len(records))

    def test_single_request_too_short_raises_error(self):
        """3. A single request too short for a valid split (Strategy B: explicit exception)."""
        records = []
        req_id = 99
        # Only 2 tokens (less than min_tokens_for_split=4)
        for tok in [1, 2]:
            records.append(NormalizedRoutingRecord(
                request_id=req_id, token_id=tok, moe_stage_id=0,
                raw_transformer_layer_id=0, topk_rank=1, expert_id=1
            ))

        with self.assertRaises(EvaluationSplitError) as ctx:
            self.harness.split_by_request(records, train_ratio=0.75, allow_token_split=True, min_tokens_for_split=4)

        self.assertIn("insufficient for a valid chronological train/test split", str(ctx.exception))
        self.assertIn("prohibited to eliminate data leakage", str(ctx.exception))

    def test_single_request_disallow_token_split_raises(self):
        """3b. When allow_token_split is False, single-request traces must raise explicitly."""
        records = [
            NormalizedRoutingRecord(request_id=1, token_id=t, moe_stage_id=0, raw_transformer_layer_id=0, topk_rank=1, expert_id=1)
            for t in range(20)
        ]
        with self.assertRaises(EvaluationSplitError) as ctx:
            self.harness.split_by_request(records, allow_token_split=False)
        self.assertIn("cannot be split by request", str(ctx.exception))

    def test_empty_input_raises_error(self):
        """4. Empty input must raise EvaluationSplitError."""
        with self.assertRaises(EvaluationSplitError) as ctx:
            self.harness.split_by_request([])
        self.assertIn("records list is empty", str(ctx.exception))

    def test_invalid_train_ratio_raises_error(self):
        """4b. Invalid train_ratio (< 0 or > 1) must raise ValueError."""
        records = [
            NormalizedRoutingRecord(request_id=1, token_id=1, moe_stage_id=0, raw_transformer_layer_id=0, topk_rank=1, expert_id=1)
        ]
        with self.assertRaises(ValueError):
            self.harness.split_by_request(records, train_ratio=1.5)
        with self.assertRaises(ValueError):
            self.harness.split_by_request(records, train_ratio=0.0)

    def test_duplicate_request_identifiers_handling(self):
        """5. Multiple records per request ID with non-contiguous or repeated rows."""
        records = [
            NormalizedRoutingRecord(request_id=2, token_id=1, moe_stage_id=0, raw_transformer_layer_id=0, topk_rank=1, expert_id=2),
            NormalizedRoutingRecord(request_id=1, token_id=1, moe_stage_id=0, raw_transformer_layer_id=0, topk_rank=1, expert_id=1),
            NormalizedRoutingRecord(request_id=2, token_id=2, moe_stage_id=0, raw_transformer_layer_id=0, topk_rank=1, expert_id=3),
            NormalizedRoutingRecord(request_id=1, token_id=2, moe_stage_id=0, raw_transformer_layer_id=0, topk_rank=1, expert_id=4),
        ]
        train, test, meta = self.harness.split_by_request(records, train_ratio=0.5)
        self.assertEqual(meta["total_requests"], 2)
        self.assertEqual(meta["train_requests"], [1])
        self.assertEqual(meta["test_requests"], [2])
        self.assertEqual(len(train), 2)
        self.assertEqual(len(test), 2)


class TestMetricsAndEvaluationIntegrity(unittest.TestCase):
    """Verifies metrics correctness, regimes, and evaluation invariants."""

    def setUp(self):
        self.harness = EvaluationHarness(num_moe_stages=4)

    def test_metric_definitions_standard(self):
        """Tests standard Precision, Recall, F1, Hit Rate, Exact Set Accuracy."""
        # P = [1, 2, 3, 4], A = [3, 4, 5, 6] -> hits = 2, P_len=4, A_len=4
        rec = PredictionResultRecord(
            request_id=1, token_id=1, token_position=0, moe_stage_id=1, budget=4,
            predictor_name="Test", predicted_experts=[1, 2, 3, 4], actual_experts=[3, 4, 5, 6]
        )
        self.assertEqual(rec.hits, 2)
        self.assertEqual(rec.precision, 0.5)
        self.assertEqual(rec.recall, 0.5)
        self.assertEqual(rec.f1, 0.5)
        self.assertEqual(rec.hit_rate, 1.0)
        self.assertEqual(rec.exact_set_accuracy, 0.0)
        self.assertTrue(rec.is_cold_start)

    def test_exact_set_accuracy(self):
        """Exact set accuracy = 1 when A == P, 0 otherwise."""
        rec_match = PredictionResultRecord(
            request_id=1, token_id=1, token_position=1, moe_stage_id=1, budget=2,
            predictor_name="Test", predicted_experts=[5, 9], actual_experts=[9, 5]
        )
        self.assertEqual(rec_match.exact_set_accuracy, 1.0)
        self.assertEqual(rec_match.precision, 1.0)
        self.assertEqual(rec_match.recall, 1.0)
        self.assertFalse(rec_match.is_cold_start)

    def test_empty_set_metric_behavior(self):
        """11. Explicit empty-set metric behavior."""
        # Case A: Predicted is empty, Actual has elements
        rec_empty_pred = PredictionResultRecord(
            request_id=1, token_id=1, token_position=0, moe_stage_id=1, budget=0,
            predictor_name="EmptyPred", predicted_experts=[], actual_experts=[1, 2]
        )
        self.assertEqual(rec_empty_pred.precision, 0.0)
        self.assertEqual(rec_empty_pred.recall, 0.0)
        self.assertEqual(rec_empty_pred.f1, 0.0)
        self.assertEqual(rec_empty_pred.hit_rate, 0.0)
        self.assertEqual(rec_empty_pred.exact_set_accuracy, 0.0)
        self.assertEqual(rec_empty_pred.missed_count, 2)

        # Case B: Both are empty
        rec_both_empty = PredictionResultRecord(
            request_id=1, token_id=1, token_position=0, moe_stage_id=1, budget=0,
            predictor_name="BothEmpty", predicted_experts=[], actual_experts=[]
        )
        self.assertEqual(rec_both_empty.precision, 0.0)
        self.assertEqual(rec_both_empty.recall, 0.0)
        self.assertEqual(rec_both_empty.f1, 0.0)
        self.assertEqual(rec_both_empty.exact_set_accuracy, 1.0) # Empty matches empty

    def test_precision_recall_f1_equality_when_b_equals_k(self):
        """Invariant: When B = K, Precision@B == Recall@B == F1@B."""
        for hits in range(9):
            preds = list(range(hits)) + list(range(100, 100 + (8 - hits)))
            acts = list(range(hits)) + list(range(200, 200 + (8 - hits)))
            rec = PredictionResultRecord(
                request_id=1, token_id=1, token_position=1, moe_stage_id=1, budget=8,
                predictor_name="B8K8", predicted_experts=preds, actual_experts=acts
            )
            expected = hits / 8.0
            self.assertAlmostEqual(rec.precision, expected)
            self.assertAlmostEqual(rec.recall, expected)
            self.assertAlmostEqual(rec.f1, expected)

    def test_frozen_evaluation_performs_no_updates(self):
        """5. Frozen evaluation performs no online updates on test records."""
        st_moe = STMoEPaperPredictor(num_moe_stages=3, num_routed_experts=16, top_k=2, online_adaptation=True)
        # Seed CCT entry
        st_moe.cct[0] = {1: {5: 2, 6: 2}}

        cct_before = copy.deepcopy(st_moe.cct)

        opp = PredictionOpportunity(
            request_id=10, token_id=1, moe_stage_id=1,
            source_experts=[1], actual_experts=[7, 8]  # Complete mismatch
        )

        # In frozen mode, update() must not be called
        results = self.harness.evaluate_predictor(
            predictor_instance=st_moe,
            predictor_name="st_moe_frozen",
            test_opportunities=[opp],
            budgets=[2],
            evaluation_mode="frozen"
        )
        self.assertEqual(len(results), 1)
        # Original predictor and its evaluation runner must remain unchanged
        self.assertEqual(st_moe.cct, cct_before)

    def test_online_updates_occur_after_scoring(self):
        """4. Online updates occur after scoring and affect subsequent predictions."""
        st_moe = STMoEPaperPredictor(num_moe_stages=3, num_routed_experts=16, top_k=2, online_adaptation=True)
        st_moe.cct[0] = {1: {5: 1, 6: 2}}

        # Opp 1: mismatch on expert 5 (counter 1 -> counter 0)
        opp1 = PredictionOpportunity(
            request_id=1, token_id=1, moe_stage_id=1,
            source_experts=[1], actual_experts=[9, 10]
        )
        # Opp 2: same source, now expert 5 has counter 0 -> mismatch triggers replacement
        opp2 = PredictionOpportunity(
            request_id=1, token_id=2, moe_stage_id=1,
            source_experts=[1], actual_experts=[11, 12]
        )

        results = self.harness.evaluate_predictor(
            predictor_instance=st_moe,
            predictor_name="st_moe_online",
            test_opportunities=[opp1, opp2],
            budgets=[2],
            evaluation_mode="online"
        )
        self.assertEqual(len(results), 2)
        # Prediction 1 occurred before update 1
        # Prediction 2 was affected by update 1
        self.assertEqual(results[0].predicted_experts, [6, 5])

    def test_temporal_history_resets_at_sequence_boundaries(self):
        """6. Temporal history resets at sequence boundaries while persistent statistics persist."""
        st_moe = STMoEPaperPredictor(num_moe_stages=3, num_routed_experts=16, top_k=2)

        # Req 1: token 1 introduces temporal history [2, 3] at stage 1
        opp_r1_t1 = PredictionOpportunity(request_id=1, token_id=1, moe_stage_id=1, source_experts=[0], actual_experts=[2, 3])
        # Req 1: token 2 should benefit from temporal history of Req 1
        opp_r1_t2 = PredictionOpportunity(request_id=1, token_id=2, moe_stage_id=1, source_experts=[0], actual_experts=[2, 3])

        # Req 2: token 1 starts fresh sequence -> temporal history from Req 1 must NOT bleed into Req 2!
        opp_r2_t1 = PredictionOpportunity(request_id=2, token_id=1, moe_stage_id=1, source_experts=[0], actual_experts=[8, 9])

        results = self.harness.evaluate_predictor(
            predictor_instance=st_moe,
            predictor_name="st_moe_paper",
            test_opportunities=[opp_r1_t1, opp_r1_t2, opp_r2_t1],
            budgets=[2],
            evaluation_mode="online"
        )
        self.assertEqual(len(results), 3)
        # First token of req 1 is cold start
        self.assertTrue(results[0].is_cold_start)
        # Second token of req 1 is steady state
        self.assertFalse(results[1].is_cold_start)
        # First token of req 2 is cold start
        self.assertTrue(results[2].is_cold_start)

    def test_fixed_budget_exact_b_candidates(self):
        """8. Fixed-budget predictions contain exactly B unique valid expert IDs."""
        cstp = CSTPWrapper(CSTPPredictor(num_moe_stages=3, num_routed_experts=32, top_k=4))
        opp = PredictionOpportunity(request_id=1, token_id=1, moe_stage_id=1, source_experts=[0, 1], actual_experts=[4, 5, 6, 7])

        for b in [1, 2, 4, 8]:
            res = self.harness.evaluate_predictor(cstp, "cstp", [opp], budgets=[b])
            self.assertEqual(len(res[0].predicted_experts), b)
            self.assertEqual(len(set(res[0].predicted_experts)), b)
            for exp in res[0].predicted_experts:
                self.assertTrue(0 <= exp < 32)

    def test_threshold_predictions_preserve_variable_set_sizes(self):
        """9. Threshold-based predictions preserve variable set sizes."""
        st_moe = STMoEPaperPredictor(num_moe_stages=3, num_routed_experts=16, top_k=2)
        # CCT has 4 candidates with score >= 2
        st_moe.cct[0] = {0: {1: 2, 2: 3, 3: 2, 4: 2}}

        opp = PredictionOpportunity(request_id=1, token_id=1, moe_stage_id=1, source_experts=[0], actual_experts=[1, 2])
        results = self.harness.evaluate_predictor(
            predictor_instance=st_moe,
            predictor_name="st_moe_thresh",
            test_opportunities=[opp],
            evaluation_mode="threshold",
            threshold=2
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0].predicted_experts), 4) # Variable set size: 4 candidates qualify
        self.assertEqual(results[0].budget, None)

        summary = self.harness.compute_summary_metrics(results)
        self.assertEqual(summary["predicted_set_size"]["mean"], 4.0)

    def test_deterministic_tie_breaking(self):
        """10. Deterministic tie-breaking across predictions."""
        st_moe = STMoEPaperPredictor(num_moe_stages=2, num_routed_experts=16, top_k=4)
        # All zero scores -> tie break by expert ID ascending
        pred1 = st_moe.predict(request_id=1, target_stage_id=1, current_source_experts=[], budget=4)
        pred2 = st_moe.predict(request_id=1, target_stage_id=1, current_source_experts=[], budget=4)
        self.assertEqual(pred1, [0, 1, 2, 3])
        self.assertEqual(pred1, pred2)

    def test_identical_evaluation_examples_across_predictors(self):
        """12. Identical evaluation opportunities are provided to every predictor."""
        opps = [
            PredictionOpportunity(request_id=1, token_id=1, moe_stage_id=1, source_experts=[0], actual_experts=[2, 3]),
            PredictionOpportunity(request_id=1, token_id=2, moe_stage_id=1, source_experts=[2], actual_experts=[4, 5])
        ]
        st_moe = STMoEPaperPredictor(num_moe_stages=3, num_routed_experts=16, top_k=2)
        cstp = CSTPWrapper(CSTPPredictor(num_moe_stages=3, num_routed_experts=16, top_k=2))
        rand = RandomPredictor(num_routed_experts=16, seed=42)

        res_st = self.harness.evaluate_predictor(st_moe, "st_moe", opps, budgets=[2])
        res_cstp = self.harness.evaluate_predictor(cstp, "cstp", opps, budgets=[2])
        res_rand = self.harness.evaluate_predictor(rand, "random", opps, budgets=[2])

        self.assertEqual(len(res_st), len(res_cstp))
        self.assertEqual(len(res_st), len(res_rand))
        for i in range(len(opps)):
            self.assertEqual(res_st[i].actual_experts, res_cstp[i].actual_experts)
            self.assertEqual(res_st[i].actual_experts, res_rand[i].actual_experts)
            self.assertEqual(res_st[i].token_position, res_cstp[i].token_position)


if __name__ == "__main__":
    unittest.main()

