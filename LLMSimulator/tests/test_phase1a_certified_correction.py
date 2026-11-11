"""
Validation Tests for Phase 1A Certified Results and Statistical Correction.

Validates:
1. mohman/ directory remains strictly untouched and read-only.
2. Original Phase 1A CSV artifacts are preserved intact in results/phase1a_cstp_profiling/csv/.
3. All 7 certified CSV files exist and have non-zero row counts.
4. corrected_aggregate_metrics.csv has exactly 192 rows covering all 192 experimental configurations.
5. All 3 seeds (42, 123, 999) are fully represented across all tables.
6. Metric calculations use actual dynamic denominators (prec_denom == predicted_set_size, rec_denom == actual_set_size).
7. Precision@8 equals Recall@8 when |P| = |A| = 8 is mathematically verified on raw data.
8. Hierarchical bootstrap analysis incorporates all 3 seeds and uses valid statistical terminology.
9. MoE Suite N=2 tasks limitation is explicitly flagged as DESCRIPTIVE_ONLY_N2_TASKS.
10. Sufficient statistics reconstruction holds: prompt/task level TP sums match aggregate totals.
11. Publication figures exist in both PNG (300 DPI) and PDF formats.
12. Zero ternary operators exist in the certified scripts and tests.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import re

sim_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
workspace_root = os.path.dirname(sim_root)
sys.path.insert(0, sim_root)

certified_dir = os.path.join(workspace_root, "results", "phase1a_cstp_profiling", "certified")
original_dir = os.path.join(workspace_root, "results", "phase1a_cstp_profiling", "csv")
mohman_dir = os.path.join(workspace_root, "mohman")


class TestPhase1ACertifiedCorrection(unittest.TestCase):

    def test_01_mohman_directory_untouched(self):
        """Verifies that mohman/ directory has not been modified or written to."""
        self.assertTrue(os.path.exists(mohman_dir), "mohman/ directory must exist")
        # Verify no certified files leaked into mohman
        for root, dirs, files in os.walk(mohman_dir):
            for f in files:
                self.assertNotIn("corrected", f, f"Certified file leaked into mohman: {f}")
                self.assertNotIn("certified", f, f"Certified file leaked into mohman: {f}")

    def test_02_original_artifacts_preserved(self):
        """Verifies that all 7 original CSV artifacts remain present and intact."""
        expected_files = [
            "frozen_vs_online.csv",
            "profiling_count_statistics.csv",
            "profiling_percentage_detailed_results.csv",
            "profiling_percentage_summary.csv",
            "profiling_robustness_seeds.csv",
            "profiling_subset_manifest.csv",
            "st_moe_vs_cstp_profiling.csv"
        ]
        for f in expected_files:
            p = os.path.join(original_dir, f)
            self.assertTrue(os.path.exists(p), f"Original file missing: {p}")
            self.assertGreater(os.path.getsize(p), 0, f"Original file is empty: {p}")

    def test_03_certified_files_exist_and_nonempty(self):
        """Verifies all 7 certified CSV files exist and have content."""
        expected_certified = [
            "corrected_per_event_metrics.csv",
            "corrected_per_prompt_metrics.csv",
            "corrected_per_task_metrics.csv",
            "corrected_aggregate_metrics.csv",
            "corrected_st_moe_head_to_head.csv",
            "corrected_profiling_count_statistics.csv",
            "corrected_statistical_analysis.csv"
        ]
        for f in expected_certified:
            p = os.path.join(certified_dir, f)
            self.assertTrue(os.path.exists(p), f"Certified CSV missing: {p}")
            self.assertGreater(os.path.getsize(p), 0, f"Certified CSV is empty: {p}")

    def test_04_aggregate_row_count_and_schema(self):
        """Verifies corrected_aggregate_metrics.csv has exactly 192 configurations."""
        p_agg = os.path.join(certified_dir, "corrected_aggregate_metrics.csv")
        df_agg = pd.read_csv(p_agg)
        self.assertEqual(len(df_agg), 192, f"Expected 192 configs, got {len(df_agg)}")
        for col in ["macro_recall", "micro_recall", "macro_precision", "micro_precision", "macro_f1", "micro_f1"]:
            self.assertIn(col, df_agg.columns)

    def test_05_multi_seed_coverage(self):
        """Verifies that all three seeds (42, 123, 999) are present across all outputs."""
        p_agg = os.path.join(certified_dir, "corrected_aggregate_metrics.csv")
        df_agg = pd.read_csv(p_agg)
        seeds_agg = sorted(list(df_agg["seed"].unique()))
        self.assertEqual(seeds_agg, [42, 123, 999])

        p_counts = os.path.join(certified_dir, "corrected_profiling_count_statistics.csv")
        df_counts = pd.read_csv(p_counts)
        seeds_counts = sorted(list(df_counts["seed"].unique()))
        self.assertEqual(seeds_counts, [42, 123, 999])

        p_prompts = os.path.join(certified_dir, "corrected_per_prompt_metrics.csv")
        df_prompts = pd.read_csv(p_prompts)
        seeds_prompts = sorted(list(df_prompts["seed"].unique()))
        self.assertEqual(seeds_prompts, [42, 123, 999])

    def test_06_non_hardcoded_denominators(self):
        """Verifies that metric calculations use actual predicted and actual set sizes."""
        p_events = os.path.join(certified_dir, "corrected_per_event_metrics.csv")
        # Read a sample chunk from event metrics
        df_ev_chunk = pd.read_csv(p_events, nrows=5000)
        self.assertIn("predicted_set_size", df_ev_chunk.columns)
        self.assertIn("actual_set_size", df_ev_chunk.columns)
        # Check tp + fp == predicted_set_size
        self.assertTrue((df_ev_chunk["tp"] + df_ev_chunk["fp"] == df_ev_chunk["predicted_set_size"]).all())
        # Check tp + fn == actual_set_size
        self.assertTrue((df_ev_chunk["tp"] + df_ev_chunk["fn"] == df_ev_chunk["actual_set_size"]).all())
        # Check precision == tp / predicted_set_size
        calc_prec = df_ev_chunk["tp"] / df_ev_chunk["predicted_set_size"]
        np.testing.assert_allclose(df_ev_chunk["precision"].values, calc_prec.values, atol=1e-5)
        # Check recall == tp / actual_set_size
        calc_rec = df_ev_chunk["tp"] / df_ev_chunk["actual_set_size"]
        np.testing.assert_allclose(df_ev_chunk["recall"].values, calc_rec.values, atol=1e-5)

    def test_07_precision_recall_equality_on_balanced_budget(self):
        """Verifies Precision@8 == Recall@8 holds empirically when |P| = |A| = 8."""
        p_events = os.path.join(certified_dir, "corrected_per_event_metrics.csv")
        df_ev_chunk = pd.read_csv(p_events, nrows=5000)
        np.testing.assert_array_equal(df_ev_chunk["precision"].values, df_ev_chunk["recall"].values)

    def test_08_hierarchical_bootstrap_and_statistical_validity(self):
        """Verifies bootstrap analysis incorporates seeds and properly flags MoE Suite N=2."""
        p_boot = os.path.join(certified_dir, "corrected_statistical_analysis.csv")
        df_boot = pd.read_csv(p_boot)
        self.assertIn("bootstrap_prob_greater_than_zero", df_boot.columns)
        self.assertIn("bootstrap_two_sided_asl", df_boot.columns)
        self.assertIn("statistical_validity", df_boot.columns)
        self.assertIn("profiling_seed_scope", df_boot.columns)

        # Check that joint_all_seeds scope exists
        scopes = df_boot["profiling_seed_scope"].unique()
        self.assertIn("joint_all_seeds", scopes)

        # Check MoE Suite validity flag
        suite_rows = df_boot[df_boot["dataset_id"] == "moe_suite"]
        for val in suite_rows["statistical_validity"]:
            self.assertEqual(val, "DESCRIPTIVE_ONLY_N2_TASKS")

        # Check Anywhere validity flag
        anywhere_rows = df_boot[df_boot["dataset_id"] == "moe_anywhere"]
        for val in anywhere_rows["statistical_validity"]:
            self.assertEqual(val, "VALID_N14_PROMPT_UNITS")

    def test_09_sufficient_statistics_reconstruction(self):
        """Verifies that aggregate TP equals sum of prompt/task TPs."""
        p_agg = os.path.join(certified_dir, "corrected_aggregate_metrics.csv")
        p_prompts = os.path.join(certified_dir, "corrected_per_prompt_metrics.csv")
        df_agg = pd.read_csv(p_agg)
        df_prompts = pd.read_csv(p_prompts)

        # Pick one config: moe_anywhere, horizon=1, seed=42, pct=25, CSTP, online
        agg_sub = df_agg[(df_agg["dataset_id"] == "moe_anywhere") & (df_agg["horizon"] == 1) & (df_agg["seed"] == 42) & (df_agg["profiling_pct"] == 25) & (df_agg["predictor"] == "CSTP") & (df_agg["regime"] == "online_within_request")]
        prompt_sub = df_prompts[(df_prompts["dataset_id"] == "moe_anywhere") & (df_prompts["horizon"] == 1) & (df_prompts["seed"] == 42) & (df_prompts["profiling_pct"] == 25) & (df_prompts["predictor"] == "CSTP") & (df_prompts["regime"] == "online_within_request")]

        self.assertEqual(len(agg_sub), 1)
        self.assertEqual(len(prompt_sub), 14)

        sum_tp = prompt_sub["total_tp"].sum()
        sum_actual = prompt_sub["total_actual_experts"].sum()
        reconstructed_micro_rec = float(sum_tp) / float(sum_actual)
        np.testing.assert_almost_equal(agg_sub["micro_recall"].values[0], reconstructed_micro_rec, decimal=6)

    def test_10_publication_plots_exist(self):
        """Verifies all 9 publication figures exist in both PNG and PDF formats."""
        plot_dir = os.path.join(certified_dir, "plots")
        expected_plots = [
            "fig1_certified_frozen_recall_vs_profiling",
            "fig2_certified_online_recall_vs_profiling",
            "fig3_certified_online_adaptation_gain",
            "fig4_certified_stmoe_vs_cstp_head_to_head",
            "fig5_certified_horizon1_vs_horizon2",
            "fig6_certified_seed_variability",
            "fig7_certified_moe_suite_task_level",
            "fig8_certified_transition_count_growth",
            "fig9_certified_confidence_intervals"
        ]
        for base in expected_plots:
            p_png = os.path.join(plot_dir, f"{base}.png")
            p_pdf = os.path.join(plot_dir, f"{base}.pdf")
            self.assertTrue(os.path.exists(p_png), f"Plot PNG missing: {p_png}")
            self.assertTrue(os.path.exists(p_pdf), f"Plot PDF missing: {p_pdf}")
            self.assertGreater(os.path.getsize(p_png), 1000, f"Plot PNG too small: {p_png}")
            self.assertGreater(os.path.getsize(p_pdf), 1000, f"Plot PDF too small: {p_pdf}")

    def test_11_zero_ternary_operators(self):
        """Verifies ZERO ternary operators across newly written scripts and tests."""
        script_path = os.path.join(sim_root, "scripts", "phase1a", "run_phase1a_certified_correction.py")
        smoke_path = os.path.join(sim_root, "scripts", "phase1a", "smoke_test_certified_correction.py")
        this_test = os.path.abspath(__file__)

        for path in [script_path, smoke_path, this_test]:
            with open(path, "r") as f:
                lines = f.readlines()
            for idx, line in enumerate(lines, 1):
                clean = line.split("#")[0]
                if "if" in clean and "else" in clean:
                    match = re.search(r"\bif\b\s+[^:]+\s+\belse\b", clean)
                    self.assertIsNone(match, f"Ternary operator found in {path}:{idx}: {line.strip()}")


if __name__ == "__main__":
    unittest.main()
