"""
Phase D Master Result Aggregation and Cross-Dataset Statistical Analysis.
Combines benchmark metrics across all 5 target workloads into unified tables,
computes predictor rankings, ACSH gains over baselines, paired statistical tests,
and online adaptation dynamics.
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Any, Dict, List

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


def run_aggregation():
    print("=" * 80)
    print("STARTING PHASE D RESULT AGGREGATION & STATISTICAL ANALYSIS")
    print("=" * 80)

    base_res_dir = os.path.join(project_root, "results", "endsem_phase_d")
    per_dataset_dir = os.path.join(base_res_dir, "per_dataset")
    out_dir = os.path.join(base_res_dir, "aggregate")
    os.makedirs(out_dir, exist_ok=True)

    datasets = [
        "deepseek_random",
        "duplex_a100",
        "correlated_synthetic",
        "moe_anywhere",
        "moe_suite"
    ]

    primary_budgets = {
        "deepseek_random": 8,
        "duplex_a100": 2,
        "correlated_synthetic": 8,
        "moe_anywhere": 8,
        "moe_suite": 8
    }

    primary_horizons = {
        "deepseek_random": 1,
        "duplex_a100": 1,
        "correlated_synthetic": 1,
        "moe_anywhere": 1,
        "moe_suite": 2
    }

    all_sweep_list = []
    for d in datasets:
        sweep_path = os.path.join(per_dataset_dir, d, "budget_sweep.csv")
        if os.path.exists(sweep_path):
            df_d = pd.read_csv(sweep_path)
            all_sweep_list.append(df_d)
        else:
            print(f"Warning: budget_sweep.csv not found for {d}")

    if len(all_sweep_list) == 0:
        print("No budget sweep results found to aggregate!")
        return

    # 1. Full Budget Sweep Table
    df_combined_sweep = pd.concat(all_sweep_list, ignore_index=True)
    sweep_csv = os.path.join(out_dir, "all_datasets_budget_sweep.csv")
    df_combined_sweep.to_csv(sweep_csv, index=False)
    print(f"Saved full budget sweep to: {sweep_csv}")

    # 2. Primary Budget Cross-Dataset Aggregate Table
    primary_rows = []
    for d in datasets:
        target_b = primary_budgets[d]
        target_h = primary_horizons[d]
        sub = df_combined_sweep[
            (df_combined_sweep["dataset_id"] == d) &
            (df_combined_sweep["budget"] == target_b) &
            (df_combined_sweep["prediction_horizon"] == target_h)
        ]
        for _, row in sub.iterrows():
            primary_rows.append(row.to_dict())

    df_primary = pd.DataFrame(primary_rows)
    primary_csv = os.path.join(out_dir, "all_datasets_aggregate.csv")
    df_primary.to_csv(primary_csv, index=False)
    print(f"Saved primary aggregate to: {primary_csv}")

    # 3. Predictor Rankings across Datasets
    ranking_rows = []
    for metric in ["macro_recall", "macro_precision", "macro_f1", "hit_rate"]:
        # Compute ranks per dataset (higher metric = rank 1)
        rank_df = df_primary.copy()
        rank_df["rank"] = rank_df.groupby("dataset_id")[metric].rank(ascending=False, method="min")
        mean_ranks = rank_df.groupby("predictor")["rank"].mean().reset_index()
        mean_metrics = rank_df.groupby("predictor")[metric].mean().reset_index()
        merged = pd.merge(mean_ranks, mean_metrics, on="predictor")
        merged["metric"] = metric
        for _, r in merged.iterrows():
            ranking_rows.append({
                "predictor": r["predictor"],
                "metric": metric,
                "mean_rank": round(float(r["rank"]), 2),
                "mean_value": round(float(r[metric]), 4)
            })

    df_rankings = pd.DataFrame(ranking_rows)
    rank_csv = os.path.join(out_dir, "predictor_rankings.csv")
    df_rankings.to_csv(rank_csv, index=False)
    print(f"Saved predictor rankings to: {rank_csv}")

    # 4. ACSH Improvement Over Baselines
    improvement_rows = []
    acsh_variants = ["acsh_global", "acsh_stage_aware", "acsh_fast_recovery"]
    baselines = ["cstp_online", "st_moe_paper_online", "cstp_st_moe_hybrid"]

    for d in datasets:
        target_b = primary_budgets[d]
        target_h = primary_horizons[d]
        d_sub = df_primary[
            (df_primary["dataset_id"] == d) &
            (df_primary["budget"] == target_b) &
            (df_primary["prediction_horizon"] == target_h)
        ]
        metric_map = {}
        for _, r in d_sub.iterrows():
            metric_map[r["predictor"]] = r["macro_recall"]

        for acsh_v in acsh_variants:
            if acsh_v not in metric_map:
                continue
            acsh_rec = metric_map[acsh_v]
            for base in baselines:
                if base not in metric_map:
                    continue
                base_rec = metric_map[base]
                abs_diff = acsh_rec - base_rec
                if base_rec > 1e-6:
                    rel_gain_pct = (abs_diff / base_rec) * 100.0
                else:
                    rel_gain_pct = 0.0

                improvement_rows.append({
                    "dataset_id": d,
                    "budget": target_b,
                    "horizon": target_h,
                    "acsh_variant": acsh_v,
                    "baseline": base,
                    "acsh_recall": round(acsh_rec, 4),
                    "baseline_recall": round(base_rec, 4),
                    "absolute_gain": round(abs_diff, 4),
                    "relative_gain_pct": round(rel_gain_pct, 2)
                })

    df_impr = pd.DataFrame(improvement_rows)
    impr_csv = os.path.join(out_dir, "acsh_improvement_over_baselines.csv")
    df_impr.to_csv(impr_csv, index=False)
    print(f"Saved ACSH improvements to: {impr_csv}")

    # 5. Statistical Paired Tests (ACSH Stage-Aware vs Baselines across all predictions)
    print("\nRunning statistical significance tests across per-prediction samples...")
    stat_rows = []
    for d in datasets:
        per_pred_path = os.path.join(per_dataset_dir, d, "per_prediction_results.csv")
        if not os.path.exists(per_pred_path):
            continue
        df_p = pd.read_csv(per_pred_path)
        target_b = primary_budgets[d]
        target_h = primary_horizons[d]

        df_p_sub = df_p[
            (df_p["budget"] == target_b) &
            (df_p["prediction_horizon"] == target_h)
        ]

        acsh_recalls = df_p_sub[df_p_sub["predictor"] == "acsh_stage_aware"]["recall"].values
        if len(acsh_recalls) == 0:
            continue

        for base in ["cstp_online", "st_moe_paper_online", "cstp_st_moe_hybrid"]:
            base_recalls = df_p_sub[df_p_sub["predictor"] == base]["recall"].values
            n_samples = min(len(acsh_recalls), len(base_recalls))
            if n_samples < 5:
                continue

            a_samp = acsh_recalls[:n_samples]
            b_samp = base_recalls[:n_samples]
            diff = a_samp - b_samp
            mean_diff = float(np.mean(diff))

            # Paired t-test
            t_res = stats.ttest_rel(a_samp, b_samp)
            t_stat = float(t_res.statistic)
            p_val = float(t_res.pvalue)

            # 95% Confidence Interval for mean difference
            se = float(stats.sem(diff))
            ci_half = 1.96 * se
            ci_lower = mean_diff - ci_half
            ci_upper = mean_diff + ci_half

            stat_rows.append({
                "dataset_id": d,
                "comparison": f"acsh_stage_aware vs {base}",
                "num_samples": n_samples,
                "mean_difference": round(mean_diff, 4),
                "ci_95_lower": round(ci_lower, 4),
                "ci_95_upper": round(ci_upper, 4),
                "t_statistic": round(t_stat, 4),
                "p_value": p_val,
                "statistically_significant": bool(p_val < 0.05)
            })

    df_stat = pd.DataFrame(stat_rows)
    stat_csv = os.path.join(out_dir, "statistical_comparison.csv")
    df_stat.to_csv(stat_csv, index=False)
    print(f"Saved statistical comparison to: {stat_csv}")

    # 6. Online Adaptation Gains
    adapt_rows = []
    for d in datasets:
        target_b = primary_budgets[d]
        target_h = primary_horizons[d]
        sub = df_primary[df_primary["dataset_id"] == d]
        p_map = {r["predictor"]: r["macro_recall"] for _, r in sub.iterrows()}

        for family, on_name, fr_name in [
            ("CSTP", "cstp_online", "cstp_frozen"),
            ("ST-MoE Paper", "st_moe_paper_online", "st_moe_paper_frozen")
        ]:
            if on_name in p_map and fr_name in p_map:
                on_val = p_map[on_name]
                fr_val = p_map[fr_name]
                delta = on_val - fr_val
                if fr_val > 1e-6:
                    rel_pct = (delta / fr_val) * 100.0
                else:
                    rel_pct = 0.0
                adapt_rows.append({
                    "dataset_id": d,
                    "predictor_family": family,
                    "online_recall": round(on_val, 4),
                    "frozen_recall": round(fr_val, 4),
                    "adaptation_delta": round(delta, 4),
                    "relative_gain_pct": round(rel_pct, 2)
                })

    df_adapt = pd.DataFrame(adapt_rows)
    adapt_csv = os.path.join(out_dir, "online_learning_gains.csv")
    df_adapt.to_csv(adapt_csv, index=False)
    print(f"Saved online learning gains to: {adapt_csv}")

    print("\n" + "=" * 80)
    print("PHASE D AGGREGATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_aggregation()

