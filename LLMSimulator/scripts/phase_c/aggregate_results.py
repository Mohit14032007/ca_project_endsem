"""
Phase C Result Aggregation and Cross-Dataset Analysis.
Combines all benchmark metrics across the 5 target workloads into unified tables
and computes adaptation gains, ablation comparisons, and Pareto tradeoffs.
"""

import os
import sys
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


def main():
    base_res_dir = os.path.join(project_root, "results", "endsem_phase_c")
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
        sweep_path = os.path.join(base_res_dir, d, "budget_sweep.csv")
        if os.path.exists(sweep_path):
            df_d = pd.read_csv(sweep_path)
            all_sweep_list.append(df_d)
        else:
            print(f"Warning: Budget sweep not found for {d} at {sweep_path}")

    if len(all_sweep_list) == 0:
        print("No budget sweep results found to aggregate.")
        return

    df_combined_sweep = pd.concat(all_sweep_list, ignore_index=True)
    combined_sweep_path = os.path.join(out_dir, "all_datasets_budget_sweep.csv")
    df_combined_sweep.to_csv(combined_sweep_path, index=False)
    print(f"Saved combined budget sweep to: {combined_sweep_path}")

    # 1. Primary budget cross-dataset aggregate table
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
    primary_agg_path = os.path.join(out_dir, "all_datasets_aggregate.csv")
    df_primary.to_csv(primary_agg_path, index=False)
    print(f"Saved all datasets aggregate to: {primary_agg_path}")

    # 2. Online adaptation gains: Online vs Frozen for CSTP and ST-MoE
    adaptation_rows = []
    for d in datasets:
        d_sub = df_combined_sweep[df_combined_sweep["dataset_id"] == d]
        horizons = d_sub["prediction_horizon"].unique()
        budgets = d_sub["budget"].unique()

        for h in horizons:
            for b in budgets:
                # CSTP
                cstp_on = d_sub[
                    (d_sub["prediction_horizon"] == h) &
                    (d_sub["budget"] == b) &
                    (d_sub["predictor"] == "cstp_online")
                ]
                cstp_fr = d_sub[
                    (d_sub["prediction_horizon"] == h) &
                    (d_sub["budget"] == b) &
                    (d_sub["predictor"] == "cstp_frozen")
                ]
                if len(cstp_on) > 0 and len(cstp_fr) > 0:
                    r_on = cstp_on.iloc[0]["macro_recall"]
                    r_fr = cstp_fr.iloc[0]["macro_recall"]
                    delta_r = r_on - r_fr
                    pct_gain = 0.0
                    if r_fr > 0.0:
                        pct_gain = (delta_r / r_fr) * 100.0

                    adaptation_rows.append({
                        "dataset_id": d,
                        "architecture": "CSTP",
                        "horizon": h,
                        "budget": b,
                        "online_recall": r_on,
                        "frozen_recall": r_fr,
                        "delta_recall": round(delta_r, 4),
                        "pct_gain_recall": round(pct_gain, 2),
                        "online_prec": cstp_on.iloc[0]["macro_precision"],
                        "frozen_prec": cstp_fr.iloc[0]["macro_precision"],
                        "delta_prec": round(cstp_on.iloc[0]["macro_precision"] - cstp_fr.iloc[0]["macro_precision"], 4),
                        "online_f1": cstp_on.iloc[0]["macro_f1"],
                        "frozen_f1": cstp_fr.iloc[0]["macro_f1"],
                        "delta_f1": round(cstp_on.iloc[0]["macro_f1"] - cstp_fr.iloc[0]["macro_f1"], 4),
                        "online_hit_rate": cstp_on.iloc[0]["hit_rate"],
                        "frozen_hit_rate": cstp_fr.iloc[0]["hit_rate"],
                        "delta_hit_rate": round(cstp_on.iloc[0]["hit_rate"] - cstp_fr.iloc[0]["hit_rate"], 4)
                    })

                # ST-MoE Paper
                st_on = d_sub[
                    (d_sub["prediction_horizon"] == h) &
                    (d_sub["budget"] == b) &
                    (d_sub["predictor"] == "st_moe_paper_online")
                ]
                st_fr = d_sub[
                    (d_sub["prediction_horizon"] == h) &
                    (d_sub["budget"] == b) &
                    (d_sub["predictor"] == "st_moe_paper_frozen")
                ]
                if len(st_on) > 0 and len(st_fr) > 0:
                    r_on = st_on.iloc[0]["macro_recall"]
                    r_fr = st_fr.iloc[0]["macro_recall"]
                    delta_r = r_on - r_fr
                    pct_gain = 0.0
                    if r_fr > 0.0:
                        pct_gain = (delta_r / r_fr) * 100.0

                    adaptation_rows.append({
                        "dataset_id": d,
                        "architecture": "ST-MoE-Paper",
                        "horizon": h,
                        "budget": b,
                        "online_recall": r_on,
                        "frozen_recall": r_fr,
                        "delta_recall": round(delta_r, 4),
                        "pct_gain_recall": round(pct_gain, 2),
                        "online_prec": st_on.iloc[0]["macro_precision"],
                        "frozen_prec": st_fr.iloc[0]["macro_precision"],
                        "delta_prec": round(st_on.iloc[0]["macro_precision"] - st_fr.iloc[0]["macro_precision"], 4),
                        "online_f1": st_on.iloc[0]["macro_f1"],
                        "frozen_f1": st_fr.iloc[0]["macro_f1"],
                        "delta_f1": round(st_on.iloc[0]["macro_f1"] - st_fr.iloc[0]["macro_f1"], 4),
                        "online_hit_rate": st_on.iloc[0]["hit_rate"],
                        "frozen_hit_rate": st_fr.iloc[0]["hit_rate"],
                        "delta_hit_rate": round(st_on.iloc[0]["hit_rate"] - st_fr.iloc[0]["hit_rate"], 4)
                    })

    df_adapt = pd.DataFrame(adaptation_rows)
    adapt_path = os.path.join(out_dir, "online_adaptation_gains.csv")
    df_adapt.to_csv(adapt_path, index=False)
    print(f"Saved online adaptation gains to: {adapt_path}")

    # 3. Ablation summary at primary budget and horizon
    ablation_rows = []
    for d in datasets:
        target_b = primary_budgets[d]
        target_h = primary_horizons[d]
        d_sub = df_combined_sweep[
            (df_combined_sweep["dataset_id"] == d) &
            (df_combined_sweep["budget"] == target_b) &
            (df_combined_sweep["prediction_horizon"] == target_h)
        ]
        cstp_row = d_sub[d_sub["predictor"] == "cstp_online"]
        base_recall = 0.0
        if len(cstp_row) > 0:
            base_recall = cstp_row.iloc[0]["macro_recall"]

        for _, row in d_sub.iterrows():
            p_name = row["predictor"]
            rec = row["macro_recall"]
            delta = rec - base_recall
            ablation_rows.append({
                "dataset_id": d,
                "horizon": target_h,
                "budget": target_b,
                "predictor": p_name,
                "learning_mode": row["learning_mode"],
                "recall": rec,
                "delta_vs_cstp_online": round(delta, 4),
                "precision": row["macro_precision"],
                "f1": row["macro_f1"],
                "hit_rate": row["hit_rate"]
            })

    df_ablation = pd.DataFrame(ablation_rows)
    ablation_path = os.path.join(out_dir, "ablation_summary.csv")
    df_ablation.to_csv(ablation_path, index=False)
    print(f"Saved ablation summary to: {ablation_path}")

    # 4. Multi-horizon comparison for MoE Anywhere
    anywhere_sub = df_combined_sweep[df_combined_sweep["dataset_id"] == "moe_anywhere"]
    if len(anywhere_sub) > 0:
        h1_sub = anywhere_sub[anywhere_sub["prediction_horizon"] == 1]
        h2_sub = anywhere_sub[anywhere_sub["prediction_horizon"] == 2]
        horizon_rows = []
        for p in anywhere_sub["predictor"].unique():
            for b in anywhere_sub["budget"].unique():
                r1_match = h1_sub[(h1_sub["predictor"] == p) & (h1_sub["budget"] == b)]
                r2_match = h2_sub[(h2_sub["predictor"] == p) & (h2_sub["budget"] == b)]
                if len(r1_match) > 0 and len(r2_match) > 0:
                    rec_h1 = r1_match.iloc[0]["macro_recall"]
                    rec_h2 = r2_match.iloc[0]["macro_recall"]
                    horizon_rows.append({
                        "dataset_id": "moe_anywhere",
                        "predictor": p,
                        "budget": b,
                        "recall_h1": rec_h1,
                        "recall_h2": rec_h2,
                        "delta_h2_minus_h1": round(rec_h2 - rec_h1, 4),
                        "prec_h1": r1_match.iloc[0]["macro_precision"],
                        "prec_h2": r2_match.iloc[0]["macro_precision"],
                        "f1_h1": r1_match.iloc[0]["macro_f1"],
                        "f1_h2": r2_match.iloc[0]["macro_f1"],
                        "hit_rate_h1": r1_match.iloc[0]["hit_rate"],
                        "hit_rate_h2": r2_match.iloc[0]["hit_rate"]
                    })
        df_horizon = pd.DataFrame(horizon_rows)
        horizon_path = os.path.join(out_dir, "multi_horizon_comparison.csv")
        df_horizon.to_csv(horizon_path, index=False)
        print(f"Saved multi-horizon comparison to: {horizon_path}")

    print("\nAggregation complete!")


if __name__ == "__main__":
    main()

