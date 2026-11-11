"""
Phase C Publication-Quality Figure Generation.
Generates comprehensive visual comparisons across all 5 datasets, 10 predictors,
and experimental regimes (Frozen, Online, Cold-Start, Shift, Overheads).
Uses matplotlib exclusively (no seaborn).
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


def configure_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "figure.autolayout": True
    })


def save_fig(fig, out_dir: str, name: str):
    png_path = os.path.join(out_dir, f"{name}.png")
    pdf_path = os.path.join(out_dir, f"{name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {name}.png and .pdf")


def main():
    configure_plot_style()
    base_res_dir = os.path.join(project_root, "results", "endsem_phase_c")
    out_dir = os.path.join(base_res_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    agg_file = os.path.join(base_res_dir, "aggregate", "all_datasets_aggregate.csv")
    sweep_file = os.path.join(base_res_dir, "aggregate", "all_datasets_budget_sweep.csv")
    gains_file = os.path.join(base_res_dir, "aggregate", "online_adaptation_gains.csv")
    ablation_file = os.path.join(base_res_dir, "aggregate", "ablation_summary.csv")
    lat_file = os.path.join(base_res_dir, "metadata", "predictor_latency.csv")
    upd_file = os.path.join(base_res_dir, "metadata", "predictor_update_latency.csv")
    mem_file = os.path.join(base_res_dir, "metadata", "predictor_memory.csv")
    shift_file = os.path.join(base_res_dir, "distribution_shift", "distribution_shift_metrics.csv")

    # Verify if aggregated files exist
    if not os.path.exists(agg_file) or not os.path.exists(sweep_file):
        print("Aggregate files not ready yet. Please run aggregate_results.py first.")
        return

    df_agg = pd.read_csv(agg_file)
    df_sweep = pd.read_csv(sweep_file)

    # ----------------------------------------------------
    # Figure 1: Overall Recall@B Comparison Across Datasets
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    target_predictors = [
        ("cstp_online", "CSTP Online", "#1f77b4", "-"),
        ("cstp_frozen", "CSTP Frozen", "#aec7e8", "--"),
        ("st_moe_paper_online", "ST-MoE Online", "#ff7f0e", "-"),
        ("st_moe_paper_frozen", "ST-MoE Frozen", "#ffbb78", "--"),
        ("frequency_only", "Frequency Only", "#2ca02c", ":"),
        ("random", "Random", "#7f7f7f", "-.")
    ]

    datasets = ["deepseek_random", "duplex_a100", "correlated_synthetic", "moe_anywhere", "moe_suite"]
    dataset_names = ["DeepSeek Random", "Duplex A100", "Correlated Synth", "MoE Anywhere", "MoE Suite"]
    x = np.arange(len(datasets))
    width = 0.13

    for idx, (p_id, p_label, col, _) in enumerate(target_predictors):
        vals = []
        for d in datasets:
            match = df_agg[(df_agg["dataset_id"] == d) & (df_agg["predictor"] == p_id)]
            if len(match) > 0:
                vals.append(match.iloc[0]["macro_recall"])
            else:
                vals.append(0.0)
        ax.bar(x + idx * width - (len(target_predictors) * width / 2.0), vals, width, label=p_label, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.set_ylabel("Macro Recall @ Primary Budget")
    ax.set_title("Figure 1: MoE Expert Prediction Recall Across 5 Evaluation Workloads")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    save_fig(fig, out_dir, "fig1_overall_recall_comparison")

    # ----------------------------------------------------
    # Figure 2: Precision@B Comparison
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, (p_id, p_label, col, _) in enumerate(target_predictors):
        vals = []
        for d in datasets:
            match = df_agg[(df_agg["dataset_id"] == d) & (df_agg["predictor"] == p_id)]
            if len(match) > 0:
                vals.append(match.iloc[0]["macro_precision"])
            else:
                vals.append(0.0)
        ax.bar(x + idx * width - (len(target_predictors) * width / 2.0), vals, width, label=p_label, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.set_ylabel("Macro Precision @ Primary Budget")
    ax.set_title("Figure 2: MoE Expert Prediction Precision Across 5 Evaluation Workloads")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    save_fig(fig, out_dir, "fig2_overall_precision_comparison")

    # ----------------------------------------------------
    # Figure 3: F1-Score@B Comparison
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, (p_id, p_label, col, _) in enumerate(target_predictors):
        vals = []
        for d in datasets:
            match = df_agg[(df_agg["dataset_id"] == d) & (df_agg["predictor"] == p_id)]
            if len(match) > 0:
                vals.append(match.iloc[0]["macro_f1"])
            else:
                vals.append(0.0)
        ax.bar(x + idx * width - (len(target_predictors) * width / 2.0), vals, width, label=p_label, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.set_ylabel("Macro F1-Score @ Primary Budget")
    ax.set_title("Figure 3: MoE Expert Prediction F1-Score Across 5 Evaluation Workloads")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    save_fig(fig, out_dir, "fig3_overall_f1_comparison")

    # ----------------------------------------------------
    # Figure 4: Hit Rate Comparison
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, (p_id, p_label, col, _) in enumerate(target_predictors):
        vals = []
        for d in datasets:
            match = df_agg[(df_agg["dataset_id"] == d) & (df_agg["predictor"] == p_id)]
            if len(match) > 0:
                vals.append(match.iloc[0]["hit_rate"])
            else:
                vals.append(0.0)
        ax.bar(x + idx * width - (len(target_predictors) * width / 2.0), vals, width, label=p_label, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.set_ylabel("Hit Rate (Fraction with >= 1 Hit)")
    ax.set_title("Figure 4: Prediction Hit Rate Across 5 Evaluation Workloads")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    save_fig(fig, out_dir, "fig4_overall_hit_rate_comparison")

    # ----------------------------------------------------
    # Figure 5 & 6: Online Learning Curves & Cold-Start Adaptation
    # ----------------------------------------------------
    # Load per-prediction results for correlated synthetic
    corr_per_pred_path = os.path.join(base_res_dir, "correlated_synthetic", "per_prediction_results.csv")
    if os.path.exists(corr_per_pred_path):
        df_corr = pd.read_csv(corr_per_pred_path)
        # Filter for primary budget B=8
        b8_sub = df_corr[df_corr["budget"] == 8]

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        curve_preds = [
            ("cstp_online", "CSTP Online", "#1f77b4", "-"),
            ("cstp_frozen", "CSTP Frozen", "#aec7e8", "--"),
            ("st_moe_paper_online", "ST-MoE Online", "#ff7f0e", "-"),
            ("st_moe_paper_frozen", "ST-MoE Frozen", "#ffbb78", "--"),
            ("cstp_cold_start", "CSTP Cold-Start", "#2ca02c", "-."),
            ("st_moe_paper_cold_start", "ST-MoE Cold-Start", "#d62728", "-.")
        ]
        for p_name, label, col, ls in curve_preds:
            p_sub = b8_sub[b8_sub["predictor"] == p_name]
            if len(p_sub) > 0:
                grp = p_sub.groupby("token_id")["recall"].mean().reset_index()
                ax.plot(grp["token_id"], grp["recall"], label=label, color=col, linestyle=ls, linewidth=2, marker="o")

        ax.set_xlabel("Token Position within Request Sequence")
        ax.set_ylabel("Mean Recall @ B=8")
        ax.set_title("Figure 5 & 6: Causal Online Adaptation & Cold-Start Convergence (Correlated Synthetic)")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right")
        save_fig(fig, out_dir, "fig5_fig6_online_learning_and_cold_start")

    # ----------------------------------------------------
    # Figure 7: Controlled Distribution Shift Recovery
    # ----------------------------------------------------
    if os.path.exists(shift_file):
        df_shift = pd.read_csv(shift_file)
        fig, ax = plt.subplots(figsize=(9, 5))

        shift_preds = [
            ("cstp_online", "CSTP Online (Adaptive)", "#1f77b4", "-"),
            ("cstp_frozen", "CSTP Frozen (Non-Adaptive)", "#aec7e8", "--"),
            ("st_moe_paper_online", "ST-MoE Online (Adaptive)", "#ff7f0e", "-"),
            ("st_moe_paper_frozen", "ST-MoE Frozen (Non-Adaptive)", "#ffbb78", "--")
        ]

        for p_name, label, col, ls in shift_preds:
            p_data = df_shift[df_shift["predictor"] == p_name].sort_values("token_id")
            if len(p_data) > 0:
                roll_rec = p_data["mean_recall"].rolling(5, min_periods=1).mean()
                ax.plot(p_data["token_id"], roll_rec, label=label, color=col, linestyle=ls, linewidth=2)

        ax.axvline(x=50, color="red", linestyle=":", linewidth=2, label="Distribution Shift (Cluster A -> B)")
        ax.set_xlabel("Token Sequence Step")
        ax.set_ylabel("Rolling Recall (5-Token Window)")
        ax.set_title("Figure 7: Predictor Resilience & Adaptation Under Distribution Shift")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper right")
        save_fig(fig, out_dir, "fig7_distribution_shift_recovery")

    # ----------------------------------------------------
    # Figure 8: Prediction Latency Breakdown
    # ----------------------------------------------------
    if os.path.exists(lat_file):
        df_lat = pd.read_csv(lat_file)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        lat_b8 = df_lat[df_lat["budget"] == 8]
        if len(lat_b8) == 0:
            lat_b8 = df_lat.iloc[:6]

        x_lat = np.arange(len(lat_b8))
        width_l = 0.25
        ax.bar(x_lat - width_l, lat_b8["mean_us"], width_l, label="Mean", color="#1f77b4")
        ax.bar(x_lat, lat_b8["p95_us"], width_l, label="P95", color="#ff7f0e")
        ax.bar(x_lat + width_l, lat_b8["p99_us"], width_l, label="P99", color="#d62728")

        ax.set_xticks(x_lat)
        ax.set_xticklabels(lat_b8["predictor"], rotation=20, ha="right")
        ax.set_ylabel("Inference Prediction Latency (microseconds)")
        ax.set_title("Figure 8: Predictor Query Latency Overhead (Budget B=8)")
        ax.legend()
        save_fig(fig, out_dir, "fig8_prediction_latency_breakdown")

    # ----------------------------------------------------
    # Figure 9: Update Latency Overhead
    # ----------------------------------------------------
    if os.path.exists(upd_file):
        df_upd = pd.read_csv(upd_file)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x_upd = np.arange(len(df_upd))
        ax.bar(x_upd, df_upd["mean_us"], color="#2ca02c", width=0.45)
        ax.set_xticks(x_upd)
        ax.set_xticklabels(df_upd["predictor"], rotation=15, ha="right")
        ax.set_ylabel("Online Update Latency (microseconds)")
        ax.set_title("Figure 9: Online Causal State Update Latency per MoE Stage")
        save_fig(fig, out_dir, "fig9_update_latency_overhead")

    # ----------------------------------------------------
    # Figure 10: Memory Footprint Comparison
    # ----------------------------------------------------
    if os.path.exists(mem_file):
        df_mem = pd.read_csv(mem_file)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x_mem = np.arange(len(df_mem))
        mem_mb = df_mem["allocated_kb"] / 1024.0
        ax.bar(x_mem, mem_mb, color="#9467bd", width=0.45)
        ax.set_xticks(x_mem)
        ax.set_xticklabels(df_mem["predictor"], rotation=20, ha="right")
        ax.set_ylabel("Memory Footprint (Megabytes)")
        ax.set_title("Figure 10: Predictor State Memory Consumption (E=256, S=57 Stages)")
        save_fig(fig, out_dir, "fig10_memory_footprint_comparison")

    # ----------------------------------------------------
    # Figure 14: Precision vs Recall Trade-off Curves Across Budgets
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    corr_sweep = df_sweep[df_sweep["dataset_id"] == "correlated_synthetic"]
    any_sweep = df_sweep[df_sweep["dataset_id"] == "moe_anywhere"]

    key_p = ["cstp_online", "cstp_frozen", "st_moe_paper_online", "st_moe_paper_frozen", "frequency_only"]
    cols = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c"]

    for p, c in zip(key_p, cols):
        sub1 = corr_sweep[corr_sweep["predictor"] == p].sort_values("budget")
        if len(sub1) > 0:
            ax1.plot(sub1["macro_recall"], sub1["macro_precision"], marker="o", label=p, color=c, linewidth=2)

        sub2 = any_sweep[(any_sweep["predictor"] == p) & (any_sweep["prediction_horizon"] == 1)].sort_values("budget")
        if len(sub2) > 0:
            ax2.plot(sub2["macro_recall"], sub2["macro_precision"], marker="s", label=p, color=c, linewidth=2)

    ax1.set_xlabel("Macro Recall")
    ax1.set_ylabel("Macro Precision")
    ax1.set_title("Synthetic Benchmark (Sweep B=4 to 64)")
    ax1.legend()

    ax2.set_xlabel("Macro Recall")
    ax2.set_ylabel("Macro Precision")
    ax2.set_title("MoE Anywhere Real Trace (Sweep B=4 to 64)")
    ax2.legend()

    fig.suptitle("Figure 14: Precision vs. Recall Trade-Off Across Prefetch Budgets")
    save_fig(fig, out_dir, "fig14_precision_vs_recall_tradeoff")

    # ----------------------------------------------------
    # Figure 15: Quality vs Latency Pareto Frontier
    # ----------------------------------------------------
    if os.path.exists(lat_file):
        df_lat = pd.read_csv(lat_file)
        fig, ax = plt.subplots(figsize=(8.5, 5))

        # Correlated synthetic primary budget
        corr_agg = df_agg[df_agg["dataset_id"] == "correlated_synthetic"]
        pareto_data = [
            ("cstp_online", "cstp_online", "#1f77b4", "o"),
            ("st_moe_paper_online", "st_moe_paper_online", "#ff7f0e", "s"),
            ("frequency_only", "frequency_only", "#2ca02c", "^"),
            ("random", "random", "#7f7f7f", "x")
        ]

        for p_pred, p_lat_key, col, mk in pareto_data:
            rec_row = corr_agg[corr_agg["predictor"] == p_pred]
            lat_row = df_lat[(df_lat["predictor"] == p_lat_key) & (df_lat["budget"] == 8)]
            if len(rec_row) > 0 and len(lat_row) > 0:
                r_val = rec_row.iloc[0]["macro_recall"]
                l_val = lat_row.iloc[0]["mean_us"]
                ax.scatter(l_val, r_val, color=col, marker=mk, s=150, label=f"{p_lat_key} (Rec={r_val:.3f}, Lat={l_val:.1f}us)")

        ax.set_xlabel("Prediction Latency (microseconds)")
        ax.set_ylabel("Macro Recall @ Budget=8")
        ax.set_title("Figure 15: Quality vs. Latency Trade-Off Frontier")
        ax.legend(loc="lower right")
        save_fig(fig, out_dir, "fig15_quality_vs_latency_frontier")

    # ----------------------------------------------------
    # Figure 16: Online Adaptation Gains
    # ----------------------------------------------------
    if os.path.exists(gains_file):
        df_gains = pd.read_csv(gains_file)
        fig, ax = plt.subplots(figsize=(9, 4.8))

        # Filter for primary budget B=8 (or B=2 for duplex)
        gains_b8 = df_gains[(df_gains["budget"] == 8) | (df_gains["dataset_id"] == "duplex_a100")]
        cstp_gains = gains_b8[gains_b8["architecture"] == "CSTP"]
        st_gains = gains_b8[gains_b8["architecture"] == "ST-MoE-Paper"]

        d_list = ["deepseek_random", "duplex_a100", "correlated_synthetic", "moe_anywhere", "moe_suite"]
        d_labels = ["DeepSeek", "Duplex", "Synthetic", "Anywhere", "MoE Suite"]
        x_g = np.arange(len(d_list))
        w_g = 0.35

        cstp_vals = []
        st_vals = []
        for d in d_list:
            c_row = cstp_gains[cstp_gains["dataset_id"] == d]
            s_row = st_gains[st_gains["dataset_id"] == d]
            if len(c_row) > 0:
                cstp_vals.append(c_row.iloc[0]["delta_recall"])
            else:
                cstp_vals.append(0.0)
            if len(s_row) > 0:
                st_vals.append(s_row.iloc[0]["delta_recall"])
            else:
                st_vals.append(0.0)

        ax.bar(x_g - w_g / 2.0, cstp_vals, w_g, label="CSTP Delta Recall", color="#1f77b4")
        ax.bar(x_g + w_g / 2.0, st_vals, w_g, label="ST-MoE Paper Delta Recall", color="#ff7f0e")

        ax.set_xticks(x_g)
        ax.set_xticklabels(d_labels)
        ax.set_ylabel("Recall Gain (Online - Frozen)")
        ax.set_title("Figure 16: Online Causal Adaptation Recall Benefits Across Datasets")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.legend()
        save_fig(fig, out_dir, "fig16_online_adaptation_gains")

    # ----------------------------------------------------
    # Figure 17: Ablation Breakdown
    # ----------------------------------------------------
    if os.path.exists(ablation_file):
        df_abl = pd.read_csv(ablation_file)
        fig, ax = plt.subplots(figsize=(9, 4.8))

        abl_preds = ["cstp_online", "spatial_only", "temporal_only", "frequency_only", "random"]
        abl_labels = ["Full CSTP", "Spatial Only", "Temporal Only", "Frequency Only", "Random"]
        synth_abl = df_abl[df_abl["dataset_id"] == "correlated_synthetic"]
        any_abl = df_abl[df_abl["dataset_id"] == "moe_anywhere"]

        x_a = np.arange(len(abl_preds))
        w_a = 0.35

        s_vals = []
        a_vals = []
        for p in abl_preds:
            s_row = synth_abl[synth_abl["predictor"] == p]
            a_row = any_abl[any_abl["predictor"] == p]
            if len(s_row) > 0:
                s_vals.append(s_row.iloc[0]["recall"])
            else:
                s_vals.append(0.0)
            if len(a_row) > 0:
                a_vals.append(a_row.iloc[0]["recall"])
            else:
                a_vals.append(0.0)

        ax.bar(x_a - w_a / 2.0, s_vals, w_a, label="Correlated Synthetic", color="#1f77b4")
        ax.bar(x_a + w_a / 2.0, a_vals, w_a, label="MoE Anywhere Real Trace", color="#2ca02c")

        ax.set_xticks(x_a)
        ax.set_xticklabels(abl_labels, rotation=15, ha="right")
        ax.set_ylabel("Macro Recall @ Budget=8")
        ax.set_title("Figure 17: Predictor Ablation Study (Spatio-Temporal Deconstruction)")
        ax.legend()
        save_fig(fig, out_dir, "fig17_ablation_comparison")

    # ----------------------------------------------------
    # Figure 18: Multi-Horizon Predictability (H=1 vs H=2)
    # ----------------------------------------------------
    horiz_file = os.path.join(base_res_dir, "aggregate", "multi_horizon_comparison.csv")
    if os.path.exists(horiz_file):
        df_horiz = pd.read_csv(horiz_file)
        fig, ax = plt.subplots(figsize=(9, 4.8))

        b8_horiz = df_horiz[df_horiz["budget"] == 8]
        preds_h = ["cstp_online", "cstp_frozen", "st_moe_paper_online", "st_moe_paper_frozen", "frequency_only"]
        labels_h = ["CSTP Online", "CSTP Frozen", "ST-MoE Online", "ST-MoE Frozen", "Frequency"]

        x_h = np.arange(len(preds_h))
        w_h = 0.35

        h1_vals = []
        h2_vals = []
        for p in preds_h:
            h_row = b8_horiz[b8_horiz["predictor"] == p]
            if len(h_row) > 0:
                h1_vals.append(h_row.iloc[0]["recall_h1"])
                h2_vals.append(h_row.iloc[0]["recall_h2"])
            else:
                h1_vals.append(0.0)
                h2_vals.append(0.0)

        ax.bar(x_h - w_h / 2.0, h1_vals, w_h, label="Horizon H=1 (Adjacent Stage)", color="#1f77b4")
        ax.bar(x_h + w_h / 2.0, h2_vals, w_h, label="Horizon H=2 (Skip Stage)", color="#e377c2")

        ax.set_xticks(x_h)
        ax.set_xticklabels(labels_h, rotation=15, ha="right")
        ax.set_ylabel("Macro Recall @ Budget=8")
        ax.set_title("Figure 18: Multi-Horizon Prediction Decay on MoE Anywhere (H=1 vs H=2)")
        ax.legend()
        save_fig(fig, out_dir, "fig18_multi_horizon_decay")

    print("\nFigure generation complete!")


if __name__ == "__main__":
    main()
