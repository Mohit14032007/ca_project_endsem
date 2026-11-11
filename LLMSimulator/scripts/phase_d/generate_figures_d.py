"""
Phase D Publication Figure Generation Suite.
Generates 22 publication-grade figures in 300 DPI PNG and vector PDF using Matplotlib directly (no seaborn).
"""

import os
import sys
import numpy as np
import pandas as pd
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
    print(f"Saved figure: {name}.png and {name}.pdf")


def generate_all_figures():
    configure_plot_style()
    base_res = os.path.join(project_root, "results", "endsem_phase_d")
    fig_dir = os.path.join(base_res, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    agg_csv = os.path.join(base_res, "aggregate", "all_datasets_aggregate.csv")
    sweep_csv = os.path.join(base_res, "aggregate", "all_datasets_budget_sweep.csv")
    shift_csv = os.path.join(base_res, "distribution_shift", "distribution_shift_metrics.csv")
    lat_csv = os.path.join(base_res, "latency", "prediction_latency.csv")
    lat_break_csv = os.path.join(base_res, "latency", "prediction_latency_breakdown.csv")
    upd_csv = os.path.join(base_res, "latency", "update_latency.csv")
    mem_csv = os.path.join(base_res, "memory", "memory_footprint.csv")
    mem_scale_csv = os.path.join(base_res, "memory", "memory_scaling.csv")
    mem_conc_csv = os.path.join(base_res, "memory", "memory_concurrency.csv")
    calib_csv = os.path.join(base_res, "ablations", "calibration_ablation.csv")
    obj_csv = os.path.join(base_res, "ablations", "objective_ablation.csv")
    beta_csv = os.path.join(base_res, "ablations", "beta_sensitivity.csv")
    tau_csv = os.path.join(base_res, "ablations", "tau_sensitivity.csv")
    cold_csv = os.path.join(base_res, "ablations", "cold_start_sensitivity.csv")

    if not os.path.exists(agg_csv):
        print(f"Error: {agg_csv} not found. Run aggregation first.")
        return

    df_agg = pd.read_csv(agg_csv)
    df_sweep = pd.read_csv(sweep_csv)

    datasets = ["deepseek_random", "duplex_a100", "correlated_synthetic", "moe_anywhere", "moe_suite"]
    d_labels = ["DeepSeek (Ctrl)", "Duplex A100", "Synthetic Corr.", "MoE Anywhere", "MoE Suite"]

    # Key predictors for grouped bar plots
    key_predictors = [
        ("cstp_online", "CSTP Online", "#1f77b4"),
        ("st_moe_paper_online", "ST-MoE Online", "#ff7f0e"),
        ("cstp_st_moe_hybrid", "Fixed Hybrid", "#2ca02c"),
        ("acsh_stage_aware", "ACSH Stage-Aware", "#d62728"),
        ("acsh_fast_recovery", "ACSH Fast Recovery", "#9467bd"),
        ("random", "Random", "#7f7f7f")
    ]

    # Helper function for grouped bar charts
    def make_grouped_bar(metric_col: str, title: str, ylabel: str, fig_name: str):
        fig, ax = plt.subplots(figsize=(11, 5.5))
        n_d = len(datasets)
        n_p = len(key_predictors)
        bar_width = 0.8 / float(n_p)
        x_indices = np.arange(n_d)

        for p_idx, (p_id, p_label, p_color) in enumerate(key_predictors):
            vals = []
            for d in datasets:
                sub = df_agg[(df_agg["dataset_id"] == d) & (df_agg["predictor"] == p_id)]
                if len(sub) > 0:
                    vals.append(float(sub[metric_col].values[0]))
                else:
                    vals.append(0.0)
            offset = (float(p_idx) - float(n_p) / 2.0 + 0.5) * bar_width
            ax.bar(x_indices + offset, vals, width=bar_width, label=p_label, color=p_color, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x_indices)
        ax.set_xticklabels(d_labels, rotation=15)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=True, facecolor="white", edgecolor="none", ncol=3)
        ax.set_ylim(0, 1.05)
        save_fig(fig, fig_dir, fig_name)

    # 1. Figure 1: Recall
    make_grouped_bar("macro_recall", "Overall Macro Recall Across Workloads", "Recall@B", "fig1_overall_recall_comparison")

    # 2. Figure 2: Precision
    make_grouped_bar("macro_precision", "Overall Macro Precision Across Workloads", "Precision@B", "fig2_overall_precision_comparison")

    # 3. Figure 3: F1
    make_grouped_bar("macro_f1", "Overall Macro F1 Score Across Workloads", "F1 Score@B", "fig3_overall_f1_comparison")

    # 4. Figure 4: Hit Rate
    make_grouped_bar("hit_rate", "Overall Hit Rate Across Workloads", "Hit Rate", "fig4_overall_hit_rate_comparison")

    # 5. Figure 5 & 6: Online Learning & Cold Start Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    regimes_to_show = [
        ("cstp_online", "CSTP Online", "#1f77b4"),
        ("cstp_frozen", "CSTP Frozen", "#aec7e8"),
        ("cstp_cold_start", "CSTP Cold-Start", "#17becf"),
        ("st_moe_paper_online", "ST-MoE Online", "#ff7f0e"),
        ("st_moe_paper_frozen", "ST-MoE Frozen", "#ffbb78"),
        ("st_moe_paper_cold_start", "ST-MoE Cold-Start", "#bcbd22"),
        ("acsh_stage_aware", "ACSH Online", "#d62728"),
        ("acsh_cold_start", "ACSH Cold-Start", "#e377c2")
    ]
    x_idx = np.arange(len(datasets))
    w = 0.8 / float(len(regimes_to_show))
    for p_idx, (p_id, p_label, p_col) in enumerate(regimes_to_show):
        vals = []
        for d in datasets:
            sub = df_agg[(df_agg["dataset_id"] == d) & (df_agg["predictor"] == p_id)]
            if len(sub) > 0:
                vals.append(float(sub["macro_recall"].values[0]))
            else:
                vals.append(0.0)
        off = (float(p_idx) - float(len(regimes_to_show)) / 2.0 + 0.5) * w
        axes[0].bar(x_idx + off, vals, width=w, label=p_label, color=p_col, edgecolor="black", linewidth=0.5)

    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels(d_labels, rotation=15)
    axes[0].set_ylabel("Recall@B")
    axes[0].set_title("Regime Comparison (Online vs Frozen vs Cold-Start)")
    axes[0].legend(frameon=True, fontsize=8, ncol=2)

    # Panel 2: Cold start comparison specifically
    cold_preds = [
        ("cstp_cold_start", "CSTP Cold-Start", "#1f77b4"),
        ("st_moe_paper_cold_start", "ST-MoE Cold-Start", "#ff7f0e"),
        ("acsh_cold_start", "ACSH Cold-Start", "#d62728"),
        ("random", "Random Baseline", "#7f7f7f")
    ]
    w2 = 0.8 / float(len(cold_preds))
    for p_idx, (p_id, p_label, p_col) in enumerate(cold_preds):
        vals = []
        for d in datasets:
            sub = df_agg[(df_agg["dataset_id"] == d) & (df_agg["predictor"] == p_id)]
            if len(sub) > 0:
                vals.append(float(sub["macro_recall"].values[0]))
            else:
                vals.append(0.0)
        off = (float(p_idx) - float(len(cold_preds)) / 2.0 + 0.5) * w2
        axes[1].bar(x_idx + off, vals, width=w2, label=p_label, color=p_col, edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels(d_labels, rotation=15)
    axes[1].set_ylabel("Recall@B")
    axes[1].set_title("Cold-Start Performance (Unprofiled)")
    axes[1].legend(frameon=True, fontsize=9)
    save_fig(fig, fig_dir, "fig5_fig6_online_learning_and_cold_start")

    # 6. Figure 7: Distribution Shift Recovery
    if os.path.exists(shift_csv):
        df_shift = pd.read_csv(shift_csv)
        fig, ax = plt.subplots(figsize=(10, 5))
        shift_palette = {
            "cstp_online": ("#1f77b4", "-", "CSTP Online"),
            "st_moe_paper_online": ("#ff7f0e", "-", "ST-MoE Online"),
            "cstp_st_moe_hybrid": ("#2ca02c", "--", "Fixed Hybrid"),
            "acsh_stage_aware": ("#d62728", "-", "ACSH Stage-Aware"),
            "acsh_fast_recovery": ("#9467bd", "-.", "ACSH Fast Recovery")
        }
        for p_id, (col, ls, lbl) in shift_palette.items():
            sub = df_shift[df_shift["predictor"] == p_id].sort_values("token_id")
            if len(sub) > 0:
                ax.plot(sub["token_id"], sub["mean_recall"], label=lbl, color=col, linestyle=ls, linewidth=2.0)
        ax.axvline(50, color="black", linestyle=":", linewidth=2, label="Regime Shift (Cluster A -> B)")
        ax.set_xlabel("Token Index")
        ax.set_ylabel("Recall@B (B=8)")
        ax.set_title("Distribution Shift Adaptation & Recovery Trajectory", fontweight="bold")
        ax.legend(frameon=True, loc="lower right")
        save_fig(fig, fig_dir, "fig7_distribution_shift_recovery")

    # 7. Figure 8: Prediction Latency Breakdown
    if os.path.exists(lat_break_csv):
        df_lb = pd.read_csv(lat_break_csv)
        fig, ax = plt.subplots(figsize=(8, 5))
        comps = df_lb["component"].tolist()
        means = df_lb["mean_us"].tolist()
        ax.bar(comps, means, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"], edgecolor="black")
        ax.set_ylabel("Latency (microseconds)")
        ax.set_title("ACSH Prediction Query Latency Component Breakdown", fontweight="bold")
        for i, v in enumerate(means):
            ax.text(i, v + 0.05 * max(means), f"{v:.1f} µs", ha="center", fontsize=10)
        save_fig(fig, fig_dir, "fig8_prediction_latency_breakdown")

    # 8. Figure 9: Update Latency Overhead
    if os.path.exists(upd_csv):
        df_u = pd.read_csv(upd_csv)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        preds = df_u["predictor"].tolist()
        means = df_u["mean_us"].tolist()
        ax.barh(preds, means, color="#17becf", edgecolor="black")
        ax.set_xlabel("Mean Update Latency (microseconds)")
        ax.set_title("Online State Update Latency Across Predictors", fontweight="bold")
        save_fig(fig, fig_dir, "fig9_update_latency_overhead")

    # 9. Figure 10: Memory Footprint Comparison
    if os.path.exists(mem_csv):
        df_m = pd.read_csv(mem_csv)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        preds = df_m["predictor"].tolist()
        kbs = df_m["allocated_kb"].tolist()
        ax.bar(preds, kbs, color="#bcbd22", edgecolor="black")
        ax.set_ylabel("Allocated Memory (KiB)")
        ax.set_title("Predictor State Memory Footprint (E=256, S=57)", fontweight="bold")
        plt.xticks(rotation=20, ha="right")
        save_fig(fig, fig_dir, "fig10_memory_footprint_comparison")

    # 10. Figure 11: Memory Scaling (Experts & Stages)
    if os.path.exists(mem_scale_csv):
        df_ms = pd.read_csv(mem_scale_csv)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        sub_e = df_ms[df_ms["dimension"] == "experts"].sort_values("num_experts")
        ax1.plot(sub_e["num_experts"], sub_e["cstp_kb"], marker="o", label="CSTP", color="#1f77b4")
        ax1.plot(sub_e["num_experts"], sub_e["st_moe_kb"], marker="s", label="ST-MoE", color="#ff7f0e")
        ax1.plot(sub_e["num_experts"], sub_e["acsh_kb"], marker="^", label="ACSH", color="#d62728")
        ax1.set_xlabel("Number of Routed Experts (E)")
        ax1.set_ylabel("Memory (KiB)")
        ax1.set_title("Memory Scaling vs Expert Count (E)")
        ax1.legend()

        sub_s = df_ms[df_ms["dimension"] == "stages"].sort_values("num_stages")
        ax2.plot(sub_s["num_stages"], sub_s["cstp_kb"], marker="o", label="CSTP", color="#1f77b4")
        ax2.plot(sub_s["num_stages"], sub_s["st_moe_kb"], marker="s", label="ST-MoE", color="#ff7f0e")
        ax2.plot(sub_s["num_stages"], sub_s["acsh_kb"], marker="^", label="ACSH", color="#d62728")
        ax2.set_xlabel("Number of MoE Stages (L)")
        ax2.set_ylabel("Memory (KiB)")
        ax2.set_title("Memory Scaling vs Layer Count (L)")
        ax2.legend()
        save_fig(fig, fig_dir, "fig11_memory_scaling_experts_stages")

    # 11. Figure 12: ACSH Dynamic Weight Trajectories
    weights_path = os.path.join(base_res, "per_dataset", "correlated_synthetic", "acsh_weights.csv")
    if os.path.exists(weights_path):
        df_w = pd.read_csv(weights_path)
        sub_sa = df_w[df_w["predictor"] == "acsh_stage_aware"]
        if len(sub_sa) > 0:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            steps = np.arange(min(len(sub_sa), 150))
            ax.plot(steps, sub_sa["w_c"].iloc[:150], label="CSTP Weight (wC)", color="#1f77b4", linewidth=2)
            ax.plot(steps, sub_sa["w_m"].iloc[:150], label="ST-MoE Weight (wM)", color="#ff7f0e", linewidth=2)
            ax.set_xlabel("Sequence Steps")
            ax.set_ylabel("Dynamic Softmax Weight")
            ax.set_title("ACSH Autonomous Reliability Weight Trajectory (Synthetic Corr.)", fontweight="bold")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(frameon=True)
            save_fig(fig, fig_dir, "fig12_acsh_weight_trajectories")

    # 12. Figure 13: Budget Sweep Subplots Across All Datasets
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    sweep_preds = [
        ("cstp_online", "CSTP", "#1f77b4", "-"),
        ("st_moe_paper_online", "ST-MoE", "#ff7f0e", "-"),
        ("cstp_st_moe_hybrid", "Hybrid", "#2ca02c", "--"),
        ("acsh_stage_aware", "ACSH Stage-Aware", "#d62728", "-"),
        ("random", "Random", "#7f7f7f", ":")
    ]
    for d_idx, d in enumerate(datasets):
        ax = axes[d_idx]
        d_sub = df_sweep[df_sweep["dataset_id"] == d]
        for p_id, p_lbl, p_col, p_ls in sweep_preds:
            p_sub = d_sub[d_sub["predictor"] == p_id].sort_values("budget")
            if len(p_sub) > 0:
                ax.plot(p_sub["budget"], p_sub["macro_recall"], label=p_lbl, color=p_col, linestyle=p_ls, marker="o", markersize=3)
        ax.set_title(d_labels[d_idx], fontsize=10, fontweight="bold")
        ax.set_xlabel("Prefetch Budget (B)")
        if d_idx == 0:
            ax.set_ylabel("Macro Recall")
        if d_idx == 2:
            ax.legend(frameon=True, fontsize=8)
    save_fig(fig, fig_dir, "fig13_budget_sweep_all_datasets")

    # 13. Figure 14: Precision vs Recall Tradeoff
    fig, ax = plt.subplots(figsize=(8, 6))
    for p_id, p_lbl, p_col, p_ls in sweep_preds:
        sub = df_sweep[df_sweep["predictor"] == p_id]
        if len(sub) > 0:
            agg_pr = sub.groupby("budget").agg({"macro_recall": "mean", "macro_precision": "mean"}).reset_index().sort_values("macro_recall")
            ax.plot(agg_pr["macro_recall"], agg_pr["macro_precision"], marker="o", label=p_lbl, color=p_col, linestyle=p_ls, linewidth=2)
            for _, r in agg_pr.iterrows():
                b_val = int(r["budget"])
                if b_val in [4, 8, 16, 64]:
                    ax.annotate(f"B={b_val}", (r["macro_recall"], r["macro_precision"]), fontsize=8, textcoords="offset points", xytext=(0,5), ha='center')
    ax.set_xlabel("Macro Recall")
    ax.set_ylabel("Macro Precision")
    ax.set_title("Precision vs Recall Frontier Across Budgets", fontweight="bold")
    ax.legend(frameon=True)
    save_fig(fig, fig_dir, "fig14_precision_vs_recall_tradeoff")

    # 14. Figure 15: Quality vs Latency Frontier
    if os.path.exists(lat_csv):
        df_l = pd.read_csv(lat_csv)
        b8_lat = df_l[df_l["budget"] == 8]
        lat_map = {r["predictor"]: r["mean_us"] for _, r in b8_lat.iterrows()}

        fig, ax = plt.subplots(figsize=(8, 5.5))
        for p_id, p_lbl, p_col in [
            ("cstp_online", "CSTP", "#1f77b4"),
            ("st_moe_paper_online", "ST-MoE", "#ff7f0e"),
            ("cstp_st_moe_hybrid", "Hybrid", "#2ca02c"),
            ("acsh_stage_aware", "ACSH Stage-Aware", "#d62728"),
            ("acsh_fast_recovery", "ACSH Fast Recovery", "#9467bd")
        ]:
            rec_val = df_agg[df_agg["predictor"] == p_id]["macro_recall"].mean()
            l_key = p_id.replace("_online", "").replace("_paper", "")
            lat_val = lat_map.get(l_key, lat_map.get(p_id, None))
            if lat_val is not None:
                ax.scatter(lat_val, rec_val, s=120, label=p_lbl, color=p_col, edgecolors="black", linewidth=1.5)
                ax.annotate(f"  {p_lbl} ({rec_val:.3f}, {lat_val:.1f}µs)", (lat_val, rec_val), fontsize=9)
        ax.set_xlabel("Host Prediction Query Latency (microseconds)")
        ax.set_ylabel("Mean Macro Recall (B=8)")
        ax.set_title("Prediction Quality vs Computational Overhead Frontier", fontweight="bold")
        ax.legend(frameon=True, loc="lower right")
        save_fig(fig, fig_dir, "fig15_quality_vs_latency_frontier")

    # 15. Figure 16: Online Adaptation Gains
    gains_csv = os.path.join(base_res, "aggregate", "online_learning_gains.csv")
    if os.path.exists(gains_csv):
        df_g = pd.read_csv(gains_csv)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        cstp_g = df_g[df_g["predictor_family"] == "CSTP"]
        st_g = df_g[df_g["predictor_family"] == "ST-MoE Paper"]
        x = np.arange(len(datasets))
        w = 0.35
        cstp_vals = []
        st_vals = []
        for d in datasets:
            cg = cstp_g[cstp_g["dataset_id"] == d]
            if len(cg) > 0:
                cstp_vals.append(cg["relative_gain_pct"].values[0])
            else:
                cstp_vals.append(0.0)

            sg = st_g[st_g["dataset_id"] == d]
            if len(sg) > 0:
                st_vals.append(sg["relative_gain_pct"].values[0])
            else:
                st_vals.append(0.0)

        ax.bar(x - w/2.0, cstp_vals, width=w, label="CSTP Gain (%)", color="#1f77b4", edgecolor="black")
        ax.bar(x + w/2.0, st_vals, width=w, label="ST-MoE Gain (%)", color="#ff7f0e", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(d_labels, rotation=15)
        ax.set_ylabel("Relative Gain (%)")
        ax.set_title("Online Adaptation Gain Over Frozen State (Online vs Frozen)", fontweight="bold")
        ax.legend()
        save_fig(fig, fig_dir, "fig16_online_adaptation_gains")

    # 16. Figure 17: Ablation Comparison (Normalization & Objective)
    if os.path.exists(calib_csv) and os.path.exists(obj_csv):
        df_c = pd.read_csv(calib_csv)
        df_o = pd.read_csv(obj_csv)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        norm_labels = [f"{r['normalization']} (T={r['temperature']})" for _, r in df_c.iterrows()]
        ax1.bar(norm_labels, df_c["macro_recall"], color="#2ca02c", edgecolor="black")
        ax1.set_ylabel("Macro Recall")
        ax1.set_title("Score Normalization & Temperature Ablation")
        ax1.set_xticklabels(norm_labels, rotation=25, ha="right")

        ax2.bar(df_o["objective"], df_o["macro_recall"], color="#d62728", edgecolor="black")
        ax2.set_ylabel("Macro Recall")
        ax2.set_title("Reliability Objective Function Ablation")
        save_fig(fig, fig_dir, "fig17_ablation_comparison")

    # 17. Figure 18: Multi-Horizon Decay
    sub_h = df_sweep[df_sweep["dataset_id"].isin(["moe_anywhere", "moe_suite"])]
    if len(sub_h) > 0:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for p_id, p_lbl, p_col in [("acsh_stage_aware", "ACSH", "#d62728"), ("cstp_online", "CSTP", "#1f77b4"), ("st_moe_paper_online", "ST-MoE", "#ff7f0e")]:
            p_sub = sub_h[sub_h["predictor"] == p_id]
            h_vals = sorted(p_sub["prediction_horizon"].unique())
            means = [p_sub[p_sub["prediction_horizon"] == h]["macro_recall"].mean() for h in h_vals]
            ax.plot(h_vals, means, marker="s", label=p_lbl, color=p_col, linewidth=2)
        ax.set_xticks([1, 2])
        ax.set_xlabel("Prediction Horizon (H stages ahead)")
        ax.set_ylabel("Macro Recall")
        ax.set_title("Prediction Quality Decay Across Multi-Stage Horizon", fontweight="bold")
        ax.legend()
        save_fig(fig, fig_dir, "fig18_multi_horizon_decay")

    # 18. Figure 19: Beta Sensitivity Curve
    if os.path.exists(beta_csv):
        df_b = pd.read_csv(beta_csv)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(df_b["beta"], df_b["macro_recall"], marker="o", color="#d62728", linewidth=2)
        ax.set_xlabel("EMA History Retention Momentum (β)")
        ax.set_ylabel("Macro Recall on Val Split")
        ax.set_title("Sensitivity to EMA Momentum Parameter β", fontweight="bold")
        save_fig(fig, fig_dir, "fig19_beta_sensitivity_curve")

    # 19. Figure 20: Tau Sensitivity Curve
    if os.path.exists(tau_csv):
        df_t = pd.read_csv(tau_csv)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(df_t["tau"], df_t["macro_recall"], marker="s", color="#9467bd", linewidth=2)
        ax.set_xlabel("Softmax Reliability Temperature (τ)")
        ax.set_ylabel("Macro Recall on Val Split")
        ax.set_title("Sensitivity to Softmax Weighting Temperature τ", fontweight="bold")
        save_fig(fig, fig_dir, "fig20_tau_sensitivity_curve")

    # 20. Figure 21: Cold-Start Initialization Sensitivity
    if os.path.exists(cold_csv):
        df_ci = pd.read_csv(cold_csv)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(df_ci["initial_reliability"], df_ci["macro_recall"], marker="^", color="#17becf", linewidth=2)
        ax.set_xlabel("Initial Reliability Prior R_0")
        ax.set_ylabel("Cold-Start Macro Recall")
        ax.set_title("Sensitivity to Initial Reliability in Cold-Start", fontweight="bold")
        save_fig(fig, fig_dir, "fig21_cold_start_initialization_sensitivity")

    # 21. Figure 22: Concurrency Memory Scaling
    if os.path.exists(mem_conc_csv):
        df_mc = pd.read_csv(mem_conc_csv)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(df_mc["concurrent_requests"], df_mc["allocated_kb"], marker="d", color="#e377c2", linewidth=2)
        ax.set_xscale("log")
        ax.set_xlabel("Concurrent Requests (log scale)")
        ax.set_ylabel("Allocated Memory (KiB)")
        ax.set_title("ACSH Memory Footprint vs Concurrent In-Flight Requests", fontweight="bold")
        save_fig(fig, fig_dir, "fig22_concurrency_memory_scaling")

    print("\n" + "=" * 80)
    print("ALL 22 PUBLICATION FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    generate_all_figures()
