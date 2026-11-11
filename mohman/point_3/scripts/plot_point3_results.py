#!/usr/bin/env python3
"""
Point 3 Plotting Script
Generates publication-quality plots illustrating trends across:
1. Batch size (B = 1, 2, 4, 8)
2. Prediction horizon (H = 1 vs H = 2)
3. Online learning progress (Tokens 0, 1, 2, 3)
4. Head-to-head comparison between CSTP, ST-MoE, and Baseline

Enforces zero ternary operators.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = "/home/mohman/ca_project"
POINT3_ROOT = os.path.join(WORKSPACE, "mohman", "point_3")
RESULTS_DIR = os.path.join(POINT3_ROOT, "results")
PLOTS_DIR = os.path.join(POINT3_ROOT, "plots")

PER_RUN_CSV = os.path.join(RESULTS_DIR, "per_run_metrics.csv")
HORIZON_CSV = os.path.join(RESULTS_DIR, "horizon_trends.csv")
LEARNING_CSV = os.path.join(RESULTS_DIR, "learning_progress_trends.csv")

# Styling configurations
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "grid.alpha": 0.4,
    "grid.linestyle": "--"
})

CONFIG_COLORS = {
    "Baseline": "#4B5563",         # Slate Gray
    "ST-MoE (H=1)": "#EF4444",     # Red
    "ST-MoE (H=2)": "#F87171",     # Light Red
    "CSTP (H=1)": "#2563EB",       # Blue
    "CSTP (H=2)": "#60A5FA",       # Light Blue
}

CONFIG_MARKERS = {
    "Baseline": "o",
    "ST-MoE (H=1)": "s",
    "ST-MoE (H=2)": "^",
    "CSTP (H=1)": "D",
    "CSTP (H=2)": "v",
}


def load_data():
    if not os.path.exists(PER_RUN_CSV):
        raise FileNotFoundError(f"Missing per_run_metrics.csv at {PER_RUN_CSV}")
    df_runs = pd.read_csv(PER_RUN_CSV)

    df_horizon = None
    if os.path.exists(HORIZON_CSV):
        df_horizon = pd.read_csv(HORIZON_CSV)

    df_learning = None
    if os.path.exists(LEARNING_CSV):
        df_learning = pd.read_csv(LEARNING_CSV)

    return df_runs, df_horizon, df_learning


def get_series(df, predictor, horizon):
    sub = df[(df["predictor"] == predictor) & (df["horizon"] == horizon)].sort_values("batch_size")
    return sub


def plot_metric_vs_batch(df_runs, metric_col, ylabel, title, output_filename, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # 1. Baseline
    sub_base = df_runs[df_runs["predictor"] == "baseline"].sort_values("batch_size")
    ax.plot(
        sub_base["batch_size"], sub_base[metric_col],
        label="Baseline", color=CONFIG_COLORS["Baseline"],
        marker=CONFIG_MARKERS["Baseline"], linewidth=2.2, markersize=8
    )

    # 2. ST-MoE H=1
    sub_st1 = get_series(df_runs, "stmoe", 1)
    ax.plot(
        sub_st1["batch_size"], sub_st1[metric_col],
        label="ST-MoE (H=1)", color=CONFIG_COLORS["ST-MoE (H=1)"],
        marker=CONFIG_MARKERS["ST-MoE (H=1)"], linewidth=2.0, markersize=7
    )

    # 3. ST-MoE H=2
    sub_st2 = get_series(df_runs, "stmoe", 2)
    ax.plot(
        sub_st2["batch_size"], sub_st2[metric_col],
        label="ST-MoE (H=2)", color=CONFIG_COLORS["ST-MoE (H=2)"],
        marker=CONFIG_MARKERS["ST-MoE (H=2)"], linewidth=1.8, markersize=7, linestyle="--"
    )

    # 4. CSTP H=1
    sub_cs1 = get_series(df_runs, "cstp", 1)
    ax.plot(
        sub_cs1["batch_size"], sub_cs1[metric_col],
        label="CSTP (H=1)", color=CONFIG_COLORS["CSTP (H=1)"],
        marker=CONFIG_MARKERS["CSTP (H=1)"], linewidth=2.2, markersize=8
    )

    # 5. CSTP H=2
    sub_cs2 = get_series(df_runs, "cstp", 2)
    ax.plot(
        sub_cs2["batch_size"], sub_cs2[metric_col],
        label="CSTP (H=2)", color=CONFIG_COLORS["CSTP (H=2)"],
        marker=CONFIG_MARKERS["CSTP (H=2)"], linewidth=1.8, markersize=7, linestyle="--"
    )

    ax.set_xlabel("Batch Size (Concurrent Sequences)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["B=1", "B=2", "B=4", "B=8"])
    ax.grid(True)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(frameon=True, loc="best")

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, output_filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_timely_and_late_rates(df_runs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Timely Rate
    for pred, h, label in [
        ("stmoe", 1, "ST-MoE (H=1)"),
        ("stmoe", 2, "ST-MoE (H=2)"),
        ("cstp", 1, "CSTP (H=1)"),
        ("cstp", 2, "CSTP (H=2)"),
    ]:
        sub = get_series(df_runs, pred, h)
        ax1.plot(
            sub["batch_size"], sub["timely_rate_of_matched_pct"],
            label=label, color=CONFIG_COLORS[label],
            marker=CONFIG_MARKERS[label], linewidth=2.0, markersize=7
        )

    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Timely Rate of Matched Prefetches (%)")
    ax1.set_title("Timely Prefetch Rate vs Batch Size", fontweight="bold")
    ax1.set_xticks([1, 2, 4, 8])
    ax1.set_xticklabels(["B=1", "B=2", "B=4", "B=8"])
    ax1.grid(True)
    ax1.legend()

    # Late Rate
    for pred, h, label in [
        ("stmoe", 1, "ST-MoE (H=1)"),
        ("stmoe", 2, "ST-MoE (H=2)"),
        ("cstp", 1, "CSTP (H=1)"),
        ("cstp", 2, "CSTP (H=2)"),
    ]:
        sub = get_series(df_runs, pred, h)
        ax2.plot(
            sub["batch_size"], sub["late_rate_of_matched_pct"],
            label=label, color=CONFIG_COLORS[label],
            marker=CONFIG_MARKERS[label], linewidth=2.0, markersize=7
        )

    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Late Attachment Rate (%)")
    ax2.set_title("Late Prefetch Rate vs Batch Size", fontweight="bold")
    ax2.set_xticks([1, 2, 4, 8])
    ax2.set_xticklabels(["B=1", "B=2", "B=4", "B=8"])
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "timely_late_rates_vs_batch.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_horizon_comparison(df_horizon):
    if df_horizon is None:
        return
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # 1. Coverage
    for pred, c_h1, c_h2, label in [
        ("stmoe", "#EF4444", "#F87171", "ST-MoE"),
        ("cstp", "#2563EB", "#60A5FA", "CSTP")
    ]:
        sub = df_horizon[df_horizon["predictor"] == pred].sort_values("batch_size")
        ax1.plot(sub["batch_size"], sub["h1_coverage_pct"], label=f"{label} H=1", color=c_h1, marker="o", linewidth=2.0)
        ax1.plot(sub["batch_size"], sub["h2_coverage_pct"], label=f"{label} H=2", color=c_h2, marker="^", linestyle="--", linewidth=2.0)

    ax1.set_title("Coverage: H=1 vs H=2", fontweight="bold")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Coverage (%)")
    ax1.set_xticks([1, 2, 4, 8])
    ax1.set_xticklabels(["B=1", "B=2", "B=4", "B=8"])
    ax1.grid(True)
    ax1.legend()

    # 2. Unused Prefetches
    for pred, c_h1, c_h2, label in [
        ("stmoe", "#EF4444", "#F87171", "ST-MoE"),
        ("cstp", "#2563EB", "#60A5FA", "CSTP")
    ]:
        sub = df_horizon[df_horizon["predictor"] == pred].sort_values("batch_size")
        ax2.plot(sub["batch_size"], sub["h1_unused_count"], label=f"{label} H=1", color=c_h1, marker="o", linewidth=2.0)
        ax2.plot(sub["batch_size"], sub["h2_unused_count"], label=f"{label} H=2", color=c_h2, marker="^", linestyle="--", linewidth=2.0)

    ax2.set_title("Unused Prefetches: H=1 vs H=2", fontweight="bold")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Unused Completed Count")
    ax2.set_xticks([1, 2, 4, 8])
    ax2.set_xticklabels(["B=1", "B=2", "B=4", "B=8"])
    ax2.grid(True)
    ax2.legend()

    # 3. Additional Traffic
    for pred, c_h1, c_h2, label in [
        ("stmoe", "#EF4444", "#F87171", "ST-MoE"),
        ("cstp", "#2563EB", "#60A5FA", "CSTP")
    ]:
        sub = df_horizon[df_horizon["predictor"] == pred].sort_values("batch_size")
        ax3.plot(sub["batch_size"], sub["h1_additional_traffic_pct"], label=f"{label} H=1", color=c_h1, marker="o", linewidth=2.0)
        ax3.plot(sub["batch_size"], sub["h2_additional_traffic_pct"], label=f"{label} H=2", color=c_h2, marker="^", linestyle="--", linewidth=2.0)

    ax3.set_title("Additional DDR5 Traffic: H=1 vs H=2", fontweight="bold")
    ax3.set_xlabel("Batch Size")
    ax3.set_ylabel("Traffic Increase vs Baseline (%)")
    ax3.set_xticks([1, 2, 4, 8])
    ax3.set_xticklabels(["B=1", "B=2", "B=4", "B=8"])
    ax3.grid(True)
    ax3.legend()

    # 4. Exposed Memory Stall
    for pred, c_h1, c_h2, label in [
        ("stmoe", "#EF4444", "#F87171", "ST-MoE"),
        ("cstp", "#2563EB", "#60A5FA", "CSTP")
    ]:
        sub = df_horizon[df_horizon["predictor"] == pred].sort_values("batch_size")
        ax4.plot(sub["batch_size"], sub["h1_stall_ms"], label=f"{label} H=1", color=c_h1, marker="o", linewidth=2.0)
        ax4.plot(sub["batch_size"], sub["h2_stall_ms"], label=f"{label} H=2", color=c_h2, marker="^", linestyle="--", linewidth=2.0)

    ax4.set_title("Exposed Memory Stall: H=1 vs H=2", fontweight="bold")
    ax4.set_xlabel("Batch Size")
    ax4.set_ylabel("Exposed Stall (ms)")
    ax4.set_xticks([1, 2, 4, 8])
    ax4.set_xticklabels(["B=1", "B=2", "B=4", "B=8"])
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "horizon_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_online_learning_trends(df_learning):
    if df_learning is None:
        return
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # Select B=4 for clear learning progression curves
    sub = df_learning[df_learning["batch_size"] == 4]

    # 1. HBM Hit Rate Progression
    for pred, h, label in [
        ("baseline", 0, "Baseline"),
        ("stmoe", 1, "ST-MoE (H=1)"),
        ("cstp", 1, "CSTP (H=1)")
    ]:
        p_sub = sub[(sub["predictor"] == pred) & (sub["horizon"] == h)].sort_values("gen_step")
        ax1.plot(
            p_sub["gen_step"], p_sub["hbm_hit_pct"],
            label=label, color=CONFIG_COLORS[label],
            marker=CONFIG_MARKERS[label], linewidth=2.0, markersize=7
        )

    ax1.set_title("HBM Hit Rate over Decode Steps (B=4)", fontweight="bold")
    ax1.set_xlabel("Decode Step (Token Generation)")
    ax1.set_ylabel("HBM Demand Hit Rate (%)")
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xticklabels(["Step 0\n(Cold)", "Step 1\n(Warm 1)", "Step 2\n(Warm 2)", "Step 3\n(Warm 3)"])
    ax1.grid(True)
    ax1.legend()

    # 2. Demand Misses per Step
    for pred, h, label in [
        ("baseline", 0, "Baseline"),
        ("stmoe", 1, "ST-MoE (H=1)"),
        ("cstp", 1, "CSTP (H=1)")
    ]:
        p_sub = sub[(sub["predictor"] == pred) & (sub["horizon"] == h)].sort_values("gen_step")
        ax2.plot(
            p_sub["gen_step"], p_sub["demand_misses"],
            label=label, color=CONFIG_COLORS[label],
            marker=CONFIG_MARKERS[label], linewidth=2.0, markersize=7
        )

    ax2.set_title("Fresh Demand Misses over Steps (B=4)", fontweight="bold")
    ax2.set_xlabel("Decode Step")
    ax2.set_ylabel("Count of Demand Misses")
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels(["Step 0", "Step 1", "Step 2", "Step 3"])
    ax2.grid(True)
    ax2.legend()

    # 3. Token Latency over Steps
    for pred, h, label in [
        ("baseline", 0, "Baseline"),
        ("stmoe", 1, "ST-MoE (H=1)"),
        ("cstp", 1, "CSTP (H=1)")
    ]:
        p_sub = sub[(sub["predictor"] == pred) & (sub["horizon"] == h)].sort_values("gen_step")
        ax3.plot(
            p_sub["gen_step"], p_sub["token_duration_ms"],
            label=label, color=CONFIG_COLORS[label],
            marker=CONFIG_MARKERS[label], linewidth=2.0, markersize=7
        )

    ax3.set_title("Token Latency over Steps (B=4)", fontweight="bold")
    ax3.set_xlabel("Decode Step")
    ax3.set_ylabel("Token Duration (ms)")
    ax3.set_xticks([0, 1, 2, 3])
    ax3.set_xticklabels(["Step 0", "Step 1", "Step 2", "Step 3"])
    ax3.grid(True)
    ax3.legend()

    # 4. Prediction Precision over Steps (ST-MoE vs CSTP)
    for pred, h, label in [
        ("stmoe", 1, "ST-MoE (H=1)"),
        ("cstp", 1, "CSTP (H=1)")
    ]:
        p_sub = sub[(sub["predictor"] == pred) & (sub["horizon"] == h)].sort_values("gen_step")
        ax4.plot(
            p_sub["gen_step"], p_sub["precision_pct"],
            label=label, color=CONFIG_COLORS[label],
            marker=CONFIG_MARKERS[label], linewidth=2.0, markersize=7
        )

    ax4.set_title("Online Prediction Precision (B=4)", fontweight="bold")
    ax4.set_xlabel("Decode Step")
    ax4.set_ylabel("Precision@Budget (%)")
    ax4.set_xticks([0, 1, 2, 3])
    ax4.set_xticklabels(["Step 0", "Step 1", "Step 2", "Step 3"])
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "online_learning_trends.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("=" * 80)
    print("POINT 3: GENERATING PUBLICATION PLOTS")
    print("=" * 80)

    df_runs, df_horizon, df_learning = load_data()

    # Plot 1: Prefetch Coverage
    plot_metric_vs_batch(
        df_runs, "coverage_pct", "Prefetch Coverage (%)",
        "Prefetch Coverage vs Concurrent Batch Size",
        "prefetch_coverage_vs_batch.png"
    )

    # Plot 2: Timely & Late Rates
    plot_timely_and_late_rates(df_runs)

    # Plot 3: Unused Prefetches
    plot_metric_vs_batch(
        df_runs, "prefetches_completed_unused", "Unused Completed Prefetches (Count)",
        "Wasted / Unused Completed Prefetches vs Batch Size",
        "unused_prefetches_vs_batch.png"
    )

    # Plot 4: HBM Hit Rate
    plot_metric_vs_batch(
        df_runs, "overall_hbm_hit_rate_pct", "HBM Hit Rate (%)",
        "HBM Demand Hit Rate vs Batch Size",
        "hbm_hit_rate_vs_batch.png"
    )

    # Plot 5: Additional Memory Traffic
    plot_metric_vs_batch(
        df_runs, "additional_traffic_pct", "Additional DDR5 Traffic vs Baseline (%)",
        "Additional Off-Chip Memory Traffic Overhead vs Batch Size",
        "additional_memory_traffic_vs_batch.png"
    )

    # Plot 6: Exposed Memory Stall
    plot_metric_vs_batch(
        df_runs, "exposed_gpu_stall_ms", "Exposed GPU Stall Time (ms)",
        "Exposed GPU Memory Stall Time vs Batch Size",
        "exposed_memory_stall_vs_batch.png"
    )

    # Plot 7: Expert Load Latency
    plot_metric_vs_batch(
        df_runs, "expert_load_mean_us", "Mean Expert Load Latency (μs)",
        "Average Expert Load Latency vs Batch Size",
        "expert_load_latency_vs_batch.png"
    )

    # Plot 8: Token Latency
    plot_metric_vs_batch(
        df_runs, "token_latency_mean_ms", "Mean Token Latency (ms)",
        "End-to-End Decode Token Latency vs Batch Size",
        "token_latency_vs_batch.png"
    )

    # Plot 9: Horizon Comparison
    plot_horizon_comparison(df_horizon)

    # Plot 10: Online Learning Trends
    plot_online_learning_trends(df_learning)

    print("\n[ALL 10 PLOTS GENERATED SUCCESSFULLY!]")


if __name__ == "__main__":
    main()

