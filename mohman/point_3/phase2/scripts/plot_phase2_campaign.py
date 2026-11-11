#!/usr/bin/env python3
"""
Publication-Quality Plotting Suite for Phase 2 Extended Prefetching Campaign.
Generates all 13 required publication plots in PNG (300 DPI) and vector PDF formats.

STRICT RULE: Zero ternary operators (ast.IfExp == 0).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = "/home/mohman/ca_project"
PHASE2_ROOT = os.path.join(WORKSPACE, "mohman", "point_3", "phase2")
RESULTS_DIR = os.path.join(PHASE2_ROOT, "results")
PLOTS_DIR = os.path.join(PHASE2_ROOT, "plots")

if "seaborn-v0_8-whitegrid" in plt.style.available:
    plt.style.use("seaborn-v0_8-whitegrid")
else:
    plt.style.use("default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

COLORS = {
    "baseline": "#4A5568",     # Slate
    "stmoe_h1": "#3182CE",     # Blue
    "stmoe_h2": "#2B6CB0",     # Dark Blue
    "cstp_h1": "#DD6B20",      # Orange
    "cstp_h2": "#C05621",      # Dark Orange
    "timely": "#38A169",       # Green
    "late": "#E53E3E",         # Red
    "unused": "#805AD5",       # Purple
}


def save_figure(fig, filename_base):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    png_path = os.path.join(PLOTS_DIR, f"{filename_base}.png")
    pdf_path = os.path.join(PLOTS_DIR, f"{filename_base}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {filename_base}.png and .pdf")


def plot_fig1_prefetch_coverage(df):
    # Fig 1: Prefetch coverage vs batch size (ST-MoE vs CSTP, H=1 vs H=2)
    sub = df[(df["cache_label"] == "35gb") & (df["tokens"] == 4)].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    batches = [1, 2, 4, 8]
    for pred, h, col, lab, m in [
        ("stmoe", 1, COLORS["stmoe_h1"], "ST-MoE (H=1)", "o"),
        ("stmoe", 2, COLORS["stmoe_h2"], "ST-MoE (H=2)", "s"),
        ("cstp", 1, COLORS["cstp_h1"], "CSTP (H=1)", "^"),
        ("cstp", 2, COLORS["cstp_h2"], "CSTP (H=2)", "D"),
    ]:
        sdf = sub[(sub["predictor"] == pred) & (sub["horizon"] == h)].sort_values("batch_size")
        if len(sdf) > 0:
            ax.plot(sdf["batch_size"], sdf["overall_coverage_pct"], marker=m, color=col, label=lab, linewidth=2, markersize=7)

    ax.set_xlabel("Batch Size (B)")
    ax.set_ylabel("Prefetch Coverage (%)")
    ax.set_title("Fig 1: Prefetch Coverage vs Batch Size (40 MoE Stages, 35 GB Cache)")
    ax.set_xticks(batches)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    save_figure(fig, "fig1_prefetch_coverage_vs_batch_size")


def plot_fig2_timely_and_late_rates(df):
    # Fig 2: Timely and late prefetch rates vs batch size
    sub = df[(df["cache_label"] == "35gb") & (df["tokens"] == 4)].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Matched rate %
    for pred, h, col, lab, m in [
        ("stmoe", 1, COLORS["stmoe_h1"], "ST-MoE H=1 Late", "o"),
        ("stmoe", 2, COLORS["stmoe_h2"], "ST-MoE H=2 Late", "s"),
        ("cstp", 1, COLORS["cstp_h1"], "CSTP H=1 Late", "^"),
        ("cstp", 2, COLORS["cstp_h2"], "CSTP H=2 Late", "D"),
    ]:
        sdf = sub[(sub["predictor"] == pred) & (sub["horizon"] == h)].sort_values("batch_size")
        if len(sdf) > 0:
            ax1.plot(sdf["batch_size"], sdf["late_rate_of_matched_pct"], marker=m, color=col, label=lab, linewidth=2)

    ax1.set_xlabel("Batch Size (B)")
    ax1.set_ylabel("Late Attachment Rate of Matched (%)")
    ax1.set_title("(a) Late Prefetch Rate vs Batch Size")
    ax1.set_xticks([1, 2, 4, 8])
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(frameon=True)

    # Right: Assisted Count
    for pred, h, col, lab, m in [
        ("stmoe", 1, COLORS["stmoe_h1"], "ST-MoE H=1", "o"),
        ("stmoe", 2, COLORS["stmoe_h2"], "ST-MoE H=2", "s"),
        ("cstp", 1, COLORS["cstp_h1"], "CSTP H=1", "^"),
        ("cstp", 2, COLORS["cstp_h2"], "CSTP H=2", "D"),
    ]:
        sdf = sub[(sub["predictor"] == pred) & (sub["horizon"] == h)].sort_values("batch_size")
        if len(sdf) > 0:
            ax2.plot(sdf["batch_size"], sdf["prefetch_assisted_demands"], marker=m, color=col, label=lab, linewidth=2)

    ax2.set_xlabel("Batch Size (B)")
    ax2.set_ylabel("Prefetch-Assisted Demands (Count)")
    ax2.set_title("(b) Assisted Demand Count vs Batch Size")
    ax2.set_xticks([1, 2, 4, 8])
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(frameon=True)

    fig.suptitle("Fig 2: Timely and Late Prefetch Rates (40 MoE Stages)", y=1.02)
    save_figure(fig, "fig2_timely_and_late_prefetch_rates")


def plot_fig3_unused_prefetches(df):
    # Fig 3: Unused prefetches (% and bytes) vs predictor, horizon, and batch size
    sub = df[(df["cache_label"] == "35gb") & (df["tokens"] == 4)].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for pred, h, col, lab, m in [
        ("stmoe", 1, COLORS["stmoe_h1"], "ST-MoE H=1", "o"),
        ("stmoe", 2, COLORS["stmoe_h2"], "ST-MoE H=2", "s"),
        ("cstp", 1, COLORS["cstp_h1"], "CSTP H=1", "^"),
        ("cstp", 2, COLORS["cstp_h2"], "CSTP H=2", "D"),
    ]:
        sdf = sub[(sub["predictor"] == pred) & (sub["horizon"] == h)].sort_values("batch_size")
        if len(sdf) > 0:
            ax1.plot(sdf["batch_size"], sdf["unused_pct"], marker=m, color=col, label=lab, linewidth=2)
            wasted_gb = sdf["completed_unused_bytes"] / (1024**3)
            ax2.plot(sdf["batch_size"], wasted_gb, marker=m, color=col, label=lab, linewidth=2)

    ax1.set_xlabel("Batch Size (B)")
    ax1.set_ylabel("Unused Completed Prefetches (%)")
    ax1.set_title("(a) Unused Prefetch Percentage")
    ax1.set_xticks([1, 2, 4, 8])
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(frameon=True)

    ax2.set_xlabel("Batch Size (B)")
    ax2.set_ylabel("Wasted Speculative Traffic (GiB)")
    ax2.set_title("(b) Wasted DDR5 Speculative Bytes")
    ax2.set_xticks([1, 2, 4, 8])
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(frameon=True)

    fig.suptitle("Fig 3: Unused Prefetch Efficiency (40 MoE Stages, 35 GB Cache)", y=1.02)
    save_figure(fig, "fig3_unused_prefetches_and_wasted_bytes")


def plot_fig4_hbm_hit_rate(df):
    # Fig 4: HBM hit rate vs batch size (baseline vs prefetching)
    sub = df[(df["cache_label"] == "35gb") & (df["tokens"] == 4)].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for pred, h, col, lab, m in [
        ("baseline", 0, COLORS["baseline"], "Baseline Demand", "x"),
        ("stmoe", 1, COLORS["stmoe_h1"], "ST-MoE H=1", "o"),
        ("cstp", 1, COLORS["cstp_h1"], "CSTP H=1", "^"),
    ]:
        sdf = sub[(sub["predictor"] == pred) & (sub["horizon"] == h)].sort_values("batch_size")
        if len(sdf) > 0:
            ax.plot(sdf["batch_size"], sdf["overall_hbm_hit_rate_pct"], marker=m, color=col, label=lab, linewidth=2, markersize=7)

    ax.set_xlabel("Batch Size (B)")
    ax.set_ylabel("HBM Demand Hit Rate (%)")
    ax.set_title("Fig 4: HBM Hit Rate Scaling Across Concurrency (40 MoE Stages, 35 GB Cache)")
    ax.set_xticks([1, 2, 4, 8])
    ax.set_ylim(60, 95)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    save_figure(fig, "fig4_hbm_hit_rate_scaling")


def plot_fig5_ddr5_memory_traffic(df):
    # Fig 5: DDR5 memory traffic (demand, speculative, total) vs batch size
    sub = df[(df["cache_label"] == "35gb") & (df["tokens"] == 4)].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    b1_base = sub[(sub["batch_size"] == 1) & (sub["predictor"] == "baseline")].iloc[0]
    b8_base = sub[(sub["batch_size"] == 8) & (sub["predictor"] == "baseline")].iloc[0]

    bar_width = 0.18
    batches = [1, 2, 4, 8]
    x = np.arange(len(batches))

    cstp_h1 = sub[(sub["predictor"] == "cstp") & (sub["horizon"] == 1)].sort_values("batch_size")
    stmoe_h1 = sub[(sub["predictor"] == "stmoe") & (sub["horizon"] == 1)].sort_values("batch_size")
    baseline = sub[sub["predictor"] == "baseline"].sort_values("batch_size")

    ax.bar(x - bar_width, baseline["total_ddr5_bytes_transferred"] / (1024**3), width=bar_width, color=COLORS["baseline"], label="Baseline Demand")
    ax.bar(x, stmoe_h1["total_ddr5_bytes_transferred"] / (1024**3), width=bar_width, color=COLORS["stmoe_h1"], label="ST-MoE H=1 (Total)")
    ax.bar(x + bar_width, cstp_h1["total_ddr5_bytes_transferred"] / (1024**3), width=bar_width, color=COLORS["cstp_h1"], label="CSTP H=1 (Total)")

    ax.set_xlabel("Batch Size (B)")
    ax.set_ylabel("Total DDR5 Traffic Transferred (GiB)")
    ax.set_title("Fig 5: DDR5 Memory Traffic Breakdown Across Batch Sizes (40 MoE Stages)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"B={b}" for b in batches])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    save_figure(fig, "fig5_ddr5_memory_traffic_breakdown")


def plot_fig6_traffic_overhead_ratio(df):
    # Fig 6: Traffic overhead ratio vs batch size
    sub = df[(df["cache_label"] == "35gb") & (df["tokens"] == 4)].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Calculate baseline traffic per batch
    base_dict = {}
    for _, row in sub[sub["predictor"] == "baseline"].iterrows():
        base_dict[row["batch_size"]] = row["total_ddr5_bytes_transferred"]

    for pred, h, col, lab, m in [
        ("stmoe", 1, COLORS["stmoe_h1"], "ST-MoE (H=1)", "o"),
        ("stmoe", 2, COLORS["stmoe_h2"], "ST-MoE (H=2)", "s"),
        ("cstp", 1, COLORS["cstp_h1"], "CSTP (H=1)", "^"),
        ("cstp", 2, COLORS["cstp_h2"], "CSTP (H=2)", "D"),
    ]:
        sdf = sub[(sub["predictor"] == pred) & (sub["horizon"] == h)].sort_values("batch_size")
        if len(sdf) > 0:
            overhead = []
            for _, r in sdf.iterrows():
                b = r["batch_size"]
                base_b = base_dict.get(b, r["total_ddr5_bytes_transferred"])
                ovh = ((r["total_ddr5_bytes_transferred"] - base_b) / base_b) * 100.0
                overhead.append(ovh)
            ax.plot(sdf["batch_size"], overhead, marker=m, color=col, label=lab, linewidth=2, markersize=7)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Batch Size (B)")
    ax.set_ylabel("Traffic Overhead Above Baseline (%)")
    ax.set_title("Fig 6: Additional DDR5 Bus Overhead vs Batch Concurrency")
    ax.set_xticks([1, 2, 4, 8])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    save_figure(fig, "fig6_traffic_overhead_ratio")


def plot_fig7_exposed_gpu_stall(df):
    # Fig 7: Exposed GPU memory-stall time vs batch size
    sub = df[(df["cache_label"] == "35gb") & (df["tokens"] == 4)].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for pred, h, col, lab, m in [
        ("baseline", 0, COLORS["baseline"], "Baseline Demand", "x"),
        ("stmoe", 1, COLORS["stmoe_h1"], "ST-MoE H=1", "o"),
        ("stmoe", 2, COLORS["stmoe_h2"], "ST-MoE H=2", "s"),
        ("cstp", 1, COLORS["cstp_h1"], "CSTP H=1", "^"),
        ("cstp", 2, COLORS["cstp_h2"], "CSTP H=2", "D"),
    ]:
        sdf = sub[(sub["predictor"] == pred) & (sub["horizon"] == h)].sort_values("batch_size")
        if len(sdf) > 0:
            ax.plot(sdf["batch_size"], sdf["exposed_gpu_stall_ms"], marker=m, color=col, label=lab, linewidth=2, markersize=7)

    ax.set_xlabel("Batch Size (B)")
    ax.set_ylabel("Exposed GPU Memory Stall Time (ms)")
    ax.set_title("Fig 7: Total Exposed Memory Stall vs Batch Size (40 MoE Stages)")
    ax.set_xticks([1, 2, 4, 8])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    save_figure(fig, "fig7_exposed_gpu_stall_reduction")


def plot_fig8_mean_token_latency(df):
    # Fig 8: Mean token latency vs batch size
    sub = df[(df["cache_label"] == "35gb") & (df["tokens"] == 4)].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for pred, h, col, lab, m in [
        ("baseline", 0, COLORS["baseline"], "Baseline Demand", "x"),
        ("stmoe", 1, COLORS["stmoe_h1"], "ST-MoE H=1", "o"),
        ("stmoe", 2, COLORS["stmoe_h2"], "ST-MoE H=2", "s"),
        ("cstp", 1, COLORS["cstp_h1"], "CSTP H=1", "^"),
        ("cstp", 2, COLORS["cstp_h2"], "CSTP H=2", "D"),
    ]:
        sdf = sub[(sub["predictor"] == pred) & (sub["horizon"] == h)].sort_values("batch_size")
        if len(sdf) > 0:
            ax.plot(sdf["batch_size"], sdf["mean_token_lat_ms"], marker=m, color=col, label=lab, linewidth=2, markersize=7)

    ax.set_xlabel("Batch Size (B)")
    ax.set_ylabel("Mean Token Latency (ms)")
    ax.set_title("Fig 8: Mean Token Latency Scaling (40 MoE Stages, 35 GB Cache)")
    ax.set_xticks([1, 2, 4, 8])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    save_figure(fig, "fig8_mean_token_latency_scaling")


def plot_fig9_tail_p95_token_latency(df):
    # Fig 9: Tail token latency (p95) vs batch size
    sub = df[(df["cache_label"] == "35gb") & (df["tokens"] == 4)].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for pred, h, col, lab, m in [
        ("baseline", 0, COLORS["baseline"], "Baseline Demand", "x"),
        ("stmoe", 1, COLORS["stmoe_h1"], "ST-MoE H=1", "o"),
        ("cstp", 1, COLORS["cstp_h1"], "CSTP H=1", "^"),
    ]:
        sdf = sub[(sub["predictor"] == pred) & (sub["horizon"] == h)].sort_values("batch_size")
        if len(sdf) > 0:
            ax.plot(sdf["batch_size"], sdf["p95_token_lat_ms"], marker=m, color=col, label=lab, linewidth=2, markersize=7)

    ax.set_xlabel("Batch Size (B)")
    ax.set_ylabel("P95 Tail Token Latency (ms)")
    ax.set_title("Fig 9: P95 Tail Token Latency vs Batch Size (40 MoE Stages)")
    ax.set_xticks([1, 2, 4, 8])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    save_figure(fig, "fig9_tail_p95_token_latency")


def plot_fig10_expert_load_latency_distributions(df):
    # Fig 10: Expert-load latency distribution across predictors and baselines
    sub = df[(df["cache_label"] == "35gb") & (df["tokens"] == 4) & (df["batch_size"] == 1)].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    predictors = ["baseline", "stmoe_h1", "stmoe_h2", "cstp_h1", "cstp_h2"]
    labels = ["Baseline", "ST-MoE H=1", "ST-MoE H=2", "CSTP H=1", "CSTP H=2"]
    means = []
    p95s = []

    for p in predictors:
        if p == "baseline":
            row = sub[sub["predictor"] == "baseline"].iloc[0]
        elif p == "stmoe_h1":
            row = sub[(sub["predictor"] == "stmoe") & (sub["horizon"] == 1)].iloc[0]
        elif p == "stmoe_h2":
            row = sub[(sub["predictor"] == "stmoe") & (sub["horizon"] == 2)].iloc[0]
        elif p == "cstp_h1":
            row = sub[(sub["predictor"] == "cstp") & (sub["horizon"] == 1)].iloc[0]
        elif p == "cstp_h2":
            row = sub[(sub["predictor"] == "cstp") & (sub["horizon"] == 2)].iloc[0]
        means.append(row["mean_expert_load_lat_us"])
        p95s.append(row["p95_expert_load_lat_us"])

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, means, width, label=r"Mean Latency ($\mu$s)", color=COLORS["stmoe_h1"])
    ax.bar(x + width/2, p95s, width, label=r"P95 Latency ($\mu$s)", color=COLORS["cstp_h1"])

    ax.set_ylabel(r"Latency ($\mu$s)")
    ax.set_title("Fig 10: Expert Load Latency Distribution (B=1, 40 MoE Stages, 35 GB Cache)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)
    save_figure(fig, "fig10_expert_load_latency_distributions")


def plot_fig11_cache_capacity_sensitivity(df):
    # Fig 11: HBM cache capacity sensitivity curve (evictions, hit rate, prefetch coverage)
    # Read cache_sweep_results.csv
    sweep_path = os.path.join(RESULTS_DIR, "cache_sweep_results.csv")
    if not os.path.exists(sweep_path):
        print(f"[SKIP] {sweep_path} not found")
        return

    sdf = pd.read_csv(sweep_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    base_sw = sdf[sdf["predictor"] == "baseline"].sort_values("cache_gb")
    cstp_sw = sdf[sdf["predictor"] == "cstp"].sort_values("cache_gb")

    # Left: Evictions vs Cache Capacity
    ax1.plot(base_sw["cache_gb"], base_sw["evicts"], marker="o", color=COLORS["baseline"], label="Baseline Evictions", linewidth=2)
    ax1.plot(cstp_sw["cache_gb"], cstp_sw["evicts"], marker="s", color=COLORS["cstp_h1"], label="CSTP Evictions", linewidth=2)
    ax1.set_xlabel("HBM Cache Capacity (GB)")
    ax1.set_ylabel("Eviction Count")
    ax1.set_title("(a) Cache Evictions vs HBM Capacity")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(frameon=True)

    # Right: Hit rate and dispatches
    ax2.plot(base_sw["cache_gb"], base_sw["hits"], marker="o", color=COLORS["timely"], label="Demand Hits", linewidth=2)
    ax2.plot(cstp_sw["cache_gb"], cstp_sw["dispatches"], marker="^", color=COLORS["cstp_h2"], label="CSTP Dispatches", linewidth=2)
    ax2.set_xlabel("HBM Cache Capacity (GB)")
    ax2.set_ylabel("Count")
    ax2.set_title("(b) Demand Hits and Speculative Dispatches")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(frameon=True)

    fig.suptitle("Fig 11: HBM Cache Capacity Sensitivity & Eviction Regimes (40 MoE Stages)", y=1.02)
    save_figure(fig, "fig11_cache_capacity_sensitivity_curve")


def plot_fig12_decode_progress_trends(df):
    # Fig 12: Decode progress trends (metrics across decode steps T=4 vs T=8)
    sub = df[(df["cache_label"] == "35gb") & (df["batch_size"] == 1)].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    labels = ["Baseline (T=4)", "Baseline (T=8)", "ST-MoE (T=4)", "ST-MoE (T=8)", "CSTP (T=4)", "CSTP (T=8)"]
    stalls = []
    evicts = []

    for pred, tok in [("baseline", 4), ("baseline", 8), ("stmoe", 4), ("stmoe", 8), ("cstp", 4), ("cstp", 8)]:
        sdf = sub[(sub["predictor"] == pred) & (sub["tokens"] == tok)]
        if len(sdf) > 0:
            row = sdf.iloc[0]
            stalls.append(row["stall_per_token_ms"])
            evicts.append(row["evicts"])
        else:
            stalls.append(0)
            evicts.append(0)

    x = np.arange(len(labels))
    ax1.bar(x, stalls, color=["#718096", "#4A5568", "#63B3ED", "#3182CE", "#F6AD55", "#DD6B20"])
    ax1.set_ylabel("Memory Stall Per Token (ms)")
    ax1.set_title("(a) Memory Stall Per Token across Generation Depth")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.bar(x, evicts, color=["#718096", "#4A5568", "#63B3ED", "#3182CE", "#F6AD55", "#DD6B20"])
    ax2.set_ylabel("Total Eviction Count")
    ax2.set_title("(b) Cumulative Cache Evictions (T=4 vs T=8)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Fig 12: Decode Progress Trends (40 MoE Stages, 35 GB Cache)", y=1.02)
    save_figure(fig, "fig12_decode_progress_trends")


def plot_fig13_stmoe_vs_cstp_head_to_head(df):
    # Fig 13: Head-to-head comparison chart: ST-MoE vs CSTP across all evaluation dimensions
    sub = df[(df["cache_label"] == "35gb") & (df["tokens"] == 4) & (df["batch_size"] == 1)].copy()
    fig, ax = plt.subplots(figsize=(8, 5))

    st_h1 = sub[(sub["predictor"] == "stmoe") & (sub["horizon"] == 1)].iloc[0]
    st_h2 = sub[(sub["predictor"] == "stmoe") & (sub["horizon"] == 2)].iloc[0]
    cs_h1 = sub[(sub["predictor"] == "cstp") & (sub["horizon"] == 1)].iloc[0]
    cs_h2 = sub[(sub["predictor"] == "cstp") & (sub["horizon"] == 2)].iloc[0]

    # Metrics normalized to ST-MoE H=1 (100%)
    categories = [
        "Prefetch-Assisted Demands",
        "Dispatched Transfers",
        "Speculative DDR5 Traffic",
        "Total DDR5 Traffic",
        "Exposed GPU Stall",
        "Inference Runtime",
    ]

    st_vals = [
        st_h1["prefetch_assisted_demands"],
        st_h1["total_dispatched_prefetches"],
        st_h1["speculative_bytes_transferred"] / 1e9,
        st_h1["total_ddr5_bytes_transferred"] / 1e9,
        st_h1["exposed_gpu_stall_ms"],
        st_h1["total_simulated_time_ms"],
    ]

    cs_vals = [
        cs_h1["prefetch_assisted_demands"],
        cs_h1["total_dispatched_prefetches"],
        cs_h1["speculative_bytes_transferred"] / 1e9,
        cs_h1["total_ddr5_bytes_transferred"] / 1e9,
        cs_h1["exposed_gpu_stall_ms"],
        cs_h1["total_simulated_time_ms"],
    ]

    x = np.arange(len(categories))
    width = 0.35
    ax.barh(x - width/2, [c / s * 100.0 for c, s in zip(cs_vals, st_vals)], width, color=COLORS["cstp_h1"], label="CSTP (Relative to ST-MoE %)")
    ax.axvline(100.0, color="gray", linestyle="--", alpha=0.8, label="ST-MoE Reference (100%)")

    ax.set_xlabel("CSTP Relative Value (%) [100% = ST-MoE]")
    ax.set_title("Fig 13: ST-MoE vs CSTP Head-to-Head Efficiency (40 MoE Stages, B=1, 35 GB)")
    ax.set_yticks(x)
    ax.set_yticklabels(categories)
    ax.set_xlim(80, 110)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, loc="lower right")
    save_figure(fig, "fig13_stmoe_vs_cstp_head_to_head")


def main():
    print("======================================================================")
    print("GENERATING PUBLICATION-QUALITY PLOTTING SUITE (13 FIGURES)")
    print("======================================================================")

    per_run_csv = os.path.join(RESULTS_DIR, "per_run_metrics.csv")
    if not os.path.exists(per_run_csv):
        print(f"[ERROR] {per_run_csv} not found! Wait for campaign to complete.")
        sys.exit(1)

    df = pd.read_csv(per_run_csv)

    plot_fig1_prefetch_coverage(df)
    plot_fig2_timely_and_late_rates(df)
    plot_fig3_unused_prefetches(df)
    plot_fig4_hbm_hit_rate(df)
    plot_fig5_ddr5_memory_traffic(df)
    plot_fig6_traffic_overhead_ratio(df)
    plot_fig7_exposed_gpu_stall(df)
    plot_fig8_mean_token_latency(df)
    plot_fig9_tail_p95_token_latency(df)
    plot_fig10_expert_load_latency_distributions(df)
    plot_fig11_cache_capacity_sensitivity(df)
    plot_fig12_decode_progress_trends(df)
    plot_fig13_stmoe_vs_cstp_head_to_head(df)

    print("\n======================================================================")
    print("ALL 13 PUBLICATION-QUALITY PLOTS GENERATED IN PNG AND PDF!")
    print("======================================================================")


if __name__ == "__main__":
    main()
