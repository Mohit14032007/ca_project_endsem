#!/usr/bin/env python3
"""
generate_plots.py
Point 3 — Publication-Quality Plot Generation Script

Generates all required plots for the final Point 3 evaluation:
  1. plots/01_total_time_vs_stall_reduction/ (Per-workload, overview, scatter trade-off, batch scaling)
  2. plots/02_prefetch_coverage/ (Strict vs associated coverage)
  3. plots/03_timeliness/ (Timely vs late rates, transit time)
  4. plots/04_unused_prefetches/ (Unused rate, wasted bytes)
  5. plots/05_hbm_hit_rate/ (Hit rate comparisons and scaling)
  6. plots/06_memory_traffic/ (DDR5 breakdown, net additional, overhead %)
  7. plots/07_exposed_memory_stall/ (Stall time, stall reduction)
  8. plots/08_expert_load_latency/ (Mean load latency, latency reduction)
  9. plots/09_end_to_end_latency/ (Mean decode step latency, p95 tail latency)
  10. plots/10_throughput/ (Tokens/sec throughput scaling)
  11. plots/comparison_dashboards/ (Multi-panel comprehensive dashboards)

Strict rule: ZERO ternary operators used throughout this script.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Color Palette: Consistent across all plots
COLOR_BASELINE = "#7f7f7f"  # Neutral grey
COLOR_STMOE = "#1f77b4"     # Deep blue
COLOR_CSTP = "#d62728"      # Crimson red

def setup_plot_style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#cccccc"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["grid.alpha"] = 0.7

def save_dual_format(fig, base_path):
    png_path = base_path + ".png"
    pdf_path = base_path + ".pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path} and {pdf_path}")

# ==============================================================================
# SECTION 1: Total Simulated Time vs Exposed Memory-Stall Reduction
# ==============================================================================

def generate_per_workload_plots(df_lw, out_dir):
    workloads = ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]
    for wid in workloads:
        df_w = df_lw[df_lw["workload_id"] == wid].copy()
        if df_w.empty:
            continue

        # Sort: baseline, stmoe, cstp
        pred_order = {"baseline": 0, "stmoe": 1, "cstp": 2}
        df_w["order"] = df_w["predictor"].map(pred_order)
        # Filter for H=2 for predictors if available, otherwise H=1
        h_filter = []
        for idx, r in df_w.iterrows():
            if r["predictor"] == "baseline":
                h_filter.append(True)
            elif r["lookahead_H"] == 2:
                h_filter.append(True)
            else:
                # Keep H=1 only if H=2 is missing
                h2_exists = len(df_w[(df_w["predictor"] == r["predictor"]) & (df_w["lookahead_H"] == 2)]) > 0
                if h2_exists:
                    h_filter.append(False)
                else:
                    h_filter.append(True)
        df_sub = df_w[h_filter].sort_values("order")

        labels = []
        sim_times = []
        stall_reds = []
        colors = []
        for idx, r in df_sub.iterrows():
            p_name = r["predictor"].upper()
            if r["predictor"] == "stmoe":
                p_name = "ST-MoE"
            elif r["predictor"] == "cstp":
                p_name = "CSTP"
            else:
                p_name = "Baseline"
            labels.append(p_name)
            sim_times.append(r["total_simulated_time_s"])
            stall_reds.append(r["exposed_memory_stall_reduction_ms"])
            if r["predictor"] == "baseline":
                colors.append(COLOR_BASELINE)
            elif r["predictor"] == "stmoe":
                colors.append(COLOR_STMOE)
            else:
                colors.append(COLOR_CSTP)

        fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
        x = np.arange(len(labels))
        width = 0.35

        # Bar for Total Simulated Time
        rects1 = ax1.bar(x - width/2, sim_times, width, label="Total Simulated Time (s)", color=colors, alpha=0.85, edgecolor="#222222")
        ax1.set_ylabel("Total Simulated Time (s)", color="#222222", fontsize=11, fontweight="bold")
        ax1.set_xlabel("System Configuration", fontsize=11, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax1.grid(True, axis="y", alpha=0.5)

        # Right Y-axis for Exposed Stall Reduction
        ax2 = ax1.twinx()
        rects2 = ax2.bar(x + width/2, stall_reds, width, label="Stall Reduction (ms)", color=colors, hatch="//", alpha=0.9, edgecolor="#222222")
        ax2.set_ylabel("Exposed Memory-Stall Reduction (ms)", color="#222222", fontsize=11, fontweight="bold")

        b_val = df_sub["batch_size"].iloc[0]
        t_val = df_sub["sequence_length"].iloc[0]
        ax1.set_title(f"Workload {wid}: B={b_val}, T={t_val}, 40 MoE Stages\nTotal Simulated Time vs Exposed Stall Reduction", fontsize=11, fontweight="bold", pad=12)

        # Value annotations
        for rect in rects1:
            h = rect.get_height()
            ax1.annotate(f"{h:.2f}s", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for rect in rects2:
            h = rect.get_height()
            if h > 0:
                ax2.annotate(f"+{h:.1f}ms", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color="#8b0000", fontweight="bold")

        base_path = os.path.join(out_dir, f"fig_01a_{wid.lower()}_total_time_vs_stall")
        save_dual_format(fig, base_path)

def generate_all_workloads_overview(df_lw, out_dir):
    workloads = ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]
    df_h2 = df_lw[df_lw["lookahead_H"].isin([0, 2])].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Simulation Time across Workloads
    ax1 = axes[0]
    x = np.arange(len(workloads))
    w = 0.25

    base_times = []
    stmoe_times = []
    cstp_times = []
    for wid in workloads:
        sub = df_h2[df_h2["workload_id"] == wid]
        b_row = sub[sub["predictor"] == "baseline"]
        s_row = sub[sub["predictor"] == "stmoe"]
        c_row = sub[sub["predictor"] == "cstp"]
        
        base_times.append(b_row["total_simulated_time_s"].iloc[0])
        stmoe_times.append(s_row["total_simulated_time_s"].iloc[0])
        cstp_times.append(c_row["total_simulated_time_s"].iloc[0])

    ax1.bar(x - w, base_times, w, label="Baseline", color=COLOR_BASELINE, edgecolor="#222222")
    ax1.bar(x, stmoe_times, w, label="ST-MoE (H=2)", color=COLOR_STMOE, edgecolor="#222222")
    ax1.bar(x + w, cstp_times, w, label="CSTP (H=2)", color=COLOR_CSTP, edgecolor="#222222")
    ax1.set_ylabel("Total Simulated Time (s)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Workload ID", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(workloads, fontsize=10, fontweight="bold")
    ax1.set_title("Total Simulated Time across Workloads (Lower is Better)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, axis="y", alpha=0.5)

    # Panel 2: Exposed Stall Reduction across Workloads
    ax2 = axes[1]
    stmoe_reds = []
    cstp_reds = []
    for wid in workloads:
        sub = df_h2[df_h2["workload_id"] == wid]
        s_row = sub[sub["predictor"] == "stmoe"]
        c_row = sub[sub["predictor"] == "cstp"]
        stmoe_reds.append(s_row["exposed_memory_stall_reduction_ms"].iloc[0])
        cstp_reds.append(c_row["exposed_memory_stall_reduction_ms"].iloc[0])

    w_half = 0.35
    ax2.bar(x - w_half/2, stmoe_reds, w_half, label="ST-MoE (H=2)", color=COLOR_STMOE, edgecolor="#222222")
    ax2.bar(x + w_half/2, cstp_reds, w_half, label="CSTP (H=2)", color=COLOR_CSTP, edgecolor="#222222")
    ax2.set_ylabel("Exposed Memory-Stall Reduction (ms)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Workload ID", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(workloads, fontsize=10, fontweight="bold")
    ax2.set_title("Exposed Memory-Stall Reduction (Higher is Better)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(True, axis="y", alpha=0.5)

    # Annotate large values on W7
    ax2.annotate(f"+{cstp_reds[-1]:.1f} ms", xy=(x[-1] + w_half/2, cstp_reds[-1]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8, fontweight="bold", color="#8b0000")
    ax2.annotate(f"+{stmoe_reds[-1]:.1f} ms", xy=(x[-1] - w_half/2, stmoe_reds[-1]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    base_path = os.path.join(out_dir, "fig_01b_all_workloads_time_and_stall_overview")
    save_dual_format(fig, base_path)

def generate_tradeoff_scatter(df_lw, out_dir):
    df_h2 = df_lw[df_lw["lookahead_H"].isin([0, 2])].copy()

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for idx, r in df_h2.iterrows():
        p = r["predictor"]
        x_val = r["total_simulated_time_s"]
        y_val = r["exposed_memory_stall_reduction_ms"]
        wid = r["workload_id"]

        col = COLOR_BASELINE
        marker = "s"
        size = 80
        if p == "stmoe":
            col = COLOR_STMOE
            marker = "o"
            size = 100
        elif p == "cstp":
            col = COLOR_CSTP
            marker = "^"
            size = 110

        ax.scatter(x_val, y_val, color=col, marker=marker, s=size, edgecolors="#222222", alpha=0.9, zorder=5)
        # Label points
        offset_y = 120
        if p == "baseline":
            offset_y = -150
        elif p == "stmoe":
            offset_y = 80
        else:
            offset_y = 150
        ax.annotate(f"{wid}:{p[:2].upper()}", (x_val, y_val), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5)

    ax.set_xlabel("Total Simulated Execution Time (s) [Lower is Better]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Exposed Memory-Stall Reduction (ms) [Higher is Better]", fontsize=11, fontweight="bold")
    ax.set_title("Point 3 Trade-Off: Simulated Time vs Memory-Stall Reduction\n(DeepSeek-V3, 40 MoE Stages, 35 GB HBM)", fontsize=11, fontweight="bold", pad=12)

    # Custom legend
    import matplotlib.lines as mlines
    leg_base = mlines.Line2D([], [], color=COLOR_BASELINE, marker="s", linestyle="None", markersize=8, label="Baseline (Stall Red = 0)")
    leg_stmoe = mlines.Line2D([], [], color=COLOR_STMOE, marker="o", linestyle="None", markersize=8, label="ST-MoE (H=2)")
    leg_cstp = mlines.Line2D([], [], color=COLOR_CSTP, marker="^", linestyle="None", markersize=9, label="CSTP (H=2)")
    ax.legend(handles=[leg_base, leg_stmoe, leg_cstp], loc="upper left", framealpha=0.95)

    ax.grid(True, linestyle="--", alpha=0.6)
    base_path = os.path.join(out_dir, "fig_01c_time_vs_stall_tradeoff_scatter")
    save_dual_format(fig, base_path)

def generate_batch_scaling_plots(df_lw, out_dir):
    # W2 (B=14), W3 (B=32), W4 (B=48), W5 (B=70) at T=16, H=2
    df_b = df_lw[(df_lw["sequence_length"] == 16) & (df_lw["lookahead_H"].isin([0, 2]))].copy()

    batches = [14, 32, 48, 70]

    # Plot D1: Total Simulated Time vs Batch Size
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for p, col, mark, lab in [("baseline", COLOR_BASELINE, "s", "Baseline"),
                              ("stmoe", COLOR_STMOE, "o", "ST-MoE (H=2)"),
                              ("cstp", COLOR_CSTP, "^", "CSTP (H=2)")]:
        sub = df_b[df_b["predictor"] == p].sort_values("batch_size")
        ax.plot(sub["batch_size"], sub["total_simulated_time_s"], marker=mark, color=col, linewidth=2, markersize=8, label=lab)

    ax.set_xlabel("Serving Batch Size (Concurrent Sequences)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Simulated Execution Time (s)", fontsize=11, fontweight="bold")
    ax.set_title("Batch Concurrency Scaling: Total Simulated Time\n(Fixed T=16, 40 MoE Stages, 35 GB HBM)", fontsize=11, fontweight="bold")
    ax.set_xticks(batches)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    base_path = os.path.join(out_dir, "fig_01d_batch_scaling_total_time")
    save_dual_format(fig, base_path)

    # Plot D2: Stall Reduction vs Batch Size
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
    for p, col, mark, lab in [("stmoe", COLOR_STMOE, "o", "ST-MoE (H=2)"),
                              ("cstp", COLOR_CSTP, "^", "CSTP (H=2)")]:
        sub = df_b[df_b["predictor"] == p].sort_values("batch_size")
        ax2.plot(sub["batch_size"], sub["exposed_memory_stall_reduction_ms"], marker=mark, color=col, linewidth=2, markersize=8, label=lab)

    ax2.set_xlabel("Serving Batch Size (Concurrent Sequences)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Exposed Memory-Stall Reduction (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("Batch Concurrency Scaling: Exposed Stall Reduction\n(Fixed T=16, 40 MoE Stages, 35 GB HBM)", fontsize=11, fontweight="bold")
    ax2.set_xticks(batches)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    base_path2 = os.path.join(out_dir, "fig_01d_batch_scaling_stall_reduction")
    save_dual_format(fig2, base_path2)

# ==============================================================================
# SECTION 2: Prefetch Coverage Plots
# ==============================================================================

def generate_coverage_plots(df_lw, out_dir):
    # W1 to W7 for H=2
    df_h2 = df_lw[df_lw["lookahead_H"] == 2].copy()
    workloads = ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]

    # 1. Strict vs Associated Coverage
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(workloads))
    w = 0.35

    stmoe_assoc = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "stmoe")]["associated_coverage_pct"].iloc[0] for wid in workloads]
    cstp_assoc = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "cstp")]["associated_coverage_pct"].iloc[0] for wid in workloads]

    rects1 = ax.bar(x - w/2, stmoe_assoc, w, label="ST-MoE Associated Coverage (%)", color=COLOR_STMOE, edgecolor="#222222")
    rects2 = ax.bar(x + w/2, cstp_assoc, w, label="CSTP Associated Coverage (%)", color=COLOR_CSTP, edgecolor="#222222")

    # Strict timely line at 0.0%
    ax.axhline(0.0, color="#000000", linewidth=1.5, linestyle="-", label="Strict Timely Coverage = 0.00% (Both)")

    ax.set_ylabel("Prefetch-Associated Demand Coverage (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Workload ID", fontsize=11, fontweight="bold")
    ax.set_title("Prefetch-Associated Coverage across Workloads (Strict Timely = 0.00%)\n(H=2, 40 MoE Stages, 35 GB HBM)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, fontsize=10, fontweight="bold")
    ax.set_ylim(-0.5, 11.5)
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.5)

    base_path = os.path.join(out_dir, "fig_02a_strict_vs_associated_coverage")
    save_dual_format(fig, base_path)

    # 2. Coverage vs Batch Size (T=16: W2, W3, W4, W5)
    df_b = df_h2[df_h2["sequence_length"] == 16].sort_values("batch_size")
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
    stmoe_sub = df_b[df_b["predictor"] == "stmoe"]
    cstp_sub = df_b[df_b["predictor"] == "cstp"]

    ax2.plot(stmoe_sub["batch_size"], stmoe_sub["associated_coverage_pct"], marker="o", color=COLOR_STMOE, linewidth=2, label="ST-MoE (H=2)")
    ax2.plot(cstp_sub["batch_size"], cstp_sub["associated_coverage_pct"], marker="^", color=COLOR_CSTP, linewidth=2, label="CSTP (H=2)")
    ax2.set_xlabel("Serving Batch Size", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Associated Demand Coverage (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Associated Coverage vs Serving Batch Size (T=16)", fontsize=11, fontweight="bold")
    ax2.set_xticks([14, 32, 48, 70])
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    base_path2 = os.path.join(out_dir, "fig_02b_coverage_vs_batch_size")
    save_dual_format(fig2, base_path2)

    # 3. Coverage vs Sequence Length (B=70: W5, W6, W7)
    df_t = df_h2[df_h2["batch_size"] == 70].sort_values("sequence_length")
    fig3, ax3 = plt.subplots(figsize=(6.5, 4.2))
    stmoe_t = df_t[df_t["predictor"] == "stmoe"]
    cstp_t = df_t[df_t["predictor"] == "cstp"]

    ax3.plot(stmoe_t["sequence_length"], stmoe_t["associated_coverage_pct"], marker="o", color=COLOR_STMOE, linewidth=2, label="ST-MoE (H=2)")
    ax3.plot(cstp_t["sequence_length"], cstp_t["associated_coverage_pct"], marker="^", color=COLOR_CSTP, linewidth=2, label="CSTP (H=2)")
    ax3.set_xlabel("Sequence Length (T decode tokens)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Associated Demand Coverage (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Associated Coverage vs Generation Sequence Length (B=70)", fontsize=11, fontweight="bold")
    ax3.set_xticks([16, 32, 64])
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()
    base_path3 = os.path.join(out_dir, "fig_02c_coverage_vs_sequence_length")
    save_dual_format(fig3, base_path3)

# ==============================================================================
# SECTION 3: Timeliness & Transit Time
# ==============================================================================

def generate_timeliness_plots(df_lw, out_dir):
    # W1 to W7: Timely Rate vs Late Rate
    workloads = ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(workloads))
    w = 0.35

    # All matched prefetches were 100% late in flight across all runs
    late_rates = [100.0 for _ in workloads]
    timely_rates = [0.0 for _ in workloads]

    ax.bar(x - w/2, timely_rates, w, label="Timely of Matched = 0.0% (Completed before demand)", color="#2ca02c", edgecolor="#222222")
    ax.bar(x + w/2, late_rates, w, label="Late of Matched = 100.0% (In-flight at demand arrival)", color="#ff7f0e", edgecolor="#222222")
    ax.set_ylabel("Percentage of Matched Prefetches (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Workload ID", fontsize=11, fontweight="bold")
    ax.set_title("Timeliness Breakdown: Timely vs Late Matched Prefetches\n(Physical transfer time 1.376 ms exceeds inter-stage compute)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.5)

    base_path = os.path.join(out_dir, "fig_03a_timely_and_late_rates")
    save_dual_format(fig, base_path)

    # Figure 3b: In-Flight Bus Transit Time at Demand Arrival (W7 Head-to-Head)
    # CSTP = 66.97 us in transit, ST-MoE = 3.81 us in transit
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    methods = ["ST-MoE (H=2)", "CSTP (H=2)"]
    times = [3.81, 66.97]
    bars = ax2.bar(methods, times, color=[COLOR_STMOE, COLOR_CSTP], width=0.45, edgecolor="#222222")
    ax2.set_ylabel("Mean Transit Time at Demand Arrival (us)", fontsize=11, fontweight="bold")
    ax2.set_title("Workload 7: Physical Bus Transit Progress at Demand\n(CSTP avoided queue head-of-line blocking: 17.6x deeper progress)", fontsize=10.5, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.5)
    for b in bars:
        h = b.get_height()
        ax2.annotate(f"{h:.2f} us", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold")

    base_path2 = os.path.join(out_dir, "fig_03b_in_flight_transit_time_comparison")
    save_dual_format(fig2, base_path2)

# ==============================================================================
# SECTION 4: Unused Prefetches
# ==============================================================================

def generate_unused_prefetches_plots(df_lw, out_dir):
    df_h2 = df_lw[df_lw["lookahead_H"] == 2].copy()
    workloads = ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]

    # 1. Unused Prefetch Rate (%)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(workloads))
    w = 0.35

    stmoe_unused = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "stmoe")]["unused_prefetch_rate_pct"].iloc[0] for wid in workloads]
    cstp_unused = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "cstp")]["unused_prefetch_rate_pct"].iloc[0] for wid in workloads]

    ax.bar(x - w/2, stmoe_unused, w, label="ST-MoE (H=2)", color=COLOR_STMOE, edgecolor="#222222")
    ax.bar(x + w/2, cstp_unused, w, label="CSTP (H=2)", color=COLOR_CSTP, edgecolor="#222222")
    ax.set_ylabel("Unused Completed Prefetch Rate (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Workload ID", fontsize=11, fontweight="bold")
    ax.set_title("Unused Speculative Prefetches across Workloads\n(Decreases with longer sequences as reuse window widens)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, fontsize=10, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.5)

    base_path = os.path.join(out_dir, "fig_04a_unused_prefetch_rate")
    save_dual_format(fig, base_path)

    # 2. Wasted Speculative Bytes (GB)
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    stmoe_wasted_gb = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "stmoe")]["wasted_speculative_gb"].iloc[0] for wid in workloads]
    cstp_wasted_gb = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "cstp")]["wasted_speculative_gb"].iloc[0] for wid in workloads]

    ax2.bar(x - w/2, stmoe_wasted_gb, w, label="ST-MoE Wasted Traffic (GB)", color=COLOR_STMOE, edgecolor="#222222")
    ax2.bar(x + w/2, cstp_wasted_gb, w, label="CSTP Wasted Traffic (GB)", color=COLOR_CSTP, edgecolor="#222222")
    ax2.set_ylabel("Wasted Speculative DDR5 Traffic (GB)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Workload ID", fontsize=11, fontweight="bold")
    ax2.set_title("Wasted Speculative Memory Bandwidth (Completed but Never Demanded)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(workloads, fontsize=10, fontweight="bold")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.5)

    base_path2 = os.path.join(out_dir, "fig_04b_wasted_speculative_bytes")
    save_dual_format(fig2, base_path2)

# ==============================================================================
# SECTION 5: HBM Hit Rate
# ==============================================================================

def generate_hbm_hit_rate_plots(df_lw, out_dir):
    workloads = ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]
    df_h2 = df_lw[df_lw["lookahead_H"].isin([0, 2])].copy()

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(workloads))
    w = 0.25

    base_hits = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "baseline")]["hbm_hit_rate_pct"].iloc[0] for wid in workloads]
    stmoe_hits = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "stmoe")]["hbm_hit_rate_pct"].iloc[0] for wid in workloads]
    cstp_hits = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "cstp")]["hbm_hit_rate_pct"].iloc[0] for wid in workloads]

    ax.bar(x - w, base_hits, w, label="Baseline (67.0%)", color=COLOR_BASELINE, edgecolor="#222222")
    ax.bar(x, stmoe_hits, w, label="ST-MoE (67.0%)", color=COLOR_STMOE, edgecolor="#222222")
    ax.bar(x + w, cstp_hits, w, label="CSTP (67.0%)", color=COLOR_CSTP, edgecolor="#222222")

    ax.set_ylabel("HBM Cache Hit Rate (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Workload ID", fontsize=11, fontweight="bold")
    ax.set_title("HBM Expert Cache Hit Rate across Workloads\n(Invariance: Late attachments do not count as resident HBM hits)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.5)

    base_path = os.path.join(out_dir, "fig_05a_hbm_hit_rate_comparison")
    save_dual_format(fig, base_path)

    # Scaling with batch size at T=16
    df_b = df_h2[df_h2["sequence_length"] == 16].sort_values("batch_size")
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
    ax2.plot(df_b[df_b["predictor"] == "baseline"]["batch_size"], df_b[df_b["predictor"] == "baseline"]["hbm_hit_rate_pct"], marker="s", color=COLOR_BASELINE, label="Baseline", linewidth=2)
    ax2.plot(df_b[df_b["predictor"] == "stmoe"]["batch_size"], df_b[df_b["predictor"] == "stmoe"]["hbm_hit_rate_pct"], marker="o", color=COLOR_STMOE, label="ST-MoE", linewidth=2)
    ax2.plot(df_b[df_b["predictor"] == "cstp"]["batch_size"], df_b[df_b["predictor"] == "cstp"]["hbm_hit_rate_pct"], marker="^", color=COLOR_CSTP, label="CSTP", linewidth=2)
    ax2.set_xlabel("Serving Batch Size", fontsize=11, fontweight="bold")
    ax2.set_ylabel("HBM Hit Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title("HBM Hit Rate Scaling with Batch Size (Fixed T=16)", fontsize=11, fontweight="bold")
    ax2.set_xticks([14, 32, 48, 70])
    ax2.set_ylim(50, 80)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    base_path2 = os.path.join(out_dir, "fig_05b_hbm_hit_rate_scaling")
    save_dual_format(fig2, base_path2)

# ==============================================================================
# SECTION 6: Memory Traffic Plots
# ==============================================================================

def generate_memory_traffic_plots(df_lw, out_dir):
    df_h2 = df_lw[df_lw["lookahead_H"].isin([0, 2])].copy()
    workloads = ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]

    # 1. Total DDR5 Traffic Breakdown
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(workloads))
    w = 0.26

    base_tot = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "baseline")]["total_ddr5_gb"].iloc[0] for wid in workloads]
    stmoe_tot = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "stmoe")]["total_ddr5_gb"].iloc[0] for wid in workloads]
    cstp_tot = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "cstp")]["total_ddr5_gb"].iloc[0] for wid in workloads]

    ax.bar(x - w, base_tot, w, label="Baseline Demand Traffic", color=COLOR_BASELINE, edgecolor="#222222")
    ax.bar(x, stmoe_tot, w, label="ST-MoE Total DDR5 Traffic", color=COLOR_STMOE, edgecolor="#222222")
    ax.bar(x + w, cstp_tot, w, label="CSTP Total DDR5 Traffic", color=COLOR_CSTP, edgecolor="#222222")
    ax.set_ylabel("Total DDR5 Traffic (GB)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Workload ID", fontsize=11, fontweight="bold")
    ax.set_title("Total Physical DDR5 Memory Traffic across Workloads\n(Demand Loads + Speculative Prefetches)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, fontsize=10, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.5)

    base_path = os.path.join(out_dir, "fig_06a_ddr5_traffic_breakdown")
    save_dual_format(fig, base_path)

    # 2. Net Additional Traffic Comparison (GB)
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    w_half = 0.35
    stmoe_net = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "stmoe")]["net_additional_memory_traffic_gb"].iloc[0] for wid in workloads]
    cstp_net = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "cstp")]["net_additional_memory_traffic_gb"].iloc[0] for wid in workloads]

    ax2.bar(x - w_half/2, stmoe_net, w_half, label="ST-MoE Net Additional Traffic (GB)", color=COLOR_STMOE, edgecolor="#222222")
    ax2.bar(x + w_half/2, cstp_net, w_half, label="CSTP Net Additional Traffic (GB)", color=COLOR_CSTP, edgecolor="#222222")
    ax2.set_ylabel("Net Additional Memory Traffic (GB)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Workload ID", fontsize=11, fontweight="bold")
    ax2.set_title("Net Additional DDR5 Traffic: (Wasted Speculative - Prevented Demand)\n(CSTP generates 71.6% less additional traffic in W7)", fontsize=10.5, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(workloads, fontsize=10, fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(True, axis="y", alpha=0.5)

    base_path2 = os.path.join(out_dir, "fig_06b_net_additional_traffic_comparison")
    save_dual_format(fig2, base_path2)

    # 3. Traffic Overhead Percentage (%)
    fig3, ax3 = plt.subplots(figsize=(8, 4.2))
    stmoe_ovh = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "stmoe")]["traffic_overhead_pct"].iloc[0] for wid in workloads]
    cstp_ovh = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "cstp")]["traffic_overhead_pct"].iloc[0] for wid in workloads]

    ax3.plot(workloads, stmoe_ovh, marker="o", color=COLOR_STMOE, linewidth=2, label="ST-MoE Overhead (%)")
    ax3.plot(workloads, cstp_ovh, marker="^", color=COLOR_CSTP, linewidth=2, label="CSTP Overhead (%)")
    ax3.set_ylabel("Traffic Overhead vs Baseline (%)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Workload ID", fontsize=11, fontweight="bold")
    ax3.set_title("DDR5 Traffic Overhead Percentage across Workloads", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()
    base_path3 = os.path.join(out_dir, "fig_06c_traffic_overhead_percentage")
    save_dual_format(fig3, base_path3)

# ==============================================================================
# SECTION 7: Exposed Memory Stall
# ==============================================================================

def generate_stall_plots(df_lw, out_dir):
    df_h2 = df_lw[df_lw["lookahead_H"].isin([0, 2])].copy()
    workloads = ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]

    # 1. Exposed Stall vs Batch Size at T=16 (W2, W3, W4, W5)
    df_b = df_h2[df_h2["sequence_length"] == 16].sort_values("batch_size")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(df_b[df_b["predictor"] == "baseline"]["batch_size"], df_b[df_b["predictor"] == "baseline"]["exposed_memory_stall_ms"] / 1000.0, marker="s", color=COLOR_BASELINE, label="Baseline", linewidth=2)
    ax.plot(df_b[df_b["predictor"] == "stmoe"]["batch_size"], df_b[df_b["predictor"] == "stmoe"]["exposed_memory_stall_ms"] / 1000.0, marker="o", color=COLOR_STMOE, label="ST-MoE (H=2)", linewidth=2)
    ax.plot(df_b[df_b["predictor"] == "cstp"]["batch_size"], df_b[df_b["predictor"] == "cstp"]["exposed_memory_stall_ms"] / 1000.0, marker="^", color=COLOR_CSTP, label="CSTP (H=2)", linewidth=2)
    ax.set_xlabel("Serving Batch Size", fontsize=11, fontweight="bold")
    ax.set_ylabel("Exposed GPU Stall Time (seconds)", fontsize=11, fontweight="bold")
    ax.set_title("Exposed GPU Stall Time vs Batch Size (Fixed T=16)", fontsize=11, fontweight="bold")
    ax.set_xticks([14, 32, 48, 70])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    base_path = os.path.join(out_dir, "fig_07a_exposed_stall_vs_batch")
    save_dual_format(fig, base_path)

    # 2. Exposed Stall Reduction Scaling (W5 -> W6 -> W7 at B=70)
    df_t = df_h2[df_h2["batch_size"] == 70].sort_values("sequence_length")
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
    ax2.plot(df_t[df_t["predictor"] == "stmoe"]["sequence_length"], df_t[df_t["predictor"] == "stmoe"]["exposed_memory_stall_reduction_ms"], marker="o", color=COLOR_STMOE, label="ST-MoE (H=2)", linewidth=2)
    ax2.plot(df_t[df_t["predictor"] == "cstp"]["sequence_length"], df_t[df_t["predictor"] == "cstp"]["exposed_memory_stall_reduction_ms"], marker="^", color=COLOR_CSTP, label="CSTP (H=2)", linewidth=2)
    ax2.set_xlabel("Generation Sequence Length (T tokens)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Exposed Stall Reduction (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("Exposed Stall Reduction vs Sequence Length (B=70)\n(CSTP exhibits super-linear scaling to 6.74s)", fontsize=11, fontweight="bold")
    ax2.set_xticks([16, 32, 64])
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    base_path2 = os.path.join(out_dir, "fig_07b_exposed_stall_reduction_scaling")
    save_dual_format(fig2, base_path2)

    # 3. Stall Reduction Percentage (%)
    fig3, ax3 = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(workloads))
    w = 0.35
    stmoe_spct = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "stmoe")]["exposed_memory_stall_reduction_pct"].iloc[0] for wid in workloads]
    cstp_spct = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "cstp")]["exposed_memory_stall_reduction_pct"].iloc[0] for wid in workloads]

    ax3.bar(x - w/2, stmoe_spct, w, label="ST-MoE Stall Red (%)", color=COLOR_STMOE, edgecolor="#222222")
    ax3.bar(x + w/2, cstp_spct, w, label="CSTP Stall Red (%)", color=COLOR_CSTP, edgecolor="#222222")
    ax3.set_ylabel("Exposed Stall Reduction Percentage (%)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Workload ID", fontsize=11, fontweight="bold")
    ax3.set_title("Exposed Memory-Stall Reduction Percentage across Workloads", fontsize=11, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(workloads, fontsize=10, fontweight="bold")
    ax3.legend()
    ax3.grid(True, axis="y", alpha=0.5)
    base_path3 = os.path.join(out_dir, "fig_07c_stall_reduction_percentage")
    save_dual_format(fig3, base_path3)

# ==============================================================================
# SECTION 8: Expert Load Latency
# ==============================================================================

def generate_expert_load_latency_plots(df_lw, out_dir):
    df_h2 = df_lw[df_lw["lookahead_H"].isin([0, 2])].copy()
    workloads = ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]

    # Mean Expert Load Latency (us)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(workloads))
    w = 0.25

    base_lat = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "baseline")]["mean_expert_load_latency_us"].iloc[0] for wid in workloads]
    stmoe_lat = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "stmoe")]["mean_expert_load_latency_us"].iloc[0] for wid in workloads]
    cstp_lat = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "cstp")]["mean_expert_load_latency_us"].iloc[0] for wid in workloads]

    ax.bar(x - w, base_lat, w, label="Baseline (458.75 us)", color=COLOR_BASELINE, edgecolor="#222222")
    ax.bar(x, stmoe_lat, w, label="ST-MoE (H=2)", color=COLOR_STMOE, edgecolor="#222222")
    ax.bar(x + w, cstp_lat, w, label="CSTP (H=2)", color=COLOR_CSTP, edgecolor="#222222")
    ax.set_ylabel("Mean Expert Load Latency (us)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Workload ID", fontsize=11, fontweight="bold")
    ax.set_title("Mean Expert Load Latency across Workloads", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, fontsize=10, fontweight="bold")
    ax.set_ylim(440, 465)
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.5)

    base_path = os.path.join(out_dir, "fig_08a_mean_expert_load_latency")
    save_dual_format(fig, base_path)

    # Expert Load Latency Reduction (us)
    fig2, ax2 = plt.subplots(figsize=(8, 4.2))
    w_half = 0.35
    stmoe_red = [base_lat[i] - stmoe_lat[i] for i in range(len(workloads))]
    cstp_red = [base_lat[i] - cstp_lat[i] for i in range(len(workloads))]

    ax2.bar(x - w_half/2, stmoe_red, w_half, label="ST-MoE Reduction (us)", color=COLOR_STMOE, edgecolor="#222222")
    ax2.bar(x + w_half/2, cstp_red, w_half, label="CSTP Reduction (us)", color=COLOR_CSTP, edgecolor="#222222")
    ax2.set_ylabel("Expert Load Latency Reduction (us)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Workload ID", fontsize=11, fontweight="bold")
    ax2.set_title("Mean Expert Load Latency Reduction (Perceived Latency Savings per Demand)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(workloads, fontsize=10, fontweight="bold")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.5)

    base_path2 = os.path.join(out_dir, "fig_08b_expert_load_latency_reduction")
    save_dual_format(fig2, base_path2)

# ==============================================================================
# SECTION 9: End-to-End Latency
# ==============================================================================

def generate_end_to_end_latency_plots(df_lw, out_dir):
    df_h2 = df_lw[df_lw["lookahead_H"].isin([0, 2])].copy()
    workloads = ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]

    # 1. Mean Decode Step Latency (ms)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(workloads))
    w = 0.25

    base_tok = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "baseline")]["mean_decode_step_latency_ms"].iloc[0] for wid in workloads]
    stmoe_tok = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "stmoe")]["mean_decode_step_latency_ms"].iloc[0] for wid in workloads]
    cstp_tok = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "cstp")]["mean_decode_step_latency_ms"].iloc[0] for wid in workloads]

    ax.bar(x - w, base_tok, w, label="Baseline", color=COLOR_BASELINE, edgecolor="#222222")
    ax.bar(x, stmoe_tok, w, label="ST-MoE (H=2)", color=COLOR_STMOE, edgecolor="#222222")
    ax.bar(x + w, cstp_tok, w, label="CSTP (H=2)", color=COLOR_CSTP, edgecolor="#222222")
    ax.set_ylabel("Mean Decode Step Latency (ms)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Workload ID", fontsize=11, fontweight="bold")
    ax.set_title("Mean Decode Step Latency across Workloads", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, fontsize=10, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.5)

    base_path = os.path.join(out_dir, "fig_09a_mean_decode_step_latency")
    save_dual_format(fig, base_path)

    # 2. Decode Step Latency Reduction in W7
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    methods = ["ST-MoE (H=2)", "CSTP (H=2)"]
    reds = [base_tok[-1] - stmoe_tok[-1], base_tok[-1] - cstp_tok[-1]]
    bars = ax2.bar(methods, reds, color=[COLOR_STMOE, COLOR_CSTP], width=0.45, edgecolor="#222222")
    ax2.set_ylabel("Decode Step Latency Reduction (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("Workload 7: Decode Step Latency Reduction\n(CSTP saves 104.95 ms per step; ST-MoE saves 35.52 ms)", fontsize=10.5, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.5)
    for b in bars:
        h = b.get_height()
        ax2.annotate(f"-{h:.2f} ms", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold")

    base_path2 = os.path.join(out_dir, "fig_09b_decode_step_latency_reduction")
    save_dual_format(fig2, base_path2)

    # 3. P95 Tail Token Latency
    fig3, ax3 = plt.subplots(figsize=(8, 4.2))
    base_p95 = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "baseline")]["p95_decode_step_latency_ms"].iloc[0] for wid in workloads]
    stmoe_p95 = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "stmoe")]["p95_decode_step_latency_ms"].iloc[0] for wid in workloads]
    cstp_p95 = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "cstp")]["p95_decode_step_latency_ms"].iloc[0] for wid in workloads]

    ax3.plot(workloads, base_p95, marker="s", color=COLOR_BASELINE, label="Baseline", linewidth=2)
    ax3.plot(workloads, stmoe_p95, marker="o", color=COLOR_STMOE, label="ST-MoE", linewidth=2)
    ax3.plot(workloads, cstp_p95, marker="^", color=COLOR_CSTP, label="CSTP", linewidth=2)
    ax3.set_ylabel("P95 Tail Decode Latency (ms)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Workload ID", fontsize=11, fontweight="bold")
    ax3.set_title("P95 Tail Decode Step Latency Scaling", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()
    base_path3 = os.path.join(out_dir, "fig_09c_p95_tail_token_latency")
    save_dual_format(fig3, base_path3)

# ==============================================================================
# SECTION 10: Throughput
# ==============================================================================

def generate_throughput_plots(df_lw, out_dir):
    df_h2 = df_lw[df_lw["lookahead_H"].isin([0, 2])].copy()
    workloads = ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]

    # 1. Aggregate Throughput (tokens/s) across Workloads
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(workloads))
    w = 0.25

    base_tps = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "baseline")]["throughput_tokens_per_sec"].iloc[0] for wid in workloads]
    stmoe_tps = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "stmoe")]["throughput_tokens_per_sec"].iloc[0] for wid in workloads]
    cstp_tps = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "cstp")]["throughput_tokens_per_sec"].iloc[0] for wid in workloads]

    ax.bar(x - w, base_tps, w, label="Baseline", color=COLOR_BASELINE, edgecolor="#222222")
    ax.bar(x, stmoe_tps, w, label="ST-MoE (H=2)", color=COLOR_STMOE, edgecolor="#222222")
    ax.bar(x + w, cstp_tps, w, label="CSTP (H=2)", color=COLOR_CSTP, edgecolor="#222222")
    ax.set_ylabel("Inference Throughput (tokens/sec)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Workload ID", fontsize=11, fontweight="bold")
    ax.set_title("Aggregate Inference Throughput across Workloads", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, fontsize=10, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.5)

    base_path = os.path.join(out_dir, "fig_10a_throughput_tokens_per_sec")
    save_dual_format(fig, base_path)

    # 2. Throughput Scaling with Batch Size at T=16
    df_b = df_h2[df_h2["sequence_length"] == 16].sort_values("batch_size")
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
    ax2.plot(df_b[df_b["predictor"] == "baseline"]["batch_size"], df_b[df_b["predictor"] == "baseline"]["throughput_tokens_per_sec"], marker="s", color=COLOR_BASELINE, label="Baseline", linewidth=2)
    ax2.plot(df_b[df_b["predictor"] == "stmoe"]["batch_size"], df_b[df_b["predictor"] == "stmoe"]["throughput_tokens_per_sec"], marker="o", color=COLOR_STMOE, label="ST-MoE (H=2)", linewidth=2)
    ax2.plot(df_b[df_b["predictor"] == "cstp"]["batch_size"], df_b[df_b["predictor"] == "cstp"]["throughput_tokens_per_sec"], marker="^", color=COLOR_CSTP, label="CSTP (H=2)", linewidth=2)
    ax2.set_xlabel("Serving Batch Size", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Throughput (tokens/sec)", fontsize=11, fontweight="bold")
    ax2.set_title("Throughput Scaling with Serving Concurrency (Fixed T=16)", fontsize=11, fontweight="bold")
    ax2.set_xticks([14, 32, 48, 70])
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    base_path2 = os.path.join(out_dir, "fig_10b_throughput_scaling_with_batch")
    save_dual_format(fig2, base_path2)

# ==============================================================================
# SECTION 11: Comparison Dashboards
# ==============================================================================

def generate_dashboards(df_lw, df_orig, out_dir):
    # Dashboard 1: Large Workload Comprehensive Overview (6 Panels)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    workloads = ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]
    df_h2 = df_lw[df_lw["lookahead_H"].isin([0, 2])].copy()
    x = np.arange(len(workloads))
    w = 0.25

    # Panel 1: Simulation Time
    ax = axes[0, 0]
    b_time = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "baseline")]["total_simulated_time_s"].iloc[0] for wid in workloads]
    s_time = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "stmoe")]["total_simulated_time_s"].iloc[0] for wid in workloads]
    c_time = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "cstp")]["total_simulated_time_s"].iloc[0] for wid in workloads]
    ax.bar(x - w, b_time, w, label="Baseline", color=COLOR_BASELINE)
    ax.bar(x, s_time, w, label="ST-MoE", color=COLOR_STMOE)
    ax.bar(x + w, c_time, w, label="CSTP", color=COLOR_CSTP)
    ax.set_ylabel("Simulated Time (s)", fontweight="bold")
    ax.set_title("A. Total Simulated Time", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.5)

    # Panel 2: Stall Reduction
    ax = axes[0, 1]
    s_stall = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "stmoe")]["exposed_memory_stall_reduction_ms"].iloc[0] for wid in workloads]
    c_stall = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "cstp")]["exposed_memory_stall_reduction_ms"].iloc[0] for wid in workloads]
    w_half = 0.35
    ax.bar(x - w_half/2, s_stall, w_half, label="ST-MoE", color=COLOR_STMOE)
    ax.bar(x + w_half/2, c_stall, w_half, label="CSTP", color=COLOR_CSTP)
    ax.set_ylabel("Stall Reduction (ms)", fontweight="bold")
    ax.set_title("B. Exposed Memory-Stall Reduction", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.5)

    # Panel 3: Associated Coverage
    ax = axes[0, 2]
    s_cov = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "stmoe")]["associated_coverage_pct"].iloc[0] for wid in workloads]
    c_cov = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "cstp")]["associated_coverage_pct"].iloc[0] for wid in workloads]
    ax.bar(x - w_half/2, s_cov, w_half, label="ST-MoE", color=COLOR_STMOE)
    ax.bar(x + w_half/2, c_cov, w_half, label="CSTP", color=COLOR_CSTP)
    ax.axhline(0, color="black", linestyle="-", label="Strict Timely = 0.0%")
    ax.set_ylabel("Associated Coverage (%)", fontweight="bold")
    ax.set_title("C. Prefetch-Associated Coverage", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.5)

    # Panel 4: Net Additional DDR5 Traffic
    ax = axes[1, 0]
    s_net = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "stmoe")]["net_additional_memory_traffic_gb"].iloc[0] for wid in workloads]
    c_net = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "cstp")]["net_additional_memory_traffic_gb"].iloc[0] for wid in workloads]
    ax.bar(x - w_half/2, s_net, w_half, label="ST-MoE", color=COLOR_STMOE)
    ax.bar(x + w_half/2, c_net, w_half, label="CSTP", color=COLOR_CSTP)
    ax.set_ylabel("Net Extra Traffic (GB)", fontweight="bold")
    ax.set_title("D. Net Additional DDR5 Traffic", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.5)

    # Panel 5: Unused Prefetches
    ax = axes[1, 1]
    s_un = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "stmoe")]["unused_prefetch_rate_pct"].iloc[0] for wid in workloads]
    c_un = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "cstp")]["unused_prefetch_rate_pct"].iloc[0] for wid in workloads]
    ax.bar(x - w_half/2, s_un, w_half, label="ST-MoE", color=COLOR_STMOE)
    ax.bar(x + w_half/2, c_un, w_half, label="CSTP", color=COLOR_CSTP)
    ax.set_ylabel("Unused Prefetches (%)", fontweight="bold")
    ax.set_title("E. Unused Completed Prefetch Rate", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.5)

    # Panel 6: Throughput
    ax = axes[1, 2]
    b_tps = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "baseline")]["throughput_tokens_per_sec"].iloc[0] for wid in workloads]
    s_tps = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "stmoe")]["throughput_tokens_per_sec"].iloc[0] for wid in workloads]
    c_tps = [df_h2[(df_h2["workload_id"] == wid) & (df_h2["predictor"] == "cstp")]["throughput_tokens_per_sec"].iloc[0] for wid in workloads]
    ax.bar(x - w, b_tps, w, label="Baseline", color=COLOR_BASELINE)
    ax.bar(x, s_tps, w, label="ST-MoE", color=COLOR_STMOE)
    ax.bar(x + w, c_tps, w, label="CSTP", color=COLOR_CSTP)
    ax.set_ylabel("Throughput (tok/s)", fontweight="bold")
    ax.set_title("F. Aggregate Inference Throughput", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.5)

    plt.suptitle("Point 3 — Large Workload Campaign Comprehensive Dashboard\n(DeepSeek-V3 40 MoE Stages, 35 GB HBM Cache, A100 Simulation)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    base_path = os.path.join(out_dir, "dashboard_large_workloads_overall")
    save_dual_format(fig, base_path)

    # Dashboard 2: Workload 7 Head-to-Head Comparison (4 Panels)
    w7_sub = df_lw[df_lw["workload_id"] == "W7"].copy()
    w7_h2 = w7_sub[w7_sub["lookahead_H"].isin([0, 2])]

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))

    # Panel 1: Simulation Time & Stall Reduction
    ax = axes2[0, 0]
    preds = ["Baseline", "ST-MoE", "CSTP"]
    sim_t = [w7_h2[w7_h2["predictor"] == "baseline"]["total_simulated_time_s"].iloc[0],
             w7_h2[w7_h2["predictor"] == "stmoe"]["total_simulated_time_s"].iloc[0],
             w7_h2[w7_h2["predictor"] == "cstp"]["total_simulated_time_s"].iloc[0]]
    cols = [COLOR_BASELINE, COLOR_STMOE, COLOR_CSTP]
    bars = ax.bar(preds, sim_t, color=cols, width=0.45, edgecolor="#222222")
    ax.set_ylabel("Total Simulated Time (s)", fontweight="bold")
    ax.set_title("A. W7 Simulated Execution Time\n(CSTP: -6.717s | ST-MoE: -2.273s)", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.5)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.2f}s", (b.get_x() + b.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)

    # Panel 2: Exposed Stall Reduction
    ax = axes2[0, 1]
    stall_reds = [0.0,
                  w7_h2[w7_h2["predictor"] == "stmoe"]["exposed_memory_stall_reduction_ms"].iloc[0],
                  w7_h2[w7_h2["predictor"] == "cstp"]["exposed_memory_stall_reduction_ms"].iloc[0]]
    bars = ax.bar(preds, stall_reds, color=cols, width=0.45, edgecolor="#222222")
    ax.set_ylabel("Exposed Stall Reduction (ms)", fontweight="bold")
    ax.set_title("B. W7 Memory-Stall Reduction\n(CSTP: 2.94x greater stall reduction)", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.5)
    for b in bars:
        h = b.get_height()
        if h > 0:
            ax.annotate(f"+{h:.1f}ms", (b.get_x() + b.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color="#8b0000")

    # Panel 3: Dispatched Prefetches vs Useful Hits
    ax = axes2[1, 0]
    preds2 = ["ST-MoE", "CSTP"]
    disp = [w7_h2[w7_h2["predictor"] == "stmoe"]["dispatched_prefetches"].iloc[0],
            w7_h2[w7_h2["predictor"] == "cstp"]["dispatched_prefetches"].iloc[0]]
    hits = [w7_h2[w7_h2["predictor"] == "stmoe"]["prefetch_assisted_demands"].iloc[0],
            w7_h2[w7_h2["predictor"] == "cstp"]["prefetch_assisted_demands"].iloc[0]]
    x2 = np.arange(2)
    w2 = 0.35
    ax.bar(x2 - w2/2, disp, w2, label="Dispatched Prefetches", color=["#4682b4", "#cd5c5c"], edgecolor="#222222")
    ax.bar(x2 + w2/2, hits, w2, label="Assisted Demands (Late Hits)", color=[COLOR_STMOE, COLOR_CSTP], edgecolor="#222222")
    ax.set_ylabel("Prefetch Event Count", fontweight="bold")
    ax.set_title("C. Dispatched vs Useful Assisted Demands\n(CSTP dispatched 47.4% fewer transfers)", fontweight="bold")
    ax.set_xticks(x2)
    ax.set_xticklabels(preds2)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.5)

    # Panel 4: DDR5 Traffic Overhead
    ax = axes2[1, 1]
    net_traffic = [w7_h2[w7_h2["predictor"] == "stmoe"]["net_additional_memory_traffic_gb"].iloc[0],
                   w7_h2[w7_h2["predictor"] == "cstp"]["net_additional_memory_traffic_gb"].iloc[0]]
    bars = ax.bar(preds2, net_traffic, color=[COLOR_STMOE, COLOR_CSTP], width=0.45, edgecolor="#222222")
    ax.set_ylabel("Net Additional DDR5 Traffic (GB)", fontweight="bold")
    ax.set_title("D. Net DDR5 Traffic Overhead\n(ST-MoE: +420.32 GB | CSTP: +119.53 GB)", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.5)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"+{h:.1f} GB", (b.get_x() + b.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Workload 7 Head-to-Head Comparison: B=70, T=64, 40 MoE Stages, 35 GB HBM\n(CSTP achieves higher stall reduction with lower traffic overhead)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    base_path2 = os.path.join(out_dir, "dashboard_w7_deepseek_v3_head_to_head")
    save_dual_format(fig2, base_path2)

    # Dashboard 3: Original Campaign Summary (4 Panels)
    if not df_orig.empty:
        fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5))
        batches_orig = [1, 2, 4, 8]
        df_orig_h1 = df_orig[df_orig["lookahead_H"].isin([0, 1])]

        # Panel 1: Exposed Stall
        ax = axes3[0]
        for p, col, mark in [("baseline", COLOR_BASELINE, "s"), ("stmoe", COLOR_STMOE, "o"), ("cstp", COLOR_CSTP, "^")]:
            sub = df_orig_h1[df_orig_h1["predictor"] == p].sort_values("batch_size")
            ax.plot(sub["batch_size"], sub["exposed_memory_stall_ms"], marker=mark, color=col, linewidth=2, label=p.upper())
        ax.set_ylabel("Exposed Stall (ms)", fontweight="bold")
        ax.set_xlabel("Batch Size", fontweight="bold")
        ax.set_title("A. Exposed Stall (Identical Curves)", fontweight="bold")
        ax.set_xticks(batches_orig)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        # Panel 2: Token Latency
        ax = axes3[1]
        for p, col, mark in [("baseline", COLOR_BASELINE, "s"), ("stmoe", COLOR_STMOE, "o"), ("cstp", COLOR_CSTP, "^")]:
            sub = df_orig_h1[df_orig_h1["predictor"] == p].sort_values("batch_size")
            ax.plot(sub["batch_size"], sub["mean_decode_step_latency_ms"], marker=mark, color=col, linewidth=2, label=p.upper())
        ax.set_ylabel("Token Latency (ms)", fontweight="bold")
        ax.set_xlabel("Batch Size", fontweight="bold")
        ax.set_title("B. Token Latency (Identical Curves)", fontweight="bold")
        ax.set_xticks(batches_orig)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        # Panel 3: DDR5 Traffic (MB)
        ax = axes3[2]
        for p, col, mark in [("baseline", COLOR_BASELINE, "s"), ("stmoe", COLOR_STMOE, "o"), ("cstp", COLOR_CSTP, "^")]:
            sub = df_orig_h1[df_orig_h1["predictor"] == p].sort_values("batch_size")
            ax.plot(sub["batch_size"], sub["total_ddr5_bytes"] / 1e6, marker=mark, color=col, linewidth=2, label=p.upper())
        ax.set_ylabel("DDR5 Traffic (MB)", fontweight="bold")
        ax.set_xlabel("Batch Size", fontweight="bold")
        ax.set_title("C. DDR5 Traffic (Predictors Wasted Traffic)", fontweight="bold")
        ax.set_xticks(batches_orig)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        plt.suptitle("Original Campaign Dashboard: 3 MoE Stages, 4 Tokens, 80 GB Cache\n(Zero Prefetching Benefit: Unconstrained Cache, Truncated Topology)", fontsize=12, fontweight="bold", y=1.03)
        plt.tight_layout()
        base_path3 = os.path.join(out_dir, "dashboard_original_campaign")
        save_dual_format(fig3, base_path3)

def main():
    base_dir = "/home/mohman/ca_project/mohman/point_3/final_results"
    cons_csv_path = os.path.join(base_dir, "data/consolidated_csv/all_experiment_results.csv")
    plots_base = os.path.join(base_dir, "plots")

    setup_plot_style()

    print("=== Generating Point 3 Final Plots ===")
    if not os.path.exists(cons_csv_path):
        print(f"[ERROR] Consolidated CSV not found: {cons_csv_path}")
        sys.exit(1)

    df_all = pd.read_csv(cons_csv_path)
    df_lw = df_all[df_all["campaign"] == "large_workload"].copy()
    df_orig = df_all[df_all["campaign"] == "original_campaign"].copy()

    # 1. Total Time vs Stall Reduction
    dir_01 = os.path.join(plots_base, "01_total_time_vs_stall_reduction")
    generate_per_workload_plots(df_lw, dir_01)
    generate_all_workloads_overview(df_lw, dir_01)
    generate_tradeoff_scatter(df_lw, dir_01)
    generate_batch_scaling_plots(df_lw, dir_01)

    # 2. Prefetch Coverage
    dir_02 = os.path.join(plots_base, "02_prefetch_coverage")
    generate_coverage_plots(df_lw, dir_02)

    # 3. Timeliness
    dir_03 = os.path.join(plots_base, "03_timeliness")
    generate_timeliness_plots(df_lw, dir_03)

    # 4. Unused Prefetches
    dir_04 = os.path.join(plots_base, "04_unused_prefetches")
    generate_unused_prefetches_plots(df_lw, dir_04)

    # 5. HBM Hit Rate
    dir_05 = os.path.join(plots_base, "05_hbm_hit_rate")
    generate_hbm_hit_rate_plots(df_lw, dir_05)

    # 6. Memory Traffic
    dir_06 = os.path.join(plots_base, "06_memory_traffic")
    generate_memory_traffic_plots(df_lw, dir_06)

    # 7. Exposed Memory Stall
    dir_07 = os.path.join(plots_base, "07_exposed_memory_stall")
    generate_stall_plots(df_lw, dir_07)

    # 8. Expert Load Latency
    dir_08 = os.path.join(plots_base, "08_expert_load_latency")
    generate_expert_load_latency_plots(df_lw, dir_08)

    # 9. End-to-End Latency
    dir_09 = os.path.join(plots_base, "09_end_to_end_latency")
    generate_end_to_end_latency_plots(df_lw, dir_09)

    # 10. Throughput
    dir_10 = os.path.join(plots_base, "10_throughput")
    generate_throughput_plots(df_lw, dir_10)

    # 11. Dashboards
    dir_dash = os.path.join(plots_base, "comparison_dashboards")
    generate_dashboards(df_lw, df_orig, dir_dash)

    print("=== All Plots Generated Successfully ===")

if __name__ == "__main__":
    main()

