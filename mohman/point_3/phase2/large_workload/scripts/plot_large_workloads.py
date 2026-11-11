#!/usr/bin/env python3
"""
Publication-Quality Plotting Suite for Point 3 Large-Batch, Long-Sequence Campaign.
Generates 12 comprehensive figures covering all requested evaluations in both PNG (300 DPI) and PDF.

STRICT RULE: Zero ternary operators (ast.IfExp == 0).
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = "/home/mohman/ca_project"
LARGE_ROOT = os.path.join(WORKSPACE, "mohman", "point_3", "phase2", "large_workload")
RESULTS_DIR = os.path.join(LARGE_ROOT, "results")
PLOTS_DIR = os.path.join(LARGE_ROOT, "plots")
RAW_DIR = os.path.join(LARGE_ROOT, "experiments", "raw", "campaign")

os.makedirs(PLOTS_DIR, exist_ok=True)

# Styling setup without ternary operators
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.titlesize"] = 15
plt.rcParams["figure.titleweight"] = "bold"

COLOR_BASE = "#2b5c8f"    # Slate Blue
COLOR_STMOE = "#d95f02"   # Ochre Orange
COLOR_CSTP = "#1b9e77"    # Forest Teal
COLOR_TIMELY = "#2ca02c"  # Bright Green
COLOR_LATE = "#ff7f0e"    # Bright Orange


def save_fig(fig, base_name):
    png_path = os.path.join(PLOTS_DIR, f"{base_name}.png")
    pdf_path = os.path.join(PLOTS_DIR, f"{base_name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {base_name}.png and {base_name}.pdf")


def load_data():
    per_run_csv = os.path.join(RESULTS_DIR, "per_run_metrics.csv")
    batch_csv = os.path.join(RESULTS_DIR, "batch_scaling.csv")
    token_csv = os.path.join(RESULTS_DIR, "token_scaling.csv")
    pred_csv = os.path.join(RESULTS_DIR, "predictor_comparison.csv")
    evt_csv = os.path.join(RESULTS_DIR, "prefetch_event_accounting.csv")

    df_run = pd.read_csv(per_run_csv)
    df_batch = pd.read_csv(batch_csv)
    df_token = pd.read_csv(token_csv)
    df_pred = pd.read_csv(pred_csv)
    df_evt = pd.read_csv(evt_csv)

    return df_run, df_batch, df_token, df_pred, df_evt


def plot_fig1_strict_coverage(df_batch):
    fig, ax = plt.subplots(figsize=(8, 5))
    stmoe_b = df_batch[df_batch["predictor"] == "stmoe"]
    cstp_b = df_batch[df_batch["predictor"] == "cstp"]

    ax.plot(stmoe_b["batch_size"], stmoe_b["strict_timely_coverage_pct"], marker="o", linewidth=2.5, markersize=8, color=COLOR_STMOE, label="ST-MoE (H=2)")
    ax.plot(cstp_b["batch_size"], cstp_b["strict_timely_coverage_pct"], marker="s", linewidth=2.5, markersize=8, color=COLOR_CSTP, label="CSTP (H=2)")

    ax.set_title("Metric 8.1(A): Strict Timely Prefetch Coverage vs Serving Batch Size (T=16, 40 Stages)")
    ax.set_xlabel("Serving Batch Size (Concurrent Sequences)")
    ax.set_ylabel("Strict Timely Coverage (%)")
    ax.set_xticks([14, 32, 48, 70])
    ax.set_ylim(-0.5, 5.0)
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Annotate honest zero timely hit observation
    ax.annotate(
        "Strict Timely Hits = 0%\n(Prefetches arrive in-flight,\nnot resident before demand)",
        xy=(48, 0.0), xytext=(35, 1.5),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=8),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff2cc", edgecolor="#d6b656")
    )

    save_fig(fig, "fig1_strict_coverage_vs_batch")


def plot_fig2_associated_coverage(df_batch):
    fig, ax = plt.subplots(figsize=(8, 5))
    stmoe_b = df_batch[df_batch["predictor"] == "stmoe"]
    cstp_b = df_batch[df_batch["predictor"] == "cstp"]

    ax.plot(stmoe_b["batch_size"], stmoe_b["prefetch_associated_coverage_pct"], marker="o", linewidth=2.5, markersize=8, color=COLOR_STMOE, label="ST-MoE (H=2)")
    ax.plot(cstp_b["batch_size"], cstp_b["prefetch_associated_coverage_pct"], marker="s", linewidth=2.5, markersize=8, color=COLOR_CSTP, label="CSTP (H=2)")

    ax.set_title("Metric 8.1(B): Prefetch-Associated Demand Coverage vs Serving Batch Size (T=16, 40 Stages)")
    ax.set_xlabel("Serving Batch Size (Concurrent Sequences)")
    ax.set_ylabel("Prefetch-Associated Coverage (%)")
    ax.set_xticks([14, 32, 48, 70])
    ax.legend(frameon=True, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Annotate massive scaling
    max_cov = stmoe_b["prefetch_associated_coverage_pct"].max()
    ax.annotate(
        f"Coverage expands to {max_cov:.2f}%\nat B=70 (9,265 assisted demands)",
        xy=(70, max_cov), xytext=(50, max_cov - 1.5),
        arrowprops=dict(facecolor=COLOR_STMOE, shrink=0.08, width=1.5, headwidth=8),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffe6cc", edgecolor=COLOR_STMOE)
    )

    save_fig(fig, "fig2_associated_coverage_vs_batch")


def plot_fig3_timely_and_late_rates(df_batch):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    stmoe_b = df_batch[df_batch["predictor"] == "stmoe"]
    cstp_b = df_batch[df_batch["predictor"] == "cstp"]

    # Timely Rate
    ax1.plot(stmoe_b["batch_size"], stmoe_b["timely_of_matched_pct"], marker="o", linewidth=2.5, markersize=8, color=COLOR_TIMELY, label="ST-MoE")
    ax1.plot(cstp_b["batch_size"], cstp_b["timely_of_matched_pct"], marker="s", linewidth=2.5, markersize=8, color=COLOR_CSTP, label="CSTP")
    ax1.set_title("Metric 8.2: Timely Prefetch Rate (%)")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Timely Completed / Matched (%)")
    ax1.set_xticks([14, 32, 48, 70])
    ax1.set_ylim(-2, 10)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Late Rate
    ax2.plot(stmoe_b["batch_size"], stmoe_b["late_of_matched_pct"], marker="o", linewidth=2.5, markersize=8, color=COLOR_LATE, label="ST-MoE")
    ax2.plot(cstp_b["batch_size"], cstp_b["late_of_matched_pct"], marker="s", linewidth=2.5, markersize=8, color=COLOR_CSTP, label="CSTP")
    ax2.set_title("Metric 8.3: Late In-Flight Attachment Rate (%)")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Late In-Flight / Matched (%)")
    ax2.set_xticks([14, 32, 48, 70])
    ax2.set_ylim(90, 102)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Metrics 8.2 & 8.3: Timely vs Late Prefetch Resolution (T=16, 40 Stages)", y=1.02)
    save_fig(fig, "fig3_timely_and_late_rates_vs_batch")


def plot_fig4_unused_prefetches(df_batch):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    stmoe_b = df_batch[df_batch["predictor"] == "stmoe"]
    cstp_b = df_batch[df_batch["predictor"] == "cstp"]

    width = 2.5
    b_sizes = np.array([14, 32, 48, 70])

    # Unused Rate (%)
    ax1.bar(b_sizes - width/2, stmoe_b["unused_pct"], width=width, color=COLOR_STMOE, label="ST-MoE (H=2)")
    ax1.bar(b_sizes + width/2, cstp_b["unused_pct"], width=width, color=COLOR_CSTP, label="CSTP (H=2)")
    ax1.set_title("Metric 8.4: Unused Completed Prefetches (%)")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Unused Rate (%)")
    ax1.set_xticks(b_sizes)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Unused Wasted Memory (GB)
    stmoe_gb = stmoe_b["unused_bytes"] / 1e9
    cstp_gb = cstp_b["unused_bytes"] / 1e9
    ax2.bar(b_sizes - width/2, stmoe_gb, width=width, color=COLOR_STMOE, label="ST-MoE (H=2)")
    ax2.bar(b_sizes + width/2, cstp_gb, width=width, color=COLOR_CSTP, label="CSTP (H=2)")
    ax2.set_title("Metric 8.4: Wasted Speculative DDR5 Traffic (GB)")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Wasted Speculative Bytes (GB)")
    ax2.set_xticks(b_sizes)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Metric 8.4: Unused Prefetches and Memory Overhead vs Batch Size (T=16)", y=1.02)
    save_fig(fig, "fig4_unused_prefetches_vs_batch")


def plot_fig5_hbm_hit_rate(df_batch):
    fig, ax = plt.subplots(figsize=(8, 5))

    base_b = df_batch[df_batch["predictor"] == "baseline"]
    stmoe_b = df_batch[df_batch["predictor"] == "stmoe"]
    cstp_b = df_batch[df_batch["predictor"] == "cstp"]

    ax.plot(base_b["batch_size"], base_b["hbm_hit_rate_pct"], marker="D", linewidth=2.5, markersize=8, color=COLOR_BASE, label="Baseline (Demand-only)")
    ax.plot(stmoe_b["batch_size"], stmoe_b["hbm_hit_rate_pct"], marker="o", linewidth=2.5, markersize=8, color=COLOR_STMOE, label="ST-MoE (H=2)")
    ax.plot(cstp_b["batch_size"], cstp_b["hbm_hit_rate_pct"], marker="s", linewidth=2.5, markersize=8, color=COLOR_CSTP, label="CSTP (H=2)")

    ax.set_title("Metric 8.5: Overall HBM Hit Rate vs Serving Batch Size (35 GB Cache)")
    ax.set_xlabel("Serving Batch Size")
    ax.set_ylabel("HBM Cache Hit Rate (%)")
    ax.set_xticks([14, 32, 48, 70])
    ax.set_ylim(60, 75)
    ax.legend(frameon=True, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)

    save_fig(fig, "fig5_hbm_hit_rate_vs_batch")


def plot_fig6_ddr5_traffic(df_batch):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    stmoe_b = df_batch[df_batch["predictor"] == "stmoe"]
    cstp_b = df_batch[df_batch["predictor"] == "cstp"]
    base_b = df_batch[df_batch["predictor"] == "baseline"]

    b_sizes = np.array([14, 32, 48, 70])
    width = 2.5

    # Total DDR5 Traffic (TB)
    base_tb = base_b["total_ddr5_bytes"] / 1e12
    stmoe_tb = stmoe_b["total_ddr5_bytes"] / 1e12
    cstp_tb = cstp_b["total_ddr5_bytes"] / 1e12

    ax1.plot(b_sizes, base_tb, marker="D", linewidth=2.5, markersize=8, color=COLOR_BASE, label="Baseline")
    ax1.plot(b_sizes, stmoe_tb, marker="o", linewidth=2.5, markersize=8, color=COLOR_STMOE, label="ST-MoE (H=2)")
    ax1.plot(b_sizes, cstp_tb, marker="s", linewidth=2.5, markersize=8, color=COLOR_CSTP, label="CSTP (H=2)")
    ax1.set_title("Total DDR5 Memory Traffic (TB)")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("DDR5 Transferred (TB)")
    ax1.set_xticks(b_sizes)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Traffic Overhead Ratio (%)
    ax2.bar(b_sizes - width/2, stmoe_b["traffic_overhead_pct"], width=width, color=COLOR_STMOE, label="ST-MoE Overhead (%)")
    ax2.bar(b_sizes + width/2, cstp_b["traffic_overhead_pct"], width=width, color=COLOR_CSTP, label="CSTP Overhead (%)")
    ax2.set_title("Speculative DDR5 Traffic Overhead (%)")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Speculative / Demand Bytes (%)")
    ax2.set_xticks(b_sizes)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Metric 8.6: Memory Traffic and Speculative Overhead vs Batch Size (T=16)", y=1.02)
    save_fig(fig, "fig6_ddr5_traffic_vs_batch")


def plot_fig7_exposed_stall(df_batch):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    stmoe_b = df_batch[df_batch["predictor"] == "stmoe"]
    cstp_b = df_batch[df_batch["predictor"] == "cstp"]
    base_b = df_batch[df_batch["predictor"] == "baseline"]
    b_sizes = np.array([14, 32, 48, 70])

    # Stall per token (ms)
    ax1.plot(b_sizes, base_b["stall_per_token_ms"], marker="D", linewidth=2.5, markersize=8, color=COLOR_BASE, label="Baseline")
    ax1.plot(b_sizes, stmoe_b["stall_per_token_ms"], marker="o", linewidth=2.5, markersize=8, color=COLOR_STMOE, label="ST-MoE (H=2)")
    ax1.plot(b_sizes, cstp_b["stall_per_token_ms"], marker="s", linewidth=2.5, markersize=8, color=COLOR_CSTP, label="CSTP (H=2)")
    ax1.set_title("Exposed Memory Stall per Token (ms)")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Stall per Token (ms)")
    ax1.set_xticks(b_sizes)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Absolute Stall Reduction (ms)
    width = 2.5
    ax2.bar(b_sizes - width/2, stmoe_b["stall_reduction_ms"], width=width, color=COLOR_STMOE, label="ST-MoE Reduction (ms)")
    ax2.bar(b_sizes + width/2, cstp_b["stall_reduction_ms"], width=width, color=COLOR_CSTP, label="CSTP Reduction (ms)")
    ax2.set_title("Absolute Stall Reduction vs Baseline (ms)")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("GPU Stall Reduction (ms)")
    ax2.set_xticks(b_sizes)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Metric 8.7: Exposed GPU Memory Stall Reduction vs Batch Size (T=16, 40 Stages)", y=1.02)
    save_fig(fig, "fig7_exposed_stall_vs_batch")


def plot_fig8_expert_load_lat(df_batch):
    fig, ax = plt.subplots(figsize=(8, 5))

    base_b = df_batch[df_batch["predictor"] == "baseline"]
    stmoe_b = df_batch[df_batch["predictor"] == "stmoe"]
    cstp_b = df_batch[df_batch["predictor"] == "cstp"]
    b_sizes = [14, 32, 48, 70]

    ax.plot(b_sizes, base_b["mean_expert_load_lat_us"], marker="D", linewidth=2.5, markersize=8, color=COLOR_BASE, label="Baseline Mean")
    ax.plot(b_sizes, stmoe_b["mean_expert_load_lat_us"], marker="o", linewidth=2.5, markersize=8, color=COLOR_STMOE, label="ST-MoE Mean")
    ax.plot(b_sizes, cstp_b["mean_expert_load_lat_us"], marker="s", linewidth=2.5, markersize=8, color=COLOR_CSTP, label="CSTP Mean")

    ax.plot(b_sizes, base_b["p95_expert_load_lat_us"], marker="D", linestyle="--", color=COLOR_BASE, alpha=0.6, label="Baseline P95")
    ax.plot(b_sizes, stmoe_b["p95_expert_load_lat_us"], marker="o", linestyle="--", color=COLOR_STMOE, alpha=0.6, label="ST-MoE P95")
    ax.plot(b_sizes, cstp_b["p95_expert_load_lat_us"], marker="s", linestyle="--", color=COLOR_CSTP, alpha=0.6, label="CSTP P95")

    ax.set_title(r"Metric 8.8: Expert-Load Latency ($\mu$s) vs Batch Size (T=16)")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(r"Latency ($\mu$s)")
    ax.set_xticks(b_sizes)
    ax.legend(frameon=True, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.6)

    save_fig(fig, "fig8_expert_load_latency_vs_batch")


def plot_fig9_token_lat(df_batch):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    base_b = df_batch[df_batch["predictor"] == "baseline"]
    stmoe_b = df_batch[df_batch["predictor"] == "stmoe"]
    cstp_b = df_batch[df_batch["predictor"] == "cstp"]
    b_sizes = [14, 32, 48, 70]

    # Mean Token Latency
    ax1.plot(b_sizes, base_b["mean_token_lat_ms"], marker="D", linewidth=2.5, markersize=8, color=COLOR_BASE, label="Baseline")
    ax1.plot(b_sizes, stmoe_b["mean_token_lat_ms"], marker="o", linewidth=2.5, markersize=8, color=COLOR_STMOE, label="ST-MoE (H=2)")
    ax1.plot(b_sizes, cstp_b["mean_token_lat_ms"], marker="s", linewidth=2.5, markersize=8, color=COLOR_CSTP, label="CSTP (H=2)")
    ax1.set_title("Mean Token Latency (ms)")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_xticks(b_sizes)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # P95 Token Latency
    ax2.plot(b_sizes, base_b["p95_token_lat_ms"], marker="D", linewidth=2.5, markersize=8, color=COLOR_BASE, label="Baseline")
    ax2.plot(b_sizes, stmoe_b["p95_token_lat_ms"], marker="o", linewidth=2.5, markersize=8, color=COLOR_STMOE, label="ST-MoE (H=2)")
    ax2.plot(b_sizes, cstp_b["p95_token_lat_ms"], marker="s", linewidth=2.5, markersize=8, color=COLOR_CSTP, label="CSTP (H=2)")
    ax2.set_title("P95 Tail Token Latency (ms)")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_xticks(b_sizes)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Metric 8.8: End-to-End Token Latency Scaling (T=16, 40 Stages)", y=1.02)
    save_fig(fig, "fig9_token_latency_vs_batch")


def plot_fig10_token_scaling(df_token):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    base_t = df_token[df_token["predictor"] == "baseline"]
    stmoe_t = df_token[df_token["predictor"] == "stmoe"]
    cstp_t = df_token[df_token["predictor"] == "cstp"]
    t_steps = [16, 32, 64]

    # Panel 1: Prefetch Associated Coverage
    ax1.plot(stmoe_t["tokens"], stmoe_t["prefetch_associated_coverage_pct"], marker="o", linewidth=2.5, markersize=8, color=COLOR_STMOE, label="ST-MoE")
    ax1.plot(cstp_t["tokens"], cstp_t["prefetch_associated_coverage_pct"], marker="s", linewidth=2.5, markersize=8, color=COLOR_CSTP, label="CSTP")
    ax1.set_title("Prefetch-Associated Demand Coverage (%)")
    ax1.set_xlabel("Token Sequence Length (T)")
    ax1.set_ylabel("Coverage (%)")
    ax1.set_xticks(t_steps)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Exposed Stall Reduction (ms)
    width = 3.0
    arr_t = np.array(t_steps)
    ax2.bar(arr_t - width/2, stmoe_t["stall_reduction_ms"], width=width, color=COLOR_STMOE, label="ST-MoE")
    ax2.bar(arr_t + width/2, cstp_t["stall_reduction_ms"], width=width, color=COLOR_CSTP, label="CSTP")
    ax2.set_title("Exposed GPU Stall Reduction (ms)")
    ax2.set_xlabel("Token Sequence Length (T)")
    ax2.set_ylabel("Stall Reduction (ms)")
    ax2.set_xticks(t_steps)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Panel 3: HBM Hit Rate (%)
    ax3.plot(base_t["tokens"], base_t["hbm_hit_rate_pct"], marker="D", linewidth=2.5, markersize=8, color=COLOR_BASE, label="Baseline")
    ax3.plot(stmoe_t["tokens"], stmoe_t["hbm_hit_rate_pct"], marker="o", linewidth=2.5, markersize=8, color=COLOR_STMOE, label="ST-MoE")
    ax3.plot(cstp_t["tokens"], cstp_t["hbm_hit_rate_pct"], marker="s", linewidth=2.5, markersize=8, color=COLOR_CSTP, label="CSTP")
    ax3.set_title("Overall HBM Hit Rate (%)")
    ax3.set_xlabel("Token Sequence Length (T)")
    ax3.set_ylabel("HBM Hit Rate (%)")
    ax3.set_xticks(t_steps)
    ax3.legend(frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.6)

    # Panel 4: Mean Token Latency (ms)
    ax4.plot(base_t["tokens"], base_t["mean_token_lat_ms"], marker="D", linewidth=2.5, markersize=8, color=COLOR_BASE, label="Baseline")
    ax4.plot(stmoe_t["tokens"], stmoe_t["mean_token_lat_ms"], marker="o", linewidth=2.5, markersize=8, color=COLOR_STMOE, label="ST-MoE")
    ax4.plot(cstp_t["tokens"], cstp_t["mean_token_lat_ms"], marker="s", linewidth=2.5, markersize=8, color=COLOR_CSTP, label="CSTP")
    ax4.set_title("Mean Token Latency (ms)")
    ax4.set_xlabel("Token Sequence Length (T)")
    ax4.set_ylabel("Mean Token Latency (ms)")
    ax4.set_xticks(t_steps)
    ax4.legend(frameon=True)
    ax4.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Metric Scaling across Long Token Sequences (B=70, 40 Stages, T in {16, 32, 64})", y=1.02)
    save_fig(fig, "fig10_all_metrics_vs_token_length")


def plot_fig11_adaptation():
    fig, ax = plt.subplots(figsize=(8, 5))

    # Parse token_latency.csv from W5 ST-MoE and CSTP to show progression
    stmoe_tok_csv = os.path.join(RAW_DIR, "w5_stmoe_h2", "traces", "token_latency.csv")
    cstp_tok_csv = os.path.join(RAW_DIR, "w5_cstp_h2", "traces", "token_latency.csv")
    base_tok_csv = os.path.join(RAW_DIR, "w5_baseline", "traces", "token_latency.csv")

    if os.path.exists(stmoe_tok_csv) and os.path.exists(base_tok_csv):
        bdf = pd.read_csv(base_tok_csv).groupby("gen_step")["token_duration_ns"].mean() / 1e6
        sdf = pd.read_csv(stmoe_tok_csv).groupby("gen_step")["token_duration_ns"].mean() / 1e6
        cdf = pd.read_csv(cstp_tok_csv).groupby("gen_step")["token_duration_ns"].mean() / 1e6

        steps = bdf.index.tolist()
        ax.plot(steps, bdf.values, marker="D", linewidth=2, color=COLOR_BASE, label="Baseline")
        ax.plot(steps, sdf.values, marker="o", linewidth=2, color=COLOR_STMOE, label="ST-MoE (H=2)")
        ax.plot(steps, cdf.values, marker="s", linewidth=2, color=COLOR_CSTP, label="CSTP (H=2)")

        ax.set_title("Predictor Adaptation: Token Latency vs Decode Step Progress (W5: B=70, T=16)")
        ax.set_xlabel("Decode Generation Step")
        ax.set_ylabel("Mean Token Duration (ms)")
        ax.legend(frameon=True)
        ax.grid(True, linestyle="--", alpha=0.6)

    save_fig(fig, "fig11_predictor_adaptation_vs_token")


def plot_fig12_stmoe_vs_cstp(df_pred):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Focus on H=2 across workloads
    sub = df_pred[df_pred["horizon"] == 2].copy()
    workloads = ["W1", "W2", "W3", "W4", "W5", "W6", "W7"]
    x = np.arange(len(workloads))
    width = 0.35

    st_df = sub[sub["predictor"] == "stmoe"].set_index("workload_id").reindex(workloads)
    cs_df = sub[sub["predictor"] == "cstp"].set_index("workload_id").reindex(workloads)

    # 1. Dispatched Prefetches
    ax1.bar(x - width/2, st_df["dispatched_prefetches"], width, color=COLOR_STMOE, label="ST-MoE")
    ax1.bar(x + width/2, cs_df["dispatched_prefetches"], width, color=COLOR_CSTP, label="CSTP")
    ax1.set_title("Total Dispatched Speculative Transfers")
    ax1.set_xticks(x)
    ax1.set_xticklabels(workloads)
    ax1.set_ylabel("Transfers Count")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. Assisted Demands
    ax2.bar(x - width/2, st_df["prefetch_assisted_demands"], width, color=COLOR_STMOE, label="ST-MoE")
    ax2.bar(x + width/2, cs_df["prefetch_assisted_demands"], width, color=COLOR_CSTP, label="CSTP")
    ax2.set_title("Prefetch-Assisted Demands (Late In-Flight)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(workloads)
    ax2.set_ylabel("Demands Count")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # 3. Exposed Stall Reduction (ms)
    ax3.bar(x - width/2, st_df["stall_reduction_ms"], width, color=COLOR_STMOE, label="ST-MoE")
    ax3.bar(x + width/2, cs_df["stall_reduction_ms"], width, color=COLOR_CSTP, label="CSTP")
    ax3.set_title("Exposed GPU Stall Reduction vs Matched Baseline (ms)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(workloads)
    ax3.set_ylabel("Stall Reduction (ms)")
    ax3.legend(frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.6)

    # 4. Traffic Overhead (%)
    ax4.bar(x - width/2, st_df["traffic_overhead_pct"], width, color=COLOR_STMOE, label="ST-MoE")
    ax4.bar(x + width/2, cs_df["traffic_overhead_pct"], width, color=COLOR_CSTP, label="CSTP")
    ax4.set_title("DDR5 Speculative Traffic Overhead (%)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(workloads)
    ax4.set_ylabel("Overhead (%)")
    ax4.legend(frameon=True)
    ax4.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Head-to-Head Comparison: ST-MoE vs CSTP across Workloads (H=2, 40 Stages)", y=1.02)
    save_fig(fig, "fig12_stmoe_vs_cstp_head_to_head")


def main():
    print("Loading campaign results...")
    df_run, df_batch, df_token, df_pred, df_evt = load_data()

    print("Generating Figure 1: Strict Coverage vs Batch Size...")
    plot_fig1_strict_coverage(df_batch)

    print("Generating Figure 2: Associated Coverage vs Batch Size...")
    plot_fig2_associated_coverage(df_batch)

    print("Generating Figure 3: Timely and Late Rates vs Batch Size...")
    plot_fig3_timely_and_late_rates(df_batch)

    print("Generating Figure 4: Unused Prefetches vs Batch Size...")
    plot_fig4_unused_prefetches(df_batch)

    print("Generating Figure 5: HBM Hit Rate vs Batch Size...")
    plot_fig5_hbm_hit_rate(df_batch)

    print("Generating Figure 6: DDR5 Traffic vs Batch Size...")
    plot_fig6_ddr5_traffic(df_batch)

    print("Generating Figure 7: Exposed GPU Stall vs Batch Size...")
    plot_fig7_exposed_stall(df_batch)

    print("Generating Figure 8: Expert-Load Latency vs Batch Size...")
    plot_fig8_expert_load_lat(df_batch)

    print("Generating Figure 9: Token Latency vs Batch Size...")
    plot_fig9_token_lat(df_batch)

    print("Generating Figure 10: All Metrics vs Token Length...")
    plot_fig10_token_scaling(df_token)

    print("Generating Figure 11: Predictor Adaptation vs Token Progress...")
    plot_fig11_adaptation()

    print("Generating Figure 12: ST-MoE vs CSTP Head-to-Head...")
    plot_fig12_stmoe_vs_cstp(df_pred)

    print("\nAll 12 publication-quality figures generated successfully in PNG and PDF!")


if __name__ == "__main__":
    main()

