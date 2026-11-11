#!/usr/bin/env python3
"""
Generate publication-quality figures for Phase 2F Synthetic Compute Sensitivity Campaign.
Zero ternary operators (enforced by AST).
"""

import ast
import os
import matplotlib.pyplot as plt
import pandas as pd

DOCS_DIR = "/home/mohman/ca_project/docs/final_campaign/phase2f"
PLOTS_DIR = os.path.join(DOCS_DIR, "plots")
SUMMARY_CSV = os.path.join(DOCS_DIR, "PHASE2F_SUMMARY.csv")


def check_no_ternary():
    with open(__file__, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=__file__)
    ternaries = []
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            ternaries.append(node.lineno)
    if ternaries:
        raise AssertionError(f"Ternary operator found at lines {ternaries} in {__file__}!")


def main():
    check_no_ternary()
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_csv(SUMMARY_CSV)

    st_df = df[df["predictor"] == "st_moe"].sort_values("multiplier")
    cstp_df = df[df["predictor"] == "cstp"].sort_values("multiplier")
    base_df = df[df["predictor"] == "none"].sort_values("multiplier")

    style_name = "default"
    if "seaborn-v0_8-whitegrid" in plt.style.available:
        style_name = "seaborn-v0_8-whitegrid"
    plt.style.use(style_name)

    # Figure 1: Stall Reduction & Speedup vs Multiplier
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(st_df["multiplier"], st_df["stall_reduction_ms"], marker="o", color="#1f77b4", linewidth=2.5, label="ST-MoE (FWD-S-H2)")
    ax1.plot(cstp_df["multiplier"], cstp_df["stall_reduction_ms"], marker="s", color="#ff7f0e", linewidth=2.5, label="CSTP (FWD-C-H2)")
    ax1.set_xlabel("Compute Time Multiplier (M)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Stall Reduction vs B(M) (ms)", fontsize=12, fontweight="bold")
    ax1.set_title("Exposed Memory Stall Reduction vs Compute Multiplier", fontsize=13, fontweight="bold")
    ax1.set_xticks([1, 4, 8, 12, 16])
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(fontsize=11)

    ax2.plot(st_df["multiplier"], st_df["speedup_pct"], marker="o", color="#1f77b4", linewidth=2.5, label="ST-MoE (FWD-S-H2)")
    ax2.plot(cstp_df["multiplier"], cstp_df["speedup_pct"], marker="s", color="#ff7f0e", linewidth=2.5, label="CSTP (FWD-C-H2)")
    ax2.set_xlabel("Compute Time Multiplier (M)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Execution Speedup vs B(M) (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Simulated Execution Speedup vs Compute Multiplier", fontsize=13, fontweight="bold")
    ax2.set_xticks([1, 4, 8, 12, 16])
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "PHASE2F_STALL_REDUCTION_VS_MULTIPLIER.png"), dpi=300)
    plt.close()

    # Figure 2: Timely Hits and Late Attachments
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(st_df["multiplier"], st_df["timely_resident_hits"], marker="o", color="#2ca02c", linewidth=2.5, label="ST-MoE (FWD-S-H2)")
    ax1.plot(cstp_df["multiplier"], cstp_df["timely_resident_hits"], marker="s", color="#d62728", linewidth=2.5, label="CSTP (FWD-C-H2)")
    ax1.set_xlabel("Compute Time Multiplier (M)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Timely Resident Hits (Count)", fontsize=12, fontweight="bold")
    ax1.set_title("Emergence of Timely Resident Hits", fontsize=13, fontweight="bold")
    ax1.set_xticks([1, 4, 8, 12, 16])
    ax1.set_yticks([0, 1, 2, 3])
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(fontsize=11)

    ax2.plot(st_df["multiplier"], st_df["late_inflight_attachments"], marker="o", color="#1f77b4", linewidth=2.5, label="ST-MoE (FWD-S-H2)")
    ax2.plot(cstp_df["multiplier"], cstp_df["late_inflight_attachments"], marker="s", color="#ff7f0e", linewidth=2.5, label="CSTP (FWD-C-H2)")
    ax2.set_xlabel("Compute Time Multiplier (M)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Late In-Flight Attachments (Count)", fontsize=12, fontweight="bold")
    ax2.set_title("Reduction in Late Attachments with Expanding Compute", fontsize=13, fontweight="bold")
    ax2.set_xticks([1, 4, 8, 12, 16])
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "PHASE2F_TIMELY_HITS_AND_ATTACHMENTS.png"), dpi=300)
    plt.close()

    # Figure 3: Queue Waiting Time Dynamics
    plt.figure(figsize=(8, 5))
    plt.plot(st_df["multiplier"], st_df["avg_queue_wait_ms"], marker="o", color="#9467bd", linewidth=2.5, label="ST-MoE Queue Wait")
    plt.plot(cstp_df["multiplier"], cstp_df["avg_queue_wait_ms"], marker="s", color="#8c564b", linewidth=2.5, label="CSTP Queue Wait")
    plt.xlabel("Compute Time Multiplier (M)", fontsize=12, fontweight="bold")
    plt.ylabel("Average DDR5 Queue Waiting Time (ms)", fontsize=12, fontweight="bold")
    plt.title("Queue Decongestion Under Expanded Compute Windows", fontsize=13, fontweight="bold")
    plt.xticks([1, 4, 8, 12, 16])
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "PHASE2F_QUEUE_WAIT_DYNAMICS.png"), dpi=300)
    plt.close()

    # Figure 4: Predictor Quality (Precision, Recall, F1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(st_df["multiplier"], st_df["prediction_precision_pct"], marker="o", color="#1f77b4", linewidth=2.5, label="ST-MoE Precision")
    ax1.plot(cstp_df["multiplier"], cstp_df["prediction_precision_pct"], marker="s", color="#ff7f0e", linewidth=2.5, label="CSTP Precision")
    ax1.set_xlabel("Compute Time Multiplier (M)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Prediction Precision (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Prediction Precision Across Multipliers", fontsize=13, fontweight="bold")
    ax1.set_xticks([1, 4, 8, 12, 16])
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(fontsize=11)

    ax2.plot(st_df["multiplier"], st_df["prediction_recall_pct"], marker="o", color="#1f77b4", linewidth=2.5, label="ST-MoE Recall")
    ax2.plot(cstp_df["multiplier"], cstp_df["prediction_recall_pct"], marker="s", color="#ff7f0e", linewidth=2.5, label="CSTP Recall")
    ax2.set_xlabel("Compute Time Multiplier (M)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Prediction Recall (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Prediction Recall Across Multipliers", fontsize=13, fontweight="bold")
    ax2.set_xticks([1, 4, 8, 12, 16])
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "PHASE2F_PREDICTOR_QUALITY.png"), dpi=300)
    plt.close()

    # Figure 5: Energy Analysis
    plt.figure(figsize=(8, 5))
    plt.plot(base_df["multiplier"], base_df["total_energy_j"], marker="^", color="#7f7f7f", linewidth=2, linestyle="--", label="Baseline B(M)")
    plt.plot(st_df["multiplier"], st_df["total_energy_j"], marker="o", color="#1f77b4", linewidth=2.5, label="ST-MoE (FWD-S-H2)")
    plt.plot(cstp_df["multiplier"], cstp_df["total_energy_j"], marker="s", color="#ff7f0e", linewidth=2.5, label="CSTP (FWD-C-H2)")
    plt.xlabel("Compute Time Multiplier (M)", fontsize=12, fontweight="bold")
    plt.ylabel("Total Simulated Energy (Joules)", fontsize=12, fontweight="bold")
    plt.title("Total System Energy Across Compute Multipliers", fontsize=13, fontweight="bold")
    plt.xticks([1, 4, 8, 12, 16])
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "PHASE2F_ENERGY_BREAKDOWN.png"), dpi=300)
    plt.close()

    print("[SUCCESS] Generated all 5 high-resolution PNG plots in docs/final_campaign/phase2f/plots/!")


if __name__ == "__main__":
    main()
