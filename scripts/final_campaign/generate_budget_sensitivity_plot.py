#!/usr/bin/env python3
import os
import ast
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    csv_path = "/home/mohman/ca_project/docs/final_campaign/sensitivity_budget/CANDIDATE_BUDGET_SENSITIVITY.csv"
    plots_dir = "/home/mohman/ca_project/docs/final_campaign/sensitivity_budget/plots"
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Subplot 1: Queue Waiting Time vs Budget at M=1 and M=12
    ax1 = axes[0]
    st_m1 = df[(df["predictor"] == "st_moe") & (df["multiplier"] == 1.0)]
    st_m12 = df[(df["predictor"] == "st_moe") & (df["multiplier"] == 12.0)]
    cstp_m12 = df[(df["predictor"] == "cstp") & (df["multiplier"] == 12.0)]

    ax1.plot(st_m1["candidate_budget"], st_m1["avg_queue_wait_ms"], marker='o', linewidth=2, color="#d95f02", label="ST-MoE (M=1.0x)")
    ax1.plot(st_m12["candidate_budget"], st_m12["avg_queue_wait_ms"], marker='s', linewidth=2, color="#1b9e77", label="ST-MoE (M=12.0x)")
    ax1.plot(cstp_m12["candidate_budget"], cstp_m12["avg_queue_wait_ms"], marker='^', linewidth=2, color="#7570b3", linestyle="--", label="CSTP (M=12.0x)")

    ax1.set_title("Queue Waiting Time vs Candidate Budget", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Candidate Budget (B)", fontsize=11)
    ax1.set_ylabel("Average Queue Waiting (ms)", fontsize=11)
    ax1.set_xticks([1, 2, 4])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9)

    # Subplot 2: Speculative Traffic vs Budget
    ax2 = axes[1]
    ax2.plot(st_m1["candidate_budget"], st_m1["speculative_traffic_mib"], marker='o', linewidth=2, color="#d95f02", label="ST-MoE Speculative Traffic")
    ax2.plot(cstp_m12["candidate_budget"], cstp_m12["speculative_traffic_mib"], marker='^', linewidth=2, color="#7570b3", linestyle="--", label="CSTP Speculative Traffic")

    ax2.set_title("DDR5 Speculative Traffic vs Budget", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Candidate Budget (B)", fontsize=11)
    ax2.set_ylabel("Speculative Traffic (MiB)", fontsize=11)
    ax2.set_xticks([1, 2, 4])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9)

    # Subplot 3: Stall Reduction at M=12 vs Budget
    ax3 = axes[2]
    ax3.bar(np.array([1, 2, 4]) - 0.15, st_m12["stall_reduction_ms"], width=0.3, color="#1b9e77", label="ST-MoE (M=12.0x)")
    ax3.bar(np.array([1, 2, 4]) + 0.15, cstp_m12["stall_reduction_ms"], width=0.3, color="#7570b3", label="CSTP (M=12.0x)")

    ax3.set_title("Stall Reduction vs Budget (M=12.0x)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Candidate Budget (B)", fontsize=11)
    ax3.set_ylabel("Exposed Stall Reduction (ms)", fontsize=11)
    ax3.set_xticks([1, 2, 4])
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(fontsize=9)

    plt.tight_layout()
    out_plot = os.path.join(plots_dir, "CANDIDATE_BUDGET_SENSITIVITY.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved plot to {out_plot}")

if __name__ == "__main__":
    main()

