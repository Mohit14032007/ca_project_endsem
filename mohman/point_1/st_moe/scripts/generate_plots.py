#!/usr/bin/env python3
"""
Generate Publication-Quality Plots for Point 1: ST-MoE Baseline Reproduction.

This script reads standardized CSVs from mohman/point_1/st_moe/results/csv/
and generates publication-quality figures (PNG at 300 DPI and PDF) in
mohman/point_1/st_moe/plots/.

Pure Matplotlib is used (NO Seaborn dependency).
"""

import os
import sys
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paths
REPO_ROOT = Path("/home/mohman/ca_project")
CSV_DIR = REPO_ROOT / "mohman" / "point_1" / "st_moe" / "results" / "csv"
PLOT_DIR = REPO_ROOT / "mohman" / "point_1" / "st_moe" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Academic styling settings
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
})

# Color palette
C_NAVY = "#1f77b4"
C_CORAL = "#ff7f0e"
C_GREEN = "#2ca02c"
C_PURPLE = "#9467bd"
C_GRAY = "#7f7f7f"
C_CYAN = "#17becf"
C_RED = "#d62728"


def save_fig(fig, base_name):
    """Saves figure in both PNG and PDF formats."""
    png_path = PLOT_DIR / f"{base_name}.png"
    pdf_path = PLOT_DIR / f"{base_name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path.name} and {pdf_path.name}")


def plot_recall_comparison():
    """
    Plot 1: Recall@8 comparison across authoritative and diagnostic datasets
    for horizons H=1 and H=2 under the primary online_within_request standard.
    """
    csv_path = CSV_DIR / "online_policy_comparison.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Focus on authoritative datasets
    data = {}
    for r in rows:
        ds = r["dataset_id"]
        h = int(r["horizon"])
        rec = float(r["within_request_recall"]) * 100.0
        if ds not in data:
            data[ds] = {}
        data[ds][h] = rec

    labels = ["MoE Anywhere (Sage-A1.5)\n40 Layers, 256 Experts", "MoE Suite (GLM-5.2)\n75 Layers, 256 Experts"]
    h1_vals = [data["moe_anywhere"][1], data["moe_suite"][1]]
    h2_vals = [data["moe_anywhere"][2], data["moe_suite"][2]]

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, h1_vals, width, label="Horizon H=1 (Next Layer)", color=C_NAVY, edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, h2_vals, width, label="Horizon H=2 (Skip Layer)", color=C_CORAL, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Recall@8 (%)")
    ax.set_title("ST-MoE Expert Prediction Recall (Primary Standard: Online Within-Request)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 60)
    ax.legend(frameon=True, loc="upper right")

    # Add direct data labels
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold", fontsize=10)
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold", fontsize=10)

    save_fig(fig, "recall_comparison")


def plot_frozen_vs_online():
    """
    Plot 2: Recall vs Budget (B in 1, 2, 4, 6, 8) comparing Frozen vs Online
    for MoE Anywhere and MoE Suite at H=1.
    """
    csv_path = CSV_DIR / "fixed_budget_summary.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Filter for H=1 authoritative
    ma_frozen = {}
    ma_online_across = {}
    ms_frozen = {}
    ms_online_across = {}

    for r in rows:
        if r["classification"] != "authoritative" or int(r["horizon"]) != 1:
            continue
        b = int(r["budget"])
        rec = float(r["recall"]) * 100.0
        ds = r["dataset_id"]
        reg = r["regime"]
        if ds == "moe_anywhere":
            if reg == "frozen":
                ma_frozen[b] = rec
            elif reg == "online_across_requests":
                ma_online_across[b] = rec
        elif ds == "moe_suite":
            if reg == "frozen":
                ms_frozen[b] = rec
            elif reg == "online_across_requests":
                ms_online_across[b] = rec

    budgets = [1, 2, 4, 6, 8]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # Left: MoE Anywhere
    ax1.plot(budgets, [ma_frozen[b] for b in budgets], marker="o", linewidth=2.0, color=C_GRAY, linestyle="--", label="Frozen (Offline Profiling Only)")
    ax1.plot(budgets, [ma_online_across[b] for b in budgets], marker="s", linewidth=2.2, color=C_NAVY, label="Online (Continuous CCT Adaptation)")
    # Mark the Primary Standard B=K within_request point (47.91%)
    ax1.scatter([8], [47.91], color=C_GREEN, s=120, zorder=5, marker="^", label="Primary Standard: Within-Request (47.91%)")

    ax1.set_title("MoE Anywhere (Sage-A1.5, H=1)")
    ax1.set_xlabel("Prediction Budget B")
    ax1.set_ylabel("Recall (%)")
    ax1.set_xticks(budgets)
    ax1.set_ylim(0, 60)
    ax1.legend(loc="upper left", frameon=True)

    # Right: MoE Suite
    ax2.plot(budgets, [ms_frozen[b] for b in budgets], marker="o", linewidth=2.0, color=C_GRAY, linestyle="--", label="Frozen (Offline Profiling Only)")
    ax2.plot(budgets, [ms_online_across[b] for b in budgets], marker="s", linewidth=2.2, color=C_CORAL, label="Online (Continuous CCT Adaptation)")
    ax2.scatter([8], [26.53], color=C_GREEN, s=120, zorder=5, marker="^", label="Primary Standard: Within-Request (26.53%)")

    ax2.set_title("MoE Suite (GLM-5.2, H=1)")
    ax2.set_xlabel("Prediction Budget B")
    ax2.set_xticks(budgets)
    ax2.legend(loc="upper left", frameon=True)

    fig.suptitle("ST-MoE Recall Scaling Across Prediction Budgets (Frozen vs Online)", y=1.02)
    save_fig(fig, "frozen_vs_online")


def plot_horizon_comparison():
    """
    Plot 3: Lookahead horizon comparison (H=1 vs H=2) for both recall and precision.
    """
    csv_path = CSV_DIR / "online_policy_comparison.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    datasets = ["moe_anywhere", "moe_suite"]
    labels = ["MoE Anywhere (Sage-A1.5)", "MoE Suite (GLM-5.2)"]

    h1_rec = []
    h2_rec = []
    h1_froz = []
    h2_froz = []

    for ds in datasets:
        r1 = [r for r in rows if r["dataset_id"] == ds and int(r["horizon"]) == 1][0]
        r2 = [r for r in rows if r["dataset_id"] == ds and int(r["horizon"]) == 2][0]
        h1_rec.append(float(r1["within_request_recall"]) * 100.0)
        h2_rec.append(float(r2["within_request_recall"]) * 100.0)
        h1_froz.append(float(r1["frozen_recall"]) * 100.0)
        h2_froz.append(float(r2["frozen_recall"]) * 100.0)

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 1.5*width, h1_froz, width, label="H=1 Frozen", color="#aec7e8", edgecolor="black", linewidth=0.6)
    ax.bar(x - 0.5*width, h1_rec, width, label="H=1 Online Within-Req", color=C_NAVY, edgecolor="black", linewidth=0.6)
    ax.bar(x + 0.5*width, h2_froz, width, label="H=2 Frozen", color="#ffbb78", edgecolor="black", linewidth=0.6)
    ax.bar(x + 1.5*width, h2_rec, width, label="H=2 Online Within-Req", color=C_CORAL, edgecolor="black", linewidth=0.6)

    ax.set_ylabel("Recall@8 (%)")
    ax.set_title("Lookahead Horizon Degradation (H=1 vs H=2 Lookahead Distance)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 60)
    ax.legend(frameon=True, loc="upper right")

    save_fig(fig, "horizon_comparison")


def plot_hit_rate_comparison():
    """
    Plot 4: Hit Rate comparison (percentage of opportunities with >= 1 correct hit).
    """
    csv_path = CSV_DIR / "fixed_budget_summary.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ma_hits_froz = {}
    ma_hits_onl = {}
    ms_hits_froz = {}
    ms_hits_onl = {}

    for r in rows:
        if r["classification"] != "authoritative" or int(r["horizon"]) != 1:
            continue
        b = int(r["budget"])
        hr = float(r["hit_rate"]) * 100.0
        ds = r["dataset_id"]
        reg = r["regime"]
        if ds == "moe_anywhere":
            if reg == "frozen":
                ma_hits_froz[b] = hr
            elif reg == "online_across_requests":
                ma_hits_onl[b] = hr
        elif ds == "moe_suite":
            if reg == "frozen":
                ms_hits_froz[b] = hr
            elif reg == "online_across_requests":
                ms_hits_onl[b] = hr

    budgets = [1, 2, 4, 6, 8]
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(budgets, [ma_hits_onl[b] for b in budgets], marker="s", linewidth=2.2, color=C_NAVY, label="MoE Anywhere (Online Across)")
    ax.plot(budgets, [ma_hits_froz[b] for b in budgets], marker="o", linewidth=1.8, linestyle="--", color="#aec7e8", label="MoE Anywhere (Frozen)")
    ax.plot(budgets, [ms_hits_onl[b] for b in budgets], marker="^", linewidth=2.2, color=C_CORAL, label="MoE Suite (Online Across)")
    ax.plot(budgets, [ms_hits_froz[b] for b in budgets], marker="v", linewidth=1.8, linestyle="--", color="#ffbb78", label="MoE Suite (Frozen)")

    # Mark the 85% claim line
    ax.axhline(85.0, color=C_RED, linestyle=":", linewidth=1.5, label="Paper Reported 'Accuracy' Anchor (~85%)")

    ax.set_xlabel("Prediction Budget B")
    ax.set_ylabel("Hit Rate (%) [P(Hits >= 1)]")
    ax.set_title("ST-MoE Hit Rate vs. Prediction Budget (H=1)")
    ax.set_xticks(budgets)
    ax.set_ylim(10, 102)
    ax.legend(frameon=True, loc="lower right")

    save_fig(fig, "hit_rate_comparison")


def plot_cold_vs_warm_recall():
    """
    Plot 5: Cold-start recall (tokens 0-4) vs Steady-state recall (tokens 5+)
    under Frozen, Within-Request, and Across-Requests policies.
    """
    csv_path = CSV_DIR / "online_policy_comparison.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Use MoE Anywhere H=1
    ma_row = [r for r in rows if r["dataset_id"] == "moe_anywhere" and int(r["horizon"]) == 1][0]

    policies = ["Frozen", "Online\n(Within-Req)", "Online\n(Across-Req)"]
    cold = [31.85, float(ma_row["within_request_cold_recall"])*100, float(ma_row["across_requests_cold_recall"])*100]
    steady = [42.75, float(ma_row["within_request_steady_recall"])*100, float(ma_row["across_requests_steady_recall"])*100]

    x = np.arange(len(policies))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))
    r1 = ax.bar(x - width/2, cold, width, label="Cold-Start (Tokens 0–4)", color=C_CYAN, edgecolor="black", linewidth=0.8)
    r2 = ax.bar(x + width/2, steady, width, label="Steady-State (Tokens 5+)", color=C_PURPLE, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Recall@8 (%)")
    ax.set_title("Cold-Start vs. Steady-State Recall (MoE Anywhere, H=1)")
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.set_ylim(0, 60)
    ax.legend(frameon=True, loc="upper left")

    for rect in r1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in r2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    save_fig(fig, "cold_vs_warm_recall")


def plot_native_threshold_tradeoff():
    """
    Plot 6: Native Threshold Tradeoff showing the severe set-size dilation
    and precision collapse of confidence_additive vs indicator_intersection.
    """
    csv_path = CSV_DIR / "native_threshold_results.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Filter for moe_anywhere H=1
    ma_thresh = [r for r in rows if r["dataset_id"] == "moe_anywhere" and int(r["horizon"]) == 1]

    modes = ["Additive (Frozen)", "Additive (Online)", "Intersection (Frozen)", "Intersection (Online)"]
    sizes = []
    recalls = []
    precisions = []

    for m_label, sm, reg in [
        ("Additive (Frozen)", "confidence_additive", "frozen"),
        ("Additive (Online)", "confidence_additive", "online_across_requests"),
        ("Intersection (Frozen)", "indicator_intersection", "frozen"),
        ("Intersection (Online)", "indicator_intersection", "online_across_requests")
    ]:
        match = [r for r in ma_thresh if r["scoring_mode"] == sm and r["regime"] == reg][0]
        sizes.append(float(match["mean_predicted_set_size"]))
        recalls.append(float(match["recall"]) * 100.0)
        precisions.append(float(match["precision"]) * 100.0)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    x = np.arange(len(modes))
    width = 0.25

    rects1 = ax1.bar(x - width, sizes, width, label="Mean Set Size |P|", color=C_RED, edgecolor="black", linewidth=0.8)
    rects2 = ax1.bar(x, recalls, width, label="Recall (%)", color=C_NAVY, edgecolor="black", linewidth=0.8)
    rects3 = ax1.bar(x + width, precisions, width, label="Precision (%)", color=C_GREEN, edgecolor="black", linewidth=0.8)

    ax1.set_ylabel("Score / Cardinality Value")
    ax1.set_title("ST-MoE Native Thresholding Tradeoff: Set-Size Dilation & Precision (MoE Anywhere, H=1)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes)
    ax1.set_ylim(0, 85)
    ax1.axhline(8.0, color="black", linestyle="--", linewidth=1.0, label="True Top-K Target (K=8)")
    ax1.legend(frameon=True, loc="upper right")

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects3:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    save_fig(fig, "native_threshold_tradeoff")


def plot_profiling_sensitivity():
    """
    Plot 7: Profiling Fraction Sensitivity (10% to 100%) showing data efficiency
    and the gap between Frozen and Online regimes.
    """
    csv_path = CSV_DIR / "profiling_sensitivity.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Filter for moe_anywhere H=1
    ma_sens = [r for r in rows if r["dataset_id"] == "moe_anywhere" and int(r["horizon"]) == 1]
    fractions = [10, 20, 40, 60, 80, 100]

    frozen_rec = []
    online_rec = []

    for f in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        rf = [r for r in ma_sens if abs(float(r["profiling_fraction"]) - f) < 1e-4 and r["regime"] == "frozen"][0]
        ro = [r for r in ma_sens if abs(float(r["profiling_fraction"]) - f) < 1e-4 and r["regime"] == "online_across_requests"][0]
        frozen_rec.append(float(rf["fixed_budget_b_k_recall"]) * 100.0)
        online_rec.append(float(ro["fixed_budget_b_k_recall"]) * 100.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fractions, frozen_rec, marker="o", linewidth=2.0, color=C_GRAY, linestyle="--", label="Frozen (CCT Profiling Only)")
    ax.plot(fractions, online_rec, marker="s", linewidth=2.2, color=C_NAVY, label="Online (CCT Adaptation Enabled)")

    ax.set_xlabel("Profiling Data Fraction (%)")
    ax.set_ylabel("Recall@8 (%)")
    ax.set_title("Offline Profiling Sensitivity (MoE Anywhere, H=1, B=K=8)")
    ax.set_xticks(fractions)
    ax.set_ylim(25, 55)
    ax.legend(frameon=True, loc="lower right")

    save_fig(fig, "profiling_sensitivity")


def main():
    print("Generating publication plots in mohman/point_1/st_moe/plots/ ...")
    plot_recall_comparison()
    plot_frozen_vs_online()
    plot_horizon_comparison()
    plot_hit_rate_comparison()
    plot_cold_vs_warm_recall()
    plot_native_threshold_tradeoff()
    plot_profiling_sensitivity()
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()

