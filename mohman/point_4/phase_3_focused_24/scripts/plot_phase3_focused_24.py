#!/usr/bin/env python3
"""
Publication-Quality Plotting Suite for Point 4 Phase 3 Focused 24-Run Evaluation.
Generates both PNG and PDF plots for the 10 mandatory figures:
  1. HBM hit rate vs capacity, separated by workload
  2. Expert reload count vs capacity
  3. Demand and speculative eviction counts vs capacity
  4. DDR5 traffic vs capacity
  5. Exposed GPU stall time vs capacity
  6. Mean token latency vs capacity
  7. CSTP prefetch coverage vs capacity
  8. Unused prefetches vs capacity
  9. Prefetch-induced evictions vs capacity
 10. FA-LRU relative improvement over LRU across the matrix
Strictly enforces ast.IfExp == 0 (zero ternary operators).
"""

import os
import sys
import ast
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = "/home/mohman/ca_project"
PHASE3_ROOT = os.path.join(WORKSPACE, "mohman", "point_4", "phase_3_focused_24")
CONSOLIDATED_DIR = os.path.join(PHASE3_ROOT, "consolidated")
PLOTS_PNG_DIR = os.path.join(PHASE3_ROOT, "plots", "png")
PLOTS_PDF_DIR = os.path.join(PHASE3_ROOT, "plots", "pdf")

ALL_CSV = os.path.join(CONSOLIDATED_DIR, "all_24_runs.csv")
PAIRED_CSV = os.path.join(CONSOLIDATED_DIR, "paired_comparisons.csv")


def verify_zero_ternary(file_path):
    with open(file_path, "r") as f:
        tree = ast.parse(f.read(), filename=file_path)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            count += 1
    if count > 0:
        raise ValueError(f"Found {count} ternary operators in {file_path}!")
    print(f"[AST_CHECK] Verified 0 ternary operators in {file_path}")


# Styling
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})

COLORS = {
    "c1": "#1f77b4", # Blue: Base LRU
    "c2": "#2ca02c", # Green: Base FA-LRU
    "c3": "#ff7f0e", # Orange: CSTP LRU
    "c4": "#d62728", # Red: CSTP FA-LRU
}

MARKERS = {
    "c1": "o",
    "c2": "s",
    "c3": "^",
    "c4": "D",
}

LABELS = {
    "c1": "Baseline LRU (No Prefetch)",
    "c2": "FA-LRU W=16 (No Prefetch)",
    "c3": "CSTP Prefetch + LRU",
    "c4": "CSTP Prefetch + FA-LRU W=16",
}

CAP_ORDER = ["cap1", "cap2", "cap3"]
CAP_LABELS = ["10 GB\n(113 exp)", "35 GiB\n(426 exp)", "80 GB\n(908 exp)"]
CAP_VALS = [10.0, 37.58, 80.0]


def save_fig(fig, base_name):
    png_path = os.path.join(PLOTS_PNG_DIR, f"{base_name}.png")
    pdf_path = os.path.join(PLOTS_PDF_DIR, f"{base_name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {base_name}.png and .pdf")


def plot_fig1_hit_rate(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    workloads = [("w2", "Workload W2 (B=4, T=16, L=40)", axes[0]),
                 ("w3", "Workload W3 (B=14, T=16, L=40)", axes[1])]

    for w_key, w_title, ax in workloads:
        sub_w = df[df["workload"] == w_key]
        for cfg in ["c1", "c2", "c3", "c4"]:
            sub_c = sub_w[sub_w["config_id"] == cfg]
            # Sort by capacity order
            sub_c = sub_c.set_index("capacity_id").reindex(CAP_ORDER).reset_index()
            y_vals = sub_c["coalesced_hbm_hit_rate"] * 100.0
            ax.plot(range(3), y_vals, marker=MARKERS[cfg], color=COLORS[cfg], label=LABELS[cfg])

        ax.set_title(w_title)
        ax.set_xticks(range(3))
        ax.set_xticklabels(CAP_LABELS)
        ax.set_xlabel("HBM Cache Capacity")
        if ax == axes[0]:
            ax.set_ylabel("Coalesced HBM Hit Rate (%)")
        ax.legend(loc="lower right")

    fig.suptitle("Figure 1: Coalesced HBM Hit Rate vs Cache Capacity", y=1.02)
    save_fig(fig, "fig1_hbm_hit_rate_vs_capacity")


def plot_fig2_reloads(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    workloads = [("w2", "Workload W2 (B=4, T=16, L=40)", axes[0]),
                 ("w3", "Workload W3 (B=14, T=16, L=40)", axes[1])]

    for w_key, w_title, ax in workloads:
        sub_w = df[df["workload"] == w_key]
        for cfg in ["c1", "c2", "c3", "c4"]:
            sub_c = sub_w[sub_w["config_id"] == cfg]
            sub_c = sub_c.set_index("capacity_id").reindex(CAP_ORDER).reset_index()
            y_vals = sub_c["reloaded_experts_count"]
            ax.plot(range(3), y_vals, marker=MARKERS[cfg], color=COLORS[cfg], label=LABELS[cfg])

        ax.set_title(w_title)
        ax.set_xticks(range(3))
        ax.set_xticklabels(CAP_LABELS)
        ax.set_xlabel("HBM Cache Capacity")
        ax.set_ylabel("Expert Reload Count")
        ax.legend(loc="upper right")

    fig.suptitle("Figure 2: Expert Reload Count vs Cache Capacity", y=1.02)
    save_fig(fig, "fig2_expert_reloads_vs_capacity")


def plot_fig3_evictions(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    workloads = [("w2", "Workload W2 (B=4, T=16, L=40)", axes[0]),
                 ("w3", "Workload W3 (B=14, T=16, L=40)", axes[1])]

    for w_key, w_title, ax in workloads:
        sub_w = df[df["workload"] == w_key]
        for cfg in ["c1", "c2", "c3", "c4"]:
            sub_c = sub_w[sub_w["config_id"] == cfg]
            sub_c = sub_c.set_index("capacity_id").reindex(CAP_ORDER).reset_index()
            y_vals = sub_c["total_evictions"]
            ax.plot(range(3), y_vals, marker=MARKERS[cfg], color=COLORS[cfg], label=LABELS[cfg])

        ax.set_title(w_title)
        ax.set_xticks(range(3))
        ax.set_xticklabels(CAP_LABELS)
        ax.set_xlabel("HBM Cache Capacity")
        ax.set_ylabel("Total Evictions Count")
        ax.legend(loc="upper right")

    fig.suptitle("Figure 3: Total Eviction Count vs Cache Capacity", y=1.02)
    save_fig(fig, "fig3_evictions_vs_capacity")


def plot_fig4_ddr5_traffic(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    workloads = [("w2", "Workload W2 (B=4, T=16, L=40)", axes[0]),
                 ("w3", "Workload W3 (B=14, T=16, L=40)", axes[1])]

    for w_key, w_title, ax in workloads:
        sub_w = df[df["workload"] == w_key]
        for cfg in ["c1", "c2", "c3", "c4"]:
            sub_c = sub_w[sub_w["config_id"] == cfg]
            sub_c = sub_c.set_index("capacity_id").reindex(CAP_ORDER).reset_index()
            y_vals = sub_c["total_ddr5_gb"]
            ax.plot(range(3), y_vals, marker=MARKERS[cfg], color=COLORS[cfg], label=LABELS[cfg])

        ax.set_title(w_title)
        ax.set_xticks(range(3))
        ax.set_xticklabels(CAP_LABELS)
        ax.set_xlabel("HBM Cache Capacity")
        ax.set_ylabel("Total DDR5 Traffic (Decimal GB, 10^9 B)")
        ax.legend(loc="upper right")

    fig.suptitle("Figure 4: Total DDR5 Memory Traffic vs Cache Capacity", y=1.02)
    save_fig(fig, "fig4_ddr5_traffic_vs_capacity")


def plot_fig5_exposed_stall(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    workloads = [("w2", "Workload W2 (B=4, T=16, L=40)", axes[0]),
                 ("w3", "Workload W3 (B=14, T=16, L=40)", axes[1])]

    for w_key, w_title, ax in workloads:
        sub_w = df[df["workload"] == w_key]
        for cfg in ["c1", "c2", "c3", "c4"]:
            sub_c = sub_w[sub_w["config_id"] == cfg]
            sub_c = sub_c.set_index("capacity_id").reindex(CAP_ORDER).reset_index()
            y_vals = sub_c["exposed_gpu_stall_ms"]
            ax.plot(range(3), y_vals, marker=MARKERS[cfg], color=COLORS[cfg], label=LABELS[cfg])

        ax.set_title(w_title)
        ax.set_xticks(range(3))
        ax.set_xticklabels(CAP_LABELS)
        ax.set_xlabel("HBM Cache Capacity")
        ax.set_ylabel("Exposed GPU Stall Time (ms)")
        ax.legend(loc="upper right")

    fig.suptitle("Figure 5: Exposed GPU Stall Time vs Cache Capacity", y=1.02)
    save_fig(fig, "fig5_exposed_stall_vs_capacity")


def plot_fig6_token_latency(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    workloads = [("w2", "Workload W2 (B=4, T=16, L=40)", axes[0]),
                 ("w3", "Workload W3 (B=14, T=16, L=40)", axes[1])]

    for w_key, w_title, ax in workloads:
        sub_w = df[df["workload"] == w_key]
        for cfg in ["c1", "c2", "c3", "c4"]:
            sub_c = sub_w[sub_w["config_id"] == cfg]
            sub_c = sub_c.set_index("capacity_id").reindex(CAP_ORDER).reset_index()
            y_vals = sub_c["mean_token_latency_ms"]
            ax.plot(range(3), y_vals, marker=MARKERS[cfg], color=COLORS[cfg], label=LABELS[cfg])

        ax.set_title(w_title)
        ax.set_xticks(range(3))
        ax.set_xticklabels(CAP_LABELS)
        ax.set_xlabel("HBM Cache Capacity")
        ax.set_ylabel("Mean Token Latency (ms)")
        ax.legend(loc="upper right")

    fig.suptitle("Figure 6: Mean Token Latency vs Cache Capacity", y=1.02)
    save_fig(fig, "fig6_token_latency_vs_capacity")


def plot_fig7_prefetch_coverage(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    workloads = [("w2", "Workload W2 (B=4, T=16, L=40)", axes[0]),
                 ("w3", "Workload W3 (B=14, T=16, L=40)", axes[1])]

    # Only C3 and C4 have prefetching
    for w_key, w_title, ax in workloads:
        sub_w = df[df["workload"] == w_key]
        for cfg in ["c3", "c4"]:
            sub_c = sub_w[sub_w["config_id"] == cfg]
            sub_c = sub_c.set_index("capacity_id").reindex(CAP_ORDER).reset_index()
            y_vals = sub_c["prefetch_coverage"] * 100.0
            ax.plot(range(3), y_vals, marker=MARKERS[cfg], color=COLORS[cfg], label=LABELS[cfg])

        ax.set_title(w_title)
        ax.set_xticks(range(3))
        ax.set_xticklabels(CAP_LABELS)
        ax.set_xlabel("HBM Cache Capacity")
        if ax == axes[0]:
            ax.set_ylabel("Prefetch Coverage (%)")
        ax.legend(loc="lower right")

    fig.suptitle("Figure 7: CSTP Prefetch Coverage vs Cache Capacity", y=1.02)
    save_fig(fig, "fig7_cstp_coverage_vs_capacity")


def plot_fig8_unused_prefetches(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    workloads = [("w2", "Workload W2 (B=4, T=16, L=40)", axes[0]),
                 ("w3", "Workload W3 (B=14, T=16, L=40)", axes[1])]

    for w_key, w_title, ax in workloads:
        sub_w = df[df["workload"] == w_key]
        for cfg in ["c3", "c4"]:
            sub_c = sub_w[sub_w["config_id"] == cfg]
            sub_c = sub_c.set_index("capacity_id").reindex(CAP_ORDER).reset_index()
            y_vals = sub_c["unused_completed_prefetches"]
            ax.plot(range(3), y_vals, marker=MARKERS[cfg], color=COLORS[cfg], label=LABELS[cfg])

        ax.set_title(w_title)
        ax.set_xticks(range(3))
        ax.set_xticklabels(CAP_LABELS)
        ax.set_xlabel("HBM Cache Capacity")
        ax.set_ylabel("Unused Completed Prefetches Count")
        ax.legend(loc="upper right")

    fig.suptitle("Figure 8: Unused Completed Prefetches vs Cache Capacity", y=1.02)
    save_fig(fig, "fig8_unused_prefetches_vs_capacity")


def plot_fig9_prefetch_induced_evictions(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    workloads = [("w2", "Workload W2 (B=4, T=16, L=40)", axes[0]),
                 ("w3", "Workload W3 (B=14, T=16, L=40)", axes[1])]

    for w_key, w_title, ax in workloads:
        sub_w = df[df["workload"] == w_key]
        for cfg in ["c3", "c4"]:
            sub_c = sub_w[sub_w["config_id"] == cfg]
            sub_c = sub_c.set_index("capacity_id").reindex(CAP_ORDER).reset_index()
            y_vals = sub_c["prefetch_induced_evictions"]
            ax.plot(range(3), y_vals, marker=MARKERS[cfg], color=COLORS[cfg], label=LABELS[cfg])

        ax.set_title(w_title)
        ax.set_xticks(range(3))
        ax.set_xticklabels(CAP_LABELS)
        ax.set_xlabel("HBM Cache Capacity")
        ax.set_ylabel("Prefetch-Induced Evictions Count")
        ax.legend(loc="upper right")

    fig.suptitle("Figure 9: Prefetch-Induced Evictions (Cache Pollution) vs Capacity", y=1.02)
    save_fig(fig, "fig9_prefetch_induced_evictions_vs_capacity")


def plot_fig10_relative_improvement(pair_df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    workloads = [("w2", "Workload W2 (B=4)", axes[0]),
                 ("w3", "Workload W3 (B=14)", axes[1])]

    # Plot Delta Hit Rate (%) for C2 vs C1 (Base FA-LRU vs LRU) and C4 vs C3 (CSTP FA-LRU vs CSTP LRU)
    for w_key, w_title, ax in workloads:
        sub_w = pair_df[pair_df["workload"] == w_key]

        c2_c1 = sub_w[sub_w["comparison_id"] == "C2_vs_C1"].set_index("capacity_id").reindex(CAP_ORDER).reset_index()
        c4_c3 = sub_w[sub_w["comparison_id"] == "C4_vs_C3"].set_index("capacity_id").reindex(CAP_ORDER).reset_index()

        y1 = c2_c1["delta_coalesced_hbm_hit_rate"] * 100.0
        y2 = c4_c3["delta_coalesced_hbm_hit_rate"] * 100.0

        ax.plot(range(3), y1, marker="s", color="#2ca02c", label="C2 vs C1: Base FA-LRU vs LRU (Δ Hit Rate %)")
        ax.plot(range(3), y2, marker="D", color="#d62728", label="C4 vs C3: CSTP FA-LRU vs LRU (Δ Hit Rate %)")
        ax.axhline(0, color="gray", linestyle=":", linewidth=1.5)

        ax.set_title(w_title)
        ax.set_xticks(range(3))
        ax.set_xticklabels(CAP_LABELS)
        ax.set_xlabel("HBM Cache Capacity")
        ax.set_ylabel("Absolute Hit Rate Change: Δ(FA-LRU - LRU) (%)")
        ax.legend(loc="upper right")

    fig.suptitle("Figure 10: FA-LRU Absolute Hit Rate Improvement Over Standard LRU", y=1.02)
    save_fig(fig, "fig10_falru_relative_improvement")


def main():
    verify_zero_ternary(__file__)
    os.makedirs(PLOTS_PNG_DIR, exist_ok=True)
    os.makedirs(PLOTS_PDF_DIR, exist_ok=True)

    if not os.path.exists(ALL_CSV) or not os.path.exists(PAIRED_CSV):
        print(f"[WAIT] Consolidated CSVs not yet available: {ALL_CSV}")
        return

    df = pd.read_csv(ALL_CSV)
    pair_df = pd.read_csv(PAIRED_CSV)

    print(f"[PLOTTING] Generating 10 publication figures from {len(df)} runs...")
    plot_fig1_hit_rate(df)
    plot_fig2_reloads(df)
    plot_fig3_evictions(df)
    plot_fig4_ddr5_traffic(df)
    plot_fig5_exposed_stall(df)
    plot_fig6_token_latency(df)
    plot_fig7_prefetch_coverage(df)
    plot_fig8_unused_prefetches(df)
    plot_fig9_prefetch_induced_evictions(df)
    plot_fig10_relative_improvement(pair_df)
    print("[PLOTTING COMPLETE] All 10 figures successfully generated in PNG and PDF!")


if __name__ == "__main__":
    main()

