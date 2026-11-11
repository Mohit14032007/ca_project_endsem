#!/usr/bin/env python3
"""
Publication Plotting Script for Big Experiment: LRU vs FA-LRU (W=64)
Generates high-resolution PNG (300 DPI) and vector PDF figures.
Enforces strictly zero ternary operators (ast.IfExp == 0).
"""

import os
import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = "/home/mohman/ca_project"
BIG_EXP_ROOT = os.path.join(WORKSPACE, "mohman", "point_4", "big_exp")
RESULTS_DIR = os.path.join(BIG_EXP_ROOT, "results")
PLOTS_PNG_DIR = os.path.join(BIG_EXP_ROOT, "figures", "png")
PLOTS_PDF_DIR = os.path.join(BIG_EXP_ROOT, "figures", "pdf")

ALL_CSV = os.path.join(RESULTS_DIR, "all_runs.csv")
PAIRED_CSV = os.path.join(RESULTS_DIR, "paired_comparisons.csv")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "grid.alpha": 0.35,
    "grid.linestyle": "--"
})


def verify_zero_ternary(file_path):
    with open(file_path, "r") as f:
        tree = ast.parse(f.read())
    ternary_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.IfExp)]
    if len(ternary_nodes) > 0:
        raise ValueError(f"Found {len(ternary_nodes)} ternary operators in {file_path}")
    print(f"[AST_CHECK] Verified 0 ternary operators in {file_path}")


def save_fig(fig, base_name):
    png_path = os.path.join(PLOTS_PNG_DIR, f"{base_name}.png")
    pdf_path = os.path.join(PLOTS_PDF_DIR, f"{base_name}.pdf")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {base_name}.png and .pdf")


def plot_fig1_hit_rate(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    workloads = [("w3", "Workload W3 (B=14, T=128)", axes[0]),
                 ("w4", "Workload W4 (B=32, T=128)", axes[1])]

    for w_key, w_title, ax in workloads:
        sub = df[(df["workload"] == w_key) & (df["repetition"] == 1)]
        # Configurations: C1 (LRU), C2 (FA-LRU), C3 (CSTP LRU), C4 (CSTP FA-LRU)
        labels = ["Base LRU", "Base FA-LRU", "CSTP LRU", "CSTP FA-LRU"]
        configs = ["c1", "c2", "c3", "c4"]
        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

        hit_rates = []
        for c in configs:
            val = sub[sub["config_id"] == c]["coalesced_hbm_hit_rate"].values[0] * 100.0
            hit_rates.append(val)

        x = np.arange(len(labels))
        bars = ax.bar(x, hit_rates, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_title(w_title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel("Coalesced HBM Hit Rate (%)")
        ax.grid(True, axis="y")

    fig.suptitle("Figure 1: Coalesced HBM Hit Rate Comparison (Standard LRU vs. FA-LRU W=64)", y=1.02)
    save_fig(fig, "fig1_hit_rate_comparison")


def plot_fig2_reloads(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    workloads = [("w3", "Workload W3 (B=14, T=128)", axes[0]),
                 ("w4", "Workload W4 (B=32, T=128)", axes[1])]

    for w_key, w_title, ax in workloads:
        sub = df[(df["workload"] == w_key) & (df["repetition"] == 1)]
        labels = ["Base LRU", "Base FA-LRU", "CSTP LRU", "CSTP FA-LRU"]
        configs = ["c1", "c2", "c3", "c4"]
        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

        reloads = []
        for c in configs:
            val = sub[sub["config_id"] == c]["reloaded_experts_count"].values[0]
            reloads.append(val)

        x = np.arange(len(labels))
        bars = ax.bar(x, reloads, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{int(height):,}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

        ax.set_title(w_title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel("Expert Reload Count")
        ax.grid(True, axis="y")

    fig.suptitle("Figure 2: Expert Reload Count Comparison", y=1.02)
    save_fig(fig, "fig2_expert_reloads_comparison")


def plot_fig3_traffic(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    workloads = [("w3", "Workload W3 (B=14, T=128)", axes[0]),
                 ("w4", "Workload W4 (B=32, T=128)", axes[1])]

    for w_key, w_title, ax in workloads:
        sub = df[(df["workload"] == w_key) & (df["repetition"] == 1)]
        labels = ["Base LRU", "Base FA-LRU", "CSTP LRU", "CSTP FA-LRU"]
        configs = ["c1", "c2", "c3", "c4"]

        dem_gb = []
        spec_gb = []
        for c in configs:
            r = sub[sub["config_id"] == c].iloc[0]
            dem_gb.append(r["demand_gb"])
            spec_gb.append(r["speculative_gb"])

        x = np.arange(len(labels))
        width = 0.55
        p1 = ax.bar(x, dem_gb, width, label="Demand Traffic (GB)", color="#1f77b4", edgecolor="black", linewidth=0.8)
        p2 = ax.bar(x, spec_gb, width, bottom=dem_gb, label="Speculative Prefetch Traffic (GB)", color="#aec7e8", edgecolor="black", linewidth=0.8)

        for i in range(len(labels)):
            tot = dem_gb[i] + spec_gb[i]
            ax.annotate(f"{tot:.1f} GB",
                        xy=(x[i], tot),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_title(w_title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel("Total DDR5 Traffic (Decimal GB = $10^9$ B)")
        ax.grid(True, axis="y")
        ax.legend(loc="upper left")

    fig.suptitle("Figure 3: DDR5 Memory Traffic Breakdown", y=1.02)
    save_fig(fig, "fig3_ddr5_traffic_comparison")


def plot_fig4_exposed_stall(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    workloads = [("w3", "Workload W3 (B=14, T=128)", axes[0]),
                 ("w4", "Workload W4 (B=32, T=128)", axes[1])]

    for w_key, w_title, ax in workloads:
        sub = df[(df["workload"] == w_key) & (df["repetition"] == 1)]
        labels = ["Base LRU", "Base FA-LRU", "CSTP LRU", "CSTP FA-LRU"]
        configs = ["c1", "c2", "c3", "c4"]
        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

        stalls = []
        for c in configs:
            val = sub[sub["config_id"] == c]["exposed_gpu_stall_ms"].values[0] / 1000.0  # seconds
            stalls.append(val)

        x = np.arange(len(labels))
        bars = ax.bar(x, stalls, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}s",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

        ax.set_title(w_title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel("Exposed GPU Memory Stall Time (seconds)")
        ax.grid(True, axis="y")

    fig.suptitle("Figure 4: Exposed Memory Stall Time Comparison", y=1.02)
    save_fig(fig, "fig4_exposed_stall_comparison")


def plot_fig5_token_latency(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    workloads = [("w3", "Workload W3 (B=14, T=128)", axes[0]),
                 ("w4", "Workload W4 (B=32, T=128)", axes[1])]

    for w_key, w_title, ax in workloads:
        sub = df[(df["workload"] == w_key) & (df["repetition"] == 1)]
        labels = ["Base LRU", "Base FA-LRU", "CSTP LRU", "CSTP FA-LRU"]
        configs = ["c1", "c2", "c3", "c4"]

        mean_lat = []
        p95_lat = []
        for c in configs:
            r = sub[sub["config_id"] == c].iloc[0]
            mean_lat.append(r["mean_token_latency_ms"])
            p95_lat.append(r["p95_token_latency_ms"])

        x = np.arange(len(labels))
        width = 0.35

        r1 = ax.bar(x - width/2, mean_lat, width, label="Mean Latency (ms)", color="#3470a3", edgecolor="black", linewidth=0.8)
        r2 = ax.bar(x + width/2, p95_lat, width, label="P95 Latency (ms)", color="#f28e2b", edgecolor="black", linewidth=0.8)

        ax.set_title(w_title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel("Simulated Token Latency (ms)")
        ax.grid(True, axis="y")
        ax.legend(loc="upper left")

    fig.suptitle("Figure 5: Mean and P95 Token Latency", y=1.02)
    save_fig(fig, "fig5_token_latency_comparison")


def plot_fig6_cstp_pollution(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    workloads = [("w3", "Workload W3 (B=14, T=128)", axes[0]),
                 ("w4", "Workload W4 (B=32, T=128)", axes[1])]

    for w_key, w_title, ax in workloads:
        sub = df[(df["workload"] == w_key) & (df["repetition"] == 1)]
        c1 = sub[sub["config_id"] == "c1"].iloc[0]
        c2 = sub[sub["config_id"] == "c2"].iloc[0]
        c3 = sub[sub["config_id"] == "c3"].iloc[0]
        c4 = sub[sub["config_id"] == "c4"].iloc[0]

        # Delta Hit Rate vs Baseline
        labels = ["LRU: CSTP vs Base", "FA-LRU: CSTP vs Base"]
        d_hit = [
            (c3["coalesced_hbm_hit_rate"] - c1["coalesced_hbm_hit_rate"]) * 100.0,
            (c4["coalesced_hbm_hit_rate"] - c2["coalesced_hbm_hit_rate"]) * 100.0
        ]
        colors = ["#d62728", "#d62728"]
        if d_hit[0] >= 0:
            colors[0] = "#2ca02c"
        if d_hit[1] >= 0:
            colors[1] = "#2ca02c"

        x = np.arange(len(labels))
        bars = ax.bar(x, d_hit, color=colors, width=0.45, edgecolor="black", linewidth=0.8)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0)

        for bar in bars:
            height = bar.get_height()
            va = "bottom"
            y_pos = height
            if height < 0:
                va = "top"
            y_offset = 4
            if height < 0:
                y_offset = -12
            ax.annotate(f"{height:+.3f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                        xytext=(0, y_offset), textcoords="offset points",
                        ha="center", va=va, fontsize=10, fontweight="bold")

        ax.set_title(w_title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Hit Rate Delta Δ(CSTP - Base) (%)")
        ax.grid(True, axis="y")

    fig.suptitle("Figure 6: Speculative CSTP Impact on Hit Rate (Cache Pollution Analysis)", y=1.02)
    save_fig(fig, "fig6_cstp_cache_pollution")


def plot_fig7_diagnostics(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    workloads = [("w3", "Workload W3 (B=14, T=128)", axes[0]),
                 ("w4", "Workload W4 (B=32, T=128)", axes[1])]

    for w_key, w_title, ax in workloads:
        sub = df[(df["workload"] == w_key) & (df["repetition"] == 1)]
        c2 = sub[sub["config_id"] == "c2"].iloc[0]
        c4 = sub[sub["config_id"] == "c4"].iloc[0]

        cats = ["Divergent Victims", "Protected Demands", "Unconsumed Spec Victims", "Protected Reused"]
        y_c2 = [
            c2["fa_lru_divergent_decisions"],
            c2["fa_lru_protected_demand_experts"],
            c2["fa_lru_unconsumed_speculative_victims"],
            c2["fa_lru_protected_experts_reused"]
        ]
        y_c4 = [
            c4["fa_lru_divergent_decisions"],
            c4["fa_lru_protected_demand_experts"],
            c4["fa_lru_unconsumed_speculative_victims"],
            c4["fa_lru_protected_experts_reused"]
        ]

        x = np.arange(len(cats))
        width = 0.35

        b1 = ax.bar(x - width/2, y_c2, width, label="C2: Base FA-LRU", color="#2ca02c", edgecolor="black", linewidth=0.8)
        b2 = ax.bar(x + width/2, y_c4, width, label="C4: CSTP FA-LRU", color="#d62728", edgecolor="black", linewidth=0.8)

        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.annotate(f"{int(h)}",
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

        ax.set_title(w_title)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=15)
        ax.set_ylabel("Event Count")
        ax.grid(True, axis="y")
        ax.legend(loc="upper right")

    fig.suptitle("Figure 7: FA-LRU (W=64) Eviction Diagnostic Events", y=1.02)
    save_fig(fig, "fig7_falru_diagnostics")


def plot_fig8_reproducibility(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    workloads = [("w3", "Workload W3 (B=14)", axes[0]),
                 ("w4", "Workload W4 (B=32)", axes[1])]

    for w_key, w_title, ax in workloads:
        sub = df[df["workload"] == w_key]
        configs = ["c1", "c2", "c3", "c4"]
        labels = ["C1", "C2", "C3", "C4"]

        r1_stalls = []
        r2_stalls = []
        for c in configs:
            val1 = sub[(sub["config_id"] == c) & (sub["repetition"] == 1)]["exposed_gpu_stall_ms"].values[0]
            val2 = sub[(sub["config_id"] == c) & (sub["repetition"] == 2)]["exposed_gpu_stall_ms"].values[0]
            r1_stalls.append(val1)
            r2_stalls.append(val2)

        x = np.arange(len(configs))
        width = 0.35

        ax.bar(x - width/2, r1_stalls, width, label="Repetition 1", color="#4e79a7", edgecolor="black", linewidth=0.8)
        ax.bar(x + width/2, r2_stalls, width, label="Repetition 2", color="#59a14f", edgecolor="black", linewidth=0.8)

        ax.set_title(w_title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Exposed GPU Stall (ms)")
        ax.grid(True, axis="y")
        ax.legend(loc="upper right")

    fig.suptitle("Figure 8: Deterministic Repetition Verification (Rep 1 vs. Rep 2)", y=1.02)
    save_fig(fig, "fig8_repetition_consistency")


def main():
    verify_zero_ternary(__file__)
    os.makedirs(PLOTS_PNG_DIR, exist_ok=True)
    os.makedirs(PLOTS_PDF_DIR, exist_ok=True)

    if not os.path.exists(ALL_CSV):
        print(f"[WAIT] Results CSV not found: {ALL_CSV}")
        return

    df = pd.read_csv(ALL_CSV)
    print(f"[PLOTTING] Generating 8 publication figures from {len(df)} runs...")

    plot_fig1_hit_rate(df)
    plot_fig2_reloads(df)
    plot_fig3_traffic(df)
    plot_fig4_exposed_stall(df)
    plot_fig5_token_latency(df)
    plot_fig6_cstp_pollution(df)
    plot_fig7_diagnostics(df)
    plot_fig8_reproducibility(df)

    print("[PLOTTING COMPLETE] All 8 figures successfully generated in PNG and PDF!")


if __name__ == "__main__":
    main()
