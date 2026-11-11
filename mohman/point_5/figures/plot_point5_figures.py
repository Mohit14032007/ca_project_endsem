#!/usr/bin/env python3
"""
Plot Publication-Quality Figures for Point 5 Comparative Evaluation.
Generates Figures 1, 2, 3, and 4 in PNG (300 DPI) and PDF formats.
Enforces strictly zero ternary operators (ast.IfExp == 0).
"""

import os
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKSPACE = "/home/mohman/ca_project"
POINT_5_ROOT = os.path.join(WORKSPACE, "mohman", "point_5")
FIGURES_DIR = os.path.join(POINT_5_ROOT, "figures")
PNG_DIR = os.path.join(FIGURES_DIR, "png")
PDF_DIR = os.path.join(FIGURES_DIR, "pdf")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "grid.alpha": 0.4,
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
    # Save directly in figures/
    png_path = os.path.join(FIGURES_DIR, f"{base_name}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{base_name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    # Mirror in png/ and pdf/
    png_sub = os.path.join(PNG_DIR, f"{base_name}.png")
    pdf_sub = os.path.join(PDF_DIR, f"{base_name}.pdf")
    fig.savefig(png_sub, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_sub, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {base_name}.png and .pdf")

def plot_fig1_predictor_effectiveness():
    """Figure 1: Predictor Effectiveness (ST-MoE vs CSTP)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Panel 1: Offline MoE-Anywhere Sage-A1.5 Metrics (Point 1)
    ax1 = axes[0]
    metrics = ["Macro Recall", "Macro Precision", "Macro F1", "Macro Hit Rate"]
    stmoe_h1 = [0.4791 * 100, 0.4791 * 100, 0.4791 * 100, 0.9694 * 100]
    cstp_h1 = [0.4551 * 100, 0.4551 * 100, 0.4551 * 100, 0.9667 * 100]
    stmoe_h2 = [0.4620 * 100, 0.4620 * 100, 0.4620 * 100, 0.9585 * 100]
    cstp_h2 = [0.4418 * 100, 0.4418 * 100, 0.4418 * 100, 0.9614 * 100]
    
    x = np.arange(len(metrics))
    width = 0.20
    
    rects1 = ax1.bar(x - 1.5 * width, stmoe_h1, width, label="ST-MoE (h=1)", color="#3498db", edgecolor="black")
    rects2 = ax1.bar(x - 0.5 * width, cstp_h1, width, label="CSTP (h=1)", color="#e67e22", edgecolor="black")
    rects3 = ax1.bar(x + 0.5 * width, stmoe_h2, width, label="ST-MoE (h=2)", color="#2980b9", edgecolor="black", alpha=0.7)
    rects4 = ax1.bar(x + 1.5 * width, cstp_h2, width, label="CSTP (h=2)", color="#d35400", edgecolor="black", alpha=0.7)
    
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Offline Evaluation (MoE-Anywhere Sage-A1.5 Trace)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=15)
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(True, axis="y")
    
    # Panel 2: In-Situ Simulation Metrics (phase_4d5c Shadow Mode)
    ax2 = axes[1]
    in_situ_metrics = ["Precision", "Recall", "F1 Score", "Hit Rate", "MRR", "NDCG"]
    stmoe_insitu = [0.3387 * 100, 0.1693 * 100, 0.2258 * 100, 0.7360 * 100, 0.5747 * 100, 0.3684 * 100]
    cstp_insitu = [0.3400 * 100, 0.1700 * 100, 0.2267 * 100, 0.7307 * 100, 0.5799 * 100, 0.3713 * 100]
    
    x2 = np.arange(len(in_situ_metrics))
    width2 = 0.35
    
    r1 = ax2.bar(x2 - width2 / 2, stmoe_insitu, width2, label="ST-MoE (Shadow Mode)", color="#2ecc71", edgecolor="black")
    r2 = ax2.bar(x2 + width2 / 2, cstp_insitu, width2, label="CSTP (Shadow Mode)", color="#9b59b6", edgecolor="black")
    
    ax2.set_ylabel("Score (%)")
    ax2.set_title("In-Situ Simulator Metrics (Historical phase_4d5c, 10 Tokens)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(in_situ_metrics, rotation=15)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(True, axis="y")
    
    fig.suptitle("Figure 1: Predictor Effectiveness — ST-MoE vs. CSTP Across Offline and In-Situ Regimes", fontsize=13, fontweight="bold", y=1.02)
    save_fig(fig, "fig1_predictor_effectiveness")

def plot_fig2_prefetching_contribution():
    """Figure 2: Prefetching Contribution (Baseline vs ST-MoE vs CSTP)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Panel 1: Exposed Memory Stall (ms) across Point 3 Large Workloads
    ax1 = axes[0]
    workloads = ["W2 (B=14)", "W3 (B=32)", "W4 (B=48)", "W5 (B=70)"]
    stall_base = [18117.03, 33639.29, 50768.71, 62304.49]
    stall_stmoe = [18067.89, 33556.78, 50668.16, 62043.92]
    stall_cstp = [18055.42, 33529.28, 50566.14, 61794.55]
    
    x = np.arange(len(workloads))
    width = 0.25
    
    ax1.bar(x - width, stall_base, width, label="Baseline (No Prefetch)", color="#7f8c8d", edgecolor="black")
    ax1.bar(x, stall_stmoe, width, label="ST-MoE Prefetch (h=2)", color="#3498db", edgecolor="black")
    ax1.bar(x + width, stall_cstp, width, label="CSTP Prefetch (h=2)", color="#e67e22", edgecolor="black")
    
    ax1.set_ylabel("Exposed GPU Memory Stall (ms)")
    ax1.set_title("Exposed Stall Reduction Across Batch Scaling (Point 3, T=16)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(workloads)
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(True, axis="y")
    
    # Panel 2: Memory Traffic Overhead (%) and Stall Reduction (ms)
    ax2 = axes[1]
    traffic_stmoe = [42.44, 42.25, 32.79, 28.53]
    traffic_cstp = [22.59, 21.78, 16.10, 13.42]
    
    ax2.plot(workloads, traffic_stmoe, marker="s", markersize=7, color="#2980b9", linewidth=2.0, label="ST-MoE Traffic Overhead (%)")
    ax2.plot(workloads, traffic_cstp, marker="o", markersize=7, color="#d35400", linewidth=2.0, label="CSTP Traffic Overhead (%)")
    
    ax2.set_ylabel("DDR5 Memory Traffic Overhead (%)")
    ax2.set_title("DDR5 Bus Traffic Overhead Comparison")
    ax2.set_ylim(0, 50)
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(True)
    
    fig.suptitle("Figure 2: Prefetching Contribution — Exposed Stall Reduction vs. Traffic Overhead Tradeoff", fontsize=13, fontweight="bold", y=1.02)
    save_fig(fig, "fig2_prefetching_contribution")

def plot_fig3_falru_contribution():
    """Figure 3: FA-LRU Contribution (CSTP + LRU vs CSTP + FA-LRU W=64)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Panel 1: Coalesced Resident Hits Comparison
    ax1 = axes[0]
    conditions = ["W3 (B=14, T=128)", "W4 (B=32, T=128)"]
    lru_hits = [1657, 1417]
    falru_hits = [1643, 1432]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    rects1 = ax1.bar(x - width / 2, lru_hits, width, label="CSTP + Standard LRU", color="#34495e", edgecolor="black")
    rects2 = ax1.bar(x + width / 2, falru_hits, width, label="CSTP + FA-LRU (W=64)", color="#27ae60", edgecolor="black")
    
    for r in rects1:
        ax1.annotate(f"{int(r.get_height()):,}", xy=(r.get_x() + r.get_width() / 2, r.get_height()),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    for r in rects2:
        ax1.annotate(f"{int(r.get_height()):,}", xy=(r.get_x() + r.get_width() / 2, r.get_height()),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
                     
    ax1.set_ylabel("Coalesced Resident Hits")
    ax1.set_title("Resident Hit Retention: W3 (-14 hits) vs. W4 (+15 hits)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions)
    ax1.set_ylim(0, 2000)
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(True, axis="y")
    
    # Panel 2: Memory Traffic and Stall Deltas in W4 (B=32)
    ax2 = axes[1]
    metrics_w4 = ["DDR5 Traffic\nSaved (GB)", "Exposed Stall\nSaved (ms)", "Execution Time\nSaved (ms)", "Protected Lines\nReused (Hits)"]
    values_w4 = [2.0258, 31.82, 31.65, 36.0]
    colors_w4 = ["#16a085", "#2980b9", "#8e44ad", "#27ae60"]
    
    x2 = np.arange(len(metrics_w4))
    bars = ax2.bar(x2, values_w4, width=0.45, color=colors_w4, edgecolor="black")
    
    for b in bars:
        h = b.get_height()
        ax2.annotate(f"+{h:.2f}", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold")
                     
    ax2.set_title("FA-LRU Incremental Physical Savings in Workload W4 (B=32, T=128)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics_w4)
    ax2.set_ylim(0, 45)
    ax2.grid(True, axis="y")
    
    fig.suptitle("Figure 3: FA-LRU (W=64) Eviction Policy Contribution Under CSTP Prefetching", fontsize=13, fontweight="bold", y=1.02)
    save_fig(fig, "fig3_falru_contribution")

def plot_fig4_overall_system_comparison():
    """Figure 4: Overall System Comparison across Conceptual Configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Primary Workload W4 (B=32, T=128, M=35 GiB)
    configs = [
        "Config A\nBaseline",
        "Config B\nPred-Only",
        "Config C2\nCSTP+LRU",
        "Config D\nComplete"
    ]
    
    # Panel 1: Simulated GPU Latency Relative to Baseline (Normalized)
    ax1 = axes[0]
    # Baseline time = 1008.204 s, Pred-Only = 1008.204 s, CSTP+LRU = 761.608 s, Complete = 761.576 s
    lat_w4 = [1.0, 1.0, 761.60755 / 1008.204, 761.57589 / 1008.204]
    lat_norm = [v * 100.0 for v in lat_w4]
    
    colors1 = ["#7f8c8d", "#95a5a6", "#e67e22", "#27ae60"]
    b1 = ax1.bar(configs, lat_norm, width=0.50, color=colors1, edgecolor="black")
    
    for bar in b1:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold")
                     
    ax1.set_ylabel("Normalized Simulated Execution Time (% of Baseline)")
    ax1.set_title("Execution Time Scaling (W4: B=32, T=128)")
    ax1.set_ylim(0, 115)
    ax1.axhline(100, color="gray", linestyle="--", linewidth=1.0)
    ax1.grid(True, axis="y")
    
    # Panel 2: Total DDR5 Memory Traffic (Normalized)
    ax2 = axes[1]
    # Baseline traffic = 48740.16 GB, Pred-only = 48740.16 GB, CSTP+LRU = 48738.84 GB, Complete = 48736.81 GB
    traf_w4 = [48740.16, 48740.16, 48738.84, 48736.81]
    traf_norm = [v / traf_w4[0] * 100.0 for v in traf_w4]
    
    b2 = ax2.bar(configs, traf_norm, width=0.50, color=colors1, edgecolor="black")
    
    for bar in b2:
        h = bar.get_height()
        ax2.annotate(f"{h:.3f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold")
                     
    ax2.set_ylabel("Normalized Total DDR5 Traffic (% of Baseline)")
    ax2.set_title("DDR5 Traffic Scaling (W4: B=32, T=128)")
    ax2.set_ylim(99.98, 100.02) # Magnified around 100%
    ax2.axhline(100, color="gray", linestyle="--", linewidth=1.0)
    ax2.grid(True, axis="y")
    
    fig.suptitle("Figure 4: Overall System Comparison — Normalized Improvements Relative to Baseline", fontsize=13, fontweight="bold", y=1.02)
    save_fig(fig, "fig4_overall_system_comparison")

def main():
    verify_zero_ternary(__file__)
    plot_fig1_predictor_effectiveness()
    plot_fig2_prefetching_contribution()
    plot_fig3_falru_contribution()
    plot_fig4_overall_system_comparison()

if __name__ == "__main__":
    main()

