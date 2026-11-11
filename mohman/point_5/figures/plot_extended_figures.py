#!/usr/bin/env python3
"""
Generate Figures 5 through 12 for Point 5 Extended Comparative Visualization.
Outputs high-resolution PNG (300 DPI) and vector PDF formats.
Enforces strictly zero ternary operators (ast.IfExp == 0).
"""

import os
import ast
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = "/home/mohman/ca_project"
POINT_5_ROOT = os.path.join(WORKSPACE, "mohman", "point_5")
FIGURES_DIR = os.path.join(POINT_5_ROOT, "figures")
PNG_DIR = os.path.join(FIGURES_DIR, "png")
PDF_DIR = os.path.join(FIGURES_DIR, "pdf")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
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
    png_path = os.path.join(FIGURES_DIR, f"{base_name}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{base_name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    png_sub = os.path.join(PNG_DIR, f"{base_name}.png")
    pdf_sub = os.path.join(PDF_DIR, f"{base_name}.pdf")
    fig.savefig(png_sub, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_sub, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {base_name}.png and .pdf")

def plot_fig5_execution_time():
    """Figure 5: End-to-End Execution Time Comparison (Workload W4)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    configs = ["Config A\nBaseline", "Config B\nPred-Only", "Config C2\nCSTP+LRU", "Config D\nComplete"]
    colors = ["#7f8c8d", "#95a5a6", "#e67e22", "#27ae60"]
    
    # Panel A: Absolute Simulated Execution Time (s)
    ax1 = axes[0]
    times_s = [1008.204, 1008.204, 761.608, 761.576]
    bars1 = ax1.bar(configs, times_s, width=0.50, color=colors, edgecolor="black", linewidth=1.0)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.2f} s", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Simulated Execution Time (seconds)")
    ax1.set_title("Panel A: Absolute Execution Time (W4: B=32, T=128)")
    ax1.set_ylim(0, 1180)
    ax1.grid(True, axis="y")
    
    # Delta callout between C and D
    ax1.annotate("FA-LRU Δ:\n-31.8 ms", xy=(3.0, 761.576), xytext=(2.5, 880),
                 ha="center", fontsize=8.5, fontweight="bold", color="#145a32",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", edgecolor="#27ae60"),
                 arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2))
                 
    # Panel B: Normalized Execution Time and Percentage Improvement
    ax2 = axes[1]
    norm_pct = [100.0, 100.0, 75.541, 75.538]
    bars2 = ax2.bar(configs, norm_pct, width=0.50, color=colors, edgecolor="black", linewidth=1.0)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.3f}%", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Normalized Execution Time (% of Baseline)")
    ax2.set_title("Panel B: Normalized Latency (% of Baseline)")
    ax2.set_ylim(0, 118)
    ax2.axhline(100, color="gray", linestyle="--", linewidth=1.0)
    ax2.grid(True, axis="y")
    
    # Overlay improvement label
    ax2.text(2.0, 40, "Prefetch Speedup:\n+24.459%", ha="center", fontsize=9, fontweight="bold", color="#d35400",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#e67e22"))
    ax2.text(3.0, 40, "Complete Speedup:\n+24.462%", ha="center", fontsize=9, fontweight="bold", color="#145a32",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#27ae60"))
             
    fig.suptitle("Figure 5: End-to-End Execution Time Comparison Across Conceptual Configurations (Workload W4)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    save_fig(fig, "fig5_execution_time_comparison")

def plot_fig6_exposed_stall():
    """Figure 6: Exposed Memory Stall Comparison (Workload W4)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    configs = ["Config A\nBaseline", "Config B\nPred-Only", "Config C2\nCSTP+LRU", "Config D\nComplete"]
    colors = ["#7f8c8d", "#95a5a6", "#e67e22", "#27ae60"]
    
    # Panel A: Absolute Exposed Memory Stall (ms)
    ax1 = axes[0]
    stalls_ms = [994459.73, 994459.73, 747863.57, 747831.75]
    bars1 = ax1.bar(configs, stalls_ms, width=0.50, color=colors, edgecolor="black", linewidth=1.0)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:,.0f} ms", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Exposed GPU Memory Stall (ms)")
    ax1.set_title("Panel A: Absolute Exposed Memory Stall (W4: B=32, T=128)")
    ax1.set_ylim(0, 1150000)
    ax1.grid(True, axis="y")
    
    # Panel B: Percentage Stall Reduction Relative to Baseline
    ax2 = axes[1]
    reductions_pct = [0.0, 0.0, 24.797, 24.800]
    bars2 = ax2.bar(configs, reductions_pct, width=0.50, color=colors, edgecolor="black", linewidth=1.0)
    for b in bars2:
        h = b.get_height()
        if h > 0:
            ax2.annotate(f"-{h:.3f}%", xy=(b.get_x() + b.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold")
        else:
            ax2.annotate("0.00%", xy=(b.get_x() + b.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax2.set_ylabel("Exposed Stall Reduction (% of Baseline)")
    ax2.set_title("Panel B: Percentage Reduction in Exposed Memory Stall")
    ax2.set_ylim(0, 30)
    ax2.grid(True, axis="y")
    
    # Transition Callouts
    ax2.annotate("Transition A -> C:\n-246,596.16 ms Stall Avoided\n(-24.80% / 4.11 min)", xy=(2.0, 24.797), xytext=(1.2, 17),
                 ha="center", fontsize=8.5, fontweight="bold", color="#d35400",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef5e7", edgecolor="#e67e22"),
                 arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.2))
                 
    ax2.annotate("Transition C -> D:\n-31.82 ms FA-LRU Savings", xy=(3.0, 24.800), xytext=(3.0, 10),
                 ha="center", fontsize=8.5, fontweight="bold", color="#145a32",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", edgecolor="#27ae60"),
                 arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2))
                 
    fig.suptitle("Figure 6: Exposed GPU Memory Stall Comparison and Transition Breakdown (Workload W4)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    save_fig(fig, "fig6_exposed_stall_comparison")

def plot_fig7_ddr5_traffic():
    """Figure 7: DDR5 Traffic Composition (Demand vs Speculative)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Panel A: Workload W3 (B=14, T=128)
    ax1 = axes[0]
    configs_w3 = ["Config A\nBaseline", "Config B\nPred-Only", "Config C2\nCSTP+LRU", "Config D\nComplete"]
    demand_w3 = [28824.39, 28824.39, 21972.27, 21978.17]
    spec_w3 = [0.0, 0.0, 7245.05, 7240.03]
    
    x1 = np.arange(len(configs_w3))
    width = 0.50
    p1 = ax1.bar(x1, demand_w3, width, label="Demand Transfer Traffic (GB)", color="#2980b9", edgecolor="black")
    p2 = ax1.bar(x1, spec_w3, width, bottom=demand_w3, label="Speculative Prefetch Traffic (GB)", color="#e74c3c", edgecolor="black")
    
    total_w3 = [d + s for d, s in zip(demand_w3, spec_w3)]
    for i, tot in enumerate(total_w3):
        ax1.annotate(f"{tot:,.1f} GB", xy=(x1[i], tot), xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=8.5, fontweight="bold")
    ax1.set_ylabel("DDR5 Traffic (decimal GB)")
    ax1.set_title("Panel A: Workload W3 (B=14, T=128, M=35 GiB)")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(configs_w3)
    ax1.set_ylim(0, 35000)
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(True, axis="y")
    
    # Panel B: Workload W4 (B=32, T=128)
    ax2 = axes[1]
    configs_w4 = ["Config A\nBaseline", "Config B\nPred-Only", "Config C2\nCSTP+LRU", "Config D\nComplete"]
    demand_w4 = [48740.16, 48740.16, 37790.54, 37808.77]
    spec_w4 = [0.0, 0.0, 10948.30, 10928.05]
    
    x2 = np.arange(len(configs_w4))
    p3 = ax2.bar(x2, demand_w4, width, label="Demand Transfer Traffic (GB)", color="#2980b9", edgecolor="black")
    p4 = ax2.bar(x2, spec_w4, width, bottom=demand_w4, label="Speculative Prefetch Traffic (GB)", color="#e74c3c", edgecolor="black")
    
    total_w4 = [d + s for d, s in zip(demand_w4, spec_w4)]
    for i, tot in enumerate(total_w4):
        ax2.annotate(f"{tot:,.1f} GB", xy=(x2[i], tot), xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=8.5, fontweight="bold")
    ax2.set_ylabel("DDR5 Traffic (decimal GB)")
    ax2.set_title("Panel B: Workload W4 (B=32, T=128, M=35 GiB)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(configs_w4)
    ax2.set_ylim(0, 58000)
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(True, axis="y")
    
    # Note on traffic neutrality and decomposition
    ax2.text(0.5, 0.05, "Note: In W4, FA-LRU reduces speculative traffic (-20.26 GB) with demand shift (+18.23 GB), saving -2.03 GB net (-23 blocks).",
             transform=ax2.transAxes, ha="center", fontsize=8.0, color="#333333", style="italic")
             
    fig.suptitle("Figure 7: DDR5 Memory Traffic Composition — Demand vs. Speculative Transfers (W3 and W4)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    save_fig(fig, "fig7_ddr5_traffic_composition")

def plot_fig8_hbm_hits_reloads():
    """Figure 8: HBM Hit Rate and Expert Reload Comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Panel A: Coalesced Resident Hits Comparison
    ax1 = axes[0]
    conditions = ["W3: LRU\n(B=14)", "W3: FA-LRU\n(B=14)", "W4: LRU\n(B=32)", "W4: FA-LRU\n(B=32)"]
    hits = [1657, 1643, 1417, 1432]
    colors = ["#34495e", "#2980b9", "#34495e", "#27ae60"]
    
    bars1 = ax1.bar(conditions, hits, width=0.45, color=colors, edgecolor="black")
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:,}", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Coalesced Resident Hits")
    ax1.set_title("Panel A: Coalesced Resident Hits (CSTP Prefetching Active)")
    ax1.set_ylim(0, 2000)
    ax1.grid(True, axis="y")
    
    # Deltas
    ax1.annotate("W3 Delta:\n-14 hits", xy=(0.5, 1643), xytext=(0.5, 1000), ha="center",
                 fontsize=8.5, fontweight="bold", color="#922b21",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#fdedec", edgecolor="#c0392b"))
    ax1.annotate("W4 Delta:\n+15 hits", xy=(2.5, 1432), xytext=(2.5, 900), ha="center",
                 fontsize=8.5, fontweight="bold", color="#145a32",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#eafaf1", edgecolor="#27ae60"))
                 
    # Panel B: Expert Reloads from DDR5
    ax2 = axes[1]
    reloads = [316227, 316241, 536034, 536019]
    bars2 = ax2.bar(conditions, reloads, width=0.45, color=colors, edgecolor="black")
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:,}", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
    ax2.set_ylabel("Expert Reloads from DDR5 (Count)")
    ax2.set_title("Panel B: Expert Reloads from DDR5 (Denominator: Actual Requests)")
    ax2.set_ylim(0, 620000)
    ax2.grid(True, axis="y")
    
    # Deltas
    ax2.annotate("W3 Delta:\n+14 reloads", xy=(0.5, 316241), xytext=(0.5, 420000), ha="center",
                 fontsize=8.5, fontweight="bold", color="#922b21",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#fdedec", edgecolor="#c0392b"))
    ax2.annotate("W4 Delta:\n-15 reloads", xy=(2.5, 536019), xytext=(2.5, 420000), ha="center",
                 fontsize=8.5, fontweight="bold", color="#145a32",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#eafaf1", edgecolor="#27ae60"))
                 
    fig.suptitle("Figure 8: HBM Expert Resident Hit Retention and DDR5 Reload Elimination",
                 fontsize=12.5, fontweight="bold", y=1.02)
    save_fig(fig, "fig8_hbm_hits_reloads")

def plot_fig9_predictor_quality():
    """Figure 9: Predictor Quality Comparison (ST-MoE vs CSTP at h=1 and h=2)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    # Exact values from Point 1 dataset
    horizons = ["Horizon h=1", "Horizon h=2"]
    x = np.arange(len(horizons))
    width = 0.35
    
    # Subplot 1: Macro Recall
    ax1 = axes[0, 0]
    stmoe_rec = [47.91, 46.20]
    cstp_rec = [45.51, 44.18]
    r1 = ax1.bar(x - width/2, stmoe_rec, width, label="ST-MoE", color="#3498db", edgecolor="black")
    r2 = ax1.bar(x + width/2, cstp_rec, width, label="CSTP", color="#e67e22", edgecolor="black")
    for b in r1:
        ax1.annotate(f"{b.get_height():.2f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    for b in r2:
        ax1.annotate(f"{b.get_height():.2f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Macro Recall (%)")
    ax1.set_title("1. Macro Recall (Exact-Set Match)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(horizons)
    ax1.set_ylim(0, 60)
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(True, axis="y")
    
    # Subplot 2: Macro Precision
    ax2 = axes[0, 1]
    stmoe_prec = [47.91, 46.20]
    cstp_prec = [45.51, 44.18]
    r3 = ax2.bar(x - width/2, stmoe_prec, width, label="ST-MoE", color="#3498db", edgecolor="black")
    r4 = ax2.bar(x + width/2, cstp_prec, width, label="CSTP", color="#e67e22", edgecolor="black")
    for b in r3:
        ax2.annotate(f"{b.get_height():.2f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    for b in r4:
        ax2.annotate(f"{b.get_height():.2f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Macro Precision (%)")
    ax2.set_title("2. Macro Precision (Exact-Set Match)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(horizons)
    ax2.set_ylim(0, 60)
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(True, axis="y")
    
    # Subplot 3: Macro F1 Score
    ax3 = axes[1, 0]
    stmoe_f1 = [47.91, 46.20]
    cstp_f1 = [45.51, 44.18]
    r5 = ax3.bar(x - width/2, stmoe_f1, width, label="ST-MoE", color="#3498db", edgecolor="black")
    r6 = ax3.bar(x + width/2, cstp_f1, width, label="CSTP", color="#e67e22", edgecolor="black")
    for b in r5:
        ax3.annotate(f"{b.get_height():.2f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    for b in r6:
        ax3.annotate(f"{b.get_height():.2f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax3.set_ylabel("Macro F1 (%)")
    ax3.set_title("3. Macro F1 Score")
    ax3.set_xticks(x)
    ax3.set_xticklabels(horizons)
    ax3.set_ylim(0, 60)
    ax3.legend(loc="upper right", fontsize=8.5)
    ax3.grid(True, axis="y")
    
    # Subplot 4: Macro Top-K Hit Rate
    ax4 = axes[1, 1]
    stmoe_hit = [96.94, 95.85]
    cstp_hit = [96.67, 96.14]
    r7 = ax4.bar(x - width/2, stmoe_hit, width, label="ST-MoE", color="#2ecc71", edgecolor="black")
    r8 = ax4.bar(x + width/2, cstp_hit, width, label="CSTP", color="#9b59b6", edgecolor="black")
    for b in r7:
        ax4.annotate(f"{b.get_height():.2f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    for b in r8:
        ax4.annotate(f"{b.get_height():.2f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax4.set_ylabel("Top-K Hit Rate (%)")
    ax4.set_title("4. Macro Top-K Hit Rate (At Least 1 Correct)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(horizons)
    ax4.set_ylim(80, 105) # Zoomed to highlight top region
    ax4.legend(loc="lower right", fontsize=8.5)
    ax4.grid(True, axis="y")
    
    fig.suptitle("Figure 9: Offline Predictor Quality Comparison (MoE-Anywhere Sage-A1.5 Dataset)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig9_predictor_quality_comparison")

def plot_fig10_prefetch_effectiveness():
    """Figure 10: Prefetch Effectiveness and Cache Pollution (ST-MoE vs CSTP)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Panel A: Prefetches Dispatched vs Late Attachments (Workload W6 and W7)
    ax1 = axes[0]
    workloads = ["W6: ST-MoE\n(B=70, T=32)", "W6: CSTP\n(B=70, T=32)", "W7: ST-MoE\n(B=70, T=64)", "W7: CSTP\n(B=70, T=64)"]
    dispatched = [35272, 17659, 102737, 54076]
    attachments = [32817, 16002, 96569, 50264]
    
    x1 = np.arange(len(workloads))
    width = 0.35
    b1 = ax1.bar(x1 - width/2, dispatched, width, label="Dispatched Prefetches", color="#3498db", edgecolor="black")
    b2 = ax1.bar(x1 + width/2, attachments, width, label="Late Attachments (Demanded In-Flight)", color="#2ecc71", edgecolor="black")
    
    for b in b1:
        ax1.annotate(f"{int(b.get_height()):,}", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
    for b in b2:
        ax1.annotate(f"{int(b.get_height()):,}", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
                     
    ax1.set_ylabel("Prefetch Count")
    ax1.set_title("Panel A: Dispatched Transfers vs. Demanded Attachments")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(workloads)
    ax1.set_ylim(0, 120000)
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(True, axis="y")
    
    # Panel B: Unused Completed Prefetches (Cache Pollution)
    ax2 = axes[1]
    unused_count = [2388, 1656, 6022, 3812]
    colors = ["#e74c3c", "#e67e22", "#e74c3c", "#e67e22"]
    
    bars2 = ax2.bar(workloads, unused_count, width=0.45, color=colors, edgecolor="black")
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:,}", xy=(b.get_x() + b.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Unused Completed Prefetches (Count)")
    ax2.set_title("Panel B: Speculative Cache Pollution (Unused Prefetches Evicted)")
    ax2.set_ylim(0, 7500)
    ax2.grid(True, axis="y")
    
    # Annotated comparison note
    ax2.annotate("CSTP Pollution Savings:\n-2,210 Unused Transfers (-36.7%)", xy=(3.0, 3812), xytext=(2.2, 5500),
                 ha="center", fontsize=8.5, fontweight="bold", color="#145a32",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", edgecolor="#27ae60"),
                 arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2))
                 
    fig.suptitle("Figure 10: Prefetch Effectiveness and Speculative Cache Pollution (Point 3 Large Workloads)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    save_fig(fig, "fig10_prefetch_effectiveness_pollution")

def plot_fig11_falru_contribution():
    """Figure 11: LRU vs FA-LRU Contribution (Big Experiment Paired Comparisons)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Panel A: Resident-Hit Comparison (W3 vs W4)
    ax1 = axes[0]
    cases = ["W3: LRU\n(B=14)", "W3: FA-LRU\n(B=14)", "W4: LRU\n(B=32)", "W4: FA-LRU\n(B=32)"]
    hits = [1657, 1643, 1417, 1432]
    colors = ["#34495e", "#c0392b", "#34495e", "#27ae60"]
    
    b1 = ax1.bar(cases, hits, width=0.45, color=colors, edgecolor="black")
    for b in b1:
        h = b.get_height()
        ax1.annotate(f"{h:,}", xy=(b.get_x() + b.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Coalesced Resident Hits")
    ax1.set_title("Panel A: Resident Hits Under CSTP Prefetching")
    ax1.set_ylim(0, 2000)
    ax1.grid(True, axis="y")
    
    ax1.text(0.5, 1200, "W3 Sign: Negative\n(-14 hits, -0.003%)", ha="center", fontsize=8.5, fontweight="bold",
             color="#922b21", bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdedec", edgecolor="#c0392b"))
    ax1.text(2.5, 1100, "W4 Sign: Positive\n(+15 hits, +0.004%)", ha="center", fontsize=8.5, fontweight="bold",
             color="#145a32", bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", edgecolor="#27ae60"))
             
    # Panel B: Incremental Physical Savings in Workload W4 (B=32)
    ax2 = axes[1]
    metrics = ["DDR5 Traffic\nSaved (GB)", "Exposed Stall\nSaved (ms)", "Execution Time\nSaved (ms)", "Protected Lines\nReused (Hits)", "Reloads\nAvoided"]
    values = [2.0258, 31.82, 31.82, 36.0, 15.0]
    bar_colors = ["#16a085", "#2980b9", "#8e44ad", "#27ae60", "#f39c12"]
    
    b2 = ax2.bar(metrics, values, width=0.45, color=bar_colors, edgecolor="black")
    for b in b2:
        h = b.get_height()
        ax2.annotate(f"+{h:.2f}", xy=(b.get_x() + b.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Physical Savings (Respective Units)")
    ax2.set_title("Panel B: FA-LRU Physical Savings in W4 (B=32, T=128)")
    ax2.set_ylim(0, 45)
    ax2.grid(True, axis="y")
    
    # Footnote distinguishing diagnostic counters from physical outcomes
    ax2.text(0.5, -0.20, "Diagnostic context: 5,999 speculative victims evicted; 36 protected lines reused yielding 15 avoided reloads.",
             transform=ax2.transAxes, ha="center", fontsize=8.0, color="#555555", style="italic")
    
    fig.suptitle("Figure 11: LRU versus FA-LRU (W=64) Contribution Across Serving Concurrency",
                 fontsize=12.5, fontweight="bold", y=1.02)
    save_fig(fig, "fig11_falru_incremental_contribution")

def plot_fig12_batch_scaling():
    """Figure 12: Batch-Size Scaling Comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Panel A: Point 3 Large Workload Campaign (T=16, 80 GB cache)
    ax1 = axes[0]
    batches_p3 = [14, 32, 48, 70]
    stall_base_p3 = [18117.03, 33639.29, 50768.71, 62304.49]
    stall_stmoe_p3 = [18067.89, 33556.78, 50668.16, 62043.92]
    stall_cstp_p3 = [18055.42, 33529.28, 50566.14, 61794.55]
    
    ax1.plot(batches_p3, stall_base_p3, marker="o", color="#7f8c8d", linewidth=2.0, label="Baseline (No Prefetch)")
    ax1.plot(batches_p3, stall_stmoe_p3, marker="s", color="#3498db", linewidth=2.0, label="ST-MoE Prefetch (h=2)")
    ax1.plot(batches_p3, stall_cstp_p3, marker="^", color="#e67e22", linewidth=2.0, label="CSTP Prefetch (h=2)")
    
    ax1.set_ylabel("Exposed GPU Memory Stall (ms)")
    ax1.set_xlabel("Batch Size (Point 3, T=16, 80 GB Cache)")
    ax1.set_title("Panel A: Point 3 Stall Scaling (T=16, 80 GB Cache)")
    ax1.set_xticks(batches_p3)
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(True)
    
    # Panel B: Big Experiment Concurrency Scaling (T=128, 35 GiB cache)
    ax2 = axes[1]
    batches_big = [14, 32]
    lat_base_big = [3586.18, 7876.59]
    lat_lru_big = [3565.37, 5950.06]
    lat_falru_big = [3565.49, 5949.81]
    
    ax2.plot(batches_big, lat_base_big, marker="o", color="#7f8c8d", linewidth=2.0, label="Config A: Baseline")
    ax2.plot(batches_big, lat_lru_big, marker="s", color="#e67e22", linewidth=2.0, label="Config C2: CSTP + LRU")
    ax2.plot(batches_big, lat_falru_big, marker="*", color="#27ae60", linewidth=2.2, markersize=10, label="Config D: Complete (FA-LRU)")
    
    ax2.set_ylabel("Mean Token Latency (ms/token)")
    ax2.set_xlabel("Batch Size (Big Experiment, T=128, 35 GiB Cache)")
    ax2.set_title("Panel B: Big Experiment Latency Scaling (T=128, 35 GiB Cache)")
    ax2.set_xticks(batches_big)
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(True)
    
    # Provenance annotations for both B=14 (W3) and B=32 (W4)
    ax2.annotate("B=14 (W3, T=128):\nBase: 459.0 s (3,586.2 ms/tok)\nCSTP: 456.4 s (3,565.4 ms/tok)\nFA-LRU: 456.4 s (3,565.5 ms/tok)",
                 xy=(14, 3586.18), xytext=(17.5, 4300),
                 ha="center", fontsize=8.0, fontweight="bold", color="#34495e",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9f9", edgecolor="#7f8c8d"),
                 arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.2))

    ax2.annotate("B=32 (W4, T=128):\nBase: 1,008.2 s (7,876.6 ms/tok)\nCSTP: 761.6 s (5,950.1 ms/tok)\nComplete: 761.6 s (5,949.8 ms/tok)",
                 xy=(32, 5949.81), xytext=(24, 7000),
                 ha="center", fontsize=8.0, fontweight="bold", color="#145a32",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", edgecolor="#27ae60"),
                 arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2))
                 
    fig.suptitle("Figure 12: Concurrency Scaling Comparison Across Experimental Campaigns",
                 fontsize=12.5, fontweight="bold", y=1.02)
    save_fig(fig, "fig12_batch_scaling_comparison")

def main():
    verify_zero_ternary(__file__)
    plot_fig5_execution_time()
    plot_fig6_exposed_stall()
    plot_fig7_ddr5_traffic()
    plot_fig8_hbm_hits_reloads()
    plot_fig9_predictor_quality()
    plot_fig10_prefetch_effectiveness()
    plot_fig11_falru_contribution()
    plot_fig12_batch_scaling()

if __name__ == "__main__":
    main()

