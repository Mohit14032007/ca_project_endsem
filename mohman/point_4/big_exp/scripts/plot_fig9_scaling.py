#!/usr/bin/env python3
"""
Plot Figure 9: Scaling Effect of FA-LRU (W=64) vs Standard LRU Under CSTP
Generates publication-quality PNG and PDF figures.
Enforces strictly zero ternary operators (ast.IfExp == 0).
"""

import os
import ast
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

WORKSPACE = "/home/mohman/ca_project"
BIG_EXP_ROOT = os.path.join(WORKSPACE, "mohman", "point_4", "big_exp")
FIGURES_DIR = os.path.join(BIG_EXP_ROOT, "figures")
PNG_DIR = os.path.join(FIGURES_DIR, "png")
PDF_DIR = os.path.join(FIGURES_DIR, "pdf")


def verify_zero_ternary(file_path):
    with open(file_path, "r") as f:
        tree = ast.parse(f.read())
    ternary_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.IfExp)]
    if len(ternary_nodes) > 0:
        raise ValueError(f"Found {len(ternary_nodes)} ternary operators in {file_path}")
    print(f"[AST_CHECK] Verified 0 ternary operators in {file_path}")


def generate_figure():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "grid.alpha": 0.4,
        "grid.linestyle": "--"
    })

    fig = plt.figure(figsize=(11, 9.2))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3.0, 1.4, 0.9], hspace=0.35)

    ax_plot = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])
    ax_text = fig.add_subplot(gs[2])

    # -------------------------------------------------------------
    # 1. Main Line Plot
    # -------------------------------------------------------------
    batch_sizes = [14, 32]
    improvements = [-0.00337, 0.00420]

    y_min = -0.0055
    y_max = 0.0060
    ax_plot.set_ylim(y_min, y_max)
    ax_plot.set_xlim(8, 38)

    # Shaded Positive and Negative Regions
    ax_plot.axhspan(0, y_max, color="#eafaf1", alpha=0.75, zorder=0)
    ax_plot.axhspan(y_min, 0, color="#fdedec", alpha=0.75, zorder=0)

    # Zero reference line
    ax_plot.axhline(0, color="#555555", linestyle="--", linewidth=1.3, zorder=1)
    ax_plot.text(37.5, 0.00015, "Zero Improvement (Parity with LRU)",
                 fontsize=8.5, color="#444444", ha="right", va="bottom", style="italic")

    # Region Labels
    ax_plot.text(9.5, 0.0050, "FA-LRU Advantage (Latency Improvement: +Δ%)",
                 fontsize=10, fontweight="bold", color="#1e8449", va="center")
    ax_plot.text(9.5, -0.0047, "LRU Advantage (Latency Degradation: -Δ%)",
                 fontsize=10, fontweight="bold", color="#922b21", va="center")

    # Plot connecting line
    ax_plot.plot(batch_sizes, improvements, color="#2c3e50", linewidth=2.2, linestyle="-", zorder=3)

    # Plot points with conditional coloring using if-else block
    for b_val, imp_val in zip(batch_sizes, improvements):
        if imp_val >= 0:
            pt_color = "#27ae60"
        else:
            pt_color = "#c0392b"
        ax_plot.scatter([b_val], [imp_val], s=110, color=pt_color, edgecolors="black", linewidth=1.2, zorder=4)

    # Annotate Point 1 (B=14)
    ax_plot.annotate(
        "-0.00337%\n(-14 resident hits)\nFA-LRU slower",
        xy=(14, -0.00337),
        xytext=(14, -0.0017),
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#78281f",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#c0392b", linewidth=1.0, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
        zorder=5
    )

    # Annotate Point 2 (B=32)
    ax_plot.annotate(
        "+0.00420%\n(+15 resident hits)\nFA-LRU faster",
        xy=(32, 0.00420),
        xytext=(32, 0.0022),
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color="#145a32",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#27ae60", linewidth=1.0, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2),
        zorder=5
    )

    # Labels and Titles
    ax_plot.set_title("Scaling Effect of FA-LRU (W=64) Relative to Standard LRU", fontsize=13, fontweight="bold", pad=8)
    ax_plot.text(0.5, 1.02, "CSTP enabled, HBM = 35 GiB, T = 128, FA-LRU W = 64",
                 transform=ax_plot.transAxes, ha="center", fontsize=10.5, color="#333333")

    ax_plot.set_xticks([14, 32])
    ax_plot.set_xticklabels(["B = 14\n(Workload W3)", "B = 32\n(Workload W4)"], fontsize=10.5, fontweight="bold")
    ax_plot.set_xlabel("Batch Size (Serving Concurrency)", fontsize=11, labelpad=6)

    ax_plot.set_ylabel("Relative Mean Token Latency Improvement (%)\n[(LRU Latency - FA-LRU Latency) / LRU Latency × 100]",
                       fontsize=10, labelpad=8)

    y_ticks = [-0.005, -0.004, -0.003, -0.002, -0.001, 0.000, 0.001, 0.002, 0.003, 0.004, 0.005]
    formatted_y_labels = []
    for val in y_ticks:
        if val == 0:
            formatted_y_labels.append(" 0.000%")
        else:
            formatted_y_labels.append(f"{val:+.3f}%")
    ax_plot.set_yticks(y_ticks)
    ax_plot.set_yticklabels(formatted_y_labels)
    ax_plot.grid(True, zorder=0)

    # Scale note
    ax_plot.text(0.015, 0.04, "Note: Magnified Y-axis (scale in 10⁻³ %). Sub-percent deltas reflect extreme cache pressure (426 slots vs 1,000+ experts).",
                 transform=ax_plot.transAxes, fontsize=8.5, color="#555555", style="italic")

    # -------------------------------------------------------------
    # 2. Compact Supporting Table
    # -------------------------------------------------------------
    ax_table.axis("off")
    table_data = [
        ["Coalesced Resident Hits", "1,657", "1,643", "-14 (-0.84%)", "1,417", "1,432", "+15 (+1.06%)"],
        ["Expert Reloads from DDR5", "316,227", "316,241", "+14 (+0.004%)", "536,034", "536,019", "-15 (-0.003%)"],
        ["Total DDR5 Traffic (GB)", "29,217.32", "29,218.20", "+0.88 (+0.003%)", "48,738.84", "48,736.81", "-2.03 (-0.004%)"],
        ["Exposed Memory Stall (ms)", "447,698.51", "447,713.80", "+15.30 (+0.003%)", "747,863.57", "747,831.75", "-31.82 (-0.004%)"],
        ["Mean Token Latency (ms)", "3,565.37", "3,565.49", "+0.12 (-0.00337%)", "5,950.06", "5,949.81", "-0.25 (+0.00420%)"]
    ]
    col_labels = [
        "Metric",
        "W3: LRU\n(B=14)",
        "W3: FA-LRU\n(B=14)",
        "W3 Delta\n(FA-LRU vs LRU)",
        "W4: LRU\n(B=32)",
        "W4: FA-LRU\n(B=32)",
        "W4 Delta\n(FA-LRU vs LRU)"
    ]

    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.45)

    # Style header and rows
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#eaeded")
            cell.set_text_props(weight="bold", color="#2c3e50")
        else:
            if col == 0:
                cell.set_text_props(weight="bold", ha="left")
                cell.set_facecolor("#f8f9f9")
            elif col == 3:
                cell.set_facecolor("#fdedec")
                cell.set_text_props(color="#922b21", weight="bold")
            elif col == 6:
                cell.set_facecolor("#eafaf1")
                cell.set_text_props(color="#196f3d", weight="bold")
            else:
                cell.set_facecolor("#ffffff")

    ax_table.set_title("Supporting Table: Absolute Metrics and Paired Deltas (CSTP Enabled, Repetition 1 & 2 Identical)",
                       fontsize=9.5, fontweight="bold", pad=2)

    # -------------------------------------------------------------
    # 3. Short Interpretation Text Box
    # -------------------------------------------------------------
    ax_text.axis("off")
    interp_text = (
        "Interpretation of Sign Inversion Across Concurrency Scaling:\n"
        "• At lower batch concurrency (B=14, Workload W3), cross-prompt expert sharing is sparse. The 40-layer sequential working set\n"
        "  displaces experts before repeated access occurs. Frequency protection in FA-LRU retains older lines with low re-use probability,\n"
        "  producing minor cache churn (-14 resident hits, +0.88 GB DDR5 traffic, -0.00337% relative latency degradation).\n"
        "• At higher batch concurrency (B=32, Workload W4), inter-sequence overlap expands substantially. FA-LRU leverages its bottom W=64\n"
        "  window to selectively evict unconsumed speculative prefetches (f=1) while protecting frequently requested demand lines (f=2).\n"
        "  This targeted retention avoids 15 DDR5 expert reloads, reduces DDR5 traffic by 2.03 GB, and yields a net latency gain (+0.00420%).\n"
        "• Determinism note: The two repetitions produce bit-exact identical values; differences represent exact architectural behavior."
    )
    ax_text.text(
        0.01, 0.90, interp_text,
        transform=ax_text.transAxes,
        fontsize=8.2,
        va="top",
        ha="left",
        color="#2c3e50",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f6f7", edgecolor="#bdc3c7", linewidth=1.0)
    )

    # Save to requested destination: mohman/point_4/big_exp/figures/
    target_png = os.path.join(FIGURES_DIR, "fig9_falru_scaling_effect.png")
    target_pdf = os.path.join(FIGURES_DIR, "fig9_falru_scaling_effect.pdf")
    fig.savefig(target_png, dpi=300, bbox_inches="tight")
    fig.savefig(target_pdf, format="pdf", bbox_inches="tight")
    print(f"[PLOT] Saved: {target_png}")
    print(f"[PLOT] Saved: {target_pdf}")

    # Also mirror into subfolders png/ and pdf/
    sub_png = os.path.join(PNG_DIR, "fig9_falru_scaling_effect.png")
    sub_pdf = os.path.join(PDF_DIR, "fig9_falru_scaling_effect.pdf")
    fig.savefig(sub_png, dpi=300, bbox_inches="tight")
    fig.savefig(sub_pdf, format="pdf", bbox_inches="tight")
    print(f"[PLOT] Mirrored: {sub_png}")
    print(f"[PLOT] Mirrored: {sub_pdf}")

    plt.close(fig)


if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    verify_zero_ternary(script_path)
    generate_figure()
