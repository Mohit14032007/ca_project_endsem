"""
Publication-Quality Figures for MoE Suite Horizon Sweep (H = 1, 2, 3, 4).
Generates Recall@8 vs H, F1@8 vs H, Latency vs H, and Pareto Frontiers.
Strictly avoids ternary operators (ast.IfExp == 0).
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_dir = os.path.join(project_root, "results", "endsem_moe_suite_horizon")
figures_dir = os.path.join(results_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# Color and style definitions
STYLE_CONFIG = {
    "acsh_stage_aware": {"color": "#1f77b4", "marker": "o", "label": "ACSH Stage-Aware (Ours)", "lw": 2.8, "ls": "-"},
    "acsh_global": {"color": "#4292c6", "marker": "s", "label": "ACSH Global", "lw": 2.0, "ls": "--"},
    "acsh_fast_recovery": {"color": "#08519c", "marker": "^", "label": "ACSH Fast-Recovery", "lw": 2.2, "ls": "-."},
    "cstp_online": {"color": "#2ca02c", "marker": "D", "label": "CSTP Online", "lw": 2.4, "ls": "-"},
    "cstp_frozen": {"color": "#74c476", "marker": "d", "label": "CSTP Frozen", "lw": 1.8, "ls": "--"},
    "st_moe_paper_online": {"color": "#d62728", "marker": "v", "label": "ST-MoE Paper Online", "lw": 2.4, "ls": "-"},
    "st_moe_paper_frozen": {"color": "#fb6a4a", "marker": "x", "label": "ST-MoE Paper Frozen", "lw": 1.8, "ls": "--"},
    "cstp_st_moe_hybrid": {"color": "#9467bd", "marker": "p", "label": "CSTP+ST-MoE Hybrid", "lw": 2.0, "ls": "-"},
    "frequency_only": {"color": "#ff7f0e", "marker": "<", "label": "Frequency Baseline", "lw": 1.6, "ls": ":"},
    "temporal_only": {"color": "#17becf", "marker": ">", "label": "Temporal-Only (α=0.0)", "lw": 1.6, "ls": ":"},
    "spatial_only": {"color": "#bcbd22", "marker": "h", "label": "Spatial-Only (α=1.0)", "lw": 1.6, "ls": ":"},
    "random": {"color": "#7f7f7f", "marker": "*", "label": "Random Baseline", "lw": 1.5, "ls": ":"},
    "acsh_cold_start": {"color": "#6baed6", "marker": "P", "label": "ACSH Cold-Start", "lw": 1.8, "ls": "-."},
    "cstp_cold_start": {"color": "#a1d99b", "marker": "X", "label": "CSTP Cold-Start", "lw": 1.6, "ls": ":"},
    "st_moe_paper_cold_start": {"color": "#fcae91", "marker": "+", "label": "ST-MoE Cold-Start", "lw": 1.6, "ls": ":"},
    "st_moe_legacy": {"color": "#8c564b", "marker": "*", "label": "ST-MoE Legacy (H=1 only)", "lw": 2.0, "ls": "None"}
}


def plot_recall_vs_horizon(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6), dpi=300)
    plt.grid(True, linestyle="--", alpha=0.6)

    horizons = [1, 2, 3, 4]
    plt.xticks(horizons, [f"H={h}\n(l \u2192 l+{h})" for h in horizons], fontsize=11)

    for pred, cfg in STYLE_CONFIG.items():
        sub = df[(df["predictor"] == pred) & (df["status"] == "completed")]
        if len(sub) == 0:
            continue

        sub = sub.sort_values("prediction_horizon")
        x_vals = sub["prediction_horizon"].values
        y_vals = sub["macro_recall"].values

        if pred == "st_moe_legacy":
            plt.scatter(
                x_vals,
                y_vals,
                color=cfg["color"],
                marker=cfg["marker"],
                s=120,
                label=cfg["label"],
                zorder=5
            )
        else:
            plt.plot(
                x_vals,
                y_vals,
                color=cfg["color"],
                marker=cfg["marker"],
                linewidth=cfg["lw"],
                linestyle=cfg["ls"],
                label=cfg["label"]
            )

    plt.xlabel("Prediction Horizon H (MoE Stages Ahead)", fontsize=12, fontweight="bold")
    plt.ylabel("Macro Recall @ Budget=8", fontsize=12, fontweight="bold")
    plt.title("Prediction Accuracy Scaling with Inter-Layer Horizon H (MoE Suite, 75 Layers, E=256)", fontsize=13, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9, frameon=True)
    plt.tight_layout()

    out_path = os.path.join(figures_dir, "recall_vs_horizon.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_f1_vs_horizon(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6), dpi=300)
    plt.grid(True, linestyle="--", alpha=0.6)

    horizons = [1, 2, 3, 4]
    plt.xticks(horizons, [f"H={h}\n(l \u2192 l+{h})" for h in horizons], fontsize=11)

    for pred, cfg in STYLE_CONFIG.items():
        sub = df[(df["predictor"] == pred) & (df["status"] == "completed")]
        if len(sub) == 0:
            continue

        sub = sub.sort_values("prediction_horizon")
        x_vals = sub["prediction_horizon"].values
        y_vals = sub["macro_f1"].values

        if pred == "st_moe_legacy":
            plt.scatter(
                x_vals,
                y_vals,
                color=cfg["color"],
                marker=cfg["marker"],
                s=120,
                label=cfg["label"],
                zorder=5
            )
        else:
            plt.plot(
                x_vals,
                y_vals,
                color=cfg["color"],
                marker=cfg["marker"],
                linewidth=cfg["lw"],
                linestyle=cfg["ls"],
                label=cfg["label"]
            )

    plt.xlabel("Prediction Horizon H (MoE Stages Ahead)", fontsize=12, fontweight="bold")
    plt.ylabel("Macro F1-Score @ Budget=8", fontsize=12, fontweight="bold")
    plt.title("Prediction F1-Score vs Inter-Layer Horizon H (MoE Suite)", fontsize=13, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9, frameon=True)
    plt.tight_layout()

    out_path = os.path.join(figures_dir, "f1_vs_horizon.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_latency_vs_horizon(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6), dpi=300)
    plt.grid(True, linestyle="--", alpha=0.6)

    horizons = [1, 2, 3, 4]
    plt.xticks(horizons, [f"H={h}" for h in horizons], fontsize=11)

    key_predictors = [
        "acsh_stage_aware",
        "acsh_global",
        "cstp_online",
        "st_moe_paper_online",
        "cstp_st_moe_hybrid",
        "frequency_only"
    ]

    for pred in key_predictors:
        cfg = STYLE_CONFIG.get(pred, {"color": "black", "marker": "o", "label": pred, "lw": 1.5, "ls": "-"})
        sub = df[(df["predictor"] == pred) & (df["status"] == "completed")]
        if len(sub) == 0:
            continue

        sub = sub.sort_values("prediction_horizon")
        x_vals = sub["prediction_horizon"].values
        y_vals = sub["query_latency_mean_us"].values

        plt.plot(
            x_vals,
            y_vals,
            color=cfg["color"],
            marker=cfg["marker"],
            linewidth=cfg["lw"],
            linestyle=cfg["ls"],
            label=cfg["label"]
        )

    plt.xlabel("Prediction Horizon H", fontsize=12, fontweight="bold")
    plt.ylabel("Mean Query Latency (\u03bcs)", fontsize=12, fontweight="bold")
    plt.title("Predictor Query Latency vs Prediction Horizon H (MoE Suite)", fontsize=13, fontweight="bold")
    plt.legend(loc="upper left", fontsize=10, frameon=True)
    plt.tight_layout()

    out_path = os.path.join(figures_dir, "latency_vs_horizon.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_pareto_frontier(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True, dpi=300)
    horizons = [1, 2, 3, 4]

    key_predictors = [
        "acsh_stage_aware",
        "acsh_global",
        "acsh_fast_recovery",
        "cstp_online",
        "cstp_frozen",
        "st_moe_paper_online",
        "st_moe_paper_frozen",
        "cstp_st_moe_hybrid",
        "frequency_only",
        "temporal_only",
        "spatial_only",
        "random"
    ]

    for idx, h in enumerate(horizons):
        ax = axes[idx]
        ax.grid(True, linestyle="--", alpha=0.6)
        sub_h = df[(df["prediction_horizon"] == h) & (df["status"] == "completed")]

        for pred in key_predictors:
            p_row = sub_h[sub_h["predictor"] == pred]
            if len(p_row) == 0:
                continue

            cfg = STYLE_CONFIG.get(pred, {"color": "black", "marker": "o", "label": pred})
            lat = p_row["query_latency_mean_us"].values[0]
            rec = p_row["macro_recall"].values[0]

            ax.scatter(
                lat,
                rec,
                color=cfg["color"],
                marker=cfg["marker"],
                s=100,
                label=cfg["label"]
            )
            ax.annotate(
                pred.replace("_", " "),
                (lat, rec),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7
            )

        ax.set_title(f"Horizon H={h} (l \u2192 l+{h})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Query Latency (\u03bcs)", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Macro Recall @ Budget=8", fontsize=11, fontweight="bold")

    plt.suptitle("Recall vs Latency Pareto Trade-off Across Prediction Horizons (MoE Suite)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(figures_dir, "pareto_frontier_horizons.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    csv_path = os.path.join(results_dir, "horizon_comparison.csv")
    if not os.path.exists(csv_path):
        print(f"Comparison CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    plot_recall_vs_horizon(df)
    plot_f1_vs_horizon(df)
    plot_latency_vs_horizon(df)
    plot_pareto_frontier(df)
    print("All horizon sweep plots successfully generated!")


if __name__ == "__main__":
    main()

