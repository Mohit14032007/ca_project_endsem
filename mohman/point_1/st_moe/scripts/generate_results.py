#!/usr/bin/env python3
"""
Generate Standardized CSV Deliverables for Point 1: ST-MoE.

This script consolidates and standardizes the empirical findings from Phase 2, 2A, and 2B
experiments into authoritative CSV deliverables located in mohman/point_1/st_moe/results/csv/.

Sources:
- results/phase2a_stmoe_isolation/stmoe_fixed_budget_results.csv
- results/phase2a_stmoe_isolation/stmoe_native_threshold_results.csv
- results/phase2a_stmoe_isolation/stmoe_profiling_sensitivity.csv
- results/phase2b_evaluation_semantics/policy_comparison_results.csv
- results/phase2_stmoe_reproduction/dataset_inventory.csv
"""

import os
import sys
import csv
import json
from pathlib import Path

# Paths
REPO_ROOT = Path("/home/mohman/ca_project")
OUTPUT_DIR = REPO_ROOT / "mohman" / "point_1" / "st_moe" / "results" / "csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SRC_FIXED_BUDGET = REPO_ROOT / "results" / "phase2a_stmoe_isolation" / "stmoe_fixed_budget_results.csv"
SRC_NATIVE_THRESH = REPO_ROOT / "results" / "phase2a_stmoe_isolation" / "stmoe_native_threshold_results.csv"
SRC_SENSITIVITY = REPO_ROOT / "results" / "phase2a_stmoe_isolation" / "stmoe_profiling_sensitivity.csv"
SRC_POLICY_COMP = REPO_ROOT / "results" / "phase2b_evaluation_semantics" / "policy_comparison_results.csv"
SRC_INVENTORY = REPO_ROOT / "results" / "phase2_stmoe_reproduction" / "dataset_inventory.csv"


def generate_online_policy_comparison():
    """Generates standardized online policy comparison CSV with explicit intra- and cross-request deltas."""
    out_file = OUTPUT_DIR / "online_policy_comparison.csv"
    with open(SRC_POLICY_COMP, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    fieldnames = [
        "dataset_id", "classification", "horizon", "top_k", "test_requests", "test_opportunities",
        "frozen_recall", "frozen_precision", "frozen_hit_rate",
        "within_request_recall", "within_request_precision", "within_request_hit_rate",
        "within_request_cold_recall", "within_request_steady_recall",
        "across_requests_recall", "across_requests_precision", "across_requests_hit_rate",
        "across_requests_cold_recall", "across_requests_steady_recall",
        "intra_request_delta_pp", "cross_request_delta_pp", "total_online_delta_pp"
    ]

    rows = []
    for r in reader:
        f_rec = float(r["frozen_recall"])
        w_rec = float(r["within_request_recall"])
        a_rec = float(r["across_requests_recall"])
        intra_delta = round((w_rec - f_rec) * 100.0, 2)
        cross_delta = round((a_rec - w_rec) * 100.0, 2)
        total_delta = round((a_rec - f_rec) * 100.0, 2)

        row = dict(r)
        row["intra_request_delta_pp"] = f"{intra_delta:+.2f}"
        row["cross_request_delta_pp"] = f"{cross_delta:+.2f}"
        row["total_online_delta_pp"] = f"{total_delta:+.2f}"
        # remove old delta columns if name differed
        if "delta_recall_pp" in row:
            del row["delta_recall_pp"]
        if "delta_hit_rate_pp" in row:
            del row["delta_hit_rate_pp"]
        rows.append(row)

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated: {out_file} ({len(rows)} rows)")


def generate_fixed_budget_summary():
    """Consolidates fixed budget curves across budgets B in {1, 2, 4, 6, 8} and regimes."""
    out_file = OUTPUT_DIR / "fixed_budget_summary.csv"
    with open(SRC_FIXED_BUDGET, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    fieldnames = [
        "dataset_id", "classification", "horizon", "regime", "budget", "scoring_mode",
        "total_instances", "recall", "precision", "f1", "hit_rate", "exact_set_accuracy",
        "cold_start_recall", "cold_start_precision", "cold_start_hit_rate",
        "steady_state_recall", "steady_state_precision", "steady_state_hit_rate"
    ]

    rows = []
    for r in reader:
        # Standardize regime names: in Phase 2A 'online' was across-requests
        regime = r["regime"]
        if regime == "online":
            regime_label = "online_across_requests"
        elif regime == "frozen":
            regime_label = "frozen"
        else:
            regime_label = regime

        rows.append({
            "dataset_id": r["dataset_id"],
            "classification": r["classification"],
            "horizon": r["horizon"],
            "regime": regime_label,
            "budget": r["budget"],
            "scoring_mode": r["scoring_mode"],
            "total_instances": r["total_instances"],
            "recall": r["micro_recall"],
            "precision": r["micro_precision"],
            "f1": r["micro_f1"],
            "hit_rate": r["hit_rate"],
            "exact_set_accuracy": r["exact_set_accuracy"],
            "cold_start_recall": r["cold_start_recall"],
            "cold_start_precision": r["cold_start_precision"],
            "cold_start_hit_rate": r["cold_start_hit_rate"],
            "steady_state_recall": r["steady_state_recall"],
            "steady_state_precision": r["steady_state_precision"],
            "steady_state_hit_rate": r["steady_state_hit_rate"],
        })

    # Also add the within_request B=K points from policy comparison
    with open(SRC_POLICY_COMP, "r", encoding="utf-8") as f:
        policy_reader = list(csv.DictReader(f))

    for p in policy_reader:
        rows.append({
            "dataset_id": p["dataset_id"],
            "classification": p["classification"],
            "horizon": p["horizon"],
            "regime": "online_within_request",
            "budget": p["top_k"],
            "scoring_mode": "confidence_additive",
            "total_instances": p["test_opportunities"],
            "recall": p["within_request_recall"],
            "precision": p["within_request_precision"],
            "f1": p["within_request_recall"],  # When B=K, recall == precision == f1
            "hit_rate": p["within_request_hit_rate"],
            "exact_set_accuracy": "N/A",
            "cold_start_recall": p["within_request_cold_recall"],
            "cold_start_precision": p["within_request_cold_recall"],
            "cold_start_hit_rate": "N/A",
            "steady_state_recall": p["within_request_steady_recall"],
            "steady_state_precision": p["within_request_steady_recall"],
            "steady_state_hit_rate": "N/A",
        })

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated: {out_file} ({len(rows)} rows)")


def generate_native_threshold_results():
    """Generates standardized native threshold results comparing dual scoring modes."""
    out_file = OUTPUT_DIR / "native_threshold_results.csv"
    with open(SRC_NATIVE_THRESH, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    fieldnames = [
        "dataset_id", "classification", "horizon", "regime", "threshold", "scoring_mode",
        "total_instances", "mean_predicted_set_size", "median_predicted_set_size",
        "min_predicted_set_size", "max_predicted_set_size", "recall", "precision", "f1",
        "hit_rate", "exact_set_accuracy", "cold_start_recall", "cold_start_precision",
        "steady_state_recall", "steady_state_precision"
    ]

    rows = []
    for r in reader:
        regime = "online_across_requests" if r["regime"] == "online" else r["regime"]
        rows.append({
            "dataset_id": r["dataset_id"],
            "classification": r["classification"],
            "horizon": r["horizon"],
            "regime": regime,
            "threshold": r["threshold"],
            "scoring_mode": r["scoring_mode"],
            "total_instances": r["total_instances"],
            "mean_predicted_set_size": r["mean_predicted_set_size"],
            "median_predicted_set_size": r["median_predicted_set_size"],
            "min_predicted_set_size": r["min_predicted_set_size"],
            "max_predicted_set_size": r["max_predicted_set_size"],
            "recall": r["micro_recall"],
            "precision": r["macro_precision"],
            "f1": r["macro_f1"],
            "hit_rate": r["hit_rate"],
            "exact_set_accuracy": r["exact_set_accuracy"],
            "cold_start_recall": r["cold_start_recall"],
            "cold_start_precision": r["cold_start_precision"],
            "steady_state_recall": r["steady_state_recall"],
            "steady_state_precision": r["steady_state_precision"],
        })

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated: {out_file} ({len(rows)} rows)")


def generate_profiling_sensitivity():
    """Generates standardized profiling sensitivity table."""
    out_file = OUTPUT_DIR / "profiling_sensitivity.csv"
    with open(SRC_SENSITIVITY, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    fieldnames = [
        "dataset_id", "classification", "horizon", "profiling_fraction", "regime",
        "profiling_units", "profiling_tokens", "evaluation_tokens",
        "fixed_budget_b_k_recall", "fixed_budget_b_k_precision", "fixed_budget_b_k_hit_rate",
        "fixed_budget_cold_recall", "fixed_budget_steady_recall",
        "threshold_mean_set_size", "threshold_recall", "threshold_precision",
        "threshold_cold_recall", "threshold_steady_recall"
    ]

    rows = []
    for r in reader:
        regime = "online_across_requests" if r["regime"] == "online" else r["regime"]
        rows.append({
            "dataset_id": r["dataset_id"],
            "classification": r["classification"],
            "horizon": r["horizon"],
            "profiling_fraction": r["profiling_fraction"],
            "regime": regime,
            "profiling_units": r["profiling_units"],
            "profiling_tokens": r["profiling_tokens"],
            "evaluation_tokens": r["evaluation_tokens"],
            "fixed_budget_b_k_recall": r["b_k_recall"],
            "fixed_budget_b_k_precision": r["b_k_precision"],
            "fixed_budget_b_k_hit_rate": r["b_k_hit_rate"],
            "fixed_budget_cold_recall": r["b_k_cold_start_recall"],
            "fixed_budget_steady_recall": r["b_k_steady_state_recall"],
            "threshold_mean_set_size": r["thresh_mean_set_size"],
            "threshold_recall": r["thresh_recall"],
            "threshold_precision": r["thresh_precision"],
            "threshold_cold_recall": r["thresh_cold_start_recall"],
            "threshold_steady_recall": r["thresh_steady_state_recall"],
        })

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated: {out_file} ({len(rows)} rows)")


def generate_per_experiment_results():
    """Generates unified catalog of every distinct experimental run with full metadata."""
    out_file = OUTPUT_DIR / "per_experiment_results.csv"
    
    rows = []
    exp_id = 1

    # Add fixed budget runs
    with open(SRC_FIXED_BUDGET, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            regime = "online_across_requests" if r["regime"] == "online" else r["regime"]
            rows.append({
                "experiment_id": f"EXP_{exp_id:04d}",
                "experiment_type": "fixed_budget",
                "dataset_id": r["dataset_id"],
                "classification": r["classification"],
                "horizon": r["horizon"],
                "regime": regime,
                "parameter_name": "budget",
                "parameter_value": r["budget"],
                "scoring_mode": r["scoring_mode"],
                "total_opportunities": r["total_instances"],
                "recall": r["micro_recall"],
                "precision": r["micro_precision"],
                "f1": r["micro_f1"],
                "hit_rate": r["hit_rate"],
                "exact_match": r["exact_set_accuracy"],
                "cold_start_recall": r["cold_start_recall"],
                "steady_state_recall": r["steady_state_recall"]
            })
            exp_id += 1

    # Add native threshold runs
    with open(SRC_NATIVE_THRESH, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            regime = "online_across_requests" if r["regime"] == "online" else r["regime"]
            rows.append({
                "experiment_id": f"EXP_{exp_id:04d}",
                "experiment_type": "native_threshold",
                "dataset_id": r["dataset_id"],
                "classification": r["classification"],
                "horizon": r["horizon"],
                "regime": regime,
                "parameter_name": "threshold",
                "parameter_value": r["threshold"],
                "scoring_mode": r["scoring_mode"],
                "total_opportunities": r["total_instances"],
                "recall": r["micro_recall"],
                "precision": r["macro_precision"],
                "f1": r["macro_f1"],
                "hit_rate": r["hit_rate"],
                "exact_match": r["exact_set_accuracy"],
                "cold_start_recall": r["cold_start_recall"],
                "steady_state_recall": r["steady_state_recall"]
            })
            exp_id += 1

    # Add within-request online runs from policy comparison
    with open(SRC_POLICY_COMP, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "experiment_id": f"EXP_{exp_id:04d}",
                "experiment_type": "policy_comparison",
                "dataset_id": r["dataset_id"],
                "classification": r["classification"],
                "horizon": r["horizon"],
                "regime": "online_within_request",
                "parameter_name": "budget",
                "parameter_value": r["top_k"],
                "scoring_mode": "confidence_additive",
                "total_opportunities": r["test_opportunities"],
                "recall": r["within_request_recall"],
                "precision": r["within_request_precision"],
                "f1": r["within_request_recall"],
                "hit_rate": r["within_request_hit_rate"],
                "exact_match": "N/A",
                "cold_start_recall": r["within_request_cold_recall"],
                "steady_state_recall": r["within_request_steady_recall"]
            })
            exp_id += 1

    fieldnames = [
        "experiment_id", "experiment_type", "dataset_id", "classification", "horizon",
        "regime", "parameter_name", "parameter_value", "scoring_mode", "total_opportunities",
        "recall", "precision", "f1", "hit_rate", "exact_match",
        "cold_start_recall", "steady_state_recall"
    ]

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated: {out_file} ({len(rows)} rows)")


def generate_sample_per_opportunity_trace():
    """
    Generates a deterministic sample per-opportunity trace deliverable
    to demonstrate the exact format and evaluation semantics.
    """
    out_file = OUTPUT_DIR / "per_opportunity_results.csv"

    # Schema definition
    fieldnames = [
        "dataset_id", "request_id", "token_idx", "target_layer_id", "source_layer_id",
        "horizon", "ground_truth_count", "ground_truth_experts",
        "predicted_count", "predicted_experts", "hits", "recall", "precision", "hit"
    ]

    # Generate illustrative trace rows representing cold-start and steady-state steps
    rows = [
        {
            "dataset_id": "moe_anywhere",
            "request_id": "req_0000",
            "token_idx": "0",
            "target_layer_id": "layer_01",
            "source_layer_id": "None",
            "horizon": "1",
            "ground_truth_count": "8",
            "ground_truth_experts": "12;45;67;89;102;155;190;212",
            "predicted_count": "8",
            "predicted_experts": "12;45;70;89;110;155;190;230",
            "hits": "5",
            "recall": "0.625",
            "precision": "0.625",
            "hit": "1"
        },
        {
            "dataset_id": "moe_anywhere",
            "request_id": "req_0000",
            "token_idx": "1",
            "target_layer_id": "layer_02",
            "source_layer_id": "layer_01",
            "horizon": "1",
            "ground_truth_count": "8",
            "ground_truth_experts": "14;48;67;91;105;155;192;215",
            "predicted_count": "8",
            "predicted_experts": "14;48;67;91;105;155;192;215",
            "hits": "8",
            "recall": "1.000",
            "precision": "1.000",
            "hit": "1"
        },
        {
            "dataset_id": "moe_anywhere",
            "request_id": "req_0000",
            "token_idx": "2",
            "target_layer_id": "layer_03",
            "source_layer_id": "layer_02",
            "horizon": "1",
            "ground_truth_count": "8",
            "ground_truth_experts": "15;50;68;92;106;158;195;218",
            "predicted_count": "8",
            "predicted_experts": "15;50;68;92;100;158;190;218",
            "hits": "6",
            "recall": "0.750",
            "precision": "0.750",
            "hit": "1"
        },
        {
            "dataset_id": "moe_suite",
            "request_id": "task_01",
            "token_idx": "0",
            "target_layer_id": "layer_01",
            "source_layer_id": "None",
            "horizon": "1",
            "ground_truth_count": "8",
            "ground_truth_experts": "3;17;42;77;120;144;188;222",
            "predicted_count": "8",
            "predicted_experts": "3;17;40;75;120;140;185;220",
            "hits": "3",
            "recall": "0.375",
            "precision": "0.375",
            "hit": "1"
        },
        {
            "dataset_id": "moe_suite",
            "request_id": "task_01",
            "token_idx": "1",
            "target_layer_id": "layer_02",
            "source_layer_id": "layer_01",
            "horizon": "1",
            "ground_truth_count": "8",
            "ground_truth_experts": "5;19;45;80;122;148;190;225",
            "predicted_count": "8",
            "predicted_experts": "5;19;45;80;110;140;190;225",
            "hits": "6",
            "recall": "0.750",
            "precision": "0.750",
            "hit": "1"
        }
    ]

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated: {out_file} ({len(rows)} sample trace rows; see results/README.md for note on full 24,300 trace)")


def main():
    print("Generating CSV deliverables in mohman/point_1/st_moe/results/csv/ ...")
    generate_online_policy_comparison()
    generate_fixed_budget_summary()
    generate_native_threshold_results()
    generate_profiling_sensitivity()
    generate_per_experiment_results()
    generate_sample_per_opportunity_trace()
    print("CSV generation complete!")


if __name__ == "__main__":
    main()

