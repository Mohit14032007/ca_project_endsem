#!/usr/bin/env python3
"""
Generate Comparative CSV Data Tables for Point 5 Figures 5 through 12.
Schema per row:
metric_name, metric_value, unit, configuration, workload, source_campaign, source_run_id, exact_source_file, measurement_type
Enforces strictly zero ternary operators (ast.IfExp == 0).
"""

import os
import ast
import pandas as pd

WORKSPACE = "/home/mohman/ca_project"
POINT_5_ROOT = os.path.join(WORKSPACE, "mohman", "point_5")
EXTRACTED_DIR = os.path.join(POINT_5_ROOT, "extracted_data")

def verify_zero_ternary(file_path):
    with open(file_path, "r") as f:
        tree = ast.parse(f.read())
    ternary_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.IfExp)]
    if len(ternary_nodes) > 0:
        raise ValueError(f"Found {len(ternary_nodes)} ternary operators in {file_path}")
    print(f"[AST_CHECK] Verified 0 ternary operators in {file_path}")

def generate_table1_execution_time():
    """Table 1: End-to-End Execution Time Comparison (Figure 5)."""
    rows = [
        # Workload W4 (B=32, T=128)
        {
            "metric_name": "simulated_execution_time",
            "metric_value": 1008.204,
            "unit": "seconds",
            "configuration": "Config A: Baseline (No Prefetch, LRU)",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c1_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "simulated_execution_time",
            "metric_value": 1008.204,
            "unit": "seconds",
            "configuration": "Config B: Prediction Only (CSTP Shadow Mode)",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment / phase_4d5c",
            "source_run_id": "inferred_w4_cstp_shadow",
            "exact_source_file": "LLMSimulator/results/phase_4d5c/production_matrix_summary.csv",
            "measurement_type": "inferred_shadow_mode_equivalence"
        },
        {
            "metric_name": "simulated_execution_time",
            "metric_value": 761.608,
            "unit": "seconds",
            "configuration": "Config C2: CSTP Prefetch + Standard LRU",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c3_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "simulated_execution_time",
            "metric_value": 761.576,
            "unit": "seconds",
            "configuration": "Config D: Complete System (CSTP + FA-LRU W=64)",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c4_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        # Workload W3 (B=14, T=128) - Authenticated Simulation Times
        {
            "metric_name": "simulated_execution_time",
            "metric_value": 459.031,
            "unit": "seconds",
            "configuration": "Config A: Baseline (No Prefetch, LRU)",
            "workload": "W3 (B=14, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c1_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "simulated_execution_time",
            "metric_value": 459.031,
            "unit": "seconds",
            "configuration": "Config B: Prediction Only (CSTP Shadow Mode)",
            "workload": "W3 (B=14, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment / phase_4d5c",
            "source_run_id": "inferred_w3_cstp_shadow",
            "exact_source_file": "LLMSimulator/results/phase_4d5c/production_matrix_summary.csv",
            "measurement_type": "inferred_shadow_mode_equivalence"
        },
        {
            "metric_name": "simulated_execution_time",
            "metric_value": 456.368,
            "unit": "seconds",
            "configuration": "Config C2: CSTP Prefetch + Standard LRU",
            "workload": "W3 (B=14, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c3_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "simulated_execution_time",
            "metric_value": 456.383,
            "unit": "seconds",
            "configuration": "Config D: Complete System (CSTP + FA-LRU W=64)",
            "workload": "W3 (B=14, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c4_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        # Normalized execution time (% of baseline)
        {
            "metric_name": "normalized_execution_time",
            "metric_value": 100.000,
            "unit": "percent",
            "configuration": "Config A: Baseline (No Prefetch, LRU)",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c1_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "calculated"
        },
        {
            "metric_name": "normalized_execution_time",
            "metric_value": 100.000,
            "unit": "percent",
            "configuration": "Config B: Prediction Only (CSTP Shadow Mode)",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment / phase_4d5c",
            "source_run_id": "inferred_w4_cstp_shadow",
            "exact_source_file": "LLMSimulator/results/phase_4d5c/production_matrix_summary.csv",
            "measurement_type": "inferred_shadow_mode_equivalence"
        },
        {
            "metric_name": "normalized_execution_time",
            "metric_value": 75.541,
            "unit": "percent",
            "configuration": "Config C2: CSTP Prefetch + Standard LRU",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c3_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "calculated"
        },
        {
            "metric_name": "normalized_execution_time",
            "metric_value": 75.538,
            "unit": "percent",
            "configuration": "Config D: Complete System (CSTP + FA-LRU W=64)",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c4_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "calculated"
        },
        # Latency Improvement relative to Baseline (%)
        {
            "metric_name": "latency_improvement_vs_baseline",
            "metric_value": 0.000,
            "unit": "percent",
            "configuration": "Config A: Baseline (No Prefetch, LRU)",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c1_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "calculated"
        },
        {
            "metric_name": "latency_improvement_vs_baseline",
            "metric_value": 0.000,
            "unit": "percent",
            "configuration": "Config B: Prediction Only (CSTP Shadow Mode)",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment / phase_4d5c",
            "source_run_id": "inferred_w4_cstp_shadow",
            "exact_source_file": "LLMSimulator/results/phase_4d5c/production_matrix_summary.csv",
            "measurement_type": "inferred_shadow_mode_equivalence"
        },
        {
            "metric_name": "latency_improvement_vs_baseline",
            "metric_value": 24.459,
            "unit": "percent",
            "configuration": "Config C2: CSTP Prefetch + Standard LRU",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c3_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "calculated"
        },
        {
            "metric_name": "latency_improvement_vs_baseline",
            "metric_value": 24.462,
            "unit": "percent",
            "configuration": "Config D: Complete System (CSTP + FA-LRU W=64)",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c4_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "calculated"
        }
    ]
    df = pd.DataFrame(rows)
    out_path = os.path.join(EXTRACTED_DIR, "execution_time_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"[SAVE] Table 1: {out_path} ({len(df)} rows)")

def generate_table2_exposed_stall():
    """Table 2: Exposed Memory Stall Comparison (Figure 6)."""
    rows = [
        # Absolute stall (ms)
        {
            "metric_name": "exposed_gpu_stall_time",
            "metric_value": 994459.73,
            "unit": "milliseconds",
            "configuration": "Config A: Baseline (No Prefetch, LRU)",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c1_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "exposed_gpu_stall_time",
            "metric_value": 994459.73,
            "unit": "milliseconds",
            "configuration": "Config B: Prediction Only (CSTP Shadow Mode)",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment / phase_4d5c",
            "source_run_id": "inferred_w4_cstp_shadow",
            "exact_source_file": "LLMSimulator/results/phase_4d5c/production_matrix_summary.csv",
            "measurement_type": "inferred_shadow_mode_equivalence"
        },
        {
            "metric_name": "exposed_gpu_stall_time",
            "metric_value": 747863.57,
            "unit": "milliseconds",
            "configuration": "Config C2: CSTP Prefetch + Standard LRU",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c3_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "exposed_gpu_stall_time",
            "metric_value": 747831.75,
            "unit": "milliseconds",
            "configuration": "Config D: Complete System (CSTP + FA-LRU W=64)",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c4_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        # Workload W3 (B=14, T=128) - Authenticated Simulation Stall Times
        {
            "metric_name": "exposed_gpu_stall_time",
            "metric_value": 450381.15,
            "unit": "milliseconds",
            "configuration": "Config A: Baseline (No Prefetch, LRU)",
            "workload": "W3 (B=14, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c1_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "exposed_gpu_stall_time",
            "metric_value": 450381.15,
            "unit": "milliseconds",
            "configuration": "Config B: Prediction Only (CSTP Shadow Mode)",
            "workload": "W3 (B=14, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment / phase_4d5c",
            "source_run_id": "inferred_w3_cstp_shadow",
            "exact_source_file": "LLMSimulator/results/phase_4d5c/production_matrix_summary.csv",
            "measurement_type": "inferred_shadow_mode_equivalence"
        },
        {
            "metric_name": "exposed_gpu_stall_time",
            "metric_value": 447698.51,
            "unit": "milliseconds",
            "configuration": "Config C2: CSTP Prefetch + Standard LRU",
            "workload": "W3 (B=14, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c3_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "exposed_gpu_stall_time",
            "metric_value": 447713.80,
            "unit": "milliseconds",
            "configuration": "Config D: Complete System (CSTP + FA-LRU W=64)",
            "workload": "W3 (B=14, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c4_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        # Percentage reduction in stall relative to baseline (%)
        {
            "metric_name": "stall_reduction_vs_baseline",
            "metric_value": 0.000,
            "unit": "percent",
            "configuration": "Config A: Baseline (No Prefetch, LRU)",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c1_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "calculated"
        },
        {
            "metric_name": "stall_reduction_vs_baseline",
            "metric_value": 0.000,
            "unit": "percent",
            "configuration": "Config B: Prediction Only (CSTP Shadow Mode)",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment / phase_4d5c",
            "source_run_id": "inferred_w4_cstp_shadow",
            "exact_source_file": "LLMSimulator/results/phase_4d5c/production_matrix_summary.csv",
            "measurement_type": "inferred_shadow_mode_equivalence"
        },
        {
            "metric_name": "stall_reduction_vs_baseline",
            "metric_value": 24.797,
            "unit": "percent",
            "configuration": "Config C2: CSTP Prefetch + Standard LRU",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c3_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "calculated"
        },
        {
            "metric_name": "stall_reduction_vs_baseline",
            "metric_value": 24.800,
            "unit": "percent",
            "configuration": "Config D: Complete System (CSTP + FA-LRU W=64)",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c4_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "calculated"
        },
        # Incremental stall reduction (C2 -> D)
        {
            "metric_name": "incremental_falru_stall_saved",
            "metric_value": 31.82,
            "unit": "milliseconds",
            "configuration": "Config D vs Config C2",
            "workload": "W4 (B=32, T=128, M=35 GiB)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c4_r1_vs_c3_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/paired_comparisons.csv",
            "measurement_type": "measured_delta"
        }
    ]
    df = pd.DataFrame(rows)
    out_path = os.path.join(EXTRACTED_DIR, "exposed_stall_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"[SAVE] Table 2: {out_path} ({len(df)} rows)")

def generate_table3_ddr5_traffic():
    """Table 3: DDR5 Traffic Composition (Figure 7)."""
    rows = [
        # Workload W4 (B=32, T=128)
        {
            "metric_name": "demand_bytes",
            "metric_value": 48740157001664,
            "unit": "bytes",
            "configuration": "Config A: Baseline",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c1_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "speculative_bytes",
            "metric_value": 0,
            "unit": "bytes",
            "configuration": "Config A: Baseline",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c1_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "total_ddr5_gb",
            "metric_value": 48740.16,
            "unit": "decimal_GB",
            "configuration": "Config A: Baseline",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c1_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "extracted"
        },
        {
            "metric_name": "demand_bytes",
            "metric_value": 48740157001664,
            "unit": "bytes",
            "configuration": "Config B: Prediction Only",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment / phase_4d5c",
            "source_run_id": "inferred_w4_cstp_shadow",
            "exact_source_file": "LLMSimulator/results/phase_4d5c/production_matrix_summary.csv",
            "measurement_type": "inferred_shadow_mode_equivalence"
        },
        {
            "metric_name": "speculative_bytes",
            "metric_value": 0,
            "unit": "bytes",
            "configuration": "Config B: Prediction Only",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment / phase_4d5c",
            "source_run_id": "inferred_w4_cstp_shadow",
            "exact_source_file": "LLMSimulator/results/phase_4d5c/production_matrix_summary.csv",
            "measurement_type": "inferred_shadow_mode_equivalence"
        },
        {
            "metric_name": "total_ddr5_gb",
            "metric_value": 48740.16,
            "unit": "decimal_GB",
            "configuration": "Config B: Prediction Only",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment / phase_4d5c",
            "source_run_id": "inferred_w4_cstp_shadow",
            "exact_source_file": "LLMSimulator/results/phase_4d5c/production_matrix_summary.csv",
            "measurement_type": "inferred_shadow_mode_equivalence"
        },
        {
            "metric_name": "demand_bytes",
            "metric_value": 37790536433664,
            "unit": "bytes",
            "configuration": "Config C2: CSTP + LRU",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c3_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "speculative_bytes",
            "metric_value": 10948303650816,
            "unit": "bytes",
            "configuration": "Config C2: CSTP + LRU",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c3_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "total_ddr5_gb",
            "metric_value": 48738.84,
            "unit": "decimal_GB",
            "configuration": "Config C2: CSTP + LRU",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c3_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "extracted"
        },
        {
            "metric_name": "demand_bytes",
            "metric_value": 37808769073152,
            "unit": "bytes",
            "configuration": "Config D: Complete System",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c4_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "speculative_bytes",
            "metric_value": 10928045162496,
            "unit": "bytes",
            "configuration": "Config D: Complete System",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c4_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "total_ddr5_gb",
            "metric_value": 48736.81,
            "unit": "decimal_GB",
            "configuration": "Config D: Complete System",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c4_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "extracted"
        },
        # Workload W3 (B=14, T=128)
        {
            "metric_name": "demand_bytes",
            "metric_value": 28824393744384,
            "unit": "bytes",
            "configuration": "Config A: Baseline",
            "workload": "W3 (B=14)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c1_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "speculative_bytes",
            "metric_value": 0,
            "unit": "bytes",
            "configuration": "Config A: Baseline",
            "workload": "W3 (B=14)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c1_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "total_ddr5_gb",
            "metric_value": 28824.39,
            "unit": "decimal_GB",
            "configuration": "Config A: Baseline",
            "workload": "W3 (B=14)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c1_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "extracted"
        },
        {
            "metric_name": "demand_bytes",
            "metric_value": 21972268351488,
            "unit": "bytes",
            "configuration": "Config C2: CSTP + LRU",
            "workload": "W3 (B=14)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c3_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "speculative_bytes",
            "metric_value": 7245051985920,
            "unit": "bytes",
            "configuration": "Config C2: CSTP + LRU",
            "workload": "W3 (B=14)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c3_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "total_ddr5_gb",
            "metric_value": 29217.32,
            "unit": "decimal_GB",
            "configuration": "Config C2: CSTP + LRU",
            "workload": "W3 (B=14)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c3_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "extracted"
        },
        {
            "metric_name": "demand_bytes",
            "metric_value": 21978169737216,
            "unit": "bytes",
            "configuration": "Config D: Complete System",
            "workload": "W3 (B=14)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c4_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "speculative_bytes",
            "metric_value": 7240031404032,
            "unit": "bytes",
            "configuration": "Config D: Complete System",
            "workload": "W3 (B=14)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c4_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "total_ddr5_gb",
            "metric_value": 29218.20,
            "unit": "decimal_GB",
            "configuration": "Config D: Complete System",
            "workload": "W3 (B=14)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c4_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "extracted"
        }
    ]
    df = pd.DataFrame(rows)
    out_path = os.path.join(EXTRACTED_DIR, "ddr5_traffic_composition.csv")
    df.to_csv(out_path, index=False)
    print(f"[SAVE] Table 3: {out_path} ({len(df)} rows)")

def generate_table4_hbm_hits_reloads():
    """Table 4: HBM Hit Rate and Expert Reload Comparison (Figure 8)."""
    rows = [
        # Workload W3 (B=14)
        {
            "metric_name": "coalesced_resident_hits",
            "metric_value": 0,
            "unit": "count",
            "configuration": "Config A: Baseline",
            "workload": "W3 (B=14)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c1_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "coalesced_hbm_hit_rate_pct",
            "metric_value": 0.000,
            "unit": "percent",
            "configuration": "Config A: Baseline",
            "workload": "W3 (B=14)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c1_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "calculated"
        },
        {
            "metric_name": "expert_reloads_count",
            "metric_value": 317775,
            "unit": "count",
            "configuration": "Config A: Baseline",
            "workload": "W3 (B=14)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c1_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "coalesced_resident_hits",
            "metric_value": 1657,
            "unit": "count",
            "configuration": "Config C2: CSTP + LRU",
            "workload": "W3 (B=14)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c3_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "coalesced_hbm_hit_rate_pct",
            "metric_value": 0.5063,
            "unit": "percent",
            "configuration": "Config C2: CSTP + LRU",
            "workload": "W3 (B=14)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c3_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "calculated"
        },
        {
            "metric_name": "expert_reloads_count",
            "metric_value": 316227,
            "unit": "count",
            "configuration": "Config C2: CSTP + LRU",
            "workload": "W3 (B=14)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c3_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "coalesced_resident_hits",
            "metric_value": 1643,
            "unit": "count",
            "configuration": "Config D: Complete System",
            "workload": "W3 (B=14)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c4_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "coalesced_hbm_hit_rate_pct",
            "metric_value": 0.5021,
            "unit": "percent",
            "configuration": "Config D: Complete System",
            "workload": "W3 (B=14)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c4_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "calculated"
        },
        {
            "metric_name": "expert_reloads_count",
            "metric_value": 316241,
            "unit": "count",
            "configuration": "Config D: Complete System",
            "workload": "W3 (B=14)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w3_c4_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        # Workload W4 (B=32)
        {
            "metric_name": "coalesced_resident_hits",
            "metric_value": 0,
            "unit": "count",
            "configuration": "Config A: Baseline",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c1_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "coalesced_hbm_hit_rate_pct",
            "metric_value": 0.000,
            "unit": "percent",
            "configuration": "Config A: Baseline",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c1_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "calculated"
        },
        {
            "metric_name": "expert_reloads_count",
            "metric_value": 537137,
            "unit": "count",
            "configuration": "Config A: Baseline",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c1_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "coalesced_resident_hits",
            "metric_value": 1417,
            "unit": "count",
            "configuration": "Config C2: CSTP + LRU",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c3_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "coalesced_hbm_hit_rate_pct",
            "metric_value": 0.2642,
            "unit": "percent",
            "configuration": "Config C2: CSTP + LRU",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c3_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "calculated"
        },
        {
            "metric_name": "expert_reloads_count",
            "metric_value": 536034,
            "unit": "count",
            "configuration": "Config C2: CSTP + LRU",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c3_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "coalesced_resident_hits",
            "metric_value": 1432,
            "unit": "count",
            "configuration": "Config D: Complete System",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c4_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        },
        {
            "metric_name": "coalesced_hbm_hit_rate_pct",
            "metric_value": 0.2670,
            "unit": "percent",
            "configuration": "Config D: Complete System",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c4_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "calculated"
        },
        {
            "metric_name": "expert_reloads_count",
            "metric_value": 536019,
            "unit": "count",
            "configuration": "Config D: Complete System",
            "workload": "W4 (B=32)",
            "source_campaign": "Point 4 Big Experiment",
            "source_run_id": "w4_c4_r1",
            "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "measurement_type": "measured"
        }
    ]
    df = pd.DataFrame(rows)
    out_path = os.path.join(EXTRACTED_DIR, "hbm_hit_reload_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"[SAVE] Table 4: {out_path} ({len(df)} rows)")

def generate_table5_predictor_quality():
    """Table 5: Predictor Quality Comparison (Figure 9)."""
    rows = [
        # Horizon h=1
        {"metric_name": "macro_recall", "metric_value": 47.91, "unit": "percent", "configuration": "ST-MoE (h=1)", "workload": "MoE-Anywhere Sage-A1.5", "source_campaign": "Point 1", "source_run_id": "p1_stmoe_h1", "exact_source_file": "mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv", "measurement_type": "measured"},
        {"metric_name": "macro_recall", "metric_value": 45.51, "unit": "percent", "configuration": "CSTP (h=1)", "workload": "MoE-Anywhere Sage-A1.5", "source_campaign": "Point 1", "source_run_id": "p1_cstp_h1", "exact_source_file": "mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv", "measurement_type": "measured"},
        {"metric_name": "macro_precision", "metric_value": 47.91, "unit": "percent", "configuration": "ST-MoE (h=1)", "workload": "MoE-Anywhere Sage-A1.5", "source_campaign": "Point 1", "source_run_id": "p1_stmoe_h1", "exact_source_file": "mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv", "measurement_type": "measured"},
        {"metric_name": "macro_precision", "metric_value": 45.51, "unit": "percent", "configuration": "CSTP (h=1)", "workload": "MoE-Anywhere Sage-A1.5", "source_campaign": "Point 1", "source_run_id": "p1_cstp_h1", "exact_source_file": "mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv", "measurement_type": "measured"},
        {"metric_name": "macro_f1", "metric_value": 47.91, "unit": "percent", "configuration": "ST-MoE (h=1)", "workload": "MoE-Anywhere Sage-A1.5", "source_campaign": "Point 1", "source_run_id": "p1_stmoe_h1", "exact_source_file": "mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv", "measurement_type": "measured"},
        {"metric_name": "macro_f1", "metric_value": 45.51, "unit": "percent", "configuration": "CSTP (h=1)", "workload": "MoE-Anywhere Sage-A1.5", "source_campaign": "Point 1", "source_run_id": "p1_cstp_h1", "exact_source_file": "mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv", "measurement_type": "measured"},
        {"metric_name": "macro_top_k_hit_rate", "metric_value": 96.94, "unit": "percent", "configuration": "ST-MoE (h=1)", "workload": "MoE-Anywhere Sage-A1.5", "source_campaign": "Point 1", "source_run_id": "p1_stmoe_h1", "exact_source_file": "mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv", "measurement_type": "measured"},
        {"metric_name": "macro_top_k_hit_rate", "metric_value": 96.67, "unit": "percent", "configuration": "CSTP (h=1)", "workload": "MoE-Anywhere Sage-A1.5", "source_campaign": "Point 1", "source_run_id": "p1_cstp_h1", "exact_source_file": "mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv", "measurement_type": "measured"},
        # Horizon h=2
        {"metric_name": "macro_recall", "metric_value": 46.20, "unit": "percent", "configuration": "ST-MoE (h=2)", "workload": "MoE-Anywhere Sage-A1.5", "source_campaign": "Point 1", "source_run_id": "p1_stmoe_h2", "exact_source_file": "mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv", "measurement_type": "measured"},
        {"metric_name": "macro_recall", "metric_value": 44.18, "unit": "percent", "configuration": "CSTP (h=2)", "workload": "MoE-Anywhere Sage-A1.5", "source_campaign": "Point 1", "source_run_id": "p1_cstp_h2", "exact_source_file": "mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv", "measurement_type": "measured"},
        {"metric_name": "macro_precision", "metric_value": 46.20, "unit": "percent", "configuration": "ST-MoE (h=2)", "workload": "MoE-Anywhere Sage-A1.5", "source_campaign": "Point 1", "source_run_id": "p1_stmoe_h2", "exact_source_file": "mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv", "measurement_type": "measured"},
        {"metric_name": "macro_precision", "metric_value": 44.18, "unit": "percent", "configuration": "CSTP (h=2)", "workload": "MoE-Anywhere Sage-A1.5", "source_campaign": "Point 1", "source_run_id": "p1_cstp_h2", "exact_source_file": "mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv", "measurement_type": "measured"},
        {"metric_name": "macro_f1", "metric_value": 46.20, "unit": "percent", "configuration": "ST-MoE (h=2)", "workload": "MoE-Anywhere Sage-A1.5", "source_campaign": "Point 1", "source_run_id": "p1_stmoe_h2", "exact_source_file": "mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv", "measurement_type": "measured"},
        {"metric_name": "macro_f1", "metric_value": 44.18, "unit": "percent", "configuration": "CSTP (h=2)", "workload": "MoE-Anywhere Sage-A1.5", "source_campaign": "Point 1", "source_run_id": "p1_cstp_h2", "exact_source_file": "mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv", "measurement_type": "measured"},
        {"metric_name": "macro_top_k_hit_rate", "metric_value": 95.85, "unit": "percent", "configuration": "ST-MoE (h=2)", "workload": "MoE-Anywhere Sage-A1.5", "source_campaign": "Point 1", "source_run_id": "p1_stmoe_h2", "exact_source_file": "mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv", "measurement_type": "measured"},
        {"metric_name": "macro_top_k_hit_rate", "metric_value": 96.14, "unit": "percent", "configuration": "CSTP (h=2)", "workload": "MoE-Anywhere Sage-A1.5", "source_campaign": "Point 1", "source_run_id": "p1_cstp_h2", "exact_source_file": "mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv", "measurement_type": "measured"}
    ]
    df = pd.DataFrame(rows)
    out_path = os.path.join(EXTRACTED_DIR, "predictor_quality_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"[SAVE] Table 5: {out_path} ({len(df)} rows)")

def generate_table6_prefetch_effectiveness():
    """Table 6: Prefetch Effectiveness and Cache Pollution (Figure 10)."""
    rows = [
        # Point 3 Large Workload W6 (B=70, T=32, h=2)
        {"metric_name": "dispatched_prefetches", "metric_value": 35272, "unit": "count", "configuration": "ST-MoE Prefetch (h=2)", "workload": "W6 (B=70, T=32)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w6_stmoe_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "dispatched_prefetches", "metric_value": 17659, "unit": "count", "configuration": "CSTP Prefetch (h=2)", "workload": "W6 (B=70, T=32)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w6_cstp_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "late_inflight_attachments", "metric_value": 32817, "unit": "count", "configuration": "ST-MoE Prefetch (h=2)", "workload": "W6 (B=70, T=32)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w6_stmoe_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "late_inflight_attachments", "metric_value": 16002, "unit": "count", "configuration": "CSTP Prefetch (h=2)", "workload": "W6 (B=70, T=32)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w6_cstp_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "unused_completed_prefetches", "metric_value": 2388, "unit": "count", "configuration": "ST-MoE Prefetch (h=2)", "workload": "W6 (B=70, T=32)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w6_stmoe_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "unused_completed_prefetches", "metric_value": 1656, "unit": "count", "configuration": "CSTP Prefetch (h=2)", "workload": "W6 (B=70, T=32)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w6_cstp_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "unused_prefetch_rate_pct", "metric_value": 6.78, "unit": "percent", "configuration": "ST-MoE Prefetch (h=2)", "workload": "W6 (B=70, T=32)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w6_stmoe_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "calculated"},
        {"metric_name": "unused_prefetch_rate_pct", "metric_value": 9.38, "unit": "percent", "configuration": "CSTP Prefetch (h=2)", "workload": "W6 (B=70, T=32)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w6_cstp_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "calculated"},
        {"metric_name": "total_evictions", "metric_value": 136898, "unit": "count", "configuration": "ST-MoE Prefetch (h=2)", "workload": "W6 (B=70, T=32)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w6_stmoe_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "total_evictions", "metric_value": 135713, "unit": "count", "configuration": "CSTP Prefetch (h=2)", "workload": "W6 (B=70, T=32)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w6_cstp_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        # Point 3 Large Workload W7 (B=70, T=64, h=2)
        {"metric_name": "dispatched_prefetches", "metric_value": 102737, "unit": "count", "configuration": "ST-MoE Prefetch (h=2)", "workload": "W7 (B=70, T=64)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w7_stmoe_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "dispatched_prefetches", "metric_value": 54076, "unit": "count", "configuration": "CSTP Prefetch (h=2)", "workload": "W7 (B=70, T=64)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w7_cstp_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "unused_completed_prefetches", "metric_value": 6022, "unit": "count", "configuration": "ST-MoE Prefetch (h=2)", "workload": "W7 (B=70, T=64)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w7_stmoe_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "unused_completed_prefetches", "metric_value": 3812, "unit": "count", "configuration": "CSTP Prefetch (h=2)", "workload": "W7 (B=70, T=64)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w7_cstp_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"}
    ]
    df = pd.DataFrame(rows)
    out_path = os.path.join(EXTRACTED_DIR, "prefetch_effectiveness_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"[SAVE] Table 6: {out_path} ({len(df)} rows)")

def generate_table7_falru_incremental():
    """Table 7: LRU vs FA-LRU Contribution (Figure 11)."""
    rows = [
        # Workload W3 (B=14)
        {"metric_name": "coalesced_resident_hits_lru", "metric_value": 1657, "unit": "count", "configuration": "CSTP + LRU", "workload": "W3 (B=14, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "w3_c3_r1", "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv", "measurement_type": "measured"},
        {"metric_name": "coalesced_resident_hits_falru", "metric_value": 1643, "unit": "count", "configuration": "CSTP + FA-LRU (W=64)", "workload": "W3 (B=14, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "w3_c4_r1", "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv", "measurement_type": "measured"},
        {"metric_name": "delta_resident_hits", "metric_value": -14, "unit": "count", "configuration": "FA-LRU vs LRU", "workload": "W3 (B=14, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "paired_w3_c4_c3", "exact_source_file": "mohman/point_4/big_exp/results/paired_comparisons.csv", "measurement_type": "measured_delta"},
        # Workload W4 (B=32)
        {"metric_name": "coalesced_resident_hits_lru", "metric_value": 1417, "unit": "count", "configuration": "CSTP + LRU", "workload": "W4 (B=32, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "w4_c3_r1", "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv", "measurement_type": "measured"},
        {"metric_name": "coalesced_resident_hits_falru", "metric_value": 1432, "unit": "count", "configuration": "CSTP + FA-LRU (W=64)", "workload": "W4 (B=32, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "w4_c4_r1", "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv", "measurement_type": "measured"},
        {"metric_name": "delta_resident_hits", "metric_value": 15, "unit": "count", "configuration": "FA-LRU vs LRU", "workload": "W4 (B=32, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "paired_w4_c4_c3", "exact_source_file": "mohman/point_4/big_exp/results/paired_comparisons.csv", "measurement_type": "measured_delta"},
        # Incremental Physical Savings in W4 (B=32)
        {"metric_name": "ddr5_traffic_saved", "metric_value": 2.0258, "unit": "decimal_GB", "configuration": "FA-LRU vs LRU", "workload": "W4 (B=32, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "paired_w4_c4_c3", "exact_source_file": "mohman/point_4/big_exp/results/paired_comparisons.csv", "measurement_type": "calculated_delta"},
        {"metric_name": "exposed_gpu_stall_saved", "metric_value": 31.82, "unit": "milliseconds", "configuration": "FA-LRU vs LRU", "workload": "W4 (B=32, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "paired_w4_c4_c3", "exact_source_file": "mohman/point_4/big_exp/results/paired_comparisons.csv", "measurement_type": "measured_delta"},
        {"metric_name": "execution_time_saved", "metric_value": 31.82, "unit": "milliseconds", "configuration": "FA-LRU vs LRU", "workload": "W4 (B=32, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "paired_w4_c4_c3", "exact_source_file": "mohman/point_4/big_exp/results/paired_comparisons.csv", "measurement_type": "measured_delta"},
        {"metric_name": "expert_reloads_avoided", "metric_value": 15, "unit": "count", "configuration": "FA-LRU vs LRU", "workload": "W4 (B=32, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "paired_w4_c4_c3", "exact_source_file": "mohman/point_4/big_exp/results/paired_comparisons.csv", "measurement_type": "measured_delta"},
        {"metric_name": "protected_lines_reused", "metric_value": 36, "unit": "count", "configuration": "FA-LRU Diagnostics", "workload": "W4 (B=32, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "w4_c4_r1", "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv", "measurement_type": "measured_diagnostic"},
        {"metric_name": "speculative_prefetches_evicted", "metric_value": 5999, "unit": "count", "configuration": "FA-LRU Diagnostics", "workload": "W4 (B=32, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "w4_c4_r1", "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv", "measurement_type": "measured_diagnostic"},
        {"metric_name": "fa_lru_divergent_decisions", "metric_value": 431909, "unit": "count", "configuration": "FA-LRU Diagnostics", "workload": "W4 (B=32, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "w4_c4_r1", "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv", "measurement_type": "measured_diagnostic"}
    ]
    df = pd.DataFrame(rows)
    out_path = os.path.join(EXTRACTED_DIR, "falru_incremental_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"[SAVE] Table 7: {out_path} ({len(df)} rows)")

def generate_table8_batch_scaling():
    """Table 8: Batch-Size Scaling Comparison (Figure 12)."""
    rows = [
        # Point 3 Large Workload Campaign (T=16, 80 GB cache)
        {"metric_name": "exposed_gpu_stall_ms", "metric_value": 18117.03, "unit": "milliseconds", "configuration": "Baseline", "workload": "W2 (B=14, T=16)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w2_baseline", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "exposed_gpu_stall_ms", "metric_value": 18067.89, "unit": "milliseconds", "configuration": "ST-MoE + LRU", "workload": "W2 (B=14, T=16)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w2_stmoe_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "exposed_gpu_stall_ms", "metric_value": 18055.42, "unit": "milliseconds", "configuration": "CSTP + LRU", "workload": "W2 (B=14, T=16)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w2_cstp_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "exposed_gpu_stall_ms", "metric_value": 33639.29, "unit": "milliseconds", "configuration": "Baseline", "workload": "W3 (B=32, T=16)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w3_baseline", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "exposed_gpu_stall_ms", "metric_value": 33556.78, "unit": "milliseconds", "configuration": "ST-MoE + LRU", "workload": "W3 (B=32, T=16)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w3_stmoe_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "exposed_gpu_stall_ms", "metric_value": 33529.28, "unit": "milliseconds", "configuration": "CSTP + LRU", "workload": "W3 (B=32, T=16)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w3_cstp_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "exposed_gpu_stall_ms", "metric_value": 50768.71, "unit": "milliseconds", "configuration": "Baseline", "workload": "W4 (B=48, T=16)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w4_baseline", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "exposed_gpu_stall_ms", "metric_value": 50668.16, "unit": "milliseconds", "configuration": "ST-MoE + LRU", "workload": "W4 (B=48, T=16)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w4_stmoe_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "exposed_gpu_stall_ms", "metric_value": 50566.14, "unit": "milliseconds", "configuration": "CSTP + LRU", "workload": "W4 (B=48, T=16)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w4_cstp_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "exposed_gpu_stall_ms", "metric_value": 62304.49, "unit": "milliseconds", "configuration": "Baseline", "workload": "W5 (B=70, T=16)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w5_baseline", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "exposed_gpu_stall_ms", "metric_value": 62043.92, "unit": "milliseconds", "configuration": "ST-MoE + LRU", "workload": "W5 (B=70, T=16)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w5_stmoe_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        {"metric_name": "exposed_gpu_stall_ms", "metric_value": 61794.55, "unit": "milliseconds", "configuration": "CSTP + LRU", "workload": "W5 (B=70, T=16)", "source_campaign": "Point 3 Large Workload", "source_run_id": "w5_cstp_h2", "exact_source_file": "mohman/point_3/phase2/large_workload/results/predictor_comparison.csv", "measurement_type": "measured"},
        # Big Experiment Campaign (T=128, 35 GiB cache) - Mean Token Latency
        {"metric_name": "mean_token_latency_ms", "metric_value": 3586.18, "unit": "milliseconds", "configuration": "Config A: Baseline", "workload": "W3 (B=14, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "w3_c1_r1", "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv", "measurement_type": "measured"},
        {"metric_name": "mean_token_latency_ms", "metric_value": 3565.37, "unit": "milliseconds", "configuration": "Config C2: CSTP + LRU", "workload": "W3 (B=14, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "w3_c3_r1", "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv", "measurement_type": "measured"},
        {"metric_name": "mean_token_latency_ms", "metric_value": 3565.49, "unit": "milliseconds", "configuration": "Config D: Complete System", "workload": "W3 (B=14, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "w3_c4_r1", "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv", "measurement_type": "measured"},
        {"metric_name": "mean_token_latency_ms", "metric_value": 7876.59, "unit": "milliseconds", "configuration": "Config A: Baseline", "workload": "W4 (B=32, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "w4_c1_r1", "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv", "measurement_type": "measured"},
        {"metric_name": "mean_token_latency_ms", "metric_value": 5950.06, "unit": "milliseconds", "configuration": "Config C2: CSTP + LRU", "workload": "W4 (B=32, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "w4_c3_r1", "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv", "measurement_type": "measured"},
        {"metric_name": "mean_token_latency_ms", "metric_value": 5949.81, "unit": "milliseconds", "configuration": "Config D: Complete System", "workload": "W4 (B=32, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "w4_c4_r1", "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv", "measurement_type": "measured"},
        # Big Experiment Campaign (T=128, 35 GiB cache) - Full Simulated Execution Time (s)
        {"metric_name": "simulated_execution_time_s", "metric_value": 459.031, "unit": "seconds", "configuration": "Config A: Baseline", "workload": "W3 (B=14, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "w3_c1_r1", "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv", "measurement_type": "measured"},
        {"metric_name": "simulated_execution_time_s", "metric_value": 456.368, "unit": "seconds", "configuration": "Config C2: CSTP + LRU", "workload": "W3 (B=14, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "w3_c3_r1", "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv", "measurement_type": "measured"},
        {"metric_name": "simulated_execution_time_s", "metric_value": 456.383, "unit": "seconds", "configuration": "Config D: Complete System", "workload": "W3 (B=14, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "w3_c4_r1", "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv", "measurement_type": "measured"},
        {"metric_name": "simulated_execution_time_s", "metric_value": 1008.204, "unit": "seconds", "configuration": "Config A: Baseline", "workload": "W4 (B=32, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "w4_c1_r1", "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv", "measurement_type": "measured"},
        {"metric_name": "simulated_execution_time_s", "metric_value": 761.608, "unit": "seconds", "configuration": "Config C2: CSTP + LRU", "workload": "W4 (B=32, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "w4_c3_r1", "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv", "measurement_type": "measured"},
        {"metric_name": "simulated_execution_time_s", "metric_value": 761.576, "unit": "seconds", "configuration": "Config D: Complete System", "workload": "W4 (B=32, T=128)", "source_campaign": "Point 4 Big Experiment", "source_run_id": "w4_c4_r1", "exact_source_file": "mohman/point_4/big_exp/results/all_runs.csv", "measurement_type": "measured"}
    ]
    df = pd.DataFrame(rows)
    out_path = os.path.join(EXTRACTED_DIR, "batch_scaling_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"[SAVE] Table 8: {out_path} ({len(df)} rows)")

def main():
    verify_zero_ternary(__file__)
    generate_table1_execution_time()
    generate_table2_exposed_stall()
    generate_table3_ddr5_traffic()
    generate_table4_hbm_hits_reloads()
    generate_table5_predictor_quality()
    generate_table6_prefetch_effectiveness()
    generate_table7_falru_incremental()
    generate_table8_batch_scaling()

if __name__ == "__main__":
    main()

