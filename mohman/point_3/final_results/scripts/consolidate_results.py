#!/usr/bin/env python3
"""
consolidate_results.py
Point 3 — Final Results Consolidation Script

Reads original CSV results from all Point 3 campaigns:
  1. Extended Large-Workload Campaign (27 runs, W1-W7, 40 stages, 35 GB HBM)
  2. Original 20-Run Campaign (20 runs, B1-B8, 3 stages, 80 GB HBM)
  3. Phase 2 40-Stage Scaling Campaign (26 runs, B1-B8, 40 stages, 30 GB HBM)

Outputs consolidated and derived CSVs to:
  - final_results/data/consolidated_csv/
  - final_results/data/derived_csv/

Zero ternary operators used throughout this script.
"""

import os
import sys
import pandas as pd
import numpy as np

def load_large_workload_metrics(csv_path):
    if not os.path.exists(csv_path):
        print(f"[WARN] Large workload CSV not found: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df["campaign"] = "large_workload"
    df["hbm_capacity_gb"] = 35.0
    df["max_outstanding_prefetches"] = 4
    df["simulation_exit_status"] = "SUCCESS (0)"
    return df

def load_original_campaign_metrics(csv_path):
    if not os.path.exists(csv_path):
        print(f"[WARN] Original campaign CSV not found: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df["campaign"] = "original_campaign"
    df["workload_id"] = "ORIG_B" + df["batch_size"].astype(str)
    df["tokens"] = 4
    df["stages"] = 3
    df["hbm_capacity_gb"] = 80.0
    df["max_outstanding_prefetches"] = 4
    df["simulation_exit_status"] = "SUCCESS (0)"
    return df

def load_phase2_metrics(csv_path):
    if not os.path.exists(csv_path):
        print(f"[WARN] Phase 2 CSV not found: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df["campaign"] = "phase2_scaling"
    df["workload_id"] = "P2_B" + df["batch_size"].astype(str)
    df["stages"] = 40
    df["hbm_capacity_gb"] = df["cache_gb"]
    df["max_outstanding_prefetches"] = 4
    df["simulation_exit_status"] = "SUCCESS (0)"
    return df

def build_all_experiment_results(df_lw, df_orig, df_p2):
    rows = []

    # 1. Process Large Workload Campaign
    if not df_lw.empty:
        for idx, r in df_lw.iterrows():
            row_dict = {
                "experiment_id": r["run_id"],
                "campaign": "large_workload",
                "workload_id": r["workload_id"],
                "batch_size": int(r["batch_size"]),
                "sequence_length": int(r["tokens"]),
                "num_moe_stages": int(r["stages"]),
                "predictor": r["predictor"],
                "lookahead_H": int(r["horizon"]),
                "prefetch_budget": int(r["budget"]),
                "max_outstanding_prefetches": int(r["max_outstanding_prefetches"]),
                "hbm_capacity_gb": float(r["hbm_capacity_gb"]),
                "total_simulated_time_s": float(r["sim_end_time_s"]),
                "sim_time_reduction_s": float(r["sim_time_reduction_s"]),
                "mean_decode_step_latency_ms": float(r["mean_token_lat_ms"]),
                "p50_decode_step_latency_ms": float(r["p50_token_lat_ms"]),
                "p95_decode_step_latency_ms": float(r["p95_token_lat_ms"]),
                "mean_expert_load_latency_us": float(r["mean_expert_load_lat_us"]),
                "exposed_memory_stall_ms": float(r["exposed_gpu_stall_ms"]),
                "exposed_memory_stall_reduction_ms": float(r["stall_reduction_ms"]),
                "exposed_memory_stall_reduction_pct": float(r["stall_reduction_pct"]),
                "prefetch_coverage_pct": float(r["strict_timely_coverage_pct"]),
                "strict_timely_coverage_pct": float(r["strict_timely_coverage_pct"]),
                "associated_coverage_pct": float(r["prefetch_associated_coverage_pct"]),
                "timely_prefetch_rate_pct": float(r["timely_of_matched_pct"]),
                "late_prefetch_rate_pct": float(r["late_of_matched_pct"]),
                "dispatched_prefetches": int(r["dispatched_prefetches"]),
                "prefetch_assisted_demands": int(r["prefetch_assisted_demands"]),
                "unused_prefetch_count": int(r["total_completed_unused"]),
                "unused_prefetch_rate_pct": float(r["unused_pct"]),
                "hbm_hit_rate_pct": float(r["hbm_hit_rate_pct"]),
                "demand_ddr5_bytes": int(r["demand_bytes"]),
                "speculative_ddr5_bytes": int(r["speculative_bytes"]),
                "total_ddr5_bytes": int(r["total_ddr5_bytes"]),
                "total_ddr5_gb": float(r["total_ddr5_bytes"]) / 1e9,
                "prevented_demand_bytes": int(r["prevented_demand_bytes"]),
                "prevented_demand_gb": float(r["prevented_demand_bytes"]) / 1e9,
                "wasted_speculative_bytes": int(r["wasted_speculative_bytes"]),
                "wasted_speculative_gb": float(r["wasted_speculative_bytes"]) / 1e9,
                "net_additional_memory_traffic_bytes": int(r["net_additional_traffic_bytes"]),
                "net_additional_memory_traffic_gb": float(r["net_additional_traffic_bytes"]) / 1e9,
                "traffic_overhead_pct": float(r["traffic_overhead_pct"]),
                "throughput_tokens_per_sec": float(r["tokens_per_sec"]),
                "simulation_exit_status": r["simulation_exit_status"]
            }
            rows.append(row_dict)

    # 2. Process Original Campaign
    if not df_orig.empty:
        # Compute baseline lookup map for stall reduction and time reduction
        baseline_stalls = {}
        baseline_times = {}
        for idx, r in df_orig.iterrows():
            if r["predictor"] == "baseline":
                b_size = int(r["batch_size"])
                baseline_stalls[b_size] = float(r["exposed_gpu_stall_ms"])
                baseline_times[b_size] = float(r["total_simulated_execution_ms"]) / 1000.0

        for idx, r in df_orig.iterrows():
            b_size = int(r["batch_size"])
            stall_ms = float(r["exposed_gpu_stall_ms"])
            sim_time_s = float(r["total_simulated_execution_ms"]) / 1000.0
            
            b_stall = baseline_stalls.get(b_size, stall_ms)
            b_time = baseline_times.get(b_size, sim_time_s)
            
            stall_red_ms = b_stall - stall_ms
            time_red_s = b_time - sim_time_s
            stall_red_pct = 0.0
            if b_stall > 0.0:
                stall_red_pct = (stall_red_ms / b_stall) * 100.0

            disp_prefetches = int(r["prefetches_completed_unused"]) + int(r["prefetches_completed_and_used"])
            net_traffic = int(r["additional_bytes_vs_baseline"])

            row_dict = {
                "experiment_id": r["run_id"],
                "campaign": "original_campaign",
                "workload_id": r["workload_id"],
                "batch_size": b_size,
                "sequence_length": 4,
                "num_moe_stages": 3,
                "predictor": r["predictor"],
                "lookahead_H": int(r["horizon"]),
                "prefetch_budget": int(r["budget"]),
                "max_outstanding_prefetches": 4,
                "hbm_capacity_gb": 80.0,
                "total_simulated_time_s": sim_time_s,
                "sim_time_reduction_s": time_red_s,
                "mean_decode_step_latency_ms": float(r["token_latency_mean_ms"]),
                "p50_decode_step_latency_ms": float(r["token_latency_p50_ms"]),
                "p95_decode_step_latency_ms": float(r["token_latency_p95_ms"]),
                "mean_expert_load_latency_us": float(r["expert_load_mean_us"]),
                "exposed_memory_stall_ms": stall_ms,
                "exposed_memory_stall_reduction_ms": stall_red_ms,
                "exposed_memory_stall_reduction_pct": stall_red_pct,
                "prefetch_coverage_pct": float(r["coverage_pct"]),
                "strict_timely_coverage_pct": float(r["coverage_pct"]),
                "associated_coverage_pct": float(r["coverage_pct"]),
                "timely_prefetch_rate_pct": float(r["timely_rate_of_matched_pct"]),
                "late_prefetch_rate_pct": float(r["late_rate_of_matched_pct"]),
                "dispatched_prefetches": disp_prefetches,
                "prefetch_assisted_demands": int(r["prefetch_assisted_demands"]),
                "unused_prefetch_count": int(r["prefetches_completed_unused"]),
                "unused_prefetch_rate_pct": float(r["unused_prefetch_pct"]),
                "hbm_hit_rate_pct": float(r["overall_hbm_hit_rate_pct"]),
                "demand_ddr5_bytes": int(r["demand_bytes"]),
                "speculative_ddr5_bytes": int(r["speculative_bytes"]),
                "total_ddr5_bytes": int(r["total_ddr5_bytes"]),
                "total_ddr5_gb": float(r["total_ddr5_bytes"]) / 1e9,
                "prevented_demand_bytes": 0,
                "prevented_demand_gb": 0.0,
                "wasted_speculative_bytes": int(r["completed_unused_bytes"]),
                "wasted_speculative_gb": float(r["completed_unused_bytes"]) / 1e9,
                "net_additional_memory_traffic_bytes": net_traffic,
                "net_additional_memory_traffic_gb": float(net_traffic) / 1e9,
                "traffic_overhead_pct": float(r["additional_traffic_pct"]),
                "throughput_tokens_per_sec": float(r["simulated_tokens_per_sec"]),
                "simulation_exit_status": "SUCCESS (0)"
            }
            rows.append(row_dict)

    # 3. Process Phase 2 Scaling Campaign
    if not df_p2.empty:
        baseline_stalls_p2 = {}
        baseline_times_p2 = {}
        baseline_traffic_p2 = {}
        for idx, r in df_p2.iterrows():
            if r["predictor"] == "baseline":
                key = (int(r["batch_size"]), int(r["tokens"]), float(r["hbm_capacity_gb"]))
                baseline_stalls_p2[key] = float(r["exposed_gpu_stall_ms"])
                baseline_times_p2[key] = float(r["total_simulated_time_ms"]) / 1000.0
                baseline_traffic_p2[key] = int(r["total_ddr5_bytes_transferred"])

        for idx, r in df_p2.iterrows():
            b_size = int(r["batch_size"])
            t_len = int(r["tokens"])
            c_gb = float(r["hbm_capacity_gb"])
            key = (b_size, t_len, c_gb)

            stall_ms = float(r["exposed_gpu_stall_ms"])
            sim_time_s = float(r["total_simulated_time_ms"]) / 1000.0
            tot_bytes = int(r["total_ddr5_bytes_transferred"])

            b_stall = baseline_stalls_p2.get(key, stall_ms)
            b_time = baseline_times_p2.get(key, sim_time_s)
            b_traffic = baseline_traffic_p2.get(key, tot_bytes)

            stall_red_ms = b_stall - stall_ms
            time_red_s = b_time - sim_time_s
            stall_red_pct = 0.0
            if b_stall > 0.0:
                stall_red_pct = (stall_red_ms / b_stall) * 100.0

            net_traffic = tot_bytes - b_traffic
            traffic_overhead = 0.0
            if b_traffic > 0:
                traffic_overhead = (net_traffic / b_traffic) * 100.0

            tok_per_sec = 0.0
            if sim_time_s > 0.0:
                tok_per_sec = (b_size * t_len) / sim_time_s

            disp = int(r["total_dispatched_prefetches"])
            unused_cnt = int(r["prefetches_completed_unused"])

            row_dict = {
                "experiment_id": r["run_id"],
                "campaign": "phase2_scaling",
                "workload_id": r["workload_id"],
                "batch_size": b_size,
                "sequence_length": t_len,
                "num_moe_stages": 40,
                "predictor": r["predictor"],
                "lookahead_H": int(r["horizon"]),
                "prefetch_budget": int(r["budget"]),
                "max_outstanding_prefetches": 4,
                "hbm_capacity_gb": c_gb,
                "total_simulated_time_s": sim_time_s,
                "sim_time_reduction_s": time_red_s,
                "mean_decode_step_latency_ms": float(r["mean_token_lat_ms"]),
                "p50_decode_step_latency_ms": float(r["p50_token_lat_ms"]),
                "p95_decode_step_latency_ms": float(r["p95_token_lat_ms"]),
                "mean_expert_load_latency_us": float(r["mean_expert_load_lat_us"]),
                "exposed_memory_stall_ms": stall_ms,
                "exposed_memory_stall_reduction_ms": stall_red_ms,
                "exposed_memory_stall_reduction_pct": stall_red_pct,
                "prefetch_coverage_pct": float(r["overall_coverage_pct"]),
                "strict_timely_coverage_pct": 0.0,
                "associated_coverage_pct": float(r["overall_coverage_pct"]),
                "timely_prefetch_rate_pct": float(r["timely_rate_of_matched_pct"]),
                "late_prefetch_rate_pct": float(r["late_rate_of_matched_pct"]),
                "dispatched_prefetches": disp,
                "prefetch_assisted_demands": int(r["prefetch_assisted_demands"]),
                "unused_prefetch_count": unused_cnt,
                "unused_prefetch_rate_pct": float(r["unused_pct"]),
                "hbm_hit_rate_pct": float(r["overall_hbm_hit_rate_pct"]),
                "demand_ddr5_bytes": int(r["demand_bytes_transferred"]),
                "speculative_ddr5_bytes": int(r["speculative_bytes_transferred"]),
                "total_ddr5_bytes": tot_bytes,
                "total_ddr5_gb": float(tot_bytes) / 1e9,
                "prevented_demand_bytes": np.nan,
                "prevented_demand_gb": np.nan,
                "wasted_speculative_bytes": int(r["completed_unused_bytes"]),
                "wasted_speculative_gb": float(r["completed_unused_bytes"]) / 1e9,
                "net_additional_memory_traffic_bytes": net_traffic,
                "net_additional_memory_traffic_gb": float(net_traffic) / 1e9,
                "traffic_overhead_pct": traffic_overhead,
                "throughput_tokens_per_sec": tok_per_sec,
                "simulation_exit_status": "SUCCESS (0)"
            }
            rows.append(row_dict)

    df_all = pd.DataFrame(rows)
    return df_all

def build_prefetch_effectiveness_table(df_all):
    cols = [
        "experiment_id", "campaign", "workload_id", "batch_size", "sequence_length",
        "num_moe_stages", "predictor", "lookahead_H", "prefetch_budget",
        "dispatched_prefetches", "prefetch_assisted_demands", "unused_prefetch_count",
        "strict_timely_coverage_pct", "associated_coverage_pct",
        "timely_prefetch_rate_pct", "late_prefetch_rate_pct",
        "unused_prefetch_rate_pct", "hbm_hit_rate_pct"
    ]
    return df_all[cols].copy()

def build_memory_traffic_table(df_all):
    cols = [
        "experiment_id", "campaign", "workload_id", "batch_size", "sequence_length",
        "predictor", "lookahead_H", "hbm_capacity_gb",
        "demand_ddr5_bytes", "speculative_ddr5_bytes", "total_ddr5_bytes", "total_ddr5_gb",
        "prevented_demand_bytes", "prevented_demand_gb",
        "wasted_speculative_bytes", "wasted_speculative_gb",
        "net_additional_memory_traffic_bytes", "net_additional_memory_traffic_gb",
        "traffic_overhead_pct"
    ]
    return df_all[cols].copy()

def build_latency_and_stall_table(df_all):
    cols = [
        "experiment_id", "campaign", "workload_id", "batch_size", "sequence_length",
        "predictor", "lookahead_H", "total_simulated_time_s", "sim_time_reduction_s",
        "exposed_memory_stall_ms", "exposed_memory_stall_reduction_ms",
        "exposed_memory_stall_reduction_pct", "mean_decode_step_latency_ms",
        "p50_decode_step_latency_ms", "p95_decode_step_latency_ms",
        "mean_expert_load_latency_us", "throughput_tokens_per_sec"
    ]
    return df_all[cols].copy()

def build_derived_comparisons(df_all, out_dir):
    # 1. Large workload comparison table
    df_lw = df_all[df_all["campaign"] == "large_workload"].copy()
    if not df_lw.empty:
        lw_file = os.path.join(out_dir, "large_workload_comparisons.csv")
        df_lw.to_csv(lw_file, index=False)
        print(f"Generated: {lw_file}")

    # 2. Batch scaling comparisons at T=16 (W2, W3, W4, W5)
    df_batch = df_lw[(df_lw["sequence_length"] == 16) & (df_lw["lookahead_H"].isin([0, 2]))].copy()
    if not df_batch.empty:
        b_file = os.path.join(out_dir, "batch_scaling_comparisons.csv")
        df_batch.to_csv(b_file, index=False)
        print(f"Generated: {b_file}")

    # 3. Token scaling comparisons at B=70 (W5, W6, W7)
    df_tok = df_lw[(df_lw["batch_size"] == 70) & (df_lw["lookahead_H"].isin([0, 2]))].copy()
    if not df_tok.empty:
        t_file = os.path.join(out_dir, "token_scaling_comparisons.csv")
        df_tok.to_csv(t_file, index=False)
        print(f"Generated: {t_file}")

    # 4. Total simulated time vs stall reduction trade-off
    tradeoff_cols = [
        "experiment_id", "campaign", "workload_id", "batch_size", "sequence_length",
        "predictor", "lookahead_H", "total_simulated_time_s", "sim_time_reduction_s",
        "exposed_memory_stall_ms", "exposed_memory_stall_reduction_ms",
        "exposed_memory_stall_reduction_pct", "throughput_tokens_per_sec"
    ]
    df_tradeoff = df_all[tradeoff_cols].copy()
    tradeoff_file = os.path.join(out_dir, "total_time_vs_stall_tradeoff.csv")
    df_tradeoff.to_csv(tradeoff_file, index=False)
    print(f"Generated: {tradeoff_file}")

def main():
    base_dir = "/home/mohman/ca_project/mohman/point_3/final_results"
    orig_csv_dir = os.path.join(base_dir, "data/original_csv")
    cons_csv_dir = os.path.join(base_dir, "data/consolidated_csv")
    derived_csv_dir = os.path.join(base_dir, "data/derived_csv")

    os.makedirs(cons_csv_dir, exist_ok=True)
    os.makedirs(derived_csv_dir, exist_ok=True)

    print("=== Consolidating Point 3 Results ===")

    lw_csv = os.path.join(orig_csv_dir, "large_workload_per_run_metrics.csv")
    orig_csv = os.path.join(orig_csv_dir, "original_campaign_per_run_metrics.csv")
    p2_csv = os.path.join(orig_csv_dir, "phase2_per_run_metrics.csv")

    df_lw = load_large_workload_metrics(lw_csv)
    df_orig = load_original_campaign_metrics(orig_csv)
    df_p2 = load_phase2_metrics(p2_csv)

    print(f"Loaded: Large Workload ({len(df_lw)} runs)")
    print(f"Loaded: Original Campaign ({len(df_orig)} runs)")
    print(f"Loaded: Phase 2 Scaling ({len(df_p2)} runs)")

    df_all = build_all_experiment_results(df_lw, df_orig, df_p2)
    print(f"Total Consolidated Experiments: {len(df_all)} runs")

    # Output A: all_experiment_results.csv
    all_file = os.path.join(cons_csv_dir, "all_experiment_results.csv")
    df_all.to_csv(all_file, index=False)
    print(f"Generated: {all_file}")

    # Output B: prefetch_effectiveness.csv
    eff_df = build_prefetch_effectiveness_table(df_all)
    eff_file = os.path.join(cons_csv_dir, "prefetch_effectiveness.csv")
    eff_df.to_csv(eff_file, index=False)
    print(f"Generated: {eff_file}")

    # Output C: memory_traffic.csv
    traffic_df = build_memory_traffic_table(df_all)
    traffic_file = os.path.join(cons_csv_dir, "memory_traffic.csv")
    traffic_df.to_csv(traffic_file, index=False)
    print(f"Generated: {traffic_file}")

    # Output D: latency_and_stall.csv
    lat_df = build_latency_and_stall_table(df_all)
    lat_file = os.path.join(cons_csv_dir, "latency_and_stall.csv")
    lat_df.to_csv(lat_file, index=False)
    print(f"Generated: {lat_file}")

    # Derived tables
    build_derived_comparisons(df_all, derived_csv_dir)

    print("=== Consolidation Completed Successfully ===")

if __name__ == "__main__":
    main()

