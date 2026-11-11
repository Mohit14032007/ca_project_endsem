#!/usr/bin/env python3
"""
Extract and Consolidate Data for Point 5 Comparative Evaluation.
Parses existing artifacts from Point 1, Point 3, Point 4, and phase_4d5c.
Enforces strictly zero ternary operators (ast.IfExp == 0).
"""

import os
import ast
import json
import pandas as pd
import numpy as np

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

def extract_big_exp_data():
    """Extracts matched runs from Point 4 Big Experiment."""
    csv_path = os.path.join(WORKSPACE, "mohman", "point_4", "big_exp", "results", "all_runs.csv")
    df = pd.read_csv(csv_path)
    
    # We use Repetition 1 as primary (Rep 2 is bit-exact identical)
    df_rep1 = df[df["repetition"] == 1].copy()
    
    extracted_rows = []
    for _, row in df_rep1.iterrows():
        cfg_id = row["config_id"]
        if cfg_id == "c1":
            conceptual_cfg = "Config_A_Baseline"
        elif cfg_id == "c2":
            conceptual_cfg = "Config_A_FA_LRU_NoPrefetch"
        elif cfg_id == "c3":
            conceptual_cfg = "Config_C2_CSTP_LRU"
        elif cfg_id == "c4":
            conceptual_cfg = "Config_D_Complete_System"
        else:
            conceptual_cfg = "Other"
            
        if row["prefetch_enabled"]:
            pred_name = "cstp"
        else:
            pred_name = "none"

        extracted_rows.append({
            "campaign": "Point_4_Big_Experiment",
            "source_file": "mohman/point_4/big_exp/results/all_runs.csv",
            "run_id": row["run_id"],
            "conceptual_config": conceptual_cfg,
            "workload": row["workload"],
            "batch_size": row["batch_size"],
            "num_tokens": row["num_tokens"],
            "num_stages": row["num_stages"],
            "top_k": 8,
            "hbm_capacity_gib": row["capacity_gib"],
            "hbm_slots": 426,
            "predictor": pred_name,
            "prefetch_enabled": row["prefetch_enabled"],
            "eviction_policy": row["eviction_policy"],
            "window_size": row["window_size"],
            "coalesced_actual_demands": row["coalesced_actual_demands"],
            "coalesced_resident_hits": row["coalesced_resident_hits"],
            "coalesced_hbm_hit_rate_pct": row["coalesced_hbm_hit_rate"] * 100.0,
            "expert_reloads": row["reloaded_experts_count"],
            "total_evictions": row["total_evictions"],
            "total_ddr5_bytes": row["total_ddr5_bytes"],
            "total_ddr5_gb": row["total_ddr5_gb"],
            "demand_bytes": row["demand_bytes"],
            "speculative_bytes": row["speculative_bytes"],
            "exposed_gpu_stall_ms": row["exposed_gpu_stall_ms"],
            "mean_token_latency_ms": row["mean_token_latency_ms"],
            "p95_token_latency_ms": row["p95_token_latency_ms"],
            "simulated_execution_time_ms": row["simulated_execution_time_ms"],
            "simulated_tokens_per_sec": row["simulated_tokens_per_second"],
            "host_wall_clock_runtime_s": row["host_wall_clock_runtime_s"],
            "fa_lru_divergent_decisions": row["fa_lru_divergent_decisions"],
            "fa_lru_protected_demand_experts": row["fa_lru_protected_demand_experts"],
            "fa_lru_protected_experts_reused": row["fa_lru_protected_experts_reused"]
        })
    return pd.DataFrame(extracted_rows)

def extract_point3_large_workload():
    """Extracts runs from Point 3 Phase 2 Large Workload Campaign."""
    csv_path = os.path.join(WORKSPACE, "mohman", "point_3", "phase2", "large_workload", "results", "per_run_metrics.csv")
    df = pd.read_csv(csv_path)
    
    extracted_rows = []
    for _, row in df.iterrows():
        pred = row["predictor"]
        if pred == "baseline":
            conceptual_cfg = "Config_A_Baseline"
        elif pred == "stmoe":
            conceptual_cfg = "Config_C1_STMoE_LRU"
        elif pred == "cstp":
            conceptual_cfg = "Config_C2_CSTP_LRU"
        else:
            conceptual_cfg = "Other"
            
        if row["predictor"] == "baseline":
            prefetch_flag = False
        else:
            prefetch_flag = True

        extracted_rows.append({
            "campaign": "Point_3_Phase2_Large_Workload",
            "source_file": "mohman/point_3/phase2/large_workload/results/per_run_metrics.csv",
            "run_id": row["run_id"],
            "conceptual_config": conceptual_cfg,
            "workload": row["workload_id"],
            "batch_size": row["batch_size"],
            "num_tokens": row["tokens"],
            "num_stages": row["stages"],
            "top_k": 8,
            "hbm_capacity_gib": 74.5, # 80 GB
            "hbm_slots": 908,
            "predictor": row["predictor"],
            "prefetch_enabled": prefetch_flag,
            "eviction_policy": "lru",
            "window_size": "NA",
            "coalesced_actual_demands": row["coalesced_invocations"],
            "coalesced_resident_hits": row["ordinary_hbm_hits"],
            "coalesced_hbm_hit_rate_pct": row["hbm_hit_rate_pct"],
            "expert_reloads": "NA",
            "total_evictions": row["evicts"],
            "total_ddr5_bytes": row["total_ddr5_bytes"],
            "total_ddr5_gb": row["total_ddr5_bytes"] / 1e9,
            "demand_bytes": row["demand_bytes"],
            "speculative_bytes": row["speculative_bytes"],
            "exposed_gpu_stall_ms": row["exposed_gpu_stall_ms"],
            "mean_token_latency_ms": row["mean_token_lat_ms"],
            "p95_token_latency_ms": row["p95_token_lat_ms"],
            "simulated_execution_time_ms": row["sim_end_time_s"] * 1000.0,
            "simulated_tokens_per_sec": row["tokens_per_sec"],
            "host_wall_clock_runtime_s": "NA",
            "fa_lru_divergent_decisions": 0,
            "fa_lru_protected_demand_experts": 0,
            "fa_lru_protected_experts_reused": 0
        })
    return pd.DataFrame(extracted_rows)

def extract_phase_4d5c_data():
    """Extracts runs from phase_4d5c (authentic shadow mode prediction-only)."""
    csv_path = os.path.join(WORKSPACE, "LLMSimulator", "results", "phase_4d5c", "production_matrix_summary.csv")
    df = pd.read_csv(csv_path)
    
    extracted_rows = []
    for _, row in df.iterrows():
        r_id = row["id"]
        if r_id == "c01_baseline_lru":
            conceptual_cfg = "Config_A_Baseline"
        elif r_id == "c02_stmoe_predonly":
            conceptual_cfg = "Config_B_STMoE_Prediction_Only"
        elif r_id == "c03_cstp_predonly":
            conceptual_cfg = "Config_B_CSTP_Prediction_Only"
        elif r_id == "c04_stmoe_prefetch_lru":
            conceptual_cfg = "Config_C1_STMoE_LRU"
        elif r_id == "c05_cstp_prefetch_lru":
            conceptual_cfg = "Config_C2_CSTP_LRU"
        else:
            continue
            
        if row["mode"] == "hardware_async":
            prefetch_flag = True
        else:
            prefetch_flag = False

        extracted_rows.append({
            "campaign": "Historical_Phase_4d5c",
            "source_file": "LLMSimulator/results/phase_4d5c/production_matrix_summary.csv",
            "run_id": row["id"],
            "conceptual_config": conceptual_cfg,
            "workload": "phase_4d5c_ms_task12",
            "batch_size": row["BATCH_SIZE"],
            "num_tokens": row["ITERATIONS"],
            "num_stages": 75,
            "top_k": 8,
            "hbm_capacity_gib": 74.5, # 80 GB
            "hbm_slots": 908,
            "predictor": row["predictor"],
            "prefetch_enabled": prefetch_flag,
            "eviction_policy": row["eviction"],
            "window_size": "NA",
            "coalesced_actual_demands": row["ROUTING_RECORDS"],
            "coalesced_resident_hits": row["hbm_demand_hits"],
            "coalesced_hbm_hit_rate_pct": row["hbm_hit_rate_pct"],
            "expert_reloads": "NA",
            "total_evictions": row["hbm_demand_evictions"] + row["hbm_speculative_evictions"],
            "total_ddr5_bytes": row["total_ddr5_bytes_transferred"],
            "total_ddr5_gb": row["total_ddr5_bytes_transferred"] / 1e9,
            "demand_bytes": row["demand_bytes_transferred"],
            "speculative_bytes": row["speculative_bytes_transferred"],
            "exposed_gpu_stall_ms": row["total_exposed_stall_ns"] / 1e6,
            "mean_token_latency_ms": row["sim_per_token_latency_avg_ms"],
            "p95_token_latency_ms": "NA",
            "simulated_execution_time_ms": row["sim_total_device_time_ns"] / 1e6,
            "simulated_tokens_per_sec": (row["BATCH_SIZE"] * row["ITERATIONS"]) / (row["sim_total_device_time_ns"] / 1e9),
            "host_wall_clock_runtime_s": row["host_runtime_sec"],
            "fa_lru_divergent_decisions": 0,
            "fa_lru_protected_demand_experts": 0,
            "fa_lru_protected_experts_reused": 0
        })
    return pd.DataFrame(extracted_rows)

def extract_predictor_metrics():
    """Extracts offline and in-situ predictor metrics."""
    # From phase_4d5c
    csv_path_4d5c = os.path.join(WORKSPACE, "LLMSimulator", "results", "phase_4d5c", "production_matrix_summary.csv")
    df_4d5c = pd.read_csv(csv_path_4d5c)
    
    rows = []
    for _, r in df_4d5c[df_4d5c["id"].isin(["c02_stmoe_predonly", "c03_cstp_predonly", "c04_stmoe_prefetch_lru", "c05_cstp_prefetch_lru"])].iterrows():
        rows.append({
            "source": "LLMSimulator/results/phase_4d5c/production_matrix_summary.csv",
            "evaluation_type": "in_situ_simulation",
            "run_id": r["id"],
            "predictor": r["predictor"],
            "mode": r["mode"],
            "horizon": 1,
            "budget": 4,
            "precision": r["pred_precision_overall"],
            "recall": r["pred_recall_overall"],
            "f1_score": r["pred_f1_overall"],
            "hit_rate": r["pred_hit_rate_overall"],
            "mrr": r["pred_mrr_overall"],
            "ndcg": r["pred_ndcg_overall"],
            "mean_pred_latency_us": r["pred_mean_latency_ns"] / 1000.0,
            "mean_update_latency_us": r["pred_mean_update_latency_ns"] / 1000.0,
            "host_overhead_pct": r["pred_host_overhead_pct"]
        })
        
    # From Point 1 corrected aggregate metrics
    p1_csv = os.path.join(WORKSPACE, "mohman", "point_1", "stmoe_cstp", "results", "csv", "corrected_aggregate_metrics.csv")
    df_p1 = pd.read_csv(p1_csv)
    # Filter for seed 42, 100% profiling, horizon 1 & 2
    sub_p1 = df_p1[(df_p1["seed"] == 42) & (df_p1["profiling_pct"] == 100) & (df_p1["regime"] == "online_within_request")]
    for _, r in sub_p1.iterrows():
        rows.append({
            "source": "mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv",
            "evaluation_type": "offline_trace_evaluation",
            "run_id": f"p1_{r['predictor'].lower()}_h{r['horizon']}",
            "predictor": r["predictor"].lower().replace("-", ""),
            "mode": "offline_evaluation",
            "horizon": r["horizon"],
            "budget": r["budget"],
            "precision": r["macro_precision"],
            "recall": r["macro_recall"],
            "f1_score": r["macro_f1"],
            "hit_rate": r["macro_hit_rate"],
            "mrr": "NA",
            "ndcg": "NA",
            "mean_pred_latency_us": "NA",
            "mean_update_latency_us": "NA",
            "host_overhead_pct": "NA"
        })
        
    return pd.DataFrame(rows)

def extract_phase3_data():
    """Extracts runs from Point 4 Phase 3 Focused 24-Run Campaign."""
    csv_path = os.path.join(WORKSPACE, "mohman", "point_4", "phase_3_focused_24", "consolidated", "all_24_runs.csv")
    df = pd.read_csv(csv_path)

    w_batch_map = {"w1": 1, "w2": 2, "w3": 4, "w4": 8}
    extracted_rows = []
    for _, row in df.iterrows():
        cfg_id = row["config_id"]
        if cfg_id == "c1":
            conceptual_cfg = "Config_A_Baseline"
        elif cfg_id == "c2":
            conceptual_cfg = "Config_A_FA_LRU_NoPrefetch"
        elif cfg_id == "c3":
            conceptual_cfg = "Config_C2_CSTP_LRU"
        elif cfg_id == "c4":
            conceptual_cfg = "Config_D_Complete_System"
        else:
            conceptual_cfg = "Other"

        w_name = str(row["workload"]).lower()
        if w_name in w_batch_map:
            b_size = w_batch_map[w_name]
        else:
            b_size = 2

        if row["prefetch_enabled"]:
            pred_name = "cstp"
        else:
            pred_name = "none"

        extracted_rows.append({
            "campaign": "Point_4_Phase_3_Focused_24",
            "source_file": "mohman/point_4/phase_3_focused_24/consolidated/all_24_runs.csv",
            "run_id": row["run_id"],
            "conceptual_config": conceptual_cfg,
            "workload": f"{row['workload']}_{row['capacity_id']}",
            "batch_size": b_size,
            "num_tokens": 20,
            "num_stages": 40,
            "top_k": 8,
            "hbm_capacity_gib": row["capacity_bytes"] / (1024**3),
            "hbm_slots": int(row["capacity_bytes"] // 88080384),
            "predictor": pred_name,
            "prefetch_enabled": row["prefetch_enabled"],
            "eviction_policy": row["eviction_policy"],
            "window_size": 16,
            "coalesced_actual_demands": row["coalesced_actual_demands"],
            "coalesced_resident_hits": row["coalesced_resident_hits"],
            "coalesced_hbm_hit_rate_pct": row["coalesced_hbm_hit_rate"] * 100.0,
            "expert_reloads": row["reloaded_experts_count"],
            "total_evictions": row["total_evictions"],
            "total_ddr5_bytes": row["total_ddr5_bytes"],
            "total_ddr5_gb": row["total_ddr5_gb"],
            "demand_bytes": row["demand_bytes"],
            "speculative_bytes": row["speculative_bytes"],
            "exposed_gpu_stall_ms": row["exposed_gpu_stall_ms"],
            "mean_token_latency_ms": row["mean_token_latency_ms"],
            "p95_token_latency_ms": row["p95_token_latency_ms"],
            "simulated_execution_time_ms": row["simulated_execution_time_ms"],
            "simulated_tokens_per_sec": row["simulated_tokens_per_second"],
            "host_wall_clock_runtime_s": row["host_wall_clock_runtime_s"],
            "fa_lru_divergent_decisions": 0,
            "fa_lru_protected_demand_experts": 0,
            "fa_lru_protected_experts_reused": 0
        })
    return pd.DataFrame(extracted_rows)

def main():
    verify_zero_ternary(__file__)
    
    df_big = extract_big_exp_data()
    df_p3 = extract_point3_large_workload()
    df_p4_p3 = extract_phase3_data()
    df_4d5c = extract_phase_4d5c_data()
    
    # Save category CSVs
    unified_df = pd.concat([df_big, df_p3, df_p4_p3, df_4d5c], ignore_index=True)
    unified_path = os.path.join(EXTRACTED_DIR, "unified_comparison.csv")
    unified_df.to_csv(unified_path, index=False)
    print(f"[SAVE] Unified comparison table: {unified_path} ({len(unified_df)} rows)")
    
    # 1. Baseline Results
    baseline_df = unified_df[unified_df["conceptual_config"] == "Config_A_Baseline"]
    baseline_path = os.path.join(EXTRACTED_DIR, "baseline_results.csv")
    baseline_df.to_csv(baseline_path, index=False)
    print(f"[SAVE] Baseline results: {baseline_path} ({len(baseline_df)} rows)")
    
    # 2. Prediction Only Results
    pred_only_df = unified_df[unified_df["conceptual_config"].str.startswith("Config_B")]
    pred_only_path = os.path.join(EXTRACTED_DIR, "prediction_only_results.csv")
    pred_only_df.to_csv(pred_only_path, index=False)
    print(f"[SAVE] Prediction-only results: {pred_only_path} ({len(pred_only_df)} rows)")
    
    # 3. Prefetch Results (Config C1 and C2)
    prefetch_df = unified_df[unified_df["conceptual_config"].isin(["Config_C1_STMoE_LRU", "Config_C2_CSTP_LRU"])]
    prefetch_path = os.path.join(EXTRACTED_DIR, "prefetch_results.csv")
    prefetch_df.to_csv(prefetch_path, index=False)
    print(f"[SAVE] Prefetch results: {prefetch_path} ({len(prefetch_df)} rows)")
    
    # 4. Complete System Results (Config D)
    complete_df = unified_df[unified_df["conceptual_config"] == "Config_D_Complete_System"]
    complete_path = os.path.join(EXTRACTED_DIR, "complete_system_results.csv")
    complete_df.to_csv(complete_path, index=False)
    print(f"[SAVE] Complete system results: {complete_path} ({len(complete_df)} rows)")
    
    # 5. Predictor Metrics
    pred_metrics_df = extract_predictor_metrics()
    pred_metrics_path = os.path.join(EXTRACTED_DIR, "predictor_metrics.csv")
    pred_metrics_df.to_csv(pred_metrics_path, index=False)
    print(f"[SAVE] Predictor metrics: {pred_metrics_path} ({len(pred_metrics_df)} rows)")
    
    # 6. Source Manifest
    manifest_rows = []
    for _, r in unified_df.iterrows():
        manifest_rows.append({
            "run_id": r["run_id"],
            "campaign": r["campaign"],
            "conceptual_config": r["conceptual_config"],
            "source_file": r["source_file"],
            "workload": r["workload"],
            "batch_size": r["batch_size"],
            "num_tokens": r["num_tokens"],
            "stages": r["num_stages"],
            "hbm_gib": r["hbm_capacity_gib"]
        })
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(EXTRACTED_DIR, "source_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"[SAVE] Source manifest: {manifest_path} ({len(manifest_df)} rows)")

if __name__ == "__main__":
    main()
