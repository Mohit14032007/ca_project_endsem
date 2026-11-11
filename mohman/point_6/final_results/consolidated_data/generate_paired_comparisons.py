#!/usr/bin/env python3
"""
Compute paired comparative deltas across Point 6 sweeps.
Strict zero ternary operators (ast.IfExp == 0).
"""

import os
import ast
import pandas as pd

WORKSPACE = "/home/mohman/ca_project"
POINT6_ROOT = os.path.join(WORKSPACE, "mohman", "point_6", "final_results")
CONSOLIDATED_ROOT = os.path.join(POINT6_ROOT, "consolidated_data")

def build_paired_comparisons():
    unified_csv = os.path.join(CONSOLIDATED_ROOT, "unified_point6_results.csv")
    if not os.path.exists(unified_csv):
        print(f"Unified CSV {unified_csv} not ready yet.")
        return
        
    df = pd.read_csv(unified_csv)
    pairs = []
    
    # 1. Batch Size Sweeps
    bs_df = df[df["suite"].isin(["batch_size", "anchor"]) & (df["context_p"] == 2048) & (df["budget_k"].isin([0, 4])) & (df["lookahead_h"].isin([0, 1]))]
    batches = sorted(bs_df["batch_size"].unique())
    
    for b in batches:
        sub = bs_df[bs_df["batch_size"] == b]
        base = sub[sub["prefetch_enabled"] == False]
        lru = sub[(sub["prefetch_enabled"] == True) & (sub["eviction_policy"] == "lru")]
        falru = sub[(sub["prefetch_enabled"] == True) & (sub["eviction_policy"] == "fa_lru")]
        
        if len(base) > 0 and len(lru) > 0:
            b_row = base.iloc[0]
            l_row = lru.iloc[0]
            pairs.append({
                "dimension": "batch_size",
                "param_name": "B",
                "param_value": b,
                "comparison": "Baseline_vs_CSTP_LRU",
                "baseline_run": b_row["run_id"],
                "target_run": l_row["run_id"],
                "base_latency_ms": b_row["mean_token_latency_ms"],
                "target_latency_ms": l_row["mean_token_latency_ms"],
                "latency_speedup_pct": (b_row["mean_token_latency_ms"] - l_row["mean_token_latency_ms"]) / b_row["mean_token_latency_ms"] * 100.0,
                "base_stall_ms": b_row["exposed_stall_ms"],
                "target_stall_ms": l_row["exposed_stall_ms"],
                "stall_reduction_pct": (b_row["exposed_stall_ms"] - l_row["exposed_stall_ms"]) / b_row["exposed_stall_ms"] * 100.0,
                "base_ddr5_gb": b_row["total_ddr5_gb"],
                "target_ddr5_gb": l_row["total_ddr5_gb"],
                "ddr5_traffic_delta_pct": (l_row["total_ddr5_gb"] - b_row["total_ddr5_gb"]) / b_row["total_ddr5_gb"] * 100.0,
                "base_hit_rate": b_row["demand_hit_rate_pct"],
                "target_hit_rate": l_row["demand_hit_rate_pct"],
                "hit_rate_gain_pct": l_row["demand_hit_rate_pct"] - b_row["demand_hit_rate_pct"],
                "reloads_saved": b_row["reloads"] - l_row["reloads"]
            })
            
        if len(lru) > 0 and len(falru) > 0:
            l_row = lru.iloc[0]
            f_row = falru.iloc[0]
            pairs.append({
                "dimension": "batch_size",
                "param_name": "B",
                "param_value": b,
                "comparison": "CSTP_LRU_vs_CSTP_FALRU",
                "baseline_run": l_row["run_id"],
                "target_run": f_row["run_id"],
                "base_latency_ms": l_row["mean_token_latency_ms"],
                "target_latency_ms": f_row["mean_token_latency_ms"],
                "latency_speedup_pct": (l_row["mean_token_latency_ms"] - f_row["mean_token_latency_ms"]) / l_row["mean_token_latency_ms"] * 100.0,
                "base_stall_ms": l_row["exposed_stall_ms"],
                "target_stall_ms": f_row["exposed_stall_ms"],
                "stall_reduction_pct": (l_row["exposed_stall_ms"] - f_row["exposed_stall_ms"]) / l_row["exposed_stall_ms"] * 100.0,
                "base_ddr5_gb": l_row["total_ddr5_gb"],
                "target_ddr5_gb": f_row["total_ddr5_gb"],
                "ddr5_traffic_delta_pct": (f_row["total_ddr5_gb"] - l_row["total_ddr5_gb"]) / l_row["total_ddr5_gb"] * 100.0,
                "base_hit_rate": l_row["demand_hit_rate_pct"],
                "target_hit_rate": f_row["demand_hit_rate_pct"],
                "hit_rate_gain_pct": f_row["demand_hit_rate_pct"] - l_row["demand_hit_rate_pct"],
                "reloads_saved": l_row["reloads"] - f_row["reloads"]
            })
            
    # 2. Context Length Sweeps
    ctx_df = df[df["suite"].isin(["context_length", "anchor"]) & (df["batch_size"] == 32) & (df["budget_k"].isin([0, 4])) & (df["lookahead_h"].isin([0, 1]))]
    contexts = sorted(ctx_df["context_p"].unique())
    
    for p in contexts:
        sub = ctx_df[ctx_df["context_p"] == p]
        base = sub[sub["prefetch_enabled"] == False]
        lru = sub[(sub["prefetch_enabled"] == True) & (sub["eviction_policy"] == "lru")]
        falru = sub[(sub["prefetch_enabled"] == True) & (sub["eviction_policy"] == "fa_lru")]
        
        if len(base) > 0 and len(lru) > 0:
            b_row = base.iloc[0]
            l_row = lru.iloc[0]
            pairs.append({
                "dimension": "context_length",
                "param_name": "P",
                "param_value": p,
                "comparison": "Baseline_vs_CSTP_LRU",
                "baseline_run": b_row["run_id"],
                "target_run": l_row["run_id"],
                "base_latency_ms": b_row["mean_token_latency_ms"],
                "target_latency_ms": l_row["mean_token_latency_ms"],
                "latency_speedup_pct": (b_row["mean_token_latency_ms"] - l_row["mean_token_latency_ms"]) / b_row["mean_token_latency_ms"] * 100.0,
                "base_stall_ms": b_row["exposed_stall_ms"],
                "target_stall_ms": l_row["exposed_stall_ms"],
                "stall_reduction_pct": (b_row["exposed_stall_ms"] - l_row["exposed_stall_ms"]) / b_row["exposed_stall_ms"] * 100.0,
                "base_ddr5_gb": b_row["total_ddr5_gb"],
                "target_ddr5_gb": l_row["total_ddr5_gb"],
                "ddr5_traffic_delta_pct": (l_row["total_ddr5_gb"] - b_row["total_ddr5_gb"]) / b_row["total_ddr5_gb"] * 100.0,
                "base_hit_rate": b_row["demand_hit_rate_pct"],
                "target_hit_rate": l_row["demand_hit_rate_pct"],
                "hit_rate_gain_pct": l_row["demand_hit_rate_pct"] - b_row["demand_hit_rate_pct"],
                "reloads_saved": b_row["reloads"] - l_row["reloads"]
            })
            
        if len(lru) > 0 and len(falru) > 0:
            l_row = lru.iloc[0]
            f_row = falru.iloc[0]
            pairs.append({
                "dimension": "context_length",
                "param_name": "P",
                "param_value": p,
                "comparison": "CSTP_LRU_vs_CSTP_FALRU",
                "baseline_run": l_row["run_id"],
                "target_run": f_row["run_id"],
                "base_latency_ms": l_row["mean_token_latency_ms"],
                "target_latency_ms": f_row["mean_token_latency_ms"],
                "latency_speedup_pct": (l_row["mean_token_latency_ms"] - f_row["mean_token_latency_ms"]) / l_row["mean_token_latency_ms"] * 100.0,
                "base_stall_ms": l_row["exposed_stall_ms"],
                "target_stall_ms": f_row["exposed_stall_ms"],
                "stall_reduction_pct": (l_row["exposed_stall_ms"] - f_row["exposed_stall_ms"]) / l_row["exposed_stall_ms"] * 100.0,
                "base_ddr5_gb": l_row["total_ddr5_gb"],
                "target_ddr5_gb": f_row["total_ddr5_gb"],
                "ddr5_traffic_delta_pct": (f_row["total_ddr5_gb"] - l_row["total_ddr5_gb"]) / l_row["total_ddr5_gb"] * 100.0,
                "base_hit_rate": l_row["demand_hit_rate_pct"],
                "target_hit_rate": f_row["demand_hit_rate_pct"],
                "hit_rate_gain_pct": f_row["demand_hit_rate_pct"] - l_row["demand_hit_rate_pct"],
                "reloads_saved": l_row["reloads"] - f_row["reloads"]
            })
            
    # 3. Lookahead Sweeps
    lh_df = df[df["suite"].isin(["lookahead", "anchor"]) & (df["prefetch_enabled"] == True) & (df["budget_k"] == 4)]
    horizons = sorted(lh_df["lookahead_h"].unique())
    # Compare each H with anchor H=1 for CSTP+LRU
    anchor_lru = lh_df[(lh_df["lookahead_h"] == 1) & (lh_df["eviction_policy"] == "lru")]
    if len(anchor_lru) > 0:
        a_row = anchor_lru.iloc[0]
        for h in horizons:
            sub = lh_df[(lh_df["lookahead_h"] == h) & (lh_df["eviction_policy"] == "lru")]
            if len(sub) > 0:
                h_row = sub.iloc[0]
                pairs.append({
                    "dimension": "lookahead",
                    "param_name": "H",
                    "param_value": h,
                    "comparison": "H1_vs_H_CSTP_LRU",
                    "baseline_run": a_row["run_id"],
                    "target_run": h_row["run_id"],
                    "base_latency_ms": a_row["mean_token_latency_ms"],
                    "target_latency_ms": h_row["mean_token_latency_ms"],
                    "latency_speedup_pct": (a_row["mean_token_latency_ms"] - h_row["mean_token_latency_ms"]) / a_row["mean_token_latency_ms"] * 100.0,
                    "base_stall_ms": a_row["exposed_stall_ms"],
                    "target_stall_ms": h_row["exposed_stall_ms"],
                    "stall_reduction_pct": (a_row["exposed_stall_ms"] - h_row["exposed_stall_ms"]) / a_row["exposed_stall_ms"] * 100.0,
                    "base_ddr5_gb": a_row["total_ddr5_gb"],
                    "target_ddr5_gb": h_row["total_ddr5_gb"],
                    "ddr5_traffic_delta_pct": (h_row["total_ddr5_gb"] - a_row["total_ddr5_gb"]) / a_row["total_ddr5_gb"] * 100.0,
                    "base_hit_rate": a_row["demand_hit_rate_pct"],
                    "target_hit_rate": h_row["demand_hit_rate_pct"],
                    "hit_rate_gain_pct": h_row["demand_hit_rate_pct"] - a_row["demand_hit_rate_pct"],
                    "reloads_saved": a_row["reloads"] - h_row["reloads"]
                })
                
    # 4. Budget Sweeps
    bg_df = df[df["suite"].isin(["prefetch_budget", "anchor"]) & (df["prefetch_enabled"] == True) & (df["lookahead_h"] == 1)]
    budgets = sorted(bg_df["budget_k"].unique())
    anchor_lru_bg = bg_df[(bg_df["budget_k"] == 4) & (bg_df["eviction_policy"] == "lru")]
    if len(anchor_lru_bg) > 0:
        a_row = anchor_lru_bg.iloc[0]
        for k in budgets:
            sub = bg_df[(bg_df["budget_k"] == k) & (bg_df["eviction_policy"] == "lru")]
            if len(sub) > 0:
                k_row = sub.iloc[0]
                pairs.append({
                    "dimension": "prefetch_budget",
                    "param_name": "K",
                    "param_value": k,
                    "comparison": "K4_vs_K_CSTP_LRU",
                    "baseline_run": a_row["run_id"],
                    "target_run": k_row["run_id"],
                    "base_latency_ms": a_row["mean_token_latency_ms"],
                    "target_latency_ms": k_row["mean_token_latency_ms"],
                    "latency_speedup_pct": (a_row["mean_token_latency_ms"] - k_row["mean_token_latency_ms"]) / a_row["mean_token_latency_ms"] * 100.0,
                    "base_stall_ms": a_row["exposed_stall_ms"],
                    "target_stall_ms": k_row["exposed_stall_ms"],
                    "stall_reduction_pct": (a_row["exposed_stall_ms"] - k_row["exposed_stall_ms"]) / a_row["exposed_stall_ms"] * 100.0,
                    "base_ddr5_gb": a_row["total_ddr5_gb"],
                    "target_ddr5_gb": k_row["total_ddr5_gb"],
                    "ddr5_traffic_delta_pct": (k_row["total_ddr5_gb"] - a_row["total_ddr5_gb"]) / a_row["total_ddr5_gb"] * 100.0,
                    "base_hit_rate": a_row["demand_hit_rate_pct"],
                    "target_hit_rate": k_row["demand_hit_rate_pct"],
                    "hit_rate_gain_pct": k_row["demand_hit_rate_pct"] - a_row["demand_hit_rate_pct"],
                    "reloads_saved": a_row["reloads"] - k_row["reloads"]
                })
                
    pair_df = pd.DataFrame(pairs)
    out_csv = os.path.join(CONSOLIDATED_ROOT, "paired_comparisons.csv")
    pair_df.to_csv(out_csv, index=False)
    print(f"Generated paired_comparisons.csv ({len(pair_df)} rows).")

if __name__ == "__main__":
    build_paired_comparisons()

