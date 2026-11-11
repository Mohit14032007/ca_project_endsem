#!/usr/bin/env python3
"""
Comprehensive telemetry parser and exporter for Point 6 Sensitivity Analysis.
Strictly ZERO ternary operators (ast.IfExp == 0).
"""

import os
import json
import ast
import pandas as pd

WORKSPACE = "/home/mohman/ca_project"
POINT6_ROOT = os.path.join(WORKSPACE, "mohman", "point_6", "final_results")
CONSOLIDATED_ROOT = os.path.join(POINT6_ROOT, "consolidated_data")
BIG_EXP_ROOT = os.path.join(WORKSPACE, "mohman", "point_4", "big_exp")

def safe_div(num, denom, default=0.0):
    if denom == 0.0 or denom == 0:
        return default
    else:
        return num / denom

def parse_run(run_id, suite, b, p, t, h, k, predictor, prefetch_enabled, policy, raw_dir):
    prefetch_summary_path = os.path.join(raw_dir, "traces", "prefetch_summary.json")
    fa_lru_path = os.path.join(raw_dir, "traces", "fa_lru_diagnostics.json")

    s_data = {}
    if os.path.exists(prefetch_summary_path):
        with open(prefetch_summary_path, "r") as f:
            s_data = json.load(f)

    fa_data = {}
    if os.path.exists(fa_lru_path):
        with open(fa_lru_path, "r") as f:
            fa_data = json.load(f)

    # Subsections in prefetch_summary.json
    hbm = s_data.get("hbm_management", {})
    stall_exec = s_data.get("exposed_stall_and_execution", {})
    traffic = s_data.get("memory_traffic", {})
    lat = s_data.get("token_latency", {})
    timely_late = s_data.get("timely_and_late_rates", {})
    unused = s_data.get("unused_prefetches", {})
    cov = s_data.get("prefetch_coverage", {})

    total_sim_exec_ns = float(stall_exec.get("total_simulated_execution_ns", 0.0))
    simulated_execution_time_s = total_sim_exec_ns / 1e9

    mean_token_lat_ns = float(lat.get("mean_ns", 0.0))
    mean_token_latency_ms = mean_token_lat_ns / 1e6
    p50_token_latency_ms = float(lat.get("p50_ns", 0.0)) / 1e6
    p90_token_latency_ms = float(lat.get("p90_ns", 0.0)) / 1e6
    p99_token_latency_ms = float(lat.get("p99_ns", 0.0)) / 1e6

    exposed_gpu_stall_ns = float(stall_exec.get("exposed_gpu_stall_ns", 0.0))
    exposed_stall_ms = exposed_gpu_stall_ns / 1e6

    compute_slack_ns = total_sim_exec_ns - exposed_gpu_stall_ns
    compute_slack_ms = compute_slack_ns / 1e6

    agg_demand_wait_ns = float(stall_exec.get("aggregate_demand_wait_ns", 0.0))
    aggregate_demand_wait_ms = agg_demand_wait_ns / 1e6

    stall_hidden_ns = float(stall_exec.get("stall_hidden_by_prefetch_ns", 0.0))
    stall_hidden_ms = stall_hidden_ns / 1e6

    total_ddr5_bytes = float(traffic.get("total_ddr5_bytes_transferred", 0.0))
    total_ddr5_gb = total_ddr5_bytes / 1e9

    demand_ddr5_bytes = float(traffic.get("demand_bytes_transferred", 0.0))
    demand_ddr5_gb = demand_ddr5_bytes / 1e9

    speculative_ddr5_bytes = float(traffic.get("speculative_bytes_transferred", 0.0))
    speculative_ddr5_gb = speculative_ddr5_bytes / 1e9

    coalesced_demands = int(hbm.get("coalesced_actual_demands", 0))
    routed_demands = int(hbm.get("routed_actual_demands", 0))
    tensor_demands = int(hbm.get("tensor_actual_demands", 0))

    resident_hits = int(hbm.get("coalesced_resident_hits", 0))
    demand_misses = int(hbm.get("coalesced_fresh_demand_misses", 0))
    reloads = int(hbm.get("reloaded_experts_count", 0))

    total_evictions = int(hbm.get("total_evictions", 0))
    demand_evictions = int(hbm.get("demand_resident_evictions", 0))
    unconsumed_prefetch_evictions = int(hbm.get("unconsumed_prefetch_evictions", 0))
    prefetch_induced_evictions = int(hbm.get("prefetch_induced_evictions", 0))

    timely_hits = int(timely_late.get("timely_resident_hits", 0))
    late_attachments = int(timely_late.get("late_inflight_attachments", 0))
    dispatched_prefetches = int(timely_late.get("total_dispatched_prefetches", 0))

    used_prefetches = int(unused.get("prefetches_completed_and_used", 0))
    unused_prefetches = int(unused.get("prefetches_completed_unused", 0))
    evicted_before_demand = int(unused.get("prefetches_evicted_before_demand", 0))
    wasted_speculative_bytes = float(unused.get("total_wasted_speculative_bytes", 0.0))
    wasted_speculative_gb = wasted_speculative_bytes / 1e9

    demand_hit_rate_pct = safe_div(resident_hits, coalesced_demands) * 100.0
    demand_miss_rate_pct = safe_div(demand_misses, coalesced_demands) * 100.0
    prefetch_timely_rate_pct = safe_div(timely_hits, coalesced_demands) * 100.0
    late_attachment_rate_pct = safe_div(late_attachments, coalesced_demands) * 100.0
    prefetch_utilization_pct = safe_div(used_prefetches, dispatched_prefetches) * 100.0

    fa_divergent = int(fa_data.get("divergent_decisions", 0))
    fa_protected = int(fa_data.get("protected_demand_experts_count", 0))
    fa_unconsumed_victims = int(fa_data.get("unconsumed_speculative_victims", 0))

    return {
        "run_id": run_id,
        "suite": suite,
        "batch_size": b,
        "context_p": p,
        "tokens_t": t,
        "lookahead_h": h,
        "budget_k": k,
        "predictor": predictor,
        "prefetch_enabled": prefetch_enabled,
        "eviction_policy": policy,
        "fa_lru_window": 64,
        "simulated_execution_time_s": round(simulated_execution_time_s, 4),
        "mean_token_latency_ms": round(mean_token_latency_ms, 2),
        "p50_token_latency_ms": round(p50_token_latency_ms, 2),
        "p90_token_latency_ms": round(p90_token_latency_ms, 2),
        "p99_token_latency_ms": round(p99_token_latency_ms, 2),
        "exposed_stall_ms": round(exposed_stall_ms, 2),
        "compute_slack_ms": round(compute_slack_ms, 2),
        "aggregate_demand_wait_ms": round(aggregate_demand_wait_ms, 2),
        "stall_hidden_ms": round(stall_hidden_ms, 2),
        "total_ddr5_bytes": int(total_ddr5_bytes),
        "demand_ddr5_bytes": int(demand_ddr5_bytes),
        "speculative_ddr5_bytes": int(speculative_ddr5_bytes),
        "total_ddr5_gb": round(total_ddr5_gb, 2),
        "demand_ddr5_gb": round(demand_ddr5_gb, 2),
        "speculative_ddr5_gb": round(speculative_ddr5_gb, 2),
        "coalesced_demands": coalesced_demands,
        "routed_demands": routed_demands,
        "tensor_demands": tensor_demands,
        "resident_hits": resident_hits,
        "demand_misses": demand_misses,
        "reloads": reloads,
        "total_evictions": total_evictions,
        "demand_evictions": demand_evictions,
        "unconsumed_prefetch_evictions": unconsumed_prefetch_evictions,
        "prefetch_induced_evictions": prefetch_induced_evictions,
        "timely_hits": timely_hits,
        "late_attachments": late_attachments,
        "dispatched_prefetches": dispatched_prefetches,
        "used_prefetches": used_prefetches,
        "unused_prefetches": unused_prefetches,
        "evicted_before_demand": evicted_before_demand,
        "wasted_speculative_gb": round(wasted_speculative_gb, 2),
        "demand_hit_rate_pct": round(demand_hit_rate_pct, 4),
        "demand_miss_rate_pct": round(demand_miss_rate_pct, 4),
        "prefetch_timely_rate_pct": round(prefetch_timely_rate_pct, 4),
        "late_attachment_rate_pct": round(late_attachment_rate_pct, 4),
        "prefetch_utilization_pct": round(prefetch_utilization_pct, 2),
        "fa_lru_divergent_decisions": fa_divergent,
        "fa_lru_protected_demand_experts": fa_protected,
        "fa_lru_unconsumed_speculative_victims": fa_unconsumed_victims,
        "source_raw_path": raw_dir
    }

def main():
    runs_to_parse = [
        # Anchor Suite (B=32, P=2048, T=128, H=1, K=4)
        ("anc_c1", "anchor", 32, 2048, 128, 0, 0, "none", False, "lru",
         os.path.join(POINT6_ROOT, "experiments", "anchor", "raw_results", "anc_c1")),
        ("anc_c3", "anchor", 32, 2048, 128, 1, 4, "CSTP", True, "lru",
         os.path.join(POINT6_ROOT, "experiments", "anchor", "raw_results", "anc_c3")),
        ("anc_c4", "anchor", 32, 2048, 128, 1, 4, "CSTP", True, "fa_lru",
         os.path.join(POINT6_ROOT, "experiments", "anchor", "raw_results", "anc_c4")),

        # Batch Size Suite (B=4, B=14)
        ("b4_c1", "batch_size", 4, 2048, 128, 0, 0, "none", False, "lru",
         os.path.join(POINT6_ROOT, "experiments", "batch_size", "raw_results", "b4_c1")),
        ("b4_c3", "batch_size", 4, 2048, 128, 1, 4, "CSTP", True, "lru",
         os.path.join(POINT6_ROOT, "experiments", "batch_size", "raw_results", "b4_c3")),
        ("b4_c4", "batch_size", 4, 2048, 128, 1, 4, "CSTP", True, "fa_lru",
         os.path.join(POINT6_ROOT, "experiments", "batch_size", "raw_results", "b4_c4")),
        ("b14_c1", "batch_size", 14, 2048, 128, 0, 0, "none", False, "lru",
         os.path.join(POINT6_ROOT, "experiments", "batch_size", "raw_results", "b14_c1")),
        ("b14_c3", "batch_size", 14, 2048, 128, 1, 4, "CSTP", True, "lru",
         os.path.join(POINT6_ROOT, "experiments", "batch_size", "raw_results", "b14_c3")),
        ("b14_c4", "batch_size", 14, 2048, 128, 1, 4, "CSTP", True, "fa_lru",
         os.path.join(POINT6_ROOT, "experiments", "batch_size", "raw_results", "b14_c4")),

        # Context Length Suite (P=128 reused, P=512, P=4096)
        ("p128_c1", "context_length", 32, 128, 128, 0, 0, "none", False, "lru",
         os.path.join(BIG_EXP_ROOT, "results", "runs", "w4_c1_r1")),
        ("p128_c3", "context_length", 32, 128, 128, 1, 4, "CSTP", True, "lru",
         os.path.join(BIG_EXP_ROOT, "results", "runs", "w4_c3_r1")),
        ("p128_c4", "context_length", 32, 128, 128, 1, 4, "CSTP", True, "fa_lru",
         os.path.join(BIG_EXP_ROOT, "results", "runs", "w4_c4_r1")),
        ("p512_c1", "context_length", 32, 512, 128, 0, 0, "none", False, "lru",
         os.path.join(POINT6_ROOT, "experiments", "context_length", "raw_results", "p512_c1")),
        ("p512_c3", "context_length", 32, 512, 128, 1, 4, "CSTP", True, "lru",
         os.path.join(POINT6_ROOT, "experiments", "context_length", "raw_results", "p512_c3")),
        ("p512_c4", "context_length", 32, 512, 128, 1, 4, "CSTP", True, "fa_lru",
         os.path.join(POINT6_ROOT, "experiments", "context_length", "raw_results", "p512_c4")),
        ("p4096_c1", "context_length", 32, 4096, 128, 0, 0, "none", False, "lru",
         os.path.join(POINT6_ROOT, "experiments", "context_length", "raw_results", "p4096_c1")),
        ("p4096_c3", "context_length", 32, 4096, 128, 1, 4, "CSTP", True, "lru",
         os.path.join(POINT6_ROOT, "experiments", "context_length", "raw_results", "p4096_c3")),
        ("p4096_c4", "context_length", 32, 4096, 128, 1, 4, "CSTP", True, "fa_lru",
         os.path.join(POINT6_ROOT, "experiments", "context_length", "raw_results", "p4096_c4")),

        # Lookahead Suite (H=2, H=4)
        ("h2_c3", "lookahead", 32, 2048, 128, 2, 4, "CSTP", True, "lru",
         os.path.join(POINT6_ROOT, "experiments", "lookahead", "raw_results", "h2_c3")),
        ("h2_c4", "lookahead", 32, 2048, 128, 2, 4, "CSTP", True, "fa_lru",
         os.path.join(POINT6_ROOT, "experiments", "lookahead", "raw_results", "h2_c4")),
        ("h4_c3", "lookahead", 32, 2048, 128, 4, 4, "CSTP", True, "lru",
         os.path.join(POINT6_ROOT, "experiments", "lookahead", "raw_results", "h4_c3")),
        ("h4_c4", "lookahead", 32, 2048, 128, 4, 4, "CSTP", True, "fa_lru",
         os.path.join(POINT6_ROOT, "experiments", "lookahead", "raw_results", "h4_c4")),

        # Prefetch Budget Suite (K=2, K=8)
        ("k2_c3", "prefetch_budget", 32, 2048, 128, 1, 2, "CSTP", True, "lru",
         os.path.join(POINT6_ROOT, "experiments", "prefetch_budget", "raw_results", "k2_c3")),
        ("k2_c4", "prefetch_budget", 32, 2048, 128, 1, 2, "CSTP", True, "fa_lru",
         os.path.join(POINT6_ROOT, "experiments", "prefetch_budget", "raw_results", "k2_c4")),
        ("k8_c3", "prefetch_budget", 32, 2048, 128, 1, 8, "CSTP", True, "lru",
         os.path.join(POINT6_ROOT, "experiments", "prefetch_budget", "raw_results", "k8_c3")),
        ("k8_c4", "prefetch_budget", 32, 2048, 128, 1, 8, "CSTP", True, "fa_lru",
         os.path.join(POINT6_ROOT, "experiments", "prefetch_budget", "raw_results", "k8_c4"))
    ]

    records = []
    manifest_records = []
    pollution_records = []
    effectiveness_records = []
    traffic_records = []

    for item in runs_to_parse:
        run_id, suite, b, p, t, h, k, predictor, prefetch_enabled, policy, raw_dir = item
        rec = parse_run(run_id, suite, b, p, t, h, k, predictor, prefetch_enabled, policy, raw_dir)
        records.append(rec)

        manifest_records.append({
            "run_id": run_id,
            "suite": suite,
            "batch_size": b,
            "context_p": p,
            "lookahead_h": h,
            "budget_k": k,
            "predictor": predictor,
            "policy": policy,
            "prefetch_summary_json": os.path.join(raw_dir, "traces", "prefetch_summary.json"),
            "fa_lru_diagnostics_json": os.path.join(raw_dir, "traces", "fa_lru_diagnostics.json"),
            "simulated_execution_time_s": rec["simulated_execution_time_s"],
            "mean_token_latency_ms": rec["mean_token_latency_ms"]
        })

        # Cache pollution records
        pollution_records.append({
            "run_id": run_id,
            "suite": suite,
            "batch_size": b,
            "context_p": p,
            "lookahead_h": h,
            "budget_k": k,
            "policy": policy,
            "total_evictions": rec["total_evictions"],
            "demand_evictions": rec["demand_evictions"],
            "unconsumed_prefetch_evictions": rec["unconsumed_prefetch_evictions"],
            "prefetch_induced_evictions": rec["prefetch_induced_evictions"],
            "unused_prefetches": rec["unused_prefetches"],
            "evicted_before_demand": rec["evicted_before_demand"],
            "wasted_speculative_gb": rec["wasted_speculative_gb"],
            "fa_lru_divergent_decisions": rec["fa_lru_divergent_decisions"],
            "fa_lru_protected_demand_experts": rec["fa_lru_protected_demand_experts"],
            "fa_lru_unconsumed_victims": rec["fa_lru_unconsumed_speculative_victims"]
        })

        # Prefetch effectiveness records
        effectiveness_records.append({
            "run_id": run_id,
            "suite": suite,
            "batch_size": b,
            "context_p": p,
            "lookahead_h": h,
            "budget_k": k,
            "policy": policy,
            "dispatched_prefetches": rec["dispatched_prefetches"],
            "timely_hits": rec["timely_hits"],
            "late_attachments": rec["late_attachments"],
            "used_prefetches": rec["used_prefetches"],
            "unused_prefetches": rec["unused_prefetches"],
            "stall_hidden_ms": rec["stall_hidden_ms"],
            "prefetch_utilization_pct": rec["prefetch_utilization_pct"],
            "prefetch_timely_rate_pct": rec["prefetch_timely_rate_pct"],
            "late_attachment_rate_pct": rec["late_attachment_rate_pct"]
        })

        # DDR5 Traffic composition records
        traffic_records.append({
            "run_id": run_id,
            "suite": suite,
            "batch_size": b,
            "context_p": p,
            "lookahead_h": h,
            "budget_k": k,
            "policy": policy,
            "demand_ddr5_bytes": rec["demand_ddr5_bytes"],
            "speculative_ddr5_bytes": rec["speculative_ddr5_bytes"],
            "total_ddr5_bytes": rec["total_ddr5_bytes"],
            "demand_ddr5_gb": rec["demand_ddr5_gb"],
            "speculative_ddr5_gb": rec["speculative_ddr5_gb"],
            "total_ddr5_gb": rec["total_ddr5_gb"],
            "speculative_traffic_ratio_pct": round(safe_div(rec["speculative_ddr5_bytes"], rec["total_ddr5_bytes"]) * 100.0, 2)
        })

    unified_df = pd.DataFrame(records)

    # Export Unified CSV
    unified_csv = os.path.join(CONSOLIDATED_ROOT, "unified_point6_results.csv")
    unified_df.to_csv(unified_csv, index=False)
    print(f"Exported: {unified_csv} ({len(unified_df)} rows)")

    # Export Source Manifest
    manifest_df = pd.DataFrame(manifest_records)
    manifest_csv = os.path.join(CONSOLIDATED_ROOT, "source_manifest.csv")
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"Exported: {manifest_csv} ({len(manifest_df)} rows)")

    # Export Cache Pollution Analysis
    pollution_df = pd.DataFrame(pollution_records)
    pollution_csv = os.path.join(CONSOLIDATED_ROOT, "cache_pollution_analysis.csv")
    pollution_df.to_csv(pollution_csv, index=False)
    print(f"Exported: {pollution_csv} ({len(pollution_df)} rows)")

    # Export Prefetch Effectiveness
    eff_df = pd.DataFrame(effectiveness_records)
    eff_csv = os.path.join(CONSOLIDATED_ROOT, "prefetch_effectiveness.csv")
    eff_df.to_csv(eff_csv, index=False)
    print(f"Exported: {eff_csv} ({len(eff_df)} rows)")

    # Export DDR5 Traffic Composition
    traffic_df = pd.DataFrame(traffic_records)
    traffic_csv = os.path.join(CONSOLIDATED_ROOT, "ddr5_traffic_composition.csv")
    traffic_df.to_csv(traffic_csv, index=False)
    print(f"Exported: {traffic_csv} ({len(traffic_df)} rows)")

    # Export per-experiment tables:
    # 1. Batch size
    bs_df = unified_df[(unified_df["suite"].isin(["batch_size", "anchor"])) & (unified_df["context_p"] == 2048) & (unified_df["budget_k"].isin([0, 4])) & (unified_df["lookahead_h"].isin([0, 1]))]
    bs_df.to_csv(os.path.join(POINT6_ROOT, "experiments", "batch_size", "batch_size_results.csv"), index=False)
    bs_df.to_csv(os.path.join(CONSOLIDATED_ROOT, "batch_size_comparison.csv"), index=False)
    print(f"Exported batch size results ({len(bs_df)} rows)")

    # 2. Context length
    ctx_df = unified_df[(unified_df["suite"].isin(["context_length", "anchor"])) & (unified_df["batch_size"] == 32) & (unified_df["budget_k"].isin([0, 4])) & (unified_df["lookahead_h"].isin([0, 1]))]
    ctx_df.to_csv(os.path.join(POINT6_ROOT, "experiments", "context_length", "context_length_results.csv"), index=False)
    ctx_df.to_csv(os.path.join(CONSOLIDATED_ROOT, "context_length_comparison.csv"), index=False)
    print(f"Exported context length results ({len(ctx_df)} rows)")

    # 3. Lookahead
    lh_df = unified_df[(unified_df["suite"].isin(["lookahead", "anchor"])) & (unified_df["prefetch_enabled"] == True) & (unified_df["budget_k"] == 4)]
    lh_df.to_csv(os.path.join(POINT6_ROOT, "experiments", "lookahead", "lookahead_results.csv"), index=False)
    lh_df.to_csv(os.path.join(CONSOLIDATED_ROOT, "lookahead_comparison.csv"), index=False)
    print(f"Exported lookahead results ({len(lh_df)} rows)")

    # 4. Prefetch budget
    bg_df = unified_df[(unified_df["suite"].isin(["prefetch_budget", "anchor"])) & (unified_df["prefetch_enabled"] == True) & (unified_df["lookahead_h"] == 1)]
    bg_df.to_csv(os.path.join(POINT6_ROOT, "experiments", "prefetch_budget", "prefetch_budget_results.csv"), index=False)
    bg_df.to_csv(os.path.join(CONSOLIDATED_ROOT, "budget_comparison.csv"), index=False)
    print(f"Exported budget results ({len(bg_df)} rows)")

if __name__ == "__main__":
    main()
