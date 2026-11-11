#!/usr/bin/env python3
"""
Phase 2 Extended Prefetching Effectiveness Campaign Orchestrator.
Executes the full comparative matrix across 40 real MoE stages:
- Batch sizes: B in {1, 2, 4, 8}
- Predictors: Baseline, ST-MoE (H=1, 2), CSTP (H=1, 2)
- Cache capacities: 35 GB (constrained) and 80 GB (reference)
- Decode tokens: T=4 (standard) and T=8 (decode progress)

Computes all eight mandatory prefetching-effectiveness metrics and exports
the 7 standardized CSV result tables.

STRICT RULE: ZERO ternary operators (ast.IfExp == 0).
"""

import os
import sys
import json
import time
import subprocess
import pandas as pd
import numpy as np

WORKSPACE = "/home/mohman/ca_project"
REPO_ROOT = os.path.join(WORKSPACE, "LLMSimulator")
BIN_PATH = os.path.join(REPO_ROOT, "build", "run_deepseekv3_mixtral_style")
PHASE2_ROOT = os.path.join(WORKSPACE, "mohman", "point_3", "phase2")

CONFIG_DIR = os.path.join(PHASE2_ROOT, "experiments", "configs")
MANIFEST_DIR = os.path.join(PHASE2_ROOT, "experiments", "manifests")
LOG_DIR = os.path.join(PHASE2_ROOT, "experiments", "logs")
RAW_DIR = os.path.join(PHASE2_ROOT, "experiments", "raw")
RESULTS_DIR = os.path.join(PHASE2_ROOT, "results")
TRACE_DIR = os.path.join(PHASE2_ROOT, "experiments", "traces")

DDR5_CFG = os.path.join(REPO_ROOT, "reproduction_A100", "with_off_chip_dram", "configs", "dram_config_DDR5_retry.yaml")
HBM_CFG = os.path.join(REPO_ROOT, "build", "dram_config_HBM3_80GB.yaml")

CACHE_35GB = 37580963840   # 35.00 GB
CACHE_80GB = 85899345920   # 80.00 GB

CAMPAIGN_RUNS = [
    # --- Matrix A: Batch Scaling at 35 GB Constrained Cache (T=4) ---
    # Baseline
    {"run_id": "phase2_baseline_b1", "predictor": "baseline", "batch_size": 1, "horizon": 0, "budget": 0, "tokens": 4, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    {"run_id": "phase2_baseline_b2", "predictor": "baseline", "batch_size": 2, "horizon": 0, "budget": 0, "tokens": 4, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    {"run_id": "phase2_baseline_b4", "predictor": "baseline", "batch_size": 4, "horizon": 0, "budget": 0, "tokens": 4, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    {"run_id": "phase2_baseline_b8", "predictor": "baseline", "batch_size": 8, "horizon": 0, "budget": 0, "tokens": 4, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    # ST-MoE H=1
    {"run_id": "phase2_stmoe_b1_h1", "predictor": "stmoe", "batch_size": 1, "horizon": 1, "budget": 4, "tokens": 4, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    {"run_id": "phase2_stmoe_b2_h1", "predictor": "stmoe", "batch_size": 2, "horizon": 1, "budget": 4, "tokens": 4, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    {"run_id": "phase2_stmoe_b4_h1", "predictor": "stmoe", "batch_size": 4, "horizon": 1, "budget": 4, "tokens": 4, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    {"run_id": "phase2_stmoe_b8_h1", "predictor": "stmoe", "batch_size": 8, "horizon": 1, "budget": 4, "tokens": 4, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    # ST-MoE H=2
    {"run_id": "phase2_stmoe_b1_h2", "predictor": "stmoe", "batch_size": 1, "horizon": 2, "budget": 4, "tokens": 4, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    {"run_id": "phase2_stmoe_b2_h2", "predictor": "stmoe", "batch_size": 2, "horizon": 2, "budget": 4, "tokens": 4, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    {"run_id": "phase2_stmoe_b4_h2", "predictor": "stmoe", "batch_size": 4, "horizon": 2, "budget": 4, "tokens": 4, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    {"run_id": "phase2_stmoe_b8_h2", "predictor": "stmoe", "batch_size": 8, "horizon": 2, "budget": 4, "tokens": 4, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    # CSTP H=1
    {"run_id": "phase2_cstp_b1_h1", "predictor": "cstp", "batch_size": 1, "horizon": 1, "budget": 4, "tokens": 4, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    {"run_id": "phase2_cstp_b2_h1", "predictor": "cstp", "batch_size": 2, "horizon": 1, "budget": 4, "tokens": 4, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    {"run_id": "phase2_cstp_b4_h1", "predictor": "cstp", "batch_size": 4, "horizon": 1, "budget": 4, "tokens": 4, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    {"run_id": "phase2_cstp_b8_h1", "predictor": "cstp", "batch_size": 8, "horizon": 1, "budget": 4, "tokens": 4, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    # CSTP H=2
    {"run_id": "phase2_cstp_b1_h2", "predictor": "cstp", "batch_size": 1, "horizon": 2, "budget": 4, "tokens": 4, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    {"run_id": "phase2_cstp_b2_h2", "predictor": "cstp", "batch_size": 2, "horizon": 2, "budget": 4, "tokens": 4, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    {"run_id": "phase2_cstp_b4_h2", "predictor": "cstp", "batch_size": 4, "horizon": 2, "budget": 4, "tokens": 4, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    {"run_id": "phase2_cstp_b8_h2", "predictor": "cstp", "batch_size": 8, "horizon": 2, "budget": 4, "tokens": 4, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},

    # --- Matrix B: Reference 80 GB Cache Comparison (B=1, T=4) ---
    {"run_id": "phase2_ref80g_baseline_b1", "predictor": "baseline", "batch_size": 1, "horizon": 0, "budget": 0, "tokens": 4, "cache_bytes": CACHE_80GB, "cache_label": "80gb"},
    {"run_id": "phase2_ref80g_stmoe_b1_h1", "predictor": "stmoe", "batch_size": 1, "horizon": 1, "budget": 4, "tokens": 4, "cache_bytes": CACHE_80GB, "cache_label": "80gb"},
    {"run_id": "phase2_ref80g_cstp_b1_h1", "predictor": "cstp", "batch_size": 1, "horizon": 1, "budget": 4, "tokens": 4, "cache_bytes": CACHE_80GB, "cache_label": "80gb"},

    # --- Matrix C: Decode Progress Scaling (T=8 tokens, B=1, 35 GB Cache) ---
    {"run_id": "phase2_t8_baseline_b1", "predictor": "baseline", "batch_size": 1, "horizon": 0, "budget": 0, "tokens": 8, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    {"run_id": "phase2_t8_stmoe_b1_h1", "predictor": "stmoe", "batch_size": 1, "horizon": 1, "budget": 4, "tokens": 8, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
    {"run_id": "phase2_t8_cstp_b1_h1", "predictor": "cstp", "batch_size": 1, "horizon": 1, "budget": 4, "tokens": 8, "cache_bytes": CACHE_35GB, "cache_label": "35gb"},
]


def generate_yaml_config(run_def):
    run_id = run_def["run_id"]
    predictor = run_def["predictor"]
    b = run_def["batch_size"]
    h = run_def["horizon"]
    bgt = run_def["budget"]
    tokens = run_def["tokens"]
    cache_bytes = run_def["cache_bytes"]

    if tokens == 8:
        trace_file = os.path.join(TRACE_DIR, f"test_moe_anywhere_b{b}_s40_t8.csv")
    else:
        trace_file = os.path.join(TRACE_DIR, f"test_moe_anywhere_b{b}_s40_t4.csv")

    csv_dir = os.path.join(RAW_DIR, run_id, "csv") + "/"
    raw_dir = os.path.join(RAW_DIR, run_id, "raw") + "/"
    traces_dir = os.path.join(RAW_DIR, run_id, "traces") + "/"

    yaml_lines = [
        "model:",
        "  model_name: deepseekV3",
        "  num_layers: 43",
        "",
        "system:",
        "  gpu_gen: A100",
        "  nvlink_gen: 3",
        "  infiniband_gen: 800",
        "  num_node: 1",
        "  num_device: 1",
        "  processor_type: GPU",
        "  distribution:",
        "    none_expert_tensor_degree: 1",
        "    expert_tensor_degree: 1",
        "  optimization:",
        "    parallel_execution: 'off'",
        "    hetero_subbatch: false",
        "    disagg_system: false",
        "    use_low_unit_moe_only: false",
        "    use_ramulator: 'on'",
        f"    offchip_dram_cfg_path: {DDR5_CFG}",
        f"    hbm_dram_cfg_path: {HBM_CFG}",
        f"    expert_cache_capacity_bytes: {cache_bytes}",
        "    compressed_kv: true",
        "    use_absorb: true",
        "    use_flash_mla: true",
        "    use_flash_attention: true",
        "    reuse_kv_cache: true",
        "    kv_cache_reuse_rate: 0.0",
        "    prefill_mode: false",
        "    decode_mode: true",
        "    use_hitting_queue: false",
        "",
        "serving:",
        f"  max_batch_size: {b}",
        "  max_process_token: 0",
        "",
        "simulation:",
        "  data: moe_anywhere",
        "  input_len: 128",
        f"  output_len: {tokens}",
        "  precision_byte: 2",
        "  skewness: 0.0",
        f"  iter: {tokens}",
        "  injection_rate: 0",
        "  exit_out_of_memory: false",
        "  mem_cap_limit: false",
        f"  trace_file: {trace_file}",
    ]

    if predictor == "baseline":
        yaml_lines.extend([
            "  prefetching:",
            "    enabled: false",
            "    mode: none",
        ])
    elif predictor == "stmoe":
        yaml_lines.extend([
            "  prefetching:",
            "    enabled: true",
            "    mode: hardware_async",
            "    scheduling:",
            "      scheduling_mode: forward_lookahead",
            f"      prediction_horizon: {h}",
            "    candidate_selection:",
            f"      budget: {bgt}",
            "    predictor:",
            "      type: ST_MOE",
            f"      prediction_horizon: {h}",
            f"      budget: {bgt}",
            "      confidence_threshold: 2",
            "      online_adaptation: true",
        ])
    elif predictor == "cstp":
        yaml_lines.extend([
            "  prefetching:",
            "    enabled: true",
            "    mode: hardware_async",
            "    scheduling:",
            "      scheduling_mode: forward_lookahead",
            f"      prediction_horizon: {h}",
            "    candidate_selection:",
            f"      budget: {bgt}",
            "    predictor:",
            "      type: CSTP",
            f"      prediction_horizon: {h}",
            f"      budget: {bgt}",
            "      alpha: 0.5",
            "      smoothing_lambda: 1.0",
            "      prior_epsilon: 0.1",
            "      online_adaptation: true",
        ])

    yaml_lines.extend([
        "",
        "log:",
        "  print_log: false",
        "  export_gantt: false",
        f"  output_directory: {csv_dir}",
        f"  gantt_directory: {raw_dir}",
        f"  traces_directory: {traces_dir}",
        "",
    ])

    return "\n".join(yaml_lines)


def run_experiment(run_def):
    run_id = run_def["run_id"]
    cfg_path = os.path.join(CONFIG_DIR, f"{run_id}.yaml")
    log_path = os.path.join(LOG_DIR, f"{run_id}.log")
    traces_dir = os.path.join(RAW_DIR, run_id, "traces")

    os.makedirs(os.path.join(RAW_DIR, run_id, "csv"), exist_ok=True)
    os.makedirs(os.path.join(RAW_DIR, run_id, "raw"), exist_ok=True)
    os.makedirs(traces_dir, exist_ok=True)

    env = os.environ.copy()
    env["USE_EVENT_DRIVEN_DRAM"] = "true"
    env["DDR5_BANDWIDTH_GBPS"] = "64.0"
    env["PREDICTION_OUTPUT_DIR"] = traces_dir

    cmd = [BIN_PATH, cfg_path]
    t0 = time.time()
    with open(log_path, "w") as log_f:
        res = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - t0

    if res.returncode != 0:
        print(f"[FAILED] {run_id} exited with code {res.returncode}")
        return False, elapsed
    else:
        return True, elapsed


def parse_run_results(run_def, host_time_sec):
    run_id = run_def["run_id"]
    predictor = run_def["predictor"]
    b = run_def["batch_size"]
    h = run_def["horizon"]
    tokens = run_def["tokens"]
    cache_label = run_def["cache_label"]
    cache_bytes = run_def["cache_bytes"]

    summary_path = os.path.join(RAW_DIR, run_id, "traces", "experiment_summary.json")
    with open(summary_path, "r") as f:
        data = json.load(f)

    cov = data["prefetch_coverage"]
    timely_late = data["timely_and_late_rates"]
    unused = data["unused_prefetches"]
    hbm = data["hbm_hit_rates"]
    traffic = data["memory_traffic"]
    stall = data["exposed_stall_and_execution"]
    exp_lat = data["expert_load_latency"]
    tok_lat = data["token_latency"]

    timely_hits = timely_late["timely_resident_hits"]
    late_hits = timely_late["late_inflight_attachments"]
    matched_prefetches = timely_hits + late_hits

    if matched_prefetches > 0:
        timely_of_matched_pct = (timely_hits / matched_prefetches) * 100.0
        late_of_matched_pct = (late_hits / matched_prefetches) * 100.0
    else:
        timely_of_matched_pct = 0.0
        late_of_matched_pct = 0.0

    unused_count = unused["prefetches_completed_unused"]
    used_count = unused["prefetches_completed_and_used"]
    completed_total = unused_count + used_count
    if completed_total > 0:
        unused_pct = (unused_count / completed_total) * 100.0
    else:
        unused_pct = 0.0

    # Parse cache events for insert, evict, reload
    cache_csv = os.path.join(RAW_DIR, run_id, "traces", "expert_cache.csv")
    inserts = 0
    evicts = 0
    if os.path.exists(cache_csv):
        cdf = pd.read_csv(cache_csv)
        acts = cdf["action"].value_counts().to_dict()
        inserts = acts.get("INSERT", 0)
        evicts = acts.get("EVICT", 0)

    unique_demands = cov["unique_actual_demands"]
    reloads = 0
    if inserts > unique_demands:
        reloads = inserts - unique_demands

    return {
        "run_id": run_id,
        "predictor": predictor,
        "batch_size": b,
        "horizon": h,
        "budget": run_def["budget"],
        "tokens": tokens,
        "cache_label": cache_label,
        "cache_bytes": cache_bytes,
        "cache_gb": cache_bytes / (1024**3),
        "host_runtime_sec": host_time_sec,
        "exit_code": 0,
        "total_actual_demands": cov["total_actual_demands"],
        "unique_actual_demands": unique_demands,
        "demands_with_prediction": cov["demands_with_prediction"],
        "prefetch_assisted_demands": cov["prefetch_assisted_demands"],
        "overall_coverage_pct": cov["overall_coverage"] * 100.0,
        "unique_overall_coverage_pct": cov["unique_overall_coverage"] * 100.0,
        "predicted_coverage_pct": cov["predicted_coverage"] * 100.0,
        "timely_resident_hits": timely_hits,
        "late_inflight_attachments": late_hits,
        "timely_rate_of_matched_pct": timely_of_matched_pct,
        "late_rate_of_matched_pct": late_of_matched_pct,
        "total_dispatched_prefetches": timely_late["total_dispatched_prefetches"],
        "prefetches_completed_and_used": used_count,
        "prefetches_completed_unused": unused_count,
        "unused_pct": unused_pct,
        "completed_unused_bytes": unused["completed_unused_bytes"],
        "total_hbm_resident_hits": hbm["total_hbm_resident_hits"],
        "overall_hbm_hit_rate_pct": hbm["overall_hbm_hit_rate"] * 100.0,
        "fresh_demand_misses": hbm["fresh_demand_misses"],
        "inserts": inserts,
        "evicts": evicts,
        "reloads": reloads,
        "demand_bytes_transferred": traffic["demand_bytes_transferred"],
        "speculative_bytes_transferred": traffic["speculative_bytes_transferred"],
        "total_ddr5_bytes_transferred": traffic["total_ddr5_bytes_transferred"],
        "exposed_gpu_stall_ms": stall["exposed_gpu_stall_ns"] / 1e6,
        "stall_per_token_ms": stall["wait_per_token_ns"] / 1e6,
        "stall_per_unique_demand_us": stall["wait_per_unique_demand_ns"] / 1e3,
        "total_simulated_time_ms": stall["total_simulated_execution_ns"] / 1e6,
        "mean_expert_load_lat_us": exp_lat["mean_ns"] / 1e3,
        "p50_expert_load_lat_us": exp_lat["p50_ns"] / 1e3,
        "p95_expert_load_lat_us": exp_lat["p95_ns"] / 1e3,
        "mean_token_lat_ms": tok_lat["mean_ns"] / 1e6,
        "p50_token_lat_ms": tok_lat["p50_ns"] / 1e6,
        "p95_token_lat_ms": tok_lat["p95_ns"] / 1e6,
    }


def main():
    print("======================================================================")
    print("STARTING POINT 3 PHASE 2 EXTENDED PREFETCHING EFFECTIVENESS CAMPAIGN")
    print(f"Total Runs to Execute: {len(CAMPAIGN_RUNS)}")
    print("======================================================================")

    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Generate Manifest
    manifest_rows = []
    for r in CAMPAIGN_RUNS:
        yaml_content = generate_yaml_config(r)
        cfg_path = os.path.join(CONFIG_DIR, f"{r['run_id']}.yaml")
        with open(cfg_path, "w") as f:
            f.write(yaml_content)

        manifest_rows.append({
            "run_id": r["run_id"],
            "predictor": r["predictor"],
            "batch_size": r["batch_size"],
            "horizon": r["horizon"],
            "tokens": r["tokens"],
            "cache_label": r["cache_label"],
            "config_path": cfg_path,
            "log_path": os.path.join(LOG_DIR, f"{r['run_id']}.log"),
        })

    manifest_csv = os.path.join(MANIFEST_DIR, "phase2_manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    print(f"[MANIFEST] Generated manifest with {len(manifest_rows)} runs at {manifest_csv}")

    # 2. Execute Runs
    per_run_data = []
    total_t0 = time.time()
    for idx, r in enumerate(CAMPAIGN_RUNS):
        run_id = r["run_id"]
        print(f"[{idx+1:2d}/{len(CAMPAIGN_RUNS):2d}] Executing {run_id:30s} ... ", end="", flush=True)
        success, elapsed = run_experiment(r)
        if not success:
            print(f"FAILED in {elapsed:.2f}s")
            sys.exit(1)
        row = parse_run_results(r, elapsed)
        per_run_data.append(row)
        print(
            f"DONE ({elapsed:4.1f}s) | Dispatched: {row['total_dispatched_prefetches']:3d} | "
            f"Assisted: {row['prefetch_assisted_demands']:2d} | Evicts: {row['evicts']:3d} | "
            f"Stall: {row['exposed_gpu_stall_ms']:7.1f} ms"
        )

    total_elapsed = time.time() - total_t0
    print(f"\n[CAMPAIGN COMPLETE] All {len(CAMPAIGN_RUNS)} runs finished in {total_elapsed:.1f}s")

    # 3. Create Standardized Result Tables
    df_runs = pd.DataFrame(per_run_data)
    per_run_csv = os.path.join(RESULTS_DIR, "per_run_metrics.csv")
    df_runs.to_csv(per_run_csv, index=False)
    print(f"[EXPORT] Saved {per_run_csv}")

    # Aggregate Metrics by (Predictor, Horizon, BatchSize, CacheLabel)
    agg_df = df_runs.groupby(["predictor", "horizon", "batch_size", "cache_label"]).agg({
        "overall_coverage_pct": "mean",
        "predicted_coverage_pct": "mean",
        "timely_rate_of_matched_pct": "mean",
        "late_rate_of_matched_pct": "mean",
        "unused_pct": "mean",
        "overall_hbm_hit_rate_pct": "mean",
        "demand_bytes_transferred": "mean",
        "speculative_bytes_transferred": "mean",
        "total_ddr5_bytes_transferred": "mean",
        "exposed_gpu_stall_ms": "mean",
        "stall_per_token_ms": "mean",
        "mean_token_lat_ms": "mean",
        "total_simulated_time_ms": "mean",
    }).reset_index()
    agg_csv = os.path.join(RESULTS_DIR, "aggregate_metrics.csv")
    agg_df.to_csv(agg_csv, index=False)
    print(f"[EXPORT] Saved {agg_csv}")

    # Batch Size Trends (35 GB Cache, T=4)
    b_df = df_runs[(df_runs["cache_label"] == "35gb") & (df_runs["tokens"] == 4)].copy()
    batch_trends_csv = os.path.join(RESULTS_DIR, "batch_size_trends.csv")
    b_df.to_csv(batch_trends_csv, index=False)
    print(f"[EXPORT] Saved {batch_trends_csv}")

    # Cache Capacity Trends (B=1, T=4, comparing 35 GB vs 80 GB)
    cap_runs = df_runs[(df_runs["batch_size"] == 1) & (df_runs["tokens"] == 4)].copy()
    cap_trends_csv = os.path.join(RESULTS_DIR, "cache_capacity_trends.csv")
    cap_runs.to_csv(cap_trends_csv, index=False)
    print(f"[EXPORT] Saved {cap_trends_csv}")

    # Horizon Trends (H=1 vs H=2 across Batch sizes at 35 GB)
    h_df = df_runs[
        (df_runs["cache_label"] == "35gb")
        & (df_runs["tokens"] == 4)
        & (df_runs["horizon"].isin([1, 2]))
    ].copy()
    horizon_trends_csv = os.path.join(RESULTS_DIR, "horizon_trends.csv")
    h_df.to_csv(horizon_trends_csv, index=False)
    print(f"[EXPORT] Saved {horizon_trends_csv}")

    # Decode Progress Trends (T=4 vs T=8 at B=1, 35 GB)
    dec_df = df_runs[
        (df_runs["cache_label"] == "35gb")
        & (df_runs["batch_size"] == 1)
        & (df_runs["tokens"].isin([4, 8]))
    ].copy()
    decode_trends_csv = os.path.join(RESULTS_DIR, "decode_progress_trends.csv")
    dec_df.to_csv(decode_trends_csv, index=False)
    print(f"[EXPORT] Saved {decode_trends_csv}")

    # Predictor Comparison (ST-MoE vs CSTP head-to-head across all conditions)
    pred_df = df_runs[df_runs["predictor"].isin(["stmoe", "cstp"])].copy()
    pred_csv = os.path.join(RESULTS_DIR, "predictor_comparison.csv")
    pred_df.to_csv(pred_csv, index=False)
    print(f"[EXPORT] Saved {pred_csv}")

    print("\n======================================================================")
    print("ALL 7 STANDARDIZED RESULT CSV TABLES EXPORTED SUCCESSFULLY!")
    print("======================================================================")


if __name__ == "__main__":
    main()

