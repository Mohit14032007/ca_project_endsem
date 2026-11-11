#!/usr/bin/env python3
"""
Point 3 Campaign Execution Script
Executes the complete 20-run prefetching effectiveness matrix on MoE-Anywhere held-out test traces.
Enforces zero ternary operators.
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
POINT3_ROOT = os.path.join(WORKSPACE, "mohman", "point_3")
BIN_PATH = os.path.join(REPO_ROOT, "build", "run_deepseekv3_mixtral_style")

CONFIG_DIR = os.path.join(POINT3_ROOT, "experiments", "configs")
MANIFEST_DIR = os.path.join(POINT3_ROOT, "experiments", "manifests")
LOG_DIR = os.path.join(POINT3_ROOT, "experiments", "logs")
RAW_DIR = os.path.join(POINT3_ROOT, "experiments", "raw")
RESULTS_DIR = os.path.join(POINT3_ROOT, "results")
PLOTS_DIR = os.path.join(POINT3_ROOT, "plots")
TRACE_DIR = os.path.join(POINT3_ROOT, "sensitivity", "traces", "batch")

DDR5_CFG = os.path.join(REPO_ROOT, "reproduction_A100", "with_off_chip_dram", "configs", "dram_config_DDR5_retry.yaml")
HBM_CFG = os.path.join(REPO_ROOT, "build", "dram_config_HBM3_80GB.yaml")

RUN_DEFINITIONS = [
    # 4 Baseline runs
    {"run_id": "baseline_b1", "predictor": "baseline", "batch_size": 1, "horizon": 0},
    {"run_id": "baseline_b2", "predictor": "baseline", "batch_size": 2, "horizon": 0},
    {"run_id": "baseline_b4", "predictor": "baseline", "batch_size": 4, "horizon": 0},
    {"run_id": "baseline_b8", "predictor": "baseline", "batch_size": 8, "horizon": 0},
    # 8 ST-MoE runs (H=1, H=2)
    {"run_id": "stmoe_b1_h1", "predictor": "stmoe", "batch_size": 1, "horizon": 1},
    {"run_id": "stmoe_b2_h1", "predictor": "stmoe", "batch_size": 2, "horizon": 1},
    {"run_id": "stmoe_b4_h1", "predictor": "stmoe", "batch_size": 4, "horizon": 1},
    {"run_id": "stmoe_b8_h1", "predictor": "stmoe", "batch_size": 8, "horizon": 1},
    {"run_id": "stmoe_b1_h2", "predictor": "stmoe", "batch_size": 1, "horizon": 2},
    {"run_id": "stmoe_b2_h2", "predictor": "stmoe", "batch_size": 2, "horizon": 2},
    {"run_id": "stmoe_b4_h2", "predictor": "stmoe", "batch_size": 4, "horizon": 2},
    {"run_id": "stmoe_b8_h2", "predictor": "stmoe", "batch_size": 8, "horizon": 2},
    # 8 CSTP runs (H=1, H=2)
    {"run_id": "cstp_b1_h1", "predictor": "cstp", "batch_size": 1, "horizon": 1},
    {"run_id": "cstp_b2_h1", "predictor": "cstp", "batch_size": 2, "horizon": 1},
    {"run_id": "cstp_b4_h1", "predictor": "cstp", "batch_size": 4, "horizon": 1},
    {"run_id": "cstp_b8_h1", "predictor": "cstp", "batch_size": 8, "horizon": 1},
    {"run_id": "cstp_b1_h2", "predictor": "cstp", "batch_size": 1, "horizon": 2},
    {"run_id": "cstp_b2_h2", "predictor": "cstp", "batch_size": 2, "horizon": 2},
    {"run_id": "cstp_b4_h2", "predictor": "cstp", "batch_size": 4, "horizon": 2},
    {"run_id": "cstp_b8_h2", "predictor": "cstp", "batch_size": 8, "horizon": 2},
]


def generate_yaml_config(run_def):
    run_id = run_def["run_id"]
    predictor = run_def["predictor"]
    b = run_def["batch_size"]
    h = run_def["horizon"]

    trace_file = os.path.join(TRACE_DIR, f"test_moe_anywhere_b{b}_t4.csv")
    csv_dir = os.path.join(RAW_DIR, run_id, "csv") + "/"
    raw_dir = os.path.join(RAW_DIR, run_id, "raw") + "/"
    traces_dir = os.path.join(RAW_DIR, run_id, "traces") + "/"

    yaml_lines = [
        "model:",
        "  model_name: deepseekV3",
        "  num_layers: 6",
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
        "    expert_cache_capacity_bytes: 85899345920",
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
        "  output_len: 4",
        "  precision_byte: 2",
        "  skewness: 0.0",
        "  iter: 4",
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
            "      budget: 4",
            "    predictor:",
            "      type: ST_MOE",
            f"      prediction_horizon: {h}",
            "      budget: 4",
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
            "      budget: 4",
            "    predictor:",
            "      type: CSTP",
            f"      prediction_horizon: {h}",
            "      budget: 4",
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
        print(f"[SUCCESS] {run_id} completed in {elapsed:.2f}s")
        return True, elapsed


def parse_run_results(run_def, host_time_sec):
    run_id = run_def["run_id"]
    predictor = run_def["predictor"]
    b = run_def["batch_size"]
    h = run_def["horizon"]
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

    # Calculate timely and late rates of matched prefetches
    timely_hits = timely_late["timely_resident_hits"]
    late_hits = timely_late["late_inflight_attachments"]
    matched_prefetches = timely_hits + late_hits

    if matched_prefetches > 0:
        timely_of_matched_pct = (timely_hits / matched_prefetches) * 100.0
        late_of_matched_pct = (late_hits / matched_prefetches) * 100.0
    else:
        timely_of_matched_pct = 0.0
        late_of_matched_pct = 0.0

    # Unused percentage of completed prefetches
    unused_count = unused["prefetches_completed_unused"]
    used_count = unused["prefetches_completed_and_used"]
    completed_total = unused_count + used_count
    if completed_total > 0:
        unused_pct = (unused_count / completed_total) * 100.0
    else:
        unused_pct = 0.0

    # Budget
    if predictor == "baseline":
        budget = 0
    else:
        budget = 4

    record = {
        "run_id": run_id,
        "predictor": predictor,
        "batch_size": b,
        "horizon": h,
        "budget": budget,
        "host_runtime_sec": host_time_sec,
        # Metric 1: Coverage
        "total_actual_demands": cov["total_actual_demands"],
        "unique_actual_demands": cov["unique_actual_demands"],
        "demands_with_prediction": cov["demands_with_prediction"],
        "prefetch_assisted_demands": cov["prefetch_assisted_demands"],
        "unique_prefetch_assisted_demands": cov["unique_prefetch_assisted_demands"],
        "coverage_pct": cov["overall_coverage"] * 100.0,
        "unique_coverage_pct": cov["unique_overall_coverage"] * 100.0,
        # Metric 2: Timely Rate
        "timely_resident_hits": timely_hits,
        "unique_timely_resident_hits": timely_late["unique_timely_resident_hits"],
        "timely_rate_of_matched_pct": timely_of_matched_pct,
        "timely_rate_of_demands_pct": timely_late["timely_rate_of_demands"] * 100.0,
        # Metric 3: Late Rate
        "late_inflight_attachments": late_hits,
        "unique_late_inflight_attachments": timely_late["unique_late_inflight_attachments"],
        "late_rate_of_matched_pct": late_of_matched_pct,
        "late_rate_of_demands_pct": timely_late["late_rate_of_demands"] * 100.0,
        # Metric 4: Unused Prefetches
        "prefetches_completed_unused": unused_count,
        "prefetches_completed_and_used": used_count,
        "unused_prefetch_pct": unused_pct,
        "completed_unused_bytes": unused["completed_unused_bytes"],
        "completed_unused_mb": unused["completed_unused_bytes"] / (1024.0 * 1024.0),
        # Metric 5: HBM Hit Rate
        "total_hbm_resident_hits": hbm["total_hbm_resident_hits"],
        "unique_resident_hits": hbm["unique_resident_hits"],
        "fresh_demand_misses": hbm["fresh_demand_misses"],
        "unique_fresh_demand_misses": hbm["unique_fresh_demand_misses"],
        "overall_hbm_hit_rate_pct": hbm["overall_hbm_hit_rate"] * 100.0,
        "unique_hbm_hit_rate_pct": hbm["unique_overall_hbm_hit_rate"] * 100.0,
        # Metric 6: Memory Traffic
        "speculative_bytes": traffic["speculative_bytes_transferred"],
        "demand_bytes": traffic["demand_bytes_transferred"],
        "total_ddr5_bytes": traffic["total_ddr5_bytes_transferred"],
        "total_ddr5_mb": traffic["total_ddr5_bytes_transferred"] / (1024.0 * 1024.0),
        "additional_bytes_vs_baseline": 0,  # Computed in post-processing
        "additional_traffic_pct": 0.0,      # Computed in post-processing
        # Metric 7: Exposed Memory Stall
        "exposed_gpu_stall_ns": stall["exposed_gpu_stall_ns"],
        "exposed_gpu_stall_ms": stall["exposed_gpu_stall_ns"] / 1e6,
        "stall_hidden_by_prefetch_ns": stall["stall_hidden_by_prefetch_ns"],
        "wait_per_token_ms": stall["wait_per_token_ns"] / 1e6,
        "total_simulated_execution_ms": stall["total_simulated_execution_ns"] / 1e6,
        # Metric 8: Latency
        "expert_load_mean_us": exp_lat["mean_ns"] / 1e3,
        "expert_load_p50_us": exp_lat["p50_ns"] / 1e3,
        "expert_load_p95_us": exp_lat["p95_ns"] / 1e3,
        "token_latency_mean_ms": tok_lat["mean_ns"] / 1e6,
        "token_latency_p50_ms": tok_lat["p50_ns"] / 1e6,
        "token_latency_p95_ms": tok_lat["p95_ns"] / 1e6,
        "simulated_tokens_per_sec": tok_lat["simulated_tokens_per_second"],
    }
    return record


def extract_learning_progress(run_def):
    run_id = run_def["run_id"]
    predictor = run_def["predictor"]
    b = run_def["batch_size"]
    h = run_def["horizon"]
    traces_dir = os.path.join(RAW_DIR, run_id, "traces")

    tok_path = os.path.join(traces_dir, "token_latency.csv")
    dem_path = os.path.join(traces_dir, "expert_demand_events.csv")
    pred_path = os.path.join(traces_dir, "prediction_events.csv")

    tok_df = pd.read_csv(tok_path)
    dem_df = pd.read_csv(dem_path)

    has_preds = os.path.exists(pred_path)
    pred_df = None
    if has_preds:
        try:
            pred_df = pd.read_csv(pred_path)
        except Exception:
            pred_df = None

    steps = sorted(tok_df["gen_step"].unique())
    rows = []
    for step in steps:
        t_sub = tok_df[tok_df["gen_step"] == step]
        token_id = int(t_sub["token_id"].iloc[0])
        tok_duration_ms = float(t_sub["token_duration_ns"].mean()) / 1e6

        d_sub = dem_df[dem_df["token_id"] == token_id]
        total_demands = len(d_sub)
        res_hits = len(d_sub[d_sub["residency_status"] == "RESIDENT_HIT"])
        dem_misses = len(d_sub[d_sub["residency_status"] == "DEMAND_MISS"])
        total_stall_ms = float(d_sub["ddr5_wait_ns"].sum()) / 1e6

        if total_demands > 0:
            hbm_hit_pct = (res_hits / total_demands) * 100.0
        else:
            hbm_hit_pct = 0.0

        p_count = 0
        hits_count = 0
        prec_pct = 0.0
        if pred_df is not None:
            p_sub = pred_df[pred_df["token_id"] == token_id]
            p_count = len(p_sub)
            if p_count > 0:
                hits_count = int(p_sub["is_hit"].sum())
                prec_pct = (hits_count / p_count) * 100.0

        rows.append({
            "run_id": run_id,
            "predictor": predictor,
            "batch_size": b,
            "horizon": h,
            "gen_step": step,
            "token_id": token_id,
            "total_demands": total_demands,
            "resident_hits": res_hits,
            "demand_misses": dem_misses,
            "hbm_hit_pct": hbm_hit_pct,
            "token_stall_ms": total_stall_ms,
            "token_duration_ms": tok_duration_ms,
            "predictions_count": p_count,
            "hits_count": hits_count,
            "precision_pct": prec_pct,
        })
    return rows


def main():
    print("=" * 80)
    print("POINT 3: PREFETCHING EFFECTIVENESS EXPERIMENTAL CAMPAIGN (20 RUNS)")
    print("=" * 80)

    # 1. Generate Configurations
    print("\n[Phase 1/4] Generating 20 Experiment YAML Configurations...")
    manifest_rows = []
    for r in RUN_DEFINITIONS:
        run_id = r["run_id"]
        yaml_content = generate_yaml_config(r)
        cfg_path = os.path.join(CONFIG_DIR, f"{run_id}.yaml")
        with open(cfg_path, "w") as f:
            f.write(yaml_content)
        manifest_rows.append({
            "run_id": run_id,
            "predictor": r["predictor"],
            "batch_size": r["batch_size"],
            "horizon": r["horizon"],
            "config_path": cfg_path,
            "trace_file": os.path.join(TRACE_DIR, f"test_moe_anywhere_b{r['batch_size']}_t4.csv"),
            "log_path": os.path.join(LOG_DIR, f"{run_id}.log"),
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = os.path.join(MANIFEST_DIR, "experiment_manifest.csv")
    manifest_json = os.path.join(MANIFEST_DIR, "experiment_manifest.json")
    manifest_df.to_csv(manifest_csv, index=False)
    with open(manifest_json, "w") as f:
        json.dump(manifest_rows, f, indent=2)
    print(f"Generated 20 YAML configs in: {CONFIG_DIR}")
    print(f"Manifest saved to: {manifest_csv}")

    # 2. Execute 20 Runs
    print("\n[Phase 2/4] Executing 20 Experiments via LLMSimulator...")
    execution_results = {}
    for idx, r in enumerate(RUN_DEFINITIONS):
        run_id = r["run_id"]
        print(f"[{idx+1:02d}/20] Running: {run_id} (Pred={r['predictor']}, B={r['batch_size']}, H={r['horizon']})...", end=" ", flush=True)
        success, elapsed = run_experiment(r)
        assert success, f"Simulation failure on run {run_id}"
        execution_results[run_id] = elapsed

    # 3. Ingest and Process Metrics
    print("\n[Phase 3/4] Processing Metrics and Computing Baseline Differentials...")
    records = []
    all_learning_rows = []
    for r in RUN_DEFINITIONS:
        run_id = r["run_id"]
        rec = parse_run_results(r, execution_results[run_id])
        records.append(rec)
        l_rows = extract_learning_progress(r)
        all_learning_rows.extend(l_rows)

    df_runs = pd.DataFrame(records)

    # Compute additional memory traffic vs baseline for matching batch size
    base_traffic_map = {}
    for b in [1, 2, 4, 8]:
        base_row = df_runs[(df_runs["predictor"] == "baseline") & (df_runs["batch_size"] == b)]
        base_bytes = int(base_row["total_ddr5_bytes"].iloc[0])
        base_traffic_map[b] = base_bytes

    for idx, row in df_runs.iterrows():
        b = int(row["batch_size"])
        b_base = base_traffic_map[b]
        tot = int(row["total_ddr5_bytes"])
        diff = tot - b_base
        df_runs.at[idx, "additional_bytes_vs_baseline"] = diff
        if b_base > 0:
            df_runs.at[idx, "additional_traffic_pct"] = (float(diff) / float(b_base)) * 100.0
        else:
            df_runs.at[idx, "additional_traffic_pct"] = 0.0

    # Save per_run_metrics.csv
    per_run_csv = os.path.join(RESULTS_DIR, "per_run_metrics.csv")
    df_runs.to_csv(per_run_csv, index=False)
    print(f"Saved: {per_run_csv} ({len(df_runs)} rows)")

    # Save learning_progress_trends.csv
    df_learning = pd.DataFrame(all_learning_rows)
    learning_csv = os.path.join(RESULTS_DIR, "learning_progress_trends.csv")
    df_learning.to_csv(learning_csv, index=False)
    print(f"Saved: {learning_csv} ({len(df_learning)} rows)")

    # 4. Generate Aggregate, Batch Size Trends, and Horizon Trends CSVs
    print("\n[Phase 4/4] Creating Aggregate and Comparative Analysis CSVs...")

    # Aggregate by (predictor, horizon)
    agg_rows = []
    groups = df_runs.groupby(["predictor", "horizon"])
    for (pred, h), g in groups:
        agg_rows.append({
            "predictor": pred,
            "horizon": h,
            "num_runs": len(g),
            "mean_coverage_pct": g["coverage_pct"].mean(),
            "mean_timely_rate_pct": g["timely_rate_of_matched_pct"].mean(),
            "mean_late_rate_pct": g["late_rate_of_matched_pct"].mean(),
            "total_unused_prefetches": g["prefetches_completed_unused"].sum(),
            "mean_hbm_hit_rate_pct": g["overall_hbm_hit_rate_pct"].mean(),
            "mean_additional_traffic_pct": g["additional_traffic_pct"].mean(),
            "mean_exposed_stall_ms": g["exposed_gpu_stall_ms"].mean(),
            "mean_token_latency_ms": g["token_latency_mean_ms"].mean(),
            "mean_expert_load_us": g["expert_load_mean_us"].mean(),
        })
    df_agg = pd.DataFrame(agg_rows)
    agg_csv = os.path.join(RESULTS_DIR, "aggregate_metrics.csv")
    df_agg.to_csv(agg_csv, index=False)
    print(f"Saved: {agg_csv}")

    # Batch size trends CSV
    batch_trends_csv = os.path.join(RESULTS_DIR, "batch_size_trends.csv")
    batch_cols = [
        "batch_size", "predictor", "horizon",
        "coverage_pct", "timely_rate_of_matched_pct", "late_rate_of_matched_pct",
        "prefetches_completed_unused", "overall_hbm_hit_rate_pct",
        "total_ddr5_mb", "additional_traffic_pct", "exposed_gpu_stall_ms",
        "wait_per_token_ms", "token_latency_mean_ms", "token_latency_p95_ms"
    ]
    df_runs.sort_values(by=["batch_size", "predictor", "horizon"])[batch_cols].to_csv(batch_trends_csv, index=False)
    print(f"Saved: {batch_trends_csv}")

    # Horizon trends CSV (comparing H=1 vs H=2 for ST-MoE and CSTP)
    horizon_rows = []
    for pred in ["stmoe", "cstp"]:
        for b in [1, 2, 4, 8]:
            r_h1 = df_runs[(df_runs["predictor"] == pred) & (df_runs["batch_size"] == b) & (df_runs["horizon"] == 1)].iloc[0]
            r_h2 = df_runs[(df_runs["predictor"] == pred) & (df_runs["batch_size"] == b) & (df_runs["horizon"] == 2)].iloc[0]
            horizon_rows.append({
                "predictor": pred,
                "batch_size": b,
                "h1_coverage_pct": r_h1["coverage_pct"],
                "h2_coverage_pct": r_h2["coverage_pct"],
                "h1_timely_pct": r_h1["timely_rate_of_matched_pct"],
                "h2_timely_pct": r_h2["timely_rate_of_matched_pct"],
                "h1_unused_count": r_h1["prefetches_completed_unused"],
                "h2_unused_count": r_h2["prefetches_completed_unused"],
                "h1_additional_traffic_pct": r_h1["additional_traffic_pct"],
                "h2_additional_traffic_pct": r_h2["additional_traffic_pct"],
                "h1_stall_ms": r_h1["exposed_gpu_stall_ms"],
                "h2_stall_ms": r_h2["exposed_gpu_stall_ms"],
                "h1_token_lat_ms": r_h1["token_latency_mean_ms"],
                "h2_token_lat_ms": r_h2["token_latency_mean_ms"],
            })
    df_horizon = pd.DataFrame(horizon_rows)
    horizon_csv = os.path.join(RESULTS_DIR, "horizon_trends.csv")
    df_horizon.to_csv(horizon_csv, index=False)
    print(f"Saved: {horizon_csv}")

    print("\n" + "=" * 80)
    print("[CAMPAIGN COMPLETE] All 20 simulations finished successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

