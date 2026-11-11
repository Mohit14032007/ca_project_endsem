#!/usr/bin/env python3
"""
Large Workload Pilot Runner (Workload W5: B=70, T=16, 40 MoE Stages).
Executes Baseline, ST-MoE (H=2), and CSTP (H=2) under 35 GB constrained cache.
Verifies event-level accounting and prefetch effectiveness metrics.

STRICT RULE: Zero ternary operators (ast.IfExp == 0).
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
LARGE_ROOT = os.path.join(WORKSPACE, "mohman", "point_3", "phase2", "large_workload")

CONFIG_DIR = os.path.join(LARGE_ROOT, "experiments", "configs", "pilot")
LOG_DIR = os.path.join(LARGE_ROOT, "experiments", "logs", "pilot")
RAW_DIR = os.path.join(LARGE_ROOT, "experiments", "raw", "pilot")
RESULTS_DIR = os.path.join(LARGE_ROOT, "results")
TRACE_FILE = os.path.join(LARGE_ROOT, "experiments", "traces", "W5.csv")

DDR5_CFG = os.path.join(REPO_ROOT, "reproduction_A100", "with_off_chip_dram", "configs", "dram_config_DDR5_retry.yaml")
HBM_CFG = os.path.join(REPO_ROOT, "build", "dram_config_HBM3_80GB.yaml")
CACHE_BYTES = 37580963840  # 35.00 GB

PILOT_CONDITIONS = [
    {"run_id": "large_pilot_w5_baseline", "predictor": "baseline", "horizon": 0, "budget": 0},
    {"run_id": "large_pilot_w5_stmoe_h2", "predictor": "stmoe", "horizon": 2, "budget": 4},
    {"run_id": "large_pilot_w5_cstp_h2", "predictor": "cstp", "horizon": 2, "budget": 4},
]


def generate_yaml(run_def):
    run_id = run_def["run_id"]
    pred = run_def["predictor"]
    h = run_def["horizon"]
    bgt = run_def["budget"]

    raw_run_dir = os.path.join(RAW_DIR, run_id)
    csv_dir = os.path.join(raw_run_dir, "csv") + "/"
    gantt_dir = os.path.join(raw_run_dir, "raw") + "/"
    traces_dir = os.path.join(raw_run_dir, "traces") + "/"

    lines = [
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
        f"    expert_cache_capacity_bytes: {CACHE_BYTES}",
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
        "  max_batch_size: 70",
        "  max_process_token: 0",
        "",
        "simulation:",
        "  data: moe_anywhere",
        "  input_len: 128",
        "  output_len: 16",
        "  precision_byte: 2",
        "  skewness: 0.0",
        "  iter: 16",
        "  injection_rate: 0",
        "  exit_out_of_memory: false",
        "  mem_cap_limit: false",
        f"  trace_file: {TRACE_FILE}",
    ]

    if pred == "baseline":
        lines.extend([
            "  prefetching:",
            "    enabled: false",
            "    mode: none",
        ])
    elif pred == "stmoe":
        lines.extend([
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
    elif pred == "cstp":
        lines.extend([
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

    lines.extend([
        "",
        "log:",
        "  print_log: false",
        "  export_gantt: false",
        f"  output_directory: {csv_dir}",
        f"  gantt_directory: {gantt_dir}",
        f"  traces_directory: {traces_dir}",
        "",
    ])

    return "\n".join(lines)


def run_pilot():
    print("======================================================================")
    print("EXECUTING STAGE D HIGH-VALUE PILOT (W5: B=70, T=16, 40 STAGES)")
    print("======================================================================")

    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    records = []
    baseline_stall_ms = 0.0

    for rdef in PILOT_CONDITIONS:
        run_id = rdef["run_id"]
        pred = rdef["predictor"]
        h = rdef["horizon"]
        bgt = rdef["budget"]

        raw_run_dir = os.path.join(RAW_DIR, run_id)
        traces_dir = os.path.join(raw_run_dir, "traces")
        log_path = os.path.join(LOG_DIR, f"{run_id}.log")

        os.makedirs(os.path.join(raw_run_dir, "csv"), exist_ok=True)
        os.makedirs(os.path.join(raw_run_dir, "raw"), exist_ok=True)
        os.makedirs(traces_dir, exist_ok=True)

        yaml_str = generate_yaml(rdef)
        cfg_path = os.path.join(CONFIG_DIR, f"{run_id}.yaml")
        with open(cfg_path, "w") as f:
            f.write(yaml_str)

        env = os.environ.copy()
        env["USE_EVENT_DRIVEN_DRAM"] = "true"
        env["DDR5_BANDWIDTH_GBPS"] = "64.0"
        env["PREDICTION_OUTPUT_DIR"] = traces_dir

        cmd = [BIN_PATH, cfg_path]
        print(f"Launching {run_id:26s} ... ", end="", flush=True)
        t0 = time.time()
        with open(log_path, "w") as log_f:
            res = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        elapsed = time.time() - t0

        if res.returncode != 0:
            print(f"FAILED (code {res.returncode}) in {elapsed:.1f}s")
            continue

        sum_json = os.path.join(traces_dir, "experiment_summary.json")
        with open(sum_json, "r") as f:
            data = json.load(f)

        cov = data["prefetch_coverage"]
        timely_late = data["timely_and_late_rates"]
        unused = data["unused_prefetches"]
        hbm = data["hbm_hit_rates"]
        traff = data["memory_traffic"]
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
        comp_total = unused_count + used_count
        if comp_total > 0:
            unused_pct = (unused_count / comp_total) * 100.0
        else:
            unused_pct = 0.0

        # Parse cache CSV for evictions and inserts
        cache_csv = os.path.join(traces_dir, "expert_cache.csv")
        inserts = 0
        evicts = 0
        if os.path.exists(cache_csv):
            cdf = pd.read_csv(cache_csv)
            acts = cdf["action"].value_counts().to_dict()
            inserts = acts.get("INSERT", 0)
            evicts = acts.get("EVICT", 0)

        total_demands = cov["total_actual_demands"]
        unique_demands = cov["unique_actual_demands"]
        assisted = cov["prefetch_assisted_demands"]

        # 8.1 Distinct Coverage Metrics: Strict Timely Coverage vs Prefetch-Associated Coverage
        strict_timely_coverage_pct = (timely_hits / total_demands) * 100.0
        prefetch_associated_coverage_pct = (assisted / total_demands) * 100.0

        current_stall_ms = stall["exposed_gpu_stall_ns"] / 1e6
        if pred == "baseline":
            baseline_stall_ms = current_stall_ms
            stall_reduction_ms = 0.0
        else:
            stall_reduction_ms = baseline_stall_ms - current_stall_ms

        rec = {
            "run_id": run_id,
            "predictor": pred,
            "horizon": h,
            "budget": bgt,
            "batch_size": 70,
            "tokens": 16,
            "host_runtime_sec": elapsed,
            "total_demands": total_demands,
            "unique_demands": unique_demands,
            "demands_with_prediction": cov["demands_with_prediction"],
            "prefetch_assisted_demands": assisted,
            "strict_timely_coverage_pct": strict_timely_coverage_pct,
            "prefetch_associated_coverage_pct": prefetch_associated_coverage_pct,
            "predicted_coverage_pct": cov["predicted_coverage"] * 100.0,
            "timely_hits": timely_hits,
            "late_hits": late_hits,
            "timely_of_matched_pct": timely_of_matched_pct,
            "late_of_matched_pct": late_of_matched_pct,
            "dispatched_prefetches": timely_late["total_dispatched_prefetches"],
            "prefetches_completed_used": used_count,
            "prefetches_completed_unused": unused_count,
            "unused_pct": unused_pct,
            "unused_bytes": unused["completed_unused_bytes"],
            "hbm_hit_rate_pct": hbm["overall_hbm_hit_rate"] * 100.0,
            "inserts": inserts,
            "evicts": evicts,
            "demand_bytes": traff["demand_bytes_transferred"],
            "speculative_bytes": traff["speculative_bytes_transferred"],
            "total_ddr5_bytes": traff["total_ddr5_bytes_transferred"],
            "exposed_gpu_stall_ms": current_stall_ms,
            "stall_per_token_ms": stall["wait_per_token_ns"] / 1e6,
            "stall_per_request_ms": (stall["exposed_gpu_stall_ns"] / 70.0) / 1e6,
            "stall_reduction_ms": stall_reduction_ms,
            "total_sim_time_ms": stall["total_simulated_execution_ns"] / 1e6,
            "mean_expert_load_lat_us": exp_lat["mean_ns"] / 1e3,
            "p50_expert_load_lat_us": exp_lat["p50_ns"] / 1e3,
            "p95_expert_load_lat_us": exp_lat["p95_ns"] / 1e3,
            "mean_token_lat_ms": tok_lat["mean_ns"] / 1e6,
            "p50_token_lat_ms": tok_lat["p50_ns"] / 1e6,
            "p95_token_lat_ms": tok_lat["p95_ns"] / 1e6,
        }
        records.append(rec)
        print(
            f"DONE ({elapsed:4.1f}s) | Dispatched: {rec['dispatched_prefetches']:4d} | "
            f"Assisted: {assisted:4d} | Timely: {timely_hits:3d} | Late: {late_hits:4d} | "
            f"Evicts: {evicts:5d} | Stall: {current_stall_ms:7.1f} ms (Delta: {stall_reduction_ms:+5.2f} ms)"
        )

    out_csv = os.path.join(RESULTS_DIR, "large_pilot_metrics.csv")
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"\n[PILOT COMPLETE] Saved metrics to {out_csv}")


if __name__ == "__main__":
    run_pilot()

