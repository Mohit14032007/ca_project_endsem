#!/usr/bin/env python3
"""
Phase 2 Controlled Pilot Evaluation Script.
Executes the 5 pilot runs on the certified 40 MoE stages topology with 35 GB constrained cache:
1. Baseline demand loading
2. ST-MoE, H=1
3. ST-MoE, H=2
4. CSTP, H=1
5. CSTP, H=2

Enforces ZERO ternary operators (ast.IfExp == 0).
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
DDR5_CFG = os.path.join(REPO_ROOT, "reproduction_A100", "with_off_chip_dram", "configs", "dram_config_DDR5_retry.yaml")
HBM_CFG = os.path.join(REPO_ROOT, "build", "dram_config_HBM3_80GB.yaml")
TRACE_FILE = os.path.join(PHASE2_ROOT, "experiments", "traces", "test_moe_anywhere_b1_s40_t4.csv")

PILOT_RUNS = [
    {"run_id": "pilot_baseline", "predictor": "baseline", "horizon": 0, "budget": 0},
    {"run_id": "pilot_stmoe_h1", "predictor": "stmoe", "horizon": 1, "budget": 4},
    {"run_id": "pilot_stmoe_h2", "predictor": "stmoe", "horizon": 2, "budget": 4},
    {"run_id": "pilot_cstp_h1", "predictor": "cstp", "horizon": 1, "budget": 4},
    {"run_id": "pilot_cstp_h2", "predictor": "cstp", "horizon": 2, "budget": 4},
]

CACHE_BYTES = 37580963840  # 35.00 GB constrained cache


def generate_yaml(run_def):
    run_id = run_def["run_id"]
    pred = run_def["predictor"]
    h = run_def["horizon"]
    bgt = run_def["budget"]

    raw_dir = os.path.join(PHASE2_ROOT, "experiments", "raw", "pilot", run_id)
    csv_dir = os.path.join(raw_dir, "csv")
    traces_dir = os.path.join(raw_dir, "traces")

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
        "  max_batch_size: 1",
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
        f"  output_directory: {csv_dir}/",
        f"  gantt_directory: {raw_dir}/",
        f"  traces_directory: {traces_dir}/",
        "",
    ])

    return "\n".join(lines)


def run_pilot():
    cfg_dir = os.path.join(PHASE2_ROOT, "experiments", "configs", "pilot")
    os.makedirs(cfg_dir, exist_ok=True)
    records = []

    for rdef in PILOT_RUNS:
        run_id = rdef["run_id"]
        pred = rdef["predictor"]
        h = rdef["horizon"]
        bgt = rdef["budget"]

        raw_dir = os.path.join(PHASE2_ROOT, "experiments", "raw", "pilot", run_id)
        csv_dir = os.path.join(raw_dir, "csv")
        traces_dir = os.path.join(raw_dir, "traces")
        log_path = os.path.join(raw_dir, f"{run_id}.log")

        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(traces_dir, exist_ok=True)

        yaml_content = generate_yaml(rdef)
        cfg_path = os.path.join(cfg_dir, f"{run_id}.yaml")
        with open(cfg_path, "w") as f:
            f.write(yaml_content)

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
            continue

        # Parse experiment_summary.json
        sum_path = os.path.join(traces_dir, "experiment_summary.json")
        with open(sum_path, "r") as f:
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
            timely_pct = (timely_hits / matched_prefetches) * 100.0
            late_pct = (late_hits / matched_prefetches) * 100.0
        else:
            timely_pct = 0.0
            late_pct = 0.0

        unused_count = unused["prefetches_completed_unused"]
        used_count = unused["prefetches_completed_and_used"]
        comp_total = unused_count + used_count
        if comp_total > 0:
            unused_pct = (unused_count / comp_total) * 100.0
        else:
            unused_pct = 0.0

        records.append({
            "run_id": run_id,
            "predictor": pred,
            "horizon": h,
            "budget": bgt,
            "host_runtime_sec": elapsed,
            "total_demands": cov["total_actual_demands"],
            "unique_demands": cov["unique_actual_demands"],
            "demands_with_prediction": cov["demands_with_prediction"],
            "prefetch_assisted_demands": cov["prefetch_assisted_demands"],
            "overall_coverage_pct": cov["overall_coverage"] * 100.0,
            "predicted_coverage_pct": cov["predicted_coverage"] * 100.0,
            "timely_hits": timely_hits,
            "late_hits": late_hits,
            "timely_of_matched_pct": timely_pct,
            "late_of_matched_pct": late_pct,
            "unused_count": unused_count,
            "unused_pct": unused_pct,
            "unused_bytes": unused["completed_unused_bytes"],
            "hbm_hit_rate_pct": hbm["overall_hbm_hit_rate"] * 100.0,
            "demand_bytes": traff["demand_bytes_transferred"],
            "speculative_bytes": traff["speculative_bytes_transferred"],
            "total_ddr5_bytes": traff["total_ddr5_bytes_transferred"],
            "exposed_gpu_stall_ms": stall["exposed_gpu_stall_ns"] / 1e6,
            "stall_per_token_ms": stall["wait_per_token_ns"] / 1e6,
            "total_sim_time_ms": stall["total_simulated_execution_ns"] / 1e6,
            "mean_expert_load_lat_us": exp_lat["mean_ns"] / 1e3,
            "p50_expert_load_lat_us": exp_lat["p50_ns"] / 1e3,
            "p95_expert_load_lat_us": exp_lat["p95_ns"] / 1e3,
            "mean_token_lat_ms": tok_lat["mean_ns"] / 1e6,
            "p50_token_lat_ms": tok_lat["p50_ns"] / 1e6,
            "p95_token_lat_ms": tok_lat["p95_ns"] / 1e6,
        })
        print(
            f"[SUCCESS] {run_id:18s} | Dispatched: {timely_late['total_dispatched_prefetches']:3d} | "
            f"Assisted: {cov['prefetch_assisted_demands']:2d} | Late: {late_hits:2d} | "
            f"Timely: {timely_hits:2d} | Unused: {unused_count:2d} | Stall: {stall['exposed_gpu_stall_ns']/1e6:7.1f} ms"
        )

    out_csv = os.path.join(PHASE2_ROOT, "results", "pilot_metrics.csv")
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"\nPilot metrics saved to {out_csv}")


if __name__ == "__main__":
    run_pilot()

