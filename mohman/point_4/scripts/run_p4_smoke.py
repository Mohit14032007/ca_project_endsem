#!/usr/bin/env python3
"""
Point 4 Phase 2 Smoke Test Campaign Runner
Executes the 4-run smoke test matrix (S1: Baseline LRU, S2: Baseline FA-LRU, S3: CSTP LRU, S4: CSTP FA-LRU)
on Prompt 60 (B=1, T=4, L=5, K=8) with constrained HBM capacity 4,000,000,000 bytes.
Strictly adheres to zero ternary operators (ast.IfExp == 0).
"""

import os
import sys
import json
import time
import subprocess
import ast
import pandas as pd

WORKSPACE = "/home/mohman/ca_project"
REPO_ROOT = os.path.join(WORKSPACE, "LLMSimulator")
POINT4_ROOT = os.path.join(WORKSPACE, "mohman", "point_4")
BIN_PATH = os.path.join(REPO_ROOT, "build", "run_deepseekv3_mixtral_style")

SMOKE_ROOT = os.path.join(POINT4_ROOT, "smoke_tests")
CONFIG_DIR = os.path.join(SMOKE_ROOT, "configs")
LOG_DIR = os.path.join(SMOKE_ROOT, "logs")
RESULTS_DIR = os.path.join(SMOKE_ROOT, "results")
MANIFEST_DIR = os.path.join(SMOKE_ROOT, "manifests")
TRACE_DIR = os.path.join(SMOKE_ROOT, "traces")
TRACE_FILE = os.path.join(TRACE_DIR, "smoke_trace_p60_b1_t4_l5.csv")

DDR5_CFG = os.path.join(REPO_ROOT, "reproduction_A100", "with_off_chip_dram", "configs", "dram_config_DDR5_retry.yaml")
HBM_CFG = os.path.join(REPO_ROOT, "build", "dram_config_HBM3_80GB.yaml")

SMOKE_RUNS = [
    {
        "run_id": "s1_baseline_lru",
        "description": "Baseline No-Prefetch with LRU eviction",
        "prefetch": False,
        "eviction_policy": "lru",
        "window_size": 16,
        "capacity_bytes": 4000000000,
    },
    {
        "run_id": "s2_baseline_fa_lru",
        "description": "Baseline No-Prefetch with FA-LRU eviction (W=16)",
        "prefetch": False,
        "eviction_policy": "fa_lru",
        "window_size": 16,
        "capacity_bytes": 4000000000,
    },
    {
        "run_id": "s3_cstp_lru",
        "description": "CSTP Prefetching with LRU eviction",
        "prefetch": True,
        "eviction_policy": "lru",
        "window_size": 16,
        "capacity_bytes": 4000000000,
    },
    {
        "run_id": "s4_cstp_fa_lru",
        "description": "CSTP Prefetching with FA-LRU eviction (W=16)",
        "prefetch": True,
        "eviction_policy": "fa_lru",
        "window_size": 16,
        "capacity_bytes": 4000000000,
    },
]


def check_zero_ternary(file_path):
    with open(file_path, "r") as f:
        tree = ast.parse(f.read(), filename=file_path)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            count += 1
    if count > 0:
        raise ValueError(f"Found {count} ternary operators in {file_path}! Strict zero ternary rule violated.")
    print(f"[AST_CHECK] Verified 0 ternary operators in {file_path}")


def generate_config(run_def):
    run_id = run_def["run_id"]
    eviction = run_def["eviction_policy"]
    window = run_def["window_size"]
    cap = run_def["capacity_bytes"]
    has_prefetch = run_def["prefetch"]

    out_csv = os.path.join(RESULTS_DIR, run_id, "csv") + "/"
    out_raw = os.path.join(RESULTS_DIR, run_id, "raw") + "/"
    out_traces = os.path.join(RESULTS_DIR, run_id, "traces") + "/"

    lines = [
        "model:",
        "  model_name: deepseekV3",
        "  num_layers: 8",
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
        f"    expert_cache_capacity_bytes: {cap}",
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
        "  prefetching:",
    ]

    if has_prefetch:
        lines.extend([
            "    enabled: true",
            "    mode: hardware_async",
            f"    eviction_policy: {eviction}",
            f"    fa_lru_window_size: {window}",
            "    scheduling:",
            "      scheduling_mode: forward_lookahead",
            "      prediction_horizon: 1",
            "    candidate_selection:",
            "      budget: 4",
            "    predictor:",
            "      type: CSTP",
            "      prediction_horizon: 1",
            "      budget: 4",
            "      alpha: 0.5",
            "      smoothing_lambda: 1.0",
            "      prior_epsilon: 0.1",
            "      online_adaptation: true",
        ])
    else:
        lines.extend([
            "    enabled: false",
            "    mode: none",
            f"    eviction_policy: {eviction}",
            f"    fa_lru_window_size: {window}",
        ])

    lines.extend([
        "",
        "log:",
        "  print_log: false",
        "  export_gantt: false",
        f"  output_directory: {out_csv}",
        f"  gantt_directory: {out_raw}",
        f"  traces_directory: {out_traces}",
        "",
    ])

    cfg_path = os.path.join(CONFIG_DIR, f"{run_id}.yaml")
    with open(cfg_path, "w") as f:
        f.write("\n".join(lines))
    return cfg_path


def execute_run(run_def):
    run_id = run_def["run_id"]
    cfg_path = generate_config(run_def)
    log_path = os.path.join(LOG_DIR, f"{run_id}.log")
    run_traces_dir = os.path.join(RESULTS_DIR, run_id, "traces")
    os.makedirs(os.path.join(RESULTS_DIR, run_id, "csv"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, run_id, "raw"), exist_ok=True)
    os.makedirs(run_traces_dir, exist_ok=True)

    env = os.environ.copy()
    env["USE_EVENT_DRIVEN_DRAM"] = "true"
    env["DDR5_BANDWIDTH_GBPS"] = "64.0"
    env["TRACE_DIR"] = run_traces_dir
    env["PREDICTION_OUTPUT_DIR"] = run_traces_dir
    env["DEBUG_CACHE_CAPACITY"] = str(run_def["capacity_bytes"])
    env["EVICTION_POLICY"] = run_def["eviction_policy"]
    env["FA_LRU_WINDOW_SIZE"] = str(run_def["window_size"])

    cmd = [BIN_PATH, cfg_path]
    print(f"\n========================================================")
    print(f"[EXECUTING] {run_id}: {run_def['description']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Output Trace: {run_traces_dir}")

    t0 = time.time()
    with open(log_path, "w") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - t0

    if res.returncode != 0:
        print(f"  [ERROR] Exited with code {res.returncode}. Log: {log_path}")
        return False, elapsed, res.returncode
    print(f"  [SUCCESS] Completed in {elapsed:.2f} seconds (Exit Code 0)")

    manifest = {
        "run_id": run_id,
        "description": run_def["description"],
        "config_path": cfg_path,
        "log_path": log_path,
        "return_code": res.returncode,
        "host_elapsed_seconds": elapsed,
        "eviction_policy": run_def["eviction_policy"],
        "window_size": run_def["window_size"],
        "capacity_bytes": run_def["capacity_bytes"],
        "prefetch": run_def["prefetch"],
    }
    manifest_path = os.path.join(MANIFEST_DIR, f"{run_id}_manifest.json")
    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)

    return True, elapsed, res.returncode


def main():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    os.makedirs(TRACE_DIR, exist_ok=True)

    check_zero_ternary(__file__)

    if not os.path.exists(TRACE_FILE):
        raise FileNotFoundError(f"Trace file not found: {TRACE_FILE}")

    all_passed = True
    summaries = []
    for rdef in SMOKE_RUNS:
        ok, elapsed, code = execute_run(rdef)
        if not ok:
            all_passed = False
        summaries.append({"run_id": rdef["run_id"], "passed": ok, "time_s": elapsed, "code": code})

    print("\n========================================================")
    print("SMOKE TEST CAMPAIGN EXECUTION SUMMARY")
    print("========================================================")
    for s in summaries:
        status_str = "PASS"
        if not s["passed"]:
            status_str = "FAIL"
        print(f"  {s['run_id']}: [{status_str}] in {s['time_s']:.2f}s (code={s['code']})")

    if not all_passed:
        print("\n[CRITICAL FAILURE] One or more smoke tests failed!")
        sys.exit(1)
    print("\n[ALL PASSED] All 4 smoke test runs executed successfully with code 0.")


if __name__ == "__main__":
    main()

