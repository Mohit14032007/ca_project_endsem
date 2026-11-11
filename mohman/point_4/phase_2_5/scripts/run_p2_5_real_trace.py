#!/usr/bin/env python3
"""
Phase 2.5 Real Trace Policy Comparison Script
Runs 16-token real trace (MoE-Anywhere Prompt 60, B=1, T=16, L=5, K=8)
comparing Standard LRU against FA-LRU (W=16) under 4.0 GB constrained HBM capacity.
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
P25_ROOT = os.path.join(WORKSPACE, "mohman", "point_4", "phase_2_5")
BIN_PATH = os.path.join(REPO_ROOT, "build", "run_deepseekv3_mixtral_style")

CONFIG_DIR = os.path.join(P25_ROOT, "configs")
LOG_DIR = os.path.join(P25_ROOT, "logs")
RESULTS_DIR = os.path.join(P25_ROOT, "results")
TRACE_FILE = os.path.join(P25_ROOT, "traces", "real_trace_p60_b1_t16_l5.csv")

DDR5_CFG = os.path.join(REPO_ROOT, "reproduction_A100", "with_off_chip_dram", "configs", "dram_config_DDR5_retry.yaml")
HBM_CFG = os.path.join(REPO_ROOT, "build", "dram_config_HBM3_80GB.yaml")


def check_zero_ternary(file_path):
    with open(file_path, "r") as f:
        tree = ast.parse(f.read(), filename=file_path)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            count += 1
    if count > 0:
        raise ValueError(f"Found {count} ternary operators in {file_path}!")
    print(f"[AST_CHECK] Verified 0 ternary operators in {file_path}")


def generate_config(run_id, policy_name, window_size=16, capacity_bytes=4000000000):
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
        f"    expert_cache_capacity_bytes: {capacity_bytes}",
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
        "  output_len: 16",
        "  precision_byte: 2",
        "  skewness: 0.0",
        "  iter: 16",
        "  injection_rate: 0",
        "  exit_out_of_memory: false",
        "  mem_cap_limit: false",
        f"  trace_file: {TRACE_FILE}",
        "  prefetching:",
        "    enabled: false",
        "    mode: none",
        f"    eviction_policy: {policy_name}",
        f"    fa_lru_window_size: {window_size}",
        "",
        "log:",
        "  print_log: false",
        "  export_gantt: false",
        f"  output_directory: {out_csv}",
        f"  gantt_directory: {out_raw}",
        f"  traces_directory: {out_traces}",
        "",
    ]

    cfg_path = os.path.join(CONFIG_DIR, f"{run_id}.yaml")
    with open(cfg_path, "w") as f:
        f.write("\n".join(lines))
    return cfg_path


def run_experiment(run_id, policy_name, window_size=16):
    cfg_path = generate_config(run_id, policy_name, window_size)
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
    env["DEBUG_CACHE_CAPACITY"] = "4000000000"
    env["EVICTION_POLICY"] = policy_name
    env["FA_LRU_WINDOW_SIZE"] = str(window_size)

    cmd = [BIN_PATH, cfg_path]
    print(f"\n[RUNNING] {run_id} ({policy_name}, W={window_size})...")
    t0 = time.time()
    with open(log_path, "w") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - t0

    if res.returncode != 0:
        print(f"  [ERROR] Exited with code {res.returncode}. Log: {log_path}")
        return False, elapsed
    print(f"  [SUCCESS] Completed in {elapsed:.2f} seconds (Exit Code 0)")
    return True, elapsed


def parse_results(run_id):
    summary_path = os.path.join(RESULTS_DIR, run_id, "traces", "prefetch_summary.json")
    with open(summary_path) as f:
        d = json.load(f)
    return d


def main():
    check_zero_ternary(__file__)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ok1, time1 = run_experiment("real_p60_lru", "lru", 16)
    ok2, time2 = run_experiment("real_p60_fa_lru", "fa_lru", 16)

    if not ok1 or not ok2:
        print("[FAILURE] One or both real trace experiments failed!")
        sys.exit(1)

    d1 = parse_results("real_p60_lru")
    d2 = parse_results("real_p60_fa_lru")

    hbm1 = d1["hbm_management"]
    hbm2 = d2["hbm_management"]

    bytes1 = d1["memory_traffic"]["total_ddr5_bytes_transferred"]
    bytes2 = d2["memory_traffic"]["total_ddr5_bytes_transferred"]

    gb1 = bytes1 / 1e9
    gb2 = bytes2 / 1e9
    gib1 = bytes1 / (1024**3)
    gib2 = bytes2 / (1024**3)

    stall1 = d1["exposed_stall_and_execution"]["exposed_gpu_stall_ns"] / 1e6
    stall2 = d2["exposed_stall_and_execution"]["exposed_gpu_stall_ns"] / 1e6

    tok_lat1 = d1["token_latency"]["mean_ns"] / 1e6
    tok_lat2 = d2["token_latency"]["mean_ns"] / 1e6

    tok_sec1 = d1["token_latency"]["simulated_tokens_per_second"]
    tok_sec2 = d2["token_latency"]["simulated_tokens_per_second"]

    print("\n======================================================================")
    print("PHASE 2.5: 16-TOKEN REAL TRACE COMPARISON (Prompt 60, 4.0 GB HBM)")
    print("======================================================================")
    print(f"| Metric | Standard LRU | FA-LRU (W=16) | Delta (FA-LRU - LRU) |")
    print(f"| :--- | :--- | :--- | :--- |")
    print(f"| Coalesced Demands | {hbm1['coalesced_actual_demands']} | {hbm2['coalesced_actual_demands']} | 0 |")
    print(f"| Routed Demands | {hbm1['routed_actual_demands']} | {hbm2['routed_actual_demands']} | 0 |")
    print(f"| Tensor Executions | {hbm1['tensor_actual_demands']} | {hbm2['tensor_actual_demands']} | 0 |")
    print(f"| Coalesced Hits | {hbm1['coalesced_resident_hits']} | {hbm2['coalesced_resident_hits']} | {hbm2['coalesced_resident_hits'] - hbm1['coalesced_resident_hits']} |")
    print(f"| Coalesced Misses | {hbm1['coalesced_fresh_demand_misses']} | {hbm2['coalesced_fresh_demand_misses']} | {hbm2['coalesced_fresh_demand_misses'] - hbm1['coalesced_fresh_demand_misses']} |")
    print(f"| Coalesced Hit Rate | {hbm1['coalesced_hbm_hit_rate']*100:.2f}% | {hbm2['coalesced_hbm_hit_rate']*100:.2f}% | {(hbm2['coalesced_hbm_hit_rate'] - hbm1['coalesced_hbm_hit_rate'])*100:.2f}% |")
    print(f"| Total Evictions | {hbm1['total_evictions']} | {hbm2['total_evictions']} | {hbm2['total_evictions'] - hbm1['total_evictions']} |")
    print(f"| Reloaded Experts Count | {hbm1['reloaded_experts_count']} | {hbm2['reloaded_experts_count']} | {hbm2['reloaded_experts_count'] - hbm1['reloaded_experts_count']} |")
    print(f"| DDR5 Traffic (GB) [10^9] | {gb1:.4f} GB | {gb2:.4f} GB | {gb2 - gb1:.4f} GB |")
    print(f"| DDR5 Traffic (GiB) [2^30] | {gib1:.4f} GiB | {gib2:.4f} GiB | {gib2 - gib1:.4f} GiB |")
    print(f"| Exposed GPU Stall | {stall1:.2f} ms | {stall2:.2f} ms | {stall2 - stall1:.2f} ms |")
    print(f"| Mean Token Latency | {tok_lat1:.2f} ms | {tok_lat2:.2f} ms | {tok_lat2 - tok_lat1:.2f} ms |")
    print(f"| Simulated Tokens/sec | {tok_sec1:.2f} | {tok_sec2:.2f} | {tok_sec2 - tok_sec1:.2f} |")
    print(f"| Total Host Runtime | {time1:.2f} s | {time2:.2f} s | {time2 - time1:.2f} s |")
    print("======================================================================")

    out_json = os.path.join(P25_ROOT, "real_trace_comparison_summary.json")
    with open(out_json, "w") as f:
        json.dump({
            "workload": "Prompt 60 (B=1, T=16, L=5, K=8)",
            "capacity_bytes": 4000000000,
            "lru": {
                "hits": hbm1["coalesced_resident_hits"],
                "misses": hbm1["coalesced_fresh_demand_misses"],
                "hit_rate_pct": hbm1["coalesced_hbm_hit_rate"] * 100,
                "evictions": hbm1["total_evictions"],
                "reloads": hbm1["reloaded_experts_count"],
                "traffic_gb": gb1,
                "traffic_gib": gib1,
                "stall_ms": stall1,
                "token_lat_ms": tok_lat1,
                "host_time_s": time1,
            },
            "fa_lru": {
                "hits": hbm2["coalesced_resident_hits"],
                "misses": hbm2["coalesced_fresh_demand_misses"],
                "hit_rate_pct": hbm2["coalesced_hbm_hit_rate"] * 100,
                "evictions": hbm2["total_evictions"],
                "reloads": hbm2["reloaded_experts_count"],
                "traffic_gb": gb2,
                "traffic_gib": gib2,
                "stall_ms": stall2,
                "token_lat_ms": tok_lat2,
                "host_time_s": time2,
            }
        }, f, indent=2)


if __name__ == "__main__":
    main()

