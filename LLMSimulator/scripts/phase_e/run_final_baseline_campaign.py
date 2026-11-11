#!/usr/bin/env python3
"""
Final Baseline Campaign Execution Harness.

Executes the approved baseline configurations for the MoE Memory Wall project:
1. Full-depth baseline matrix (A1-A8, B1-B4, C1-C3) with authentic Ramulator DDR5 timing.
2. Representative screening baselines (A1-rep, A5-rep, A7-rep) for calibration.

Extracts all 17 required baseline metrics directly from simulation traces.
Generates all authoritative deliverable reports and matrices.

ENFORCEMENT: ZERO TERNARY OPERATORS (ast.IfExp == 0).
"""

import ast
import csv
import glob
import os
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd


# Directory layout
SIMULATOR_ROOT = "/home/mohman/ca_project/LLMSimulator"
EXECUTABLE = os.path.join(SIMULATOR_ROOT, "build/run_deepseekv3_mixtral_style")
OUTPUT_BASE = os.path.join(SIMULATOR_ROOT, "results/endsem_phase_e/final_baseline")
CONFIGS_DIR = os.path.join(OUTPUT_BASE, "configs")
RUNS_DIR = os.path.join(OUTPUT_BASE, "runs")
TRACES_DIR = os.path.join(OUTPUT_BASE, "traces")
DOCS_DIR = os.path.join(SIMULATOR_ROOT, "docs/phase_e")
MIRROR_DOCS_DIR = "/home/mohman/ca_project/docs/phase_e"
MIRROR_RESULTS_DIR = "/home/mohman/ca_project/results/endsem_phase_e/final_baseline"


def generate_yaml(config_id, num_layers, batch_size, input_len, output_len, iter_steps, trace_file, bypass_cache, run_dir):
    csv_dir = os.path.join(run_dir, "csv")
    traces_dir = os.path.join(run_dir, "traces")
    raw_dir = os.path.join(run_dir, "raw")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(traces_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    if bypass_cache:
        bypass_str = "true"
    else:
        bypass_str = "false"

    if trace_file != "":
        trace_line = f"  trace_file: {trace_file}\n"
    else:
        trace_line = ""

    yaml_text = f"""log:
  export_gantt: false
  gantt_directory: {raw_dir}/
  output_directory: {csv_dir}/
  print_log: false
  traces_directory: {traces_dir}/
model:
  model_name: deepseekV3
  num_layers: {num_layers}
serving:
  max_batch_size: {batch_size}
  max_process_token: 0
simulation:
  data: synthesis
  exit_out_of_memory: false
  injection_rate: 0
  input_len: {input_len}
  iter: {iter_steps}
  mem_cap_limit: false
  output_len: {output_len}
  precision_byte: 2
  skewness: 0.0
{trace_line}  warm_hbm_preload: false
system:
  distribution:
    expert_tensor_degree: 1
    none_expert_tensor_degree: 1
  gpu_gen: A100
  infiniband_gen: 800
  num_device: 1
  num_node: 1
  nvlink_gen: 3
  optimization:
    bypass_offchip_execution_cache: {bypass_str}
    compressed_kv: true
    decode_mode: true
    disagg_system: false
    expert_cache_capacity_bytes: 80000000000
    hbm_dram_cfg_path: reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/config_snapshot/dram_config_HBM3_80GB.yaml
    hetero_subbatch: false
    kv_cache_reuse_rate: 0.0
    offchip_dram_cfg_path: reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/config_snapshot/dram_config_DDR5_retry.yaml
    parallel_execution: 'off'
    prefill_mode: false
    reuse_kv_cache: true
    use_absorb: true
    use_flash_attention: true
    use_flash_mla: true
    use_hitting_queue: false
    use_low_unit_moe_only: false
    use_ramulator: 'on'
  processor_type: GPU
"""
    yaml_path = os.path.join(CONFIGS_DIR, f"{config_id}.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_text)
    return yaml_path


def parse_simulation_metrics(run_dir, batch_size, iter_steps):
    csv_dir = os.path.join(run_dir, "csv")
    traces_dir = os.path.join(run_dir, "traces")

    routing_path = os.path.join(traces_dir, "expert_routing.csv")
    cache_path = os.path.join(traces_dir, "expert_cache.csv")
    load_path = os.path.join(traces_dir, "expert_load.csv")
    csv_files = glob.glob(os.path.join(csv_dir, "*_ramul_decode.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No summary CSV found in {csv_dir}")

    df_route = pd.read_csv(routing_path)
    df_cache = pd.read_csv(cache_path)
    df_load = pd.read_csv(load_path)
    df_summary = pd.read_csv(csv_files[0])

    # 1. Actual expert demand count
    demand_count = int(len(df_route))

    # 2. Unique expert count
    unique_experts = int(len(df_route[["layer_id", "expert_id"]].drop_duplicates()))

    # 3. Temporal consecutive-token expert overlap
    tokens = sorted(df_route["token_id"].unique())
    overlaps = []
    for i in range(len(tokens) - 1):
        t1 = tokens[i]
        t2 = tokens[i + 1]
        s1 = set(zip(df_route[df_route["token_id"] == t1]["layer_id"], df_route[df_route["token_id"] == t1]["expert_id"]))
        s2 = set(zip(df_route[df_route["token_id"] == t2]["layer_id"], df_route[df_route["token_id"] == t2]["expert_id"]))
        u = s1.union(s2)
        if len(u) > 0:
            overlaps.append(len(s1.intersection(s2)) / len(u))
    if len(overlaps) > 0:
        temporal_overlap = float(np.mean(overlaps))
    else:
        temporal_overlap = 0.0

    # 4. Cross-request expert overlap (where B > 1)
    if batch_size > 1:
        req_overlaps = []
        requests = sorted(df_route["request_id"].unique())
        for t in tokens:
            t_df = df_route[df_route["token_id"] == t]
            for r1_idx in range(len(requests)):
                for r2_idx in range(r1_idx + 1, len(requests)):
                    r1 = requests[r1_idx]
                    r2 = requests[r2_idx]
                    s1 = set(zip(t_df[t_df["request_id"] == r1]["layer_id"], t_df[t_df["request_id"] == r1]["expert_id"]))
                    s2 = set(zip(t_df[t_df["request_id"] == r2]["layer_id"], t_df[t_df["request_id"] == r2]["expert_id"]))
                    u = s1.union(s2)
                    if len(u) > 0:
                        req_overlaps.append(len(s1.intersection(s2)) / len(u))
        if len(req_overlaps) > 0:
            cross_req_overlap = float(np.mean(req_overlaps))
        else:
            cross_req_overlap = 0.0
    else:
        cross_req_overlap = 0.0

    # 5. Expert-level HBM hit rate
    expert_hits = int(len(df_cache[df_cache["action"] == "HIT"]))
    expert_misses = int(len(df_cache[df_cache["action"] == "MISS"]))
    total_expert_lookups = expert_hits + expert_misses
    if total_expert_lookups > 0:
        expert_hbm_hit_rate = float(expert_hits / total_expert_lookups)
    else:
        expert_hbm_hit_rate = 0.0

    # 6. Matrix-level HBM hit rate
    df_matrix = df_load[df_load["weight_tensor"] != "all_weights"]
    matrix_hits = int(len(df_matrix[df_matrix["cache_hit_or_miss"] == "hit"]))
    matrix_total = int(len(df_matrix))
    if matrix_total > 0:
        matrix_hbm_hit_rate = float(matrix_hits / matrix_total)
    else:
        matrix_hbm_hit_rate = 0.0

    # 7. DDR5 demand misses
    ddr5_demand_misses = expert_misses

    # 8. DDR5 bytes transferred (88,080,384 bytes per FP16 DeepSeek expert)
    ddr5_bytes_transferred = int(ddr5_demand_misses * 88080384)

    # 9. DDR5 burst count (64 bytes per DRAM burst)
    ddr5_burst_count = int(ddr5_bytes_transferred // 64)

    # 10. Expert-load latency (sum of duration of full expert transfers from offchip DRAM in ms)
    df_expert_transfers = df_load[(df_load["weight_tensor"] == "all_weights") & (df_load["cache_hit_or_miss"] == "miss")]
    expert_load_latency_ms = float(df_expert_transfers["duration"].sum() / 1e6)

    # 11. Compute time (ms)
    compute_time_ms = float((df_summary["fc_comp"] + df_summary["attn_comp"] + df_summary["moe_comp"]).sum() / 1e6)

    # 12. Exposed memory-stall time (ms)
    exposed_memory_stall_ms = float(df_summary["moe_dram"].sum() / 1e6)

    # 13. End-to-end execution time (ms)
    end_to_end_sim_time_ms = float(df_summary["time"].iloc[-1] / 1e6)

    # 14. Per-token latency (ms/token)
    if iter_steps > 0:
        per_token_latency_ms = float(end_to_end_sim_time_ms / iter_steps)
    else:
        per_token_latency_ms = 0.0

    # 15. Token throughput (tokens/second)
    if end_to_end_sim_time_ms > 0:
        token_throughput = float((batch_size * iter_steps) / (end_to_end_sim_time_ms / 1000.0))
    else:
        token_throughput = 0.0

    # 16. HBM occupancy (GB)
    max_resident_bytes = int(df_cache["resident_bytes"].max())
    hbm_occupancy_gb = float(max_resident_bytes / (1024 ** 3))

    # 17. Natural LRU eviction count
    natural_lru_evictions = int(len(df_cache[df_cache["action"] == "EVICT"]))

    return {
        "actual_expert_demands": demand_count,
        "unique_experts": unique_experts,
        "temporal_consecutive_overlap": temporal_overlap,
        "cross_request_overlap": cross_req_overlap,
        "expert_level_hbm_hit_rate": expert_hbm_hit_rate,
        "matrix_level_hbm_hit_rate": matrix_hbm_hit_rate,
        "ddr5_demand_misses": ddr5_demand_misses,
        "ddr5_bytes_transferred": ddr5_bytes_transferred,
        "ddr5_burst_count": ddr5_burst_count,
        "expert_load_latency_ms": expert_load_latency_ms,
        "compute_time_ms": compute_time_ms,
        "exposed_memory_stall_ms": exposed_memory_stall_ms,
        "end_to_end_sim_time_ms": end_to_end_sim_time_ms,
        "per_token_latency_ms": per_token_latency_ms,
        "token_throughput": token_throughput,
        "hbm_occupancy_gb": hbm_occupancy_gb,
        "natural_lru_evictions": natural_lru_evictions,
    }


def execute_configuration(config_spec):
    cfg_id = config_spec["config_id"]
    tbl = config_spec["table"]
    ds = config_spec["dataset"]
    T = config_spec["T"]
    B = config_spec["B"]
    input_len = config_spec["input_len"]
    output_len = config_spec["output_len"]
    iter_steps = config_spec["iter"]
    moe_stages = config_spec["moe_stages"]
    num_layers = moe_stages + 3
    trace_file = config_spec["trace_file"]
    bypass_cache = config_spec["bypass_cache"]
    scope = config_spec["scope"]

    run_dir = os.path.join(RUNS_DIR, cfg_id)
    yaml_path = generate_yaml(cfg_id, num_layers, B, input_len, output_len, iter_steps, trace_file, bypass_cache, run_dir)

    print(f"\n======================================================================")
    print(f"RUNNING CONFIGURATION: {cfg_id} | {ds} | T={T}, B={B}, Lin={input_len}, MoE={moe_stages} ({scope})")
    print(f"  YAML: {yaml_path}")
    print(f"  Trace: {trace_file}")
    print(f"  Bypass Cache: {bypass_cache}")
    print(f"======================================================================")

    start_time = time.time()
    cmd = f"cd {SIMULATOR_ROOT} && {EXECUTABLE} {yaml_path}"
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end_time = time.time()
    host_wall_time = end_time - start_time

    log_path = os.path.join(run_dir, "stdout_stderr.log")
    with open(log_path, "w") as f:
        f.write("=== STDOUT ===\n")
        f.write(proc.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(proc.stderr)

    if proc.returncode != 0:
        print(f"  [ERROR] Execution failed with return code {proc.returncode}!")
        print(f"  STDERR TAIL:\n{proc.stderr[-1000:]}")
        return {
            "config_id": cfg_id,
            "status": "FAILED",
            "host_wall_time_sec": host_wall_time,
            "error": proc.stderr[-200:],
        }

    print(f"  [SUCCESS] Completed in {host_wall_time:.2f} host seconds.")
    metrics = parse_simulation_metrics(run_dir, B, iter_steps)
    metrics["config_id"] = cfg_id
    metrics["table"] = tbl
    metrics["dataset"] = ds
    metrics["sequence_length_T"] = T
    metrics["batch_size_B"] = B
    metrics["input_len"] = input_len
    metrics["output_len"] = output_len
    metrics["iter"] = iter_steps
    metrics["moe_stages"] = moe_stages
    metrics["total_layers"] = num_layers
    metrics["scope"] = scope
    metrics["bypass_cache"] = bypass_cache
    metrics["status"] = "COMPLETED"
    metrics["host_wall_time_sec"] = host_wall_time
    metrics["yaml_path"] = yaml_path
    metrics["trace_path"] = trace_file

    print(f"  Metrics: Demands={metrics['actual_expert_demands']}, Unique={metrics['unique_experts']}, "
          f"ExpHitRate={metrics['expert_level_hbm_hit_rate']*100:.2f}%, Stall={metrics['exposed_memory_stall_ms']:.1f}ms, "
          f"SimTime={metrics['end_to_end_sim_time_ms']:.1f}ms, Evictions={metrics['natural_lru_evictions']}")
    return metrics


def build_experiment_specs():
    specs = []

    # 1. Full Depth Matrix (A1-A8, B1-B4, C1-C3)
    # Table A
    specs.append({
        "config_id": "A1", "table": "A", "dataset": "MoE Suite",
        "T": 128, "B": 1, "input_len": 128, "output_len": 128, "iter": 128, "moe_stages": 75,
        "trace_file": os.path.join(TRACES_DIR, "ms_task12_t128_l75.csv"), "bypass_cache": False, "scope": "full_depth"
    })
    specs.append({
        "config_id": "A2", "table": "A", "dataset": "MoE Suite",
        "T": 512, "B": 1, "input_len": 128, "output_len": 512, "iter": 512, "moe_stages": 75,
        "trace_file": os.path.join(TRACES_DIR, "ms_task12_t512_l75.csv"), "bypass_cache": False, "scope": "full_depth"
    })
    specs.append({
        "config_id": "A3", "table": "A", "dataset": "MoE Suite",
        "T": 1024, "B": 1, "input_len": 128, "output_len": 1024, "iter": 1024, "moe_stages": 75,
        "trace_file": os.path.join(TRACES_DIR, "ms_task12_t1024_l75.csv"), "bypass_cache": False, "scope": "full_depth"
    })
    specs.append({
        "config_id": "A4", "table": "A", "dataset": "MoE Suite",
        "T": 2048, "B": 1, "input_len": 128, "output_len": 2048, "iter": 2048, "moe_stages": 75,
        "trace_file": os.path.join(TRACES_DIR, "ms_task12_t2048_l75.csv"), "bypass_cache": False, "scope": "full_depth"
    })
    specs.append({
        "config_id": "A5", "table": "A", "dataset": "MoE Anywhere",
        "T": 128, "B": 1, "input_len": 128, "output_len": 128, "iter": 128, "moe_stages": 40,
        "trace_file": os.path.join(TRACES_DIR, "ma_p60_t128_l40.csv"), "bypass_cache": False, "scope": "full_depth"
    })
    specs.append({
        "config_id": "A6", "table": "A", "dataset": "MoE Anywhere",
        "T": 256, "B": 1, "input_len": 128, "output_len": 256, "iter": 256, "moe_stages": 40,
        "trace_file": os.path.join(TRACES_DIR, "ma_p60_t256_l40.csv"), "bypass_cache": False, "scope": "full_depth"
    })
    specs.append({
        "config_id": "A7", "table": "A", "dataset": "DeepSeek-style synthetic",
        "T": 128, "B": 1, "input_len": 128, "output_len": 128, "iter": 128, "moe_stages": 57,
        "trace_file": "", "bypass_cache": False, "scope": "full_depth"
    })
    specs.append({
        "config_id": "A8", "table": "A", "dataset": "DeepSeek-style synthetic",
        "T": 512, "B": 1, "input_len": 128, "output_len": 512, "iter": 512, "moe_stages": 57,
        "trace_file": "", "bypass_cache": False, "scope": "full_depth"
    })

    # Table B
    specs.append({
        "config_id": "B1", "table": "B", "dataset": "MoE Anywhere",
        "T": 128, "B": 2, "input_len": 128, "output_len": 128, "iter": 128, "moe_stages": 40,
        "trace_file": os.path.join(TRACES_DIR, "ma_b2_t128_l40.csv"), "bypass_cache": False, "scope": "full_depth"
    })
    specs.append({
        "config_id": "B2", "table": "B", "dataset": "MoE Anywhere",
        "T": 128, "B": 4, "input_len": 128, "output_len": 128, "iter": 128, "moe_stages": 40,
        "trace_file": os.path.join(TRACES_DIR, "ma_b4_t128_l40.csv"), "bypass_cache": False, "scope": "full_depth"
    })
    specs.append({
        "config_id": "B3", "table": "B", "dataset": "MoE Anywhere",
        "T": 256, "B": 2, "input_len": 128, "output_len": 256, "iter": 256, "moe_stages": 40,
        "trace_file": os.path.join(TRACES_DIR, "ma_b2_t256_l40.csv"), "bypass_cache": False, "scope": "full_depth"
    })
    specs.append({
        "config_id": "B4", "table": "B", "dataset": "MoE Anywhere",
        "T": 256, "B": 4, "input_len": 128, "output_len": 256, "iter": 256, "moe_stages": 40,
        "trace_file": os.path.join(TRACES_DIR, "ma_b4_t256_l40.csv"), "bypass_cache": False, "scope": "full_depth"
    })

    # Table C
    specs.append({
        "config_id": "C1", "table": "C", "dataset": "MoE Suite",
        "T": 128, "B": 1, "input_len": 512, "output_len": 128, "iter": 128, "moe_stages": 75,
        "trace_file": os.path.join(TRACES_DIR, "ms_task12_t128_l75.csv"), "bypass_cache": False, "scope": "full_depth"
    })
    specs.append({
        "config_id": "C2", "table": "C", "dataset": "MoE Suite",
        "T": 128, "B": 1, "input_len": 1024, "output_len": 128, "iter": 128, "moe_stages": 75,
        "trace_file": os.path.join(TRACES_DIR, "ms_task12_t128_l75.csv"), "bypass_cache": False, "scope": "full_depth"
    })
    specs.append({
        "config_id": "C3", "table": "C", "dataset": "MoE Suite",
        "T": 128, "B": 1, "input_len": 2048, "output_len": 128, "iter": 128, "moe_stages": 75,
        "trace_file": os.path.join(TRACES_DIR, "ms_task12_t128_l75.csv"), "bypass_cache": False, "scope": "full_depth"
    })

    # 2. Representative Screening Reference Baselines (2 stages, cache bypass)
    specs.append({
        "config_id": "A1-rep", "table": "A-rep", "dataset": "MoE Suite",
        "T": 128, "B": 1, "input_len": 128, "output_len": 128, "iter": 128, "moe_stages": 2,
        "trace_file": os.path.join(TRACES_DIR, "ms_task12_t128_l2.csv"), "bypass_cache": True, "scope": "representative_screening"
    })
    specs.append({
        "config_id": "A5-rep", "table": "A-rep", "dataset": "MoE Anywhere",
        "T": 128, "B": 1, "input_len": 128, "output_len": 128, "iter": 128, "moe_stages": 2,
        "trace_file": os.path.join(TRACES_DIR, "ma_p60_t128_l2.csv"), "bypass_cache": True, "scope": "representative_screening"
    })
    specs.append({
        "config_id": "A7-rep", "table": "A-rep", "dataset": "DeepSeek-style synthetic",
        "T": 128, "B": 1, "input_len": 128, "output_len": 128, "iter": 128, "moe_stages": 2,
        "trace_file": "", "bypass_cache": True, "scope": "representative_screening"
    })

    return specs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline configurations")
    parser.add_argument("--configs", type=str, default="all", help="Comma-separated list of config IDs to run, or 'all'")
    args = parser.parse_args()

    # Assert AST zero ternary operators
    with open(__file__, "r") as f:
        tree = ast.parse(f.read(), filename=__file__)
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            raise ValueError(f"Ternary operator found at line {node.lineno}: {ast.dump(node)}")
    print("[AST CHECK] Verified zero ternary operators (ast.IfExp == 0).")

    all_specs = build_experiment_specs()
    if args.configs == "all":
        specs = all_specs
    else:
        target_ids = [c.strip() for c in args.configs.split(",") if c.strip()]
        specs = [s for s in all_specs if s["config_id"] in target_ids]

    manifest_path = os.path.join(DOCS_DIR, "FINAL_BASELINE_EXECUTION_MANIFEST.csv")
    res_path = os.path.join(DOCS_DIR, "FINAL_BASELINE_RESULTS.csv")
    met_path = os.path.join(DOCS_DIR, "FINAL_BASELINE_METRICS.csv")

    manifest_dict = {}
    if os.path.exists(manifest_path):
        m_df = pd.read_csv(manifest_path)
        for _, row in m_df.iterrows():
            manifest_dict[row["config_id"]] = row.to_dict()
    else:
        for s in all_specs:
            manifest_dict[s["config_id"]] = {
                "config_id": s["config_id"],
                "table": s["table"],
                "dataset": s["dataset"],
                "sequence_length_T": s["T"],
                "batch_size_B": s["B"],
                "input_len": s["input_len"],
                "output_len": s["output_len"],
                "iter": s["iter"],
                "moe_stages": s["moe_stages"],
                "total_layers": s["moe_stages"] + 3,
                "scope": s["scope"],
                "bypass_cache": s["bypass_cache"],
                "trace_file": s["trace_file"],
                "status": "QUEUED"
            }

    results_dict = {}
    if os.path.exists(res_path):
        r_df = pd.read_csv(res_path)
        for _, row in r_df.iterrows():
            results_dict[row["config_id"]] = row.to_dict()

    metrics_dict = {}
    if os.path.exists(met_path):
        met_df = pd.read_csv(met_path)
        for _, row in met_df.iterrows():
            metrics_dict[row["config_id"]] = row.to_dict()

    print(f"[CAMPAIGN START] Total configurations to execute: {len(specs)}")

    total_campaign_start = time.time()
    for spec in specs:
        if spec["config_id"] in manifest_dict:
            manifest_dict[spec["config_id"]]["status"] = "RUNNING"
        else:
            manifest_dict[spec["config_id"]] = {
                "config_id": spec["config_id"],
                "table": spec["table"],
                "dataset": spec["dataset"],
                "sequence_length_T": spec["T"],
                "batch_size_B": spec["B"],
                "input_len": spec["input_len"],
                "output_len": spec["output_len"],
                "iter": spec["iter"],
                "moe_stages": spec["moe_stages"],
                "total_layers": spec["moe_stages"] + 3,
                "scope": spec["scope"],
                "bypass_cache": spec["bypass_cache"],
                "trace_file": spec["trace_file"],
                "status": "RUNNING"
            }
        pd.DataFrame(list(manifest_dict.values())).to_csv(manifest_path, index=False)
        shutil.copy(manifest_path, os.path.join(MIRROR_DOCS_DIR, "FINAL_BASELINE_EXECUTION_MANIFEST.csv"))

        res = execute_configuration(spec)
        manifest_dict[spec["config_id"]]["status"] = res["status"]

        if res["status"] == "COMPLETED":
            results_dict[res["config_id"]] = {
                "config_id": res["config_id"],
                "table": res["table"],
                "dataset": res["dataset"],
                "sequence_length_T": res["sequence_length_T"],
                "batch_size_B": res["batch_size_B"],
                "input_len": res["input_len"],
                "iter": res["iter"],
                "moe_stages": res["moe_stages"],
                "scope": res["scope"],
                "simulated_time_ms": round(res["end_to_end_sim_time_ms"], 2),
                "host_time_sec": round(res["host_wall_time_sec"], 2),
                "token_throughput": round(res["token_throughput"], 3),
                "expert_hbm_hit_rate_pct": round(res["expert_level_hbm_hit_rate"] * 100.0, 2),
                "matrix_hbm_hit_rate_pct": round(res["matrix_level_hbm_hit_rate"] * 100.0, 2),
                "ddr5_traffic_mb": round(res["ddr5_bytes_transferred"] / (1024 * 1024), 2),
                "memory_stall_ms": round(res["exposed_memory_stall_ms"], 2),
                "natural_lru_evictions": res["natural_lru_evictions"],
                "status": res["status"]
            }

            metrics_dict[res["config_id"]] = {
                "config_id": res["config_id"],
                "table": res["table"],
                "dataset": res["dataset"],
                "scope": res["scope"],
                "1_actual_expert_demands": res["actual_expert_demands"],
                "2_unique_experts": res["unique_experts"],
                "3_temporal_consecutive_overlap": round(res["temporal_consecutive_overlap"], 4),
                "4_cross_request_overlap": round(res["cross_request_overlap"], 4),
                "5_expert_level_hbm_hit_rate": round(res["expert_level_hbm_hit_rate"], 4),
                "6_matrix_level_hbm_hit_rate": round(res["matrix_level_hbm_hit_rate"], 4),
                "7_ddr5_demand_misses": res["ddr5_demand_misses"],
                "8_ddr5_bytes_transferred": res["ddr5_bytes_transferred"],
                "9_ddr5_burst_count": res["ddr5_burst_count"],
                "10_expert_load_latency_ms": round(res["expert_load_latency_ms"], 2),
                "11_compute_time_ms": round(res["compute_time_ms"], 2),
                "12_exposed_memory_stall_ms": round(res["exposed_memory_stall_ms"], 2),
                "13_end_to_end_sim_time_ms": round(res["end_to_end_sim_time_ms"], 2),
                "14_per_token_latency_ms": round(res["per_token_latency_ms"], 2),
                "15_token_throughput": round(res["token_throughput"], 3),
                "16_hbm_occupancy_gb": round(res["hbm_occupancy_gb"], 2),
                "17_natural_lru_evictions": res["natural_lru_evictions"],
                "host_wall_time_sec": round(res["host_wall_time_sec"], 2)
            }

        # Save interim progress
        pd.DataFrame(list(manifest_dict.values())).to_csv(manifest_path, index=False)
        shutil.copy(manifest_path, os.path.join(MIRROR_DOCS_DIR, "FINAL_BASELINE_EXECUTION_MANIFEST.csv"))

        df_res = pd.DataFrame(list(results_dict.values()))
        df_res.to_csv(res_path, index=False)
        shutil.copy(res_path, os.path.join(MIRROR_DOCS_DIR, "FINAL_BASELINE_RESULTS.csv"))
        shutil.copy(res_path, os.path.join(OUTPUT_BASE, "FINAL_BASELINE_RESULTS.csv"))

        df_met = pd.DataFrame(list(metrics_dict.values()))
        df_met.to_csv(met_path, index=False)
        shutil.copy(met_path, os.path.join(MIRROR_DOCS_DIR, "FINAL_BASELINE_METRICS.csv"))
        shutil.copy(met_path, os.path.join(OUTPUT_BASE, "FINAL_BASELINE_METRICS.csv"))

    total_campaign_end = time.time()
    total_campaign_duration = total_campaign_end - total_campaign_start
    print(f"\n======================================================================")
    print(f"BATCH COMPLETE! Host Wall Time: {total_campaign_duration:.2f} seconds ({total_campaign_duration/3600.0:.2f} hours)")
    print(f"======================================================================")

    print(f"[DELIVERABLES SAVED]")
    print(f"  {res_path}")
    print(f"  {met_path}")
    print(f"  {manifest_path}")


if __name__ == "__main__":
    main()
