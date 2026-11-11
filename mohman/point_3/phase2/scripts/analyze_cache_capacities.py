#!/usr/bin/env python3
"""
Cache Capacity and Working-Set Sensitivity Profiler for Phase 2.
Analyzes eviction counts, reload counts, HBM hit rates, and prefetch behavior
across various HBM cache capacities (e.g., 20 GB, 30 GB, 40 GB, 80 GB).

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


def run_cache_experiment(exp_name, predictor, cache_bytes, num_tokens=4):
    cfg_dir = os.path.join(PHASE2_ROOT, "experiments", "configs", "cache_sweep")
    raw_dir = os.path.join(PHASE2_ROOT, "experiments", "raw", "cache_sweep", exp_name)
    csv_dir = os.path.join(raw_dir, "csv")
    traces_dir = os.path.join(raw_dir, "traces")
    log_path = os.path.join(raw_dir, f"{exp_name}.log")

    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(traces_dir, exist_ok=True)

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
        "  max_batch_size: 1",
        "  max_process_token: 0",
        "",
        "simulation:",
        "  data: moe_anywhere",
        "  input_len: 128",
        f"  output_len: {num_tokens}",
        "  precision_byte: 2",
        "  skewness: 0.0",
        f"  iter: {num_tokens}",
        "  injection_rate: 0",
        "  exit_out_of_memory: false",
        "  mem_cap_limit: false",
        f"  trace_file: {TRACE_FILE}",
    ]

    if predictor == "baseline":
        yaml_lines.extend([
            "  prefetching:",
            "    enabled: false",
            "    mode: none",
        ])
    elif predictor == "cstp":
        yaml_lines.extend([
            "  prefetching:",
            "    enabled: true",
            "    mode: hardware_async",
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
    elif predictor == "stmoe":
        yaml_lines.extend([
            "  prefetching:",
            "    enabled: true",
            "    mode: hardware_async",
            "    scheduling:",
            "      scheduling_mode: forward_lookahead",
            "      prediction_horizon: 1",
            "    candidate_selection:",
            "      budget: 4",
            "    predictor:",
            "      type: ST_MOE",
            "      prediction_horizon: 1",
            "      budget: 4",
            "      confidence_threshold: 2",
            "      online_adaptation: true",
        ])

    yaml_lines.extend([
        "",
        "log:",
        "  print_log: false",
        "  export_gantt: false",
        f"  output_directory: {csv_dir}/",
        f"  gantt_directory: {raw_dir}/",
        f"  traces_directory: {traces_dir}/",
        "",
    ])

    cfg_path = os.path.join(cfg_dir, f"{exp_name}.yaml")
    with open(cfg_path, "w") as f:
        f.write("\n".join(yaml_lines))

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
        print(f"[FAILED] {exp_name} exited with code {res.returncode}")
        return None

    # Parse expert_cache.csv
    cache_csv = os.path.join(traces_dir, "expert_cache.csv")
    insert_count = 0
    evict_count = 0
    hit_count = 0
    if os.path.exists(cache_csv):
        cdf = pd.read_csv(cache_csv)
        actions = cdf["action"].value_counts().to_dict()
        insert_count = actions.get("INSERT", 0)
        evict_count = actions.get("EVICT", 0)
        hit_count = actions.get("HIT", 0)

    # Parse experiment_summary.json if exists
    sum_json = os.path.join(traces_dir, "experiment_summary.json")
    cov_pct = 0.0
    timely_pct = 0.0
    late_pct = 0.0
    hbm_hit_pct = 0.0
    dispatches = 0
    if os.path.exists(sum_json):
        with open(sum_json, "r") as f:
            sdata = json.load(f)
        cov = sdata.get("prefetch_coverage", {})
        cov_pct = cov.get("prefetch_coverage_pct", 0.0)
        timely_late = sdata.get("timely_and_late_rates", {})
        timely_hits = timely_late.get("timely_resident_hits", 0)
        late_hits = timely_late.get("late_inflight_attachments", 0)
        matched = timely_hits + late_hits
        if matched > 0:
            timely_pct = (timely_hits / matched) * 100.0
            late_pct = (late_hits / matched) * 100.0
        hbm_data = sdata.get("hbm_hit_rates", {})
        hbm_hit_pct = hbm_data.get("hbm_demand_hit_rate_pct", 0.0)
        traff = sdata.get("memory_traffic", {})
        dispatches = traff.get("speculative_transfers_count", 0)

    res_dict = {
        "exp_name": exp_name,
        "predictor": predictor,
        "cache_gb": cache_bytes / (1024**3),
        "host_sec": elapsed,
        "inserts": insert_count,
        "evicts": evict_count,
        "hits": hit_count,
        "hbm_hit_pct": hbm_hit_pct,
        "dispatches": dispatches,
        "coverage_pct": cov_pct,
        "timely_pct": timely_pct,
        "late_pct": late_pct,
    }
    print(
        f"[DONE] {exp_name:25s} | Cache: {res_dict['cache_gb']:5.1f} GB | "
        f"Inserts: {insert_count:4d} | Evicts: {evict_count:4d} | "
        f"Hits: {hit_count:4d} | HBM Hit: {hbm_hit_pct:5.1f}% | "
        f"Dispatches: {dispatches:3d} | Cov: {cov_pct:5.2f}%"
    )
    return res_dict


def main():
    print("======================================================================")
    print("HBM CACHE CAPACITY SWEEP ON 40 MOE STAGES (4 TOKENS, B=1)")
    print("======================================================================")

    # Test capacities:
    # 80 GB = 85899345920
    # 50 GB = 53687091200
    # 35 GB = 37580963840
    # 25 GB = 26843545600
    # 15 GB = 16106127360
    capacities = [
        ("80gb", 85899345920),
        ("50gb", 53687091200),
        ("35gb", 37580963840),
        ("25gb", 26843545600),
    ]

    results = []
    for cap_label, cap_bytes in capacities:
        for pred in ["baseline", "cstp"]:
            exp_id = f"sweep_{pred}_{cap_label}"
            r = run_cache_experiment(exp_id, pred, cap_bytes, num_tokens=4)
            if r is not None:
                results.append(r)

    out_csv = os.path.join(PHASE2_ROOT, "results", "cache_sweep_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    main()

