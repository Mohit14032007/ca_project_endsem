#!/usr/bin/env python3
"""
Phase 3 Focused 24-Run Campaign Runner.
Executes the Cartesian product of:
  - 4 Configurations (C1: Base LRU, C2: Base FA-LRU, C3: CSTP LRU, C4: CSTP FA-LRU)
  - 3 Capacities (CAP1: 10 GB, CAP2: 35 GiB, CAP3: 80 GB)
  - 2 Workloads (W2: B=4, W3: B=14) across all 40 MoE stages.
Strictly adheres to zero ternary operators (ast.IfExp == 0).
"""

import os
import sys
import ast
import time
import json
import subprocess
import pandas as pd

WORKSPACE = "/home/mohman/ca_project"
REPO_ROOT = os.path.join(WORKSPACE, "LLMSimulator")
PHASE3_ROOT = os.path.join(WORKSPACE, "mohman", "point_4", "phase_3_focused_24")

BIN_PATH = os.path.join(REPO_ROOT, "build", "run_deepseekv3_mixtral_style")
CONFIG_DIR = os.path.join(PHASE3_ROOT, "configs")
MANIFEST_DIR = os.path.join(PHASE3_ROOT, "manifests")
TRACES_DIR = os.path.join(PHASE3_ROOT, "traces")
LOG_DIR = os.path.join(PHASE3_ROOT, "logs")
RESULTS_DIR = os.path.join(PHASE3_ROOT, "raw_results")
CONSOLIDATED_DIR = os.path.join(PHASE3_ROOT, "consolidated")

DDR5_CFG = os.path.join(
    REPO_ROOT, "reproduction_A100", "with_off_chip_dram", "configs", "dram_config_DDR5_retry.yaml"
)
HBM_CFG = os.path.join(REPO_ROOT, "build", "dram_config_HBM3_80GB.yaml")

# Matrix Definitions
WORKLOADS = {
    "w2": {
        "batch_size": 4,
        "trace_file": os.path.join(TRACES_DIR, "w2.csv"),
        "prompts": "60-63 (tools, summarize)",
        "expected_routed": 4 * 16 * 40 * 8, # 20,480
    },
    "w3": {
        "batch_size": 14,
        "trace_file": os.path.join(TRACES_DIR, "w3.csv"),
        "prompts": "60-73 (100% held-out test split)",
        "expected_routed": 14 * 16 * 40 * 8, # 71,680
    }
}

CAPACITIES = {
    "cap1": {
        "label": "10 GB",
        "bytes": 10000000000,
        "experts": 113,
    },
    "cap2": {
        "label": "35 GiB",
        "bytes": 37580963840,
        "experts": 426,
    },
    "cap3": {
        "label": "80 GB",
        "bytes": 80000000000,
        "experts": 908,
    }
}

CONFIGS = {
    "c1": {
        "prefetch": False,
        "mode": "none",
        "eviction": "lru",
        "desc": "Baseline LRU (No Prefetch)",
    },
    "c2": {
        "prefetch": False,
        "mode": "none",
        "eviction": "fa_lru",
        "desc": "FA-LRU (W=16, No Prefetch)",
    },
    "c3": {
        "prefetch": True,
        "mode": "hardware_async",
        "eviction": "lru",
        "desc": "CSTP Prefetch + Standard LRU",
    },
    "c4": {
        "prefetch": True,
        "mode": "hardware_async",
        "eviction": "fa_lru",
        "desc": "CSTP Prefetch + FA-LRU (W=16)",
    }
}


def verify_zero_ternary(file_path):
    with open(file_path, "r") as f:
        tree = ast.parse(f.read(), filename=file_path)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            count += 1
    if count > 0:
        raise ValueError(f"Found {count} ternary operators in {file_path}!")
    print(f"[AST_CHECK] Verified 0 ternary operators in {file_path}")


def generate_yaml(run_id, w_key, cap_key, cfg_key):
    w_info = WORKLOADS[w_key]
    cap_info = CAPACITIES[cap_key]
    cfg_info = CONFIGS[cfg_key]

    out_csv = os.path.join(RESULTS_DIR, run_id, "csv") + "/"
    out_raw = os.path.join(RESULTS_DIR, run_id, "raw") + "/"
    out_traces = os.path.join(RESULTS_DIR, run_id, "traces") + "/"

    p_enabled_str = "false"
    if cfg_info["prefetch"]:
        p_enabled_str = "true"

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
        f"    expert_cache_capacity_bytes: {cap_info['bytes']}",
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
        f"  max_batch_size: {w_info['batch_size']}",
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
        f"  trace_file: {w_info['trace_file']}",
    ]
    prefetch_lines = []
    if cfg_info["prefetch"]:
        prefetch_lines.extend([
            "  prefetching:",
            "    enabled: true",
            f"    mode: {cfg_info['mode']}",
            f"    eviction_policy: {cfg_info['eviction']}",
            "    fa_lru_window_size: 16",
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
        prefetch_lines.extend([
            "  prefetching:",
            "    enabled: false",
            "    mode: none",
            f"    eviction_policy: {cfg_info['eviction']}",
            "    fa_lru_window_size: 16",
        ])

    lines.extend(prefetch_lines)
    lines.extend([
        "",
        "log:",
        "  print_log: false",
        "  export_gantt: false",
        f"  output_directory: {out_csv}",
        f"  gantt_directory: {out_raw}",
        f"  traces_directory: {out_traces}",
        ""
    ])

    cfg_path = os.path.join(CONFIG_DIR, f"{run_id}.yaml")
    with open(cfg_path, "w") as f:
        f.write("\n".join(lines))
    return cfg_path


def run_single_simulation(run_id, w_key, cap_key, cfg_key):
    cfg_path = generate_yaml(run_id, w_key, cap_key, cfg_key)
    log_path = os.path.join(LOG_DIR, f"{run_id}.log")
    run_traces_dir = os.path.join(RESULTS_DIR, run_id, "traces")

    os.makedirs(os.path.join(RESULTS_DIR, run_id, "csv"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, run_id, "raw"), exist_ok=True)
    os.makedirs(run_traces_dir, exist_ok=True)

    cap_info = CAPACITIES[cap_key]
    cfg_info = CONFIGS[cfg_key]

    env = os.environ.copy()
    env["USE_EVENT_DRIVEN_DRAM"] = "true"
    env["DDR5_BANDWIDTH_GBPS"] = "64.0"
    env["TRACE_DIR"] = run_traces_dir
    env["PREDICTION_OUTPUT_DIR"] = run_traces_dir
    env["DEBUG_CACHE_CAPACITY"] = str(cap_info["bytes"])
    env["EVICTION_POLICY"] = cfg_info["eviction"]
    env["FA_LRU_WINDOW_SIZE"] = "16"

    cmd = [BIN_PATH, cfg_path]
    print(f"\n[EXEC] Running {run_id} ({w_key.upper()}, {cap_info['label']}, {cfg_key.upper()})...")
    t0 = time.time()
    with open(log_path, "w") as log_f:
        res = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - t0

    if res.returncode != 0:
        print(f"  [ERROR] {run_id} FAILED with return code {res.returncode}. Log: {log_path}")
        return False, elapsed

    print(f"  [SUCCESS] {run_id} completed in {elapsed:.2f}s (Exit code 0)")
    return True, elapsed


def validate_and_parse_run(run_id, w_key, cap_key, cfg_key, host_runtime):
    summary_path = os.path.join(RESULTS_DIR, run_id, "traces", "prefetch_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing prefetch_summary.json for {run_id}")

    with open(summary_path) as f:
        data = json.load(f)

    hbm = data["hbm_management"]
    mem = data["memory_traffic"]
    stall = data["exposed_stall_and_execution"]
    tok_lat = data["token_latency"]
    cov = data.get("prefetch_coverage", {})
    timely = data.get("timely_and_late_rates", {})
    unused = data.get("unused_prefetches", {})

    cap_bytes = CAPACITIES[cap_key]["bytes"]

    # 1. Capacity Conservation
    final_res_bytes = hbm.get("final_hbm_resident_bytes", 0)
    if final_res_bytes == 0:
        # Read from log if not in json
        log_path = os.path.join(LOG_DIR, f"{run_id}.log")
        with open(log_path) as lf:
            for line in lf:
                if "Final Resident Memory:" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        val_str = parts[1].strip().split()[0]
                        final_res_bytes = int(val_str)
    if final_res_bytes > cap_bytes:
        raise ValueError(f"Capacity violated in {run_id}: {final_res_bytes} > {cap_bytes}")

    # 2. Eviction Conservation
    tot_evict = hbm["total_evictions"]
    unconsumed = hbm["unconsumed_prefetch_evictions"]
    demand_evict = hbm["demand_resident_evictions"]
    pabi_evict = hbm["pabi_speculative_evictions"]
    if tot_evict != (unconsumed + demand_evict + pabi_evict):
        raise ValueError(f"Eviction mutual exclusion failed in {run_id}: {tot_evict} != {unconsumed} + {demand_evict} + {pabi_evict}")

    # 3. Prefetch-induced attribution
    induced = hbm["prefetch_induced_evictions"]
    if induced > demand_evict:
        raise ValueError(f"Prefetch induced > demand resident in {run_id}: {induced} > {demand_evict}")

    # 4. Traffic Conservation
    tot_bytes = mem["total_ddr5_bytes_transferred"]
    dem_bytes = mem["demand_bytes_transferred"]
    spec_bytes = mem["speculative_bytes_transferred"]
    if tot_bytes != (dem_bytes + spec_bytes):
        raise ValueError(f"Traffic conservation failed in {run_id}: {tot_bytes} != {dem_bytes} + {spec_bytes}")

    # Units: Decimal GB (10^9) and Binary GiB (2^30)
    tot_gb = tot_bytes / 1e9
    tot_gib = tot_bytes / (1024**3)
    dem_gb = dem_bytes / 1e9
    dem_gib = dem_bytes / (1024**3)
    spec_gb = spec_bytes / 1e9
    spec_gib = spec_bytes / (1024**3)

    stall_ms = stall["exposed_gpu_stall_ns"] / 1e6
    mean_tok_lat_ms = tok_lat["mean_ns"] / 1e6
    p50_tok_lat_ms = tok_lat["p50_ns"] / 1e6
    p95_tok_lat_ms = tok_lat["p95_ns"] / 1e6
    tok_per_sec = tok_lat["simulated_tokens_per_second"]
    sim_time_ms = stall["total_simulated_execution_ns"] / 1e6

    row = {
        "run_id": run_id,
        "workload": w_key,
        "capacity_id": cap_key,
        "capacity_label": CAPACITIES[cap_key]["label"],
        "capacity_bytes": cap_bytes,
        "config_id": cfg_key,
        "config_desc": CONFIGS[cfg_key]["desc"],
        "eviction_policy": CONFIGS[cfg_key]["eviction"],
        "prefetch_enabled": CONFIGS[cfg_key]["prefetch"],
        # HBM Metrics
        "coalesced_actual_demands": hbm["coalesced_actual_demands"],
        "routed_actual_demands": hbm["routed_actual_demands"],
        "tensor_actual_demands": hbm["tensor_actual_demands"],
        "coalesced_resident_hits": hbm["coalesced_resident_hits"],
        "coalesced_fresh_demand_misses": hbm["coalesced_fresh_demand_misses"],
        "coalesced_hbm_hit_rate": hbm["coalesced_hbm_hit_rate"],
        "total_evictions": tot_evict,
        "unconsumed_prefetch_evictions": unconsumed,
        "demand_resident_evictions": demand_evict,
        "pabi_speculative_evictions": pabi_evict,
        "prefetch_induced_evictions": induced,
        "reloaded_experts_count": hbm["reloaded_experts_count"],
        "final_resident_bytes": final_res_bytes,
        # Memory Traffic
        "total_ddr5_bytes": tot_bytes,
        "total_ddr5_gb": tot_gb,
        "total_ddr5_gib": tot_gib,
        "demand_bytes": dem_bytes,
        "demand_gb": dem_gb,
        "demand_gib": dem_gib,
        "speculative_bytes": spec_bytes,
        "speculative_gb": spec_gb,
        "speculative_gib": spec_gib,
        "ddr5_bus_utilization": mem.get("ddr5_bus_utilization", 0.0),
        # Prefetch Effectiveness
        "prefetch_coverage": cov.get("overall_coverage", 0.0),
        "timely_resident_hits": timely.get("timely_resident_hits", 0),
        "late_inflight_attachments": timely.get("late_inflight_attachments", 0),
        "unused_completed_prefetches": unused.get("prefetches_completed_unused", 0),
        "prefetches_evicted_before_demand": unused.get("prefetches_evicted_before_demand", 0),
        "transfers_cancelled": timely.get("transfers_cancelled", 0),
        # Performance
        "exposed_gpu_stall_ms": stall_ms,
        "mean_token_latency_ms": mean_tok_lat_ms,
        "p50_token_latency_ms": p50_tok_lat_ms,
        "p95_token_latency_ms": p95_tok_lat_ms,
        "simulated_execution_time_ms": sim_time_ms,
        "simulated_tokens_per_second": tok_per_sec,
        "host_wall_clock_runtime_s": host_runtime,
        "status": "VALID",
    }
    return row


def generate_paired_comparisons(all_df):
    rows = []
    # Pairs to compare:
    # 1. C2 vs C1: FA-LRU vs LRU (Baseline)
    # 2. C4 vs C3: FA-LRU vs LRU (CSTP)
    # 3. C3 vs C1: CSTP vs Baseline (LRU)
    # 4. C4 vs C2: CSTP vs Baseline (FA-LRU)
    pair_defs = [
        ("C2_vs_C1", "c2", "c1", "FA-LRU impact without prefetch"),
        ("C4_vs_C3", "c4", "c3", "FA-LRU impact with CSTP"),
        ("C3_vs_C1", "c3", "c1", "CSTP impact under LRU"),
        ("C4_vs_C2", "c4", "c2", "CSTP impact under FA-LRU"),
    ]

    metrics_to_diff = [
        "coalesced_hbm_hit_rate",
        "coalesced_resident_hits",
        "coalesced_fresh_demand_misses",
        "total_evictions",
        "demand_resident_evictions",
        "reloaded_experts_count",
        "total_ddr5_gb",
        "demand_gb",
        "speculative_gb",
        "exposed_gpu_stall_ms",
        "mean_token_latency_ms",
        "simulated_tokens_per_second",
    ]

    for w_key in ["w2", "w3"]:
        for cap_key in ["cap1", "cap2", "cap3"]:
            sub = all_df[(all_df["workload"] == w_key) & (all_df["capacity_id"] == cap_key)]
            for pair_id, cfg_a, cfg_b, desc in pair_defs:
                row_a = sub[sub["config_id"] == cfg_a]
                row_b = sub[sub["config_id"] == cfg_b]
                if row_a.empty or row_b.empty:
                    continue
                ra = row_a.iloc[0]
                rb = row_b.iloc[0]

                entry = {
                    "comparison_id": pair_id,
                    "description": desc,
                    "workload": w_key,
                    "capacity_id": cap_key,
                    "capacity_label": CAPACITIES[cap_key]["label"],
                    "config_A": cfg_a,
                    "config_B": cfg_b,
                }
                for m in metrics_to_diff:
                    val_a = ra[m]
                    val_b = rb[m]
                    delta = val_a - val_b
                    pct_delta = 0.0
                    if val_b != 0.0:
                        pct_delta = (delta / val_b) * 100.0
                    entry[f"delta_{m}"] = delta
                    entry[f"pct_delta_{m}"] = pct_delta
                rows.append(entry)

    pair_df = pd.DataFrame(rows)
    out_csv = os.path.join(CONSOLIDATED_DIR, "paired_comparisons.csv")
    pair_df.to_csv(out_csv, index=False)
    print(f"[CONSOLIDATION] Paired comparisons saved to {out_csv}")
    return pair_df


def main():
    verify_zero_ternary(__file__)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CONSOLIDATED_DIR, exist_ok=True)

    # 1. Generate Manifest
    manifest_rows = []
    run_matrix = []

    # Groups:
    # G1: W2, CAP1 | G2: W2, CAP2 | G3: W2, CAP3
    # G4: W3, CAP1 | G5: W3, CAP2 | G6: W3, CAP3
    group_map = {
        ("w2", "cap1"): "G1",
        ("w2", "cap2"): "G2",
        ("w2", "cap3"): "G3",
        ("w3", "cap1"): "G4",
        ("w3", "cap2"): "G5",
        ("w3", "cap3"): "G6",
    }

    for w_key in ["w2", "w3"]:
        for cap_key in ["cap1", "cap2", "cap3"]:
            gid = group_map[(w_key, cap_key)]
            for cfg_key in ["c1", "c2", "c3", "c4"]:
                run_id = f"{w_key}_{cap_key}_{cfg_key}"
                run_matrix.append((run_id, gid, w_key, cap_key, cfg_key))

                manifest_rows.append({
                    "run_id": run_id,
                    "group_id": gid,
                    "workload": w_key,
                    "batch_size": WORKLOADS[w_key]["batch_size"],
                    "num_tokens": 16,
                    "num_stages": 40,
                    "capacity_id": cap_key,
                    "capacity_label": CAPACITIES[cap_key]["label"],
                    "capacity_bytes": CAPACITIES[cap_key]["bytes"],
                    "config_id": cfg_key,
                    "config_desc": CONFIGS[cfg_key]["desc"],
                    "eviction_policy": CONFIGS[cfg_key]["eviction"],
                    "prefetch_mode": CONFIGS[cfg_key]["mode"],
                    "trace_file": WORKLOADS[w_key]["trace_file"],
                    "prompts": WORKLOADS[w_key]["prompts"],
                    "random_seed": 42,
                    "status": "PENDING"
                })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(MANIFEST_DIR, "campaign_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"[MANIFEST] Generated campaign manifest with {len(run_matrix)} runs: {manifest_path}")

    # 2. Execute 24 Runs
    results_rows = []
    start_time = time.time()
    for idx, (run_id, gid, w_key, cap_key, cfg_key) in enumerate(run_matrix, 1):
        print(f"\n======================================================================")
        print(f"RUN {idx}/24: {run_id} (Group {gid})")
        print(f"======================================================================")
        ok, host_time = run_single_simulation(run_id, w_key, cap_key, cfg_key)
        if not ok:
            print(f"[FATAL] Run {run_id} failed. Aborting campaign.")
            sys.exit(1)

        row = validate_and_parse_run(run_id, w_key, cap_key, cfg_key, host_time)
        results_rows.append(row)

    total_time = time.time() - start_time
    print(f"\n======================================================================")
    print(f"[ALL RUNS COMPLETE] Successfully executed 24/24 runs in {total_time:.2f}s!")
    print(f"======================================================================")

    # 3. Save Consolidated CSV
    all_df = pd.DataFrame(results_rows)
    all_csv_path = os.path.join(CONSOLIDATED_DIR, "all_24_runs.csv")
    all_df.to_csv(all_csv_path, index=False)
    print(f"[CONSOLIDATION] Saved all 24 runs to {all_csv_path}")

    # 4. Paired Comparisons
    generate_paired_comparisons(all_df)

    # 5. Save Summary JSON
    summary_dict = {
        "total_runs": 24,
        "completed_runs": len(results_rows),
        "failed_runs": 0,
        "total_campaign_wall_clock_time_s": total_time,
        "runs": results_rows
    }
    summary_json_path = os.path.join(CONSOLIDATED_DIR, "experiment_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"[CONSOLIDATION] Saved campaign summary JSON to {summary_json_path}")


if __name__ == "__main__":
    main()
