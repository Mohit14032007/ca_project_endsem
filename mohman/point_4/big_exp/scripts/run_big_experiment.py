#!/usr/bin/env python3
"""
Big Experiment Runner: LRU vs FA-LRU (W=64) Under Large MoE Workloads.
Evaluates 16 simulation runs across W3 (B=14, T=128) and W4 (B=32, T=128),
4 configurations (C1, C2, C3, C4), and 2 repetitions.

Enforces:
- Zero ternary operators (ast.IfExp == 0)
- Strict capacity, eviction, and traffic conservation checks
- Collection of all standard metrics and FA-LRU C.4 diagnostics
- Detailed logging and error trapping
"""

import os
import sys
import ast
import json
import time
import subprocess
import pandas as pd

WORKSPACE = "/home/mohman/ca_project"
BIG_EXP_ROOT = os.path.join(WORKSPACE, "mohman", "point_4", "big_exp")
CONFIG_DIR = os.path.join(BIG_EXP_ROOT, "configs")
RESULTS_DIR = os.path.join(BIG_EXP_ROOT, "results")
LOG_DIR = os.path.join(RESULTS_DIR, "raw_logs")
TRACES_DIR = os.path.join(BIG_EXP_ROOT, "traces")

BIN_PATH = os.path.join(WORKSPACE, "LLMSimulator", "build", "run_deepseekv3_mixtral_style")

HBM_CAPACITY_BYTES = 37580963840  # 35 GiB
EXPERT_SIZE_BYTES = 88080384

WORKLOADS = {
    "w3": {
        "name": "W3",
        "batch_size": 14,
        "tokens": 128,
        "stages": 40,
        "top_k": 8,
        "trace_file": os.path.join(TRACES_DIR, "w3_b14_t128.csv"),
    },
    "w4": {
        "name": "W4",
        "batch_size": 32,
        "tokens": 128,
        "stages": 40,
        "top_k": 8,
        "trace_file": os.path.join(TRACES_DIR, "w4_b32_t128.csv"),
    }
}

CONFIGS = {
    "c1": {
        "desc": "Baseline Standard LRU (No Prefetch)",
        "prefetch": False,
        "mode": "none",
        "eviction": "lru",
        "window_size": 64
    },
    "c2": {
        "desc": "Baseline FA-LRU (W=64, No Prefetch)",
        "prefetch": False,
        "mode": "none",
        "eviction": "fa_lru",
        "window_size": 64
    },
    "c3": {
        "desc": "CSTP Prefetch + Standard LRU",
        "prefetch": True,
        "mode": "hardware_async",
        "eviction": "lru",
        "window_size": 64
    },
    "c4": {
        "desc": "CSTP Prefetch + FA-LRU (W=64)",
        "prefetch": True,
        "mode": "hardware_async",
        "eviction": "fa_lru",
        "window_size": 64
    }
}


def verify_zero_ternary(file_path):
    with open(file_path, "r") as f:
        tree = ast.parse(f.read())
    ternary_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.IfExp)]
    if len(ternary_nodes) > 0:
        raise ValueError(f"Found {len(ternary_nodes)} ternary operators in {file_path}")
    print(f"[AST_CHECK] Verified 0 ternary operators in {file_path}")


def generate_yaml(run_id, w_key, cfg_key):
    w_info = WORKLOADS[w_key]
    c_info = CONFIGS[cfg_key]

    raw_results_dir = os.path.join(RESULTS_DIR, "runs", run_id)
    csv_dir = os.path.join(raw_results_dir, "csv")
    raw_dir = os.path.join(raw_results_dir, "raw")
    traces_dir = os.path.join(raw_results_dir, "traces")

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(traces_dir, exist_ok=True)

    if c_info["prefetch"]:
        prefetch_block = f"""  prefetching:
    enabled: true
    mode: {c_info['mode']}
    eviction_policy: {c_info['eviction']}
    fa_lru_window_size: {c_info['window_size']}
    scheduling:
      scheduling_mode: forward_lookahead
      prediction_horizon: 1
    candidate_selection:
      budget: 4
    predictor:
      type: CSTP
      prediction_horizon: 1
      budget: 4
      alpha: 0.5
      smoothing_lambda: 1.0
      prior_epsilon: 0.1
      online_adaptation: true"""
    else:
        prefetch_block = f"""  prefetching:
    enabled: false
    mode: none
    eviction_policy: {c_info['eviction']}
    fa_lru_window_size: {c_info['window_size']}"""

    yaml_content = f"""model:
  model_name: deepseekV3
  num_layers: 43

system:
  gpu_gen: A100
  nvlink_gen: 3
  infiniband_gen: 800
  num_node: 1
  num_device: 1
  processor_type: GPU
  distribution:
    none_expert_tensor_degree: 1
    expert_tensor_degree: 1
  optimization:
    parallel_execution: 'off'
    hetero_subbatch: false
    disagg_system: false
    use_low_unit_moe_only: false
    use_ramulator: 'on'
    offchip_dram_cfg_path: /home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/configs/dram_config_DDR5_retry.yaml
    hbm_dram_cfg_path: /home/mohman/ca_project/LLMSimulator/build/dram_config_HBM3_80GB.yaml
    expert_cache_capacity_bytes: {HBM_CAPACITY_BYTES}
    compressed_kv: true
    use_absorb: true
    use_flash_mla: true
    use_flash_attention: true
    reuse_kv_cache: true
    kv_cache_reuse_rate: 0.0
    prefill_mode: false
    decode_mode: true
    use_hitting_queue: false

serving:
  max_batch_size: {w_info['batch_size']}
  max_process_token: 0

simulation:
  data: moe_anywhere
  input_len: 128
  output_len: {w_info['tokens']}
  precision_byte: 2
  skewness: 0.0
  iter: {w_info['tokens']}
  injection_rate: 0
  exit_out_of_memory: false
  mem_cap_limit: false
  trace_file: {w_info['trace_file']}
{prefetch_block}

log:
  print_log: false
  export_gantt: false
  output_directory: {csv_dir}/
  gantt_directory: {raw_dir}/
  traces_directory: {traces_dir}/
"""
    yaml_path = os.path.join(CONFIG_DIR, f"{run_id}.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return yaml_path


def run_single_simulation(run_id, w_key, cfg_key, rep):
    cfg_path = generate_yaml(run_id, w_key, cfg_key)
    log_path = os.path.join(LOG_DIR, f"{run_id}.log")
    run_traces_dir = os.path.join(RESULTS_DIR, "runs", run_id, "traces")

    c_info = CONFIGS[cfg_key]

    env = os.environ.copy()
    env["USE_EVENT_DRIVEN_DRAM"] = "true"
    env["DDR5_BANDWIDTH_GBPS"] = "64.0"
    env["TRACE_DIR"] = run_traces_dir
    env["PREDICTION_OUTPUT_DIR"] = run_traces_dir
    env["DEBUG_CACHE_CAPACITY"] = str(HBM_CAPACITY_BYTES)
    env["EVICTION_POLICY"] = c_info["eviction"]
    env["FA_LRU_WINDOW_SIZE"] = str(c_info["window_size"])

    cmd = [BIN_PATH, cfg_path]
    print(f"\n[EXEC] Running {run_id} ({w_key.upper()}, {cfg_key.upper()}, Rep {rep})...")
    t0 = time.time()
    with open(log_path, "w") as log_f:
        res = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - t0

    if res.returncode != 0:
        print(f"  [ERROR] {run_id} FAILED with return code {res.returncode}. Log: {log_path}")
        return False, elapsed

    print(f"  [SUCCESS] {run_id} completed in {elapsed:.2f}s (Exit code 0)")
    return True, elapsed


def validate_and_parse_run(run_id, w_key, cfg_key, rep, host_runtime):
    run_dir = os.path.join(RESULTS_DIR, "runs", run_id, "traces")
    summary_path = os.path.join(run_dir, "prefetch_summary.json")
    diag_path = os.path.join(run_dir, "fa_lru_diagnostics.json")

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

    # 1. Capacity Conservation
    final_res_bytes = hbm.get("final_hbm_resident_bytes", 0)
    if final_res_bytes == 0:
        log_path = os.path.join(LOG_DIR, f"{run_id}.log")
        with open(log_path) as lf:
            for line in lf:
                if "Resident Experts in HBM:" in line or "Resident Experts:" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        val_str = parts[-1].strip().split()[0]
                        final_res_bytes = int(val_str) * EXPERT_SIZE_BYTES
    if final_res_bytes > HBM_CAPACITY_BYTES:
        raise ValueError(f"Capacity violated in {run_id}: {final_res_bytes} > {HBM_CAPACITY_BYTES}")

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

    # 5. Denominators
    routed_demands = hbm["routed_actual_demands"]
    tensor_demands = hbm["tensor_actual_demands"]
    if tensor_demands != routed_demands * 3:
        raise ValueError(f"Tensor demands != 3 * routed demands in {run_id}: {tensor_demands} != 3 * {routed_demands}")

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

    # Load FA-LRU Diagnostics
    fa_diag = {}
    if os.path.exists(diag_path):
        with open(diag_path) as df:
            fa_diag = json.load(df)

    div_decisions = fa_diag.get("divergent_decisions", 0)
    prot_demands = fa_diag.get("protected_demand_experts_count", 0)
    uncons_victims = fa_diag.get("unconsumed_speculative_victims", 0)
    prot_reused = fa_diag.get("protected_experts_reused", 0)
    prot_never_reused = fa_diag.get("protected_experts_never_reused", 0)
    fallbacks = fa_diag.get("fallback_beyond_window_count", 0)
    avg_scan_cost = fa_diag.get("average_candidate_window_scan_cost", 0.0)
    max_scan_cost = fa_diag.get("max_candidate_window_scan_depth", 0)

    row = {
        "run_id": run_id,
        "workload": w_key,
        "repetition": rep,
        "batch_size": WORKLOADS[w_key]["batch_size"],
        "num_tokens": WORKLOADS[w_key]["tokens"],
        "num_stages": WORKLOADS[w_key]["stages"],
        "capacity_bytes": HBM_CAPACITY_BYTES,
        "capacity_gib": 35.0,
        "config_id": cfg_key,
        "config_desc": CONFIGS[cfg_key]["desc"],
        "eviction_policy": CONFIGS[cfg_key]["eviction"],
        "window_size": CONFIGS[cfg_key]["window_size"],
        "prefetch_enabled": CONFIGS[cfg_key]["prefetch"],
        # Demands & Hit Rates
        "coalesced_actual_demands": hbm["coalesced_actual_demands"],
        "routed_actual_demands": routed_demands,
        "tensor_actual_demands": tensor_demands,
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
        # DDR5 Memory Traffic
        "total_ddr5_bytes": tot_bytes,
        "total_ddr5_gb": tot_gb,
        "total_ddr5_gib": tot_gib,
        "demand_bytes": dem_bytes,
        "demand_gb": dem_gb,
        "demand_gib": dem_gib,
        "speculative_bytes": spec_bytes,
        "speculative_gb": spec_gb,
        "speculative_gib": spec_gib,
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
        # FA-LRU Specific Diagnostics (Section C.4)
        "fa_lru_divergent_decisions": div_decisions,
        "fa_lru_protected_demand_experts": prot_demands,
        "fa_lru_unconsumed_speculative_victims": uncons_victims,
        "fa_lru_protected_experts_reused": prot_reused,
        "fa_lru_protected_experts_never_reused": prot_never_reused,
        "fa_lru_fallback_beyond_window": fallbacks,
        "fa_lru_avg_window_scan_cost": avg_scan_cost,
        "fa_lru_max_window_scan_cost": max_scan_cost,
        "status": "VALID"
    }
    return row


def generate_paired_comparisons(all_df):
    rows = []
    # Comparison 1: W3 without prefetch (C2 vs C1)
    # Comparison 2: W3 with CSTP (C4 vs C3)
    # Comparison 3: W4 without prefetch (C2 vs C1)
    # Comparison 4: W4 with CSTP (C4 vs C3)
    pair_defs = [
        ("W3_NoPrefetch", "w3", "c2", "c1", "W3: FA-LRU W=64 vs LRU (No Prefetch)"),
        ("W3_CSTP", "w3", "c4", "c3", "W3: FA-LRU W=64 vs LRU (CSTP Enabled)"),
        ("W4_NoPrefetch", "w4", "c2", "c1", "W4: FA-LRU W=64 vs LRU (No Prefetch)"),
        ("W4_CSTP", "w4", "c4", "c3", "W4: FA-LRU W=64 vs LRU (CSTP Enabled)"),
    ]

    metrics = [
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
        "p95_token_latency_ms",
        "simulated_tokens_per_second",
        "fa_lru_divergent_decisions",
        "fa_lru_protected_demand_experts",
        "fa_lru_protected_experts_reused"
    ]

    for comp_id, w_key, cfg_A, cfg_B, desc in pair_defs:
        for rep in [1, 2]:
            sub_A = all_df[(all_df["workload"] == w_key) & (all_df["config_id"] == cfg_A) & (all_df["repetition"] == rep)]
            sub_B = all_df[(all_df["workload"] == w_key) & (all_df["config_id"] == cfg_B) & (all_df["repetition"] == rep)]

            if len(sub_A) == 0 or len(sub_B) == 0:
                continue

            r_A = sub_A.iloc[0]
            r_B = sub_B.iloc[0]

            comp_row = {
                "comparison_id": comp_id,
                "repetition": rep,
                "workload": w_key,
                "description": desc,
                "config_FA_LRU": cfg_A,
                "config_LRU": cfg_B,
            }

            for m in metrics:
                val_A = r_A[m]
                val_B = r_B[m]
                delta = val_A - val_B
                if abs(val_B) > 1e-9:
                    pct = (delta / val_B) * 100.0
                else:
                    pct = 0.0
                comp_row[f"delta_{m}"] = delta
                comp_row[f"pct_delta_{m}"] = pct

            rows.append(comp_row)

    comp_df = pd.DataFrame(rows)
    comp_path = os.path.join(RESULTS_DIR, "paired_comparisons.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"[CONSOLIDATION] Saved paired comparisons -> {comp_path}")
    return comp_df


def main():
    verify_zero_ternary(__file__)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Build 16-run execution matrix
    # 2 workloads x 4 configs x 2 repetitions
    run_matrix = []
    for w_key in ["w3", "w4"]:
        for cfg_key in ["c1", "c2", "c3", "c4"]:
            for rep in [1, 2]:
                run_id = f"{w_key}_{cfg_key}_r{rep}"
                run_matrix.append((run_id, w_key, cfg_key, rep))

    print(f"======================================================================")
    print(f"BIG EXPERIMENT CAMPAIGN: {len(run_matrix)} RUNS")
    print(f"Workloads: W3 (B=14, T=128), W4 (B=32, T=128)")
    print(f"Capacity: 35 GiB, Window Size: W=64, Repetitions: 2")
    print(f"======================================================================")

    # 1. Manifest Generation
    manifest_rows = []
    for run_id, w_key, cfg_key, rep in run_matrix:
        manifest_rows.append({
            "run_id": run_id,
            "workload": w_key,
            "repetition": rep,
            "batch_size": WORKLOADS[w_key]["batch_size"],
            "tokens": WORKLOADS[w_key]["tokens"],
            "stages": WORKLOADS[w_key]["stages"],
            "capacity_gib": 35.0,
            "config_id": cfg_key,
            "config_desc": CONFIGS[cfg_key]["desc"],
            "eviction_policy": CONFIGS[cfg_key]["eviction"],
            "window_size": CONFIGS[cfg_key]["window_size"],
            "prefetch_enabled": CONFIGS[cfg_key]["prefetch"],
            "trace_file": WORKLOADS[w_key]["trace_file"],
            "seed": 42
        })
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(RESULTS_DIR, "campaign_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"[MANIFEST] Generated campaign manifest ({len(run_matrix)} runs) -> {manifest_path}")

    # 2. Execute Runs
    results_rows = []
    start_total = time.time()
    for idx, (run_id, w_key, cfg_key, rep) in enumerate(run_matrix, 1):
        print(f"\n======================================================================")
        print(f"RUN {idx}/16: {run_id}")
        print(f"======================================================================")
        ok, host_time = run_single_simulation(run_id, w_key, cfg_key, rep)
        if not ok:
            print(f"[FATAL] Simulation {run_id} failed. Aborting campaign.")
            sys.exit(1)

        row = validate_and_parse_run(run_id, w_key, cfg_key, rep, host_time)
        results_rows.append(row)

    total_wall_clock = time.time() - start_total
    print(f"\n======================================================================")
    print(f"[CAMPAIGN COMPLETE] All 16 runs succeeded in {total_wall_clock:.2f}s!")
    print(f"======================================================================")

    # 3. Save Consolidated CSV
    all_df = pd.DataFrame(results_rows)
    all_csv_path = os.path.join(RESULTS_DIR, "all_runs.csv")
    all_df.to_csv(all_csv_path, index=False)
    print(f"[CONSOLIDATION] Saved all 16 runs -> {all_csv_path}")

    # 4. Save Paired Comparisons
    generate_paired_comparisons(all_df)

    # 5. Save Summary JSON
    summary_dict = {
        "total_planned_runs": 16,
        "completed_runs": len(results_rows),
        "failed_runs": 0,
        "total_campaign_wall_clock_time_s": total_wall_clock,
        "capacity_bytes": HBM_CAPACITY_BYTES,
        "capacity_gib": 35.0,
        "window_size": 64,
        "runs": results_rows
    }
    summary_json_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"[CONSOLIDATION] Saved experiment summary JSON -> {summary_json_path}")


if __name__ == "__main__":
    main()

