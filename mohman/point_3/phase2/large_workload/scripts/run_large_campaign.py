#!/usr/bin/env python3
"""
Full Campaign Execution Script for Point 3 Large Workloads.
Executes all 27 target workload runs across W1..W7 under 35 GB constrained HBM cache.
Evaluates:
  1. No-prefetch demand baseline
  2. Online-adaptive ST-MoE (H=1 and H=2)
  3. Online-adaptive CSTP (H=1 and H=2)

Calculates the Eight Mandatory Metrics, generates:
  - per_run_metrics.csv
  - batch_scaling.csv
  - token_scaling.csv
  - predictor_comparison.csv
  - prefetch_event_accounting.csv

STRICT RULE: Zero ternary operators (ast.IfExp == 0).
"""

import os
import sys
import json
import time
import subprocess
import argparse
import pandas as pd
import numpy as np

WORKSPACE = "/home/mohman/ca_project"
REPO_ROOT = os.path.join(WORKSPACE, "LLMSimulator")
BIN_PATH = os.path.join(REPO_ROOT, "build", "run_deepseekv3_mixtral_style")
LARGE_ROOT = os.path.join(WORKSPACE, "mohman", "point_3", "phase2", "large_workload")

CONFIG_DIR = os.path.join(LARGE_ROOT, "experiments", "configs", "campaign")
LOG_DIR = os.path.join(LARGE_ROOT, "experiments", "logs", "campaign")
RAW_DIR = os.path.join(LARGE_ROOT, "experiments", "raw", "campaign")
RESULTS_DIR = os.path.join(LARGE_ROOT, "results")
TRACES_DIR = os.path.join(LARGE_ROOT, "experiments", "traces")
MANIFEST_FILE = os.path.join(LARGE_ROOT, "experiments", "manifests", "workload_prompt_mapping.csv")

DDR5_CFG = os.path.join(REPO_ROOT, "reproduction_A100", "with_off_chip_dram", "configs", "dram_config_DDR5_retry.yaml")
HBM_CFG = os.path.join(REPO_ROOT, "build", "dram_config_HBM3_80GB.yaml")
CACHE_BYTES = 37580963840  # 35.00 GB

# 27 experimental runs
CAMPAIGN_RUNS = [
    # W1: B=14, T=4 (Held-out test only)
    {"run_id": "w1_baseline", "workload_id": "W1", "batch_size": 14, "tokens": 4, "predictor": "baseline", "horizon": 0, "budget": 0},
    {"run_id": "w1_stmoe_h2", "workload_id": "W1", "batch_size": 14, "tokens": 4, "predictor": "stmoe", "horizon": 2, "budget": 4},
    {"run_id": "w1_cstp_h2", "workload_id": "W1", "batch_size": 14, "tokens": 4, "predictor": "cstp", "horizon": 2, "budget": 4},
    {"run_id": "w1_stmoe_h1", "workload_id": "W1", "batch_size": 14, "tokens": 4, "predictor": "stmoe", "horizon": 1, "budget": 4},
    {"run_id": "w1_cstp_h1", "workload_id": "W1", "batch_size": 14, "tokens": 4, "predictor": "cstp", "horizon": 1, "budget": 4},

    # W2: B=14, T=16 (Held-out test only)
    {"run_id": "w2_baseline", "workload_id": "W2", "batch_size": 14, "tokens": 16, "predictor": "baseline", "horizon": 0, "budget": 0},
    {"run_id": "w2_stmoe_h2", "workload_id": "W2", "batch_size": 14, "tokens": 16, "predictor": "stmoe", "horizon": 2, "budget": 4},
    {"run_id": "w2_cstp_h2", "workload_id": "W2", "batch_size": 14, "tokens": 16, "predictor": "cstp", "horizon": 2, "budget": 4},
    {"run_id": "w2_stmoe_h1", "workload_id": "W2", "batch_size": 14, "tokens": 16, "predictor": "stmoe", "horizon": 1, "budget": 4},
    {"run_id": "w2_cstp_h1", "workload_id": "W2", "batch_size": 14, "tokens": 16, "predictor": "cstp", "horizon": 1, "budget": 4},

    # W3: B=32, T=16 (Mixed batch)
    {"run_id": "w3_baseline", "workload_id": "W3", "batch_size": 32, "tokens": 16, "predictor": "baseline", "horizon": 0, "budget": 0},
    {"run_id": "w3_stmoe_h2", "workload_id": "W3", "batch_size": 32, "tokens": 16, "predictor": "stmoe", "horizon": 2, "budget": 4},
    {"run_id": "w3_cstp_h2", "workload_id": "W3", "batch_size": 32, "tokens": 16, "predictor": "cstp", "horizon": 2, "budget": 4},

    # W4: B=48, T=16 (Mixed batch)
    {"run_id": "w4_baseline", "workload_id": "W4", "batch_size": 48, "tokens": 16, "predictor": "baseline", "horizon": 0, "budget": 0},
    {"run_id": "w4_stmoe_h2", "workload_id": "W4", "batch_size": 48, "tokens": 16, "predictor": "stmoe", "horizon": 2, "budget": 4},
    {"run_id": "w4_cstp_h2", "workload_id": "W4", "batch_size": 48, "tokens": 16, "predictor": "cstp", "horizon": 2, "budget": 4},

    # W5: B=70, T=16 (Mixed batch)
    {"run_id": "w5_baseline", "workload_id": "W5", "batch_size": 70, "tokens": 16, "predictor": "baseline", "horizon": 0, "budget": 0},
    {"run_id": "w5_stmoe_h2", "workload_id": "W5", "batch_size": 70, "tokens": 16, "predictor": "stmoe", "horizon": 2, "budget": 4},
    {"run_id": "w5_cstp_h2", "workload_id": "W5", "batch_size": 70, "tokens": 16, "predictor": "cstp", "horizon": 2, "budget": 4},
    {"run_id": "w5_stmoe_h1", "workload_id": "W5", "batch_size": 70, "tokens": 16, "predictor": "stmoe", "horizon": 1, "budget": 4},
    {"run_id": "w5_cstp_h1", "workload_id": "W5", "batch_size": 70, "tokens": 16, "predictor": "cstp", "horizon": 1, "budget": 4},

    # W6: B=70, T=32 (Mixed batch, long sequence)
    {"run_id": "w6_baseline", "workload_id": "W6", "batch_size": 70, "tokens": 32, "predictor": "baseline", "horizon": 0, "budget": 0},
    {"run_id": "w6_stmoe_h2", "workload_id": "W6", "batch_size": 70, "tokens": 32, "predictor": "stmoe", "horizon": 2, "budget": 4},
    {"run_id": "w6_cstp_h2", "workload_id": "W6", "batch_size": 70, "tokens": 32, "predictor": "cstp", "horizon": 2, "budget": 4},

    # W7: B=70, T=64 (Mixed batch, maximum sequence)
    {"run_id": "w7_baseline", "workload_id": "W7", "batch_size": 70, "tokens": 64, "predictor": "baseline", "horizon": 0, "budget": 0},
    {"run_id": "w7_stmoe_h2", "workload_id": "W7", "batch_size": 70, "tokens": 64, "predictor": "stmoe", "horizon": 2, "budget": 4},
    {"run_id": "w7_cstp_h2", "workload_id": "W7", "batch_size": 70, "tokens": 64, "predictor": "cstp", "horizon": 2, "budget": 4},
]


def generate_yaml(run_def):
    run_id = run_def["run_id"]
    wid = run_def["workload_id"]
    b = run_def["batch_size"]
    t = run_def["tokens"]
    pred = run_def["predictor"]
    h = run_def["horizon"]
    bgt = run_def["budget"]

    trace_file = os.path.join(TRACES_DIR, f"{wid}.csv")
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
        f"  max_batch_size: {b}",
        "  max_process_token: 0",
        "",
        "simulation:",
        "  data: moe_anywhere",
        "  input_len: 128",
        f"  output_len: {t}",
        "  precision_byte: 2",
        "  skewness: 0.0",
        f"  iter: {t}",
        "  injection_rate: 0",
        "  exit_out_of_memory: false",
        "  mem_cap_limit: false",
        f"  trace_file: {trace_file}",
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


def run_campaign(force=False):
    print("======================================================================")
    print("EXECUTING POINT 3 LARGE WORKLOAD CAMPAIGN (27 RUNS, 40 STAGES, 35 GB)")
    print("======================================================================")

    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    manifest_df = pd.read_csv(MANIFEST_FILE)

    raw_records = []
    baseline_stalls = {}

    total_runs = len(CAMPAIGN_RUNS)
    for idx, rdef in enumerate(CAMPAIGN_RUNS, 1):
        run_id = rdef["run_id"]
        wid = rdef["workload_id"]
        b = rdef["batch_size"]
        t = rdef["tokens"]
        pred = rdef["predictor"]
        h = rdef["horizon"]
        bgt = rdef["budget"]

        raw_run_dir = os.path.join(RAW_DIR, run_id)
        traces_dir = os.path.join(raw_run_dir, "traces")
        log_path = os.path.join(LOG_DIR, f"{run_id}.log")
        sum_json = os.path.join(traces_dir, "experiment_summary.json")

        skip_run = False
        if not force:
            if os.path.exists(sum_json):
                skip_run = True

        elapsed = 0.0
        if not skip_run:
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
            print(f"[{idx:2d}/{total_runs:2d}] Launching {run_id:16s} ... ", end="", flush=True)
            t0 = time.time()
            with open(log_path, "w") as log_f:
                res = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
            elapsed = time.time() - t0

            if res.returncode != 0:
                print(f"FAILED (code {res.returncode}) in {elapsed:.1f}s")
                continue
        else:
            print(f"[{idx:2d}/{total_runs:2d}] Reusing   {run_id:16s} ... ", end="", flush=True)

        # Parse experiment_summary.json
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

        strict_timely_coverage_pct = (timely_hits / total_demands) * 100.0
        prefetch_associated_coverage_pct = (assisted / total_demands) * 100.0

        current_stall_ms = stall["exposed_gpu_stall_ns"] / 1e6
        if pred == "baseline":
            baseline_stalls[wid] = current_stall_ms
            stall_reduction_ms = 0.0
            stall_reduction_pct = 0.0
        else:
            matched_base_stall = baseline_stalls.get(wid, current_stall_ms)
            stall_reduction_ms = matched_base_stall - current_stall_ms
            if matched_base_stall > 0:
                stall_reduction_pct = (stall_reduction_ms / matched_base_stall) * 100.0
            else:
                stall_reduction_pct = 0.0

        # Split breakdown from demand events
        demand_csv = os.path.join(traces_dir, "expert_demand_events.csv")
        test_demands = 0
        non_test_demands = 0
        test_late_hits = 0
        non_test_late_hits = 0
        test_resident_hits = 0
        non_test_resident_hits = 0
        test_stall_ms = 0.0
        non_test_stall_ms = 0.0

        if os.path.exists(demand_csv):
            wid_map = manifest_df[manifest_df["workload_id"] == wid].set_index("seq_id")["split"].to_dict()
            ddf = pd.read_csv(demand_csv)
            ddf["split"] = ddf["seq_id"].map(wid_map)

            test_mask = (ddf["split"] == "test")
            non_test_mask = (ddf["split"] != "test")

            test_demands = int(test_mask.sum())
            non_test_demands = int(non_test_mask.sum())

            test_sub = ddf[test_mask]
            non_test_sub = ddf[non_test_mask]

            test_res_counts = test_sub["residency_status"].value_counts().to_dict()
            non_test_res_counts = non_test_sub["residency_status"].value_counts().to_dict()

            test_resident_hits = test_res_counts.get("RESIDENT_HIT", 0)
            non_test_resident_hits = non_test_res_counts.get("RESIDENT_HIT", 0)

            test_late_hits = test_res_counts.get("LATE_IN_FLIGHT", 0)
            non_test_late_hits = non_test_res_counts.get("LATE_IN_FLIGHT", 0)

            test_stall_ms = float(test_sub["ddr5_wait_ns"].sum() / 1e6)
            non_test_stall_ms = float(non_test_sub["ddr5_wait_ns"].sum() / 1e6)

        total_ddr5_bytes = traff["total_ddr5_bytes_transferred"]
        demand_bytes = traff["demand_bytes_transferred"]
        speculative_bytes = traff["speculative_bytes_transferred"]
        if demand_bytes > 0:
            traffic_overhead_pct = (speculative_bytes / demand_bytes) * 100.0
        else:
            traffic_overhead_pct = 0.0

        total_sim_time_ms = stall["total_simulated_execution_ns"] / 1e6
        stall_per_tok_ms = stall["wait_per_token_ns"] / 1e6
        stall_per_req_ms = (stall["exposed_gpu_stall_ns"] / float(b)) / 1e6

        rec = {
            "run_id": run_id,
            "workload_id": wid,
            "batch_size": b,
            "tokens": t,
            "stages": 40,
            "predictor": pred,
            "horizon": h,
            "budget": bgt,
            "host_runtime_sec": elapsed,
            "total_sim_time_ms": total_sim_time_ms,
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
            "ordinary_hbm_hits": hbm["total_hbm_resident_hits"],
            "inserts": inserts,
            "evicts": evicts,
            "demand_bytes": demand_bytes,
            "speculative_bytes": speculative_bytes,
            "total_ddr5_bytes": total_ddr5_bytes,
            "traffic_overhead_pct": traffic_overhead_pct,
            "exposed_gpu_stall_ms": current_stall_ms,
            "stall_per_token_ms": stall_per_tok_ms,
            "stall_per_request_ms": stall_per_req_ms,
            "stall_reduction_ms": stall_reduction_ms,
            "stall_reduction_pct": stall_reduction_pct,
            "mean_expert_load_lat_us": exp_lat["mean_ns"] / 1e3,
            "p50_expert_load_lat_us": exp_lat["p50_ns"] / 1e3,
            "p95_expert_load_lat_us": exp_lat["p95_ns"] / 1e3,
            "mean_token_lat_ms": tok_lat["mean_ns"] / 1e6,
            "p50_token_lat_ms": tok_lat["p50_ns"] / 1e6,
            "p95_token_lat_ms": tok_lat["p95_ns"] / 1e6,
            "tokens_per_sec": tok_lat.get("simulated_tokens_per_second", 0.0),
            "test_demands": test_demands,
            "non_test_demands": non_test_demands,
            "test_late_hits": test_late_hits,
            "non_test_late_hits": non_test_late_hits,
            "test_resident_hits": test_resident_hits,
            "non_test_resident_hits": non_test_resident_hits,
            "test_stall_ms": test_stall_ms,
            "non_test_stall_ms": non_test_stall_ms,
        }
        raw_records.append(rec)
        print(
            f"DONE ({elapsed:4.1f}s) | Dispatched: {rec['dispatched_prefetches']:5d} | "
            f"Assisted: {assisted:5d} | Timely: {timely_hits:3d} | Late: {late_hits:5d} | "
            f"Evicts: {evicts:5d} | Stall: {current_stall_ms:8.1f} ms (Delta: {stall_reduction_ms:+7.2f} ms)"
        )

    df = pd.DataFrame(raw_records)

    # 1. per_run_metrics.csv
    per_run_csv = os.path.join(RESULTS_DIR, "per_run_metrics.csv")
    df.to_csv(per_run_csv, index=False)
    print(f"\n[OUTPUT 1] Saved per-run metrics to {per_run_csv}")

    # 2. batch_scaling.csv (T=16, H=2, Baseline vs ST-MoE vs CSTP across B={14, 32, 48, 70})
    b_df = df[
        (df["tokens"] == 16) &
        ((df["horizon"] == 2) | (df["predictor"] == "baseline")) &
        (df["workload_id"].isin(["W2", "W3", "W4", "W5"]))
    ].copy()
    b_df = b_df.sort_values(by=["batch_size", "predictor"])
    batch_csv = os.path.join(RESULTS_DIR, "batch_scaling.csv")
    b_df.to_csv(batch_csv, index=False)
    print(f"[OUTPUT 2] Saved batch scaling to {batch_csv}")

    # 3. token_scaling.csv (B=70, H=2, Baseline vs ST-MoE vs CSTP across T={16, 32, 64})
    t_df = df[
        (df["batch_size"] == 70) &
        ((df["horizon"] == 2) | (df["predictor"] == "baseline")) &
        (df["workload_id"].isin(["W5", "W6", "W7"]))
    ].copy()
    t_df = t_df.sort_values(by=["tokens", "predictor"])
    token_csv = os.path.join(RESULTS_DIR, "token_scaling.csv")
    t_df.to_csv(token_csv, index=False)
    print(f"[OUTPUT 3] Saved token scaling to {token_csv}")

    # 4. predictor_comparison.csv (Head-to-head ST-MoE vs CSTP across all workloads)
    p_df = df[df["predictor"].isin(["stmoe", "cstp"])].copy()
    p_df = p_df.sort_values(by=["workload_id", "horizon", "predictor"])
    pred_csv = os.path.join(RESULTS_DIR, "predictor_comparison.csv")
    p_df.to_csv(pred_csv, index=False)
    print(f"[OUTPUT 4] Saved predictor comparison to {pred_csv}")

    # 5. prefetch_event_accounting.csv
    evt_cols = [
        "run_id", "workload_id", "batch_size", "tokens", "predictor", "horizon",
        "total_demands", "unique_demands", "dispatched_prefetches",
        "prefetch_assisted_demands", "timely_hits", "late_hits",
        "strict_timely_coverage_pct", "prefetch_associated_coverage_pct",
        "prefetches_completed_used", "prefetches_completed_unused",
        "unused_pct", "unused_bytes", "speculative_bytes", "total_ddr5_bytes",
        "traffic_overhead_pct", "stall_reduction_ms", "stall_reduction_pct"
    ]
    evt_df = df[evt_cols].copy()
    evt_csv = os.path.join(RESULTS_DIR, "prefetch_event_accounting.csv")
    evt_df.to_csv(evt_csv, index=False)
    print(f"[OUTPUT 5] Saved event accounting to {evt_csv}")

    print("\n======================================================================")
    print("ALL 5 REQUIRED RESULTS TABLES GENERATED SUCCESSFULLY")
    print("======================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run large workload campaign")
    parser.add_argument("--force", action="store_true", help="Force rerun existing runs")
    args = parser.parse_args()
    run_campaign(force=args.force)

