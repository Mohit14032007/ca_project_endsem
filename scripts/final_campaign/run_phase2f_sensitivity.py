#!/usr/bin/env python3
"""
Phase 2F: Synthetic Compute-Window Sensitivity Campaign Runner.

Evaluates forward-lookahead prefetching across a sweep of synthetic compute multipliers:
M in {1.0, 4.0, 8.0, 12.0, 16.0}
Predictors: None (Demand-only baseline B), ST-MoE (FWD-S-H2), CSTP (FWD-C-H2)
Total configurations: 15 runs.

Strictly adheres to:
1. Zero ternary operators (ast.IfExp verified).
2. Strict matched baseline comparisons: FWD(M) evaluated strictly against B(M).
3. Monotonic inter-token simulated timebase (clock continuity).
4. Full conservation invariant validation for demand outcomes and transfer lifecycles.
"""

import ast
import json
import os
import shutil
import subprocess
import sys
import time
import pandas as pd
import yaml

REPO_ROOT = "/home/mohman/ca_project"
SIMULATOR_DIR = os.path.join(REPO_ROOT, "LLMSimulator")
EXECUTABLE = os.path.join(SIMULATOR_DIR, "build", "run_deepseekv3_mixtral_style")
OUTPUT_BASE = os.path.join(REPO_ROOT, "results", "final_comparative_campaign", "phase2f")
CONFIGS_DIR = os.path.join(OUTPUT_BASE, "configs")
DOCS_DIR = os.path.join(REPO_ROOT, "docs", "final_campaign", "phase2f")

HISTORICAL_BASELINE_DIR = os.path.join(
    REPO_ROOT, "results", "final_comparative_campaign", "phase2e3_warm_diagnostic", "B"
)

CALIBRATED_DDR5_BW_GBPS = 22.578
EXPERT_SIZE_BYTES = 88080384

WORKLOAD_A1 = {
    "name": "Workload A1",
    "task": "ms_task12",
    "model_name": "deepseekV3",
    "num_layers": 78,
    "moe_layers": 75,
    "top_k": 8,
    "input_len": 128,
    "output_len": 128,
    "batch_size": 1,
    "iter": 4,
    "trace_file": "/home/mohman/ca_project/LLMSimulator/results/endsem_phase_e/final_baseline/traces/ms_task12_t128_l75.csv"
}

MULTIPLIERS = [1.0, 4.0, 8.0, 12.0, 16.0]

PREDICTOR_FAMILIES = [
    {
        "id_prefix": "B",
        "name_prefix": "Demand-only Baseline (LRU)",
        "enabled": False,
        "mode": "none",
        "predictor": "none",
        "scheduling_mode": "legacy_current_stage",
        "horizon": 1,
        "budget": 0,
        "eviction": "lru",
        "quarantine": False,
        "pinning": False,
        "max_pinned": 0,
        "spec_fraction": 0.0
    },
    {
        "id_prefix": "FWD-S-H2",
        "name_prefix": "ST-MoE Forward Lookahead H=2",
        "enabled": True,
        "mode": "hardware_async",
        "predictor": "st_moe",
        "scheduling_mode": "forward_lookahead",
        "horizon": 2,
        "budget": 4,
        "eviction": "pabi",
        "quarantine": True,
        "pinning": True,
        "max_pinned": 4,
        "spec_fraction": 0.20
    },
    {
        "id_prefix": "FWD-C-H2",
        "name_prefix": "CSTP Forward Lookahead H=2",
        "enabled": True,
        "mode": "hardware_async",
        "predictor": "cstp",
        "scheduling_mode": "forward_lookahead",
        "horizon": 2,
        "budget": 4,
        "eviction": "pabi",
        "quarantine": True,
        "pinning": True,
        "max_pinned": 4,
        "spec_fraction": 0.20
    }
]


def check_no_ternary_operators():
    with open(__file__, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=__file__)
    ternaries = []
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            ternaries.append(node.lineno)
    if ternaries:
        raise AssertionError(f"Ternary operator found at lines {ternaries} in {__file__}!")
    print("[PRE-FLIGHT CHECK] AST check passed: 0 ternary operators in Python runner.")


def generate_yaml_config(run_id, family, multiplier, run_dir):
    w = WORKLOAD_A1
    csv_dir = os.path.join(run_dir, "csv")
    raw_dir = os.path.join(run_dir, "raw")
    traces_dir = os.path.join(run_dir, "traces")

    doc = {
        "model": {
            "model_name": w["model_name"],
            "num_layers": w["num_layers"]
        },
        "system": {
            "gpu_gen": "A100",
            "nvlink_gen": 3,
            "infiniband_gen": 800,
            "num_node": 1,
            "num_device": 1,
            "processor_type": "GPU",
            "distribution": {
                "none_expert_tensor_degree": 1,
                "expert_tensor_degree": 1
            },
            "optimization": {
                "parallel_execution": "off",
                "hetero_subbatch": False,
                "disagg_system": False,
                "use_low_unit_moe_only": False,
                "use_ramulator": "on",
                "use_event_driven_dram": True,
                "bypass_offchip_execution_cache": False,
                "expert_cache_capacity_bytes": 80000000000,
                "offchip_dram_cfg_path": "reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/config_snapshot/dram_config_DDR5_retry.yaml",
                "hbm_dram_cfg_path": "reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/config_snapshot/dram_config_HBM3_80GB.yaml",
                "ddr5_bandwidth_gbps": CALIBRATED_DDR5_BW_GBPS,
                "compute_time_multiplier": float(multiplier),
                "clock_continuity": True,
                "compressed_kv": True,
                "use_absorb": True,
                "use_flash_mla": True,
                "use_flash_attention": True,
                "reuse_kv_cache": True,
                "kv_cache_reuse_rate": 0.0,
                "prefill_mode": False,
                "decode_mode": True,
                "use_hitting_queue": False
            }
        },
        "serving": {
            "max_batch_size": w["batch_size"],
            "max_process_token": 0
        },
        "simulation": {
            "data": "synthesis",
            "input_len": w["input_len"],
            "output_len": w["output_len"],
            "precision_byte": 2,
            "skewness": 0.0,
            "iter": w["iter"],
            "injection_rate": 0,
            "exit_out_of_memory": False,
            "mem_cap_limit": False,
            "warm_hbm_preload": False,
            "trace_file": w["trace_file"],
            "optimization": {
                "compute_time_multiplier": float(multiplier)
            }
        },
        "log": {
            "print_log": False,
            "export_gantt": False,
            "output_directory": csv_dir + "/",
            "gantt_directory": raw_dir + "/",
            "traces_directory": traces_dir + "/"
        }
    }

    if not family["enabled"]:
        doc["simulation"]["prefetching"] = {
            "enabled": False,
            "mode": "none",
            "predictor": {
                "type": "none"
            }
        }
    else:
        prefetch_sec = {
            "enabled": True,
            "mode": family["mode"],
            "predictor": {
                "type": family["predictor"],
                "online_adaptation": True
            },
            "candidate_selection": {
                "policy": "top_k",
                "budget": family["budget"]
            },
            "scheduling": {
                "scheduling_mode": family["scheduling_mode"],
                "prediction_horizon": family["horizon"],
                "lookahead_tokens": 0,
                "demand_priority": True,
                "queue_enabled": True,
                "queue_capacity": 1024,
                "priority_policy": "deadline_score_rank",
                "duplicate_policy": "retain_highest_priority",
                "overflow_policy": "reject_lowest_priority"
            },
            "transfer": {
                "model": "event_driven",
                "asynchronous": True
            },
            "eviction_policy": family["eviction"]
        }

        if family["predictor"] == "st_moe":
            prefetch_sec["predictor"]["confidence_threshold"] = 2
        elif family["predictor"] == "cstp":
            prefetch_sec["predictor"]["alpha"] = 0.5
            prefetch_sec["predictor"]["smoothing_lambda"] = 1.0

        if family["quarantine"]:
            prefetch_sec["pabi"] = {
                "quarantine_enabled": True,
                "speculative_fraction": family["spec_fraction"],
                "pinning_enabled": family["pinning"],
                "max_pinned_experts": family["max_pinned"],
                "pin_selection_policy": "top_ranked",
                "pin_expiration_policy": "stage_deadline"
            }

        doc["simulation"]["prefetching"] = prefetch_sec

    os.makedirs(CONFIGS_DIR, exist_ok=True)
    yaml_path = os.path.join(CONFIGS_DIR, f"{run_id}.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(doc, f, sort_keys=False, default_flow_style=False)

    manifest_dir = os.path.join(run_dir, "manifest")
    os.makedirs(manifest_dir, exist_ok=True)
    with open(os.path.join(manifest_dir, "effective_config.yaml"), "w") as f:
        yaml.dump(doc, f, sort_keys=False, default_flow_style=False)

    return yaml_path


def parse_completion_status(status_file_path):
    d = {}
    if os.path.exists(status_file_path):
        with open(status_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    d[k] = v
    return d


def parse_step_summary_csv(csv_dir):
    metrics = {
        "sim_total_device_time_ns": 0,
        "sim_total_device_time_ms": 0.0,
        "sim_per_token_latency_avg_ms": 0.0,
        "sim_per_token_latency_std_ms": 0.0,
        "sim_total_moe_dram_ms": 0.0,
        "sim_total_moe_comp_ms": 0.0,
        "steps_count": 0,
        "per_token_latencies": [],
        "total_energy_j": 0.0,
        "act_energy_j": 0.0,
        "read_energy_j": 0.0,
        "write_energy_j": 0.0
    }
    if not os.path.exists(csv_dir):
        return metrics

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    if not csv_files:
        return metrics

    csv_file = os.path.join(csv_dir, csv_files[0])
    try:
        df = pd.read_csv(csv_file)
        if len(df) > 0 and "latency" in df.columns:
            metrics["steps_count"] = len(df)
            tot_latency_ns = df["latency"].sum()
            metrics["sim_total_device_time_ns"] = int(tot_latency_ns)
            metrics["sim_total_device_time_ms"] = float(tot_latency_ns / 1e6)
            metrics["sim_per_token_latency_avg_ms"] = float(df["latency"].mean() / 1e6)
            metrics["per_token_latencies"] = [float(v / 1e6) for v in df["latency"]]
            std_ms = 0.0
            if len(df) > 1:
                std_ms = float(df["latency"].std() / 1e6)
            metrics["sim_per_token_latency_std_ms"] = std_ms

            if "moe_dram" in df.columns:
                metrics["sim_total_moe_dram_ms"] = float(df["moe_dram"].sum() / 1e6)
            if "moe_comp" in df.columns:
                metrics["sim_total_moe_comp_ms"] = float(df["moe_comp"].sum() / 1e6)

            if "total_energy" in df.columns:
                metrics["total_energy_j"] = float(df["total_energy"].sum() / 1e9)
            if "act_energy" in df.columns:
                metrics["act_energy_j"] = float(df["act_energy"].sum() / 1e9)
            if "read_energy" in df.columns:
                metrics["read_energy_j"] = float(df["read_energy"].sum() / 1e9)
            if "write_energy" in df.columns:
                metrics["write_energy_j"] = float(df["write_energy"].sum() / 1e9)
    except Exception as e:
        print(f"[WARN] Error reading step summary {csv_file}: {e}")

    return metrics


def parse_expert_load_csv(load_csv_path):
    metrics = {
        "demand_lookups_total": 0,
        "demand_hits": 0,
        "demand_misses": 0,
        "expert_hbm_hit_rate_pct": 0.0,
        "physical_demand_transfers": 0,
        "physical_demand_bytes": 0,
        "physical_demand_traffic_mb": 0.0,
        "exposed_stall_ns": 0.0,
        "exposed_stall_ms": 0.0,
        "avg_demand_transfer_latency_ns": 0.0
    }
    if not os.path.exists(load_csv_path):
        return metrics

    try:
        df = pd.read_csv(load_csv_path)
        if "weight_tensor" in df.columns and "cache_hit_or_miss" in df.columns:
            exp_df = df[df["weight_tensor"] == "all_weights"]
            n_all = len(exp_df)
            metrics["demand_lookups_total"] = int(n_all)

            exp_hit_df = exp_df[exp_df["cache_hit_or_miss"] == "hit"]
            exp_miss_df = exp_df[exp_df["cache_hit_or_miss"] == "miss"]
            n_hits = len(exp_hit_df)
            n_misses = len(exp_miss_df)
            metrics["demand_hits"] = int(n_hits)
            metrics["demand_misses"] = int(n_misses)

            if n_all > 0:
                metrics["expert_hbm_hit_rate_pct"] = float(n_hits / n_all * 100.0)

            metrics["physical_demand_transfers"] = int(n_misses)
            phys_bytes = n_misses * EXPERT_SIZE_BYTES
            metrics["physical_demand_bytes"] = int(phys_bytes)
            metrics["physical_demand_traffic_mb"] = float(phys_bytes / (1024.0 * 1024.0))

            stall_ns = float(exp_miss_df["duration"].sum())
            metrics["exposed_stall_ns"] = stall_ns
            metrics["exposed_stall_ms"] = stall_ns / 1e6

            if n_misses > 0:
                metrics["avg_demand_transfer_latency_ns"] = float(exp_miss_df["duration"].mean())
    except Exception as e:
        print(f"[WARN] Error reading {load_csv_path}: {e}")

    return metrics


def parse_prefetch_effectiveness_csv(eff_csv_path):
    metrics = {
        "timely_resident_hits": 0,
        "late_inflight_attachments": 0,
        "unprefetched_misses": 0,
        "total_demand_evaluations": 0,
        "spec_demanded": 0,
        "spec_unused": 0,
        "spec_evicted": 0,
        "spec_in_flight_at_end": 0,
        "spec_cancelled": 0,
        "total_speculative_transfers": 0,
        "avg_remaining_transfer_ns": 0.0,
        "avg_demand_arrival_to_ready_ns": 0.0,
        "speculative_bytes": 0,
        "speculative_mb": 0.0,
        "avg_queue_wait_ns": 0.0
    }
    if not os.path.exists(eff_csv_path):
        return metrics

    try:
        df = pd.read_csv(eff_csv_path)
        if "event_type" in df.columns and "outcome" in df.columns:
            dem_df = df[df["event_type"] == "DEMAND_ACCESS"]
            metrics["total_demand_evaluations"] = len(dem_df)
            d_counts = dem_df["outcome"].value_counts().to_dict()
            metrics["timely_resident_hits"] = int(d_counts.get("TIMELY_RESIDENT", 0))
            metrics["late_inflight_attachments"] = int(d_counts.get("LATE_IN_FLIGHT", 0))
            metrics["unprefetched_misses"] = int(d_counts.get("NOT_PREFETCHED", 0))

            late_df = dem_df[dem_df["outcome"] == "LATE_IN_FLIGHT"]
            if len(late_df) > 0 and "remaining_transfer_ns" in late_df.columns:
                metrics["avg_remaining_transfer_ns"] = float(late_df["remaining_transfer_ns"].mean())
            if len(late_df) > 0 and "demand_arrival_to_ready_ns" in late_df.columns:
                metrics["avg_demand_arrival_to_ready_ns"] = float(late_df["demand_arrival_to_ready_ns"].mean())

            spec_df = df[df["event_type"] == "SPECULATIVE_TRANSFER"]
            metrics["total_speculative_transfers"] = len(spec_df)
            s_counts = spec_df["outcome"].value_counts().to_dict()
            metrics["spec_demanded"] = int(s_counts.get("DEMANDED", 0))
            metrics["spec_unused"] = int(s_counts.get("UNUSED", 0))
            metrics["spec_evicted"] = int(s_counts.get("EVICTED_BEFORE_DEMAND", 0))
            metrics["spec_in_flight_at_end"] = int(s_counts.get("STILL_IN_FLIGHT_AT_END", 0))
            metrics["spec_cancelled"] = int(s_counts.get("CANCELLED", 0))

            metrics["speculative_bytes"] = metrics["total_speculative_transfers"] * EXPERT_SIZE_BYTES
            metrics["speculative_mb"] = float(metrics["speculative_bytes"] / (1024.0 * 1024.0))

            if len(spec_df) > 0 and "dispatch_ts" in spec_df.columns and "transfer_start_ts" in spec_df.columns:
                wait_series = spec_df["transfer_start_ts"] - spec_df["dispatch_ts"]
                valid_wait = wait_series[wait_series >= 0.0]
                if len(valid_wait) > 0:
                    metrics["avg_queue_wait_ns"] = float(valid_wait.mean())
    except Exception as e:
        print(f"[WARN] Error reading {eff_csv_path}: {e}")

    return metrics


def parse_prediction_csv(pred_csv_path):
    metrics = {
        "prediction_precision_pct": 0.0,
        "prediction_recall_pct": 0.0,
        "prediction_f1_pct": 0.0,
        "total_predictions": 0,
        "hits_at_k": 0
    }
    if not os.path.exists(pred_csv_path):
        return metrics

    try:
        df = pd.read_csv(pred_csv_path)
        if "hit" in df.columns:
            n_pred = len(df)
            hits = int((df["hit"] == 1).sum())
            metrics["total_predictions"] = n_pred
            metrics["hits_at_k"] = hits
            if n_pred > 0:
                metrics["prediction_precision_pct"] = float(hits / n_pred * 100.0)

            stage_col = None
            if "target_stage_id" in df.columns:
                stage_col = "target_stage_id"
            elif "target_stage" in df.columns:
                stage_col = "target_stage"

            if stage_col is not None and "token_id" in df.columns:
                n_target_stages = len(df[["token_id", stage_col]].drop_duplicates())
                total_target_routed = n_target_stages * 8
                if total_target_routed > 0:
                    metrics["prediction_recall_pct"] = float(hits / total_target_routed * 100.0)

            p = metrics["prediction_precision_pct"]
            r = metrics["prediction_recall_pct"]
            if (p + r) > 0.0:
                metrics["prediction_f1_pct"] = float(2 * p * r / (p + r))
    except Exception as e:
        print(f"[WARN] Error reading {pred_csv_path}: {e}")

    return metrics


def run_single_experiment(run_id, family, multiplier):
    run_dir = os.path.join(OUTPUT_BASE, run_id)
    w = WORKLOAD_A1

    print("\n" + "=" * 90)
    print(f"RUNNING: {run_id} | Multiplier M={multiplier} | Family: {family['name_prefix']}")
    print(f"Target Directory: {run_dir}")
    print("=" * 90)

    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    os.makedirs(os.path.join(run_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "traces"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "queue"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "manifest"), exist_ok=True)

    yaml_path = generate_yaml_config(run_id, family, multiplier, run_dir)

    env = os.environ.copy()
    env.pop("DEBUG_CACHE_CAPACITY", None)
    env["TRACE_DIR"] = os.path.join(run_dir, "traces")
    env["PREDICTION_OUTPUT_DIR"] = run_dir
    env["PREFETCH_QUEUE_OUTPUT_DIR"] = os.path.join(run_dir, "queue")
    env["USE_EVENT_DRIVEN_DRAM"] = "1"
    env["BYPASS_OFFCHIP_EXECUTION_CACHE"] = "0"
    env["DDR5_BANDWIDTH_GBPS"] = str(CALIBRATED_DDR5_BW_GBPS)
    env["COMPUTE_MULTIPLIER"] = str(multiplier)
    env["CLOCK_CONTINUITY"] = "1"
    env["EVICTION_POLICY"] = family["eviction"]

    if family["enabled"]:
        env["PREFETCH_ENABLED"] = "1"
        env["PREFETCH_MODE"] = family["mode"]
        env["PREFETCH_PREDICTOR_TYPE"] = family["predictor"]
        env["PREFETCH_SCHEDULING_MODE"] = family["scheduling_mode"]
        env["PREFETCH_PREDICTION_HORIZON"] = str(family["horizon"])
    else:
        env["PREFETCH_ENABLED"] = "0"
        env["PREFETCH_MODE"] = "none"
        env["PREFETCH_PREDICTOR_TYPE"] = "none"
        env["PREFETCH_SCHEDULING_MODE"] = "legacy_current_stage"
        env["PREFETCH_PREDICTION_HORIZON"] = "1"

    if family["quarantine"]:
        env["PABI_QUARANTINE_ENABLED"] = "1"
    else:
        env["PABI_QUARANTINE_ENABLED"] = "0"

    if family["pinning"]:
        env["PABI_PINNING_ENABLED"] = "1"
        env["PABI_MAX_PINNED_EXPERTS"] = str(family["max_pinned"])
    else:
        env["PABI_PINNING_ENABLED"] = "0"
        env["PABI_MAX_PINNED_EXPERTS"] = "0"

    cmd = [EXECUTABLE, yaml_path]

    t0 = time.perf_counter()
    start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res = subprocess.run(cmd, cwd=SIMULATOR_DIR, env=env, capture_output=True, text=True)
    t1 = time.perf_counter()
    end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    host_sec = t1 - t0

    with open(os.path.join(run_dir, "logs", "stdout.log"), "w") as f:
        f.write(res.stdout)
    with open(os.path.join(run_dir, "logs", "stderr.log"), "w") as f:
        f.write(res.stderr)

    if res.returncode != 0:
        print(f"[FATAL ERROR] Run {run_id} failed with exit code {res.returncode}!")
        print(f"Stderr:\n{res.stderr[-500:]}")
        sys.exit(res.returncode)

    completion_status = parse_completion_status(os.path.join(run_dir, "logs", "completion_status.txt"))
    step_metrics = parse_step_summary_csv(os.path.join(run_dir, "csv"))
    load_metrics = parse_expert_load_csv(os.path.join(run_dir, "traces", "expert_load.csv"))
    eff_metrics = parse_prefetch_effectiveness_csv(os.path.join(run_dir, "traces", "prefetch_effectiveness.csv"))

    pred_csv = os.path.join(run_dir, "prediction_records.csv")
    pred_metrics = parse_prediction_csv(pred_csv)

    manifest_record = {
        "run_id": run_id,
        "multiplier": multiplier,
        "family": family["id_prefix"],
        "predictor": family["predictor"],
        "horizon": family["horizon"],
        "budget": family["budget"],
        "host_seconds": host_sec,
        "start_time": start_time_iso,
        "end_time": end_time_iso,
        "exit_code": res.returncode,
        "sim_exec_time_ms": step_metrics["sim_total_device_time_ms"],
        "exposed_stall_ms": load_metrics["exposed_stall_ms"],
        "hbm_hit_rate_pct": load_metrics["expert_hbm_hit_rate_pct"],
        "demand_hits": load_metrics["demand_hits"],
        "demand_misses": load_metrics["demand_misses"],
        "timely_resident_hits": eff_metrics["timely_resident_hits"],
        "late_inflight_attachments": eff_metrics["late_inflight_attachments"],
        "unprefetched_misses": eff_metrics["unprefetched_misses"],
        "spec_demanded": eff_metrics["spec_demanded"],
        "spec_unused": eff_metrics["spec_unused"],
        "spec_evicted": eff_metrics["spec_evicted"],
        "spec_in_flight_at_end": eff_metrics["spec_in_flight_at_end"],
        "spec_cancelled": eff_metrics["spec_cancelled"],
        "total_speculative": eff_metrics["total_speculative_transfers"],
        "speculative_traffic_mb": eff_metrics["speculative_mb"],
        "demand_traffic_mb": load_metrics["physical_demand_traffic_mb"],
        "avg_queue_wait_ms": eff_metrics["avg_queue_wait_ns"] / 1e6,
        "avg_remaining_transfer_ms": eff_metrics["avg_remaining_transfer_ns"] / 1e6,
        "prediction_precision_pct": pred_metrics["prediction_precision_pct"],
        "prediction_recall_pct": pred_metrics["prediction_recall_pct"],
        "prediction_f1_pct": pred_metrics["prediction_f1_pct"],
        "total_energy_j": step_metrics["total_energy_j"],
        "per_token_latencies": step_metrics["per_token_latencies"]
    }

    with open(os.path.join(run_dir, "manifest", "run_manifest.json"), "w") as f:
        json.dump(manifest_record, f, indent=2)

    print(f"  [DONE] {run_id} | Host: {host_sec:.2f}s | Sim Exec: {step_metrics['sim_total_device_time_ms']:.2f} ms | Stall: {load_metrics['exposed_stall_ms']:.2f} ms | Timely Hits: {eff_metrics['timely_resident_hits']} | Late Attach: {eff_metrics['late_inflight_attachments']}")
    return manifest_record


def validate_baseline_gate(bm1_record):
    print("\n" + "#" * 90)
    print("EXECUTING PART B2: STRICT BASELINE EQUIVALENCE GATE")
    print("#" * 90)

    # Reference values from Phase 2E.3 B
    ref_exec_ms = 8003.70
    ref_stall_ms = 7884.24
    ref_hits = 5179
    ref_misses = 2021
    ref_hit_rate = 71.93

    actual_exec_ms = bm1_record["sim_exec_time_ms"]
    actual_stall_ms = bm1_record["exposed_stall_ms"]
    actual_hits = bm1_record["demand_hits"]
    actual_misses = bm1_record["demand_misses"]
    actual_hit_rate = bm1_record["hbm_hit_rate_pct"]

    print(f"Reference Phase 2E.3 B: Exec={ref_exec_ms:.2f} ms | Stall={ref_stall_ms:.2f} ms | Hits={ref_hits} | Misses={ref_misses} | HitRate={ref_hit_rate:.2f}%")
    print(f"Actual Phase 2F B-M1:   Exec={actual_exec_ms:.2f} ms | Stall={actual_stall_ms:.2f} ms | Hits={actual_hits} | Misses={actual_misses} | HitRate={actual_hit_rate:.2f}%")

    diff_exec = abs(actual_exec_ms - ref_exec_ms)
    diff_stall = abs(actual_stall_ms - ref_stall_ms)

    assert diff_exec < 0.1, f"Baseline execution time mismatch: {actual_exec_ms} vs {ref_exec_ms}"
    assert diff_stall < 0.1, f"Baseline exposed stall mismatch: {actual_stall_ms} vs {ref_stall_ms}"
    assert actual_hits == ref_hits, f"Demand hits mismatch: {actual_hits} vs {ref_hits}"
    assert actual_misses == ref_misses, f"Demand misses mismatch: {actual_misses} vs {ref_misses}"

    print("[GATE RESULT: PASS] 100% Strict Baseline Equivalence verified against historical baseline anchor!")


def main():
    check_no_ternary_operators()

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    all_records = []
    baseline_records_by_mult = {}

    # Step 1: Run B-M1 first for validation gate
    bm1_family = PREDICTOR_FAMILIES[0]
    bm1_record = run_single_experiment("B-M1", bm1_family, 1.0)
    validate_baseline_gate(bm1_record)
    all_records.append(bm1_record)
    baseline_records_by_mult[1.0] = bm1_record

    # Step 2: Run remaining 14 experiments
    for m in MULTIPLIERS:
        for fam in PREDICTOR_FAMILIES:
            run_id = f"{fam['id_prefix']}-M{int(m)}"
            if run_id == "B-M1":
                continue  # Already executed
            rec = run_single_experiment(run_id, fam, m)
            all_records.append(rec)
            if fam["id_prefix"] == "B":
                baseline_records_by_mult[m] = rec

    print("\n" + "#" * 90)
    print("ALL 15 EXPERIMENTAL RUNS COMPLETE. GENERATING SUMMARY DELIVERABLES & CSVs...")
    print("#" * 90)

    # Process and compute matched baseline metrics
    summary_rows = []
    manifest_rows = []
    validation_rows = []
    detailed_token_rows = []
    queue_diag_rows = []

    for r in all_records:
        m = r["multiplier"]
        fam = r["family"]
        run_id = r["run_id"]
        b_rec = baseline_records_by_mult[m]

        b_exec = b_rec["sim_exec_time_ms"]
        b_stall = b_rec["exposed_stall_ms"]

        stall_reduction_ms = 0.0
        stall_reduction_pct = 0.0
        speedup_pct = 0.0

        if fam != "B":
            stall_reduction_ms = b_stall - r["exposed_stall_ms"]
            if b_stall > 0.0:
                stall_reduction_pct = (stall_reduction_ms / b_stall) * 100.0
            if b_exec > 0.0:
                speedup_pct = ((b_exec - r["sim_exec_time_ms"]) / b_exec) * 100.0

        conversion_rate_pct = 0.0
        tot_eff = r["timely_resident_hits"] + r["late_inflight_attachments"] + r["spec_unused"]
        if tot_eff > 0:
            conversion_rate_pct = (r["timely_resident_hits"] / tot_eff) * 100.0

        # Demand conservation: natural_hits + timely_hits + late_attachments + unprefetched_misses = 7200
        # In baseline, timely_hits = 0, late_attachments = 0, so natural_hits = demand_hits, unprefetched = demand_misses
        natural_hits = r["demand_hits"] - r["timely_resident_hits"]
        demand_conservation_sum = natural_hits + r["timely_resident_hits"] + r["late_inflight_attachments"] + r["unprefetched_misses"]
        demand_conservation_pass = (demand_conservation_sum == 7200)

        # Transfer lifecycle conservation: submitted = demanded + unused + evicted + still_in_flight + cancelled
        transfer_conservation_sum = r["spec_demanded"] + r["spec_unused"] + r["spec_evicted"] + r["spec_in_flight_at_end"] + r["spec_cancelled"]
        transfer_conservation_pass = (transfer_conservation_sum == r["total_speculative"])

        summary_rows.append({
            "run_id": run_id,
            "multiplier": m,
            "predictor": r["predictor"],
            "horizon": r["horizon"],
            "budget": r["budget"],
            "sim_exec_time_ms": round(r["sim_exec_time_ms"], 2),
            "exposed_stall_ms": round(r["exposed_stall_ms"], 2),
            "stall_reduction_ms": round(stall_reduction_ms, 2),
            "stall_reduction_pct": round(stall_reduction_pct, 2),
            "speedup_pct": round(speedup_pct, 2),
            "hbm_hit_rate_pct": round(r["hbm_hit_rate_pct"], 2),
            "timely_resident_hits": r["timely_resident_hits"],
            "late_inflight_attachments": r["late_inflight_attachments"],
            "unprefetched_misses": r["unprefetched_misses"],
            "prefetch_conversion_pct": round(conversion_rate_pct, 2),
            "avg_queue_wait_ms": round(r["avg_queue_wait_ms"], 2),
            "avg_remaining_transfer_ms": round(r["avg_remaining_transfer_ms"], 2),
            "speculative_traffic_mb": round(r["speculative_traffic_mb"], 2),
            "demand_traffic_mb": round(r["demand_traffic_mb"], 2),
            "prediction_precision_pct": round(r["prediction_precision_pct"], 2),
            "prediction_recall_pct": round(r["prediction_recall_pct"], 2),
            "prediction_f1_pct": round(r["prediction_f1_pct"], 2),
            "total_energy_j": round(r["total_energy_j"], 4)
        })

        manifest_rows.append({
            "run_id": run_id,
            "multiplier": m,
            "family": fam,
            "predictor": r["predictor"],
            "horizon": r["horizon"],
            "budget": r["budget"],
            "start_time": r["start_time"],
            "end_time": r["end_time"],
            "host_seconds": round(r["host_seconds"], 2),
            "exit_code": r["exit_code"],
            "status": "SUCCESS"
        })

        validation_rows.append({
            "run_id": run_id,
            "multiplier": m,
            "total_demand_lookups": 7200,
            "natural_hits": natural_hits,
            "timely_hits": r["timely_resident_hits"],
            "late_attachments": r["late_inflight_attachments"],
            "unprefetched_misses": r["unprefetched_misses"],
            "demand_sum": demand_conservation_sum,
            "demand_conservation_valid": demand_conservation_pass,
            "speculative_submitted": r["total_speculative"],
            "spec_demanded": r["spec_demanded"],
            "spec_unused": r["spec_unused"],
            "spec_evicted": r["spec_evicted"],
            "spec_in_flight_at_end": r["spec_in_flight_at_end"],
            "spec_cancelled": r["spec_cancelled"],
            "transfer_sum": transfer_conservation_sum,
            "transfer_conservation_valid": transfer_conservation_pass
        })

        queue_diag_rows.append({
            "run_id": run_id,
            "multiplier": m,
            "predictor": r["predictor"],
            "total_speculative_transfers": r["total_speculative"],
            "avg_queue_wait_ms": round(r["avg_queue_wait_ms"], 2),
            "avg_remaining_transfer_ms": round(r["avg_remaining_transfer_ms"], 2),
            "spec_demanded": r["spec_demanded"],
            "spec_unused": r["spec_unused"],
            "spec_in_flight_at_end": r["spec_in_flight_at_end"]
        })

        for t_idx, lat in enumerate(r["per_token_latencies"]):
            detailed_token_rows.append({
                "run_id": run_id,
                "multiplier": m,
                "predictor": r["predictor"],
                "token_id": t_idx,
                "latency_ms": round(lat, 2)
            })

    # Write CSV deliverables
    pd.DataFrame(manifest_rows).to_csv(os.path.join(DOCS_DIR, "PHASE2F_RUN_MANIFEST.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(DOCS_DIR, "PHASE2F_SUMMARY.csv"), index=False)
    pd.DataFrame(validation_rows).to_csv(os.path.join(DOCS_DIR, "PHASE2F_METRIC_VALIDATION.csv"), index=False)
    pd.DataFrame(detailed_token_rows).to_csv(os.path.join(DOCS_DIR, "PHASE2F_DETAILED_PER_TOKEN.csv"), index=False)
    pd.DataFrame(queue_diag_rows).to_csv(os.path.join(DOCS_DIR, "PHASE2F_QUEUE_DIAGNOSTICS.csv"), index=False)

    print("\n[SUCCESS] Generated all 5 machine-readable CSVs in docs/final_campaign/phase2f/!")
    print(pd.DataFrame(summary_rows)[["run_id", "sim_exec_time_ms", "exposed_stall_ms", "stall_reduction_ms", "speedup_pct", "timely_resident_hits", "late_inflight_attachments", "avg_queue_wait_ms"]].to_string())


if __name__ == "__main__":
    main()

