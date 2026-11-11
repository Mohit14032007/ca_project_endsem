#!/usr/bin/env python3
"""
Phase 2E.3: Modified Baseline Validation and Controlled Multi-Token Prefetch Diagnostic Runner.

Executes:
1. B (Demand-only baseline, 4 tokens)
2. FWD-S-H2 (ST-MoE Forward Lookahead H=2, 4 tokens)
3. FWD-C-H2 (CSTP Forward Lookahead H=2, 4 tokens)

Workload: Workload A1 (ms_task12_t128_l75.csv, DeepSeek-V3 Mixtral-style, 4 tokens)
Output: results/final_comparative_campaign/phase2e3_warm_diagnostic/
"""

import ast
import hashlib
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
OUTPUT_BASE = os.path.join(REPO_ROOT, "results", "final_comparative_campaign", "phase2e3_warm_diagnostic")
CONFIGS_DIR = os.path.join(OUTPUT_BASE, "configs")
DOCS_DIR = os.path.join(REPO_ROOT, "docs", "final_campaign")

GIT_BASELINE_ANCHOR = "d5bf61ff0c066c06f72173189f9924329f613a88"
CALIBRATED_DDR5_BW_GBPS = 22.578
EXPERT_SIZE_BYTES = 88080384

WORKLOAD_A1 = {
    "name": "Workload A1",
    "task": "ms_task12",
    "model_name": "deepseekV3",
    "num_layers": 78,
    "first_k_dense": 3,
    "moe_layers": 75,
    "top_k": 8,
    "input_len": 128,
    "output_len": 128,
    "batch_size": 1,
    "iter": 4,
    "trace_file": "LLMSimulator/results/endsem_phase_e/final_baseline/traces/ms_task12_t128_l75.csv",
    "trace_sha256": "5a635dece5747e78be7f600082c888a7f9e9d2ee50e1ae829d338f75f235ea1c",
    "first_k_dense": 0
}

CONFIGURATIONS = {
    "B": {
        "id": "B",
        "name": "Demand-only Baseline (LRU)",
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
    "FWD-S-H2": {
        "id": "FWD-S-H2",
        "name": "ST-MoE Forward Lookahead H=2",
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
    "FWD-C-H2": {
        "id": "FWD-C-H2",
        "name": "CSTP Forward Lookahead H=2",
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
}

RUN_ORDER = ["B", "FWD-S-H2", "FWD-C-H2"]


def check_zero_ternary(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    tree = ast.parse(code, filename=file_path)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            count += 1
    if count > 0:
        raise RuntimeError(f"FATAL: Found {count} ternary expressions (ast.IfExp) in {file_path}")
    print(f"[VERIFY] Zero ternary expressions verified in {file_path}")


def compute_sha256(file_path):
    if not os.path.exists(file_path):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def generate_yaml_config(config_id, run_dir, custom_iter=None):
    w = WORKLOAD_A1
    c = CONFIGURATIONS[config_id]

    csv_dir = os.path.join(run_dir, "csv")
    raw_dir = os.path.join(run_dir, "raw")
    traces_dir = os.path.join(run_dir, "traces")

    iter_count = w["iter"]
    if custom_iter is not None:
        iter_count = custom_iter

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
            "iter": iter_count,
            "injection_rate": 0,
            "exit_out_of_memory": False,
            "mem_cap_limit": False,
            "warm_hbm_preload": False,
            "trace_file": os.path.join(REPO_ROOT, w["trace_file"])
        },
        "log": {
            "print_log": False,
            "export_gantt": False,
            "output_directory": csv_dir + "/",
            "gantt_directory": raw_dir + "/",
            "traces_directory": traces_dir + "/"
        }
    }

    if not c["enabled"]:
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
            "mode": c["mode"],
            "predictor": {
                "type": c["predictor"],
                "online_adaptation": True
            },
            "candidate_selection": {
                "policy": "top_k",
                "budget": c["budget"]
            },
            "scheduling": {
                "scheduling_mode": c["scheduling_mode"],
                "prediction_horizon": c["horizon"],
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
            "eviction_policy": c["eviction"]
        }

        if c["predictor"] == "st_moe":
            prefetch_sec["predictor"]["confidence_threshold"] = 2
        elif c["predictor"] == "cstp":
            prefetch_sec["predictor"]["alpha"] = 0.5
            prefetch_sec["predictor"]["smoothing_lambda"] = 1.0

        if c["quarantine"]:
            prefetch_sec["pabi"] = {
                "quarantine_enabled": True,
                "speculative_fraction": c["spec_fraction"],
                "pinning_enabled": c["pinning"],
                "max_pinned_experts": c["max_pinned"],
                "pin_selection_policy": "top_ranked",
                "pin_expiration_policy": "stage_deadline"
            }

        doc["simulation"]["prefetching"] = prefetch_sec

    os.makedirs(CONFIGS_DIR, exist_ok=True)
    yaml_path = os.path.join(CONFIGS_DIR, f"{config_id}.yaml")
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


def count_file_lines(path):
    if not os.path.exists(path):
        return 0
    cnt = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            cnt += 1
    return cnt


def parse_expert_cache_csv(cache_csv_path):
    metrics = {
        "hbm_demand_hits": 0,
        "hbm_demand_misses": 0,
        "hbm_speculative_inserts": 0,
        "hbm_demand_inserts": 0,
        "hbm_promotions": 0,
        "hbm_speculative_evictions": 0,
        "hbm_demand_evictions": 0,
        "hbm_pin_intents_created": 0,
        "hbm_pins_applied": 0,
        "hbm_pins_expired": 0,
        "hbm_pins_cleared_on_promotion": 0,
        "hbm_pinned_eviction_skipped": 0,
        "hbm_speculative_rejects": 0
    }
    if not os.path.exists(cache_csv_path):
        return metrics

    try:
        df = pd.read_csv(cache_csv_path)
        if "action" in df.columns:
            counts = df["action"].value_counts().to_dict()
            metrics["hbm_demand_hits"] = int(counts.get("HIT", 0))
            metrics["hbm_demand_misses"] = int(counts.get("MISS", 0))
            metrics["hbm_speculative_inserts"] = int(counts.get("SPECULATIVE_INSERT", 0))
            metrics["hbm_demand_inserts"] = int(counts.get("DEMAND_INSERT", 0))
            metrics["hbm_promotions"] = int(counts.get("PROMOTION", 0))
            metrics["hbm_speculative_evictions"] = int(counts.get("SPECULATIVE_EVICT", 0))
            metrics["hbm_demand_evictions"] = int(counts.get("DEMAND_EVICT", 0)) + int(counts.get("EVICT", 0))
            metrics["hbm_pin_intents_created"] = int(counts.get("PIN_INTENT_CREATED", 0))
            metrics["hbm_pins_applied"] = int(counts.get("PIN_APPLIED", 0))
            metrics["hbm_pins_expired"] = int(counts.get("PIN_EXPIRED", 0))
            metrics["hbm_pins_cleared_on_promotion"] = int(counts.get("PIN_CLEARED_ON_PROMOTION", 0))
            metrics["hbm_pinned_eviction_skipped"] = int(counts.get("PINNED_EVICTION_SKIPPED", 0))
            metrics["hbm_speculative_rejects"] = int(counts.get("SPECULATIVE_REJECT", 0))
    except Exception as e:
        print(f"[WARN] Error reading {cache_csv_path}: {e}")

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
        "avg_demand_transfer_latency_ns": 0.0,
        "has_transfer_id_column": False,
        "valid_transfer_ids_count": 0
    }
    if not os.path.exists(load_csv_path):
        return metrics

    try:
        df = pd.read_csv(load_csv_path)
        if "transfer_id" in df.columns:
            metrics["has_transfer_id_column"] = True
            metrics["valid_transfer_ids_count"] = int((df["transfer_id"] > 0).sum())

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


def parse_step_summary_csv(csv_dir):
    metrics = {
        "sim_total_device_time_ns": 0,
        "sim_total_device_time_ms": 0.0,
        "sim_per_token_latency_avg_ms": 0.0,
        "sim_per_token_latency_std_ms": 0.0,
        "sim_total_moe_dram_ms": 0.0,
        "sim_total_moe_comp_ms": 0.0,
        "steps_count": 0
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
            std_ms = 0.0
            if len(df) > 1:
                std_ms = float(df["latency"].std() / 1e6)
            metrics["sim_per_token_latency_std_ms"] = std_ms

            if "moe_dram" in df.columns:
                metrics["sim_total_moe_dram_ms"] = float(df["moe_dram"].sum() / 1e6)
            if "moe_comp" in df.columns:
                metrics["sim_total_moe_comp_ms"] = float(df["moe_comp"].sum() / 1e6)
    except Exception as e:
        print(f"[WARN] Error reading step summary {csv_file}: {e}")

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

            bytes_val = len(spec_df) * EXPERT_SIZE_BYTES
            metrics["speculative_bytes"] = bytes_val
            metrics["speculative_mb"] = float(bytes_val / (1024.0 * 1024.0))

            # Queue wait time: transfer_start_ts - dispatch_ts
            if len(spec_df) > 0 and "transfer_start_ts" in spec_df.columns and "dispatch_ts" in spec_df.columns:
                started = spec_df[spec_df["transfer_start_ts"] > 0]
                if len(started) > 0:
                    wait_series = started["transfer_start_ts"] - started["dispatch_ts"]
                    metrics["avg_queue_wait_ns"] = float(wait_series.mean())
    except Exception as e:
        print(f"[WARN] Error reading {eff_csv_path}: {e}")

    return metrics


def parse_live_predictions_csv(pred_csv_path):
    metrics = {
        "total_predictions": 0,
        "prediction_hits": 0,
        "prediction_precision_pct": 0.0,
        "prediction_recall_pct": 0.0,
        "prediction_f1_pct": 0.0
    }
    if not os.path.exists(pred_csv_path):
        return metrics

    try:
        df = pd.read_csv(pred_csv_path)
        if "is_hit" in df.columns:
            total_preds = len(df)
            hits = int(df["is_hit"].sum())
            metrics["total_predictions"] = total_preds
            metrics["prediction_hits"] = hits
            if total_preds > 0:
                metrics["prediction_precision_pct"] = float(hits / total_preds * 100.0)

            # Ground truth: total routed experts across evaluated stages
            # In workload A1: 4 tokens * 73 forward stages * 8 experts = 2336 target experts
            # Or calculate recall against evaluated target stages
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


def execute_single_run(config_id, custom_dir=None, custom_iter=None):
    c = CONFIGURATIONS[config_id]
    w = WORKLOAD_A1
    run_dir = os.path.join(OUTPUT_BASE, config_id)
    if custom_dir is not None:
        run_dir = custom_dir

    iter_count = w["iter"]
    if custom_iter is not None:
        iter_count = custom_iter

    expected_records = (iter_count * w["moe_layers"] * 8) + 1

    print("\n" + "#" * 90)
    print(f"STARTING RUN: {config_id} | Name: {c['name']} | Tokens: {iter_count}")
    print(f"Target Directory: {run_dir}")
    print("#" * 90)

    if os.path.exists(run_dir):
        print(f"[CLEANUP] Removing previous run directory {run_dir} for fresh execution...")
        shutil.rmtree(run_dir)

    os.makedirs(os.path.join(run_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "traces"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "queue"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "manifest"), exist_ok=True)

    yaml_path = generate_yaml_config(config_id, run_dir, custom_iter=custom_iter)

    env = os.environ.copy()
    env.pop("DEBUG_CACHE_CAPACITY", None)
    env["TRACE_DIR"] = os.path.join(run_dir, "traces")
    env["PREDICTION_OUTPUT_DIR"] = run_dir
    env["PREFETCH_QUEUE_OUTPUT_DIR"] = os.path.join(run_dir, "queue")
    env["USE_EVENT_DRIVEN_DRAM"] = "1"
    env["BYPASS_OFFCHIP_EXECUTION_CACHE"] = "0"
    env["DDR5_BANDWIDTH_GBPS"] = str(CALIBRATED_DDR5_BW_GBPS)
    env["EVICTION_POLICY"] = c["eviction"]

    if c["enabled"]:
        env["PREFETCH_ENABLED"] = "1"
        env["PREFETCH_MODE"] = c["mode"]
        env["PREFETCH_PREDICTOR_TYPE"] = c["predictor"]
        env["PREFETCH_SCHEDULING_MODE"] = c["scheduling_mode"]
        env["PREFETCH_PREDICTION_HORIZON"] = str(c["horizon"])
    else:
        env["PREFETCH_ENABLED"] = "0"
        env["PREFETCH_MODE"] = "none"
        env["PREFETCH_PREDICTOR_TYPE"] = "none"
        env["PREFETCH_SCHEDULING_MODE"] = "legacy_current_stage"
        env["PREFETCH_PREDICTION_HORIZON"] = "1"

    if c["quarantine"]:
        env["PABI_QUARANTINE_ENABLED"] = "1"
    else:
        env["PABI_QUARANTINE_ENABLED"] = "0"

    if c["pinning"]:
        env["PABI_PINNING_ENABLED"] = "1"
        env["PABI_MAX_PINNED_EXPERTS"] = str(c["max_pinned"])
    else:
        env["PABI_PINNING_ENABLED"] = "0"
        env["PABI_MAX_PINNED_EXPERTS"] = "0"

    env_vars_list = [
        f"TRACE_DIR={env['TRACE_DIR']}",
        f"PREDICTION_OUTPUT_DIR={env['PREDICTION_OUTPUT_DIR']}",
        f"PREFETCH_QUEUE_OUTPUT_DIR={env['PREFETCH_QUEUE_OUTPUT_DIR']}",
        f"USE_EVENT_DRIVEN_DRAM={env['USE_EVENT_DRIVEN_DRAM']}",
        f"BYPASS_OFFCHIP_EXECUTION_CACHE={env['BYPASS_OFFCHIP_EXECUTION_CACHE']}",
        f"DDR5_BANDWIDTH_GBPS={env['DDR5_BANDWIDTH_GBPS']}",
        f"EVICTION_POLICY={env['EVICTION_POLICY']}",
        f"PREFETCH_ENABLED={env['PREFETCH_ENABLED']}",
        f"PREFETCH_MODE={env['PREFETCH_MODE']}",
        f"PREFETCH_PREDICTOR_TYPE={env['PREFETCH_PREDICTOR_TYPE']}",
        f"PREFETCH_SCHEDULING_MODE={env['PREFETCH_SCHEDULING_MODE']}",
        f"PREFETCH_PREDICTION_HORIZON={env['PREFETCH_PREDICTION_HORIZON']}",
        f"PABI_QUARANTINE_ENABLED={env['PABI_QUARANTINE_ENABLED']}",
        f"PABI_PINNING_ENABLED={env['PABI_PINNING_ENABLED']}",
        f"PABI_MAX_PINNED_EXPERTS={env['PABI_MAX_PINNED_EXPERTS']}"
    ]
    env_str = " ".join(env_vars_list)
    cmd = [EXECUTABLE, yaml_path]
    full_cmd_str = f"env {env_str} {' '.join(cmd)}"

    with open(os.path.join(run_dir, "manifest", "command.sh"), "w") as f:
        f.write("#!/bin/bash\n")
        f.write(full_cmd_str + "\n")

    print(f"Configuration: {config_id} ({c['name']})")
    print(f"Executing: {full_cmd_str[:120]}...")

    t0 = time.perf_counter()
    start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res = subprocess.run(cmd, cwd=SIMULATOR_DIR, env=env, capture_output=True, text=True)
    t1 = time.perf_counter()
    end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    host_sec = t1 - t0

    stdout_log = os.path.join(run_dir, "logs", "stdout.log")
    stderr_log = os.path.join(run_dir, "logs", "stderr.log")
    with open(stdout_log, "w") as f:
        f.write(res.stdout)
    with open(stderr_log, "w") as f:
        f.write(res.stderr)

    if res.returncode != 0:
        print(f"[FATAL] Subprocess exited with return code {res.returncode}")
        print(f"STDERR tail:\n{res.stderr[-1000:]}")
        raise RuntimeError(f"Run {config_id} failed with exit code {res.returncode}!")

    status_path = os.path.join(run_dir, "logs", "completion_status.txt")
    if not os.path.exists(status_path):
        raise RuntimeError(f"Run {config_id} missing completion_status.txt")
    status_dict = parse_completion_status(status_path)
    if status_dict.get("STATUS") != "SUCCESS":
        raise RuntimeError(f"Run {config_id} completion status is {status_dict.get('STATUS')}")

    routing_lines = count_file_lines(os.path.join(run_dir, "traces", "expert_routing.csv"))
    if routing_lines != expected_records:
        raise ValueError(f"Run {config_id} routing records {routing_lines} != expected {expected_records}")

    manifest_info = {
        "config_id": config_id,
        "config_name": c["name"],
        "iterations": iter_count,
        "start_time_iso": start_time_iso,
        "end_time_iso": end_time_iso,
        "host_runtime_sec": host_sec,
        "git_baseline_anchor": GIT_BASELINE_ANCHOR,
        "executable": EXECUTABLE,
        "yaml_path": yaml_path,
        "predictor": c["predictor"],
        "scheduling_mode": c["scheduling_mode"],
        "horizon": c["horizon"],
        "budget": c["budget"],
        "eviction": c["eviction"],
        "quarantine": c["quarantine"],
        "pinning": c["pinning"],
        "exit_code": res.returncode,
        "validation_status": "PASSED"
    }
    with open(os.path.join(run_dir, "manifest", "run_manifest.json"), "w") as f:
        json.dump(manifest_info, f, indent=2)

    print(f"  -> SUCCESS: Run {config_id} finished with ExitCode 0 in {host_sec:.2f} host seconds.")
    return True, run_dir, host_sec


def preflight_verification():
    print("=" * 90)
    print("PHASE 2E.3: PREFLIGHT AUDIT & VERIFICATION")
    print("=" * 90)

    if not os.path.exists(EXECUTABLE):
        raise FileNotFoundError(f"FATAL: Executable not found at {EXECUTABLE}")
    if not os.access(EXECUTABLE, os.X_OK):
        raise PermissionError(f"FATAL: Executable {EXECUTABLE} is not marked executable")
    exec_hash = compute_sha256(EXECUTABLE)
    print(f"[PREFLIGHT 1] Executable verified: {EXECUTABLE} (SHA256: {exec_hash[:16]}...)")

    trace_path = os.path.join(REPO_ROOT, WORKLOAD_A1["trace_file"])
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"FATAL: Trace not found at {trace_path}")
    trace_hash = compute_sha256(trace_path)
    if trace_hash != WORKLOAD_A1["trace_sha256"]:
        raise ValueError(f"FATAL: Trace checksum mismatch: got {trace_hash}, expected {WORKLOAD_A1['trace_sha256']}")
    print(f"[PREFLIGHT 2] Workload A1 trace verified: {trace_path} (SHA-256: {trace_hash[:16]}...)")

    dram_path = os.path.join(SIMULATOR_DIR, "reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/config_snapshot/dram_config_DDR5_retry.yaml")
    hbm_path = os.path.join(SIMULATOR_DIR, "reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/config_snapshot/dram_config_HBM3_80GB.yaml")
    if not os.path.exists(dram_path) or not os.path.exists(hbm_path):
        raise FileNotFoundError("FATAL: Memory configs missing")
    print("[PREFLIGHT 3] DRAM memory configs verified.")

    check_zero_ternary(__file__)

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    print(f"[PREFLIGHT 5] Output directory ready: {OUTPUT_BASE}")
    print("=" * 90)


def run_phase2e3():
    preflight_verification()

    print("\n" + "=" * 90)
    print("PHASE 2E.3: CONTROLLED MULTI-TOKEN DIAGNOSTIC EXPERIMENT (4 TOKENS)")
    print("=" * 90)

    summary_rows = []

    for config_id in RUN_ORDER:
        c = CONFIGURATIONS[config_id]
        ok, rdir, hsec = execute_single_run(config_id)

        step_metrics = parse_step_summary_csv(os.path.join(rdir, "csv"))
        cache_metrics = parse_expert_cache_csv(os.path.join(rdir, "traces", "expert_cache.csv"))
        load_metrics = parse_expert_load_csv(os.path.join(rdir, "traces", "expert_load.csv"))
        eff_metrics = parse_prefetch_effectiveness_csv(os.path.join(rdir, "traces", "prefetch_effectiveness.csv"))
        pred_metrics = parse_live_predictions_csv(os.path.join(rdir, "live_predictions.csv"))

        # Verify Stage-ID alignment:
        routing_csv = os.path.join(rdir, "traces", "expert_routing.csv")
        load_csv = os.path.join(rdir, "traces", "expert_load.csv")
        df_r = pd.read_csv(routing_csv)
        df_l = pd.read_csv(load_csv)
        r_stages = sorted(df_r["layer_id"].unique().tolist())
        l_stages = sorted(df_l["layer_id"].unique().tolist())
        stage_align_ok = (r_stages == l_stages)

        # Total lookups conservation
        total_lookups = load_metrics["demand_lookups_total"]
        demand_hits = load_metrics["demand_hits"]
        demand_misses = load_metrics["demand_misses"]
        timely_hits = eff_metrics["timely_resident_hits"]
        late_attachments = eff_metrics["late_inflight_attachments"]
        not_prefetched = eff_metrics["unprefetched_misses"]

        # Natural demand hits = total demand hits - timely prefetch hits
        natural_hits = demand_hits - timely_hits

        # Conservation check:
        # Total lookups = natural_hits + timely_hits + late_attachments + not_prefetched
        lookup_conservation = (total_lookups == (natural_hits + timely_hits + late_attachments + not_prefetched))
        if not c["enabled"]:
            lookup_conservation = (total_lookups == (demand_hits + demand_misses))

        # Transfer lifecycle conservation:
        # spec_total = spec_demanded + spec_unused + spec_evicted + spec_in_flight + spec_cancelled
        spec_total = eff_metrics["total_speculative_transfers"]
        spec_dem = eff_metrics["spec_demanded"]
        spec_unused = eff_metrics["spec_unused"]
        spec_evicted = eff_metrics["spec_evicted"]
        spec_inflight = eff_metrics["spec_in_flight_at_end"]
        spec_cancelled = eff_metrics["spec_cancelled"]
        transfer_conservation = (spec_total == (spec_dem + spec_unused + spec_evicted + spec_inflight + spec_cancelled))

        row = {
            "config_id": config_id,
            "config_name": c["name"],
            "tokens": WORKLOAD_A1["iter"],
            "host_runtime_sec": round(hsec, 2),
            "sim_exec_time_ms": round(step_metrics["sim_total_device_time_ms"], 2),
            "sim_moe_dram_ms": round(step_metrics["sim_total_moe_dram_ms"], 2),
            "sim_moe_comp_ms": round(step_metrics["sim_total_moe_comp_ms"], 2),
            "exposed_stall_ms": round(load_metrics["exposed_stall_ms"], 2),
            "demand_lookups_total": total_lookups,
            "demand_hits_total": demand_hits,
            "demand_misses_total": demand_misses,
            "hbm_hit_rate_pct": round(load_metrics["expert_hbm_hit_rate_pct"], 2),
            "natural_demand_hits": natural_hits,
            "timely_resident_hits": timely_hits,
            "late_inflight_attachments": late_attachments,
            "unprefetched_misses": not_prefetched,
            "spec_transfers_submitted": spec_total,
            "spec_demanded": spec_dem,
            "spec_unused": spec_unused,
            "spec_evicted": spec_evicted,
            "spec_in_flight_at_end": spec_inflight,
            "spec_cancelled": spec_cancelled,
            "ddr5_demand_bytes": load_metrics["physical_demand_bytes"],
            "ddr5_demand_mb": round(load_metrics["physical_demand_traffic_mb"], 2),
            "ddr5_speculative_bytes": eff_metrics["speculative_bytes"],
            "ddr5_speculative_mb": round(eff_metrics["speculative_mb"], 2),
            "total_ddr5_mb": round(load_metrics["physical_demand_traffic_mb"] + eff_metrics["speculative_mb"], 2),
            "avg_queue_wait_ms": round(eff_metrics["avg_queue_wait_ns"] / 1e6, 3),
            "avg_remaining_transfer_ms": round(eff_metrics["avg_remaining_transfer_ns"] / 1e6, 3),
            "avg_demand_arrival_to_ready_ms": round(eff_metrics["avg_demand_arrival_to_ready_ns"] / 1e6, 3),
            "prediction_precision_pct": round(pred_metrics["prediction_precision_pct"], 2),
            "prediction_recall_pct": round(pred_metrics["prediction_recall_pct"], 2),
            "prediction_f1_pct": round(pred_metrics["prediction_f1_pct"], 2),
            "stage_id_alignment": "UNKNOWN",
            "demand_lookup_conservation": "UNKNOWN",
            "transfer_lifecycle_conservation": "UNKNOWN"
        }
        # Multi-line if-else to preserve zero-ternary rule
        if stage_align_ok:
            row["stage_id_alignment"] = "PASSED"
        else:
            row["stage_id_alignment"] = "FAILED"

        if lookup_conservation:
            row["demand_lookup_conservation"] = "VERIFIED"
        else:
            row["demand_lookup_conservation"] = "FAILED"

        if transfer_conservation:
            row["transfer_lifecycle_conservation"] = "VERIFIED"
        else:
            row["transfer_lifecycle_conservation"] = "FAILED"

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(OUTPUT_BASE, "phase2e3_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"\n[SUMMARY] Saved consolidated CSV to: {summary_csv_path}")

    # Copy to docs/final_campaign/PHASE2E3_SUMMARY.csv
    docs_csv_path = os.path.join(DOCS_DIR, "PHASE2E3_SUMMARY.csv")
    shutil.copyfile(summary_csv_path, docs_csv_path)
    print(f"[SUMMARY] Synced CSV to: {docs_csv_path}")

    # Print summary table
    print("\n" + "=" * 120)
    print("PHASE 2E.3 CONSOLIDATED EXPERIMENT SUMMARY")
    print("=" * 120)
    cols_to_print = [
        "config_id", "sim_exec_time_ms", "exposed_stall_ms", "hbm_hit_rate_pct",
        "natural_demand_hits", "timely_resident_hits", "late_inflight_attachments",
        "spec_demanded", "spec_unused", "total_ddr5_mb"
    ]
    print(df_summary[cols_to_print].to_string(index=False))
    print("=" * 120)

    return df_summary


if __name__ == "__main__":
    run_phase2e3()
