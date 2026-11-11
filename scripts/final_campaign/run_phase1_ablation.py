#!/usr/bin/env python3
"""
Stage 2B: Phase 1 Component Ablation Ladder Execution Runner.
Evaluates 12 runs: Workloads A1 and B2 across P-S, P-C, PF-S, PF-C, QUAR-S, QUAR-C.

Compliance:
- Authoritative baseline anchor: d5bf61ff0c066c06f72173189f9924329f613a88
- Calibrated DDR5 bandwidth: 22.578 GB/s (3.90116 ms / expert)
- Strictly ZERO ternary operators (ast.IfExp == 0)
- Detects incomplete outputs, refuses silent overwrites, supports crash recovery
- Separates exposed memory stall from total DDR5 service time
- Separates expert-level hit rate from matrix-level hit rate
"""

import ast
import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import yaml
import numpy as np
import pandas as pd

REPO_ROOT = "/home/mohman/ca_project/LLMSimulator"
EXECUTABLE = os.path.join(REPO_ROOT, "build", "run_deepseekv3_mixtral_style")
OUTPUT_BASE = "/home/mohman/ca_project/results/final_comparative_campaign"
ABLATION_DIR = os.path.join(OUTPUT_BASE, "ablation")
CONFIGS_DIR = os.path.join(OUTPUT_BASE, "configs", "ablation")
DOCS_DIR = "/home/mohman/ca_project/docs/final_campaign"

GIT_BASELINE_ANCHOR = "d5bf61ff0c066c06f72173189f9924329f613a88"
CALIBRATED_DDR5_BW_GBPS = 22.578
EXPERT_SIZE_BYTES = 88080384
EXPERT_SERVICE_TIME_NS = 88080384.0 / CALIBRATED_DDR5_BW_GBPS

WORKLOADS = {
    "A1": {
        "id": "A1",
        "name": "MoE_Suite_T128_B1_FullDepth",
        "model_name": "deepseekV3",
        "num_layers": 78,
        "first_k_dense": 3,
        "moe_layers": 75,
        "batch_size": 1,
        "input_len": 128,
        "output_len": 128,
        "iter": 128,
        "trace_file": "results/endsem_phase_e/final_baseline/traces/ms_task12_t128_l75.csv",
        "trace_sha256": "5a635dece5747e78be7f600082c888a7f9e9d2ee50e1ae829d338f75f235ea1c",
        "expected_routing_records": 76801,
        "baseline_stall_ms": 399846.43,
        "baseline_hit_rate_pct": 72.73,
        "baseline_sim_time_ms": 248927.0,
        "baseline_host_sec": 210.91
    },
    "B2": {
        "id": "B2",
        "name": "MoE_Anywhere_T128_B4_FullDepth",
        "model_name": "deepseekV3",
        "num_layers": 43,
        "first_k_dense": 3,
        "moe_layers": 40,
        "batch_size": 4,
        "input_len": 128,
        "output_len": 128,
        "iter": 128,
        "trace_file": "results/endsem_phase_e/final_baseline/traces/ma_b4_t128_l40.csv",
        "trace_sha256": "d53226b68966d79ecbbab4edab99f6f7f5506edb1bc97f62fd06d2a0365e86a0",
        "expected_routing_records": 163841,
        "baseline_stall_ms": 717441.46,
        "baseline_hit_rate_pct": 67.99,
        "baseline_sim_time_ms": 506134.0,
        "baseline_host_sec": 118.74
    }
}

CONFIGURATIONS = {
    "P-S": {
        "id": "P-S",
        "name": "ST-MoE Prediction-Only",
        "mode": "prediction_only",
        "predictor": "st_moe",
        "eviction": "lru",
        "quarantine": False,
        "pinning": False,
        "budget": 4,
        "horizon": 2,
        "spec_fraction": 0.0,
        "max_pinned": 0
    },
    "P-C": {
        "id": "P-C",
        "name": "CSTP Prediction-Only",
        "mode": "prediction_only",
        "predictor": "cstp",
        "eviction": "lru",
        "quarantine": False,
        "pinning": False,
        "budget": 4,
        "horizon": 2,
        "spec_fraction": 0.0,
        "max_pinned": 0
    },
    "PF-S": {
        "id": "PF-S",
        "name": "ST-MoE + LRU Prefetch",
        "mode": "hardware_async",
        "predictor": "st_moe",
        "eviction": "lru",
        "quarantine": False,
        "pinning": False,
        "budget": 4,
        "horizon": 2,
        "spec_fraction": 0.0,
        "max_pinned": 0
    },
    "PF-C": {
        "id": "PF-C",
        "name": "CSTP + LRU Prefetch",
        "mode": "hardware_async",
        "predictor": "cstp",
        "eviction": "lru",
        "quarantine": False,
        "pinning": False,
        "budget": 4,
        "horizon": 2,
        "spec_fraction": 0.0,
        "max_pinned": 0
    },
    "QUAR-S": {
        "id": "QUAR-S",
        "name": "ST-MoE + PABI Quarantine",
        "mode": "hardware_async",
        "predictor": "st_moe",
        "eviction": "pabi",
        "quarantine": True,
        "pinning": False,
        "budget": 4,
        "horizon": 2,
        "spec_fraction": 0.20,
        "max_pinned": 0
    },
    "QUAR-C": {
        "id": "QUAR-C",
        "name": "CSTP + PABI Quarantine",
        "mode": "hardware_async",
        "predictor": "cstp",
        "eviction": "pabi",
        "quarantine": True,
        "pinning": False,
        "budget": 4,
        "horizon": 2,
        "spec_fraction": 0.20,
        "max_pinned": 0
    }
}

RUN_MATRIX = [
    ("A1", "P-S"),
    ("A1", "P-C"),
    ("A1", "PF-S"),
    ("A1", "PF-C"),
    ("A1", "QUAR-S"),
    ("A1", "QUAR-C"),
    ("B2", "P-S"),
    ("B2", "P-C"),
    ("B2", "PF-S"),
    ("B2", "PF-C"),
    ("B2", "QUAR-S"),
    ("B2", "QUAR-C")
]


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


def generate_yaml_config(workload_id, config_id, run_dir):
    w = WORKLOADS[workload_id]
    c = CONFIGURATIONS[config_id]
    run_id = f"{workload_id}_{config_id}"

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
            "trace_file": os.path.join(REPO_ROOT, w["trace_file"]),
            "warm_hbm_preload": False
        },
        "log": {
            "print_log": False,
            "export_gantt": False,
            "output_directory": csv_dir + "/",
            "gantt_directory": raw_dir + "/",
            "traces_directory": traces_dir + "/"
        }
    }

    prefetch_sec = {
        "enabled": True,
        "mode": c["mode"]
    }

    if c["predictor"] == "st_moe":
        prefetch_sec["predictor"] = {
            "type": "st_moe",
            "confidence_threshold": 2,
            "online_adaptation": True
        }
    elif c["predictor"] == "cstp":
        prefetch_sec["predictor"] = {
            "type": "cstp",
            "alpha": 0.5,
            "smoothing_lambda": 1.0,
            "online_adaptation": True
        }
    else:
        prefetch_sec["predictor"] = {"type": "none"}

    prefetch_sec["candidate_selection"] = {
        "policy": "top_k",
        "budget": c["budget"]
    }
    prefetch_sec["scheduling"] = {
        "prediction_horizon": c["horizon"],
        "lookahead_tokens": 0,
        "demand_priority": True,
        "queue_enabled": True,
        "queue_capacity": 1024,
        "priority_policy": "deadline_score_rank",
        "duplicate_policy": "retain_highest_priority",
        "overflow_policy": "reject_lowest_priority"
    }
    prefetch_sec["transfer"] = {
        "model": "event_driven",
        "asynchronous": True
    }
    prefetch_sec["eviction_policy"] = c["eviction"]

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
    yaml_path = os.path.join(CONFIGS_DIR, f"{run_id}.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(doc, f, sort_keys=False, default_flow_style=False)

    manifest_dir = os.path.join(run_dir, "manifest")
    os.makedirs(manifest_dir, exist_ok=True)
    with open(os.path.join(manifest_dir, "effective_config.yaml"), "w") as f:
        yaml.dump(doc, f, sort_keys=False, default_flow_style=False)

    return yaml_path


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
        "demand_transfers_count": 0,
        "demand_bytes_transferred": 0,
        "total_exposed_stall_ns": 0.0,
        "avg_demand_transfer_latency_ns": 0.0,
        "matrix_hits": 0,
        "matrix_misses": 0,
        "matrix_hbm_hit_rate_pct": 0.0
    }
    if not os.path.exists(load_csv_path):
        return metrics

    try:
        df = pd.read_csv(load_csv_path)
        if "cache_hit_or_miss" in df.columns:
            miss_df = df[df["cache_hit_or_miss"] == "miss"]
            metrics["demand_transfers_count"] = int(len(miss_df))
            metrics["demand_bytes_transferred"] = int(miss_df["size_bytes"].sum())
            metrics["total_exposed_stall_ns"] = float(miss_df["duration"].sum())
            if len(miss_df) > 0:
                metrics["avg_demand_transfer_latency_ns"] = float(miss_df["duration"].mean())

            if "weight_tensor" in df.columns:
                matrix_df = df[df["weight_tensor"] != "all_weights"]
                if len(matrix_df) > 0:
                    m_hits = int(len(matrix_df[matrix_df["cache_hit_or_miss"] == "hit"]))
                    m_miss = int(len(matrix_df[matrix_df["cache_hit_or_miss"] == "miss"]))
                    metrics["matrix_hits"] = m_hits
                    metrics["matrix_misses"] = m_miss
                    total_m = m_hits + m_miss
                    if total_m > 0:
                        metrics["matrix_hbm_hit_rate_pct"] = float(m_hits / total_m * 100.0)
    except Exception as e:
        print(f"[WARN] Error reading {load_csv_path}: {e}")

    return metrics


def parse_step_summary_csv(summary_csv_path):
    metrics = {
        "sim_total_device_time_ns": 0,
        "sim_per_token_latency_avg_ms": 0.0,
        "sim_per_token_latency_std_ms": 0.0,
        "sim_total_moe_dram_ms": 0.0,
        "sim_total_moe_comp_ms": 0.0,
        "sim_total_attn_comp_ms": 0.0,
        "sim_total_fc_comp_ms": 0.0,
        "steps_count": 0
    }
    if not os.path.exists(summary_csv_path):
        return metrics

    try:
        df = pd.read_csv(summary_csv_path)
        if "latency" in df.columns:
            latencies_ns = df["latency"].tolist()
            latencies_ms = [l / 1e6 for l in latencies_ns]
            metrics["sim_total_device_time_ns"] = int(df["time"].iloc[-1])
            metrics["sim_per_token_latency_avg_ms"] = float(pd.Series(latencies_ms).mean())
            metrics["sim_per_token_latency_std_ms"] = float(pd.Series(latencies_ms).std())
            metrics["steps_count"] = len(df)
            if "moe_dram" in df.columns:
                metrics["sim_total_moe_dram_ms"] = float(df["moe_dram"].sum() / 1e6)
            if "moe_comp" in df.columns:
                metrics["sim_total_moe_comp_ms"] = float(df["moe_comp"].sum() / 1e6)
            if "attn_comp" in df.columns:
                metrics["sim_total_attn_comp_ms"] = float(df["attn_comp"].sum() / 1e6)
            if "fc_comp" in df.columns:
                metrics["sim_total_fc_comp_ms"] = float(df["fc_comp"].sum() / 1e6)
    except Exception as e:
        print(f"[WARN] Error reading {summary_csv_path}: {e}")

    return metrics


def parse_live_prediction_summary_json(pred_json_path):
    metrics = {
        "pred_events_total": 0,
        "pred_updates_total": 0,
        "pred_precision_overall": 0.0,
        "pred_recall_overall": 0.0,
        "pred_f1_overall": 0.0,
        "pred_hit_rate_overall": 0.0,
        "pred_mrr_overall": 0.0,
        "pred_ndcg_overall": 0.0,
        "pred_mean_latency_ns": 0.0,
        "pred_mean_update_latency_ns": 0.0,
        "pred_host_overhead_pct": 0.0
    }
    if not os.path.exists(pred_json_path):
        return metrics

    try:
        with open(pred_json_path, "r") as f:
            data = json.load(f)
            metrics["pred_events_total"] = data.get("total_prediction_events", 0)
            metrics["pred_updates_total"] = data.get("total_update_events", 0)
            metrics["pred_host_overhead_pct"] = data.get("prediction_overhead_percent", 0.0)
            overall = data.get("overall", {})
            metrics["pred_precision_overall"] = overall.get("precision", 0.0)
            metrics["pred_recall_overall"] = overall.get("recall", 0.0)
            metrics["pred_f1_overall"] = overall.get("f1", 0.0)
            metrics["pred_hit_rate_overall"] = overall.get("hit_rate", 0.0)
            metrics["pred_mrr_overall"] = overall.get("mrr", 0.0)
            metrics["pred_ndcg_overall"] = overall.get("ndcg", 0.0)
            metrics["pred_mean_latency_ns"] = overall.get("mean_pred_latency_ns", 0.0)
            metrics["pred_mean_update_latency_ns"] = overall.get("mean_upd_latency_ns", 0.0)
    except Exception as e:
        print(f"[WARN] Error reading {pred_json_path}: {e}")

    return metrics


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


def validate_run_outputs(workload_id, config_id, run_dir):
    w = WORKLOADS[workload_id]
    c = CONFIGURATIONS[config_id]
    run_id = f"{workload_id}_{config_id}"

    errors = []
    status_path = os.path.join(run_dir, "logs", "completion_status.txt")
    if not os.path.exists(status_path):
        errors.append(f"Missing {status_path}")
        return False, errors

    status_dict = parse_completion_status(status_path)
    if status_dict.get("STATUS") != "SUCCESS":
        errors.append(f"STATUS is not SUCCESS in {status_path} (got {status_dict.get('STATUS')})")

    iter_done = int(status_dict.get("ITERATIONS", 0))
    if iter_done != w["iter"]:
        errors.append(f"Completed iterations {iter_done} != expected {w['iter']}")

    routing_path = os.path.join(run_dir, "traces", "expert_routing.csv")
    r_lines = count_file_lines(routing_path)
    if r_lines != w["expected_routing_records"]:
        errors.append(f"expert_routing.csv lines {r_lines} != expected {w['expected_routing_records']}")

    cache_path = os.path.join(run_dir, "traces", "expert_cache.csv")
    c_lines = count_file_lines(cache_path)
    if c_lines < 100:
        errors.append(f"expert_cache.csv has only {c_lines} lines (expected > 100)")

    load_path = os.path.join(run_dir, "traces", "expert_load.csv")
    l_lines = count_file_lines(load_path)
    if l_lines < 100:
        errors.append(f"expert_load.csv has only {l_lines} lines (expected > 100)")

    # Config-specific checks
    if c["mode"] == "prediction_only":
        pred_json = os.path.join(run_dir, "live_prediction_summary.json")
        if not os.path.exists(pred_json):
            errors.append(f"Prediction-only missing {pred_json}")
        else:
            p_data = parse_live_prediction_summary_json(pred_json)
            if p_data["pred_events_total"] == 0:
                errors.append("Prediction-only recorded 0 prediction events")
        sub = int(status_dict.get("PREFETCH_TRANSFERS_SUBMITTED", 0))
        if sub != 0:
            errors.append(f"Prediction-only submitted {sub} speculative transfers (expected 0)")
        ins = int(status_dict.get("PREFETCHES_INSERTED_HBM", 0))
        if ins != 0:
            errors.append(f"Prediction-only inserted {ins} speculative prefetches to HBM (expected 0)")

    if c["mode"] == "hardware_async":
        disp = int(status_dict.get("CANDIDATES_DISPATCHED", 0))
        if disp == 0:
            errors.append("hardware_async dispatched 0 candidates")
        sub = int(status_dict.get("PREFETCH_TRANSFERS_SUBMITTED", 0))
        if sub == 0:
            errors.append("hardware_async submitted 0 transfers")
        comp = int(status_dict.get("PREFETCH_TRANSFERS_COMPLETED", 0))
        if comp == 0:
            errors.append("hardware_async completed 0 transfers")

        if not c["quarantine"]:
            # Standard LRU prefetch: speculative insertions should be 0
            spec_ins = int(status_dict.get("SPECULATIVE_INSERTIONS", 0))
            if spec_ins != 0:
                errors.append(f"PF-S/C (LRU) had {spec_ins} speculative quarantine insertions (expected 0)")
        else:
            # PABI quarantine: speculative insertions should be > 0
            spec_ins = int(status_dict.get("SPECULATIVE_INSERTIONS", 0))
            if spec_ins == 0:
                errors.append("QUAR-S/C had 0 speculative insertions into quarantine buffer")

        pin_intents = int(status_dict.get("PIN_INTENTS_CREATED", 0))
        if pin_intents != 0:
            errors.append(f"Pinning disabled in Phase 1, but got {pin_intents} pin intents created")

    if len(errors) == 0:
        return True, []
    return False, errors


def execute_single_run(workload_id, config_id, dry_run=False):
    w = WORKLOADS[workload_id]
    c = CONFIGURATIONS[config_id]
    run_id = f"{workload_id}_{config_id}"
    run_dir = os.path.join(ABLATION_DIR, run_id)

    print("\n" + "=" * 90)
    print(f"PREPARING RUN: {run_id} | Workload: {w['name']} | Config: {c['name']}")
    print(f"Directory: {run_dir}")
    print("=" * 90)

    # Crash recovery & refusal to overwrite
    if os.path.exists(run_dir):
        is_valid, validation_errors = validate_run_outputs(workload_id, config_id, run_dir)
        if is_valid:
            print(f"[RECOVERY] Run {run_id} is already COMPLETE and fully validated. Skipping duplicate run.")
            return True, run_dir, 0.0
        else:
            print(f"[WARNING] Run directory {run_dir} exists but outputs are INCOMPLETE or INVALID:")
            for err in validation_errors:
                print(f"   -> {err}")
            print(f"[CLEANUP] Removing incomplete outputs for clean restart of {run_id}...")
            shutil.rmtree(run_dir)

    os.makedirs(os.path.join(run_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "traces"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "queue"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "manifest"), exist_ok=True)

    yaml_path = generate_yaml_config(workload_id, config_id, run_dir)

    # Verify input trace exists and checksum matches
    full_trace_path = os.path.join(REPO_ROOT, w["trace_file"])
    if not os.path.exists(full_trace_path):
        raise FileNotFoundError(f"Input trace file missing: {full_trace_path}")
    trace_hash = compute_sha256(full_trace_path)
    if trace_hash != w["trace_sha256"]:
        raise ValueError(f"Trace checksum mismatch for {full_trace_path}: got {trace_hash}, expected {w['trace_sha256']}")

    exec_hash = compute_sha256(EXECUTABLE)
    yaml_hash = compute_sha256(yaml_path)

    # Setup environment
    env = os.environ.copy()
    env.pop("DEBUG_CACHE_CAPACITY", None)
    env["TRACE_DIR"] = os.path.join(run_dir, "traces")
    env["PREDICTION_OUTPUT_DIR"] = run_dir
    env["PREFETCH_QUEUE_OUTPUT_DIR"] = os.path.join(run_dir, "queue")
    env["USE_EVENT_DRIVEN_DRAM"] = "1"
    env["BYPASS_OFFCHIP_EXECUTION_CACHE"] = "0"
    env["DDR5_BANDWIDTH_GBPS"] = str(CALIBRATED_DDR5_BW_GBPS)
    env["EVICTION_POLICY"] = c["eviction"]

    if c["quarantine"]:
        env["PABI_QUARANTINE_ENABLED"] = "1"
    else:
        env["PABI_QUARANTINE_ENABLED"] = "0"

    env["PABI_PINNING_ENABLED"] = "0"
    env["PABI_MAX_PINNED_EXPERTS"] = "0"

    # Pre-execution settings printout
    print(f"Pre-flight Verification:")
    print(f"  - Trace: {w['trace_file']} (SHA256: {trace_hash[:16]}...)")
    print(f"  - Executable: {EXECUTABLE} (SHA256: {exec_hash[:16]}...)")
    print(f"  - Mode: {c['mode']} | Predictor: {c['predictor']} | Horizon: {c['horizon']} | Budget: {c['budget']}")
    print(f"  - Eviction: {c['eviction']} | Quarantine: {c['quarantine']} | Pinning: {c['pinning']}")
    print(f"  - DRAM BW: {CALIBRATED_DDR5_BW_GBPS} GB/s | Event-driven: 1")

    if dry_run:
        print("[DRY-RUN] Verification successful. Subprocess execution bypassed.")
        return True, run_dir, 0.0

    cmd = [EXECUTABLE, yaml_path]
    print(f"Launching subprocess: {' '.join(cmd)}")
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    t1 = time.perf_counter()
    host_sec = t1 - t0

    # Write logs
    stdout_log = os.path.join(run_dir, "logs", "stdout.log")
    stderr_log = os.path.join(run_dir, "logs", "stderr.log")
    with open(stdout_log, "w") as f:
        f.write(res.stdout)
    with open(stderr_log, "w") as f:
        f.write(res.stderr)

    if res.returncode != 0:
        print(f"[FATAL] Subprocess exited with return code {res.returncode}")
        print(f"STDERR tail:\n{res.stderr[-1000:]}")
        raise RuntimeError(f"Run {run_id} failed!")

    # Verify outputs
    is_valid, validation_errors = validate_run_outputs(workload_id, config_id, run_dir)
    if not is_valid:
        print(f"[FATAL] Validation failed for {run_id}:")
        for err in validation_errors:
            print(f"   -> {err}")
        raise RuntimeError(f"Validation failed for run {run_id}")

    # Create run manifest
    manifest = {
        "run_id": run_id,
        "workload_id": workload_id,
        "workload_name": w["name"],
        "configuration_id": config_id,
        "configuration_name": c["name"],
        "git_baseline_anchor": GIT_BASELINE_ANCHOR,
        "calibrated_ddr5_bw_gbps": CALIBRATED_DDR5_BW_GBPS,
        "expert_size_bytes": EXPERT_SIZE_BYTES,
        "start_time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "host_runtime_sec": host_sec,
        "exit_code": res.returncode,
        "checksums": {
            "trace_sha256": trace_hash,
            "executable_sha256": exec_hash,
            "yaml_sha256": yaml_hash
        },
        "effective_configuration": {
            "mode": c["mode"],
            "predictor": c["predictor"],
            "horizon": c["horizon"],
            "budget": c["budget"],
            "eviction_policy": c["eviction"],
            "quarantine_enabled": c["quarantine"],
            "pinning_enabled": c["pinning"],
            "speculative_fraction": c["spec_fraction"]
        },
        "workload_parameters": {
            "model_name": w["model_name"],
            "num_layers": w["num_layers"],
            "moe_layers": w["moe_layers"],
            "batch_size": w["batch_size"],
            "input_len": w["input_len"],
            "output_len": w["output_len"],
            "iter": w["iter"]
        },
        "validation_status": "VALIDATED_SUCCESS"
    }

    manifest_path = os.path.join(run_dir, "manifest", "run_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  -> SUCCESS in {host_sec:.2f} s. Manifest recorded.")
    return True, run_dir, host_sec


def extract_all_metrics():
    print("\n" + "=" * 90)
    print("EXTRACTING METRICS ACROSS ALL 12 PHASE 1 RUNS")
    print("=" * 90)

    combined_rows = []
    prediction_rows = []
    prefetch_rows = []
    hbm_rows = []
    ddr5_rows = []
    perf_rows = []

    for workload_id, config_id in RUN_MATRIX:
        run_id = f"{workload_id}_{config_id}"
        run_dir = os.path.join(ABLATION_DIR, run_id)
        w = WORKLOADS[workload_id]
        c = CONFIGURATIONS[config_id]

        status_path = os.path.join(run_dir, "logs", "completion_status.txt")
        status_dict = parse_completion_status(status_path)

        manifest_path = os.path.join(run_dir, "manifest", "run_manifest.json")
        host_sec = 0.0
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                mf = json.load(f)
                host_sec = mf.get("host_runtime_sec", 0.0)

        cache_m = parse_expert_cache_csv(os.path.join(run_dir, "traces", "expert_cache.csv"))
        load_m = parse_expert_load_csv(os.path.join(run_dir, "traces", "expert_load.csv"))
        pred_m = parse_live_prediction_summary_json(os.path.join(run_dir, "live_prediction_summary.json"))

        csv_files = glob.glob(os.path.join(run_dir, "csv", "*_ramul_decode.csv"))
        step_m = {}
        if len(csv_files) > 0:
            step_m = parse_step_summary_csv(csv_files[0])

        # Calculation of hit rates
        expert_hits = cache_m["hbm_demand_hits"]
        expert_misses = cache_m["hbm_demand_misses"]
        total_expert_lookups = expert_hits + expert_misses
        expert_hit_rate_pct = 0.0
        if total_expert_lookups > 0:
            expert_hit_rate_pct = (expert_hits / total_expert_lookups) * 100.0

        # Prefetch counts
        candidates_eval = int(status_dict.get("CANDIDATES_EVALUATED", 0))
        candidates_disp = int(status_dict.get("CANDIDATES_DISPATCHED", 0))
        transfers_sub = int(status_dict.get("PREFETCH_TRANSFERS_SUBMITTED", 0))
        transfers_comp = int(status_dict.get("PREFETCH_TRANSFERS_COMPLETED", 0))
        prefetches_ins = int(status_dict.get("PREFETCHES_INSERTED_HBM", 0))
        res_skipped = int(status_dict.get("RESIDENT_PREFETCHES_SKIPPED", 0))
        dup_prevented = int(status_dict.get("DUPLICATE_PREFETCHES_PREVENTED", 0))
        useful_hits = int(status_dict.get("PREFETCH_USEFUL_HITS", 0))
        inflight_attach = int(status_dict.get("DEMAND_INFLIGHT_ATTACHMENTS", 0))
        cancelled = int(status_dict.get("CANCELLED_PREFETCHES", 0))
        unused = prefetches_ins - useful_hits
        if unused < 0:
            unused = 0

        # DDR5 calculations
        demand_transfers = load_m["demand_transfers_count"]
        demand_bytes = load_m["demand_bytes_transferred"]
        spec_completed = transfers_comp
        spec_bytes = spec_completed * EXPERT_SIZE_BYTES
        total_ddr5_bytes = demand_bytes + spec_bytes
        total_ddr5_service_ns = (demand_transfers + spec_completed) * EXPERT_SERVICE_TIME_NS
        exposed_stall_ns = load_m["total_exposed_stall_ns"]
        exposed_stall_ms = exposed_stall_ns / 1e6

        # Simulation times
        sim_device_ms = step_m.get("sim_total_device_time_ns", 0) / 1e6
        per_token_ms = step_m.get("sim_per_token_latency_avg_ms", 0.0)
        per_token_std = step_m.get("sim_per_token_latency_std_ms", 0.0)

        # Baseline comparisons
        base_stall = w["baseline_stall_ms"]
        base_sim = w["baseline_sim_time_ms"]
        stall_reduction_pct = 0.0
        if base_stall > 0:
            stall_reduction_pct = ((base_stall - exposed_stall_ms) / base_stall) * 100.0

        sim_speedup = 0.0
        if sim_device_ms > 0:
            sim_speedup = base_sim / sim_device_ms

        # Combined row
        c_row = {
            "run_id": run_id,
            "workload_id": workload_id,
            "configuration_id": config_id,
            "configuration_name": c["name"],
            "predictor_type": c["predictor"],
            "prefetch_mode": c["mode"],
            "eviction_policy": c["eviction"],
            "quarantine_enabled": c["quarantine"],
            "pinning_enabled": c["pinning"],
            "host_runtime_sec": host_sec,
            "sim_device_time_ms": sim_device_ms,
            "sim_per_token_avg_ms": per_token_ms,
            "total_exposed_stall_ms": exposed_stall_ms,
            "stall_reduction_pct_vs_base": stall_reduction_pct,
            "sim_speedup_vs_base": sim_speedup,
            "expert_hbm_hit_rate_pct": expert_hit_rate_pct,
            "matrix_hbm_hit_rate_pct": load_m["matrix_hbm_hit_rate_pct"],
            "expert_demand_hits": expert_hits,
            "expert_demand_misses": expert_misses,
            "demand_transfers_count": demand_transfers,
            "demand_bytes_transferred": demand_bytes,
            "prefetch_dispatched": candidates_disp,
            "prefetch_transfers_submitted": transfers_sub,
            "prefetch_transfers_completed": transfers_comp,
            "prefetch_useful_hits": useful_hits,
            "demand_inflight_attachments": inflight_attach,
            "speculative_insertions": cache_m["hbm_speculative_inserts"],
            "speculative_promotions": cache_m["hbm_promotions"],
            "speculative_evictions": cache_m["hbm_speculative_evictions"],
            "pred_precision": pred_m["pred_precision_overall"],
            "pred_recall": pred_m["pred_recall_overall"],
            "pred_f1": pred_m["pred_f1_overall"],
            "pred_hit_rate": pred_m["pred_hit_rate_overall"],
            "pred_mean_latency_ns": pred_m["pred_mean_latency_ns"],
            "pred_host_overhead_pct": pred_m["pred_host_overhead_pct"]
        }
        combined_rows.append(c_row)

        # 1. Prediction metrics row
        p_row = {
            "run_id": run_id,
            "workload_id": workload_id,
            "configuration_id": config_id,
            "predictor_type": c["predictor"],
            "total_prediction_events": pred_m["pred_events_total"],
            "total_update_events": pred_m["pred_updates_total"],
            "precision": pred_m["pred_precision_overall"],
            "recall": pred_m["pred_recall_overall"],
            "f1": pred_m["pred_f1_overall"],
            "hit_rate": pred_m["pred_hit_rate_overall"],
            "mrr": pred_m["pred_mrr_overall"],
            "ndcg": pred_m["pred_ndcg_overall"],
            "mean_pred_latency_ns": pred_m["pred_mean_latency_ns"],
            "mean_upd_latency_ns": pred_m["pred_mean_update_latency_ns"],
            "host_overhead_pct": pred_m["pred_host_overhead_pct"]
        }
        prediction_rows.append(p_row)

        # 2. Prefetch lifecycle row
        pf_row = {
            "run_id": run_id,
            "workload_id": workload_id,
            "configuration_id": config_id,
            "candidates_evaluated": candidates_eval,
            "candidates_dispatched": candidates_disp,
            "transfers_submitted": transfers_sub,
            "transfers_completed": transfers_comp,
            "prefetches_inserted_hbm": prefetches_ins,
            "resident_skipped": res_skipped,
            "duplicates_prevented": dup_prevented,
            "useful_hits": useful_hits,
            "inflight_attachments": inflight_attach,
            "cancelled_prefetches": cancelled,
            "unused_prefetches": unused
        }
        prefetch_rows.append(pf_row)

        # 3. HBM metrics row
        h_row = {
            "run_id": run_id,
            "workload_id": workload_id,
            "configuration_id": config_id,
            "eviction_policy": c["eviction"],
            "quarantine_enabled": c["quarantine"],
            "expert_demand_hits": expert_hits,
            "expert_demand_misses": expert_misses,
            "expert_hbm_hit_rate_pct": expert_hit_rate_pct,
            "matrix_hits": load_m["matrix_hits"],
            "matrix_misses": load_m["matrix_misses"],
            "matrix_hbm_hit_rate_pct": load_m["matrix_hbm_hit_rate_pct"],
            "demand_inserts": cache_m["hbm_demand_inserts"],
            "demand_evictions": cache_m["hbm_demand_evictions"],
            "speculative_inserts": cache_m["hbm_speculative_inserts"],
            "speculative_promotions": cache_m["hbm_promotions"],
            "speculative_evictions": cache_m["hbm_speculative_evictions"],
            "speculative_rejects": cache_m["hbm_speculative_rejects"],
            "pin_intents_created": cache_m["hbm_pin_intents_created"],
            "pins_applied": cache_m["hbm_pins_applied"],
            "pins_expired": cache_m["hbm_pins_expired"]
        }
        hbm_rows.append(h_row)

        # 4. DDR5 metrics row
        d_row = {
            "run_id": run_id,
            "workload_id": workload_id,
            "configuration_id": config_id,
            "demand_transfers_count": demand_transfers,
            "demand_bytes": demand_bytes,
            "speculative_transfers_completed": spec_completed,
            "speculative_bytes": spec_bytes,
            "total_bytes_transferred": total_ddr5_bytes,
            "total_ddr5_service_time_ns": total_ddr5_service_ns,
            "total_ddr5_service_time_ms": total_ddr5_service_ns / 1e6,
            "exposed_stall_ns": exposed_stall_ns,
            "exposed_stall_ms": exposed_stall_ms,
            "avg_demand_transfer_latency_ns": load_m["avg_demand_transfer_latency_ns"],
            "avg_demand_transfer_latency_ms": load_m["avg_demand_transfer_latency_ns"] / 1e6
        }
        ddr5_rows.append(d_row)

        # 5. Performance metrics row
        perf_row = {
            "run_id": run_id,
            "workload_id": workload_id,
            "configuration_id": config_id,
            "host_runtime_sec": host_sec,
            "sim_total_device_time_ms": sim_device_ms,
            "sim_per_token_avg_ms": per_token_ms,
            "sim_per_token_std_ms": per_token_std,
            "sim_moe_dram_ms": step_m.get("sim_total_moe_dram_ms", 0.0),
            "sim_moe_comp_ms": step_m.get("sim_total_moe_comp_ms", 0.0),
            "sim_attn_comp_ms": step_m.get("sim_total_attn_comp_ms", 0.0),
            "sim_fc_comp_ms": step_m.get("sim_total_fc_comp_ms", 0.0),
            "baseline_sim_time_ms": base_sim,
            "baseline_stall_ms": base_stall,
            "speedup_vs_baseline": sim_speedup,
            "stall_reduction_pct": stall_reduction_pct
        }
        perf_rows.append(perf_row)

    # Convert to DataFrames and save
    df_combined = pd.DataFrame(combined_rows)
    df_pred = pd.DataFrame(prediction_rows)
    df_pf = pd.DataFrame(prefetch_rows)
    df_hbm = pd.DataFrame(hbm_rows)
    df_ddr5 = pd.DataFrame(ddr5_rows)
    df_perf = pd.DataFrame(perf_rows)

    os.makedirs(ABLATION_DIR, exist_ok=True)
    df_combined.to_csv(os.path.join(ABLATION_DIR, "phase1_combined_results.csv"), index=False)
    df_pred.to_csv(os.path.join(ABLATION_DIR, "phase1_prediction_metrics.csv"), index=False)
    df_pf.to_csv(os.path.join(ABLATION_DIR, "phase1_prefetch_lifecycle.csv"), index=False)
    df_hbm.to_csv(os.path.join(ABLATION_DIR, "phase1_hbm_metrics.csv"), index=False)
    df_ddr5.to_csv(os.path.join(ABLATION_DIR, "phase1_ddr5_metrics.csv"), index=False)
    df_perf.to_csv(os.path.join(ABLATION_DIR, "phase1_performance_metrics.csv"), index=False)

    print(f"CSVs successfully exported to {ABLATION_DIR}/")
    return df_combined, df_pred, df_pf, df_hbm, df_ddr5, df_perf


def generate_checksum_manifest():
    lines = []
    # Workload traces
    for wid, w in WORKLOADS.items():
        tp = os.path.join(REPO_ROOT, w["trace_file"])
        lines.append(f"{compute_sha256(tp)}  {w['trace_file']}")
    # Executable
    lines.append(f"{compute_sha256(EXECUTABLE)}  build/run_deepseekv3_mixtral_style")
    # Output CSVs
    csv_names = [
        "phase1_combined_results.csv",
        "phase1_prediction_metrics.csv",
        "phase1_prefetch_lifecycle.csv",
        "phase1_hbm_metrics.csv",
        "phase1_ddr5_metrics.csv",
        "phase1_performance_metrics.csv"
    ]
    for cname in csv_names:
        cpath = os.path.join(ABLATION_DIR, cname)
        if os.path.exists(cpath):
            lines.append(f"{compute_sha256(cpath)}  results/final_comparative_campaign/ablation/{cname}")

    out_chk = os.path.join(ABLATION_DIR, "checksums.sha256")
    with open(out_chk, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Checksum manifest saved to {out_chk}")


def print_summary_table(df_combined):
    print("\n" + "=" * 130)
    print("PHASE 1 COMPONENT ABLATION LADDER EXECUTION SUMMARY")
    print("=" * 130)
    header = (
        f"{'Run ID':<10} | {'Config Name':<25} | {'Disp':<6} | {'Comp':<6} | "
        f"{'Hits':<6} | {'HBM Hit %':<9} | {'Stall (ms)':<10} | {'E2E (ms)':<10} | {'Speedup':<8} | {'Host (s)':<8}"
    )
    print(header)
    print("-" * len(header))
    for _, r in df_combined.iterrows():
        rid = str(r["run_id"])
        cname = str(r["configuration_name"])
        disp = str(r["prefetch_dispatched"])
        comp = str(r["prefetch_transfers_completed"])
        hits = str(r["prefetch_useful_hits"])
        hit_pct = f"{r['expert_hbm_hit_rate_pct']:.2f}%"
        stall_ms = f"{r['total_exposed_stall_ms']:.1f}"
        e2e_ms = f"{r['sim_device_time_ms']:.1f}"
        spdup = f"{r['sim_speedup_vs_base']:.3f}x"
        host_s = f"{r['host_runtime_sec']:.1f}s"
        print(f"{rid:<10} | {cname:<25} | {disp:<6} | {comp:<6} | "
              f"{hits:<6} | {hit_pct:<9} | {stall_ms:<10} | {e2e_ms:<10} | {spdup:<8} | {host_s:<8}")
    print("=" * 130)


def main():
    check_zero_ternary(__file__)

    dry_run = "--dry-run" in sys.argv
    verify_only = "--verify-only" in sys.argv
    reparse_only = "--reparse-only" in sys.argv

    target_run = None
    if "--run" in sys.argv:
        idx = sys.argv.index("--run")
        if idx + 1 < len(sys.argv):
            target_run = sys.argv[idx + 1]

    target_workload = None
    if "--workload" in sys.argv:
        idx = sys.argv.index("--workload")
        if idx + 1 < len(sys.argv):
            target_workload = sys.argv[idx + 1]

    active_matrix = []
    for wid, cid in RUN_MATRIX:
        rid = f"{wid}_{cid}"
        include = True
        if target_run is not None:
            if rid != target_run:
                include = False
        if target_workload is not None:
            if wid != target_workload:
                include = False
        if include:
            active_matrix.append((wid, cid))

    if reparse_only:
        df_combined, _, _, _, _, _ = extract_all_metrics()
        generate_checksum_manifest()
        print_summary_table(df_combined)
        return

    print("\n" + "=" * 90)
    print("STAGE 2B: PHASE 1 COMPONENT ABLATION LADDER EXECUTION")
    print(f"Total Runs in Batch: {len(active_matrix)}")
    print(f"Git Anchor: {GIT_BASELINE_ANCHOR}")
    print(f"Calibrated DDR5 Bandwidth: {CALIBRATED_DDR5_BW_GBPS} GB/s")
    print("=" * 90)

    total_start_time = time.perf_counter()
    runtimes = []

    for wid, cid in active_matrix:
        success, run_dir, host_sec = execute_single_run(wid, cid, dry_run=dry_run)
        runtimes.append((f"{wid}_{cid}", host_sec))

    total_end_time = time.perf_counter()
    actual_total_sec = total_end_time - total_start_time
    actual_total_min = actual_total_sec / 60.0

    print("\n" + "=" * 90)
    print(f"BATCH COMPLETED ({len(active_matrix)} runs) in {actual_total_min:.2f} minutes ({actual_total_sec:.1f} seconds).")
    print("=" * 90)

    if not dry_run and len(active_matrix) == len(RUN_MATRIX):
        df_combined, _, _, _, _, _ = extract_all_metrics()
        generate_checksum_manifest()
        print_summary_table(df_combined)


if __name__ == "__main__":
    main()
