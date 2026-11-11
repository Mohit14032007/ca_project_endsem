#!/usr/bin/env python3
"""
Phase 4D.5C: 10-Token Bounded Production Evaluation Matrix Runner
Evaluates 9 configurations across ST-MoE, CSTP, LRU, and PABI on ms_task12_t128_l75.csv.
Authoritative DDR5 Bandwidth: 22.578 GB/s (Calibrated).
Zero-ternary operator compliance enforced.
"""

import os
import sys
import subprocess
import shutil
import time
import json
import ast
import pandas as pd
import yaml

REPO_ROOT = "/home/mohman/ca_project/LLMSimulator"
CONFIGS_DIR = os.path.join(REPO_ROOT, "configs", "phase_4d5c")
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "phase_4d5c")
EXECUTABLE = os.path.join(REPO_ROOT, "build", "run_deepseekv3_mixtral_style")
TRACE_FILE = "results/endsem_phase_e/final_baseline/traces/ms_task12_t128_l75.csv"

# Authoritative Ramulator2-calibrated DDR5 bandwidth
AUTHORITATIVE_DDR5_BW_GBPS = 22.578

CONFIGURATIONS = [
    {
        "id": "c01_baseline_lru",
        "name": "Baseline (None + LRU)",
        "mode": "none",
        "predictor": "none",
        "eviction": "lru",
        "quarantine": False,
        "pinning": False,
        "max_pinned": 0,
        "spec_fraction": 0.0
    },
    {
        "id": "c02_stmoe_predonly",
        "name": "ST-MoE Prediction-Only",
        "mode": "prediction_only",
        "predictor": "st_moe",
        "eviction": "lru",
        "quarantine": False,
        "pinning": False,
        "max_pinned": 0,
        "spec_fraction": 0.0
    },
    {
        "id": "c03_cstp_predonly",
        "name": "CSTP Prediction-Only",
        "mode": "prediction_only",
        "predictor": "cstp",
        "eviction": "lru",
        "quarantine": False,
        "pinning": False,
        "max_pinned": 0,
        "spec_fraction": 0.0
    },
    {
        "id": "c04_stmoe_prefetch_lru",
        "name": "ST-MoE + LRU Prefetch",
        "mode": "hardware_async",
        "predictor": "st_moe",
        "eviction": "lru",
        "quarantine": False,
        "pinning": False,
        "max_pinned": 0,
        "spec_fraction": 0.0
    },
    {
        "id": "c05_cstp_prefetch_lru",
        "name": "CSTP + LRU Prefetch",
        "mode": "hardware_async",
        "predictor": "cstp",
        "eviction": "lru",
        "quarantine": False,
        "pinning": False,
        "max_pinned": 0,
        "spec_fraction": 0.0
    },
    {
        "id": "c06_stmoe_pabi_quarantine",
        "name": "ST-MoE + PABI Quarantine",
        "mode": "hardware_async",
        "predictor": "st_moe",
        "eviction": "pabi",
        "quarantine": True,
        "pinning": False,
        "max_pinned": 0,
        "spec_fraction": 0.20
    },
    {
        "id": "c07_cstp_pabi_quarantine",
        "name": "CSTP + PABI Quarantine",
        "mode": "hardware_async",
        "predictor": "cstp",
        "eviction": "pabi",
        "quarantine": True,
        "pinning": False,
        "max_pinned": 0,
        "spec_fraction": 0.20
    },
    {
        "id": "c08_stmoe_pabi_pinning",
        "name": "ST-MoE + PABI Pinning",
        "mode": "hardware_async",
        "predictor": "st_moe",
        "eviction": "pabi",
        "quarantine": True,
        "pinning": True,
        "max_pinned": 4,
        "spec_fraction": 0.20
    },
    {
        "id": "c09_cstp_pabi_pinning",
        "name": "CSTP + PABI Pinning",
        "mode": "hardware_async",
        "predictor": "cstp",
        "eviction": "pabi",
        "quarantine": True,
        "pinning": True,
        "max_pinned": 4,
        "spec_fraction": 0.20
    }
]


def check_zero_ternary(script_path):
    with open(script_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=script_path)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            count += 1
    if count > 0:
        raise RuntimeError(f"FATAL: Found {count} ternary expressions in {script_path}")
    print(f"[VERIFY] Zero ternary expressions verified in {script_path}")


def generate_yaml(cfg):
    cid = cfg["id"]
    out_dir = os.path.join(RESULTS_DIR, cid)
    
    doc = {
        "model": {
            "model_name": "deepseekV3",
            "num_layers": 78  # 3 dense + 75 MoE layers
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
            "max_batch_size": 1,
            "max_process_token": 0
        },
        "simulation": {
            "data": "synthesis",
            "input_len": 128,
            "output_len": 128,
            "precision_byte": 2,
            "skewness": 0.0,
            "iter": 10,
            "injection_rate": 0,
            "exit_out_of_memory": False,
            "mem_cap_limit": False,
            "trace_file": TRACE_FILE,
            "warm_hbm_preload": False
        },
        "log": {
            "print_log": False,
            "export_gantt": False,
            "output_directory": os.path.join(out_dir, "csv") + "/",
            "gantt_directory": os.path.join(out_dir, "raw") + "/",
            "traces_directory": os.path.join(out_dir, "traces") + "/"
        }
    }

    prefetch_sec = {}
    if cfg["mode"] == "none":
        prefetch_sec["enabled"] = False
        prefetch_sec["mode"] = "none"
        prefetch_sec["predictor"] = {"type": "none"}
    else:
        prefetch_sec["enabled"] = True
        prefetch_sec["mode"] = cfg["mode"]
        if cfg["predictor"] == "st_moe":
            prefetch_sec["predictor"] = {
                "type": "st_moe",
                "confidence_threshold": 2,
                "online_adaptation": True
            }
        elif cfg["predictor"] == "cstp":
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
            "budget": 4
        }
        prefetch_sec["scheduling"] = {
            "prediction_horizon": 1,
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
        prefetch_sec["eviction_policy"] = cfg["eviction"]

        if cfg["quarantine"]:
            prefetch_sec["pabi"] = {
                "quarantine_enabled": True,
                "speculative_fraction": cfg["spec_fraction"],
                "pinning_enabled": cfg["pinning"],
                "max_pinned_experts": cfg["max_pinned"],
                "pin_selection_policy": "top_ranked",
                "pin_expiration_policy": "stage_deadline"
            }

    doc["simulation"]["prefetching"] = prefetch_sec

    os.makedirs(CONFIGS_DIR, exist_ok=True)
    yaml_path = os.path.join(CONFIGS_DIR, f"{cid}.yaml")
    with open(yaml_path, "w") as f:
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
        "hbm_speculative_rejects": 0,
        "hbm_speculative_reject_pinned_cap": 0
    }
    if not os.path.exists(cache_csv_path):
        return metrics

    try:
        df = pd.read_csv(cache_csv_path)
        if "action" in df.columns:
            actions = df["action"].value_counts().to_dict()
            metrics["hbm_demand_hits"] = int(actions.get("HIT", 0))
            metrics["hbm_demand_misses"] = int(actions.get("MISS", 0))
            metrics["hbm_speculative_inserts"] = int(actions.get("SPECULATIVE_INSERT", 0))
            metrics["hbm_demand_inserts"] = int(actions.get("DEMAND_INSERT", 0))
            metrics["hbm_promotions"] = int(actions.get("PROMOTION", 0))
            metrics["hbm_speculative_evictions"] = int(actions.get("SPECULATIVE_EVICT", 0))
            metrics["hbm_demand_evictions"] = int(actions.get("DEMAND_EVICT", 0)) + int(actions.get("EVICT", 0))
            metrics["hbm_pin_intents_created"] = int(actions.get("PIN_INTENT_CREATED", 0))
            metrics["hbm_pins_applied"] = int(actions.get("PIN_APPLIED", 0))
            metrics["hbm_pins_expired"] = int(actions.get("PIN_EXPIRED", 0))
            metrics["hbm_pins_cleared_on_promotion"] = int(actions.get("PIN_CLEARED_ON_PROMOTION", 0))
            metrics["hbm_pinned_eviction_skipped"] = int(actions.get("PINNED_EVICTION_SKIPPED", 0))
            metrics["hbm_speculative_rejects"] = int(actions.get("SPECULATIVE_REJECT", 0))
            metrics["hbm_speculative_reject_pinned_cap"] = int(actions.get("SPECULATIVE_REJECT_PINNED_CAPACITY", 0))
    except Exception as e:
        print(f"[WARN] Error reading {cache_csv_path}: {e}")

    return metrics


def parse_expert_load_csv(load_csv_path):
    metrics = {
        "demand_transfers_count": 0,
        "demand_bytes_transferred": 0,
        "total_exposed_stall_ns": 0.0,
        "avg_demand_transfer_latency_ns": 0.0
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
    except Exception as e:
        print(f"[WARN] Error reading {load_csv_path}: {e}")

    return metrics


def parse_step_summary_csv(summary_csv_path):
    metrics = {
        "sim_total_device_time_ns": 0,
        "sim_per_token_latency_avg_ms": 0.0,
        "sim_per_token_latency_std_ms": 0.0,
        "sim_per_token_latencies_ms": [],
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
            metrics["sim_per_token_latencies_ms"] = latencies_ms
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


def run_configuration(cfg, yaml_path):
    cid = cfg["id"]
    out_dir = os.path.join(RESULTS_DIR, cid)
    
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.join(out_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "traces"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "queue"), exist_ok=True)

    env = os.environ.copy()
    env.pop("DEBUG_CACHE_CAPACITY", None)
    
    env["TRACE_DIR"] = os.path.join(out_dir, "traces")
    env["PREDICTION_OUTPUT_DIR"] = out_dir
    env["PREFETCH_QUEUE_OUTPUT_DIR"] = os.path.join(out_dir, "queue")
    env["USE_EVENT_DRIVEN_DRAM"] = "1"
    env["BYPASS_OFFCHIP_EXECUTION_CACHE"] = "0"
    env["DDR5_BANDWIDTH_GBPS"] = str(AUTHORITATIVE_DDR5_BW_GBPS)

    env["EVICTION_POLICY"] = cfg["eviction"]
    if cfg["quarantine"]:
        env["PABI_QUARANTINE_ENABLED"] = "1"
    else:
        env["PABI_QUARANTINE_ENABLED"] = "0"

    if cfg["pinning"]:
        env["PABI_PINNING_ENABLED"] = "1"
        env["PABI_MAX_PINNED_EXPERTS"] = str(cfg["max_pinned"])
    else:
        env["PABI_PINNING_ENABLED"] = "0"
        env["PABI_MAX_PINNED_EXPERTS"] = "0"

    print("\n" + "=" * 80)
    print(f"RUNNING CONFIGURATION: {cfg['name']} ({cid})")
    print(f"Workload: 10 Tokens, 75 MoE Layers, Top-8 Routing")
    print(f"Bandwidth: {AUTHORITATIVE_DDR5_BW_GBPS} GB/s (Calibrated)")
    print(f"Output directory: {out_dir}")
    print("=" * 80)

    cmd = [EXECUTABLE, yaml_path]
    t_start = time.perf_counter()
    res = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    t_end = time.perf_counter()
    host_runtime_sec = t_end - t_start

    stdout_path = os.path.join(out_dir, "logs", "stdout.log")
    stderr_path = os.path.join(out_dir, "logs", "stderr.log")
    with open(stdout_path, "w") as f:
        f.write(res.stdout)
    with open(stderr_path, "w") as f:
        f.write(res.stderr)

    if res.returncode != 0:
        print(f"[FATAL] Configuration {cid} FAILED with return code {res.returncode}")
        print(f"STDERR:\n{res.stderr[-1000:]}")
        raise RuntimeError(f"Configuration {cid} execution failed!")

    # Parse status file
    status_path = os.path.join(out_dir, "logs", "completion_status.txt")
    status_dict = {}
    if os.path.exists(status_path):
        with open(status_path, "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    status_dict[k] = v

    status_dict["success"] = True
    status_dict["host_runtime_sec"] = host_runtime_sec

    # Parse detailed CSV and JSON artifacts
    cache_metrics = parse_expert_cache_csv(os.path.join(out_dir, "traces", "expert_cache.csv"))
    load_metrics = parse_expert_load_csv(os.path.join(out_dir, "traces", "expert_load.csv"))
    pred_metrics = parse_live_prediction_summary_json(os.path.join(out_dir, "live_prediction_summary.json"))

    # Locate step summary CSV in csv/ directory
    csv_dir = os.path.join(out_dir, "csv")
    csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith(".csv")]
    step_metrics = {}
    if len(csv_files) > 0:
        step_metrics = parse_step_summary_csv(csv_files[0])

    record = dict(cfg)
    record["yaml_config_path"] = yaml_path
    record["host_runtime_sec"] = host_runtime_sec
    record.update(status_dict)
    record.update(cache_metrics)
    record.update(load_metrics)
    record.update(pred_metrics)
    record.update(step_metrics)

    # Derived summary metrics
    dispatched = int(status_dict.get("CANDIDATES_DISPATCHED", 0))
    submitted = int(status_dict.get("PREFETCH_TRANSFERS_SUBMITTED", 0))
    completed = int(status_dict.get("PREFETCH_TRANSFERS_COMPLETED", 0))
    inserted = int(status_dict.get("PREFETCHES_INSERTED_HBM", 0))
    useful_hits = int(status_dict.get("PREFETCH_USEFUL_HITS", 0))
    inflight_attach = int(status_dict.get("DEMAND_INFLIGHT_ATTACHMENTS", 0))
    cancelled = int(status_dict.get("CANCELLED_PREFETCHES", 0))

    transfers_still_inflight = submitted - completed - cancelled
    if transfers_still_inflight < 0:
        transfers_still_inflight = 0
    record["transfers_still_inflight"] = transfers_still_inflight

    unused_prefetches = inserted - useful_hits
    if unused_prefetches < 0:
        unused_prefetches = 0
    record["unused_prefetches"] = unused_prefetches

    expert_size_bytes = 88080384
    record["speculative_bytes_transferred"] = completed * expert_size_bytes
    record["total_ddr5_bytes_transferred"] = record["demand_bytes_transferred"] + record["speculative_bytes_transferred"]

    total_demands = record["hbm_demand_hits"] + record["hbm_demand_misses"]
    hbm_hit_rate = 0.0
    if total_demands > 0:
        hbm_hit_rate = (record["hbm_demand_hits"] / total_demands) * 100.0
    record["hbm_hit_rate_pct"] = hbm_hit_rate

    print(f"  -> SUCCESS in {host_runtime_sec:.2f} s")
    print(f"     Simulated E2E Device Latency: {record.get('sim_total_device_time_ns', 0) / 1e6:.2f} ms")
    print(f"     Average Per-Token Latency:    {record.get('sim_per_token_latency_avg_ms', 0.0):.2f} ms")
    print(f"     Exposed Memory Stall Time:    {record.get('total_exposed_stall_ns', 0.0) / 1e6:.2f} ms")
    print(f"     HBM Hit Rate:                 {record.get('hbm_hit_rate_pct', 0.0):.2f}% "
          f"({record.get('hbm_demand_hits', 0)} hits / {record.get('hbm_demand_misses', 0)} misses)")
    print(f"     Prefetch Dispatched:          {dispatched} (Submitted: {submitted}, Completed: {completed})")
    print(f"     Useful Prefetch Hits:         {useful_hits} (In-flight Attach: {inflight_attach})")
    print(f"     Speculative Insertions:       {record.get('hbm_speculative_inserts', 0)} "
          f"(Promotions: {record.get('hbm_promotions', 0)}, Evictions: {record.get('hbm_speculative_evictions', 0)})")
    print(f"     Pin Intents:                  {record.get('hbm_pin_intents_created', 0)} "
          f"(Expired: {record.get('hbm_pins_expired', 0)})")

    return record


def reparse_existing_results():
    prev_json_path = os.path.join(RESULTS_DIR, "production_matrix_summary.json")
    prev_runtimes = {}
    if os.path.exists(prev_json_path):
        try:
            with open(prev_json_path, "r") as f:
                old_list = json.load(f)
                for item in old_list:
                    prev_runtimes[item["id"]] = item.get("host_runtime_sec", 0.0)
        except Exception as e:
            print(f"[WARN] Failed to read previous runtimes: {e}")

    summary_results = []
    for cfg in CONFIGURATIONS:
        cid = cfg["id"]
        out_dir = os.path.join(RESULTS_DIR, cid)
        yaml_path = os.path.join(CONFIGS_DIR, f"{cid}.yaml")
        status_path = os.path.join(out_dir, "logs", "completion_status.txt")
        status_dict = {}
        if os.path.exists(status_path):
            with open(status_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line:
                        k, v = line.split("=", 1)
                        status_dict[k] = v

        status_dict["success"] = True
        host_runtime_sec = prev_runtimes.get(cid, 0.0)
        status_dict["host_runtime_sec"] = host_runtime_sec

        cache_metrics = parse_expert_cache_csv(os.path.join(out_dir, "traces", "expert_cache.csv"))
        load_metrics = parse_expert_load_csv(os.path.join(out_dir, "traces", "expert_load.csv"))
        pred_metrics = parse_live_prediction_summary_json(os.path.join(out_dir, "live_prediction_summary.json"))

        csv_dir = os.path.join(out_dir, "csv")
        csv_files = []
        if os.path.exists(csv_dir):
            for fname in os.listdir(csv_dir):
                if fname.endswith(".csv"):
                    csv_files.append(os.path.join(csv_dir, fname))

        step_metrics = {}
        if len(csv_files) > 0:
            step_metrics = parse_step_summary_csv(csv_files[0])

        record = dict(cfg)
        record["yaml_config_path"] = yaml_path
        record["host_runtime_sec"] = host_runtime_sec
        record.update(status_dict)
        record.update(cache_metrics)
        record.update(load_metrics)
        record.update(pred_metrics)
        record.update(step_metrics)

        dispatched = int(status_dict.get("CANDIDATES_DISPATCHED", 0))
        submitted = int(status_dict.get("PREFETCH_TRANSFERS_SUBMITTED", 0))
        completed = int(status_dict.get("PREFETCH_TRANSFERS_COMPLETED", 0))
        inserted = int(status_dict.get("PREFETCHES_INSERTED_HBM", 0))
        useful_hits = int(status_dict.get("PREFETCH_USEFUL_HITS", 0))
        inflight_attach = int(status_dict.get("DEMAND_INFLIGHT_ATTACHMENTS", 0))
        cancelled = int(status_dict.get("CANCELLED_PREFETCHES", 0))

        transfers_still_inflight = submitted - completed - cancelled
        if transfers_still_inflight < 0:
            transfers_still_inflight = 0
        record["transfers_still_inflight"] = transfers_still_inflight

        unused_prefetches = inserted - useful_hits
        if unused_prefetches < 0:
            unused_prefetches = 0
        record["unused_prefetches"] = unused_prefetches

        spec_bytes = completed * 88080384
        record["speculative_bytes_transferred"] = spec_bytes

        total_bytes = int(record.get("demand_bytes_transferred", 0)) + spec_bytes
        record["total_ddr5_bytes_transferred"] = total_bytes

        hits = int(record.get("hbm_demand_hits", 0))
        misses = int(record.get("hbm_demand_misses", 0))
        total_acc = hits + misses
        hit_rate = 0.0
        if total_acc > 0:
            hit_rate = (hits / total_acc) * 100.0
        record["hbm_hit_rate_pct"] = hit_rate

        summary_results.append(record)

    return summary_results


def main():
    check_zero_ternary(__file__)
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary_results = []
    if "--reparse" in sys.argv:
        print("[INFO] Re-parsing existing matrix results without rerunning simulations...")
        summary_results = reparse_existing_results()
    else:
        for cfg in CONFIGURATIONS:
            yaml_path = generate_yaml(cfg)
            record = run_configuration(cfg, yaml_path)
            summary_results.append(record)

    # Save consolidated JSON
    json_path = os.path.join(RESULTS_DIR, "production_matrix_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_results, f, indent=2)

    # Save consolidated CSV
    csv_path = os.path.join(RESULTS_DIR, "production_matrix_summary.csv")
    df = pd.DataFrame(summary_results)
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 120)
    print("PHASE 4D.5C 10-TOKEN BOUNDED PRODUCTION EVALUATION RESULTS MATRIX")
    print("=" * 120)
    header = (
        f"{'#':<3} | {'Configuration':<26} | {'Dispatched':<10} | {'Completed':<9} | "
        f"{'Hits':<5} | {'HBM Hit %':<9} | {'Stall (ms)':<10} | {'E2E (ms)':<10} | {'TokLat(ms)':<10} | {'Host (s)':<8}"
    )
    print(header)
    print("-" * len(header))

    idx = 1
    for r in summary_results:
        cid = r["id"]
        name = r["name"]
        dispatched = r.get("CANDIDATES_DISPATCHED", "0")
        completed = r.get("PREFETCH_TRANSFERS_COMPLETED", "0")
        hits = r.get("PREFETCH_USEFUL_HITS", "0")
        hit_pct = f"{r.get('hbm_hit_rate_pct', 0.0):.2f}%"
        stall_ms = f"{r.get('total_exposed_stall_ns', 0.0) / 1e6:.2f}"
        e2e_ms = f"{r.get('sim_total_device_time_ns', 0) / 1e6:.2f}"
        tok_ms = f"{r.get('sim_per_token_latency_avg_ms', 0.0):.2f}"
        sec = f"{r.get('host_runtime_sec', 0.0):.2f}"
        print(f"{idx:<3} | {name:<26} | {dispatched:<10} | {completed:<9} | "
              f"{hits:<5} | {hit_pct:<9} | {stall_ms:<10} | {e2e_ms:<10} | {tok_ms:<10} | {sec:<8}")
        idx += 1

    print("=" * 120)
    print(f"Consolidated JSON saved to: {json_path}")
    print(f"Consolidated CSV saved to:  {csv_path}\n")


if __name__ == "__main__":
    main()
