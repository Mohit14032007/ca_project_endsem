#!/usr/bin/env python3
"""
Phase 4D.5B: Bounded 4-Token Smoke Test Matrix Runner
Runs all 9 configurations across ST-MoE, CSTP, LRU, and PABI.
Zero-ternary operator compliance enforced.
"""

import os
import sys
import subprocess
import shutil
import time
import json
import ast
import yaml

REPO_ROOT = "/home/mohman/ca_project/LLMSimulator"
CONFIGS_DIR = os.path.join(REPO_ROOT, "configs", "phase_4d5b")
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "phase_4d5b")
EXECUTABLE = os.path.join(REPO_ROOT, "build", "run_deepseekv3_mixtral_style")
TRACE_FILE = "results/endsem_phase_e/final_baseline/traces/ms_task12_t128_l2.csv"

# Configuration metadata matrix
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
            "num_layers": 5
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
            "iter": 4,
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

    # Prefetch section
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

    yaml_path = os.path.join(CONFIGS_DIR, f"{cid}.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(doc, f, sort_keys=False, default_flow_style=False)
    return yaml_path


def run_configuration(cfg, yaml_path):
    cid = cfg["id"]
    out_dir = os.path.join(RESULTS_DIR, cid)
    
    # Clean output dir
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.join(out_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "traces"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "queue"), exist_ok=True)

    # Prepare isolated environment
    env = os.environ.copy()
    
    # Clear any previous overrides
    env.pop("DEBUG_CACHE_CAPACITY", None)
    
    # Explicitly set all parameters
    env["TRACE_DIR"] = os.path.join(out_dir, "traces")
    env["PREDICTION_OUTPUT_DIR"] = out_dir
    env["PREFETCH_QUEUE_OUTPUT_DIR"] = os.path.join(out_dir, "queue")
    env["USE_EVENT_DRIVEN_DRAM"] = "1"
    env["BYPASS_OFFCHIP_EXECUTION_CACHE"] = "0"
    env["DDR5_BANDWIDTH_GBPS"] = "64.0"

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

    print(f"\n======================================================================")
    print(f"RUNNING CONFIGURATION: {cfg['name']} ({cid})")
    print(f"Mode: {cfg['mode']}, Predictor: {cfg['predictor']}, Eviction: {cfg['eviction']}, "
          f"Quarantine: {cfg['quarantine']}, Pinning: {cfg['pinning']}")
    print(f"Output directory: {out_dir}")
    print(f"======================================================================")

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
        print(f"[ERROR] Run failed with returncode {res.returncode}")
        print(f"STDERR:\n{res.stderr[-500:]}")
        return {
            "success": False,
            "returncode": res.returncode,
            "host_runtime_sec": host_runtime_sec
        }

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

    # Print summary highlights
    print(f"  -> SUCCESS (Host runtime: {host_runtime_sec:.3f} s)")
    print(f"     Candidates evaluated: {status_dict.get('CANDIDATES_EVALUATED', '0')}")
    print(f"     Candidates dispatched: {status_dict.get('CANDIDATES_DISPATCHED', '0')}")
    print(f"     Prefetches submitted: {status_dict.get('PREFETCH_TRANSFERS_SUBMITTED', '0')}")
    print(f"     Prefetches completed: {status_dict.get('PREFETCH_TRANSFERS_COMPLETED', '0')}")
    print(f"     Prefetches inserted HBM: {status_dict.get('PREFETCHES_INSERTED_HBM', '0')}")
    print(f"     Prefetch useful hits: {status_dict.get('PREFETCH_USEFUL_HITS', '0')}")
    print(f"     Demand inflight attach: {status_dict.get('DEMAND_INFLIGHT_ATTACHMENTS', '0')}")
    print(f"     Speculative insertions: {status_dict.get('SPECULATIVE_INSERTIONS', '0')}")
    print(f"     Pin intents created: {status_dict.get('PIN_INTENTS_CREATED', '0')}")
    print(f"     Resident pinned: {status_dict.get('RESIDENT_EXPERTS_PINNED', '0')}")
    print(f"     Pin expirations: {status_dict.get('PIN_EXPIRATIONS', '0')}")

    return status_dict


def main():
    check_zero_ternary(__file__)
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary_results = []
    for cfg in CONFIGURATIONS:
        yaml_path = generate_yaml(cfg)
        status = run_configuration(cfg, yaml_path)
        cfg_summary = dict(cfg)
        cfg_summary.update(status)
        summary_results.append(cfg_summary)

    # Write summary JSON
    summary_path = os.path.join(RESULTS_DIR, "smoke_matrix_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_results, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 4D.5B 4-TOKEN SMOKE TEST MATRIX RESULTS SUMMARY")
    print("=" * 80)
    header = (
        f"{'#':<3} | {'Configuration':<26} | {'Dispatched':<10} | {'Submitted':<9} | "
        f"{'Complete':<8} | {'HBM Ins':<7} | {'Hits':<5} | {'Pins':<5} | {'Exp':<5} | {'Sec':<6}"
    )
    print(header)
    print("-" * len(header))

    idx = 1
    for r in summary_results:
        dispatched = r.get("CANDIDATES_DISPATCHED", "0")
        submitted = r.get("PREFETCH_TRANSFERS_SUBMITTED", "0")
        completed = r.get("PREFETCH_TRANSFERS_COMPLETED", "0")
        inserted = r.get("PREFETCHES_INSERTED_HBM", "0")
        hits = r.get("PREFETCH_USEFUL_HITS", "0")
        pins = r.get("PIN_INTENTS_CREATED", "0")
        exp = r.get("PIN_EXPIRATIONS", "0")
        sec = f"{r.get('host_runtime_sec', 0.0):.2f}"
        name = r["name"]
        print(f"{idx:<3} | {name:<26} | {dispatched:<10} | {submitted:<9} | "
              f"{completed:<8} | {inserted:<7} | {hits:<5} | {pins:<5} | {exp:<5} | {sec:<6}")
        idx += 1

    print("=" * 80)
    print(f"Summary JSON saved to: {summary_path}\n")


if __name__ == "__main__":
    main()

