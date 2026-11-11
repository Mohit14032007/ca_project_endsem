#!/usr/bin/env python3
import os
import sys
import time
import shutil
import subprocess
import yaml
import ast
import pandas as pd
import numpy as np

WORKSPACE_DIR = "/home/mohman/ca_project"
SIMULATOR_DIR = os.path.join(WORKSPACE_DIR, "LLMSimulator")
EXECUTABLE = os.path.join(SIMULATOR_DIR, "build", "run_deepseekv3_mixtral_style")
OUTPUT_BASE = os.path.join(WORKSPACE_DIR, "results", "final_comparative_campaign", "sensitivity_budget")
CONFIGS_DIR = os.path.join(OUTPUT_BASE, "configs")
DOCS_DIR = os.path.join(WORKSPACE_DIR, "docs", "final_campaign", "sensitivity_budget")

CALIBRATED_DDR5_BW_GBPS = 22.578
EXPERT_SIZE_BYTES = 88080384
EXPERT_SIZE_MIB = 84.0
EXPERT_SIZE_MB = 88.080384

WORKLOAD_A1 = {
    "name": "MoE_Suite_Task12_A1",
    "model_name": "deepseekV3",
    "num_layers": 78,
    "moe_stages": 75,
    "dense_layers": 3,
    "top_k": 8,
    "iter": 4,
    "input_len": 128,
    "output_len": 128,
    "batch_size": 1,
    "trace_file": os.path.join(SIMULATOR_DIR, "results/endsem_phase_e/final_baseline/traces/ms_task12_t128_l75.csv")
}

def generate_yaml(run_id, predictor_type, horizon, budget, multiplier, run_dir):
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
            },
            "prefetching": {
                "enabled": True,
                "mode": "hardware_async",
                "predictor": {
                    "type": predictor_type,
                    "online_adaptation": True
                },
                "candidate_selection": {
                    "policy": "top_k",
                    "budget": budget
                },
                "scheduling": {
                    "scheduling_mode": "forward_lookahead",
                    "prediction_horizon": horizon,
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
                "eviction_policy": "pabi",
                "pabi": {
                    "quarantine_enabled": True,
                    "speculative_fraction": 0.20,
                    "pinning_enabled": True,
                    "max_pinned_experts": 4,
                    "pin_selection_policy": "top_ranked",
                    "pin_expiration_policy": "stage_deadline"
                }
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

    if predictor_type == "st_moe":
        doc["simulation"]["prefetching"]["predictor"]["confidence_threshold"] = 2
    elif predictor_type == "cstp":
        doc["simulation"]["prefetching"]["predictor"]["alpha"] = 0.5
        doc["simulation"]["prefetching"]["predictor"]["smoothing_lambda"] = 1.0

    os.makedirs(CONFIGS_DIR, exist_ok=True)
    yaml_path = os.path.join(CONFIGS_DIR, f"{run_id}.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(doc, f, sort_keys=False, default_flow_style=False)

    manifest_dir = os.path.join(run_dir, "manifest")
    os.makedirs(manifest_dir, exist_ok=True)
    with open(os.path.join(manifest_dir, "effective_config.yaml"), "w") as f:
        yaml.dump(doc, f, sort_keys=False, default_flow_style=False)

    return yaml_path

def run_experiment(run_id, predictor_type, horizon, budget, multiplier):
    run_dir = os.path.join(OUTPUT_BASE, run_id)
    print(f"\n[STARTING] {run_id} (Predictor={predictor_type}, H={horizon}, B={budget}, M={multiplier})")

    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    os.makedirs(os.path.join(run_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "traces"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "queue"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "manifest"), exist_ok=True)

    yaml_path = generate_yaml(run_id, predictor_type, horizon, budget, multiplier, run_dir)

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
    env["EVICTION_POLICY"] = "pabi"
    env["PREFETCH_ENABLED"] = "1"
    env["PREFETCH_MODE"] = "hardware_async"
    env["PREFETCH_PREDICTOR_TYPE"] = predictor_type
    env["PREFETCH_SCHEDULING_MODE"] = "forward_lookahead"
    env["PREFETCH_PREDICTION_HORIZON"] = str(horizon)
    env["PABI_QUARANTINE_ENABLED"] = "1"
    env["PABI_PINNING_ENABLED"] = "1"
    env["PABI_MAX_PINNED_EXPERTS"] = "4"

    cmd = [EXECUTABLE, yaml_path]

    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=SIMULATOR_DIR, env=env, capture_output=True, text=True)
    t1 = time.perf_counter()
    host_sec = t1 - t0

    with open(os.path.join(run_dir, "logs", "stdout.log"), "w") as f:
        f.write(res.stdout)
    with open(os.path.join(run_dir, "logs", "stderr.log"), "w") as f:
        f.write(res.stderr)

    if res.returncode != 0:
        print(f"[ERROR] Run {run_id} failed with exit code {res.returncode}")
        print(res.stderr[-500:])
        sys.exit(res.returncode)

    print(f"[DONE] {run_id} finished in {host_sec:.2f} host seconds.")
    return run_dir

def main():
    runs_to_execute = [
        # Nominal M=1.0
        ("FWD-S-H2-B1-M1", "st_moe", 2, 1, 1.0),
        ("FWD-S-H2-B2-M1", "st_moe", 2, 2, 1.0),
        ("FWD-C-H2-B1-M1", "cstp", 2, 1, 1.0),
        ("FWD-C-H2-B2-M1", "cstp", 2, 2, 1.0),
        # Decoupled Window M=12.0
        ("FWD-S-H2-B1-M12", "st_moe", 2, 1, 12.0),
        ("FWD-S-H2-B2-M12", "st_moe", 2, 2, 12.0),
        ("FWD-C-H2-B1-M12", "cstp", 2, 1, 12.0),
        ("FWD-C-H2-B2-M12", "cstp", 2, 2, 12.0),
    ]

    for run_id, pred, h, b, m in runs_to_execute:
        run_experiment(run_id, pred, h, b, m)

    print("\n[SUCCESS] All 8 candidate budget sensitivity runs finished successfully.")

if __name__ == "__main__":
    main()
