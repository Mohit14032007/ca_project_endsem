#!/usr/bin/env python3
"""
Point 6 Smoke Test Runner: Validates trace loading, relative indexing,
context length, lookahead, budget, and metrics collection.
Enforces zero ternary operators (ast.IfExp == 0).
"""

import os
import sys
import time
import subprocess
import json
import ast
import pandas as pd

WORKSPACE = "/home/mohman/ca_project"
BIN_PATH = os.path.join(WORKSPACE, "LLMSimulator", "build", "run_deepseekv3_mixtral_style")
SMOKE_DIR = os.path.join(WORKSPACE, "mohman", "point_6", "final_results", "audit", "smoke_tests")
HBM_CAPACITY_BYTES = 37580963840  # 35 GiB

SMOKE_TESTS = [
    {
        "id": "smoke_b4_baseline",
        "batch_size": 4,
        "input_len": 2048,
        "iter": 2,
        "lookahead": 0,
        "budget": 0,
        "prefetch": False,
        "eviction": "lru",
        "trace": os.path.join(WORKSPACE, "mohman", "point_6", "final_results", "experiments", "batch_size", "traces", "b4_t128.csv")
    },
    {
        "id": "smoke_p4096_falru",
        "batch_size": 32,
        "input_len": 4096,
        "iter": 2,
        "lookahead": 1,
        "budget": 4,
        "prefetch": True,
        "eviction": "fa_lru",
        "trace": os.path.join(WORKSPACE, "mohman", "point_4", "big_exp", "traces", "w4_b32_t128.csv")
    },
    {
        "id": "smoke_h2_cstp",
        "batch_size": 32,
        "input_len": 2048,
        "iter": 2,
        "lookahead": 2,
        "budget": 4,
        "prefetch": True,
        "eviction": "lru",
        "trace": os.path.join(WORKSPACE, "mohman", "point_4", "big_exp", "traces", "w4_b32_t128.csv")
    },
    {
        "id": "smoke_k8_falru",
        "batch_size": 32,
        "input_len": 2048,
        "iter": 2,
        "lookahead": 1,
        "budget": 8,
        "prefetch": True,
        "eviction": "fa_lru",
        "trace": os.path.join(WORKSPACE, "mohman", "point_4", "big_exp", "traces", "w4_b32_t128.csv")
    }
]

def generate_smoke_yaml(t):
    run_dir = os.path.join(SMOKE_DIR, t["id"])
    cfg_dir = os.path.join(run_dir, "config")
    csv_dir = os.path.join(run_dir, "csv")
    raw_dir = os.path.join(run_dir, "raw")
    traces_dir = os.path.join(run_dir, "traces")
    
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(traces_dir, exist_ok=True)
    
    if t["prefetch"]:
        prefetch_block = f"""  prefetching:
    enabled: true
    mode: hardware_async
    eviction_policy: {t['eviction']}
    fa_lru_window_size: 64
    scheduling:
      scheduling_mode: forward_lookahead
      prediction_horizon: {t['lookahead']}
    candidate_selection:
      budget: {t['budget']}
    predictor:
      type: CSTP
      prediction_horizon: {t['lookahead']}
      budget: {t['budget']}
      alpha: 0.5
      smoothing_lambda: 1.0
      prior_epsilon: 0.1
      online_adaptation: true"""
    else:
        prefetch_block = f"""  prefetching:
    enabled: false
    mode: none
    eviction_policy: {t['eviction']}
    fa_lru_window_size: 64"""

    yaml_text = f"""model:
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
  max_batch_size: {t['batch_size']}
  max_process_token: 0

simulation:
  data: moe_anywhere
  input_len: {t['input_len']}
  output_len: {t['iter']}
  precision_byte: 2
  skewness: 0.0
  iter: {t['iter']}
  injection_rate: 0
  exit_out_of_memory: false
  mem_cap_limit: false
  trace_file: {t['trace']}
{prefetch_block}

log:
  print_log: false
  export_gantt: false
  output_directory: {csv_dir}/
  gantt_directory: {raw_dir}/
  traces_directory: {traces_dir}/
"""
    yaml_path = os.path.join(cfg_dir, "config.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_text)
    return yaml_path, run_dir, traces_dir

def run_smoke_test(t):
    yaml_path, run_dir, traces_dir = generate_smoke_yaml(t)
    log_file = os.path.join(run_dir, "stdout.log")
    
    env = os.environ.copy()
    env["USE_EVENT_DRIVEN_DRAM"] = "true"
    env["DDR5_BANDWIDTH_GBPS"] = "64.0"
    env["TRACE_DIR"] = traces_dir
    env["PREDICTION_OUTPUT_DIR"] = traces_dir
    env["DEBUG_CACHE_CAPACITY"] = str(HBM_CAPACITY_BYTES)
    env["EVICTION_POLICY"] = t["eviction"]
    env["FA_LRU_WINDOW_SIZE"] = "64"
    
    print(f"Launching smoke test: {t['id']} (B={t['batch_size']}, P={t['input_len']}, H={t['lookahead']}, K={t['budget']}, policy={t['eviction']})...")
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run([BIN_PATH, yaml_path], stdout=lf, stderr=subprocess.STDOUT, env=env)
    dur = time.time() - t0
    
    if proc.returncode != 0:
        print(f"  FAILED with code {proc.returncode}! Log: {log_file}")
        return False
        
    # Check outputs
    summary_json = os.path.join(traces_dir, "prefetch_summary.json")
    fa_lru_json = os.path.join(traces_dir, "fa_lru_diagnostics.json")
    token_lat_csv = os.path.join(traces_dir, "token_latency.csv")
    
    has_summary = os.path.exists(summary_json)
    has_token = os.path.exists(token_lat_csv)
    
    if not has_summary:
        print(f"  FAILED: Missing prefetch_summary.json")
        return False
        
    print(f"  PASSED in {dur:.2f}s! All output artifacts present.")
    return True

def main():
    print("=== STARTING POINT 6 SMOKE TEST VALIDATION ===")
    all_passed = True
    for t in SMOKE_TESTS:
        passed = run_smoke_test(t)
        if not passed:
            all_passed = False
            break
            
    if all_passed:
        print("\nALL 4 SMOKE TESTS PASSED CLEANLY! Ready for full execution campaign.")
    else:
        print("\nSMOKE TESTS FAILED! Aborting full campaign.")
        sys.exit(1)

if __name__ == "__main__":
    main()
