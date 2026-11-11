import os
import sys
import yaml
import subprocess
import json
import csv

BASE_SIMULATOR = "/home/mohman/ca_project/LLMSimulator/build/run_deepseekv3_mixtral_style"
BASE_DIR = "/home/mohman/ca_project/mohman/point_3"

def make_yaml_config(
    model_name="deepseekV3",
    num_layers=5,
    max_batch_size=1,
    data_name="moe_anywhere",
    input_len=128,
    output_len=4,
    iter_count=4,
    trace_file="/home/mohman/ca_project/mohman/point_3/traces/moe_anywhere_b1_t128.csv",
    cache_capacity_bytes=85899345920,
    prefetch_enabled=False,
    predictor_type="none",
    horizon=1,
    budget=4,
    out_dir="/home/mohman/ca_project/mohman/point_3/results/temp"
):
    prefetch_mode = "none"
    if prefetch_enabled:
        prefetch_mode = "hardware_async"

    cfg = {
        "model": {
            "model_name": model_name,
            "num_layers": num_layers
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
                "offchip_dram_cfg_path": "/home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/configs/dram_config_DDR5_retry.yaml",
                "hbm_dram_cfg_path": "/home/mohman/ca_project/LLMSimulator/build/dram_config_HBM3_80GB.yaml",
                "expert_cache_capacity_bytes": cache_capacity_bytes,
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
            "max_batch_size": max_batch_size,
            "max_process_token": 0
        },
        "simulation": {
            "data": data_name,
            "input_len": input_len,
            "output_len": output_len,
            "precision_byte": 2,
            "skewness": 0.0,
            "iter": iter_count,
            "injection_rate": 0,
            "exit_out_of_memory": False,
            "mem_cap_limit": False,
            "trace_file": trace_file,
            "prefetching": {
                "enabled": prefetch_enabled,
                "mode": prefetch_mode,
                "scheduling": {
                    "scheduling_mode": "forward_lookahead",
                    "prediction_horizon": horizon
                },
                "candidate_selection": {
                    "budget": budget
                },
                "predictor": {
                    "type": predictor_type,
                    "prediction_horizon": horizon,
                    "budget": budget
                }
            }
        },
        "log": {
            "print_log": False,
            "export_gantt": False,
            "output_directory": os.path.join(out_dir, "csv/"),
            "gantt_directory": os.path.join(out_dir, "raw/"),
            "traces_directory": os.path.join(out_dir, "traces/")
        }
    }
    return cfg

def run_experiment(config_path):
    cmd = [BASE_SIMULATOR, config_path]
    env = os.environ.copy()
    env["USE_EVENT_DRIVEN_DRAM"] = "true"
    env["DDR5_BANDWIDTH_GBPS"] = "64.0"
    
    print("[RUNNING] " + config_path)
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print("[FAIL] " + config_path + " returned exit code " + str(res.returncode))
        print("STDERR: " + res.stderr[-500:])
        return False
    print("[SUCCESS] " + config_path)
    return True

def generate_and_run_all():
    experiments = []

    # 1. Budget Sensitivity (Budgets 1, 2, 4, 8)
    for b in [1, 2, 4, 8]:
        cfg_name = "budget_" + str(b)
        cfg_path = os.path.join(BASE_DIR, "configs/sensitivity/budget", cfg_name + ".yaml")
        out_dir = os.path.join(BASE_DIR, "results/sensitivity/budget", cfg_name)
        cfg_data = make_yaml_config(
            data_name="moe_anywhere",
            iter_count=4,
            trace_file=os.path.join(BASE_DIR, "traces/moe_anywhere_b1_t128.csv"),
            prefetch_enabled=True,
            predictor_type="CSTP",
            horizon=1,
            budget=b,
            out_dir=out_dir
        )
        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
        with open(cfg_path, 'w') as f:
            yaml.dump(cfg_data, f, default_flow_style=False)
        experiments.append(("budget", cfg_name, cfg_path, out_dir))

    # 2. Batch Size Sensitivity (Batches 1, 2, 4)
    for bs in [1, 2, 4]:
        for mode in ["baseline", "cstp"]:
            cfg_name = "batch" + str(bs) + "_" + mode
            cfg_path = os.path.join(BASE_DIR, "configs/sensitivity/batch", cfg_name + ".yaml")
            out_dir = os.path.join(BASE_DIR, "results/sensitivity/batch", cfg_name)
            trace_file = os.path.join(BASE_DIR, "traces", "moe_anywhere_b" + str(bs) + "_t128.csv")
            
            p_enabled = False
            p_type = "none"
            if mode == "cstp":
                p_enabled = True
                p_type = "CSTP"

            cfg_data = make_yaml_config(
                max_batch_size=bs,
                data_name="moe_anywhere",
                iter_count=4,
                trace_file=trace_file,
                prefetch_enabled=p_enabled,
                predictor_type=p_type,
                horizon=1,
                budget=4,
                out_dir=out_dir
            )
            os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
            with open(cfg_path, 'w') as f:
                yaml.dump(cfg_data, f, default_flow_style=False)
            experiments.append(("batch", cfg_name, cfg_path, out_dir))

    # 3. Cross-Dataset Validation (MoE-Anywhere, MoE-Suite 12, MoE-Suite 13)
    datasets = [
        ("moe_anywhere", os.path.join(BASE_DIR, "traces/moe_anywhere_b1_t128.csv")),
        ("moe_suite_12", os.path.join(BASE_DIR, "traces/moe_suite_task12_b1_t128.csv")),
        ("moe_suite_13", os.path.join(BASE_DIR, "traces/moe_suite_task13_b1_t128.csv"))
    ]
    for d_name, d_trace in datasets:
        for p_name in ["baseline", "stmoe", "cstp"]:
            cfg_name = d_name + "_" + p_name
            cfg_path = os.path.join(BASE_DIR, "configs/sensitivity/cross_dataset", cfg_name + ".yaml")
            out_dir = os.path.join(BASE_DIR, "results/sensitivity/cross_dataset", cfg_name)
            
            p_enabled = False
            actual_predictor = "none"
            if p_name == "stmoe":
                p_enabled = True
                actual_predictor = "ST_MOE"
            elif p_name == "cstp":
                p_enabled = True
                actual_predictor = "CSTP"

            cfg_data = make_yaml_config(
                data_name=d_name,
                iter_count=4,
                trace_file=d_trace,
                prefetch_enabled=p_enabled,
                predictor_type=actual_predictor,
                horizon=1,
                budget=4,
                out_dir=out_dir
            )
            os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
            with open(cfg_path, 'w') as f:
                yaml.dump(cfg_data, f, default_flow_style=False)
            experiments.append(("cross_dataset", cfg_name, cfg_path, out_dir))

    # 4. Decode Length Sensitivity (Iter 2, 4, 8, 16)
    for it in [2, 4, 8, 16]:
        for mode in ["baseline", "cstp"]:
            cfg_name = "iter" + str(it) + "_" + mode
            cfg_path = os.path.join(BASE_DIR, "configs/sensitivity/decode_length", cfg_name + ".yaml")
            out_dir = os.path.join(BASE_DIR, "results/sensitivity/decode_length", cfg_name)
            
            p_enabled = False
            actual_predictor = "none"
            if mode == "cstp":
                p_enabled = True
                actual_predictor = "CSTP"

            cfg_data = make_yaml_config(
                data_name="moe_anywhere",
                iter_count=it,
                trace_file=os.path.join(BASE_DIR, "traces/moe_anywhere_b1_t128.csv"),
                prefetch_enabled=p_enabled,
                predictor_type=actual_predictor,
                horizon=1,
                budget=4,
                out_dir=out_dir
            )
            os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
            with open(cfg_path, 'w') as f:
                yaml.dump(cfg_data, f, default_flow_style=False)
            experiments.append(("decode_length", cfg_name, cfg_path, out_dir))

    # 5. Constrained Cache Capacity (20 experts = 1.76 GB, 30 experts = 2.64 GB)
    for exp_cap in [20, 30]:
        cap_bytes = exp_cap * 88080384
        for mode in ["baseline", "cstp"]:
            cfg_name = "cap" + str(exp_cap) + "_" + mode
            cfg_path = os.path.join(BASE_DIR, "configs/sensitivity/cache_size", cfg_name + ".yaml")
            out_dir = os.path.join(BASE_DIR, "results/sensitivity/cache_size", cfg_name)
            
            p_enabled = False
            actual_predictor = "none"
            if mode == "cstp":
                p_enabled = True
                actual_predictor = "CSTP"

            cfg_data = make_yaml_config(
                data_name="moe_anywhere",
                iter_count=4,
                trace_file=os.path.join(BASE_DIR, "traces/moe_anywhere_b1_t128.csv"),
                cache_capacity_bytes=cap_bytes,
                prefetch_enabled=p_enabled,
                predictor_type=actual_predictor,
                horizon=1,
                budget=4,
                out_dir=out_dir
            )
            os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
            with open(cfg_path, 'w') as f:
                yaml.dump(cfg_data, f, default_flow_style=False)
            experiments.append(("cache_size", cfg_name, cfg_path, out_dir))

    print("Total sensitivity experiments planned: " + str(len(experiments)))
    
    # Run all experiments
    success_count = 0
    for category, name, cfg_path, out_dir in experiments:
        ok = run_experiment(cfg_path)
        if ok:
            success_count = success_count + 1

    print("Successfully completed " + str(success_count) + " / " + str(len(experiments)) + " experiments.")
    return experiments

def aggregate_sensitivity_metrics(experiments):
    rows = []
    for category, name, cfg_path, out_dir in experiments:
        sum_file = os.path.join(out_dir, "traces/experiment_summary.json")
        if not os.path.exists(sum_file):
            print("[WARN] Summary missing for " + name)
            continue
        with open(sum_file, 'r') as f:
            data = json.load(f)

        meta = data.get("experiment_metadata", {})
        cov = data.get("prefetch_coverage", {})
        rates = data.get("timely_and_late_rates", {})
        unused = data.get("unused_prefetches", {})
        hits = data.get("hbm_hit_rates", {})
        mem = data.get("memory_traffic", {})
        stall = data.get("exposed_stall_and_execution", {})
        exp_lat = data.get("expert_load_latency", {})
        tok_lat = data.get("token_latency", {})

        row = {
            "category": category,
            "experiment": name,
            "dataset": meta.get("dataset", ""),
            "batch_size": meta.get("batch_size", 1),
            "predictor": meta.get("predictor_type", "none"),
            "horizon": meta.get("horizon", 0),
            "budget": meta.get("budget", 0),
            "total_demands_proj": cov.get("total_actual_demands", 0),
            "unique_demands": cov.get("unique_actual_demands", 0),
            "unique_coverage": cov.get("unique_overall_coverage", 0.0),
            "unique_timely_hits": rates.get("unique_timely_resident_hits", 0),
            "unique_late_attachments": rates.get("unique_late_inflight_attachments", 0),
            "unique_resident_hits": hits.get("unique_resident_hits", 0),
            "unique_fresh_misses": hits.get("unique_fresh_demand_misses", 0),
            "unique_hbm_hit_rate": hits.get("unique_overall_hbm_hit_rate", 0.0),
            "overall_hbm_hit_rate_proj": hits.get("overall_hbm_hit_rate", 0.0),
            "total_ddr5_bytes": mem.get("total_ddr5_bytes_transferred", 0),
            "speculative_bytes": mem.get("speculative_bytes_transferred", 0),
            "demand_bytes": mem.get("demand_bytes_transferred", 0),
            "wasted_speculative_bytes": unused.get("total_wasted_speculative_bytes", 0),
            "aggregate_demand_wait_ns": stall.get("aggregate_demand_wait_ns", 0.0),
            "exposed_gpu_stall_ns": stall.get("exposed_gpu_stall_ns", 0.0),
            "stall_hidden_by_prefetch_ns": stall.get("stall_hidden_by_prefetch_ns", 0.0),
            "wait_per_token_ns": stall.get("wait_per_token_ns", 0.0),
            "wait_per_unique_demand_ns": stall.get("wait_per_unique_demand_ns", 0.0),
            "token_count": tok_lat.get("count", 0),
            "mean_token_latency_ns": tok_lat.get("mean_ns", 0.0),
            "p50_token_latency_ns": tok_lat.get("p50_ns", 0.0),
            "tokens_per_second": tok_lat.get("simulated_tokens_per_second", 0.0)
        }
        rows.append(row)

    out_csv = os.path.join(BASE_DIR, "reports/sensitivity_metrics.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if len(rows) > 0:
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("[SUCCESS] Sensitivity metrics consolidated to: " + out_csv)

if __name__ == "__main__":
    exps = generate_and_run_all()
    aggregate_sensitivity_metrics(exps)

