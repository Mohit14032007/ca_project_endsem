import os
import csv
import glob
import yaml

BASE_DIR = "/home/mohman/ca_project/mohman/point_3"
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
MANIFEST_PATH = os.path.join(BASE_DIR, "experiment_manifest.csv")

def extract_meta(cfg):
    p_cfg = cfg.get("simulation", {}).get("prefetching", {})
    p_enabled = p_cfg.get("enabled", False)
    predictor = "none"
    horizon = 0
    budget = 0
    if p_enabled:
        predictor = p_cfg.get("predictor", {}).get("type", "none")
        horizon = p_cfg.get("scheduling", {}).get("prediction_horizon", 1)
        budget = p_cfg.get("candidate_selection", {}).get("budget", 4)
    return predictor, horizon, budget

def main():
    rows = []
    
    # Core experiments
    core_configs = glob.glob(os.path.join(CONFIGS_DIR, "core_*.yaml"))
    for cp in sorted(core_configs):
        with open(cp, 'r') as f:
            cfg = yaml.safe_load(f)
        meta_exp = os.path.splitext(os.path.basename(cp))[0]
        pred, hor, bud = extract_meta(cfg)
        rows.append({
            "experiment_id": meta_exp,
            "category": "core",
            "config_path": cp,
            "dataset": cfg["simulation"]["data"],
            "batch_size": cfg["serving"]["max_batch_size"],
            "predictor": pred,
            "horizon": hor,
            "budget": bud,
            "cache_capacity_bytes": cfg["system"]["optimization"]["expert_cache_capacity_bytes"],
            "iterations": cfg["simulation"]["iter"],
            "trace_file": cfg["simulation"]["trace_file"],
            "status": "COMPLETED"
        })

    # Smoke experiments
    smoke_configs = glob.glob(os.path.join(CONFIGS_DIR, "smoke_*.yaml"))
    for cp in sorted(smoke_configs):
        with open(cp, 'r') as f:
            cfg = yaml.safe_load(f)
        meta_exp = os.path.splitext(os.path.basename(cp))[0]
        pred, hor, bud = extract_meta(cfg)
        rows.append({
            "experiment_id": meta_exp,
            "category": "smoke",
            "config_path": cp,
            "dataset": cfg["simulation"]["data"],
            "batch_size": cfg["serving"]["max_batch_size"],
            "predictor": pred,
            "horizon": hor,
            "budget": bud,
            "cache_capacity_bytes": cfg["system"]["optimization"]["expert_cache_capacity_bytes"],
            "iterations": cfg["simulation"]["iter"],
            "trace_file": cfg["simulation"]["trace_file"],
            "status": "COMPLETED"
        })

    # Sensitivity experiments
    sens_configs = glob.glob(os.path.join(CONFIGS_DIR, "sensitivity", "*", "*.yaml"))
    for cp in sorted(sens_configs):
        with open(cp, 'r') as f:
            cfg = yaml.safe_load(f)
        subcat = os.path.basename(os.path.dirname(cp))
        meta_exp = os.path.splitext(os.path.basename(cp))[0]
        pred, hor, bud = extract_meta(cfg)
        rows.append({
            "experiment_id": f"{subcat}_{meta_exp}",
            "category": subcat,
            "config_path": cp,
            "dataset": cfg["simulation"]["data"],
            "batch_size": cfg["serving"]["max_batch_size"],
            "predictor": pred,
            "horizon": hor,
            "budget": bud,
            "cache_capacity_bytes": cfg["system"]["optimization"]["expert_cache_capacity_bytes"],
            "iterations": cfg["simulation"]["iter"],
            "trace_file": cfg["simulation"]["trace_file"],
            "status": "COMPLETED"
        })

    if len(rows) > 0:
        with open(MANIFEST_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[SUCCESS] Wrote {len(rows)} experiment manifest rows to {MANIFEST_PATH}")

if __name__ == "__main__":
    main()
