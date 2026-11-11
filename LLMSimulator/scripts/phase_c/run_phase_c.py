"""
Phase C Master Experiment Orchestrator.
Executes the comprehensive multi-dataset benchmark across all 5 target workloads,
10 predictor configurations, and 3 experimental regimes (Frozen, Online, Cold-Start).
"""

import os
import sys
import json
import time
import subprocess
import hashlib
from typing import Any, Dict, List
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from scripts.phase_c.run_online_vs_frozen import evaluate_dataset


def get_git_commit() -> str:
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        commit = subprocess.check_output(cmd, cwd=project_root).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown_commit"


def get_file_checksum(fpath: str) -> str:
    if not os.path.exists(fpath):
        return "missing"
    hasher = hashlib.sha256()
    with open(fpath, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def run_full_phase_c() -> None:
    start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start_time_sec = time.time()

    print("=" * 80)
    print("STARTING PHASE C FULL MULTI-DATASET BENCHMARK")
    print(f"Timestamp: {start_time_iso}")
    print(f"Git Commit: {get_git_commit()}")
    print("=" * 80)

    meta_dir = os.path.join(project_root, "results", "endsem_phase_c", "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    # Workload specifications
    workload_configs = [
        {
            "dataset_id": "deepseek_random",
            "horizons": [1],
            "regimes": ["frozen", "online", "cold_start"],
            "max_units": None,
            "max_tokens": None
        },
        {
            "dataset_id": "duplex_a100",
            "horizons": [1],
            "regimes": ["frozen", "online", "cold_start"],
            "max_units": None,
            "max_tokens": None
        },
        {
            "dataset_id": "correlated_synthetic",
            "horizons": [1],
            "regimes": ["frozen", "online", "cold_start"],
            "max_units": None,
            "max_tokens": None
        },
        {
            "dataset_id": "moe_anywhere",
            "horizons": [1, 2],
            "regimes": ["frozen", "online", "cold_start"],
            "max_units": 14,
            "max_tokens": 20
        },
        {
            "dataset_id": "moe_suite",
            "horizons": [2],
            "regimes": ["frozen", "online", "cold_start"],
            "max_units": 2,
            "max_tokens": 50
        }
    ]

    execution_manifest: Dict[str, Any] = {
        "phase": "Phase C",
        "start_time": start_time_iso,
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "workloads_evaluated": []
    }

    for cfg in workload_configs:
        d_id = cfg["dataset_id"]
        t0 = time.time()
        print(f"\n>>> Running Full Benchmark for Workload: {d_id} <<<")
        res = evaluate_dataset(
            dataset_id=d_id,
            small_scale=False,
            horizons=cfg["horizons"],
            regimes=cfg["regimes"],
            max_test_units=cfg["max_units"],
            max_tokens_per_unit=cfg["max_tokens"]
        )
        duration_s = time.time() - t0
        print(f">>> Completed {d_id} in {duration_s:.2f}s ({res['total_predictions']} predictions recorded) <<<")

        execution_manifest["workloads_evaluated"].append({
            "dataset_id": d_id,
            "horizons": cfg["horizons"],
            "regimes": cfg["regimes"],
            "total_predictions": res["total_predictions"],
            "duration_seconds": round(duration_s, 2)
        })

    end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    total_elapsed = time.time() - start_time_sec
    execution_manifest["end_time"] = end_time_iso
    execution_manifest["total_elapsed_seconds"] = round(total_elapsed, 2)
    execution_manifest["status"] = "SUCCESS"

    # Save run manifest
    manifest_path = os.path.join(meta_dir, "run_manifest.json")
    with open(manifest_path, "w") as fp:
        json.dump(execution_manifest, fp, indent=2)

    print("\n" + "=" * 80)
    print("PHASE C FULL BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Total Elapsed Time: {total_elapsed:.2f}s")
    print(f"Run Manifest saved to: {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_full_phase_c()

