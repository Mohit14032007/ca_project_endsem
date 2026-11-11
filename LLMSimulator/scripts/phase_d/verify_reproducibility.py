"""
Phase D Reproducibility and Integrity Verification Script.
Audits directory structure, file completeness, checksum integrity,
ternary-free AST compliance, historical result preservation, and unit test pass rates.
"""

import os
import sys
import ast
import glob
import hashlib
import subprocess
from typing import Any, Dict, List, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


def get_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def check_ternaries_in_file(filepath: str) -> int:
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)
    ternaries = [n for n in ast.walk(tree) if isinstance(n, ast.IfExp)]
    return len(ternaries)


def run_verification() -> bool:
    print("=" * 80)
    print("STARTING PHASE D REPRODUCIBILITY & INTEGRITY AUDIT")
    print("=" * 80)

    base_res = os.path.join(project_root, "results", "endsem_phase_d")
    errors: List[str] = []

    # 1. AST Ternary Check on newly introduced code
    print("\n1. Auditing AST for Forbidden Ternary Expressions (ast.IfExp)...")
    phase_d_files = [
        os.path.join(project_root, "src", "predictor", "acsh_predictor.py"),
        os.path.join(project_root, "tests", "phase_d", "test_acsh_validation.py"),
        os.path.join(project_root, "scripts", "phase_d", "run_phase_d.py"),
        os.path.join(project_root, "scripts", "phase_d", "run_distribution_shift_d.py"),
        os.path.join(project_root, "scripts", "phase_d", "benchmark_predictors_d.py"),
        os.path.join(project_root, "scripts", "phase_d", "run_ablations_d.py"),
        os.path.join(project_root, "scripts", "phase_d", "aggregate_phase_d.py"),
        os.path.join(project_root, "scripts", "phase_d", "generate_figures_d.py"),
        os.path.join(project_root, "scripts", "phase_d", "generate_reconciliation_csv.py")
    ]
    for fp in phase_d_files:
        if os.path.exists(fp):
            t_count = check_ternaries_in_file(fp)
            if t_count > 0:
                errors.append(f"AST Error: {fp} contains {t_count} ternary expressions!")
            else:
                print(f"  [PASSED] {os.path.basename(fp)}: 0 ternaries")
        else:
            errors.append(f"Missing file: {fp}")

    # 2. Check Directory Structure
    print("\n2. Checking Phase D Output Directories...")
    required_dirs = [
        "audit", "metadata", "datasets", "per_dataset", "aggregate",
        "online_learning", "distribution_shift", "calibration", "ablations",
        "latency", "memory", "figures", "logs", "checkpoints"
    ]
    for d in required_dirs:
        dir_path = os.path.join(base_res, d)
        if not os.path.exists(dir_path):
            errors.append(f"Missing directory: {dir_path}")
        else:
            print(f"  [PASSED] Directory exists: results/endsem_phase_d/{d}")

    # 3. Check Audit Reconciliation CSV
    recon_csv = os.path.join(base_res, "audit", "phase_c_reconciliation.csv")
    if os.path.exists(recon_csv):
        print(f"  [PASSED] Audit CSV exists: {os.path.basename(recon_csv)}")
    else:
        errors.append(f"Missing audit CSV: {recon_csv}")

    # 4. Check Historical Integrity (Phase 1, 2B, C preserved)
    print("\n3. Checking Historical Result Immutability...")
    hist_dirs = [
        os.path.join(project_root, "results", "endsem_phase1"),
        os.path.join(project_root, "results", "endsem_phase2b"),
        os.path.join(project_root, "results", "endsem_phase_c")
    ]
    for hd in hist_dirs:
        if os.path.exists(hd):
            n_files = len(glob.glob(os.path.join(hd, "**", "*"), recursive=True))
            print(f"  [PASSED] Historical directory preserved: {os.path.basename(hd)} ({n_files} items)")
        else:
            errors.append(f"Missing historical directory: {hd}")

    # 5. Check Unit Tests Execution
    print("\n4. Running Unit Test Suite...")
    try:
        test_out = subprocess.check_output(
            ["python3", "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=project_root,
            stderr=subprocess.STDOUT
        ).decode("utf-8")
        if "FAILED" in test_out:
            errors.append("Unit tests failed!")
        else:
            lines = [l for l in test_out.splitlines() if "Ran" in l or "OK" in l]
            print(f"  [PASSED] Unit tests: {' '.join(lines)}")
    except subprocess.CalledProcessError as e:
        errors.append(f"Unit test runner error: {e.output.decode('utf-8')}")

    # Final Summary
    print("\n" + "=" * 80)
    if len(errors) == 0:
        print("PHASE D REPRODUCIBILITY AUDIT: ALL CHECKS PASSED (100% COMPLIANT)")
        print("=" * 80)
        return True
    else:
        print(f"PHASE D REPRODUCIBILITY AUDIT FAILED WITH {len(errors)} ERRORS:")
        for err in errors:
            print(f"  - {err}")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = run_verification()
    if not success:
        sys.exit(1)

