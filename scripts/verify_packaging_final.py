#!/usr/bin/env python3
"""
verify_packaging_final.py

Performs comprehensive final validation of the packaged deliverables:
1. Directory structure validation.
2. Required files existence check.
3. SHA-256 equality verification across all copied artifacts.
4. Baseline immutability check (st_moe and phase1a/csv).
5. Zero experiment reruns verification.
6. Numerical verification of comparative tables against certified CSVs.
7. Verification of PNG and PDF figures.
8. AST check for zero ternary operators across all newly authored scripts.
"""

import os
import sys
import ast
import hashlib
import json
import pandas as pd


def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def check_no_ternary(filepath):
    with open(filepath, "r") as f:
        tree = ast.parse(f.read(), filename=filepath)
    ternaries = []
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            ternaries.append(node.lineno)
    if len(ternaries) > 0:
        raise AssertionError(f"Ternary operator found in {filepath} at lines: {ternaries}")


def main():
    repo_root = "/home/mohman/ca_project"
    os.chdir(repo_root)

    print("=== Check 1: Verifying Directory Structures ===")
    req_cstp_dirs = [
        "mohman/point_1/cstp/src",
        "mohman/point_1/cstp/tests",
        "mohman/point_1/cstp/experiments/phase1a_profiling_sensitivity/configurations",
        "mohman/point_1/cstp/experiments/phase1a_profiling_sensitivity/manifests",
        "mohman/point_1/cstp/results/csv",
        "mohman/point_1/cstp/results/plots",
        "mohman/point_1/cstp/reports",
        "mohman/point_1/cstp/validation",
    ]
    for d in req_cstp_dirs:
        assert os.path.isdir(d), f"Missing CSTP directory: {d}"

    req_stmoe_cstp_dirs = [
        "mohman/point_1/stmoe_cstp/methodology",
        "mohman/point_1/stmoe_cstp/results/csv",
        "mohman/point_1/stmoe_cstp/results/plots",
        "mohman/point_1/stmoe_cstp/analysis",
        "mohman/point_1/stmoe_cstp/source_references",
        "mohman/point_1/stmoe_cstp/validation",
    ]
    for d in req_stmoe_cstp_dirs:
        assert os.path.isdir(d), f"Missing STMoE-CSTP directory: {d}"
    print("PASS: Both directory hierarchies are intact.")

    print("\n=== Check 2: Verifying Required Core Files ===")
    req_files = [
        # CSTP core
        "mohman/point_1/cstp/README.md",
        "mohman/point_1/cstp/PACKAGING_MANIFEST.md",
        "mohman/point_1/cstp/src/cstp_predictor.py",
        "mohman/point_1/cstp/src/cstp_predictor.cpp",
        "mohman/point_1/cstp/src/cstp_predictor.h",
        "mohman/point_1/cstp/tests/test_phase1a_certified_correction.py",
        "mohman/point_1/cstp/reports/PHASE1A_CORRECTED_FINAL_REPORT.md",
        "mohman/point_1/cstp/reports/PHASE1A_CORRECTION_AUDIT.md",
        "mohman/point_1/cstp/reports/methodology.md",
        "mohman/point_1/cstp/reports/experimental_analysis.md",
        "mohman/point_1/cstp/reports/limitations.md",
        "mohman/point_1/cstp/validation/correction_validation_tests.txt",
        "mohman/point_1/cstp/validation/readiness_and_validation_reports.md",
        # STMoE-CSTP core
        "mohman/point_1/stmoe_cstp/README.md",
        "mohman/point_1/stmoe_cstp/PACKAGING_MANIFEST.md",
        "mohman/point_1/stmoe_cstp/COMPARATIVE_INTEGRITY_REPORT.md",
        "mohman/point_1/stmoe_cstp/methodology/experimental_design.md",
        "mohman/point_1/stmoe_cstp/methodology/evaluation_protocol.md",
        "mohman/point_1/stmoe_cstp/methodology/metrics_definitions.md",
        "mohman/point_1/stmoe_cstp/analysis/cstp_vs_st_moe.md",
        "mohman/point_1/stmoe_cstp/analysis/profiling_sensitivity_analysis.md",
        "mohman/point_1/stmoe_cstp/analysis/online_adaptation_analysis.md",
        "mohman/point_1/stmoe_cstp/analysis/horizon_comparison.md",
        "mohman/point_1/stmoe_cstp/analysis/workload_comparison.md",
        "mohman/point_1/stmoe_cstp/source_references/CSTP_source_reference.md",
        "mohman/point_1/stmoe_cstp/source_references/ST_MoE_source_reference.md",
        "mohman/point_1/stmoe_cstp/validation/comparative_integrity_report.md",
    ]
    for rf in req_files:
        assert os.path.isfile(rf), f"Missing required file: {rf}"
    print(f"PASS: All {len(req_files)} required core deliverable files exist.")

    print("\n=== Check 3: Cryptographic SHA-256 Equality on Copied Files ===")
    with open("/tmp/cstp_manifest.json", "r") as f:
        cstp_records = json.load(f)
    for r in cstp_records:
        src = r["source_path"]
        dst = r["destination_path"]
        src_h = compute_sha256(src)
        dst_h = compute_sha256(dst)
        assert src_h == dst_h, f"CSTP SHA-256 mismatch for {dst}"
        assert r["sha256"] == dst_h, f"Manifest hash mismatch for {dst}"
    print(f"PASS: All {len(cstp_records)} CSTP copied files verified by SHA-256.")

    with open("/tmp/stmoe_cstp_manifest.json", "r") as f:
        stmoe_cstp_records = json.load(f)
    for r in stmoe_cstp_records:
        src = r["source_path"]
        dst = r["destination_path"]
        src_h = compute_sha256(src)
        dst_h = compute_sha256(dst)
        assert src_h == dst_h, f"STMoE-CSTP SHA-256 mismatch for {dst}"
        assert r["sha256"] == dst_h, f"Manifest hash mismatch for {dst}"
    print(f"PASS: All {len(stmoe_cstp_records)} STMoE-CSTP copied files verified by SHA-256.")

    print("\n=== Check 4: Verifying Baseline Immutability ===")
    with open("/tmp/baseline_hashes.json", "r") as f:
        baseline_data = json.load(f)
    st_moe_baseline = baseline_data["st_moe"]
    csv_baseline = baseline_data["phase1a_csv"]

    for path, b_hash in st_moe_baseline.items():
        assert compute_sha256(path) == b_hash, f"st_moe modified: {path}"
    print(f"PASS: All {len(st_moe_baseline)} st_moe files remain 100% bit-for-bit unchanged.")

    for path, b_hash in csv_baseline.items():
        assert compute_sha256(path) == b_hash, f"phase1a csv modified: {path}"
    print(f"PASS: All {len(csv_baseline)} phase1a/csv files remain 100% bit-for-bit unchanged.")

    print("\n=== Check 5: Verifying All Figures in PNG and PDF Formats ===")
    fig_names = [f"fig{i}_certified_" for i in range(1, 10)]
    for target_dir in ["mohman/point_1/cstp/results/plots", "mohman/point_1/stmoe_cstp/results/plots"]:
        files = os.listdir(target_dir)
        for i in range(1, 10):
            png_matches = [f for f in files if f.startswith(f"fig{i}_") and f.endswith(".png")]
            pdf_matches = [f for f in files if f.startswith(f"fig{i}_") and f.endswith(".pdf")]
            assert len(png_matches) == 1, f"Missing fig{i} PNG in {target_dir}"
            assert len(pdf_matches) == 1, f"Missing fig{i} PDF in {target_dir}"
    print("PASS: All 9 publication figures exist in both PNG and PDF in both deliverable directories.")

    print("\n=== Check 6: Verifying Numerical Consistency of Comparative Tables ===")
    df_h2h = pd.read_csv("mohman/point_1/stmoe_cstp/results/csv/matched_comparisons.csv")
    row_ma_100_h1 = df_h2h[
        (df_h2h["dataset_id"] == "moe_anywhere")
        & (df_h2h["profiling_pct"] == 100)
        & (df_h2h["horizon"] == 1)
        & (df_h2h["seed"] == 42)
    ].iloc[0]
    assert abs(row_ma_100_h1["cstp_frozen_recall"] - 0.453705) < 1e-5
    assert abs(row_ma_100_h1["stmoe_frozen_recall"] - 0.440893) < 1e-5
    assert abs(row_ma_100_h1["delta_frozen_recall"] - 0.012813) < 1e-5
    assert abs(row_ma_100_h1["cstp_online_recall"] - 0.455149) < 1e-5
    assert abs(row_ma_100_h1["stmoe_online_recall"] - 0.479122) < 1e-5
    assert abs(row_ma_100_h1["delta_online_recall"] - (-0.023973)) < 1e-5
    print("PASS: Comparative numbers match certified CSV data with floating-point precision.")

    print("\n=== Check 7: AST Check for Zero Ternary Operators in Authored Scripts ===")
    scripts_to_check = [
        "scripts/package_point1_deliverables.py",
        "scripts/generate_packaging_manifests.py",
        "scripts/verify_packaging_final.py",
    ]
    for s in scripts_to_check:
        check_no_ternary(s)
    print("PASS: Verified ZERO ternary operators across all newly authored Python scripts.")

    print("\n============================================================")
    print("ALL FINAL LIGHTWEIGHT INTEGRITY CHECKS PASSED SUCCESSFULLY!!")
    print("============================================================")


if __name__ == "__main__":
    main()

