#!/usr/bin/env python3
"""
package_point1_deliverables.py

Safely packages existing Phase 1A certified artifacts and source files
into mohman/point_1/cstp/ and mohman/point_1/stmoe_cstp/.

Enforces:
1. Zero experiment execution.
2. Zero modification of existing st_moe/ directory.
3. Zero modification of existing phase1a_cstp_profiling/csv/ directory.
4. Cryptographic SHA-256 verification of all copied files.
5. Zero ternary operators.
"""

import os
import sys
import shutil
import hashlib
import json


def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def get_dir_hashes(directory):
    records = {}
    if not os.path.exists(directory):
        return records
    for root, dirs, files in os.walk(directory):
        for f in files:
            path = os.path.join(root, f)
            records[path] = compute_sha256(path)
    return records


def verify_untouched(baseline_hashes, current_hashes, dir_name):
    print("Verifying directory integrity for: " + dir_name)
    if len(baseline_hashes) != len(current_hashes):
        raise RuntimeError(
            "File count mismatch in "
            + dir_name
            + ": baseline="
            + str(len(baseline_hashes))
            + ", current="
            + str(len(current_hashes))
        )
    for path, b_hash in baseline_hashes.items():
        if path not in current_hashes:
            raise RuntimeError("Missing file in " + dir_name + ": " + path)
        c_hash = current_hashes[path]
        if b_hash != c_hash:
            raise RuntimeError("Hash mismatch for " + path + ": " + b_hash + " != " + c_hash)
    print("PASS: Directory " + dir_name + " is completely untouched.")


def main():
    repo_root = "/home/mohman/ca_project"
    os.chdir(repo_root)

    print("=== Step 1: Loading Baseline Hashes ===")
    with open("/tmp/baseline_hashes.json", "r") as f:
        baseline_data = json.load(f)
    st_moe_baseline = baseline_data["st_moe"]
    csv_baseline = baseline_data["phase1a_csv"]

    verify_untouched(
        st_moe_baseline,
        get_dir_hashes("mohman/point_1/st_moe"),
        "mohman/point_1/st_moe"
    )
    verify_untouched(
        csv_baseline,
        get_dir_hashes("results/phase1a_cstp_profiling/csv"),
        "results/phase1a_cstp_profiling/csv"
    )

    print("\n=== Step 2: Defining File Copy Specifications ===")

    # Define CSTP file mappings
    cstp_copies = [
        # Source code
        (
            "LLMSimulator/src/predictor/cstp_predictor.py",
            "mohman/point_1/cstp/src/cstp_predictor.py"
        ),
        (
            "LLMSimulator/src/prefetch/cstp_predictor.cpp",
            "mohman/point_1/cstp/src/cstp_predictor.cpp"
        ),
        (
            "LLMSimulator/src/prefetch/cstp_predictor.h",
            "mohman/point_1/cstp/src/cstp_predictor.h"
        ),
        (
            "LLMSimulator/src/prefetch/predictor_interface.h",
            "mohman/point_1/cstp/src/predictor_interface.h"
        ),
        (
            "LLMSimulator/src/predictor/base_predictor.py",
            "mohman/point_1/cstp/src/base_predictor.py"
        ),
        (
            "LLMSimulator/src/predictor/evaluation.py",
            "mohman/point_1/cstp/src/evaluation.py"
        ),
        (
            "LLMSimulator/src/predictor/trace_parser.py",
            "mohman/point_1/cstp/src/trace_parser.py"
        ),
        # Tests
        (
            "LLMSimulator/tests/test_cstp.py",
            "mohman/point_1/cstp/tests/test_cstp.py"
        ),
        (
            "LLMSimulator/tests/test_cstp_profiling_sensitivity.py",
            "mohman/point_1/cstp/tests/test_cstp_profiling_sensitivity.py"
        ),
        (
            "LLMSimulator/tests/test_phase1a_certified_correction.py",
            "mohman/point_1/cstp/tests/test_phase1a_certified_correction.py"
        ),
        (
            "LLMSimulator/tests/test_evaluation.py",
            "mohman/point_1/cstp/tests/test_evaluation.py"
        ),
        (
            "LLMSimulator/tests/test_evaluation_semantics.py",
            "mohman/point_1/cstp/tests/test_evaluation_semantics.py"
        ),
        # Experiments
        (
            "LLMSimulator/scripts/phase1a/run_phase1a_certified_correction.py",
            "mohman/point_1/cstp/experiments/phase1a_profiling_sensitivity/run_phase1a_certified_correction.py"
        ),
        (
            "LLMSimulator/scripts/phase1a/run_cstp_profiling_sensitivity.py",
            "mohman/point_1/cstp/experiments/phase1a_profiling_sensitivity/run_cstp_profiling_sensitivity.py"
        ),
        (
            "LLMSimulator/scripts/phase1a/smoke_test_certified_correction.py",
            "mohman/point_1/cstp/experiments/phase1a_profiling_sensitivity/smoke_test_certified_correction.py"
        ),
        (
            "results/phase1a_cstp_profiling/csv/profiling_subset_manifest.csv",
            "mohman/point_1/cstp/experiments/phase1a_profiling_sensitivity/manifests/profiling_subset_manifest.csv"
        ),
        # Certified CSV results
        (
            "results/phase1a_cstp_profiling/certified/corrected_aggregate_metrics.csv",
            "mohman/point_1/cstp/results/csv/corrected_aggregate_metrics.csv"
        ),
        (
            "results/phase1a_cstp_profiling/certified/corrected_per_task_metrics.csv",
            "mohman/point_1/cstp/results/csv/corrected_per_task_metrics.csv"
        ),
        (
            "results/phase1a_cstp_profiling/certified/corrected_per_prompt_metrics.csv",
            "mohman/point_1/cstp/results/csv/corrected_per_prompt_metrics.csv"
        ),
        (
            "results/phase1a_cstp_profiling/certified/corrected_st_moe_head_to_head.csv",
            "mohman/point_1/cstp/results/csv/corrected_st_moe_head_to_head.csv"
        ),
        (
            "results/phase1a_cstp_profiling/certified/corrected_profiling_count_statistics.csv",
            "mohman/point_1/cstp/results/csv/corrected_profiling_count_statistics.csv"
        ),
        (
            "results/phase1a_cstp_profiling/certified/corrected_statistical_analysis.csv",
            "mohman/point_1/cstp/results/csv/corrected_statistical_analysis.csv"
        ),
        (
            "results/phase1a_cstp_profiling/certified/corrected_per_event_metrics.csv",
            "mohman/point_1/cstp/results/csv/corrected_per_event_metrics.csv"
        ),
        # Certified figures PNG & PDF
        (
            "results/phase1a_cstp_profiling/certified/plots/fig1_certified_frozen_recall_vs_profiling.png",
            "mohman/point_1/cstp/results/plots/fig1_certified_frozen_recall_vs_profiling.png"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig1_certified_frozen_recall_vs_profiling.pdf",
            "mohman/point_1/cstp/results/plots/fig1_certified_frozen_recall_vs_profiling.pdf"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig2_certified_online_recall_vs_profiling.png",
            "mohman/point_1/cstp/results/plots/fig2_certified_online_recall_vs_profiling.png"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig2_certified_online_recall_vs_profiling.pdf",
            "mohman/point_1/cstp/results/plots/fig2_certified_online_recall_vs_profiling.pdf"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig3_certified_online_adaptation_gain.png",
            "mohman/point_1/cstp/results/plots/fig3_certified_online_adaptation_gain.png"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig3_certified_online_adaptation_gain.pdf",
            "mohman/point_1/cstp/results/plots/fig3_certified_online_adaptation_gain.pdf"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig4_certified_stmoe_vs_cstp_head_to_head.png",
            "mohman/point_1/cstp/results/plots/fig4_certified_stmoe_vs_cstp_head_to_head.png"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig4_certified_stmoe_vs_cstp_head_to_head.pdf",
            "mohman/point_1/cstp/results/plots/fig4_certified_stmoe_vs_cstp_head_to_head.pdf"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig5_certified_horizon1_vs_horizon2.png",
            "mohman/point_1/cstp/results/plots/fig5_certified_horizon1_vs_horizon2.png"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig5_certified_horizon1_vs_horizon2.pdf",
            "mohman/point_1/cstp/results/plots/fig5_certified_horizon1_vs_horizon2.pdf"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig6_certified_seed_variability.png",
            "mohman/point_1/cstp/results/plots/fig6_certified_seed_variability.png"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig6_certified_seed_variability.pdf",
            "mohman/point_1/cstp/results/plots/fig6_certified_seed_variability.pdf"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig7_certified_moe_suite_task_level.png",
            "mohman/point_1/cstp/results/plots/fig7_certified_moe_suite_task_level.png"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig7_certified_moe_suite_task_level.pdf",
            "mohman/point_1/cstp/results/plots/fig7_certified_moe_suite_task_level.pdf"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig8_certified_transition_count_growth.png",
            "mohman/point_1/cstp/results/plots/fig8_certified_transition_count_growth.png"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig8_certified_transition_count_growth.pdf",
            "mohman/point_1/cstp/results/plots/fig8_certified_transition_count_growth.pdf"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig9_certified_confidence_intervals.png",
            "mohman/point_1/cstp/results/plots/fig9_certified_confidence_intervals.png"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig9_certified_confidence_intervals.pdf",
            "mohman/point_1/cstp/results/plots/fig9_certified_confidence_intervals.pdf"
        ),
        # Reports
        (
            "PHASE1A_CORRECTED_FINAL_REPORT.md",
            "mohman/point_1/cstp/reports/PHASE1A_CORRECTED_FINAL_REPORT.md"
        ),
        (
            "PHASE1A_CORRECTION_AUDIT.md",
            "mohman/point_1/cstp/reports/PHASE1A_CORRECTION_AUDIT.md"
        ),
        (
            "CSTP_AUDIT_REPORT.md",
            "mohman/point_1/cstp/reports/CSTP_AUDIT_REPORT.md"
        ),
        (
            "CSTP_PROFILING_PERCENTAGE_REPORT.md",
            "mohman/point_1/cstp/reports/CSTP_PROFILING_PERCENTAGE_REPORT.md"
        ),
        # Validation logs
        (
            "correction_validation_tests.txt",
            "mohman/point_1/cstp/validation/correction_validation_tests.txt"
        ),
    ]

    # Define stmoe_cstp comparative file mappings
    stmoe_cstp_copies = [
        # Certified CSV results
        (
            "results/phase1a_cstp_profiling/certified/corrected_st_moe_head_to_head.csv",
            "mohman/point_1/stmoe_cstp/results/csv/matched_comparisons.csv"
        ),
        (
            "results/phase1a_cstp_profiling/certified/corrected_aggregate_metrics.csv",
            "mohman/point_1/stmoe_cstp/results/csv/aggregate_metrics.csv"
        ),
        (
            "results/phase1a_cstp_profiling/certified/corrected_profiling_count_statistics.csv",
            "mohman/point_1/stmoe_cstp/results/csv/count_statistics.csv"
        ),
        (
            "results/phase1a_cstp_profiling/certified/corrected_statistical_analysis.csv",
            "mohman/point_1/stmoe_cstp/results/csv/statistical_analysis.csv"
        ),
        (
            "results/phase1a_cstp_profiling/certified/corrected_per_task_metrics.csv",
            "mohman/point_1/stmoe_cstp/results/csv/per_task_metrics.csv"
        ),
        (
            "results/phase1a_cstp_profiling/certified/corrected_per_prompt_metrics.csv",
            "mohman/point_1/stmoe_cstp/results/csv/per_prompt_metrics.csv"
        ),
        (
            "results/phase1a_cstp_profiling/certified/corrected_st_moe_head_to_head.csv",
            "mohman/point_1/stmoe_cstp/results/csv/corrected_st_moe_head_to_head.csv"
        ),
        (
            "results/phase1a_cstp_profiling/certified/corrected_aggregate_metrics.csv",
            "mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv"
        ),
        (
            "results/phase1a_cstp_profiling/certified/corrected_profiling_count_statistics.csv",
            "mohman/point_1/stmoe_cstp/results/csv/corrected_profiling_count_statistics.csv"
        ),
        (
            "results/phase1a_cstp_profiling/certified/corrected_statistical_analysis.csv",
            "mohman/point_1/stmoe_cstp/results/csv/corrected_statistical_analysis.csv"
        ),
        # Certified figures PNG & PDF
        (
            "results/phase1a_cstp_profiling/certified/plots/fig1_certified_frozen_recall_vs_profiling.png",
            "mohman/point_1/stmoe_cstp/results/plots/fig1_certified_frozen_recall_vs_profiling.png"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig1_certified_frozen_recall_vs_profiling.pdf",
            "mohman/point_1/stmoe_cstp/results/plots/fig1_certified_frozen_recall_vs_profiling.pdf"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig2_certified_online_recall_vs_profiling.png",
            "mohman/point_1/stmoe_cstp/results/plots/fig2_certified_online_recall_vs_profiling.png"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig2_certified_online_recall_vs_profiling.pdf",
            "mohman/point_1/stmoe_cstp/results/plots/fig2_certified_online_recall_vs_profiling.pdf"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig3_certified_online_adaptation_gain.png",
            "mohman/point_1/stmoe_cstp/results/plots/fig3_certified_online_adaptation_gain.png"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig3_certified_online_adaptation_gain.pdf",
            "mohman/point_1/stmoe_cstp/results/plots/fig3_certified_online_adaptation_gain.pdf"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig4_certified_stmoe_vs_cstp_head_to_head.png",
            "mohman/point_1/stmoe_cstp/results/plots/fig4_certified_stmoe_vs_cstp_head_to_head.png"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig4_certified_stmoe_vs_cstp_head_to_head.pdf",
            "mohman/point_1/stmoe_cstp/results/plots/fig4_certified_stmoe_vs_cstp_head_to_head.pdf"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig5_certified_horizon1_vs_horizon2.png",
            "mohman/point_1/stmoe_cstp/results/plots/fig5_certified_horizon1_vs_horizon2.png"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig5_certified_horizon1_vs_horizon2.pdf",
            "mohman/point_1/stmoe_cstp/results/plots/fig5_certified_horizon1_vs_horizon2.pdf"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig6_certified_seed_variability.png",
            "mohman/point_1/stmoe_cstp/results/plots/fig6_certified_seed_variability.png"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig6_certified_seed_variability.pdf",
            "mohman/point_1/stmoe_cstp/results/plots/fig6_certified_seed_variability.pdf"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig7_certified_moe_suite_task_level.png",
            "mohman/point_1/stmoe_cstp/results/plots/fig7_certified_moe_suite_task_level.png"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig7_certified_moe_suite_task_level.pdf",
            "mohman/point_1/stmoe_cstp/results/plots/fig7_certified_moe_suite_task_level.pdf"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig8_certified_transition_count_growth.png",
            "mohman/point_1/stmoe_cstp/results/plots/fig8_certified_transition_count_growth.png"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig8_certified_transition_count_growth.pdf",
            "mohman/point_1/stmoe_cstp/results/plots/fig8_certified_transition_count_growth.pdf"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig9_certified_confidence_intervals.png",
            "mohman/point_1/stmoe_cstp/results/plots/fig9_certified_confidence_intervals.png"
        ),
        (
            "results/phase1a_cstp_profiling/certified/plots/fig9_certified_confidence_intervals.pdf",
            "mohman/point_1/stmoe_cstp/results/plots/fig9_certified_confidence_intervals.pdf"
        ),
        # Validation logs
        (
            "correction_validation_tests.txt",
            "mohman/point_1/stmoe_cstp/validation/correction_validation_tests.txt"
        ),
    ]

    all_copies = [("cstp", cstp_copies), ("stmoe_cstp", stmoe_cstp_copies)]

    print("\n=== Step 3: Verifying Source File Existence ===")
    for domain, copy_list in all_copies:
        for src, dst in copy_list:
            if not os.path.exists(src):
                raise FileNotFoundError("Source file does not exist: " + src)
    print("PASS: All source files exist.")

    print("\n=== Step 4: Copying Files and Verifying SHA-256 ===")
    manifest_records = {"cstp": [], "stmoe_cstp": []}

    for domain, copy_list in all_copies:
        print("Processing domain: " + domain + " (" + str(len(copy_list)) + " files)")
        for src, dst in copy_list:
            dst_dir = os.path.dirname(dst)
            os.makedirs(dst_dir, exist_ok=True)
            
            # Compute source hash and size
            src_hash = compute_sha256(src)
            src_size = os.path.getsize(src)

            # Copy file preserving metadata
            shutil.copy2(src, dst)

            # Compute destination hash and size
            dst_hash = compute_sha256(dst)
            dst_size = os.path.getsize(dst)

            if src_hash != dst_hash:
                raise RuntimeError("Integrity verification failed for: " + dst)
            if src_size != dst_size:
                raise RuntimeError("Size verification failed for: " + dst)

            record = {
                "source_path": src,
                "destination_path": dst,
                "file_size_bytes": dst_size,
                "sha256": dst_hash,
                "status": "VERIFIED_MATCH"
            }
            manifest_records[domain].append(record)

    print("PASS: All files copied and cryptographically verified.")

    # Save JSON manifests for audit
    with open("/tmp/cstp_manifest.json", "w") as f:
        json.dump(manifest_records["cstp"], f, indent=2)
    with open("/tmp/stmoe_cstp_manifest.json", "w") as f:
        json.dump(manifest_records["stmoe_cstp"], f, indent=2)

    print("\n=== Step 5: Verifying Untouched Baselines ===")
    verify_untouched(
        st_moe_baseline,
        get_dir_hashes("mohman/point_1/st_moe"),
        "mohman/point_1/st_moe"
    )
    verify_untouched(
        csv_baseline,
        get_dir_hashes("results/phase1a_cstp_profiling/csv"),
        "results/phase1a_cstp_profiling/csv"
    )
    print("PASS: Both baseline directories remain 100% untouched.")

    print("\n=== Packaging Execution Completed Successfully ===")


if __name__ == "__main__":
    main()

