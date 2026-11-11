#!/usr/bin/env python3
import os
import csv
import ast

WORKSPACE_DIR = "/home/mohman/ca_project"
SUBMISSION_DIR = os.path.join(WORKSPACE_DIR, "final_submission")
MASTER_DIR = os.path.join(SUBMISSION_DIR, "00_MASTER")

manifest_data = [
    {
        "file": "LLMSimulator/CMakeLists.txt",
        "original_status": "Tracked in baseline d5bf61f",
        "final_status": "Modified (+2 lines)",
        "modification_type": "Build Configuration",
        "reason": "Includes src/prefetch subdirectory in simulator build",
        "related_requirement": "REQ-1, REQ-2, REQ-8"
    },
    {
        "file": "LLMSimulator/config.yaml",
        "original_status": "Tracked in baseline d5bf61f",
        "final_status": "Modified (+1 line)",
        "modification_type": "Default Configuration",
        "reason": "Enables offchip DRAM execution cache bypass default flag",
        "related_requirement": "REQ-2, REQ-8"
    },
    {
        "file": "LLMSimulator/eval/CMakeLists.txt",
        "original_status": "Tracked in baseline d5bf61f",
        "final_status": "Modified (+64 lines)",
        "modification_type": "Build Configuration",
        "reason": "Adds executable targets for all unit, regression, and validation suites",
        "related_requirement": "REQ-1, REQ-2, REQ-4, REQ-6, REQ-8"
    },
    {
        "file": "LLMSimulator/eval/test.cpp",
        "original_status": "Tracked in baseline d5bf61f",
        "final_status": "Modified (+14 lines)",
        "modification_type": "Test Harness",
        "reason": "Exposes test hook for cache validation",
        "related_requirement": "REQ-8"
    },
    {
        "file": "LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp",
        "original_status": "Tracked in baseline d5bf61f",
        "final_status": "Modified (+220 lines, -4 lines)",
        "modification_type": "Core Driver",
        "reason": "Integrates live prediction engine, prefetch candidate queue, forward lookahead, clock continuity, and per-token step latency isolation",
        "related_requirement": "REQ-1, REQ-2, REQ-3, REQ-6, REQ-8"
    },
    {
        "file": "LLMSimulator/src/CMakeLists.txt",
        "original_status": "Tracked in baseline d5bf61f",
        "final_status": "Modified (+2 lines)",
        "modification_type": "Build Configuration",
        "reason": "Links prefetch library into libllm_system.so",
        "related_requirement": "REQ-8"
    },
    {
        "file": "LLMSimulator/src/hardware/cluster.cpp",
        "original_status": "Tracked in baseline d5bf61f",
        "final_status": "Modified (+21 lines, -5 lines)",
        "modification_type": "Hardware Simulation",
        "reason": "Preserves monotonic device time across decoding tokens without double-counting step latency; guards stat.latency in setStat",
        "related_requirement": "REQ-3, REQ-6, REQ-7"
    },
    {
        "file": "LLMSimulator/src/hardware/cluster.h",
        "original_status": "Tracked in baseline d5bf61f",
        "final_status": "Modified (+3 lines)",
        "modification_type": "Header Definition",
        "reason": "Declares clock continuity helper and step timing members",
        "related_requirement": "REQ-3, REQ-6"
    },
    {
        "file": "LLMSimulator/src/hardware/device.cpp",
        "original_status": "Tracked in baseline d5bf61f",
        "final_status": "Modified (+1377 lines, -26 lines)",
        "modification_type": "Core Memory & Prefetch Engine",
        "reason": "Implements candidate dispatch, PABI quarantine (20% cap), Belady-style pinning, DDR5 transfer coordination, demand attachment, logical stage alignment",
        "related_requirement": "REQ-2, REQ-3, REQ-4, REQ-5"
    },
    {
        "file": "LLMSimulator/src/hardware/device.h",
        "original_status": "Tracked in baseline d5bf61f",
        "final_status": "Modified (+250 lines, -4 lines)",
        "modification_type": "Header Definition",
        "reason": "Declares prefetch statistics counters, PABI structures, DDR5 transfer engine pointer, and stage state tracking",
        "related_requirement": "REQ-2, REQ-3, REQ-4"
    },
    {
        "file": "LLMSimulator/src/hardware/hardware_config.h",
        "original_status": "Tracked in baseline d5bf61f",
        "final_status": "Modified (+81 lines, -10 lines)",
        "modification_type": "Configuration System",
        "reason": "Adds compute_time_multiplier and clock_continuity to SystemConfig constructors and default hardware profiles with env overrides",
        "related_requirement": "REQ-6, REQ-8"
    },
    {
        "file": "LLMSimulator/src/hardware/layer_impl.cpp",
        "original_status": "Tracked in baseline d5bf61f",
        "final_status": "Modified (+32 lines, -3 lines)",
        "modification_type": "Layer Implementation",
        "reason": "Passes device configuration and multiplier context to layer executions",
        "related_requirement": "REQ-6"
    },
    {
        "file": "LLMSimulator/src/hardware/linear_impl.cpp",
        "original_status": "Tracked in baseline d5bf61f",
        "final_status": "Modified (+167 lines, -2 lines)",
        "modification_type": "Timing Model",
        "reason": "Decouples on-chip duration scaling from DDR5 miss penalty; preserves exact baseline max(compute, memory) formula when M=1.0",
        "related_requirement": "REQ-2, REQ-6"
    },
    {
        "file": "LLMSimulator/src/module/linear.cpp",
        "original_status": "Tracked in baseline d5bf61f",
        "final_status": "Modified (+8 lines, -1 line)",
        "modification_type": "Module Layer",
        "reason": "Propagates stage identifiers and weight metadata to LinearExecutionGPU",
        "related_requirement": "REQ-2, REQ-3"
    },
    {
        "file": "LLMSimulator/src/module/module_graph.cpp",
        "original_status": "Tracked in baseline d5bf61f",
        "final_status": "Modified (+3 lines)",
        "modification_type": "Graph Execution",
        "reason": "Guards empty module graph assertion in restart_graph",
        "related_requirement": "REQ-8"
    },
    {
        "file": "LLMSimulator/src/module/route.cpp",
        "original_status": "Tracked in baseline d5bf61f",
        "final_status": "Modified (+3 lines)",
        "modification_type": "Router Interface",
        "reason": "Authoritative routing preservation hook; publishes historical routing sequence to prediction engine after stage completion",
        "related_requirement": "REQ-1, REQ-2"
    },
    {
        "file": "LLMSimulator/src/scheduler/scheduler.cpp",
        "original_status": "Tracked in baseline d5bf61f",
        "final_status": "Modified (+127 lines)",
        "modification_type": "Scheduler",
        "reason": "Supports batched sequence scheduling and prefetch queue synchronisation",
        "related_requirement": "REQ-2, REQ-6"
    },
    {
        "file": "LLMSimulator/src/scheduler/scheduler.h",
        "original_status": "Tracked in baseline d5bf61f",
        "final_status": "Modified (+18 lines)",
        "modification_type": "Header Definition",
        "reason": "Declares queue synchronization interfaces",
        "related_requirement": "REQ-2"
    },
    {
        "file": "LLMSimulator/src/scheduler/sequence.cpp",
        "original_status": "Tracked in baseline d5bf61f",
        "final_status": "Modified (+89 lines, -31 lines)",
        "modification_type": "Trace Loader",
        "reason": "Loads real recorded MoE-Suite and MoE-Anywhere routing traces; parses CSV trace files and assigns expert sequences",
        "related_requirement": "BONUS, REQ-6"
    },
    {
        "file": "LLMSimulator/src/scheduler/sequence.h",
        "original_status": "Tracked in baseline d5bf61f",
        "final_status": "Modified (+3 lines)",
        "modification_type": "Header Definition",
        "reason": "Declares trace metadata members",
        "related_requirement": "BONUS"
    },
    {
        "file": "LLMSimulator/src/prefetch/st_moe_predictor.cpp",
        "original_status": "New file (untracked in baseline)",
        "final_status": "Implemented (288 lines)",
        "modification_type": "New Subsystem",
        "reason": "Implements ST-MoE frequency and spatial co-activation tracking with 4-bit saturating counters",
        "related_requirement": "REQ-1"
    },
    {
        "file": "LLMSimulator/src/prefetch/cstp_predictor.cpp",
        "original_status": "New file (untracked in baseline)",
        "final_status": "Implemented (320 lines)",
        "modification_type": "New Subsystem",
        "reason": "Implements CSTP Laplace-smoothed spatio-temporal transition Markov model",
        "related_requirement": "REQ-1"
    },
    {
        "file": "LLMSimulator/src/prefetch/live_prediction_engine.cpp",
        "original_status": "New file (untracked in baseline)",
        "final_status": "Implemented (547 lines)",
        "modification_type": "New Subsystem",
        "reason": "Manages forward lookahead (L -> L+H), causal isolation, and streaming statistics logging",
        "related_requirement": "REQ-1, REQ-2, REQ-3"
    },
    {
        "file": "LLMSimulator/src/prefetch/prefetch_candidate_queue.cpp",
        "original_status": "New file (untracked in baseline)",
        "final_status": "Implemented (440 lines)",
        "modification_type": "New Subsystem",
        "reason": "Candidate queue with priority ranking, duplicate suppression, and ADMITTED state filtering",
        "related_requirement": "REQ-2, REQ-6"
    },
    {
        "file": "LLMSimulator/src/prefetch/ddr5_transfer_engine.cpp",
        "original_status": "New file (untracked in baseline)",
        "final_status": "Implemented (800 lines)",
        "modification_type": "New Subsystem",
        "reason": "Asynchronous DDR5 transfer engine: 22.578 GB/s bandwidth, demand preemption, pause/resume, in-flight attachment",
        "related_requirement": "REQ-2, REQ-3"
    }
]

def check_no_ternary():
    with open(__file__, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=__file__)
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            raise AssertionError("Ternary operator found!")
    print("[PRE-FLIGHT] Zero ternary operators verified.")

def main():
    check_no_ternary()
    os.makedirs(MASTER_DIR, exist_ok=True)

    # 1. Write CSV
    csv_path = os.path.join(MASTER_DIR, "FINAL_SOURCE_CHANGE_MANIFEST.csv")
    fieldnames = ["file", "original_status", "final_status", "modification_type", "reason", "related_requirement"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for r in manifest_data:
            writer.writerow(r)
    print(f"[SUCCESS] Wrote {csv_path}")

    # 2. Write Markdown Audit
    md_lines = [
        "# Final Source Code Change Audit",
        "**MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference**\n",
        "### Baseline Git Anchor: `d5bf61ff0c066c06f72173189f9924329f613a88`\n",
        "This audit establishes the exact lineage and technical justification for all modifications made relative to the authoritative baseline anchor.\n",
        "| File | Original Baseline Status | Final Status | Modification Type | Technical Reason | Requirement |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for r in manifest_data:
        md_lines.append(f"| `{r['file']}` | {r['original_status']} | `{r['final_status']}` | {r['modification_type']} | {r['reason']} | **{r['related_requirement']}** |")

    md_path = os.path.join(MASTER_DIR, "FINAL_SOURCE_CHANGE_AUDIT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[SUCCESS] Wrote {md_path}")

if __name__ == "__main__":
    main()

