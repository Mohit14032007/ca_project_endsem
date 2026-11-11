"""
Generates the Prefetch Algorithm Inventory CSV for the Current Implementation Audit.
Strictly zero ternary operators (ast.IfExp == 0).
"""

import os
import sys
import pandas as pd

PROJECT_ROOT = "/home/mohman/ca_project/LLMSimulator"

inventory_data = [
    {
        "component_name": "ACSH Master Predictor",
        "subsystem": "Predictor Layer",
        "implementation_language": "Python",
        "status": "IMPLEMENTED",
        "source_file_references": "src/predictor/acsh_predictor.py:35-375",
        "input_interface": "request_id, target_stage_id, current_source_experts, budget, token_id, source_stage_id",
        "output_interface": "List[int] (Top-B expert IDs directly)",
        "hardware_awareness": "NONE (No bandwidth, transfer time, deadline, or HBM capacity awareness)",
        "evaluation_metric": "Recall@B, Precision@B, Hit Rate (Algorithmic accuracy)",
        "current_limitations": "Outputs fixed Top-B without confidence thresholding; does not output probabilities to callers"
    },
    {
        "component_name": "CSTP Predictor",
        "subsystem": "Predictor Layer",
        "implementation_language": "Python",
        "status": "IMPLEMENTED",
        "source_file_references": "src/predictor/cstp_predictor.py:20-210",
        "input_interface": "request_id, current_stage, current_experts, budget",
        "output_interface": "List[int] (Top-B expert IDs)",
        "hardware_awareness": "NONE (Pure statistical Dirichlet count model)",
        "evaluation_metric": "Recall@B, Precision@B, Hit Rate",
        "current_limitations": "Spatial transition degrades at lookahead H >= 2; high state memory (57.97 MiB heap)"
    },
    {
        "component_name": "ST-MoE Paper Predictor",
        "subsystem": "Predictor Layer",
        "implementation_language": "Python",
        "status": "IMPLEMENTED",
        "source_file_references": "src/predictor/st_moe_paper.py:30-360",
        "input_interface": "request_id, current_stage, current_experts, budget",
        "output_interface": "List[int] (Top-B expert IDs or thresholded candidate set)",
        "hardware_awareness": "NONE (Candidate Co-activation Table and 2-bit counters)",
        "evaluation_metric": "Recall@B, Precision@B, Hit Rate",
        "current_limitations": "Lacks continuous probability estimation; saturates at 2-bit counter maximum (11)"
    },
    {
        "component_name": "Predictor-to-Prefetch Selection Logic",
        "subsystem": "Scheduler / Prefetch Policy",
        "implementation_language": "Python",
        "status": "PARTIALLY_IMPLEMENTED",
        "source_file_references": "src/predictor/acsh_predictor.py:320-335; src/predictor/evaluation.py:80-120",
        "input_interface": "Score vectors (p_acsh) and integer budget B",
        "output_interface": "Slices top B elements: acsh_sorted[:min(budget, num_exp)]",
        "hardware_awareness": "NONE (Always transfers exactly B candidates; no cost-benefit policy)",
        "evaluation_metric": "Theoretical prefetch accounting (Predicted, Actual, Missed, Wasted)",
        "current_limitations": "No separate prefetch scheduler; assumes all B candidates are transferred into HBM"
    },
    {
        "component_name": "Python-to-C++ Bridge / IPC",
        "subsystem": "Inter-Language Communication",
        "implementation_language": "None",
        "status": "NOT_IMPLEMENTED_PROPOSED_ONLY",
        "source_file_references": "docs/NEXT_IMPLEMENTATION_ROADMAP.md:65-75 (Proposed for Phase E)",
        "input_interface": "N/A (No IPC, sockets, files, or pybind11 bindings exist)",
        "output_interface": "N/A",
        "hardware_awareness": "NONE",
        "evaluation_metric": "N/A",
        "current_limitations": "Python predictors run completely decoupled from C++ execution simulator"
    },
    {
        "component_name": "Asynchronous DDR5-to-HBM Prefetch Queue",
        "subsystem": "C++ Memory Subsystem",
        "implementation_language": "C++",
        "status": "NOT_IMPLEMENTED_PROPOSED_ONLY",
        "source_file_references": "src/dram/dram_interface.cpp:125-135; docs/NEXT_IMPLEMENTATION_ROADMAP.md:85-105",
        "input_interface": "Synchronous DRAMRequest::Ptr via issueRamulatorBatch()",
        "output_interface": "Blocking execution loop: memory_system->tick() until finished",
        "hardware_awareness": "Cycle-accurate Ramulator DRAM timing (Synchronous demand-only)",
        "evaluation_metric": "Memory duration (ns), DRAM read/write command counts",
        "current_limitations": "Blocking calls halt simulator; zero DMA concurrency or non-blocking transfer queues"
    },
    {
        "component_name": "HBM Residency & Eviction Manager",
        "subsystem": "C++ Hardware Subsystem",
        "implementation_language": "C++",
        "status": "NOT_IMPLEMENTED_PROPOSED_ONLY",
        "source_file_references": "src/hardware/layer_impl.cpp:20-100; docs/NEXT_IMPLEMENTATION_ROADMAP.md:110-130",
        "input_interface": "N/A (C++ models timing per layer; no dynamic expert weight table)",
        "output_interface": "N/A",
        "hardware_awareness": "NONE (Memoization execution cache keys by tensor shape/type, not physical residency)",
        "evaluation_metric": "N/A",
        "current_limitations": "No tracking of resident expert weights, buffer capacity limits, or replacement policies"
    },
    {
        "component_name": "PCIe Bandwidth & Contention Arbiter",
        "subsystem": "C++ Hardware Subsystem",
        "implementation_language": "C++",
        "status": "NOT_IMPLEMENTED_PROPOSED_ONLY",
        "source_file_references": "src/hardware/cluster.cpp:80-140; docs/NEXT_IMPLEMENTATION_ROADMAP.md:95-105",
        "input_interface": "N/A",
        "output_interface": "N/A",
        "hardware_awareness": "NONE (Static analytical bandwidth limits only)",
        "evaluation_metric": "N/A",
        "current_limitations": "Cannot model contention between concurrent prefetch transfers and demand loads"
    },
    {
        "component_name": "Evaluation Harness & Metric Accounting",
        "subsystem": "Evaluation Layer",
        "implementation_language": "Python",
        "status": "IMPLEMENTED",
        "source_file_references": "scripts/phase_d/benchmark_predictors_d.py:280-450; src/predictor/evaluation.py:20-140",
        "input_interface": "Predicted candidate set vs true ground-truth expert set",
        "output_interface": "Macro/Micro Recall, Precision, F1, Hit Rate, Wasted/Missed counts",
        "hardware_awareness": "NONE (Measures algorithmic quality; does not simulate physical memory stalls avoided)",
        "evaluation_metric": "Recall@B, Precision@B, F1, Hit Rate",
        "current_limitations": "Cannot measure physical stall cycles, late prefetches, or end-to-end wall-clock speedup"
    }
]

def main():
    df = pd.DataFrame(inventory_data)
    
    out_dir_1 = os.path.join(PROJECT_ROOT, "results", "endsem_phase_d", "audit")
    os.makedirs(out_dir_1, exist_ok=True)
    out_csv_1 = os.path.join(out_dir_1, "prefetch_algorithm_inventory.csv")
    df.to_csv(out_csv_1, index=False)
    print(f"Saved prefetch algorithm inventory to: {out_csv_1}")

    out_dir_2 = "/home/mohman/ca_project/results/endsem_phase_d/audit"
    os.makedirs(out_dir_2, exist_ok=True)
    out_csv_2 = os.path.join(out_dir_2, "prefetch_algorithm_inventory.csv")
    df.to_csv(out_csv_2, index=False)
    print(f"Saved mirrored prefetch algorithm inventory to: {out_csv_2}")

if __name__ == "__main__":
    main()
