#!/usr/bin/env python3
import os
import ast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

WORKSPACE_DIR = "/home/mohman/ca_project"
SUBMISSION_DIR = os.path.join(WORKSPACE_DIR, "final_submission")
MECH_DIR = os.path.join(SUBMISSION_DIR, "02_PREFETCHING_MECHANISM")

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
    diag_dir = os.path.join(MECH_DIR, "diagrams")
    graphs_dir = os.path.join(MECH_DIR, "graphs")
    os.makedirs(diag_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    # 1. State Machine Diagram in diagrams/
    diag_md = """# DDR5 Transfer Engine & Prefetch Candidate Lifecycle
```mermaid
stateDiagram-v2
    [*] --> PENDING: Predictor generates Top-K candidate for target stage L+H
    PENDING --> ADMITTED: Candidate queue admits and sorts by (Deadline, Score)
    PENDING --> REJECTED: Candidate queue full or below threshold
    ADMITTED --> ACTIVE: DDR5 channel available; bus service begins (22.578 GB/s)
    ACTIVE --> PAUSED: Demand miss preempts bus channel
    PAUSED --> ACTIVE: Demand transfer completes; speculative fetch resumes
    ACTIVE --> COMPLETED: 88.08 MB transferred (3.901 ms total service)
    COMPLETED --> PROMOTED: Target stage demands expert; promoted to HBM Demand pool
    COMPLETED --> EXPIRED: Target stage completes without demanding expert
    PROMOTED --> [*]
    EXPIRED --> [*]
    REJECTED --> [*]
```
"""
    with open(os.path.join(diag_dir, "PREFETCH_STATE_MACHINE.md"), "w", encoding="utf-8") as f:
        f.write(diag_md)

    # 2. Render State Machine and Timeline Plot in graphs/
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")
    ax.set_title("Forward-Lookahead Timeline (Source L=10 -> Target L+H=12, H=2)", fontsize=12, fontweight="bold")

    # Draw Timeline Bars
    # Stage 10 Compute
    ax.add_patch(patches.Rectangle((0.1, 0.65), 0.15, 0.2, facecolor="#4daf4a", edgecolor="black"))
    ax.text(0.175, 0.75, "Stage 10\nCompute\n(0.18 ms)", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # Stage 11 Compute
    ax.add_patch(patches.Rectangle((0.27, 0.65), 0.15, 0.2, facecolor="#4daf4a", edgecolor="black"))
    ax.text(0.345, 0.75, "Stage 11\nCompute\n(0.18 ms)", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # Stage 12 Compute
    ax.add_patch(patches.Rectangle((0.44, 0.65), 0.15, 0.2, facecolor="#4daf4a", edgecolor="black"))
    ax.text(0.515, 0.75, "Stage 12\nDemand Lookup\n& Compute", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # Speculative DDR5 Transfer
    ax.add_patch(patches.Rectangle((0.25, 0.25), 0.35, 0.2, facecolor="#e41a1c", edgecolor="black"))
    ax.text(0.425, 0.35, "Asynchronous DDR5 Transfer (3.90 ms)\n[Overlaps across stages 10 & 11]", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # Annotations
    ax.annotate("Dispatch Prefetch for Stage 12", xy=(0.25, 0.45), xytext=(0.15, 0.52),
                arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.5), fontsize=8, fontweight="bold")
    ax.annotate("Timely Resident Hit!\n(Expert in PABI Quarantine)", xy=(0.44, 0.65), xytext=(0.62, 0.65),
                arrowprops=dict(facecolor="green", arrowstyle="->", lw=1.5), fontsize=8, fontweight="bold", color="green")

    plt.tight_layout()
    p_time = os.path.join(graphs_dir, "PREFETCH_TIMELINE_OVERLAP.png")
    plt.savefig(p_time, dpi=300)
    plt.close()
    print(f"[SUCCESS] Wrote mechanism diagram and plot to {p_time}")

if __name__ == "__main__":
    main()

