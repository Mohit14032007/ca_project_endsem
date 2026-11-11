"""
Visualization generator for MoE Expert Predictors (CSTP vs ST-MoE).

Generates high-resolution diagnostic diagrams illustrating:
1. ST-MoE CCT structure (with 2-bit counter states and replacement eligibility)
2. ST-MoE History Table (multi-request temporal tracking)
3. CSTP Spatial transition counts and Dirichlet-smoothed probabilities
4. CSTP Temporal transition matrix
5. CSTP Marginal frequency prior distribution
6. Side-by-side prediction scoring comparison for running example
7. Side-by-side state update lifecycle comparison
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Output directory
OUT_DIR = "/home/mohman/ca_project/LLMSimulator/docs/predictor_comparison/structure_visualization"
os.makedirs(OUT_DIR, exist_ok=True)


def plot_vis1_st_moe_cct():
    """Visualization 1: ST-MoE Cross-Layer Correlation Table (CCT) organization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Draw title
    ax.text(0.5, 0.95, "Visualization 1: ST-MoE Cross-Layer Correlation Table (CCT)",
            fontsize=14, weight='bold', ha='center')
    ax.text(0.5, 0.90, "Structure: self.cct[source_stage][source_expert] -> {dest_expert: counter (0..3)}",
            fontsize=11, style='italic', color='#4b5563', ha='center')

    # Table headers
    headers = ["Source Stage", "Source Expert", "Dest Candidate", "Counter", "State", "Status & Eligibility"]
    col_x = [0.08, 0.24, 0.40, 0.54, 0.68, 0.88]
    y_start = 0.80

    for x, h in zip(col_x, headers):
        ax.text(x, y_start, h, fontsize=11, weight='bold', ha='center')

    ax.plot([0.02, 0.98], [y_start - 0.02, y_start - 0.02], color='#1f2937', lw=1.5)

    # Illustrative entries
    rows = [
        ("Stage 0 (L)", "Expert 0", "Expert 1", "3 (11)", "Strongly Preferred", "#15803d", "Retained in CCT"),
        ("Stage 0 (L)", "Expert 0", "Expert 2", "2 (10)", "Preferred (Initial)", "#22c55e", "Retained in CCT"),
        ("Stage 0 (L)", "Expert 1", "Expert 2", "3 (11)", "Strongly Preferred", "#15803d", "Retained in CCT"),
        ("Stage 0 (L)", "Expert 1", "Expert 4", "2 (10)", "Preferred (Initial)", "#22c55e", "Retained in CCT"),
        ("Stage 0 (L)", "Expert 2", "Expert 6", "1 (01)", "Weakly Not Preferred", "#eab308", "Decremented once"),
        ("Stage 0 (L)", "Expert 2", "Expert 7", "0 (00)", "Strongly Not Preferred", "#dc2626", "ELIGIBLE FOR REPLACEMENT"),
    ]

    curr_y = y_start - 0.08
    for stage, src, dst, cnt, state, color, status in rows:
        # Background box for replacement eligibility
        if cnt.startswith("0"):
            rect = patches.Rectangle((0.02, curr_y - 0.03), 0.96, 0.06,
                                     facecolor='#fee2e2', edgecolor='#ef4444', alpha=0.5, lw=1)
            ax.add_patch(rect)
        elif cnt.startswith("3"):
            rect = patches.Rectangle((0.02, curr_y - 0.03), 0.96, 0.06,
                                     facecolor='#f0fdf4', edgecolor='#86efac', alpha=0.5, lw=0.5)
            ax.add_patch(rect)

        status_color = '#374151'
        if cnt.startswith("0"):
            status_color = '#b91c1c'

        ax.text(col_x[0], curr_y, stage, fontsize=10, ha='center')
        ax.text(col_x[1], curr_y, src, fontsize=10, weight='bold', ha='center')
        ax.text(col_x[2], curr_y, dst, fontsize=10, ha='center')
        ax.text(col_x[3], curr_y, cnt, fontsize=10, weight='bold', ha='center')
        ax.text(col_x[4], curr_y, state, fontsize=9, color=color, weight='bold', ha='center')
        ax.text(col_x[5], curr_y, status, fontsize=9, weight='bold', ha='center',
                color=status_color)

        curr_y -= 0.07

    # Explanatory footer note
    footer_text = (
        "Key Notes on ST-MoE CCT:\n"
        "- Initialized with binary 10 (decimal 2) during offline profiling (Algorithm 1) and replacement (Algorithm 3).\n"
        "- When counter reaches 0 (binary 00) upon mismatch, candidate is replaced by un-stored actual target expert.\n"
        "- Counter is strictly bounded between 0 and 3 (2-bit saturating logic)."
    )
    ax.text(0.05, 0.15, footer_text, fontsize=9.5, bbox=dict(facecolor='#f3f4f6', edgecolor='#d1d5db', boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "vis1_st_moe_cct.png"), dpi=200)
    plt.close()


def plot_vis2_st_moe_ht():
    """Visualization 2: ST-MoE History Table (HT) multi-request architecture."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis('off')

    ax.text(0.5, 0.95, "Visualization 2: ST-MoE History Table (HT) Organization",
            fontsize=14, weight='bold', ha='center')
    ax.text(0.5, 0.90, "Structure: self.history_table[request_id][stage_id] -> List[expert_ids]",
            fontsize=11, style='italic', color='#4b5563', ha='center')

    # Draw Request Containers
    boxes = [
        ("Request ID: 101 (Active Sequence A)", 0.05, 0.52, 0.42, 0.32, "#eff6ff", "#3b82f6"),
        ("Request ID: 202 (Active Sequence B)", 0.53, 0.52, 0.42, 0.32, "#f5f3ff", "#8b5cf6"),
    ]

    for title, x, y, w, h, bg, border in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                      facecolor=bg, edgecolor=border, lw=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.05, title, fontsize=11, weight='bold', ha='center', color=border)

    # Content for Req 101
    ax.text(0.08, 0.70, "• Stage 0: Actual token t-1 experts -> {0, 4}", fontsize=10)
    ax.text(0.08, 0.63, "• Stage 1: Actual token t-1 experts -> {2, 3}", fontsize=10, weight='bold', color='#1d4ed8')
    ax.text(0.08, 0.56, "• Stage 2: Actual token t-1 experts -> {5, 7}", fontsize=10)

    # Content for Req 202
    ax.text(0.56, 0.70, "• Stage 0: Actual token t-1 experts -> {1, 6}", fontsize=10)
    ax.text(0.56, 0.63, "• Stage 1: Actual token t-1 experts -> {12, 18, 21}", fontsize=10, weight='bold', color='#6d28d9')
    ax.text(0.56, 0.56, "• Stage 2: Actual token t-1 experts -> {3, 9}", fontsize=10)

    # Temporal score injection note
    ax.text(0.5, 0.38, "Prediction at Stage 1 for Request 101 reads ONLY Request 101's Stage 1 HT: {2, 3}",
            fontsize=11, weight='bold', ha='center', color='#1e3a8a')

    # Formula card
    formula_box = patches.FancyBboxPatch((0.08, 0.08), 0.84, 0.24, boxstyle="round,pad=0.02",
                                         facecolor='#f8fafc', edgecolor='#94a3b8', lw=1)
    ax.add_patch(formula_box)
    desc = (
        "History Table Properties & Lifecycle:\n"
        "1. Strictly Request-Isolated: Stored per request_id; reset() clears only the target request.\n"
        "2. Exact Previous Token: Records actual Top-K selections revealed from decoding token t-1.\n"
        "3. Uniform Temporal Score: Every candidate in HT receives an implicit confidence bonus of +2.\n"
        "4. Cold-Start Handling: At token t=0, HT[request_id] is empty -> temporal bonus is 0 for all candidates."
    )
    ax.text(0.10, 0.26, desc, fontsize=9.5, va='top')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "vis2_st_moe_ht.png"), dpi=200)
    plt.close()


def plot_vis3_cstp_spatial():
    """Visualization 3: CSTP Spatial Transition Tensor & Dirichlet Conditional Probability."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Illustrative 8x8 spatial co-activation count matrix for Stage 0 -> Stage 1
    np.random.seed(42)
    spatial_counts = np.array([
        [0, 8, 5, 0, 1, 0, 0, 0],
        [0, 0, 9, 0, 6, 0, 1, 0],
        [1, 0, 0, 4, 0, 7, 0, 0],
        [0, 0, 0, 0, 5, 0, 3, 1],
        [0, 2, 0, 0, 0, 6, 0, 4],
        [3, 0, 1, 0, 0, 0, 5, 0],
        [0, 4, 0, 2, 0, 0, 0, 7],
        [1, 0, 3, 0, 2, 0, 0, 0]
    ], dtype=np.float64)

    # Compute smoothed conditional probability with prior
    prior = np.ones(8) / 8.0
    smoothing_lambda = 1.0
    p_cond = np.zeros((8, 8))
    for u in range(8):
        row = spatial_counts[u]
        total = np.sum(row)
        p_cond[u] = (row + (smoothing_lambda * prior)) / (total + smoothing_lambda)

    # Plot counts
    im1 = ax1.imshow(spatial_counts, cmap='Blues')
    ax1.set_title("CSTP Raw Co-activation Counts\nself.spatial_counts[stage=0, u, v]", fontsize=11, weight='bold')
    ax1.set_xlabel("Target Expert v at Stage 1 (0..7)", weight='bold')
    ax1.set_ylabel("Source Expert u at Stage 0 (0..7)", weight='bold')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Plot smoothed probabilities
    im2 = ax2.imshow(p_cond, cmap='YlGnBu')
    ax2.set_title("Dirichlet-Smoothed Spatial Score\nS_spatial(v | u) with lambda=1.0", fontsize=11, weight='bold')
    ax2.set_xlabel("Target Expert v at Stage 1 (0..7)", weight='bold')
    ax2.set_ylabel("Source Expert u at Stage 0 (0..7)", weight='bold')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle("Visualization 3: CSTP Spatial Transition Modeling (3D Tensor Slice)", fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "vis3_cstp_spatial.png"), dpi=200)
    plt.close()


def plot_vis4_cstp_temporal():
    """Visualization 4: CSTP Temporal Transition Tensor across Decoding Steps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Illustrative 8x8 temporal transition count matrix for Stage 1 (token t-1 -> token t)
    temporal_counts = np.array([
        [4, 1, 0, 2, 0, 0, 0, 1],
        [0, 5, 1, 0, 0, 1, 0, 0],
        [1, 0, 6, 2, 0, 4, 0, 0],
        [0, 0, 1, 7, 0, 0, 2, 0],
        [0, 0, 0, 0, 5, 1, 0, 3],
        [1, 2, 0, 0, 0, 6, 1, 0],
        [0, 0, 3, 0, 1, 0, 4, 1],
        [0, 1, 0, 0, 2, 0, 0, 5]
    ], dtype=np.float64)

    prior = np.ones(8) / 8.0
    smoothing_lambda = 1.0
    p_trans = np.zeros((8, 8))
    for w in range(8):
        row = temporal_counts[w]
        total = np.sum(row)
        p_trans[w] = (row + (smoothing_lambda * prior)) / (total + smoothing_lambda)

    im1 = ax1.imshow(temporal_counts, cmap='Purples')
    ax1.set_title("CSTP Temporal Transition Counts\nself.temporal_counts[stage=1, w, v]", fontsize=11, weight='bold')
    ax1.set_xlabel("Next Token Expert v at Token t (0..7)", weight='bold')
    ax1.set_ylabel("Prev Token Expert w at Token t-1 (0..7)", weight='bold')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(p_trans, cmap='Oranges')
    ax2.set_title("Smoothed Temporal Transition Prob\nS_temporal(v | w) with lambda=1.0", fontsize=11, weight='bold')
    ax2.set_xlabel("Next Token Expert v at Token t (0..7)", weight='bold')
    ax2.set_ylabel("Prev Token Expert w at Token t-1 (0..7)", weight='bold')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle("Visualization 4: CSTP Temporal Transition Dynamics (Cross-Token)", fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "vis4_cstp_temporal.png"), dpi=200)
    plt.close()


def plot_vis5_cstp_marginal():
    """Visualization 5: CSTP Target-Stage Marginal Frequency Prior."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    experts = np.arange(8)
    # Illustrative counts showing realistic popularity skew at Stage 1
    raw_counts = np.array([4, 12, 28, 9, 3, 18, 5, 11], dtype=np.float64)

    epsilon = 0.1
    denom = np.sum(raw_counts) + (len(experts) * epsilon)
    prior_prob = (raw_counts + epsilon) / denom

    ax1.bar(experts, raw_counts, color='#3b82f6', width=0.55, edgecolor='#1d4ed8')
    ax1.set_title("Stage 1 Marginal Selections\nself.stage_marginal_counts[stage=1]", fontsize=11, weight='bold')
    ax1.set_xlabel("Expert ID (0..7)", weight='bold')
    ax1.set_ylabel("Total Observation Count", weight='bold')
    ax1.set_xticks(experts)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    ax2.bar(experts, prior_prob, color='#10b981', width=0.55, edgecolor='#047857')
    ax2.set_title("Dirichlet-Smoothed Marginal Prior\nP_prior(v | stage=1) with epsilon=0.1", fontsize=11, weight='bold')
    ax2.set_xlabel("Expert ID (0..7)", weight='bold')
    ax2.set_ylabel("Prior Probability", weight='bold')
    ax2.set_xticks(experts)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    fig.suptitle("Visualization 5: CSTP Stage Marginal Prior Distribution", fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "vis5_cstp_marginal.png"), dpi=200)
    plt.close()


def plot_vis6_prediction_pipeline():
    """Visualization 6: End-to-End Prediction Scoring on the Running Example."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    experts = np.arange(8)

    # ST-MoE Running Example intermediate scores
    # Sources = [0, 1]. CCT[0]={1:3, 2:2}, CCT[1]={2:3, 4:2}
    # HT = {2, 3} (+2 each)
    st_moe_spatial = np.array([0, 3, 5, 0, 2, 0, 0, 0])
    st_moe_temporal = np.array([0, 0, 2, 2, 0, 0, 0, 0])
    st_moe_total = st_moe_spatial + st_moe_temporal

    # Plot ST-MoE stacked bars
    ax1.bar(experts, st_moe_spatial, label='Spatial (CCT)', color='#3b82f6', width=0.55)
    ax1.bar(experts, st_moe_temporal, bottom=st_moe_spatial, label='Temporal (HT +2)', color='#f59e0b', width=0.55)
    ax1.axhline(2.0, color='#dc2626', linestyle='--', lw=1.5, label='Threshold (>= 2)')
    ax1.set_title("ST-MoE Paper: Alg 2 & Eq 1 Scoring\nThreshold Set: {1, 2, 3, 4} | Budget B=3: [2, 1, 3]",
                  fontsize=11, weight='bold')
    ax1.set_xlabel("Candidate Expert ID (0..7)", weight='bold')
    ax1.set_ylabel("Integer Confidence Score", weight='bold')
    ax1.set_xticks(experts)
    ax1.set_ylim(0, 9)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # CSTP Running Example continuous scores
    # alpha = 0.5. Spatial from u in {0, 1}. Temporal from w in {2, 3}.
    # Illustrative smoothed scores
    cstp_spatial = np.array([0.02, 0.38, 0.45, 0.01, 0.12, 0.01, 0.01, 0.00])
    cstp_temporal = np.array([0.03, 0.05, 0.35, 0.32, 0.02, 0.18, 0.04, 0.01])
    cstp_composite = (0.5 * cstp_spatial) + (0.5 * cstp_temporal)

    w_bar = 0.25
    ax2.bar(experts - w_bar/2, cstp_spatial, width=w_bar, label='S_spatial(v)', color='#60a5fa')
    ax2.bar(experts + w_bar/2, cstp_temporal, width=w_bar, label='S_temporal(v)', color='#fbbf24')
    ax2.plot(experts, cstp_composite, color='#dc2626', marker='o', lw=2, label='Composite Score (alpha=0.5)')
    ax2.set_title("CSTP: Dirichlet Composite Scoring\nBudget B=3: [2, 3, 1]",
                  fontsize=11, weight='bold')
    ax2.set_xlabel("Candidate Expert ID (0..7)", weight='bold')
    ax2.set_ylabel("Continuous Probability Score", weight='bold')
    ax2.set_xticks(experts)
    ax2.set_ylim(0, 0.5)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    fig.suptitle("Visualization 6: Prediction Scoring Comparison on Identical Inputs (Experts 0..7, B=3)",
                 fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "vis6_prediction_pipeline.png"), dpi=200)
    plt.close()


def plot_vis7_update_lifecycle():
    """Visualization 7: Side-by-side Table Update Lifecycle Diagram."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis('off')

    ax.text(0.5, 0.95, "Visualization 7: State Update Lifecycle when Actual Routing F_j = {2, 5} Revealed",
            fontsize=13, weight='bold', ha='center')

    # ST-MoE Update Box
    rect1 = patches.FancyBboxPatch((0.05, 0.20), 0.42, 0.68, boxstyle="round,pad=0.03",
                                   facecolor='#f0fdf4', edgecolor='#16a34a', lw=1.5)
    ax.add_patch(rect1)
    ax.text(0.26, 0.84, "ST-MoE Online Update (Algorithm 3)", fontsize=11, weight='bold', color='#15803d', ha='center')

    st_moe_steps = (
        "1. CCT Counter Update for source u in {0, 1}:\n"
        "   • Candidate 2 matches actual {2, 5}:\n"
        "     CCT[0][2]: 2 -> 3 (increment)\n"
        "     CCT[1][2]: 3 -> 3 (saturated max)\n"
        "   • Candidates 1 & 4 mismatch {2, 5}:\n"
        "     CCT[0][1]: 3 -> 2 (decrement)\n"
        "     CCT[1][4]: 2 -> 1 (decrement)\n\n"
        "2. Candidate Replacement:\n"
        "   • Neither candidate reached 0 -> No replacement.\n"
        "   • If score had been 0, slot would be replaced by\n"
        "     un-stored expert 5, initialized to 2 (10).\n\n"
        "3. History Table (HT) Overwrite:\n"
        "   • HT[request_id][stage=1] <- [2, 5]"
    )
    ax.text(0.08, 0.78, st_moe_steps, fontsize=9.2, va='top')

    # CSTP Update Box
    rect2 = patches.FancyBboxPatch((0.53, 0.20), 0.42, 0.68, boxstyle="round,pad=0.03",
                                   facecolor='#eff6ff', edgecolor='#2563eb', lw=1.5)
    ax.add_patch(rect2)
    ax.text(0.74, 0.84, "CSTP Update Lifecycle", fontsize=11, weight='bold', color='#1d4ed8', ha='center')

    cstp_steps = (
        "1. History Table (HT) Update (Always Active):\n"
        "   • HT[request_id][stage=1] <- {2, 5}\n"
        "   • Ready for token t+1 temporal conditioning.\n\n"
        "2. Online Adaptation (if online_adaptation=True):\n"
        "   • Marginal Counts: self.stage_marginal_counts[1, 2] += 1\n"
        "                    self.stage_marginal_counts[1, 5] += 1\n"
        "   • Spatial Co-activations (Stage 0 -> Stage 1):\n"
        "     For u in {0, 1} and v in {2, 5}:\n"
        "     self.spatial_counts[0, u, v] += 1.0\n"
        "   • Temporal Transitions (Token t-1 -> Token t):\n"
        "     For w in prev {2, 3} and v in curr {2, 5}:\n"
        "     self.temporal_counts[1, w, v] += 1.0\n\n"
        "3. Offline Profiling (Default Mode):\n"
        "   • Transition tensors remain fixed during testing."
    )
    ax.text(0.56, 0.78, cstp_steps, fontsize=9.2, va='top')

    # Footer note
    ax.text(0.5, 0.08, "Both predictors execute updates strictly AFTER prediction metrics are logged (Causal Sequencing).",
            fontsize=10, weight='bold', ha='center', color='#1f2937',
            bbox=dict(facecolor='#f3f4f6', edgecolor='#9ca3af', boxstyle='round,pad=0.4'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "vis7_update_lifecycle.png"), dpi=200)
    plt.close()


def main():
    print("Generating Visualizations 1 through 7...")
    plot_vis1_st_moe_cct()
    plot_vis2_st_moe_ht()
    plot_vis3_cstp_spatial()
    plot_vis4_cstp_temporal()
    plot_vis5_cstp_marginal()
    plot_vis6_prediction_pipeline()
    plot_vis7_update_lifecycle()
    print("All 7 visualizations successfully generated in:", OUT_DIR)


if __name__ == "__main__":
    main()
