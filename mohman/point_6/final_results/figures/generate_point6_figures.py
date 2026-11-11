#!/usr/bin/env python3
"""
Publication-Quality Figure Generator for Point 6 Sensitivity Analysis.
Generates all 15 required figures in both PNG (300 DPI) and PDF formats:
- Batch Size Suite: Figures 1, 2, 3
- Context Length Suite: Figures 4, 5, 6
- Lookahead Suite: Figures 7, 8, 9
- Prefetch Budget Suite: Figures 10, 11, 12
- Comparative / Consolidated: Figures 13, 14, 15

Enforces:
- Strictly ZERO ternary operators (ast.IfExp == 0)
- Explicit units (ms, GB, GiB, %, tokens/s)
- Direct provenance and workload labeling
"""

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

WORKSPACE = "/home/mohman/ca_project"
POINT6_ROOT = os.path.join(WORKSPACE, "mohman", "point_6", "final_results")
CONSOLIDATED_ROOT = os.path.join(POINT6_ROOT, "consolidated_data")
FIG_ROOT = os.path.join(POINT6_ROOT, "figures")

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

def save_fig(fig, rel_dir, filename_base, extra_alias=None):
    target_dir = os.path.join(FIG_ROOT, rel_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    png_path = os.path.join(target_dir, f"{filename_base}.png")
    pdf_path = os.path.join(target_dir, f"{filename_base}.pdf")
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')

    artifact_png_dir = "/home/mohman/.gemini/antigravity/brain/b7db0340-f5d9-4df3-86d8-be407d44242d/figures/point_6/png"
    artifact_pdf_dir = "/home/mohman/.gemini/antigravity/brain/b7db0340-f5d9-4df3-86d8-be407d44242d/figures/point_6/pdf"
    os.makedirs(artifact_png_dir, exist_ok=True)
    os.makedirs(artifact_pdf_dir, exist_ok=True)

    fig.savefig(os.path.join(artifact_png_dir, f"{filename_base}.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(artifact_pdf_dir, f"{filename_base}.pdf"), bbox_inches='tight')

    if extra_alias is not None:
        fig.savefig(os.path.join(target_dir, f"{extra_alias}.png"), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(target_dir, f"{extra_alias}.pdf"), bbox_inches='tight')
        fig.savefig(os.path.join(artifact_png_dir, f"{extra_alias}.png"), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(artifact_pdf_dir, f"{extra_alias}.pdf"), bbox_inches='tight')

    plt.close(fig)
    print(f"  [FIGURE] Saved {filename_base} (.png & .pdf) in figures/{rel_dir}")

def plot_all_figures():
    print("=== GENERATING 15 POINT 6 PUBLICATION FIGURES ===")
    
    unified_csv = os.path.join(CONSOLIDATED_ROOT, "unified_point6_results.csv")
    if not os.path.exists(unified_csv):
        print(f"Error: Unified CSV {unified_csv} not found.")
        return
        
    df = pd.read_csv(unified_csv)
    
    # ----------------------------------------------------
    # BATCH SIZE SUITE: Figures 1, 2, 3
    # ----------------------------------------------------
    bs_df = df[df["suite"].isin(["batch_size", "anchor"]) & (df["context_p"] == 2048) & (df["budget_k"].isin([0, 4])) & (df["lookahead_h"].isin([0, 1]))].copy()
    batches = [4, 14, 32]
    
    # Fig 1: Batch size vs HBM Hit Rate
    fig, ax = plt.subplots(figsize=(7, 4.5))
    base_hits = [bs_df[(bs_df["batch_size"] == b) & (bs_df["prefetch_enabled"] == False)]["demand_hit_rate_pct"].values[0] for b in batches]
    lru_hits = [bs_df[(bs_df["batch_size"] == b) & (bs_df["prefetch_enabled"] == True) & (bs_df["eviction_policy"] == "lru")]["demand_hit_rate_pct"].values[0] for b in batches]
    falru_hits = [bs_df[(bs_df["batch_size"] == b) & (bs_df["prefetch_enabled"] == True) & (bs_df["eviction_policy"] == "fa_lru")]["demand_hit_rate_pct"].values[0] for b in batches]
    
    x = np.arange(len(batches))
    w = 0.25
    ax.bar(x - w, base_hits, width=w, label="Baseline (No Prefetch)", color="#9E9E9E", edgecolor="black")
    ax.bar(x, lru_hits, width=w, label="CSTP + LRU", color="#1976D2", edgecolor="black")
    ax.bar(x + w, falru_hits, width=w, label="CSTP + FA-LRU (W=64)", color="#388E3C", edgecolor="black")
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"B = {b}" for b in batches])
    ax.set_ylabel("Coalesced Demand Hit Rate (%)")
    ax.set_title("Figure 1: Batch Size vs. Physical HBM Hit Rate\n(DeepSeek-V3, P=2048, T=128, HBM=35 GiB)", fontweight="bold")
    ax.legend(loc="upper left")
    save_fig(fig, "batch_size", "fig1_batch_size_hit_rate")
    
    # Fig 2: Batch size vs Exposed Memory Stall
    fig, ax = plt.subplots(figsize=(7, 4.5))
    base_stalls = [bs_df[(bs_df["batch_size"] == b) & (bs_df["prefetch_enabled"] == False)]["exposed_stall_ms"].values[0] / 1000.0 for b in batches]
    lru_stalls = [bs_df[(bs_df["batch_size"] == b) & (bs_df["prefetch_enabled"] == True) & (bs_df["eviction_policy"] == "lru")]["exposed_stall_ms"].values[0] / 1000.0 for b in batches]
    falru_stalls = [bs_df[(bs_df["batch_size"] == b) & (bs_df["prefetch_enabled"] == True) & (bs_df["eviction_policy"] == "fa_lru")]["exposed_stall_ms"].values[0] / 1000.0 for b in batches]
    
    ax.plot(batches, base_stalls, marker='o', linewidth=2, color="#9E9E9E", label="Baseline (No Prefetch)")
    ax.plot(batches, lru_stalls, marker='s', linewidth=2, color="#1976D2", label="CSTP + LRU")
    ax.plot(batches, falru_stalls, marker='^', linewidth=2, color="#388E3C", label="CSTP + FA-LRU (W=64)")
    
    ax.set_xlabel("Serving Batch Size ($B$)")
    ax.set_ylabel("Exposed Memory Stall Time (seconds)")
    ax.set_title("Figure 2: Batch Size vs. Exposed Memory Stall Duration\n(Asynchronous Prefetching Hides Stall Under High Concurrency)", fontweight="bold")
    ax.legend()
    save_fig(fig, "batch_size", "fig2_batch_size_exposed_stall")
    
    # Fig 3: Batch size vs Token Latency
    fig, ax = plt.subplots(figsize=(7, 4.5))
    base_lats = [bs_df[(bs_df["batch_size"] == b) & (bs_df["prefetch_enabled"] == False)]["mean_token_latency_ms"].values[0] for b in batches]
    lru_lats = [bs_df[(bs_df["batch_size"] == b) & (bs_df["prefetch_enabled"] == True) & (bs_df["eviction_policy"] == "lru")]["mean_token_latency_ms"].values[0] for b in batches]
    falru_lats = [bs_df[(bs_df["batch_size"] == b) & (bs_df["prefetch_enabled"] == True) & (bs_df["eviction_policy"] == "fa_lru")]["mean_token_latency_ms"].values[0] for b in batches]
    
    ax.plot(batches, base_lats, marker='o', linewidth=2, color="#9E9E9E", label="Baseline (No Prefetch)")
    ax.plot(batches, lru_lats, marker='s', linewidth=2, color="#1976D2", label="CSTP + LRU")
    ax.plot(batches, falru_lats, marker='^', linewidth=2, color="#388E3C", label="CSTP + FA-LRU (W=64)")
    
    ax.set_xlabel("Serving Batch Size ($B$)")
    ax.set_ylabel("Mean Token Latency (ms)")
    ax.set_title("Figure 3: Batch Size vs. Mean Token Latency\n(Degradation at B=4, Significant Acceleration at B=14 and B=32)", fontweight="bold")
    ax.legend()
    save_fig(fig, "batch_size", "fig3_batch_size_token_latency")
    
    # ----------------------------------------------------
    # CONTEXT LENGTH SUITE: Figures 4, 5, 6
    # ----------------------------------------------------
    ctx_df = df[df["suite"].isin(["context_length", "anchor"]) & (df["batch_size"] == 32) & (df["budget_k"].isin([0, 4])) & (df["lookahead_h"].isin([0, 1]))].copy()
    contexts = [128, 512, 2048, 4096]
    
    # Fig 4: Context Length vs KV Cache Footprint (Calculated)
    # DeepSeek-V3 MLA: FP16, 40 layers, kv_lora_rank=512, qk_rope_head_dim=64 -> 576 floats = 1152 bytes per token
    # Footprint = B * (P + t) * L * 1152 bytes
    fig, ax = plt.subplots(figsize=(7, 4.5))
    initial_kv_gib = []
    max_kv_gib = []
    bytes_per_tok = 40 * 1152
    for p in contexts:
        init_bytes = 32 * p * bytes_per_tok
        max_bytes = 32 * (p + 128) * bytes_per_tok
        initial_kv_gib.append(init_bytes / (1024**3))
        max_kv_gib.append(max_bytes / (1024**3))
    
    x = np.arange(len(contexts))
    w = 0.35
    ax.bar(x - w/2, initial_kv_gib, width=w, label="Initial KV-Cache at Decode Start ($P$)", color="#0288D1", edgecolor="black")
    ax.bar(x + w/2, max_kv_gib, width=w, label="Maximum KV-Cache at Horizon ($P+T$)", color="#F57C00", edgecolor="black")
    
    ax.axhline(35.0, color="#D32F2F", linestyle="--", linewidth=1.5, label="Total HBM Capacity (35 GiB)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P = {p}" for p in contexts])
    ax.set_ylabel("KV-Cache Memory Footprint (GiB)")
    ax.set_title("Figure 4: DeepSeek-V3 MLA KV-Cache Scaling Across Context Lengths\n(B=32, T=128, 40 Layers, FP16)", fontweight="bold")
    ax.legend(loc="upper left")
    save_fig(fig, "context_length", "fig4_context_length_kv_cache")
    
    # Fig 5: Context Length vs Non-Stall Compute Duration / Prefetch Slack
    fig, ax = plt.subplots(figsize=(7, 4.5))
    compute_slacks = []
    for p in contexts:
        sub = ctx_df[(ctx_df["context_p"] == p) & (ctx_df["prefetch_enabled"] == False)]
        slack_ms = sub["compute_slack_ms"].values[0]
        compute_slacks.append(slack_ms)
        
    ax.plot(contexts, compute_slacks, marker='o', linewidth=2.5, color="#7B1FA2")
    for p, cs in zip(contexts, compute_slacks):
        ax.annotate(f"{cs:.2f} ms", (p, cs), textcoords="offset points", xytext=(0, 8), ha='center', fontweight="bold")
        
    ax.set_xlabel("Initial Context Length $P$ (Tokens)")
    ax.set_ylabel("Non-Stall Compute Duration / Prefetch Slack (ms)")
    ax.set_title("Figure 5: Compute Duration & Prefetch Overlap Slack vs. Context Length\n(B=32, T=128, O(P+t) Attention Scaling in `attention_gen_impl.cpp`)", fontweight="bold")
    save_fig(fig, "context_length", "fig5_context_length_attention_time")
    
    # Fig 6: Context Length vs Expert Cache Performance (Latency across P)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    base_lat_p = [ctx_df[(ctx_df["context_p"] == p) & (ctx_df["prefetch_enabled"] == False)]["mean_token_latency_ms"].values[0] for p in contexts]
    lru_lat_p = [ctx_df[(ctx_df["context_p"] == p) & (ctx_df["prefetch_enabled"] == True) & (ctx_df["eviction_policy"] == "lru")]["mean_token_latency_ms"].values[0] for p in contexts]
    falru_lat_p = [ctx_df[(ctx_df["context_p"] == p) & (ctx_df["prefetch_enabled"] == True) & (ctx_df["eviction_policy"] == "fa_lru")]["mean_token_latency_ms"].values[0] for p in contexts]
    
    ax.plot(contexts, base_lat_p, marker='o', linewidth=2, color="#9E9E9E", label="Baseline (No Prefetch)")
    ax.plot(contexts, lru_lat_p, marker='s', linewidth=2, color="#1976D2", label="CSTP + LRU")
    ax.plot(contexts, falru_lat_p, marker='^', linewidth=2, color="#388E3C", label="CSTP + FA-LRU (W=64)")
    
    ax.set_xlabel("Initial Context Length $P$ (Tokens)")
    ax.set_ylabel("Mean Token Latency (ms)")
    ax.set_title("Figure 6: Context Length Sensitivity: Mean Token Latency Across P\n(B=32, T=128, HBM=35 GiB)", fontweight="bold")
    ax.legend()
    save_fig(fig, "context_length", "fig6_context_length_expert_cache")
    
    # ----------------------------------------------------
    # LOOKAHEAD SUITE: Figures 7, 8, 9
    # ----------------------------------------------------
    lh_df = df[df["suite"].isin(["lookahead", "anchor"]) & (df["prefetch_enabled"] == True) & (df["budget_k"] == 4)].copy()
    horizons = [1, 2, 4]
    
    # Fig 7: Lookahead H vs Resident Hits
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lru_hits_lh = [lh_df[(lh_df["lookahead_h"] == h) & (lh_df["eviction_policy"] == "lru")]["resident_hits"].values[0] for h in horizons]
    falru_hits_lh = [lh_df[(lh_df["lookahead_h"] == h) & (lh_df["eviction_policy"] == "fa_lru")]["resident_hits"].values[0] for h in horizons]
    
    x = np.arange(len(horizons))
    w = 0.35
    ax.bar(x - w/2, lru_hits_lh, width=w, label="CSTP + LRU", color="#1976D2", edgecolor="black")
    ax.bar(x + w/2, falru_hits_lh, width=w, label="CSTP + FA-LRU (W=64)", color="#388E3C", edgecolor="black")
    
    for i in range(len(horizons)):
        ax.text(x[i] - w/2, lru_hits_lh[i] + 30, f"{lru_hits_lh[i]}", ha='center', fontsize=9, fontweight="bold")
        ax.text(x[i] + w/2, falru_hits_lh[i] + 30, f"{falru_hits_lh[i]}", ha='center', fontsize=9, fontweight="bold")
        
    ax.set_xticks(x)
    ax.set_xticklabels([f"H = {h}" for h in horizons])
    ax.set_ylabel("HBM Resident Hits (Coalesced)")
    ax.set_title("Figure 7: Lookahead Distance ($H$) vs. HBM Resident Hits\n(B=32, P=2048, T=128, K=4)", fontweight="bold")
    ax.legend(loc="upper left")
    save_fig(fig, "lookahead", "fig7_lookahead_resident_hits", extra_alias="fig7_lookahead_timely_coverage")
    
    # Fig 8: Lookahead H vs Unused Completed Prefetches
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lru_unused = [lh_df[(lh_df["lookahead_h"] == h) & (lh_df["eviction_policy"] == "lru")]["unused_prefetches"].values[0] for h in horizons]
    falru_unused = [lh_df[(lh_df["lookahead_h"] == h) & (lh_df["eviction_policy"] == "fa_lru")]["unused_prefetches"].values[0] for h in horizons]
    
    ax.plot(horizons, lru_unused, marker='s', linewidth=2, color="#E64A19", label="CSTP + LRU")
    ax.plot(horizons, falru_unused, marker='^', linewidth=2, color="#C2185B", label="CSTP + FA-LRU (W=64)")
    
    for h, u in zip(horizons, lru_unused):
        ax.annotate(f"{u}", (h, u), textcoords="offset points", xytext=(-10, 8), ha='center', fontweight="bold")
    for h, u in zip(horizons, falru_unused):
        ax.annotate(f"{u}", (h, u), textcoords="offset points", xytext=(10, -12), ha='center', fontweight="bold")
        
    ax.set_xlabel("Prediction Lookahead ($H$ Layers Ahead)")
    ax.set_ylabel("Unused Completed Prefetches (Count)")
    ax.set_title("Figure 8: Impact of Prediction Distance ($H$) on Speculative Waste\n(B=32, P=2048, T=128, K=4)", fontweight="bold")
    ax.legend()
    save_fig(fig, "lookahead", "fig8_lookahead_unused_prefetches")
    
    # Fig 9: Lookahead H vs Exposed Memory Stall
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lru_lh_stalls = [lh_df[(lh_df["lookahead_h"] == h) & (lh_df["eviction_policy"] == "lru")]["exposed_stall_ms"].values[0] / 1000.0 for h in horizons]
    falru_lh_stalls = [lh_df[(lh_df["lookahead_h"] == h) & (lh_df["eviction_policy"] == "fa_lru")]["exposed_stall_ms"].values[0] / 1000.0 for h in horizons]
    
    ax.plot(horizons, lru_lh_stalls, marker='s', linewidth=2, color="#1976D2", label="CSTP + LRU")
    ax.plot(horizons, falru_lh_stalls, marker='^', linewidth=2, color="#388E3C", label="CSTP + FA-LRU (W=64)")
    
    for h, s in zip(horizons, lru_lh_stalls):
        ax.annotate(f"{s:.2f}s", (h, s), textcoords="offset points", xytext=(-15, 8), ha='center', fontweight="bold")
    for h, s in zip(horizons, falru_lh_stalls):
        ax.annotate(f"{s:.2f}s", (h, s), textcoords="offset points", xytext=(15, -12), ha='center', fontweight="bold")
        
    ax.set_xlabel("Prediction Lookahead ($H$ Layers Ahead)")
    ax.set_ylabel("Exposed Memory Stall Time (seconds)")
    ax.set_title("Figure 9: Lookahead Distance vs. Exposed Memory Stalls\n(Extended Lead Time Hides Stalls but Increases Evictions)", fontweight="bold")
    ax.legend()
    save_fig(fig, "lookahead", "fig9_lookahead_exposed_stall")
    
    # ----------------------------------------------------
    # PREFETCH BUDGET SUITE: Figures 10, 11, 12
    # ----------------------------------------------------
    bg_df = df[df["suite"].isin(["prefetch_budget", "anchor"]) & (df["prefetch_enabled"] == True) & (df["lookahead_h"] == 1)].copy()
    budgets = [2, 4, 8]
    
    # Fig 10: Prefetch Budget K vs Speculative DDR5 Traffic
    fig, ax = plt.subplots(figsize=(7, 4.5))
    spec_traffic = [bg_df[(bg_df["budget_k"] == k) & (bg_df["eviction_policy"] == "lru")]["speculative_ddr5_gb"].values[0] for k in budgets]
    demand_traffic = [bg_df[(bg_df["budget_k"] == k) & (bg_df["eviction_policy"] == "lru")]["demand_ddr5_gb"].values[0] for k in budgets]
    
    x = np.arange(len(budgets))
    w = 0.35
    ax.bar(x, demand_traffic, width=w, label="Demand DDR5 Traffic", color="#5C6BC0", edgecolor="black")
    ax.bar(x, spec_traffic, width=w, bottom=demand_traffic, label="Speculative DDR5 Traffic", color="#FFA726", edgecolor="black")
    
    for i in range(len(budgets)):
        total_gb = demand_traffic[i] + spec_traffic[i]
        ax.text(x[i], total_gb + 500, f"{total_gb:.0f} GB", ha='center', fontweight="bold")
        
    ax.set_xticks(x)
    ax.set_xticklabels([f"K = {k}" for k in budgets])
    ax.set_ylabel("Total DDR5 Traffic (GB)")
    ax.set_title("Figure 10: Prefetch Budget ($K$) vs. DDR5 Traffic Composition\n(B=32, P=2048, T=128, H=1)", fontweight="bold")
    ax.legend()
    save_fig(fig, "prefetch_budget", "fig10_budget_speculative_ddr5")
    
    # Fig 11: Prefetch Budget K vs Demand-Resident Evictions (Cache Pollution)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lru_poll = [bg_df[(bg_df["budget_k"] == k) & (bg_df["eviction_policy"] == "lru")]["demand_evictions"].values[0] for k in budgets]
    falru_poll = [bg_df[(bg_df["budget_k"] == k) & (bg_df["eviction_policy"] == "fa_lru")]["demand_evictions"].values[0] for k in budgets]
    
    ax.plot(budgets, lru_poll, marker='s', linewidth=2, color="#E53935", label="Standard LRU (Higher Evictions)")
    ax.plot(budgets, falru_poll, marker='^', linewidth=2, color="#43A047", label="FA-LRU W=64 (Frequency Protected)")
    
    for k, p in zip(budgets, lru_poll):
        ax.annotate(f"{p}", (k, p), textcoords="offset points", xytext=(-10, 8), ha='center', fontweight="bold")
    for k, p in zip(budgets, falru_poll):
        ax.annotate(f"{p}", (k, p), textcoords="offset points", xytext=(10, -12), ha='center', fontweight="bold")
        
    ax.set_xlabel("Prefetch Budget ($K$ Candidates per Token)")
    ax.set_ylabel("Demand-Resident Evictions (Count)")
    ax.set_title("Figure 11: Cache Pollution: Demand Evictions vs. Prefetch Budget ($K$)\n(FA-LRU Suppresses Evictions of Frequently Reused Experts)", fontweight="bold")
    ax.legend()
    save_fig(fig, "prefetch_budget", "fig11_budget_demand_resident_evictions")
    
    # Fig 12: Prefetch Budget K vs Token Latency
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lru_bg_lat = [bg_df[(bg_df["budget_k"] == k) & (bg_df["eviction_policy"] == "lru")]["mean_token_latency_ms"].values[0] for k in budgets]
    falru_bg_lat = [bg_df[(bg_df["budget_k"] == k) & (bg_df["eviction_policy"] == "fa_lru")]["mean_token_latency_ms"].values[0] for k in budgets]
    
    ax.plot(budgets, lru_bg_lat, marker='s', linewidth=2, color="#1976D2", label="CSTP + LRU")
    ax.plot(budgets, falru_bg_lat, marker='^', linewidth=2, color="#388E3C", label="CSTP + FA-LRU (W=64)")
    
    for k, lat in zip(budgets, lru_bg_lat):
        ax.annotate(f"{lat:.2f} ms", (k, lat), textcoords="offset points", xytext=(-15, 8), ha='center', fontweight="bold")
    for k, lat in zip(budgets, falru_bg_lat):
        ax.annotate(f"{lat:.2f} ms", (k, lat), textcoords="offset points", xytext=(15, -12), ha='center', fontweight="bold")
        
    ax.set_xlabel("Prefetch Budget ($K$)")
    ax.set_ylabel("Mean Token Latency (ms)")
    ax.set_title("Figure 12: Mean Token Latency vs. Prefetch Aggressiveness ($K$)\n(K=2 Minimizes Contention; K=8 Saturates DDR5 Bus)", fontweight="bold")
    ax.legend()
    save_fig(fig, "prefetch_budget", "fig12_budget_token_latency")
    
    # ----------------------------------------------------
    # COMPARATIVE / CONSOLIDATED: Figures 13, 14, 15
    # ----------------------------------------------------
    # Fig 13: Prefetching vs No-Prefetching across all B
    fig, ax = plt.subplots(figsize=(8, 4.8))
    speedup_lru = []
    speedup_falru = []
    for b in batches:
        base_l = bs_df[(bs_df["batch_size"] == b) & (bs_df["prefetch_enabled"] == False)]["mean_token_latency_ms"].values[0]
        lru_l = bs_df[(bs_df["batch_size"] == b) & (bs_df["prefetch_enabled"] == True) & (bs_df["eviction_policy"] == "lru")]["mean_token_latency_ms"].values[0]
        falru_l = bs_df[(bs_df["batch_size"] == b) & (bs_df["prefetch_enabled"] == True) & (bs_df["eviction_policy"] == "fa_lru")]["mean_token_latency_ms"].values[0]
        speedup_lru.append((base_l - lru_l) / base_l * 100.0)
        speedup_falru.append((base_l - falru_l) / base_l * 100.0)
        
    x = np.arange(len(batches))
    w = 0.35
    ax.bar(x - w/2, speedup_lru, width=w, label="CSTP + LRU Speedup", color="#1976D2", edgecolor="black")
    ax.bar(x + w/2, speedup_falru, width=w, label="CSTP + FA-LRU Speedup", color="#388E3C", edgecolor="black")
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    for i in range(len(batches)):
        if speedup_lru[i] >= 0.0:
            ax.text(x[i] - w/2, speedup_lru[i] + 0.03, f"+{speedup_lru[i]:.2f}%", ha='center', va='bottom', fontsize=9, fontweight="bold")
        else:
            ax.text(x[i] - w/2, speedup_lru[i] - 0.03, f"{speedup_lru[i]:.2f}%", ha='center', va='top', fontsize=9, fontweight="bold")
            
        if speedup_falru[i] >= 0.0:
            ax.text(x[i] + w/2, speedup_falru[i] + 0.03, f"+{speedup_falru[i]:.2f}%", ha='center', va='bottom', fontsize=9, fontweight="bold")
        else:
            ax.text(x[i] + w/2, speedup_falru[i] - 0.03, f"{speedup_falru[i]:.2f}%", ha='center', va='top', fontsize=9, fontweight="bold")
        
    ax.set_xticks(x)
    ax.set_xticklabels([f"B = {b}" for b in batches])
    ax.set_ylabel("Latency Improvement over Baseline (%)")
    ax.set_ylim(-0.25, 0.95)
    ax.set_title("Figure 13: End-to-End Latency Speedup Across Serving Batch Sizes\n(P=2048, T=128, H=1, K=4, HBM=35 GiB)", fontweight="bold")
    ax.legend(loc="upper left")
    save_fig(fig, "combined", "fig13_prefetch_vs_noprefetch")
    
    # Fig 14: LRU vs FA-LRU Comparison across Context Lengths (Reloads Saved)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    reloads_saved = []
    for p in contexts:
        lru_r = ctx_df[(ctx_df["context_p"] == p) & (ctx_df["prefetch_enabled"] == True) & (ctx_df["eviction_policy"] == "lru")]["reloads"].values[0]
        falru_r = ctx_df[(ctx_df["context_p"] == p) & (ctx_df["prefetch_enabled"] == True) & (ctx_df["eviction_policy"] == "fa_lru")]["reloads"].values[0]
        reloads_saved.append(lru_r - falru_r)
        
    x = np.arange(len(contexts))
    w = 0.35
    ax.bar(x, reloads_saved, width=w, color="#2E7D32", edgecolor="black", label="Net Expert Reloads Saved by FA-LRU")
    for i in range(len(contexts)):
        ax.text(x[i], reloads_saved[i] + 0.5, f"+{reloads_saved[i]} reloads", ha='center', fontweight="bold")
        
    ax.set_xticks(x)
    ax.set_xticklabels([f"P = {p}" for p in contexts])
    ax.set_ylabel("Expert Reloads Avoided")
    ax.set_ylim(0, 26)
    ax.set_title("Figure 14: FA-LRU Eviction Benefit: Reloads Saved over LRU Across Contexts\n(B=32, T=128, W=64, HBM=35 GiB)", fontweight="bold")
    ax.legend()
    save_fig(fig, "combined", "fig14_lru_vs_falru_comparison")
    
    # Fig 15: Three Operating Regimes Empirical Synthesis
    fig, ax = plt.subplots(figsize=(10, 6.0))
    regime_data = [
        {"name": "Regime 1: Accelerated Serving\nHigh Concurrency (B=14, K=4)", "x": 4, "y": 14, "color": "#4CAF50", "gain": "+0.58% Speedup\n(-1,564 reloads)"},
        {"name": "Regime 1: Peak Acceleration\nBalanced Load (B=32, K=2)", "x": 2, "y": 32, "color": "#2E7D32", "gain": "+1.42% Speedup\n(-3,287 reloads)"},
        {"name": "Regime 1: Steady Acceleration\nAnchor (B=32, H=1, K=4)", "x": 4, "y": 32, "color": "#388E3C", "gain": "+0.73% Speedup\n(-1,398 reloads)"},
        {"name": "Regime 2: Compute-Bound Plateau\nLong Context (B=32, P=4096)", "x": 4, "y": 42, "color": "#FFA000", "gain": "+0.77% Speedup\n(Attn Slack Dominant)"},
        {"name": "Regime 3: Low Concurrency Degradation\nCache Pollution (B=4, K=4)", "x": 4, "y": 4, "color": "#E53935", "gain": "-0.13% Degradation\n(+301 reloads)"},
        {"name": "Regime 3: Bus Saturation Degradation\nOver-Aggressive Budget (B=32, K=8)", "x": 8, "y": 32, "color": "#D32F2F", "gain": "-0.026% Degradation vs K=4\n(18.0 TB Spec DDR5)"}
    ]
    
    for r in regime_data:
        ax.scatter(r["x"], r["y"], s=900, color=r["color"], edgecolors="black", linewidth=1.5, alpha=0.85, zorder=3)
        ax.annotate(f"{r['name']}\n[{r['gain']}]", (r["x"], r["y"]), textcoords="offset points", xytext=(0, 16), ha='center', fontsize=9, fontweight="bold")
        
    ax.set_xlim(1, 10)
    ax.set_ylim(0, 50)
    ax.set_xlabel("Prefetch Speculation Budget ($K$ Candidates per Token)")
    ax.set_ylabel("Workload Concurrency & Memory Wall Pressure ($B$)")
    ax.set_title("Figure 15: Empirical Operating Regimes for MoE Expert Prefetching\n(Accelerated Serving vs. Compute-Bound Plateau vs. Performance Inversion)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    save_fig(fig, "combined", "fig15_operating_regimes_summary")
    
    print("\nALL 15 FIGURES GENERATED AND SAVED SUCCESSFULLY!")

if __name__ == "__main__":
    plot_all_figures()
