#!/usr/bin/env python3
"""
Generate publication-quality figures for Point 7 Energy Analysis.

Plots Figures 1 to 12 based strictly on verified CSV data from extracted_data/:
- Figure 1: Total Modeled System Energy across Anchor Configurations (A, B, C, D)
- Figure 2: Energy per Generated Token across Anchor Configurations (A, B, C, D)
- Figure 3: Demand vs Speculative DDR5 Active Energy across Prefetch Budgets (K=0, 2, 4, 8)
- Figure 4: Total DDR5 Traffic vs DDR5 Active Energy (Analytical Model Verification)
- Figure 5: Exposed Memory Stall vs Total System Energy (Energy-Performance Trade-off)
- Figure 6: Energy-Delay Product (EDP) across Anchor Configurations and Budget Sweep
- Figure 7: Energy and Energy/Token Scaling across Batch Sizes (B=4, 14, 32)
- Figure 8: Prefetch Budget Sweet Spot (Delta Energy & Relative EDP vs Baseline)
- Figure 9: Energy Dynamics across Prefetch Lookahead (H=1, 2, 4)
- Figure 10: Energy Scaling across Context Lengths (P=128, 512, 2048, 4096)
- Figure 11: Energy-Performance Pareto Frontier (Simulated Execution Time vs System Energy)
- Figure 12: Component-wise System Energy Breakdown for Complete System D

Outputs both PNG (300 DPI) and vector PDF to:
  1. mohman/point_7/figures/png/ and mohman/point_7/figures/pdf/
  2. Artifact directory: /home/mohman/.gemini/antigravity/brain/b7db0340-f5d9-4df3-86d8-be407d44242d/figures/point_7/

CRITICAL RULE: STRICTLY ZERO TERNARY OPERATORS (ast.IfExp == 0).
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Style configuration
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#CCCCCC'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.7

POINT7_DIR = "/home/mohman/ca_project/mohman/point_7"
DATA_DIR = os.path.join(POINT7_DIR, "extracted_data")
FIG_DIR = os.path.join(POINT7_DIR, "figures")
PNG_DIR = os.path.join(FIG_DIR, "png")
PDF_DIR = os.path.join(FIG_DIR, "pdf")

ARTIFACT_DIR = "/home/mohman/.gemini/antigravity/brain/b7db0340-f5d9-4df3-86d8-be407d44242d/figures/point_7"
ART_PNG_DIR = os.path.join(ARTIFACT_DIR, "png")
ART_PDF_DIR = os.path.join(ARTIFACT_DIR, "pdf")

for d in [PNG_DIR, PDF_DIR, ART_PNG_DIR, ART_PDF_DIR]:
    os.makedirs(d, exist_ok=True)

# Load data
df_unified = pd.read_csv(os.path.join(DATA_DIR, "unified_energy_results.csv"))
df_breakdown = pd.read_csv(os.path.join(DATA_DIR, "energy_component_breakdown.csv"))
df_traffic = pd.read_csv(os.path.join(DATA_DIR, "energy_traffic_reconciliation.csv"))
df_token = pd.read_csv(os.path.join(DATA_DIR, "energy_per_token.csv"))
df_edp = pd.read_csv(os.path.join(DATA_DIR, "energy_delay_product.csv"))
df_paired = pd.read_csv(os.path.join(DATA_DIR, "paired_energy_comparisons.csv"))
df_sens = pd.read_csv(os.path.join(DATA_DIR, "energy_sensitivity_summary.csv"))

def save_fig(fig, base_name):
    png_path = os.path.join(PNG_DIR, base_name + ".png")
    pdf_path = os.path.join(PDF_DIR, base_name + ".pdf")
    art_png_path = os.path.join(ART_PNG_DIR, base_name + ".png")
    art_pdf_path = os.path.join(ART_PDF_DIR, base_name + ".pdf")
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    shutil.copy2(png_path, art_png_path)
    shutil.copy2(pdf_path, art_pdf_path)
    plt.close(fig)
    print("Saved: " + base_name)

# -----------------------------------------------------------------------------
# Figure 1: Total Modeled System Energy across Anchor Configurations (A, B, C, D)
# -----------------------------------------------------------------------------
def plot_fig1():
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    runs = ["anc_c1", "anc_c2", "anc_c3", "anc_c4"]
    labels = ["A: Baseline\n(No Prefetch)", "B: Prediction-Only\n(Shadow)", "C: Prefetch\n+ LRU", "D: Complete System\n(FA-LRU W=64)"]
    sub = df_unified[df_unified['run_id'].isin(runs)].set_index('run_id').loc[runs]
    
    x = np.arange(len(runs))
    width = 0.52
    
    hbm = sub['hbm_total_j'].values
    mac = sub['gpu_mac_j'].values
    ddr5 = sub['ddr5_total_j'].values
    static = sub['static_dram_j'].values
    total = sub['system_total_j'].values
    
    base_val = total[0]
    
    c_hbm = "#1f77b4"
    c_mac = "#2ca02c"
    c_ddr5 = "#ff7f0e"
    c_static = "#9467bd"
    
    b_mac = hbm
    b_ddr5 = hbm + mac
    b_static = hbm + mac + ddr5
    
    ax.bar(x, hbm, width, label="GPU HBM2e Active DRAM", color=c_hbm, edgecolor='black', linewidth=0.5)
    ax.bar(x, mac, width, bottom=b_mac, label="GPU MAC Compute", color=c_mac, edgecolor='black', linewidth=0.5)
    ax.bar(x, ddr5, width, bottom=b_ddr5, label="Off-Chip DDR5 Bus Active (0.1 nJ/byte)", color=c_ddr5, edgecolor='black', linewidth=0.5)
    ax.bar(x, static, width, bottom=b_static, label="Static DRAM Power (4.5 W)", color=c_static, edgecolor='black', linewidth=0.5)
    
    for i in range(len(runs)):
        val = total[i]
        delta_val = val - base_val
        delta_pct = (delta_val / base_val) * 100.0
        delta_str = ""
        if i == 0:
            delta_str = "Ref (100%)"
        else:
            if abs(delta_val) < 0.01:
                delta_str = "+0.000%"
            else:
                delta_str = f"{delta_pct:+.3f}%\n({delta_val:+.2f} J)"
        ax.annotate(f"{val:,.1f} J\n({delta_str})",
                    xy=(x[i], val),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    
    ax.set_ylabel("Modeled System Total Energy (Joules)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 1: Modeled System Total Energy across Anchor Configurations\n(B=32, P=2048, T=128, K=4, H=1)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 11800)
    ax.grid(axis='y')
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    save_fig(fig, "fig1_anchor_system_energy")

# -----------------------------------------------------------------------------
# Figure 2: Energy per Generated Token across Anchor Configurations (A, B, C, D)
# -----------------------------------------------------------------------------
def plot_fig2():
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    runs = ["anc_c1", "anc_c2", "anc_c3", "anc_c4"]
    labels = ["A: Baseline", "B: Prediction-Only", "C: Prefetch + LRU", "D: Complete System"]
    sub = df_unified[df_unified['run_id'].isin(runs)].set_index('run_id').loc[runs]
    
    x = np.arange(len(runs))
    width = 0.35
    
    native_tok = sub['native_energy_per_token_mj'].values
    system_tok = sub['system_energy_per_token_mj'].values
    
    r1 = ax.bar(x - width/2, native_tok, width, label="Native GPU Energy (HBM + MAC)", color="#2b5c8f", edgecolor='black', linewidth=0.5)
    r2 = ax.bar(x + width/2, system_tok, width, label="Modeled System Energy (GPU + DDR5 + Static)", color="#d95f02", edgecolor='black', linewidth=0.5)
    
    for i in range(len(runs)):
        ax.annotate(f"{native_tok[i]:.2f}",
                    xy=(x[i] - width/2, native_tok[i]),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        ax.annotate(f"{system_tok[i]:.2f}",
                    xy=(x[i] + width/2, system_tok[i]),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    
    ax.set_ylabel("Energy per Generated Token (mJ / token)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 2: Energy per Generated Token across Anchor Configurations\n(B=32, P=2048, T=128, Total Generated Tokens = 4,096)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 2750)
    ax.grid(axis='y')
    ax.legend(loc='upper right', frameon=True, fontsize=9.5)
    
    # Text annotation explaining native vs system
    ax.text(0.03, 0.45, "Native GPU energy reflects on-chip HBM bank accesses & MACs (~272 mJ/tok).\nModeled System energy includes off-chip DDR5 transfers and background static power,\nshowing modest +0.31% prefetch overhead (+7.09 mJ/tok).",
            transform=ax.transAxes, fontsize=8.5, bbox=dict(boxstyle="round,pad=0.5", facecolor="#e6f2ff", edgecolor="#2b5c8f", alpha=0.9))
    
    save_fig(fig, "fig2_anchor_energy_per_token")

# -----------------------------------------------------------------------------
# Figure 3: Demand vs Speculative DDR5 Active Energy across Prefetch Budgets
# -----------------------------------------------------------------------------
def plot_fig3():
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    runs = ["anc_c1", "k2_c4", "anc_c4", "k8_c4"]
    labels = ["K=0\n(Baseline)", "K=2\n(Conservative)", "K=4\n(Anchor)", "K=8\n(Aggressive)"]
    sub = df_unified[df_unified['run_id'].isin(runs)].set_index('run_id').loc[runs]
    
    x = np.arange(len(runs))
    width = 0.5
    
    demand = sub['ddr5_demand_j'].values
    spec = sub['ddr5_speculative_j'].values
    total = sub['ddr5_total_j'].values
    
    c_demand = "#377eb8"
    c_spec = "#e41a1c"
    
    ax.bar(x, demand, width, label="Demand DDR5 Active Energy (0.1 nJ/byte)", color=c_demand, edgecolor='black', linewidth=0.5)
    ax.bar(x, spec, width, bottom=demand, label="Speculative Prefetch DDR5 Active Energy", color=c_spec, edgecolor='black', linewidth=0.5)
    
    for i in range(len(runs)):
        pct_spec = (spec[i] / total[i]) * 100.0
        ax.annotate(f"Total: {total[i]:,.1f} J\n(Spec: {pct_spec:.1f}%)",
                    xy=(x[i], total[i]),
                    xytext=(0, 6), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel("DDR5 Active Transfer Energy (Joules)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 3: Demand vs. Speculative DDR5 Active Transfer Energy\nacross Prefetch Budgets (B=32, P=2048, T=128, H=1)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 5800)
    ax.grid(axis='y')
    ax.legend(loc='lower left', frameon=True, fontsize=9.5)
    save_fig(fig, "fig3_prefetch_budget_ddr5_breakdown")

# -----------------------------------------------------------------------------
# Figure 4: Total DDR5 Traffic vs DDR5 Active Energy
# -----------------------------------------------------------------------------
def plot_fig4():
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    
    # Decimal TB: 1 TB = 1e12 bytes
    traffic_tb = df_unified['total_ddr5_bytes'].values / 1e12
    ddr5_energy = df_unified['ddr5_total_j'].values
    
    scatter = ax.scatter(traffic_tb, ddr5_energy, c=df_unified['batch_size'], cmap='viridis', s=65, edgecolors='black', linewidth=0.5, zorder=4)
    
    # Analytical accounting line (0.1 nJ/byte = 100 J/TB)
    x_line = np.linspace(10, 52, 100)
    y_line = x_line * 100.0
    ax.plot(x_line, y_line, color='red', linestyle='--', linewidth=1.2, label="Analytical Transfer Model (100 J / TB)", zorder=2)
    
    # Annotate specific anchor runs
    anc_d = df_unified[df_unified['run_id'] == 'anc_c4'].iloc[0]
    anc_a = df_unified[df_unified['run_id'] == 'anc_c1'].iloc[0]
    k2 = df_unified[df_unified['run_id'] == 'k2_c4'].iloc[0]
    k8 = df_unified[df_unified['run_id'] == 'k8_c4'].iloc[0]
    
    ax.annotate("Anchor D (K=4)\n48.74 TB, 4874 J", xy=(anc_d['total_ddr5_bytes']/1e12, anc_d['ddr5_total_j']),
                xytext=(15, -20), textcoords="offset points", arrowprops=dict(arrowstyle="->", color="black"), fontsize=8.5, fontweight='bold')
    ax.annotate("Baseline A (K=0)\n48.21 TB, 4821 J", xy=(anc_a['total_ddr5_bytes']/1e12, anc_a['ddr5_total_j']),
                xytext=(-105, 15), textcoords="offset points", arrowprops=dict(arrowstyle="->", color="black"), fontsize=8.5, fontweight='bold')
    ax.annotate("K=2: 48.39 TB", xy=(k2['total_ddr5_bytes']/1e12, k2['ddr5_total_j']),
                xytext=(-85, -25), textcoords="offset points", arrowprops=dict(arrowstyle="->", color="black"), fontsize=8.5)
    ax.annotate("K=8: 48.78 TB", xy=(k8['total_ddr5_bytes']/1e12, k8['ddr5_total_j']),
                xytext=(15, 10), textcoords="offset points", arrowprops=dict(arrowstyle="->", color="black"), fontsize=8.5)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Batch Size (B)", fontsize=10, fontweight='bold')
    
    ax.set_xlabel("Total Off-Chip DDR5 Traffic (Decimal TB, $10^{12}$ Bytes)", fontsize=11, fontweight='bold')
    ax.set_ylabel("DDR5 Active Transfer Energy (Joules)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 4: Off-Chip DDR5 Traffic vs. Active Energy Accounting\n(Analytical Evaluation: 0.1 nJ/byte = 100 J/TB across 28 Runs)", fontsize=12, fontweight='bold', pad=12)
    ax.grid(True)
    ax.legend(loc='upper left', frameon=True, fontsize=9.5)
    save_fig(fig, "fig4_ddr5_traffic_vs_energy")

# -----------------------------------------------------------------------------
# Figure 5: Exposed Memory Stall vs Total System Energy (Energy-Performance Trade-off)
# -----------------------------------------------------------------------------
def plot_fig5():
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    
    # Filter B=32 runs (Context P=2048)
    sub = df_unified[(df_unified['batch_size'] == 32) & (df_unified['context_p'] == 2048)].copy()
    
    colors = []
    markers = []
    for idx, row in sub.iterrows():
        p = row['prefetch_enabled']
        pol = row['eviction_policy']
        if not p:
            colors.append('#377eb8') # blue for baseline
            markers.append('o')
        else:
            if pol == 'lru':
                colors.append('#e41a1c') # red for LRU
                markers.append('^')
            else:
                colors.append('#4daf4a') # green for FA-LRU
                markers.append('s')
                
    for i in range(len(sub)):
        row = sub.iloc[i]
        c = colors[i]
        m = markers[i]
        ax.scatter(row['exposed_stall_ms']/1000.0, row['system_total_j'], color=c, marker=m, s=75, edgecolors='black', linewidth=0.6, zorder=3)
        
    # Annotate specific runs
    annot_runs = {
        'anc_c1': 'Baseline A (No Prefetch)',
        'k2_c4': 'K=2 (Sweet Spot: -27.68 J Net Energy)',
        'anc_c4': 'Anchor D (K=4, FA-LRU)',
        'k8_c4': 'K=8 (Over-Aggressive)',
        'h4_c4': 'H=4, K=4 (Lowest Stall: 746.15 s)'
    }
    for rid, lbl in annot_runs.items():
        r = sub[sub['run_id'] == rid].iloc[0]
        x_pt = r['exposed_stall_ms'] / 1000.0
        y_pt = r['system_total_j']
        offset_x = 10
        offset_y = 5
        if rid == 'k2_c4':
            offset_x = -135
            offset_y = -22
        else:
            if rid == 'anc_c1':
                offset_x = -140
                offset_y = 10
            else:
                if rid == 'h4_c4':
                    offset_x = 10
                    offset_y = -15
        ax.annotate(lbl, xy=(x_pt, y_pt), xytext=(offset_x, offset_y), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8), fontsize=8.5, fontweight='bold')
        
    # Legend proxies
    ax.scatter([], [], color='#377eb8', marker='o', s=60, label='Baseline A / Pred B')
    ax.scatter([], [], color='#e41a1c', marker='^', s=60, label='Prefetch + LRU')
    ax.scatter([], [], color='#4daf4a', marker='s', s=60, label='Prefetch + FA-LRU (W=64)')
    
    ax.set_xlabel("Exposed Memory Stall Time (seconds)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Modeled System Total Energy (Joules)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 5: Exposed Memory Stall vs. Total System Energy (B=32 Sweeps)\nBalancing Stall Latency Against Speculative Energy", fontsize=12, fontweight='bold', pad=12)
    ax.grid(True)
    ax.legend(loc='center left', frameon=True, fontsize=9.5)
    save_fig(fig, "fig5_stall_reduction_vs_energy_overhead")

# -----------------------------------------------------------------------------
# Figure 6: Energy-Delay Product (EDP) across Anchor Configurations and Budget Sweep
# -----------------------------------------------------------------------------
def plot_fig6():
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    runs = ["anc_c1", "anc_c2", "anc_c3", "anc_c4", "k2_c4", "k8_c4"]
    labels = ["Baseline A\n(K=0)", "Pred-Only B\n(Shadow)", "Prefetch C\n(K=4, LRU)", "Complete D\n(K=4, FA-LRU)", "K=2 Sweet Spot\n(FA-LRU)", "K=8 Aggressive\n(FA-LRU)"]
    sub = df_unified[df_unified['run_id'].isin(runs)].set_index('run_id').loc[runs]
    
    base_edp = sub.loc['anc_c1', 'system_edp_j_s']
    edp_vals = sub['system_edp_j_s'].values / 1e6 # Million J*s
    deltas_pct = ((sub['system_edp_j_s'].values - base_edp) / base_edp) * 100.0
    
    x = np.arange(len(runs))
    width = 0.52
    
    colors = []
    for d in deltas_pct:
        if d < -0.01:
            colors.append('#2ca02c') # green improvement
        else:
            if d > 0.01:
                colors.append('#d62728') # red degradation
            else:
                colors.append('#1f77b4') # neutral blue
            
    bars = ax.bar(x, edp_vals, width, color=colors, edgecolor='black', linewidth=0.6)
    
    for i in range(len(runs)):
        val = edp_vals[i]
        d_pct = deltas_pct[i]
        lbl = f"{val:.3f}M\n({d_pct:+.2f}%)"
        if i == 0:
            lbl = f"{val:.3f}M\n(Ref)"
        ax.annotate(lbl, xy=(x[i], val), xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        
    ax.axhline(base_edp / 1e6, color='black', linestyle='--', linewidth=0.9, alpha=0.7, label='Baseline Reference EDP')
    ax.set_ylabel("System Energy-Delay Product ($10^6$ Joules $\\cdot$ s)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 6: System Energy-Delay Product (EDP) across Configurations\n(B=32, P=2048, T=128)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(6.5, 7.5)
    ax.grid(axis='y')
    ax.legend(loc='lower left', frameon=True, fontsize=9.5)
    save_fig(fig, "fig6_energy_delay_product")

# -----------------------------------------------------------------------------
# Figure 7: Energy and Energy/Token Scaling across Batch Sizes (B=4, 14, 32)
# -----------------------------------------------------------------------------
def plot_fig7():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    
    batches = [4, 14, 32]
    base_runs = ["b4_c1", "b14_c1", "anc_c1"]
    comp_runs = ["b4_c4", "b14_c4", "anc_c4"]
    
    sub_base = df_unified[df_unified['run_id'].isin(base_runs)].set_index('run_id').loc[base_runs]
    sub_comp = df_unified[df_unified['run_id'].isin(comp_runs)].set_index('run_id').loc[comp_runs]
    
    x = np.arange(len(batches))
    width = 0.35
    
    # Panel 1: Total System Energy
    ax1.bar(x - width/2, sub_base['system_total_j'], width, label="Baseline A (No Prefetch)", color="#377eb8", edgecolor='black', linewidth=0.5)
    ax1.bar(x + width/2, sub_comp['system_total_j'], width, label="Complete System D (FA-LRU)", color="#4daf4a", edgecolor='black', linewidth=0.5)
    
    for i in range(len(batches)):
        b_val = sub_base['system_total_j'].values[i]
        c_val = sub_comp['system_total_j'].values[i]
        pct = ((c_val - b_val) / b_val) * 100.0
        ax1.annotate(f"{b_val:,.0f} J", xy=(x[i] - width/2, b_val), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=8.5)
        ax1.annotate(f"{c_val:,.0f} J\n({pct:+.2f}%)", xy=(x[i] + width/2, c_val), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        
    ax1.set_xlabel("Batch Size (B)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Total System Energy (Joules)", fontsize=11, fontweight='bold')
    ax1.set_title("(a) Total System Energy vs. Batch Size", fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"B={b}" for b in batches], fontsize=10)
    ax1.set_ylim(0, 11500)
    ax1.grid(axis='y')
    ax1.legend(loc='upper left', frameon=True, fontsize=9)
    
    # Panel 2: Energy per Generated Token
    ax2.bar(x - width/2, sub_base['system_energy_per_token_mj'], width, label="Baseline A (No Prefetch)", color="#377eb8", edgecolor='black', linewidth=0.5)
    ax2.bar(x + width/2, sub_comp['system_energy_per_token_mj'], width, label="Complete System D (FA-LRU)", color="#4daf4a", edgecolor='black', linewidth=0.5)
    
    for i in range(len(batches)):
        b_tok = sub_base['system_energy_per_token_mj'].values[i]
        c_tok = sub_comp['system_energy_per_token_mj'].values[i]
        pct = ((c_tok - b_tok) / b_tok) * 100.0
        ax2.annotate(f"{b_tok:.1f}", xy=(x[i] - width/2, b_tok), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=8.5)
        ax2.annotate(f"{c_tok:.1f}\n({pct:+.2f}%)", xy=(x[i] + width/2, c_tok), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        
    ax2.set_xlabel("Batch Size (B)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("System Energy per Token (mJ / token)", fontsize=11, fontweight='bold')
    ax2.set_title("(b) Concurrency Amortization of Energy per Token", fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"B={b}" for b in batches], fontsize=10)
    ax2.set_ylim(0, 5500)
    ax2.grid(axis='y')
    ax2.legend(loc='upper right', frameon=True, fontsize=9)
    
    fig.suptitle("Figure 7: Energy Scaling and Concurrency Amortization across Batch Sizes", fontsize=12, fontweight='bold', y=0.98)
    fig.tight_layout()
    save_fig(fig, "fig7_batch_size_energy_scaling")

# -----------------------------------------------------------------------------
# Figure 8: Prefetch Budget Sweet Spot (Delta Energy & Relative EDP vs Baseline)
# -----------------------------------------------------------------------------
def plot_fig8():
    fig, ax1 = plt.subplots(figsize=(8.8, 5.6))
    runs = ["anc_c1", "k2_c4", "anc_c4", "k8_c4"]
    k_vals = [0, 2, 4, 8]
    sub = df_unified[df_unified['run_id'].isin(runs)].set_index('run_id').loc[runs]
    
    base_energy = sub.loc['anc_c1', 'system_total_j']
    base_edp = sub.loc['anc_c1', 'system_edp_j_s']
    
    delta_energy = sub['system_total_j'].values - base_energy
    delta_edp_pct = ((sub['system_edp_j_s'].values - base_edp) / base_edp) * 100.0
    
    color1 = '#1f77b4'
    color2 = '#d95f02'
    
    ax1.set_xlabel("Prefetch Budget (K candidates per token)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Net Change in System Energy (Joules vs. Baseline)", color=color1, fontsize=11, fontweight='bold')
    l1 = ax1.plot(k_vals, delta_energy, marker='o', linewidth=2.2, color=color1, label="Delta Total Energy (J)")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(0, color='gray', linestyle=':', linewidth=1)
    
    for i, txt in enumerate(delta_energy):
        offset = (0, 8)
        if i == 1:
            offset = (0, -18)
        ax1.annotate(f"{txt:+.2f} J", (k_vals[i], delta_energy[i]), xytext=offset, textcoords="offset points",
                     ha='center', fontsize=9, fontweight='bold', color=color1)
        
    ax2 = ax1.twinx()
    ax2.set_ylabel("Relative EDP Change (%)", color=color2, fontsize=11, fontweight='bold')
    l2 = ax2.plot(k_vals, delta_edp_pct, marker='s', linewidth=2.2, linestyle='--', color=color2, label="Relative EDP Change (%)")
    ax2.tick_params(axis='y', labelcolor=color2)
    
    for i, txt in enumerate(delta_edp_pct):
        offset = (0, -16)
        if i == 1:
            offset = (0, 8)
        ax2.annotate(f"{txt:+.2f}%", (k_vals[i], delta_edp_pct[i]), xytext=offset, textcoords="offset points",
                     ha='center', fontsize=9, fontweight='bold', color=color2)
        
    ax1.set_xticks(k_vals)
    ax1.set_xticklabels(["K=0\n(Baseline)", "K=2\n(Sweet Spot)", "K=4\n(Anchor)", "K=8\n(Diminishing)"], fontsize=10)
    
    # Highlight sweet spot
    ax1.axvspan(1.5, 2.5, color='green', alpha=0.12, label="Energy & EDP Sweet Spot (K=2)")
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', frameon=True, fontsize=9.5)
    
    ax1.set_title("Figure 8: Prefetch Budget Sweet Spot: Net Energy & EDP Trade-off\n(B=32, P=2048, T=128, H=1)", fontsize=12, fontweight='bold', pad=12)
    ax1.grid(True)
    save_fig(fig, "fig8_budget_sweet_spot")

# -----------------------------------------------------------------------------
# Figure 9: Energy Dynamics across Prefetch Lookahead (H=1, 2, 4)
# -----------------------------------------------------------------------------
def plot_fig9():
    fig, ax1 = plt.subplots(figsize=(8.8, 5.6))
    runs = ["anc_c4", "h2_c4", "h4_c4"]
    h_vals = [1, 2, 4]
    sub = df_unified[df_unified['run_id'].isin(runs)].set_index('run_id').loc[runs]
    
    stalls = sub['exposed_stall_ms'].values / 1000.0 # seconds
    system_e = sub['system_total_j'].values
    
    c1 = '#2b5c8f'
    c2 = '#d95f02'
    
    ax1.set_xlabel("Prefetch Lookahead (H steps ahead)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Exposed Memory Stall (seconds)", color=c1, fontsize=11, fontweight='bold')
    l1 = ax1.plot(h_vals, stalls, marker='o', linewidth=2.0, color=c1, label="Exposed Stall (s)")
    ax1.tick_params(axis='y', labelcolor=c1)
    
    for i in range(len(h_vals)):
        ax1.annotate(f"{stalls[i]:.2f} s", (h_vals[i], stalls[i]), xytext=(0, 6), textcoords="offset points",
                     ha='center', fontsize=9, fontweight='bold', color=c1)
        
    ax2 = ax1.twinx()
    ax2.set_ylabel("Modeled System Total Energy (Joules)", color=c2, fontsize=11, fontweight='bold')
    l2 = ax2.plot(h_vals, system_e, marker='^', linewidth=2.0, linestyle='--', color=c2, label="Total System Energy (J)")
    ax2.tick_params(axis='y', labelcolor=c2)
    
    for i in range(len(h_vals)):
        ax2.annotate(f"{system_e[i]:,.2f} J", (h_vals[i], system_e[i]), xytext=(0, -15), textcoords="offset points",
                     ha='center', fontsize=9, fontweight='bold', color=c2)
        
    ax1.set_xticks(h_vals)
    ax1.set_xticklabels(["H=1", "H=2", "H=4"], fontsize=10)
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True, fontsize=9.5)
    
    ax1.set_title("Figure 9: Energy and Stall Sensitivity to Prefetch Lookahead H\n(B=32, P=2048, T=128, K=4)", fontsize=12, fontweight='bold', pad=12)
    ax1.grid(True)
    save_fig(fig, "fig9_lookahead_energy_dynamics")

# -----------------------------------------------------------------------------
# Figure 10: Energy Scaling across Context Lengths (P=128, 512, 2048, 4096)
# -----------------------------------------------------------------------------
def plot_fig10():
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    runs = ["p128_c4", "p512_c4", "anc_c4", "p4096_c4"]
    p_vals = [128, 512, 2048, 4096]
    sub = df_unified[df_unified['run_id'].isin(runs)].set_index('run_id').loc[runs]
    
    attn_comp = sub['attn_comp_j'].values
    attn_dram = sub['attn_dram_j'].values
    moe_dram = sub['moe_dram_j'].values
    moe_comp = sub['moe_comp_j'].values
    
    x = np.arange(len(p_vals))
    width = 0.5
    
    ax.bar(x, moe_dram, width, label="MoE HBM DRAM Energy", color="#3182bd", edgecolor='black', linewidth=0.5)
    ax.bar(x, moe_comp, width, bottom=moe_dram, label="MoE MAC Compute Energy", color="#6baed6", edgecolor='black', linewidth=0.5)
    ax.bar(x, attn_dram, width, bottom=moe_dram+moe_comp, label="Attention HBM DRAM Energy (KV Cache)", color="#fd8d3c", edgecolor='black', linewidth=0.5)
    ax.bar(x, attn_comp, width, bottom=moe_dram+moe_comp+attn_dram, label="Attention MAC Compute Energy", color="#e6550d", edgecolor='black', linewidth=0.5)
    
    total_onchip = moe_dram + moe_comp + attn_dram + attn_comp
    for i in range(len(p_vals)):
        attn_pct = ((attn_comp[i] + attn_dram[i]) / total_onchip[i]) * 100.0
        ax.annotate(f"{total_onchip[i]:.1f} J\n(Attn: {attn_pct:.1f}%)",
                    xy=(x[i], total_onchip[i]),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        
    ax.set_ylabel("On-Chip GPU Energy (Joules)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Initial Prompt Context Length (P tokens)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 10: Growth of Attention KV Cache & Compute Energy with Context Length\n(B=32, T=128, K=4, H=1, Complete System D)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P={p}" for p in p_vals], fontsize=10)
    ax.set_ylim(0, 1300)
    ax.grid(axis='y')
    ax.legend(loc='upper left', frameon=True, fontsize=9.5)
    save_fig(fig, "fig10_context_length_energy_scaling")

# -----------------------------------------------------------------------------
# Figure 11: Energy-Performance Pareto Frontier (Execution Time vs System Energy)
# -----------------------------------------------------------------------------
def plot_fig11():
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    
    b32_runs = df_unified[(df_unified['batch_size'] == 32) & (df_unified['context_p'] == 2048)].copy()
    
    for idx, row in b32_runs.iterrows():
        p = row['prefetch_enabled']
        pol = row['eviction_policy']
        c = '#377eb8'
        m = 'o'
        if not p:
            c = '#377eb8'
            m = 'o'
        else:
            if pol == 'lru':
                c = '#e41a1c'
                m = '^'
            else:
                c = '#2ca02c'
                m = 's'
        ax.scatter(row['simulated_execution_time_s'], row['system_total_j'], color=c, marker=m, s=80, edgecolors='black', linewidth=0.6, zorder=3)
        
    pts = {
        'anc_c1': 'Baseline A (No Prefetch)',
        'k2_c4': 'Pareto Optimal: K=2 Sweet Spot\n(Lowest Energy: 9,359.9 J)',
        'anc_c4': 'Anchor D (K=4, FA-LRU)',
        'anc_c3': 'Anchor C (K=4, LRU)',
        'h4_c4': 'Pareto Optimal: H=4, K=4\n(Lowest Latency: 760.20 s)',
        'k8_c4': 'Sub-optimal: K=8\n(Bus Contention)'
    }
    
    for rid, lbl in pts.items():
        r = b32_runs[b32_runs['run_id'] == rid].iloc[0]
        x_pt = r['simulated_execution_time_s']
        y_pt = r['system_total_j']
        ox = 10
        oy = 5
        if rid == 'k2_c4':
            ox = -160
            oy = -25
        else:
            if rid == 'anc_c1':
                ox = 12
                oy = 5
            else:
                if rid == 'h4_c4':
                    ox = -155
                    oy = 10
                else:
                    if rid == 'anc_c3':
                        ox = 10
                        oy = -15
                    else:
                        if rid == 'k8_c4':
                            ox = -120
                            oy = 10
        ax.annotate(lbl, xy=(x_pt, y_pt), xytext=(ox, oy), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8), fontsize=8.5, fontweight='bold')
        
    # Highlight Pareto Curve between K=2 and H=4
    r_k2 = b32_runs[b32_runs['run_id'] == 'k2_c4'].iloc[0]
    r_h2 = b32_runs[b32_runs['run_id'] == 'h2_c4'].iloc[0]
    r_h4 = b32_runs[b32_runs['run_id'] == 'h4_c4'].iloc[0]
    pareto_x = [r_k2['simulated_execution_time_s'], r_h2['simulated_execution_time_s'], r_h4['simulated_execution_time_s']]
    pareto_y = [r_k2['system_total_j'], r_h2['system_total_j'], r_h4['system_total_j']]
    ax.plot(pareto_x, pareto_y, color='gold', linestyle='-', linewidth=2.5, label='Pareto Optimal Frontier', zorder=2)
    
    ax.scatter([], [], color='#377eb8', marker='o', s=70, label='Baseline A / Pred B')
    ax.scatter([], [], color='#e41a1c', marker='^', s=70, label='Prefetch + LRU (C)')
    ax.scatter([], [], color='#2ca02c', marker='s', s=70, label='Complete System FA-LRU (D)')
    
    ax.set_xlabel("Simulated Execution Time (seconds)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Modeled System Total Energy (Joules)", fontsize=11, fontweight='bold')
    ax.set_title("Figure 11: Energy-Performance Trade-off and Pareto Frontier\n(B=32, P=2048, T=128 Workloads)", fontsize=12, fontweight='bold', pad=12)
    ax.grid(True)
    ax.legend(loc='lower left', frameon=True, fontsize=9.5)
    save_fig(fig, "fig11_energy_performance_pareto")

# -----------------------------------------------------------------------------
# Figure 12: Component-wise System Energy Breakdown for Complete System D
# -----------------------------------------------------------------------------
def plot_fig12():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4))
    
    row_d = df_unified[df_unified['run_id'] == 'anc_c4'].iloc[0]
    
    # 4 Macro components
    labels1 = [
        f"Off-Chip DDR5 Bus Active\n({row_d['ddr5_total_j']:,.1f} J)",
        f"Static DRAM Background\n({row_d['static_dram_j']:,.1f} J)",
        f"GPU HBM2e Active\n({row_d['hbm_total_j']:,.1f} J)",
        f"GPU MAC Compute\n({row_d['gpu_mac_j']:,.1f} J)"
    ]
    vals1 = [row_d['ddr5_total_j'], row_d['static_dram_j'], row_d['hbm_total_j'], row_d['gpu_mac_j']]
    colors1 = ['#ff7f0e', '#9467bd', '#1f77b4', '#2ca02c']
    explode1 = (0.05, 0.05, 0.05, 0.1)
    
    wedges1, texts1, autotexts1 = ax1.pie(vals1, labels=labels1, autopct='%1.1f%%', startangle=140,
                                           colors=colors1, explode=explode1, textprops=dict(fontsize=8.5))
    for at in autotexts1:
        at.set_fontweight('bold')
    ax1.set_title(f"(a) Macro System Breakdown\nTotal: {row_d['system_total_j']:,.1f} Joules", fontsize=11, fontweight='bold')
    
    # Sub-breakdown of DDR5: Demand vs Speculative
    labels2 = [
        f"Demand DDR5\n({row_d['ddr5_demand_j']:,.1f} J)",
        f"Speculative Prefetch DDR5\n({row_d['ddr5_speculative_j']:,.1f} J)"
    ]
    vals2 = [row_d['ddr5_demand_j'], row_d['ddr5_speculative_j']]
    colors2 = ['#377eb8', '#e41a1c']
    explode2 = (0, 0.08)
    
    wedges2, texts2, autotexts2 = ax2.pie(vals2, labels=labels2, autopct='%1.1f%%', startangle=90,
                                           colors=colors2, explode=explode2, textprops=dict(fontsize=9))
    for at in autotexts2:
        at.set_fontweight('bold')
    ax2.set_title(f"(b) Off-Chip DDR5 Bus Energy Split\nTotal DDR5: {row_d['ddr5_total_j']:,.1f} Joules", fontsize=11, fontweight='bold')
    
    fig.suptitle("Figure 12: Comprehensive Energy Breakdown for Complete System D (Anchor)", fontsize=12, fontweight='bold', y=0.98)
    fig.tight_layout()
    save_fig(fig, "fig12_component_pie_breakdown")

def main():
    print("Generating Figure 1...")
    plot_fig1()
    print("Generating Figure 2...")
    plot_fig2()
    print("Generating Figure 3...")
    plot_fig3()
    print("Generating Figure 4...")
    plot_fig4()
    print("Generating Figure 5...")
    plot_fig5()
    print("Generating Figure 6...")
    plot_fig6()
    print("Generating Figure 7...")
    plot_fig7()
    print("Generating Figure 8...")
    plot_fig8()
    print("Generating Figure 9...")
    plot_fig9()
    print("Generating Figure 10...")
    plot_fig10()
    print("Generating Figure 11...")
    plot_fig11()
    print("Generating Figure 12...")
    plot_fig12()
    print("All 12 Point 7 figures generated successfully!")

if __name__ == '__main__':
    main()
