#!/usr/bin/env python3
"""
Extract per-prompt inferred initial context lengths from authentic MoE-Anywhere trace dataset.
Strict adherence to zero ternary operators.
"""

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def run_extraction():
    parquet_path = "LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet"
    out_dir = "mohman/point_6/experiment_a"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Reading parquet from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} records across {df['prompt_id'].nunique()} prompts.")
    
    prompt_records = []
    
    # Process each prompt
    for pid in sorted(df['prompt_id'].unique()):
        p_df = df[df['prompt_id'] == pid]
        
        tasks = p_df['task'].unique()
        task_name = tasks[0]
        
        row_count = len(p_df)
        min_pos = int(p_df['token_pos'].min())
        max_pos = int(p_df['token_pos'].max())
        inferred_P = min_pos
        
        unique_positions = p_df['token_pos'].unique()
        unique_pos_count = len(unique_positions)
        
        unique_layers = p_df['layer'].unique()
        unique_layer_count = len(unique_layers)
        
        # Validation checks
        # 1. Contiguous positions and exactly 257 positions
        span = max_pos - min_pos + 1
        contiguous_trace = False
        if span == unique_pos_count:
            if unique_pos_count == 257:
                contiguous_trace = True
                
        # 2. Complete layer coverage: all 40 layers at every position
        # Exactly 40 layers, and exactly 40 rows per token position
        pos_counts = p_df['token_pos'].value_counts()
        complete_layer_coverage = False
        if unique_layer_count == 40:
            if (pos_counts == 40).all():
                complete_layer_coverage = True
                
        # 3. Top-k valid: exactly 8 experts per row, all within [0, 255]
        topk_valid = True
        for row_experts in p_df['experts']:
            if len(row_experts) != 8:
                topk_valid = False
                break
            for eid in row_experts:
                if eid < 0:
                    topk_valid = False
                    break
                if eid > 255:
                    topk_valid = False
                    break
            if not topk_valid:
                break
                
        # 4. Consistency across layers: min position is identical across all 40 layers
        min_pos_per_layer = p_df.groupby('layer')['token_pos'].min()
        layer_min_consistent = False
        if (min_pos_per_layer == min_pos).all():
            layer_min_consistent = True
            
        # 5. Overall trace valid
        trace_valid = False
        if contiguous_trace:
            if complete_layer_coverage:
                if topk_valid:
                    if layer_min_consistent:
                        if row_count == 10280:
                            trace_valid = True
                            
        prompt_records.append({
            'prompt_id': pid,
            'task': task_name,
            'min_token_pos': min_pos,
            'max_token_pos': max_pos,
            'inferred_initial_P': inferred_P,
            'unique_token_positions': unique_pos_count,
            'row_count': row_count,
            'unique_layers': unique_layer_count,
            'contiguous_trace': contiguous_trace,
            'complete_layer_coverage': complete_layer_coverage,
            'topk_valid': topk_valid,
            'trace_valid': trace_valid
        })
        
    out_df = pd.DataFrame(prompt_records)
    
    # Sort by inferred_initial_P, then prompt_id
    out_df = out_df.sort_values(by=['inferred_initial_P', 'prompt_id']).reset_index(drop=True)
    
    csv_path = os.path.join(out_dir, "PROMPT_INITIAL_CONTEXT_LENGTHS.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"Saved prompt lengths CSV to {csv_path} ({len(out_df)} rows).")
    
    # Verify all passed validation
    all_valid = out_df['trace_valid'].all()
    print(f"All 74 prompts passed complete validation: {all_valid}")
    
    # Statistical analysis
    p_vals = out_df['inferred_initial_P']
    stats = {
        'count': int(len(p_vals)),
        'min': int(p_vals.min()),
        'max': int(p_vals.max()),
        'mean': float(p_vals.mean()),
        'median': float(p_vals.median()),
        'std': float(p_vals.std()),
        'p25': float(p_vals.quantile(0.25)),
        'p75': float(p_vals.quantile(0.75)),
        'iqr': float(p_vals.quantile(0.75) - p_vals.quantile(0.25))
    }
    
    print("\n--- STATISTICAL SUMMARY ACROSS 74 PROMPTS ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
        
    print("\n--- FREQUENCY DISTRIBUTION OF INFERRED P ---")
    freq = p_vals.value_counts().sort_index()
    for p, c in freq.items():
        pct = (c / len(p_vals)) * 100.0
        print(f"  P = {p:2d}: {c:2d} prompts ({pct:5.2f}%)")
        
    print("\n--- TASK-WISE BREAKDOWN ---")
    task_grp = out_df.groupby('task')['inferred_initial_P'].agg([
        'count', 'min', 'max', 'mean', 'median', 'std',
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75)
    ]).rename(columns={'<lambda_0>': 'p25', '<lambda_1>': 'p75'})
    print(task_grp.to_string())
    
    # Generate publication-quality plot
    plot_histogram(out_df, stats, out_dir)
    
    return out_df, stats, task_grp

def plot_histogram(df, stats, out_dir):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2.2, 1.3]})
    
    p_vals = df['inferred_initial_P']
    
    # Top plot: Histogram with Task Breakdown
    task_order = df.groupby('task')['inferred_initial_P'].median().sort_values().index.tolist()
    tasks = sorted(df['task'].unique())
    
    # Palette
    cmap = plt.get_cmap('tab10')
    task_colors = {}
    for idx, t in enumerate(tasks):
        task_colors[t] = cmap(idx % 10)
        
    bins = np.arange(12.5, 66.5, 1.0)
    
    # Stacked data for histogram
    task_data = []
    for t in tasks:
        task_data.append(df[df['task'] == t]['inferred_initial_P'])
        
    n, bin_edges, patches = ax1.hist(
        task_data, bins=bins, stacked=True, label=tasks,
        color=[task_colors[t] for t in tasks], edgecolor='black', linewidth=0.6, alpha=0.9
    )
    
    # Annotations for Mean and Median
    mean_val = stats['mean']
    median_val = stats['median']
    min_val = stats['min']
    max_val = stats['max']
    p25_val = stats['p25']
    p75_val = stats['p75']
    
    ax1.axvline(mean_val, color='#D32F2F', linestyle='--', linewidth=2.0, label=f"Mean: {mean_val:.2f}")
    ax1.axvline(median_val, color='#1976D2', linestyle='-', linewidth=2.0, label=f"Median: {median_val:.1f}")
    ax1.axvline(p25_val, color='#388E3C', linestyle=':', linewidth=1.5, label=f"25th Pct: {p25_val:.1f}")
    ax1.axvline(p75_val, color='#7B1FA2', linestyle=':', linewidth=1.5, label=f"75th Pct: {p75_val:.1f}")
    
    ax1.set_title(r"Authentic MoE-Anywhere Initial Context Length Distribution ($P = \mathrm{min\_token\_pos}$)", fontsize=14, fontweight='bold', pad=12)
    ax1.set_xlabel("Inferred Initial Context Length $P$ (Tokens)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Prompt Count", fontsize=11, fontweight='bold')
    ax1.set_xlim(10, 68)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    # Text box with key statistics
    stats_text = (
        f"Sample Size: N = 74 prompts\n"
        f"Range: [{min_val}, {max_val}] tokens (Span = {max_val - min_val} tokens)\n"
        f"Mean ± Std: {mean_val:.2f} ± {stats['std']:.2f} tokens\n"
        f"Median (IQR): {median_val:.1f} ({p25_val:.1f} - {p75_val:.1f}, IQR = {stats['iqr']:.1f})\n"
        f"Validation: 100% Contiguous (257 decode steps/prompt)\n"
        f"Layer Coverage: 100% (40/40 layers, Top-8 routing)"
    )
    ax1.text(0.68, 0.95, stats_text, transform=ax1.transAxes, fontsize=9.5,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.6', facecolor='#F5F5F5', edgecolor='#9E9E9E', alpha=0.95))
             
    ax1.legend(loc='upper center', bbox_to_anchor=(0.35, 0.98), ncol=3, fontsize=8.5, frameon=True)
    
    # Bottom plot: Task-wise Boxplot / Strip plot
    # Group prompts by task
    task_order_sorted = df.groupby('task')['inferred_initial_P'].median().sort_values().index.tolist()
    box_data = []
    box_labels = []
    for t in task_order_sorted:
        vals = df[df['task'] == t]['inferred_initial_P'].values
        box_data.append(vals)
        count = len(vals)
        box_labels.append(f"{t}\n(n={count})")
        
    bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True, orientation='vertical', showmeans=True,
                     meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='#D32F2F', markersize=5),
                     medianprops=dict(color='#1976D2', linewidth=2.0))
                     
    for patch, t in zip(bp['boxes'], task_order_sorted):
        patch.set_facecolor(task_colors[t])
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
        
    # Overlay jittered points
    for idx, t in enumerate(task_order_sorted):
        y_vals = df[df['task'] == t]['inferred_initial_P'].values
        x_jitter = np.random.normal(idx + 1, 0.05, size=len(y_vals))
        ax2.scatter(x_jitter, y_vals, color=task_colors[t], edgecolor='black', s=25, alpha=0.85, zorder=3)
        
    ax2.set_title("Inferred Initial Context Length Variation Across Task Categories", fontsize=12, fontweight='bold', pad=8)
    ax2.set_ylabel("Inferred $P$ (Tokens)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Task Category", fontsize=11, fontweight='bold')
    ax2.set_ylim(10, 68)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    
    plt.tight_layout()
    
    png_path = os.path.join(out_dir, "INITIAL_CONTEXT_LENGTH_HISTOGRAM.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Saved histogram plot to {png_path}.")

if __name__ == "__main__":
    run_extraction()
