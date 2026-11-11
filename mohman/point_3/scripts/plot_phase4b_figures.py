import os
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "/home/mohman/ca_project/mohman/point_3"
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Styling settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

def read_csv_rows(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def plot_fig1_core_coverage_and_hit_rate(core_rows):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    labels = []
    proj_hits = []
    uniq_hits = []
    for r in core_rows:
        cfg = r['config']
        if cfg == 'baseline':
            labels.append('Baseline\n(Demand)')
        elif cfg == 'stmoe_h1':
            labels.append('ST-MoE\n(H=1, B=4)')
        elif cfg == 'stmoe_h2':
            labels.append('ST-MoE\n(H=2, B=4)')
        elif cfg == 'cstp_h1':
            labels.append('CSTP\n(H=1, B=4)')
        elif cfg == 'cstp_h2':
            labels.append('CSTP\n(H=2, B=4)')
        else:
            labels.append(cfg)

        proj_hits.append(float(r['hbm_hit_rate_proj']) * 100.0)
        uniq_hits.append(float(r['unique_hbm_hit_rate']) * 100.0)

    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax.bar(x - width/2.0, proj_hits, width, label='Projection-Level Hit Rate (W1/W3/W2)', color='#2b5c8f', edgecolor='black')
    rects2 = ax.bar(x + width/2.0, uniq_hits, width, label='Unique-Expert Hit Rate (Deduplicated)', color='#d95f02', edgecolor='black')

    ax.set_ylabel('HBM Hit Rate (%)', fontweight='bold')
    ax.set_title('Figure 1: Core Comparison - Projection-Level vs Unique-Expert HBM Hit Rate\n(MoE-Anywhere Trace, B=1, 4 Iterations)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)

    for r in rects1:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2.0, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for r in rects2:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2.0, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig1_core_coverage_and_hit_rate.png"), dpi=300)
    plt.savefig(os.path.join(PLOTS_DIR, "fig1_core_coverage_and_hit_rate.pdf"))
    plt.close()
    print("[PLOTTED] fig1_core_coverage_and_hit_rate")

def plot_fig2_core_memory_traffic(core_rows):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    labels = []
    demand_gb = []
    spec_gb = []
    wasted_gb = []

    for r in core_rows:
        cfg = r['config']
        if cfg == 'baseline':
            labels.append('Baseline\n(Demand)')
        elif cfg == 'stmoe_h1':
            labels.append('ST-MoE\n(H=1, B=4)')
        elif cfg == 'stmoe_h2':
            labels.append('ST-MoE\n(H=2, B=4)')
        elif cfg == 'cstp_h1':
            labels.append('CSTP\n(H=1, B=4)')
        elif cfg == 'cstp_h2':
            labels.append('CSTP\n(H=2, B=4)')
        else:
            labels.append(cfg)

        d_gb = float(r['demand_bytes']) / (1024.0 ** 3)
        s_gb = float(r['speculative_bytes']) / (1024.0 ** 3)
        w_gb = float(r['total_wasted_speculative_bytes']) / (1024.0 ** 3)

        demand_gb.append(d_gb)
        spec_gb.append(s_gb)
        wasted_gb.append(w_gb)

    x = np.arange(len(labels))
    width = 0.45

    b1 = ax.bar(x, demand_gb, width, label='Demand DDR5 Traffic (GB)', color='#1f77b4', edgecolor='black')
    b2 = ax.bar(x, spec_gb, width, bottom=demand_gb, label='Speculative DDR5 Traffic (GB)', color='#ff7f0e', edgecolor='black')

    ax.set_ylabel('Total DDR5 Traffic Transferred (GB)', fontweight='bold')
    ax.set_title('Figure 2: Memory Traffic Volume and Speculative Overhead\n(MoE-Anywhere, B=1, 4 Iterations)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.grid(axis='y')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)

    for i in range(len(labels)):
        tot = demand_gb[i] + spec_gb[i]
        over = 0.0
        if demand_gb[0] > 0.0:
            over = ((tot - demand_gb[0]) / demand_gb[0]) * 100.0
        ax.annotate(f"{tot:.2f} GB\n(+{over:.1f}%)", xy=(x[i], tot), xytext=(0, 4),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig2_core_memory_traffic_and_overhead.png"), dpi=300)
    plt.savefig(os.path.join(PLOTS_DIR, "fig2_core_memory_traffic_and_overhead.pdf"))
    plt.close()
    print("[PLOTTED] fig2_core_memory_traffic_and_overhead")

def plot_fig3_core_exposed_stall_and_latency(core_rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    
    labels = []
    stall_ms = []
    token_lat_ms = []
    tokens_sec = []

    for r in core_rows:
        cfg = r['config']
        if cfg == 'baseline':
            labels.append('Baseline')
        elif cfg == 'stmoe_h1':
            labels.append('ST-MoE (H=1)')
        elif cfg == 'stmoe_h2':
            labels.append('ST-MoE (H=2)')
        elif cfg == 'cstp_h1':
            labels.append('CSTP (H=1)')
        elif cfg == 'cstp_h2':
            labels.append('CSTP (H=2)')
        else:
            labels.append(cfg)

        stall_ms.append(float(r['exposed_gpu_stall_ns']) / 1e6)
        token_lat_ms.append(float(r['token_lat_mean_ns']) / 1e6)
        tokens_sec.append(float(r['tokens_per_second']))

    x = np.arange(len(labels))
    width = 0.5

    ax1.bar(x, stall_ms, width, color='#d62728', edgecolor='black')
    ax1.set_ylabel('Exposed GPU Stall Time (ms)', fontweight='bold')
    ax1.set_title('Exposed Memory Stall Duration', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha='right', fontweight='bold')
    ax1.grid(axis='y')
    for i, v in enumerate(stall_ms):
        ax1.annotate(f"{v:.1f} ms", xy=(x[i], v), xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold', fontsize=9)

    ax2.bar(x, token_lat_ms, width, color='#2ca02c', edgecolor='black')
    ax2.set_ylabel('Mean Per-Token Latency (ms)', fontweight='bold')
    ax2.set_title('Simulated Per-Token Execution Time', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha='right', fontweight='bold')
    ax2.grid(axis='y')
    for i, v in enumerate(token_lat_ms):
        ax2.annotate(f"{v:.1f} ms\n({tokens_sec[i]:.1f} tok/s)", xy=(x[i], v), xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold', fontsize=9)

    plt.suptitle('Figure 3: Core Timing Comparison - Exposed Memory Stall & Token Latency', fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig3_core_exposed_stall_and_token_latency.png"), dpi=300)
    plt.savefig(os.path.join(PLOTS_DIR, "fig3_core_exposed_stall_and_token_latency.pdf"))
    plt.close()
    print("[PLOTTED] fig3_core_exposed_stall_and_token_latency")

def plot_fig4_expert_load_latency_distributions():
    summary_path = os.path.join(BASE_DIR, "results/core/cstp_h1/traces/experiment_summary.json")
    if not os.path.exists(summary_path):
        return
    with open(summary_path, 'r') as f:
        data = json.load(f)

    dist = data.get("expert_load_latency_distributions", {})
    categories = ['resident_hits', 'timely_prefetch', 'late_attachment', 'demand_miss']
    display_names = ['Resident Hits\n(HBM Hit)', 'Timely Prefetch\n(Arrived in HBM)', 'Late Attachment\n(In-Flight Demanded)', 'Fresh Demand\n(Full Miss)']
    
    means = []
    p50s = []
    p95s = []
    counts = []

    for cat in categories:
        info = dist.get(cat, {})
        counts.append(info.get('count', 0))
        means.append(info.get('mean_ns', 0.0) / 1e6)
        p50s.append(info.get('p50_ns', 0.0) / 1e6)
        p95s.append(info.get('p95_ns', 0.0) / 1e6)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(categories))
    width = 0.25

    ax.bar(x - width, means, width, label='Mean Latency (ms)', color='#386cb0', edgecolor='black')
    ax.bar(x, p50s, width, label='Median / P50 (ms)', color='#7fc97f', edgecolor='black')
    ax.bar(x + width, p95s, width, label='P95 Latency (ms)', color='#beaed4', edgecolor='black')

    ax.set_ylabel('Expert Load Latency (ms)', fontweight='bold')
    ax.set_title('Figure 4: Category-Separated Expert Load Latency Distributions\n(CSTP H=1, DDR5-64GB/s, 88.08 MB Expert Weight)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{name}\n(N={count})" for name, count in zip(display_names, counts)], fontweight='bold')
    ax.grid(axis='y')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig4_expert_load_latency_distributions.png"), dpi=300)
    plt.savefig(os.path.join(PLOTS_DIR, "fig4_expert_load_latency_distributions.pdf"))
    plt.close()
    print("[PLOTTED] fig4_expert_load_latency_distributions")

def plot_sensitivity_figures(sens_rows):
    # 1. Budget Sensitivity
    b_rows = [r for r in sens_rows if r['category'] == 'budget']
    if len(b_rows) > 0:
        b_rows.sort(key=lambda r: int(r['budget']))
        budgets = [r['budget'] for r in b_rows]
        wasted_mb = [float(r['wasted_speculative_bytes']) / (1024.0 * 1024.0) for r in b_rows]
        hit_rates = [float(r['unique_hbm_hit_rate']) * 100.0 for r in b_rows]

        fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
        ax2 = ax1.twinx()

        p1 = ax1.bar(budgets, wasted_mb, width=0.4, color='#e41a1c', alpha=0.8, edgecolor='black', label='Wasted Speculative Traffic (MB)')
        p2 = ax2.plot(budgets, hit_rates, color='#377eb8', marker='o', linewidth=2.5, markersize=8, label='Unique Hit Rate (%)')

        ax1.set_xlabel('Prefetch Candidate Budget (B)', fontweight='bold')
        ax1.set_ylabel('Wasted Memory Traffic (MB)', color='#e41a1c', fontweight='bold')
        ax2.set_ylabel('Unique HBM Hit Rate (%)', color='#377eb8', fontweight='bold')
        ax1.set_title('Figure 5: Candidate Budget Sensitivity Analysis\n(CSTP H=1, MoE-Anywhere, B=1, 4 Iterations)', fontweight='bold')
        ax1.grid(True)

        for rect in p1:
            h = rect.get_height()
            ax1.annotate(f"{h:.0f} MB", xy=(rect.get_x() + rect.get_width()/2.0, h), xytext=(0, 3),
                         textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "fig5_budget_sensitivity.png"), dpi=300)
        plt.savefig(os.path.join(PLOTS_DIR, "fig5_budget_sensitivity.pdf"))
        plt.close()
        print("[PLOTTED] fig5_budget_sensitivity")

    # 2. Batch Size Scaling
    bs_rows = [r for r in sens_rows if r['category'] == 'batch']
    if len(bs_rows) > 0:
        batches = [1, 2, 4]
        base_bytes = []
        cstp_bytes = []
        for b in batches:
            for r in bs_rows:
                if int(r['batch_size']) == b:
                    if r['predictor'] == 'none':
                        base_bytes.append(float(r['total_ddr5_bytes']) / (1024.0 ** 3))
                    elif r['predictor'] == 'CSTP':
                        cstp_bytes.append(float(r['total_ddr5_bytes']) / (1024.0 ** 3))

        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        x = np.arange(len(batches))
        width = 0.35

        ax.bar(x - width/2.0, base_bytes, width, label='Baseline Demand Traffic (GB)', color='#4daf4a', edgecolor='black')
        ax.bar(x + width/2.0, cstp_bytes, width, label='CSTP (H=1) Traffic (GB)', color='#984ea3', edgecolor='black')

        ax.set_xlabel('Batch Size (Sequences)', fontweight='bold')
        ax.set_ylabel('Total DDR5 Traffic (GB)', fontweight='bold')
        ax.set_title('Figure 6: Batch Size Scaling and Memory Traffic Demand\n(MoE-Anywhere Multi-Sequence Replay)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"B={b}" for b in batches], fontweight='bold')
        ax.grid(axis='y')
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "fig6_batch_size_scaling.png"), dpi=300)
        plt.savefig(os.path.join(PLOTS_DIR, "fig6_batch_size_scaling.pdf"))
        plt.close()
        print("[PLOTTED] fig6_batch_size_scaling")

    # 3. Cross-Dataset Comparison
    ds_rows = [r for r in sens_rows if r['category'] == 'cross_dataset']
    if len(ds_rows) > 0:
        datasets = ['moe_anywhere', 'moe_suite_12', 'moe_suite_13']
        d_labels = ['MoE-Anywhere', 'MoE-Suite Task 12', 'MoE-Suite Task 13']
        base_hits = []
        stmoe_hits = []
        cstp_hits = []

        for d in datasets:
            for r in ds_rows:
                if r['dataset'] == d:
                    if r['predictor'] == 'none':
                        base_hits.append(float(r['overall_hbm_hit_rate_proj']) * 100.0)
                    elif r['predictor'] == 'ST_MOE':
                        stmoe_hits.append(float(r['overall_hbm_hit_rate_proj']) * 100.0)
                    elif r['predictor'] == 'CSTP':
                        cstp_hits.append(float(r['overall_hbm_hit_rate_proj']) * 100.0)

        fig, ax = plt.subplots(figsize=(8.5, 5))
        x = np.arange(len(datasets))
        width = 0.25

        ax.bar(x - width, base_hits, width, label='Baseline Demand', color='#a6cee3', edgecolor='black')
        ax.bar(x, stmoe_hits, width, label='ST-MoE (H=1)', color='#1f78b4', edgecolor='black')
        ax.bar(x + width, cstp_hits, width, label='CSTP (H=1)', color='#b2df8a', edgecolor='black')

        ax.set_ylabel('Projection HBM Hit Rate (%)', fontweight='bold')
        ax.set_title('Figure 7: Cross-Dataset Hit Rate Evaluation\n(4 Iterations, B=1, 80 GB Cache)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(d_labels, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y')
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "fig7_cross_dataset_comparison.png"), dpi=300)
        plt.savefig(os.path.join(PLOTS_DIR, "fig7_cross_dataset_comparison.pdf"))
        plt.close()
        print("[PLOTTED] fig7_cross_dataset_comparison")

    # 4. Decode Length Evolution
    dec_rows = [r for r in sens_rows if r['category'] == 'decode_length']
    if len(dec_rows) > 0:
        iters = [2, 4, 8, 16]
        cstp_hits = []
        tokens_sec = []
        for it in iters:
            for r in dec_rows:
                if int(r['token_count']) == it and r['predictor'] == 'CSTP':
                    cstp_hits.append(float(r['overall_hbm_hit_rate_proj']) * 100.0)
                    tokens_sec.append(float(r['tokens_per_second']))

        fig, ax1 = plt.subplots(figsize=(8, 4.8))
        ax2 = ax1.twinx()

        l1 = ax1.plot(iters, cstp_hits, marker='s', color='#ff7f00', linewidth=2.5, markersize=8, label='HBM Hit Rate (%)')
        l2 = ax2.plot(iters, tokens_sec, marker='o', color='#377eb8', linewidth=2.5, markersize=8, label='Tokens / Second')

        ax1.set_xlabel('Decoding Iterations (Generated Tokens)', fontweight='bold')
        ax1.set_ylabel('HBM Hit Rate (%)', color='#ff7f00', fontweight='bold')
        ax2.set_ylabel('Throughput (Tokens / Sec)', color='#377eb8', fontweight='bold')
        ax1.set_title('Figure 8: Decoding Evolution and Steady-State Convergence\n(CSTP H=1, MoE-Anywhere, B=1)', fontweight='bold')
        ax1.set_xticks(iters)
        ax1.grid(True)

        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='white', framealpha=0.9)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "fig8_decode_length_evolution.png"), dpi=300)
        plt.savefig(os.path.join(PLOTS_DIR, "fig8_decode_length_evolution.pdf"))
        plt.close()
        print("[PLOTTED] fig8_decode_length_evolution")

    # 5. Constrained Cache Behavior
    cap_rows = [r for r in sens_rows if r['category'] == 'cache_size']
    if len(cap_rows) > 0:
        caps = [20, 30]
        base_stalls = []
        cstp_stalls = []
        for c in caps:
            for r in cap_rows:
                if f"cap{c}_baseline" in r['experiment']:
                    base_stalls.append(float(r['exposed_gpu_stall_ns']) / 1e6)
                elif f"cap{c}_cstp" in r['experiment']:
                    cstp_stalls.append(float(r['exposed_gpu_stall_ns']) / 1e6)

        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        x = np.arange(len(caps))
        width = 0.35

        ax.bar(x - width/2.0, base_stalls, width, label='Baseline Stall (ms)', color='#e41a1c', edgecolor='black')
        ax.bar(x + width/2.0, cstp_stalls, width, label='CSTP (H=1) Stall (ms)', color='#4daf4a', edgecolor='black')

        ax.set_xlabel('Cache Capacity (# Resident Experts)', fontweight='bold')
        ax.set_ylabel('Exposed GPU Stall Time (ms)', fontweight='bold')
        ax.set_title('Figure 9: Impact of Constrained Cache Capacity on Memory Stall\n(MoE-Anywhere, B=1, 4 Iterations)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c} Experts\n({c*88.08/1024:.2f} GB)" for c in caps], fontweight='bold')
        ax.grid(axis='y')
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "fig9_constrained_cache_behavior.png"), dpi=300)
        plt.savefig(os.path.join(PLOTS_DIR, "fig9_constrained_cache_behavior.pdf"))
        plt.close()
        print("[PLOTTED] fig9_constrained_cache_behavior")

def main():
    core_csv = os.path.join(BASE_DIR, "reports/consolidated_metrics.csv")
    core_rows = read_csv_rows(core_csv)
    if len(core_rows) > 0:
        plot_fig1_core_coverage_and_hit_rate(core_rows)
        plot_fig2_core_memory_traffic(core_rows)
        plot_fig3_core_exposed_stall_and_latency(core_rows)
        plot_fig4_expert_load_latency_distributions()

    sens_csv = os.path.join(BASE_DIR, "reports/sensitivity_metrics.csv")
    sens_rows = read_csv_rows(sens_csv)
    if len(sens_rows) > 0:
        plot_sensitivity_figures(sens_rows)

if __name__ == "__main__":
    main()

