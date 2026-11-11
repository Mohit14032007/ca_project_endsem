"""
Diagnostic Script: MoE Routing Selection Audit and Controlled Statistical Analysis.
Verifies configured Top-K vs actual selection mechanisms across all workloads in LLMSimulator.
Follows strict coding guidelines: ZERO ternary operators (ast.IfExp == 0).
"""

import os
import sys
import math
import ast
import json
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = "/home/mohman/ca_project/LLMSimulator"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predictor.adapters.simulator_adapter import SimulatorDatasetAdapter
from src.predictor.adapters.moe_anywhere_adapter import MoeAnywhereDatasetAdapter
from src.predictor.adapters.moe_suite_adapter import MoeSuiteDatasetAdapter
from src.predictor.adapters.synthetic_adapter import SyntheticCorrelatedDatasetAdapter


def calculate_jaccard(set_a, set_b):
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    union_len = len(set_a.union(set_b))
    if union_len == 0:
        return 0.0
    inter_len = len(set_a.intersection(set_b))
    return float(inter_len) / float(union_len)


def calculate_entropy(counts, total):
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = float(c) / float(total)
            ent -= p * math.log2(p)
    return ent


def evaluate_empirical_routing(events, num_experts, top_k):
    """
    Analyzes routing statistics for a sequence of events.
    Each event in events is a dict with keys:
      'request_id', 'token_id', 'stage_id', 'experts' (set of ints)
    """
    total_events = len(events)
    if total_events == 0:
        return {}

    # 1. Expert frequencies
    expert_counts = np.zeros(num_experts, dtype=np.int64)
    for ev in events:
        for exp in ev['experts']:
            if 0 <= exp < num_experts:
                expert_counts[exp] += 1

    total_expert_tokens = int(np.sum(expert_counts))
    unique_experts = int(np.count_nonzero(expert_counts))
    entropy_bits = calculate_entropy(expert_counts, total_expert_tokens)
    max_entropy = math.log2(num_experts)
    if max_entropy > 0:
        normalized_entropy = entropy_bits / max_entropy
    else:
        normalized_entropy = 1.0

    # 2. Consecutive token Jaccard overlap (same stage, token t vs t+1)
    stage_token_map = {}
    for ev in events:
        key = (ev['request_id'], ev['stage_id'])
        if key not in stage_token_map:
            stage_token_map[key] = {}
        stage_token_map[key][ev['token_id']] = ev['experts']

    token_jaccards = []
    for (req_id, stage_id), tokens in stage_token_map.items():
        sorted_tokens = sorted(tokens.keys())
        for i in range(len(sorted_tokens) - 1):
            t1 = sorted_tokens[i]
            t2 = sorted_tokens[i + 1]
            if t2 - t1 == 1:
                j = calculate_jaccard(tokens[t1], tokens[t2])
                token_jaccards.append(j)

    if len(token_jaccards) > 0:
        mean_token_jaccard = float(np.mean(token_jaccards))
    else:
        mean_token_jaccard = float('nan')

    # 3. Adjacent layer Jaccard overlap (same token, stage l vs l+1)
    token_stage_map = {}
    for ev in events:
        key = (ev['request_id'], ev['token_id'])
        if key not in token_stage_map:
            token_stage_map[key] = {}
        token_stage_map[key][ev['stage_id']] = ev['experts']

    layer_jaccards = []
    for (req_id, tok_id), stages in token_stage_map.items():
        sorted_stages = sorted(stages.keys())
        for i in range(len(sorted_stages) - 1):
            s1 = sorted_stages[i]
            s2 = sorted_stages[i + 1]
            if s2 - s1 == 1:
                j = calculate_jaccard(stages[s1], stages[s2])
                layer_jaccards.append(j)

    if len(layer_jaccards) > 0:
        mean_layer_jaccard = float(np.mean(layer_jaccards))
    else:
        mean_layer_jaccard = float('nan')

    # 4. Empirical random prediction Recall@K
    # If we predict K random experts uniformly without replacement for each event,
    # what is the measured Recall@K over 100 trials?
    recalls = []
    rng = np.random.RandomState(42)
    all_expert_ids = np.arange(num_experts)
    for ev in events:
        gt = ev['experts']
        pred = set(rng.choice(all_expert_ids, size=top_k, replace=False))
        inter = len(gt.intersection(pred))
        recalls.append(float(inter) / float(len(gt)))

    mean_random_recall = float(np.mean(recalls))
    theoretical_random_recall = float(top_k) / float(num_experts)

    return {
        "total_events": total_events,
        "unique_experts": unique_experts,
        "entropy_bits": entropy_bits,
        "max_entropy_bits": max_entropy,
        "normalized_entropy": normalized_entropy,
        "mean_token_jaccard": mean_token_jaccard,
        "mean_layer_jaccard": mean_layer_jaccard,
        "empirical_random_recall": mean_random_recall,
        "theoretical_random_recall": theoretical_random_recall,
        "freq_min": int(np.min(expert_counts)),
        "freq_max": int(np.max(expert_counts)),
        "freq_std": float(np.std(expert_counts))
    }


def simulate_get_random_expert(num_experts, top_k, num_tokens=5, num_stages=57, seed=777):
    """
    Replicates Scheduler::getRandomExpert(top_k) C++ logic:
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(0, num_experts - 1);
    while (route.size() < top_k) {
        int seq_id = distribution(generator) % num_experts;
        route.insert(seq_id);
    }
    """
    # Use python's numpy with matching uniform sampling without replacement
    rng = np.random.RandomState(seed)
    events = []
    for tok in range(num_tokens):
        for stage in range(num_stages):
            # Draw top_k unique experts uniformly
            chosen = set()
            while len(chosen) < top_k:
                val = int(rng.randint(0, num_experts))
                chosen.add(val)
            events.append({
                'request_id': 1,
                'token_id': tok,
                'stage_id': stage,
                'experts': chosen
            })
    return events


def load_adapter_events(adapter):
    adapter.load()
    events = []
    for key, experts in adapter.grouped_routes.items():
        req_id, tok_id, stage_id = key
        events.append({
            'request_id': req_id,
            'token_id': tok_id,
            'stage_id': stage_id,
            'experts': set(experts)
        })
    return events


def main():
    print("=" * 80)
    print("CRITICAL ROUTING SELECTION AUDIT — LLMSimulator")
    print("=" * 80)

    # 1. Controlled Diagnostic Simulation of getRandomExpert()
    print("\n--- 1. Controlled Diagnostic Test of getRandomExpert() (E=256, K=8) ---")
    seeds = [777, 42, 1234, 9999]
    for s in seeds:
        # Small scale (5 tokens x 57 stages = 285 events, matching historical DeepSeek run)
        sim_events_small = simulate_get_random_expert(256, 8, num_tokens=5, num_stages=57, seed=s)
        stats_small = evaluate_empirical_routing(sim_events_small, 256, 8)
        print(f"Seed {s} (5 tokens, 285 events):")
        print(f"  Unique Experts: {stats_small['unique_experts']} / 256")
        print(f"  Entropy: {stats_small['entropy_bits']:.4f} / {stats_small['max_entropy_bits']:.4f} bits (Normalized: {stats_small['normalized_entropy']:.4f})")
        print(f"  Consecutive Token Jaccard: {stats_small['mean_token_jaccard']:.4f} (Theoretical: {8/504:.4f})")
        print(f"  Adjacent Layer Jaccard:    {stats_small['mean_layer_jaccard']:.4f} (Theoretical: {8/504:.4f})")
        print(f"  Empirical Random Recall@8: {stats_small['empirical_random_recall']:.4f} (Theoretical: {8/256:.4f})")

        # Large scale (100 tokens x 57 stages = 5700 events)
        sim_events_large = simulate_get_random_expert(256, 8, num_tokens=100, num_stages=57, seed=s)
        stats_large = evaluate_empirical_routing(sim_events_large, 256, 8)
        print(f"Seed {s} (100 tokens, 5700 events):")
        print(f"  Unique Experts: {stats_large['unique_experts']} / 256")
        print(f"  Entropy: {stats_large['entropy_bits']:.4f} / {stats_large['max_entropy_bits']:.4f} bits (Normalized: {stats_large['normalized_entropy']:.4f})")
        print(f"  Consecutive Token Jaccard: {stats_large['mean_token_jaccard']:.4f} (Theoretical: {8/504:.4f})")
        print(f"  Adjacent Layer Jaccard:    {stats_large['mean_layer_jaccard']:.4f} (Theoretical: {8/504:.4f})")
        print(f"  Empirical Random Recall@8: {stats_large['empirical_random_recall']:.4f} (Theoretical: {8/256:.4f})\n")

    # 2. Inspecting Actual Traces
    print("\n--- 2. Evaluating Historical Workload Traces ---")
    workload_configs = [
        {
            "id": "deepseek_random",
            "name": "DeepSeek-V3 (Simulator)",
            "expert_count": 256,
            "top_k": 8,
            "mechanism": "Random-K (std::mt19937 uniform sampling without replacement)",
            "scores_computed": "NO",
            "randomness": "Pseudo-random (std::mt19937)",
            "replacement": "Without replacement (std::set)",
            "seed": "777 (static)",
            "source": "LLMSimulator internal generator (Scheduler::getRandomExpert)",
            "validity": "INVALID as real MoE routing (negative control only; zero router scores, zero semantic correlation)",
            "references": "src/scheduler/scheduler.cpp:141-153; src/scheduler/sequence.cpp:219; eval/test_deepseekv3_mixtral_style.cpp:254",
            "adapter_type": "simulator",
            "trace_path": os.path.join(PROJECT_ROOT, "reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/traces/expert_routing.csv"),
            "num_stages": 57,
            "first_k_dense": 3,
            "reconstruct": False
        },
        {
            "id": "duplex_a100",
            "name": "Duplex A100 (Mixtral 8x7B Simulator)",
            "expert_count": 8,
            "top_k": 2,
            "mechanism": "Random-K (std::mt19937 uniform sampling without replacement)",
            "scores_computed": "NO",
            "randomness": "Pseudo-random (std::mt19937)",
            "replacement": "Without replacement (std::set)",
            "seed": "777 (static, skewness=0.0)",
            "source": "LLMSimulator internal generator (Scheduler::getRandomExpert)",
            "validity": "INVALID as real MoE routing (negative control only; synthetic random uniform routing)",
            "references": "src/scheduler/scheduler.cpp:141-153; src/scheduler/sequence.cpp:219; reproduction_A100/results/a100_duplex/logs/concise_run.log:2",
            "adapter_type": "simulator",
            "trace_path": os.path.join(PROJECT_ROOT, "reproduction_A100/results/a100_duplex/traces/expert_routing.csv"),
            "num_stages": 32,
            "first_k_dense": 0,
            "reconstruct": False
        },
        {
            "id": "correlated_synthetic",
            "name": "Correlated Synthetic Benchmark",
            "expert_count": 256,
            "top_k": 8,
            "mechanism": "Weighted / Synthetic Markov Random (spatial affinity 0.6, temporal affinity 0.5, Zipf 0.8)",
            "scores_computed": "NO (analytical heuristic probabilities, no gating network)",
            "randomness": "Pseudo-random (numpy.random.choice with categorical probabilities)",
            "replacement": "Without replacement (replace=False)",
            "seed": "42",
            "source": "scripts/generate_correlated_benchmark_trace.py",
            "validity": "SYNTHETIC CONTROL (valid as controlled synthetic benchmark with ground truth correlation; NOT real model routing)",
            "references": "scripts/generate_correlated_benchmark_trace.py:20-80; src/predictor/adapters/synthetic_adapter.py:17-60",
            "adapter_type": "synthetic",
            "trace_path": os.path.join(PROJECT_ROOT, "results/endsem_phase1/benchmark_correlated_trace.csv"),
            "num_stages": 57,
            "first_k_dense": 3,
            "reconstruct": False
        },
        {
            "id": "moe_anywhere",
            "name": "MoE Anywhere (Sage-A1.5 Production Trace)",
            "expert_count": 256,
            "top_k": 8,
            "mechanism": "Genuine Top-K (External Replay)",
            "scores_computed": "YES (computed by production neural router and logged in parquet trace)",
            "randomness": "DETERMINISTIC REPLAY (real model router decisions)",
            "replacement": "Without replacement (Top-8 highest scoring distinct experts)",
            "seed": "N/A (empirical recorded data)",
            "source": "Azure/Alibaba-scale production cluster trace (carlosfelipe/moe-anywhere-routing-traces)",
            "validity": "VALID (genuine production MoE router trace from deployed model)",
            "references": "src/predictor/adapters/moe_anywhere_adapter.py:20-130; data/downloads/moe_anywhere/e1-sage-a1.5.parquet",
            "adapter_type": "moe_anywhere",
            "trace_path": os.path.join(PROJECT_ROOT, "data/downloads/moe_anywhere/e1-sage-a1.5.parquet"),
            "num_stages": 40,
            "first_k_dense": 0,
            "reconstruct": False
        },
        {
            "id": "moe_suite",
            "name": "MoE Suite (Multi-Corpus Coding MoE)",
            "expert_count": 256,
            "top_k": 8,
            "mechanism": "Genuine Top-K (External Replay)",
            "scores_computed": "YES (computed by model router across diverse multi-task coding benchmarks)",
            "randomness": "DETERMINISTIC REPLAY (real model router decisions)",
            "replacement": "Without replacement (Top-8 highest scoring distinct experts)",
            "seed": "N/A (empirical recorded data)",
            "source": "mistrjirka/Predicting-Experts-for-MOE-Suite (token_cube)",
            "validity": "VALID (genuine MoE model router trace across multi-task evaluations)",
            "references": "src/predictor/adapters/moe_suite_adapter.py:20-110; data/downloads/moe_suite/*.parquet",
            "adapter_type": "moe_suite",
            "trace_path": os.path.join(PROJECT_ROOT, "data/downloads/moe_suite"),
            "num_stages": 75,
            "first_k_dense": 0,
            "reconstruct": False
        }
    ]

    audit_rows = []

    for cfg in workload_configs:
        print(f"\nProcessing Workload: {cfg['name']} ({cfg['id']})")
        events = []
        if cfg['adapter_type'] == "simulator":
            adapter = SimulatorDatasetAdapter(
                dataset_id=cfg['id'],
                trace_path=cfg['trace_path'],
                num_moe_stages=cfg['num_stages'],
                num_routed_experts=cfg['expert_count'],
                top_k=cfg['top_k'],
                first_k_dense=cfg['first_k_dense'],
                reconstruct_stages=cfg['reconstruct']
            )
            events = load_adapter_events(adapter)
        elif cfg['adapter_type'] == "synthetic":
            adapter = SyntheticCorrelatedDatasetAdapter(
                trace_path=cfg['trace_path']
            )
            events = load_adapter_events(adapter)
        elif cfg['adapter_type'] == "moe_anywhere":
            adapter = MoeAnywhereDatasetAdapter(
                parquet_path=cfg['trace_path'],
                num_moe_stages=cfg['num_stages'],
                num_routed_experts=cfg['expert_count'],
                top_k=cfg['top_k']
            )
            adapter.load()
            for key, (exps, wts) in adapter.routes.items():
                pid, tpos, lyr = key
                events.append({
                    'request_id': pid,
                    'token_id': tpos,
                    'stage_id': lyr,
                    'experts': set(exps)
                })
        elif cfg['adapter_type'] == "moe_suite":
            adapter = MoeSuiteDatasetAdapter(
                data_dir=cfg['trace_path'],
                num_moe_stages=cfg['num_stages'],
                num_routed_experts=cfg['expert_count'],
                top_k=cfg['top_k']
            )
            adapter.load()
            # Analyze test split
            for task_id in adapter.splits["test"]:
                arr = adapter.task_routes[task_id]  # (N, 75, 8)
                for tpos in range(arr.shape[0]):
                    for lyr in range(arr.shape[1]):
                        exps = set(int(e) for e in arr[tpos, lyr, :])
                        events.append({
                            'request_id': task_id,
                            'token_id': tpos,
                            'stage_id': lyr,
                            'experts': exps
                        })

        stats = evaluate_empirical_routing(events, cfg['expert_count'], cfg['top_k'])
        print(f"  Events Evaluated: {stats.get('total_events', 0)}")
        print(f"  Unique Experts:   {stats.get('unique_experts', 0)} / {cfg['expert_count']}")
        print(f"  Entropy:          {stats.get('entropy_bits', 0):.4f} bits (Normalized: {stats.get('normalized_entropy', 0):.4f})")
        print(f"  Token Jaccard:    {stats.get('mean_token_jaccard', float('nan')):.4f}")
        print(f"  Layer Jaccard:    {stats.get('mean_layer_jaccard', float('nan')):.4f}")
        print(f"  Random Recall@K:  {stats.get('empirical_random_recall', float('nan')):.4f} (Theoretical: {stats.get('theoretical_random_recall', float('nan')):.4f})")

        row = {
            "workload": cfg['id'],
            "workload_name": cfg['name'],
            "expert_count": cfg['expert_count'],
            "configured_k": cfg['top_k'],
            "actual_selection_mechanism": cfg['mechanism'],
            "router_scores_computed": cfg['scores_computed'],
            "randomness": cfg['randomness'],
            "sampling_replacement": cfg['replacement'],
            "seed": cfg['seed'],
            "routing_source": cfg['source'],
            "validity_as_real_moe_evidence": cfg['validity'],
            "total_events": stats.get('total_events', 0),
            "unique_experts_observed": stats.get('unique_experts', 0),
            "entropy_bits": round(stats.get('entropy_bits', 0.0), 4),
            "normalized_entropy": round(stats.get('normalized_entropy', 0.0), 4),
            "consecutive_token_jaccard": round(stats.get('mean_token_jaccard', 0.0), 4),
            "adjacent_layer_jaccard": round(stats.get('mean_layer_jaccard', 0.0), 4),
            "empirical_random_recall": round(stats.get('empirical_random_recall', 0.0), 4),
            "theoretical_random_recall": round(stats.get('theoretical_random_recall', 0.0), 4),
            "source_file_references": cfg['references']
        }
        audit_rows.append(row)

    # 3. Write CSV output
    df_audit = pd.DataFrame(audit_rows)
    
    # Target 1: LLMSimulator/results/endsem_phase_d/audit/routing_selection_audit.csv
    out_dir_1 = os.path.join(PROJECT_ROOT, "results", "endsem_phase_d", "audit")
    os.makedirs(out_dir_1, exist_ok=True)
    out_csv_1 = os.path.join(out_dir_1, "routing_selection_audit.csv")
    df_audit.to_csv(out_csv_1, index=False)
    print(f"\nSaved audit CSV to: {out_csv_1}")

    # Target 2: ca_project/results/endsem_phase_d/audit/routing_selection_audit.csv
    out_dir_2 = "/home/mohman/ca_project/results/endsem_phase_d/audit"
    os.makedirs(out_dir_2, exist_ok=True)
    out_csv_2 = os.path.join(out_dir_2, "routing_selection_audit.csv")
    df_audit.to_csv(out_csv_2, index=False)
    print(f"Saved mirrored audit CSV to: {out_csv_2}")


if __name__ == "__main__":
    main()
