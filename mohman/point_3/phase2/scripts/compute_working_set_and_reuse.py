#!/usr/bin/env python3
"""
Working Set and Expert Reuse Distance Profiler for MoE-Anywhere Held-Out Split.
Analyzes exact demand streams across 40 MoE stages, calculating:
1. Working set growth (unique experts, resident bytes) vs token count (1..128)
2. Working set vs batch size (B = 1, 2, 4, 8)
3. Expert reuse-distance distributions (intervening demand events between successive accesses)
4. Cache residency requirements for hit generation.

Enforces ZERO ternary operators (ast.IfExp == 0).
Optimized with O(1) hash table indexing.
"""

import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WORKSPACE = "/home/mohman/ca_project"
PARQUET_PATH = os.path.join(
    WORKSPACE, "LLMSimulator", "data", "downloads", "moe_anywhere", "e1-sage-a1.5.parquet"
)
OUTPUT_DIR = os.path.join(WORKSPACE, "mohman", "point_3", "phase2", "results")
EXPERT_BYTES = 88080384  # 84.00 MiB / 88.08 MB FP16


def compute_working_set_scaling(df, test_prompts):
    records = []
    token_steps = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
    batch_sizes = [1, 2, 4, 8]

    # Pre-build lookup dictionary: (prompt_id, token_pos) -> list of (layer, experts)
    p_sub = df[df["prompt_id"].isin(test_prompts[:8])]
    grouped = {}
    for row in p_sub.itertuples():
        key = (row.prompt_id, row.token_pos)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append((row.layer, row.experts))

    # Pre-collect sorted unique token_pos per prompt
    prompt_tokens = {}
    for pid in test_prompts[:8]:
        pdf = p_sub[p_sub["prompt_id"] == pid]
        prompt_tokens[pid] = sorted(pdf["token_pos"].unique())

    for b in batch_sizes:
        prompts = test_prompts[:b]
        for t_count in token_steps:
            seen_tuples = set()
            total_demands = 0
            for pid in prompts:
                tpos_list = prompt_tokens[pid][:t_count]
                for tpos in tpos_list:
                    items = grouped.get((pid, tpos), [])
                    for layer, experts in items:
                        for e in experts:
                            seen_tuples.add((layer, int(e)))
                            total_demands += 1

            unique_tuples = len(seen_tuples)
            ws_bytes = unique_tuples * EXPERT_BYTES
            ws_gib = ws_bytes / (1024**3)
            ws_gb = ws_bytes / (1000**3)

            records.append({
                "batch_size": b,
                "token_count": t_count,
                "total_demand_accesses": total_demands,
                "unique_stage_expert_bundles": unique_tuples,
                "working_set_bytes": ws_bytes,
                "working_set_gib": ws_gib,
                "working_set_gb": ws_gb,
            })

    out_csv = os.path.join(OUTPUT_DIR, "working_set_scaling.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    res_df = pd.DataFrame(records)
    res_df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Saved working set scaling curve to {out_csv}")
    return res_df


def compute_reuse_distances(df, test_prompts):
    records = []
    token_count = 64

    # Build O(1) hash map: (pid, token_pos, layer) -> experts
    lookup = {}
    prompt_tokens = {}
    p_sub = df[df["prompt_id"].isin(test_prompts[:8])]
    for row in p_sub.itertuples():
        lookup[(row.prompt_id, row.token_pos, row.layer)] = row.experts

    for pid in test_prompts[:8]:
        pdf = p_sub[p_sub["prompt_id"] == pid]
        prompt_tokens[pid] = sorted(pdf["token_pos"].unique())[:token_count]

    for b in [1, 8]:
        prompts = test_prompts[:b]
        demand_stream = []

        # Reconstruct execution order: Token -> Stage -> Sequences in Batch -> Experts -> Projections
        for t_idx in range(token_count):
            for stage in range(40):
                for pid in prompts:
                    tpos = prompt_tokens[pid][t_idx]
                    experts = lookup.get((pid, tpos, stage), None)
                    if experts is not None:
                        for e in experts:
                            tuple_key = (stage, int(e))
                            demand_stream.append(tuple_key)
                            demand_stream.append(tuple_key)
                            demand_stream.append(tuple_key)

        last_seen = {}
        distances_intra = []
        distances_inter_token = []
        all_distances = []

        for event_idx, tup in enumerate(demand_stream):
            if tup in last_seen:
                dist = event_idx - last_seen[tup]
                all_distances.append(dist)
                if dist <= 3:
                    distances_intra.append(dist)
                else:
                    distances_inter_token.append(dist)
            last_seen[tup] = event_idx

        all_arr = np.array(all_distances)
        inter_arr = np.array(distances_inter_token)

        records.append({
            "batch_size": b,
            "total_demand_events": len(demand_stream),
            "total_reuse_events": len(all_distances),
            "intra_step_sister_reuse_count": len(distances_intra),
            "inter_token_reuse_count": len(distances_inter_token),
            "all_reuse_p50": float(np.percentile(all_arr, 50)),
            "all_reuse_mean": float(np.mean(all_arr)),
            "all_reuse_p95": float(np.percentile(all_arr, 95)),
            "all_reuse_max": int(np.max(all_arr)),
            "inter_token_p50": float(np.percentile(inter_arr, 50)),
            "inter_token_mean": float(np.mean(inter_arr)),
            "inter_token_p95": float(np.percentile(inter_arr, 95)),
            "inter_token_max": int(np.max(inter_arr)),
        })

    out_csv = os.path.join(OUTPUT_DIR, "reuse_distance_distribution.csv")
    res_df = pd.DataFrame(records)
    res_df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Saved reuse distance distribution to {out_csv}")
    return res_df


def main():
    print("======================================================================")
    print("WORKING SET & REUSE DISTANCE ANALYSIS (40 MOE STAGES)")
    print("======================================================================")
    table = pq.read_table(PARQUET_PATH)
    df = table.to_pandas()
    test_prompts = list(range(60, 74))

    compute_working_set_scaling(df, test_prompts)
    compute_reuse_distances(df, test_prompts)


if __name__ == "__main__":
    main()
