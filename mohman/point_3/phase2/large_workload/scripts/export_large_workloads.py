#!/usr/bin/env python3
"""
Large Workload Trace Exporter for Point 3 Phase 2.
Exports real MoE routing traces from MoE-Anywhere (e1-sage-a1.5.parquet)
for target workloads W1 through W7 across up to 70 independent prompts.

Enforces:
- Strict split provenance tracking (test, validation, train).
- Zero fallback to random routing.
- Zero ternary operators (ast.IfExp == 0).
- Full 40 MoE stages.
- No repeated or synthetic routing records.
"""

import os
import sys
import argparse
import pyarrow.parquet as pq
import pandas as pd

WORKSPACE = "/home/mohman/ca_project"
PARQUET_PATH = os.path.join(
    WORKSPACE, "LLMSimulator", "data", "downloads", "moe_anywhere", "e1-sage-a1.5.parquet"
)
LARGE_WORKLOAD_ROOT = os.path.join(WORKSPACE, "mohman", "point_3", "phase2", "large_workload")
TRACES_DIR = os.path.join(LARGE_WORKLOAD_ROOT, "experiments", "traces")
MANIFEST_DIR = os.path.join(LARGE_WORKLOAD_ROOT, "experiments", "manifests")


def get_prompt_split(pid):
    if pid >= 60 and pid <= 73:
        return "test"
    elif pid >= 52 and pid <= 59:
        return "validation"
    else:
        return "train"


def export_single_workload(df, workload_id, prompt_ids, num_stages, num_tokens):
    output_csv = os.path.join(TRACES_DIR, f"{workload_id}.csv")
    print(f"\n--- Exporting {workload_id} ({len(prompt_ids)} seqs, {num_tokens} tokens, {num_stages} stages) ---")

    # Build O(1) hash map: (prompt_id, token_pos, layer) -> experts
    p_sub = df[df["prompt_id"].isin(prompt_ids)].copy()
    lookup = {}
    task_map = {}
    for row in p_sub.itertuples():
        lookup[(row.prompt_id, row.token_pos, row.layer)] = row.experts
        task_map[row.prompt_id] = row.task

    rows = []
    rows.append("seq_id,token_id,layer_id,topk_rank,expert_id\n")

    mapping_records = []
    for seq_idx, pid in enumerate(prompt_ids):
        split = get_prompt_split(pid)
        task = task_map.get(pid, "unknown")

        pdf = p_sub[p_sub["prompt_id"] == pid]
        unique_tokens = sorted(list(pdf["token_pos"].unique()))
        total_available = len(unique_tokens)

        if num_tokens > total_available:
            raise ValueError(
                f"Requested {num_tokens} tokens, but prompt {pid} only has {total_available} tokens"
            )

        tokens_to_use = unique_tokens[:num_tokens]
        min_tpos = min(unique_tokens)

        mapping_records.append({
            "workload_id": workload_id,
            "seq_id": seq_idx,
            "prompt_id": pid,
            "split": split,
            "task": task,
            "prefill_len": min_tpos,
            "num_tokens": num_tokens,
        })

        for t_rel_idx, tpos in enumerate(tokens_to_use):
            for stage in range(num_stages):
                experts = lookup.get((pid, tpos, stage), None)
                if experts is None:
                    raise ValueError(f"Missing routing record for pid={pid}, tpos={tpos}, stage={stage}")

                top_k = len(experts)
                for rank_idx in range(top_k):
                    exp_id = int(experts[rank_idx])
                    topk_rank = rank_idx + 1
                    rows.append(f"{seq_idx},{t_rel_idx},{stage},{topk_rank},{exp_id}\n")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w") as f:
        f.writelines(rows)

    total_records = len(rows) - 1
    expected_records = len(prompt_ids) * num_tokens * num_stages * 8
    if total_records != expected_records:
        raise RuntimeError(
            f"Record mismatch: got {total_records}, expected {expected_records}"
        )

    print(f"[SUCCESS] Exported {total_records} records to {output_csv}")
    return mapping_records


def main():
    print("======================================================================")
    print("LARGE WORKLOAD TRACE EXPORTER (W1 THROUGH W7)")
    print("======================================================================")

    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Parquet file not found: {PARQUET_PATH}")

    table = pq.read_table(PARQUET_PATH)
    df = table.to_pandas()

    test_prompts = list(range(60, 74))         # 14 test prompts
    val_prompts = list(range(52, 60))           # 8 validation prompts
    train_prompts = list(range(0, 48))          # 48 train prompts

    prompts_b14 = test_prompts                  # 14 test prompts (Pure test)
    prompts_b32 = test_prompts + val_prompts + train_prompts[:10]   # 14 + 8 + 10 = 32
    prompts_b48 = test_prompts + val_prompts + train_prompts[:26]   # 14 + 8 + 26 = 48
    prompts_b70 = test_prompts + val_prompts + train_prompts        # 14 + 8 + 48 = 70

    workloads = [
        ("W1", prompts_b14, 40, 4),
        ("W2", prompts_b14, 40, 16),
        ("W3", prompts_b32, 40, 16),
        ("W4", prompts_b48, 40, 16),
        ("W5", prompts_b70, 40, 16),
        ("W6", prompts_b70, 40, 32),
        ("W7", prompts_b70, 40, 64),
    ]

    all_mappings = []
    for wid, plist, stages, toks in workloads:
        maps = export_single_workload(df, wid, plist, stages, toks)
        all_mappings.extend(maps)

    os.makedirs(MANIFEST_DIR, exist_ok=True)
    map_csv = os.path.join(MANIFEST_DIR, "workload_prompt_mapping.csv")
    pd.DataFrame(all_mappings).to_csv(map_csv, index=False)
    print(f"\n[SUCCESS] Saved complete workload prompt mapping to {map_csv}")
    print("======================================================================")


if __name__ == "__main__":
    main()

