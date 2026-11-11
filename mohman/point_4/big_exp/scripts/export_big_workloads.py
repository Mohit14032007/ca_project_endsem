#!/usr/bin/env python3
"""
Big Experiment Trace Exporter: W3 (B=14, T=128) and W4 (B=32, T=128)
Extracts authentic routing traces from MoE-Anywhere (e1-sage-a1.5.parquet)
Enforces:
- Zero synthetic data
- Zero prompt duplication
- Full 40 MoE stages
- Zero ternary operators (AST verified)
"""

import os
import sys
import ast
import pandas as pd
import pyarrow.parquet as pq

WORKSPACE = "/home/mohman/ca_project"
PARQUET_PATH = os.path.join(
    WORKSPACE, "LLMSimulator", "data", "downloads", "moe_anywhere", "e1-sage-a1.5.parquet"
)
OUTPUT_DIR = os.path.join(WORKSPACE, "mohman", "point_4", "big_exp", "traces")


def get_prompt_split(pid):
    if pid >= 60 and pid <= 73:
        return "test"
    elif pid >= 52 and pid <= 59:
        return "validation"
    else:
        return "train"


def verify_zero_ternary(file_path):
    with open(file_path, "r") as f:
        tree = ast.parse(f.read())
    ternary_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.IfExp)]
    if len(ternary_nodes) > 0:
        raise ValueError(f"Found {len(ternary_nodes)} ternary operators in {file_path}")
    print(f"[AST_CHECK] Verified 0 ternary operators in {file_path}")


def export_trace(df, workload_id, prompt_ids, num_stages, num_tokens):
    output_csv = os.path.join(OUTPUT_DIR, f"{workload_id}.csv")
    print(f"\n[EXPORT] Exporting {workload_id}: {len(prompt_ids)} sequences, {num_tokens} tokens, {num_stages} stages...")

    sub_df = df[df["prompt_id"].isin(prompt_ids)].copy()
    lookup = {}
    task_map = {}
    for row in sub_df.itertuples():
        lookup[(row.prompt_id, row.token_pos, row.layer)] = row.experts
        task_map[row.prompt_id] = row.task

    rows = []
    rows.append("seq_id,token_id,layer_id,topk_rank,expert_id\n")

    manifest_records = []
    for seq_idx, pid in enumerate(prompt_ids):
        split = get_prompt_split(pid)
        task = task_map.get(pid, "unknown")

        pdf = sub_df[sub_df["prompt_id"] == pid]
        unique_tokens = sorted(list(pdf["token_pos"].unique()))
        total_avail = len(unique_tokens)

        if num_tokens > total_avail:
            raise ValueError(f"Prompt {pid} has only {total_avail} tokens, needed {num_tokens}")

        tokens_to_use = unique_tokens[:num_tokens]
        min_pos = min(unique_tokens)
        max_pos = tokens_to_use[-1]

        manifest_records.append({
            "workload_id": workload_id,
            "seq_id": seq_idx,
            "prompt_id": pid,
            "split": split,
            "task": task,
            "prefill_len": min_pos,
            "first_token_pos": min_pos,
            "last_token_pos": max_pos,
            "num_tokens": num_tokens
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_csv, "w") as f:
        f.writelines(rows)

    total_records = len(rows) - 1
    expected_records = len(prompt_ids) * num_tokens * num_stages * 8
    if total_records != expected_records:
        raise RuntimeError(f"Record count mismatch: got {total_records}, expected {expected_records}")

    print(f"[SUCCESS] Exported {workload_id} ({total_records} routing records) -> {output_csv}")
    return manifest_records


def main():
    verify_zero_ternary(__file__)
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Missing parquet file: {PARQUET_PATH}")

    table = pq.read_table(PARQUET_PATH)
    df = table.to_pandas()

    p_w3 = list(range(60, 74))                                   # 14 test prompts
    p_w4 = list(range(60, 74)) + list(range(52, 60)) + list(range(0, 10))  # 14 test + 8 val + 10 train = 32

    all_manifest = []
    m_w3 = export_trace(df, "w3_b14_t128", p_w3, 40, 128)
    all_manifest.extend(m_w3)

    m_w4 = export_trace(df, "w4_b32_t128", p_w4, 40, 128)
    all_manifest.extend(m_w4)

    manifest_df = pd.DataFrame(all_manifest)
    manifest_path = os.path.join(OUTPUT_DIR, "trace_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\n[MANIFEST] Saved trace manifest ({len(all_manifest)} sequences) -> {manifest_path}")


if __name__ == "__main__":
    main()

