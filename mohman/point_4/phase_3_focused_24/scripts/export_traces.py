#!/usr/bin/env python3
"""
Trace Exporter for Point 4 Phase 3 Focused 24-Run Evaluation.
Exports authentic routing traces from MoE-Anywhere (e1-sage-a1.5.parquet)
for:
  - W2: B=4, T=16, L=40, Top-K=8 (Prompts 60, 61, 62, 63)
  - W3: B=14, T=16, L=40, Top-K=8 (Prompts 60 through 73)
Strictly enforces ast.IfExp == 0 (zero ternary operators).
"""

import os
import sys
import ast
import pyarrow.parquet as pq
import pandas as pd

WORKSPACE = "/home/mohman/ca_project"
PARQUET_PATH = os.path.join(
    WORKSPACE, "LLMSimulator", "data", "downloads", "moe_anywhere", "e1-sage-a1.5.parquet"
)
PHASE3_ROOT = os.path.join(WORKSPACE, "mohman", "point_4", "phase_3_focused_24")
TRACES_DIR = os.path.join(PHASE3_ROOT, "traces")
MANIFEST_DIR = os.path.join(PHASE3_ROOT, "manifests")


def verify_zero_ternary(file_path):
    with open(file_path, "r") as f:
        tree = ast.parse(f.read(), filename=file_path)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            count += 1
    if count > 0:
        raise ValueError(f"Found {count} ternary operators in {file_path}!")
    print(f"[AST_CHECK] Verified 0 ternary operators in {file_path}")


def get_prompt_split(pid):
    if pid >= 60 and pid <= 73:
        return "test"
    if pid >= 52 and pid <= 59:
        return "validation"
    return "train"


def export_workload_trace(df, workload_id, prompt_ids, num_stages, num_tokens):
    output_csv = os.path.join(TRACES_DIR, f"{workload_id}.csv")
    print(f"[EXPORTING] {workload_id}: {len(prompt_ids)} sequences, {num_tokens} tokens, {num_stages} stages...")

    p_sub = df[df["prompt_id"].isin(prompt_ids)].copy()
    lookup = {}
    task_map = {}
    for row in p_sub.itertuples():
        lookup[(row.prompt_id, row.token_pos, row.layer)] = row.experts
        task_map[row.prompt_id] = row.task

    rows = ["seq_id,token_id,layer_id,topk_rank,expert_id\n"]
    manifest_rows = []

    for seq_idx, pid in enumerate(prompt_ids):
        split = get_prompt_split(pid)
        task = task_map.get(pid, "unknown")

        pdf = p_sub[p_sub["prompt_id"] == pid]
        unique_tokens = sorted(list(pdf["token_pos"].unique()))
        total_available = len(unique_tokens)

        if num_tokens > total_available:
            raise ValueError(f"Prompt {pid} only has {total_available} tokens, requested {num_tokens}")

        tokens_to_use = unique_tokens[:num_tokens]
        min_tpos = min(unique_tokens)

        manifest_rows.append({
            "workload_id": workload_id,
            "seq_id": seq_idx,
            "prompt_id": pid,
            "split": split,
            "task": task,
            "min_token_pos": min_tpos,
            "num_tokens": num_tokens,
            "num_stages": num_stages,
            "top_k": 8
        })

        for t_rel_idx, tpos in enumerate(tokens_to_use):
            for stage in range(num_stages):
                experts = lookup.get((pid, tpos, stage), None)
                if experts is None:
                    raise ValueError(f"Missing routing for pid={pid}, tpos={tpos}, stage={stage}")
                for rank_idx in range(len(experts)):
                    exp_id = int(experts[rank_idx])
                    topk_rank = rank_idx + 1
                    rows.append(f"{seq_idx},{t_rel_idx},{stage},{topk_rank},{exp_id}\n")

    with open(output_csv, "w") as f:
        f.writelines(rows)

    expected_rows = len(prompt_ids) * num_tokens * num_stages * 8
    actual_rows = len(rows) - 1
    if actual_rows != expected_rows:
        raise RuntimeError(f"Row count mismatch: expected {expected_rows}, got {actual_rows}")

    print(f"  [SUCCESS] Written {actual_rows} routing rows to {output_csv}")
    return manifest_rows


def main():
    verify_zero_ternary(__file__)
    os.makedirs(TRACES_DIR, exist_ok=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)

    print(f"[LOADING] Loading parquet dataset from {PARQUET_PATH}...")
    table = pq.read_table(PARQUET_PATH, columns=["prompt_id", "task", "token_pos", "layer", "experts"])
    df = table.to_pandas()

    w2_prompts = [60, 61, 62, 63]
    w3_prompts = list(range(60, 74))

    all_manifest = []
    w2_meta = export_workload_trace(df, "w2", w2_prompts, num_stages=40, num_tokens=16)
    all_manifest.extend(w2_meta)

    w3_meta = export_workload_trace(df, "w3", w3_prompts, num_stages=40, num_tokens=16)
    all_manifest.extend(w3_meta)

    manifest_df = pd.DataFrame(all_manifest)
    manifest_csv = os.path.join(MANIFEST_DIR, "workload_prompt_mapping.csv")
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"[MANIFEST] Exported prompt mapping manifest to {manifest_csv}")


if __name__ == "__main__":
    main()

