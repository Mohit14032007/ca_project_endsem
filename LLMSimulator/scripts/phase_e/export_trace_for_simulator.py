#!/usr/bin/env python3
"""
Export Real MoE Routing Traces for LLMSimulator.

Extracts Top-K expert routing decisions from official test splits of:
1. MoE Suite (corpus-12-typescript-migration-planner, corpus-13-go-fenced-leases)
2. MoE Anywhere (e1-sage-a1.5.parquet, test prompts 60..73)

Enforces:
- Strict split isolation (held-out test split only).
- Zero fallback to random routing.
- Preservation of original token order, layer order, and Top-K ranking.
- Support for independent multi-sequence batches.
- Zero ternary operators (ast.IfExp == 0).
"""

import argparse
import os
import sys
import pyarrow.parquet as pq
import numpy as np


def export_moe_suite(task_ids, num_layers, num_tokens, output_path):
    base_dir = "LLMSimulator/data/downloads/moe_suite"
    if not os.path.exists(base_dir):
        base_dir = "data/downloads/moe_suite"

    rows = []
    # Header: seq_id,token_id,layer_id,topk_rank,expert_id
    rows.append("seq_id,token_id,layer_id,topk_rank,expert_id\n")

    seq_idx = 0
    for task_id in task_ids:
        parquet_path = os.path.join(base_dir, f"{task_id}.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"MoE Suite task file not found: {parquet_path}")

        table = pq.read_table(parquet_path, columns=["route_selected"])
        # route_selected: (N_tokens, 75, 8)
        raw_routes = table["route_selected"].to_pylist()
        total_tokens = len(raw_routes)

        if num_tokens > total_tokens:
            raise ValueError(f"Requested {num_tokens} tokens, but {task_id} only has {total_tokens} tokens")

        for token_id in range(num_tokens):
            token_routes = raw_routes[token_id]
            for layer_id in range(num_layers):
                stage_routes = token_routes[layer_id]
                top_k = len(stage_routes)
                for rank_idx in range(top_k):
                    exp_id = stage_routes[rank_idx]
                    topk_rank = rank_idx + 1
                    rows.append(f"{seq_idx},{token_id},{layer_id},{topk_rank},{exp_id}\n")
        seq_idx += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(rows)

    total_records = len(rows) - 1
    print(f"[EXPORT_MOE_SUITE] Exported {total_records} records across {len(task_ids)} sequences to {output_path}")


def export_moe_anywhere(prompt_ids, num_layers, num_tokens, output_path):
    parquet_path = "LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet"
    if not os.path.exists(parquet_path):
        parquet_path = "data/downloads/moe_anywhere/e1-sage-a1.5.parquet"
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"MoE Anywhere parquet file not found: {parquet_path}")

    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    rows = []
    rows.append("seq_id,token_id,layer_id,topk_rank,expert_id\n")

    seq_idx = 0
    for pid in prompt_ids:
        pdf = df[df["prompt_id"] == pid]
        if len(pdf) == 0:
            raise ValueError(f"Prompt ID {pid} not found in MoE Anywhere dataset")

        unique_tokens = sorted(list(pdf["token_pos"].unique()))
        total_tokens = len(unique_tokens)

        if num_tokens > total_tokens:
            raise ValueError(f"Requested {num_tokens} tokens, but prompt {pid} only has {total_tokens} tokens")

        tokens_to_use = unique_tokens[:num_tokens]

        for token_rel_idx in range(len(tokens_to_use)):
            tpos = tokens_to_use[token_rel_idx]
            for layer_id in range(num_layers):
                row_match = pdf[(pdf["token_pos"] == tpos) & (pdf["layer"] == layer_id)]
                if len(row_match) == 0:
                    raise ValueError(f"Missing routing record for prompt {pid}, token {tpos}, layer {layer_id}")

                experts = row_match["experts"].iloc[0]
                top_k = len(experts)
                for rank_idx in range(top_k):
                    exp_id = experts[rank_idx]
                    topk_rank = rank_idx + 1
                    rows.append(f"{seq_idx},{token_rel_idx},{layer_id},{topk_rank},{exp_id}\n")
        seq_idx += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(rows)

    total_records = len(rows) - 1
    print(f"[EXPORT_MOE_ANYWHERE] Exported {total_records} records across {len(prompt_ids)} sequences to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export real MoE routing traces for LLMSimulator")
    parser.add_argument("--dataset", type=str, required=True, choices=["moe_suite", "moe_anywhere"])
    parser.add_argument("--tasks", type=str, default="corpus-12-typescript-migration-planner")
    parser.add_argument("--prompts", type=str, default="60")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    if args.dataset == "moe_suite":
        task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
        export_moe_suite(task_list, args.num_layers, args.tokens, args.output)
    else:
        prompt_list = [int(p.strip()) for p in args.prompts.split(",") if p.strip()]
        export_moe_anywhere(prompt_list, args.num_layers, args.tokens, args.output)


if __name__ == "__main__":
    main()

