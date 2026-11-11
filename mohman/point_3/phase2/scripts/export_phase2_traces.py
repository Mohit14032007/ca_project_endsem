#!/usr/bin/env python3
"""
Trace Exporter for Point 3 Phase 2 Extended Prefetching Campaign.
Extracts real MoE routing traces from MoE-Anywhere (e1-sage-a1.5.parquet)
for certified held-out test prompts (prompts 60..73).

Supports:
- Arbitrary number of MoE stages (e.g. 15, 20, 40)
- Arbitrary token counts (e.g. 2, 4, 16, 32, 64, 128)
- Arbitrary batch sizes (B = 1, 2, 4, 8) using distinct held-out prompts
- 100% trace fidelity: Top-8 rankings, exact expert IDs, zero random fallback
- STRICT RULE: Zero ternary operators (ast.IfExp == 0).
"""

import os
import sys
import argparse
import pyarrow.parquet as pq
import pandas as pd


def export_traces(parquet_path, output_path, prompt_ids, num_stages, num_tokens):
    """
    Exports routing records for specified prompts, stages, and tokens.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    rows = []
    rows.append("seq_id,token_id,layer_id,topk_rank,expert_id\n")

    seq_idx = 0
    for pid in prompt_ids:
        pdf = df[df["prompt_id"] == pid]
        if len(pdf) == 0:
            raise ValueError(f"Prompt ID {pid} not found in parquet dataset")

        # Sort uniquely by token_pos to obtain consistent sequential ordering
        unique_tokens = sorted(list(pdf["token_pos"].unique()))
        total_available_tokens = len(unique_tokens)

        if num_tokens > total_available_tokens:
            raise ValueError(
                f"Requested {num_tokens} tokens, but prompt {pid} only has {total_available_tokens} tokens"
            )

        tokens_to_use = unique_tokens[:num_tokens]

        # Pre-filter for the tokens and stages needed to ensure maximum export performance
        pdf_filtered = pdf[
            (pdf["token_pos"].isin(tokens_to_use)) & (pdf["layer"] < num_stages)
        ].copy()
        
        # Sort by token_pos ascending, then layer ascending
        pdf_sorted = pdf_filtered.sort_values(by=["token_pos", "layer"])

        # Map token_pos to relative token_id (0, 1, 2, ...)
        tpos_to_rel = {}
        for idx, tpos in enumerate(tokens_to_use):
            tpos_to_rel[tpos] = idx

        for _, row in pdf_sorted.iterrows():
            tpos = row["token_pos"]
            layer = row["layer"]
            token_rel_idx = tpos_to_rel[tpos]
            experts = row["experts"]
            top_k = len(experts)
            for rank_idx in range(top_k):
                exp_id = int(experts[rank_idx])
                topk_rank = rank_idx + 1
                rows.append(
                    f"{seq_idx},{token_rel_idx},{layer},{topk_rank},{exp_id}\n"
                )

        seq_idx += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(rows)

    total_records = len(rows) - 1
    expected_records = len(prompt_ids) * num_tokens * num_stages * 8
    if total_records != expected_records:
        raise RuntimeError(
            f"Record count mismatch: got {total_records}, expected {expected_records}"
        )

    print(
        f"[EXPORT_SUCCESS] Exported {total_records} records to {output_path} "
        f"(Seqs={len(prompt_ids)}, Tokens={num_tokens}, Stages={num_stages}, Top-K=8)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export real MoE traces from MoE-Anywhere parquet."
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default="/home/mohman/ca_project/LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet",
        help="Path to e1-sage-a1.5.parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="60",
        help="Comma-separated list of prompt IDs (e.g. '60' or '60,61,62,63')",
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        default=40,
        help="Number of MoE stages to export (0 to num_stages - 1)",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=4,
        help="Number of tokens to export per sequence",
    )

    args = parser.parse_args()

    prompt_list = [int(p.strip()) for p in args.prompts.split(",") if p.strip()]
    export_traces(
        parquet_path=args.parquet,
        output_path=args.output,
        prompt_ids=prompt_list,
        num_stages=args.num_stages,
        num_tokens=args.num_tokens,
    )


if __name__ == "__main__":
    main()

