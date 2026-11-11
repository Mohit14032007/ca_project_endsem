"""
Generates a calibrated multi-request synthetic routing trace with realistic
spatial correlation, temporal persistence, and Zipfian popularity.
Used to validate predictor learning behavior in the presence of real correlation.
"""

import os
import sys
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)


def generate_benchmark_trace(
    output_path: str,
    num_requests: int = 16,
    num_tokens: int = 5,
    num_stages: int = 57,
    num_experts: int = 256,
    top_k: int = 8,
    first_k_dense: int = 3,
    spatial_affinity: float = 0.6,
    temporal_affinity: float = 0.5,
    seed: int = 42
) -> None:
    """Generates synthetic multi-request trace with spatio-temporal structure."""
    np.random.seed(seed)
    rows = []

    # Create Zipfian popularity distribution over experts
    zipf_weights = 1.0 / np.power(np.arange(1, num_experts + 1), 0.8)
    base_probs = zipf_weights / np.sum(zipf_weights)

    # Spatial cluster mapping: experts partitioned into 16 thematic clusters of 16 experts
    cluster_size = 16
    num_clusters = num_experts // cluster_size

    for req_id in range(1, num_requests + 1):
        # Each request has an affinity cluster
        req_cluster = (req_id - 1) % num_clusters
        prev_token_routed = {}

        for tok_idx in range(num_tokens):
            tok_id = 1000 + tok_idx

            for stage_idx in range(num_stages):
                raw_layer = stage_idx + first_k_dense

                # Compute sampling probability for this stage
                prob = base_probs.copy()

                # Boost cluster affinity (spatial thematic bias)
                c_start = req_cluster * cluster_size
                c_end = c_start + cluster_size
                prob[c_start:c_end] += spatial_affinity / float(cluster_size)

                # Temporal boost: if previous token routed experts at this stage, boost them
                if stage_idx in prev_token_routed:
                    prev_set = prev_token_routed[stage_idx]
                    for pe in prev_set:
                        prob[pe] += temporal_affinity

                # Normalize probability
                prob = prob / np.sum(prob)

                # Draw top_k unique experts
                selected = np.random.choice(
                    num_experts,
                    size=top_k,
                    replace=False,
                    p=prob
                )
                prev_token_routed[stage_idx] = set(selected)

                for rank_idx, exp_id in enumerate(selected):
                    row = {
                        "request_id": req_id,
                        "token_id": tok_id,
                        "layer_id": stage_idx,
                        "topk_rank": rank_idx + 1,
                        "expert_id": int(exp_id)
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated correlated benchmark trace at {output_path} ({len(df)} rows).")


if __name__ == "__main__":
    out_file = os.path.join(
        project_root,
        "results",
        "endsem_phase1",
        "benchmark_correlated_trace.csv"
    )
    generate_benchmark_trace(out_file)

