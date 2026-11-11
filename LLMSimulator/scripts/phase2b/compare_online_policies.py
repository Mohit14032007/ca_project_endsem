"""
Phase 2B: Comparative Evaluation of Online Adaptation Policies.

Compares:
1. online_across_requests (preserves persistent CCT updates across requests)
2. online_within_request (restores profiling CCT before each request)
3. frozen (zero updates, profiling baseline reference)

Across identical test opportunities and datasets.
"""

import os
import sys
import json
import time
from typing import Any, Dict, List, Set, Tuple
import numpy as np
import pandas as pd

sim_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
workspace_root = os.path.dirname(sim_root) if os.path.basename(sim_root) == "LLMSimulator" else sim_root
if os.path.basename(sim_root) != "LLMSimulator":
    sim_root = os.path.join(workspace_root, "LLMSimulator")

sys.path.insert(0, sim_root)

from src.predictor.st_moe_paper import STMoEPaperPredictor
from src.predictor.evaluation import (
    EvaluationHarness,
    PredictionOpportunity,
    PredictionResultRecord
)
from src.predictor.adapters.moe_anywhere_adapter import MoeAnywhereDatasetAdapter
from src.predictor.adapters.moe_suite_adapter import MoeSuiteDatasetAdapter
from src.predictor.adapters.simulator_adapter import SimulatorDatasetAdapter
from src.predictor.adapters.synthetic_adapter import SyntheticCorrelatedDatasetAdapter
from scripts.phase2a.run_stmoe_isolated import (
    compute_co_matrices_moe_anywhere,
    compute_co_matrices_moe_suite,
    compute_co_matrices_generic,
    extract_test_opportunities_anywhere,
    extract_test_opportunities_suite,
    extract_test_opportunities_generic,
    populate_cct_from_co_matrix
)


def run_comparison():
    out_dir = os.path.join(workspace_root, "results", "phase2b_evaluation_semantics")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("PHASE 2B: ONLINE ADAPTATION POLICIES COMPARISON")
    print(f"Results Directory: {out_dir}")
    print("=" * 80)

    # 1. Load Adapters
    print("\nLoading Dataset Adapters...")
    p_anywhere = os.path.join(sim_root, "data", "downloads", "moe_anywhere", "e1-sage-a1.5.parquet")
    adapter_anywhere = MoeAnywhereDatasetAdapter(p_anywhere)
    adapter_anywhere.load()

    d_suite = os.path.join(sim_root, "data", "downloads", "moe_suite")
    adapter_suite = MoeSuiteDatasetAdapter(d_suite)
    adapter_suite.load()

    p_duplex = os.path.join(sim_root, "reproduction_A100", "results", "a100_duplex", "traces", "expert_routing.csv")
    adapter_duplex = SimulatorDatasetAdapter(
        dataset_id="duplex_a100",
        trace_path=p_duplex,
        num_moe_stages=32,
        num_routed_experts=8,
        top_k=2,
        first_k_dense=0,
        train_ratio=0.75
    )
    adapter_duplex.load()

    p_synth = os.path.join(sim_root, "results", "endsem_phase1", "benchmark_correlated_trace.csv")
    adapter_synth = SyntheticCorrelatedDatasetAdapter(p_synth)
    adapter_synth.load()

    workloads = [
        ("moe_anywhere", adapter_anywhere, 1, 8, "authoritative"),
        ("moe_anywhere", adapter_anywhere, 2, 8, "authoritative"),
        ("moe_suite", adapter_suite, 2, 8, "authoritative"),
        ("moe_suite", adapter_suite, 1, 8, "authoritative"),
        ("duplex_a100", adapter_duplex, 1, 2, "diagnostic"),
        ("correlated_synthetic", adapter_synth, 1, 8, "diagnostic")
    ]

    rows = []

    for name, adapter, horizon, top_k, classification in workloads:
        print(f"\n--- Workload: {name} (H={horizon}, Class={classification}) ---")

        if name == "moe_anywhere":
            train_units = set(adapter.get_splits()["train"])
            test_opps, eval_tokens = extract_test_opportunities_anywhere(adapter, horizon, max_tokens_per_prompt=30)
            co_matrices, train_tokens = compute_co_matrices_moe_anywhere(adapter, train_units, horizon)
            num_stages = adapter.num_moe_stages
            num_experts = adapter.num_routed_experts
        elif name == "moe_suite":
            train_units = set(adapter.splits["train"])
            test_opps, eval_tokens = extract_test_opportunities_suite(adapter, horizon, max_tokens_per_task=50)
            co_matrices, train_tokens = compute_co_matrices_moe_suite(adapter, train_units, horizon)
            num_stages = adapter.num_moe_stages
            num_experts = adapter.num_routed_experts
        else:
            num_stages = getattr(adapter, "num_moe_stages", getattr(adapter, "num_stages", 57))
            num_experts = getattr(adapter, "num_routed_experts", getattr(adapter, "num_experts", 256))
            train_units = set(adapter.splits["train"])
            test_opps, eval_tokens = extract_test_opportunities_generic(adapter, horizon)
            co_matrices, train_tokens = compute_co_matrices_generic(adapter, train_units, horizon)

        unique_test_requests = sorted(list(set(op.request_id for op in test_opps)))
        test_req_count = len(unique_test_requests)
        total_opps_count = len(test_opps)

        print(f"  Test Requests/Prompts: {test_req_count}")
        print(f"  Total Opportunities:   {total_opps_count}")

        harness = EvaluationHarness(num_moe_stages=num_stages)
        num_transitions = max(1, num_stages - horizon)

        def make_fresh():
            pred = STMoEPaperPredictor(
                num_moe_stages=num_stages,
                num_routed_experts=num_experts,
                top_k=top_k,
                cct_capacity=top_k,
                confidence_threshold=2,
                online_adaptation=True,
                prediction_horizon=horizon,
                scoring_mode="confidence_additive"
            )
            for s in range(num_transitions):
                populate_cct_from_co_matrix(pred, s, co_matrices[s])
            return pred

        # Evaluate across the three policies
        policies = ["frozen", "online_within_request", "online_across_requests"]
        policy_summaries = {}

        for pol in policies:
            pred_inst = make_fresh()
            res = harness.evaluate_predictor(
                predictor_instance=pred_inst,
                predictor_name=f"st_moe_{pol}",
                test_opportunities=test_opps,
                budgets=[top_k],
                evaluation_mode=pol
            )
            policy_summaries[pol] = harness.compute_summary_metrics(res)

        sm_froz = policy_summaries["frozen"]
        sm_within = policy_summaries["online_within_request"]
        sm_across = policy_summaries["online_across_requests"]

        comp_row = {
            "dataset_id": name,
            "classification": classification,
            "horizon": horizon,
            "top_k": top_k,
            "test_requests": test_req_count,
            "test_opportunities": total_opps_count,
            # Frozen Reference
            "frozen_recall": round(sm_froz["macro_recall"], 4),
            "frozen_precision": round(sm_froz["macro_precision"], 4),
            "frozen_hit_rate": round(sm_froz["hit_rate"], 4),
            # Online Within-Request
            "within_request_recall": round(sm_within["macro_recall"], 4),
            "within_request_precision": round(sm_within["macro_precision"], 4),
            "within_request_hit_rate": round(sm_within["hit_rate"], 4),
            "within_request_cold_recall": round(sm_within["cold_start"]["recall"], 4),
            "within_request_steady_recall": round(sm_within["steady_state"]["recall"], 4),
            # Online Across-Requests
            "across_requests_recall": round(sm_across["macro_recall"], 4),
            "across_requests_precision": round(sm_across["macro_precision"], 4),
            "across_requests_hit_rate": round(sm_across["hit_rate"], 4),
            "across_requests_cold_recall": round(sm_across["cold_start"]["recall"], 4),
            "across_requests_steady_recall": round(sm_across["steady_state"]["recall"], 4),
            # Delta (Across - Within)
            "delta_recall_pp": round((sm_across["macro_recall"] - sm_within["macro_recall"]) * 100, 2),
            "delta_hit_rate_pp": round((sm_across["hit_rate"] - sm_within["hit_rate"]) * 100, 2),
        }
        rows.append(comp_row)
        print(f"  [Within-Request]  Recall={comp_row['within_request_recall']:.4f}, HitRate={comp_row['within_request_hit_rate']:.4f}")
        print(f"  [Across-Requests] Recall={comp_row['across_requests_recall']:.4f}, HitRate={comp_row['across_requests_hit_rate']:.4f}")
        print(f"  Delta: {comp_row['delta_recall_pp']:+.2f} percentage points")

    df = pd.DataFrame(rows)
    p_csv = os.path.join(out_dir, "policy_comparison_results.csv")
    df.to_csv(p_csv, index=False)
    print(f"\nSaved Policy Comparison Table -> {p_csv}")


if __name__ == "__main__":
    run_comparison()

