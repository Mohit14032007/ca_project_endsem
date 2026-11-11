"""
Main Execution Script for Phase 1 Offline Predictor Evaluation.

Executes 8 offline experiments:
- Exp 1: ST-MoE vs CSTP across prefetch budgets B in {1, 2, 4, 8, 16, 32}
- Exp 2: CSTP alpha sensitivity in [0.0, 1.0]
- Exp 3: CSTP lambda sensitivity in [0.1, 20.0]
- Exp 4: Marginal prior ablation (frequency-aware prior vs flat/no prior)
- Exp 5: Spatial-only vs Temporal-only vs Combined prediction
- Exp 6: Cold-start (token 0) vs established history (tokens >= 1)
- Exp 7: Layer-wise prediction performance across MoE stages
- Exp 8: Prefetch-oriented breakdown (Predicted, Actual, Missed, Unused, Coverage)

Outputs all structured CSV tables and JSON summaries into results/endsem_phase1/.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.predictor.trace_parser import (
    RoutingTraceParser,
    ModelArchitectureConfig,
    DEEPSEEK_V3_CONFIG,
    MIXTRAL_8X7B_CONFIG
)
from src.predictor.st_moe_predictor import STMoEPredictor
from src.predictor.cstp_predictor import CSTPPredictor
from src.predictor.evaluation import EvaluationHarness, PredictionResultRecord


def run_experiments(
    trace_path: str,
    output_dir: str,
    config: ModelArchitectureConfig,
    reconstruct_stages: bool = False
) -> Dict:
    """Executes all 8 experiments on the given trace and persists results."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n==================================================================")
    print(f"Loading and validating trace: {trace_path}")
    print(f"Architecture: {config.name} ({config.num_moe_stages} MoE stages, {config.num_routed_experts} experts, top-{config.top_k})")
    print(f"==================================================================")

    parser = RoutingTraceParser(config)
    records = parser.parse_file(trace_path, reconstruct_stages_from_sequence=reconstruct_stages)
    validation_report = parser.get_validation_report()

    print(f"Loaded {len(records)} normalized records.")
    print(f"Requests found: {validation_report['requests_found']}")
    print(f"Tokens found: {validation_report['tokens_found']}")
    if validation_report["modulo_60_drift_detected"]:
        print("[WARNING] Modulo-60 index drift detected in recorded layer_id.")
    if validation_report["is_single_request"]:
        print("[LIMITATION] Trace contains only 1 request ID. Request-level train/test split impossible.")

    harness = EvaluationHarness(num_moe_stages=config.num_moe_stages)

    # Split dataset
    if validation_report["is_single_request"]:
        # Temporal split on the single request (tokens 0..2 train, tokens 3..4 test)
        unique_tokens = sorted(list(set(r.token_id for r in records)))
        split_idx = max(1, int(len(unique_tokens) * 0.6))
        train_tokens = set(unique_tokens[:split_idx])
        test_tokens = set(unique_tokens[split_idx:])

        train_records = [r for r in records if r.token_id in train_tokens]
        test_records = [r for r in records if r.token_id in test_tokens]
        split_meta = {
            "total_requests": 1,
            "train_requests": [1],
            "test_requests": [1],
            "train_tokens": sorted(list(train_tokens)),
            "test_tokens": sorted(list(test_tokens)),
            "train_record_count": len(train_records),
            "test_record_count": len(test_records),
            "is_single_request": True
        }
    else:
        train_records, test_records, split_meta = harness.split_by_request(records, train_ratio=0.75)

    print(f"Split: {split_meta['train_record_count']} train records, {split_meta['test_record_count']} test records.")

    test_opportunities = harness.prepare_opportunities(test_records)
    print(f"Evaluation opportunities prepared: {len(test_opportunities)}")

    budgets = [1, 2, 4, 8, 16, 32]
    # Filter budgets that exceed total experts
    budgets = [b for b in budgets if b <= config.num_routed_experts]

    # Pre-train / profile models
    # 1. Profile ST-MoE on training records
    st_moe = STMoEPredictor(
        num_moe_stages=config.num_moe_stages,
        num_routed_experts=config.num_routed_experts,
        top_k=config.top_k,
        cct_capacity=8,
        confidence_threshold=2
    )
    # Calculate profiling priors for ST-MoE
    prior_counts = {}
    for stage in range(config.num_moe_stages):
        prior_counts[stage] = {}
    for r in train_records:
        if r.expert_id not in prior_counts[r.moe_stage_id]:
            prior_counts[r.moe_stage_id][r.expert_id] = 0
        prior_counts[r.moe_stage_id][r.expert_id] += 1

    st_priors = {}
    for stage in range(config.num_moe_stages):
        sorted_priors = sorted(
            prior_counts[stage].keys(),
            key=lambda e: (-prior_counts[stage][e], e)
        )
        st_priors[stage] = sorted_priors
    st_moe.set_profiling_prior(st_priors)

    # Profile ST-MoE CCT on training opportunities
    train_opportunities = harness.prepare_opportunities(train_records)
    for op in train_opportunities:
        st_moe.update(op.request_id, op.moe_stage_id, op.source_experts, op.actual_experts)
    st_moe.history_table.clear()

    # 2. Profile CSTP on training records
    cstp_default = CSTPPredictor(
        num_moe_stages=config.num_moe_stages,
        num_routed_experts=config.num_routed_experts,
        top_k=config.top_k,
        alpha=0.5,
        smoothing_lambda=1.0,
        prior_epsilon=0.1
    )
    cstp_default.fit_profile(train_records)

    # -------------------------------------------------------------
    # Experiment 1: ST-MoE versus CSTP at identical budgets
    # -------------------------------------------------------------
    print("\n--- Running Experiment 1: ST-MoE vs CSTP across Budgets ---")
    exp1_rows = []

    # No prediction baseline
    no_pred_results = harness.evaluate_predictor(None, "NoPrediction", test_opportunities, budgets)

    # ST-MoE
    st_moe_results = harness.evaluate_predictor(st_moe, "ST-MoE", test_opportunities, budgets)

    # CSTP
    cstp_results = harness.evaluate_predictor(cstp_default, "CSTP", test_opportunities, budgets)

    for b in budgets:
        # No prediction
        np_sub = [r for r in no_pred_results if r.budget == b]
        np_metrics = harness.compute_summary_metrics(np_sub)
        np_row = {
            "predictor": "NoPrediction",
            "budget": b,
            "macro_precision": np_metrics.get("macro_precision", 0.0),
            "macro_recall": np_metrics.get("macro_recall", 0.0),
            "macro_f1": np_metrics.get("macro_f1", 0.0),
            "hit_rate": np_metrics.get("hit_rate", 0.0),
            "micro_precision": np_metrics.get("micro_precision", 0.0),
            "micro_recall": np_metrics.get("micro_recall", 0.0),
            "coverage": np_metrics.get("coverage", 0.0)
        }
        exp1_rows.append(np_row)

        # ST-MoE
        st_sub = [r for r in st_moe_results if r.budget == b]
        st_metrics = harness.compute_summary_metrics(st_sub)
        st_row = {
            "predictor": "ST-MoE",
            "budget": b,
            "macro_precision": st_metrics.get("macro_precision", 0.0),
            "macro_recall": st_metrics.get("macro_recall", 0.0),
            "macro_f1": st_metrics.get("macro_f1", 0.0),
            "hit_rate": st_metrics.get("hit_rate", 0.0),
            "micro_precision": st_metrics.get("micro_precision", 0.0),
            "micro_recall": st_metrics.get("micro_recall", 0.0),
            "coverage": st_metrics.get("coverage", 0.0)
        }
        exp1_rows.append(st_row)

        # CSTP
        cstp_sub = [r for r in cstp_results if r.budget == b]
        cstp_metrics = harness.compute_summary_metrics(cstp_sub)
        cstp_row = {
            "predictor": "CSTP",
            "budget": b,
            "macro_precision": cstp_metrics.get("macro_precision", 0.0),
            "macro_recall": cstp_metrics.get("macro_recall", 0.0),
            "macro_f1": cstp_metrics.get("macro_f1", 0.0),
            "hit_rate": cstp_metrics.get("hit_rate", 0.0),
            "micro_precision": cstp_metrics.get("micro_precision", 0.0),
            "micro_recall": cstp_metrics.get("micro_recall", 0.0),
            "coverage": cstp_metrics.get("coverage", 0.0)
        }
        exp1_rows.append(cstp_row)

    df_exp1 = pd.DataFrame(exp1_rows)
    df_exp1.to_csv(os.path.join(output_dir, "experiment1_stmoe_vs_cstp.csv"), index=False)
    print("Saved experiment1_stmoe_vs_cstp.csv")

    # -------------------------------------------------------------
    # Experiment 2: CSTP alpha sensitivity
    # -------------------------------------------------------------
    print("\n--- Running Experiment 2: CSTP Alpha Sensitivity ---")
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    exp2_rows = []
    eval_budget = min(config.top_k, config.num_routed_experts)

    for alpha_val in alphas:
        cstp_alpha = CSTPPredictor(
            num_moe_stages=config.num_moe_stages,
            num_routed_experts=config.num_routed_experts,
            top_k=config.top_k,
            alpha=alpha_val,
            smoothing_lambda=1.0,
            prior_epsilon=0.1
        )
        cstp_alpha.fit_profile(train_records)
        res = harness.evaluate_predictor(cstp_alpha, f"CSTP_a{alpha_val}", test_opportunities, [eval_budget])
        m = harness.compute_summary_metrics(res)
        row = {
            "alpha": alpha_val,
            "budget": eval_budget,
            "macro_precision": m.get("macro_precision", 0.0),
            "macro_recall": m.get("macro_recall", 0.0),
            "macro_f1": m.get("macro_f1", 0.0),
            "hit_rate": m.get("hit_rate", 0.0),
            "micro_precision": m.get("micro_precision", 0.0),
            "micro_recall": m.get("micro_recall", 0.0),
            "coverage": m.get("coverage", 0.0)
        }
        exp2_rows.append(row)

    df_exp2 = pd.DataFrame(exp2_rows)
    df_exp2.to_csv(os.path.join(output_dir, "experiment2_alpha_sensitivity.csv"), index=False)
    print("Saved experiment2_alpha_sensitivity.csv")

    # -------------------------------------------------------------
    # Experiment 3: CSTP lambda sensitivity
    # -------------------------------------------------------------
    print("\n--- Running Experiment 3: CSTP Lambda Sensitivity ---")
    lambdas = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0]
    exp3_rows = []

    for lam_val in lambdas:
        cstp_lam = CSTPPredictor(
            num_moe_stages=config.num_moe_stages,
            num_routed_experts=config.num_routed_experts,
            top_k=config.top_k,
            alpha=0.5,
            smoothing_lambda=lam_val,
            prior_epsilon=0.1
        )
        cstp_lam.fit_profile(train_records)
        res = harness.evaluate_predictor(cstp_lam, f"CSTP_lam{lam_val}", test_opportunities, [eval_budget])
        m = harness.compute_summary_metrics(res)
        row = {
            "lambda": lam_val,
            "budget": eval_budget,
            "macro_precision": m.get("macro_precision", 0.0),
            "macro_recall": m.get("macro_recall", 0.0),
            "macro_f1": m.get("macro_f1", 0.0),
            "hit_rate": m.get("hit_rate", 0.0),
            "coverage": m.get("coverage", 0.0)
        }
        exp3_rows.append(row)

    df_exp3 = pd.DataFrame(exp3_rows)
    df_exp3.to_csv(os.path.join(output_dir, "experiment3_lambda_sensitivity.csv"), index=False)
    print("Saved experiment3_lambda_sensitivity.csv")

    # -------------------------------------------------------------
    # Experiment 4: Target-layer frequency prior ablation
    # -------------------------------------------------------------
    print("\n--- Running Experiment 4: Target-Layer Prior Ablation ---")
    exp4_rows = []
    prior_variants = [
        ("With_Frequency_Prior", 0.1),
        ("High_Smoothing_Prior", 10.0),
        ("Near_Zero_Epsilon", 1e-6)
    ]

    for label, eps_val in prior_variants:
        cstp_prior = CSTPPredictor(
            num_moe_stages=config.num_moe_stages,
            num_routed_experts=config.num_routed_experts,
            top_k=config.top_k,
            alpha=0.5,
            smoothing_lambda=1.0,
            prior_epsilon=eps_val
        )
        cstp_prior.fit_profile(train_records)
        res = harness.evaluate_predictor(cstp_prior, label, test_opportunities, [eval_budget])
        m = harness.compute_summary_metrics(res)
        row = {
            "prior_variant": label,
            "epsilon": eps_val,
            "budget": eval_budget,
            "macro_precision": m.get("macro_precision", 0.0),
            "macro_recall": m.get("macro_recall", 0.0),
            "macro_f1": m.get("macro_f1", 0.0),
            "hit_rate": m.get("hit_rate", 0.0),
            "coverage": m.get("coverage", 0.0)
        }
        exp4_rows.append(row)

    df_exp4 = pd.DataFrame(exp4_rows)
    df_exp4.to_csv(os.path.join(output_dir, "experiment4_prior_ablation.csv"), index=False)
    print("Saved experiment4_prior_ablation.csv")

    # -------------------------------------------------------------
    # Experiment 5: Spatial-only vs Temporal-only vs Combined
    # -------------------------------------------------------------
    print("\n--- Running Experiment 5: Spatial vs Temporal vs Combined ---")
    exp5_rows = []
    modes = [
        ("Spatial_Only", 1.0),
        ("Temporal_Only", 0.0),
        ("Combined_SpatioTemporal", 0.5)
    ]

    for mode_name, a_val in modes:
        cstp_mode = CSTPPredictor(
            num_moe_stages=config.num_moe_stages,
            num_routed_experts=config.num_routed_experts,
            top_k=config.top_k,
            alpha=a_val,
            smoothing_lambda=1.0,
            prior_epsilon=0.1
        )
        cstp_mode.fit_profile(train_records)
        res = harness.evaluate_predictor(cstp_mode, mode_name, test_opportunities, budgets)
        for b in budgets:
            sub = [r for r in res if r.budget == b]
            m = harness.compute_summary_metrics(sub)
            row = {
                "mode": mode_name,
                "alpha": a_val,
                "budget": b,
                "macro_precision": m.get("macro_precision", 0.0),
                "macro_recall": m.get("macro_recall", 0.0),
                "macro_f1": m.get("macro_f1", 0.0),
                "hit_rate": m.get("hit_rate", 0.0),
                "coverage": m.get("coverage", 0.0)
            }
            exp5_rows.append(row)

    df_exp5 = pd.DataFrame(exp5_rows)
    df_exp5.to_csv(os.path.join(output_dir, "experiment5_spatial_temporal_ablation.csv"), index=False)
    print("Saved experiment5_spatial_temporal_ablation.csv")

    # -------------------------------------------------------------
    # Experiment 6: Cold-start vs Established history
    # -------------------------------------------------------------
    print("\n--- Running Experiment 6: Cold-Start vs Established History ---")
    exp6_rows = []
    for pred_label, res_list in [("ST-MoE", st_moe_results), ("CSTP", cstp_results)]:
        for b in budgets:
            sub = [r for r in res_list if r.budget == b]

            cold_sub = [r for r in sub if r.token_position == 0]
            warm_sub = [r for r in sub if r.token_position > 0]

            m_cold = harness.compute_summary_metrics(cold_sub)
            m_warm = harness.compute_summary_metrics(warm_sub)

            row_cold = {
                "predictor": pred_label,
                "budget": b,
                "regime": "Cold_Start_Token_0",
                "instance_count": len(cold_sub),
                "macro_precision": m_cold.get("macro_precision", 0.0),
                "macro_recall": m_cold.get("macro_recall", 0.0),
                "macro_f1": m_cold.get("macro_f1", 0.0),
                "hit_rate": m_cold.get("hit_rate", 0.0),
                "coverage": m_cold.get("coverage", 0.0)
            }
            row_warm = {
                "predictor": pred_label,
                "budget": b,
                "regime": "Established_History_Token_Ge_1",
                "instance_count": len(warm_sub),
                "macro_precision": m_warm.get("macro_precision", 0.0),
                "macro_recall": m_warm.get("macro_recall", 0.0),
                "macro_f1": m_warm.get("macro_f1", 0.0),
                "hit_rate": m_warm.get("hit_rate", 0.0),
                "coverage": m_warm.get("coverage", 0.0)
            }
            exp6_rows.append(row_cold)
            exp6_rows.append(row_warm)

    df_exp6 = pd.DataFrame(exp6_rows)
    df_exp6.to_csv(os.path.join(output_dir, "experiment6_cold_start.csv"), index=False)
    print("Saved experiment6_cold_start.csv")

    # -------------------------------------------------------------
    # Experiment 7: Prediction performance across MoE stages
    # -------------------------------------------------------------
    print("\n--- Running Experiment 7: Performance Across MoE Stages ---")
    exp7_rows = []
    for stage_idx in range(config.num_moe_stages):
        raw_layer = stage_idx + config.first_k_dense

        # CSTP at budget = top_k
        cstp_stage_sub = [r for r in cstp_results if r.budget == eval_budget and r.moe_stage_id == stage_idx]
        m_cstp = harness.compute_summary_metrics(cstp_stage_sub)

        # ST-MoE at budget = top_k
        st_stage_sub = [r for r in st_moe_results if r.budget == eval_budget and r.moe_stage_id == stage_idx]
        m_st = harness.compute_summary_metrics(st_stage_sub)

        row = {
            "moe_stage_id": stage_idx,
            "raw_transformer_layer_id": raw_layer,
            "budget": eval_budget,
            "cstp_macro_recall": m_cstp.get("macro_recall", 0.0),
            "cstp_macro_precision": m_cstp.get("macro_precision", 0.0),
            "cstp_hit_rate": m_cstp.get("hit_rate", 0.0),
            "st_moe_macro_recall": m_st.get("macro_recall", 0.0),
            "st_moe_macro_precision": m_st.get("macro_precision", 0.0),
            "st_moe_hit_rate": m_st.get("hit_rate", 0.0)
        }
        exp7_rows.append(row)

    df_exp7 = pd.DataFrame(exp7_rows)
    df_exp7.to_csv(os.path.join(output_dir, "experiment7_per_stage.csv"), index=False)
    print("Saved experiment7_per_stage.csv")

    # -------------------------------------------------------------
    # Experiment 8: Prefetch-oriented breakdown across budgets
    # -------------------------------------------------------------
    print("\n--- Running Experiment 8: Prefetch-Oriented Breakdown ---")
    exp8_rows = []
    for pred_label, res_list in [("ST-MoE", st_moe_results), ("CSTP", cstp_results)]:
        for b in budgets:
            sub = [r for r in res_list if r.budget == b]
            m = harness.compute_summary_metrics(sub)
            row = {
                "predictor": pred_label,
                "budget": b,
                "total_opportunities": m.get("total_instances", 0),
                "total_predicted_experts": m.get("total_predicted", 0),
                "total_actual_experts": m.get("total_actual", 0),
                "total_hits": m.get("total_hits", 0),
                "total_missed": m.get("total_missed", 0),
                "total_unused_prefetches": m.get("total_unused", 0),
                "coverage": m.get("coverage", 0.0),
                "micro_precision": m.get("micro_precision", 0.0),
                "micro_recall": m.get("micro_recall", 0.0)
            }
            exp8_rows.append(row)

    df_exp8 = pd.DataFrame(exp8_rows)
    df_exp8.to_csv(os.path.join(output_dir, "experiment8_budget_scaling.csv"), index=False)
    print("Saved experiment8_budget_scaling.csv")

    summary_manifest = {
        "trace_evaluated": trace_path,
        "model_name": config.name,
        "num_moe_stages": config.num_moe_stages,
        "num_routed_experts": config.num_routed_experts,
        "top_k": config.top_k,
        "split_metadata": split_meta,
        "validation_report": validation_report,
        "output_directory": output_dir,
        "note": "Offline evaluation: Dynamic interconnect stall latency and prefetch lead time require cycle-level execution timeline; timestamps are omitted to prevent fabrication."
    }

    with open(os.path.join(output_dir, "summary_report.json"), "w") as fp:
        json.dump(summary_manifest, fp, indent=2)

    print(f"\nAll 8 experiments completed successfully for {config.name}.")
    print(f"Results stored under: {output_dir}\n")
    return summary_manifest


def main():
    parser = argparse.ArgumentParser(description="Evaluate offline ST-MoE and CSTP predictors.")
    parser.add_argument("--trace", type=str, default=None, help="Path to routing trace CSV")
    parser.add_argument("--dataset_type", type=str, default="both", choices=["deepseekv3", "duplex", "both"])
    args = parser.parse_args()

    results_base = os.path.join(project_root, "results", "endsem_phase1")

    if args.dataset_type in ["deepseekv3", "both"]:
        ds_trace = os.path.join(
            project_root,
            "reproduction_A100",
            "with_off_chip_dram",
            "deepseekv3",
            "experiments",
            "deepseekv3_mixtral_style_20261011",
            "full_run_5step_20261011",
            "traces",
            "expert_routing.csv"
        )
        if os.path.exists(ds_trace):
            ds_out = os.path.join(results_base, "deepseekv3_5step")
            run_experiments(
                trace_path=ds_trace,
                output_dir=ds_out,
                config=DEEPSEEK_V3_CONFIG,
                reconstruct_stages=True
            )
        else:
            print(f"[WARN] DeepSeek-V3 trace not found at {ds_trace}")

    if args.dataset_type in ["duplex", "both"]:
        duplex_trace = os.path.join(
            project_root,
            "reproduction_A100",
            "results",
            "a100_duplex",
            "traces",
            "expert_routing.csv"
        )
        if os.path.exists(duplex_trace):
            duplex_out = os.path.join(results_base, "duplex_a100_32req")
            run_experiments(
                trace_path=duplex_trace,
                output_dir=duplex_out,
                config=MIXTRAL_8X7B_CONFIG,
                reconstruct_stages=False
            )
        else:
            print(f"[WARN] Duplex A100 trace not found at {duplex_trace}")


if __name__ == "__main__":
    main()
