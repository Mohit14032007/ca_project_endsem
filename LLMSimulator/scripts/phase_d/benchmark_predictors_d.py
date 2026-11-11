"""
Comprehensive Host-Side Predictor Latency and Memory Overhead Benchmarking for Phase D.
Measures query latency, update latency, initialization latency, component breakdowns,
and memory scaling across expert counts, stage counts, and concurrent requests.
"""

import os
import sys
import copy
import time
import tracemalloc
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.predictor.cstp_predictor import CSTPPredictor
from src.predictor.st_moe_paper import STMoEPaperPredictor
from src.predictor.st_moe_predictor import STMoEPredictor
from src.predictor.acsh_predictor import ACSHPredictor
from src.predictor.base_predictor import (
    BasePredictor,
    CSTPWrapper,
    STMoELegacyWrapper,
    RandomPredictor,
    FrequencyPredictor,
    TemporalOnlyPredictor,
    SpatialOnlyPredictor,
    CSTPSTMoEHybridPredictor
)


def compute_timing_stats(latencies_ns: List[int]) -> Dict[str, float]:
    arr_us = np.array(latencies_ns, dtype=np.float64) / 1000.0
    return {
        "mean_us": round(float(np.mean(arr_us)), 3),
        "median_us": round(float(np.median(arr_us)), 3),
        "std_us": round(float(np.std(arr_us)), 3),
        "p95_us": round(float(np.percentile(arr_us, 95)), 3),
        "p99_us": round(float(np.percentile(arr_us, 99)), 3),
        "max_us": round(float(np.max(arr_us)), 3),
        "min_us": round(float(np.min(arr_us)), 3),
        "num_calls": len(latencies_ns)
    }


def run_benchmark():
    print("=" * 80)
    print("STARTING PHASE D PREDICTOR BENCHMARK (LATENCY & MEMORY)")
    print("=" * 80)

    latency_dir = os.path.join(project_root, "results", "endsem_phase_d", "latency")
    memory_dir = os.path.join(project_root, "results", "endsem_phase_d", "memory")
    os.makedirs(latency_dir, exist_ok=True)
    os.makedirs(memory_dir, exist_ok=True)

    num_stages = 57
    num_experts = 256
    top_k = 8
    budgets = [4, 8, 16, 24, 32, 48, 64]
    n_warmup = 100
    n_timed = 1000

    np.random.seed(42)

    class DummyRecord:
        def __init__(self, req, tok, stage, exp):
            self.request_id = req
            self.token_id = tok
            self.moe_stage_id = stage
            self.expert_id = exp

    profile_records = []
    for t in range(5):
        for s in range(num_stages):
            for e in np.random.choice(num_experts, size=top_k, replace=False):
                profile_records.append(DummyRecord(1, t, s, int(e)))

    # -------------------------------------------------------------
    # 1. Measure Initialization & Profiling Latency
    # -------------------------------------------------------------
    print("\n1. Measuring Initialization and Profiling Latency...")
    init_rows = []

    # CSTP
    t_list = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        cstp = CSTPPredictor(num_moe_stages=num_stages, num_routed_experts=num_experts, top_k=top_k)
        cstp.fit_profile(profile_records)
        t1 = time.perf_counter_ns()
        t_list.append(t1 - t0)
    s_init = compute_timing_stats(t_list)
    s_init["predictor"] = "cstp"
    init_rows.append(s_init)

    # ST-MoE Paper
    t_list = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        st_moe = STMoEPaperPredictor(num_moe_stages=num_stages, num_routed_experts=num_experts, top_k=top_k)
        st_moe.build_cct_from_profile(profile_records)
        t1 = time.perf_counter_ns()
        t_list.append(t1 - t0)
    s_init = compute_timing_stats(t_list)
    s_init["predictor"] = "st_moe_paper"
    init_rows.append(s_init)

    # Hybrid
    t_list = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        cstp = CSTPPredictor(num_moe_stages=num_stages, num_routed_experts=num_experts, top_k=top_k)
        cstp.fit_profile(profile_records)
        st_moe = STMoEPaperPredictor(num_moe_stages=num_stages, num_routed_experts=num_experts, top_k=top_k)
        st_moe.build_cct_from_profile(profile_records)
        _ = CSTPSTMoEHybridPredictor(cstp, st_moe)
        t1 = time.perf_counter_ns()
        t_list.append(t1 - t0)
    s_init = compute_timing_stats(t_list)
    s_init["predictor"] = "cstp_st_moe_hybrid"
    init_rows.append(s_init)

    # ACSH Global
    t_list = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        cstp = CSTPPredictor(num_moe_stages=num_stages, num_routed_experts=num_experts, top_k=top_k)
        cstp.fit_profile(profile_records)
        st_moe = STMoEPaperPredictor(num_moe_stages=num_stages, num_routed_experts=num_experts, top_k=top_k)
        st_moe.build_cct_from_profile(profile_records)
        _ = ACSHPredictor(cstp, st_moe, variant="global")
        t1 = time.perf_counter_ns()
        t_list.append(t1 - t0)
    s_init = compute_timing_stats(t_list)
    s_init["predictor"] = "acsh_global"
    init_rows.append(s_init)

    # ACSH Stage-Aware
    t_list = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        cstp = CSTPPredictor(num_moe_stages=num_stages, num_routed_experts=num_experts, top_k=top_k)
        cstp.fit_profile(profile_records)
        st_moe = STMoEPaperPredictor(num_moe_stages=num_stages, num_routed_experts=num_experts, top_k=top_k)
        st_moe.build_cct_from_profile(profile_records)
        _ = ACSHPredictor(cstp, st_moe, variant="stage_aware")
        t1 = time.perf_counter_ns()
        t_list.append(t1 - t0)
    s_init = compute_timing_stats(t_list)
    s_init["predictor"] = "acsh_stage_aware"
    init_rows.append(s_init)

    # ACSH Fast Recovery
    t_list = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        cstp = CSTPPredictor(num_moe_stages=num_stages, num_routed_experts=num_experts, top_k=top_k)
        cstp.fit_profile(profile_records)
        st_moe = STMoEPaperPredictor(num_moe_stages=num_stages, num_routed_experts=num_experts, top_k=top_k)
        st_moe.build_cct_from_profile(profile_records)
        _ = ACSHPredictor(cstp, st_moe, variant="fast_recovery")
        t1 = time.perf_counter_ns()
        t_list.append(t1 - t0)
    s_init = compute_timing_stats(t_list)
    s_init["predictor"] = "acsh_fast_recovery"
    init_rows.append(s_init)

    df_init = pd.DataFrame(init_rows)
    df_init.to_csv(os.path.join(latency_dir, "initialization_latency.csv"), index=False)
    print("  Initialization latency saved to latency/initialization_latency.csv")

    # -------------------------------------------------------------
    # 2. Measure Memory Footprint
    # -------------------------------------------------------------
    print("\n2. Measuring State Memory Footprint...")
    mem_rows = []

    def measure_memory_bytes(factory, name: str) -> Dict[str, Any]:
        tracemalloc.start()
        tracemalloc.reset_peak()
        obj = factory()
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "predictor": name,
            "allocated_kb": round(float(cur) / 1024.0, 2),
            "peak_kb": round(float(peak) / 1024.0, 2),
            "allocated_bytes": cur,
            "peak_bytes": peak
        }

    mem_rows.append(measure_memory_bytes(lambda: CSTPPredictor(num_moe_stages=57, num_routed_experts=256), "cstp"))
    mem_rows.append(measure_memory_bytes(lambda: STMoEPaperPredictor(num_moe_stages=57, num_routed_experts=256), "st_moe_paper"))
    mem_rows.append(measure_memory_bytes(lambda: CSTPSTMoEHybridPredictor(
        CSTPPredictor(num_moe_stages=57, num_routed_experts=256),
        STMoEPaperPredictor(num_moe_stages=57, num_routed_experts=256)
    ), "cstp_st_moe_hybrid"))
    mem_rows.append(measure_memory_bytes(lambda: ACSHPredictor(
        CSTPPredictor(num_moe_stages=57, num_routed_experts=256),
        STMoEPaperPredictor(num_moe_stages=57, num_routed_experts=256),
        variant="global"
    ), "acsh_global"))
    mem_rows.append(measure_memory_bytes(lambda: ACSHPredictor(
        CSTPPredictor(num_moe_stages=57, num_routed_experts=256),
        STMoEPaperPredictor(num_moe_stages=57, num_routed_experts=256),
        variant="stage_aware"
    ), "acsh_stage_aware"))
    mem_rows.append(measure_memory_bytes(lambda: ACSHPredictor(
        CSTPPredictor(num_moe_stages=57, num_routed_experts=256),
        STMoEPaperPredictor(num_moe_stages=57, num_routed_experts=256),
        variant="fast_recovery"
    ), "acsh_fast_recovery"))

    df_mem = pd.DataFrame(mem_rows)
    df_mem.to_csv(os.path.join(memory_dir, "memory_footprint.csv"), index=False)
    print("  Memory footprint saved to memory/memory_footprint.csv")

    # -------------------------------------------------------------
    # 2b. Memory Scaling with E and L
    # -------------------------------------------------------------
    print("\n2b. Measuring Memory Scaling with Experts and Stages...")
    scale_rows = []
    expert_grid = [8, 16, 32, 64, 128, 256]
    stage_grid = [16, 32, 57, 75]

    for e in expert_grid:
        c_bytes = measure_memory_bytes(lambda: CSTPPredictor(num_moe_stages=57, num_routed_experts=e), "cstp")
        s_bytes = measure_memory_bytes(lambda: STMoEPaperPredictor(num_moe_stages=57, num_routed_experts=e), "st_moe_paper")
        a_bytes = measure_memory_bytes(lambda: ACSHPredictor(
            CSTPPredictor(num_moe_stages=57, num_routed_experts=e),
            STMoEPaperPredictor(num_moe_stages=57, num_routed_experts=e),
            variant="stage_aware"
        ), "acsh_stage_aware")
        scale_rows.append({
            "dimension": "experts",
            "num_experts": e,
            "num_stages": 57,
            "cstp_kb": c_bytes["allocated_kb"],
            "st_moe_kb": s_bytes["allocated_kb"],
            "acsh_kb": a_bytes["allocated_kb"]
        })

    for s in stage_grid:
        c_bytes = measure_memory_bytes(lambda: CSTPPredictor(num_moe_stages=s, num_routed_experts=256), "cstp")
        s_bytes = measure_memory_bytes(lambda: STMoEPaperPredictor(num_moe_stages=s, num_routed_experts=256), "st_moe_paper")
        a_bytes = measure_memory_bytes(lambda: ACSHPredictor(
            CSTPPredictor(num_moe_stages=s, num_routed_experts=256),
            STMoEPaperPredictor(num_moe_stages=s, num_routed_experts=256),
            variant="stage_aware"
        ), "acsh_stage_aware")
        scale_rows.append({
            "dimension": "stages",
            "num_experts": 256,
            "num_stages": s,
            "cstp_kb": c_bytes["allocated_kb"],
            "st_moe_kb": s_bytes["allocated_kb"],
            "acsh_kb": a_bytes["allocated_kb"]
        })

    df_scale = pd.DataFrame(scale_rows)
    df_scale.to_csv(os.path.join(memory_dir, "memory_scaling.csv"), index=False)
    print("  Memory scaling saved to memory/memory_scaling.csv")

    # -------------------------------------------------------------
    # 2c. Concurrent Request Memory Scaling
    # -------------------------------------------------------------
    print("\n2c. Measuring Memory with Concurrent Requests...")
    concurrency_rows = []
    concurrency_levels = [1, 10, 100, 1000]

    base_cstp = CSTPPredictor(num_moe_stages=57, num_routed_experts=256)
    base_st = STMoEPaperPredictor(num_moe_stages=57, num_routed_experts=256)
    acsh_inst = ACSHPredictor(base_cstp, base_st, variant="stage_aware")

    for n_req in concurrency_levels:
        tracemalloc.start()
        tracemalloc.reset_peak()
        for r in range(n_req):
            acsh_inst.rel_c_stage[r] = {(s, 1): 0.5 for s in range(57)}
            acsh_inst.rel_m_stage[r] = {(s, 1): 0.5 for s in range(57)}
            acsh_inst.pending_predictions[(r, 10, 0)] = {
                "cstp_candidates": [1, 2, 3],
                "st_moe_candidates": [2, 3, 4],
                "horizon": 1,
                "budget": 8
            }
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        concurrency_rows.append({
            "concurrent_requests": n_req,
            "allocated_kb": round(float(cur) / 1024.0, 2),
            "allocated_per_req_kb": round((float(cur) / 1024.0) / float(n_req), 3)
        })

    df_conc = pd.DataFrame(concurrency_rows)
    df_conc.to_csv(os.path.join(memory_dir, "memory_concurrency.csv"), index=False)
    print("  Memory concurrency saved to memory/memory_concurrency.csv")

    # -------------------------------------------------------------
    # 3. Measure Prediction and Update Latency
    # -------------------------------------------------------------
    print("\n3. Measuring Prediction and Update Latency...")
    pred_lat_rows = []
    upd_lat_rows = []
    pred_breakdown_rows = []
    upd_breakdown_rows = []

    cstp_b = CSTPPredictor(num_moe_stages=num_stages, num_routed_experts=num_experts, top_k=top_k, online_adaptation=True)
    cstp_b.fit_profile(profile_records)

    st_moe_b = STMoEPaperPredictor(num_moe_stages=num_stages, num_routed_experts=num_experts, top_k=top_k, online_adaptation=True)
    st_moe_b.build_cct_from_profile(profile_records)

    predictors_to_bench = [
        ("cstp", CSTPWrapper(cstp_b)),
        ("st_moe_paper", st_moe_b),
        ("cstp_st_moe_hybrid", CSTPSTMoEHybridPredictor(copy.deepcopy(cstp_b), copy.deepcopy(st_moe_b))),
        ("acsh_global", ACSHPredictor(copy.deepcopy(cstp_b), copy.deepcopy(st_moe_b), variant="global")),
        ("acsh_stage_aware", ACSHPredictor(copy.deepcopy(cstp_b), copy.deepcopy(st_moe_b), variant="stage_aware")),
        ("acsh_fast_recovery", ACSHPredictor(copy.deepcopy(cstp_b), copy.deepcopy(st_moe_b), variant="fast_recovery"))
    ]

    source_exps_list = [list(np.random.choice(num_experts, size=top_k, replace=False)) for _ in range(n_timed)]
    target_exps_list = [list(np.random.choice(num_experts, size=top_k, replace=False)) for _ in range(n_timed)]

    for name, pred in predictors_to_bench:
        print(f"  Benchmarking: {name}")

        for b in budgets:
            # Warmup
            for i in range(n_warmup):
                pred.predict(
                    request_id=1,
                    target_stage_id=10,
                    current_source_experts=source_exps_list[i],
                    budget=b,
                    token_id=0,
                    source_stage_id=9
                )

            # Timed prediction
            p_times = []
            for i in range(n_timed):
                src = source_exps_list[i]
                t0 = time.perf_counter_ns()
                _ = pred.predict(
                    request_id=1,
                    target_stage_id=10,
                    current_source_experts=src,
                    budget=b,
                    token_id=0,
                    source_stage_id=9
                )
                t1 = time.perf_counter_ns()
                p_times.append(t1 - t0)

            stats = compute_timing_stats(p_times)
            stats["predictor"] = name
            stats["budget"] = b
            pred_lat_rows.append(stats)

        # Update latency
        for i in range(n_warmup):
            pred.update(
                request_id=1,
                target_stage_id=10,
                source_experts=source_exps_list[i],
                actual_target_experts=target_exps_list[i],
                token_id=0,
                source_stage_id=9
            )

        u_times = []
        for i in range(n_timed):
            src = source_exps_list[i]
            act = target_exps_list[i]
            t0 = time.perf_counter_ns()
            pred.update(
                request_id=1,
                target_stage_id=10,
                source_experts=src,
                actual_target_experts=act,
                token_id=0,
                source_stage_id=9
            )
            t1 = time.perf_counter_ns()
            u_times.append(t1 - t0)

        u_stats = compute_timing_stats(u_times)
        u_stats["predictor"] = name
        upd_lat_rows.append(u_stats)

    df_p_lat = pd.DataFrame(pred_lat_rows)
    df_p_lat.to_csv(os.path.join(latency_dir, "prediction_latency.csv"), index=False)

    df_u_lat = pd.DataFrame(upd_lat_rows)
    df_u_lat.to_csv(os.path.join(latency_dir, "update_latency.csv"), index=False)
    print("  Prediction and update latencies saved to latency/")

    # -------------------------------------------------------------
    # 4. Latency Breakdown for ACSH Stage-Aware
    # -------------------------------------------------------------
    print("\n4. Measuring ACSH Component Latency Breakdown...")
    acsh_for_breakdown = ACSHPredictor(copy.deepcopy(cstp_b), copy.deepcopy(st_moe_b), variant="stage_aware")
    b = 8

    t_cstp_scoring = []
    t_st_lookup = []
    t_comb = []
    t_topb = []

    for i in range(n_timed):
        src = source_exps_list[i]
        req_id = 1
        target_stage = 10
        source_stage = 9

        # 1. CSTP component
        t0 = time.perf_counter_ns()
        s_spatial = acsh_for_breakdown.cstp.compute_spatial_score(
            target_stage_id=target_stage,
            source_experts=src,
            source_stage_id=source_stage
        )
        s_temporal = acsh_for_breakdown.cstp.compute_temporal_score(req_id, target_stage)
        raw_cstp = (acsh_for_breakdown.cstp.alpha * s_spatial) + ((1.0 - acsh_for_breakdown.cstp.alpha) * s_temporal)
        t1 = time.perf_counter_ns()
        t_cstp_scoring.append(t1 - t0)

        # 2. ST-MoE component
        t0 = time.perf_counter_ns()
        st_dict = acsh_for_breakdown.st_moe.compute_confidence_scores(
            request_id=req_id,
            target_stage_id=target_stage,
            current_source_experts=src,
            source_stage_id=source_stage
        )
        raw_st_moe = np.zeros(num_experts, dtype=np.float64)
        for exp_id, val in st_dict.items():
            if exp_id >= 0 and exp_id < num_experts:
                raw_st_moe[exp_id] = float(val)
        t1 = time.perf_counter_ns()
        t_st_lookup.append(t1 - t0)

        # 3. Softmax & weighting
        t0 = time.perf_counter_ns()
        p_c, p_m = acsh_for_breakdown._normalize_scores(raw_cstp, raw_st_moe)
        w_c, w_m = acsh_for_breakdown.get_fusion_weights(req_id, target_stage, 1)
        p_acsh = (w_c * p_c) + (w_m * p_m)
        t1 = time.perf_counter_ns()
        t_comb.append(t1 - t0)

        # 4. Top-B selection
        t0 = time.perf_counter_ns()
        all_experts = list(range(num_experts))
        acsh_sorted = sorted(all_experts, key=lambda idx: (-p_acsh[idx], idx))
        _ = acsh_sorted[:b]
        t1 = time.perf_counter_ns()
        t_topb.append(t1 - t0)

    pred_breakdown_rows.append({"component": "cstp_scoring", **compute_timing_stats(t_cstp_scoring)})
    pred_breakdown_rows.append({"component": "st_moe_lookup", **compute_timing_stats(t_st_lookup)})
    pred_breakdown_rows.append({"component": "acsh_combination", **compute_timing_stats(t_comb)})
    pred_breakdown_rows.append({"component": "acsh_topb_selection", **compute_timing_stats(t_topb)})

    df_p_breakdown = pd.DataFrame(pred_breakdown_rows)
    df_p_breakdown.to_csv(os.path.join(latency_dir, "prediction_latency_breakdown.csv"), index=False)
    print("  Prediction breakdown saved to latency/prediction_latency_breakdown.csv")

    print("\n" + "=" * 80)
    print("PHASE D BENCHMARKING COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()

