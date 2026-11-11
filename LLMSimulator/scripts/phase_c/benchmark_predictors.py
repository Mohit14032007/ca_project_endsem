"""
Comprehensive Host-Side Predictor Latency and Memory Benchmarking for Phase C.
Uses time.perf_counter_ns() for high-resolution timing and tracemalloc for state memory.
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
from src.predictor.base_predictor import (
    BasePredictor,
    STMoELegacyWrapper,
    CSTPWrapper,
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


def benchmark_all_predictors() -> None:
    print("=" * 70)
    print("PHASE C: PREDICTOR LATENCY AND MEMORY OVERHEAD BENCHMARKING")
    print("=" * 70)

    out_dir = os.path.join(project_root, "results", "endsem_phase_c", "metadata")
    os.makedirs(out_dir, exist_ok=True)

    num_stages = 57
    num_experts = 256
    top_k = 8
    budgets = [4, 8, 16, 24, 32, 48, 64]
    n_warmup = 100
    n_timed = 1000

    np.random.seed(42)

    # 1. Measure Initialization Latency
    print("\n1. Measuring Initialization and Profiling Latency...")
    init_rows = []

    # Create synthetic profiling records
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

    # CSTP Init
    t_cstp_list = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        cstp = CSTPPredictor(num_moe_stages=num_stages, num_routed_experts=num_experts, top_k=top_k)
        cstp.fit_profile(profile_records)
        t1 = time.perf_counter_ns()
        t_cstp_list.append(t1 - t0)
    s_cstp_init = compute_timing_stats(t_cstp_list)
    s_cstp_init["predictor"] = "CSTP"
    init_rows.append(s_cstp_init)

    # ST-MoE Paper Init
    t_st_list = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        st_moe = STMoEPaperPredictor(num_moe_stages=num_stages, num_routed_experts=num_experts, top_k=top_k)
        st_moe.build_cct_from_profile(profile_records)
        t1 = time.perf_counter_ns()
        t_st_list.append(t1 - t0)
    s_st_init = compute_timing_stats(t_st_list)
    s_st_init["predictor"] = "ST-MoE Paper"
    init_rows.append(s_st_init)

    # Frequency Init
    t_freq_list = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        freq = FrequencyPredictor(num_moe_stages=num_stages, num_routed_experts=num_experts)
        freq.fit_profile(profile_records)
        t1 = time.perf_counter_ns()
        t_freq_list.append(t1 - t0)
    s_freq_init = compute_timing_stats(t_freq_list)
    s_freq_init["predictor"] = "Frequency Only"
    init_rows.append(s_freq_init)

    df_init = pd.DataFrame(init_rows)
    df_init.to_csv(os.path.join(out_dir, "predictor_initialization_latency.csv"), index=False)
    print("  Initialization latency saved.")

    # 2. Measure Memory Footprint
    print("\n2. Measuring State Memory Footprint via tracemalloc...")
    mem_rows = []

    def measure_memory(predictor_factory, name: str) -> Dict[str, Any]:
        tracemalloc.start()
        tracemalloc.reset_peak()
        p = predictor_factory()
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "predictor": name,
            "allocated_kb": round(float(cur) / 1024.0, 2),
            "peak_kb": round(float(peak) / 1024.0, 2)
        }

    mem_rows.append(measure_memory(lambda: CSTPPredictor(num_moe_stages=57, num_routed_experts=256), "CSTP (E=256, S=57)"))
    mem_rows.append(measure_memory(lambda: STMoEPaperPredictor(num_moe_stages=57, num_routed_experts=256), "ST-MoE Paper (E=256, S=57)"))
    mem_rows.append(measure_memory(lambda: FrequencyPredictor(num_moe_stages=57, num_routed_experts=256), "Frequency Only (E=256, S=57)"))
    mem_rows.append(measure_memory(lambda: RandomPredictor(num_routed_experts=256), "Random (E=256)"))
    mem_rows.append(measure_memory(lambda: CSTPPredictor(num_moe_stages=75, num_routed_experts=256), "CSTP MoE Suite (E=256, S=75)"))
    mem_rows.append(measure_memory(lambda: STMoEPaperPredictor(num_moe_stages=75, num_routed_experts=256), "ST-MoE MoE Suite (E=256, S=75)"))

    df_mem = pd.DataFrame(mem_rows)
    df_mem.to_csv(os.path.join(out_dir, "predictor_memory.csv"), index=False)
    print("  Memory measurements saved.")

    # 3. Measure Prediction and Update Latency across Budgets
    print("\n3. Measuring Prediction and Update Latencies...")
    pred_latency_rows = []
    upd_latency_rows = []

    # Prepare initialized predictor instances
    cstp_inst = CSTPPredictor(num_moe_stages=num_stages, num_routed_experts=num_experts, top_k=top_k, online_adaptation=True)
    cstp_inst.fit_profile(profile_records)

    st_moe_inst = STMoEPaperPredictor(num_moe_stages=num_stages, num_routed_experts=num_experts, top_k=top_k, online_adaptation=True)
    st_moe_inst.build_cct_from_profile(profile_records)

    freq_inst = FrequencyPredictor(num_moe_stages=num_stages, num_routed_experts=num_experts)
    freq_inst.fit_profile(profile_records)

    test_configs = [
        ("cstp_online", CSTPWrapper(cstp_inst)),
        ("cstp_frozen", CSTPWrapper(copy.deepcopy(cstp_inst))),
        ("st_moe_paper_online", st_moe_inst),
        ("st_moe_paper_frozen", copy.deepcopy(st_moe_inst)),
        ("st_moe_legacy", STMoELegacyWrapper(STMoEPredictor(num_moe_stages=num_stages, num_routed_experts=num_experts, top_k=top_k))),
        ("frequency_only", freq_inst),
        ("temporal_only", TemporalOnlyPredictor(copy.deepcopy(cstp_inst))),
        ("spatial_only", SpatialOnlyPredictor(copy.deepcopy(cstp_inst))),
        ("random", RandomPredictor(num_routed_experts=num_experts, seed=42)),
        ("cstp_st_moe_hybrid", CSTPSTMoEHybridPredictor(copy.deepcopy(cstp_inst), copy.deepcopy(st_moe_inst)))
    ]
    # Set frozen mode explicitly
    test_configs[1][1].inner.online_adaptation = False
    test_configs[3][1].online_adaptation = False

    # Pre-generate synthetic evaluation inputs
    source_exps_list = [list(np.random.choice(num_experts, size=top_k, replace=False)) for _ in range(n_timed)]
    target_exps_list = [list(np.random.choice(num_experts, size=top_k, replace=False)) for _ in range(n_timed)]

    for name, predictor in test_configs:
        print(f"  Benchmarking: {name}")

        # Prediction latency across budgets
        for b in budgets:
            # Warmup
            for i in range(n_warmup):
                predictor.predict(
                    request_id=1,
                    target_stage_id=10,
                    current_source_experts=source_exps_list[i],
                    budget=b
                )

            # Timed prediction
            times = []
            for i in range(n_timed):
                src = source_exps_list[i]
                t0 = time.perf_counter_ns()
                _ = predictor.predict(
                    request_id=1,
                    target_stage_id=10,
                    current_source_experts=src,
                    budget=b
                )
                t1 = time.perf_counter_ns()
                times.append(t1 - t0)

            stats = compute_timing_stats(times)
            stats["predictor"] = name
            stats["budget"] = b
            pred_latency_rows.append(stats)

        # Update latency
        # Warmup
        for i in range(n_warmup):
            predictor.update(
                request_id=1,
                target_stage_id=10,
                source_experts=source_exps_list[i],
                actual_target_experts=target_exps_list[i]
            )

        # Timed update
        upd_times = []
        for i in range(n_timed):
            src = source_exps_list[i]
            act = target_exps_list[i]
            t0 = time.perf_counter_ns()
            predictor.update(
                request_id=1,
                target_stage_id=10,
                source_experts=src,
                actual_target_experts=act
            )
            t1 = time.perf_counter_ns()
            upd_times.append(t1 - t0)

        upd_stats = compute_timing_stats(upd_times)
        upd_stats["predictor"] = name
        upd_latency_rows.append(upd_stats)

    df_pred_lat = pd.DataFrame(pred_latency_rows)
    df_pred_lat.to_csv(os.path.join(out_dir, "predictor_latency.csv"), index=False)

    df_upd_lat = pd.DataFrame(upd_latency_rows)
    df_upd_lat.to_csv(os.path.join(out_dir, "predictor_update_latency.csv"), index=False)

    print(f"\nAll latency and memory benchmarks saved to {out_dir}/")


if __name__ == "__main__":
    benchmark_all_predictors()

