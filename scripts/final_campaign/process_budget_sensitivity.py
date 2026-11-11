#!/usr/bin/env python3
import os
import sys
import json
import pandas as pd
import numpy as np

OUTPUT_BASE = "/home/mohman/ca_project/results/final_comparative_campaign/sensitivity_budget"
DOCS_DIR = "/home/mohman/ca_project/docs/final_campaign/sensitivity_budget"
PHASE2F_BASE = "/home/mohman/ca_project/results/final_comparative_campaign/phase2f"

EXPERT_SIZE_BYTES = 88080384
EXPERT_SIZE_MIB = 84.0
EXPERT_SIZE_MB = 88.080384

def parse_run(run_dir):
    csv_dir = os.path.join(run_dir, "csv")
    traces_dir = os.path.join(run_dir, "traces")

    sim_exec_ms = 0.0
    if os.path.exists(csv_dir):
        files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
        if files:
            df_csv = pd.read_csv(os.path.join(csv_dir, files[0]))
            if "latency" in df_csv.columns:
                sim_exec_ms = float(df_csv["latency"].sum() / 1e6)

    load_csv = os.path.join(traces_dir, "expert_load.csv")
    exposed_stall_ms = 0.0
    demand_misses = 0
    demand_hits = 0
    if os.path.exists(load_csv):
        df_load = pd.read_csv(load_csv)
        if "weight_tensor" in df_load.columns:
            exp_df = df_load[df_load["weight_tensor"] == "all_weights"]
            miss_df = exp_df[exp_df["cache_hit_or_miss"] == "miss"]
            hit_df = exp_df[exp_df["cache_hit_or_miss"] == "hit"]
            demand_misses = len(miss_df)
            demand_hits = len(hit_df)
            exposed_stall_ms = float(miss_df["duration"].sum() / 1e6)

    eff_csv = os.path.join(traces_dir, "prefetch_effectiveness.csv")
    timely_hits = 0
    late_attachments = 0
    total_spec = 0
    avg_queue_wait_ms = 0.0
    avg_rem_ms = 0.0
    unused = 0

    if os.path.exists(eff_csv):
        df_eff = pd.read_csv(eff_csv)
        if "event_type" in df_eff.columns and "outcome" in df_eff.columns:
            dem_df = df_eff[df_eff["event_type"] == "DEMAND_ACCESS"]
            d_counts = dem_df["outcome"].value_counts().to_dict()
            timely_hits = int(d_counts.get("TIMELY_RESIDENT", 0))
            late_attachments = int(d_counts.get("LATE_IN_FLIGHT", 0))

            late_df = dem_df[dem_df["outcome"] == "LATE_IN_FLIGHT"]
            if len(late_df) > 0 and "remaining_transfer_ns" in late_df.columns:
                avg_rem_ms = float(late_df["remaining_transfer_ns"].mean() / 1e6)

            spec_df = df_eff[df_eff["event_type"] == "SPECULATIVE_TRANSFER"]
            total_spec = len(spec_df)
            s_counts = spec_df["outcome"].value_counts().to_dict()
            unused = int(s_counts.get("UNUSED", 0))

            if len(spec_df) > 0 and "dispatch_ts" in spec_df.columns and "transfer_start_ts" in spec_df.columns:
                wait_series = spec_df["transfer_start_ts"] - spec_df["dispatch_ts"]
                valid_wait = wait_series[wait_series >= 0.0]
                if len(valid_wait) > 0:
                    avg_queue_wait_ms = float(valid_wait.mean() / 1e6)

    pred_json = os.path.join(run_dir, "live_prediction_summary.json")
    prec = 0.0
    rec = 0.0
    f1 = 0.0
    if os.path.exists(pred_json):
        with open(pred_json, "r") as f_in:
            data = json.load(f_in)
            overall = data.get("overall", {})
            prec = float(overall.get("precision", 0.0) * 100.0)
            rec = float(overall.get("recall", 0.0) * 100.0)
            f1 = float(overall.get("f1", 0.0) * 100.0)

    spec_mib = total_spec * EXPERT_SIZE_MIB
    spec_mb = total_spec * EXPERT_SIZE_MB
    dem_mib = demand_misses * EXPERT_SIZE_MIB
    dem_mb = demand_misses * EXPERT_SIZE_MB

    return {
        "sim_exec_ms": round(sim_exec_ms, 2),
        "exposed_stall_ms": round(exposed_stall_ms, 2),
        "demand_hits": demand_hits,
        "demand_misses": demand_misses,
        "timely_hits": timely_hits,
        "late_attachments": late_attachments,
        "unused_prefetches": unused,
        "total_speculative": total_spec,
        "avg_queue_wait_ms": round(avg_queue_wait_ms, 2),
        "avg_remaining_transfer_ms": round(avg_rem_ms, 2),
        "spec_mib": round(spec_mib, 2),
        "spec_mb": round(spec_mb, 2),
        "dem_mib": round(dem_mib, 2),
        "dem_mb": round(dem_mb, 2),
        "precision_pct": round(prec, 2),
        "recall_pct": round(rec, 2),
        "f1_pct": round(f1, 2)
    }

def main():
    os.makedirs(DOCS_DIR, exist_ok=True)

    rows = []

    b_m1 = parse_run(os.path.join(PHASE2F_BASE, "B-M1"))
    b_m12 = parse_run(os.path.join(PHASE2F_BASE, "B-M12"))

    experiments = [
        # Nominal M=1.0
        ("FWD-S-H2-B1-M1", "st_moe", 2, 1, 1.0, os.path.join(OUTPUT_BASE, "FWD-S-H2-B1-M1"), b_m1),
        ("FWD-S-H2-B2-M1", "st_moe", 2, 2, 1.0, os.path.join(OUTPUT_BASE, "FWD-S-H2-B2-M1"), b_m1),
        ("FWD-S-H2-B4-M1", "st_moe", 2, 4, 1.0, os.path.join(PHASE2F_BASE, "FWD-S-H2-M1"), b_m1),
        ("FWD-C-H2-B1-M1", "cstp", 2, 1, 1.0, os.path.join(OUTPUT_BASE, "FWD-C-H2-B1-M1"), b_m1),
        ("FWD-C-H2-B2-M1", "cstp", 2, 2, 1.0, os.path.join(OUTPUT_BASE, "FWD-C-H2-B2-M1"), b_m1),
        ("FWD-C-H2-B4-M1", "cstp", 2, 4, 1.0, os.path.join(PHASE2F_BASE, "FWD-C-H2-M1"), b_m1),
        # Decoupled Window M=12.0
        ("FWD-S-H2-B1-M12", "st_moe", 2, 1, 12.0, os.path.join(OUTPUT_BASE, "FWD-S-H2-B1-M12"), b_m12),
        ("FWD-S-H2-B2-M12", "st_moe", 2, 2, 12.0, os.path.join(OUTPUT_BASE, "FWD-S-H2-B2-M12"), b_m12),
        ("FWD-S-H2-B4-M12", "st_moe", 2, 4, 12.0, os.path.join(PHASE2F_BASE, "FWD-S-H2-M12"), b_m12),
        ("FWD-C-H2-B1-M12", "cstp", 2, 1, 12.0, os.path.join(OUTPUT_BASE, "FWD-C-H2-B1-M12"), b_m12),
        ("FWD-C-H2-B2-M12", "cstp", 2, 2, 12.0, os.path.join(OUTPUT_BASE, "FWD-C-H2-B2-M12"), b_m12),
        ("FWD-C-H2-B4-M12", "cstp", 2, 4, 12.0, os.path.join(PHASE2F_BASE, "FWD-C-H2-M12"), b_m12),
    ]

    for run_id, pred, h, b, m, path, base_metrics in experiments:
        m_dict = parse_run(path)
        stall_red = round(base_metrics["exposed_stall_ms"] - m_dict["exposed_stall_ms"], 2)
        speedup = 0.0
        if base_metrics["sim_exec_ms"] > 0:
            speedup = round((base_metrics["sim_exec_ms"] - m_dict["sim_exec_ms"]) / base_metrics["sim_exec_ms"] * 100.0, 3)

        rows.append({
            "run_id": run_id,
            "multiplier": m,
            "predictor": pred,
            "horizon": h,
            "candidate_budget": b,
            "sim_exec_time_ms": m_dict["sim_exec_ms"],
            "exposed_stall_ms": m_dict["exposed_stall_ms"],
            "stall_reduction_ms": stall_red,
            "speedup_pct": speedup,
            "timely_resident_hits": m_dict["timely_hits"],
            "late_inflight_attachments": m_dict["late_attachments"],
            "unused_prefetches": m_dict["unused_prefetches"],
            "total_speculative_transfers": m_dict["total_speculative"],
            "avg_queue_wait_ms": m_dict["avg_queue_wait_ms"],
            "avg_remaining_transfer_ms": m_dict["avg_remaining_transfer_ms"],
            "speculative_traffic_mib": m_dict["spec_mib"],
            "speculative_traffic_mb": m_dict["spec_mb"],
            "demand_traffic_mib": m_dict["dem_mib"],
            "demand_traffic_mb": m_dict["dem_mb"],
            "precision_pct": m_dict["precision_pct"],
            "recall_pct": m_dict["recall_pct"],
            "f1_pct": m_dict["f1_pct"]
        })

    df_out = pd.DataFrame(rows)
    csv_out_path = os.path.join(DOCS_DIR, "CANDIDATE_BUDGET_SENSITIVITY.csv")
    df_out.to_csv(csv_out_path, index=False)
    print(f"[SUCCESS] Wrote candidate budget sensitivity CSV to {csv_out_path}\n")
    print(df_out[["run_id", "multiplier", "predictor", "candidate_budget", "avg_queue_wait_ms", "speculative_traffic_mib", "timely_resident_hits", "late_inflight_attachments", "stall_reduction_ms"]].to_string())

if __name__ == "__main__":
    main()
