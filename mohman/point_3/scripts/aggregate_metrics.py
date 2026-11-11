import json
import os
import glob
import csv

def load_json(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    base_dir = "/home/mohman/ca_project/mohman/point_3/results/core"
    configs = ["baseline", "stmoe_h1", "stmoe_h2", "cstp_h1", "cstp_h2"]
    
    rows = []
    baseline_bytes = 0
    baseline_token_lat = 0.0

    # First pass: find baseline values for ratio computations
    base_summary_path = os.path.join(base_dir, "baseline", "traces", "experiment_summary.json")
    base_data = load_json(base_summary_path)
    if base_data is not None:
        baseline_bytes = base_data.get("memory_traffic", {}).get("total_ddr5_bytes_transferred", 0)
        baseline_token_lat = base_data.get("token_latency", {}).get("mean_ns", 0.0)

    for cfg in configs:
        summary_path = os.path.join(base_dir, cfg, "traces", "experiment_summary.json")
        data = load_json(summary_path)
        if data is None:
            print("[WARN] Missing summary for " + cfg)
            continue
            
        meta = data.get("experiment_metadata", {})
        cov = data.get("prefetch_coverage", {})
        rates = data.get("timely_and_late_rates", {})
        unused = data.get("unused_prefetches", {})
        hits = data.get("hbm_hit_rates", {})
        mem = data.get("memory_traffic", {})
        stall = data.get("exposed_stall_and_execution", {})
        exp_lat = data.get("expert_load_latency", {})
        exp_dist = data.get("expert_load_latency_distributions", {})
        u_exp_dist = data.get("unique_expert_load_latency_distributions", {})
        tok_lat = data.get("token_latency", {})
        req_lat = data.get("request_latency", {})
        
        # Calculate speedup and traffic overhead vs baseline
        total_bytes = mem.get("total_ddr5_bytes_transferred", 0)
        traffic_overhead = 0.0
        if baseline_bytes > 0:
            traffic_overhead = (total_bytes - baseline_bytes) / float(baseline_bytes)

        mean_tok = tok_lat.get("mean_ns", 0.0)
        speedup = 1.0
        if mean_tok > 0.0:
            if baseline_token_lat > 0.0:
                speedup = baseline_token_lat / mean_tok

        row = {
            "config": cfg,
            "predictor": meta.get("predictor_type", "none"),
            "horizon": meta.get("horizon", 0),
            "budget": meta.get("budget", 0),
            # Metric 1: Coverage
            "total_demands_proj": cov.get("total_actual_demands", 0),
            "unique_demands": cov.get("unique_actual_demands", 0),
            "overall_coverage_proj": cov.get("overall_coverage", 0.0),
            "unique_coverage": cov.get("unique_overall_coverage", 0.0),
            "completed_utilization": cov.get("completed_prefetch_utilization", 0.0),
            # Metric 2: Timely & Late Rates
            "timely_hits_proj": rates.get("timely_resident_hits", 0),
            "unique_timely_hits": rates.get("unique_timely_resident_hits", 0),
            "late_attachments_proj": rates.get("late_inflight_attachments", 0),
            "unique_late_attachments": rates.get("unique_late_inflight_attachments", 0),
            "timely_rate_proj": rates.get("timely_rate_of_demands", 0.0),
            "unique_timely_rate": rates.get("unique_timely_rate_of_demands", 0.0),
            "late_rate_proj": rates.get("late_rate_of_demands", 0.0),
            "unique_late_rate": rates.get("unique_late_rate_of_demands", 0.0),
            "avg_remaining_late_wait_ns": rates.get("avg_remaining_late_wait_ns", 0.0),
            "dispatched_prefetches": rates.get("total_dispatched_prefetches", 0),
            # Metric 3: Unused Prefetches
            "prefetches_completed_and_used": unused.get("prefetches_completed_and_used", 0),
            "prefetches_completed_unused": unused.get("prefetches_completed_unused", 0),
            "prefetches_evicted_before_demand": unused.get("prefetches_evicted_before_demand", 0),
            "prefetches_still_inflight_at_end": unused.get("prefetches_still_inflight_at_end", 0),
            "total_wasted_speculative_bytes": unused.get("total_wasted_speculative_bytes", 0),
            # Metric 4: Hit Rates
            "resident_hits_proj": hits.get("total_hbm_resident_hits", 0),
            "unique_resident_hits": hits.get("unique_resident_hits", 0),
            "fresh_misses_proj": hits.get("fresh_demand_misses", 0),
            "unique_fresh_misses": hits.get("unique_fresh_demand_misses", 0),
            "hbm_hit_rate_proj": hits.get("overall_hbm_hit_rate", 0.0),
            "unique_hbm_hit_rate": hits.get("unique_overall_hbm_hit_rate", 0.0),
            "fresh_miss_rate_proj": hits.get("fresh_demand_miss_rate", 0.0),
            "unique_fresh_miss_rate": hits.get("unique_fresh_demand_miss_rate", 0.0),
            # Metric 5: Memory Traffic
            "speculative_bytes": mem.get("speculative_bytes_transferred", 0),
            "demand_bytes": mem.get("demand_bytes_transferred", 0),
            "total_ddr5_bytes": total_bytes,
            "traffic_overhead_vs_baseline": traffic_overhead,
            # Metric 6: Stall
            "aggregate_demand_wait_ns": stall.get("aggregate_demand_wait_ns", 0.0),
            "exposed_gpu_stall_ns": stall.get("exposed_gpu_stall_ns", 0.0),
            "stall_hidden_by_prefetch_ns": stall.get("stall_hidden_by_prefetch_ns", 0.0),
            "wait_per_token_ns": stall.get("wait_per_token_ns", 0.0),
            "wait_per_unique_demand_ns": stall.get("wait_per_unique_demand_ns", 0.0),
            "total_simulated_execution_ns": stall.get("total_simulated_execution_ns", 0.0),
            # Metric 7: Expert Load Latency
            "exp_lat_mean_ns": exp_lat.get("mean_ns", 0.0),
            "exp_lat_p50_ns": exp_lat.get("p50_ns", 0.0),
            "exp_lat_p90_ns": exp_lat.get("p90_ns", 0.0),
            "exp_lat_p95_ns": exp_lat.get("p95_ns", 0.0),
            "exp_lat_p99_ns": exp_lat.get("p99_ns", 0.0),
            "exp_lat_max_ns": exp_lat.get("max_ns", 0.0),
            "demand_miss_lat_mean_ns": exp_dist.get("demand_miss", {}).get("mean_ns", 0.0),
            # Metric 8: Token Latency
            "token_count": tok_lat.get("count", 0),
            "token_lat_mean_ns": mean_tok,
            "token_lat_p50_ns": tok_lat.get("p50_ns", 0.0),
            "token_lat_p90_ns": tok_lat.get("p90_ns", 0.0),
            "token_lat_p95_ns": tok_lat.get("p95_ns", 0.0),
            "token_lat_p99_ns": tok_lat.get("p99_ns", 0.0),
            "token_lat_max_ns": tok_lat.get("max_ns", 0.0),
            "tokens_per_second": tok_lat.get("simulated_tokens_per_second", 0.0),
            "speedup_vs_baseline": speedup
        }
        rows.append(row)

    out_csv = "/home/mohman/ca_project/mohman/point_3/reports/consolidated_metrics.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if len(rows) > 0:
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("[SUCCESS] Consolidated metrics written to: " + out_csv)

if __name__ == "__main__":
    main()

