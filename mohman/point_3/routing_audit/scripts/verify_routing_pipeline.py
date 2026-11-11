#!/usr/bin/env python3
"""
Routing Pipeline Forensic Diagnostic Tool

Parses and cross-verifies:
1. Ground truth trace CSV
2. Simulator generated expert_routing.csv
3. Simulator generated expert_load.csv
4. Simulator generated prefetch_events.csv

Emits bounded diagnostic report verifying:
- Request ID, Seq ID, Token ID, Layer ID / MoE Stage
- Routing source (synthetic or real_trace)
- Actual Top-K expert IDs from trace vs expert_routing.csv
- Expert IDs dispatched to GEMM execution
- Expert IDs supplied as source context to predictor
- Predicted expert IDs
- Admitted candidates
- Dispatched candidates

Zero ternary operators rule strictly enforced.
"""

import os
import sys
import csv
import json


def parse_trace_csv(trace_path):
    trace_data = {}
    if not os.path.exists(trace_path):
        return trace_data
    with open(trace_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            seq_id = int(row[0])
            token_id = int(row[1])
            layer_id = int(row[2])
            rank = int(row[3])
            expert_id = int(row[4])
            key = (seq_id, token_id, layer_id)
            if key not in trace_data:
                trace_data[key] = []
            trace_data[key].append((rank, expert_id))
    for k in trace_data:
        trace_data[k].sort(key=lambda x: x[0])
        trace_data[k] = [x[1] for x in trace_data[k]]
    return trace_data


def parse_routing_csv(routing_path):
    routing_data = {}
    if not os.path.exists(routing_path):
        return routing_data
    with open(routing_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            req_id = int(row[0])
            token_id = int(row[1])
            layer_id = int(row[2])
            rank = int(row[3])
            expert_id = int(row[4])
            key = (req_id, token_id, layer_id)
            if key not in routing_data:
                routing_data[key] = []
            routing_data[key].append((rank, expert_id))
    for k in routing_data:
        routing_data[k].sort(key=lambda x: x[0])
        routing_data[k] = [x[1] for x in routing_data[k]]
    return routing_data


def parse_expert_load_csv(load_path):
    demands = {}
    if not os.path.exists(load_path):
        return demands
    with open(load_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            req_id = int(row[0])
            seq_id = int(row[1])
            token_id = int(row[2])
            layer_id = int(row[3])
            expert_id = int(row[4])
            weight_tensor = row[5]
            if weight_tensor != "all_weights":
                continue
            key = (seq_id, token_id, layer_id)
            if key not in demands:
                demands[key] = []
            if expert_id not in demands[key]:
                demands[key].append(expert_id)
    return demands


def parse_prefetch_events(events_path):
    predictions = {}
    admissions = {}
    dispatches = {}
    if not os.path.exists(events_path):
        return predictions, admissions, dispatches

    with open(events_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            ev_id = row[0]
            ts = float(row[1])
            ev_type = row[2]
            req_id = int(row[3])
            seq_id = int(row[4])
            token_id = int(row[5])
            layer_id = int(row[6])
            expert_id = int(row[7])
            cand_id = int(row[8])
            tr_id = int(row[9])
            details = row[10]

            key = (seq_id, token_id, layer_id)
            if ev_type == "PREDICTION":
                if key not in predictions:
                    predictions[key] = []
                predictions[key].append(expert_id)
            elif ev_type == "CANDIDATE_ADMITTED":
                if key not in admissions:
                    admissions[key] = []
                admissions[key].append(expert_id)
            elif ev_type == "CANDIDATE_DISPATCHED":
                if key not in dispatches:
                    dispatches[key] = []
                dispatches[key].append(expert_id)

    return predictions, admissions, dispatches


def generate_diagnostic(trace_path, run_dir, output_report_path, prompt_len=128):
    trace_data = parse_trace_csv(trace_path)
    routing_path = os.path.join(run_dir, "traces", "expert_routing.csv")
    load_path = os.path.join(run_dir, "traces", "expert_load.csv")
    events_path = os.path.join(run_dir, "traces", "prefetch_events.csv")

    routing_data = parse_routing_csv(routing_path)
    demands = parse_expert_load_csv(load_path)
    predictions, admissions, dispatches = parse_prefetch_events(events_path)

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, 'w') as out:
        out.write("# Routing and Prefetch Pipeline Forensic Diagnostic\n\n")
        out.write(f"**Run Directory:** `{run_dir}`  \n")
        out.write(f"**Ground Truth Trace:** `{trace_path}`  \n\n")
        out.write("| Seq | RelTok | SimTok | MoE Stage | Routing Source | Ground Truth Top-K | update_expert() Routing | GEMM Demanded Experts | Match? | Predicted Targets | Admitted | Dispatched |\n")
        out.write("|---|---|---|---|---|---|---|---|---|---|---|---||\n")

        # Sort keys by token and stage
        all_stage_keys = sorted(list(trace_data.keys()))
        if len(all_stage_keys) == 0 and len(routing_data) > 0:
            # Synthetic mode: derive keys from routing_data
            for rk in sorted(list(routing_data.keys())):
                req_id, sim_tok, stage_id = rk
                rel_tok = sim_tok - prompt_len
                seq_id = req_id - 1
                if seq_id < 0:
                    seq_id = 0
                sim_routed = routing_data[rk]
                dem_key = (seq_id, sim_tok, stage_id)
                gemm_demanded = demands.get(dem_key, [])

                match_status = "MISMATCH"
                if sorted(sim_routed) == sorted(gemm_demanded):
                    match_status = "EXACT_MATCH"

                pred_key = (seq_id, sim_tok, stage_id)
                preds = predictions.get(pred_key, [])
                adm = admissions.get(pred_key, [])
                disp = dispatches.get(pred_key, [])

                gt_str = "N/A (SYNTHETIC)"
                sim_str = ";".join(str(e) for e in sim_routed)
                dem_str = ";".join(str(e) for e in sorted(gemm_demanded))
                pred_str = ";".join(str(e) for e in preds)
                adm_str = ";".join(str(e) for e in adm)
                disp_str = ";".join(str(e) for e in disp)
                source_type = "SYNTHETIC"

                out.write(f"| {seq_id} | {rel_tok} | {sim_tok} | {stage_id} | {source_type} | {gt_str} | {sim_str} | {dem_str} | {match_status} | {pred_str} | {adm_str} | {disp_str} |\n")
        else:
            sorted_reqs = sorted(list(set(k[0] for k in routing_data.keys())))
            active_seq_ids = sorted(list(set(k[0] for k in all_stage_keys)))
            for key in all_stage_keys:
                seq_id, rel_tok, stage_id = key
                sim_tok = prompt_len + rel_tok
                gt_experts = trace_data.get(key, [])
                
                # Find in routing_data (keyed by req_id, sim_tok, stage_id)
                sim_routed = []
                r_cands = []
                if seq_id < len(sorted_reqs):
                    r_cands.append(sorted_reqs[seq_id])
                r_cands.extend([1, 0, seq_id, seq_id + 1])
                for r_cand in r_cands:
                    rk = (r_cand, sim_tok, stage_id)
                    if rk in routing_data:
                        sim_routed = routing_data[rk]
                        break

                # Find in demands: in batch execution, demand is attributed across batch sequences
                gemm_demanded_set = set()
                for s in active_seq_ids:
                    dem_key = (s, sim_tok, stage_id)
                    if dem_key in demands:
                        gemm_demanded_set.update(demands[dem_key])
                gemm_demanded = sorted(list(gemm_demanded_set))

                # Check match
                match_status = "MISMATCH"
                if sorted(gt_experts) == sorted(sim_routed):
                    match_status = "EXACT_MATCH"

                # Check prefetch predictions for this target stage
                pred_key = (seq_id, sim_tok, stage_id)
                preds = predictions.get(pred_key, [])
                adm = admissions.get(pred_key, [])
                disp = dispatches.get(pred_key, [])

                gt_str = ";".join(str(e) for e in gt_experts)
                sim_str = ";".join(str(e) for e in sim_routed)
                dem_str = ";".join(str(e) for e in sorted(gemm_demanded))
                pred_str = ";".join(str(e) for e in preds)
                adm_str = ";".join(str(e) for e in adm)
                disp_str = ";".join(str(e) for e in disp)

                source_type = "REAL_TRACE"
                if len(gt_experts) == 0:
                    source_type = "SYNTHETIC"

                out.write(f"| {seq_id} | {rel_tok} | {sim_tok} | {stage_id} | {source_type} | {gt_str} | {sim_str} | {dem_str} | {match_status} | {pred_str} | {adm_str} | {disp_str} |\n")

    print(f"[DIAGNOSTIC] Generated report: {output_report_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: verify_routing_pipeline.py <trace_path> <run_dir> <output_report_path> [prompt_len]")
        sys.exit(1)
    t_path = sys.argv[1]
    r_dir = sys.argv[2]
    o_path = sys.argv[3]
    p_len = 128
    if len(sys.argv) >= 5:
        p_len = int(sys.argv[4])
    generate_diagnostic(t_path, r_dir, o_path, p_len)

