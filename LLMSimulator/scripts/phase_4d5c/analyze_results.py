#!/usr/bin/env python3
"""
scripts/phase_4d5c/analyze_results.py
Comparative Analysis and Table Generator for Phase 4D.5C 10-Token Evaluation.

Strict constraint: ZERO ternary operators.
"""

import ast
import json
import os
import sys
import pandas as pd


def check_zero_ternary(filename):
    with open(filename, "r") as f:
        tree = ast.parse(f.read(), filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            raise AssertionError(f"Ternary operator found at line {node.lineno} in {filename}")
    print(f"[VERIFY] Zero ternary expressions verified in {filename}")


def load_matrix():
    results_dir = "/home/mohman/ca_project/LLMSimulator/results/phase_4d5c"
    json_path = os.path.join(results_dir, "production_matrix_summary.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def pct_change(base, new):
    if base == 0:
        return 0.0
    return ((new - base) / base) * 100.0


def latency_reduction_pct(base, new):
    if base == 0:
        return 0.0
    return ((base - new) / base) * 100.0


def main():
    check_zero_ternary(__file__)
    records = load_matrix()

    c1 = records["c01_baseline_lru"]
    c2 = records["c02_stmoe_predonly"]
    c3 = records["c03_cstp_predonly"]
    c4 = records["c04_stmoe_prefetch_lru"]
    c5 = records["c05_cstp_prefetch_lru"]
    c6 = records["c06_stmoe_pabi_quarantine"]
    c7 = records["c07_cstp_pabi_quarantine"]
    c8 = records["c08_stmoe_pabi_pinning"]
    c9 = records["c09_cstp_pabi_pinning"]

    print("\n--- 1. Baseline vs Prediction-Only ---")
    for base, pred, name in [(c1, c2, "ST-MoE"), (c1, c3, "CSTP")]:
        e2e_base = base["sim_total_device_time_ns"] / 1e6
        e2e_pred = pred["sim_total_device_time_ns"] / 1e6
        stall_base = base["total_exposed_stall_ns"] / 1e6
        stall_pred = pred["total_exposed_stall_ns"] / 1e6
        host_base = base["host_runtime_sec"]
        host_pred = pred["host_runtime_sec"]
        print(f"{name}: E2E diff: {e2e_pred - e2e_base:.2f} ms ({pct_change(e2e_base, e2e_pred):.2f}%), "
              f"Stall diff: {stall_pred - stall_base:.2f} ms, Host diff: {host_pred - host_base:.2f} s")

    print("\n--- 2. Prediction-Only vs Hardware Async Prefetch ---")
    for pred, pref, name in [(c2, c4, "ST-MoE (LRU)"), (c3, c5, "CSTP (LRU)")]:
        e2e_p = pred["sim_total_device_time_ns"] / 1e6
        e2e_pf = pref["sim_total_device_time_ns"] / 1e6
        stall_p = pred["total_exposed_stall_ns"] / 1e6
        stall_pf = pref["total_exposed_stall_ns"] / 1e6
        print(f"{name}: E2E reduction: {latency_reduction_pct(e2e_p, e2e_pf):.4f}% ({e2e_p - e2e_pf:.2f} ms), "
              f"Stall reduction: {latency_reduction_pct(stall_p, stall_pf):.4f}% ({stall_p - stall_pf:.2f} ms)")

    print("\n--- 3. LRU vs PABI Quarantine ---")
    for lru, pabi, name in [(c4, c6, "ST-MoE"), (c5, c7, "CSTP")]:
        e2e_l = lru["sim_total_device_time_ns"] / 1e6
        e2e_pb = pabi["sim_total_device_time_ns"] / 1e6
        stall_l = lru["total_exposed_stall_ns"] / 1e6
        stall_pb = pabi["total_exposed_stall_ns"] / 1e6
        dem_ins_l = int(lru["DEMAND_INSERTIONS"])
        dem_ins_pb = int(pabi["DEMAND_INSERTIONS"])
        print(f"{name}: Demand Insertions {dem_ins_l} -> {dem_ins_pb} ({dem_ins_pb - dem_ins_l}), "
              f"Stall: {stall_l:.2f} -> {stall_pb:.2f} ms, E2E: {e2e_l:.2f} -> {e2e_pb:.2f} ms")

    print("\n--- 4. PABI Quarantine vs PABI Pinning ---")
    for pabi, pin, name in [(c6, c8, "ST-MoE"), (c7, c9, "CSTP")]:
        e2e_pb = pabi["sim_total_device_time_ns"] / 1e6
        e2e_pin = pin["sim_total_device_time_ns"] / 1e6
        stall_pb = pabi["total_exposed_stall_ns"] / 1e6
        stall_pin = pin["total_exposed_stall_ns"] / 1e6
        print(f"{name}: Stall {stall_pb:.2f} -> {stall_pin:.2f} ms, E2E: {e2e_pb:.2f} -> {e2e_pin:.2f} ms, "
              f"Pin Intents: {pin.get('hbm_pin_intents_created', 0)}, Expired: {pin.get('hbm_pins_expired', 0)}")

    print("\n--- 5. ST-MoE vs CSTP Head-to-Head ---")
    pairs = [
        ("Pred-Only", c2, c3),
        ("LRU Prefetch", c4, c5),
        ("PABI Quarantine", c6, c7),
        ("PABI Pinning", c8, c9)
    ]
    for mode, st, cs in pairs:
        e2e_st = st["sim_total_device_time_ns"] / 1e6
        e2e_cs = cs["sim_total_device_time_ns"] / 1e6
        stall_st = st["total_exposed_stall_ns"] / 1e6
        stall_cs = cs["total_exposed_stall_ns"] / 1e6
        disp_st = int(st.get("CANDIDATES_DISPATCHED", 0))
        disp_cs = int(cs.get("CANDIDATES_DISPATCHED", 0))
        print(f"{mode:<16} | CSTP vs ST-MoE Stall: {stall_cs - stall_st:+.2f} ms, E2E: {e2e_cs - e2e_st:+.2f} ms, "
              f"Dispatched: CS={disp_cs} vs ST={disp_st}")


if __name__ == "__main__":
    main()

