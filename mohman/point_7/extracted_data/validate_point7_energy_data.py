#!/usr/bin/env python3
"""
Validation script for Point 7 Energy Analysis datasets.
Verifies all microarchitectural and mathematical accounting invariants across all CSVs.

Invariants checked:
1. total_ddr5_bytes == demand_ddr5_bytes + speculative_ddr5_bytes
2. ddr5_total_j == ddr5_demand_j + ddr5_speculative_j
3. ddr5_total_j == total_ddr5_bytes * 1e-10
4. static_dram_j == simulated_execution_time_s * 4.5
5. system_total_j == native_total_j + ddr5_total_j + static_dram_j
6. system_energy_per_token_mj == (system_total_j / total_tokens) * 1000
7. system_edp_j_s == system_total_j * simulated_execution_time_s

RULE: Strictly ZERO ternary operators (ast.IfExp == 0).
"""

import os
import sys
import pandas as pd
import numpy as np

DATA_DIR = "/home/mohman/ca_project/mohman/point_7/extracted_data"

def validate_unified_results():
    path = os.path.join(DATA_DIR, "unified_energy_results.csv")
    df = pd.read_csv(path)
    print(f"[VALIDATION] Checking {path} ({len(df)} rows)...")
    
    # 28 rows check
    if len(df) != 28:
        raise ValueError(f"Expected 28 rows, found {len(df)}")
        
    for idx, row in df.iterrows():
        rid = row['run_id']
        
        # 1. Byte conservation
        b_tot = row['total_ddr5_bytes']
        b_dem = row['demand_ddr5_bytes']
        b_spec = row['speculative_ddr5_bytes']
        if b_tot != (b_dem + b_spec):
            raise AssertionError(f"Byte mismatch in {rid}: {b_tot} != {b_dem} + {b_spec}")
            
        # 2. DDR5 Energy conservation
        e_ddr5_tot = row['ddr5_total_j']
        e_ddr5_dem = row['ddr5_demand_j']
        e_ddr5_spec = row['ddr5_speculative_j']
        diff_ddr5 = abs(e_ddr5_tot - (e_ddr5_dem + e_ddr5_spec))
        if diff_ddr5 > 1e-9:
            raise AssertionError(f"DDR5 energy mismatch in {rid}: {diff_ddr5}")
            
        # 3. DDR5 coefficient (0.1 nJ/byte = 1e-10 J/byte)
        diff_coeff = abs(e_ddr5_tot - (b_tot * 1e-10))
        if diff_coeff > 1e-9:
            raise AssertionError(f"DDR5 coefficient mismatch in {rid}: {diff_coeff}")
            
        # 4. Static DRAM energy (4.5 W * time_s)
        t_sim = row['simulated_execution_time_s']
        e_static = row['static_dram_j']
        diff_static = abs(e_static - (t_sim * 4.5))
        if diff_static > 1e-9:
            raise AssertionError(f"Static DRAM energy mismatch in {rid}: {diff_static}")
            
        # 5. System total energy
        e_native = row['native_total_j']
        e_sys = row['system_total_j']
        diff_sys = abs(e_sys - (e_native + e_ddr5_tot + e_static))
        if diff_sys > 1e-9:
            raise AssertionError(f"System total energy mismatch in {rid}: {diff_sys}")
            
        # 6. Energy per token
        tok_tot = row['total_tokens']
        e_tok_mj = row['system_energy_per_token_mj']
        expected_tok_mj = (e_sys / tok_tot) * 1000.0
        diff_tok = abs(e_tok_mj - expected_tok_mj)
        if diff_tok > 1e-6:
            raise AssertionError(f"Energy/token mismatch in {rid}: {diff_tok}")
            
        # 7. System EDP
        edp = row['system_edp_j_s']
        expected_edp = e_sys * t_sim
        diff_edp = abs(edp - expected_edp)
        if diff_edp > 1e-4:
            raise AssertionError(f"EDP mismatch in {rid}: {diff_edp}")
            
    print("[VALIDATION] unified_energy_results.csv passed all 7 microarchitectural invariant tests!")

def validate_paired_comparisons():
    path = os.path.join(DATA_DIR, "paired_energy_comparisons.csv")
    df_pair = pd.read_csv(path)
    df_uni = pd.read_csv(os.path.join(DATA_DIR, "unified_energy_results.csv")).set_index('run_id')
    print(f"[VALIDATION] Checking {path} ({len(df_pair)} rows)...")
    
    for idx, row in df_pair.iterrows():
        pid = row['pair_id']
        b_id = row['baseline_run_id']
        p_id = row['proposed_run_id']
        
        b_row = df_uni.loc[b_id]
        p_row = df_uni.loc[p_id]
        
        # Check workload compatibility
        if b_row['batch_size'] != p_row['batch_size']:
            raise AssertionError(f"Incompatible batch size in pair {pid}: {b_row['batch_size']} vs {p_row['batch_size']}")
        if b_row['context_p'] != p_row['context_p']:
            raise AssertionError(f"Incompatible context_p in pair {pid}: {b_row['context_p']} vs {p_row['context_p']}")
        if b_row['tokens_t'] != p_row['tokens_t']:
            raise AssertionError(f"Incompatible tokens_t in pair {pid}: {b_row['tokens_t']} vs {p_row['tokens_t']}")
            
        # Check delta execution time
        expected_time_delta = p_row['simulated_execution_time_s'] - b_row['simulated_execution_time_s']
        if abs(row['execution_time_delta_s'] - expected_time_delta) > 1e-4:
            raise AssertionError(f"Time delta mismatch in pair {pid}")
            
        # Check delta system energy
        expected_sys_delta = p_row['system_total_j'] - b_row['system_total_j']
        if abs(row['system_energy_delta_j'] - expected_sys_delta) > 1e-4:
            raise AssertionError(f"System energy delta mismatch in pair {pid}")
            
    print("[VALIDATION] paired_energy_comparisons.csv passed all compatibility and delta checks!")

def main():
    validate_unified_results()
    validate_paired_comparisons()
    print("[VALIDATION] All Point 7 dataset validations PASSED with ZERO errors!")

if __name__ == '__main__':
    main()

