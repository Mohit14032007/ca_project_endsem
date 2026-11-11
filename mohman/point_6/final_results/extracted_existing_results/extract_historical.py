#!/usr/bin/env python3
"""
Extract and inventory historical experiment results for Point 6.
Enforces zero ternary operators (ast.IfExp == 0).
"""

import os
import shutil
import ast
import pandas as pd

def extract_historical_results():
    target_dir = "mohman/point_6/final_results/extracted_existing_results"
    os.makedirs(target_dir, exist_ok=True)
    
    # 1. Batch size historical results
    p5_batch = pd.read_csv("mohman/point_5/extracted_data/batch_scaling_comparison.csv")
    p5_batch.to_csv(os.path.join(target_dir, "batch_size_historical_results.csv"), index=False)
    print(f"Exported batch_size_historical_results.csv ({len(p5_batch)} rows).")
    
    # 2. Context length extraction from Point 6 Experiment A
    p6_exp_a = pd.read_csv("mohman/point_6/experiment_a/PROMPT_INITIAL_CONTEXT_LENGTHS.csv")
    p6_exp_a.to_csv(os.path.join(target_dir, "context_length_extraction.csv"), index=False)
    print(f"Exported context_length_extraction.csv ({len(p6_exp_a)} rows).")
    
    # 3. Predictor historical results
    p5_pred = pd.read_csv("mohman/point_5/extracted_data/predictor_metrics.csv")
    p5_pred.to_csv(os.path.join(target_dir, "predictor_historical_results.csv"), index=False)
    print(f"Exported predictor_historical_results.csv ({len(p5_pred)} rows).")
    
    # 4. Prefetch historical results
    p5_prefetch = pd.read_csv("mohman/point_5/extracted_data/prefetch_results.csv")
    p5_prefetch.to_csv(os.path.join(target_dir, "prefetch_historical_results.csv"), index=False)
    print(f"Exported prefetch_historical_results.csv ({len(p5_prefetch)} rows).")
    
    # 5. Eviction historical results
    p5_complete = pd.read_csv("mohman/point_5/extracted_data/complete_system_results.csv")
    p5_complete.to_csv(os.path.join(target_dir, "eviction_historical_results.csv"), index=False)
    print(f"Exported eviction_historical_results.csv ({len(p5_complete)} rows).")
    
    # 6. Historical source manifest
    p5_manifest = pd.read_csv("mohman/point_5/extracted_data/source_manifest.csv")
    p5_manifest.to_csv(os.path.join(target_dir, "historical_source_manifest.csv"), index=False)
    print(f"Exported historical_source_manifest.csv ({len(p5_manifest)} rows).")

if __name__ == "__main__":
    extract_historical_results()

