#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np

def main():
    phase2f_dir = "/home/mohman/ca_project/results/final_comparative_campaign/phase2f"
    summary_path = "/home/mohman/ca_project/docs/final_campaign/phase2f/PHASE2F_SUMMARY.csv"

    df = pd.read_csv(summary_path)

    # Add or update columns with precise units and real predictor metrics
    precisions = []
    recalls = []
    f1s = []
    spec_mibs = []
    spec_mbs = []
    dem_mibs = []
    dem_mbs = []

    for idx, row in df.iterrows():
        run_id = row["run_id"]
        run_dir = os.path.join(phase2f_dir, run_id)
        json_path = os.path.join(run_dir, "live_prediction_summary.json")

        p = 0.0
        r = 0.0
        f = 0.0

        if os.path.exists(json_path):
            with open(json_path, "r") as f_in:
                data = json.load(f_in)
                overall = data.get("overall", {})
                p = float(overall.get("precision", 0.0) * 100.0)
                r = float(overall.get("recall", 0.0) * 100.0)
                f = float(overall.get("f1", 0.0) * 100.0)

        precisions.append(round(p, 2))
        recalls.append(round(r, 2))
        f1s.append(round(f, 2))

        # Traffic conversions: 1 expert = 84 MiB = 88.080384 MB
        # row["speculative_traffic_mb"] was actually MiB in the raw output
        raw_spec_mib = float(row["speculative_traffic_mb"])
        raw_dem_mib = float(row["demand_traffic_mb"])

        spec_mibs.append(round(raw_spec_mib, 2))
        spec_mbs.append(round(raw_spec_mib * (88.080384 / 84.0), 2))
        dem_mibs.append(round(raw_dem_mib, 2))
        dem_mbs.append(round(raw_dem_mib * (88.080384 / 84.0), 2))

    df["prediction_precision_pct"] = precisions
    df["prediction_recall_pct"] = recalls
    df["prediction_f1_pct"] = f1s
    df["speculative_traffic_mib"] = spec_mibs
    df["speculative_traffic_mb"] = spec_mbs
    df["demand_traffic_mib"] = dem_mibs
    df["demand_traffic_mb"] = dem_mbs

    df.to_csv(summary_path, index=False)
    print(f"[SUCCESS] Updated {summary_path} with real predictor metrics and reconciled units.")

if __name__ == "__main__":
    main()

