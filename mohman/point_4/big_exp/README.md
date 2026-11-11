# Big Experiment: LRU vs. FA-LRU ($W=64$) Under Large MoE Workloads

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**HBM Cache Capacity:** 35 GiB ($37,580,963,840$ Bytes, 426 slots)  
**Evaluated Workloads:** W3 ($B=14, T=128$) and W4 ($B=32, T=128$) across 40 MoE stages  
**Status:** COMPLETE AND FULLY CERTIFIED (16/16 Runs Succeeded)  
**Date:** 2026-11-10  

---

## 1. Directory Structure

```text
mohman/point_4/big_exp/
├── audit/
│   ├── TRACE_AVAILABILITY_AUDIT.md       # Audit of authentic T=128 token traces
│   ├── CONFIGURATION_AUDIT.md            # Hardware, cache, and policy configuration audit
│   └── EXPERIMENT_READINESS_REPORT.md     # Readiness checklist and GO certification
├── configs/                              # 16 YAML simulation configuration files
├── figures/                              # 8 Publication-quality figures (300 DPI PNG & vector PDF)
│   ├── png/
│   └── pdf/
├── reports/
│   ├── FINAL_BIG_EXPERIMENT_REPORT.md    # Master 13-section technical report answering all core questions
│   ├── DETAILED_METRIC_ANALYSIS.md       # Metric tables, conservation proofs, and C.4 diagnostics
│   └── REPRODUCIBILITY_REPORT.md         # Repetition 1 vs. Repetition 2 bit-exact determinism audit
├── results/
│   ├── all_runs.csv                      # Consolidated 16-run dataset
│   ├── paired_comparisons.csv            # Pairwise deltas (C2 vs C1, C4 vs C3)
│   ├── campaign_manifest.csv             # Execution parameters and seed manifest
│   ├── experiment_summary.json           # Machine-readable telemetry summary
│   └── raw_logs/                         # Standard output logs for all 16 runs
├── scripts/
│   ├── export_big_workloads.py           # Trace exporter from e1-sage-a1.5.parquet
│   ├── run_big_experiment.py             # 16-run execution and validation runner
│   └── plot_big_experiment.py            # Matplotlib publication figure generator
└── traces/
    ├── trace_manifest.csv                # Sequence and prompt metadata
    ├── w3_b14_t128.csv                   # W3 routing trace (573,440 records)
    └── w4_b32_t128.csv                   # W4 routing trace (1,310,720 records)
```

---

## 2. Key Findings Summary

1. **FA-LRU ($W=64$) Wins Under High-Concurrency Workload W4 ($B=32, T=128$):**
   - **Hit Rate Improvement:** Increased coalesced resident hits from 1,417 to **1,432** (**$+15\text{ additional hits}$**, $+1.04\%$ relative).
   - **Reduced Reloads:** Reduced expert reloads from DDR5 from 536,034 to **536,019** (**$-15\text{ reloads}$**).
   - **Lower Memory Traffic:** Saved **$2.0258\text{ GB}$ of DDR5 traffic** ($48,738.84\text{ GB} \to 48,736.81\text{ GB}$).
   - **Exposed Stall Reduction:** Decreased exposed GPU memory stall from $747,863.57\text{ ms}$ to **$747,831.75\text{ ms}$** (**$-31.82\text{ ms}$**).
   - **Lower Token Latency:** Improved mean token latency from $5,950.06\text{ ms}$ to **$5,949.81\text{ ms}$** ($-0.25\text{ ms/token}$).
2. **Mitigation of Speculative Cache Pollution:**
   - Under active CSTP prefetching, FA-LRU diverged from LRU in **$78.12\%$ of eviction decisions** (431,909 decisions in W4).
   - FA-LRU protected demand lines ($f=2$) by prioritizing the eviction of **5,999 unconsumed speculative prefetches** ($f=1$).
3. **Pure Demand Identity:**
   - Without prefetching (C1 vs. C2), FA-LRU and LRU produce bit-exact identical metrics across both workloads because working-set thrashing causes all candidates in the bottom-64 window to decay uniformly to $f=1$.
4. **100% Deterministic Reproducibility:**
   - Repetition 1 vs. Repetition 2 produced identical metrics across all 16 runs with zero drift.

---

## 3. Quick Links to Key Deliverables

- **Master Synthesis Report:** [`reports/FINAL_BIG_EXPERIMENT_REPORT.md`](file:///home/mohman/ca_project/mohman/point_4/big_exp/reports/FINAL_BIG_EXPERIMENT_REPORT.md)
- **Conservation and Metric Proofs:** [`reports/DETAILED_METRIC_ANALYSIS.md`](file:///home/mohman/ca_project/mohman/point_4/big_exp/reports/DETAILED_METRIC_ANALYSIS.md)
- **Determinism Audit:** [`reports/REPRODUCIBILITY_REPORT.md`](file:///home/mohman/ca_project/mohman/point_4/big_exp/reports/REPRODUCIBILITY_REPORT.md)
- **Consolidated CSV:** [`results/all_runs.csv`](file:///home/mohman/ca_project/mohman/point_4/big_exp/results/all_runs.csv)
- **Paired Comparisons CSV:** [`results/paired_comparisons.csv`](file:///home/mohman/ca_project/mohman/point_4/big_exp/results/paired_comparisons.csv)
- **Figures:** [`figures/png/`](file:///home/mohman/ca_project/mohman/point_4/big_exp/figures/png/) and [`figures/pdf/`](file:///home/mohman/ca_project/mohman/point_4/big_exp/figures/pdf/)

