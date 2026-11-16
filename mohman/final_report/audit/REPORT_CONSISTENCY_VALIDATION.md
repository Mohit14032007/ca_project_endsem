# REPORT CONSISTENCY VALIDATION REPORT

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference
**Total Automated Checks:** 53
**Passed Checks:** 53
**Failed Checks:** 0
**Overall Test Suite Status:** PASS (100% Consistent)

---

## Automated Test Gate Breakdown

### Gate 1: Figure Provenance
- **Passed:** 32 | **Failed:** 0
  - [PASS] Data provenance: figure_3_1_data.csv
  - [PASS] Data provenance: figure_3_2_data.csv
  - [PASS] Data provenance: figure_3_3_data.csv
  - [PASS] Data provenance: figure_3_4_data.csv
  - [PASS] Data provenance: figure_3_5_data.csv
  - [PASS] Data provenance: figure_5_1_data.csv
  - [PASS] Data provenance: figure_5_2_data.csv
  - [PASS] Data provenance: figure_5_3_data.csv
  - [PASS] Data provenance: figure_5_4_data.csv
  - [PASS] Data provenance: figure_5_5_data.csv
  - [PASS] Data provenance: figure_5_6_data.csv
  - [PASS] Data provenance: figure_5_7_data.csv
  - [PASS] Data provenance: figure_5_8_data.csv
  - [PASS] Data provenance: figure_5_9_data.csv
  - [PASS] Data provenance: figure_5_10_data.csv
  - [PASS] Data provenance: figure_5_11_data.csv
  - [PASS] Data provenance: figure_5_12_data.csv
  - [PASS] Data provenance: figure_5_13_data.csv
  - [PASS] Data provenance: figure_5_14_data.csv
  - [PASS] Data provenance: figure_6_1_data.csv
  - [PASS] Data provenance: figure_7_1_data.csv
  - [PASS] Data provenance: figure_7_2_data.csv
  - [PASS] Data provenance: figure_7_3_data.csv
  - [PASS] Data provenance: figure_8_1_data.csv
  - [PASS] Data provenance: figure_8_2_data.csv
  - [PASS] Data provenance: figure_8_3_data.csv
  - [PASS] Data provenance: figure_8_4_data.csv
  - [PASS] Data provenance: figure_8_5_data.csv
  - [PASS] Data provenance: figure_8_6_data.csv
  - [PASS] Data provenance: figure_9_1_data.csv
  - [PASS] Data provenance: figure_9_2_data.csv
  - [PASS] Data provenance: figure_9_3_data.csv

### Gate 2: Graph vs Table
- **Passed:** 3 | **Failed:** 0
  - [PASS] Fig 7.2 vs Table 7.1 Config A to D stall reduction
  - [PASS] Fig 7.3 DDR5 traffic sum for Config C (37.81 + 10.93 = 48.74 GB)
  - [PASS] Fig 9.1 vs Table 9.2 Baseline energy components sum to 9387.61 J

### Gate 3: Derived Mathematical Calculations
- **Passed:** 5 | **Failed:** 0
  - [PASS] 84 MiB transfer duration at 22.578 GB/s == 3.901 ms
  - [PASS] Table 4.2 clock-exact late-attachment timeline arithmetic
  - [PASS] Config A EDP = Energy * Time == 7202.85 kJ·s
  - [PASS] Config D-Opt K=2 EDP == 7079.32 kJ·s
  - [PASS] DeepSeek MLA KV Cache footprint at P=2048 matches formula

### Gate 4: Experiment Campaign Separation
- **Passed:** 3 | **Failed:** 0
  - [PASS] Table 8.1 excludes Point 3 W7 6.717s value
  - [PASS] Table 8.1 includes authoritative Point 6 H=1 hidden stall (3,568.90 ms)
  - [PASS] Table 5.1 retains W7 authoritative large-workload values

### Gate 5: Metric Naming & Definitions
- **Passed:** 3 | **Failed:** 0
  - [PASS] Table 3.2 uses percentage points (pp) for gain column
  - [PASS] Figure 3.3 accurately reflects 3 adaptation policies
  - [PASS] Figure 3.5 removes stale threshold sweep description

### Gate 6: Unit Consistency
- **Passed:** 3 | **Failed:** 0
  - [PASS] Table 4.3 states exact 88,080,384 bytes for 84 MiB expert
  - [PASS] Figure 7.3 DDR5 traffic consistently labeled in GB
  - [PASS] Table 9.2 EDP column labeled in kJ·s

### Gate 7: Report Claims Verification
- **Passed:** 2 | **Failed:** 0
  - [PASS] All executive summary claims match authoritative datasets
  - [PASS] Reconciliation matrix comprehensive (120 items >= 100)

### Gate 8: 32-Figure Graph Inventory
- **Passed:** 2 | **Failed:** 0
  - [PASS] All 32 expected figure data exports present (32/32)
  - [PASS] All 32 figures have detailed 10-point academic explanations (32/32)

---

## Numerical Discrepancy & Resolution Summary

✓ **Zero unresolved discrepancies across all 32 figures, 15 tables, derivations, and text claims.**

1. **Transfer Latency Reconciled:** 84 MiB @ 22.578 GB/s is mathematically 3.901 ms (88,080,384 B / 22.578e9 B/s). Table 4.2 timeline sequence completely synchronized with microsecond precision.
2. **Campaign Separation Enforced:** Point 3 W7 large workload (6.717 s execution time savings) is strictly separated from Point 6 lookahead sensitivity (H=1 hidden stall: 3,568.90 ms, H=2: 3,873.24 ms, H=4: 4,005.17 ms).
3. **Prediction Accuracy Disambiguated:** CSTP 82.02% vs. ST-MoE 68.12% is strictly reported as a +13.90 percentage-point difference (pp) and +20.40% relative improvement.
4. **Figure 3.3 and Figure 3.5 Semantic Alignment:** Stale threshold-sweep and 40-layer descriptions replaced with exact plotted data (Figure 3.3 evaluates Frozen, Online Within-Req, and Online Across-Req; Figure 3.5 evaluates Additive vs. Intersection scoring modes).
5. **Energy & EDP Components Reconciled:** Subsystem energy breakdown (Static DRAM 3,452.72 J, DDR5 4,820.77 J, HBM 1,042.87 J, GPU MAC 71.25 J) sums to 9,387.61 J. K=2 sweet spot verified at 9,359.93 J (-27.68 J / -0.29% net savings) with best EDP of 7,079.32 kJ·s (-1.72%).