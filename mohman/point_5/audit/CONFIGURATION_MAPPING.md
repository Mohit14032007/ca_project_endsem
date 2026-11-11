# Configuration Mapping: Four Conceptual Configurations

## 1. Overview of the Four Required Configurations

Point 5 requires evaluating the MoE inference architecture across four conceptual configurations:
1. **Configuration A: Baseline Without Prefetching** (Demand-only expert loading under Standard LRU).
2. **Configuration B: Prediction Without Prefetching** (Predictor runs in shadow mode; no speculative memory transfers issued; HBM cache unchanged).
3. **Configuration C: Prediction-Driven Prefetching** (Asynchronous prefetching under Standard LRU; evaluated for both ST-MoE and CSTP).
4. **Configuration D: Complete Implementation** (CSTP predictor + asynchronous prefetching + Frequency-Aware LRU with $W=64$).

This document maps our existing experimental campaigns to these four configurations.

---

## 2. Conceptual Configuration Definitions and Requirements

```
                                  Architecture Mapping Matrix

  Configuration      Predictor     Prefetching    Eviction Policy   DDR5 Speculative Transfers   HBM Alterations
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Config A           None          Disabled       Standard LRU      None (0 Bytes)               Demand Misses Only
  Config B           ST-MoE/CSTP   Shadow Mode    Standard LRU      None (0 Bytes)               Demand Misses Only
  Config C1          ST-MoE        Enabled        Standard LRU      Active (DDR5 -> HBM)         Polluted by Speculation
  Config C2          CSTP          Enabled        Standard LRU      Active (DDR5 -> HBM)         Polluted by Speculation
  Config D           CSTP          Enabled        FA-LRU (W=64)     Active (DDR5 -> HBM)         Speculative Lines Filtered
```

---

## 3. Campaign Mapping and Coverage Matrix

| Experimental Campaign | Workload Scale | Config A (Baseline) | Config B (Pred-Only) | Config C1 (ST-MoE LRU) | Config C2 (CSTP LRU) | Config D (Complete System) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Point 4 Big Experiment** | $B \in \{14, 32\}, T=128, L=40, M=35\text{ GiB}$ | `c1` | Missing In-Situ | Not Executed | `c3` | `c4` ($W=64$) |
| **Point 4 Phase 3 Focused** | $B \in \{1, 2, 4, 8\}, T=20, L=40, M \in [10, 35]\text{ GiB}$ | `c1` | Missing In-Situ | Not Executed | `c3` | `c4` ($W=16$) |
| **Point 3 Phase 2 Large** | $B \in [14, 70], T \in [4, 64], L=40, M=80\text{ GB}$ | `baseline` | Missing In-Situ | `stmoe_h1/h2` | `cstp_h1/h2` | Not Available |
| **Historical Phase 4d5c** | $B=1, T=10, L=78, M=80\text{ GB}$ | `c01` | `c02`/`c03` | `c04` | `c05` | `c06`-`c09` (PABI) |
| **Point 1 Trace Evaluation** | Sage-A1.5, 74 prompts $\times 257$ tokens | N/A | Offline Extracted | Offline Extracted | Offline Extracted | N/A |

---

## 4. Detailed Mapping of Each Configuration

### Configuration A: Baseline Without Prefetching
- **Definition**: Demand-only expert loading from DDR5 into HBM upon execution miss. Eviction managed by true Least Recently Used (LRU) timestamp order. Prefetching framework explicitly disabled (`prefetching.enabled: false`).
- **Primary Matched Source**:
  - Workload W3 ($B=14, T=128$): `mohman/point_4/big_exp/results/all_runs.csv` (`w3_c1_r1`, `w3_c1_r2`).
  - Workload W4 ($B=32, T=128$): `mohman/point_4/big_exp/results/all_runs.csv` (`w4_c1_r1`, `w4_c1_r2`).
- **Supporting Sources**:
  - Point 3 Large Workload: `w1_baseline` through `w7_baseline`.
  - Point 4 Phase 3: `w1_cap5_c1`, `w2_cap1_c1` through `w2_cap5_c1`, `w3_cap5_c1`, `w4_cap5_c1`.
  - Historical `phase_4d5c`: `c01_baseline_lru`.

### Configuration B: Prediction Without Prefetching (Shadow Mode)
- **Definition**: Predictor (ST-MoE or CSTP) executes concurrently with inference. Predictions are generated, evaluated against ground truth, and recorded for precision/recall. Speculative prefetch transfers are completely suppressed (`mode: prediction_only`). HBM cache contents and DDR5 bus traffic remain strictly identical to Configuration A.
- **Primary In-Situ Source**:
  - Historical `phase_4d5c`:
    - `c02_stmoe_predonly`: Evaluates ST-MoE in shadow mode. Precision: 0.3387, Recall: 0.1693, F1: 0.2258, Hit Rate: 0.7360, Predictor latency: $42.45\text{ }\mu\text{s}$, Host overhead: $0.25\%$. DDR5 bytes and device execution latency are identical to `c01_baseline_lru` (17,514,000,000 ns).
    - `c03_cstp_predonly`: Evaluates CSTP in shadow mode. Precision: 0.3400, Recall: 0.1700, F1: 0.2267, Hit Rate: 0.7307, Predictor latency: $62.74\text{ }\mu\text{s}$, Host overhead: $0.25\%$. DDR5 bytes and device execution latency are identical to `c01_baseline_lru` (17,514,000,000 ns).
- **Offline Trace Evaluation Source**:
  - Point 1: `mohman/point_1/stmoe_cstp/results/csv/corrected_aggregate_metrics.csv` (measures Macro Recall: 0.4791 for ST-MoE vs. 0.4551 for CSTP; Hit Rate: 0.9694 vs. 0.9667 on the full Sage-A1.5 dataset).
- **Status for Big Experiment Workloads (W3/W4, $T=128$)**:
  - Not executed in-situ during the Big Experiment campaign.
  - Because shadow mode suppresses all memory transfers by definition, its simulated GPU execution latency, DDR5 traffic, and HBM resident hit counts are mathematically identical to Configuration A (`w3_c1` and `w4_c1`). Only CPU host overhead ($\approx 0.25\%$) differs.

### Configuration C: Prediction-Driven Prefetching
- **Definition**: Predictor dispatches speculative expert prefetch requests to the asynchronous DDR5 transfer engine ahead of execution. Standard LRU manages HBM cache residency.
- **Sub-configuration C1 (ST-MoE + LRU)**:
  - Primary Source: Point 3 Large Workload Campaign (`w1_stmoe_h2` to `w7_stmoe_h2`).
  - Historical In-Situ: `c04_stmoe_prefetch_lru` in `phase_4d5c`.
- **Sub-configuration C2 (CSTP + LRU)**:
  - Primary Matched Source: Point 4 Big Experiment (`w3_c3_r1`/`r2` for $B=14$; `w4_c3_r1`/`r2` for $B=32$).
  - Point 3 Large Workload Source: `w1_cstp_h2` to `w7_cstp_h2`.
  - Point 4 Phase 3 Source: `w2_cap1_c3` to `w2_cap5_c3`, `w1_cap5_c3`, `w3_cap5_c3`, `w4_cap5_c3`.
  - Historical In-Situ: `c05_cstp_prefetch_lru` in `phase_4d5c`.

### Configuration D: Complete Implementation
- **Definition**: CSTP predictor generating speculative prefetches into the event-driven DDR5 transfer engine, coupled with Frequency-Aware LRU (FA-LRU, $W=64$) managing HBM cache residency. Demand-only frequency updating shields repeatedly accessed experts and preferentially evicts unconsumed speculative prefetch lines.
- **Primary Matched Source**:
  - Workload W3 ($B=14, T=128$): Point 4 Big Experiment (`w3_c4_r1`, `w3_c4_r2`).
  - Workload W4 ($B=32, T=128$): Point 4 Big Experiment (`w4_c4_r1`, `w4_c4_r2`).
- **Supporting Source**:
  - Point 4 Phase 3 Focused Campaign ($W=16$): `w2_cap1_c4` to `w2_cap5_c4`, `w1_cap5_c4`, `w3_cap5_c4`, `w4_cap5_c4`.

---

## 5. Direct Paired Comparison Pathways

To maintain strict scientific integrity, comparisons must be partitioned into strictly matched pairs and cross-campaign supporting trends:

```
               Primary Strictly Matched Comparison Hierarchy (Big Experiment W3/W4)
               
   [Config A: Baseline LRU]
           │
           │  (Combined effect of CSTP prefetching under LRU: ΔStall, +Traffic, +Hits)
           ▼
   [Config C2: CSTP + LRU]
           │
           │  (Incremental contribution of FA-LRU: -Reloads, -Traffic, -Stall)
           ▼
   [Config D: CSTP + FA-LRU (W=64)]
   
   ─────────────────────────────────────────────────────────────────────────────────────
   
              Cross-Predictor Comparison Pathway (Point 3 Large Workload W1–W7)
              
   [Config A: Baseline] ────────► [Config C1: ST-MoE + LRU] vs [Config C2: CSTP + LRU]
                                (Isolates predictor accuracy & stall reduction trade-off)
```

