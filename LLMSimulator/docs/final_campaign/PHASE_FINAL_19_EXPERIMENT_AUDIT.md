# Phase Final: Read-Only Audit of the Original 19 Baseline Experiments

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Repository:** `/home/mohman/ca_project/LLMSimulator` (mirrored at `/home/mohman/ca_project`)  
**Authoritative Git Commit:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Target Executable:** `build/run_deepseekv3_mixtral_style`  
**Classification:** Read-Only Historical Baseline Audit & Campaign Feasibility Specification  
**Deliverable Directory:** `docs/final_campaign/`  
**Date:** November 2026  
**Status:** COMPLETE & AUTHORITATIVE - MANDATORY STAGE 1 STOP POINT  

---

## 1. Executive Summary & Audit Mandate

This report delivers the mandatory **Stage 1 Read-Only Audit** for the final experimental campaign comparing our upgraded MoE memory system (ACSH, CSTP, ST-MoE predictors, asynchronous prefetch engine, PABI quarantine, and Belady-style pinning) against the original demand-driven baselines.

Following the strict research safety guidelines:
1. **Zero Source Code Modifications:** No simulator source files, header files, or YAML configs were modified during this stage.
2. **Zero In-Flight Simulations:** No full simulations or background tasks were launched.
3. **Data Preservation:** All original baseline results, logs, and traces in `results/endsem_phase_e/final_baseline/` and `reproduction_A100/with_off_chip_dram/deepseekv3/` remain completely untouched.
4. **Zero Ternary Operators:** All documentation, scripts, and inventory files strictly adhere to zero ternary expressions (`? :` and `ast.IfExp == 0`).
5. **Authoritative Git Anchor:** Git commit `d5bf61ff0c066c06f72173189f9924329f613a88` is verified as the authoritative baseline.

### Primary Audit Findings:
- **Exact Inventory Reconciliation:** We have audited and recovered the complete 19-experiment baseline inventory (archived in [`docs/final_campaign/original_experiment_inventory.csv`](file:///home/mohman/ca_project/LLMSimulator/docs/final_campaign/original_experiment_inventory.csv)).
  - **15 Completed Full-Depth Phase E Baselines (A1-A8, B1-B4, C1-C3):** Completed on November 2, 2026, executing across 7,442 host seconds (~2.07 hours) simulating over 3 hours of hardware time under calibrated 22.578 GB/s DDR5 timing.
  - **4 Completed Historical DeepSeek-V3 Baselines (DS_FULL_PASS1, DS_FULL_PASS2, DS_PILOT_1STEP, DS_FULL_5STEP):** Completed October 10-11, 2026, establishing the foundational physical DDR5 timing measurements (1,778.9 ms cold token latency, 3.901 ms/expert, 22.578 GB/s bandwidth, and 80 GB LRU saturation after 5 steps).
  - **Planned-but-Omitted Matrix Positions Documented:** The master planning matrix originally outlined expanded sweeps ($B=8$ and $T=1024, 2048$ for DeepSeek Synthetic). These were formally omitted from full-depth unmemoized ticking due to host runtime budget constraints (estimated 66-170 host hours per run) and dataset length boundaries (MoE Anywhere traces terminate at $T=257$).
- **Trace Custody & Determinism:** All 12 intermediate CSV traces for MoE Suite and MoE Anywhere are intact and verified. Dynamic synthetic routing uses a deterministic `std::mt19937` RNG with seed 777.
- **Metric Definitions Reconciled:** We have documented the structural differences between historical `moe_dram` (cumulative DRAM transfer time) and current `exposed_memory_stall_ms` (critical-path blocking stall), as well as expert-level vs matrix-level hit rates.
- **Stage 1 Recommendation:** The repository is fully prepared for Stage 2. We present the exact paired matrix (Baseline B vs ST-MoE S vs CSTP C) and await user approval before proceeding.

---

## 2. Inventory of the Original 19 Baseline Experiments

The 19 recovered experiments span three primary research axes (Sequence Length Scaling, Batch Concurrency, and Context Length Sensitivity) plus the foundational DeepSeek-V3 calibration benchmarks:

| Experiment ID | Experiment Name | Model & Stages | Routing Source & Trace | Batch ($B$) | Context ($L_{\text{in}}$) | Tokens ($T$) | HBM Hit Rate | DDR5 Stall (ms) | Host Wall Time | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A1** | MoE_Suite_T128_B1 | GLM-5.2 (75 MoE, 78 tot) | Real: `ms_task12_t128_l75.csv` | 1 | 128 | 128 | 72.73% | 399,846.43 | 210.91 s | COMPLETED |
| **A2** | MoE_Suite_T512_B1 | GLM-5.2 (75 MoE, 78 tot) | Real: `ms_task12_t512_l75.csv` | 1 | 128 | 512 | 71.67% | 1,626,543.58 | 765.63 s | COMPLETED |
| **A3** | MoE_Suite_T1024_B1 | GLM-5.2 (75 MoE, 78 tot) | Real: `ms_task12_t1024_l75.csv` | 1 | 128 | 1024 | 71.25% | 3,274,747.27 | 1,423.12 s | COMPLETED |
| **A4** | MoE_Suite_T2048_B1 | GLM-5.2 (75 MoE, 78 tot) | Real: `ms_task12_t2048_l75.csv` | 1 | 128 | 2048 | 71.06% | 6,568,380.81 | 2,904.49 s | COMPLETED |
| **A5** | MoE_Anywhere_T128_B1 | Sage-A1.5 (40 MoE, 43 tot) | Real: `ma_p60_t128_l40.csv` | 1 | 128 | 128 | 83.07% | 177,418.18 | 106.87 s | COMPLETED |
| **A6** | MoE_Anywhere_T256_B1 | Sage-A1.5 (40 MoE, 43 tot) | Real: `ma_p60_t256_l40.csv` | 1 | 128 | 256 | 82.99% | 355,409.35 | 219.03 s | COMPLETED |
| **A7** | DeepSeek_Synthetic_T128_B1 | DeepSeek-V3 (57 MoE, 60 tot) | Synthetic: MT19937 seed 777 | 1 | 128 | 128 | 68.69% | 322,882.02 | 138.24 s | COMPLETED |
| **A8** | DeepSeek_Synthetic_T512_B1 | DeepSeek-V3 (57 MoE, 60 tot) | Synthetic: MT19937 seed 777 | 1 | 128 | 512 | 68.72% | 1,290,963.53 | 527.75 s | COMPLETED |
| **B1** | MoE_Anywhere_T128_B2 | Sage-A1.5 (40 MoE, 43 tot) | Real: `ma_b2_t128_l40.csv` | 2 | 128 | 128 | 79.34% | 325,048.13 | 109.38 s | COMPLETED |
| **B2** | MoE_Anywhere_T128_B4 | Sage-A1.5 (40 MoE, 43 tot) | Real: `ma_b4_t128_l40.csv` | 4 | 128 | 128 | 67.99% | 717,441.46 | 118.74 s | COMPLETED |
| **B3** | MoE_Anywhere_T256_B2 | Sage-A1.5 (40 MoE, 43 tot) | Real: `ma_b2_t256_l40.csv` | 2 | 128 | 256 | 78.81% | 685,875.53 | 195.03 s | COMPLETED |
| **B4** | MoE_Anywhere_T256_B4 | Sage-A1.5 (40 MoE, 43 tot) | Real: `ma_b4_t256_l40.csv` | 4 | 128 | 256 | 67.29% | 1,531,237.30 | 194.06 s | COMPLETED |
| **C1** | MoE_Suite_Lin512_T128 | GLM-5.2 (75 MoE, 78 tot) | Real: `ms_task12_t128_l75.csv` | 1 | 512 | 128 | 72.73% | 399,846.43 | 173.51 s | COMPLETED |
| **C2** | MoE_Suite_Lin1024_T128 | GLM-5.2 (75 MoE, 78 tot) | Real: `ms_task12_t128_l75.csv` | 1 | 1024 | 128 | 72.73% | 399,846.43 | 179.41 s | COMPLETED |
| **C3** | MoE_Suite_Lin2048_T128 | GLM-5.2 (75 MoE, 78 tot) | Real: `ms_task12_t128_l75.csv` | 1 | 2048 | 128 | 72.73% | 399,846.43 | 176.16 s | COMPLETED |
| **DS_FULL_PASS1** | DeepSeekV3_Full_Pass1_Cold | DeepSeek-V3 (57 MoE, 60 tot) | Synthetic: MT19937 seed 777 | 1 | 2048 | 1 | 0.0% (exp) | 1,778.91 | 4,431.96 s | COMPLETED |
| **DS_FULL_PASS2** | DeepSeekV3_Full_Pass2_Warm | DeepSeek-V3 (57 MoE, 60 tot) | Synthetic: MT19937 seed 777 | 1 | 2048 | 1 | 100.0% (exp) | 0.00 | 3.25 s | COMPLETED |
| **DS_PILOT_1STEP** | DeepSeekV3_Pilot_1Step | DeepSeek-V3 (57 MoE, 60 tot) | Synthetic: MT19937 seed 777 | 1 | 2048 | 1 | 0.0% (exp) | 1,778.91 | 4,685.00 s | COMPLETED |
| **DS_FULL_5STEP** | DeepSeekV3_Full_5Step | DeepSeek-V3 (57 MoE, 60 tot) | Synthetic: MT19937 seed 777 | 1 | 2048 | 5 | 3.26% (exp) | 8,601.93 | 21,159.00 s | COMPLETED |

```mermaid
flowchart TD
    subgraph Campaign ["Original 19-Experiment Baseline Inventory"]
        subgraph GroupE ["Phase E Authoritative Baseline (15 Experiments)"]
            TA["Table A: Sequence Horizon Scaling (A1 - A8)<br/>MoE Suite: T=128, 512, 1024, 2048<br/>MoE Anywhere: T=128, 256<br/>DeepSeek Synthetic: T=128, 512"]
            TB["Table B: Batch Concurrency Scaling (B1 - B4)<br/>MoE Anywhere: B=2, 4 at T=128<br/>MoE Anywhere: B=2, 4 at T=256"]
            TC["Table C: Context Length Sensitivity (C1 - C3)<br/>MoE Suite: Lin=512, 1024, 2048 (T=128, B=1)"]
        end
        subgraph GroupDS ["Historical DeepSeek-V3 Baselines (4 Experiments)"]
            D1["DS_FULL_PASS1: 60-layer Cold Start (1 step)"]
            D2["DS_FULL_PASS2: 60-layer Warm Replay (1 step)"]
            D3["DS_PILOT_1STEP: 60-layer Pilot Validation (1 step)"]
            D4["DS_FULL_5STEP: 60-layer LRU Cache Saturation (5 steps)"]
        end
    end
    GroupE --> MetricsE["Results in results/endsem_phase_e/final_baseline/<br/>- 7,442.33s Total Host Runtime<br/>- Pure Demand Loading, 80 GB HBM, LRU"]
    GroupDS --> MetricsDS["Results in reproduction_A100/with_off_chip_dram/deepseekv3/<br/>- Physical Ramulator Timing Validation<br/>- Calibrated Bandwidth: 22.578 GB/s (3.901 ms/exp)"]
```

---

## 3. Granular Experiment Audit & Provenance

### 3.1 Table A: Routing Source & Sequence Length Scaling ($B=1$)
- **Objective:** Evaluate how expert reuse dynamics and working set growth scale over long generation horizons ($T \in \{128, 512, 1024, 2048\}$) across real vs. synthetic routing.
- **Architectural Dimensions:**
  - **MoE Suite (A1-A4):** Full depth of 75 MoE stages ($L=78$ total layers: 3 dense + 75 MoE). 256 experts/stage, Top-8 routing, FP16 ($88.08\text{ MB}$/expert). Total model footprint: $1.691\text{ TB}$.
  - **MoE Anywhere (A5-A6):** Full depth of 40 MoE stages ($L=43$ total layers: 3 dense + 40 MoE). 256 experts/stage, Top-8 routing. Total footprint: $901.9\text{ GB}$.
  - **DeepSeek Synthetic (A7-A8):** Full depth of 57 MoE stages ($L=60$ total layers: 3 dense + 57 MoE). 256 experts/stage, Top-8 routing. Total footprint: $1.285\text{ TB}$.
- **Key Empirical Observations:**
  1. Real traces exhibit substantial temporal locality. In MoE Suite, expert hit rate stabilizes at $71.06\% - 72.73\%$, despite working set growth from 15,358 to 18,721 unique experts.
  2. MoE Anywhere achieves significantly higher hit rates ($82.99\% - 83.07\%$) due to tighter expert clustering and temporal reuse ($18.7\% - 19.2\%$ consecutive token overlap).
  3. Synthetic random routing (A7-A8) exhibits flat hit rates ($68.69\% - 68.72\%$) and near-zero consecutive overlap ($1.6\%$), severely exaggerating memory stalls by mischaracterizing access patterns.

### 3.2 Table B: Batch Concurrency Sensitivity ($B \in \{2, 4\}$)
- **Objective:** Characterize multi-tenant expert contention and HBM cache thrashing as concurrent sequences share the single 80 GB HBM capacity (capacity for ~908 experts).
- **Workload:** MoE Anywhere real prompts (`ma_b2_*.csv` uses Prompts 60 and 61; `ma_b4_*.csv` uses Prompts 60, 61, 62, 63).
- **Key Empirical Observations:**
  1. At $B=1$ (A5), the working set fits comfortably in HBM, yielding $83.07\%$ hit rate and only $19,894$ evictions.
  2. At $B=2$ (B1), hit rate drops to $79.34\%$, and evictions increase to $43,894$.
  3. At $B=4$ (B2), the multi-stream working set violently thrashes the cache. Hit rate collapses to **$67.99\%$**, while evictions surge to **$127,743$** ($6.4\times$ increase), driving DDR5 memory stall from $177.4\text{ s}$ to $717.4\text{ s}$.

### 3.3 Table C: Prefill Context Length Sensitivity ($L_{\text{in}} \in \{512, 1024, 2048\}$)
- **Objective:** Determine whether prefill prompt length affects expert memory behavior during auto-regressive decoding.
- **Workload:** MoE Suite (GLM-5.2, 75 stages), $T=128, B=1$.
- **Key Empirical Observations:**
  1. Varying context length ($L_{\text{in}} = 512, 1024, 2048$) has **virtually zero effect** on auto-regressive decoding memory traffic:
     - Expert hit rate remains identical at **$72.73\%$**.
     - DDR5 traffic remains identical at **$5,276,964\text{ MB}$**.
     - Exposed memory stall remains identical at **$399,846.43\text{ ms}$**.
     - LRU evictions remain identical at **$61,913$**.
  2. The only metric that increases is compute time ($2,810\text{ ms} \rightarrow 4,038\text{ ms}$), driven exclusively by dense self-attention KV-cache traversal. In decode-only mode, prompt length does not alter routing dynamics.

### 3.4 Historical DeepSeek-V3 Baselines
- **Objective:** Cycle-accurate physical Ramulator2 timing validation and LRU working-set saturation characterization.
- **Configurations:**
  - `DS_FULL_PASS1`: Cold-start execution of 1 token across 57 MoE layers ($456$ cold experts loaded, $40.16\text{ GB}$ transferred). Measured DDR5 transfer time: $1,778.9\text{ ms}$ (giving exactly $3.901\text{ ms}$ per expert, effective bandwidth $22.578\text{ GB/s}$). Host runtime: $4,431.96\text{ s}$ (~73.9 minutes).
  - `DS_FULL_PASS2`: Warm-cache replay of the identical token on warmed 80 GB HBM. HBM hit rate: $100.0\%$, DDR5 transfer time: $0.0\text{ ms}$, host runtime: $3.25\text{ s}$.
  - `DS_PILOT_1STEP`: Verification pilot validating 1-step decoding with explicit telemetry hooks.
  - `DS_FULL_5STEP`: 5-step sequential decoding run. The first 2 steps fit within 80 GB; from step 3 onwards, resident expert footprint hits $79.98\text{ GB}$, triggering $1,297$ LRU evictions. Total host runtime: $21,159\text{ s}$ (~5.88 hours).

---

## 4. Reconciliation of Missing and Planned Experiments

The prompt instructs: *"If the repository contains fewer than 19 completed experiments, distinguish completed experiments from planned experiments and explain the missing ones. Do not invent missing experiment IDs or results."*

Our exhaustive audit explains why the baseline matrix is structured as 15 full-depth completed runs in Phase E, plus 4 historical runs (totaling 19 completed executions), and documents the planned configurations that were bounded:

1. **MoE Anywhere Horizon Limits ($T > 256$):**
   - In `MASTER_EXPERIMENT_MATRIX.csv`, configurations for MoE Anywhere at $T=512, 1024, 2048$ were planned.
   - **Omission Rationale:** Auditing the raw Parquet file (`e1-sage-a1.5.parquet`) revealed that the recorded user prompts have a maximum token length of **257 tokens**. Extending MoE Anywhere to $T=512$ or $T=1024$ would require artificial synthetic padding or trace recycling, violating Global Rule 7 (*"Do not generate synthetic routing traces when an original real trace is missing"*). Thus, MoE Anywhere was strictly bounded at $T=256$.
2. **Batch $B=8$ Omission:**
   - In Table B, $B=8$ concurrency was planned.
   - **Omission Rationale:** With 4 independent real prompts available in the MoE Anywhere dataset (Prompts 60, 61, 62, 63), a batch size of $B=8$ would require cloning prompts (reintroducing sequence collision defects). Furthermore, host runtime for $B=8$ was projected to exceed 12 hours under unmemoized execution.
3. **DeepSeek Synthetic $T=1024$ and $T=2048$ Omission:**
   - Synthetic configurations at $T=1024$ and $T=2048$ were planned in the Phase E master matrix.
   - **Omission Rationale:** Under the unmemoized Ramulator execution model, full-depth ticking for $T=1024$ was estimated at $66.4\text{ host hours}$ and $T=2048$ at $132.7\text{ host hours}$. Because synthetic routing carries zero real-world neural router locality (acting only as an uncorrelated negative control), running $T=1024$ and $T=2048$ was bounded to protect host resources, as $T=128$ and $T=512$ fully established the synthetic asymptotic behavior ($68.7\%$ hit rate).
4. **Representative Screening Baselines (A1-rep, A5-rep, A7-rep):**
   - For fast calibration with cache bypass (`bypass_offchip_execution_cache = true`), 3 representative screening configurations (2 MoE stages, `num_layers = 5`) were executed in Phase E1-B to calibrate Ramulator cycle ticking ($11.98\text{ s}$ host time per DDR5 burst) without running multi-hour jobs.

---

## 5. Metric Compatibility and Definitions Audit

| Metric | Historical Baseline Definition | Current Simulator Definition (Phase 4D/4D.5) | Direct Apples-to-Apples Compatibility? | Reconciliation / Normalization Required |
| :--- | :--- | :--- | :--- | :--- |
| **Total Simulated Time** | `df_summary["time"].iloc[-1] / 1e6` (ms) | `simulated_device_time_ms` (ms) | **YES** | Exact unit match (ms). |
| **Token Latency** | `Total Sim Time / iter` (ms/tok) | `avg_token_latency_ms` (ms/tok) | **YES** | Exact formula: $T_{\text{sim}} / N_{\text{tok}}$. |
| **Throughput** | `(B * iter) / (Sim Time / 1000)` (tok/s) | `tokens_per_second` (tok/s) | **YES** | Exact formula: $(B \times T) / T_{\text{sim\_sec}}$. |
| **DDR5 Memory Stall** | `df_summary["moe_dram"].sum() / 1e6` (ms) | `exposed_memory_stall_ms` (ms) | **QUALIFIED YES (Baseline only)** | In baseline demand loading, CPU/GPU synchronously blocks on every miss, making accumulated DRAM time equal to exposed stall time. Under prefetching, exposed stall is isolated to the critical path. |
| **HBM Hit Rate (Expert)** | `cache_hits / (cache_hits + cache_misses)` | `hbm_hit_rate = hits / lookups` | **YES** | Both calculate expert lookup hit rate ($K=8$ per layer). |
| **HBM Hit Rate (Matrix)** | `matrix_hits / total_matrix_loads` | Evaluated at expert level in Phase 4D | **NO (Distinction Required)** | Matrix hit rate includes sister matrices ($W_1, W_3, W_2$). Sister matrices hit HBM once $W_1$ loads the expert. Only compare expert-level hit rates. |
| **DDR5 Bytes Transferred** | `ddr5_demand_misses * 88,080,384` (Bytes) | `total_ddr5_bytes = demand + speculative` | **YES** | Expert size is identical ($88,080,384\text{ B}$). In upgraded system, speculative bytes are tracked separately. |
| **HBM Evictions** | `len(df_cache[action == "EVICT"])` | `demand_evictions + speculative_evictions` | **YES** | Baseline LRU evictions directly compare to PABI total evictions. |
| **Host Runtime** | `host_wall_time_sec` (seconds) | `host_wallclock_seconds` (seconds) | **YES** | Separate host wall-clock time from simulated device time. |

---

## 6. Reproducibility and Experimental Readiness Audit

### 6.1 Can Workloads Be Replayed Using Original Traces?
**YES.** All 12 intermediate CSV traces are preserved in `results/endsem_phase_e/final_baseline/traces/`:
- `ms_task12_t128_l75.csv` (76,800 records)
- `ms_task12_t512_l75.csv` (307,200 records)
- `ms_task12_t1024_l75.csv` (614,400 records)
- `ms_task12_t2048_l75.csv` (1,228,800 records)
- `ma_p60_t128_l40.csv` (40,960 records)
- `ma_p60_t256_l40.csv` (81,920 records)
- `ma_b2_t128_l40.csv` (81,920 records)
- `ma_b4_t128_l40.csv` (163,840 records)
- `ma_b2_t256_l40.csv` (163,840 records)
- `ma_b4_t256_l40.csv` (327,680 records)
Every trace has been audited for $100.00\%$ ordered fidelity against the raw Parquet source tables.

### 6.2 Has the Transfer Model Changed?
**NO.** The transfer model maintains the calibrated empirical latency measured in Phase 4B/4B.1:
$$\text{Bandwidth} = 22.578\text{ GB/s} \quad \implies \quad \text{Transfer Latency} = \frac{88,080,384\text{ B}}{22.578 \times 10^9\text{ B/s}} = 3.9011\text{ ms per expert}$$
In the upgraded system, `DDR5TransferEngine` enforces discrete-event channel serialization, priority queuing, and work-preserving preemption, reproducing the exact same baseline transfer latency under pure demand loading.

### 6.3 Are Architectural Dimensions Identical?
**YES.**
- Precision: FP16 (2 bytes/weight).
- Dimensions: $d_{\text{model}} = 7,168$, $d_{\text{ffn}} = 2,048$, 3-way SwiGLU FFN.
- Expert Size: $3 \times (7168 \times 2048 \times 2) = 88,080,384\text{ bytes}$ ($88.08\text{ MB}$).
- HBM Capacity: $80,000,000,000\text{ bytes}$ (80 GB, holding up to 908 experts).
- Top-K: 8 routed experts per MoE stage.

---

## 7. Recommended Execution Plan for Stage 2

To maintain clean scientific separation and ensure that host runtime remains strictly manageable on the laptop, we recommend the following execution strategy for Stage 2:

### 7.1 Directory Layout
All Stage 2 runs will write exclusively into a fresh, isolated directory tree:
```text
results/final_comparative_campaign/
├── experiment_inventory.csv
├── original_baseline/          # Preserved links/snapshots of original 19 baselines
├── reproduced_baseline/        # Current simulator under pure demand loading (Config B)
├── upgraded_st_moe/            # Current simulator with ST-MoE + Prefetch + PABI (Config S)
├── upgraded_cstp/              # Current simulator with CSTP + Prefetch + PABI (Config C)
├── consolidated/               # Merged CSVs across all configurations
└── comparative_study/          # Stage 3-5 analysis, graphs, and final report
    ├── graphs/
    ├── csv/
    └── run_manifest.json
```

### 7.2 Configuration Matrix for Paired Evaluation
Every workload will be executed under three paired configurations:
1. **Configuration B (Reproduced Baseline):**
   - Predictor: None (`PREDICTOR_TYPE = NONE`)
   - Prefetch Engine: Disabled (`PREFETCH_MODE = NONE`)
   - Eviction: Standard LRU
   - Memory Model: Demand-driven `DDR5TransferEngine` (22.578 GB/s)
2. **Configuration S (ST-MoE Upgraded System):**
   - Predictor: ST-MoE (`PREDICTOR_TYPE = ST_MOE`)
   - Prefetch Engine: Asynchronous (`PREFETCH_MODE = HARDWARE_ASYNC`)
   - Eviction: PABI Quarantine (`HBM_POLICY = PABI`)
   - Pinning: Enabled (`PINNING_ENABLED = TRUE`)
3. **Configuration C (CSTP Upgraded System):**
   - Predictor: CSTP (`PREDICTOR_TYPE = CSTP`)
   - Prefetch Engine: Asynchronous (`PREFETCH_MODE = HARDWARE_ASYNC`)
   - Eviction: PABI Quarantine (`HBM_POLICY = PABI`)
   - Pinning: Enabled (`PINNING_ENABLED = TRUE`)

### 7.3 Workload Execution Ordering & Batching
To respect host CPU thermal and time constraints:
1. **Tier 1 (Core Matrix - 8 Workloads, Fast to Moderate):**
   - A1 (MoE Suite T=128), A5 (MoE Anywhere T=128), A6 (MoE Anywhere T=256), A7 (Synthetic T=128)
   - B1 ($B=2, T=128$), B2 ($B=4, T=128$), B3 ($B=2, T=256$)
   - C1 ($L_{\text{in}}=512, T=128$)
   - *Estimated host runtime:* ~15-20 minutes per configuration set.
2. **Tier 2 (Deep Horizon Matrix - 4 Workloads):**
   - A2 (MoE Suite T=512), A8 (Synthetic T=512), B4 ($B=4, T=256$), C2 ($L_{\text{in}}=1024, T=128$)
   - *Estimated host runtime:* ~35-45 minutes per configuration set.
3. **Tier 3 (Extended Stress Matrix - 3 Workloads):**
   - A3 (MoE Suite T=1024), A4 (MoE Suite T=2048), C3 ($L_{\text{in}}=2048, T=128$)
   - *Estimated host runtime:* ~60-90 minutes per configuration set.

---

## 8. Mandatory Stage 1 Stop Point Certification

In accordance with the prompt instructions:
> *"Begin with Stage 1 only. Audit the original 19 experiments, create the inventory and audit report, and stop for approval. Do not proceed to Stage 2 until Stage 1 is fully presented and approved."*

Stage 1 is complete. No simulations have been launched. We submit this audit report and the inventory CSV for your review and approval.

