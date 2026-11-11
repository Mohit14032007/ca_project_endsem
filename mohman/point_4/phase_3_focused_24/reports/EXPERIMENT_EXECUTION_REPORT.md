# Point 4 Phase 3: Experiment Execution Report — Focused 24-Run Campaign

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Harness Executable:** `run_deepseekv3_mixtral_style`  
**Execution Date:** 2026-11-10  
**Status:** COMPLETE (24/24 runs succeeded with Exit Code 0)  
**Total Wall-Clock Execution Time:** 318.10 seconds  

---

## 1. Executive Summary

This report documents the end-to-end execution of the focused 24-run experimental campaign for **Point 4: HBM Management (LRU vs. Frequency-Aware LRU Evaluation)**. Following the certified Part A validation (Phase 2.6), all 24 simulation runs were executed using the full-depth 40-stage MoE model, event-driven DRAM modeling with calibrated 64.0 GB/s DDR5 bandwidth, real routing traces from MoE-Anywhere (`e1-sage-a1.5.parquet`), and explicit environment controls for eviction policies and exact HBM capacities.

Every single simulation run finished cleanly with exit code 0, without process crashes, memory allocation errors, or timeout conditions. All generated metrics were verified against the fundamental conservation invariants of memory capacity, eviction classification mutual exclusion, prefetch-induced attribution, and physical memory transfer traffic.

---

## 2. Experimental Matrix and Configuration Parameters

The experiment matrix evaluates the complete Cartesian product:

$$\text{2 Workloads (W2, W3)} \times \text{3 HBM Capacities (CAP1, CAP2, CAP3)} \times \text{4 Configurations (C1, C2, C3, C4)} = 24\text{ Runs}$$

### 2.1 Configuration Definitions

| Config ID | Prefetching Engine | Eviction Policy | Window Size ($W$) | Predictor Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **C1** | Disabled (`mode: none`) | Standard LRU | N/A | None |
| **C2** | Disabled (`mode: none`) | FA-LRU | 16 | None |
| **C3** | CSTP (`mode: cstp`) | Standard LRU | N/A | Horizon $H=1$, Budget $K=4$ |
| **C4** | CSTP (`mode: cstp`) | FA-LRU | 16 | Horizon $H=1$, Budget $K=4$ |

### 2.2 HBM Capacity Definitions

Exact byte budgets were enforced through simulation configuration files and the `DEBUG_CACHE_CAPACITY` environment override:

| Capacity ID | Nominal Label | Exact Capacity (Bytes) | Exact GiB ($2^{30}$) | Expert Slots ($\approx 88.08\text{ MB/expert}$) |
| :--- | :--- | :--- | :--- | :--- |
| **CAP1** | 10 GB | $10,000,000,000$ | $9.3132\text{ GiB}$ | 113 slots |
| **CAP2** | 35 GiB | $37,580,963,840$ | $35.0000\text{ GiB}$ | 426 slots |
| **CAP3** | 80 GB | $80,000,000,000$ | $74.5058\text{ GiB}$ | 908 slots |

### 2.3 Workload Specifications

- **Workload W2 (Medium Batch):** Batch size $B=4$, sequence generation tokens $T=16$, layers $L=40$, Top-$K=8$. Replaying prompts 60, 61, 62, 63 from `e1-sage-a1.5.parquet` (20,480 trace records).
- **Workload W3 (Held-Out Large Batch):** Batch size $B=14$, sequence generation tokens $T=16$, layers $L=40$, Top-$K=8$. Replaying prompts 60–73 from `e1-sage-a1.5.parquet` (71,680 trace records).

---

## 3. Manifest and Execution Log Audit

The campaign manifest was generated at `mohman/point_4/phase_3_focused_24/manifests/campaign_manifest.csv` prior to launching execution. Every run produced an individual execution log under `mohman/point_4/phase_3_focused_24/logs/`.

### Table 1: Comprehensive 24-Run Execution Audit

| Run Index | Run ID | Group | Workload | Capacity | Config | Exit Code | Host Time (s) | Simulated GPU Time (ms) | Tokens / s | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `w2_cap1_c1` | G1 | W2 | 10 GB | C1 | 0 | 12.15 | 11,716.30 | 5.46 | PASS |
| 2 | `w2_cap1_c2` | G1 | W2 | 10 GB | C2 | 0 | 13.56 | 11,716.30 | 5.46 | PASS |
| 3 | `w2_cap1_c3` | G1 | W2 | 10 GB | C3 | 0 | 12.61 | 11,706.75 | 5.47 | PASS |
| 4 | `w2_cap1_c4` | G1 | W2 | 10 GB | C4 | 0 | 12.15 | 11,706.75 | 5.47 | PASS |
| 5 | `w2_cap2_c1` | G2 | W2 | 35 GiB | C1 | 0 | 13.53 | 10,100.27 | 6.34 | PASS |
| 6 | `w2_cap2_c2` | G2 | W2 | 35 GiB | C2 | 0 | 12.65 | 10,100.27 | 6.34 | PASS |
| 7 | `w2_cap2_c3` | G2 | W2 | 35 GiB | C3 | 0 | 12.82 | 10,629.59 | 6.02 | PASS |
| 8 | `w2_cap2_c4` | G2 | W2 | 35 GiB | C4 | 0 | 12.44 | 10,629.59 | 6.02 | PASS |
| 9 | `w2_cap3_c1` | G3 | W2 | 80 GB | C1 | 0 | 11.96 | 6,977.38 | 9.17 | PASS |
| 10 | `w2_cap3_c2` | G3 | W2 | 80 GB | C2 | 0 | 11.61 | 6,977.38 | 9.17 | PASS |
| 11 | `w2_cap3_c3` | G3 | W2 | 80 GB | C3 | 0 | 12.22 | 6,989.00 | 9.16 | PASS |
| 12 | `w2_cap3_c4` | G3 | W2 | 80 GB | C4 | 0 | 12.09 | 6,989.00 | 9.16 | PASS |
| 13 | `w3_cap1_c1` | G4 | W3 | 10 GB | C1 | 0 | 13.87 | 18,565.28 | 12.07 | PASS |
| 14 | `w3_cap1_c2` | G4 | W3 | 10 GB | C2 | 0 | 13.07 | 18,565.28 | 12.07 | PASS |
| 15 | `w3_cap1_c3` | G4 | W3 | 10 GB | C3 | 0 | 14.18 | 18,550.24 | 12.08 | PASS |
| 16 | `w3_cap1_c4` | G4 | W3 | 10 GB | C4 | 0 | 15.08 | 18,550.24 | 12.08 | PASS |
| 17 | `w3_cap2_c1` | G5 | W3 | 35 GiB | C1 | 0 | 14.03 | 18,565.28 | 12.07 | PASS |
| 18 | `w3_cap2_c2` | G5 | W3 | 35 GiB | C2 | 0 | 13.82 | 18,565.28 | 12.07 | PASS |
| 19 | `w3_cap2_c3` | G5 | W3 | 35 GiB | C3 | 0 | 15.51 | 18,531.51 | 12.09 | PASS |
| 20 | `w3_cap2_c4` | G5 | W3 | 35 GiB | C4 | 0 | 13.90 | 18,531.51 | 12.09 | PASS |
| 21 | `w3_cap3_c1` | G6 | W3 | 80 GB | C1 | 0 | 12.62 | 14,227.64 | 15.74 | PASS |
| 22 | `w3_cap3_c2` | G6 | W3 | 80 GB | C2 | 0 | 12.68 | 14,227.64 | 15.74 | PASS |
| 23 | `w3_cap3_c3` | G6 | W3 | 80 GB | C3 | 0 | 14.94 | 14,349.06 | 15.61 | PASS |
| 24 | `w3_cap3_c4` | G6 | W3 | 80 GB | C4 | 0 | 14.58 | 14,349.06 | 15.61 | PASS |

---

## 4. Hardware Verification and Startup Parameter Certification

A forensic review of the startup lines across all 24 logs certifies:

1. **Exact Capacity Enforcement:**
   - CAP1 runs logged: `HBM Local Cache Budget: 10000000000 bytes (9.31 GB)`.
   - CAP2 runs logged: `HBM Local Cache Budget: 37580963840 bytes (35.00 GB)`.
   - CAP3 runs logged: `HBM Local Cache Budget: 80000000000 bytes (74.51 GB)`.
2. **Eviction Policy Enforcement:**
   - C1 and C3 runs logged: `Eviction Policy: lru`.
   - C2 and C4 runs logged: `Eviction Policy: fa_lru (fa_lru_window_size=16)`.
3. **Trace Ingestion Integrity:**
   - Workload W2 logged: `Sequences: 4, Replay Tokens: 16, Layers: 40, Top-K: 8, Total Records: 20480 (100% verified, 0 fallback to random)`.
   - Workload W3 logged: `Sequences: 14, Replay Tokens: 16, Layers: 40, Top-K: 8, Total Records: 71680 (100% verified, 0 fallback to random)`.
4. **Host vs. Simulated Runtime Separation:**
   - Host CPU execution wall-clock runtimes ranged between 11.61 s and 15.51 s per run.
   - Simulated GPU execution times ranged between 6,977.38 ms and 18,565.28 ms, strictly derived from GPU compute cycles and DDR5 memory stall latencies.

---

## 5. Artifact Preservation

All generated execution artifacts have been persisted and organized under:
- Manifests: `mohman/point_4/phase_3_focused_24/manifests/`
- YAML Configurations: `mohman/point_4/phase_3_focused_24/configs/`
- Standard Output Logs: `mohman/point_4/phase_3_focused_24/logs/`
- Raw Telemetry Traces: `mohman/point_4/phase_3_focused_24/raw_results/`
- Consolidated Datasets: `mohman/point_4/phase_3_focused_24/consolidated/`

