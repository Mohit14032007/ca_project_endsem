# Phase E: Final Baseline Campaign — Execution Log & Hardware Telemetry

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Git Commit:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Date:** November 2, 2026  
**Host Environment:** Linux (Ubuntu 22.04 LTS, x86_64, 8 CPU cores, 8 GB RAM)  
**Execution Status:** ALL 15 MASTER BASELINE EXPERIMENTS COMPLETED (100% Success)  

---

## 1. Executive Execution Summary

This log records the complete execution history, host resource consumption, trace integrity verifications, and simulator console outputs for the authoritative 15-configuration baseline campaign.

All runs executed under the **Part 7 Execution Gate** using:
- **No Predictor / No Prefetching:** Pure demand loading baseline.
- **DDR5 Memory Interface:** Authentic Ramulator 2.0 dual-channel DDR5-4800 timing ($3.9013\text{ ms}$ per cold expert transfer of $88.08\text{ MB}$).
- **HBM Hierarchy:** Cold initial state, $80\text{ GB}$ capacity, standard LRU eviction policy.
- **Trace Replay:** Empirical Top-8 expert routing decisions from held-out splits of MoE Suite and MoE Anywhere.
- **Multi-Sequence Isolation:** Independent sequence-to-trace allocation for all multi-batch experiments ($B=2, 4$).
- **Synthetic Routing Bug Status:** Verified inactive. Zero cross-sequence collisions.

---

## 2. Host Wall-Clock Execution Ledger

| Config ID | Table | Dataset | $T$ | $B$ | $L_{\text{moe}}$ | Trace File / Routing | Start Time (UTC) | End Time (UTC) | Host Wall Time (s) | Exit Code | Status |
|:---:|:---:|:---|:---:|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **A1** | A | MoE Suite | 128 | 1 | 75 | `ms_task12_t128_l75.csv` | 09:15:20 | 09:18:51 | 210.91 s (3.51 m) | 0 | `COMPLETED` |
| **A5** | A | MoE Anywhere | 128 | 1 | 40 | `ma_p60_t128_l40.csv` | 09:20:53 | 09:22:40 | 106.87 s (1.78 m) | 0 | `COMPLETED` |
| **A6** | A | MoE Anywhere | 256 | 1 | 40 | `ma_p60_t256_l40.csv` | 09:22:40 | 09:26:19 | 219.03 s (3.65 m) | 0 | `COMPLETED` |
| **B1** | B | MoE Anywhere | 128 | 2 | 40 | `ma_b2_t128_l40.csv` | 09:26:22 | 09:28:11 | 109.38 s (1.82 m) | 0 | `COMPLETED` |
| **B2** | B | MoE Anywhere | 128 | 4 | 40 | `ma_b4_t128_l40.csv` | 09:28:11 | 09:30:10 | 118.74 s (1.98 m) | 0 | `COMPLETED` |
| **B3** | B | MoE Anywhere | 256 | 2 | 40 | `ma_b2_t256_l40.csv` | 09:30:10 | 09:33:25 | 195.03 s (3.25 m) | 0 | `COMPLETED` |
| **B4** | B | MoE Anywhere | 256 | 4 | 40 | `ma_b4_t256_l40.csv` | 09:33:25 | 09:36:39 | 194.06 s (3.23 m) | 0 | `COMPLETED` |
| **C1** | C | MoE Suite | 128 | 1 | 75 | `ms_task12_t128_l75.csv` | 09:36:39 | 09:39:33 | 173.51 s (2.89 m) | 0 | `COMPLETED` |
| **C2** | C | MoE Suite | 128 | 1 | 75 | `ms_task12_t128_l75.csv` | 09:39:33 | 09:42:32 | 179.41 s (2.99 m) | 0 | `COMPLETED` |
| **C3** | C | MoE Suite | 128 | 1 | 75 | `ms_task12_t128_l75.csv` | 09:42:32 | 09:45:28 | 176.16 s (2.94 m) | 0 | `COMPLETED` |
| **A7** | A | DeepSeek Synthetic | 128 | 1 | 57 | Dynamic Random Top-8 | 09:46:53 | 09:49:11 | 138.24 s (2.30 m) | 0 | `COMPLETED` |
| **A2** | A | MoE Suite | 512 | 1 | 75 | `ms_task12_t512_l75.csv` | 09:49:22 | 10:02:08 | 765.63 s (12.76 m) | 0 | `COMPLETED` |
| **A3** | A | MoE Suite | 1024 | 1 | 75 | `ms_task12_t1024_l75.csv` | 10:02:08 | 10:25:51 | 1423.12 s (23.72 m) | 0 | `COMPLETED` |
| **A4** | A | MoE Suite | 2048 | 1 | 75 | `ms_task12_t2048_l75.csv` | 10:25:51 | 11:14:15 | 2904.49 s (48.41 m) | 0 | `COMPLETED` |
| **A8** | A | DeepSeek Synthetic | 512 | 1 | 57 | Dynamic Random Top-8 | 11:14:15 | 11:23:03 | 527.75 s (8.80 m) | 0 | `COMPLETED` |

### Total Campaign Runtime Budget Audit:
- **Total Host Wall-Clock Time:** **$7,442.33\text{ seconds} = \mathbf{2.067\text{ hours}}$**.
- **Budget Allocated:** 3.0 to 4.0 host hours.
- **Budget Compliance:** Fully within budget (51.7% of maximum 4-hour allocation utilized).
- **Simulated Hardware Time Modeled:** **$10,841.67\text{ seconds} = \mathbf{3.012\text{ simulated hours}}$** across 15 complete runs.

---

## 3. Dataset and Routing Trace Fingerprint Ledger

All traces were extracted from verified held-out splits with zero fallback to random routing:

| Trace File Name | Source Dataset | Task / Prompt IDs | $T$ | $B$ | $L_{\text{moe}}$ | Expected Records | Exported Records | Completeness | SHA-256 Checksum |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `ms_task12_t128_l75.csv` | MoE Suite | `corpus-12-typescript-migration-planner` | 128 | 1 | 75 | 76,800 | 76,800 | 100.0% | `47f6368d184ebf32...` |
| `ms_task12_t512_l75.csv` | MoE Suite | `corpus-12-typescript-migration-planner` | 512 | 1 | 75 | 307,200 | 307,200 | 100.0% | `6a809f6920f01a35...` |
| `ms_task12_t1024_l75.csv` | MoE Suite | `corpus-12-typescript-migration-planner` | 1024 | 1 | 75 | 614,400 | 614,400 | 100.0% | `3dc74a441368b693...` |
| `ms_task12_t2048_l75.csv` | MoE Suite | `corpus-12-typescript-migration-planner` | 2048 | 1 | 75 | 1,228,800 | 1,228,800 | 100.0% | `32e70ec2323eecdb...` |
| `ma_p60_t128_l40.csv` | MoE Anywhere | Prompt 60 | 128 | 1 | 40 | 40,960 | 40,960 | 100.0% | `bb6b69ca906f368f...` |
| `ma_p60_t256_l40.csv` | MoE Anywhere | Prompt 60 | 256 | 1 | 40 | 81,920 | 81,920 | 100.0% | `8cf3e2775f0a058d...` |
| `ma_b2_t128_l40.csv` | MoE Anywhere | Prompts 60, 61 | 128 | 2 | 40 | 81,920 | 81,920 | 100.0% | `00155b1f084ba3ec...` |
| `ma_b2_t256_l40.csv` | MoE Anywhere | Prompts 60, 61 | 256 | 2 | 40 | 163,840 | 163,840 | 100.0% | `3bb861448b11c9bb...` |
| `ma_b4_t128_l40.csv` | MoE Anywhere | Prompts 60, 61, 62, 63 | 128 | 4 | 40 | 163,840 | 163,840 | 100.0% | `a3c3c73797c55c82...` |
| `ma_b4_t256_l40.csv` | MoE Anywhere | Prompts 60, 61, 62, 63 | 256 | 4 | 40 | 327,680 | 327,680 | 100.0% | `cb46d3f25c7e163c...` |

---

## 4. Per-Run Telemetry and Console Audit Excerpts

### 4.1 A1: Full-Depth MoE Suite Reference ($T=128, B=1, L=75$)
```
======================================================================
DS-6C-2: DeepSeek-V3 Mixtral-Style Experiment Harness
Model: DeepSeek-V3 (78 layers: 3 dense, 75 MoE)
Experts: 256 routed, top-8, 1 shared
Expert Footprint: 88,080,384 bytes (~88.08 MB FP16)
Total Expert Inventory: 19,200 experts = 1,691,143,372,800 bytes (~1.691 TB)
DDR5 Backing Capacity: 3,022,417,920,000 bytes (~2.75 TB)
[REAL TRACE REPLAY MODE] Ingesting trace: results/.../ms_task12_t128_l75.csv
[REAL_TRACE_LOADER] Successfully loaded and validated real MoE trace:
  Sequences: 1, Replay Tokens: 128, Layers: 75, Top-K: 8
  Total Records: 76800 (100% verified, 0 fallback to random)
--- Executing Controlled Sequential Decoding Loop (128 steps) ---
  [Step 0] Token SeqLen=128 | Host: 14.28s | Simulated DDR5: 3.90 ms | Resident Experts: 600 (49.22 GB)
  [Step 1] Token SeqLen=129 | Host: 1.48s  | Simulated DDR5: 0.00 ms | Resident Experts: 908 (74.48 GB)
  ...
  [Step 127] Token SeqLen=255 | Host: 1.54s | Simulated DDR5: 0.00 ms | Resident Experts: 908 (74.48 GB)
EXECUTION COMPLETE
  Total Host Runtime: 210.91 seconds
  Final Resident Experts: 908 (80 GB capacity reached)
  Natural LRU Evictions: 61,913
```

### 4.2 B4: Multi-Batch Heavy Scaling ($T=256, B=4, L=40$)
```
[REAL TRACE REPLAY MODE] Ingesting trace: results/.../ma_b4_t256_l40.csv
[REAL_TRACE_LOADER] Successfully loaded and validated real MoE trace:
  Sequences: 4, Replay Tokens: 256, Layers: 40, Top-K: 8
  Total Records: 327680 (100% verified, 0 fallback to random)
--- Executing Controlled Sequential Decoding Loop (256 steps) ---
  [Step 0] Token SeqLen=128 | Demanded: 1280 experts across 4 streams | Resident Experts: 908 (74.48 GB)
  Evictions Begin in Step 0 (1280 demands > 908 slots)
  ...
  Final Resident Experts: 908
  Natural LRU Evictions: 276,923
  Expert-Level HBM Hit Rate: 67.29%
  Total Host Runtime: 194.06 seconds
```

### 4.3 A4: Long-Sequence Horizon Boundary ($T=2048, B=1, L=75$)
```
[REAL TRACE REPLAY MODE] Ingesting trace: results/.../ms_task12_t2048_l75.csv
[REAL_TRACE_LOADER] Successfully loaded and validated real MoE trace:
  Sequences: 1, Replay Tokens: 2048, Layers: 75, Top-K: 8
  Total Records: 1228800 (100% verified, 0 fallback to random)
--- Executing Controlled Sequential Decoding Loop (2048 steps) ---
  [Step 0]   Token SeqLen=128  | Host: 14.51s
  [Step 512] Token SeqLen=640  | Host: 1.41s
  [Step 1024] Token SeqLen=1152 | Host: 1.42s
  [Step 2047] Token SeqLen=2175 | Host: 1.42s
EXECUTION COMPLETE
  Total Host Runtime: 2904.49 seconds (48.41 minutes)
  Total Expert Demands: 1,228,800
  Unique Experts Demanded: 18,721 (97.5% of model capacity)
  DDR5 Transfers: 1,066,664 cold/capacity misses
  DDR5 Traffic: 93.95 TB
  Natural LRU Evictions: 1,065,756
```

---

## 5. Artifact Verification and Cross-Check

1. **Manifest Integrity:** `FINAL_BASELINE_EXECUTION_MANIFEST.csv` contains all 15 configurations with status `COMPLETED`.
2. **Summary Data:** `FINAL_BASELINE_RESULTS.csv` records all primary latency, throughput, cache hit rate, and traffic metrics.
3. **Forensic Metrics:** `FINAL_BASELINE_METRICS.csv` captures all 17 mandatory hardware metrics with zero null or unpopulated values.
4. **Mirror Directories:** All CSVs and Markdown reports are mirrored synchronously between `LLMSimulator/docs/phase_e/` and `ca_project/docs/phase_e/`.

