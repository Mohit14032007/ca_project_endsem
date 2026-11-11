# Point 6 — Comprehensive Inventory of Existing Experiments

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Sensitivity Analysis Consolidation  
**Author:** Antigravity Engineering  
**Date:** November 10, 2026  
**Status:** Authoritative & Completed  

---

## 1. Executive Summary

This document inventories all 59 relevant historical simulation runs across Points 3, 4, and 5. Each run is mapped to its exact configuration parameters, architectural boundaries, and comparability status relative to Point 6's reference configuration.

### Reference Configuration for Point 6:
- **Model:** DeepSeek-V3 (40 MoE stages, 256 routed experts, Top-8 routing)
- **HBM Capacity:** 35 GiB ($37,580,963,840$ bytes)
- **DDR5 Bandwidth:** 64 GB/s
- **Predictor:** CSTP ($H=1, K=4, \alpha=0.5, \lambda=1.0$)
- **HBM Eviction Policy:** Standard LRU vs. FA-LRU ($W=64$)
- **Generation Horizon:** $T = 128$ tokens

---

## 2. Complete Historical Run Inventory Table

| Source Point | Experiment ID | $B$ | $P$ | $T$ | $L$ | Top-K | $H$ | Budget $K$ | Predictor | Policy | $W$ | HBM Capacity | Trace Source | Comparability |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :--- | :--- | :--- |
| Point 4 Big Experiment | `w3_c1_r1` | 14 | 128 | 128 | 40 | 8 | 0 | 0 | None | LRU | 64 | 37580963840 B (35.0 GiB) | MoE-Anywhere (Prompts 60-73 (Test)) | **Directly comparable (P=128 reference)** |
| Point 4 Big Experiment | `w3_c2_r1` | 14 | 128 | 128 | 40 | 8 | 0 | 0 | None | FA-LRU | 64 | 37580963840 B (35.0 GiB) | MoE-Anywhere (Prompts 60-73 (Test)) | **Directly comparable (P=128 reference)** |
| Point 4 Big Experiment | `w3_c3_r1` | 14 | 128 | 128 | 40 | 8 | 1 | 4 | CSTP | LRU | 64 | 37580963840 B (35.0 GiB) | MoE-Anywhere (Prompts 60-73 (Test)) | **Directly comparable (P=128 reference)** |
| Point 4 Big Experiment | `w3_c4_r1` | 14 | 128 | 128 | 40 | 8 | 1 | 4 | CSTP | FA-LRU | 64 | 37580963840 B (35.0 GiB) | MoE-Anywhere (Prompts 60-73 (Test)) | **Directly comparable (P=128 reference)** |
| Point 4 Big Experiment | `w4_c1_r1` | 32 | 128 | 128 | 40 | 8 | 0 | 0 | None | LRU | 64 | 37580963840 B (35.0 GiB) | MoE-Anywhere (Prompts 60-73, 52-59, 0-9) | **Directly comparable (P=128 reference)** |
| Point 4 Big Experiment | `w4_c2_r1` | 32 | 128 | 128 | 40 | 8 | 0 | 0 | None | FA-LRU | 64 | 37580963840 B (35.0 GiB) | MoE-Anywhere (Prompts 60-73, 52-59, 0-9) | **Directly comparable (P=128 reference)** |
| Point 4 Big Experiment | `w4_c3_r1` | 32 | 128 | 128 | 40 | 8 | 1 | 4 | CSTP | LRU | 64 | 37580963840 B (35.0 GiB) | MoE-Anywhere (Prompts 60-73, 52-59, 0-9) | **Directly comparable (P=128 reference)** |
| Point 4 Big Experiment | `w4_c4_r1` | 32 | 128 | 128 | 40 | 8 | 1 | 4 | CSTP | FA-LRU | 64 | 37580963840 B (35.0 GiB) | MoE-Anywhere (Prompts 60-73, 52-59, 0-9) | **Directly comparable (P=128 reference)** |
| Point 3 Large Workload | `w1_baseline` | 14 | 2048 | 4 | 40 | 8 | 0 | 0 | None | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w1_stmoe_h2` | 14 | 2048 | 4 | 40 | 8 | 2 | 4 | ST-MoE | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w1_cstp_h2` | 14 | 2048 | 4 | 40 | 8 | 2 | 4 | CSTP | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w1_stmoe_h1` | 14 | 2048 | 4 | 40 | 8 | 1 | 4 | ST-MoE | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w1_cstp_h1` | 14 | 2048 | 4 | 40 | 8 | 1 | 4 | CSTP | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w2_baseline` | 14 | 2048 | 16 | 40 | 8 | 0 | 0 | None | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w2_stmoe_h2` | 14 | 2048 | 16 | 40 | 8 | 2 | 4 | ST-MoE | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w2_cstp_h2` | 14 | 2048 | 16 | 40 | 8 | 2 | 4 | CSTP | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w2_stmoe_h1` | 14 | 2048 | 16 | 40 | 8 | 1 | 4 | ST-MoE | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w2_cstp_h1` | 14 | 2048 | 16 | 40 | 8 | 1 | 4 | CSTP | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w3_baseline` | 32 | 2048 | 16 | 40 | 8 | 0 | 0 | None | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w3_stmoe_h2` | 32 | 2048 | 16 | 40 | 8 | 2 | 4 | ST-MoE | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w3_cstp_h2` | 32 | 2048 | 16 | 40 | 8 | 2 | 4 | CSTP | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w4_baseline` | 48 | 2048 | 16 | 40 | 8 | 0 | 0 | None | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w4_stmoe_h2` | 48 | 2048 | 16 | 40 | 8 | 2 | 4 | ST-MoE | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w4_cstp_h2` | 48 | 2048 | 16 | 40 | 8 | 2 | 4 | CSTP | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w5_baseline` | 70 | 2048 | 16 | 40 | 8 | 0 | 0 | None | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w5_stmoe_h2` | 70 | 2048 | 16 | 40 | 8 | 2 | 4 | ST-MoE | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w5_cstp_h2` | 70 | 2048 | 16 | 40 | 8 | 2 | 4 | CSTP | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w5_stmoe_h1` | 70 | 2048 | 16 | 40 | 8 | 1 | 4 | ST-MoE | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w5_cstp_h1` | 70 | 2048 | 16 | 40 | 8 | 1 | 4 | CSTP | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w6_baseline` | 70 | 2048 | 32 | 40 | 8 | 0 | 0 | None | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w6_stmoe_h2` | 70 | 2048 | 32 | 40 | 8 | 2 | 4 | ST-MoE | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w6_cstp_h2` | 70 | 2048 | 32 | 40 | 8 | 2 | 4 | CSTP | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w7_baseline` | 70 | 2048 | 64 | 40 | 8 | 0 | 0 | None | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w7_stmoe_h2` | 70 | 2048 | 64 | 40 | 8 | 2 | 4 | ST-MoE | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 3 Large Workload | `w7_cstp_h2` | 70 | 2048 | 64 | 40 | 8 | 2 | 4 | CSTP | LRU | 0 | 80000000000 B (74.5 GiB) | Synthetic / MoE-Anywhere | **Partially comparable (T=16, HBM=80GB)** |
| Point 4 Focused 24 | `w2_cap1_c1` | 2 | 128 | 20 | 40 | 8 | 0 | 0 | None | LRU | 16 | 9.3 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w2_cap1_c2` | 2 | 128 | 20 | 40 | 8 | 0 | 0 | None | FA-LRU | 16 | 9.3 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w2_cap1_c3` | 2 | 128 | 20 | 40 | 8 | 1 | 4 | CSTP | LRU | 16 | 9.3 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w2_cap1_c4` | 2 | 128 | 20 | 40 | 8 | 1 | 4 | CSTP | FA-LRU | 16 | 9.3 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w2_cap2_c1` | 2 | 128 | 20 | 40 | 8 | 0 | 0 | None | LRU | 16 | 35.0 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w2_cap2_c2` | 2 | 128 | 20 | 40 | 8 | 0 | 0 | None | FA-LRU | 16 | 35.0 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w2_cap2_c3` | 2 | 128 | 20 | 40 | 8 | 1 | 4 | CSTP | LRU | 16 | 35.0 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w2_cap2_c4` | 2 | 128 | 20 | 40 | 8 | 1 | 4 | CSTP | FA-LRU | 16 | 35.0 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w2_cap3_c1` | 2 | 128 | 20 | 40 | 8 | 0 | 0 | None | LRU | 16 | 74.5 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w2_cap3_c2` | 2 | 128 | 20 | 40 | 8 | 0 | 0 | None | FA-LRU | 16 | 74.5 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w2_cap3_c3` | 2 | 128 | 20 | 40 | 8 | 1 | 4 | CSTP | LRU | 16 | 74.5 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w2_cap3_c4` | 2 | 128 | 20 | 40 | 8 | 1 | 4 | CSTP | FA-LRU | 16 | 74.5 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w3_cap1_c1` | 4 | 128 | 20 | 40 | 8 | 0 | 0 | None | LRU | 16 | 9.3 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w3_cap1_c2` | 4 | 128 | 20 | 40 | 8 | 0 | 0 | None | FA-LRU | 16 | 9.3 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w3_cap1_c3` | 4 | 128 | 20 | 40 | 8 | 1 | 4 | CSTP | LRU | 16 | 9.3 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w3_cap1_c4` | 4 | 128 | 20 | 40 | 8 | 1 | 4 | CSTP | FA-LRU | 16 | 9.3 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w3_cap2_c1` | 4 | 128 | 20 | 40 | 8 | 0 | 0 | None | LRU | 16 | 35.0 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w3_cap2_c2` | 4 | 128 | 20 | 40 | 8 | 0 | 0 | None | FA-LRU | 16 | 35.0 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w3_cap2_c3` | 4 | 128 | 20 | 40 | 8 | 1 | 4 | CSTP | LRU | 16 | 35.0 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w3_cap2_c4` | 4 | 128 | 20 | 40 | 8 | 1 | 4 | CSTP | FA-LRU | 16 | 35.0 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w3_cap3_c1` | 4 | 128 | 20 | 40 | 8 | 0 | 0 | None | LRU | 16 | 74.5 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w3_cap3_c2` | 4 | 128 | 20 | 40 | 8 | 0 | 0 | None | FA-LRU | 16 | 74.5 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w3_cap3_c3` | 4 | 128 | 20 | 40 | 8 | 1 | 4 | CSTP | LRU | 16 | 74.5 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |
| Point 4 Focused 24 | `w3_cap3_c4` | 4 | 128 | 20 | 40 | 8 | 1 | 4 | CSTP | FA-LRU | 16 | 74.5 GiB | MoE-Anywhere Prompts 0-3 | **Partially comparable (T=20, W=16)** |

---

## 3. Reusability Analysis for Point 6 Campaigns

### 3.1 Directly Reusable Point 4 Big Experiment Runs
The 16 runs from Point 4 Big Experiment (`w3_c1` to `w3_c4` and `w4_c1` to `w4_c4`) match Point 6's reference configuration at $P=128, T=128$, HBM=35GiB:
- `w4_c1` (Baseline LRU), `w4_c3` (CSTP+LRU), and `w4_c4` (CSTP+FA-LRU W=64) provide the authentic $P=128, B=32$ baseline for Experiment 2 (Context-Length Sensitivity).
- `w3_c1` to `w3_c4` provide the validated $B=14$ reference point at $P=128$.

### 3.2 Identified Experimental Gaps (Newly Required Simulations)
To complete Point 6 without gaps, the following 23 controlled simulations must be executed:
1. **Experiment 1 (Batch-Size Sensitivity at Anchor $P=2048$):** $B \in \{4, 14, 32\}$ for Systems A, C, D (8 new runs, sharing the anchor $B=32$ run).
2. **Experiment 2 (Context-Length Sensitivity at $B=32$):** $P \in \{512, 2048, 4096\}$ for Systems A, C, D (6 new runs, reusing $P=128$ from Big Exp).
3. **Experiment 3 (Lookahead Sensitivity at $B=32, P=2048$):** $H \in \{2, 4\}$ for Systems C and D (4 new runs, sharing $H=1$ anchor).
4. **Experiment 4 (Prefetch Budget Sensitivity at $B=32, P=2048$):** $K \in \{2, 8\}$ for Systems C and D (4 new runs, sharing $K=4$ anchor).
5. **Anchor Run:** $B=32, P=2048, T=128, H=1, K=4$ for Systems A, C, D (3 runs, shared across all 4 experiments).
