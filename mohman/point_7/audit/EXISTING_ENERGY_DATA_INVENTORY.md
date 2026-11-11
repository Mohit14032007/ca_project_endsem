# Point 7 — Phase B: Master Inventory of Existing Experimental Energy Data

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 7 — Energy Analysis  
**Author:** Antigravity Engineering  
**Date:** November 11, 2026  
**Status:** Complete Audit & Verification  

---

## 1. Inventory Summary

An exhaustive audit was conducted across all existing experimental directories (`mohman/point_3/`, `mohman/point_4/`, `mohman/point_5/`, `mohman/point_6/`).

### Key Inventory Findings:
1. **Point 6 Sensitivity Analysis Suite (23 Newly Executed Simulations):**
   - **Status:** **100% USABLE & COMPLETE**.
   - Every single one of the 23 runs contains a valid, populated 128-step decode CSV (`deepseekV3_*_decode.csv`) with all 14 energy columns non-zero and mathematically verified.
   - Corresponding `prefetch_summary.json` files provide exact byte-level DDR5 demand and speculative traffic.
2. **Point 4 Big Experiment Suite (8 Complete 128-Step Simulations):**
   - **Status:** **100% USABLE & COMPLETE**.
   - Provides the mandatory 4-configuration comparison ($A, B, C, D$) under high concurrency ($B=32$ and $B=14$):
     - `w4_c1_r1`: Baseline A (No prefetch, LRU)
     - `w4_c2_r1`: Prediction-only B (CSTP shadow mode, LRU)
     - `w4_c3_r1`: Prefetch C (CSTP prefetching, LRU)
     - `w4_c4_r1`: Complete D (CSTP prefetching, FA-LRU W=64)
3. **Data Integrity:** Zero missing columns, zero corrupted CSVs, and exact byte conservation across all 23 Point 6 runs and 8 Point 4 Big Exp runs.

---

## 2. Comprehensive Master Inventory Table

| Run ID | Campaign | Suite | B | P | T | H | K | Policy | Decode CSV Status | Step Rows | HBM Energy (J) | MAC Energy (J) | Total DDR5 Traffic (GB) | Speculative Traffic (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `anc_c1` | Point 6 | Anchor | 32 | 2048 | 128 | 0 | 0 | LRU | Complete | 128 | 1,042.87 | 71.25 | 48,207.72 | 0.00 |
| `anc_c3` | Point 6 | Anchor | 32 | 2048 | 128 | 1 | 4 | LRU | Complete | 128 | 1,044.16 | 71.25 | 48,741.13 | 10,949.01 |
| `anc_c4` | Point 6 | Anchor | 32 | 2048 | 128 | 1 | 4 | FA-LRU | Complete | 128 | 1,044.18 | 71.25 | 48,739.10 | 10,928.22 |
| `b4_c1` | Point 6 | Batch Size | 4 | 2048 | 128 | 0 | 0 | LRU | Complete | 128 | 318.03 | 8.91 | 11,744.90 | 0.00 |
| `b4_c3` | Point 6 | Batch Size | 4 | 2048 | 128 | 1 | 4 | LRU | Complete | 128 | 317.86 | 8.91 | 12,057.15 | 3,495.69 |
| `b4_c4` | Point 6 | Batch Size | 4 | 2048 | 128 | 1 | 4 | FA-LRU | Complete | 128 | 317.86 | 8.91 | 12,056.98 | 3,495.53 |
| `b14_c1` | Point 6 | Batch Size | 14 | 2048 | 128 | 0 | 0 | LRU | Complete | 128 | 654.11 | 31.17 | 28,958.82 | 0.00 |
| `b14_c3` | Point 6 | Batch Size | 14 | 2048 | 128 | 1 | 4 | LRU | Complete | 128 | 655.54 | 31.17 | 34,923.75 | 6,959.55 |
| `b14_c4` | Point 6 | Batch Size | 14 | 2048 | 128 | 1 | 4 | FA-LRU | Complete | 128 | 655.54 | 31.17 | 34,923.23 | 6,959.02 |
| `p128_c1` (`w4_c1`) | Point 4 | Context P | 32 | 128 | 128 | 0 | 0 | LRU | Complete | 128 | 1,031.53 | 49.58 | 48,207.72 | 0.00 |
| `p128_c2` (`w4_c2`) | Point 4 | Context P | 32 | 128 | 128 | 0 | 0 | LRU | Complete | 128 | 1,031.53 | 49.58 | 48,207.72 | 0.00 |
| `p128_c3` (`w4_c3`) | Point 4 | Context P | 32 | 128 | 128 | 1 | 4 | LRU | Complete | 128 | 1,032.75 | 49.58 | 48,741.13 | 10,949.01 |
| `p128_c4` (`w4_c4`) | Point 4 | Context P | 32 | 128 | 128 | 1 | 4 | FA-LRU | Complete | 128 | 1,032.76 | 49.58 | 48,739.10 | 10,928.22 |
| `p512_c1` | Point 6 | Context P | 32 | 512 | 128 | 0 | 0 | LRU | Complete | 128 | 1,033.86 | 53.91 | 48,207.72 | 0.00 |
| `p512_c3` | Point 6 | Context P | 32 | 512 | 128 | 1 | 4 | LRU | Complete | 128 | 1,035.10 | 53.91 | 48,741.13 | 10,949.01 |
| `p512_c4` | Point 6 | Context P | 32 | 512 | 128 | 1 | 4 | FA-LRU | Complete | 128 | 1,035.11 | 53.91 | 48,739.10 | 10,928.22 |
| `p4096_c1` | Point 6 | Context P | 32 | 4096 | 128 | 0 | 0 | LRU | Complete | 128 | 1,055.13 | 94.35 | 48,207.72 | 0.00 |
| `p4096_c3` | Point 6 | Context P | 32 | 4096 | 128 | 1 | 4 | LRU | Complete | 128 | 1,056.53 | 94.35 | 48,741.13 | 10,949.01 |
| `p4096_c4` | Point 6 | Context P | 32 | 4096 | 128 | 1 | 4 | FA-LRU | Complete | 128 | 1,056.54 | 94.35 | 48,739.10 | 10,928.22 |
| `h2_c3` | Point 6 | Lookahead | 32 | 2048 | 128 | 2 | 4 | LRU | Complete | 128 | 1,044.44 | 71.25 | 48,741.13 | 10,949.01 |
| `h2_c4` | Point 6 | Lookahead | 32 | 2048 | 128 | 2 | 4 | FA-LRU | Complete | 128 | 1,044.44 | 71.25 | 48,739.10 | 10,928.22 |
| `h4_c3` | Point 6 | Lookahead | 32 | 2048 | 128 | 4 | 4 | LRU | Complete | 128 | 1,044.80 | 71.25 | 48,741.13 | 10,949.01 |
| `h4_c4` | Point 6 | Lookahead | 32 | 2048 | 128 | 4 | 4 | FA-LRU | Complete | 128 | 1,044.80 | 71.25 | 48,739.10 | 10,928.22 |
| `k2_c3` | Point 6 | Budget | 32 | 2048 | 128 | 1 | 2 | LRU | Complete | 128 | 1,045.74 | 71.25 | 48,395.41 | 6,083.98 |
| `k2_c4` | Point 6 | Budget | 32 | 2048 | 128 | 1 | 2 | FA-LRU | Complete | 128 | 1,045.74 | 71.25 | 48,394.01 | 6,060.11 |
| `k8_c3` | Point 6 | Budget | 32 | 2048 | 128 | 1 | 8 | LRU | Complete | 128 | 1,045.68 | 71.25 | 48,776.19 | 18,037.10 |
| `k8_c4` | Point 6 | Budget | 32 | 2048 | 128 | 1 | 8 | FA-LRU | Complete | 128 | 1,045.67 | 71.25 | 48,778.12 | 18,019.84 |

---

## 3. Data Reuse and Execution Gap Assessment

1. **Four-Configuration Anchor ($A, B, C, D$):**
   - For $B=32, P=128, T=128$, all four configurations ($A, B, C, D$) are fully available from Point 4 Big Exp (`w4_c1_r1`, `w4_c2_r1`, `w4_c3_r1`, `w4_c4_r1`).
   - For $B=32, P=2048, T=128$, configurations $A$ (`anc_c1`), $C$ (`anc_c3`), and $D$ (`anc_c4`) are available. Configuration $B$ (CSTP shadow mode) exhibits identical GPU behavior to $A$ (as proven by `w4_c1` vs `w4_c2` and `w3_c1` vs `w3_c2`), because shadow prediction executes exclusively on host CPU.
2. **Sensitivity Sweeps:**
   - Batch size ($B \in \{4, 14, 32\}$): fully populated with valid decode CSVs.
   - Context length ($P \in \{128, 512, 2048, 4096\}$): fully populated with valid decode CSVs.
   - Lookahead ($H \in \{1, 2, 4\}$): fully populated with valid decode CSVs.
   - Budget ($K \in \{2, 4, 8\}$): fully populated with valid decode CSVs.
3. **Conclusion:** All required experimental configurations have valid, verified, and complete simulation results. Zero new simulations are required, eliminating wasteful redundant execution.

