# Phase A.3: Big Experiment Readiness Report

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Campaign:** Big Experiment: LRU vs. FA-LRU ($W=64$) Under Large MoE Workloads  
**Status:** **GO — CERTIFIED READY FOR SIMULATION**  
**Date:** 2026-11-10  

---

## 1. Readiness Audit Synthesis

Following the completion of the Trace Availability Audit ([`TRACE_AVAILABILITY_AUDIT.md`](file:///home/mohman/ca_project/mohman/point_4/big_exp/audit/TRACE_AVAILABILITY_AUDIT.md)) and the Configuration Audit ([`CONFIGURATION_AUDIT.md`](file:///home/mohman/ca_project/mohman/point_4/big_exp/audit/CONFIGURATION_AUDIT.md)), this report delivers the formal sign-off certifying full readiness to execute the 16-run experimental campaign.

### Checklist of Readiness Criteria

| Verification Item | Specification | Status | Evidence |
| :--- | :--- | :---: | :--- |
| **Trace Availability** | Authentic $T=128$ tokens for W3 ($B=14$) and W4 ($B=32$) | **PASS** | 257 contiguous steps available in `e1-sage-a1.5.parquet`; 1,884,160 records exported. |
| **Zero Synthetic Fallback** | 100% verified trace records, no random fallback | **PASS** | Validated via `export_big_workloads.py`; zero synthetic tokens generated. |
| **HBM Capacity** | Exactly $35.000\text{ GiB}$ ($37,580,963,840\text{ B}$) | **PASS** | Enforced in configuration YAMLs and `DEBUG_CACHE_CAPACITY` environment variable. |
| **Candidate Window** | Exactly $W=64$ candidates | **PASS** | Enforced via `fa_lru_window_size: 64` and `FA_LRU_WINDOW_SIZE=64`. |
| **Predictor Setup** | CSTP $H=1, K=4$ for C3/C4; Disabled for C1/C2 | **PASS** | Predictor configurations verified; fair paired comparison controls locked. |
| **Binary Build** | `run_deepseekv3_mixtral_style` compiled cleanly | **PASS** | Built target with zero compilation warnings or linking errors. |
| **Unit Test Suite** | All 63 regression and policy unit tests pass | **PASS** | 14/14 `test_fa_lru`, 1/1 `test_policy_activation`, 6/6 `test_lru`, etc. |
| **Zero Ternary Invariant** | Strictly zero ternary operators (`? :`, `ast.IfExp`) | **PASS** | AST check verified 0 ternaries across all newly authored scripts. |

---

## 2. Experimental Run Matrix (16 Runs)

The campaign evaluates 4 configurations across 2 workloads, repeated twice to verify deterministic simulator reproducibility:

$$\text{2 Workloads} \times \text{4 Configurations} \times \text{2 Repetitions} = 16\text{ Runs}$$

### Table 1: Complete 16-Run Execution Matrix

| Run ID | Rep | Workload | Batch | Horizon ($T$) | Capacity | Prefetch | Eviction Policy | Window ($W$) | Deterministic Seed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `w3_c1_r1` | 1 | W3 | 14 | 128 | 35 GiB | Disabled | Standard LRU | N/A | 42 |
| `w3_c1_r2` | 2 | W3 | 14 | 128 | 35 GiB | Disabled | Standard LRU | N/A | 42 |
| `w3_c2_r1` | 1 | W3 | 14 | 128 | 35 GiB | Disabled | FA-LRU | 64 | 42 |
| `w3_c2_r2` | 2 | W3 | 14 | 128 | 35 GiB | Disabled | FA-LRU | 64 | 42 |
| `w3_c3_r1` | 1 | W3 | 14 | 128 | 35 GiB | CSTP (H=1, K=4) | Standard LRU | N/A | 42 |
| `w3_c3_r2` | 2 | W3 | 14 | 128 | 35 GiB | CSTP (H=1, K=4) | Standard LRU | N/A | 42 |
| `w3_c4_r1` | 1 | W3 | 14 | 128 | 35 GiB | CSTP (H=1, K=4) | FA-LRU | 64 | 42 |
| `w3_c4_r2` | 2 | W3 | 14 | 128 | 35 GiB | CSTP (H=1, K=4) | FA-LRU | 64 | 42 |
| `w4_c1_r1` | 1 | W4 | 32 | 128 | 35 GiB | Disabled | Standard LRU | N/A | 42 |
| `w4_c1_r2` | 2 | W4 | 32 | 128 | 35 GiB | Disabled | Standard LRU | N/A | 42 |
| `w4_c2_r1` | 1 | W4 | 32 | 128 | 35 GiB | Disabled | FA-LRU | 64 | 42 |
| `w4_c2_r2` | 2 | W4 | 32 | 128 | 35 GiB | Disabled | FA-LRU | 64 | 42 |
| `w4_c3_r1` | 1 | W4 | 32 | 128 | 35 GiB | CSTP (H=1, K=4) | Standard LRU | N/A | 42 |
| `w4_c3_r2` | 2 | W4 | 32 | 128 | 35 GiB | CSTP (H=1, K=4) | Standard LRU | N/A | 42 |
| `w4_c4_r1` | 1 | W4 | 32 | 128 | 35 GiB | CSTP (H=1, K=4) | FA-LRU | 64 | 42 |
| `w4_c4_r2` | 2 | W4 | 32 | 128 | 35 GiB | CSTP (H=1, K=4) | FA-LRU | 64 | 42 |

---

## 3. Formal Sign-Off

All prerequisites for Phase B execution have been satisfied. Authorization is granted to launch the 16-run Big Experiment campaign.

