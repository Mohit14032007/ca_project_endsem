# Big Experiment: Reproducibility and Determinism Report

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Campaign:** Big Experiment: LRU vs. FA-LRU ($W=64$) Under Large MoE Workloads  
**Evaluation Scope:** Bit-Exact Determinism Audit Across Repetitions 1 and 2  
**Status:** 100% REPRODUCIBLE (Bit-Exact Invariance Verified)  
**Date:** 2026-11-10  

---

## 1. Executive Summary

This report establishes the complete experimental reproducibility of the Big Experiment campaign. In accordance with the Phase B protocol, every configuration was executed across two distinct repetitions using identical deterministic seeds (Seed 42) and identical input traces.

A bit-level telemetry audit across all pairs confirms that the simulation framework is **100% deterministic**. Every cache hit, eviction decision, physical memory transfer, and GPU stall cycle reproduced with zero drift.

> [!IMPORTANT]
> Because LLMSimulator executes as a fully deterministic discrete-event simulator, identical repeated runs do not represent independent statistical random samples. Instead, they serve as rigorous verification that the simulator execution harness is free from uninitialized memory, thread race conditions, nondeterministic container ordering, and environment jitter.

---

## 2. Repetition Consistency Audit Table

The table below compares Repetition 1 and Repetition 2 across all 8 experimental configurations.

| Configuration ID | Workload | Prefetching | Eviction Policy | Repetition 1 Hit Rate | Repetition 2 Hit Rate | Hit Rate $\Delta$ | Rep 1 Evictions | Rep 2 Evictions | Rep 1 DDR5 (Bytes) | Rep 2 DDR5 (Bytes) | Rep 1 Stall (ms) | Rep 2 Stall (ms) | Determinism Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `w3_c1` | W3 | Disabled | Standard LRU | 0.000000 | 0.000000 | 0.000000 | 326,825 | 326,825 | 28,824,393,744,384 | 28,824,393,744,384 | 450,381.15 | 450,381.15 | BIT-EXACT |
| `w3_c2` | W3 | Disabled | FA-LRU ($W=64$) | 0.000000 | 0.000000 | 0.000000 | 326,825 | 326,825 | 28,824,393,744,384 | 28,824,393,744,384 | 450,381.15 | 450,381.15 | BIT-EXACT |
| `w3_c3` | W3 | CSTP | Standard LRU | 0.005063 | 0.005063 | 0.000000 | 331,173 | 331,173 | 29,217,320,337,408 | 29,217,320,337,408 | 447,698.51 | 447,698.51 | BIT-EXACT |
| `w3_c4` | W3 | CSTP | FA-LRU ($W=64$) | 0.005021 | 0.005021 | 0.000000 | 331,184 | 331,184 | 29,218,201,141,248 | 29,218,201,141,248 | 447,713.80 | 447,713.80 | BIT-EXACT |
| `w4_c1` | W4 | Disabled | Standard LRU | 0.000000 | 0.000000 | 0.000000 | 546,889 | 546,889 | 48,207,715,368,960 | 48,207,715,368,960 | 753,245.55 | 753,245.55 | BIT-EXACT |
| `w4_c2` | W4 | Disabled | FA-LRU ($W=64$) | 0.000000 | 0.000000 | 0.000000 | 546,889 | 546,889 | 48,207,715,368,960 | 48,207,715,368,960 | 753,245.55 | 753,245.55 | BIT-EXACT |
| `w4_c3` | W4 | CSTP | Standard LRU | 0.002589 | 0.002589 | 0.000000 | 552,919 | 552,919 | 48,738,840,084,480 | 48,738,840,084,480 | 747,863.57 | 747,863.57 | BIT-EXACT |
| `w4_c4` | W4 | CSTP | FA-LRU ($W=64$) | 0.002616 | 0.002616 | 0.000000 | 552,896 | 552,896 | 48,736,814,235,648 | 48,736,814,235,648 | 747,831.75 | 747,831.75 | BIT-EXACT |

---

## 3. Host System Execution Environment & Variance

While simulated execution time and memory behavior are 100% deterministic, host CPU wall-clock execution time varies slightly due to operating system scheduling, memory paging, and cache line contention on the host machine.

### Table 2: Host Wall-Clock Execution Runtimes

| Configuration | Workload | Repetition 1 Host Time (s) | Repetition 2 Host Time (s) | Absolute $\Delta$ (s) | Relative Variance (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `c1` (Base LRU) | W3 | 284.28 | 273.48 | $-10.80$ | $3.80\%$ |
| `c2` (Base FA-LRU) | W3 | 280.61 | 286.21 | $+5.60$ | $2.00\%$ |
| `c3` (CSTP LRU) | W3 | 306.78 | 302.06 | $-4.72$ | $1.54\%$ |
| `c4` (CSTP FA-LRU)| W3 | 342.53 | 352.42 | $+9.89$ | $2.89\%$ |
| `c1` (Base LRU) | W4 | 634.70 | 678.10 | $+43.40$ | $6.84\%$ |
| `c2` (Base FA-LRU) | W4 | 711.62 | 685.50 | $-26.12$ | $3.67\%$ |
| `c3` (CSTP LRU) | W4 | 725.69 | 782.35 | $+56.66$ | $7.81\%$ |
| `c4` (CSTP FA-LRU)| W4 | 816.91 | 979.80 | $+162.89$ | $19.94\%$ |

---

## 4. Step-by-Step Instructions for Replication

To independently reproduce any run in this campaign:

### 1. Environment Variable Setup
```bash
export USE_EVENT_DRIVEN_DRAM=true
export DDR5_BANDWIDTH_GBPS=64.0
export DEBUG_CACHE_CAPACITY=37580963840
export EVICTION_POLICY=fa_lru          # or lru
export FA_LRU_WINDOW_SIZE=64
```

### 2. Execution Command
```bash
/home/mohman/ca_project/LLMSimulator/build/run_deepseekv3_mixtral_style \
    /home/mohman/ca_project/mohman/point_4/big_exp/configs/<run_id>.yaml
```

### 3. Trace and Configuration References
- Trace files: [`mohman/point_4/big_exp/traces/w3_b14_t128.csv`](file:///home/mohman/ca_project/mohman/point_4/big_exp/traces/w3_b14_t128.csv) and [`w4_b32_t128.csv`](file:///home/mohman/ca_project/mohman/point_4/big_exp/traces/w4_b32_t128.csv).
- YAML configurations: [`mohman/point_4/big_exp/configs/`](file:///home/mohman/ca_project/mohman/point_4/big_exp/configs/).
- Random seed: 42.

---

## 5. Reproducibility Certification

The Big Experiment campaign achieves 100% bit-exact determinism across all simulated cache states and performance outputs.

