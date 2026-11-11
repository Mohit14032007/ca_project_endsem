# Phase A.2: Configuration and Source Implementation Audit

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Campaign:** Big Experiment: LRU vs. FA-LRU ($W=64$) Under Large MoE Workloads  
**Evaluation Scope:** Verification of Simulator Implementation, Cache Mechanisms, and Control Invariants  
**Status:** AUDIT PASSED (All Architectural Invariants Verified)  
**Date:** 2026-11-10  

---

## 1. Executive Summary

This audit verifies the implementation and exact configuration parameters for the Big Experiment evaluating Standard LRU against Frequency-Aware LRU with window size $W=64$. Every subsystem governing memory allocation, eviction victim selection, frequency accounting, speculative prefetch admission, and DDR5 transfer timing was forensically audited directly against the LLMSimulator source code.

All operational rules and conservation invariants required for Phase B execution are strictly satisfied.

---

## 2. Experimental Parameter Configuration

The Big Experiment enforces identical hardware and model parameters across all runs, varying solely the eviction policy and the presence of CSTP prefetching:

### Table 1: Audited Simulation Configuration Parameters

| Subsystem | Parameter | Exact Audited Value | Source Code / Configuration Verification |
| :--- | :--- | :--- | :--- |
| **HBM Cache** | Capacity Budget ($M_{\text{capacity}}$) | $37,580,963,840\text{ Bytes}$ ($35.000\text{ GiB}$) | Enforced via `hbm.capacity_bytes` & `DEBUG_CACHE_CAPACITY` |
| | Resident Slots Available | 426 expert slots | $\lfloor 37,580,963,840 / 88,080,384 \rfloor = 426$ |
| | Eviction Policy Window ($W$) | Exactly $W=64$ candidates | `fa_lru_window_size: 64` & `FA_LRU_WINDOW_SIZE=64` |
| **Model Topology** | Model Architecture | DeepSeek-V3 Mixtral-style | 43 total layers (3 dense, 40 MoE stages) |
| | Active MoE Stages ($L$) | 40 stages (Stages 0 to 39) | `num_moe_layers = 40` |
| | Routed Experts Per Stage | 256 experts | `num_routed_expert = 256` |
| | Routing Degree (Top-$K$) | 8 experts per token | `top_k = 8` |
| | Expert Footprint | $88,080,384\text{ Bytes}$ ($\approx 83.999\text{ MiB}$) | Total expert parameter size (Gate, Up, Down FP16) |
| **DRAM System** | Transfer Model | Calibrated Event-Driven DRAM | `use_event_driven_dram: true` |
| | Backing Memory Bandwidth | $64.0\text{ GB/s}$ | `transfer_rate_gbps: 64.0` |
| | Memory Scale Factor | $0.5$ (B100 HBM3E calibrated) | `memory_scale_factor = 0.5` |
| **CSTP Engine** | Prediction Horizon ($H$) | $H=1$ stage lookahead | `prediction_horizon: 1` |
| | Prefetch Budget ($K$) | $K=4$ candidates per stage | `budget: 4` |
| | Transfer Coordination | Event-driven DDR5 engine | Prefetches issued asynchronously ahead of demand |

---

## 3. Subsystem Implementation Audit

### 3.1 Standard LRU Baseline
- **Implementation:** [`LLMSimulator/src/hardware/device.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L858-L872)
- **Mechanics:** Manages resident entries in doubly linked list `lru_expert_list`. On hit, entries are spliced to MRU (back). When evicting, the list is traversed from `begin()` (LRU). The first unpinned entry is selected as the victim.

### 3.2 FA-LRU Architecture and $W=64$ Candidate Window
- **Implementation:** [`LLMSimulator/src/hardware/device.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L701-L749)
- **Candidate Window Scan:** In `Device::select_fa_lru_victim()`, FA-LRU scans at most $W=64$ entries starting from `lru_expert_list.begin()`.
- **Victim Selection:** For each entry, lazy decay is materialized. The unpinned entry possessing the strict minimum decayed frequency (`freq < min_freq`) is selected.
- **Tie-Breaking Order:** Because `<` is used, when entries share identical frequencies, `best_it` is never overwritten, strictly preserving the oldest (LRU) entry in the candidate window.
- **Pass 2 Fallback:** If all 64 entries in the bottom window are pinned, Pass 2 traverses beyond the window to evict the first available unpinned entry.

### 3.3 Frequency Tracking and Lazy Epoch Discounting
- **Implementation:** [`LLMSimulator/src/hardware/device.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L682-L699)
- **Epoch Boundary:** Incremented in `Device::set_current_stage(int stage)` when `stage == 0 && current_stage_id > 0` ([`device.h#L338`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.h#L338)). This guarantees that an epoch corresponds to one full 40-stage decode generation step.
- **Lazy Evaluation:** Bit-shift halving (`access_frequency >> epoch_diff`) is computed lazily upon entry touch or victim evaluation, avoiding $O(N)$ full-cache sweeps.

### 3.4 Demand-Only Frequency Updating & Speculative Probe Invariant
- **Implementation:** [`LLMSimulator/src/hardware/device.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L761-L774)
- **Speculative Probe Immunity:** Line 762 enforces:
  ```cpp
  if (!is_demand_access) {
    return true; // Speculative probe: no promotion, no frequency update, no decay
  }
  ```
- **Primary vs. Secondary Demands:**
  - Gate GEMM projection has `is_primary = true`. On first demand hit: `access_frequency = 2, was_demanded = true`. On subsequent hits: `access_frequency++`.
  - Up and Down sister GEMM projections have `is_primary = false`. They do NOT increment `access_frequency`.
  - Speculative prefetch insertions enter with `access_frequency = 1` and `was_demanded = false`.

### 3.5 Pinning Invariant
- In both LRU and FA-LRU passes, lines 721 and 865 explicitly check:
  ```cpp
  if (map_it->second.is_pinned) {
    continue; // Pinned entries can never be evicted
  }
  ```
  Pinned lines are strictly immune from eviction until their pin deadline expires.

### 3.6 Memory Traffic and Eviction Accounting
- Evictions are categorized into mutually exclusive primary bins:
  - `DEMAND_RESIDENT`: Demand-verified lines evicted.
  - `UNCONSUMED_PREFETCH`: Speculative lines evicted prior to demand arrival.
  - `PABI_SPECULATIVE`: Quarantine partition evictions.
- Secondary attribution `PREFETCH_INDUCED` tracks demand lines displaced specifically by incoming prefetch lines without double counting.

---

## 4. Audit Conclusion

The configuration and implementation audit is certified. All policy rules, parameter settings, and accounting mechanics are verified for the $W=64$ Big Experiment.

