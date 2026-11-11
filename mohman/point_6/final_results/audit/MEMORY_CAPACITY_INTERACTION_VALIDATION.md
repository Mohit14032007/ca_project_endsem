# Memory Capacity Interaction Validation Report: Dedicated Expert Cache vs. Dynamic KV-Cache Growth

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Memory Hierarchy Code Audit  
**Author:** Antigravity Engineering  
**Date:** November 11, 2026  
**Status:** Code-Level Verified  

---

## 1. Executive Summary

This report delivers a definitive code-level audit of how `LLMSimulator` accounts for GPU memory capacity, specifically validating whether dynamic Key-Value (KV) cache growth competes with or displaces the High Bandwidth Memory (HBM) expert weight cache.

### Core Audit Conclusion:
**In `LLMSimulator`, the expert cache budget is a dedicated, static partition of $35\text{ GiB}$ (`cache_capacity_bytes = 37,580,963,840 bytes`). KV-cache growth does NOT dynamically shrink the expert cache capacity, decrement the available expert budget, or trigger expert evictions.**

Consequently:
- The context length sensitivity experiment isolates the pure computational overlap slack enabled by attention scaling ($\mathcal{O}(P+t)$).
- It does **not** model a unified memory architecture where growing KV cache physically crowds out expert weights.

---

## 2. Source Code Architecture & Tracing

### 2.1 Initialization of Expert Cache Capacity (`device.cpp` and `device.h`)
The expert cache capacity is declared and managed independently in `llm_system::Device`:

```cpp
// In LLMSimulator/src/hardware/device.h (line 301)
long long cache_capacity_bytes = 80000000000LL; // Default 80 GB

// In LLMSimulator/src/hardware/device.cpp (lines 102-108)
if (system_config.optimization_config.expert_cache_capacity_bytes > 0) {
    cache_capacity_bytes = system_config.optimization_config.expert_cache_capacity_bytes;
}
```
In our simulation configuration:
`expert_cache_capacity_bytes: 37580963840` (exactly $35\text{ GiB}$).

### 2.2 Admission and Eviction Logic (`device.cpp`)
All admission checks and eviction triggers compare current resident expert bytes (`resident_bytes`) solely against `cache_capacity_bytes`:

```cpp
// In LLMSimulator/src/hardware/device.cpp (lines 958, 1134, 1225)
while (resident_bytes + expert_size > cache_capacity_bytes && !lru_expert_list.empty()) {
    // Evict victim expert from HBM
    ...
}
```
Every expert in DeepSeek-V3 occupies exactly $88,080,384\text{ bytes}$ ($84.0\text{ MiB}$). The maximum resident capacity is:
$$\text{Max Resident Experts} = \left\lfloor \frac{37,580,963,840}{88,080,384} \right\rfloor = 426\text{ experts}$$

### 2.3 Independent Accounting of KV Cache (`layer.cpp` and `attention_gen_impl.cpp`)
In `LLMSimulator`, the KV cache is managed as independent tensor objects (`k_cache` and `v_cache`):
- `k_cache->setShape({accumul_len, head_dim * num_kv_heads});`
- In `attention_gen_impl.cpp`, memory reads to the KV cache are issued to the memory model via `issueRamulator()` or `getIdealMemoryStatus()`.
- **Zero code links `k_cache` allocation to `resident_bytes`.**
- **Zero code modifies `device->cache_capacity_bytes` dynamically during execution.**

---

## 3. Physical GPU vs. Simulator Divergence Disclosure

To ensure scientific honesty, the relationship between simulator accounting and physical hardware must be explicitly disclosed:

```
+---------------------------------------------------------------------------------------------------+
|                           MEMORY ALLOCATION ARCHITECTURE COMPARISON                               |
+--------------------------+------------------------------------+-----------------------------------+
| Metric / Feature         | Physical GPU (e.g. NVIDIA A100 80G)| LLMSimulator Configuration        |
+--------------------------+------------------------------------+-----------------------------------+
| Memory Architecture      | Single Unified Physical Pool (80G) | Dedicated Expert Cache Partition  |
| Expert Capacity          | Shares pool with KV & Activations  | Dedicated 35 GiB Static Budget    |
| Impact of Long Context   | KV cache shrinks available expert  | KV cache increases attention time;|
|                          | memory; forces early evictions     | expert capacity remains 35 GiB    |
| Primary Evaluated Effect | Memory footprint compression       | Temporal overlap slack expansion  |
+--------------------------+------------------------------------+-----------------------------------+
```

### Statement for Academic Publication:
> **"In our simulation configuration, LLMSimulator allocates a dedicated 35 GiB cache partition exclusively for expert weights. Dynamic KV-cache expansion increases attention computational duration and DRAM traffic, but does not contract the expert cache budget. This setup directly isolates the beneficial effect of attention compute slack on asynchronous prefetch completion, without the confounding variable of dynamic capacity contraction."**

