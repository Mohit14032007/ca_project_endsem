# Eviction Policy Extension Audit — LLMSimulator

## MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference

---

## 1. Executive Summary

This audit identifies the exact extension points within `LLMSimulator` required to integrate configurable expert eviction policies—specifically **Frequency-Aware LRU (FA-LRU)**—without modifying the timing model, DDR5 transfer engine, Ramulator, interconnect, or GPU compute accounting.

---

## 2. Extension Points in the Codebase

### 2.1 Configuration Layer: Enum & Parser Extension
The framework already contains an `EvictionPolicy` enum designed for policy extension.

- **Header File:** [`src/prefetch/prefetch_config.h:51-58`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_config.h#L51-L58)
  ```cpp
  enum class EvictionPolicy {
      LRU,                // Standard Least Recently Used eviction (baseline)
      LRU_PINNED,         // LRU with pin protection for in-flight/active experts (stub)
      PREFETCH_AWARE,     // Prefetch-guided eviction policy (stub)
      PABI,               // Predictor-Aware Buffer Insertion (Phase 4D.3)
      FA_LRU              // [EXTENSION POINT] Frequency-Aware LRU
  };
  ```

- **Source File:** [`src/prefetch/prefetch_config.cpp:142-167`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_config.cpp#L142-L167)
  Add string serialization and parsing:
  ```cpp
  // to_string
  case EvictionPolicy::FA_LRU: return "fa_lru";
  
  // parse_eviction_policy
  if (s == "fa_lru" || s == "frequency_aware_lru") return EvictionPolicy::FA_LRU;
  ```

- **YAML Ingestion:** [`src/prefetch/prefetch_config.cpp:402-404`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_config.cpp#L402-L404)
  Already parses `node["eviction_policy"]` via `parse_eviction_policy()`.
  Supports YAML configuration:
  ```yaml
  simulation:
    prefetching:
      eviction_policy: fa_lru
  ```

- **Environment Override:** [`src/hardware/device.cpp:119-125`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L119-L125)
  Already supports runtime override via `EVICTION_POLICY=fa_lru`.

---

### 2.2 Metadata Layer: `ResidentExpertEntry` Enhancement
To make eviction frequency-aware, each resident expert must maintain activation statistics.

- **Header File:** [`src/hardware/device.h:134-164`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.h#L134-L164)
  ```cpp
  struct ResidentExpertEntry {
    int layer_id = -1;
    int expert_id = -1;
    long long size_bytes = 0;
    ExpertClassification classification = ExpertClassification::DEMAND_VERIFIED;
    std::list<std::tuple<int, int, long long>>::iterator list_it;
    
    // Existing fields...
    bool is_prefetched = false;
    bool was_demanded = false;

    // [EXTENSION POINT] Frequency and Age Metadata for FA-LRU
    uint32_t access_frequency = 1;         // Cumulative demand access count
    double last_access_timestamp_ns = 0.0; // Simulated time of most recent access
    uint32_t insertion_epoch = 0;          // Token/stage epoch when expert entered HBM
  };
  ```

---

### 2.3 Access Path: Recency & Frequency Update
When an expert is accessed, its frequency and recency must be updated.

- **Source File:** [`src/hardware/device.cpp:710-747`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L710-L747)
  Inside `Device::is_expert_resident()`:
  ```cpp
  // Demand hit path:
  if (eviction_policy == EvictionPolicy::FA_LRU) {
    it->second.access_frequency += 1;
    it->second.last_access_timestamp_ns = get_time();
    // Splice to MRU back
    auto node = *(it->second.list_it);
    lru_expert_list.erase(it->second.list_it);
    lru_expert_list.push_back(node);
    it->second.list_it = std::prev(lru_expert_list.end());
  }
  ```
  > [!NOTE]
  > Speculative probing passes `is_demand_access = false`, which already short-circuits at line 681. Therefore, unverified prefetches never artificially increment `access_frequency`.

---

### 2.4 Insertion & Victim Selection Path: `Device::mark_expert_resident()`
Currently, victim selection in standard LRU is hardcoded to pop unconditionally from the front of `lru_expert_list`:
- **Current LRU Code:** [`src/hardware/device.cpp:761-764`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L761-L764)
  ```cpp
  while (resident_bytes + expert_size > cache_capacity_bytes && !lru_expert_list.empty()) {
    auto lru_expert = lru_expert_list.front();
    lru_expert_list.pop_front();
    // evict lru_expert...
  }
  ```

- **[EXTENSION POINT] Configurable Victim Selector:**
  Replace the unconditional `lru_expert_list.front()` with a modular victim selection dispatch:
  ```cpp
  while (resident_bytes + expert_size > cache_capacity_bytes && !lru_expert_list.empty()) {
    std::list<std::tuple<int, int, long long>>::iterator victim_it;
    if (eviction_policy == EvictionPolicy::FA_LRU) {
      victim_it = select_fa_lru_victim();
    } else {
      victim_it = lru_expert_list.begin(); // Standard LRU
    }
    // Evict *victim_it and erase from lru_expert_list
  }
  ```

---

### 2.5 Metrics Layer: Un-stubbing Collector Hooks
To capture evictions and residency transitions accurately in the evaluation reports:

- **Source File:** [`src/prefetch/prefetch_metrics_collector.cpp:328-352`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.cpp#L328-L352)
  Implement `record_hbm_resident()` and `record_hbm_evicted()`:
  - Aggregate `total_hbm_evictions`.
  - Track `evictions_by_reason` (`CAPACITY_EVICTION`, `UNCONSUMED_PREFETCH`).
  - Track `reloaded_experts_count` (how often an evicted `{layer_id, expert_id}` is re-demanded).

---

## 3. Modularity and Isolation Assessment

### 3.1 Can FA-LRU be implemented without altering the memory architecture?
**Yes.** The eviction policy interacts exclusively with:
1. `resident_experts` (hash map membership).
2. `lru_expert_list` (recency list ordering).
3. `resident_bytes` (byte accounting).

It **does not affect**:
- `DDR5TransferEngine`: Transfer submission, queueing, active transmission, preemption, and completion timing remain 100% identical.
- `RamulatorInterface`: Off-chip DRAM timing, rank/bank scheduling, and PHY calibration remain untouched.
- `TopModuleGraph` / Compute Timing: GEMM execution duration, parallel tensor scheduling, and device clock progression remain unchanged.
- Interconnect: ICT bandwidth and cross-device latency remain unchanged.

### 3.2 Side Effects and Precautions
- **Iterators:** `ResidentExpertEntry::list_it` must always point to a valid node in `lru_expert_list`. Whenever a node is spliced, erased, or re-inserted, `list_it` must be updated atomically.
- **Complexity:** Selecting an FA-LRU victim must not degrade simulation speed. Scanning the entire 900-entry cache on every eviction would incur $O(N)$ overhead. A bounded candidate window (e.g., inspecting the $K=16$ oldest LRU entries) maintains $O(1)$ constant-time execution while providing robust frequency protection.

