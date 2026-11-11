# Audit Gaps, Ambiguities, and Limitations — Point 4

## MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference

---

## 1. Executive Summary

This document enumerates all architectural gaps, uninstrumented hooks, configuration ambiguities, and implementation limitations identified during the Phase 1 read-only audit of `LLMSimulator`. These gaps must be explicitly recognized and addressed in the Phase 2 implementation.

---

## 2. Identified Codebase Gaps & Ambiguities

### Gap 1: Disconnect Between YAML Cache Capacity and Runtime Device Sizing
- **Finding:** In several experiment configuration files and documentation, `expert_cache_capacity_bytes` or `hbm_capacity_bytes` is specified under `simulation.prefetching`.
- **Code Audit:** In [`src/hardware/device.cpp:101-107`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L101-L107), `cache_capacity_bytes` is initialized to the hardcoded default `80000000000LL` (80 GB) and only overridden by:
  ```cpp
  const char* debug_cap = std::getenv("DEBUG_CACHE_CAPACITY");
  if (debug_cap != nullptr) {
    try {
      cache_capacity_bytes = std::stoll(debug_cap);
    } catch (...) {
    }
  }
  ```
- **Consequence:** If an experimenter sets `expert_cache_capacity_bytes: 35000000000` in YAML but forgets to export `DEBUG_CACHE_CAPACITY=35000000000` in bash, the simulator silently runs with 80 GB capacity!
- **Remediation for Phase 2:** Ensure runner harnesses explicitly pass `DEBUG_CACHE_CAPACITY`, and wire `prefetch_config.h` capacity directly into `Device::Device`.

---

### Gap 2: No-Op Collector Stubs for HBM Residency and Eviction Events
- **Finding:** In [`src/prefetch/prefetch_metrics_collector.cpp:328-352`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.cpp#L328-L352):
  ```cpp
  void PrefetchMetricsCollector::record_hbm_resident(...) {
    if (!enabled_) return;
    (void)layer_id; (void)expert_id; (void)transfer_id; ...
  }
  void PrefetchMetricsCollector::record_hbm_evicted(...) {
    if (!enabled_) return;
    (void)layer_id; (void)expert_id; (void)reason; ...
  }
  ```
- **Consequence:** While `Device::mark_expert_resident()` dutifully calls these methods, `PrefetchMetricsCollector` discards the information. The generated `prefetch_summary.json` and `prefetch_summary.csv` do not contain official cache eviction tallies or eviction reason breakdowns.
- **Remediation for Phase 2:** Un-stub these methods to aggregate `total_evictions`, `evictions_by_reason`, and `reloaded_experts_count` directly into the final metrics collector.

---

### Gap 3: Absence of Per-Expert Access Frequency in `ResidentExpertEntry`
- **Finding:** In [`src/hardware/device.h:134-164`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.h#L134-L164), `ResidentExpertEntry` contains `list_it`, `size_bytes`, `classification`, and prefetch flags, but **no counter for access frequency**.
- **Consequence:** Under standard LRU, the simulator only knows recency (via list order). The historical demand activation frequency of an expert is discarded.
- **Remediation for Phase 2:** Add `uint32_t access_frequency = 1;` and `double last_access_timestamp_ns = 0.0;` to `ResidentExpertEntry`.

---

### Gap 4: Absence of Expert Reload / Churn Tracking
- **Finding:** The simulator tracks total demand misses, but does not distinguish between:
  1. **Cold Miss:** The expert is demanded for the first time in the simulation.
  2. **Reload Miss (Churn):** The expert was previously resident, was evicted due to cache capacity pressure, and is now re-demanded.
- **Consequence:** Without reload tracking, cache thrashing can only be inferred indirectly from overall miss rates.
- **Remediation for Phase 2:** Maintain a lightweight historical set or map of evicted `{layer_id, expert_id}` pairs to measure genuine cache churn.

---

### Gap 5: Eviction Invariants Under Concurrent Prefetch Admissions
- **Finding:** When multiple prefetches complete at the exact same device timestamp in `process_completed_prefetches()` ([`src/hardware/device.cpp:1412-1483`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L1412-L1483)), they are inserted sequentially. Under standard LRU, each completed prefetch immediately triggers capacity evictions if the cache is full.
- **Ambiguity:** If a prefetch displaces another freshly admitted prefetch that hasn't been demanded yet, it increments `unconsumed_prefetches`.
- **Verification Requirement:** In Phase 2, verify that FA-LRU does not prioritize unconsumed prefetches over active demand-verified experts.

---

### Gap 6: Belády Pinning is Quarantined to PABI Speculative Entries Only
- **Finding:** In [`src/hardware/device.cpp:828-832`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L828-L832), Belády score pinning is only active when `eviction_policy == EvictionPolicy::PABI && pabi_quarantine_enabled`.
- **Consequence:** Standard `LRU` mode has zero pin protection. Demand-loaded experts in use by current batch sequences can theoretically be evicted if intervening loads exceed cache capacity.
- **Scope Boundary:** FA-LRU will focus strictly on improving victim selection in the general cache pool without altering PABI quarantine structures.

