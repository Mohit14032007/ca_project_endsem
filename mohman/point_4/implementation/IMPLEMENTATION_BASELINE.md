# Implementation Baseline: Point 4 (FA-LRU HBM Eviction Policy)

## 1. Executive Summary

This document establishes the verified codebase baseline for **Point 4: Frequency-Aware LRU (FA-LRU) Eviction Policy Implementation and Validation** in `LLMSimulator`. All source files, git commit references, compiler configurations, and baseline build statuses have been audited and verified prior to modifying any source code.

---

## 2. Git Repository Status

- **Repository Root:** `/home/mohman/ca_project/LLMSimulator`
- **Active Branch:** `main`
- **Head Commit Hash:** `d5bf61ff0c066c06f72173189f9924329f613a88`
- **Upstream Tracking:** Up to date with `origin/main`

### Existing Uncommitted Changes (Preserved from Earlier Points)
The following files were modified during Points 1 through 3 and Phase 4D/5 developments and are strictly preserved:
- [`CMakeLists.txt`](file:///home/mohman/ca_project/LLMSimulator/CMakeLists.txt)
- [`config.yaml`](file:///home/mohman/ca_project/LLMSimulator/config.yaml)
- [`eval/CMakeLists.txt`](file:///home/mohman/ca_project/LLMSimulator/eval/CMakeLists.txt)
- [`eval/test.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test.cpp)
- [`eval/test_deepseekv3_mixtral_style.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp)
- [`src/CMakeLists.txt`](file:///home/mohman/ca_project/LLMSimulator/src/CMakeLists.txt)
- [`src/hardware/cluster.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/cluster.cpp)
- [`src/hardware/cluster.h`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/cluster.h)
- [`src/hardware/device.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp)
- [`src/hardware/device.h`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.h)
- [`src/hardware/hardware_config.h`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/hardware_config.h)
- [`src/hardware/layer_impl.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/layer_impl.cpp)
- [`src/hardware/linear_impl.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/linear_impl.cpp)
- [`src/module/linear.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/module/linear.cpp)
- [`src/module/module_graph.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/module/module_graph.cpp)
- [`src/module/route.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/module/route.cpp)
- [`src/module/tensor.h`](file:///home/mohman/ca_project/LLMSimulator/src/module/tensor.h)
- [`src/scheduler/scheduler.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp)
- [`src/scheduler/scheduler.h`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.h)
- [`src/scheduler/sequence.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp)
- [`src/scheduler/sequence.h`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.h)

**Preservation Guarantee:** No existing unstaged changes or untracked test fixtures from earlier points will be reverted, deleted, or overwritten.

---

## 3. Compilation Baseline Verification

The baseline codebase was compiled from a clean build directory prior to implementation:
- **Build Directory:** `/home/mohman/ca_project/LLMSimulator/build`
- **Compiler:** `g++` (GCC 13.x with C++17 support)
- **Compilation Command:**
  ```bash
  cd /home/mohman/ca_project/LLMSimulator/build && make -j$(nproc)
  ```
- **Exit Status:** `0` (Success, 100% built targets)
- **Verification Timestamp:** `2026-11-10T13:44:15Z`

---

## 4. Source Files Targeted for Phase 2 Modifications

The following 9 files are modified to implement the FA-LRU eviction policy, instrumentation, and test coverage:

1. [`src/prefetch/prefetch_config.h`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_config.h):
   - Add `FA_LRU` to `EvictionPolicy` enum.
   - Add `fa_lru_window_size` (default: 16) to `PrefetchConfig`.

2. [`src/prefetch/prefetch_config.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_config.cpp):
   - Add string conversion and parsing for `fa_lru` in `to_string(EvictionPolicy)` and `parse_eviction_policy()`.
   - Add YAML parsing for `fa_lru_window_size` and environment variable overrides `EVICTION_POLICY`, `FA_LRU_WINDOW_SIZE`.
   - Validate $W \ge 1$ in `PrefetchConfig::validate()`.

3. [`src/module/tensor.h`](file:///home/mohman/ca_project/LLMSimulator/src/module/tensor.h):
   - Add `bool is_primary_expert_weight = false;` to `Tensor`.

4. [`src/module/expert.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/module/expert.cpp):
   - Flag $W_1$ (`gate_proj`) as `is_primary_expert_weight = true;`.
   - Explicitly flag $W_3$ (`up_proj`) and $W_2$ (`down_proj`) as `is_primary_expert_weight = false;`.

5. [`src/module/linear.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/module/linear.cpp):
   - Pass `A->is_primary_expert_weight` to `device->is_expert_resident()` in `Linear::launch`.

6. [`src/hardware/device.h`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.h):
   - Extend `ResidentExpertEntry` with `uint32_t access_frequency = 1;` and `uint32_t last_epoch = 0;`.
   - Add `uint32_t global_epoch = 0;`, `int fa_lru_window_size = 16;`.
   - Add `std::unordered_set<std::pair<int, int>, PairHash> previously_evicted_experts;` and `uint64_t reloaded_experts_count = 0;`.
   - Declare `materialize_lazy_decay()` and `select_fa_lru_victim()`.
   - Update `set_current_stage()` with wrap-around epoch advancement.

7. [`src/hardware/device.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp):
   - Implement `materialize_lazy_decay()` with shift-safe clamping ($\ge 31 \implies 1$).
   - Implement `select_fa_lru_victim()` with bottom-$W$ scanning, lazy decay, strict tie-breaking to oldest, and $W=1$ equivalence.
   - Update `is_expert_resident()` to materialize decay, increment frequency only on primary demands, and splice to MRU.
   - Update `mark_expert_resident()` to invoke `select_fa_lru_victim()` when `eviction_policy == EvictionPolicy::FA_LRU`, record mutually exclusive eviction reasons, record prefetch-induced evictions, and track reload history.

8. [`src/prefetch/prefetch_metrics_collector.h`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.h) & [`src/prefetch/prefetch_metrics_collector.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.cpp):
   - Instrument coalesced demand metrics ($D_{\text{coalesced}}$, $D_{\text{routed}}$, $D_{\text{tensor}}$).
   - Implement active tracking in `record_hbm_evicted()`, `record_hbm_resident()`, and `record_prefetch_induced_eviction()`.
   - Export comprehensive HBM eviction and hit statistics in `prefetch_summary.json` and `prefetch_metrics.csv`.

9. [`eval/test_deepseekv3_mixtral_style.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp):
   - Add startup logging for eviction policy and window size $W$.
   - Pass HBM cache metrics to `PrefetchMetricsCollector` on finalization.

---

## 5. Architectural Correctness Invariants

1. **Zero Ternary Operator Standard:** Neither `? :` in C++, `ast.IfExp` in Python, nor ternary prose syntax shall be introduced.
2. **Backward Compatibility:** When `eviction_policy == EvictionPolicy::LRU`, cache eviction behavior is 100% bit-identical to the baseline.
3. **PABI Isolation:** When `eviction_policy == EvictionPolicy::PABI`, speculative quarantine buffer and Belády pinning mechanisms remain unaltered.
4. **Residency Conservation Identity:** At all times:
   $$\text{demand\_insertions} + \text{prefetch\_insertions} - \text{total\_evictions} = |\text{resident\_experts}|$$
   $$\text{total\_evictions} = \text{unconsumed\_prefetch\_evictions} + \text{demand\_resident\_evictions} + \text{pabi\_speculative\_evictions}$$

