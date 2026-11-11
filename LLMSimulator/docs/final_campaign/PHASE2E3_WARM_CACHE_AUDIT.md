# Phase 2E.3 — Warm-Cache Behavior Audit

## 1. Audit Objective and Scope

This audit inspects the cache initialization, persistence, and reset code paths in the `LLMSimulator` repository to answer the six required behavioral questions before launching experiments.

- **Authoritative Baseline Anchor:** `d5bf61ff0c066c06f72173189f9924329f613a88`
- **Source Files Inspected:**
  - `LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp`
  - `LLMSimulator/src/hardware/device.h`
  - `LLMSimulator/src/hardware/device.cpp`
  - `LLMSimulator/src/hardware/cluster.cpp`
  - `LLMSimulator/src/hardware/linear_impl.cpp`
  - `LLMSimulator/src/scheduler/sequence.cpp`

---

## 2. Answers to the Six Mandatory Audit Inquiries

### Question 1: Whether HBM starts empty at the beginning of each token or persists across tokens.
- **Finding:** **HBM persists across tokens.**
- **Code Evidence:**
  In `test_deepseekv3_mixtral_style.cpp:335`, `device->reset_expert_cache()` is called **once** prior to the start of the sequential decoding loop (`for (int step = 0; step < iter; step++)`).
  At the start of each step/token (line 433), `cluster->run(metadata)` invokes `Cluster::restartModuleGraph()`.
  In `src/hardware/cluster.cpp:44-51`:
  ```cpp
  void Cluster::restartModuleGraph() {
    for (int device_rank = 0; device_rank < num_total_device; device_rank++) {
      Device::Ptr device = get_device(device_rank);
      device->restartGraph();
      device->reset_status();
      device->reset_timeboard();
    }
  }
  ```
  `device->reset_status()` resets `status = StatusBoard();` (which resets per-token timing), but does **not** touch `resident_experts`, `lru_expert_list`, or `resident_bytes`.
  Therefore, the HBM expert cache persists across all sequential decoding tokens.

### Question 2: Whether demand-loaded experts remain resident across decoding tokens.
- **Finding:** **Yes, demand-loaded experts remain resident across decoding tokens.**
- **Code Evidence:**
  When an expert is demanded in `LinearImpl::forward_with_cache()` (`src/hardware/linear_impl.cpp:103`), `device->mark_expert_resident(weight->expert_layer_id, weight->expert_id, total_expert_size)` inserts the expert into `resident_experts` and `lru_expert_list`.
  The expert remains resident until evicted by the LRU (or PABI) replacement policy when `resident_bytes + expert_size > cache_capacity_bytes`.
  In Workload A1 with an 80 GB HBM capacity, up to $\approx 908$ expert instances ($88.08\text{ MB}$ each) can reside concurrently. Since 75 stages $\times 8$ routed experts $= 600$ lookups per token, a significant fraction of demanded experts remain resident in HBM across subsequent tokens.

### Question 3: Whether cache state is reset between requests.
- **Finding:** **Cache state is NOT reset between requests in the multi-token decoding loop.**
- **Code Evidence:**
  `test_deepseekv3_mixtral_style.cpp` processes sequential decoding tokens within a single persistent `Cluster` and `Device` instance. No request-boundary reset hook is called inside `runIteration` or the step loop. The cache state only resets if a test explicitly invokes `device->reset_expert_cache()`, which only occurs before step 0.

### Question 4: Whether `warm_hbm_preload` actually exists and what it loads.
- **Finding:** **`warm_hbm_preload` exists, but it operates as an oracle preload using future trace lookahead.**
- **Code Evidence:**
  In `test_deepseekv3_mixtral_style.cpp:401-421`:
  ```cpp
  if (warm_hbm_preload) {
    std::cout << "[WARM HBM PRELOAD] Preloading replay working set into HBM cache...\n";
    long long total_expert_size = 88080384LL;
    if (!trace_file.empty()) {
      for (int s = 0; s < max_batch_size; s++) {
        for (int t = 0; t < iter; t++) {
          for (int l = 0; l < num_moe_layers; l++) {
            size_t table_idx = (size_t)t * num_moe_layers + l;
            int target_layer = (l + 1) % num_moe_layers;
            for (int e_id : scheduler->sequences_info[s]->expert_list[table_idx]) {
              if (!device->is_expert_resident(target_layer, e_id)) {
                device->mark_expert_resident(target_layer, e_id, total_expert_size);
              }
            }
          }
        }
      }
    }
  }
  ```
  When enabled, `warm_hbm_preload` reads all future tokens ($t = 0 \dots \text{iter}-1$) from the offline trace and populates the cache before execution begins.
  Furthermore, line 409 uses `int target_layer = (l + 1) % num_moe_layers;`, which preloads experts with a 1-stage post-increment offset.
  Therefore, **`warm_hbm_preload` must remain `false`** for all scientifically rigorous comparative evaluations.

### Question 5: Whether all configurations receive identical initial cache states.
- **Finding:** **Yes, under standard configuration (`warm_hbm_preload: false`), all configurations start with an identically empty HBM cache.**
- **Code Evidence:**
  In all configurations (`B`, `FWD-S-H2`, `FWD-C-H2`), `warm_hbm_preload` is set to `false`. Each run invokes `device->reset_expert_cache()` at line 335, resulting in:
  - `resident_experts.size() == 0`
  - `resident_bytes == 0`
  - `lru_expert_list.empty() == true`
  - `inflight_registry.empty() == true`
  - `ddr5_transfer_engine` queues empty.
  Initial state equivalence is guaranteed.

### Question 6: Whether cache state can be initialized from previously demanded experts without using future routing information.
- **Finding:** **Yes. Natural sequential multi-token decoding inherently provides this without using future routing information.**
- **Code Evidence:**
  In a 4-token execution ($t = 0, 1, 2, 3$):
  - In Token 0 ($t = 0$), the cache is cold. Demand misses populate HBM with demanded experts.
  - In Tokens 1, 2, 3 ($t > 0$), previously demanded experts that remain in HBM serve as natural warm-cache hits.
  - This develops natural cache residency without reading future tokens or employing synthetic preloading.

---

## 3. Conclusion and Recommendations for Phase 2E.3

1. Cache persistence across sequential decoding tokens is verified.
2. Natural warm-cache residency develops after Token 0 without needing `warm_hbm_preload`.
3. `warm_hbm_preload` must remain `false` across all configurations to prevent future trace leakage.
4. Part C can proceed with a clean 4-token run starting from cold cache, allowing natural cache residency to develop across Tokens 1–3.
