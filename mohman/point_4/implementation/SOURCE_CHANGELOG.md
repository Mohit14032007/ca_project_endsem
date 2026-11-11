# Point 4 Phase 2: Source Code Changelog & File Modifications

## MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference

---

## 1. Overview of Modified Files

The following files in `LLMSimulator` were modified or added to implement the FA-LRU eviction policy, candidate windowing, shift-safe lazy decay, primary-demand frequency isolation, and eviction accounting:

| File Path | Component | Type of Change | Purpose |
| :--- | :--- | :--- | :--- |
| `src/prefetch/prefetch_config.h` | Configuration | Modified | Added `FA_LRU` enum value, `fa_lru_window_size` parameter. |
| `src/prefetch/prefetch_config.cpp` | Configuration | Modified | YAML parser for `fa_lru`, env overrides (`EVICTION_POLICY`, `FA_LRU_WINDOW_SIZE`), validation. |
| `src/module/tensor.h` | Module Graph | Modified | Added `is_primary_expert_weight` flag to `Tensor`. |
| `src/module/expert.cpp` | Model Structure | Modified | Set `is_primary_expert_weight = true` on $W_1$, `false` on $W_3, W_2$. |
| `src/module/linear.cpp` | Execution | Modified | Propagated `is_primary_expert_weight` into `is_expert_resident()` demand check. |
| `src/hardware/linear_impl.cpp` | Hardware Sim | Modified | Propagated `is_primary_expert_weight` into hardware `is_expert_resident()`. |
| `src/hardware/device.h` | Hardware Cache | Modified | Added `access_frequency`, `last_epoch`, `select_fa_lru_victim`, `materialize_lazy_decay`. |
| `src/hardware/device.cpp` | Hardware Cache | Modified | Implemented FA-LRU candidate selection, lazy decay, eviction categorization, reload tracking. |
| `src/prefetch/prefetch_metrics_collector.h` | Telemetry | Modified | Added coalesced demand and eviction breakdown fields and API methods. |
| `src/prefetch/prefetch_metrics_collector.cpp` | Telemetry | Modified | Telemetry processing, artifact export (`prefetch_metrics.csv`, `prefetch_summary.json`). |
| `eval/test_deepseekv3_mixtral_style.cpp` | Runner Binary | Modified | Telemetry linkage at finalization, startup policy logging. |
| `eval/CMakeLists.txt` | Build System | Modified | Added `test_fa_lru` executable target. |
| `eval/test_fa_lru.cpp` | Validation | **New File** | 12-criterion unit test suite validating FA-LRU semantics and invariants. |

---

## 2. Contiguous Diff Breakdown by File

### 2.1 `src/prefetch/prefetch_config.h`

```cpp
// Enum addition
enum class EvictionPolicy {
    LRU,
    LRU_PINNED,
    PREFETCH_AWARE,
    PABI,
    FA_LRU          // Point 4: Frequency-Aware LRU
};

// PrefetchConfig struct addition
struct PrefetchConfig {
    ...
    EvictionPolicy eviction_policy = EvictionPolicy::LRU;
    int fa_lru_window_size = 16;  // Point 4: Default candidate window W = 16
    ...
};
```

---

### 2.2 `src/prefetch/prefetch_config.cpp`

```cpp
// String serialization
std::string to_string(EvictionPolicy policy) {
    if (policy == EvictionPolicy::FA_LRU) return "fa_lru";
    ...
}

// String parsing
EvictionPolicy parse_eviction_policy(const std::string& str) {
    std::string s = to_lower(str);
    if (s == "fa_lru" || s == "falru" || s == "frequency_aware_lru") return EvictionPolicy::FA_LRU;
    ...
}

// Validation & Environment Overrides
if (config.eviction_policy == EvictionPolicy::FA_LRU) {
    if (config.fa_lru_window_size < 1) {
        throw std::invalid_argument("[CONFIG_ERROR] fa_lru_window_size must be >= 1");
    }
}

const char* env_eviction = std::getenv("EVICTION_POLICY");
if (env_eviction != nullptr) {
    config.eviction_policy = parse_eviction_policy(std::string(env_eviction));
}

const char* env_window = std::getenv("FA_LRU_WINDOW_SIZE");
if (env_window != nullptr) {
    config.fa_lru_window_size = std::stoi(std::string(env_window));
}
```

---

### 2.3 `src/module/tensor.h` & `src/module/expert.cpp`

```cpp
// In tensor.h
class Tensor {
    ...
    bool is_primary_expert_weight = false; // Point 4: true only for W1
};

// In expert.cpp (Expert::build_expert_subgraph)
// W1 (gate_proj): primary projection
linear1_weight->is_primary_expert_weight = true;

// W3 (up_proj) and W2 (down_proj): secondary projections
linear3_weight->is_primary_expert_weight = false;
linear2_weight->is_primary_expert_weight = false;
```

---

### 2.4 `src/hardware/device.h`

```cpp
struct ResidentExpertEntry {
    ...
    uint32_t access_frequency = 1;  // Point 4: Raw frequency counter
    uint32_t last_epoch = 0;        // Point 4: Epoch of last access
};

class Device {
    ...
    int global_epoch = 0;
    int fa_lru_window_size = 16;
    std::unordered_set<std::pair<int, int>, PairHash> previously_evicted_experts;
    uint64_t reloaded_experts_count = 0;

    void materialize_lazy_decay(ResidentExpertEntry& entry);
    std::pair<int, int> select_fa_lru_victim();
    ...
};
```

---

### 2.5 `src/hardware/device.cpp`

```cpp
// Shift-safe lazy decay
void Device::materialize_lazy_decay(ResidentExpertEntry& entry) {
    if (global_epoch <= (int)entry.last_epoch) {
        return;
    }
    uint32_t epoch_diff = (uint32_t)(global_epoch - (int)entry.last_epoch);
    if (epoch_diff >= 31) {
        entry.access_frequency = 0;
    } else {
        entry.access_frequency = (entry.access_frequency >> epoch_diff);
    }
    entry.last_epoch = (uint32_t)global_epoch;
}

// Candidate window victim selection
std::pair<int, int> Device::select_fa_lru_victim() {
    if (lru_expert_list.empty()) return {-1, -1};
    if (fa_lru_window_size <= 1) return lru_expert_list.front();

    auto it = lru_expert_list.begin();
    int count = 0;
    uint32_t min_freq = UINT32_MAX;
    std::pair<int, int> best_victim = {-1, -1};

    while (it != lru_expert_list.end() && count < fa_lru_window_size) {
        auto key = *it;
        auto map_it = resident_experts.find(key);
        if (map_it != resident_experts.end()) {
            if (!map_it->second.is_pinned) {
                materialize_lazy_decay(map_it->second);
                uint32_t f = map_it->second.access_frequency;
                if (f < min_freq) {
                    min_freq = f;
                    best_victim = key;
                }
            }
        }
        it++;
        count++;
    }
    if (best_victim.first == -1) return lru_expert_list.front();
    return best_victim;
}
```

---

### 2.6 `src/prefetch/prefetch_metrics_collector.cpp`

```cpp
// Coalesced stage deduplication in finalize()
void PrefetchMetricsCollector::finalize(double total_sim_time_ns, uint64_t final_hbm_occupancy_bytes) {
    ...
    std::unordered_map<int, std::unordered_map<int, std::unordered_set<int>>> stage_experts;
    for (const auto& ev : demand_events_) {
        stage_experts[ev.token_idx][ev.layer_id].insert(ev.expert_id);
    }
    uint64_t coalesced_cnt = 0;
    for (const auto& tok_pair : stage_experts) {
        for (const auto& l_pair : tok_pair.second) {
            coalesced_cnt += l_pair.second.size();
        }
    }
    coalesced_actual_demands_ = coalesced_cnt;
    routed_actual_demands_ = coalesced_cnt;
    tensor_actual_demands_ = coalesced_cnt * 3;
    ...
}
```

