# Point 4 Phase 2.6: Source Code Changelog & Modifications

## MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference

---

## 1. Executive Summary

This document details all code modifications applied during **Phase 2.6** to resolve the demand accounting discrepancy, ensure sequence-aware decode-step tracking, prevent missing decode-step event collapsing, and extend the unit test suite.

All modifications strictly obey the **zero-ternary operator rule**:
- C++ codebase: Exactly 0 ternary operators (`? :`).
- Python scripts: Exactly 0 conditional expressions (`ast.IfExp`).

---

## 2. File-by-File Detailed Changelog

### 2.1 `src/prefetch/prefetch_metrics_collector.h`

#### 2.1.1 Default Decode Step in `CollectorDemandRecord`
- Changed default `decode_step` from `0` to `-1` to ensure uninitialized records are not erroneously attributed to decode step 0.
```cpp
struct CollectorDemandRecord {
    uint64_t demand_id = 0;
    int request_id = 0;
    int seq_id = 0;
    int token_id = 0;
    int decode_step = -1; // Updated from 0
    int layer_id = -1;
    // ...
};
```

#### 2.1.2 Added `invalid_decode_step_demands` Telemetry Counter
- Added counter to `AggregatedPrefetchMetrics` to explicitly flag and track demands that lacked valid decode-step attribution.
```cpp
struct AggregatedPrefetchMetrics {
    // ...
    uint64_t coalesced_actual_demands = 0;
    uint64_t routed_actual_demands = 0;
    uint64_t tensor_actual_demands = 0;
    uint64_t invalid_decode_step_demands = 0; // Added in Phase 2.6
    uint64_t coalesced_resident_hits = 0;
    // ...
};
```

#### 2.1.3 Added Sequence Decode-Step Lookup & Public Accessors
- Added `token_to_gen_step_` mapping and accessor functions:
```cpp
    const std::vector<CollectorDemandRecord>& get_demands() const { return demands_; }
    int get_current_decode_step() const { return current_decode_step_; }
    int lookup_token_gen_step(int request_id, int seq_id, int token_id) const;

private:
    int current_decode_step_ = -1; // Default to -1
    std::unordered_map<std::tuple<int, int, int>, int, struct Key3TupleHash> token_to_gen_step_;
```

---

### 2.2 `src/prefetch/prefetch_metrics_collector.cpp`

#### 2.2.1 Sequence-Aware Resolution in `record_expert_demanded()`
- Replaced unsafe fallback to `token_id` with sequence-aware lookup in `token_to_gen_step_`:
```cpp
    int d_step = decode_step;
    if (d_step < 0) {
        auto t_key = std::make_tuple(request_id, seq_id, token_id);
        auto t_it = token_to_gen_step_.find(t_key);
        if (t_it != token_to_gen_step_.end()) {
            d_step = t_it->second;
        } else if (current_decode_step_ >= 0) {
            d_step = current_decode_step_;
        } else {
            d_step = -1;
        }
    }
```

#### 2.2.2 Sequence Registration in `record_token_start()`
- Updated `record_token_start()` to register the token in `token_to_gen_step_`:
```cpp
void PrefetchMetricsCollector::record_token_start(
    int request_id, int seq_id, int token_id, int gen_step, double start_ts) {
    if (!enabled_) {
        return;
    }

    current_decode_step_ = gen_step;
    auto key = std::make_tuple(request_id, seq_id, token_id);
    token_start_times_[key] = start_ts;
    token_to_gen_step_[key] = gen_step;
}
```

#### 2.2.3 Prevention of Missing Decode Step Collapsing in `finalize()`
- Demands with `decode_step < 0` receive strictly unique negative keys so they never collapse together:
```cpp
    std::map<std::tuple<int, int, int>, CollectorDemandRecord> stage_coalesced_demands;
    uint64_t invalid_decode_step_count = 0;
    int invalid_seq_counter = -1;
    for (const auto& dem : demands_) {
        int step = dem.decode_step;
        if (step < 0) {
            invalid_decode_step_count = invalid_decode_step_count + 1;
            step = invalid_seq_counter;
            invalid_seq_counter = invalid_seq_counter - 1;
        }
        auto coalesced_key = std::make_tuple(step, dem.layer_id, dem.expert_id);
        if (stage_coalesced_demands.find(coalesced_key) == stage_coalesced_demands.end()) {
            stage_coalesced_demands[coalesced_key] = dem;
        }
    }

    metrics_.coalesced_actual_demands = stage_coalesced_demands.size();
    metrics_.invalid_decode_step_demands = invalid_decode_step_count;
```

#### 2.2.4 Telemetry Exporters
- Added `invalid_decode_step_demands` to JSON and CSV export routines.

---

### 2.3 `eval/test_fa_lru.cpp`

#### 2.3.1 Value Copy Fix for Metric Snapshots
- Replaced `const auto& m_b = col.get_metrics();` with `AggregatedPrefetchMetrics m_b = col.get_metrics();` to eliminate reference aliasing.

#### 2.3.2 Added Scenarios F, G, H, and Test 14
- **Scenario F:** Multi-sequence batch with different prompt lengths ($L=10$ vs $L=30$) coalescing into 1 demand.
- **Scenario G:** Unassigned decode step events not collapsing.
- **Scenario H:** Direct inspection of individual `CollectorDemandRecord::decode_step` fields.
- **Test 14:** Full simulator batch decode-step propagation across 4 sequences and 2 generation steps ($B=4, T=2$: Tensor=24, Routed=8, Coalesced=5).

---

## 3. Zero-Ternary Compliance Verification

A static analysis was conducted on all modified files:
- `git diff | grep -E "^\+[^\+].*\?"`: **0 matches.**
- All branching logic uses explicit `if/else` constructs.

