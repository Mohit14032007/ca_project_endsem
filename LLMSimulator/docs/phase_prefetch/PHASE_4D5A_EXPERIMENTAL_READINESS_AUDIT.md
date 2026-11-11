# Phase 4D.5A — Read-Only Experimental Readiness Audit

## Executive Summary

Phase 4D.5A performs a **strictly read-only architectural and operational audit** of the `LLMSimulator` framework following the completion of Phase 4D.4 (Belády-style predictor-score pinning and stage-deadline expiration). The objective is to determine whether the simulator is ready for controlled end-to-end comparative experiments across:
1. Baseline (No predictor, no prefetch, LRU)
2. ST-MoE Prediction-Only
3. CSTP Prediction-Only
4. ST-MoE with Prefetching and LRU
5. CSTP with Prefetching and LRU
6. ST-MoE with PABI Quarantine
7. CSTP with PABI Quarantine
8. ST-MoE with PABI Quarantine and Pinning
9. CSTP with PABI Quarantine and Pinning

### Executive Readiness Verdict: CONDITIONAL GO / NO-GO

- **Prediction-Only Experiments (Configurations 1, 2, 3)**: **GO**.
  - The live prediction pipeline (`LivePredictionEngine`), predictor registry, trace loading, and metric logging are fully integrated and mathematically validated against offline replay (100% parity verified in Phase 2C).
- **Asynchronous Hardware Prefetching Experiments (Configurations 4 through 9)**: **NO-GO (Pending 4 Targeted Integration Wiring Points in Phase 4D.5B)**.
  - While all core subsystems (`DDR5TransferEngine`, `InflightRegistry`, `PABI` dual-list quarantine, Belády pin-selection, and stage-deadline expiration) are fully implemented and passed 100% of unit/integration tests (186/186 tests passing), **they are currently decoupled from the production simulation loop (`test_deepseekv3_mixtral_style.cpp` / `Cluster::run`)**.
  - Specifically, candidate dispatch (`dispatch_candidates`), pin expiration (`expire_pins_for_stage`), stage tracking (`set_current_stage`), and `Scheduler::init_prediction_engine()` operating mode support have not yet been wired into the live execution harness.
  - Launching full experiments under Configurations 4–9 prior to wiring these hooks would produce passive demand-only execution without speculative transfers or active pin protection.

The sections below present the forensic audit findings with exact file paths, line numbers, and metric equations.

---

## 1. Audit 1 — Configuration and Experiment Control

### 1.1 The 9 Target Configurations Matrix

To evaluate MoE expert prefetching systematically, the experiment matrix spans 9 configurations. Below are the exact YAML fields and environment variable overrides required for each.

| # | Configuration Name | Mode | Predictor | Eviction Policy | Quarantine | Pinning | Transfer Model |
|---|---|---|---|---|---|---|---|
| **1** | Baseline | `none` | `none` | `lru` | Disabled | Disabled | `event_driven` |
| **2** | ST-MoE Prediction-Only | `prediction_only` | `st_moe` | `lru` | Disabled | Disabled | N/A (No Xfer) |
| **3** | CSTP Prediction-Only | `prediction_only` | `cstp` | `lru` | Disabled | Disabled | N/A (No Xfer) |
| **4** | ST-MoE + Prefetch + LRU | `hardware_async` | `st_moe` | `lru` | Disabled | Disabled | `event_driven` |
| **5** | CSTP + Prefetch + LRU | `hardware_async` | `cstp` | `lru` | Disabled | Disabled | `event_driven` |
| **6** | ST-MoE + PABI Quarantine | `hardware_async` | `st_moe` | `pabi` | Enabled (20%) | Disabled | `event_driven` |
| **7** | CSTP + PABI Quarantine | `hardware_async` | `cstp` | `pabi` | Enabled (20%) | Disabled | `event_driven` |
| **8** | ST-MoE + PABI + Pinning | `hardware_async` | `st_moe` | `pabi` | Enabled (20%) | Enabled (K=4) | `event_driven` |
| **9** | CSTP + PABI + Pinning | `hardware_async` | `cstp` | `pabi` | Enabled (20%) | Enabled (K=4) | `event_driven` |

### 1.2 Exact YAML Configuration Specifications

#### Configuration 1: Baseline (No Predictor, No Prefetch, LRU)
```yaml
simulation:
  prefetching:
    enabled: false
    mode: "none"
    predictor:
      type: "none"
system:
  optimization:
    use_event_driven_dram: true
    bypass_offchip_execution_cache: false
```
*Environment Overrides*: `EVICTION_POLICY=lru`, `PABI_QUARANTINE_ENABLED=0`, `PABI_PINNING_ENABLED=0`, `USE_EVENT_DRIVEN_DRAM=1`.

#### Configuration 2: ST-MoE Prediction-Only
```yaml
simulation:
  prefetching:
    enabled: true
    mode: "prediction_only"
    predictor:
      type: "st_moe"
      confidence_threshold: 2
    candidate_selection:
      policy: "top_k"
      budget: 4
    scheduling:
      prediction_horizon: 2
```
*Environment Overrides*: `PREDICTION_OUTPUT_DIR=results/exp2_st_moe_pred_only`.

#### Configuration 3: CSTP Prediction-Only
```yaml
simulation:
  prefetching:
    enabled: true
    mode: "prediction_only"
    predictor:
      type: "cstp"
      alpha: 0.5
      smoothing_lambda: 1.0
    candidate_selection:
      policy: "top_k"
      budget: 4
    scheduling:
      prediction_horizon: 2
```
*Environment Overrides*: `PREDICTION_OUTPUT_DIR=results/exp3_cstp_pred_only`.

#### Configuration 4: ST-MoE with Prefetching and Standard LRU
```yaml
simulation:
  prefetching:
    enabled: true
    mode: "hardware_async"
    predictor:
      type: "st_moe"
      confidence_threshold: 2
    candidate_selection:
      policy: "top_k"
      budget: 4
    scheduling:
      prediction_horizon: 2
      demand_priority: true
    transfer:
      model: "event_driven"
      asynchronous: true
    eviction_policy: "lru"
system:
  optimization:
    use_event_driven_dram: true
```
*Environment Overrides*: `EVICTION_POLICY=lru`, `PABI_QUARANTINE_ENABLED=0`, `PABI_PINNING_ENABLED=0`, `USE_EVENT_DRIVEN_DRAM=1`.

#### Configuration 5: CSTP with Prefetching and Standard LRU
```yaml
simulation:
  prefetching:
    enabled: true
    mode: "hardware_async"
    predictor:
      type: "cstp"
      alpha: 0.5
      smoothing_lambda: 1.0
    candidate_selection:
      policy: "top_k"
      budget: 4
    scheduling:
      prediction_horizon: 2
      demand_priority: true
    transfer:
      model: "event_driven"
      asynchronous: true
    eviction_policy: "lru"
system:
  optimization:
    use_event_driven_dram: true
```
*Environment Overrides*: `EVICTION_POLICY=lru`, `PABI_QUARANTINE_ENABLED=0`, `PABI_PINNING_ENABLED=0`, `USE_EVENT_DRIVEN_DRAM=1`.

#### Configuration 6: ST-MoE with PABI Quarantine
```yaml
simulation:
  prefetching:
    enabled: true
    mode: "hardware_async"
    predictor:
      type: "st_moe"
      confidence_threshold: 2
    candidate_selection:
      policy: "top_k"
      budget: 4
    scheduling:
      prediction_horizon: 2
      demand_priority: true
    transfer:
      model: "event_driven"
      asynchronous: true
    eviction_policy: "pabi"
    pabi:
      quarantine_enabled: true
      speculative_fraction: 0.20
      pinning_enabled: false
system:
  optimization:
    use_event_driven_dram: true
```
*Environment Overrides*: `EVICTION_POLICY=pabi`, `PABI_QUARANTINE_ENABLED=1`, `PABI_PINNING_ENABLED=0`, `USE_EVENT_DRIVEN_DRAM=1`.

#### Configuration 7: CSTP with PABI Quarantine
```yaml
simulation:
  prefetching:
    enabled: true
    mode: "hardware_async"
    predictor:
      type: "cstp"
      alpha: 0.5
      smoothing_lambda: 1.0
    candidate_selection:
      policy: "top_k"
      budget: 4
    scheduling:
      prediction_horizon: 2
      demand_priority: true
    transfer:
      model: "event_driven"
      asynchronous: true
    eviction_policy: "pabi"
    pabi:
      quarantine_enabled: true
      speculative_fraction: 0.20
      pinning_enabled: false
system:
  optimization:
    use_event_driven_dram: true
```
*Environment Overrides*: `EVICTION_POLICY=pabi`, `PABI_QUARANTINE_ENABLED=1`, `PABI_PINNING_ENABLED=0`, `USE_EVENT_DRIVEN_DRAM=1`.

#### Configuration 8: ST-MoE with PABI Quarantine and Pinning
```yaml
simulation:
  prefetching:
    enabled: true
    mode: "hardware_async"
    predictor:
      type: "st_moe"
      confidence_threshold: 2
    candidate_selection:
      policy: "top_k"
      budget: 4
    scheduling:
      prediction_horizon: 2
      demand_priority: true
    transfer:
      model: "event_driven"
      asynchronous: true
    eviction_policy: "pabi"
    pabi:
      quarantine_enabled: true
      speculative_fraction: 0.20
      pinning_enabled: true
      max_pinned_experts: 4
      pin_selection_policy: "top_ranked"
      pin_expiration_policy: "stage_deadline"
system:
  optimization:
    use_event_driven_dram: true
```
*Environment Overrides*: `EVICTION_POLICY=pabi`, `PABI_QUARANTINE_ENABLED=1`, `PABI_PINNING_ENABLED=1`, `PABI_MAX_PINNED_EXPERTS=4`, `USE_EVENT_DRIVEN_DRAM=1`.

#### Configuration 9: CSTP with PABI Quarantine and Pinning
```yaml
simulation:
  prefetching:
    enabled: true
    mode: "hardware_async"
    predictor:
      type: "cstp"
      alpha: 0.5
      smoothing_lambda: 1.0
    candidate_selection:
      policy: "top_k"
      budget: 4
    scheduling:
      prediction_horizon: 2
      demand_priority: true
    transfer:
      model: "event_driven"
      asynchronous: true
    eviction_policy: "pabi"
    pabi:
      quarantine_enabled: true
      speculative_fraction: 0.20
      pinning_enabled: true
      max_pinned_experts: 4
      pin_selection_policy: "top_ranked"
      pin_expiration_policy: "stage_deadline"
system:
  optimization:
    use_event_driven_dram: true
```
*Environment Overrides*: `EVICTION_POLICY=pabi`, `PABI_QUARANTINE_ENABLED=1`, `PABI_PINNING_ENABLED=1`, `PABI_MAX_PINNED_EXPERTS=4`, `USE_EVENT_DRIVEN_DRAM=1`.

### 1.3 Accidental Feature Contamination Audit

We audited whether switching between configurations could inadvertently leave another feature active:

1. **Eviction Policy Switching**:
   - `Device::is_pinning_active()` in `src/hardware/device.cpp:397-405` strictly checks `eviction_policy == EvictionPolicy::PABI`.
   - If a YAML file sets `pinning_enabled: true` but `eviction_policy: lru`, `is_pinning_active()` returns `false`.
   - However, if the process inherits `PABI_PINNING_ENABLED=1` from an environment variable across shell sessions, any configuration with `eviction_policy: pabi` will silently activate pinning unless explicitly overridden with `PABI_PINNING_ENABLED=0`.
   - **Recommendation**: Test runner scripts must explicitly set or unset all environment overrides (`EVICTION_POLICY`, `PABI_QUARANTINE_ENABLED`, `PABI_PINNING_ENABLED`, `PABI_MAX_PINNED_EXPERTS`) before each run.
2. **Predictor Type and Global Toggle**:
   - `PrefetchConfig::validate()` in `src/prefetch/prefetch_config.cpp:229-235` throws `std::invalid_argument` if `enabled: false` but `predictor.type != none`.
   - This prevents accidental predictor instantiation when prefetching is disabled.
3. **Execution Cache Bypass Consistency**:
   - `SystemConfig::apply_environment_overrides()` in `src/hardware/hardware_config.h:157-169` checks `BYPASS_OFFCHIP_EXECUTION_CACHE`.
   - When running with `use_event_driven_dram: true`, `LinearImpl::run_linear` in `src/hardware/linear_impl.cpp:75` takes the event-driven path and does not call `issueRamulatorBatch()`, ensuring that stale Ramulator execution cache state cannot pollute event-driven timing.

---

## 2. Audit 2 — Trace Comparability

### 2.1 Available Real Routing Trace Inventory

The repository contains real routing traces in `results/endsem_phase_e/final_baseline/traces/`:

| Trace File Name | Sequences | Replay Tokens | MoE Layers | Top-K | Total Records | File Size |
|---|---|---|---|---|---|---|
| `ms_task12_t128_l2.csv` | 1 | 128 | 2 | 8 | 2,048 | 26 KB |
| `ma_p60_t128_l2.csv` | 1 | 128 | 2 | 8 | 2,048 | 26 KB |
| `ms_task12_t128_l75.csv` | 1 | 128 | 75 | 8 | 76,800 | 1,019 KB |
| `ma_b2_t128_l40.csv` | 2 | 128 | 40 | 8 | 81,920 | 1.1 MB |
| `ma_b4_t128_l40.csv` | 4 | 128 | 40 | 8 | 163,840 | 2.2 MB |
| `ms_task12_t512_l75.csv` | 1 | 512 | 75 | 8 | 307,200 | 4.2 MB |
| `ms_task12_t1024_l75.csv` | 1 | 1,024 | 75 | 8 | 614,400 | 8.5 MB |
| `ms_task12_t2048_l75.csv` | 1 | 2,048 | 75 | 8 | 1,228,800 | 18 MB |

### 2.2 Trace Ingestion & Synthetic Fallback Audit

- **Zero Silent Fallback**: In `Scheduler::loadRealTrace()` (`src/scheduler/scheduler.cpp:715-725`), the loader explicitly verifies that every `(sequence, token, layer)` tuple contains exactly `top_k` records. If any record is missing, it throws a fatal `std::runtime_error`:
  ```cpp
  throw std::runtime_error("[ERROR] Trace incomplete in " + trace_path + ... + ". Zero fallback to random allowed!");
  ```
- **Deterministic Replay**: In `Sequence::update_expert()` (`src/scheduler/sequence.cpp:211-235`), expert routing is read directly from `sequences_info[seq_id]->expert_list[table_idx]`. When `get_expert_from_list` is `true`, no random number generator is called.
- **Identical Events Across Predictors**: In `BatchedSequence::update_expert()` (`src/scheduler/sequence.cpp:204-266`), the exact same `request_id`, `token_id`, and `cur_layer` are passed to `on_stage_routing_start()` and `on_stage_routing_complete()`. Thus, ST-MoE, CSTP, ACSH, and Baseline consume identical events in identical order.

### 2.3 Scheduler Off-By-One & Sequence Popping Defect

In `src/scheduler/sequence.cpp:15`:
```cpp
total_len = input_len + output_len - 1;
```
In `src/scheduler/scheduler.cpp:356`:
```cpp
if (seq->current_len == seq->total_len) {
  batch->pop(seq);
}
```
**Forensic Consequence**:
- When a sequence runs in decode mode with `input_len` and generates decoding tokens, `current_len` increments by 1 on each step.
- When `step = output_len - 1`, `current_len` reaches `input_len + output_len - 1 == total_len`.
- The sequence is popped from `running_queue`.
- On the subsequent step (`step = output_len`), `running_queue` becomes empty, causing `scheduler->fillRunningQueue()` to pop a *new* sequence from `sequence_queue` (or spawn an uninitialized request). In trace `A1`, `request_id` abruptly changed to `2` at step 128, resetting relative token index to 0.
- **Required Protocol for Phase 4D.5B Experiments**:
  - To simulate $N$ tokens without triggering the off-by-one popping defect, configuration files must specify `output_len >= N + 1` (e.g. for an evaluation run of 10 tokens, set `input_len: 128`, `output_len: 128`, `iter: 10`).

### 2.4 Warm-Up and Cache Reset

- In `eval/test_deepseekv3_mixtral_style.cpp:331-332`:
  ```cpp
  device->cache_capacity_bytes = cache_capacity;
  device->reset_expert_cache();
  ```
  `reset_expert_cache()` resets all dual recency lists, byte accounting, in-flight registries, pin intents, and transfer engine queues.
- `warm_hbm_preload`: When enabled, preloads the working set into HBM before the decode loop starts (`test_deepseekv3_mixtral_style.cpp:395-415`). For cold-start comparison across configurations, `warm_hbm_preload` must be set to `false`.

---

## 3. Audit 3 — Metric Correctness

| Metric Name | Source Variable / Counter | Exact Location | Denominator | Granularity | Double-Count Risk | Cross-Config Validity |
|---|---|---|---|---|---|---|
| **Prediction Precision@K** | `m.precision` | `live_prediction_engine.cpp:307` | `predicted_candidates.size()` | Per-stage event | None (unique event_id) | **Valid** |
| **Prediction Recall@K** | `m.recall` | `live_prediction_engine.cpp:313` | `actual_target_experts.size()` | Per-stage event | None (unique event_id) | **Valid** |
| **Prefetch Coverage** | `useful_prefetch_hits / total_demands` | Derived from `device.h:73, 90` | `useful_hits + demand_misses` | Global / Per-token | None | **Valid** |
| **Timely Prefetches** | Prefetches completed before demand (`weight_memory_cost == 0`) | `linear_impl.cpp:96` | Total prefetch transfers | Per-expert transfer | None | **Valid** |
| **Late Prefetches** | In-flight attachments (`demand_inflight_attachments`) | `device.cpp:1012` | Total prefetch transfers | Per-expert transfer | None | **Valid** |
| **Unused Prefetches** | `speculative_evictions` + unpromoted speculative | `device.cpp:752, 793, 856` | Total prefetches inserted | Global | None | **Valid** |
| **Useful Prefetch Hits** | `pabi_metrics.useful_prefetch_hits` | `device.cpp:600, 646` | Total expert demands | Global / Cumulative | None | **Valid** |
| **HBM Hit Rate** | `1.0 - (demand_misses / total_demands)` | `linear_impl.cpp:66-67` | Total expert accesses | Per-step & Global | None | **Valid** |
| **Demand Misses** | `pabi_metrics.demand_insertions` | `device.cpp:697, 902` | N/A | Global / Cumulative | None | **Valid** |
| **DDR5 Bytes Transferred** | `metrics.total_bytes_transferred` | `ddr5_transfer_engine.cpp:211, 281` | N/A | Global / Cumulative | None | **Valid** |
| **Speculative Evictions** | `pabi_metrics.speculative_evictions` | `device.cpp:752, 793, 856` | N/A | Global / Cumulative | None | **Valid** |
| **Demand Evictions** | `pabi_metrics.demand_evictions` | `device.cpp:681, 881` | N/A | Global / Cumulative | None | **Valid** |
| **Pin Expirations** | `pabi_metrics.pin_expirations` | `device.cpp:528, 541, 1149` | N/A | Global / Cumulative | None | **Valid** |
| **Pinned-Capacity Rejections** | `speculative_reject_pinned_capacity` | `device.cpp:733` | N/A | Global / Cumulative | None | **Valid** |
| **Exposed Memory Stall** | `weight_memory_cost = max(0, tr->completion - start_time)` | `linear_impl.cpp:94` | N/A | Per-expert & Per-step | None (isolated start_time) | **CRITICAL: Only valid stall metric** |
| **Expert-Load Latency** | `tr->completion_time_ns - tr->arrival_time_ns` | `linear_impl.cpp:150-160` | N/A | Per-transfer | None | **Valid** |
| **Token Latency** | `stat.latency = step_device_time` | `test_deepseekv3_mixtral_style.cpp:467` | N/A | Per-token step | None | **Valid** |
| **End-to-End Latency** | `cumulative_device_time` | `test_deepseekv3_mixtral_style.cpp:460` | N/A | Cumulative run | None | **Valid** |
| **Host Execution Time** | `total_host_sec` | `test_deepseekv3_mixtral_style.cpp:557` | N/A | Whole run wall clock | None | **Valid (CPU host time)** |

### 3.1 Exposed Critical-Path Stall vs. Accumulated DDR5 Service Time

> [!CAUTION]
> **Metric Invalidation Warning**: `ddr5_transfer_engine->get_metrics().demand_service_time_ns` and `demand_queue_waiting_time_ns` record the internal duration of the DDR5 bus transfers.
> In an asynchronous system with prefetching, compute overlaps with prefetch transfers.
> If an expert transfer takes 3.901 ms, but completed 2.0 ms before the layer started execution, the exposed stall is **0.0 ms**.
> Using accumulated DDR5 service time instead of exposed critical-path stall (`weight_memory_cost` in `linear_impl.cpp:94`) would falsely report memory stall and mask prefetching benefits.
> All comparative analyses in Phase 4D.5B must strictly evaluate **`stat.latency`** and **`exec_status.memory_duration`**.

---

## 4. Audit 4 — Phase 4D.4 Pinning Correctness

A code-level verification of the pinning implementation in `src/hardware/device.cpp` revealed the following findings:

### 4.1 Multiple Valid Pin Intents for Same Expert
- In `select_pin_candidates_for_stage()` (`src/hardware/device.cpp:448-456`), candidates within the same target stage are deduplicated by `expert_id`. The candidate with highest rank / score wins.
- DeepSeek-V3 experts are indexed by `(layer_id, expert_id)`. Experts with the same `expert_id` in different layers represent physically distinct weight tensors and map to different keys `{layer_id, expert_id}` in `resident_experts` and `active_pin_intents`. No collision occurs.

### 4.2 Multi-Stage Expiration Isolation
- In `expire_pins_for_stage(int target_stage_id)` (`src/hardware/device.cpp:524, 537, 548`), expiration strictly filters by:
  ```cpp
  if (kv.second.pin_expiration_stage_id == target_stage_id)
  ```
  Expiring stage 1 leaves stage 2 pins untouched. Verified by Test 21 in `test_pabi_pinning.cpp`.

### 4.3 Transfer Completion at Deadline
- In `process_completed_prefetches()` (`src/hardware/device.cpp:1146`):
  ```cpp
  if (it->second.pin_expiration_stage_id < current_stage_id)
  ```
  If transfer completes when `pin_expiration_stage_id == current_stage_id`, it is accepted as pinned. If `pin_expiration_stage_id < current_stage_id`, the deadline has passed, so the expert is inserted into quarantine as unpinned.

### 4.4 Demand Promotion While Pin Active
- In `promote_speculative_to_demand()` (`src/hardware/device.cpp:573-585`), demanding a pinned speculative resident clears `is_pinned = false`, resets target/expiration stage IDs, increments `pins_cleared_on_promotion` and `pinned_expert_demand_hits`, removes the intent from `active_pin_intents`, and moves the expert to the MRU of `demand_lru_list`.

### 4.5 Demand Allocation Under 100% Pinned Speculative Saturation
- In `mark_expert_resident()` (`src/hardware/device.cpp:824-886`), cold demand misses skip all pinned entries in `speculative_lru_list`. If all speculative entries are pinned, eviction falls back to `demand_lru_list.front()`. Pinned speculative entries are protected against eviction.

### 4.6 Correctness Issues Identified in Audit 4

> [!WARNING]
> **Issue 4A: `device->current_stage_id` is never updated in production simulation.**
> - `set_current_stage(int stage)` is defined in `src/hardware/device.h:255`.
> - However, `set_current_stage()` is never invoked inside `src/module/linear.cpp` or `src/hardware/linear_impl.cpp`.
> - `current_stage_id` remains `0` indefinitely during production inference.
> - As a result, line 1146 (`it->second.pin_expiration_stage_id < current_stage_id`) can never detect an expired deadline for transfers completing in later stages.
> - **Fix required in Phase 4D.5B**: Add `device->set_current_stage(sequences_metadata->cur_layer)` in `src/module/linear.cpp:50`.

---

## 5. Audit 5 — Event-Driven DDR5 Model

### 5.1 Model Classification & Accuracy
- The DDR5 memory model implemented in `src/prefetch/ddr5_transfer_engine.cpp` is a **Ramulator2-calibrated discrete-event single-server queue abstraction**.
- It does **not** execute Ramulator2 cycle-by-cycle for every transfer during the simulation.
- It is calibrated directly from the Phase 4B/4B.1 empirical transfer campaign:
  - Bandwidth: $B_{\text{DDR5}} = 22.578\text{ GB/s}$ ($22.578\text{ bytes/ns}$).
  - Expert footprint: $88,080,384\text{ bytes}$ (~88.08 MB FP16).
  - Single-expert transfer duration: $\tau_{\text{single}} = 3,901,159.7\text{ ns} \approx 3.901\text{ ms}$.
- It models strict software-level demand priority and work-preserving preemption of speculative prefetch transfers.

### 5.2 Simulated Hardware Clock vs. CPU Host Time
- **Simulated Hardware Clock (`time_ns`, `status.device_time`)**:
  - Incremented by exact compute durations (FLOPs / peak throughput) and memory stalls (`tr->completion_time_ns - start_time`).
  - Completely decoupled from host CPU speed.
- **CPU Host Time (`std::chrono::high_resolution_clock`)**:
  - Measures wall-clock execution time of the simulator on the host workstation.

### 5.3 Overlap Scheduling Boundaries
- Compute/transfer overlap occurs strictly when an asynchronous prefetch is submitted at a scheduling boundary (e.g. stage routing start) and executes concurrently while preceding modules (`Attention`, `Linear`, `LayerNorm`) advance `device->status.device_time`.
- If the prefetch completes before the MoE stage reaches the expert (`completion_time_ns <= start_time`), overlap is 100% and memory stall is 0.
- If prefetch transfer is still in progress when the stage reaches the expert, the engine promotes the transfer to demand, and the remaining transfer latency is exposed as memory stall.

---

## 6. Audit 6 — Experiment Feasibility

### 6.1 Smoke-Test Workload Proposal
- **Purpose**: Verify end-to-end execution, CSV output generation, and zero runtime crashes in < 15 seconds.
- **Trace**: `results/endsem_phase_e/final_baseline/traces/ms_task12_t128_l2.csv` (1 sequence, 2 MoE layers, top-8).
- **Parameters**: `iter: 4`, `input_len: 128`, `output_len: 128`, `max_batch_size: 1`.
- **Expected Host Runtime**: ~3 seconds per configuration (~27 seconds for all 9 configurations).

### 6.2 Bounded Evaluation Workload Proposal
- **Purpose**: Scientifically valid comparative evaluation across all 9 configurations under realistic 75-layer MoE cache pressure.
- **Trace**: `results/endsem_phase_e/final_baseline/traces/ms_task12_t128_l75.csv` (1 sequence, 75 MoE layers, top-8).
- **Parameters**: `iter: 10`, `input_len: 128`, `output_len: 128`, `max_batch_size: 1`.
- **Cache Budget**: 80 GB ($C_{\text{HBM}} = 80,000,000,000$ bytes).
- **Total Expert Working Set per Token**: $75 \text{ layers} \times 8 \text{ experts} \times 88.08 \text{ MB} = 52.85 \text{ GB}$.
- **Expected Host Runtime**: ~35 seconds per configuration (~5.5 minutes for all 9 configurations).
- **Disk Space**: ~25 MB total for all 9 output directories.

### 6.3 Larger Evaluation Workload (Post-Validation)
- **Trace**: `results/endsem_phase_e/final_baseline/traces/ms_task12_t128_l75.csv`.
- **Parameters**: `iter: 64`, `input_len: 128`, `output_len: 128`, `max_batch_size: 1`.
- **Expected Host Runtime**: ~3.5 minutes per configuration (~32 minutes total).

### 6.4 Deterministic Experiment Naming Convention

All Phase 4D.5B runs must follow the standardized naming convention:
```
results/phase_4d5b/<workload>_<config_id>_<predictor>_<policy>/
```
Examples:
- `results/phase_4d5b/dsV3_t10_c01_baseline_lru/`
- `results/phase_4d5b/dsV3_t10_c02_stmoe_predonly/`
- `results/phase_4d5b/dsV3_t10_c04_stmoe_prefetch_lru/`
- `results/phase_4d5b/dsV3_t10_c06_stmoe_pabi_quarantine/`
- `results/phase_4d5b/dsV3_t10_c08_stmoe_pabi_pinning/`
- `results/phase_4d5b/dsV3_t10_c09_cstp_pabi_pinning/`

---

## 7. Known Blockers for End-to-End Simulation

Before Phase 4D.5B can execute Configurations 4 through 9, the following 4 wiring points must be resolved:

### Blocker 1: `Scheduler::init_prediction_engine()` restricts mode to `PREDICTION_ONLY`
- **Location**: `src/scheduler/scheduler.cpp:27-28`
- **Code**:
  ```cpp
  if (system_config.prefetch_config.enabled && 
      system_config.prefetch_config.mode == PrefetchMode::PREDICTION_ONLY) {
  ```
- **Consequence**: When YAML sets `mode: "hardware_async"`, `init_prediction_engine()` evaluates to `false` and leaves `prediction_engine = nullptr`. No predictions are generated, candidate queue is not created, and no prefetches can be dispatched.
- **Required Fix**: Change condition to:
  ```cpp
  if (system_config.prefetch_config.enabled && 
      (system_config.prefetch_config.mode == PrefetchMode::PREDICTION_ONLY ||
       system_config.prefetch_config.mode == PrefetchMode::HARDWARE_ASYNC))
  ```

### Blocker 2: Missing candidate dispatch hook in simulation loop
- **Location**: `src/scheduler/sequence.cpp:204` or `src/module/linear.cpp:52`
- **Consequence**: `LivePredictionEngine::dispatch_candidates(device, current_time)` is implemented and tested in unit tests, but is not called inside `BatchedSequence::update_expert()` or `Linear::forward()`. Candidates enqueued in `candidate_queue_` remain dormant and are never submitted to `Device::submit_prefetch_transfer()`.
- **Required Fix**: In `src/scheduler/sequence.cpp:207`, invoke candidate dispatch to `device` after `on_stage_routing_start()`.

### Blocker 3: Missing pin expiration hook in stage completion loop
- **Location**: `src/scheduler/sequence.cpp:266` or `src/module/linear.cpp:60`
- **Consequence**: `LivePredictionEngine::expire_pins_for_stage(device, cur_layer)` is never invoked as layers finish. Pins would never expire at stage deadlines.
- **Required Fix**: In `src/scheduler/sequence.cpp:266`, invoke `scheduler->prediction_engine->expire_pins_for_stage(device, cur_layer)` upon stage completion.

### Blocker 4: Missing `Device::set_current_stage()` update
- **Location**: `src/module/linear.cpp:50`
- **Consequence**: `device->current_stage_id` is never set to `cur_layer`, preventing late-transfer deadline comparisons.
- **Required Fix**: In `src/module/linear.cpp:50`, call `device->set_current_stage(layer_id)`.

---

## 8. Recommended Bounded Experiment Plan

Once the 4 wiring points in Section 7 are implemented in Phase 4D.5B, the experiment plan should proceed in two strictly bounded steps:

### Step 1: Smoke-Test Validation (2 Layers, 4 Tokens)
Run all 9 configurations on `ms_task12_t128_l2.csv` for 4 tokens to verify:
1. Candidate dispatch produces > 0 `candidates_dispatched`.
2. Prefetches complete and enter `speculative_lru_list`.
3. Pinned entries receive pins and expire at stage deadline.
4. Summary CSVs, traces, and metrics are written cleanly.

### Step 2: Bounded Production Run (75 Layers, 10 Tokens)
Run all 9 configurations on `ms_task12_t128_l75.csv` for 10 tokens:
- Evaluate Precision@4, Recall@4, Useful Prefetch Hits, HBM Hit Rate, Exposed Memory Stall Duration, and Per-Token Latency.
- Compare LRU vs. PABI Quarantine vs. PABI Pinning.

---

## 9. Exact Commands for Running Experiments (Do Not Execute Yet)

Below are the exact commands prepared for Phase 4D.5B execution:

```bash
# 1. Build harness
cd /home/mohman/ca_project/LLMSimulator && cmake -B build && cmake --build build -j$(nproc)

# 2. Configuration 1: Baseline
./build/run_deepseekv3_mixtral_style configs/phase_4d5b/c01_baseline_lru.yaml

# 3. Configuration 2: ST-MoE Prediction-Only
./build/run_deepseekv3_mixtral_style configs/phase_4d5b/c02_stmoe_predonly.yaml

# 4. Configuration 3: CSTP Prediction-Only
./build/run_deepseekv3_mixtral_style configs/phase_4d5b/c03_cstp_predonly.yaml

# 5. Configuration 4: ST-MoE + Prefetch + LRU
./build/run_deepseekv3_mixtral_style configs/phase_4d5b/c04_stmoe_prefetch_lru.yaml

# 6. Configuration 5: CSTP + Prefetch + LRU
./build/run_deepseekv3_mixtral_style configs/phase_4d5b/c05_cstp_prefetch_lru.yaml

# 7. Configuration 6: ST-MoE + PABI Quarantine
./build/run_deepseekv3_mixtral_style configs/phase_4d5b/c06_stmoe_pabi_quarantine.yaml

# 8. Configuration 7: CSTP + PABI Quarantine
./build/run_deepseekv3_mixtral_style configs/phase_4d5b/c07_cstp_pabi_quarantine.yaml

# 9. Configuration 8: ST-MoE + PABI + Pinning
./build/run_deepseekv3_mixtral_style configs/phase_4d5b/c08_stmoe_pabi_pinning.yaml

# 10. Configuration 9: CSTP + PABI + Pinning
./build/run_deepseekv3_mixtral_style configs/phase_4d5b/c09_cstp_pabi_pinning.yaml
```

---

## 10. Final Decision for Phase 4D.5B

### DECISION: **NO-GO FOR DIRECT EXPERIMENT LAUNCH; GO FOR PHASE 4D.5B INTEGRATION WIRING FIRST**

1. **DO NOT** execute large simulation runs immediately under Configurations 4–9 because prefetch dispatch, stage expiration, and `current_stage_id` are not yet connected to the main simulation loop.
2. **PHASE 4D.5B MUST FIRST**:
   - Update `Scheduler::init_prediction_engine()` to support `PrefetchMode::HARDWARE_ASYNC`.
   - Wire `dispatch_candidates()` in `Sequence::update_expert()`.
   - Wire `expire_pins_for_stage()` in `Sequence::update_expert()`.
   - Wire `set_current_stage()` in `Linear::forward()`.
   - Run the 4-token smoke test across all 9 configurations to confirm dispatch and pinning activity before initiating the 10-token evaluation campaign.

