# Phase 4D.1 Report: Production Integration of DDR5TransferEngine (Demand-Only)

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Authoritative Baseline Commit:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Phase:** 4D.1 — Production Integration of DDR5TransferEngine (Demand-Only)  
**Status:** COMPLETE (10/10 Correctness Tests Passed; 101/101 Global Regression Tests Passed)  
**Ternary Operator Policy:** Strictly Zero Ternary Operators in any C++, Python, or documentation file.

---

## 1. Executive Summary

Phase 4D.1 integrates the calibrated, event-driven `DDR5TransferEngine` into the production demand-loading path of LLMSimulator.

In prior phases, demand misses on off-chip DRAM directly called `issueRamulatorBatch(..., MemoryTarget::OFFCHIP_DRAM)`, which invoked the blocking Ramulator2 memory simulator. In Phase 4D.1, when `use_event_driven_dram` is enabled, actual demand misses bypass Ramulator off-chip calls and are serviced by the discrete-event `DDR5TransferEngine` using the calibrated DDR5 transfer rate of $22.578\text{ B/ns}$ ($22.578\text{ GB/s}$) and single-server serialized channel semantics.

Strict phase scoping has been enforced:
- **Demand-Only:** Speculative prefetch candidate dispatch, PABI dual-list buffering, quarantine, and promotion are deferred to subsequent subphases.
- **Baseline Preserved:** Configurations without `use_event_driven_dram: true` continue executing the original blocking Ramulator2 demand path.
- **Timing Invariants:** Single expert demand transfer latency is verified at $3.90116\text{ ms}$, two serialized demands at $3.90116\text{ ms}$ and $7.80232\text{ ms}$, memory duration is charged exactly once, and zero double-charging occurs.

---

## 2. Exact Files Modified and Why

| File | Type | Reason for Modification |
| :--- | :--- | :--- |
| `src/prefetch/ddr5_transfer_engine.h` | Header | Added `submit_demand_id` (returns allocated `transfer_id`) and `run_until_transfer_completed(transfer_id)`. |
| `src/prefetch/ddr5_transfer_engine.cpp` | Source | Implemented `submit_demand_id` and discrete-event progression in `run_until_transfer_completed`. |
| `src/hardware/hardware_config.h` | Header | Added `use_event_driven_dram` (default `false`) and `ddr5_bandwidth_gbps` (default `22.578`) to `SystemConfig` constructor, members, and `apply_environment_overrides()`. |
| `src/hardware/device.h` | Header | Added per-device `ddr5_transfer_engine` pointer, `inflight_experts` map, and transfer management method declarations (`submit_demand_transfer`, `wait_for_demand_transfer`, `is_expert_inflight`, `clear_inflight_expert`). Removed pre-existing ternary in `cache_capacity_bytes`. |
| `src/hardware/device.cpp` | Source | Instantiated `ddr5_transfer_engine` in `Device` constructor when enabled; implemented transfer management methods; replaced pre-existing ternary operators in `run_ramulator_batch` with standard `if/else`. |
| `src/module/linear.cpp` | Source | Updated off-chip DRAM activation guard to check `(!device->config.offchip_dram_cfg_path.empty() \|\| device->config.use_event_driven_dram)`. |
| `src/hardware/linear_impl.cpp` | Source | Diverted off-chip demand misses to `device->submit_demand_transfer` and `device->wait_for_demand_transfer`, bypassing `issueRamulatorBatch(..., OFFCHIP_DRAM)`. Replaced legacy ternary operators in logging logic. |
| `eval/CMakeLists.txt` | Build | Registered `test_demand_transfer_integration` executable target. |
| `eval/test_demand_transfer_integration.cpp` | Test | Created dedicated 10-test correctness suite covering single/multi demand misses, serialization, failure handling, LRU preservation, device isolation, clock sync, and baseline preservation. |

---

## 3. Production Call Flow: Before vs. After

### Before (Authoritative Baseline)
```
Linear::forward()
 ├── Check is_expert_resident(layer_id, expert_id)
 └── If miss and offchip_dram_cfg_path present:
      ├── weight->weight_target = MemoryTarget::OFFCHIP_DRAM
      ├── weight->load_sisters = true
      └── Device::execution(LinearExecutionGPU)
           ├── issueRamulator(input, MemoryTarget::HBM)
           ├── issueRamulatorBatch(batch_tensors, MemoryTarget::OFFCHIP_DRAM)  <-- Ramulator2 B
           │    └── weight_memory_cost = weight_status.memory_duration (~3.901 ms)
           ├── mark_expert_resident(layer_id, expert_id, total_expert_size)
           ├── issueRamulator(output, MemoryTarget::HBM)
           └── exec_status.total_duration = max(compute_duration, memory_duration)
```

### After (Phase 4D.1 Demand-Only Event-Driven Integration)
```
Linear::forward()
 ├── Check is_expert_resident(layer_id, expert_id)
 └── If miss and (!offchip_dram_cfg_path.empty() || use_event_driven_dram):
      ├── weight->weight_target = MemoryTarget::OFFCHIP_DRAM
      ├── weight->load_sisters = true
      └── Device::execution(LinearExecutionGPU)
           ├── issueRamulator(input, MemoryTarget::HBM)
           ├── Check: is_demand_miss && device->config.use_event_driven_dram
           │    ├── TRUE (Event-driven mode):
           │    │    ├── uint64_t tid = device->submit_demand_transfer(layer, expert, start_time, total_expert_size)
           │    │    ├── bool ok = device->wait_for_demand_transfer(tid)
           │    │    ├── weight_memory_cost = tr->completion_time_ns - start_time (~3.90116 ms)
           │    │    ├── mark_expert_resident(layer, expert, total_expert_size)
           │    │    ├── clear_inflight_expert(layer, expert)
           │    │    └── Bypasses issueRamulatorBatch(..., OFFCHIP_DRAM) completely!
           │    └── FALSE (Baseline mode):
           │         ├── issueRamulatorBatch(batch_tensors, OFFCHIP_DRAM)
           │         └── mark_expert_resident(layer, expert, total_expert_size)
           ├── issueRamulator(output, MemoryTarget::HBM)
           └── exec_status.total_duration = max(compute_duration, memory_duration)
```

---

## 4. Exact New Function Signatures and Responsibilities

### `DDR5TransferEngine` Extensions
- `uint64_t submit_demand_id(int request_id, int layer_id, int expert_id, double current_time_ns, uint64_t bytes = DEFAULT_EXPERT_BYTES, uint64_t transfer_id = 0);`
  - Submits a demand transfer to the discrete-event queue at timestamp `current_time_ns`.
  - Automatically assigns and returns the unique `transfer_id` (or returns 0 on failure/rejection).
- `bool run_until_transfer_completed(uint64_t transfer_id);`
  - Steps the discrete-event simulation queue (`step()`) until the transfer with ID `transfer_id` transitions to `COMPLETED` or `CANCELLED`.
  - Returns `true` if completed successfully, `false` if cancelled, missing, or stalled.

### `Device` Extensions
- `uint64_t submit_demand_transfer(int layer_id, int expert_id, double current_time_ns, uint64_t bytes = DDR5TransferEngine::DEFAULT_EXPERT_BYTES);`
  - Checks `inflight_experts` for `{layer_id, expert_id}` to prevent duplicate simultaneous submissions.
  - Submits transfer to `ddr5_transfer_engine` and stores `transfer_id` in `inflight_experts`.
- `bool wait_for_demand_transfer(uint64_t transfer_id);`
  - Invokes `ddr5_transfer_engine->run_until_transfer_completed(transfer_id)`.
- `bool is_expert_inflight(int layer_id, int expert_id) const;`
  - Queries whether `{layer_id, expert_id}` currently has an active or queued transfer.
- `void clear_inflight_expert(int layer_id, int expert_id);`
  - Erases `{layer_id, expert_id}` from `inflight_experts` upon transfer completion or failure.

---

## 5. Timeline Synchronization and Determinism

Two timelines exist in LLMSimulator:
1. `Device::status.device_time`: The GPU device simulated clock (nanoseconds).
2. `DDR5TransferEngine::current_time_ns_`: The discrete-event DDR5 channel clock (nanoseconds).

The coordination rules enforced in Phase 4D.1 are:
1. **Monotonic Submission:** Demand transfers are submitted at `start_time = device->get_time()`. Inside `DDR5TransferEngine::submit_transfer`, if `current_time_ns > current_time_ns_`, the engine advances to `current_time_ns`.
2. **Channel Progression:** Calling `wait_for_demand_transfer(tid)` runs `step()` until `tr->completion_time_ns`. At that point, `ddr5_transfer_engine->get_current_time_ns() >= tr->completion_time_ns`.
3. **Duration Accounting:** The latency `weight_memory_cost = tr->completion_time_ns - start_time` is accumulated into `exec_status.memory_duration`.
4. **Device Clock Advance:** `LinearExecutionGPU` computes `exec_status.total_duration = std::max(exec_status.compute_duration, exec_status.memory_duration)`. When `ModuleGraph::execution()` finishes the layer, it increments `device->status.device_time += exec_status.total_duration`.
5. **No Regressive Clocks:** Because `exec_status.total_duration >= weight_memory_cost`, the updated `device_time >= tr->completion_time_ns`. Consequently, future transfers arrive at $t \ge \text{current\_time\_ns\_}$, ensuring neither clock ever moves backward.

---

## 6. Prevention of Double-Charging

Double-charging is strictly prevented through three structural safeguards:
1. **Branch Exclusion:** When `is_demand_miss && device->config.use_event_driven_dram` is true, the call to `issueRamulatorBatch(..., MemoryTarget::OFFCHIP_DRAM)` is bypassed entirely. No Ramulator request is constructed, submitted, or accounted for off-chip DRAM.
2. **Hit Path Isolation:** When an expert is resident in HBM (`weight->weight_target == MemoryTarget::HBM`), `is_demand_miss` is false. The event-driven engine is not called; instead, standard HBM access latency is measured via `issueRamulatorBatch(..., MemoryTarget::HBM)`.
3. **Exact Duration Accumulation:** `weight_status.memory_duration = weight_memory_cost` is added to `exec_status.memory_duration` exactly once via `exec_status += weight_status`.

---

## 7. Per-Device Isolation

Multi-device MoE configurations (e.g. 8 devices on an A100/H100 node) have independent memory controllers and DRAM channels.
- `DDR5TransferEngine` is owned per-instance by `Device` (`std::shared_ptr<DDR5TransferEngine> ddr5_transfer_engine`).
- Device 0 and Device 1 each instantiate their own transfer engine.
- Simultaneous demand misses on Device 0 and Device 1 execute concurrently without cross-device head-of-line blocking (validated in Test 6).

---

## 8. Baseline Preservation

The original Ramulator baseline mode is preserved:
- `SystemConfig::use_event_driven_dram` defaults to `false`.
- Existing YAML configs that do not set `use_event_driven_dram: true` continue executing the baseline blocking Ramulator2 demand path.
- Environment variable `USE_EVENT_DRIVEN_DRAM=1` enables event-driven mode dynamically without altering existing YAML configurations.
- Environment variable `DDR5_BANDWIDTH_GBPS` allows configuring the calibrated DDR5 transfer rate (defaults to 22.578).

---

## 9. Correctness Test Results

The dedicated integration test suite `eval/test_demand_transfer_integration.cpp` was compiled and executed. All 10 tests passed:

```text
======================================================================
PHASE 4D.1: Production Integration of DDR5TransferEngine (Demand-Only)
Calibrated DDR5 Bandwidth: 22.578 B/ns | Single-Server Event Queue
======================================================================

[RUNNING] Test 1: Single demand miss completes and marks expert resident...
[PASS] Test 1: Single demand miss completes and marks expert resident (3.90116 ms latency)
[RUNNING] Test 2: Repeated demand for resident expert (HBM hit, no transfer)...
[PASS] Test 2: Repeated demand correctly hits HBM without second transfer
[RUNNING] Test 3: Two demand misses serialized FIFO (3.90116 ms and 7.80232 ms)...
[PASS] Test 3: Serialized FIFO completions verified (T1=3.90116 ms, T2=7.80232 ms)
[RUNNING] Test 4: Transfer failure handling (expert not marked resident)...
[ERROR][DDR5TransferEngine] Rejection: Zero-byte transfer submission attempted.
[PASS] Test 4: Transfer failure correctly prevents resident insertion
[RUNNING] Test 5: HBM capacity eviction preservation (LRU policy verified)...
[PASS] Test 5: Existing LRU eviction behavior strictly preserved
[RUNNING] Test 6: Per-device isolation (independent DDR5 transfer engines)...
[PASS] Test 6: Independent devices maintain isolated transfer channels
[RUNNING] Test 7: Clock synchronization between device time and engine time...
[PASS] Test 7: Device time and engine clock advance consistently
[RUNNING] Test 8: No double-charging verification...
[PASS] Test 8: Transfer duration contributes exactly once to execution duration
[RUNNING] Test 9: Original Ramulator mode preservation...
[PASS] Test 9: Original blocking Ramulator path remains default and preserved
[RUNNING] Test 10: Event-driven mode verified (Ramulator bypassed on miss)...
[PASS] Test 10: Event-driven engine successfully bypasses Ramulator on miss

======================================================================
TEST SUITE SUMMARY: 10 PASSED, 0 FAILED
======================================================================
```

### Global Regression Test Summary
All 101 tests across the entire repository passed with zero failures:
- Phase 1: 17/17 passed
- Phase 2A: 40/40 passed
- Phase 2C: 2/2 passed
- Phase 3: 20/20 passed
- Phase 4C: 12/12 passed
- Phase 4D.1: 10/10 passed
- **Total: 101/101 PASSED**

---

## 10. Quantitative Timing Validation

Using the calibrated Phase 4B reference values:
- Expert footprint: $88,080,384\text{ bytes}$ (bundle of W1, W2, W3).
- DDR5 Bandwidth: $22.578\text{ B/ns}$ ($22.578\text{ GB/s}$).

$$\text{Theoretical Latency} = \frac{88,080,384\text{ B}}{22.578\text{ B/ns}} = 3,901,159.713\text{ ns} \approx 3.90116\text{ ms}$$

### Measured Latency vs. Expected Values

| Scenario | Transfer ID | Start Time (ns) | Expected Completion (ns) | Measured Completion (ns) | Delta (ns) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Single Demand Miss (Test 1) | 1 | $1,000,000.0$ | $4,901,159.713$ | $4,901,159.713$ | $< 0.001$ | **MATCH** |
| Serialized Demand 1 (Test 3) | 1 | $0.0$ | $3,901,159.713$ | $3,901,159.713$ | $< 0.001$ | **MATCH** |
| Serialized Demand 2 (Test 3) | 2 | $0.0$ | $7,802,319.426$ | $7,802,319.426$ | $< 0.001$ | **MATCH** |
| Multi-Device Dev 0 (Test 6) | 1 | $0.0$ | $3,901,159.713$ | $3,901,159.713$ | $< 0.001$ | **MATCH** |
| Multi-Device Dev 1 (Test 6) | 1 | $0.0$ | $3,901,159.713$ | $3,901,159.713$ | $< 0.001$ | **MATCH** |

Timing precision is exact to within floating-point epsilon ($< 1\text{ ps}$ numerical deviation).

---

## 11. Explicitly Deferred Work (Boundary Constraints)

As mandated by Phase 4D.1 instructions, the following items are **strictly deferred**:
1. Candidate queue dispatch (`PrefetchCandidateQueue::select_ready_candidates()`).
2. Speculative prefetch transfer submission to `DDR5TransferEngine`.
3. PABI quarantine buffer implementation.
4. Speculative-to-demand promotion.
5. Dual-list LRU policy.
6. Predictor score-based pinning / Belády approximation.
7. Prefetch cancellation on misprediction.

---

## 12. Ternary Operator Compliance Audit

A strict Python script was executed across all modified and newly created files in Phase 4D.1.
- Total ternary operators found: **0**
- All pre-existing ternary operators in modified sections of `device.h`, `device.cpp`, and `linear_impl.cpp` were refactored into explicit `if/else` control blocks.

---

## 13. Conclusion and Phase 4D.2 Readiness

Phase 4D.1 is complete. The production demand-loading path now reliably routes off-chip misses through the calibrated discrete-event DDR5 transfer engine.

**Recommendation:** Proceed to Phase 4D.2 (Demand/Prefetch Coordination and In-Flight Expert Handling) upon user approval.
