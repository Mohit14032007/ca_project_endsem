# Phase 4A Call Graph Documentation: Asynchronous Prefetch Transfer Engine

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Baseline Git Commit:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Date:** November 2, 2026  
**Status:** **READ-ONLY AUDIT COMPLETE (Zero Code Modifications)**

---

## 1. Overview

This document provides exact, function-level architectural call graphs for:
1. **The current synchronous demand-loading path** in LLMSimulator.
2. **The proposed asynchronous prefetch transfer path** for Phase 4.
3. **Every new integration point** required to support asynchronous prefetching.
4. **Existing critical functions that must remain completely unchanged** to guarantee baseline preservation.

---

## 2. Current Synchronous Demand-Loading Call Graph

The diagram below maps the exact execution path of an expert during sequential decoding in LLMSimulator.

```mermaid
sequenceDiagram
    autonumber
    participant Loop as Sequential Decoding Loop (test_deepseekv3_mixtral_style.cpp:427)
    participant Cluster as Cluster::run() (src/hardware/cluster.cpp:1101)
    participant Route as Route::forward() (src/module/route.cpp:148)
    participant Seq as BatchedSequence::update_expert() (src/scheduler/sequence.cpp:153)
    participant Linear as Linear::forward() (src/module/linear.cpp:21)
    participant Dev as Device (src/hardware/device.cpp)
    participant Exec as LinearExecutionGPU() (src/hardware/linear_impl.cpp:16)
    participant RamulBatch as issueRamulatorBatch() (src/hardware/layer_impl.cpp:57)
    participant DRAM as DRAMInterface (src/dram/dram_interface.cpp:36)

    Loop->>Cluster: cluster->run(metadata)
    Cluster->>Route: Route::forward(input, metadata)
    Route->>Seq: update_expert(num_expert, top_k, true)
    Note over Seq: Router assigns actual_target_experts for cur_layer
    Note over Seq: Predictions generated & enqueued (Phase 3)
    Seq-->>Route: num_token_in_expert updated
    Route-->>Cluster: Route outputs returned
    
    Cluster->>Linear: Linear::forward(input, metadata)
    Note over Linear: cur_layer = metadata->cur_layer
    Linear->>Dev: is_expert_resident(layer_id, expert_id)
    
    alt Expert is Resident (HBM Cache HIT)
        Dev-->>Linear: true (Moved to MRU; "HIT" written to expert_cache.csv)
        Linear->>Linear: A->weight_target = MemoryTarget::HBM
    else Expert is Absent (HBM Cache MISS)
        Dev-->>Linear: false ("MISS" written to expert_cache.csv)
        Linear->>Linear: A->weight_target = MemoryTarget::OFFCHIP_DRAM
        Linear->>Linear: A->load_sisters = true
    end
    
    Linear->>Dev: execution(LayerType::LINEAR, tensor_list, ...)
    Dev->>Exec: LinearExecutionGPU(device, input, weight, output, use_ramulator)
    
    alt weight_target == MemoryTarget::OFFCHIP_DRAM (Demand Stall)
        Exec->>RamulBatch: issueRamulatorBatch(device, ..., batch_tensors, OFFCHIP_DRAM)
        RamulBatch->>DRAM: HandleRequest(requests, 0)
        Note over DRAM: Blocks in DRAMInterface::run() until all requests finish
        DRAM-->>RamulBatch: duration (~3.90 ms)
        RamulBatch-->>Exec: ExecStatus (memory_duration = 3.90 ms)
        Exec->>Dev: mark_expert_resident(layer_id, expert_id, total_size)
        Note over Dev: Evicts LRU experts if full; appends INSERT to expert_cache.csv
        Exec->>Exec: Log to expert_load.csv
        Note over Exec: Stalls simulated clock: cumulative_device_time += 3.90 ms
    else weight_target == MemoryTarget::HBM (Hit Path)
        Exec->>RamulBatch: issueRamulatorBatch(device, ..., batch_tensors, HBM)
        RamulBatch-->>Exec: ExecStatus (memory_duration = 0 ms / cached)
    end
    
    Exec-->>Linear: ExecStatus returned
    Linear-->>Cluster: Layer execution complete
```

### Exact Source File and Line References (Synchronous Path)

1. `eval/test_deepseekv3_mixtral_style.cpp:444`:  
   Calls `cluster->run(metadata)`.
2. `src/hardware/cluster.cpp:1101-1109`:  
   `Cluster::run()` executes each graph node in sequence.
3. `src/module/route.cpp:160-163`:  
   `Route::forward()` calls `scheduler->running_queue[i]->update_expert(num_expert, top_k, true)`.
4. `src/scheduler/sequence.cpp:204-207`:  
   Calls `scheduler->prediction_engine->on_stage_routing_start(request_id, token_id, cur_layer)`.
5. `src/prefetch/live_prediction_engine.cpp:112-120`:  
   Enqueues candidates via `candidate_queue_->enqueue_candidates(...)`.
6. `src/scheduler/sequence.cpp:226-234`:  
   Authoritative router determines actual targets and logs to `expert_routing.csv`.
7. `src/scheduler/sequence.cpp:263-266`:  
   Calls `scheduler->prediction_engine->on_stage_routing_complete(...)` to reconcile stage.
8. `src/module/linear.cpp:48-57`:  
   Checks `if (A->is_expert_weight && input->getSize() > 0)` and queries `device->is_expert_resident(layer_id, A->expert_id)`.
9. `src/hardware/device.cpp:272-301`:  
   `Device::is_expert_resident(layer_id, expert_id)` mutates `lru_expert_list` and logs to `expert_cache.csv`.
10. `src/hardware/linear_impl.cpp:58-71`:  
    `LinearExecutionGPU()` invokes `issueRamulatorBatch(...)`.
11. `src/hardware/layer_impl.cpp:57-114`:  
    `issueRamulatorBatch()` checks `execution_cache` and calls `device->run_ramulator_batch(requests, target)`.
12. `src/hardware/device.cpp:218-224`:  
    `Device::run_ramulator_batch()` routes to `offchip_dram_interface->HandleRequest(requests, 0)`.
13. `src/dram/dram_interface.cpp:36-79`:  
    `DRAMInterface::HandleRequest()` calls `run()`.
14. `src/dram/dram_interface.cpp:125-132`:  
    `DRAMInterface::run()` blocks in a tight loop until `memory_system->is_finished() == true`.
15. `src/hardware/device.cpp:303-335`:  
    `Device::mark_expert_resident()` evicts LRU experts and inserts the new expert.

---

## 3. Proposed Asynchronous Prefetch Path (Phase 4)

The diagram below details the proposed asynchronous transfer execution path.

```mermaid
sequenceDiagram
    autonumber
    participant Route as Route::forward()
    participant Queue as PrefetchCandidateQueue
    participant Engine as AsyncTransferEngine (Phase 4)
    participant Linear as Linear::forward()
    participant Dev as Device (with has_expert_resident)
    participant Exec as LinearExecutionGPU()

    Note over Route: Stage S Execution Begins
    Route->>Queue: Enqueue predictions for Stage (S + H)
    Route->>Engine: poll_and_dispatch_transfers(current_time)
    
    rect rgb(240, 248, 255)
        Note over Engine,Queue: Prefetch Dispatch Pipeline
        Engine->>Queue: select_ready_candidates(available_slots)
        Queue-->>Engine: candidates [e.g. Expert 15 for Stage S+1]
        loop For each candidate
            Engine->>Dev: has_expert_resident(target_stage, expert_id) [READ-ONLY QUERY]
            alt Already in HBM
                Engine->>Engine: Suppress transfer (redundant)
            else Not in HBM and Not in Flight
                Engine->>Engine: Register in_flight_transfers_.insert({stage, expert})
                Engine->>Engine: Submit asynchronous transfer (fluid-flow bandwidth model)
                Engine->>Queue: mark_candidate_dispatched(cand_id)
            end
        end
    end

    Note over Route,Linear: Stage S Compute & Memory Execution Runs
    Route-->>Linear: Expert Linear Modules Dispatched
    
    rect rgb(255, 250, 240)
        Note over Linear,Exec: Demand Stage Routing (Stage S)
        Linear->>Dev: is_expert_resident(stage_S, expert_id)
        alt Expert Was Prefetched in Stage (S-H) and Completed
            Dev-->>Linear: true (HBM HIT!)
            Note over Linear: ZERO DDR5 STALL! Complete latency hiding!
        else Expert Prefetch Still In Flight (Partial Overlap)
            Dev-->>Linear: false (In-Flight Hit)
            Linear->>Engine: wait_for_in_flight(stage_S, expert_id)
            Note over Engine: Stalls pipeline ONLY for remaining transfer duration!
            Engine-->>Linear: Transfer completed
            Linear->>Dev: mark_expert_resident(stage_S, expert_id, size)
        else Unpredicted Expert (Demand Miss)
            Dev-->>Linear: false (Full Miss)
            Linear->>Engine: submit_demand_transfer(stage_S, expert_id)
            Note over Engine: Preempts prefetch bandwidth; serves demand with 100% BW
            Engine-->>Linear: Demand completed (~3.90 ms stall)
            Linear->>Dev: mark_expert_resident(stage_S, expert_id, size)
        end
    end

    Note over Route: Stage S Finishes; Stage S+1 Begins
```

---

## 4. Itemized New Integration Points for Phase 4

The following integration points must be introduced in Phase 4:

### Integration Point 1: Read-Only Residency Query
- **Location:** `src/hardware/device.h` and `src/hardware/device.cpp`.
- **Signature:** `bool has_expert_resident(int layer_id, int expert_id) const;`.
- **Purpose:** Allows the prefetch dispatcher to check whether an expert is already resident before issuing a speculative DDR5 transfer.
- **Invariant:** Must **not** mutate `lru_expert_list` and must **not** write to `expert_cache.csv`.

### Integration Point 2: Asynchronous Transfer Engine
- **Location:** `src/prefetch/async_transfer_engine.h` and `src/prefetch/async_transfer_engine.cpp`.
- **Responsibilities:**
  1. Maintain active transfer list with remaining byte accounting.
  2. Implement fluid-flow bandwidth sharing derived from calibration tables.
  3. Track in-flight transfers via `std::unordered_set<std::pair<int, int>, PairHash>`.
  4. Enforce strict demand priority over speculative prefetch transfers.
  5. Provide `advance_time(current_time_ns)` and `wait_for_in_flight(layer_id, expert_id)`.

### Integration Point 3: Candidate Dispatch Hook
- **Location:** `src/scheduler/scheduler.cpp` or `src/module/route.cpp`.
- **Trigger:** At the beginning of each MoE routing step, after candidates are enqueued into `PrefetchCandidateQueue`.
- **Action:** Calls `candidate_queue->select_ready_candidates(free_slots)`, filters redundant transfers via `has_expert_resident()`, and dispatches qualified candidates to `AsyncTransferEngine`.

### Integration Point 4: Demand Arbitration & In-Flight Interception
- **Location:** `src/module/linear.cpp` and `src/hardware/linear_impl.cpp`.
- **Action:** Before executing a full $3.90$ ms demand miss via Ramulator, query `AsyncTransferEngine::is_in_flight(layer_id, expert_id)`.
  - If in flight: Wait for the remaining transfer time, avoiding duplicate DDR5 fetch and reducing stall latency.
  - If not in flight: Issue demand transfer through `AsyncTransferEngine` with full channel priority.

### Integration Point 5: Calibration Artifact & Loader
- **Location:** `configs/prefetch/ddr5_transfer_calibration.csv` and `src/prefetch/prefetch_config.h/.cpp`.
- **Purpose:** Supplies calibrated empirical bandwidths and contention penalties to `AsyncTransferEngine`.

---

## 5. Existing Functions That Must Remain Completely Unchanged

To guarantee baseline reproducibility, preserve existing tests, and prevent regression, the following functions must **not** be modified:

1. `BatchedSequence::update_expert()`:  
   Router logic, trace extraction, and `num_token_in_expert` accumulation must remain 100% authoritative.
2. `Route::forward()`:  
   Module graph topology, activation sizing, and parallel dispatch rules must remain identical.
3. `Device::is_expert_resident()`:  
   Existing demand hit/miss reporting, LRU promotion, and `expert_cache.csv` formatting must remain intact for authoritative execution.
4. `Device::mark_expert_resident()`:  
   Core LRU eviction loop and `expert_cache.csv` `INSERT`/`EVICT` records must retain their exact operational semantics.
5. `DRAMInterface::HandleRequest()` and `DRAMInterface::run()`:  
   The synchronous Ramulator interface must remain intact for baseline execution and calibration runs.
6. `cluster->execution_time_cache` & `checkExecutionCache()`:  
   The execution memoization cache must remain functional for baseline configurations where prefetching is disabled.
7. Predictor Implementations (`ST-MoE`, `CSTP`, `ACSH`):  
   All Phase 2A/2B prediction algorithms must remain untouched.
8. Phase 3 Queue Implementation (`PrefetchCandidateQueue`):  
   Core priority comparisons, duplicate suppression, and stage reconciliation must remain untouched.

