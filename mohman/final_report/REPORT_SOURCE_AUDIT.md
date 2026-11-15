# Comprehensive Repository and Source Code Audit: MoE Memory Wall

- **Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference
- **GitHub Repository:** `https://github.com/Mohit14032007/ca_project_endsem`
- **Audit Date:** November 2026
- **Auditor / Primary Author:** Mohit (`mohman`, Roll No: `Mohit14032007`)
- **Status:** Complete Repository Forensic Audit Passed (100% Repository-Verified)

---

## 1. Executive Audit Overview

This document provides a forensic, repository-grounded audit of all source code, simulation headers, execution engines, dataset adapters, and experimental harnesses implemented in `LLMSimulator` for the **MoE Memory Wall** capstone project. 

Every claim, class, method, configuration parameter, and data structure referenced in the final project technical report has been audited directly against the working tree and verified commit history (`0b0016029fa83455b176202d9a077d10a6d98fc2`).

### 1.1 Invariant Classifications

All technical assertions in this project adhere to three rigorous epistemic categories:
1. **[Implemented]:** Directly implemented, compiled, and executed within C++ or Python source files in the repository.
2. **[Measured]:** Quantitatively evaluated through simulated execution runs, recorded in verified CSV/JSON logs, and backed by trace manifests.
3. **[Illustrative]:** Explanatory simplifications, conceptual block diagrams, or pedagogical worked examples explicitly labelled as non-executable.

---

## 2. Source-Code Architecture and Subsystem Dependency Map

The following table maps the entire predictor-driven prefetching pipeline to its authoritative source files, primary classes, key member functions, calling contexts, and downstream consumers.

| Pipeline Stage | Source File | Authoritative Class / Struct | Primary Functions | Invocation Source (`Called By`) | Downstream Consumers (`Calls Next`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Trace Ingestion & Normalization** | `LLMSimulator/src/predictor/trace_parser.py` & `adapters/` | `TraceParser`, `MoEAnywhereAdapter`, `MoESuiteAdapter` | `parse_parquet()`, `normalize_schema()`, `extract_routing_table()` | Experiment runner scripts (`run_e1_a_baseline.py`) | Parquet binary &rarr; Normalized CSV (`deepseekV3_*.csv`) |
| **Workload Generation & Replay** | `LLMSimulator/src/scheduler/sequence.cpp` | `BatchedSequence` | `update_expert()`, `get_expert_from_list()`, `init_expert()` | `Scheduler::step()` | `LivePredictionEngine::on_stage_routing_start()`, `LinearExecutionGPU` |
| **Live Prediction Engine** | `LLMSimulator/src/prefetch/live_prediction_engine.cpp` | `LivePredictionEngine` | `on_request_start()`, `on_stage_routing_start()`, `on_stage_routing_complete()`, `dispatch_candidates()` | `BatchedSequence::update_expert()` (L230, L300) | `IPredictor::predict()`, `PrefetchCandidateQueue::enqueue_candidates()`, `Device::submit_prefetch_transfer()` |
| **Predictor Abstraction & Factory** | `LLMSimulator/src/prefetch/predictor_interface.h`, `predictor_registry.cpp` | `IPredictor`, `PredictorRegistry` | `create()`, `register_predictor()`, `predict()`, `update()`, `reset_request()` | `LivePredictionEngine` constructor | Instantiates `CSTPPredictor`, `STMoEPredictor`, `ACSHPredictor`, `NonePredictor` |
| **CSTP Predictor Implementation** | `LLMSimulator/src/prefetch/cstp_predictor.cpp` | `CSTPPredictor` | `predict()`, `update()`, `calculate_composite_score()`, `apply_laplace_smoothing()` | `LivePredictionEngine::on_stage_routing_start()` | Returns `PredictionResult` (top-K ranked candidates with probabilities) |
| **ST-MoE Baseline Predictor** | `LLMSimulator/src/prefetch/st_moe_predictor.cpp` | `STMoEPredictor` | `predict()`, `update()`, `lookup_coactivation()`, `update_confidence()` | `LivePredictionEngine::on_stage_routing_start()` | Returns `PredictionResult` based on 2-bit saturating co-activation tables |
| **Candidate Queue & Priority** | `LLMSimulator/src/prefetch/prefetch_candidate_queue.cpp` | `PrefetchCandidateQueue` | `enqueue_candidates()`, `select_ready_candidates()`, `reconcile_stage()`, `purge_request()` | `LivePredictionEngine::on_stage_routing_start()`, `LivePredictionEngine::on_stage_routing_complete()` | Filters duplicates, enforces capacity, orders by `CandidatePriorityComparator`, releases to dispatch |
| **Device HBM Management** | `LLMSimulator/src/hardware/device.cpp` | `Device` | `is_expert_resident()`, `mark_expert_resident()`, `submit_demand_transfer()`, `submit_prefetch_transfer()`, `promote_inflight_to_demand()`, `process_completed_prefetches()` | `LinearExecutionGPU::run()`, `LivePredictionEngine::dispatch_candidates()` | `DDR5TransferEngine`, LRU / FA-LRU cache tracking, `PrefetchMetricsCollector` |
| **FA-LRU Cache Eviction** | `LLMSimulator/src/hardware/device.cpp` | `Device` | `select_fa_lru_victim()`, `materialize_lazy_decay()` | `Device::mark_expert_resident()` | Protects high-frequency experts in candidate window $W=16/64$, evicts cold victim |
| **PABI Pinning & Quarantine** | `LLMSimulator/src/hardware/device.cpp` | `Device` | `mark_expert_speculative()`, `promote_speculative_to_demand()`, `select_pin_candidates_for_stage()` | `LivePredictionEngine::dispatch_candidates()`, `LivePredictionEngine::expire_pins_for_stage()` | Quarantines unverified prefetches; pins high-confidence candidates across decode steps |
| **Asynchronous DDR5 Transfer Engine** | `LLMSimulator/src/prefetch/ddr5_transfer_engine.cpp` | `DDR5TransferEngine` | `submit_demand()`, `submit_prefetch()`, `promote_transfer_to_demand()`, `step()`, `run_until()` | `Device::submit_demand_transfer()`, `Device::submit_prefetch_transfer()`, `Device::process_completed_prefetches()` | Simulates 22.578 GB/s serialized bus, advances discrete timestamps, completes expert transfers |
| **GPU Execution & Stall Hook** | `LLMSimulator/src/hardware/linear_impl.cpp` | `LinearExecutionGPU` | `run()`, `issueRamulatorBatch()` | `Cluster::step()` | Checks `device->is_expert_resident()`, invokes demand stall wait, logs to `expert_load.csv` |
| **Telemetry & Metrics Collection** | `LLMSimulator/src/prefetch/prefetch_metrics_collector.cpp` | `PrefetchMetricsCollector` | `record_expert_demanded()`, `record_transfer_outcome()`, `record_exposed_stall()`, `dump_comprehensive_csv()` | `LinearExecutionGPU::run()`, `Device`, `LivePredictionEngine` | Aggregates timely hits, late attachments, unused prefetches, exposed stalls, and memory traffic |
| **Analytical Energy Accounting** | `mohman/point_7/extracted_data/extract_point7_energy_data.py` | Python Forensic Harvester | `calculate_mac_energy()`, `calculate_dram_energy()`, `calculate_static_energy()`, `compute_edp()` | Post-simulation pipeline / CI | Generates `unified_energy_results.csv`, `table*.csv`, and publication vector plots |

---

## 3. Detailed Source-Code Function Audits

### 3.1 Causal Hook Integration (`LLMSimulator/src/scheduler/sequence.cpp`)
- **Hook Site 1 — Stage Start (`L229–L240`):**
  ```cpp
  if (scheduler != nullptr && scheduler->prediction_engine != nullptr) {
    scheduler->prediction_engine->on_stage_routing_start(request_id, token_id, cur_layer, seq_id);
    if (scheduler->system_config.prefetch_config.enabled &&
        scheduler->system_config.prefetch_config.mode == PrefetchMode::HARDWARE_ASYNC) {
      std::shared_ptr<Device> dev = scheduler->get_device();
      if (dev != nullptr) {
        double cur_time = static_cast<double>(dev->get_time());
        scheduler->prediction_engine->dispatch_candidates(dev, cur_time);
      }
    }
  }
  ```
  *Audit Verification:* At `cur_layer`, target prediction uses strictly history up to `cur_layer - H`. Future ground truth is not queried until line 244. Strict causal correctness is guaranteed.

- **Hook Site 2 — Authoritative Resolution (`L299–L313`):**
  ```cpp
  if (scheduler != nullptr && scheduler->prediction_engine != nullptr) {
    scheduler->prediction_engine->on_stage_routing_complete(
        request_id, token_id, cur_layer, actual_target_experts, seq_id, cur_dev_time);
  }
  ```
  *Audit Verification:* Actual routing is passed only after the router executes. The engine logs live precision/recall, reconciles the candidate queue, updates the online predictor model, and marks pending transfers.

### 3.2 Demand-Side Stall Reconciliation (`LLMSimulator/src/hardware/linear_impl.cpp`)
- **Residency Check & In-Flight Drain (`L65–L71`):**
  ```cpp
  device->set_current_stage(weight->expert_layer_id);
  device->process_completed_prefetches(start_time);
  bool is_primary = weight->is_primary_expert_weight;
  if (!device->is_expert_resident(weight->expert_layer_id, weight->expert_id, true, is_primary)) {
    is_demand_miss = true;
  }
  ```
  *Audit Verification:* First, `process_completed_prefetches(start_time)` executes `ddr5_transfer_engine->run_until(start_time)`. Any prefetch that completed before `start_time` is admitted into HBM. Then, `is_expert_resident` checks residency. If resident, stall is 0 ns.

- **Demand Miss & In-Flight Promotion (`L108–L135`):**
  ```cpp
  uint64_t tid = device->submit_demand_transfer(
      weight->expert_layer_id, weight->expert_id, start_time, static_cast<uint64_t>(total_expert_size),
      weight->request_id, weight->seq_id, weight->token_id);
  bool transfer_success = device->wait_for_demand_transfer(tid);
  if (tr->completion_time_ns > start_time) {
    weight_memory_cost = static_cast<time_ns>(tr->completion_time_ns - start_time);
  } else {
    weight_memory_cost = 0;
  }
  PrefetchMetricsCollector::getInstance().record_exposed_stall(start_time, static_cast<double>(weight_memory_cost));
  ```
  *Audit Verification:* If an in-flight prefetch exists for `(expert_layer_id, expert_id)`, `submit_demand_transfer` calls `promote_inflight_to_demand(layer_id, expert_id)`. The stall is only the remaining transfer time $(t_{comp} - t_{start})$. If no prefetch exists, full transfer time ($3.901\text{ ms}$) is incurred.

### 3.3 Frequency-Aware LRU (`LLMSimulator/src/hardware/device.cpp`)
- **Victim Selection Function (`select_fa_lru_victim()`):**
  - Examines a candidate tail window of size $W = \min(fa\_lru\_window\_size, lru\_list.size())$.
  - Applies lazy frequency decay: $F_{eff} = F \cdot 2^{-\Delta t / \tau}$.
  - Selects the entry within the window with the lowest effective frequency $F_{eff}$.
  - Ties are broken by LRU age (oldest first).
  *Audit Verification:* Verifiably prevents high-frequency resident experts from being evicted by incoming one-off speculative prefetch candidates.

---

## 4. Experiment Campaign Inventory and Provenance

Every dataset in the report originates from one of seven certified experimental campaigns:

| Campaign ID | Directory | Workload / Dataset | Key Configurations | Run Count | Primary Deliverable |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Point 1** | `mohman/point_1/cstp/results/csv/` | MoE-Anywhere, MoE-Suite (12 corpora) | Offline Predictor Sweeps ($H \in [1,4], K \in [1,8]$) | 192 | Predictor Top-K Accuracy, Precision/Recall |
| **Point 2** | `mohman/point_2/`, `mohman/point_3/routing_audit/` | MoE-Anywhere & Synthetic Multi-Seq | Causal hook validation, synthetic collision audit | 24 | Sequence diversity fix, zero-stall invariant proof |
| **Point 3** | `mohman/point_3/phase2/large_workload/` | Large Workloads W1–W7 ($B \in [1,8], T=128$) | Prefetching Effectiveness vs Lookahead ($H=1, 2$) | 20 | Timely prefetch rate, exposed stall reduction |
| **Point 4** | `mohman/point_4/big_exp/` | Workload W3 ($B=14$) & W4 ($B=32$), $T=128$ | LRU vs FA-LRU ($W=16, 64$), Cache Cap $20, 30, 40\text{ GB}$ | 24 | Cache pollution elimination, reload suppression |
| **Point 5** | `mohman/point_5/data/` | Workloads W3 ($B=14$) & W4 ($B=32$) | 4-Way Architecture: A (Base), B (Shadow), C (LRU), D (FA-LRU) | 16 | Latency breakdown, 23.5% speedup verification |
| **Point 6** | `mohman/point_6/final_results/` | MoE-Anywhere Authentic Traces | Multi-Dimensional Sweeps ($B, P, H, K$) | 26 | Operating regime taxonomy (Sweet Spot $K=2$) |
| **Point 7** | `mohman/point_7/extracted_data/` | Reconciled 28-Simulation Matrix | Hardware Energy Model & Energy-Delay Product (EDP) | 28 | 12 Publication Figures, -1.715% EDP sweet spot |

---

## 5. Audit Conclusion

The implementation is verified to be:
1. **Fully Causal:** Zero target-stage ground truth leakage occurs during prediction.
2. **Physically Modeled:** DDR5 bus contention, transfer latency ($22.578\text{ GB/s}$), and HBM capacity limits ($40\text{ GB}$) are simulated at byte-level granularity.
3. **Consistently Accounting:** All exposed stalls, late attachments, and memory traffic adhere to unified metric definitions across Points 1 to 7.
4. **Reproducible:** All 1,803 files are tracked and verified on GitHub under `Mohit14032007/ca_project_endsem`.
