# Phase 0: Forensic Audit of Predictor-Driven Expert Prefetching

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Git Baseline Anchor:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Audit Date:** November 9, 2026  
**Deliverables:**
- Master Audit Report: `docs/phase_prefetch/CURRENT_PREFETCH_IMPLEMENTATION_AUDIT.md`
- Implementation File Map: `docs/phase_prefetch/CURRENT_PREFETCH_FILE_MAP.csv`
- Mirrored Copies in: `LLMSimulator/docs/phase_prefetch/`

---

## 1. Executive Summary & Verdict

### Core Verdict
**YES.** We can conclusively demonstrate a complete, causally correct, predictor-driven forward-lookahead prefetch mechanism that transfers experts into GPU HBM while computation is actively executing.

### Ground Truth Proof and Evidence
1. **Architectural Completeness & Integration:**
   - **Forward-Lookahead Pipeline:** Fully integrated across `src/scheduler/sequence.cpp:240-310`, `src/prefetch/live_prediction_engine.cpp:270-303`, `src/prefetch/prefetch_candidate_queue.cpp:94-170`, `src/hardware/device.cpp:1418-1474`, `src/prefetch/ddr5_transfer_engine.cpp:120-210`, and `src/hardware/linear_impl.cpp:64-106`.
   - **Strict Causality:** Forecasts target stage $T = L + H$ using observed routing from source stage $L$ post-routing. Zero future ground-truth routing is queried, passed, or leaked. Actual router selections remain authoritative; predictions purely trigger physical memory transfers.
   - **True Asynchronous Overlap:** The simulation engine drives DDR5 transfers via a discrete-event engine (`DDR5TransferEngine::run_until()`), where the simulation clock advances as GPU GEMM computations execute (`src/module/module_graph.cpp:249`). Transfers execute concurrently with intervening compute stages $L, L+1, \dots, L+H-1$.
   - **In-Flight Demand Coordination:** When an expert demanded at stage $L+H$ is already in transit (`inflight_registry`), `Device::submit_demand_transfer()` promotes the transfer to `DEMAND` priority and halts the device only for the remaining transfer latency (`tr->completion_time_ns - current_time_ns`), rather than stalling for the full $3.901\text{ ms}$. If the transfer completed prior to demand arrival, memory stall is exactly $0\text{ ns}$.
   - **HBM Isolation and Protection:** Speculative transfers are quarantined in a dedicated $16\text{ GB}$ PABI buffer, preventing LRU pollution of demand-verified experts. Belády pin intents protect candidates from premature eviction across intermediate stages until their deadline stage $L+H$ executes.

2. **Empirical & Experimental Demonstration:**
   - **Single-Token Cold-Cache Regime ($M = 1.0\times$):** As documented in `PHASE2E2_VALIDATION_RESULTS.md`, single-batch decode GEMMs execute in $\sim 0.183\text{ ms}$ per stage ($0.366\text{ ms}$ for $H = 2$). Because physical DDR5 transfer of an $88.08\text{ MB}$ expert requires $3.901\text{ ms}$, the available compute window is only $\sim 9.4\%$ of the transfer duration. Prefetches arrive partially complete (`LATE_IN_FLIGHT`), achieving partial stall reduction but $0$ timely resident hits.
   - **Batched / Scaled Compute Regime ($M \ge 12\times$):** In Phase 2F (`PHASE2F_EXECUTIVE_SUMMARY.md`, `PHASE2F_FULL_RESULTS.md`), scaling the compute window across $M \in \{1\times, 4\times, 8\times, 12\times, 16\times\}$ expanded available compute time to $4.396\text{ ms}$ ($M=12\times$) and $5.861\text{ ms}$ ($M=16\times$). Crossing the $3.901\text{ ms}$ transfer threshold **unmasked timely resident hits ($H_{\text{timely}} = 1 \text{ to } 2$)**, eliminating memory stalls and delivering up to **$12.86\text{ ms}$ of exposed memory stall reduction**.

3. **Critical Configuration Trap:**
   - In `struct PrefetchConfig` (`src/prefetch/prefetch_config.h:127`), `scheduling_mode` defaults to `SchedulingMode::LEGACY_CURRENT_STAGE` (co-located dispatch).
   - Forward-lookahead prefetching is **ONLY** activated if explicitly enabled in YAML via `scheduling.scheduling_mode: forward_lookahead` or via the environment override `PREFETCH_SCHEDULING_MODE=forward_lookahead`. Running the simulator without this explicit configuration activates legacy co-located dispatch, yielding $0$ overlap and $0$ timely hits.

---

## 2. Current Repository Integrity and Historical Reconciliation

### Git Commit and Working Tree Status
- **Baseline Git Anchor:** Commit `d5bf61ff0c066c06f72173189f9924329f613a88` on branch `main` (`Finalize LLMSimulator: Add Group 13 master reference documentation...`).
- **Working Tree State:** Clean with respect to core baseline logic, containing 20 modified tracked files in `LLMSimulator` and untracked research modules.
- **Modified Tracked Files:**
  1. `CMakeLists.txt`, `config.yaml`
  2. `eval/CMakeLists.txt`, `eval/test.cpp`, `eval/test_deepseekv3_mixtral_style.cpp`
  3. `src/CMakeLists.txt`
  4. `src/hardware/cluster.cpp`, `src/hardware/cluster.h`
  5. `src/hardware/device.cpp`, `src/hardware/device.h`, `src/hardware/hardware_config.h`
  6. `src/hardware/layer_impl.cpp`, `src/hardware/linear_impl.cpp`
  7. `src/module/linear.cpp`, `src/module/module_graph.cpp`, `src/module/route.cpp`
  8. `src/scheduler/scheduler.cpp`, `src/scheduler/scheduler.h`
  9. `src/scheduler/sequence.cpp`, `src/scheduler/sequence.h`
- **Untracked Additions:**
  - `src/prefetch/` (11 C++ header and source files implementing predictors, candidate queue, transfer engine, and configuration parser).
  - `eval/` test suites (`test_ddr5_transfer_engine.cpp`, `test_forward_lookahead_prefetch.cpp`, `test_phase2e2_remediation.cpp`, `test_phase2f_multiplier.cpp`, etc.).
  - `docs/` and `results/` containing comprehensive campaign logs and test artifacts.

### Reconciliation with Historical Reports
1. **`PREFETCH_SCHEDULING_AUDIT.md`:** Correctly diagnosed the initial legacy temporal inversion where prefetch dispatch occurred at stage $L$ targeting stage $L$ using source $L-H$, leaving only $\sim 10-50\,\mu\text{s}$ before demand arrival.
2. **`PHASE2C_FORWARD_LOOKAHEAD_DESIGN.md` & `PHASE2D_FORWARD_LOOKAHEAD_AUDIT.md`:** Designed and audited the corrected forward-lookahead architecture ($L \to L+H$). Verified 10/10 test cases in `test_forward_lookahead_prefetch.cpp`.
3. **`PHASE2E1_REMEDIATION_PLAN.md` & `PHASE2E2_VALIDATION_RESULTS.md`:** Identified and rectified a 1-stage indexing mismatch in `sequence.cpp`/`linear.cpp` by introducing `sequences_metadata->current_moe_stage_id`, filtered candidate queue dispatch to `ADMITTED` candidates only, and added `transfer_id` as the 12th column in `expert_load.csv`.
4. **`PHASE2F_EXECUTIVE_SUMMARY.md` & `PHASE2F_FULL_RESULTS.md`:** Proved that when compute windows are scaled to realistic multi-token workloads ($M \ge 12\times$), prefetching achieves timely resident hits ($H_{\text{timely}} = 1 \text{ to } 2$) and reduces stall by up to $12.86\text{ ms}$.

---

## 3. YAML Configuration Inventory

The prefetching subsystem is fully configurable via the YAML configuration file under the `prefetching` section. Parsing is implemented in `src/prefetch/prefetch_config.cpp:218-472`.

| YAML Key Path | Parser Location | C++ Destination Variable | Actual Default | Accepted Values | Runtime Effect | Environment Override |
|---|---|---|---|---|---|---|
| `prefetching.enabled` | `prefetch_config.cpp:222` | `PrefetchConfig::enabled` | `false` | `true`, `false` | Master toggle for prediction and prefetching engine | None |
| `prefetching.mode` | `prefetch_config.cpp:227` | `PrefetchConfig::mode` | `PrefetchMode::NONE` | `none`, `prediction_only`, `hardware_sync`, `hardware_async` | Governs execution mode (`hardware_async` activates transfer engine) | None |
| `prefetching.predictor.type` | `prefetch_config.cpp:233` | `PrefetchConfig::predictor_type` | `PredictorType::NONE` | `none`, `st_moe`, `cstp`, `acsh_global`, `acsh_stage_aware`, `acsh_fast_recovery`, `frequency`, `temporal`, `spatial`, `random`, `oracle` | Selects active C++ predictor algorithm | None |
| `prefetching.predictor.alpha` | `prefetch_config.cpp:238` | `PrefetchConfig::alpha` | `0.5` | Double in `[0.0, 1.0]` | Balance between spatial ($S$) and temporal ($T$) scores in CSTP | None |
| `prefetching.predictor.smoothing_lambda` | `prefetch_config.cpp:243` | `PrefetchConfig::smoothing_lambda` | `1.0` | Double `> 0.0` | Dirichlet prior smoothing constant in CSTP | None |
| `prefetching.predictor.confidence_threshold` | `prefetch_config.cpp:248` | `PrefetchConfig::confidence_threshold` | `2` | Integer `[0, 3]` | ST-MoE saturating 2-bit counter threshold | None |
| `prefetching.predictor.online_adaptation` | `prefetch_config.cpp:253` | `PrefetchConfig::online_adaptation` | `true` | `true`, `false` | Enables online CCT and spatial matrix updates during inference | None |
| `prefetching.candidate_selection.policy` | `prefetch_config.cpp:302` | `PrefetchConfig::candidate_selection_policy` | `CandidateSelectionPolicy::TOP_K` | `top_k`, `threshold_filtered`, `adaptive` | Filter policy for candidate generation | None |
| `prefetching.candidate_selection.budget` | `prefetch_config.cpp:307` | `PrefetchConfig::budget` | `4` | Integer `[1, 64]` | Maximum prefetch candidates $B$ per stage | None |
| `prefetching.scheduling.scheduling_mode` | `prefetch_config.cpp:315` | `PrefetchConfig::scheduling_mode` | `SchedulingMode::LEGACY_CURRENT_STAGE` | `legacy_current_stage`, `forward_lookahead` | **CRITICAL:** Selects forward-lookahead vs legacy dispatch | `PREFETCH_SCHEDULING_MODE` |
| `prefetching.scheduling.prediction_horizon` | `prefetch_config.cpp:320` | `PrefetchConfig::prediction_horizon` | `1` | Integer `[1, 10]` | Layer lookahead distance $H$ ($T = L + H$) | `PREFETCH_PREDICTION_HORIZON` |
| `prefetching.scheduling.demand_priority` | `prefetch_config.cpp:330` | `PrefetchConfig::demand_priority` | `true` | `true`, `false` | Enforces demand preemption over prefetch transfers | None |
| `prefetching.scheduling.queue_enabled` | `prefetch_config.cpp:335` | `PrefetchConfig::queue_enabled` | `true` | `true`, `false` | Enables multi-stage prefetch candidate queue | None |
| `prefetching.scheduling.queue_capacity` | `prefetch_config.cpp:340` | `PrefetchConfig::queue_capacity` | `1024` | Integer `[1, 65536]` | Maximum candidate capacity in queue | None |
| `prefetching.scheduling.priority_policy` | `prefetch_config.cpp:345` | `PrefetchConfig::priority_policy` | `PrefetchPriorityPolicy::DEADLINE_SCORE_RANK` | `deadline_score_rank`, `deadline_only`, `score_only` | Sorting key for candidate dispatch priority | None |
| `prefetching.transfer.model` | `prefetch_config.cpp:365` | `PrefetchConfig::transfer_model` | `TransferModel::EVENT_DRIVEN` | `event_driven`, `ramulator_full`, `fixed_latency` | Memory transfer simulation model | None |
| `prefetching.transfer.asynchronous` | `prefetch_config.cpp:370` | `PrefetchConfig::asynchronous` | `true` | `true`, `false` | Enables asynchronous non-blocking memory transfers | None |
| `prefetching.transfer.max_outstanding_transfers` | `prefetch_config.cpp:375` | `PrefetchConfig::max_outstanding_transfers` | `4` | Integer `[1, 64]` | Max concurrent transfers allowed in queue | None |
| `system.ddr5_bandwidth_gbps` | `hardware_config.h:352` | `SystemConfig::ddr5_bandwidth_gbps` | `22.578` | Double `> 0.0` | Calibrated DDR5 physical bandwidth | `DDR5_BANDWIDTH_GBPS` |
| `system.use_event_driven_dram` | `hardware_config.h:345` | `SystemConfig::use_event_driven_dram` | `false` | `true`, `false` | Activates `DDR5TransferEngine` in device | `USE_EVENT_DRIVEN_DRAM` |
| `system.bypass_offchip_execution_cache` | `hardware_config.h:348` | `SystemConfig::bypass_offchip_execution_cache` | `false` | `true`, `false` | Bypasses execution cache for off-chip transfers | `BYPASS_OFFCHIP_EXECUTION_CACHE` |
| `system.compute_time_multiplier` | `hardware_config.h:356` | `SystemConfig::compute_time_multiplier` | `1.0` | Double `> 0.0` | Synthetic scaling factor for compute window | `COMPUTE_MULTIPLIER` |
| `prefetching.hbm.eviction_policy` | `prefetch_config.cpp:390` | `PrefetchConfig::eviction_policy` | `EvictionPolicy::LRU` | `lru`, `pabi`, `lru_pinned` | HBM cache eviction policy | `EVICTION_POLICY` |
| `prefetching.pabi.quarantine_enabled` | `prefetch_config.cpp:402` | `PrefetchConfig::pabi_quarantine_enabled` | `true` | `true`, `false` | Isolates speculative experts in dedicated buffer | `PABI_QUARANTINE_ENABLED` |
| `prefetching.pabi.speculative_buffer_capacity_bytes` | `prefetch_config.cpp:407` | `PrefetchConfig::pabi_speculative_buffer_capacity_bytes` | `16000000000LL` (16 GB) | Int64 `> 0` | Physical byte capacity of speculative buffer | `PABI_SPECULATIVE_BUFFER_CAPACITY_BYTES` |
| `prefetching.pabi.pinning_enabled` | `prefetch_config.cpp:417` | `PrefetchConfig::pabi_pinning_enabled` | `false` | `true`, `false` | Enables Belády pin protection until target stage | `PABI_PINNING_ENABLED` |
| `prefetching.pabi.max_pinned_experts` | `prefetch_config.cpp:422` | `PrefetchConfig::pabi_max_pinned_experts` | `4` | Integer `[1, 32]` | Maximum concurrent pinned experts | `PABI_MAX_PINNED_EXPERTS` |

### Environment Variable Precedence Rules
Environment variables strictly override YAML values at startup:
1. `PREFETCH_SCHEDULING_MODE` overrides `prefetching.scheduling.scheduling_mode`.
2. `PREFETCH_PREDICTION_HORIZON` overrides `prefetching.scheduling.prediction_horizon`.
3. `USE_EVENT_DRIVEN_DRAM` overrides `system.use_event_driven_dram`.
4. `DDR5_BANDWIDTH_GBPS` overrides `system.ddr5_bandwidth_gbps`.
5. `COMPUTE_MULTIPLIER` overrides `system.compute_time_multiplier`.
6. `PABI_QUARANTINE_ENABLED`, `PABI_PINNING_ENABLED`, `PABI_SPECULATIVE_BUFFER_CAPACITY_BYTES`, and `EVICTION_POLICY` override their respective HBM settings in `Device::Device()`.

---

## 4. Architecture and Component Inventory

The prefetch subsystem comprises seven primary modular C++ components:

```
+-----------------------------------------------------------------------------------+
|                                  Scheduler                                        |
|  (src/scheduler/sequence.cpp: BatchedSequence::update_expert)                     |
+-----------------------------------------+-----------------------------------------+
                                          |
                +-------------------------+-------------------------+
                | on_stage_routing_complete(L, actual_experts)      | dispatch_candidates()
                v                                                   v
+-------------------------------+                         +-------------------------+
|     LivePredictionEngine      |                         |         Device          |
| (src/prefetch/live_prediction)|                         | (src/hardware/device.cpp|
+---------------+---------------+                         +------------+------------+
                |                                                      |
                | queries predict(L, L+H)                              | submits speculative
                v                                                      v
+---------------+---------------+                         +------------+------------+
|      Concrete Predictor       |                         |    DDR5TransferEngine   |
| (STMoE / CSTP / ACSH / None)  |                         | (src/prefetch/ddr5_eng) |
+---------------+---------------+                         +------------+------------+
                |                                                      |
                | returns CandidateScores                              | discrete-event
                v                                                      | timeline (22.58 GB/s)
+---------------+---------------+                                      v
|    PrefetchCandidateQueue     |                         +------------+------------+
| (src/prefetch/candidate_queue)|                         |        HBM Cache        |
|  - ADMITTED / READY filtering |------------------------>|  - Speculative 16GB     |
|  - DEADLINE_SCORE_RANK sort   |   dispatched to Dev     |  - Demand-Verified LRU  |
+-------------------------------+                         |  - Belady Pins          |
+-----------------------------------------------------------------------------------+
```

1. **`LivePredictionEngine` (`src/prefetch/live_prediction_engine.{h,cpp}`):** Orchestrates live simulation interactions. In `FORWARD_LOOKAHEAD` mode, skips prediction at `on_stage_routing_start()`, generates forward predictions for stage $L+H$ at `on_stage_routing_complete()`, tracks pending predictions, and computes precision, recall, and F1 metrics.
2. **`IPredictor` Subsystem (`src/prefetch/predictor_interface.h`):** Defines the standard virtual C++ interface (`predict`, `update`, `reset`). Concrete implementations include:
   - `STMoEPredictor`: 2-bit saturating counters in a Cross-Layer Correlation Table (CCT) and History Table.
   - `CSTPPredictor`: Dirichlet-smoothed spatial transition matrix $P(e_j|e_i)$ blended with temporal frequency $\alpha S + (1-\alpha) T$.
   - `ACSHPredictor`: Adaptive softmax fusion between CSTP and ST-MoE with fast error cooldown.
   - `ControlPredictors`: Empirical Frequency, Temporal, Spatial, Random, and Oracle baselines.
3. **`PrefetchCandidateQueue` (`src/prefetch/prefetch_candidate_queue.{h,cpp}`):** Multi-stage priority queue. Implements candidate admission, deadline-aware sorting (`DEADLINE_SCORE_RANK`), duplicate suppression, and capacity overflow management.
4. **`DDR5TransferEngine` (`src/prefetch/ddr5_transfer_engine.{h,cpp}`):** Discrete-event simulation of the physical off-chip DDR5 bus. Models calibrated bandwidth ($22.578\text{ GB/s}$), single-channel serialization, demand preemption of active speculative transfers, and pause/resume accounting.
5. **`Device` Cache & Coordination (`src/hardware/device.{h,cpp}`):** Manages HBM cache residency, in-flight transfer tracking (`inflight_registry`), PABI speculative buffer quarantine ($16\text{ GB}$ cap), Belády pinning (`active_pin_intents`), and candidate dispatch coordination.
6. **`LinearExecutionGPU` (`src/hardware/linear_impl.cpp`):** Executes GPU linear layers, checks expert HBM residency, issues demand transfers on misses, attaches to in-flight transfers, accounts for exposed memory stall, and logs the 12-column `expert_load.csv`.
7. **`BatchedSequence` (`src/scheduler/sequence.{h,cpp}`):** Manages decoding sequences, evaluates router outputs, and invokes prefetch engine hooks at exact sequence boundaries.

---

## 5. Exact Production Call Graph

The end-to-end production execution sequence flows across 12 precisely verified steps:

```
[Step 1: Router Evaluates Top-K Experts]
   |  src/scheduler/sequence.cpp:242-294 (BatchedSequence::update_expert)
   v
[Step 2: Routing Completion Notified to Engine]
   |  src/scheduler/sequence.cpp:295 (calls pe->on_stage_routing_complete)
   v
[Step 3: Engine Validates Boundaries and Updates Pending Stage L]
   |  src/prefetch/live_prediction_engine.cpp:141-265 (computes metrics, updates predictor tables)
   v
[Step 4: Predictor Generates Target L+H Predictions]
   |  src/prefetch/live_prediction_engine.cpp:270-282 -> src/prefetch/st_moe_predictor.cpp:175-208 / cstp_predictor.cpp:223-258
   v
[Step 5: Candidate Selection & Budget Enforcement]
   |  src/prefetch/live_prediction_engine.cpp:277-280 (budget B enforced, candidate scores sorted)
   v
[Step 6: Candidate Queue Admits Candidates]
   |  src/prefetch/prefetch_candidate_queue.cpp:94-170 (checks capacity, assigns DEADLINE_SCORE_RANK)
   v
[Step 7: Immediate Candidate Dispatch Post-Routing]
   |  src/scheduler/sequence.cpp:301-310 -> src/prefetch/live_prediction_engine.cpp:528-537 -> src/hardware/device.cpp:1418-1474
   v
[Step 8: DDR5 Engine Enqueues and Begins Physical Transfer]
   |  src/hardware/device.cpp:1219-1289 -> src/prefetch/ddr5_transfer_engine.cpp:120-210 (calculates 3.901 ms transfer time)
   v
[Step 9: Intervening Compute Advances Simulation Clock]
   |  src/module/module_graph.cpp:249 (status.device_time += exec_status.total_duration)
   |  src/hardware/device.cpp:1322-1415 (calls ddr5_transfer_engine->run_until(device_time))
   v
[Step 10: Demand Arrives at Target Stage L+H]
   |  src/module/linear.cpp:48-60 & src/hardware/linear_impl.cpp:64-74 (checks is_expert_resident)
   v
[Step 11: Residency Resolution (Resident Hit, In-Flight Attachment, or Demand Miss)]
   |  src/hardware/linear_impl.cpp:76-106 & src/hardware/device.cpp:1119-1215
   v
[Step 12: HBM Eviction, Quarantine, Promotion, and Pin Expiration]
      src/hardware/device.cpp:450-580, 700-820, 1450-1530 (PABI buffer enforcement, pin expiration)
```

### Line-by-Line Execution Details
- **Step 1 (`sequence.cpp:242-294`):** Extracts Top-K routed experts (`actual_target_experts`) from preloaded trace (`sequences_info[seq_id]->expert_list`) or PRNG.
- **Step 2 (`sequence.cpp:295`):** Passes `actual_target_experts` for `current_moe_stage_id` to `pe->on_stage_routing_complete()`.
- **Step 3 (`live_prediction_engine.cpp:141-265`):** Evaluates ground truth for stage $L$, logs metrics (`precision`, `recall`, `f1`), and invokes `predictor_->update()` using actual routed experts.
- **Step 4 (`live_prediction_engine.cpp:270-282`):** Verifies `future_target_stage_id = target_stage_id + horizon_ < num_stages_`. Passes source stage $L$ and `actual_target_experts` to `predictor_->predict()`.
- **Step 5 (`live_prediction_engine.cpp:277-280`):** C++ predictor evaluates spatial/temporal scoring matrices, returning top $B$ candidate expert IDs and scores.
- **Step 6 (`prefetch_candidate_queue.cpp:94-170`):** Enqueues candidates, verifies queue capacity, applies duplicate policy, and transitions state to `CandidateState::ADMITTED`.
- **Step 7 (`sequence.cpp:301-310` & `device.cpp:1418-1474`):** `dispatch_candidates(dev, cur_time)` invokes `Device::dispatch_prefetch_candidates()`. Selects `ADMITTED` candidates, checks residency/in-flight duplicates, and registers Belády pins.
- **Step 8 (`device.cpp:1219-1289` & `ddr5_transfer_engine.cpp:120-210`):** Enforces PABI quarantine buffer limit ($16\text{ GB}$). Submits prefetch to `DDR5TransferEngine`. Calculates transfer duration: $88,080,384\text{ bytes} / 22.578\text{ bytes/ns} = 3,901,160\text{ ns}$.
- **Step 9 (`module_graph.cpp:249` & `device.cpp:1322-1415`):** Simulated GPU execution of GEMMs increments `status.device_time`. At stage boundaries and before linear access, `device->process_completed_prefetches(start_time)` executes `ddr5_transfer_engine->run_until(start_time)`.
- **Step 10 (`linear.cpp:48-60` & `linear_impl.cpp:64-74`):** Stage $L+H$ linear execution queries `device->is_expert_resident(layer_id, expert_id)`.
- **Step 11 (`linear_impl.cpp:76-106` & `device.cpp:1119-1215`):**
  - **Resident Hit:** Expert already in HBM. Memory stall $= 0\text{ ns}$. If in PABI speculative buffer, it is promoted to demand-verified LRU list.
  - **In-Flight Attachment:** Expert in `inflight_registry`. Promoted to `DEMAND`. Memory stall $= \text{remaining\_duration\_ns} = tr\to\text{completion\_time\_ns} - \text{start\_time}$.
  - **Demand Miss:** Absent from HBM. Submits demand transfer preempting speculative transfers. Full memory stall $= 3.90116\text{ ms}$.
- **Step 12 (`device.cpp:450-580`, `700-820`):** Evictions from PABI speculative buffer never evict demand-verified experts. Pinned experts are protected until `expire_pins_for_stage(current_moe_stage_id)` is invoked at stage completion.

---

## 6. Forward-Lookahead Scheduling Verification

A critical priority of this audit was verifying the elimination of the temporal co-located scheduling defect identified in earlier audits.

### Causality and Target Formulation Audit
1. **Source Stage Selection:** Verified in `live_prediction_engine.cpp:273-274`. Source stage is explicitly set to `source_stage_id = target_stage_id` (the stage that just finished routing).
2. **Trigger Timing:** Verified in `sequence.cpp:295-310`. The forward prediction trigger occurs **after** actual routing evaluation completes (`on_stage_routing_complete`), and candidate dispatch occurs immediately post-routing.
3. **Target Stage Formulation:** Verified in `live_prediction_engine.cpp:271`. Target stage is formulated strictly as:
   $$T = L + H$$
   Tested and verified for $H \in \{1, 2, 3\}$.
4. **Boundary Condition Handling:** Verified in `live_prediction_engine.cpp:272`. If $L + H \ge S$ (where $S$ is the total MoE stage count), candidate generation and dispatch are cleanly suppressed. Verified in Unit Test 2 of `test_forward_lookahead_prefetch.cpp`.
5. **Strict Causality Enforcement:** Neither `predict()` in `st_moe_predictor.cpp` nor `cstp_predictor.cpp` takes or accesses future routing vectors. Only historical transition tables (`cct_`, `spatial_counts_`, `history_table_`) are queried.
6. **Router Authoritativeness:** `actual_target_experts` evaluated by the sequence router (`sequence.cpp:254-267`) directly determine token routing and sequence state. Prefetch candidates initiate memory transfers only and never alter token-expert routing.
7. **Logical Stage Decoupling:** In Phase 2E.1, an indexing bug was identified where physical layer counter `cur_layer++` advanced faster than logical MoE stages in multi-stage execution. This was decoupled in `sequence.cpp` and `linear.cpp` by binding execution to `sequences_metadata->current_moe_stage_id`.

---

## 7. DDR5 Transfer and Computation-Overlap Explanation

### Simulation Model Architecture
The DDR5 transfer engine (`src/prefetch/ddr5_transfer_engine.cpp`) is implemented as a **discrete-event simulation model** operating on simulated nanoseconds, **not** as a concurrent background host thread.

- **Event Queue:** Events (`TransferEvent`) are scheduled in an ordered priority queue (`std::priority_queue<TransferEvent, ..., std::greater>`).
- **Tie-Breaking:** Completions are strictly scheduled before new arrivals on exact timestamp collisions (`ddr5_transfer_engine.h:130-158`), eliminating race conditions.
- **Physical Bandwidth:** Calibrated from cycle-accurate Ramulator2 traces at:
  $$\text{Bandwidth} = 22.578\text{ GB/s} = 22.578\text{ bytes/ns}$$
- **Transfer Duration:** For an $88,080,384\text{ byte}$ expert ($3 \times 29,360,128\text{ byte}$ tensors):
  $$T_{\text{transfer}} = \frac{88,080,384\text{ bytes}}{22.578\text{ bytes/ns}} = 3,901,160\text{ ns} = 3.90116\text{ ms}$$
- **Single-Channel Serialization:** The physical DDR5 memory bus is a single shared resource. If candidate #1 is transmitting, candidate #2 cannot begin transmission until candidate #1 completes. For budget $B = 4$, the required bus transmission times are:
  - Candidate 1: $3.901\text{ ms}$
  - Candidate 2: $7.802\text{ ms}$
  - Candidate 3: $11.703\text{ ms}$
  - Candidate 4: $15.605\text{ ms}$
- **Preemption & Resumption:** High-priority demand transfers preempt active speculative transfers immediately (`ddr5_transfer_engine.cpp:240-310`). The speculative transfer transitions to `PAUSED`, byte progress is saved, and it resumes once the demand transfer finishes.

### Timeline Reconstruction: Predicted Expert Overlap
Below is the concrete reconstructed timeline for a prefetch candidate dispatched at stage $L$ with horizon $H = 2$:

```
Timeline (ns)
0.0 ms:  Stage L Routing Complete. Top candidate dispatched.
         ddr5_engine schedules COMPLETION event at t = 3,901,160 ns.
0.0 - 0.183 ms: Stage L Linear GEMM executes on GPU (duration = 183,000 ns).
0.183 ms: Stage L completes. Simulation clock = 183,000 ns.
0.183 - 0.366 ms: Stage L+1 Linear GEMM executes on GPU (duration = 183,000 ns).
0.366 ms: Stage L+1 completes. Simulation clock = 366,000 ns.
          Actual Demand arrives for Stage L+2!

RESIDENCY LOOKUP AT t = 366,000 ns:
- Total Elapsed Compute Window: 366,000 ns (0.366 ms)
- Transfer Completion Target:   3,901,160 ns (3.901 ms)
- Transfer State: ACTIVE (In-Flight)
- Progress Transferred: (366,000 ns * 22.578 bytes/ns) = 8,263,548 bytes (~9.4%)
- Remaining Transfer Time: 3,901,160 - 366,000 = 3,535,160 ns (3.535 ms)

OUTCOME:
- Outcome Classification: LATE_IN_FLIGHT attachment
- Device Stalls For: 3.535 ms (rather than full 3.901 ms)
- Stall Reduction Achieved: 0.366 ms (100% of available compute overlap window)
```

**Why Timely Hits Emerge Under Scaled Compute ($M \ge 12\times$):**
When the compute multiplier $M$ scales GEMM durations by $12\times$ ($2.198\text{ ms}$ per stage, or $4.396\text{ ms}$ for $H = 2$):
- Compute window ($4.396\text{ ms}$) $>$ Transfer time ($3.901\text{ ms}$).
- At $t = 3.901\text{ ms}$, `ddr5_transfer_engine->run_until()` fires the `COMPLETION` event.
- Candidate is inserted into HBM as `SPECULATIVE` and pinned.
- When demand arrives at $t = 4.396\text{ ms}$, **the expert is already RESIDENT**.
- **Memory Stall $= 0.00\text{ ms}$! Timely resident hit confirmed.**

---

## 8. HBM Lifecycle and Eviction Behavior

### Expert State Machine
An expert transitions across five discrete states during simulation:

```
                  +-------------------+
                  |      ABSENT       |
                  +---------+---------+
                            |
           +----------------+----------------+
           | Prefetch Dispatch               | Demand Miss
           v                                 v
   +---------------+                 +---------------+
   |   IN_FLIGHT   |                 |   IN_FLIGHT   |
   | (SPECULATIVE) |                 |   (DEMAND)    |
   +-------+-------+                 +-------+-------+
           |                                 |
           | Transfer Completes              | Transfer Completes
           v                                 v
   +---------------+  Demand Hit     +---------------+
   |   RESIDENT    |---------------->|   RESIDENT    |
   | (SPECULATIVE) | (Promotion)     |   (DEMAND)    |
   +-------+-------+                 +-------+-------+
           |                                 |
           | Evicted (Speculative LRU)       | Evicted (Demand LRU)
           v                                 v
   +---------------+                 +---------------+
   |    ABSENT     |                 |    ABSENT     |
   +---------------+                 +---------------+
```

### Invariants Enforced in Source Code
1. **PABI Quarantine Buffer (`device.cpp:108-116`, `1348-1379`):** Speculative prefetch insertions are quarantined in `speculative_lru_list`, strictly limited to $16\text{ GB}$ (20% of 80 GB HBM). If the speculative buffer is full, the oldest speculative expert is evicted. **Speculative experts can never evict demand-verified experts.**
2. **Belády Pinning (`device.cpp:1428-1438`, `1359-1378`):** Admitted candidates receive a `PinIntent` with expiration stage $T = L + H$. Once resident, `is_pinned = true` prevents eviction during intervening stages. At stage boundaries, `expire_pins_for_stage(current_stage)` unpins experts whose deadline has passed.
3. **Demand Promotion (`device.cpp:700-750`):** When a demand access hits a speculative resident expert, `promote_speculative_to_demand()` removes it from `speculative_lru_list` and moves it to `demand_lru_list`, clearing its pin and conferring permanent LRU protection.
4. **Duplicate Suppression (`device.cpp:1228-1238`, `1451-1461`):** Before submitting a prefetch, `Device` verifies that the expert is neither resident (`has_resident_expert`) nor already in transit (`inflight_registry`). Duplicates are skipped without consuming memory bandwidth.

---

## 9. Test Coverage and Historical Experiment Reconciliation

The repository contains an extensive suite of pre-built verification executables in `LLMSimulator/build/`:

| Executable Path | Test Scope | Assertions / Cases | Historical Status | Current Verdict |
|---|---|---|---|---|
| `build/test_ddr5_transfer_engine` | Single-channel DDR5 bus, serialization, preemption, bandwidth | 12 Test Cases | 12/12 PASSED | **VERIFIED** |
| `build/test_forward_lookahead_prefetch` | Forward causality ($L \to L+H$), boundary suppression, priority order | 10 Test Cases | 10/10 PASSED | **VERIFIED** |
| `build/test_phase2e2_remediation` | Stage decoupling (`current_moe_stage_id`), `ADMITTED` filter, 12-col CSV | 6 Tests (74 Assertions) | 6/6 PASSED | **VERIFIED** |
| `build/test_phase2f_multiplier` | Synthetic compute window scaling ($M \in \{1, 4, 8, 12, 16\}$) | Multiplier Sweep | Complete | **VERIFIED** |
| `build/test_pabi_pinning` | Belády pinning, deadline expiration, pin collisions, capacity | 25 Test Cases | 25/25 PASSED | **VERIFIED** |
| `build/test_pabi_quarantine` | Speculative buffer isolation ($16\text{ GB}$ cap), eviction segregation | Full Suite | PASSED | **VERIFIED** |
| `build/test_demand_transfer_integration` | In-flight attachment, priority promotion, remaining duration | Full Suite | PASSED | **VERIFIED** |
| `build/test_prefetch_dispatch_coordination` | Prediction engine to candidate queue to device dispatch flow | Full Suite | PASSED | **VERIFIED** |
| `build/test_prefetch_scheduler` | Candidate queue policies (`DEADLINE_SCORE_RANK`, overflow) | Full Suite | PASSED | **VERIFIED** |
| `build/test_live_prediction` | LivePredictionEngine metric accumulators (P, R, F1, MRR, nDCG) | Full Suite | PASSED | **VERIFIED** |
| `build/test_production_prefetch_integration` | End-to-end integration test with DeepSeek-V3 configuration | End-to-End | PASSED | **VERIFIED** |
| `build/calibrate_ddr5_transfers` | Calibration test deriving 22.578 GB/s bandwidth | Standalone | PASSED | **VERIFIED** |
| `build/replay_predictors` | C++ vs Python trace replay parity for ST-MoE and CSTP | Full Replay | PASSED | **VERIFIED** |

---

## 10. Confirmed Working Components

The following components are fully implemented, verified, integrated, and empirically validated:
1. **C++ Predictors (`VERIFIED`):** `STMoEPredictor` and `CSTPPredictor` produce identical candidate rankings to their Python counterparts and update online CCT and spatial tables causally.
2. **Forward-Lookahead Trigger (`VERIFIED`):** Dispatches candidates for target stage $T = L + H$ post-routing without temporal leakage.
3. **Prefetch Candidate Queue (`VERIFIED`):** Enforces budget $B$, priority sorting (`DEADLINE_SCORE_RANK`), duplicate rejection, and `ADMITTED` state filtering.
4. **Calibrated DDR5 Transfer Engine (`VERIFIED`):** Discrete-event simulation accurately enforcing physical bus serialization ($22.578\text{ GB/s}$) and preemption.
5. **In-Flight Demand Coordination (`VERIFIED`):** Demands attach to in-flight transfers, reducing stall to remaining duration.
6. **PABI Quarantine Buffer (`VERIFIED`):** Isolates speculative transfers within $16\text{ GB}$, preventing demand cache pollution.
7. **Belády Pinning Engine (`VERIFIED`):** Protects admitted candidates from eviction across intermediate stages until target deadline.
8. **12-Column CSV Telemetry (`VERIFIED`):** `expert_load.csv` accurately records per-transfer durations, hit/miss classifications, and `transfer_id`.

---

## 11. Defects, Incomplete Components, and Traps

### Traps and Architectural Bottlenecks
1. **The Default Configuration Trap (`ISSUE`):**
   - In `src/prefetch/prefetch_config.h:127`, `PrefetchConfig::scheduling_mode` defaults to `SchedulingMode::LEGACY_CURRENT_STAGE`.
   - Running the simulator without explicitly specifying `scheduling.scheduling_mode: forward_lookahead` in YAML or setting `PREFETCH_SCHEDULING_MODE=forward_lookahead` silently reverts the system to the legacy co-located dispatch defect ($L \to L$), resulting in $0$ overlap and $0$ timely hits.
2. **Physical Bandwidth Bottleneck at Single-Batch Decode (`ARCHITECTURAL LIMITATION`):**
   - At batch size 1 ($M = 1.0\times$), single-token GEMM compute duration is $\sim 0.183\text{ ms}$ per stage. An $88.08\text{ MB}$ expert requires $3.901\text{ ms}$ to transfer over DDR5.
   - The available compute window for $H=2$ ($0.366\text{ ms}$) provides only $\sim 9.4\%$ of the necessary transfer time. While prefetching hides $0.366\text{ ms}$ of stall via in-flight attachment, **timely resident hits cannot physically occur at batch size 1 under DDR5 bandwidth limits**. Timely hits require either wider horizons ($H \ge 22$), batched inference ($M \ge 12\times$), or faster interconnects (e.g., PCIe Gen5 / CXL).
3. **Queue Congestion Under Fixed Candidate Budget $B = 4$ (`ISSUE`):**
   - Dispatched candidates serialize on the single DDR5 bus. A budget of $B=4$ queues $15.60\text{ ms}$ of work per stage. Lower-ranked candidates suffer up to $11.7\text{ ms}$ of queue waiting time, causing them to arrive late even when the compute window expands. Dynamic budget throttling is required to prevent channel saturation.

---

## 12. Recommended Next Implementation Phase

To build upon the verified forward-lookahead foundation, the next implementation phase should execute the following targeted enhancements:
1. **Default Configuration Hardening:** Change the default value of `PrefetchConfig::scheduling_mode` in `src/prefetch/prefetch_config.h` from `LEGACY_CURRENT_STAGE` to `FORWARD_LOOKAHEAD`.
2. **Dynamic Channel-Aware Budget Throttling:** Implement bandwidth-aware candidate admission in `PrefetchCandidateQueue` that dynamically limits budget $B$ based on the available compute window $\Delta T_{\text{comp}}$ to eliminate queue serialization delays.
3. **Multi-Token Batched Campaign:** Execute production evaluation runs with realistic batch sizes ($B_{\text{seq}} \in \{8, 16, 32\}$) where GEMM compute durations naturally exceed $3.901\text{ ms}$, demonstrating sustained timely resident hits and overall end-to-end token latency speedups without synthetic multipliers.
