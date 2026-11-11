# Phase 4D.2 — Prediction-Driven Prefetch Dispatch and In-Flight Demand Coordination: Implementation Report

## Executive Summary

Phase 4D.2 establishes end-to-end connectivity between the predictive scheduling framework (Phases 1–3) and the production event-driven memory transfer path (Phases 4C & 4D.1). Admitted prefetch candidates from `PrefetchCandidateQueue` are now asynchronously dispatched to the per-device `DDR5TransferEngine` without blocking GPU computation. A unified in-flight registry tracks expert transfers across both demand and speculative classes. When demand arrives for an in-flight prefetch, the transfer is promoted to demand priority in place, preserving transfer progress and eliminating duplicate transfers.

All 18 designated verification tests passed with 100% success. All 101 pre-existing regression tests from Phases 1–4D.1 continue to pass (119 total tests passed, 0 failed). Zero ternary operators were introduced across all codebases, headers, tests, and documentation.

---

## 1. Exact Files Modified and Created

### Modified Files:
1. `src/hardware/device.h`
   - Added `struct InflightExpertEntry` tracking transfer ID, class, dispatch time, and expert keys.
   - Added `struct PrefetchPipelineMetrics` tracking evaluated, dispatched, completed, promoted, and hit counters.
   - Added `inflight_registry` (`unordered_map<{layer, expert}, InflightExpertEntry>`).
   - Added `prefetched_resident_experts` (`unordered_set<{layer, expert}>`).
   - Declared `submit_prefetch_transfer`, `promote_inflight_to_demand`, `process_completed_prefetches`, and `dispatch_prefetch_candidates`.
   - Updated `reset_expert_cache()` to reset all prefetch state, metrics, and transfer registries.

2. `src/hardware/device.cpp`
   - Included `prefetch/prefetch_candidate_queue.h`.
   - Implemented `submit_prefetch_transfer`: validates residency and in-flight status, dispatches to `DDR5TransferEngine::submit_prefetch_id()`, and populates `inflight_registry`.
   - Implemented `promote_inflight_to_demand`: invokes `DDR5TransferEngine::promote_transfer_to_demand()`, updating transfer class in place.
   - Updated `submit_demand_transfer`: coordinates with `inflight_registry`; if an in-flight prefetch exists, promotes it to demand priority and returns the existing transfer ID with zero duplicate transfers.
   - Implemented `process_completed_prefetches`: advances engine clock to current device time, inserts completed transfers into HBM, records metrics, and purges in-flight tracking.
   - Implemented `dispatch_prefetch_candidates`: filters candidates by validity, HBM residency, and in-flight status, dispatches transfers, marks candidates dispatched, and records pipeline metrics.
   - Updated `is_expert_resident`: detects when a demand hits a resident prefetched expert, recording useful prefetch hits.
   - Updated `mark_expert_resident`: cleans up eviction entries from `prefetched_resident_experts`.
   - Updated `clear_inflight_expert` and `is_expert_inflight`: dual-maintains `inflight_registry` and `inflight_experts`.

3. `src/prefetch/ddr5_transfer_engine.h` & `src/prefetch/ddr5_transfer_engine.cpp`
   - Added `submit_prefetch_id(...)` returning newly allocated transfer ID.
   - Implemented `promote_transfer_to_demand(uint64_t transfer_id, double current_time_ns)`: promotes queued or active prefetches to demand priority while preserving transferred bytes, transfer state, and remaining duration.

4. `src/prefetch/live_prediction_engine.h` & `src/prefetch/live_prediction_engine.cpp`
   - Added `dispatch_candidates(std::shared_ptr<Device> device, double current_time_ns, int budget = -1)` delegating to device dispatch logic.

5. `src/module/linear.cpp`
   - Synchronized transfer progression before residency evaluation: called `device->process_completed_prefetches(device->get_time())` prior to checking `device->is_expert_resident()`.

6. `src/hardware/linear_impl.cpp`
   - In `LinearExecutionGPU`, called `device->process_completed_prefetches(start_time)` before residency check.
   - Handled demand latency calculation for promoted prefetches: `weight_memory_cost = tr->completion_time_ns - start_time` (guarded strictly by `if/else`, zero ternaries).

7. `eval/CMakeLists.txt`
   - Registered executable target `test_prefetch_dispatch_coordination`.

### Newly Created Files:
8. `eval/test_prefetch_dispatch_coordination.cpp`
   - 18 comprehensive, deterministic unit and integration test cases covering the entire prefetch dispatch, overlap, promotion, and timeline coordination lifecycle.

---

## 2. Actual Production Call Flow After Phase 4D.2

```
1. BatchedSequence::update_expert(num_expert, top_k, ...)
   │
   ├─► LivePredictionEngine::on_stage_routing_start(req, token, cur_layer)
   │     ├─► Predictor::predict(...) -> PredictionResult
   │     └─► PrefetchCandidateQueue::enqueue_candidates(...) [State: ADMITTED]
   │
   ├─► Router selects actual target experts
   │
   └─► LivePredictionEngine::on_stage_routing_complete(req, token, cur_layer, actual_experts)
         └─► PrefetchCandidateQueue::reconcile_stage(...) [State: RECONCILED_HIT or EXPIRED_MISS]

2. Prefetch Candidate Dispatch (Device Execution Phase)
   │
   ├─► Device::dispatch_prefetch_candidates(queue, current_device_time, budget)
   │     ├─► Device::process_completed_prefetches(current_device_time)
   │     ├─► queue->select_ready_candidates(budget)
   │     ├─► For each candidate:
   │     │     ├─► Check is_expert_resident(layer, expert) -> skip if hit
   │     │     ├─► Check is_expert_inflight(layer, expert) -> skip if already inflight
   │     │     ├─► Device::submit_prefetch_transfer(layer, expert, current_device_time)
   │     │     │     └─► DDR5TransferEngine::submit_prefetch_id(...) [State: QUEUED or ACTIVE]
   │     │     ├─► Track in inflight_registry[{layer, expert}]
   │     │     └─► queue->mark_candidate_dispatched(candidate_id) [State: READY]
   │     └─► Return dispatched_count

3. Module Execution & Time Progression (Linear::forward / LinearExecutionGPU)
   │
   ├─► Linear::forward:
   │     ├─► Device::process_completed_prefetches(device->get_time())
   │     │     ├─► DDR5TransferEngine::run_until(device->get_time())
   │     │     └─► Completed prefetches inserted into HBM via mark_expert_resident()
   │     │
   │     └─► Check is_expert_resident(layer, expert):
   │           ├─► TRUE: HBM hit! (useful_hits incremented, zero stall)
   │           └─► FALSE: Weight target set to OFFCHIP_DRAM -> demand miss
   │
   └─► LinearExecutionGPU:
         ├─► start_time = device->get_time()
         ├─► If demand miss:
         │     ├─► Device::process_completed_prefetches(start_time)
         │     ├─► If still miss:
         │     │     ├─► Device::submit_demand_transfer(layer, expert, start_time)
         │     │     │     ├─► Check inflight_registry:
         │     │     │     │     ├─► Found PREFETCH:
         │     │     │     │     │     ├─► DDR5TransferEngine::promote_transfer_to_demand(...)
         │     │     │     │     │     ├─► Update class to DEMAND
         │     │     │     │     │     └─► Return existing transfer ID (ZERO duplicate transfer)
         │     │     │     │     └─► Not Found:
         │     │     │     │           └─► Submit new demand transfer
         │     │     ├─► Device::wait_for_demand_transfer(tid)
         │     │     ├─► Charged stall latency = tr->completion_time_ns - start_time
         │     │     ├─► Device::mark_expert_resident(layer, expert, size)
         │     │     └─► Device::clear_inflight_expert(layer, expert)
         │     └─► status.device_time += weight_memory_cost
```

---

## 3. Where Candidate Dispatch Occurs and Why

Candidate dispatch is initiated when the `Device` is active and aware of its current simulation timeline:
- Candidates are admitted during `BatchedSequence::update_expert()` for target stage $T = cur\_layer + H$.
- Candidate dispatch can be invoked via `device->dispatch_prefetch_candidates(queue, current_time, budget)` or `scheduler->prediction_engine->dispatch_candidates(device, current_time)`.
- **Why this is safe:**
  1. Prediction generation and candidate admission for horizon $H \ge 1$ occur strictly before execution of future target stages.
  2. Dispatch evaluates candidates against the exact simulated device timestamp `current_time`.
  3. Pre-existing resident experts and in-flight transfers are filtered out before transfer submission.
  4. Marking candidates `READY` in the queue ensures deterministic dispatch without duplicate admission.

---

## 4. How Device Identity is Selected

Each GPU device in `Cluster` has a unique integer rank:
- In production, each `Device` maintains its own private `ddr5_transfer_engine`, its own private `resident_experts` hash map, and its own private `inflight_registry`.
- The `request_id` passed to the transfer engine uses `device_total_rank` to maintain device affinity.
- Transfers on Device 0 and Device 1 are completely isolated: prefetching on Device 0 does not consume Device 1 DDR5 channel bandwidth or occupy Device 1 HBM capacity.

---

## 5. Synchronization of Asynchronous Transfers with Device Time

In LLMSimulator, execution time is modeled as a monotonically increasing nanosecond clock: `device->status.device_time` (queried via `device->get_time()`).

### Progression Semantics:
1. Asynchronous prefetch submission registers a transfer at `device->get_time()` without blocking.
2. Transfer events advance when `process_completed_prefetches(current_device_time)` is invoked:
   - It calls `ddr5_transfer_engine->run_until(current_device_time)`.
   - Discrete events (completions, preemptions, resumptions) up to `current_device_time` are executed in strict timestamp order.
3. Overlap Support:
   - GPU computation duration from preceding layers advances `status.device_time`.
   - When the next layer begins, `process_completed_prefetches` processes transfers that finished during that compute interval.
   - If transfer duration $\le$ compute duration, the expert is already resident when queried, achieving **100% compute/prefetch overlap with 0 ns memory stall**.

---

## 6. How Completed Prefetches Enter HBM

When `process_completed_prefetches(current_device_time)` executes:
1. It queries `ddr5_transfer_engine` for all transfer IDs in `inflight_registry`.
2. For each transfer where `tr->is_completed() == true`:
   - Checks if the expert is already resident. If not, calls `mark_expert_resident(layer_id, expert_id, total_bytes)`.
   - `mark_expert_resident()` enforces standard LRU capacity: if cache capacity is exceeded, LRU experts are evicted.
   - The key `{layer_id, expert_id}` is inserted into `prefetched_resident_experts`.
   - Metric counters `prefetches_inserted_hbm` and `prefetch_transfers_completed` are incremented.
   - The entry is erased from `inflight_registry` and `inflight_experts`.
3. If a transfer was cancelled (`state == CANCELLED`), it is erased from `inflight_registry` without HBM insertion, ensuring **incomplete weights are never inserted into HBM**.

---

## 7. How the Unified In-Flight Registry Works

`inflight_registry` is defined as:
```cpp
std::unordered_map<std::pair<int, int>, InflightExpertEntry, PairHash> inflight_registry;
```
where `InflightExpertEntry` encapsulates:
- `uint64_t transfer_id`: engine transfer handle.
- `TransferClass transfer_class`: `DEMAND` or `PREFETCH`.
- `double dispatch_time_ns`: timestamp when transfer was submitted.
- `int layer_id`, `int expert_id`: composite expert identity.

### Coordination Rules:
- Keyed strictly by `{layer_id, expert_id}` (device-local, shared across tokens/requests).
- Any attempt to submit prefetch for an existing in-flight key returns 0 immediately.
- Any attempt to submit demand for an in-flight key attaches to the existing transfer ID.

---

## 8. Exact Behavior When Demand Arrives During Active or Queued Prefetch

### Scenario A: Demand Arrives While Prefetch is ACTIVE
- **Setup:** Prefetch for Layer 12, Expert 5 begins at $t = 0.0$ (size: 88,080,384 B, duration: ~3.90116 ms).
- **At $t = 1.0\text{ ms}$:** Router demands Layer 12, Expert 5.
- **Behavior:**
  1. `submit_demand_transfer` discovers `{12, 5}` in `inflight_registry`.
  2. Promotes transfer class from `PREFETCH` to `DEMAND` via `promote_transfer_to_demand()`.
  3. Progress is preserved: transferred bytes = $1,000,000 \times 22.578 = 22,578,000\text{ B}$.
  4. Remaining bytes = $88,080,384 - 22,578,000 = 65,502,384\text{ B}$.
  5. Completion time remains $3,901,159.71\text{ ns}$ ($3.90116\text{ ms}$).
  6. Demand attaches to transfer and waits: charged wait latency is **$2,901,159.71\text{ ns}$ ($2.90116\text{ ms}$)** instead of $3.90116\text{ ms}$.
  7. Transferred bytes remain exactly 88,080,384 B. Zero duplicate bytes.

### Scenario B: Demand Arrives While Prefetch is QUEUED
- **Setup:** Prefetch 1 is ACTIVE on channel. Prefetch 2 is QUEUED in `prefetch_queue_`.
- **Behavior:**
  1. Router demands Prefetch 2.
  2. `promote_transfer_to_demand()` removes Prefetch 2 from `prefetch_queue_` and inserts into `demand_queue_`.
  3. Because Prefetch 1 is speculative, Prefetch 1 is preempted to `PAUSED` state.
  4. Promoted demand Prefetch 2 immediately takes the channel and becomes `ACTIVE`.
  5. When Prefetch 2 finishes, Prefetch 1 resumes from its paused progress.

---

## 9. Evidence That No Duplicate Transfers Occur

Verified across Test 6, Test 7, Test 8, Test 9, Test 10, and Test 11:
1. In Test 6 & 8: Single transfer ID is returned to both prefetch dispatch and subsequent demand arrival.
2. In Test 9: `metrics.total_bytes_transferred == 88080384ULL` (exact size of 1 expert, 0 bytes duplicated).
3. In Test 10: Three simultaneous requests requesting `{2, 10}` receive the identical transfer ID. Total transfers submitted to engine = 1.
4. In Test 11: Demand arriving after transfer completion but before insertion processing attaches to the existing transfer ID (`metrics.total_demand_submitted == 0`).

---

## 10. All Test Cases and Actual Results

| Test # | Test Name | Target Behavior Verified | Result |
|---|---|---|:---:|
| 1 | `test_1_candidate_admission_and_dispatch` | Candidates admitted from queue transition to in-flight transfers | **PASS** |
| 2 | `test_2_prefetch_submission_asynchronous` | Non-blocking async submission; clock does not advance | **PASS** |
| 3 | `test_3_prefetch_completion_hbm_insertion` | Completing prefetch enters HBM and clears in-flight | **PASS** |
| 4 | `test_4_already_resident_expert_not_prefetched` | Resident experts are filtered out; 0 transfer submitted | **PASS** |
| 5 | `test_5_already_inflight_expert_not_prefetched` | In-flight duplicate prefetch is rejected and logged | **PASS** |
| 6 | `test_6_demand_arriving_active_prefetch_promotes` | Active prefetch promoted to demand; progress preserved | **PASS** |
| 7 | `test_7_demand_arriving_queued_prefetch_promotes` | Queued prefetch promoted to demand; preempts active prefetch | **PASS** |
| 8 | `test_8_remaining_transfer_time_charged_correctly` | Demand charged only remaining 2.90116 ms latency | **PASS** |
| 9 | `test_9_transfer_byte_count_not_duplicated` | Exact 88,080,384 bytes transferred; zero duplication | **PASS** |
| 10 | `test_10_multi_request_share_inflight_transfer` | 3 requests share 1 in-flight transfer | **PASS** |
| 11 | `test_11_completed_transfer_before_insertion_no_duplicate` | Completed transfer before insertion avoids second load | **PASS** |
| 12 | `test_12_candidate_expiration_safety` | Expired candidate miss does not abort active transfer | **PASS** |
| 13 | `test_13_failed_cancelled_transfer_never_inserts` | Cancelled transfer never inserts incomplete expert | **PASS** |
| 14 | `test_14_lru_behavior_preserved_after_prefetch` | Standard LRU eviction operates identically on prefetched experts | **PASS** |
| 15 | `test_15_device_isolation` | Device 0 and Device 1 remain strictly isolated | **PASS** |
| 16 | `test_16_original_blocking_ramulator_baseline_unchanged` | Default baseline flags and Ramulator path preserved | **PASS** |
| 17 | `test_17_demand_only_regression_phase4d1` | Demand-only mode from Phase 4D.1 continues to pass | **PASS** |
| 18 | `test_18_demand_preempts_unrelated_active_prefetch` | Demand preempts unrelated prefetch; prefetch resumes cleanly | **PASS** |

**Summary: 18 Passed, 0 Failed.**

---

## 11. Timing Calculations and Actual Measured Values

- **Channel Bandwidth:** $22.578\text{ GB/s} = 22.578\text{ B/ns}$ (calibrated Ramulator2 rate).
- **Expert Footprint:** $88,080,384\text{ bytes}$ ($3 \times 29,360,128\text{ B}$ projection tensors).
- **Single Transfer Duration:**
  $$\Delta t = \frac{88,080,384\text{ B}}{22.578\text{ B/ns}} = 3,901,159.713\text{ ns} \approx 3.90116\text{ ms}$$
- **Measured Active Prefetch Promotion at $t = 1.0\text{ ms}$:**
  - Progress at preemption: $1,000,000\text{ ns} \times 22.578\text{ B/ns} = 22,578,000\text{ bytes}$.
  - Remaining bytes: $88,080,384 - 22,578,000 = 65,502,384\text{ bytes}$.
  - Measured remaining wait time:
    $$\Delta t_{\text{wait}} = \frac{65,502,384}{22.578} = 2,901,159.713\text{ ns} \approx 2.90116\text{ ms}$$
  - Measured completion timestamp: $3,901,159.713\text{ ns}$.
  - Latency saved on demand: $1.0\text{ ms}$ ($25.63\%$ stall reduction).

---

## 12. Prefetch Metrics and Precise Definitions

| Metric Name | Type | Definition |
|---|---|---|
| `candidates_evaluated` | `uint64_t` | Total candidate recommendations inspected by dispatch engine. |
| `candidates_dispatched` | `uint64_t` | Candidates passing residency and inflight checks, submitted to transfer engine. |
| `prefetch_transfers_submitted` | `uint64_t` | Number of speculative prefetch transfer requests submitted to DDR5 engine. |
| `prefetch_transfers_completed` | `uint64_t` | Prefetch transfers that finished all bytes without cancellation. |
| `prefetches_inserted_hbm` | `uint64_t` | Prefetched experts successfully inserted into HBM resident cache. |
| `prefetch_useful_hits` | `uint64_t` | Demands arriving for an expert currently resident via prefetch. |
| `prefetch_promotions_to_demand` | `uint64_t` | In-flight prefetches promoted to demand upon actual router demand. |
| `duplicate_prefetches_prevented`| `uint64_t` | Candidate dispatches suppressed because expert was already in flight. |
| `resident_prefetches_skipped` | `uint64_t` | Candidate dispatches suppressed because expert was already resident. |
| `demand_inflight_attachments` | `uint64_t` | Demands attached to existing in-flight transfers (demand or prefetch). |
| `cancelled_prefetches` | `uint64_t` | Prefetch transfers cancelled before completion (e.g. request abort). |

---

## 13. Baseline Preservation and Complete Regression Suite

All 119 unit/integration tests across all phases pass with 100% success:
- `test_prefetch_dispatch_coordination`: 18/18 PASS (Phase 4D.2)
- `test_demand_transfer_integration`: 10/10 PASS (Phase 4D.1)
- `test_ddr5_transfer_engine`: 12/12 PASS (Phase 4C)
- `test_prefetch_scheduler`: 20/20 PASS (Phase 3)
- `test_live_prediction`: 20/20 PASS (Phase 2C)
- `test_phase2a_predictors`: 26/26 PASS (Phase 2A)
- `test_prefetch_framework`: 13/13 PASS (Phase 1)

---

## 14. Git Diff Summary

```
 LLMSimulator/eval/CMakeLists.txt                        |   4 +
 LLMSimulator/eval/test_prefetch_dispatch_coordination.cpp| 512 ++++++++++++++++++++
 LLMSimulator/src/hardware/device.cpp                    | 178 ++++++-
 LLMSimulator/src/hardware/device.h                      |  39 +-
 LLMSimulator/src/hardware/linear_impl.cpp               |  13 +-
 LLMSimulator/src/module/linear.cpp                      |   3 +
 LLMSimulator/src/prefetch/ddr5_transfer_engine.cpp      |  82 ++++
 LLMSimulator/src/prefetch/ddr5_transfer_engine.h        |   4 +
 LLMSimulator/src/prefetch/live_prediction_engine.cpp    |  11 +
 LLMSimulator/src/prefetch/live_prediction_engine.h      |   5 +
 10 files changed, 836 insertions(+), 15 deletions(-)
```

---

## 15. Remaining Limitations and Overlap Boundary Clarification

1. **Overlap Granularity:**
   - The simulator maintains a discrete nanosecond timeline where device execution advances per module/operator (e.g. Attention, LayerNorm, Linear).
   - DDR5 transfer progress is synchronized at operator/linear execution boundaries (`device->get_time()`).
   - Intra-GEMM sub-cycle DMA preemption is not modeled by the high-level LLMSimulator, but inter-layer compute/transfer overlap is strictly represented on the timeline.
2. **HBM Policy Status:**
   - Standard LRU remains the active replacement policy. Prefetched experts enter the MRU end of standard LRU.
   - PABI (Predictor-Aware Buffer Insertion), dual-list quarantine, and speculative pinning have **not** been implemented in this subphase and remain cleanly isolated for Phase 4D.3.

---

## Phase 4D.2 Conclusion: READY FOR PHASE 4D.3

Phase 4D.2 is fully complete and verified. As instructed, execution is halted here awaiting approval before Phase 4D.3.
