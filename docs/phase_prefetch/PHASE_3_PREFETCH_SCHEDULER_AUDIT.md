# Phase 3 Architecture Audit: Prefetch Candidate Queue and Scheduling Architecture

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Baseline Git Commit:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Date:** November 2, 2026  
**Status:** **AUDIT COMPLETE — READY FOR IMPLEMENTATION**

---

## 1. Executive Summary and Scope Boundaries

Phase 3 designs and implements the prefetch scheduling and candidate queue layer for LLMSimulator. This layer acts as the intermediate control buffer between the live prediction framework implemented in Phase 2C and the asynchronous transfer engine to be implemented in Phase 4.

### Strict Scope Boundaries (What Phase 3 Does and Does NOT Do)
- **IN SCOPE:**
  - Structured prefetch candidate representations.
  - Multi-request and multi-token isolation.
  - Exact composite candidate identity without lossy hashing.
  - Deterministic priority ordering based on target-stage deadline, prediction score, and rank.
  - Configurable queue capacity and overflow handling.
  - Queue-level duplicate candidate suppression.
  - Target-stage deadline reconciliation and candidate retirement upon routing resolution.
  - Request termination and token purge lifecycle events.
  - Independent instrumentation, metric tracking, and CSV/JSON logging.
  - Unit and integration testing across 20+ scenarios.
- **STRICTLY OUT OF SCOPE (Deferred to Phase 4 / Phase 5):**
  - No DDR5 memory requests or Ramulator cycle scheduling.
  - No asynchronous memory transfers or simulated completion events.
  - No HBM cache insertions, evictions, reservations, or pinning.
  - No modification to existing demand expert loading or LRU cache behavior.
  - No claims of memory-stall reduction or inference latency speedup.

---

## 2. Forensic Trace of the Live Routing Pipeline

### 2.1 Function-Level Call Sequence

A comprehensive trace of the active simulator codebase reveals the following chronological call flow during sequential decoding:

```text
Experiment Runner (eval/test_deepseekv3_mixtral_style.cpp)
    |
    v
Scheduler::Create() (src/scheduler/scheduler.cpp:20)
    |-- Scheduler::initExpertList() (src/scheduler/scheduler.cpp:22)
    |-- Scheduler::init_prediction_engine() (src/scheduler/scheduler.cpp:26)
    |       \-- LivePredictionEngine::LivePredictionEngine() (src/prefetch/live_prediction_engine.cpp:20)
    |               \-- PredictorRegistry::create_predictor() (src/prefetch/predictor_registry.cpp)
    \-- Scheduler::initRunningQueue() (src/scheduler/scheduler.cpp:56)
            \-- BatchedSequence::Create() (src/scheduler/sequence.cpp:59)
                    \-- Binds Scheduler pointer (scheduler = getPtr())
```

During each decoding step in `test_deepseekv3_mixtral_style.cpp`:

```text
Harness Loop (eval/test_deepseekv3_mixtral_style.cpp:395)
    |
    +-> Scheduler::updateScheduler(step_device_time) (src/scheduler/scheduler.cpp:463)
    |
    +-> Scheduler::fillSequenceQueue() (src/scheduler/scheduler.cpp:464)
    |
    +-> Scheduler::fillRunningQueue() (src/scheduler/scheduler.cpp:465)
    |
    +-> Scheduler::setMetadata() (src/scheduler/scheduler.cpp:410)
    |       \-- BatchedSequence::update_expert() (src/scheduler/sequence.cpp:153)
    |               |
    |               +-> [POINT 1 & 2] Hook 1: on_stage_routing_start() (src/scheduler/sequence.cpp:205)
    |               |       \-- LivePredictionEngine::on_stage_routing_start()
    |               |               \-- IPredictor::predict()
    |               |               \-- [NEW PHASE 3 HOOK]: PrefetchScheduler::enqueue_candidates()
    |               |
    |               +-> [POINT 3] Authoritative Router Resolution (src/scheduler/sequence.cpp:211-261)
    |               |       |-- Real trace lookup: scheduler->sequences_info[seq_id]->expert_list[table_idx]
    |               |       \-- Synthetic routing: scheduler->getRandomExpert(top_k)
    |               |
    |               \-> [POINT 4 & 5] Hook 2: on_stage_routing_complete() (src/scheduler/sequence.cpp:264)
    |                       \-- LivePredictionEngine::on_stage_routing_complete()
    |                               \-- IPredictor::update()
    |                               \-- [NEW PHASE 3 HOOK]: PrefetchScheduler::reconcile_stage()
    |
    +-> Cluster::run(metadata) (eval/test_deepseekv3_mixtral_style.cpp:420)
    |       \-- Device::run() -> Device::execution()
    |               \-- Linear::forward() (src/module/linear.cpp:48-56)
    |                       \-- Demand miss check: Device::is_expert_resident()
    |
    \-> [POINT 6] Run Termination / Sequence Completion
            \-- [NEW PHASE 3 HOOK]: PrefetchScheduler::purge_request() / finalize()
```

### 2.2 Critical Integration Points Identified

1. **Prediction Generation [Point 1]:**
   - **File:** `src/prefetch/live_prediction_engine.cpp:94-95`
   - **Function:** `LivePredictionEngine::on_stage_routing_start()`
   - **Mechanism:** `predictor_->predict(...)` returns `PredictionResult` containing candidate expert IDs and prediction scores.
2. **Candidate Queue Ingestion [Point 2]:**
   - **File:** `src/prefetch/live_prediction_engine.cpp:113`
   - **Opportunity:** Immediately after `predictor_->predict()` produces candidates, they must be formatted into `PrefetchCandidate` entries and offered to the candidate queue.
3. **Target Routing Revelation [Point 3]:**
   - **File:** `src/scheduler/sequence.cpp:227` and `254`
   - **Mechanism:** `actual_target_experts` vector is populated either from `scheduler->sequences_info` or `getRandomExpert()`.
4. **Target Deadline Identification [Point 4]:**
   - **Semantic:** A candidate predicted for target stage $S_{\text{target}}$ on token $T$ has a strict deadline: it must be resident before stage $S_{\text{target}}$ begins computation for token $T$.
   - When the execution loop enters stage $S_{\text{target}}$, the opportunity window for prefetching that stage has closed.
5. **Target Reconciliation and Candidate Retirement [Point 5]:**
   - **File:** `src/prefetch/live_prediction_engine.cpp:120`
   - **Function:** `LivePredictionEngine::on_stage_routing_complete()`
   - **Mechanism:** The authoritative target experts for `(request_id, token_id, target_stage_id)` are now revealed. All queued candidates for this target stage are evaluated:
     - Candidates in `actual_target_experts` are classified as `RECONCILED_HIT`.
     - Candidates not in `actual_target_experts` are classified as `EXPIRED_MISS`.
     - All matching candidates are retired from the active queue.
6. **Request Cleanup [Point 6]:**
   - **File:** `src/prefetch/live_prediction_engine.cpp:58-63`
   - **Function:** `LivePredictionEngine::on_request_start()` and sequence reset.
   - **Mechanism:** When a sequence terminates or resets, all active candidates belonging to `request_id` are removed to prevent stale entries across requests.

---

## 3. Forensic Audit of Existing Hardware & Memory Capabilities

A comprehensive read of `src/hardware/device.h`, `src/hardware/device.cpp`, `src/hardware/layer_impl.cpp`, and `src/module/linear.cpp` was conducted to determine what hardware queries are safely available.

### 3.1 Audit Findings Matrix

| Hardware Capability | Exists in Source? | Inspection Detail | Safety for Phase 3 |
| :--- | :---: | :--- | :--- |
| **Read-Only Residency Query** | **NO** | `Device::is_expert_resident()` exists, but it has severe side effects: it promotes the expert to the MRU position in `lru_expert_list` and logs `HIT`/`MISS` to `expert_cache.csv`. | **UNSAFE:** Calling it from the prefetch scheduler would corrupt LRU order and pollute simulation traces. Phase 3 must remain independent of residency. |
| **Reliable Expert Identity** | **YES** | Experts are uniquely identified by `(layer_id, expert_id)` where $0 \le \text{expert\_id} < N_{\text{routed\_expert}}$. | **SAFE:** Fully supported. |
| **Expert Footprint Size** | **YES** | For DeepSeek-V3, expert footprint is 88,080,384 bytes (~88.08 MB FP16). | **SAFE:** Available via `model_config`. |
| **In-Flight Transfer Tracking** | **NO** | The existing simulator has no in-flight transfer state machine or pending DDR5 queue. | **OUT OF SCOPE:** Belongs to Phase 4. |
| **Demand-Load Notification Hook**| **NO** | `Linear::forward()` triggers demand DRAM transfers directly without broadcasting events. | **OUT OF SCOPE:** Belongs to Phase 4. |
| **HBM Pinning / Reservation** | **NO** | `resident_experts` is an unmodified `std::unordered_map` iterator cache. No pin counts exist. | **OUT OF SCOPE:** Belongs to Phase 5. |

### 3.2 Residency Independence Mandate
Because `Device::is_expert_resident()` is inherently state-mutating, **Phase 3 will NOT query hardware residency**. The candidate queue will track candidate lifecycle and priority purely based on algorithmic and temporal scheduling properties. Phase 4 will introduce safe, read-only hardware coordination before issuing physical transfers.

---

## 4. Architectural Decisions for Phase 3

### 4.1 Queue Ownership and Lifecycle
- **Decision:** A new standalone class `PrefetchCandidateQueue` will be created under `src/prefetch/`.
- **Ownership:** `LivePredictionEngine` will own an instance of `PrefetchCandidateQueue`.
- **Scheduler Access:** `Scheduler` will provide access to the queue via `scheduler->get_prediction_engine()->get_candidate_queue()`.
- **Rationale:** This maintains strict separation of concerns:
  - `Predictor`: Pure mathematical algorithm.
  - `PrefetchCandidateQueue`: Pure priority scheduling, capacity, and admission data structure.
  - `LivePredictionEngine`: Coordinator connecting causal routing hooks to both predictor and queue.
  - `Hardware/Device`: Intact and untouched.

### 4.2 Candidate Identity and Composite Key
To prevent any possibility of hash collision or false candidate merging across requests, tokens, or pipeline stages, a 128-bit composite key will be used:

```cpp
struct CandidateKey {
    int request_id = 0;
    int token_id = 0;
    int target_stage_id = 0;
    int expert_id = 0;

    bool operator==(const CandidateKey& o) const {
        return request_id == o.request_id &&
               token_id == o.token_id &&
               target_stage_id == o.target_stage_id &&
               expert_id == o.expert_id;
    }
};

struct CandidateKeyHash {
    std::size_t operator()(const CandidateKey& k) const {
        std::size_t h1 = std::hash<int>{}(k.request_id);
        std::size_t h2 = std::hash<int>{}(k.token_id);
        std::size_t h3 = std::hash<int>{}(k.target_stage_id);
        std::size_t h4 = std::hash<int>{}(k.expert_id);
        std::size_t seed = h1;
        seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= h4 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};
```

This guarantees that:
- Request 1 predicting expert 15 at stage 4 will **never** collide with Request 2 predicting expert 15 at stage 4.
- Predicting expert 15 for stage 4 will **never** overwrite or merge with predicting expert 15 for stage 8.

### 4.3 Candidate Structure & Lifecycle States
The candidate record contains comprehensive metadata:

```cpp
enum class CandidateState {
    ADMITTED,       // Admitted to queue and pending scheduling
    READY,          // Prioritized and ready for future Phase 4 transfer selection
    RECONCILED_HIT, // Authoritatively verified as an actual routing hit at target stage
    EXPIRED_MISS,   // Reached target stage without being selected by router (false positive)
    CANCELLED,      // Cancelled due to request completion or explicit preemption
    EVICTED         // Evicted due to queue capacity overflow
};

struct PrefetchCandidate {
    uint64_t candidate_id = 0;          // Monotonic unique insertion identifier
    int request_id = 0;
    int token_id = 0;
    int source_stage_id = -1;
    int target_stage_id = 0;
    int expert_id = 0;
    int prediction_rank = 0;            // 1-indexed rank from predictor (1 to B)
    double prediction_score = 0.0;      // Associated prediction probability or score
    int prediction_horizon = 1;         // Horizon H
    uint64_t insertion_sequence = 0;    // Insertion sequence number for deterministic tie-breaking
    CandidateState state = CandidateState::ADMITTED;
    double admission_timestamp_ns = 0.0;
};
```

### 4.4 Scheduling Priority Ordering
The priority comparator orders candidates deterministically:
1. **Target-Stage Deadline Urgency (Primary):**
   Earlier target stages execute sooner.
   $$\text{Deadline} = \text{token\_id} \times 1{,}000{,}000 + \text{target\_stage\_id}$$
   Smaller deadline values have higher priority.
2. **Prediction Score (Secondary):**
   Higher score indicates higher predictor confidence and takes priority.
3. **Prediction Rank (Tertiary):**
   Lower rank ($1 < 2 < 3$) takes precedence over higher rank.
4. **Insertion Sequence (Quaternary):**
   Earlier admitted candidates take precedence (FIFO stability).
5. **Expert ID (Deterministic Tie-Breaker):**
   Smaller expert ID breaks any residual tie.

```cpp
struct CandidatePriorityComparator {
    bool operator()(const PrefetchCandidate& a, const PrefetchCandidate& b) const {
        int64_t deadline_a = static_cast<int64_t>(a.token_id) * 1000000LL + a.target_stage_id;
        int64_t deadline_b = static_cast<int64_t>(b.token_id) * 1000000LL + b.target_stage_id;
        if (deadline_a != deadline_b) {
            return deadline_a < deadline_b;
        }
        if (std::abs(a.prediction_score - b.prediction_score) > 1e-9) {
            return a.prediction_score > b.prediction_score;
        }
        if (a.prediction_rank != b.prediction_rank) {
            return a.prediction_rank < b.prediction_rank;
        }
        if (a.insertion_sequence != b.insertion_sequence) {
            return a.insertion_sequence < b.insertion_sequence;
        }
        return a.expert_id < b.expert_id;
    }
};
```

### 4.5 Duplicate Policy
When a candidate `(request_id, token_id, target_stage_id, expert_id)` is submitted:
- If an active entry already exists in the queue:
  - Under `RETAIN_HIGHEST_PRIORITY`: If the new submission has a strictly higher score (or better rank), the existing entry's score, rank, and source stage are updated. If the existing entry has equal or higher priority, the new submission is suppressed and counted in `duplicates_suppressed`.
  - Under `REJECT_DUPLICATE`: The new submission is immediately discarded and counted in `duplicates_suppressed`.

### 4.6 Capacity and Overflow Policy
- Configured via `scheduling.queue_capacity` (default: 1024 candidates).
- When the active queue reaches capacity:
  - Under `REJECT_NEW`: Incoming candidates are rejected, incrementing `capacity_rejected`.
  - Under `REJECT_LOWEST_PRIORITY`: The candidate is compared with the lowest-priority candidate currently in the queue. If the new candidate has higher priority, the lowest-priority entry is evicted (marked `EVICTED`, incrementing `overflow_evicted`) and the new candidate is admitted. Otherwise, the new candidate is rejected.

### 4.7 Target-Stage Reconciliation
When `on_stage_routing_complete()` executes for `(request_id, token_id, target_stage_id)` with `actual_target_experts`:
1. The queue locates all active candidates matching `(request_id, token_id, target_stage_id)`.
2. For each candidate:
   - If `expert_id` $\in \text{actual\_target\_experts}$, mark `RECONCILED_HIT` and increment `reconciled_hits`.
   - Else, mark `EXPIRED_MISS` and increment `expired_misses`.
3. The candidate is retired from the active priority index, reducing active queue occupancy.

### 4.8 Future Phase 4 Integration Interface
The queue will provide a non-destructive popping / selection method for Phase 4:
```cpp
std::vector<PrefetchCandidate> select_ready_candidates(int max_count);
void mark_candidate_dispatched(uint64_t candidate_id);
```
Phase 4 will consume candidates from `select_ready_candidates()`, check available DDR5 transfer bandwidth, and dispatch them to the event-driven transfer engine while maintaining complete preemption for demand misses.

---

## 5. Configuration Updates

Extend `PrefetchConfig` in `src/prefetch/prefetch_config.h` under the `scheduling` block:

```yaml
simulation:
  prefetching:
    enabled: true
    mode: "prediction_only"
    predictor:
      type: "st_moe"
    candidate_selection:
      budget: 4
    scheduling:
      prediction_horizon: 1
      queue_enabled: true
      queue_capacity: 1024
      priority_policy: "deadline_score_rank"
      duplicate_policy: "retain_highest_priority"
      overflow_policy: "reject_lowest_priority"
```

Default behavior when `queue_enabled: false`: The candidate queue is bypassed entirely, matching Phase 2C behavior bit-for-bit.

---

## 6. Implementation Roadmap

1. Implement `src/prefetch/prefetch_candidate_queue.h` and `src/prefetch/prefetch_candidate_queue.cpp`.
2. Extend `src/prefetch/prefetch_config.h` and `src/prefetch/prefetch_config.cpp` with queue settings.
3. Integrate queue into `src/prefetch/live_prediction_engine.h` and `live_prediction_engine.cpp`.
4. Expose queue accessor in `src/scheduler/scheduler.h`.
5. Implement verification suite `eval/test_prefetch_scheduler.cpp` covering tests 1–20.
6. Run live workloads and verify zero regression against Phase 2C.
7. Author `docs/phase_prefetch/PHASE_3_PREFETCH_SCHEDULER_REPORT.md`.

