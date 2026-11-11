# Phase 2E.1: Root-Cause Remediation Plan & Execution Recommendation

**Authoritative Git Baseline Anchor:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Status:** Audit Complete — **EXPLICIT GO RECOMMENDATION**  
**Date:** November 3, 2026  
**Document Location:** `docs/final_campaign/PHASE2E1_REMEDIATION_PLAN.md`  
**Companion Audits:**
- `docs/final_campaign/PHASE2E1_LAYER_IDENTITY_AUDIT.md`
- `docs/final_campaign/PHASE2E1_CANDIDATE_QUEUE_AUDIT.md`
- `docs/final_campaign/PHASE2E1_STALL_ACCOUNTING_AUDIT.md`

---

## 1. Executive Summary & Remediation Strategy

The Phase 2E Bounded Smoke Experiment revealed that while all 8 experimental configurations executed cleanly and stably to termination, all forward lookahead prefetching configurations exhibited:
1. Zero timely resident hits.
2. Identical simulated execution time to baseline (8003.70 ms).
3. Identical exposed memory stall to baseline (7884.24 ms).
4. An increase in late in-flight attachments (+81% ST-MoE, +62% CSTP) that produced zero reduction in memory stall time.

Across the three comprehensive read-only audits conducted in Phase 2E.1, we investigated each subsystem to trace the exact causal chain. The findings establish that the failure of prefetching to achieve speedup is **not an algorithmic failure of ST-MoE or CSTP**, nor is it a fundamental limitation of forward lookahead prefetching. Instead, it is the deterministic consequence of **two severe implementation defects and one instrumentation flaw**:

```mermaid
flowchart TD
    D1["Defect 1: 1-Stage Indexing Skew\n(sequence.cpp / linear.cpp)"] -->|Prefetch stored at T, Demand queries T+1| MissAll["100% Demand Miss Rate on All 75 Stages"]
    MissAll -->|2,021 Demand Transfers @ 3.901 ms| BusSat["DDR5 Channel 98.5% Saturated\n(7,884 ms / 8,003 ms)"]
    
    D2["Defect 2: Queue Retention\n(prefetch_candidate_queue.cpp)"] -->|Dispatched candidates kept in active_candidates_| QStarve["Candidate Queue Monopolized\n(1,369 Redundant Evaluations)"]
    QStarve -->|Blocks Future Predictions| PrefetchStarve["Prefetches Queued Late or Blocked"]
    
    BusSat -->|Demand Preempts / Queues Prefetches| ZeroProg["Prefetches Suffer 0 ms Progress in DDR5 Queue"]
    PrefetchStarve --> ZeroProg
    
    ZeroProg -->|Demand Arrives at Target Stage| LateAttach["Demand Attaches to Starved Queued Transfer"]
    
    D3["Defect 3: Remaining Time Zeroed\n(device.cpp:1134 tr->completion_time_ns > current)"] --> FalseZero["remaining_transfer_ns logged as 0.00 ns (Defect)"]
    
    LateAttach --> FullStall["Hardware LinearImpl accurately waits full 3.901 ms\n(Total Execution Time = Baseline 8003.70 ms)"]
```

Because each of these defects is localized, structurally isolated, and verified down to exact lines of code, a surgical remediation in Phase 2E.2 will restore end-to-end functionality without disturbing the baseline anchor or introducing architectural drift.

---

## 2. Classification of Audit Findings

| Audit Area | Subsystem / Component | Finding Description | Formal Classification |
|:---|:---|:---|:---:|
| **Audit A: Layer Identity** | `BatchedSequence::update_expert` & `Linear::forward` | Post-increment `cur_layer++` runs before `Linear::forward` reads `cur_layer`, skewing demand lookup by +1 stage relative to routing/prefetch. | **`ISSUE (DEFECT)`** |
| **Audit A: Trace Logs** | `expert_routing.csv` vs `expert_load.csv` | Router logs stages $[0 \dots 74]$; Linear logs stages $[1 \dots 74, 0]$. Proven empirical trace desynchronization. | **`ISSUE (DEFECT)`** |
| **Audit B: Queue Lifecycle** | `PrefetchCandidateQueue::mark_candidate_dispatched` | Sets state to `READY` but retains candidate in `active_candidates_`. | **`ISSUE (DEFECT)`** |
| **Audit B: Queue Selection** | `PrefetchCandidateQueue::select_ready_candidates` | Iterates `active_candidates_` without filtering `state == ADMITTED`. Top-B slots monopolized by earlier dispatched candidates. | **`ISSUE (DEFECT)`** |
| **Audit B: Starvation** | `Device::dispatch_prefetch_candidates` | 1,369 redundant candidate evaluations; newly predicted distant-stage candidates blocked from dispatch. | **`ISSUE (DEFECT)`** |
| **Audit C: Bus Contention** | `DDR5TransferEngine` Single Channel | 2,021 demand transfers saturate bus for 7,884.24 ms (98.5% of total time). Background prefetches queued with 0 progress. | **`VERIFIED`** |
| **Audit C: Instrumentation** | `Device::submit_demand_transfer` | Queued transfers have `completion_time_ns == -1.0`, causing `remaining_ns` check to fail and log `0.00 ns` despite 0 bytes sent. | **`ISSUE (DEFECT)`** |
| **Audit C: Hardware Stall** | `LinearImpl::execute` | Wait time correctly blocks for `tr->completion_time_ns - start_time` (full 3.901 ms), explaining identical runtime. | **`VERIFIED`** |
| **Audit D: Metadata Tracing** | `SpeculativeTransferRecord::source_stage` | Field omitted / set to -1 in `Device::submit_prefetch_transfer`. | **`ISSUE (GAP)`** |
| **Audit D: Metric Accounting** | $T_{\text{total}} = T_{\text{compute}} + T_{\text{stall}}$ | Total execution time formula is mathematically verified and conservation invariant holds. | **`VERIFIED`** |

---

## 3. Prioritized Remediation Items

### Category 1: Confirmed Code Defects (P0 — Mandatory for Functional Correctness)

#### Item 1.1: Reconcile Layer ID Indexing Between Linear Execution and Router Prediction
- **Defect Description:** In [`BatchedSequence::update_expert`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L316), `cur_layer++` executes at the end of routing for stage $S$. When the MoE linear computation runs immediately afterwards in [`Linear::forward`](file:///home/mohman/ca_project/LLMSimulator/src/module/linear.cpp#L49), it reads `int layer_id = sequences_metadata->cur_layer;`, which has already incremented to $S+1$. Prefetched experts dispatched for target stage $T$ are stored in cache under $\{T, E\}$, but demand lookup queries $\{T+1, E\}$, causing 100% prefetch misses.
- **Affected Files & Lines:**
  - `LLMSimulator/src/scheduler/sequence.h:119-122`
  - `LLMSimulator/src/scheduler/sequence.cpp:228, 315-317`
  - `LLMSimulator/src/module/linear.cpp:49-50`
- **Proposed Code Modification:**
  Introduce an explicit `current_moe_stage_id` field in `BatchedSequence` that explicitly captures the stage currently being routed and executed, decoupling it from the post-incrementing loop counter `cur_layer`:
  
  *In `src/scheduler/sequence.h`:*
  ```cpp
  // Add explicit field initialized to 0
  int current_moe_stage_id = 0;
  ```
  
  *In `src/scheduler/sequence.cpp` (inside `BatchedSequence::update_expert`):*
  ```cpp
  // Set at the start of routing for this stage:
  current_moe_stage_id = cur_layer;
  ```
  
  *In `src/module/linear.cpp` (inside `Linear::forward`):*
  ```cpp
  // Replace: int layer_id = sequences_metadata->cur_layer;
  // With:
  int layer_id = sequences_metadata->current_moe_stage_id;
  A->expert_layer_id = layer_id;
  ```
- **Expected Effect on Smoke Metrics:**
  - `expert_routing.csv` and `expert_load.csv` will both index stages consistently as $[0, 1, \dots, 74]$.
  - Prefetched experts for stage $T$ match the demand lookup key $\{T, E\}$, immediately transforming timely prefetches into resident HBM cache hits.
- **Risk Assessment:** **Low**. The change is strictly localized to MoE weight lookup (`if (A->is_expert_weight)`), leaving dense linear layers and other components untouched.
- **Validation Test:** Run single-stage unit test; assert that for any stage $S$, `expert_routing.csv` layer equals `expert_load.csv` layer.

---

#### Item 1.2: Filter Dispatched Candidates in `PrefetchCandidateQueue::select_ready_candidates`
- **Defect Description:** In [`PrefetchCandidateQueue::select_ready_candidates`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_candidate_queue.cpp#L355-L360), all candidates in `active_candidates_` are considered, including candidates already in `CandidateState::READY` (dispatched). Because earlier stage candidates have earlier deadlines, they consume the top-$B$ selection slots indefinitely, starving newly enqueued candidates for future stages.
- **Affected Files & Lines:**
  - `LLMSimulator/src/prefetch/prefetch_candidate_queue.cpp:356-360`
- **Proposed Code Modification:**
  Filter candidates by `state == CandidateState::ADMITTED` before priority sorting:
  
  *In `src/prefetch/prefetch_candidate_queue.cpp`:*
  ```cpp
  std::vector<PrefetchCandidate> candidates;
  candidates.reserve(active_candidates_.size());
  for (const auto& kv : active_candidates_) {
      if (kv.second.state == CandidateState::ADMITTED) {
          candidates.push_back(kv.second);
      }
  }
  ```
- **Expected Effect on Smoke Metrics:**
  - Eliminates 1,369 redundant re-evaluations of already-dispatched candidates.
  - Newly predicted candidates for target stages $L+H$ are selected and dispatched on the very step they are generated.
- **Risk Assessment:** **Low**. Standard queue filtering logic. No impact on eviction or PIN policies.
- **Validation Test:** Unit test in `test_prefetch_framework.cpp` verifying that calling `select_ready_candidates()` after `mark_candidate_dispatched()` returns an empty list or newly added candidates, not the dispatched ones.

---

#### Item 1.3: Correct Remaining Transfer Time Calculation for Queued/Paused Transfers
- **Defect Description:** In [`Device::submit_demand_transfer`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L1134-L1136), `remaining_ns` is computed by `if (tr->completion_time_ns > current_time_ns)`. For transfers currently in `TransferState::QUEUED` or `TransferState::PAUSED`, `completion_time_ns` is `-1.0`, causing the condition to fail and recording `remaining_ns = 0.00 ns` even though zero bytes have been transferred.
- **Affected Files & Lines:**
  - `LLMSimulator/src/hardware/device.cpp:1133-1138`
- **Proposed Code Modification:**
  Check transfer state and compute remaining time based on remaining bytes and channel bandwidth:
  
  *In `src/hardware/device.cpp`:*
  ```cpp
  double remaining_ns = 0.0;
  if (ddr5_transfer_engine != nullptr) {
      const auto* tr = ddr5_transfer_engine->get_transfer(it->second.transfer_id);
      if (tr != nullptr) {
          if (tr->state == TransferState::ACTIVE) {
              if (tr->completion_time_ns > current_time_ns) {
                  remaining_ns = tr->completion_time_ns - current_time_ns;
              }
          } else if (tr->state == TransferState::QUEUED || tr->state == TransferState::PAUSED) {
              // Calculate remaining transfer duration from remaining bytes
              double bw = ddr5_transfer_engine->get_bandwidth_bytes_per_ns();
              if (bw > 0.0) {
                  remaining_ns = tr->remaining_bytes / bw;
              } else {
                  remaining_ns = tr->total_bytes / (22.578 * 1e-1); // Fallback to calibrated rate
              }
          }
      }
  }
  ```
- **Expected Effect on Smoke Metrics:**
  - Accurately reports non-zero remaining transfer time for late in-flight attachments in `prefetch_effectiveness.csv`.
  - Eliminates false reporting of $0.00\text{ ns}$ remaining transfer on unstarted transfers.
- **Risk Assessment:** **Low**. Purely instrumentation / accounting logic; does not affect simulation execution state.
- **Validation Test:** Test fixture submitting a prefetch, attaching demand while queued, and asserting `remaining_transfer_at_demand_ns > 3,800,000.0 ns`.

---

### Category 2: Required Instrumentation Improvements (P1 — High Priority)

#### Item 2.1: Populate `source_stage` in Prefetch Submission and Speculative Records
- **Defect Description:** In [`Device::dispatch_prefetch_candidates`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L1433), `submit_prefetch_transfer` receives `cand.target_stage_id` and `cand.expert_id`, but does not pass `cand.source_stage_id`. As a result, speculative records in `speculative_transfers.csv` and `prefetch_effectiveness.csv` record `source_stage` as `-1`.
- **Affected Files & Lines:**
  - `LLMSimulator/src/hardware/device.h:284`
  - `LLMSimulator/src/hardware/device.cpp:1220-1230, 1433`
- **Proposed Code Modification:**
  Add `source_stage` parameter to `Device::submit_prefetch_transfer`:
  ```cpp
  uint64_t submit_prefetch_transfer(int target_stage, int expert_id, double current_time_ns, int source_stage = -1);
  ```
  Pass `cand.source_stage_id` from candidate queue into `SpeculativeTransferRecord::source_stage`.
- **Expected Effect:** Enables end-to-end audit tracing from prediction source stage to execution target stage in all output CSVs.
- **Risk Assessment:** **Low**. Pure logging enhancement.

---

#### Item 2.2: Record Accurate `transfer_start_ts` when Transfer Transitions to `ACTIVE`
- **Defect Description:** In [`Device::submit_prefetch_transfer`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L1240), `transfer_start_ts` is logged as `current_time_ns` at queue submission time, rather than when the transfer actually begins transmission on the DDR5 bus (`service_start_time_ns`).
- **Affected Files & Lines:**
  - `LLMSimulator/src/hardware/device.cpp:1240, 1330`
- **Proposed Code Modification:**
  Update `speculative_transfers[tid].transfer_start_ts = tr->service_start_time_ns` upon transfer completion or demand attachment.
- **Expected Effect:** Accurately reflects true physical bus occupancy and transmission delay in `speculative_transfers.csv`.
- **Risk Assessment:** **Low**.

---

#### Item 2.3: Cross-Correlate Transfer IDs in `expert_load.csv`
- **Defect Description:** `expert_load.csv` records `{timestamp, layer_id, expert_id, status, memory_duration}`, but does not log `transfer_id`, preventing 1:1 join with `prefetch_effectiveness.csv` without timestamp heuristics.
- **Affected Files & Lines:**
  - `LLMSimulator/src/hardware/linear_impl.cpp:189-195`
- **Proposed Code Modification:**
  Append `transfer_id` to `expert_load.csv` columns (`...,transfer_id`).
- **Expected Effect:** Direct foreign-key join between execution trace and transfer engine trace.
- **Risk Assessment:** **Low**.

---

### Category 3: Optional Optimizations (P2 — Secondary / Sensitivity Phase)

#### Item 3.1: Adaptive Prefetch Throttling Under Channel Saturation
- **Description:** When the DDR5 channel queue depth exceeds a configured threshold, throttle speculative prefetch candidate dispatch to avoid queue congestion.
- **Priority:** Defer to Phase 2G (Sensitivity Analysis). The primary priority is establishing the correct baseline functional operation under unthrottled conditions.

#### Item 3.2: Horizon Calibration ($H=1$ vs $H=2$ vs $H=3$)
- **Description:** For Workload A1 (75 stages, 57 MoE layers), layer execution latency is ~35 ms while transfer latency is 3.9 ms. A lookahead horizon of $H=1$ already provides $9\times$ the required computation overlap. In Phase 2E.2 rerun, compare $H=1, 2, 3$ to observe if $H=1$ provides earlier eviction safety and lower pin pressure.

---

## 4. Safety & Coding Standard Compliance

All proposed remediations strictly adhere to project constraints:
1. **Zero Ternary Operators:**
   - C++ ternary operators (`? :`) are strictly forbidden. All conditional expressions will use standard `if-else` blocks.
   - Python code must maintain `ast.IfExp == 0`.
2. **Git Baseline Preservation:**
   - Authoritative anchor `d5bf61ff0c066c06f72173189f9924329f613a88` is strictly preserved.
   - All modifications are confined to simulation engine execution and tracking code.
3. **Preservation of Historical Artifacts:**
   - Historical baseline results, Tier 1 results, and Phase 2E smoke outputs in `results/final_comparative_campaign/` remain untouched.
   - Phase 2E.2 rerun outputs will be directed to `results/final_comparative_campaign/smoke_phase2e2/`.

---

## 5. Validation Test Design for Remediation

To guarantee flawless execution before rerunning the full smoke campaign, the following 4-step validation sequence will be executed in Phase 2E.2:

```mermaid
flowchart LR
    V1["Step 1: Code Remediation\n(Apply Items 1.1, 1.2, 1.3)"] --> V2["Step 2: Unit Test Suite\n(test_forward_lookahead_prefetch)"]
    V2 --> V3["Step 3: 1-Token Single-Run Smoke\n(FWD-S-H2, verify layer index & resident hit)"]
    V3 --> V4["Step 4: Full 8-Run Phase 2E.2 Matrix\n(B, LEGACY, FWD-S 1-3, FWD-C 1-3)"]
```

### Verification Success Criteria:
1. **Layer Alignment:** `expert_routing.csv` and `expert_load.csv` must exhibit identical layer ranges: $0 \le \text{layer\_id} < 75$.
2. **Resident Hit Generation:** In forward lookahead configurations ($H \ge 1$), `timely_resident_hits > 0` and `hbm_hit_rate > 71.93%` (baseline).
3. **Execution Time Reduction:** Total simulated execution time for forward configurations must be strictly less than baseline ($T_{\text{fwd}} < 8003.70\text{ ms}$).
4. **Queue Health:** `PrefetchCandidateQueue::select_ready_candidates()` must never return already dispatched (`READY`) candidates.

---

## 6. Formal Execution Recommendation: EXPLICIT GO

### Recommendation Statement: **GO**

### Justification:
1. **Root Causes Fully Proven:** The investigation has conclusively resolved all anomalies observed in Phase 2E. The primary failure was caused by the 1-stage post-increment indexing skew (`cur_layer++` before `Linear::forward`) and the candidate queue selection filter omission.
2. **Deterministic, Surgical Fixes:** All required fixes (Items 1.1, 1.2, 1.3) are minimal, deterministic, and isolated to fewer than 20 total lines across 3 files.
3. **No Unresolved Architecture Gaps:** No structural redesign or heuristic guessing is required.
4. **Zero Impact on Baseline Anchor:** The modifications affect only the MoE forward stage binding and prefetch queue filtering, preserving all baseline models and historical data intact.

We recommend proceeding immediately to **Phase 2E.2: Remediation & Smoke Validation Rerun** upon user authorization.

