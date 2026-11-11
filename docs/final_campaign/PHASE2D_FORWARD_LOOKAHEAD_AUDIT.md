# Phase 2D: Forward-Lookahead Correctness & Timing Audit

**Authoritative Git Baseline Anchor:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Audit Status:** Complete & Verified  
**Date:** November 3, 2026  
**Document Location:** `docs/final_campaign/PHASE2D_FORWARD_LOOKAHEAD_AUDIT.md`  
**Classification Standard:** `VERIFIED`, `ISSUE`, `NEEDS CLARIFICATION`, `NOT IMPLEMENTED`, `IMPLEMENTED BUT NOT EVALUATED`

---

## 1. Executive Summary

Phase 2D performs a strictly read-only correctness and architectural audit of Phase 2C ("Correct Forward-Lookahead Prefetch Scheduling"). The primary objective is to verify that the temporal co-located scheduling defect identified in earlier audits has been eliminated, that no causal violations or information leakages exist, that transfer timing overlap behaves consistently with physical DDR5 serialization constraints, and that PABI quarantine and Belády pinning provide unbroken protection for prefetched experts.

### Summary of Audit Findings
1. **Causality & Ordering:** Fully `VERIFIED`. Source routing is immutable before prediction starts; predictions strictly forecast target stage $T = L + H$; no future routing information is leaked.
2. **Transfer Timing & Overlap:** `VERIFIED`. Single-candidate prefetches achieve genuine timely resident hits when intervening computation exceeds $3.901\text{ ms}$. Multi-candidate prefetch batches ($B \ge 2$) serialize on the DDR5 bus, meaning candidates ranked 2..B require proportionally larger overlap windows ($7.802\text{ ms}$, $11.703\text{ ms}$, $15.605\text{ ms}$).
3. **PABI & Pinning Protection:** `VERIFIED`. Pins assigned with target deadline $T = L + H$ remain active across intervening stages $L+1 \dots T-1$. PABI buffer isolation prevents premature eviction under intermediate capacity pressure.
4. **Metric Accounting:** `VERIFIED` with one `ISSUE`. Prefetch transfer lifecycles adhere to strict conservation laws. In `prefetch_effectiveness.csv`, demand accesses record `TIMELY_RESIDENT`, `LATE_IN_FLIGHT`, and `NOT_PREFETCHED`, but natural hits on previously resident demand-verified experts (`NATURAL_HIT`) are logged only in `expert_cache.csv`.

---

## 2. Technical Audit: Part A1 — Forward-Lookahead Causality

| Audit Item | Verification Target | Source Location | Audit Finding | Classification |
|---|---|---|---|---|
| **A1.1** | Source-stage routing is available before prediction | `src/scheduler/sequence.cpp:240-297` | Actual router evaluation produces `actual_target_experts` at lines 242-292 prior to invoking `on_stage_routing_complete` at line 295. | **VERIFIED** |
| **A1.2** | Predictor receives only observed source routing | `src/prefetch/live_prediction_engine.cpp:273-278` | Engine sets `source_stage_id = target_stage_id` and passes `source_experts = actual_target_experts` to `predictor_->predict(...)`. | **VERIFIED** |
| **A1.3** | Prediction target is exactly $L + H$ | `src/prefetch/live_prediction_engine.cpp:271-272` | `future_target_stage_id = target_stage_id + horizon_`. Tested for $H \in \{1, 2, 3\}$. | **VERIFIED** |
| **A1.4** | No future routing information accessed | `src/prefetch/st_moe_predictor.cpp:175-208`<br>`src/prefetch/cstp_predictor.cpp:223-258` | `predict()` queries only correlation tables (`cct_`, `spatial_counts_`) and history tables (`history_table_`). Future routing vectors are neither passed nor queried. | **VERIFIED** |
| **A1.5** | Dispatch occurs after source routing completes | `src/scheduler/sequence.cpp:299-307` | `dispatch_candidates(dev, cur_time)` is invoked after `on_stage_routing_complete`. At `on_stage_routing_start`, forward mode skips dispatch. | **VERIFIED** |
| **A1.6** | Actual router remains authoritative | `src/scheduler/sequence.cpp:252-265` | Model execution uses `actual_target_experts` directly from sequence trace/routing. Predicted candidates are dispatched purely as memory transfers and never modify sequence execution routing. | **VERIFIED** |
| **A1.7** | Consistency for $H \in \{1, 2, 3\}$ and boundary checks | `src/prefetch/live_prediction_engine.cpp:271-272`<br>`eval/test_forward_lookahead_prefetch.cpp:122-168` | Target stage equals $L + H$. When $L + H \ge S$ (where $S$ is stage count), boundary check suppresses dispatch. Verified in Test 2. | **VERIFIED** |
| **A1.8** | Predictor update ordering and isolation | `src/prefetch/live_prediction_engine.cpp:158-171`<br>`src/prefetch/live_prediction_engine.cpp:270-303` | Sequential ordering: (1) pending predictions targeting stage $L$ are updated with ground truth $L$, then (2) forward predictions for stage $L+H$ are computed using source $L$. Zero temporal leakage. | **VERIFIED** |

---

## 3. Technical Audit: Part A2 — Actual Transfer Timing & Overlap

### 3.1 Transfer Timeline & Mathematical Overlap Window
Under the calibrated DDR5 transfer engine (`ddr5_bandwidth_gbps = 22.578`):
$$\text{Single Expert Transfer Duration} = \frac{88,080,384\text{ bytes}}{22.578\text{ bytes/ns}} = 3,901,160\text{ ns} = 3.90116\text{ ms}$$

The execution timeline unfolds as follows:
1. **$T_{\text{dispatch}}$:** Stage $L$ completes router evaluation ($\approx 0.05 - 0.10\text{ ms}$ after stage start).
2. **$T_{\text{transfer\_start}}$:** First prefetch candidate enters active service immediately if DDR5 channel is idle.
3. **$T_{\text{transfer\_complete}}$:** First candidate completes at $T_{\text{dispatch}} + 3.90116\text{ ms}$.
4. **$T_{\text{demand}}$:** Stage $L + H$ evaluates routing and issues demand accesses:
   $$T_{\text{demand}} = T_{\text{dispatch}} + T_{\text{compute}}(L) + \sum_{k=1}^{H-1} \Big( T_{\text{stage}}(L+k) \Big) + T_{\text{routing\_eval}}(L+H)$$

### 3.2 Overlap Analysis Across Lookahead Horizons ($H=1, 2, 3$)
Assuming typical DeepSeek-V3 GEMM computation time of $T_{\text{compute}} \approx 1.80\text{ ms}$ per stage and routing evaluation of $0.05\text{ ms}$:

| Lookahead Horizon | Available Overlap Window $\Delta T_{\text{overlap}}$ | Transfer Duration ($B=1$) | Overlap Margin | Expected Outcome (Rank 1 Candidate) |
|---|---|---|---|---|
| **Legacy ($H=0$)** | $\approx 0.05\text{ ms}$ | $3.901\text{ ms}$ | $-3.851\text{ ms}$ | **0% Timely Hits** (99% late stall) |
| **Forward $H=1$** | $1.80 + 0.05 \approx 1.85\text{ ms}$ | $3.901\text{ ms}$ | $-2.051\text{ ms}$ | **Late Attachment** (Stall reduced by $\sim 47\%$) |
| **Forward $H=2$** | $1.80 + 1.85 + 0.05 \approx 3.70\text{ ms}$ | $3.901\text{ ms}$ | $-0.201\text{ ms}$ | **Near Timely** (Stall reduced by $\sim 95\%$) |
| **Forward $H=3$** | $1.80 + 1.85 + 1.85 + 0.05 \approx 5.55\text{ ms}$ | $3.901\text{ ms}$ | $+1.649\text{ ms}$ | **100% Timely Resident Hit** (0 ns stall) |

### 3.3 Channel Contention & Candidate Serialization
A critical finding of this audit is that **DDR5 transfers are serialized across candidates**:
- When budget $B=4$ candidates are enqueued at stage $L$:
  - Candidate 1 finishes at: $T_{\text{dispatch}} + 3.901\text{ ms}$
  - Candidate 2 finishes at: $T_{\text{dispatch}} + 7.802\text{ ms}$
  - Candidate 3 finishes at: $T_{\text{dispatch}} + 11.703\text{ ms}$
  - Candidate 4 finishes at: $T_{\text{dispatch}} + 15.605\text{ ms}$
- **Finding:** Under horizon $H=2$ or $H=3$, only Candidate 1 (and potentially Candidate 2 if batch size/compute increases) can complete in time to produce a `TIMELY_RESIDENT` hit. Candidates 3 and 4 will inevitably be `LATE_IN_FLIGHT` or complete after target stage execution (`UNUSED`).
- **Classification:** `VERIFIED`. The system behaves realistically according to physical memory bandwidth limits without unphysical parallel DDR5 buses.

---

## 4. Technical Audit: Part A3 — PABI Quarantine and Belády Pinning

| Subsystem Component | Source Reference | Mechanism Audited | Verification Status |
|---|---|---|---|
| **Speculative Insertion** | `src/hardware/device.cpp:1324-1333` | Completed prefetch enters `resident_experts` with classification `SPECULATIVE` and is enqueued in `speculative_lru_list`. Speculative bytes tracked in `resident_speculative_bytes`. | **VERIFIED** |
| **Pin Assignment** | `src/hardware/device.cpp:1342-1353` | When prefetch transfer completes, if `has_pin_intent == true` and `pin_expiration_stage_id >= current_stage_id`, sets `is_pinned = true` and records expiration stage. | **VERIFIED** |
| **Cross-Stage Protection** | `src/hardware/device.cpp:780-820` | During intermediate stages $L+1 \dots L+H-1$, speculative buffer eviction scans `speculative_lru_list` and skips pinned entries (`pinned_eviction_skipped++`), evicting only unpinned speculative entries. | **VERIFIED** |
| **Demand Promotion** | `src/hardware/device.cpp:592-658` | When stage $L+H$ issues demand access, `is_expert_resident` calls `promote_speculative_to_demand()`. It clears pin metadata, transitions classification to `DEMAND_VERIFIED`, moves entry to `demand_lru_list`, and records `timely_resident_hits++`. | **VERIFIED** |
| **Deadline Expiration** | `src/hardware/device.cpp:550-575` | At stage completion in `sequence.cpp:315`, `expire_pins_for_stage(cur_layer)` expires pins with `pin_expiration_stage_id <= cur_layer`. Unpromoted entries become unpinned and eligible for eviction. | **VERIFIED** |
| **Late Transfer Arrival** | `src/hardware/device.cpp:1335-1340` | If transfer finishes after the target stage has already passed (`pin_expiration_stage_id < current_stage_id`), the expert enters unpinned and counts as `pin_expirations++`. | **VERIFIED** |

---

## 5. Technical Audit: Part A4 — Metric Correctness & Reconciliation

### 5.1 Prefetch Transfer Lifecycle Conservation Law
Every prefetch transfer submitted to `DDR5TransferEngine` is tracked in `Device::speculative_transfers` and is assigned exactly one mutually exclusive outcome upon simulation completion:
$$\text{Submitted Transfers} = \text{DEMANDED} + \text{UNUSED} + \text{EVICTED\_BEFORE\_DEMAND} + \text{STILL\_IN\_FLIGHT\_AT\_END} + \text{CANCELLED}$$

- `DEMANDED`: Either promoted upon timely arrival or attached while in flight.
- `UNUSED`: Completed into speculative buffer but never demanded by target stage before pin expired.
- `EVICTED_BEFORE_DEMAND`: Evicted from speculative buffer under buffer capacity pressure before target stage demand.
- `STILL_IN_FLIGHT_AT_END`: Submitted during late stages that remained active/queued when the simulation terminated.
- `CANCELLED`: Cancelled due to sequence preemption or duplicate suppression.
- **Audit Result:** Verified in `device.cpp:1480-1501` and `eval/test_forward_lookahead_prefetch.cpp:473-524`. Conservation holds with 0 discrepancies. Classification: `VERIFIED`.

### 5.2 Demand Outcome Partition & The Natural Hit Gap
Every expert requested during MoE execution by `sequence.cpp` falls into one of four physical states:
$$\text{Total Demands} = \text{NATURAL\_HIT} + \text{TIMELY\_RESIDENT} + \text{LATE\_IN\_FLIGHT} + \text{NOT\_PREFETCHED}$$

- **Audit Finding (Issue Identified):**
  - In `src/hardware/device.h:74-78`: `enum class DemandOutcome { TIMELY_RESIDENT, LATE_IN_FLIGHT, NOT_PREFETCHED };`.
  - In `src/hardware/device.cpp:686-704`: When a demand access hits a `DEMAND_VERIFIED` expert that was already resident in HBM (a natural hit), `is_expert_resident()` returns `true` and logs `"HIT"` in `expert_cache.csv`, but **does not push an entry to `demand_outcomes`**.
  - Consequently, `prefetch_effectiveness.csv` currently logs records for all demand misses and prefetch hits, but omits natural hits.
  - Total `DEMAND_ACCESS` row count in `prefetch_effectiveness.csv` equals $\text{Total Demands} - \text{Natural Hits}$.
  - **Classification:** `ISSUE` (Minor accounting omission in CSV emission; does not affect simulation execution or correctness, but should be documented or completed for seamless CSV summing).

---

## 6. Audit Conclusion & Readiness Decision

The Phase 2C forward-lookahead prefetch implementation is architecturally sound, causally strict, and correctly handles both timely resident completions and late in-flight attachments.

**Audit Recommendation:** **CONDITIONAL GO** (Fully ready for bounded smoke validation, with natural-hit accounting documented).

