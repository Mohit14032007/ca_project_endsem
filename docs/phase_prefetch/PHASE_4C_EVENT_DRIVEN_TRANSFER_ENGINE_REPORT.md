# Phase 4C Implementation Report: Calibrated Event-Driven DDR5 Transfer Engine

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Authoritative Baseline Commit:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Date:** November 2, 2026  
**Status:** **PASS** (12/12 Deterministic Unit Tests Passed, Total Host Execution Time: **0.074 ms**)  
**Deliverable:** `docs/phase_prefetch/PHASE_4C_EVENT_DRIVEN_TRANSFER_ENGINE_REPORT.md`

---

## 1. Executive Summary & Architectural Overview

Phase 4C implements the core discrete-event transfer engine for the hybrid DDR5 off-chip memory subsystem, resolving the findings of the Phase 4B calibration campaign and Phase 4B.1 forensic audit.

The engine models a **serialized, single-server transfer channel** calibrated directly from cycle-accurate Ramulator2 DDR5-3200BN measurements. It enforces **strict software-level demand priority** and **work-preserving preemption** of speculative prefetch transfers without relying on tick-by-tick simulation or host wall-clock delays.

```
 +-----------------------------------------------------------------------------------+
 |                             Phase 4C Architecture                                 |
 +-----------------------------------------------------------------------------------+
                                          |
               +--------------------------+--------------------------+
               |                                                     |
               v                                                     v
   +-----------------------+                             +-----------------------+
   |  Demand Request Queue | (Strict High Priority)      | Prefetch Queue (FIFO) |
   +-----------------------+                             +-----------------------+
               |                                                     |
               +--------------------------+--------------------------+
                                          |
                                          v
                         +---------------------------------+
                         |      Channel Arbiter / DES      |
                         |  - Single-server serialization  |
                         |  - Immediate preemption         |
                         |  - Generation-based validity    |
                         +---------------------------------+
                                          |
                                          v
                         +---------------------------------+
                         |      DDR5 Transfer Channel      |
                         |     Rate: 22.578 bytes / ns     |
                         |    (22.578 GB/s Saturation)     |
                         +---------------------------------+
                                          |
                                          v
                         +---------------------------------+
                         |        Transfer Events          |
                         |  (Min-Heap by timestamp_ns)     |
                         +---------------------------------+
```

### Scope Invariant Enforcement
1. **Core Transfer Engine Only:** This phase implements and verifies the transfer engine in isolation. No integration into the production demand path, HBM cache residency structures, or GPU execution pipeline was performed.
2. **Authoritative Baseline Preservation:** Baseline configurations, traces, and Phase 1–4B.1 deliverables remain 100% untouched.
3. **No Tick-by-Tick Stepping:** Transfers advance via exact discrete completion and preemption events calculated as `current_time + remaining_bytes / 22.578`.
4. **Strict Coding Invariant:** **Zero ternary operators** across all newly authored C++ sources, headers, build definitions, and documentation.

---

## 2. File Inventory & Modifications

| File Path | Operation | Description |
| :--- | :--- | :--- |
| `src/prefetch/ddr5_transfer_engine.h` | **Created** | Declares `DDR5TransferEngine`, `MemoryTransfer`, `TransferClass`, `TransferState`, `TransferEvent`, and `TransferEngineMetrics`. |
| `src/prefetch/ddr5_transfer_engine.cpp` | **Created** | Implements the discrete-event transfer engine, work-preserving preemption, generation-based stale-event rejection, and invariant checks. |
| `src/prefetch/CMakeLists.txt` | **Modified** | Added `ddr5_transfer_engine.cpp` to the `prefetch` OBJECT library target. |
| `eval/test_ddr5_transfer_engine.cpp` | **Created** | Standalone deterministic test harness implementing Tests A through L and conservation invariant assertions. |
| `eval/CMakeLists.txt` | **Modified** | Added build target `test_ddr5_transfer_engine` linked against `llm_system`. |
| `docs/phase_prefetch/PHASE_4C_EVENT_DRIVEN_TRANSFER_ENGINE_REPORT.md` | **Created** | Comprehensive implementation report and formal verification record. |

---

## 3. Calibration Constants & Unit Conversions

The transfer engine is parameterized by the empirical findings of Phase 4B and Phase 4B.1:

### 3.1 Rate & Dimension Specifications
- **Calibrated Saturated Bandwidth:**
  $$BW = 22.578 \times 10^9 \text{ bytes/second} = 22.578 \text{ bytes/nanosecond}$$
  *Unit Invariant Check:* $22.578\text{ B/ns}$ is used consistently. The inverted factor $0.022578\text{ B/ns}$ is strictly rejected.
- **DeepSeek-V3 Routed Expert Parameter Footprint:**
  $$M_{\text{expert}} = 88,080,384 \text{ bytes} \quad (\approx 88.08 \text{ MB FP16})$$
- **Single Projection Weight Tensor Footprint:**
  $$M_{\text{tensor}} = 29,360,128 \text{ bytes} \quad (\approx 29.36 \text{ MB FP16})$$
- **Tensors Per Expert:** Exactly $3$ projection matrices ($W_1$ gate, $W_2$ up, $W_3$ down).

### 3.2 Expected Isolated Transfer Latencies
- **Single Tensor Transfer Latency:**
  $$T_{\text{tensor}} = \frac{29,360,128 \text{ bytes}}{22.578 \text{ bytes/ns}} = 1,300,386.571 \text{ ns} \approx 1.30039 \text{ ms}$$
- **Single Complete Expert Transfer Latency:**
  $$T_{\text{expert}} = \frac{88,080,384 \text{ bytes}}{22.578 \text{ bytes/ns}} = 3,901,159.713 \text{ ns} \approx 3.90116 \text{ ms}$$

All internal engine timing calculations use simulated time in **nanoseconds (`double`)**.

---

## 4. Transfer Lifecycle & State Machine

### 4.1 State Definitions
The transfer engine tracks each memory transfer through five explicit lifecycle states:

```
        +---------+
        | Initial |
        +---------+
             |
             v
        +---------+                 (Preempted by Demand)
        | QUEUED  | <-----------------------------------+
        +---------+                                     |
             |                                          |
             | (Channel Idle & Selected)                |
             v                                          |
        +---------+                                     |
        | ACTIVE  | ------------------------------------+
        +---------+       (Paused state preserved)
             |
             +--------------------+
             |                    |
             v                    v
      +------------+       +-------------+
      | COMPLETED  |       |  CANCELLED  |
      +------------+       +-------------+
```

1. **`QUEUED`:** Transfer is admitted and waiting in either `demand_queue_` or `prefetch_queue_`. Transferred bytes = $0$.
2. **`ACTIVE`:** Transfer currently owns the DDR5 channel and receives $22.578\text{ B/ns}$. Exactly one transfer can be in this state at any instant.
3. **`PAUSED`:** Speculative prefetch transfer was preempted by an arriving demand request. Its progress is frozen and preserved in `paused_prefetch_queue_`.
4. **`COMPLETED`:** Terminal state reached when remaining bytes reach $0$.
5. **`CANCELLED`:** Terminal state if cancelled prior to completion.

### 4.2 Formal State Transition Table
| From State | To State | Trigger Condition | Valid? |
| :--- | :--- | :--- | :--- |
| `QUEUED` | `ACTIVE` | Channel idle and transfer selected by arbiter | **YES** |
| `QUEUED` | `CANCELLED` | Explicit cancellation while waiting | **YES** |
| `ACTIVE` | `PAUSED` | Demand request arrives while prefetch is active | **YES** |
| `ACTIVE` | `COMPLETED` | Simulated time reaches completion event timestamp | **YES** |
| `ACTIVE` | `CANCELLED` | Explicit cancellation while active | **YES** |
| `PAUSED` | `ACTIVE` | All queued demand requests cleared; prefetch resumes | **YES** |
| `PAUSED` | `CANCELLED` | Explicit cancellation while paused | **YES** |
| `COMPLETED` | *Any* | Any transition attempted on a completed transfer | **NO (Rejected)** |
| `CANCELLED` | *Any* | Any transition attempted on a cancelled transfer | **NO (Rejected)** |

---

## 5. Service Model, Demand Priority & Preemption Semantics

### 5.1 Single-Server Serialized Draining
Consistent with the Phase 4B empirical data:
- At any point in simulated time, at most **one transfer** receives channel bandwidth.
- The active transfer receives the full calibrated rate ($22.578\text{ B/ns}$).
- Queued and paused transfers receive $0\text{ B/ns}$.
- There is **no fluid-flow equal sharing** between active streams.

### 5.2 Priority Hierarchy
1. **Demand Requests:** Highest priority. Serviced in strict FIFO order.
2. **Paused Prefetch Resumptions:** Second priority. Resumed as soon as all demand requests complete.
3. **Queued Prefetch Candidates:** Third priority. Admitted only when demand and paused queues are empty.

### 5.3 Work-Preserving Preemption Algorithm
When a demand request arrives at simulated timestamp $t_{\text{demand}}$ while a speculative prefetch $P$ is active:
1. **Advance Elapsed Time:** The active transfer is updated to $t_{\text{demand}}$:
   $$\Delta t = t_{\text{demand}} - t_{\text{last\_start}}$$
   $$\Delta B = \Delta t \times 22.578$$
2. **Update Transferred & Remaining Bytes:**
   $$B_{\text{transferred}} \leftarrow B_{\text{transferred}} + \Delta B$$
   $$B_{\text{remaining}} \leftarrow B_{\text{total}} - B_{\text{transferred}}$$
3. **State Transition & Invalidation:**
   - If $B_{\text{remaining}} \le 10^{-9}\text{ bytes}$, the prefetch completed exactly at $t_{\text{demand}}$ and transitions to `COMPLETED`.
   - Otherwise, state transitions `ACTIVE -> PAUSED`.
   - The generation counter is incremented ($G \leftarrow G + 1$), rendering its pending completion event in the event queue invalid.
   - The paused prefetch is pushed to the head of `paused_prefetch_queue_`.
4. **Demand Takeover:** The arriving demand request immediately transitions to `ACTIVE` at $t_{\text{demand}}$.
5. **Resumption:** When all demand transfers complete at $t_{\text{demand\_done}}$, prefetch $P$ resumes from $B_{\text{remaining}}$:
   $$T_{\text{remaining\_duration}} = \frac{B_{\text{remaining}}}{22.578}$$
   $$T_{\text{new\_completion}} = t_{\text{demand\_done}} + T_{\text{remaining\_duration}}$$
   **Invariant:** Zero work is discarded. Transferred bytes are never retransmitted.

---

## 6. Discrete-Event Scheduling & Stale-Event Protection

### 6.1 Event Representation & Deterministic Ordering
Discrete events are scheduled in a min-heap priority queue (`std::priority_queue<TransferEvent, ..., std::greater<TransferEvent>>`).

```cpp
struct TransferEvent {
    uint64_t event_id = 0;
    double timestamp_ns = 0.0;
    TransferEventType type = TransferEventType::COMPLETION;
    uint64_t transfer_id = 0;
    uint64_t generation = 0;
};
```

#### Documented Tie-Breaking Rule on Shared Timestamps
When multiple events share the exact same timestamp ($|t_1 - t_2| < 10^{-9}\text{ ns}$):
1. **`COMPLETION` executes before `ARRIVAL`:** This ensures that an active transfer that completes at $t$ cleanly finalizes its remaining bytes to $0$ and vacates the channel before an arrival at $t$ begins.
2. **Monotonic Sequence Number:** If two events share timestamp and type, the tie is broken by ascending `event_id`.

### 6.2 Generation-Based Stale-Event Protection
In a discrete-event simulator with preemption, when an active transfer is paused, its previously scheduled completion event remains inside the priority queue. Removing arbitrary elements from a heap is inefficient.

The engine implements **generation-based validity verification**:
- Every transfer carries a monotonic `generation` integer.
- Whenever a transfer starts or is scheduled, its event records `event.generation = transfer.generation`.
- When a transfer is paused, cancelled, or completed, its `generation` increments.
- When an event pops from the priority queue:
  ```cpp
  if (transfer.state != TransferState::ACTIVE ||
      transfer.generation != event.generation ||
      transfer.transfer_id != active_transfer_id_) {
      metrics_.stale_events_ignored++;
      return true; // Discard stale event without side effects
  }
  ```
- This guarantees 100% mathematical immunity to stale completion events.

---

## 7. Numerical Precision Strategy

1. **Double-Precision Floating Point:** All timestamps, elapsed intervals, remaining bytes, and transferred bytes are maintained as standard IEEE 754 64-bit `double`.
2. **Epsilon Clamping:** When $B_{\text{remaining}} \le 10^{-9}\text{ bytes}$, it is clamped to exactly $0.0$ to prevent negative remaining work from floating-point roundoff.
3. **Conservation Tolerance:** The byte conservation equation:
   $$\left| B_{\text{total}} - (B_{\text{transferred}} + B_{\text{remaining}}) \right| < 10^{-6} \text{ bytes}$$
   holds with zero loss across all preemptions and fractional timestamp boundaries.

---

## 8. Unit Test Suite Execution & Results

The unit test suite (`eval/test_ddr5_transfer_engine.cpp`) was executed. All 12 tests passed deterministically:

| Test ID | Test Description | Configuration | Expected Latency | Actual Latency | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Test A** | Single demand transfer | 1 Demand ($88.08\text{ MB}$) at $t=0$ | $3.90116\text{ ms}$ | **$3.90116\text{ ms}$** | **PASS** |
| **Test B** | Two simultaneous demands | 2 Demands ($88.08\text{ MB}$ each) at $t=0$ | D1: $3.90116\text{ ms}$<br>D2: $7.80232\text{ ms}$ | **D1: $3.90116\text{ ms}$**<br>**D2: $7.80232\text{ ms}$** | **PASS** |
| **Test C** | Four simultaneous demands | 4 Demands ($88.08\text{ MB}$ each) at $t=0$ | D1: $3.90116\text{ ms}$<br>D2: $7.80232\text{ ms}$<br>D3: $11.70348\text{ ms}$<br>D4: $15.60464\text{ ms}$ | **D1: $3.90116\text{ ms}$**<br>**D2: $7.80232\text{ ms}$**<br>**D3: $11.70348\text{ ms}$**<br>**D4: $15.60464\text{ ms}$** | **PASS** |
| **Test D** | Two prefetch transfers | 2 Prefetches ($88.08\text{ MB}$ each) at $t=0$ | P1: $3.90116\text{ ms}$<br>P2: $7.80232\text{ ms}$ | **P1: $3.90116\text{ ms}$**<br>**P2: $7.80232\text{ ms}$** | **PASS** |
| **Test E** | Demand before prefetch service | Prefetch at $t=0$, Demand at $t=0$ | Demand: $3.90116\text{ ms}$<br>Prefetch: $7.80232\text{ ms}$ | **Demand: $3.90116\text{ ms}$**<br>**Prefetch: $7.80232\text{ ms}$** | **PASS** |
| **Test F** | Demand preempts active prefetch | Prefetch at $t=0$, Demand arrives at $t=1.0\text{ ms}$ | Demand: $4.90116\text{ ms}$<br>Prefetch: $7.80232\text{ ms}$ | **Demand: $4.90116\text{ ms}$**<br>**Prefetch: $7.80232\text{ ms}$** | **PASS** |
| **Test G** | Multiple demands during prefetch | Prefetch at $t=0$, D1 at $1\text{ ms}$, D2 at $2\text{ ms}$ | D1: $4.90116\text{ ms}$<br>D2: $8.80232\text{ ms}$<br>Prefetch: $11.70348\text{ ms}$ | **D1: $4.90116\text{ ms}$**<br>**D2: $8.80232\text{ ms}$**<br>**Prefetch: $11.70348\text{ ms}$** | **PASS** |
| **Test H** | Repeated preemption | Prefetch preempted at $1\text{ ms}$ and again $1\text{ ms}$ after resuming | 2 Preemptions, 2 Resumptions<br>Final: $11.70348\text{ ms}$ | **2 Preemptions, 2 Resumptions**<br>**Final: $11.70348\text{ ms}$** | **PASS** |
| **Test I** | Exact timestamp collision | Demand arrives at exact prefetch completion ($3.90116\text{ ms}$) | Prefetch: $3.90116\text{ ms}$<br>Demand: $7.80232\text{ ms}$ | **Prefetch: $3.90116\text{ ms}$**<br>**Demand: $7.80232\text{ ms}$** | **PASS** |
| **Test J** | Stale completion event | Prefetch preempted; old event pops during demand execution | Stale event ignored ($count=1$)<br>Prefetch ends at $7.80232\text{ ms}$ | **Stale event ignored ($count=1$)**<br>**Prefetch ends at $7.80232\text{ ms}$** | **PASS** |
| **Test K** | Fractional progress | Preempted at fractional $t=123,456.789\text{ ns}$ | Transferred: $2,787,407.38\text{ B}$<br>Delta $< 10^{-6}\text{ B}$ | **Transferred: $2,787,407.38\text{ B}$**<br>**Delta $< 10^{-6}\text{ B}$** | **PASS** |
| **Test L** | Invalid operations | Zero-byte, duplicate ID, cancellation | All invalid actions rejected<br>Cancellations verified | **Zero-byte rejected**<br>**Duplicate ID rejected**<br>**Cancellations verified** | **PASS** |

### Test Suite Execution Summary
- **Tests Passed:** **12 / 12 (100.0%)**
- **Tests Failed:** **0**
- **Host Execution Wall-Clock Time:** **0.074 ms** (74 microseconds)

---

## 9. Conservation Invariants Verification

The method `verify_conservation_invariants()` was executed across all test runs, confirming:
1. **Byte Conservation:** For every transfer $i$, $\left| B_{\text{total}} - (B_{\text{transferred}} + B_{\text{remaining}}) \right| < 10^{-6} \text{ bytes}$.
2. **No Double Completion:** No transfer completed more than once; `completed_transfer_count` strictly matches completed transfers.
3. **No Service While Paused/Queued:** Active transfer ID is $0$ when the channel is idle; non-zero only when state is `ACTIVE`.
4. **Single-Server Mutex:** Exactly one transfer is active at any point in simulated time.
5. **Demand Queue Invariant:** No prefetch transfer is active while `demand_queue_` has waiting entries.
6. **Non-Negative Work:** Remaining and transferred bytes are strictly non-negative.
7. **Service Time Fidelity:** Total active service time matches total bytes transferred divided by $22.578\text{ B/ns}$.

---

## 10. Known Limitations & Extrapolation Boundaries

1. **Ramulator-Calibrated Abstraction:** The engine models transfer latency at $22.578\text{ B/ns}$ based on Ramulator2 calibration. It does not invoke Ramulator2 cycle-by-cycle during inference.
2. **Continuous Preemption vs DRAM Burst Chop:** The engine models continuous byte-level preemption. In physical hardware, DRAM transactions occur in 64-byte burst chops. However, a 64-byte burst takes $< 2.83\text{ ns}$, which is negligible for LLM inference time scales.
3. **No Link Contention Outside DDR5:** The engine models the DDR5 memory channel bandwidth saturation limit. It assumes the PCIe Gen5 / CXL interconnect does not throttle below $22.578\text{ GB/s}$.

---

## 11. Git Working Tree State & Invariant Checks

### 11.1 Pre- and Post-Implementation Git Status
```
Changes not staged for commit:
	modified:   LLMSimulator/CMakeLists.txt
	modified:   LLMSimulator/config.yaml
	modified:   LLMSimulator/eval/CMakeLists.txt
	modified:   LLMSimulator/eval/test.cpp
	modified:   LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp
	modified:   LLMSimulator/src/CMakeLists.txt
	modified:   LLMSimulator/src/hardware/hardware_config.h
	modified:   LLMSimulator/src/hardware/layer_impl.cpp
	modified:   LLMSimulator/src/prefetch/CMakeLists.txt
	modified:   LLMSimulator/src/scheduler/scheduler.cpp
	modified:   LLMSimulator/src/scheduler/scheduler.h
	modified:   LLMSimulator/src/scheduler/sequence.cpp

Untracked files:
	LLMSimulator/eval/test_ddr5_transfer_engine.cpp
	LLMSimulator/src/prefetch/ddr5_transfer_engine.cpp
	LLMSimulator/src/prefetch/ddr5_transfer_engine.h
```

### 11.2 Verification of Non-Regression & Preserved Work
All existing unit tests and integration tests from earlier phases were re-executed:
- Phase 1 tests (`test_prefetch_framework`): **17 / 17 PASSED**
- Phase 2A tests (`test_phase2a_predictors`): **40 / 40 PASSED**
- Phase 2C tests (`test_live_prediction`): **2 / 2 PASSED**
- Phase 3 tests (`test_prefetch_scheduler`): **20 / 20 PASSED**
- Phase 4C tests (`test_ddr5_transfer_engine`): **12 / 12 PASSED**
- **Total Test Suite:** **91 / 91 PASSED (0 FAILURES)**

### 11.3 Strict Coding Invariant: Zero Ternary Operators
The entire codebase and documentation were scanned using Python AST and ternary regex verification:
- `src/prefetch/ddr5_transfer_engine.h`: **0 ternary operators**
- `src/prefetch/ddr5_transfer_engine.cpp`: **0 ternary operators**
- `eval/test_ddr5_transfer_engine.cpp`: **0 ternary operators**
- `docs/phase_prefetch/PHASE_4C_EVENT_DRIVEN_TRANSFER_ENGINE_REPORT.md`: **0 ternary operators**

---
**End of Phase 4C Implementation Report.**  
*Awaiting user review before initiating Phase 4D.*
