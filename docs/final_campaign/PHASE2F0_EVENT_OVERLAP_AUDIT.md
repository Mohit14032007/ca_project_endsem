# Phase 2F.0 — Event Overlap and Queue Congestion Audit

## 1. Executive Summary

This read-only audit examines the concurrent execution mechanics between GPU computation and off-chip memory transfers in `LLMSimulator`, focusing on `DDR5TransferEngine` and the execution graph in `TopModuleGraph`. It verifies whether computation and DDR5 transfers genuinely execute concurrently in simulated time, evaluates the preemption and cancellation capabilities of the transfer queues, and investigates the structural dynamics of queue congestion under speculative dispatch.

### Key Audit Findings
1. **Genuine Concurrent Progression:** The simulator is an event-driven timing model where compute duration and memory transfer progress concurrently in simulated time. Compute duration advances the device clock, and calls to `run_until(device->get_time())` advance active DDR5 transfers by the exact elapsed simulated time.
2. **Zero Host-Level Blocking:** Computation does not block host CPU execution, spin-wait, or sleep. It computes an analytical latency and increments simulated device time.
3. **Full Lifecycle Cancellation Support:** `DDR5TransferEngine::cancel_transfer` supports cancellation across all non-terminal states: `QUEUED`, `PAUSED`, and `ACTIVE`. When an `ACTIVE` transfer is cancelled, bytes already transmitted are credited, the channel is marked idle, and the next eligible transfer is scheduled.
4. **Demand Preemption with Work Preservation:** Demand requests immediately preempt active speculative transfers via `preempt_active_prefetch()`. Transferred bytes are preserved, and the prefetch is placed in `paused_prefetch_queue_` for resumption once demand transfers complete.
5. **No Stage-Expiration Awareness:** `DDR5TransferEngine` operates as a strict FIFO queue that lacks awareness of transformer stages. Stale speculative transfers whose target stages have retired in the past remain in the queue and continue to consume DDR5 bus bandwidth unless explicitly cancelled.

---

## 2. Event Overlap Mechanics: Compute and Transfer Concurrency

To understand how computation and asynchronous prefetching overlap, consider the lifecycle of an expert transfer dispatched at stage $L$ targeting stage $L+2$ ($H=2$).

```
Timeline: Stage L Dispatch -> Stage L Compute -> Stage L+1 Compute -> Stage L+2 Demand
-----------------------------------------------------------------------------------------
Simulated Clock: T_0                                                       T_demand
GPU Compute:     [Stage L: 0.183 ms] ----------> [Stage L+1: 0.183 ms] ----> | Demand Check
                 |                                                           |
DDR5 Channel:    |================= Speculative Transfer ================>   | (3.901 ms)
                 |                                                           |
Progress:        Byte transfer advances during [Stage L + Stage L+1] compute |
-----------------------------------------------------------------------------------------
```

### 2.1 The Two Clocks: Device Time and Transfer Engine Time
The simulator maintains two distinct time counters:
1. `device->status.device_time`: Represents the simulated progress of the GPU compute cores and sequential module graph execution.
2. `ddr5_transfer_engine->current_time_ns_`: Represents the simulated progress of the physical DDR5 memory channel and its discrete event queue.

### 2.2 Progression Sequence
When stage $L$ executes:
1. **Prefetch Candidate Dispatch:** At timestamp $T_{\text{dispatch}} = \text{device->get\_time()}$, the prediction engine submits a prefetch transfer to `DDR5TransferEngine` via `submit_prefetch_transfer` (`src/hardware/device.cpp:1215-1260`).
2. **Transfer Engine Initialization:** If the channel is idle, `start_transfer()` (`src/prefetch/ddr5_transfer_engine.cpp:505-539`) transitions the transfer to `ACTIVE`, records `service_start_time_ns = T_{\text{dispatch}}`, and schedules a `TransferEventType::COMPLETION` event at:
   $$T_{\text{complete}} = T_{\text{dispatch}} + \frac{\text{total\_bytes}}{\text{bandwidth}} = T_{\text{dispatch}} + 3.90116\text{ ms}$$
3. **GPU Computation Advances Device Clock:** The GPU executes the remaining operators of stage $L$ and the operators of stage $L+1$ (LayerNorm, MLA Attention, Gate, Up, Down GEMMs, Act, Residuals). Each module pop adds its execution duration to `device->status.device_time`:
   $$T_{\text{device}} = T_{\text{dispatch}} + \Delta t_{\text{compute}}$$
4. **Synchronization at Demand Check:** When execution reaches stage $L+2$, `Linear::forward` (`src/module/linear.cpp:54`) calls:
   ```cpp
   device->process_completed_prefetches(device->get_time());
   ```
   This delegates to `DDR5TransferEngine::run_until(current_device_time)` (`src/prefetch/ddr5_transfer_engine.cpp:620-634`).
5. **Discrete Event Drain:** In `run_until(target_time_ns)`:
   - While `event_queue_.top().timestamp_ns <= target_time_ns`, `step()` pops each completion event, marks the transfer `COMPLETED`, updates metrics, and triggers `schedule_next_transfer()`.
   - If the active transfer has not completed, `update_active_transfer_progress(target_time_ns)` (`lines 375-419`) calculates the bytes transferred during the elapsed compute window:
     $$\Delta \text{bytes} = (T_{\text{device}} - T_{\text{last\_start}}) \times \text{bandwidth}$$
     and reduces `remaining_bytes` accordingly.

### 2.3 Verification: Does Extending Compute Duration Enable Physical Overlap?
**YES.** If the simulated compute duration $\Delta t_{\text{compute}}$ is extended such that:
$$\Delta t_{\text{compute}} \ge \frac{\text{total\_bytes}}{\text{bandwidth}} = 3.90116\text{ ms}$$
then $T_{\text{device}} \ge T_{\text{complete}}$. When `run_until(T_{\text{device}})` is called at the demand lookup, the completion event will execute, the transfer will be marked `COMPLETED`, and `device->process_completed_prefetches` will insert the expert into HBM.
The demand lookup then finds the expert in HBM, generating a **timely resident hit** and avoiding a demand stall.

### 2.4 Identification of Blocking Operations
The audit confirms that there are **no blocking operations** in the execution loop:
- No synchronous POSIX `sleep()`, `usleep()`, or `std::this_thread::sleep_for()`.
- No host pthread condition variables or busy mutex wait loops.
- No CUDA synchronization calls (`cudaDeviceSynchronize()`).
The execution engine is fully asynchronous and event-driven.

---

## 3. Queue Congestion and Channel Scheduling Audit

The behavior of `DDR5TransferEngine` under heavy speculative traffic was audited by inspecting `src/prefetch/ddr5_transfer_engine.h` and `ddr5_transfer_engine.cpp`.

### 3.1 Can Queued Speculative Requests Be Cancelled?
**YES.**
`DDR5TransferEngine::cancel_transfer(uint64_t transfer_id, double current_time_ns)` (`src/prefetch/ddr5_transfer_engine.cpp:306-373`) provides explicit cancellation functionality.
- It verifies the transfer exists and is not already `COMPLETED` or `CANCELLED`.
- It advances the engine to `current_time_ns`.
- It sets `transfer.state = TransferState::CANCELLED`.
- It increments `transfer.generation` to invalidate any pending discrete events in `event_queue_`.
- It increments `metrics_.cancelled_transfers`.

### 3.2 Is Cancellation Possible Only Before Physical Service Starts?
**NO. Cancellation is supported across all non-terminal states:**
1. **`QUEUED` State (`lines 333-355`):** The transfer is removed from `demand_queue_` or `prefetch_queue_`. It never touches the bus.
2. **`PAUSED` State (`lines 357-370`):** The transfer is removed from `paused_prefetch_queue_`.
3. **`ACTIVE` State (`lines 322-331`):**
   ```cpp
   if (transfer.state == TransferState::ACTIVE) {
       update_active_transfer_progress(current_time_ns);
       transfer.state = TransferState::CANCELLED;
       transfer.generation++;
       channel_busy_ = false;
       active_transfer_id_ = 0;
       metrics_.cancelled_transfers++;
       schedule_next_transfer();
       return true;
   }
   ```
   Transferred bytes up to `current_time_ns` are credited to metrics, the channel is immediately freed, and the next queued request is scheduled.

> [!IMPORTANT]
> Although `cancel_transfer()` is fully implemented and tested in the unit test suite, **it is never called by `Device` or `Linear` during simulation**. Consequently, speculative transfers that miss their target window are never purged.

### 3.3 Can Active Transfers Be Paused and Resumed?
**YES.**
Work-preserving preemption and resumption are implemented in `preempt_active_prefetch()` (`lines 421-455`) and `schedule_next_transfer()` (`lines 569-575`):
- When an active prefetch is preempted by a demand request, `update_active_transfer_progress()` records exactly how many bytes were transferred prior to preemption.
- The state transitions to `PAUSED`.
- The transfer is placed at the front of `paused_prefetch_queue_`.
- When the demand transfer finishes, `schedule_next_transfer()` inspects `paused_prefetch_queue_` with Priority 2 (immediately after demand requests, ahead of fresh prefetches) and resumes transmission of the remaining bytes without re-transmitting completed data.

### 3.4 Do Demand Requests Preempt Speculative Transfers?
**YES.**
In `DDR5TransferEngine::submit_transfer` (`lines 148-156`):
```cpp
if (transfer.transfer_class == TransferClass::DEMAND) {
    metrics_.total_demand_submitted++;

    // If an active prefetch is currently transmitting, preempt it immediately
    if (channel_busy_ && active_transfer_id_ != 0) {
        auto it = transfers_.find(active_transfer_id_);
        if (it != transfers_.end()) {
            if (it->second.transfer_class == TransferClass::PREFETCH) {
                preempt_active_prefetch(current_time_ns);
            }
        }
    }

    demand_queue_.push_back(tid);
}
```
Demand arrivals immediately preempt any active speculative transfer on the channel.

### 3.5 Is the Transfer Queue FIFO?
**YES.**
The queues in `DDR5TransferEngine` are implemented as `std::deque<uint64_t>`:
- `demand_queue_`: Serviced strictly FIFO (`front()` then `pop_front()`).
- `paused_prefetch_queue_`: Serviced LIFO/FIFO for resumption.
- `prefetch_queue_`: Serviced strictly FIFO (`front()` then `pop_front()`).
No priority reordering occurs within the prefetch queue based on target stage or urgency.

### 3.6 Can the Queue Distinguish Expired Target-Stage Requests?
**NO.**
`DDR5TransferEngine` stores `MemoryTransfer` objects containing `layer_id` and `expert_id`, but the engine itself has no awareness of the current logical stage of the model or which stages have already completed execution.
If stage 5 finishes execution while prefetch requests for stage 5 are still sitting in `prefetch_queue_`, the engine will continue to pull them from the queue and transmit them over DDR5 in future stages, wasting bandwidth on data that is already obsolete.

---

## 4. Architectural Summary: Queue Dynamics and Compute Scaling

The findings of this audit demonstrate that:
1. **Overlap is physically and logically supported:** When simulated compute time expands, the DDR5 transfer engine progresses concurrently via discrete event advancement.
2. **Queue oversubscription is the bottleneck:** At candidate budget $B=4$, four expert transfers ($15.60\text{ ms}$ of transmission) are queued every stage.
   Even if compute time is scaled by $12\times$ ($0.183 \times 12 = 2.198\text{ ms}$ compute per stage), $15.60\text{ ms}$ of work is generated every $2.198\text{ ms}$.
   Unless the compute multiplier is large enough to absorb all queued candidates ($M \ge 42.6\times$), or budget is throttled, FIFO queue delay will continue to cause candidates to arrive late.
3. **Cancellation is the missing link:** Because `cancel_transfer()` is already implemented in `DDR5TransferEngine`, connecting stage retirement to `cancel_transfer()` would eliminate obsolete requests and enable prefetching to operate efficiently at realistic compute multipliers.

