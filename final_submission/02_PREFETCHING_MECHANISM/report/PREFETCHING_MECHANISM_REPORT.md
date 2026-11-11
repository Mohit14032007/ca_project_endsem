# Requirement 2: Predictor-Driven Asynchronous Prefetching Mechanism

## 1. System Architecture

```text
       +-------------------------------------------------------------+
       |               MoE Router (Authoritative)                    |
       +-------------------------------------------------------------+
                                      | (Source Stage L Routing Event)
                                      v
       +-------------------------------------------------------------+
       |             ST-MoE / CSTP Prediction Engine                 |
       |  - Causal isolation: observes stage <= L                    |
       |  - Predicts Top-K candidate experts for target stage L + H  |
       +-------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |               Prefetch Candidate Queue                      |
       |  - State: PENDING -> ADMITTED -> DISPATCHED / REJECTED      |
       |  - Duplicate suppression & deadline tracking                |
       +-------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |                 DDR5 Transfer Engine                        |
       |  - Dual-channel DDR5-4800 (Calibrated 22.578 GB/s)          |
       |  - Isolated expert transfer latency: 3.90116 ms (88.08 MB)  |
       |  - Demand preemption, Pause / Resume, In-flight attachment  |
       +-------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |                     HBM Cache Region                        |
       |  - Demand-Verified Partition (80% capacity)                 |
       |  - PABI Quarantine Partition (20% capacity cap = 16 GB)     |
       |  - Belady-Style Pinning (deadline-expired unpinning)        |
       +-------------------------------------------------------------+
```

## 2. Demand Path Correctness & Lifecycle States
The prefetch framework handles all 10 potential execution scenarios on the demand path:
1. **Natural Hit:** Expert already resident in demand HBM partition.
2. **Timely Prefetch Hit:** Prefetch transfer completed before stage demand arrival; expert resides in PABI quarantine; promoted to demand partition.
3. **In-Flight Attachment (Late Prefetch):** Expert transfer actively progressing on DDR5 bus; demand request attaches to existing transfer, avoiding a redundant transfer.
4. **Queued Attachment:** Expert queued in DDR5 engine; promoted to demand priority immediately.
5. **Demand Miss (Unprefetched):** Traditional demand fetch initiated with highest priority.
6. **Duplicate Prediction:** Discarded cleanly by `PrefetchCandidateQueue::insert_candidates()`.
7. **Candidate Expiration:** Discarded if current stage exceeds deadline ($S_{\text{curr}} \ge S_{\text{target}}$).
8. **Transfer Cancellation:** Preempted cleanly if superseded.
9. **Demand Preemption:** Active speculative transfer paused; channel reassigned to demand transfer; resumed upon demand completion.
10. **Transfer Resumption:** Paused speculative transfer resumes with exact remaining bytes intact.

All state transitions are formally verified in [`test_ddr5_transfer_engine.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test_ddr5_transfer_engine.cpp) and [`test_demand_transfer_integration.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test_demand_transfer_integration.cpp).
