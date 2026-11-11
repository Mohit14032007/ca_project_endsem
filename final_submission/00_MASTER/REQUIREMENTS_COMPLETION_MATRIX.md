# Requirements Completion Matrix
**MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference**

| ID | Requirement Name | Professor Specification | Implementation | Evaluation | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `REQ-1` | **Prediction Technique** | ST-MoE reproduction, CSTP, training-free, precision/recall/F1, real MoE routing traces | `IMPLEMENTED` | `VERIFIED` | **COMPLETE** |
| `REQ-2` | **Prefetching Mechanism** | Predictor-driven candidate generation, forward lookahead, overlap with computation, DDR5 engine, HBM insertion, demand attachment/promotion | `IMPLEMENTED` | `VERIFIED` | **COMPLETE** |
| `REQ-3` | **Prefetching Effectiveness** | Coverage, timely and late prefetch rates, unused prefetches, HBM hit rate, additional traffic, exposed stall, expert-load latency, end-to-end latency | `IMPLEMENTED` | `VERIFIED` | **COMPLETE** |
| `REQ-4` | **HBM Management** | LRU baseline, PABI quarantine, Belady-inspired pinning, speculative vs demand residency, eviction behavior, empirical advantage over LRU | `IMPLEMENTED` | `VERIFIED` | **COMPLETE** |
| `REQ-5` | **Comparative Evaluation** | Controlled comparison: Baseline, Prediction-only (ST-MoE, CSTP), Forward Prefetch (ST-MoE, CSTP), LRU vs PABI, Pinning on/off on identical workloads | `IMPLEMENTED` | `VERIFIED` | **COMPLETE** |
| `REQ-6` | **Sensitivity Analysis** | Batch size (1, 2, 4, 8), context length, horizon H in {1,2,3}, candidate budget K in {2,4,8}, compute multiplier in {1,4,8,12,16} | `IMPLEMENTED` | `VERIFIED` | **COMPLETE** |
| `REQ-7` | **Energy Analysis** | Energy counters implemented/usable, DDR5 traffic energy, HBM energy, predictor overhead, extra speculative traffic energy translation | `IMPLEMENTED` | `VERIFIED` | **COMPLETE** |
| `REQ-8` | **Documentation and Reproducibility** | Git repository link, complete source, setup/build instructions, reproduction commands, configs, trace provenance, results/plots, report, ZIP | `IMPLEMENTED` | `VERIFIED` | **COMPLETE** |
| `BONUS` | **Realistic MoE Routing Traces** | Verify real recorded traces vs synthetic traffic; faithfully reproduce routing sequences | `IMPLEMENTED` | `VERIFIED` | **COMPLETE** |
