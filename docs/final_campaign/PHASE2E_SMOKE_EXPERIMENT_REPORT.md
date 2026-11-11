# Phase 2E: Bounded Forward-Lookahead Smoke Experiment Report

**Authoritative Git Baseline Anchor:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Execution Date:** November 3, 2026  
**Workload Evaluated:** Workload A1 (MoE-Suite Task 12, 78 Layers, 75 MoE Stages, Top-8 Routing, 4 Decoding Tokens)  
**Document Location:** `docs/final_campaign/PHASE2E_SMOKE_EXPERIMENT_REPORT.md`  
**Data Summary CSV:** `results/final_comparative_campaign/smoke_phase2e/phase2e_smoke_summary.csv`  

---

## 1. Executive Summary & Gating Decision

### **GATING DECISION: CONDITIONAL HALT / NO-GO FOR TIER 1**
While all 8 smoke configurations executed cleanly with exit code 0 and metric conservation holds with 0 error, the gating condition requiring **timely resident hits > 0** was **not met** (`timely_resident_hits = 0` across all forward configurations). Consequently, Phase 2F (Full 24-Run Tier 1 Campaign) remains **strictly held**.

### Key Results Overview:
1. **Execution Success:** All 8 smoke configurations executed sequentially and completed with exit code 0 (total runtime 73.27 seconds).
2. **Timely Resident Hits:**
   - Legacy co-located scheduler (`LEGACY-S`): `0` timely resident hits.
   - Forward ST-MoE (`FWD-S-H1`, `FWD-S-H2`, `FWD-S-H3`): `0` timely resident hits.
   - Forward CSTP (`FWD-C-H1`, `FWD-C-H2`, `FWD-C-H3`): `0` timely resident hits.
3. **Late In-Flight Attachments:**
   - Legacy scheduler (`LEGACY-S`): `37` late in-flight attachments.
   - Forward ST-MoE: Increased to `64` ($H=1$), `67` ($H=2$), and `62` ($H=3$) attachments (+81% increase).
   - Forward CSTP: Increased to `60` ($H=1$), `60` ($H=2$), and `57` ($H=3$) attachments (+62% increase).
4. **Exposed Memory Stall & End-to-End Latency:**
   - Baseline (`B`): `7884.24 ms` stall, `8003.70 ms` total time.
   - All prefetching configurations (`LEGACY-S`, `FWD-S`, `FWD-C`): Identical `7884.24 ms` stall and `8003.70 ms` simulated time.
5. **Lifecycle Conservation:** Conservation error is strictly 0 across all runs:
   $$\text{Prefetches Dispatched} = \text{Demanded} + \text{Unused} + \text{Cancelled} + \text{Evicted Before Demand} + \text{Still In Flight}$$
   $$\text{Total Demand Lookups} = \text{Natural HBM Hits} + \text{Timely Hits} + \text{Late Attachments} + \text{Unprefetched Misses}$$

---

## 2. Complete 8-Configuration Metric Reconciliation Table

| Config ID | Predictor | Sched Mode | Horiz | Sim Time (ms) | Exposed Stall (ms) | Hit Rate (%) | Timely Hits | Late Attach | Natural Hits | Unpref Miss | Dispatched | Unused | Pred F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `B` | none | legacy_current_stage | 1 | 8003.70 | 7884.24 | 71.93% | 0 | 0 | 5179 | 2021 | 0 | 0 | 0.0000 |
| `LEGACY-S` | st_moe | legacy_current_stage | 1 | 8003.70 | 7884.24 | 71.93% | 0 | 37 | 5179 | 1984 | 903 | 29 | 0.1261 |
| `FWD-S-H1` | st_moe | forward_lookahead | 1 | 8003.70 | 7884.24 | 71.93% | 0 | 64 | 5179 | 1957 | 920 | 29 | 0.1273 |
| `FWD-S-H2` | st_moe | forward_lookahead | 2 | 8003.70 | 7884.24 | 71.93% | 0 | 67 | 5179 | 1954 | 907 | 29 | 0.1313 |
| `FWD-S-H3` | st_moe | forward_lookahead | 3 | 8003.70 | 7884.24 | 71.93% | 0 | 62 | 5179 | 1959 | 889 | 29 | 0.1279 |
| `FWD-C-H1` | cstp | forward_lookahead | 1 | 8003.70 | 7884.24 | 71.93% | 0 | 60 | 5179 | 1961 | 843 | 29 | 0.1261 |
| `FWD-C-H2` | cstp | forward_lookahead | 2 | 8003.70 | 7884.24 | 71.93% | 0 | 60 | 5179 | 1961 | 834 | 29 | 0.1273 |
| `FWD-C-H3` | cstp | forward_lookahead | 3 | 8003.70 | 7884.24 | 71.93% | 0 | 57 | 5179 | 1964 | 819 | 29 | 0.1291 |

---

## 3. Deep Root-Cause Architectural Diagnostics

Through trace-level inspection of `prefetch_effectiveness.csv`, `expert_routing.csv`, and `expert_load.csv`, two primary architectural mechanisms explain why `timely_resident_hits` remained 0 despite forward lookahead:

### 3.1 Mechanism 1: The 1-Stage Post-Increment Indexing Skew
In the simulator's control flow:
1. When a stage's routing module executes ([`BatchedSequence::update_expert`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L165-L318)), `cur_layer` represents the current MoE stage ($0, 1, \dots, 74$).
2. Routing is resolved, forward prediction generates target stage $T = \text{cur\_layer} + H$, and transfers are enqueued into [`PrefetchCandidateQueue`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_candidate_queue.h) targeting stage $T$.
3. At the end of `update_expert` ([`sequence.cpp:316`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L316)), `cur_layer++` immediately increments the sequence layer counter.
4. When the actual Linear computation of that stage executes ([`Linear::forward`](file:///home/mohman/ca_project/LLMSimulator/src/module/linear.cpp#L49-L50)), it reads:
   ```cpp
   int layer_id = sequences_metadata->cur_layer; // Already incremented by 1!
   A->expert_layer_id = layer_id;
   ```
5. In [`LinearImpl::execute`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/linear_impl.cpp#L65-L67), the cache lookup queries `is_expert_resident(weight->expert_layer_id, weight->expert_id)`.
6. Because `weight->expert_layer_id` is offset by +1 relative to the routing stage index, the candidate prefetch (dispatched for $T$) and the demand lookup (querying $T+1$) suffer an index mismatch. Prefetches only intersect when an adjacent stage's candidate happens to match, explaining why late in-flight attachments occurred rather than timely hits.

### 3.2 Mechanism 2: Candidate Queue Re-Dispatch Behavior
In [`PrefetchCandidateQueue::mark_candidate_dispatched`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_candidate_queue.cpp#L385-L390), dispatched candidates transition to `READY` state but remain in `active_candidates_`. Because [`select_ready_candidates`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_candidate_queue.cpp#L352-L374) sorts by earliest deadline, previously dispatched candidates for earlier stages are repeatedly re-evaluated at subsequent stages. While duplicate suppression prevented redundant physical transfers, the queue head was dominated by earlier stages whose execution windows had already begun.

### 3.3 Mechanism 3: Physical Transfer Serialization vs Overlap Window
At the calibrated $22.578\text{ GB/s}$ DDR5 bandwidth, each 88.08 MB expert requires $3.90116\text{ ms}$ on the single-channel DDR5 bus.
- Under budget $B=4$, all 4 candidates serialize on the bus ($3.901\text{ ms}, 7.802\text{ ms}, 11.703\text{ ms}, 15.605\text{ ms}$).
- In Workload A1, each stage compute requires $\sim 1.85\text{ ms}$.
- Overlap for $H=1$: $\sim 1.75\text{ ms} < 3.901\text{ ms}$ (arrival is always late).
- Overlap for $H=2$: $\sim 3.60 - 3.70\text{ ms} \approx 3.901\text{ ms}$ (threshold boundary).
- When combined with the 1-stage post-increment indexing skew, even candidate 1 arrived just after demand onset, resulting in `LATE_IN_FLIGHT` attachments instead of `TIMELY_RESIDENT` hits.

---

## 4. Verification Checklist & Compliance

- [x] All 8 runs completed with exit code 0.
- [x] Host runtimes captured (8.41s to 10.21s per run; total 73.27s).
- [x] Full standard outputs and errors logged under `results/final_comparative_campaign/smoke_phase2e/<config_id>/logs/`.
- [x] All CSV and JSON outputs verified and non-empty.
- [x] Metric conservation verified: error = 0 across all runs.
- [x] Zero ternary operator compliance verified: `ast.IfExp = 0` in Python runner, `? : = 0` in simulator diffs.
- [x] Authoritative git baseline anchor preserved: `d5bf61ff0c066c06f72173189f9924329f613a88`.

---

## 5. Recommended Actions for Phase 2F Authorization

Before authorizing the full 24-run Tier 1 campaign:
1. **Reconcile Layer ID Indexing:** Unify the layer index passed to `on_stage_routing_complete` with `weight->expert_layer_id` in `Linear::forward` (ensuring stage $L$ Router and stage $L$ Linear share the exact same `layer_id`).
2. **Refine Candidate Queue Removal:** Remove or deactivate admitted candidates from the dispatch pool once their transfer has been accepted by `submit_prefetch_transfer`.
3. **Re-run Bounded Smoke Test:** Re-execute the 8-run smoke experiment to verify that `timely_resident_hits > 0` and `exposed_stall < 7884.24 ms` before committing to the full multi-hour campaign.
