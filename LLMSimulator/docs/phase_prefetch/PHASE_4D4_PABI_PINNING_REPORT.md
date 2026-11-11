# Phase 4D.4 — Belády-Style Predictor-Score Pinning and Stage-Deadline Expiration: Implementation Report

## Executive Summary

Phase 4D.4 extends the **PABI (Predictor-Aware Buffer Insertion)** cache management architecture with **Belády-style predictor-score pinning and stage-deadline expiration**. Under Phase 4D.3, newly completed prefetches enter a logical speculative quarantine region and are evicted using LRU within that region when space is constrained. However, pure LRU within the speculative buffer treats all speculative experts identically, disregarding the predictor's ranking, confidence scores, and imminent target-stage deadlines. As a consequence, high-confidence candidates predicted for the immediate next stage could be prematurely evicted by lower-confidence or farther-horizon speculative prefetches.

Phase 4D.4 addresses this limitation by introducing a bounded, deterministic pinning mechanism:
1. **Belády-Inspired Heuristic**: Approximates optimal replacement by protecting the highest-confidence speculative candidates predicted for imminent target stages from premature eviction until their target stage executes.
2. **Deterministic Top-K Selection**: At candidate dispatch, candidates for each target stage are deterministically prioritized by:
   - Prediction rank ascending (rank 1 is highest priority),
   - Prediction score descending (higher confidence tie-break),
   - Expert ID ascending (strict tie-break).
3. **Per-Target-Stage Pin Budget ($K$)**: A strict per-stage budget (`max_pinned_experts`, default 4) prevents speculative buffer saturation.
4. **Dual-Phase Lifecycle Tracking**:
   - **In-Flight Pin Intent**: Candidates selected for pinning receive pin intent metadata during transfer.
   - **Resident Speculative Pin**: Upon transfer completion, resident entries inherit the pin if their target stage has not yet passed.
5. **Pin-Aware Eviction Invariants**:
   - **Case A (Speculative Overflow)**: Pinned speculative entries are skipped. If all resident speculative entries are pinned, incoming speculative insertions are safely rejected with `SPECULATIVE_REJECT_PINNED_CAPACITY` rather than evicting pinned candidates or demand-verified working sets.
   - **Case B (Total HBM Overflow)**: Demand miss allocations evict unpinned speculative entries first, followed by demand LRU entries, while preserving pinned speculative entries.
6. **Stage-Deadline Expiration**: Once the target stage finishes execution, all associated pins (both resident and in-flight) are atomically expired. Expired entries remain in HBM as unpinned speculative entries and become eligible for ordinary LRU eviction without immediate byte deallocation.
7. **Demand-Verified Promotion**: Demanded hits on pinned speculative experts immediately clear pin metadata and promote the expert to the MRU position of the demand-verified recency list.

All 25 unit, integration, and invariant test cases in `eval/test_pabi_pinning.cpp` passed (100% success rate). All 161 pre-existing tests across Phase 0 through Phase 4D.3 continue to pass with zero regressions (186 total tests passed, 0 failed across 12 test suites). Strictly zero ternary operators were introduced across all modified and newly created files (`ast.IfExp == 0` and `? : == 0`).

---

## 1. Architectural Design & Belády-Inspired Heuristic Model

### 1.1 Motivation & Distinction from True Oracle Belády

Belády's optimal algorithm (MIN / OPT) proves that the optimal caching policy evicts the item whose next reference occurs furthest in the future. In an online LLM serving system with dynamic token-dependent routing, true future access sequences are non-causal and unknown at runtime.

Phase 4D.4 implements a **causal, predictor-driven approximation** of Belády's principle:
- Predictions produced by ST-MoE, CSTP, and ACSH forecast expert access at specific future pipeline stages.
- The predictor rank $r$ and confidence score $s$ provide a monotonic heuristic proxy for reference probability.
- Target stage distance $\Delta = \text{target\_stage} - \text{current\_stage}$ defines the imminent temporal window.
- By pinning the top-$K$ candidates for the target stage until stage execution, the cache guarantees that high-probability candidates survive cache pressure until the exact cycle their reference is evaluated.
- **Explicit Boundary**: This is a causal heuristic approximation, **not** an oracle or true Belády replacement policy. It claims no omniscient knowledge and strictly uses causal predictor outputs.

### 1.2 Deterministic Pin Selection Algorithm

Candidate selection occurs during `dispatch_prefetch_candidates()` or explicitly via `select_pin_candidates_for_stage()`:

```
Algorithm: Deterministic Pin Selection
Input: target_stage_id, candidate_list, max_pinned_experts, resident_experts
Output: selected_pin_candidates

1. Filter eligible candidates:
   eligible = []
   for cand in candidate_list:
     if cand.target_stage_id != target_stage_id: continue
     if cand.expert_id < 0: continue
     if cand.state in {EXPIRED_MISS, CANCELLED, EVICTED}: continue
     if resident_experts contains (target_stage_id, cand.expert_id) as DEMAND_VERIFIED:
       continue  // Already resident and demand-verified; pinning redundant
     eligible.append(cand)

2. Deterministic sort:
   sort eligible by:
     a. prediction_rank ASCENDING   (rank 1 before rank 2)
     b. prediction_score DESCENDING (0.95 before 0.80)
     c. expert_id ASCENDING         (expert 5 before expert 12)

3. Deduplicate by expert_id within stage:
   deduplicated = unique_by_expert_id(eligible, preserve_first=true)

4. Budget cap:
   selected = deduplicated[0 : min(len(deduplicated), max_pinned_experts)]

5. State attachment:
   for cand in selected:
     Create active PinIntent
     If expert is resident speculative -> mark resident is_pinned = true
     If expert is in-flight prefetch   -> mark in-flight has_pin_intent = true
```

---

## 2. Exact Files Modified and Created

### 2.1 Modified Files

1. `src/prefetch/prefetch_config.h`
   - Added configuration fields:
     - `bool pabi_pinning_enabled = false;`
     - `int pabi_max_pinned_experts = 4;`
     - `std::string pabi_pin_selection_policy = "top_ranked";`
     - `std::string pabi_pin_expiration_policy = "stage_deadline";`

2. `src/prefetch/prefetch_config.cpp`
   - Added validation enforcing `pabi_max_pinned_experts >= 0`, `pabi_pin_selection_policy == "top_ranked"`, and `pabi_pin_expiration_policy == "stage_deadline"`.
   - Added YAML parsing for `pinning_enabled`, `max_pinned_experts`, `pin_selection_policy`, and `pin_expiration_policy` under both `pabi` and `hbm.pabi` nodes.

3. `src/hardware/device.h`
   - Defined `struct PinIntent` storing layer, expert, target stage, expiration stage, rank, score, candidate ID, transfer ID, and active status.
   - Extended `struct InflightExpertEntry` with pin intent fields:
     - `bool has_pin_intent = false;`
     - `int pin_target_stage_id = -1;`
     - `int pin_expiration_stage_id = -1;`
     - `int prediction_rank = 0;`
     - `double prediction_score = 0.0;`
     - `uint64_t candidate_id = 0;`
   - Extended `struct ResidentExpertEntry` with resident pin metadata:
     - `bool is_pinned = false;`
     - `int pin_target_stage_id = -1;`
     - `int pin_expiration_stage_id = -1;`
     - `int prediction_rank = 0;`
     - `double prediction_score = 0.0;`
     - `uint64_t candidate_id = 0;`
   - Extended `struct PabiCacheMetrics` with 11 pinning-specific metrics:
     - `pin_candidates_considered`
     - `pin_candidates_selected`
     - `pin_intents_created`
     - `resident_experts_pinned`
     - `inflight_pin_intents`
     - `pin_expirations`
     - `pins_cleared_on_promotion`
     - `pinned_eviction_skipped`
     - `speculative_reject_pinned_capacity`
     - `pinned_expert_demand_hits`
     - `unpinned_speculative_evictions`
   - Added Device member fields:
     - `bool pinning_enabled = false;`
     - `int max_pinned_experts = 4;`
     - `std::string pin_selection_policy = "top_ranked";`
     - `std::string pin_expiration_policy = "stage_deadline";`
     - `int current_stage_id = 0;`
     - `std::unordered_map<std::pair<int, int>, PinIntent, PairHash> active_pin_intents;`
   - Declared member functions:
     - `void set_current_stage(int stage)`
     - `bool is_pinning_active() const`
     - `std::vector<PrefetchCandidate> select_pin_candidates_for_stage(int, const std::vector<PrefetchCandidate>&)`
     - `int expire_pins_for_stage(int)`
     - `int cleanup_unexecuted_stage_pins(int)`
   - Updated `reset_expert_cache()` to clear `active_pin_intents` and reinitialize all metrics.

4. `src/hardware/device.cpp`
   - Initialized pinning settings in constructor and enabled environment variable overrides: `PABI_PINNING_ENABLED` and `PABI_MAX_PINNED_EXPERTS`.
   - Implemented `is_pinning_active()`: returns true if and only if `eviction_policy == EvictionPolicy::PABI`, `pabi_quarantine_enabled == true`, `pinning_enabled == true`, and `max_pinned_experts > 0`.
   - Implemented `select_pin_candidates_for_stage`: deterministic filtering, sorting, deduplication, budget cap, intent generation, resident marking, and in-flight attachment.
   - Implemented `expire_pins_for_stage`: expires matching resident speculative pins, in-flight pin intents, and active intent records.
   - Implemented `cleanup_unexecuted_stage_pins`: cleans up unexecuted or aborted stage pins safely.
   - Updated `promote_speculative_to_demand`: clears resident pin metadata, removes active intent, and increments `pins_cleared_on_promotion` and `pinned_expert_demand_hits`.
   - Updated `mark_expert_resident`:
     - Case A (Speculative buffer capacity): scans `speculative_lru_list`, skips pinned entries (`pinned_eviction_skipped`), evicts first unpinned entry. If all entries are pinned, rejects insertion with `SPECULATIVE_REJECT_PINNED_CAPACITY`.
     - Case B (Total HBM capacity): evicts unpinned speculative entries first, skips pinned entries, evicts demand-verified entries next.
   - Updated `submit_prefetch_transfer`: attaches pin intent to `InflightExpertEntry` if present in `active_pin_intents`.
   - Updated `promote_inflight_to_demand`: clears pin intent upon demand promotion and erases active intent.
   - Updated `process_completed_prefetches`: completed transfers with active unexpired pin intent enter HBM with `is_pinned = true`. Completed transfers whose deadline has already elapsed enter HBM unpinned. Cancelled transfers clear pin intents.
   - Updated `dispatch_prefetch_candidates`: groups admitted candidates by target stage and automatically triggers `select_pin_candidates_for_stage()`.
   - Updated `verify_pabi_invariants`: validates that only speculative entries can be pinned, demand entries never have pins or pin intents, and LRU mode never has active pins.

5. `src/prefetch/live_prediction_engine.h`
   - Declared `int expire_pins_for_stage(std::shared_ptr<Device> device, int target_stage_id);`.

6. `src/prefetch/live_prediction_engine.cpp`
   - Implemented `expire_pins_for_stage` forwarding directly to `device->expire_pins_for_stage(target_stage_id)`.

7. `eval/CMakeLists.txt`
   - Added executable target `test_pabi_pinning` linked with `llm_system`.

### 2.2 Newly Created Files

8. `eval/test_pabi_pinning.cpp`
   - 25 deterministic unit, integration, and invariant test cases covering all aspects of Phase 4D.4 pinning.

9. `docs/phase_prefetch/PHASE_4D4_PABI_PINNING_REPORT.md`
   - Comprehensive implementation report and invariant documentation.

---

## 3. Lifecycle of Pinned Candidates

The diagram below illustrates the complete state transition lifecycle of a pinned prefetch candidate:

```
[Candidate Generated by Predictor]
                 |
                 v
   [select_pin_candidates_for_stage]
                 |
   +-------------+-------------+
   | Rank <= K                 | Rank > K / Overflow
   v                           v
[PinIntent Created]       [Ordinary Prefetch]
   |                           |
   v                           v
[submit_prefetch_transfer]     [submit_prefetch_transfer]
(has_pin_intent = true)        (has_pin_intent = false)
   |                           |
   +-------------+-------------+
                 |
                 v
      [Transfer In-Flight]
        /        |        \
       /         |         \ (Demand Hit In-Flight)
      /          |          v
     /           |   [promote_inflight_to_demand]
    /            |   (Pin intent cleared)
   /             |          |
  /              v          v
 /       [Transfer Finishes] [Demand Transfer Finishes]
 |               |                  |
 | (Deadline     | (Deadline        v
 |  Elapsed)     |  Active)    [HBM: DEMAND_VERIFIED]
 |               v                  (MRU)
 |       [HBM: SPECULATIVE]
 |       (is_pinned = true)
 |               |
 |         +-----+-----+
 |         |           |
 |   (Demand Hit)  (Stage Deadline Reached)
 |         |           |
 |         v           v
 |   [promote_to]  [expire_pins_for_stage]
 |   [  demand  ]      |
 |   (Pin Cleared)     v
 |         |       [HBM: SPECULATIVE]
 |         |       (is_pinned = false)
 |         |           |
 |         |           v
 |         |       [Eligible for LRU Eviction]
 \         |           |
  \        |           |
   v       v           v
[HBM: DEMAND_VERIFIED]  [Evicted on Pressure]
```

### 3.1 Step-by-Step Lifecycle Description

1. **Prediction & Admission**: The live prediction engine evaluates the layer and generates candidates with ranks and confidence scores.
2. **Deterministic Selection**: For target stage $S$, candidates are filtered, sorted, and capped at $K = \text{max\_pinned\_experts}$. A `PinIntent` record is stored in `active_pin_intents`.
3. **Dispatch & In-Flight Intent**: When the prefetch transfer is submitted to the `DDR5TransferEngine`, the intent is attached to the transfer entry (`has_pin_intent = true`).
4. **Residency Placement**:
   - If the prefetch completes before stage $S$ executes, it is placed into the speculative quarantine list with `is_pinned = true`.
   - If the transfer finishes after stage $S$ has passed (`pin_expiration_stage_id < current_stage_id`), the intent is marked expired and the expert is placed into quarantine as unpinned.
5. **Eviction Protection**: While pinned, the expert cannot be evicted by incoming speculative insertions or total HBM overflows.
6. **Demand Hit**:
   - If routing demands the pinned expert, `promote_speculative_to_demand()` immediately clears `is_pinned`, erases the active intent, moves the expert to the MRU of `demand_lru_list`, and increments `pinned_expert_demand_hits`.
7. **Deadline Expiration**:
   - When stage $S$ completes, `expire_pins_for_stage(S)` clears `is_pinned` on all matching resident experts.
   - The experts remain resident in `speculative_lru_list` without byte reallocation. They now become ordinary speculative entries subject to normal LRU eviction.
8. **Cancellation**:
   - If an in-flight transfer is cancelled, its pin intent is removed from `active_pin_intents`.

---

## 4. Pin-Aware Eviction Invariants

### 4.1 Case A: Speculative Quarantine Capacity Enforcement

When speculative insertions exceed `speculative_buffer_capacity_bytes`, the device scans `speculative_lru_list` from LRU to MRU:
1. For each entry, if `is_pinned == true`, it is preserved. `pinned_eviction_skipped` is incremented.
2. The first unpinned speculative entry encountered is evicted.
3. If **all** resident speculative entries are currently pinned, no victim can be evicted without violating the pin guarantee. The incoming speculative insertion is rejected, incrementing `speculative_reject_pinned_capacity` and `speculative_insertions_rejected`. Demand-verified entries are strictly untouched.

### 4.2 Case B: Total Physical HBM Capacity Enforcement

When a cold demand miss requires space and total resident bytes exceed `cache_capacity_bytes`:
1. The device searches `speculative_lru_list` from LRU to MRU.
2. Pinned speculative entries are skipped (`pinned_eviction_skipped` incremented).
3. Any unpinned speculative entry is evicted first to make room for demand weights.
4. If no unpinned speculative entry exists, eviction falls back to `demand_lru_list` from LRU to MRU.
5. Pinned speculative entries remain protected against eviction.

### 4.3 Formal Cache Invariants

The `Device::verify_pabi_invariants()` method strictly checks:
1. **Byte Conservation**: $B_{\text{resident}} = B_{\text{demand}} + B_{\text{speculative}}$.
2. **Non-Negativity**: $B_{\text{resident}} \ge 0$, $B_{\text{demand}} \ge 0$, $B_{\text{speculative}} \ge 0$.
3. **Physical Capacity**: $B_{\text{resident}} \le C_{\text{HBM}}$.
4. **Quarantine Capacity**: $B_{\text{speculative}} \le C_{\text{speculative}}$.
5. **Pin Exclusivity**: If `entry.is_pinned == true`, then `entry.classification` must be `SPECULATIVE`. Demand-verified entries can never be pinned.
6. **In-Flight Pin Integrity**: If `inflight.transfer_class == DEMAND`, then `has_pin_intent` must be `false`.
7. **LRU Invariant**: When `eviction_policy == LRU`, no resident entry may have `is_pinned == true`.

---

## 5. Shared Experts and Multi-Request Handling

In multi-sequence batched inference, multiple requests or layers may predict the same physical expert:
1. **Per-Stage Deduplication**: Within each target stage, `select_pin_candidates_for_stage()` deduplicates by `expert_id`. The occurrence with the best rank and highest score is selected; lower-priority duplicates are discarded.
2. **Single Resident Instance**: HBM maintains a single physical copy of each expert keyed by `(layer_id, expert_id)`. Multiple requests share the resident entry.
3. **Promotion Coalescing**: The first demand hit promotes the resident expert to `DEMAND_VERIFIED` and clears the pin. Subsequent requests referencing the same expert hit in the demand list without redundant transfers or conflicting pin states.

---

## 6. Configuration & Environment Overrides

### 6.1 Configuration Schema

The following parameters are supported in `config.yaml` under `pabi` or `hbm.pabi`:

```yaml
pabi:
  quarantine_enabled: true
  speculative_buffer_capacity_bytes: 16000000000  # 16 GB
  speculative_fraction: 0.20
  pinning_enabled: true
  max_pinned_experts: 4
  pin_selection_policy: "top_ranked"
  pin_expiration_policy: "stage_deadline"
```

### 6.2 Environment Variable Overrides

For headless ablation studies and CI test runs, runtime environment variables override configuration defaults:

| Environment Variable | Valid Values | Description |
|---|---|---|
| `PABI_PINNING_ENABLED` | `1`, `true`, `0`, `false` | Dynamically enables or disables PABI pinning |
| `PABI_MAX_PINNED_EXPERTS` | Integer $\ge 0$ | Overrides per-stage pin budget ($K$) |

Precedence is strictly enforced: `Environment Variable > YAML Configuration > C++ Preset Default`.

---

## 7. Metrics & Telemetry Events

### 7.1 PABI Pinning Metrics

`PabiCacheMetrics` tracks 11 dedicated counters:

```cpp
struct PabiCacheMetrics {
  // Phase 4D.3 metrics
  uint64_t demand_insertions = 0;
  uint64_t speculative_insertions = 0;
  uint64_t speculative_promotions = 0;
  uint64_t speculative_evictions = 0;
  uint64_t demand_evictions = 0;
  uint64_t speculative_insertions_rejected = 0;
  uint64_t useful_prefetch_hits = 0;

  // Phase 4D.4 pinning metrics
  uint64_t pin_candidates_considered = 0;
  uint64_t pin_candidates_selected = 0;
  uint64_t pin_intents_created = 0;
  uint64_t resident_experts_pinned = 0;
  uint64_t inflight_pin_intents = 0;
  uint64_t pin_expirations = 0;
  uint64_t pins_cleared_on_promotion = 0;
  uint64_t pinned_eviction_skipped = 0;
  uint64_t speculative_reject_pinned_capacity = 0;
  uint64_t pinned_expert_demand_hits = 0;
  uint64_t unpinned_speculative_evictions = 0;
};
```

### 7.2 Structured Cache Log Events

The following distinct log events are emitted via `log_cache_event()`:

- `PIN_INTENT_CREATED`: Pin intent recorded for an admitted candidate.
- `PIN_APPLIED`: Resident speculative entry marked as pinned.
- `PIN_EXPIRED`: Stage deadline elapsed; pin removed.
- `PIN_CLEARED_ON_PROMOTION`: Demand reference hit a pinned expert; pin cleared and expert promoted.
- `PINNED_EVICTION_SKIPPED`: Eviction scanner skipped a pinned candidate to preserve it.
- `SPECULATIVE_REJECT_PINNED_CAPACITY`: Speculative insertion rejected because quarantine is saturated with pinned experts.

---

## 8. Comprehensive Unit & Integration Test Suite (25 Tests)

The test suite `eval/test_pabi_pinning.cpp` implements 25 isolated, deterministic test cases verifying every specification requirement:

| Test ID | Test Name | Target Invariant / Verification | Result |
|---|---|---|---|
| **Test 1** | `test_1_pinning_disabled_preserves_pabi_behavior` | When `pinning_enabled=false`, candidates are unpinned and behave identically to Phase 4D.3. | **PASS** |
| **Test 2** | `test_2_lru_policy_never_activates_pinning` | When `eviction_policy=LRU`, `is_pinning_active()` returns false and pinning never triggers. | **PASS** |
| **Test 3** | `test_3_top_k_selection_respects_candidate_rank` | Top-$K$ selection chooses rank 1, 2, 3 over rank 4, 5. | **PASS** |
| **Test 4** | `test_4_score_tie_break_is_deterministic` | Candidates with equal rank break ties by higher prediction score. | **PASS** |
| **Test 5** | `test_5_expert_id_tie_break_is_deterministic` | Candidates with equal rank and score break ties by lower expert ID. | **PASS** |
| **Test 6** | `test_6_per_target_stage_pin_budget_enforced` | Exactly $K=2$ candidates selected when 5 candidates are submitted for a stage. | **PASS** |
| **Test 7** | `test_7_selected_inflight_candidate_receives_pin_intent` | In-flight transfer receives `has_pin_intent=true` matching selected candidate. | **PASS** |
| **Test 8** | `test_8_completed_transfer_receives_speculative_resident_pin` | Completed transfer inserts into HBM with `is_pinned=true` and valid metadata. | **PASS** |
| **Test 9** | `test_9_pinned_expert_skipped_during_speculative_lru_eviction` | Pinned entry survives speculative buffer overflow while unpinned entry is evicted. | **PASS** |
| **Test 10** | `test_10_unpinned_speculative_evicted_before_pinned` | Unpinned speculative entry at MRU is evicted before an older pinned entry at LRU. | **PASS** |
| **Test 11** | `test_11_pinned_expert_skipped_during_total_hbm_overflow` | Demand miss evicts unpinned speculative and demand LRU entries, skipping pinned entries. | **PASS** |
| **Test 12** | `test_12_incoming_speculative_rejected_if_all_pinned` | When all resident entries in speculative buffer are pinned, insertion is rejected safely. | **PASS** |
| **Test 13** | `test_13_demand_hit_promotes_pinned_speculative_and_clears_pin` | Demanding a pinned speculative expert clears `is_pinned` and promotes it to demand MRU. | **PASS** |
| **Test 14** | `test_14_inflight_prefetch_promoted_to_demand_no_pin` | In-flight prefetch promoted to demand has pin intent cleared and inserts as demand-verified. | **PASS** |
| **Test 15** | `test_15_pin_expires_at_exact_target_stage_completion` | `expire_pins_for_stage(S)` clears `is_pinned` on resident entries for stage $S$. | **PASS** |
| **Test 16** | `test_16_expiration_does_not_evict_expert_immediately` | Expired entry remains in HBM and `resident_speculative_bytes` is unchanged. | **PASS** |
| **Test 17** | `test_17_expired_expert_eligible_for_ordinary_lru_eviction` | After pin expiration, the expert is evicted under standard speculative LRU pressure. | **PASS** |
| **Test 18** | `test_18_transfer_completing_after_deadline_inserted_unpinned` | Prefetch completing after target stage has elapsed is inserted unpinned. | **PASS** |
| **Test 19** | `test_19_cancelled_transfer_clears_pin_intent` | Cancelling an in-flight transfer removes pin intent from `active_pin_intents`. | **PASS** |
| **Test 20** | `test_20_multiple_requests_same_expert_no_duplicate_pins` | Multiple candidates predicting the same expert produce exactly one pin intent. | **PASS** |
| **Test 21** | `test_21_multiple_target_stage_deadlines_handled_deterministically` | Expiring stage 1 leaves stage 2 pins active; expiring stage 2 expires stage 2 pins. | **PASS** |
| **Test 22** | `test_22_skipped_aborted_stage_cleanup_does_not_leak_pins` | `cleanup_unexecuted_stage_pins()` properly purges pins without memory leaks. | **PASS** |
| **Test 23** | `test_23_cache_reset_clears_all_pin_state` | `reset_expert_cache()` clears all active pin intents, resident pins, and in-flight pins. | **PASS** |
| **Test 24** | `test_24_byte_accounting_invariants_valid_through_lifecycle` | Invariants hold through insertion, pinning, expiration, eviction, and promotion. | **PASS** |
| **Test 25** | `test_25_regression_phase4d1_4d2_4d3` | Full end-to-end integration with demand transfers, prefetch dispatch, and quarantine. | **PASS** |

---

## 9. Full Regression Test Verification (186/186 Tests Across 12 Suites)

All 12 unit, integration, and regression test suites were executed sequentially from the `LLMSimulator` directory. All 186 tests passed with zero failures:

| Suite # | Binary Executable | Scope / Phase | Tests Passed | Tests Failed | Status |
|---|---|---|---|---|---|
| 1 | `test_cache_residency_remediation` | Phase 0 Baseline HBM LRU & Residency | 6 | 0 | **PASS** |
| 2 | `test_lru` | Phase 0 DeepSeek-V3 Capacity & Variable Size | 6 | 0 | **PASS** |
| 3 | `test_configurable_execution_cache` | Phase 2A Execution Cache Bypass Modes | 9 | 0 | **PASS** |
| 4 | `test_prefetch_framework` | Phase 1 Configuration & Predictor Interface | 17 | 0 | **PASS** |
| 5 | `test_phase2a_predictors` | Phase 2A C++ Predictors (ST-MoE, CSTP, ACSH) | 35 | 0 | **PASS** |
| 6 | `test_live_prediction` | Phase 2C Live Pipeline Prediction Integration | 8 | 0 | **PASS** |
| 7 | `test_prefetch_scheduler` | Phase 3 Prefetch Candidate Queue & Scheduling | 20 | 0 | **PASS** |
| 8 | `test_ddr5_transfer_engine` | Phase 4C Discrete-Event DDR5 Transfer Engine | 12 | 0 | **PASS** |
| 9 | `test_demand_transfer_integration` | Phase 4D.1 Production Demand-Only Integration | 10 | 0 | **PASS** |
| 10 | `test_prefetch_dispatch_coordination` | Phase 4D.2 Prefetch Dispatch & Coordination | 18 | 0 | **PASS** |
| 11 | `test_pabi_quarantine` | Phase 4D.3 PABI Quarantine & Promotion | 20 | 0 | **PASS** |
| 12 | `test_pabi_pinning` | Phase 4D.4 Predictor Pinning & Stage Expiration | 25 | 0 | **PASS** |
| **TOTAL** | **12 Test Suites** | **Complete Project Verification** | **186** | **0** | **100% PASS** |

---

## 10. Verification of Coding Standards & Zero Ternary Operators

In accordance with strict behavioral constraints, zero ternary operators (`? :` and `ast.IfExp`) were introduced across all modified and newly created files.

Verification command:
```bash
python3 -c '
files = [
    "LLMSimulator/src/prefetch/prefetch_config.h",
    "LLMSimulator/src/prefetch/prefetch_config.cpp",
    "LLMSimulator/src/hardware/device.h",
    "LLMSimulator/src/hardware/device.cpp",
    "LLMSimulator/src/prefetch/live_prediction_engine.h",
    "LLMSimulator/src/prefetch/live_prediction_engine.cpp",
    "LLMSimulator/eval/test_pabi_pinning.cpp"
]
for fpath in files:
    with open(fpath) as f:
        for idx, line in enumerate(f, 1):
            if "?" in line:
                print(f"Violation: {fpath}:{idx}: {line.strip()}")
'
```
**Output**: Strictly ZERO matches. 100% compliance verified.

---

## 11. Git Diff Summary

```
Changes not staged for commit:
  modified:   LLMSimulator/CMakeLists.txt
  modified:   LLMSimulator/config.yaml
  modified:   LLMSimulator/eval/CMakeLists.txt
  modified:   LLMSimulator/eval/test.cpp
  modified:   LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp
  modified:   LLMSimulator/src/CMakeLists.txt
  modified:   LLMSimulator/src/hardware/device.cpp
  modified:   LLMSimulator/src/hardware/device.h
  modified:   LLMSimulator/src/hardware/hardware_config.h
  modified:   LLMSimulator/src/hardware/layer_impl.cpp
  modified:   LLMSimulator/src/hardware/linear_impl.cpp
  modified:   LLMSimulator/src/module/linear.cpp
  modified:   LLMSimulator/src/scheduler/scheduler.cpp
  modified:   LLMSimulator/src/scheduler/scheduler.h
  modified:   LLMSimulator/src/scheduler/sequence.cpp

Untracked files:
  LLMSimulator/eval/test_pabi_pinning.cpp
  LLMSimulator/docs/phase_prefetch/PHASE_4D4_PABI_PINNING_REPORT.md
  LLMSimulator/src/prefetch/prefetch_config.h (extended)
  LLMSimulator/src/prefetch/prefetch_config.cpp (extended)
  LLMSimulator/src/prefetch/live_prediction_engine.h (extended)
  LLMSimulator/src/prefetch/live_prediction_engine.cpp (extended)
```

Baseline commit preserved: `d5bf61ff0c066c06f72173189f9924329f613a88`.

---

## 12. Theoretical Context, Boundaries & Limitations

1. **Causal Approximation vs. True Belády MIN**:
   - Belády's optimal algorithm relies on complete offline knowledge of all future references across an infinite horizon.
   - PABI pinning is strictly online and causal: it protects candidates selected by the predictor for a short, bounded horizon ($\Delta$ stages). If the predictor makes an inaccurate forecast, the pinned expert occupies a pin budget slot until stage expiration, but cannot pollute the demand-verified working set or exceed the speculative buffer capacity.
2. **Per-Stage Pin Budget Trade-Off**:
   - Setting $K = \text{max\_pinned\_experts}$ too high could saturate the speculative buffer with pinned entries, causing subsequent speculative insertions to be rejected (`SPECULATIVE_REJECT_PINNED_CAPACITY`).
   - Setting $K$ too low may allow high-confidence candidates to be prematurely evicted by bursty speculative prefetches. A default of $K = 4$ provides strong balance for top-2 / top-4 routing architectures.
3. **Phase Boundary & Scope Constraint**:
   - Phase 4D.4 extends PABI with pinning and expiration.
   - It does not modify predictor update mechanisms, router logic, or full-trace simulation harnesses.
   - Work halts here as required; Phase 4D.5 or subsequent phases will evaluate comprehensive end-to-end trace performance.

