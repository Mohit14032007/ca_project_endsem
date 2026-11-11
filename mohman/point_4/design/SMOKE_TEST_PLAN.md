# Smoke Test Plan — LRU vs FA-LRU Validation

## MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference

---

## 1. Objectives & Scope

Before executing the full experimental campaign, a minimal, fast, deterministic **Smoke Test** must be conducted to validate the correctness of the FA-LRU eviction policy implementation.

The smoke test validates that:
1. FA-LRU compiles and links without syntax errors, missing symbols, or linker issues.
2. The simulation completes successfully (exit code 0) on a real routing trace.
3. HBM cache capacity constraints are strictly preserved ($resident\_bytes \le cache\_capacity\_bytes$).
4. Evictions occur actively and are logged accurately.
5. Invariant integrity between `resident_experts` map and recency ordering lists is preserved.
6. Demand routing decisions are identical across comparable LRU and FA-LRU runs.
7. No memory leaks, segfaults, or dangling iterators occur.

> [!CAUTION]
> This smoke test is **not executed during Phase 1**. It is fully specified here and will be run immediately following implementation approval in Phase 2.

---

## 2. Workload & Test Configuration

To force immediate, active eviction pressure within a short execution window without requiring lengthy simulation time:

- **Workload Parameters:**
  - Model: DeepSeek-V3 architecture ($N_{\text{routed}} = 256$, Top-$K=8$, $88.08 \text{ MB}$ per expert).
  - Trace File: MoE-Anywhere Prompt 60 (`moe_anywhere_prompts_60_to_67/prompt_60.csv`).
  - Batch Size: $B = 1$.
  - Output Tokens: $T = 4$ tokens.
  - Active MoE Stages: 5 stages (stages 0 to 4).
  - Total Demand Events: $1 \text{ seq} \times 4 \text{ tokens} \times 5 \text{ stages} \times 8 \text{ Top-}K = 160 \text{ expert demands}$.

- **Deliberately Constrained HBM Capacity:**
  - Parameter: `DEBUG_CACHE_CAPACITY=2000000000` (2.0 GB).
  - Maximum Resident Capacity: $\lfloor 2,000,000,000 / 88,080,384 \rfloor = 22 \text{ experts}$.
  - *Rationale:* With 160 expert demands and a cache capacity of only 22 experts, the cache will fill completely within token 1, guaranteeing that tokens 2, 3, and 4 will execute dozens of active evictions and reloads under heavy cache pressure.

---

## 3. Test Matrix (4 Runs)

| Run ID | Configuration | Prefetching | Eviction Policy | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **ST-1** | Baseline LRU | Disabled (`mode: none`) | Standard `LRU` | Reference baseline with active evictions |
| **ST-2** | Baseline FA-LRU | Disabled (`mode: none`) | `FA_LRU` ($W=16$) | Pure eviction policy verification |
| **ST-3** | CSTP LRU | CSTP ($H=1, B=4$) | Standard `LRU` | Prefetching reference |
| **ST-4** | CSTP FA-LRU | CSTP ($H=1, B=4$) | `FA_LRU` ($W=16$) | Prefetch + FA-LRU interaction check |

---

## 4. Verification Checklist & Invariant Assertions

The post-run validation script ([`scripts/verify_smoke_test.py`](file:///home/mohman/ca_project/mohman/point_4/scripts/verify_smoke_test.py)) will check the following 8 invariants:

1. **Exit Code:**
   Every simulation process terminates with returncode `0`.
2. **Deterministic Routing Integrity:**
   The exact sequence of requested `{layer_id, expert_id}` pairs in `expert_cache.csv` is identical across ST-1 and ST-2.
3. **Strict Capacity Upper Bound:**
   $$\forall t, \quad \text{resident\_bytes}(t) \le \text{cache\_capacity\_bytes} \quad (2,000,000,000 \text{ bytes})$$
4. **Residency Table & List Size Parity:**
   $$|\text{resident\_experts}| = |\text{lru\_expert\_list}|$$
5. **Conservation of Demand Accounting:**
   $$\text{demand\_insertions} - \text{demand\_evictions} = |\text{resident\_experts}|$$
6. **No Phantom Residency:**
   Every expert marked as evicted in `expert_cache.csv` with action `EVICT` is removed from `resident_experts` and subsequent demands correctly register a `MISS` or fresh DDR5 load.
7. **Zero Duplicate Entries:**
   No `(layer_id, expert_id)` key appears more than once in `lru_expert_list`.
8. **Active Eviction Verification:**
   Total evictions in ST-1 and ST-2 must both be greater than 50 ($E_{\text{total}} \ge 50$), confirming that the constrained capacity successfully triggered genuine eviction pressure.

---

## 5. Decision Criteria

At the conclusion of the smoke test, the results will be classified into one of three outcomes:

### **GO Decision**
- All 8 correctness and invariant checks pass.
- Active evictions ($E_{\text{total}} \ge 50$) occur under both policies.
- Metrics show a logical and promising trend in at least one meaningful metric (e.g., higher HBM hit rate, fewer reloads, lower DDR5 traffic, or lower exposed stall time under FA-LRU) without unacceptable regressions.
- *Action:* Proceed to full Phase 3 experimental campaign across W1–W7.

### **INVESTIGATE Decision**
- Correctness and invariant checks pass, but eviction counts are near zero or metrics show completely flat behavior across both policies.
- *Root Cause Analysis:* Cache capacity was too large, trace sequence was too short, or frequency decay parameter $N_{\text{decay}}$ needs adjustment.
- *Action:* Adjust capacity constraint or trace length and re-run smoke test.

### **NO-GO Decision**
- Any correctness or invariant assertion fails (e.g., segfault, capacity overflow, routing mismatch, corrupted recency iterator, or severe unexplained performance regression).
- *Action:* Halt immediately. Root-cause bug in `device.cpp` or policy dispatch. Do not proceed to full experiments.

