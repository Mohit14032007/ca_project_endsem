# Point 4 Phase 2.5: Eviction Accounting & Pinning Protection Validation

## MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference

---

## 1. Executive Summary & Problem Formulation

In modern memory hierarchy management with predictive prefetching, evictions arise from two competing traffic classes: demand traffic and speculative prefetch traffic. 

During the Phase 2 implementation audit, two potential hazards were identified:
1. **Eviction Category Double-Counting:** The classification `PREFETCH_INDUCED_EVICTION` was displayed in parallel with primary eviction categories (`DEMAND_RESIDENT_EVICTION`, `UNCONSUMED_PREFETCH_EVICTION`, and `PABI_SPECULATIVE_EVICTION`). Summing these categories resulted in double-counting, because a prefetch-induced eviction is fundamentally a demand-resident eviction triggered by an incoming prefetch transfer.
2. **Pinned-Entry Eviction Vulnerability:** When Belády-style or PABI stage-deadline pinning is enabled, resident entries can be pinned in HBM to guarantee zero-latency execution. In an under-constrained victim search, an unpinned candidate could be overlooked or, worse, a pinned entry evicted, violating execution guarantees. If all resident entries are pinned, an incoming prefetch could overrun the physical memory capacity.

Both issues have been remediated, formally specified, and verified through unit tests in `eval/test_fa_lru.cpp`.

---

## 2. Strict Mutual Exclusion of Primary Eviction Categories

### 2.1 The Conservation Identity
Every evicted expert from HBM must belong to exactly **one** primary eviction classification:

$$E_{\text{total}} = E_{\text{unconsumed\_prefetch}} + E_{\text{demand\_resident}} + E_{\text{PABI\_speculative}}$$

Where:
- **$E_{\text{unconsumed\_prefetch}}$ (`UNCONSUMED_PREFETCH_EVICTION`):** An expert that was brought into HBM speculatively by a prefetcher, but was evicted before any demand access ever consumed it (wasted speculative traffic / cache pollution).
- **$E_{\text{demand\_resident}}$ (`DEMAND_RESIDENT_EVICTION`):** An expert that had previously been demanded by compute (or promoted upon demand arrival) and was subsequently chosen as an eviction victim.
- **$E_{\text{PABI\_speculative}}$ (`PABI_SPECULATIVE_EVICTION`):** An expert evicted from the dedicated speculative quarantine buffer under the PABI policy.

### 2.2 Orthogonal Attribution: Prefetch-Induced Evictions
`PREFETCH_INDUCED_EVICTION` is **not** a fourth additive category. Instead, it is an orthogonal attribution flag:
$$E_{\text{prefetch\_induced}} \subseteq E_{\text{demand\_resident}}$$
When an incoming speculative prefetch arrives and finds the cache full, it forces an eviction. If the victim was a demand-verified resident expert, the primary counter $E_{\text{demand\_resident}}$ increments, and the secondary attribution counter $E_{\text{prefetch\_induced}}$ also increments:

$$\text{assert}(E_{\text{prefetch\_induced}} \le E_{\text{demand\_resident}})$$
$$\text{assert}(E_{\text{total}} == E_{\text{unconsumed\_prefetch}} + E_{\text{demand\_resident}} + E_{\text{PABI\_speculative}})$$

### 2.3 Verification via Unit Test 13
Test 13 in `eval/test_fa_lru.cpp` explicitly simulates one event of each type:
1. One unconsumed prefetch eviction ($E_{\text{unconsumed}} = 1$).
2. One demand-resident eviction caused by a demand miss ($E_{\text{demand}} = 1, E_{\text{induced}} = 0$).
3. One demand-resident eviction caused by an admitted prefetch ($E_{\text{demand}} = 2, E_{\text{induced}} = 1$).
4. One PABI speculative quarantine eviction ($E_{\text{PABI\_spec}} = 1$).

**Outcome:**
$$E_{\text{total}} = 1 + 2 + 1 = 4$$
$$E_{\text{prefetch\_induced}} = 1 \le E_{\text{demand\_resident}} (2)$$
The conservation identity holds with 100% mathematical precision.

---

## 3. Pinned-Entry Protection Architecture in FA-LRU

When an expert is actively pinned (e.g., during critical stage execution), it must **never** be selected as an eviction victim. 

### 3.1 Two-Pass Selection Algorithm
`Device::select_fa_lru_victim()` implements a robust, two-pass search:

```cpp
std::list<std::tuple<int, int, long long>>::iterator Device::select_fa_lru_victim() {
    if (lru_expert_list.empty()) {
        return lru_expert_list.end();
    }

    // PASS 1: Search bottom-W window for unpinned candidate with minimum decayed frequency
    int count = 0;
    auto it = lru_expert_list.begin();
    auto best_it = lru_expert_list.end();
    uint32_t min_freq = UINT32_MAX;

    while (it != lru_expert_list.end() && count < fa_lru_window_size) {
        int l = std::get<0>(*it);
        int e = std::get<1>(*it);
        auto rentry_it = resident_experts.find(std::make_pair(l, e));
        if (rentry_it != resident_experts.end()) {
            if (!rentry_it->second.is_pinned) {
                materialize_lazy_decay(rentry_it->second);
                uint32_t f = rentry_it->second.frequency;
                if (f < min_freq) {
                    min_freq = f;
                    best_it = it;
                }
            }
        }
        it++;
        count++;
    }

    if (best_it != lru_expert_list.end()) {
        return best_it;
    }

    // PASS 2: Fallback beyond bottom-W window if all window entries are pinned
    it = lru_expert_list.begin();
    while (it != lru_expert_list.end()) {
        int l = std::get<0>(*it);
        int e = std::get<1>(*it);
        auto rentry_it = resident_experts.find(std::make_pair(l, e));
        if (rentry_it != resident_experts.end() && !rentry_it->second.is_pinned) {
            return it;
        }
        it++;
    }

    // ALL entries in cache are pinned: return end()
    return lru_expert_list.end();
}
```

### 3.2 Handling Fully Pinned Cache Capacity
If all resident entries in HBM are pinned, `select_fa_lru_victim()` safely returns `lru_expert_list.end()`. 

In `Device::mark_expert_resident()`:
```cpp
if (victim_it == lru_expert_list.end()) {
    // If admitting speculative prefetch, reject admission to protect capacity
    if (is_admitting_prefetch) {
        prefetch_metrics.candidate_admission_rejected++;
        return;
    }
}
```
If an incoming speculative prefetch arrives when all entries are pinned, prefetch admission is rejected, the transfer is discarded, and the physical capacity bound is strictly maintained.

---

## 4. Unit Test Verification (Tests 9, 10, 11)

The pinning protection mechanisms were verified across three comprehensive tests in `eval/test_fa_lru.cpp`:

### 4.1 Test 9: Oldest Entry Pinned in Window
- **Setup:** Window $W=4$. Cache contains $(0, 1)$ [oldest, $F=1$], $(0, 2)$ [$F=2$], $(0, 3)$ [$F=3$], $(0, 4)$ [$F=4$].
- **Action:** Expert $(0, 1)$ is marked `is_pinned = true`.
- **Victim Selection:** Pass 1 skips pinned $(0, 1)$ and selects $(0, 2)$ (the unpinned candidate with lowest frequency $F=2$).
- **Result:** `[PASS]` Pinned oldest entry protected.

### 4.2 Test 10: All Window Candidates Pinned
- **Setup:** Window $W=2$. Cache contains 4 entries: $(0, 1), (0, 2), (0, 3), (0, 4)$. Both window candidates $(0, 1)$ and $(0, 2)$ are marked pinned.
- **Action:** Select victim.
- **Victim Selection:** Pass 1 finds 0 unpinned candidates in $W=2$. Pass 2 activates, searches beyond window, and selects $(0, 3)$ (first unpinned candidate).
- **Result:** `[PASS]` Fallback to unpinned candidate beyond window verified.

### 4.3 Test 11: 100% Pinned Cache & Prefetch Rejection
- **Setup:** Cache capacity = 2 entries. Both entries $(0, 1)$ and $(0, 2)$ are marked pinned.
- **Action 1:** Select victim. Returns `end()`.
- **Action 2:** Call `mark_expert_resident(..., is_admitting_prefetch = true)` for expert $(0, 3)$.
- **Verification:**
  - Neither $(0, 1)$ nor $(0, 2)$ is evicted.
  - Expert $(0, 3)$ is **not** admitted to cache.
  - `candidate_admission_rejected` metric increments.
  - Cache size remains exactly 2 entries.
- **Result:** `[PASS]` 100% pinned cache invariant preserved without corruption or overflow.

---

## 5. Summary Matrix of Eviction & Pinning Guarantees

| Invariant / Property | Mechanism | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| **Mutual Exclusion** | Single classification per eviction event | Unit Test 13 | VERIFIED |
| **Attribution Distinction** | $E_{\text{induced}} \subseteq E_{\text{demand\_resident}}$ | Unit Test 13 & Smoke S3/S4 | VERIFIED |
| **Pinning Priority** | Pass 1 skips `is_pinned == true` | Unit Test 9 | VERIFIED |
| **Window Fallback** | Pass 2 inspects remainder of LRU list | Unit Test 10 | VERIFIED |
| **Capacity Preservation** | Rejects prefetch if all pinned | Unit Test 11 | VERIFIED |

