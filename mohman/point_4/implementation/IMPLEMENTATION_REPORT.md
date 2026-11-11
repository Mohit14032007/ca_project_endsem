# Point 4 Phase 2: Frequency-Aware LRU (FA-LRU) Implementation Report

## MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference

---

## 1. Executive Summary

In Point 4 Phase 2, we designed, implemented, and validated the **Frequency-Aware Least Recently Used (FA-LRU)** HBM cache eviction policy in `LLMSimulator`. This implementation directly resolves the memory-wall bottleneck in Mixture-of-Experts (MoE) inference by protecting frequently routed experts from eviction while retaining the cache-friendly spatial and temporal dynamics of standard LRU.

The implementation strictly adheres to the finalized Phase 1.6 architecture specifications:
1. **Dynamic Victim Windowing ($W=16$):** Scans the bottom $W$ least recently used resident experts, identifying the candidate with minimal access frequency.
2. **Lazy Shift-Safe Epoch Discounting:** Implements exponential half-life frequency decay without unbounded scanning overhead, safe against undefined shift behaviors for epoch gaps $\ge 31$.
3. **Primary Weight Hit Isolation:** Disentangles composite tensor weight accesses ($W_1$, $W_3$, $W_2$) so that expert access frequency increments exactly once per stage activation.
4. **Mutually Exclusive Eviction Accounting:** Classifies evictions into strict, orthogonal categories (`UNCONSUMED_PREFETCH_EVICTION`, `DEMAND_RESIDENT_EVICTION`, `PABI_SPECULATIVE_EVICTION`, and `PREFETCH_INDUCED_EVICTION`).
5. **Zero-Ternary Compliance:** Verified zero ternary operators across all C++ source files, headers, and Python harness scripts.

---

## 2. Architecture & Design Rationale

### 2.1 The Need for Frequency Awareness Beyond Standard LRU

Standard LRU evicts the entry at the head of the recency list regardless of how frequently it has been reused. In MoE workloads like DeepSeek-V3 ($N_{\text{routed}}=256, \text{Top-}8$), expert routing exhibits high skewness: a subset of "heavy-hitter" experts are routed across multiple consecutive tokens, while ephemeral experts are routed only once. Under standard LRU with constrained cache capacity:
- A burst of ephemeral experts evicts heavy-hitter experts that were accessed multiple times in preceding stages.
- When those heavy-hitter experts are immediately requested again in subsequent tokens, the GPU experiences costly off-chip DDR5 reload stalls ($1.376\text{ ms}$ per expert load).

FA-LRU addresses this vulnerability by combining **recency-bounded candidate selection** with **frequency evaluation**:
- Instead of unconditionally evicting the absolute least recently used entry, FA-LRU inspects a candidate window consisting of the bottom $W$ entries of the LRU queue.
- Among these $W$ least recently used experts, FA-LRU selects the expert with the lowest decayed access frequency.
- By bounding the window to $W=16$, recently accessed experts in the upper positions of the cache are strictly shielded from evaluation and eviction.

### 2.2 Shift-Safe Lazy Epoch Discounting

To ensure that stale frequency accumulated in early tokens does not indefinitely pin an expert in cache, access frequency undergoes exponential decay across token generations (epochs):
$$F_{\text{decayed}} = \left\lfloor \frac{F_{\text{raw}}}{2^{\Delta e}} \right\rfloor$$
where $\Delta e = e_{\text{current}} - e_{\text{last}}$.

#### Undefined Behavior Avoidance
In ISO C++ (`[expr.shift]`), executing a right shift operation `value >> shift` where `shift >= 32` for a 32-bit integer is undefined behavior (yielding hardware-dependent modulo shifts or CPU traps). We implement shift clamping:
- If $\Delta e \ge 31$, $F_{\text{decayed}} = 0$.
- If $1 \le \Delta e \le 30$, $F_{\text{decayed}} = F_{\text{raw}} >> \Delta e$.
- If $\Delta e = 0$, $F_{\text{decayed}} = F_{\text{raw}}$ (no decay, idempotent).

### 2.3 Primary vs Secondary Demand Disentanglement

DeepSeek-V3 executes each routed expert via three feed-forward projections in sequence:
1. $W_1$ (`gate_proj`)
2. $W_3$ (`up_proj`)
3. $W_2$ (`down_proj`)

Each projection executes `Device::is_expert_resident()` to confirm residency before compute:
- If frequency were naively incremented on every `is_expert_resident()` call, a single stage activation would artificially inflate frequency by $+3$.
- In our implementation, `Tensor` includes a boolean property `is_primary_expert_weight`.
- In `Expert::build_expert_subgraph()`, `is_primary_expert_weight` is set to `true` exclusively for $W_1$, and `false` for $W_3$ and $W_2$.
- In `Device::is_expert_resident()`, the frequency increment occurs if and only if `is_primary_demand == true`.
- Consequently, each stage activation increments frequency exactly once ($+1$).

---

## 3. Data Structures & Algorithmic Flow

### 3.1 Extended Residency Metadata

In `src/hardware/device.h`, `ResidentExpertEntry` is extended:

```cpp
struct ResidentExpertEntry {
    std::pair<int, int> expert_key;
    long long byte_size = 0;
    std::list<std::pair<int, int>>::iterator lru_iter;
    bool is_speculative = false;
    bool was_demanded = false;
    bool is_pinned = false;
    
    // Point 4: FA-LRU Metadata
    uint32_t access_frequency = 1;
    uint32_t last_epoch = 0;
};
```

### 3.2 Candidate Window Victim Selection (`select_fa_lru_victim`)

The victim selection algorithm scans backwards from the LRU end (front of `lru_expert_list`):

```cpp
std::pair<int, int> Device::select_fa_lru_victim() {
    if (lru_expert_list.empty()) {
        return {-1, -1};
    }
    if (fa_lru_window_size <= 1) {
        return lru_expert_list.front();
    }

    auto it = lru_expert_list.begin();
    int count = 0;
    uint32_t min_freq = UINT32_MAX;
    std::pair<int, int> best_victim = {-1, -1};

    while (it != lru_expert_list.end() && count < fa_lru_window_size) {
        auto key = *it;
        auto map_it = resident_experts.find(key);
        if (map_it != resident_experts.end()) {
            if (!map_it->second.is_pinned) {
                materialize_lazy_decay(map_it->second);
                uint32_t f = map_it->second.access_frequency;
                if (f < min_freq) {
                    min_freq = f;
                    best_victim = key;
                }
            }
        }
        it++;
        count++;
    }

    if (best_victim.first == -1) {
        return lru_expert_list.front();
    }
    return best_victim;
}
```

#### Deterministic Tie-Breaking Guarantee
Because `it` traverses from the oldest candidate (`lru_expert_list.begin()`) toward younger candidates, and the condition is strict inequality (`f < min_freq`), when all candidates share identical frequency, `best_victim` retains the first encountered candidate—the absolute oldest entry. This guarantees:
- Complete backward compatibility with standard LRU when frequencies are identical.
- Deterministic and reproducible victim choice without arbitrary hash-iteration jitter.

---

## 4. Instrumentation & Metric Pipeline

### 4.1 Eviction Categorization

Evictions are partitioned into mutually exclusive counters:
1. `UNCONSUMED_PREFETCH_EVICTION`: Speculative prefetch evicted before receiving demand verification (`was_demanded == false`).
2. `DEMAND_RESIDENT_EVICTION`: Demand-verified expert evicted under cache capacity pressure.
3. `PABI_SPECULATIVE_EVICTION`: Evicted from dedicated PABI quarantine buffer.
4. `PREFETCH_INDUCED_EVICTION`: Secondary tracking counter incremented when an incoming speculative prefetch forces eviction of an existing resident expert.

### 4.2 Coalesced vs Routed vs Tensor Demand Tracking

Metrics collector tracks demand granularity across all three architectural levels:
- $D_{\text{routed}} = B \times T \times L \times K$: Raw token-expert routing assignments.
- $D_{\text{coalesced}} = \sum_{t, l} |\bigcup_{b} E_{b, t, l}|$: Unique experts activated per stage per token.
- $D_{\text{tensor}} = 3 \times D_{\text{coalesced}}$: Individual linear layer executions ($W_1, W_3, W_2$).

---

## 5. Summary of Achievements

- Zero regressions across existing baseline, prefetch framework, and timing unit tests.
- 100% pass rate across 12 newly developed Point 4 unit tests (`test_fa_lru`).
- 100% pass rate across all 8 invariant assertions in the 4-run smoke test campaign.
- Absolute conservation of memory residency:
  $$\text{insertions} - \text{evictions} = |\text{resident\_experts}|$$
- Zero ternary operators across all C++ code, headers, and Python harness scripts.

