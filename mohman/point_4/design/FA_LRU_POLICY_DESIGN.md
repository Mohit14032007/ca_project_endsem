# Frequency-Aware LRU (FA-LRU) Eviction Policy Design

## MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference

---

## 1. Motivation & Problem Formulation

In Mixture-of-Experts (MoE) workloads, expert activation distributions are strongly skewed:
- A small subset of experts ("heavy hitters" or hot experts) account for a disproportionately large fraction of all routing decisions across tokens.
- A long tail of experts are accessed rarely or intermittently.

Standard Least Recently Used (LRU) cache management operates under the assumption that temporal recency is a sufficient surrogate for future access probability:
$$\mathbb{P}(\text{Access at } t + \Delta t) \propto \frac{1}{\text{Time since last access}}$$

However, in autoregressive MoE inference:
1. Each decoding token routes tokens through all MoE stages sequentially ($L = 0, \dots, 39$).
2. In stage $L$, Top-$K$ routing selects $K$ experts (e.g., $K=8$ out of 256).
3. If an expert was accessed in token $T$, it may not be accessed in token $T+1$ if intermediate stages or other tokens temporarily route elsewhere.
4. Under standard LRU, if cache capacity is constrained (e.g., 35 GB holding ~397 experts), bursty routing demands in intervening layers will push the heavy hitter toward the LRU eviction front.
5. LRU blindly evicts this high-frequency expert simply because its recency was temporarily eclipsed, causing an expensive DDR5 reload miss when it is demanded again just one or two tokens later. This phenomenon is known as **cache thrashing**.

**Objective of FA-LRU:**
Balance **recency** (to adapt to dynamic conversational topic shifts) and **frequency** (to protect hot experts from premature eviction).

---

## 2. Mathematical Formulation & Policy Rules

### 2.1 Metadata Representation
For each resident expert key $k = (l, e)$ resident in HBM, the policy maintains:
- $F(k) \in \mathbb{N}^+$: Effective access frequency (activation count).
- $t_{\text{last}}(k) \in \mathbb{R}^+$: Simulated timestamp (ns) of the most recent access.
- $t_{\text{insert}}(k) \in \mathbb{R}^+$: Timestamp when the expert was loaded into HBM.
- $e_{\text{epoch}}(k) \in \mathbb{N}$: Epoch counter at admission.

### 2.2 Access & Frequency Update Rules

#### Rule 1: Demand Access Hit
When expert $k$ is accessed by an actual computation demand:
$$F(k) \leftarrow F(k) + 1$$
$$t_{\text{last}}(k) \leftarrow t_{\text{sim}}$$
- Splicing: The expert's node in `lru_expert_list` is moved to the back (MRU position).
- Time Complexity: $O(1)$.

#### Rule 2: Demand Miss Admission (Fresh DDR5 Load)
When expert $k$ misses in HBM and is loaded from DDR5:
$$F(k) \leftarrow 1$$
$$t_{\text{last}}(k) \leftarrow t_{\text{sim}}$$
$$t_{\text{insert}}(k) \leftarrow t_{\text{sim}}$$
- Position: Inserted at `lru_expert_list.back()` (MRU position).
- Time Complexity: $O(1)$.

#### Rule 3: Speculative Prefetch Admission
When a speculative prefetch completes before actual demand:
$$F(k) \leftarrow 1$$
$$t_{\text{last}}(k) \leftarrow t_{\text{sim}}$$
$$\text{was\_demanded}(k) \leftarrow \text{false}$$
> [!IMPORTANT]
> Speculative prefetches enter with baseline frequency $F(k) = 1$. They **do not** receive boosted frequency. Only when an actual token GEMM demands the prefetched expert is Rule 1 invoked, elevating $F(k) \leftarrow 2$. This prevents unverified speculative prefetches from polluting the high-frequency cache tiers.

#### Rule 4: Speculative Probing (In-Flight / Cache Check)
When the predictor or scheduler probes HBM residency with `is_demand_access = false`:
- Neither $F(k)$ nor $t_{\text{last}}(k)$ is modified.
- Recency list ordering is strictly preserved.

---

## 3. Aging & Frequency Decay Mechanism

Without decay, early-phase heavy hitters would accumulate high frequency counts and permanently resist eviction long after the model's routing focus has shifted ("frequency pollution").

### 3.1 Periodic Epoch Halving
To ensure responsiveness to dynamic routing transitions, FA-LRU employs an **epoch decay mechanism**:
- Every $N_{\text{epoch}}$ execution stages (or every $N_{\text{tokens}}$ generated tokens), an epoch decay event is triggered:
$$F(k) \leftarrow \max\left(1, \left\lfloor \frac{F(k)}{2} \right\rfloor \right) \quad \forall k \in \text{resident\_experts}$$
- **Decay Parameter:** $N_{\text{decay}} = 40 \text{ stages}$ (equivalent to 1 full token pass across all MoE layers).
- **Properties:**
  - Preserves frequency rank ordering.
  - Bounds the maximum frequency value, preventing integer overflow.
  - Ensures that recently activated experts can out-compete historically active but currently idle experts.

---

## 4. Victim Selection Algorithms

When `resident_bytes + expert_size > cache_capacity_bytes`, one or more victims must be evicted. We analyze three architectural alternatives and present the recommended design.

### Alternative A: Window-Based Bottom-$W$ Sampled FA-LRU (Recommended)

#### Mechanism
Rather than scanning all $N$ entries in HBM ($N \approx 400\text{--}900$), FA-LRU inspects a window of $W$ candidates from the least-recently used end of `lru_expert_list` (the front of the list):
$$\mathcal{C}_W = \{ \text{lru\_expert\_list}[0], \text{lru\_expert\_list}[1], \dots, \text{lru\_expert\_list}[W-1] \}$$
From $\mathcal{C}_W$, select the candidate with the lowest effective frequency:
$$k^* = \arg\min_{k \in \mathcal{C}_W} F(k)$$

#### Tie-Breaking Rule
If multiple candidates in $\mathcal{C}_W$ have the identical minimum frequency:
$$\text{Select the candidate closest to the front of } \text{lru\_expert\_list} \text{ (oldest recency)}.$$

#### Window Size Calibration
- If $W = 1$: Degenerates identically to standard LRU.
- If $W = 16$: Inspects the 16 coldest entries in recency. A heavy hitter with $F=15$ sitting in the bottom 16 is protected; a cold expert with $F=1$ is evicted instead.
- If $W = 32$: Stronger frequency protection.
- **Recommended Window:** $W = 16$. With $W = 16$, victim selection requires at most 16 pointer dereferences, executing in strictly bounded $O(1)$ time ($< 1\,\mu\text{s}$ per eviction).

### Alternative B: Segmented Two-Queue FA-LRU (SLRU-Style)
- Divides cache into two physical lists:
  1. **Probationary List ($L_1$):** Allocated $30\%$ of capacity. New admissions enter here.
  2. **Protected List ($L_2$):** Allocated $70\%$ of capacity. Experts with $F \ge 2$ reside here.
- On a hit in $L_1$: Promoted to $L_2$.
- When $L_2$ overflows: Demotes LRU entry of $L_2$ to $L_1$.
- Evictions always occur from the front of $L_1$.
- *Trade-off:* Requires managing two distinct capacity thresholds and handling multi-list synchronization.

### Alternative C: Unified Cost Scoring Function
- Compute score:
$$S(k) = \alpha \cdot \frac{t_{\text{sim}} - t_{\text{last}}(k)}{\Delta t_{\max}} - (1 - \alpha) \cdot \frac{F(k)}{F_{\max}}$$
- Evict candidate with maximum $S(k)$.
- *Trade-off:* Requires heap maintenance or $O(N)$ full-cache sweeps, creating noticeable simulation overhead.

---

## 5. Architectural Recommendation & Algorithm Specification

We recommend **Window-Based Bottom-$W$ Sampled FA-LRU ($W=16$)** with **Epoch Halving**:
1. **Zero Data Structure Divergence:** Utilizes the existing `lru_expert_list` and `resident_experts` map without requiring auxiliary heaps or segmented buffers.
2. **Strict $O(1)$ Complexity:** Bounded to exactly $\min(W, |\text{lru\_expert\_list}|)$ iterations.
3. **Graceful Degeneration:** Seamlessly falls back to pure LRU by setting $W = 1$.
4. **Provable Determinism:** Complete reproducibility across simulation runs with zero nondeterministic floating-point hazards.

### Detailed C++ Algorithm Pseudocode
```cpp
// Victim selection within mark_expert_resident
std::list<std::tuple<int, int, long long>>::iterator Device::select_fa_lru_victim() {
    const size_t window_size = 16;
    auto best_it = lru_expert_list.begin();
    uint32_t min_freq = UINT32_MAX;
    
    size_t count = 0;
    for (auto it = lru_expert_list.begin(); it != lru_expert_list.end(); ++it) {
        if (count >= window_size) {
            break;
        }
        int l = std::get<0>(*it);
        int e = std::get<1>(*it);
        auto map_it = resident_experts.find({l, e});
        if (map_it != resident_experts.end()) {
            uint32_t freq = map_it->second.access_frequency;
            // Strict inequality ensures tie-breaks favor earlier (colder) entries
            if (freq < min_freq) {
                min_freq = freq;
                best_it = it;
            }
        }
        count += 1;
    }
    return best_it;
}
```

---

## 6. Determinism & Coding Standards

1. **Zero Ternary Operators:** No `? :` in C++ code, no `ast.IfExp` in Python harnesses, and no ternary prose.
2. **Integer Arithmetic:** Frequency counting and epoch decay utilize unsigned 32-bit integers (`uint32_t`).
3. **Invariants:**
   - $\sum_{(l,e)} \text{size\_bytes} = \text{resident\_bytes} \le \text{cache\_capacity\_bytes}$.
   - $|\text{resident\_experts}| = |\text{lru\_expert\_list}|$.
   - For every entry in `resident_experts`, `list_it` dereferences to the identical `(layer_id, expert_id)`.

