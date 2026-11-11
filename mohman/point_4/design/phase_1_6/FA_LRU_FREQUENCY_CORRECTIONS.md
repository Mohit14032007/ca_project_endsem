# FA-LRU Frequency Accounting Corrections

## Point 4 — Phase 1.6 Consistency Audit & Design Corrections

---

## 1. Executive Summary & Audit Context

To prevent frequency corruption under the multi-projection architecture of DeepSeek-V3, this document verifies the exact call path through [`Linear::launch`](file:///home/mohman/ca_project/LLMSimulator/src/module/linear.cpp#L49-L115) and [`Device::is_expert_resident`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L670-L747). It formalizes an explicit primary-demand dispatch mechanism to guarantee that frequency increments occur **exactly once per stage activation** without depending on fragile string comparisons.

---

## 2. Forensic Call-Path Verification

Tracing the execution of an expert forward pass through `FeedForward3Way::forward` ([`src/module/layer.cpp:415-437`](file:///home/mohman/ca_project/LLMSimulator/src/module/layer.cpp#L415-L437)):
```cpp
Tensor::Ptr gate_out = (*gate_proj)(input, sequences_metadata);   // W1_gate
Tensor::Ptr act_out = (*act_fn)(gate_out, sequences_metadata);
Tensor::Ptr up_proj_out = (*up_proj)(input, sequences_metadata); // W3_up
Tensor::Ptr ffn_out = (*down_proj)(up_proj_out, sequences_metadata); // W2_down
```

### Call Sequence Properties
1. **$W_1$ (`gate_proj`) Execution:**
   - In `Linear::launch` ([`linear.cpp:106`](file:///home/mohman/ca_project/LLMSimulator/src/module/linear.cpp#L106)), $W_1$ calls `device->is_expert_resident(layer_id, A->expert_id)`.
   - **Case A (Miss):** `is_expert_resident()` returns `false`. $W_1$ marks `A->weight_target = OFFCHIP_DRAM` and `A->load_sisters = true`. In `linear_impl.cpp:54-60`, $W_1, W_3, W_2$ are batched into a single DDR5 transfer. Upon completion, `device->mark_expert_resident()` is called once for the entire expert ($88.08 \text{ MB}$).
   - **Case B (Hit):** `is_expert_resident()` returns `true`.
2. **$W_3$ (`up_proj`) and $W_2$ (`down_proj`) Execution:**
   - When $W_3$ runs, it calls `device->is_expert_resident(layer_id, A->expert_id)`.
   - Because $W_1$ already marked the expert resident (or loaded it), $W_3$ hits in HBM.
   - When $W_2$ runs, it calls `device->is_expert_resident(layer_id, A->expert_id)`. It also hits in HBM.

**Verified Architectural Fact:**
$W_1$ is **always and deterministically the first projection executed** for an expert within an MoE stage. $W_3$ and $W_2$ always follow sequentially.

---

## 3. String Comparison vs. Explicit Primary-Demand Flag

In Phase 1.5, checking `A->name == "W1_gate"` was considered. We now evaluate the safety and robustness of this approach against an explicit flag.

### Trade-Off Analysis

| Dimension | Option A: String Comparison (`A->name == "W1_gate"`) | Option B: Explicit Flag (`is_primary_demand = true`) |
| :--- | :--- | :--- |
| **Robustness** | **Fragile.** Fails if tensor naming conventions change or if other modules rename projection weights. | **Rock Solid.** Semantically tied to the projection topology in `Tensor` and `Device`. |
| **Performance** | Incurs `std::string::operator==` string comparison overhead on every GEMM launch. | Bounded integer/boolean register comparison; zero string operations. |
| **API Clarity** | Implicit; caller cannot control whether an access is primary. | Explicit; caller explicitly defines demand semantics in function signature. |

### Decision: Adopt Explicit Primary-Demand Flag (Option B)

1. In [`src/module/tensor.h`](file:///home/mohman/ca_project/LLMSimulator/src/module/tensor.h), add:
   ```cpp
   bool is_primary_expert_weight = false;
   ```
2. In [`src/module/expert.cpp:108-114`](file:///home/mohman/ca_project/LLMSimulator/src/module/expert.cpp#L108-L114):
   ```cpp
   w1->is_primary_expert_weight = true;
   w3->is_primary_expert_weight = false;
   w2->is_primary_expert_weight = false;
   ```
3. Update `Device::is_expert_resident` signature in [`src/hardware/device.h`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.h):
   ```cpp
   bool is_expert_resident(int layer_id, int expert_id, 
                           bool is_demand_access = true, 
                           bool is_primary_demand = true);
   ```
4. In [`src/module/linear.cpp:106`](file:///home/mohman/ca_project/LLMSimulator/src/module/linear.cpp#L106):
   ```cpp
   bool is_primary = A->is_primary_expert_weight;
   if (!device->is_expert_resident(layer_id, A->expert_id, true, is_primary)) {
   ```

---

## 4. Verification of Lifecycle State Transitions

Using the explicit `is_primary_demand` flag, we trace all expert lifecycle states:

### 1. Demand Miss Insertion
- When $W_1$ misses, it loads all three projection tensors and calls `mark_expert_resident()`.
- `mark_expert_resident()` initializes `access_frequency = 1` and `last_epoch = global_epoch`.
- When $W_3$ and $W_2$ execute, they pass `is_primary_demand = false`.
- `is_expert_resident()` sees `is_primary_demand == false`, splices the entry to MRU, but **does not increment frequency**.
- **Result:** After the stage finishes, `access_frequency` is **exactly 1**. Zero unintended inflation occurs.

### 2. Subsequent Demand Hits on Resident Expert
- On a subsequent token, $W_1$ hits and passes `is_primary_demand = true`.
- `is_expert_resident()` increments `access_frequency` by 1 ($1 \to 2$) and splices to MRU.
- When $W_3$ and $W_2$ hit, they pass `is_primary_demand = false` and do not increment frequency.
- **Result:** `access_frequency` increments by **exactly 1 per stage activation**.

### 3. Resident Prefetched Expert (First Demand Hit)
- Completed prefetch enters HBM via `mark_expert_demand()`, setting `access_frequency = 1` and `was_demanded = false`.
- First demand arrives at $W_1$: hits in HBM with `is_primary_demand = true`.
- `is_expert_resident()` detects `was_demanded == false`, sets `was_demanded = true`, increments frequency to $F = 2$, and records a timely prefetch hit.
- $W_3$ and $W_2$ hit with `is_primary_demand = false`; frequency remains 2.
- **Result:** Frequency is incremented **exactly once** on first consumption.

### 4. In-Flight Prefetch Promoted to Demand
- Demand arrives while transfer is active/queued in `DDR5TransferEngine`.
- Attaches to transfer via `promote_inflight_to_demand()`.
- When transfer finishes in `process_completed_prefetches()`, it calls `mark_expert_resident()` with $F = 1$ and `was_demanded = true`.
- When $W_3$ and $W_2$ run, they pass `is_primary_demand = false`, preserving $F = 1$.
- **Result:** Transfer runs physically once; frequency is initialized to 1.

### 5. Speculative Probing
- Prefetch candidate evaluation calls `is_expert_resident(..., is_demand_access = false)`.
- Line 681 explicitly short-circuits:
  ```cpp
  if (!is_demand_access) {
    return true;
  }
  ```
- **Result:** Speculative probes **never** modify frequency, recency, or cache metadata.

