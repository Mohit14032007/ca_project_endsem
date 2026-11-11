# Point 4 Phase 2.5: Demand Accounting Validation

## MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference

---

## 1. Executive Summary & Problem Formulation

In Mixture-of-Experts (MoE) architectures such as DeepSeek-V3, memory demands occur across three distinct structural and temporal granularities:
1. **Logical Token Routing ($D_{\text{routed}}$):** Each token in batch $B$ routes to $\text{Top-}K$ experts across $L$ layers ($B \times T \times L \times K$).
2. **Coalesced Physical Demands ($D_{\text{coalesced}}$):** Within a single decode step $s$, multiple tokens in batch $B$ may select the identical expert $(l, e)$. Modern hardware engines coalesce these demands into a single physical DRAM/HBM check and potential memory transfer.
3. **Execution Tensor Sub-Operations ($D_{\text{tensor}}$):** Each expert feed-forward block comprises three dense GEMM projections: gate projection ($W_1$), up projection ($W_3$), and down projection ($W_2$). Hence, each activated expert undergoes three tensor execution events.

### Phase 2 Accounting Vulnerabilities
During the Phase 2 implementation audit, two critical accounting ambiguities were uncovered:
- **Multiplying by 3 Fallacy:** In previous reporting, tensor executions were estimated by multiplying coalesced demands by 3 ($3 \times D_{\text{coalesced}}$), which is invalid when tensor events are recorded asynchronously or when execution pipelines drop or coalesce sub-projections.
- **Flawed Batch Coalescing Key:** Demands were previously deduplicated using `(token_id, layer_id, expert_id)`. In multi-sequence batch serving, tokens across different sequences may share identical relative `token_id`s (e.g., token index 0 of sequence A and token index 0 of sequence B) while being processed in the same decode step, or conversely have different `token_id`s in the same batch step due to variable prompt lengths. Using `token_id` failed to properly coalesce batched demands.

---

## 2. Formal Mathematical Definitions of Denominators

To ensure mathematical precision and eliminate ambiguities, LLMSimulator's `PrefetchMetricsCollector` now strictly disentangles and records three distinct demand counters:

### 2.1 Execution-Level Tensor Events ($D_{\text{tensor}}$)
$$D_{\text{tensor}} = |\text{demands\_}|$$
$D_{\text{tensor}}$ represents the raw count of all recorded execution events in `CollectorDemandRecord`. Every call to `record_expert_demanded()` adds exactly one record. There is no synthetic multiplier; it reflects actual hardware tensor invocations.

### 2.2 Logical Routed Demands ($D_{\text{routed}}$)
$$D_{\text{routed}} = \left| \bigcup_{i=0}^{|\text{demands\_}|-1} \Big( \text{req\_id}_i, \text{seq\_id}_i, \text{tok\_id}_i, \text{layer\_id}_i, \text{expert\_id}_i \Big) \right|$$
$D_{\text{routed}}$ aggregates individual token-to-expert routing assignments, deduplicating the three projection passes ($W_1, W_3, W_2$) belonging to the same token, sequence, and request. For a model with batch size $B$, sequence length $T$, layers $L$, and $\text{Top-}K$, the theoretical logical routed demands are:
$$D_{\text{routed}} = B \times T \times L \times K$$

### 2.3 Coalesced Physical Demands ($D_{\text{coalesced}}$)
$$D_{\text{coalesced}} = \left| \bigcup_{i=0}^{|\text{demands\_}|-1} \Big( \text{decode\_step}_i, \text{layer\_id}_i, \text{expert\_id}_i \Big) \right|$$
$D_{\text{coalesced}}$ represents the unique physical expert activations within each generation step $s$. If $N$ tokens in batch $B$ route to expert $e$ in layer $l$ during decode step $s$, they coalesce into exactly **one** physical demand event $(s, l, e)$.

---

## 3. Hit Rate Formulations

Corresponding to each denominator, the hit rates are rigorously defined:

| Metric Name | Hit Rate Formula | Description |
| :--- | :--- | :--- |
| **Coalesced HBM Hit Rate** | $$H_{\text{coalesced}} = \frac{\text{coalesced\_resident\_hits}}{D_{\text{coalesced}}}$$ | Primary hardware cache metric: fraction of physical batch expert accesses resolved in HBM without DDR5 stalls. |
| **Routed Demand Hit Rate** | $$H_{\text{routed}} = \frac{\text{routed\_resident\_hits}}{D_{\text{routed}}}$$ | Token-level routing perspective: fraction of token expert assignments that find their expert resident. |
| **Tensor Execution Hit Rate** | $$H_{\text{tensor}} = \frac{\text{tensor\_resident\_hits}}{D_{\text{tensor}}}$$ | Sub-operation perspective: fraction of $W_1, W_3, W_2$ kernel executions where the expert weights are resident. |

> [!NOTE]
> When an expert is fetched for $W_1$, subsequent sub-projections $W_3$ and $W_2$ within the same execution block always hit in cache unless evicted by an aggressive speculative prefetch. Therefore, $H_{\text{tensor}}$ is naturally higher than $H_{\text{coalesced}}$. To avoid misleading performance claims, **$H_{\text{coalesced}}$ is our authoritative primary figure of merit**.

---

## 4. Empirical Validation: Scenarios A–E

To rigorously verify the implementation, `eval/test_fa_lru.cpp` (Test 12) implements five targeted operational scenarios:

```
Scenario Summary:
-----------------------------------------------------------------------------------------
Scenario | Description                        | Raw Events | Routed | Coalesced | Status
-----------------------------------------------------------------------------------------
A        | B=1, T=1 single expert (W1,W3,W2)  | 3          | 1      | 1         | PASS
B        | B=2 same expert in decode step     | 6          | 2      | 1         | PASS
C        | B=2 diff experts in decode step    | 6          | 2      | 2         | PASS
D        | Diff decode steps, same token_idx  | 6          | 2      | 2         | PASS
E        | Repeated decode steps, same req    | 6          | 2      | 2         | PASS
-----------------------------------------------------------------------------------------
```

### 4.1 Scenario A: Single Token Base Case ($B=1, T=1, L=1, K=1$)
- **Events:** Expert $(0, 1)$ demanded with $W_1, W_3, W_2$ in decode step 0.
- **Verification:**
  - $D_{\text{tensor}} = 3$ (raw records)
  - $D_{\text{routed}} = 1$ (dedup across $W_1, W_3, W_2$ on `req=1, seq=1, tok=0, layer=0, exp=1`)
  - $D_{\text{coalesced}} = 1$ (single pair $(0, 1)$ at decode step 0)

### 4.2 Scenario B: Batch Coalescing ($B=2$, Identical Expert)
- **Events:** In decode step 1, Token A (`req=1, seq=1, tok=1`) and Token B (`req=2, seq=2, tok=1`) both route to Expert $(0, 2)$. Each produces 3 projection events ($W_1, W_3, W_2$).
- **Verification:**
  - $D_{\text{tensor}} = 6$
  - $D_{\text{routed}} = 2$ (Token A and Token B represent two distinct routing decisions)
  - $D_{\text{coalesced}} = 1$ (both tokens share $(0, 2)$ in decode step 1; physical memory requires only one load check)
- **Significance:** Proves that batch coalescing properly collapses inter-token redundancy within the batch.

### 4.3 Scenario C: Batch Diversity ($B=2$, Different Experts)
- **Events:** In decode step 2, Token A demands Expert $(0, 3)$ and Token B demands Expert $(0, 4)$.
- **Verification:**
  - $D_{\text{tensor}} = 6$
  - $D_{\text{routed}} = 2$
  - $D_{\text{coalesced}} = 2$ (two distinct experts needed in parallel)

### 4.4 Scenario D: Temporal Isolation Across Decode Steps
- **Events:** Sequence A demands Expert $(0, 5)$ at `tok=5` in decode step 3. Sequence B demands Expert $(0, 5)$ at `tok=5` in decode step 4.
- **Verification:**
  - $D_{\text{tensor}} = 6$
  - $D_{\text{routed}} = 2$
  - $D_{\text{coalesced}} = 2$ (even though `tok_id=5` is identical, they occur in decode steps 3 and 4, requiring independent temporal memory checks)
- **Significance:** Demonstrates why keying by `decode_step` fixes the vulnerability of keying by `token_id`.

### 4.5 Scenario E: Multi-Step Sequential Execution
- **Events:** Sequence 1 demands Expert $(0, 6)$ at `decode_step=5, tok=6`, and again demands Expert $(0, 6)$ at `decode_step=6, tok=7`.
- **Verification:**
  - $D_{\text{tensor}} = 6$
  - $D_{\text{routed}} = 2$
  - $D_{\text{coalesced}} = 2$ (temporal re-access across distinct inference iterations)

---

## 5. Telemetry Schema Verification

The telemetry exporter in `PrefetchMetricsCollector::finalize()` outputs these fields to both `prefetch_summary.json` and `prefetch_metrics.csv`:

```json
{
  "hbm_management": {
    "tensor_actual_demands": 1920,
    "routed_actual_demands": 640,
    "coalesced_actual_demands": 640,
    "coalesced_resident_hits": 189,
    "coalesced_fresh_demand_misses": 451,
    "coalesced_hbm_hit_rate": 0.2953125,
    "routed_hbm_hit_rate": 0.2953125,
    "tensor_hbm_hit_rate": 0.76510416
  }
}
```

This guarantees complete traceability from raw tensor invocations down to physical DDR5 demand transfers.

