# Demand Accounting Corrections: Routed vs. Coalesced vs. Tensor Executions

## Point 4 — Phase 1.6 Consistency Audit & Design Corrections

---

## 1. Executive Summary & Problem Statement

In the previous Phase 1.5 workload documentation, total demand events were calculated as:
$$D_{\text{routed}} = B \times T \times L \times K$$
While this formula correctly counts the total number of **individual token-to-expert routing selections**, it does **not** represent the physical expert loading demands experienced by the GPU HBM cache subsystem.

In batched inference ($B > 1$), multiple concurrent sequences in the same serving batch frequently route to the same expert in the same MoE stage. In `LLMSimulator` ([`src/module/expert.cpp:208-215`](file:///home/mohman/ca_project/LLMSimulator/src/module/expert.cpp#L208-L215)), the MoE module executes that expert **once** for the combined batch using batched matrix multiplication ($M = \text{num\_token\_in\_expert}[e]$).

Therefore, we must rigorously formalize and distinguish three distinct operational levels:
1. **Routed Token-Expert Selections ($D_{\text{routed}}$):** Algorithmic routing decisions made by Top-$K$ gating.
2. **Unique Coalesced Expert Activations ($D_{\text{coalesced}}$):** Distinct $(l, e)$ expert pairs demanded by the batch in a given stage, representing physical weight transfers ($88.08 \text{ MB}$).
3. **Tensor Projection Executions ($D_{\text{tensor}}$):** Individual projection GEMMs ($W_1, W_3, W_2$) executed on GPU hardware ($D_{\text{tensor}} = 3 \times D_{\text{coalesced}}$).

---

## 2. Mathematical Formalization

### 2.1 Definitions

#### 1. Routed Demands ($D_{\text{routed}}$)
$$D_{\text{routed}} = \sum_{t=0}^{T-1} \sum_{l=0}^{L-1} \sum_{s=0}^{B-1} K = B \times T \times L \times K$$
Counts every token-to-expert routing assignment across sequences, tokens, stages, and Top-$K$ ranks.

#### 2. Unique Coalesced Expert Activations ($D_{\text{coalesced}}$)
$$D_{\text{coalesced}} = \sum_{t=0}^{T-1} \sum_{l=0}^{L-1} \left| \bigcup_{s=0}^{B-1} \mathcal{E}(s, t, l) \right|$$
where $\mathcal{E}(s, t, l)$ is the set of $K$ experts chosen by sequence $s$ at token $t$ in MoE stage $l$.
- This is the **true physical demand denominator** for HBM cache hit rate, DDR5 demand transfers, and memory traffic.
- If all sequences in a batch route to distinct experts, $D_{\text{coalesced}} = D_{\text{routed}}$.
- When sequences share experts, $D_{\text{coalesced}} < D_{\text{routed}}$.

#### 3. Tensor Projection Executions ($D_{\text{tensor}}$)
$$D_{\text{tensor}} = 3 \times D_{\text{coalesced}}$$
Because DeepSeek-V3 employs FeedForward3Way ($W_1$ gate, $W_3$ up, $W_2$ down), each coalesced expert activation executes three GEMMs on the GPU.

---

## 3. Ground-Truth Trace Analysis & Workload Counts

Using the authentic MoE-Anywhere dataset ([`LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet`](file:///home/mohman/ca_project/LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet)), we extracted the exact routing selections and calculated the true values of $D_{\text{routed}}$, $D_{\text{coalesced}}$, and $D_{\text{tensor}}$:

| Workload ID | Batch ($B$) | Tokens ($T$) | Stages ($L$) | Top-$K$ ($K$) | $D_{\text{routed}}$ | $D_{\text{coalesced}}$ | $D_{\text{tensor}}$ | Expert Sharing Ratio | Avg Unique Experts / Stage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P4-SMOKE** | 1 | 4 | 5 | 8 | **160** | **160** | **480** | **0.00%** ($1.000\times$) | 8.00 / 8 |
| **P4-W1** | 1 | 16 | 40 | 8 | **5,120** | **5,120** | **15,360** | **0.00%** ($1.000\times$) | 8.00 / 8 |
| **P4-W2** | 4 | 16 | 40 | 8 | **20,480** | **8,275** | **24,825** | **59.59%** ($2.475\times$) | 12.93 / 32 |
| **P4-W3** | 14 | 16 | 40 | 8 | **71,680** | **13,164** | **39,492** | **81.64%** ($5.445\times$) | 20.57 / 112 |
| **P4-W4** | 32 | 16 | 40 | 8 | **163,840** | **25,431** | **76,293** | **84.48%** ($6.443\times$) | 39.74 / 256 |

### Architectural Insights from Trace Data
1. **Single-Sequence Baseline ($B=1$):** In P4-SMOKE and P4-W1, because there is only one sequence, no cross-sequence expert sharing can occur. Therefore, $D_{\text{coalesced}} = D_{\text{routed}}$.
2. **Massive Sharing Under Batching ($B \ge 4$):**
   - At $B=4$ (P4-W2), 59.59% of routed selections are shared. On average, only 12.93 unique experts are needed per stage out of 32 allocated slots.
   - At $B=14$ (P4-W3), 81.64% of selections are shared. Only 20.57 unique experts are needed per stage out of 112 slots.
   - At $B=32$ (P4-W4), 84.48% of selections are shared. 163,840 token selections collapse into just 25,431 physical expert activations.

---

## 4. Denominator Corrections for Point 4 Metrics

### 4.1 HBM Hit Rate Denominators
In all subsequent Point 4 analyses, two distinct hit rates will be reported:

1. **Physical Coalesced HBM Hit Rate ($H_{\text{coalesced}}$):**
   $$H_{\text{coalesced}} = \frac{N_{\text{coalesced\_resident\_hits}} + N_{\text{coalesced\_timely\_hits}}}{D_{\text{coalesced}}}$$
   - **Authoritative Denominator:** $D_{\text{coalesced}}$.
   - **Significance:** Measures the fraction of physical expert weight loads ($88.08 \text{ MB}$) avoided by HBM residency.

2. **Algorithmic Routing Hit Rate ($H_{\text{routed}}$):**
   $$H_{\text{routed}} = \frac{\sum_{s, t, l, k} \mathbb{I}(\text{resident})}{D_{\text{routed}}}$$
   - **Denominator:** $D_{\text{routed}} = B \times T \times L \times K$.
   - **Significance:** Measures the fraction of token-level routing requests that encounter resident experts.

---

## 5. Source-Code Instrumentation Requirement

In [`src/prefetch/prefetch_metrics_collector.h`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.h) and [`src/prefetch/prefetch_metrics_collector.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/prefetch_metrics_collector.cpp):
- Current code tracks:
  - `total_actual_demands`: Incremented on every projection GEMM ($D_{\text{tensor}}$).
  - `unique_actual_demands`: Deduplicated by `(request_id, seq_id, token_id, layer_id, expert_id)` ($D_{\text{routed}}$).
- **Required New Instrumentation for Phase 2:**
  Add a stage-coalesced demand map:
  ```cpp
  std::map<std::tuple<int, int, int>, CollectorDemandRecord> stage_coalesced_demands;
  ```
  keyed on `(token_id, layer_id, expert_id)`.
  This guarantees that at simulation finalization, $D_{\text{coalesced}} = \text{stage\_coalesced\_demands.size()}$ is measured directly from runtime events, matching the analytical trace calculations with 100% precision.

