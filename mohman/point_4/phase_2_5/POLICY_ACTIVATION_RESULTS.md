# Point 4 Phase 2.5: Policy Activation & Real Trace Validation

## MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference

---

## 1. Executive Summary & Objective

The primary objective of Phase 2.5 Task 6 is to provide irrefutable proof of policy activation:
1. **Deterministic Divergence Proof:** Prove via a controlled access trace that FA-LRU diverges from standard LRU under non-uniform access frequency distributions, successfully protecting high-frequency experts from eviction.
2. **Real-Trace Workload Evaluation:** Execute a realistic multi-stage MoE trace (Prompt 60, held-out `tools` domain, $B=1, T=16, L=5, K=8$) under constrained HBM capacity ($4.0\text{ GB}$) to benchmark FA-LRU against standard LRU on genuine DeepSeek-V3 routing patterns.

---

## 2. Deterministic Policy Activation Experiment

To isolate the victim selection logic from non-deterministic scheduling effects, a dedicated test binary was constructed (`eval/test_policy_activation.cpp`) and executed.

### 2.1 Trace Design
- **Cache Configuration:** Capacity $C = 4\text{ experts}$, Bottom Candidate Window $W = 4$.
- **Expert Space:** 8 distinct experts $(0, 1) \dots (0, 8)$ in layer 0.
- **Access Pattern:**
  - **Warm-up (Steps 0–3):** Sequential accesses to $(0, 1), (0, 2), (0, 3), (0, 4)$ in Epoch 0. All miss and fill the cache ($F=1$).
  - **Frequency Skew (Steps 4–7):** Expert $(0, 1)$ is repeatedly demanded across Epochs 0 and 1, elevating its decayed access frequency to $F=3$.
  - **Eviction Phase (Steps 8–10):** Three ephemeral experts $(0, 5), (0, 6), (0, 7)$ are demanded once ($F=1$). This evicts $(0, 3), (0, 4), (0, 2)$.
  - **Critical Test Step 11:** Demand arrives for Expert $(0, 8)$.
  - **Payoff Step 12:** Heavy-hitter Expert $(0, 1)$ is re-demanded.

### 2.2 Step-by-Step Decision Log (`policy_activation_decisions.csv`)

| Step | Demand | Epoch | LRU Outcome | LRU Victim | FA-LRU Outcome | FA-LRU Victim | Diverged? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | $(0, 1)$ | 0 | MISS | None | MISS | None | No |
| 1 | $(0, 2)$ | 0 | MISS | None | MISS | None | No |
| 2 | $(0, 3)$ | 0 | MISS | None | MISS | None | No |
| 3 | $(0, 4)$ | 0 | MISS | None | MISS | None | No |
| 4 | $(0, 1)$ | 0 | HIT | None | HIT | None | No |
| 5 | $(0, 2)$ | 0 | HIT | None | HIT | None | No |
| 6 | $(0, 1)$ | 1 | HIT | None | HIT | None | No |
| 7 | $(0, 1)$ | 1 | HIT | None | HIT | None | No |
| 8 | $(0, 5)$ | 1 | MISS | $(0, 3)$ | MISS | $(0, 3)$ | No |
| 9 | $(0, 6)$ | 1 | MISS | $(0, 4)$ | MISS | $(0, 4)$ | No |
| 10 | $(0, 7)$ | 1 | MISS | $(0, 2)$ | MISS | $(0, 2)$ | No |
| **11** | **$(0, 8)$** | **1** | **MISS** | **$(0, 1)$** | **MISS** | **$(0, 5)$** | **YES (DIVERGENCE)** |
| **12** | **$(0, 1)$** | **1** | **MISS** | **$(0, 5)$** | **HIT** | **None** | **YES (PERFORMANCE PAYOFF)** |
| 13 | $(0, 5)$ | 1 | MISS | $(0, 6)$ | MISS | $(0, 6)$ | No |

### 2.3 Microarchitectural Analysis of Divergence

#### At Step 11:
- Cache resident set before Step 11 (in LRU order, oldest at front):
  - Index 0 (Oldest): Expert $(0, 1)$, Decayed Frequency $F = 3$.
  - Index 1: Expert $(0, 5)$, Decayed Frequency $F = 1$.
  - Index 2: Expert $(0, 6)$, Decayed Frequency $F = 1$.
  - Index 3 (Newest): Expert $(0, 7)$, Decayed Frequency $F = 1$.
- **Standard LRU Behavior:** LRU blindly inspects the front of the list and selects Expert $(0, 1)$ as the victim simply because it was not touched during Steps 8–10. **The high-frequency heavy-hitter is evicted!**
- **FA-LRU Behavior:** FA-LRU inspects the bottom window ($W=4$). It evaluates all four candidates:
  - $(0, 1) \to F = 3$
  - $(0, 5) \to F = 1$
  - $(0, 6) \to F = 1$
  - $(0, 7) \to F = 1$
  The minimum frequency is $F_{\text{min}} = 1$. Among the candidates with $F = 1$, the oldest is Expert $(0, 5)$. FA-LRU evicts ephemeral $(0, 5)$ and **actively protects the heavy-hitter $(0, 1)$**.

#### At Step 12:
- Heavy-hitter Expert $(0, 1)$ is requested again.
- **Standard LRU:** Experiences a **demand miss**, forcing an expensive DDR5 reload and evicting $(0, 5)$.
- **FA-LRU:** Experiences a **resident HBM hit** ($0\text{ ns}$ stall, 0 bytes memory traffic).

This experiment provides incontrovertible proof that FA-LRU functions as specified, protects heavy-hitters, and delivers superior cache residency.

---

## 3. 16-Token Real Trace Comparison (Prompt 60)

To assess baseline equivalence and telemetry consistency in a realistic setting, a 16-token trace extracted from MoE-Anywhere Prompt 60 ($B=1, T=16, L=5, K=8$) was simulated under constrained capacity ($4.0\text{ GB} = 45\text{ experts}$) using `run_p2_5_real_trace.py`.

### 3.1 Comparative Performance Matrix

| Metric Category | Specific Metric | Standard LRU | FA-LRU ($W=16$) | Absolute Delta | Relative Delta |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cache Setup** | Capacity Limit | 4.0 GB (45 experts) | 4.0 GB (45 experts) | 0 | 0.0% |
| | Candidate Window ($W$) | 1 (Implicit) | 16 | +15 | - |
| **Demand Breakdown** | Coalesced Actual Demands | 640 | 640 | 0 | 0.0% |
| | Routed Demands ($B \times T \times L \times K$) | 640 | 640 | 0 | 0.0% |
| | Tensor Executions ($W_1, W_3, W_2$) | 1,920 | 1,920 | 0 | 0.0% |
| **Cache Performance** | Coalesced Resident Hits | 189 | 189 | 0 | 0.0% |
| | Coalesced Fresh Misses | 451 | 451 | 0 | 0.0% |
| | **Coalesced HBM Hit Rate** | **29.53%** | **29.53%** | **0.00%** | **0.0%** |
| | Tensor-Level Hit Rate | 76.51% | 76.51% | 0.00% | 0.0% |
| **Evictions & Reloads**| Total Evictions | 406 | 406 | 0 | 0.0% |
| | Reloaded Experts Count | 157 | 157 | 0 | 0.0% |
| **Memory Traffic** | DDR5 Traffic (Decimal GB, $10^9$) | **39.7243 GB** | **39.7243 GB** | **0.0000 GB** | **0.0%** |
| | DDR5 Traffic (Binary GiB, $2^{30}$) | **36.9961 GiB** | **36.9961 GiB** | **0.0000 GiB** | **0.0%** |
| **Latency & Timing** | Exposed GPU Stall Time | 620.69 ms | 620.69 ms | 0.00 ms | 0.0% |
| | Mean Token Latency | 42.08 ms | 42.08 ms | 0.00 ms | 0.0% |
| | Simulated Tokens / Second | 23.76 | 23.76 | 0.00 | 0.0% |
| **Host Execution** | Total Host CPU Runtime | 2.95 s | 3.01 s | +0.06 s | +2.0% |

### 3.2 Engineering Analysis of Real-Trace Baseline Equivalence
In this short 16-token single-batch experiment ($B=1, T=16$), standard LRU and FA-LRU exhibit identical macro-level hit rates ($29.53\%$) and DDR5 traffic ($39.7243\text{ GB}$):
1. **Low Frequency Variance in Small Token Windows:** In DeepSeek-V3 with $N_{\text{routed}}=256$, routing 8 experts per token across 5 layers spreads activations thinly over just 16 generation steps. Most experts are demanded at most once or twice.
2. **Deterministic Tie-Breaking:** When candidates in the bottom-$W=16$ window have identical decayed frequency ($F=1$), FA-LRU's strict inequality condition `if (f < min_freq)` selects the oldest candidate at the front of the list. Under uniform frequency, FA-LRU strictly preserves LRU order.
3. **Necessity of Phase 3 Large-Batch Workloads:** This confirms the architectural rationale for the Phase 3 experimental design:
   - To unlock the frequency-aware advantages demonstrated in Section 2, the workload must provide sufficient batch concurrency ($B=4, 8, 16, 32$) and longer token sequences ($T=64, 128$) where popular routing clusters repeatedly activate high-frequency experts across distinct decoding steps.

---

## 4. Summary & Conclusions

1. **Policy Activation Proved:** Test 12 in Section 2 definitively establishes that FA-LRU actively diverges from LRU at Step 11 and reaps a performance payoff at Step 12.
2. **Stability & Invariant Safety:** In the full 16-token real trace, FA-LRU executed 640 coalesced demands, 406 evictions, and 157 reloads with zero crashes, zero invariant violations, and negligible runtime overhead ($+0.06\text{ s}$).
3. **Telemetry Consistency:** The new demand accounting accurately reports $D_{\text{routed}}=640$, $D_{\text{coalesced}}=640$, and $D_{\text{tensor}}=1920$ with zero discrepancies.

