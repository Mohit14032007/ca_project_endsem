# Final Experiment Matrix & Campaign Specification

## Point 4: HBM Management Beyond Standard LRU

---

## 1. Executive Summary

This document specifies the complete, finalized 120-run experimental campaign for Point 4. It defines the 6 evaluation configurations, 5 HBM capacity tiers, 4 representative workloads, naming conventions, and controlled variable isolations required to rigorously evaluate **FA-LRU vs Standard LRU** across no-prefetch baseline, ST-MoE, and CSTP prefetchers.

---

## 2. Experimental Configurations (6 Strategies)

To prevent ambiguity between serving batch size and prefetch candidate budget, we adopt strict mathematical notation:
- **$B \in \{1, 4, 14, 32\}$:** Serving Batch Size (concurrent inference sequences).
- **$K = 4$:** Prefetch Candidate Budget (maximum speculative candidates dispatched per stage).
- **$H = 1$:** Prediction Lookahead Horizon (number of MoE stages predicted ahead).

| Config ID | Strategy Name | Prefetch Engine | Eviction Policy | Parameters | Primary Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C1** | `baseline_lru` | None (`mode: none`) | Standard `LRU` | None | Pure demand-only baseline reference under standard LRU |
| **C2** | `baseline_falru` | None (`mode: none`) | `FA_LRU` | $W=16$, Epoch Halving | Isolates the pure effect of frequency-aware eviction on demand hits and stalls |
| **C3** | `stmoe_lru` | ST-MoE | Standard `LRU` | $H=1, K=4$, adapt=true | Standard prefetching baseline under LRU |
| **C4** | `stmoe_falru` | ST-MoE | `FA_LRU` | $H=1, K=4, W=16$ | Evaluates synergy between ST-MoE prefetching and frequency-aware cache retention |
| **C5** | `cstp_lru` | CSTP | Standard `LRU` | $H=1, K=4$, adapt=true | Confidence-filtered prefetching baseline under LRU |
| **C6** | `cstp_falru` | CSTP | `FA_LRU` | $H=1, K=4, W=16$ | Evaluates synergy between CSTP confidence filtering and frequency-aware eviction |

---

## 3. HBM Capacity Sweep (5 Tiers)

To evaluate cache behavior across the full spectrum of memory pressure, five capacity tiers are defined:

| Capacity Tier | Bytes Setting | Equivalent Capacity | Max Resident Experts | % of DeepSeek-V3 Weights | Cache Pressure Regime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cap-10G** | `10000000000` | 10.00 GB | 113 experts | 1.10% | **Extreme Pressure:** Evictions occur continuously on every token step. Strong test for heavy-hitter survival. |
| **Cap-20G** | `20000000000` | 20.00 GB | 227 experts | 2.22% | **High Pressure:** Heavy thrashing regime under standard LRU; significant headroom for FA-LRU gains. |
| **Cap-35G** | `37580963840` | 35.00 GB | 426 experts | 4.16% | **Standard Reference:** Directly matches Point 3 large-workload campaign setting. |
| **Cap-50G** | `50000000000` | 50.00 GB | 567 experts | 5.54% | **Moderate Pressure:** Working set retention expands; tests whether frequency awareness remains advantageous. |
| **Cap-80G** | `80000000000` | 80.00 GB | 908 experts | 8.87% | **Low Pressure / Unconstrained:** Physical A100/H100 80GB HBM capacity baseline. |

*Note on Expert Sizing:* DeepSeek-V3 expert size $= 88,080,384 \text{ bytes}$. $\lfloor \text{Capacity} / 88,080,384 \rfloor = \text{Max Resident Experts}$.

---

## 4. Workload Definitions (4 Point 4 Workloads)

To avoid confusion with Point 3 workloads ($W_1\dots W_7$), Point 4 workloads use the prefix **P4-W**:

| Workload ID | Serving Batch ($B$) | Tokens ($T$) | Stages ($L$) | Independent Prompts | Total Stage Demands | Characterization |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P4-W1** | 1 | 16 | 40 | 1 (Prompt 60, Test) | 5,120 demands | **Interactive Latency:** Single-sequence autoregressive decode. Low concurrency, tests temporal heavy-hitter retention. |
| **P4-W2** | 4 | 16 | 40 | 4 (Prompts 60–63, Test) | 20,480 demands | **Small Concurrent Batch:** Moderate expert overlap across sequences. |
| **P4-W3** | 14 | 16 | 40 | 14 (Prompts 60–73, Complete Test) | 71,680 demands | **Full Held-Out Batch:** All 14 independent test prompts running concurrently. High routing diversity across 4 distinct tasks. |
| **P4-W4** | 32 | 16 | 40 | 32 (14 Test + 8 Val + 10 Train) | 163,840 demands | **High-Throughput Serving:** Zero prompt duplication. Intense eviction contention across all 40 layers. |

---

## 5. Matrix Multiplicity & Run Justification

$$\text{Total Runs} = 6 \text{ Configurations} \times 5 \text{ Capacities} \times 4 \text{ Workloads} = 120 \text{ Runs}$$

### Are all 120 runs valid and necessary?
**Yes.** Every quadrant of this matrix serves a specific, non-redundant scientific purpose:
1. **Capacity Axis (5 points):** Provides the definitive "Hit Rate vs HBM Capacity" curve, proving whether FA-LRU reduces the physical HBM footprint required to achieve a target hit rate (e.g., whether 35 GB FA-LRU matches or beats 50 GB LRU).
2. **Workload Axis (4 points):** Tests batch scalability from $B=1$ to $B=32$, verifying whether expert sharing in larger batches magnifies or diminishes frequency benefits.
3. **Strategy Axis (6 points):**
   - Paired comparison $(C_1, C_2)$ proves the standalone efficacy of FA-LRU.
   - Paired comparisons $(C_3, C_4)$ and $(C_5, C_6)$ determine whether FA-LRU resolves the "unconsumed prefetch pollution" problem observed in Point 3.

---

## 6. Strict Paired-Comparison Invariants

Within every matched pair (e.g., $C_1$ vs $C_2$, $C_3$ vs $C_4$, or $C_5$ vs $C_6$ at identical workload and capacity), the following variables **must remain 100% identical**:
1. **Trace Input:** Exact same routing trace file (`P4-W1.csv`, `P4-W2.csv`, `P4-W3.csv`, `P4-W4.csv`).
2. **Seed & Sequence Order:** Identical prompt ordering and sequence metadata.
3. **Hardware Model:** Identical H100 GPU compute peak (989.4 TFLOPS) and DDR5 bandwidth (64.0 GB/s).
4. **Interconnect & Timing:** `CLOCK_CONTINUITY=1`, `USE_EVENT_DRIVEN_DRAM=true`, `COMPUTE_MULTIPLIER=1.0`.
5. **Predictor Budget & Horizon:** Identical $H=1, K=4$ where prefetching is active.

The **only** permitted difference between paired runs is the eviction policy selection: `eviction_policy: lru` versus `eviction_policy: fa_lru`.

---

## 7. Run Identification & File Naming Schema

Every run in the campaign is uniquely and deterministically identified by the schema:
$$\text{Run ID} = \langle\text{workload}\rangle\_\langle\text{strategy}\rangle\_\langle\text{capacity}\rangle$$

Examples:
- `p4w1_baseline_lru_cap10g`
- `p4w1_baseline_falru_cap10g`
- `p4w3_cstp_lru_cap35g`
- `p4w3_cstp_falru_cap35g`
- `p4w4_stmoe_falru_cap80g`

Output artifacts for each run will be stored in isolated directories:
`mohman/point_4/experiments/raw/<run_id>/traces/`

