# Phase 2E.3 — Authoritative GO / NO-GO Recommendation

## 1. Executive Decision: NO-GO

**Authoritative Decision: NO-GO for Tier 1 Full Comparative Campaign Execution**

Under the current forward-lookahead prefetching architecture ($H \in \{1, 2, 3\}$, Budget $B=4$, DDR5 bandwidth $22.578\text{ GB/s}$), asynchronous prefetching **fails all five mandatory operational criteria**:
- **Timely Resident Hits:** $0$ ($0.00\%$ timely hit rate across $7,200$ demand lookups).
- **Stall Reduction:** $0.00\text{ ms}$ ($0.00\%$ reduction; stall remains identical to demand baseline at $7,884.24\text{ ms}$).
- **Execution Speedup:** $0.00\text{ ms}$ ($0.00\%$ speedup; simulated execution time remains identical at $8,003.70\text{ ms}$).
- **Host Runtime:** $+11.1\%$ slowdown ($8.40\text{ s}$ vs $7.56\text{ s}$) due to queue tracking and predictor overhead.
- **DDR5 Bandwidth Overhead:** $+19.5\%$ to $+22.6\%$ pure bandwidth waste ($+33.2\text{ GB}$ to $+38.4\text{ GB}$ of speculative traffic that produces zero latency reduction).

Launching the full 24-run Tier 1 campaign under these parameters would consume substantial compute time and storage across all eight workloads (A1, A5, A6, A7, B1, B2, B3, C1) while producing unvarying $0.00\%$ speedup and uniform $0$ timely resident hits across all 16 prefetch configurations.

---

## 2. Evaluation Against the Five Mandatory Criteria

| Criterion | Evaluation Metric | Baseline (`B`) | ST-MoE (`FWD-S-H2`) | CSTP (`FWD-C-H2`) | Criterion Met? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Criterion 1: Timely Resident Hits** | Timely resident hits $> 0$ | $0$ | **$0$** | **$0$** | **FAILED** |
| **Criterion 2: Correct Prediction Hits** | Timely hits correspond to correct predictions | N/A | **$0$** | **$0$** | **FAILED** |
| **Criterion 3: Exposed Stall Reduction** | Exposed stall reduction ($ms$) | Baseline ($0$) | **$0.00\text{ ms}$ ($0.0\%$)** | **$0.00\text{ ms}$ ($0.0\%$)** | **FAILED** |
| **Criterion 4: Execution Time Reduction**| Wallclock execution time reduction ($ms$) | Baseline ($0$) | **$0.00\text{ ms}$ ($0.0\%$)** | **$0.00\text{ ms}$ ($0.0\%$)** | **FAILED** |
| **Criterion 5: Bandwidth Justification** | Speculative DDR5 traffic overhead | $0.0\text{ MB}$ ($0.0\%$) | **$+33,180.0\text{ MB}$ ($+19.5\%$)** | **$+38,388.0\text{ MB}$ ($+22.6\%$)** | **FAILED** |

### Detailed Criterion Breakdown

### Criterion 1: Do Timely Resident Hits Occur?
**Answer: NO.**
Across $7,200$ total expert demand lookups over 4 decoding tokens in Workload A1:
- Demand-only Baseline (`B`): $0$ timely hits.
- ST-MoE Forward Lookahead $H=2$ (`FWD-S-H2`): **$0$ timely hits**.
- CSTP Forward Lookahead $H=2$ (`FWD-C-H2`): **$0$ timely hits**.

Not a single prefetch transfer completed prior to the arrival of demand lookup for that expert at that stage.

### Criterion 2: Do Timely Hits Correspond to Correctly Predicted Experts?
**Answer: NO.**
Because timely resident hits are strictly $0$, zero timely hits correspond to correctly predicted experts. 
While predictor precision is $19.69\%$ for ST-MoE and $19.09\%$ for CSTP ($230$ and $223$ hits out of $1,168$ predictions respectively), every correctly predicted expert that was not already naturally resident in HBM was demanded *before* its physical transfer could complete. These arrived as:
- **Late In-Flight Attachments:** $59$ for `FWD-S-H2`, $68$ for `FWD-C-H2`.
- **Unused Speculative Transfers:** $29$ for `FWD-S-H2`, $29$ for `FWD-C-H2`.
- **Trapped In-Flight at End:** $307$ for `FWD-S-H2`, $360$ for `FWD-C-H2`.

### Criterion 3: Does Exposed Memory Stall Decrease?
**Answer: NO.**
Exposed memory stall is bitwise identical across all three configurations:
- Baseline (`B`): $7,884.24\text{ ms}$
- ST-MoE (`FWD-S-H2`): $7,884.24\text{ ms}$ ($\Delta = 0.00\text{ ms}$, $0.00\%$)
- CSTP (`FWD-C-H2`): $7,884.24\text{ ms}$ ($\Delta = 0.00\text{ ms}$, $0.00\%$)

Late in-flight attachments ($59$ in ST-MoE, $68$ in CSTP) yielded zero stall hiding. In `DDR5TransferEngine`, when demand attaches to a speculative transfer, the speculative transfer has been waiting in the queue or just started physical service ($3.901\text{ ms}$ remaining). Because demand execution stalls until the transfer is completely finished and the GPU computation time between stages is negligible, the pipeline remains completely stalled.

### Criterion 4: Does Total Simulated Execution Time Decrease?
**Answer: NO.**
Simulated execution time is identical across all three configurations:
- Baseline (`B`): $8,003.70\text{ ms}$
- ST-MoE (`FWD-S-H2`): $8,003.70\text{ ms}$ ($\Delta = 0.00\text{ ms}$, $0.00\%$)
- CSTP (`FWD-C-H2`): $8,003.70\text{ ms}$ ($\Delta = 0.00\text{ ms}$, $0.00\%$)

Total simulation time is the sum of exposed compute and memory stall. With zero compute-overlap and zero stall reduction, end-to-end execution latency remains invariant.

### Criterion 5: Is Additional Speculative DDR5 Traffic Justified?
**Answer: NO.**
Speculative prefetching injects severe channel overhead with zero performance return:
- `FWD-S-H2` submitted $395$ speculative transfers, transmitting $34.79\text{ GB}$ ($+19.5\%$) of extra DDR5 traffic.
- `FWD-C-H2` submitted $457$ speculative transfers, transmitting $40.25\text{ GB}$ ($+22.6\%$) of extra DDR5 traffic.

Because this additional bandwidth produced $0.00\text{ ms}$ stall reduction, the speculative traffic is $100\%$ overhead waste that consumes energy and congests the memory subsystem.

---

## 3. Root-Cause Analysis: The Architectural Dilemma

The failure of forward lookahead prefetching to produce timely hits is governed by three fundamental physical and architectural bottlenecks:

```
+-----------------------------------------------------------------------------------------+
|                               PHYSICAL TIMING MISMATCH                                  |
|                                                                                         |
| Available Compute Window (H=2):                                                         |
| [Stage L Compute: 0.183 ms] -> [Stage L+1 Compute: 0.183 ms] = TOTAL: 0.366 ms         |
|                                                                                         |
| Physical Transfer Latency (1 Expert = 84 MB @ 22.578 GB/s):                             |
| [==================================== 3.901 ms =====================================]   |
|                                                                                         |
| Ratio: 3.901 ms / 0.366 ms = 10.66x Deficit!                                           |
| (An expert takes >10x longer to transfer than the lookahead window provides)            |
+-----------------------------------------------------------------------------------------+
```

### 1. Severe Compute-to-Transfer Latency Mismatch
- **Physical expert transfer latency:** An $84.0\text{ MB}$ expert transferred over a single DDR5 channel at $22.578\text{ GB/s}$ requires:
  $$t_{\text{transfer}} = \frac{84.0 \times 10^6 \text{ bytes}}{22.578 \times 10^9 \text{ bytes/s}} = 3.901\text{ ms}$$
- **Available compute window:** In DeepSeek-V3 Mixtral-style, MoE computation per stage is $\approx 0.183\text{ ms}$.
- Over a forward lookahead horizon of $H=2$, the compute window elapsed before Stage $L+2$ is demanded is:
  $$t_{\text{window}} = 2 \times 0.183\text{ ms} = 0.366\text{ ms}$$
- **Deficit:** $t_{\text{window}} \ll t_{\text{transfer}}$. Transferring a single expert requires more than $10\times$ the time provided by an $H=2$ horizon! Even with $H=3$, the compute window is only $0.549\text{ ms}$ ($14.1\%$ of the transfer time).
- For a single prefetch to complete purely during computation without stalling, the lookahead horizon would need to be:
  $$H_{\text{min}} = \left\lceil \frac{3.901\text{ ms}}{0.183\text{ ms}} \right\rceil = 22 \text{ stages!}$$

### 2. Speculative Queue Congestion (FIFO Head-of-Line Blocking)
- At budget $B=4$, four candidates are dispatched per stage ($4 \times 3.901\text{ ms} = 15.60\text{ ms}$ of transfer time requested every $0.183\text{ ms}$).
- As a result, the `DDR5TransferEngine` FIFO queue explodes. The average queue waiting time for speculative candidates reaches:
  - `FWD-S-H2`: **$238.64\text{ ms}$**
  - `FWD-C-H2`: **$222.33\text{ ms}$**
- Candidates sit waiting in the queue for hundreds of milliseconds. By the time a candidate reaches the physical DDR5 bus, its target stage executed hundreds of milliseconds in the past.

### 3. Lack of Speculative Transfer Expiration / Cancellation
- When stage $L$ finishes and retires, speculative transfers queued for stage $L$ are never cancelled or pruned from the DDR5 queue.
- These stale transfers continue to occupy DDR5 bus cycles long after their target stage has completed, actively delaying subsequent speculative transfers and demand requests.

### 4. Inter-Token Simulation Clock Reset
- In `test_deepseekv3_mixtral_style.cpp:464`, `device->status.device_time` is reset to $0$ at the start of each decoding step.
- Speculative transfers dispatched in the final stages of token $k$ cannot cleanly bridge into token $k+1$, truncating any potential inter-token prefetch pipeline.

---

## 4. Prerequisites for a Future "GO"

To achieve timely resident hits and genuine stall reduction in future research iterations, the prefetching subsystem would require the following architectural enhancements:

1. **Stale Transfer Cancellation on Stage Expiration:**
   - When a stage finishes execution, any queued speculative transfers targeted for that stage that have not yet started physical DDR5 transfer must be immediately purged from `DDR5TransferEngine`.
2. **Aggressive Budget Throttling / Rate Limiting:**
   - Instead of injecting $B=4$ candidates per stage, limit injection to at most 1 high-confidence candidate every $N$ stages (matching the bandwidth capacity: $1$ transfer per $\approx 4\text{ ms}$ of execution).
3. **Horizon Expansion or Interconnect Upgrade:**
   - Either extend the lookahead horizon to $H \ge 22$ (inter-stage lookahead or token-level lookahead), or evaluate hardware configurations with significantly higher off-chip bandwidth (e.g., PCIe Gen5 $64\text{ GB/s}$ or CXL $128\text{ GB/s}$, which drops $t_{\text{transfer}}$ to $0.65\text{ ms}$).
4. **Inter-Token Continuous Clocking:**
   - Preserve `device_time` across token boundaries so prefetch transfers initiated at the end of token $k$ can overlap with prompt/KV operations and arrive timely for early stages of token $k+1$.
5. **Partial-Stall Overlap Instrumentation:**
   - If late in-flight attachments are intended to hide a fraction of memory stall (e.g., hiding $1.5\text{ ms}$ of a $3.9\text{ ms}$ transfer), the simulator's stall calculation logic in `linear_impl.cpp` must be audited and refined to credit physical transfer elapsed time before demand arrival.

---

## 5. Formal Conclusion

Proceeding with the full Tier 1 comparative campaign at this stage is **not justified**. The Phase 2E.3 diagnostic conclusively proves that forward-lookahead prefetching with current horizons and budgets produces **zero timely resident hits, zero stall reduction, zero speedup, and $+19.5\%$ to $+22.6\%$ pure bandwidth waste**.

The authoritative recommendation is **NO-GO**. Future efforts should focus on solving the architectural compute-to-transfer latency mismatch and implementing speculative queue cancellation before launching large-scale comparative campaigns.

