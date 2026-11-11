# Phase 2E.3 — Controlled Multi-Token Prefetch Diagnostic Results

## 1. Executive Summary

This report documents the results of the controlled 4-token multi-token experiment designed to evaluate forward-lookahead prefetching once HBM residency develops naturally.

- **Authoritative Baseline Anchor:** `d5bf61ff0c066c06f72173189f9924329f613a88`
- **Workload:** Workload A1 (`ms_task12_t128_l75.csv`, DeepSeek-V3 Mixtral-style, 4 tokens, input_len=128, output_len=128, batch_size=1)
- **Matrix Evaluated:**
  1. `B`: Demand-only Baseline (LRU)
  2. `FWD-S-H2`: ST-MoE Predictor, Forward Lookahead $H=2$, Budget $B=4$, PABI Quarantine + Belády Pinning
  3. `FWD-C-H2`: CSTP Predictor, Forward Lookahead $H=2$, Budget $B=4$, PABI Quarantine + Belády Pinning
- **Output Target:** `results/final_comparative_campaign/phase2e3_warm_diagnostic/`
- **Summary CSV:** `results/final_comparative_campaign/phase2e3_warm_diagnostic/phase2e3_summary.csv` and synced to `docs/final_campaign/PHASE2E3_SUMMARY.csv`.

---

## 2. Comprehensive Multi-Token Metric Matrix

The table below reports all required metrics separately, strictly maintaining conservation laws and distinguishing natural HBM hits from timely prefetch hits:

| Metric Category | Metric Name | Baseline (`B`) | ST-MoE Forward (`FWD-S-H2`) | CSTP Forward (`FWD-C-H2`) |
| :--- | :--- | :---: | :---: | :---: |
| **Execution & Latency** | Simulated Execution Time ($ms$) | $8,003.70$ | $8,003.70$ | $8,003.70$ |
| | Exposed Memory Stall ($ms$) | $7,884.24$ | $7,884.24$ | $7,884.24$ |
| | MoE DRAM Time ($ms$) | $5,297.89$ | $5,297.89$ | $5,297.89$ |
| | MoE Compute Time ($ms$) | $54.95$ | $54.95$ | $54.95$ |
| | Host Wallclock Runtime ($s$) | $7.56$ | $8.40$ | $7.85$ |
| **Demand Lookups & Cache** | Total Demand Lookups | $7,200$ | $7,200$ | $7,200$ |
| | Natural Demand HBM Hits | $5,179$ | $5,179$ | $5,179$ |
| | HBM Hit Rate ($\%$) | $71.93\%$ | $71.93\%$ | $71.93\%$ |
| | Total Demand Misses | $2,021$ | $2,021$ | $2,021$ |
| **Prefetch Effectiveness** | Timely Resident Hits | **$0$** | **$0$** | **$0$** |
| | Late In-Flight Attachments | $0$ | **$59$** | **$68$** |
| | Unprefetched Demand Misses | $2,021$ | $1,962$ | $1,953$ |
| | Prefetches Completed Before Demand | $0$ | $29$ | $29$ |
| | Unused Speculative Prefetches | $0$ | $29$ | $29$ |
| | Evicted Speculative Prefetches | $0$ | $0$ | $0$ |
| | Still In-Flight at End | $0$ | $307$ | $360$ |
| | Cancelled Prefetches | $0$ | $0$ | $0$ |
| | Total Speculative Submitted | $0$ | $395$ | $457$ |
| **DDR5 Bandwidth & Traffic** | Physical Demand Transfers | $2,021$ | $2,021$ | $2,021$ |
| | DDR5 Demand Bytes | $178,010,456,064$ | $178,010,456,064$ | $178,010,456,064$ |
| | DDR5 Demand Traffic ($MB$) | $169,764.0$ | $169,764.0$ | $169,764.0$ |
| | DDR5 Speculative Bytes | $0$ | $34,791,751,680$ | $40,252,735,488$ |
| | DDR5 Speculative Traffic ($MB$) | $0.0$ | $33,180.0$ | $38,388.0$ |
| | Total DDR5 Traffic ($MB$) | $169,764.0$ | **$202,944.0$** ($+19.5\%$) | **$208,152.0$** ($+22.6\%$) |
| **Timing & Service Queuing** | Average Queue Waiting Time ($ms$) | $0.000$ | $238.639$ | $222.334$ |
| | Avg Remaining Service Time at Demand ($ms$) | $0.000$ | $3.901$ | $3.901$ |
| | Avg Demand Arrival to Ready ($ms$) | $3.901$ | $7.263$ | $6.818$ |
| **Predictor Accuracy** | Total Predictions Evaluated | $0$ | $1,168$ | $1,168$ |
| | Prediction Hits | $0$ | $230$ | $223$ |
| | Prediction Precision ($\%$) | $0.00\%$ | **$19.69\%$** | **$19.09\%$** |
| | Prediction Recall ($\%$) | $0.00\%$ | **$9.85\%$** | **$9.55\%$** |
| | Prediction F1 Score ($\%$) | $0.00\%$ | **$13.13\%$** | **$12.73\%$** |
| **Conservation Audits** | Demand-Outcome Conservation | **VERIFIED** | **VERIFIED** | **VERIFIED** |
| | Transfer-Lifecycle Conservation | **VERIFIED** | **VERIFIED** | **VERIFIED** |
| | Stage-ID Alignment Check | **PASSED** | **PASSED** | **PASSED** |

---

## 3. Strict Conservation Verification

### 1. Demand-Outcome Conservation
Every demand lookup in `expert_load.csv` ($7,200$ total expert lookups across 4 tokens) is strictly conserved into mutually exclusive outcomes:
$$\text{Demand Lookups} = \text{Natural HBM Hits} + \text{Timely Prefetch Hits} + \text{Late In-Flight Attachments} + \text{Unprefetched Misses}$$

- **Baseline (`B`):** $5,179 + 0 + 0 + 2,021 = 7,200$ (**VERIFIED**).
- **`FWD-S-H2`:** $5,179 + 0 + 59 + 1,962 = 7,200$ (**VERIFIED**).
- **`FWD-C-H2`:** $5,179 + 0 + 68 + 1,953 = 7,200$ (**VERIFIED**).

### 2. Transfer-Lifecycle Conservation
Every speculative transfer submitted to `Device` is strictly accounted for:
$$\text{Speculative Submitted} = \text{Demanded} + \text{Unused} + \text{Evicted} + \text{Still In-Flight at End} + \text{Cancelled}$$

- **`FWD-S-H2`:** $59 + 29 + 0 + 307 + 0 = 395$ (**VERIFIED**).
- **`FWD-C-H2`:** $68 + 29 + 0 + 360 + 0 = 457$ (**VERIFIED**).

---

## 4. In-Depth Analysis of Multi-Token Behavior

### 1. Natural Cache Residency Development
As established in the Warm-Cache Audit, HBM cache residency develops naturally across tokens:
- **Token 0:** Completely cold cache ($0$ initial resident experts). $576$ unique experts loaded, reaching $47.25\text{ GB}$.
- **Token 1:** Natural warm hits occur. HBM capacity reaches its $80\text{ GB}$ ceiling ($908$ resident experts, $74.48\text{ GB}$).
- **Tokens 2 & 3:** Steady-state LRU eviction and reuse maintain a $71.93\%$ natural hit rate ($5,179$ hits out of $7,200$ lookups).

### 2. Why Zero Timely Resident Hits Occurred Even in Warm Tokens
Despite natural HBM residency reaching $71.93\%$, **timely resident prefetch hits remained zero** across all configurations:

1. **Persistent Demand Dominance:**
   Even in warm tokens, the remaining $28.07\%$ demand misses ($2,021$ total across 4 tokens) dominate the DDR5 channel. Each demand miss generates an urgent $3.901\text{ ms}$ demand transfer.
2. **Compute Window Insufficiency Under $H=2$:**
   In DeepSeek-V3, GPU computation per stage is $\approx 0.183\text{ ms}$. Over a forward lookahead horizon of $H=2$, the available compute window before the target stage executes is only $2 \times 0.183\text{ ms} = 0.366\text{ ms}$.
   Transferring an $84.0\text{ MB}$ expert over a $22.578\text{ GB/s}$ DDR5 bus requires a physical minimum of $3.901\text{ ms}$.
   Because $0.366\text{ ms} \ll 3.901\text{ ms}$, speculative transfers cannot physically complete within the 2-stage window.
3. **Queue Delay Behind Preceding Speculative Transfers:**
   At budget $B=4$, four candidates are dispatched per stage. Due to FIFO scheduling in `DDR5TransferEngine`, candidates experience an average queue waiting time of $238.64\text{ ms}$ (ST-MoE) and $222.33\text{ ms}$ (CSTP). Useful prefetches spend their entire lookahead window sitting in the DDR5 transfer queue.
4. **Inter-Token Time Reset Anomaly:**
   In `test_deepseekv3_mixtral_style.cpp:464`, `device->status.device_time` is reset to $0$ at the start of each decoding step, while `cumulative_device_time` is tracked separately. Consequently, prefetch transfers dispatched near the end of a token do not bridge smoothly into the next token's timeline.

### 3. Latency Stall and DDR5 Overhead Comparison
- **Exposed Memory Stall:** Remained completely identical at $7,884.24\text{ ms}$ across `B`, `FWD-S-H2`, and `FWD-C-H2`.
- **Execution Time:** Remained completely identical at $8,003.70\text{ ms}$.
- **DDR5 Traffic Overhead:**
  - `FWD-S-H2` transmitted $33,180.0\text{ MB}$ of speculative traffic ($+19.5\%$), yielding $0.00\text{ ms}$ stall reduction.
  - `FWD-C-H2` transmitted $38,388.0\text{ MB}$ of speculative traffic ($+22.6\%$), yielding $0.00\text{ ms}$ stall reduction.

---

## 5. Summary Table for Decision

| Criterion | Target Metric | Baseline (`B`) | `FWD-S-H2` | `FWD-C-H2` | Goal Achieved? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Timely Hits** | Timely resident hits $> 0$ | $0$ | **$0$** | **$0$** | **NO** |
| **Stall Reduction** | Exposed stall reduction ($ms$) | Baseline ($0$) | **$0.00\text{ ms}$** | **$0.00\text{ ms}$** | **NO** |
| **Execution Speedup** | Total execution time reduction ($ms$) | Baseline ($0$) | **$0.00\text{ ms}$** | **$0.00\text{ ms}$** | **NO** |
| **Bandwidth Efficiency** | DDR5 Traffic Overhead | $169.8\text{ GB}$ | **$202.9\text{ GB}$ ($+19.5\%$)** | **$208.2\text{ GB}$ ($+22.6\%$)** | **NO (Pure waste)** |
| **Prediction Accuracy** | Precision / Recall / F1 | N/A | $19.69\% / 9.85\% / 13.13\%$ | $19.09\% / 9.55\% / 12.73\%$ | **Low utility** |

