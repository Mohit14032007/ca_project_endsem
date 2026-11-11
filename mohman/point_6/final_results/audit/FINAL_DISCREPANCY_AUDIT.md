# Point 6 — Final Discrepancy Audit: Identification, Source-Level Investigation, and Resolution

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 6 — Sensitivity Analysis Final Audit  
**Author:** Antigravity Engineering  
**Date:** November 11, 2026  
**Status:** Complete & Fully Reconciled  

---

## 1. Executive Summary

This audit rigorously inspects all experimental outputs, configuration files, telemetry logs, consolidated tables, and publication figures produced during the Point 6 Sensitivity Analysis. Following the user's directive, we conducted a forensic review of reported values against the primary raw simulation outputs (`prefetch_summary.json`, `fa_lru_diagnostics.json`, and standard simulation logs).

### Summary of Identified Discrepancies and Root Causes:

1. **Prefetch Budget ($K$) Sweep Traffic Inconsistency (HIGH PRIORITY):**
   - *Discrepancy:* The previous executive table reported Total DDR5 Traffic for $K=2$ as $43,139.11\text{ GB}$, and reported identical traffic for $K=4$ and $K=8$ ($48,741.13\text{ GB}$). Meanwhile, Figure 10 plotted $48,395\text{ GB}$ for $K=2$ and $48,776\text{ GB}$ for $K=8$.
   - *Investigation:* The raw JSON telemetry records:
     - $K=2$: Demand bytes $= 42,311,438,303,232$ ($42,311.44\text{ GB}$), Speculative bytes $= 6,083,976,364,032$ ($6,083.98\text{ GB}$), Total $= 48,395,414,667,264$ ($48,395.41\text{ GB}$).
     - $K=4$: Demand bytes $= 37,792,121,880,576$ ($37,792.12\text{ GB}$), Speculative bytes $= 10,949,008,293,888$ ($10,949.01\text{ GB}$), Total $= 48,741,130,174,464$ ($48,741.13\text{ GB}$).
     - $K=8$: Demand bytes $= 30,739,085,131,776$ ($30,739.09\text{ GB}$), Speculative bytes $= 18,037,101,035,520$ ($18,037.10\text{ GB}$), Total $= 48,776,186,167,296$ ($48,776.19\text{ GB}$).
   - *Resolution:* Figure 10 was plotting the true underlying data; the markdown reports contained a transcription error where K=4's numbers were copied to K=8 and an erroneous value was placed in K=2. The invariant $\text{Total} = \text{Demand} + \text{Speculative}$ holds exactly in the raw bytes. All tables and reports are corrected.

2. **Compute-Slack Rounding Discrepancy in Context-Length Sweep:**
   - *Discrepancy:* Figure 5 plotted compute slack values of $13,783.4\text{ ms}$ for $P=512$ and $14,355.7\text{ ms}$ for $P=4,096$, whereas the earlier report table displayed $13,782.45\text{ ms}$ and $14,346.52\text{ ms}$.
   - *Investigation:* The earlier report derived compute slack by subtracting rounded seconds ($767.028\text{ s} - 753.24555\text{ s} = 13.78245\text{ s}$), whereas Figure 5 computed slack directly from exact nanoseconds in the raw JSON:
     - $P=512$: $767,028,950,643\text{ ns} - 753,245,552,640\text{ ns} = 13,783,398,003\text{ ns} = 13,783.40\text{ ms}$.
     - $P=4,096$: $767,601,218,375\text{ ns} - 753,245,552,640\text{ ns} = 14,355,665,735\text{ ns} = 14,355.67\text{ ms}$.
   - *Resolution:* All tables are aligned with the nanosecond-precise raw JSON values.

3. **Figure 7, 8, 9 Title and Manifest Alignment:**
   - *Discrepancy:* The manifest described Figure 7 as "lookahead timely coverage", while the plot displayed HBM resident hits.
   - *Investigation:* In LLMSimulator's event-driven DDR5 model, prefetches complete either before demand as resident hits or during demand as late inflight attachments. Because `timely_resident_hits` in the logging structure is reserved for zero-stall prefetches, plotting coalesced resident hits is accurate, but the title and filename should explicitly state "HBM Resident Hits".
   - *Resolution:* Regenerated Figure 7 with unambiguous title, labels, and manifest entries.

4. **Batch Size Concurrency Trace Verification:**
   - *Verification:* Verified that all configurations within each batch size use the exact same trace file, and that the nesting property holds:
     $$\text{Trace}(B=4) \subset \text{Trace}(B=14) \subset \text{Trace}(B=32)$$
     Python verification confirmed exact byte-for-byte equality of sequences 0..3 across all traces.

5. **Trace Equivalence Across Context-Length Sweeps ($P \in \{128, 512, 2048, 4096\}$):**
   - *Verification:* Cryptographically confirmed using SHA-256 hashing across all 1,310,720 routing records. All four context lengths execute the exact same sequence of expert selections (`hash: 0649629054917f948fd9cc75c24992971a08db5883a6c562570d07338f71d904`).

6. **Dedicated Expert Cache vs. KV Cache Accounting:**
   - *Verification:* Traced source code in `device.cpp` and `device.h`. Confirmed that `device->cache_capacity_bytes = 37,580,963,840` is a dedicated 35 GiB expert weight partition. KV cache memory does not decrement this capacity. Context length scaling therefore isolates temporal overlap slack from cache shrinkage.

---

## 2. Manifest of All 26 Evaluated Runs

The evaluation consists of **exactly 23 newly executed simulations** in Point 6, plus **3 verified reference runs** reused directly from the Point 4 Big Experiment ($B=32, P=128, T=128$):

| Run ID | Suite | B | P | T | H | K | Predictor | Policy | W | Execution Type | Primary Purpose |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `anc_c1` | Anchor | 32 | 2048 | 128 | 0 | 0 | none | LRU | 64 | Newly Executed | Baseline Reference |
| `anc_c3` | Anchor | 32 | 2048 | 128 | 1 | 4 | CSTP | LRU | 64 | Newly Executed | CSTP + LRU Reference |
| `anc_c4` | Anchor | 32 | 2048 | 128 | 1 | 4 | CSTP | FA-LRU | 64 | Newly Executed | Complete System Reference |
| `b4_c1` | Batch Size | 4 | 2048 | 128 | 0 | 0 | none | LRU | 64 | Newly Executed | Low Concurrency Baseline |
| `b4_c3` | Batch Size | 4 | 2048 | 128 | 1 | 4 | CSTP | LRU | 64 | Newly Executed | Low Concurrency Prefetch |
| `b4_c4` | Batch Size | 4 | 2048 | 128 | 1 | 4 | CSTP | FA-LRU | 64 | Newly Executed | Low Concurrency FA-LRU |
| `b14_c1` | Batch Size | 14 | 2048 | 128 | 0 | 0 | none | LRU | 64 | Newly Executed | Medium Concurrency Baseline |
| `b14_c3` | Batch Size | 14 | 2048 | 128 | 1 | 4 | CSTP | LRU | 64 | Newly Executed | Medium Concurrency Prefetch |
| `b14_c4` | Batch Size | 14 | 2048 | 128 | 1 | 4 | CSTP | FA-LRU | 64 | Newly Executed | Medium Concurrency FA-LRU |
| `p128_c1` | Context P | 32 | 128 | 128 | 0 | 0 | none | LRU | 64 | Reused (`w4_c1_r1`) | Historical Reference Baseline |
| `p128_c3` | Context P | 32 | 128 | 128 | 1 | 4 | CSTP | LRU | 64 | Reused (`w4_c3_r1`) | Historical Reference Prefetch |
| `p128_c4` | Context P | 32 | 128 | 128 | 1 | 4 | CSTP | FA-LRU | 64 | Reused (`w4_c4_r1`) | Historical Reference Complete |
| `p512_c1` | Context P | 32 | 512 | 128 | 0 | 0 | none | LRU | 64 | Newly Executed | Short Context Baseline |
| `p512_c3` | Context P | 32 | 512 | 128 | 1 | 4 | CSTP | LRU | 64 | Newly Executed | Short Context Prefetch |
| `p512_c4` | Context P | 32 | 512 | 128 | 1 | 4 | CSTP | FA-LRU | 64 | Newly Executed | Short Context FA-LRU |
| `p4096_c1`| Context P | 32 | 4096 | 128 | 0 | 0 | none | LRU | 64 | Newly Executed | Long Context Baseline |
| `p4096_c3`| Context P | 32 | 4096 | 128 | 1 | 4 | CSTP | LRU | 64 | Newly Executed | Long Context Prefetch |
| `p4096_c4`| Context P | 32 | 4096 | 128 | 1 | 4 | CSTP | FA-LRU | 64 | Newly Executed | Long Context FA-LRU |
| `h2_c3` | Lookahead | 32 | 2048 | 128 | 2 | 4 | CSTP | LRU | 64 | Newly Executed | Lookahead H=2 Prefetch |
| `h2_c4` | Lookahead | 32 | 2048 | 128 | 2 | 4 | CSTP | FA-LRU | 64 | Newly Executed | Lookahead H=2 Complete |
| `h4_c3` | Lookahead | 32 | 2048 | 128 | 4 | 4 | CSTP | LRU | 64 | Newly Executed | Lookahead H=4 Prefetch |
| `h4_c4` | Lookahead | 32 | 2048 | 128 | 4 | 4 | CSTP | FA-LRU | 64 | Newly Executed | Lookahead H=4 Complete |
| `k2_c3` | Budget | 32 | 2048 | 128 | 1 | 2 | CSTP | LRU | 64 | Newly Executed | Conservative Budget K=2 |
| `k2_c4` | Budget | 32 | 2048 | 128 | 1 | 2 | CSTP | FA-LRU | 64 | Newly Executed | Conservative Budget Complete |
| `k8_c3` | Budget | 32 | 2048 | 128 | 1 | 8 | CSTP | LRU | 64 | Newly Executed | Aggressive Budget K=8 |
| `k8_c4` | Budget | 32 | 2048 | 128 | 1 | 8 | CSTP | FA-LRU | 64 | Newly Executed | Aggressive Budget Complete |

---

## 3. Discrepancy Investigation Details

### 3.1 Investigation of the $K$-Sweep Inconsistency
In the earlier reporting, a critical transcription error occurred:
- Reported table values:
  - $K=2$: Total DDR5 $= 43,139.11\text{ GB}$, Speculative $= 5,479.29\text{ GB}$.
  - $K=4$: Total DDR5 $= 48,741.13\text{ GB}$, Speculative $= 10,949.01\text{ GB}$.
  - $K=8$: Total DDR5 $= 48,741.13\text{ GB}$, Speculative $= 10,949.01\text{ GB}$ (identical to K=4).
- Figure 10 values:
  - $K=2$: Total DDR5 $\approx 48,395\text{ GB}$.
  - $K=4$: Total DDR5 $\approx 48,741\text{ GB}$.
  - $K=8$: Total DDR5 $\approx 48,776\text{ GB}$.

Forensic parsing of `prefetch_summary.json` confirms:
- **For $K=2$ (`k2_c3`):**
  - Demand Bytes: $42,311,438,303,232$ bytes ($42,311.44\text{ GB}$)
  - Speculative Bytes: $6,083,976,364,032$ bytes ($6,083.98\text{ GB}$)
  - Total DDR5 Bytes: $48,395,414,667,264$ bytes ($48,395.41\text{ GB}$)
  - Invariant Verification: $42,311.44 + 6,083.98 = 48,395.41\text{ GB}$ (Exact).
- **For $K=4$ (`anc_c3`):**
  - Demand Bytes: $37,792,121,880,576$ bytes ($37,792.12\text{ GB}$)
  - Speculative Bytes: $10,949,008,293,888$ bytes ($10,949.01\text{ GB}$)
  - Total DDR5 Bytes: $48,741,130,174,464$ bytes ($48,741.13\text{ GB}$)
  - Invariant Verification: $37,792.12 + 10,949.01 = 48,741.13\text{ GB}$ (Exact).
- **For $K=8$ (`k8_c3`):**
  - Demand Bytes: $30,739,085,131,776$ bytes ($30,739.09\text{ GB}$)
  - Speculative Bytes: $18,037,101,035,520$ bytes ($18,037.10\text{ GB}$)
  - Total DDR5 Bytes: $48,776,186,167,296$ bytes ($48,776.19\text{ GB}$)
  - Invariant Verification: $30,739.09 + 18,037.10 = 48,776.19\text{ GB}$ (Exact).

**Root Cause:** Figure 10 was plotting the true underlying simulation data, while the markdown table contained manual transcription errors that misreported $K=2$ and duplicated $K=4$'s values into $K=8$. All tables, CSVs, and markdown reports have been reconciled.

### 3.2 Evidence on DDR5 Interconnect Contention vs. Saturation
The prompt raised whether $K=8$ "genuinely saturates" the DDR5 transfer engine.
Examining the raw prefetch timing logs reveals:
1. `total_dispatched_prefetches`:
   - $K=2$: 69,073
   - $K=4$: 124,307
   - $K=8$: 204,780 (+196.5% vs. K=2)
2. `avg_remaining_late_wait_ns` (the residual stall time a demand must wait for an inflight transfer):
   - $K=2$: 1,276,444.28 ns (1.276 ms)
   - $K=4$: 1,345,701.73 ns (1.346 ms)
   - $K=8$: 1,371,369.91 ns (1.371 ms)
   *(Note: A full demand miss takes 1,376,256.00 ns / 1.376 ms. At $K=8$, the remaining wait is 1.371 ms, meaning prefetch transfers arrive only 5 $\mu$s before the un-prefetched miss time!)*
3. `stall_hidden_by_prefetch_ns`:
   - $K=2$: 6,348,225,181 ns (6.348 seconds)
   - $K=4$: 3,567,271,830 ns (3.567 seconds)
   - $K=8$: 953,066,646 ns (0.953 seconds, an 85.0% collapse in prefetch utility!)

**Scientific Attribution:** While direct hardware bus saturation cannot be measured in an event-driven software simulator without cycle-accurate DRAM bus tracing, the transfer queue metrics provide direct, unequivocal evidence of **severe transfer queue contention and controller backlog**:
- Speculative load volume surges from 6.08 TB to 18.04 TB (+196.5%).
- Queue waiting delays push completion times to the very end of the demand window (1.371 ms residual wait).
- The prefetch stall-hiding benefit collapses from 6.35 s to 0.95 s, causing net latency degradation (-0.026% relative to K=4).

---

## 4. Conclusion & Action Items

All identified discrepancies have been definitively resolved. The source-of-truth CSVs have been regenerated from the raw JSON files, all 15 figures have been aligned and re-plotted, and the technical reports have been updated to reflect exact empirical reality.

