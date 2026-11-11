# Phase 2E.3 — Modified-Code Baseline Validation Report

## 1. Executive Summary

This report validates that the Phase 2E.2 source code modifications (stage-ID decoupling, candidate queue filtering, remaining transfer duration accounting, trace metadata propagation, physical start timestamps, and the 12th `transfer_id` column in `expert_load.csv`) strictly preserve the behavior of the demand-only baseline (`B`).

The demand-only baseline was executed using the modified simulator code on Workload A1 (DeepSeek-V3 Mixtral-style, 4 decoding tokens, 75 MoE layers, top-8 routing) and compared against the historical Phase 2E baseline anchor (`results/final_comparative_campaign/smoke_phase2e/B`).

**Conclusion:** The modified codebase matches the historical Phase 2E baseline **identically to 6 significant figures** across execution time, exposed memory stall, HBM hits, demand misses, hit rate, and physical DDR5 transfers. No performance drift, timing distortion, or semantic regression occurred.

---

## 2. Quantitative Baseline Comparison: Phase 2E.3 vs. Historical Phase 2E

The table below contrasts the modified-code execution against the historical Phase 2E baseline:

| Metric | Historical Phase 2E Baseline (`smoke_phase2e/B`) | Modified Phase 2E.3 Baseline (`phase2e3_warm_diagnostic/B`) | Absolute Difference | Relative Difference | Validation Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Simulated Execution Time** | $8,003.70\text{ ms}$ | $8,003.70\text{ ms}$ | $0.0000\text{ ms}$ | **$0.0000\%$** | **MATCH** |
| **Exposed Memory Stall** | $7,884.24\text{ ms}$ | $7,884.24\text{ ms}$ | $0.0000\text{ ms}$ | **$0.0000\%$** | **MATCH** |
| **MoE DRAM Time** | $5,297.89\text{ ms}$ | $5,297.89\text{ ms}$ | $0.0000\text{ ms}$ | **$0.0000\%$** | **MATCH** |
| **MoE Compute Time** | $54.95\text{ ms}$ | $54.95\text{ ms}$ | $0.0000\text{ ms}$ | **$0.0000\%$** | **MATCH** |
| **Total Demand Lookups** | $7,200$ | $7,200$ | $0$ | **$0.0000\%$** | **MATCH** |
| **Demand HBM Hits** | $5,179$ | $5,179$ | $0$ | **$0.0000\%$** | **MATCH** |
| **Demand HBM Misses** | $2,021$ | $2,021$ | $0$ | **$0.0000\%$** | **MATCH** |
| **HBM Hit Rate** | $71.93\%$ | $71.93\%$ | $0.0000\%$ | **$0.0000\%$** | **MATCH** |
| **Physical DDR5 Transfers** | $2,021$ | $2,021$ | $0$ | **$0.0000\%$** | **MATCH** |
| **Physical DDR5 Demand Bytes** | $178,010,456,064\text{ B}$ | $178,010,456,064\text{ B}$ | $0\text{ B}$ | **$0.0000\%$** | **MATCH** |
| **DDR5 Demand Traffic** | $169,764.0\text{ MB}$ | $169,764.0\text{ MB}$ | $0.0\text{ MB}$ | **$0.0000\%$** | **MATCH** |
| **Average Demand Latency** | $3,901,160.0\text{ ns}$ | $3,901,160.0\text{ ns}$ | $0.0\text{ ns}$ | **$0.0000\%$** | **MATCH** |
| **Routing Records Count** | $2,401$ lines | $2,401$ lines | $0$ lines | **$0.0000\%$** | **MATCH** |
| **Router / Linear Stage Alignment** | $[0 \dots 74] == [0 \dots 74]$ | $[0 \dots 74] == [0 \dots 74]$ | $0$ mismatch | **Identical** | **PASSED** |

---

## 3. Per-Token Latency and Stall Profile

Both historical and modified baselines execute identical per-token step timelines:

| Step / Token | Sequence Length | Step Latency (Simulated) | Cumulative Time | Resident Experts in HBM | Resident Bytes in HBM |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Token 0** | 128 | $2,369.47\text{ ms}$ | $2,369.47\text{ ms}$ | 600 experts | $52.85\text{ GB}$ |
| **Token 1** | 129 | $1,696.57\text{ ms}$ | $4,066.04\text{ ms}$ | 908 experts | $74.48\text{ GB}$ (Capacity reached) |
| **Token 2** | 130 | $2,062.18\text{ ms}$ | $6,128.22\text{ ms}$ | 908 experts | $74.48\text{ GB}$ |
| **Token 3** | 131 | $1,875.48\text{ ms}$ | $8,003.70\text{ ms}$ | 908 experts | $74.48\text{ GB}$ |
| **Total (4 Tokens)** | — | **$8,003.70\text{ ms}$** | **$8,003.70\text{ ms}$** | **908 experts** | **$74.48\text{ GB}$** |

---

## 4. Architectural Analysis: Why Baseline Behavior is Identical

The Phase 2E.2 modifications were designed with strict architectural isolation:

1. **`current_moe_stage_id` Decoupling:**
   In `BatchedSequence::update_expert()`, `current_moe_stage_id` is assigned the current stage index. For the demand-only baseline (`B`), Linear lookups now query `current_moe_stage_id`, which identically equals the routing stage index $[0 \dots 74]$. In baseline `B`, demand lookups were already executing layer by layer; the decoupling ensures that routing and Linear execution reference the exact same stage index without introducing any timing skew or layer skipping.
2. **Transfer ID Column in `expert_load.csv`:**
   Appending `transfer_id` as the 12th column in `expert_load.csv` provides complete traceability while strictly preserving the existing 11 columns, column formats, floating-point precision, and row sequencing. For baseline `B`, each demand miss receives a valid sequential transfer ID, while hits receive `0`.
3. **Inactive Prefetch Paths:**
   In configuration `B`, `prefetch_enabled` is `false`. All candidate queue filtering (P0.2), speculative transfer creation (P1.4), physical service-start timestamps (P1.5), and PABI quarantine/pinning remain completely dormant.

---

## 5. Verification Checklist

- [x] Execution time matches within $\pm 1\%$ (actual difference: $0.0000\%$).
- [x] Exposed stall matches within $\pm 1\%$ (actual difference: $0.0000\%$).
- [x] Demand HBM hits and misses match identically ($5,179$ hits, $2,021$ misses).
- [x] Physical DDR5 transfers match identically ($2,021$ transfers).
- [x] Stage-ID alignment verified ($[0 \dots 74]$ on Router identically equals $[0 \dots 74]$ on Linear).
- [x] Zero ternary operators in C++ and Python AST.

**Validation Status: FULLY PASSED.**
