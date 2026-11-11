# Phase 2D: Bounded Forward-Lookahead Smoke Experiment Design (Phase 2E Protocol)

**Authoritative Git Baseline Anchor:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Status:** Designed & Ready for Authorization  
**Date:** November 3, 2026  
**Document Location:** `docs/final_campaign/PHASE2D_SMOKE_EXPERIMENT_DESIGN.md`

---

## 1. Objective & Scope

This document specifies the exact protocol for **Phase 2E: Bounded Forward-Lookahead Smoke Experiment**. 

The primary objective is to empirically confirm that the Phase 2C forward-lookahead scheduler achieves non-zero timely resident hits and reduces exposed memory stall time relative to the legacy co-located baseline, without launching long simulations or full multi-workload matrix campaigns.

### Key Evaluation Criteria
1. **Zero Runtime Crashes:** All 8 smoke configurations must complete successfully with exit code 0.
2. **Timely Resident Hit Emergence:** Forward ST-MoE and CSTP configurations under $H=2$ and $H=3$ must produce non-zero `timely_resident_hits` in `prefetch_effectiveness.csv`.
3. **Stall Reduction:** Exposed memory stall in forward configurations must show a measurable reduction compared to both the demand baseline and the legacy co-located scheduler.
4. **Conservation Integrity:** Metric conservation must be strictly satisfied across all runs:
   $$\text{Prefetches Dispatched} = \text{Demanded} + \text{Unused} + \text{Evicted Before Demand} + \text{Still In Flight} + \text{Cancelled}$$

---

## 2. Workload & Model Selection

### 2.1 Recommended Workload: Workload A1 (Bounded 4-Token Slice)
- **Model:** GLM-5.2 / DeepSeek-V3 Full Depth (78 layers: 3 dense + 75 MoE stages, 256 experts/stage, Top-8 routing, 88.08 MB FP16 expert footprint).
- **Trace Provenance:** Real recorded routing trace from MoE-Suite Task 12:  
  `LLMSimulator/results/endsem_phase_e/final_baseline/traces/ms_task12_t128_l75.csv` (1,019 KB, verified present).
- **Token Count:** **4 decoding tokens** (`iter: 4`).
- **Total MoE Stage Executions:** $4 \text{ tokens} \times 75 \text{ MoE stages} = 300\text{ stage events}$.
- **Total Routed Expert Demands:** $300 \times 8 = 2,400\text{ expert demand accesses}$.
- **Execution Wall-Clock Time:** $\sim 6 - 8\text{ seconds}$ per run ($\approx 1\text{ minute}$ for all 8 runs).

### Why Workload A1 with 4 Tokens?
1. It is completely deterministic and uses an authoritative real recorded trace from MoE-Suite.
2. 75 sequential MoE stages provide a deep pipeline where lookahead horizons $H=1, 2, 3$ have ample opportunity to demonstrate stage-to-stage overlap.
3. 4 tokens allow the online predictors to advance through cold-start ($T=0$) into warm adaptation ($T=1..3$), exercising both prediction tables and cache state.
4. Bounded execution avoids excessive host compute time while exercising 100% of the production prefetch code paths.

---

## 3. The 8-Configuration Experiment Matrix

| Config ID | Configuration Name | Prefetch Mode | Scheduling Mode | Predictor | Horizon $H$ | Budget $B$ | Eviction Policy | PABI Quarantine | Belády Pinning |
|---|---|---|---|---|:---:|:---:|---|:---:|:---:|
| **B** | Baseline | `none` | N/A | None | N/A | 0 | LRU | False | False |
| **LEGACY-S** | Legacy ST-MoE | `hardware_async` | `legacy_current_stage` | `st_moe` | 1 | 4 | PABI | True | True |
| **FWD-S-H1** | Forward ST-MoE H1 | `hardware_async` | `forward_lookahead` | `st_moe` | 1 | 4 | PABI | True | True |
| **FWD-S-H2** | Forward ST-MoE H2 | `hardware_async` | `forward_lookahead` | `st_moe` | 2 | 4 | PABI | True | True |
| **FWD-S-H3** | Forward ST-MoE H3 | `hardware_async` | `forward_lookahead` | `st_moe` | 3 | 4 | PABI | True | True |
| **FWD-C-H1** | Forward CSTP H1 | `hardware_async` | `forward_lookahead` | `cstp` | 1 | 4 | PABI | True | True |
| **FWD-C-H2** | Forward CSTP H2 | `hardware_async` | `forward_lookahead` | `cstp` | 2 | 4 | PABI | True | True |
| **FWD-C-H3** | Forward CSTP H3 | `hardware_async` | `forward_lookahead` | `cstp` | 3 | 4 | PABI | True | True |

---

## 4. Execution Commands & Environment Overrides

Each configuration runs as an isolated subprocess using the compiled `run_deepseekv3_mixtral_style` binary:

```bash
# Output Root
OUTPUT_ROOT="results/final_comparative_campaign/smoke_phase2e"
mkdir -p "${OUTPUT_ROOT}"

# Common Hardware Parameters
COMMON_ENV="USE_EVENT_DRIVEN_DRAM=1 DDR5_BANDWIDTH_GBPS=22.578 PREFETCH_BUDGET=4 PABI_MAX_PINNED_EXPERTS=4"

# 1. Baseline (B)
PREFETCH_ENABLED=0 EVICTION_POLICY=lru TRACE_DIR="${OUTPUT_ROOT}/B" \
  ${COMMON_ENV} ./build/run_deepseekv3_mixtral_style configs/A1_smoke_4token.yaml

# 2. Legacy ST-MoE (LEGACY-S)
PREFETCH_ENABLED=1 PREFETCH_MODE=hardware_async PREFETCH_PREDICTOR_TYPE=st_moe \
  PREFETCH_SCHEDULING_MODE=legacy_current_stage PREFETCH_PREDICTION_HORIZON=1 \
  EVICTION_POLICY=pabi PABI_QUARANTINE_ENABLED=1 PABI_PINNING_ENABLED=1 \
  TRACE_DIR="${OUTPUT_ROOT}/LEGACY-S" ${COMMON_ENV} ./build/run_deepseekv3_mixtral_style configs/A1_smoke_4token.yaml

# 3. Forward ST-MoE H=1 (FWD-S-H1)
PREFETCH_ENABLED=1 PREFETCH_MODE=hardware_async PREFETCH_PREDICTOR_TYPE=st_moe \
  PREFETCH_SCHEDULING_MODE=forward_lookahead PREFETCH_PREDICTION_HORIZON=1 \
  EVICTION_POLICY=pabi PABI_QUARANTINE_ENABLED=1 PABI_PINNING_ENABLED=1 \
  TRACE_DIR="${OUTPUT_ROOT}/FWD-S-H1" ${COMMON_ENV} ./build/run_deepseekv3_mixtral_style configs/A1_smoke_4token.yaml

# 4. Forward ST-MoE H=2 (FWD-S-H2)
PREFETCH_ENABLED=1 PREFETCH_MODE=hardware_async PREFETCH_PREDICTOR_TYPE=st_moe \
  PREFETCH_SCHEDULING_MODE=forward_lookahead PREFETCH_PREDICTION_HORIZON=2 \
  EVICTION_POLICY=pabi PABI_QUARANTINE_ENABLED=1 PABI_PINNING_ENABLED=1 \
  TRACE_DIR="${OUTPUT_ROOT}/FWD-S-H2" ${COMMON_ENV} ./build/run_deepseekv3_mixtral_style configs/A1_smoke_4token.yaml

# 5. Forward ST-MoE H=3 (FWD-S-H3)
PREFETCH_ENABLED=1 PREFETCH_MODE=hardware_async PREFETCH_PREDICTOR_TYPE=st_moe \
  PREFETCH_SCHEDULING_MODE=forward_lookahead PREFETCH_PREDICTION_HORIZON=3 \
  EVICTION_POLICY=pabi PABI_QUARANTINE_ENABLED=1 PABI_PINNING_ENABLED=1 \
  TRACE_DIR="${OUTPUT_ROOT}/FWD-S-H3" ${COMMON_ENV} ./build/run_deepseekv3_mixtral_style configs/A1_smoke_4token.yaml

# 6. Forward CSTP H=1 (FWD-C-H1)
PREFETCH_ENABLED=1 PREFETCH_MODE=hardware_async PREFETCH_PREDICTOR_TYPE=cstp \
  PREFETCH_SCHEDULING_MODE=forward_lookahead PREFETCH_PREDICTION_HORIZON=1 \
  EVICTION_POLICY=pabi PABI_QUARANTINE_ENABLED=1 PABI_PINNING_ENABLED=1 \
  TRACE_DIR="${OUTPUT_ROOT}/FWD-C-H1" ${COMMON_ENV} ./build/run_deepseekv3_mixtral_style configs/A1_smoke_4token.yaml

# 7. Forward CSTP H=2 (FWD-C-H2)
PREFETCH_ENABLED=1 PREFETCH_MODE=hardware_async PREFETCH_PREDICTOR_TYPE=cstp \
  PREFETCH_SCHEDULING_MODE=forward_lookahead PREFETCH_PREDICTION_HORIZON=2 \
  EVICTION_POLICY=pabi PABI_QUARANTINE_ENABLED=1 PABI_PINNING_ENABLED=1 \
  TRACE_DIR="${OUTPUT_ROOT}/FWD-C-H2" ${COMMON_ENV} ./build/run_deepseekv3_mixtral_style configs/A1_smoke_4token.yaml

# 8. Forward CSTP H=3 (FWD-C-H3)
PREFETCH_ENABLED=1 PREFETCH_MODE=hardware_async PREFETCH_PREDICTOR_TYPE=cstp \
  PREFETCH_SCHEDULING_MODE=forward_lookahead PREFETCH_PREDICTION_HORIZON=3 \
  EVICTION_POLICY=pabi PABI_QUARANTINE_ENABLED=1 PABI_PINNING_ENABLED=1 \
  TRACE_DIR="${OUTPUT_ROOT}/FWD-C-H3" ${COMMON_ENV} ./build/run_deepseekv3_mixtral_style configs/A1_smoke_4token.yaml
```

---

## 5. Required Output Artifacts & Validation Checklist

Each configuration directory `${OUTPUT_ROOT}/<config_id>/` must produce:
1. `summary.csv`: Overall simulated execution time, total stall time, total traffic, and HBM hit rate.
2. `expert_cache.csv`: Line-by-line event log of cache hits, misses, speculative insertions, promotions, and evictions.
3. `prefetch_effectiveness.csv`: Complete lifecycle accounting of all prefetch transfers and demand outcomes.
4. `live_predictions.csv` & `live_prediction_summary.json`: Predictor precision, recall, F1, MRR, nDCG, and latencies.

### Gating Conditions for Campaign Authorization:
- [ ] All 8 runs finish with exit code 0.
- [ ] No NaN, negative latency, or zero-byte anomaly in any output CSV.
- [ ] In `FWD-S-H2`, `FWD-S-H3`, `FWD-C-H2`, `FWD-C-H3`: `timely_resident_hits > 0`.
- [ ] `timely_resident_hits(FWD-S-H2) > timely_resident_hits(LEGACY-S)`.
- [ ] Metric conservation check holds: `Submitted == Demanded + Unused + EvictedBeforeDemand + InFlightAtEnd + Cancelled`.

---
*Deliverable approved for Phase 2D.*

