# ST-MoE Configuration Manifest and Environment Specification

**Directory:** [`mohman/point_1/st_moe/configs/`](file:///home/mohman/ca_project/mohman/point_1/st_moe/configs/)

---

## Configuration Overview

This directory provides the authoritative configuration specifications for Point 1 (ST-MoE baseline reproduction). All experimental conditions, dataset partitions, predictor hyperparameters, and evaluation protocols are machine-readable in [`experiment_manifest.json`](file:///home/mohman/ca_project/mohman/point_1/st_moe/configs/experiment_manifest.json).

### Hyperparameters Summary

| Parameter | Value | Reference / Justification |
| :--- | :---: | :--- |
| `cct_capacity` | 8 | Top candidate correlation capacity per source expert (Paper Section 4.2) |
| `confidence_bits` | 2 | 2-bit saturating counter ($0 \dots 3$) (Paper Section 4.2) |
| `confidence_init` | 1 | Weakly active initialization upon insertion |
| `confidence_max` | 3 | Maximum saturation bound |
| `confidence_threshold` | 2 | Native threshold $\tau = 2$ for dynamic admission (Paper Eq. 1) |
| `history_table_capacity` | 8 | Per-request temporal recency buffer |
| `fixed_budgets` | $\{1, 2, 4, 6, 8\}$ | Bounded prefetch budgets for hardware memory constraint sweeps |
| `horizons` | $\{1, 2\}$ | $H=1$ (adjacent next layer) and $H=2$ (skip-layer lookahead) |
| `primary_online_policy` | `online_within_request` | Profiled state restored at request boundaries for request-order invariance |

### Deterministic Splits

1. **`moe_anywhere` (Sage-A1.5, 40 MoE Layers, 256 Experts, Top-8):**
   - 74 total prompts partitioned into 52 train (profiling), 8 validation, 14 test.
   - Evaluated across 30 evaluation tokens per prompt = **16,800 opportunities**.
2. **`moe_suite` (GLM-5.2, 75 MoE Layers, 256 Experts, Top-8):**
   - 12 official tasks partitioned into 8 train (profiling), 2 validation, 2 test.
   - Evaluated across 50 evaluation tokens per task = **7,500 opportunities**.

