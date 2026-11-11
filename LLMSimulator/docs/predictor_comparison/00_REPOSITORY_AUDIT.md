# Phase A: Repository Audit & Predictor Architecture Baseline

## 1. Overview
This audit establishes the starting baseline for the controlled evaluation between **CSTP** (Confidence- and Frequency-Aware Spatio-Temporal Predictor) and **ST-MoE** (Spatio-Temporal Expert Prefetching, Zhao et al., arXiv:2606.15453v1).

---

## 2. Component Reusability Matrix

| Component | Path | Status | Reusability in Phase B & Beyond |
| :--- | :--- | :--- | :--- |
| **Trace Parser** | `src/predictor/trace_parser.py` | Verified | 100% reusable. Parses CSV traces, validates monotonicity, detects modulo-60 drift, validates expert bounds. |
| **Evaluation Framework** | `src/predictor/evaluation.py` | Verified | High reusability. Enforces request-level split isolation, causal sequencing, and standard metrics (Precision@B, Recall@B, F1@B). |
| **CSTP Implementation** | `src/predictor/cstp_predictor.py` | Verified | 100% preserved. Implements Dirichlet-smoothed conditional spatial + temporal + marginal prior scoring. |
| **Legacy ST-MoE** | `src/predictor/st_moe_predictor.py` | Verified | Preserved as `st_moe_legacy`. Deviates from paper in counter initialization (1 vs 2) and fallback priority union. |
| **Paper-Faithful ST-MoE** | `src/predictor/st_moe_paper.py` | New (Phase B) | Newly implemented to faithfully follow Algorithms 1–3, Equation 1, and 2-bit counter transitions. |
| **Common Interface** | `src/predictor/base_predictor.py` | New (Phase B) | Unified abstract base class and adapter wrappers for `st_moe_paper`, `st_moe_legacy`, and `cstp`. |
| **Phase 2A Off-Chip Cache** | `src/hardware/layer_impl.cpp` | Verified | Fully preserved. Configurable execution cache bypass enabled by default (`bypass_offchip_execution_cache = false`). |

---

## 3. Strict Causality and Data Flow Checks
1. **Prediction Before Actual Routing**: At prediction time for layer $L+1$, the predictor only receives:
   - Observed routing for layer $L$ of the current token.
   - History Table entries from token $t-1$ (actual observed routing from the immediately preceding token).
   - Offline profiling state (CCT or frequency priors).
2. **Post-Prediction State Update**: The predictor's internal state (CCT counters, History Table) is updated strictly after the actual routing for layer $L+1$ has been revealed and scored.
3. **Request Isolation**: Each request sequence has an independent History Table. State does not leak across distinct request IDs.

