# ST-MoE Experiment Configuration and Isolation Protocol

**Deliverable:** Point 1 — Paper-Faithful ST-MoE Baseline  
**Runner Scripts:**
- [`LLMSimulator/scripts/phase2a/run_stmoe_isolated.py`](file:///home/mohman/ca_project/LLMSimulator/scripts/phase2a/run_stmoe_isolated.py)
- [`LLMSimulator/scripts/phase2b/compare_online_policies.py`](file:///home/mohman/ca_project/LLMSimulator/scripts/phase2b/compare_online_policies.py)  
**Evaluator Implementation:** [`LLMSimulator/src/predictor/evaluation.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/evaluation.py)  
**Configuration Manifest:** [`mohman/point_1/st_moe/configs/experiment_manifest.json`](file:///home/mohman/ca_project/mohman/point_1/st_moe/configs/experiment_manifest.json)

---

## 1. Experimental Integrity Architecture

### 1.1 The Cross-Experiment Contamination Hazard (Resolved in Phase 2A)
In early experimental runners, a single `STMoEPaperPredictor` instance was constructed per dataset and reused across sequential evaluations:
1. First, fixed-budget sweeps were run.
2. Next, native threshold runs were evaluated on the same instance.
3. Finally, profiling sensitivity sweeps were executed.

Because the ST-MoE update rule (Algorithm 3) mutates the 2-bit saturating counters and replaces CCT candidates during online evaluation, subsequent runs inherited an altered CCT state. This violated experimental isolation, creating hidden order-dependence and invalidating comparisons between frozen and online regimes.

### 1.2 The Isolation Guarantee
To guarantee absolute experimental integrity, the runner was refactored in Phase 2A:
1. **Fresh Predictor per Condition:** Every combination of dataset, horizon $H$, budget $B$, and evaluation policy receives a freshly initialized `STMoEPaperPredictor` instance.
2. **Fresh Profiling Calibration:** Every fresh instance profiles the identical training trace from scratch to establish an identical initial state.
3. **Verified Invariant:** Regression tests in [`LLMSimulator/tests/test_stmoe_isolation.py`](file:///home/mohman/ca_project/LLMSimulator/tests/test_stmoe_isolation.py) verify (6/6 passing) that running online evaluations on one condition produces zero side-effects on subsequent frozen or online evaluations.

---

## 2. Online Adaptation Policies (Harden in Phase 2B)

Phase 2B standardized the exact semantics of online evaluation by introducing and formally defining two distinct online policies:

```
=============================================================================
Policy 1: online_within_request (PRIMARY STANDARD)
=============================================================================
Request 1: [Profiled CCT] ---> Token 0 -> Token 1 -> ... -> [Mutated CCT]
                                                                |
                                             RESET TO PROFILED CCT
                                                                v
Request 2: [Profiled CCT] ---> Token 0 -> Token 1 -> ... -> [Mutated CCT]

Properties:
* Order-invariant across requests (shuffle test prompts -> identical metric).
* Strict multi-tenant isolation.
* Measures pure intra-prompt semantic context adaptation.

=============================================================================
Policy 2: online_across_requests (SECONDARY BENCHMARK)
=============================================================================
Request 1: [Profiled CCT] ---> Token 0 -> Token 1 -> ... -> [CCT State 1]
                                                                |
                                                    CUMULATIVE CCT TRANSFER
                                                                v
Request 2: [CCT State 1]  ---> Token 0 -> Token 1 -> ... -> [CCT State 2]

Properties:
* Evaluates lifelong, cumulative online adaptation.
* Sensitive to request presentation order.
=============================================================================
```

### Protocol Comparison
- **`online_within_request` (Primary Standard):** Deep-copies the initial profiled predictor state before each request. Within the request, CCT updates occur causally after each token. When the request terminates, the mutated state is discarded, and the original profiled predictor is restored for the next request.
- **`online_across_requests`:** CCT mutations persist continuously across request boundaries.
- **`frozen`:** The CCT is populated purely during offline profiling; all online update mechanisms are completely disabled during test serving.

---

## 3. Causal Boundaries and Metadata Integrity

### 3.1 Strict Causal Execution Pipeline
To prevent data snooping or lookahead leakage, every prediction event adheres strictly to the causal cycle:

$$\text{Predict}(L_{\text{tgt}}) \longrightarrow \text{Score}(P \text{ vs } G) \longrightarrow \text{Receive } G(L_{\text{tgt}}) \longrightarrow \text{Update CCT / HT}$$

The target ground truth routing set $G(L_{\text{tgt}})$ is never exposed to the predictor until after the prediction set $P$ has been locked and scored.

### 3.2 Sentinel Source-Stage Metadata Handling
When lookahead horizon $H \ge 1$, initial layers do not possess a preceding source layer within the active token. Specifically:
- For $H=1$: Layer 0 has no source layer.
- For $H=2$: Layers 0 and 1 have no source layer.

In earlier harnesses, a naïve clamp `max(0, stage - H)` falsely assigned Layer 0 as the source for itself, creating synthetic self-referential correlation lookups.

This was resolved by introducing a formal `_SENTINEL_SOURCE_STAGE`:
```python
if stage_idx < horizon:
    source_stage_id = None
else:
    source_stage_id = stages[stage_idx - horizon].id
```
When `source_stage_id is None`, the predictor falls back to temporal history or global priors without executing an invalid spatial lookup.

---

## 4. Configuration Parameters

The official hyperparameters applied across all ST-MoE experiments are summarized below:

| Component | Configuration Parameter | Setting | Description |
| :--- | :--- | :---: | :--- |
| **CCT Architecture** | `cct_capacity` | `8` | Maximum candidate experts tracked per source expert |
| | `confidence_bits` | `2` | Saturating counter resolution ($0, 1, 2, 3$) |
| | `init_confidence` | `1` | Initial counter state upon candidate insertion |
| | `max_confidence` | `3` | Saturation upper bound |
| | `min_confidence` | `0` | Saturation lower bound |
| **History Table** | `ht_capacity` | `8` | Request-local temporal recency buffer size |
| | `decay_on_step` | `True` | Counters decay if not refreshed in current step |
| **Scoring & Threshold**| `confidence_threshold` | `2` | Paper threshold $\tau = 2$ |
| | `scoring_mode` | `confidence_additive` | Equation 1 literal ($\text{CCT} + \text{HT}$) |
| | `ablation_mode` | `indicator_intersection`| Indicator intersection ($\mathbb{I}(\text{CCT}) + \mathbb{I}(\text{HT})$) |
| **Budgets Evaluated** | `fixed_budgets` | `[1, 2, 4, 6, 8]` | Top-$B$ candidate ranking constraints |
| **Horizons** | `horizons` | `[1, 2]` | Lookahead distance in MoE layers |
| **RNG Seed** | `seed` | `42` | Global deterministic seed |

