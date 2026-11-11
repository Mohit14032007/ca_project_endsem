# Point 1: Correlation-based Spatio-Temporal Predictor (CSTP)

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Deliverable:** Point 1 — Standalone CSTP Expert Prediction Package  
**Target Repository Path:** [`mohman/point_1/cstp/`](file:///home/mohman/ca_project/mohman/point_1/cstp/)  
**Baseline Git Anchor:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Primary Engine:** [`src/cstp_predictor.py`](file:///home/mohman/ca_project/mohman/point_1/cstp/src/cstp_predictor.py) & [`src/cstp_predictor.cpp`](file:///home/mohman/ca_project/mohman/point_1/cstp/src/cstp_predictor.cpp)  
**Evaluator:** [`src/evaluation.py`](file:///home/mohman/ca_project/mohman/point_1/cstp/src/evaluation.py)  

---

## 1. Executive Summary & Predictor Overview

The **Correlation-based Spatio-Temporal Predictor (CSTP)** is a training-free expert prediction framework formulated to forecast routed experts in Mixture-of-Experts (MoE) Large Language Models $H$ stages in advance ($H \in \{1, 2\}$). 

In modern massive MoE architectures (e.g., 256 routed experts per layer across 40–75 layers), transferring expert parameter matrices across host-accelerator interconnects (PCIe / CXL) after router gating computation incurs significant memory stall penalties. CSTP mitigates this memory wall by computing continuous probabilistic rankings over candidate experts ahead of gating execution.

CSTP synthesizes three complementary signals:
1. **Marginal Frequency Prior:** Base empirical probability of expert activation per pipeline stage.
2. **Spatial Co-Activation:** Inter-layer conditional activation probabilities between source layer experts and target layer experts.
3. **Temporal Sequential Continuity:** Intra-layer Markovian transition probabilities capturing sequential expert reuse across successive decoding tokens.

---

## 2. Core Prediction Components

```
                   +---------------------------------------------+
                   |           Current Token Execution           |
                   | Source Experts at Layer L_src: {e_1, ...}   |
                   +----------------------+----------------------+
                                          |
                +-------------------------+-------------------------+
                |                                                   |
                v                                                   v
    +-----------------------+                           +-----------------------+
    | Spatial Predictor     |                           | Temporal Predictor    |
    | P_spatial(e_tgt|e_src)|                           | P_temporal(e_tgt|hist)|
    | Inter-Layer Transitions                           | Intra-Layer History   |
    | Smoothed by Marginal  |                           | Smoothed by Marginal  |
    +-----------+-----------+                           +-----------+-----------+
                |                                                   |
                | Conditional Probabilities                         | Conditional Probabilities
                +-------------------------+-------------------------+
                                          |
                                          v
                    +---------------------------------------------+
                    | Composite Probabilistic Scoring             |
                    | S(e_tgt) = alpha * Spatial + (1-alpha) * Temp|
                    +----------------------+----------------------+
                                          |
                                          v
                    +---------------------------------------------+
                    | Top-B Prefetch Ranking (e.g., B = 8)        |
                    +---------------------------------------------+
```

1. **Frequency-Aware Marginal Prior:** Captures global skew in expert popularity. Highly specialized experts with near-zero base activation receive low baseline priors, while popular hub experts receive higher baseline scores.
2. **Spatial Transition Component:** Models feedforward routing dependencies across layers $L_{\text{src}} \to L_{\text{tgt}} = L_{\text{src}} + H$. Given the activated set $E_{\text{src}}$ at layer $L_{\text{src}}$, it calculates the expected conditional activation probability for every candidate expert $e_j$ at $L_{\text{tgt}}$.
3. **Temporal Transition Component:** Models sequential recurrence within the same layer across autoregressive decoding steps. Given recently activated experts $E_{\text{hist}}$ at layer $L_{\text{tgt}}$, it forecasts continued intra-layer activation for token $t$.

---

## 3. Mathematical Formulation

Let $L_{\text{src}}$ denote the source stage, $L_{\text{tgt}} = L_{\text{src}} + H$ denote the target stage, $M$ denote the total number of experts per stage ($M = 256$), and $E_{\text{src}} \subset \{0, \dots, M-1\}$ denote the set of active source experts ($|E_{\text{src}}| = K = 8$).

### 3.1 Stage Marginal Frequency Prior
The smoothed marginal prior for expert $e_j$ at target stage $L_{\text{tgt}}$ is defined as:

$$P_{\text{marginal}}(e_j^{(L_{\text{tgt}})}) = \frac{C_{\text{marginal}}(L_{\text{tgt}}, e_j) + \epsilon}{\sum_{e'=0}^{M-1} C_{\text{marginal}}(L_{\text{tgt}}, e') + M \cdot \epsilon}$$

where $C_{\text{marginal}}(L_{\text{tgt}}, e_j)$ is the cumulative activation count of expert $e_j$ at stage $L_{\text{tgt}}$, and $\epsilon > 0$ is a Laplace smoothing parameter ensuring non-zero support.

### 3.2 Smoothed Spatial Conditional Probability
The transition probability from source expert $e_i \in E_{\text{src}}$ to target expert $e_j$ is given by Dirichlet-smoothed conditional estimation:

$$P_{\text{spatial}}(e_j^{(L_{\text{tgt}})} \mid e_i^{(L_{\text{src}})}) = \frac{C_{\text{spatial}}(L_{\text{src}}, e_i, L_{\text{tgt}}, e_j) + \lambda \cdot P_{\text{marginal}}(e_j^{(L_{\text{tgt}})})}{\sum_{e'=0}^{M-1} C_{\text{spatial}}(L_{\text{src}}, e_i, L_{\text{tgt}}, e') + \lambda}$$

where $C_{\text{spatial}}$ tracks co-occurrences of $e_i$ at $L_{\text{src}}$ and $e_j$ at $L_{\text{tgt}}$, and $\lambda > 0$ regulates shrinkage toward the marginal prior when empirical pair transitions are sparse.

The aggregated spatial score over all active source experts is the arithmetic mean:

$$S_{\text{spatial}}(e_j^{(L_{\text{tgt}})}) = \frac{1}{|E_{\text{src}}|} \sum_{e_i \in E_{\text{src}}} P_{\text{spatial}}(e_j^{(L_{\text{tgt}})} \mid e_i^{(L_{\text{src}})})$$

### 3.3 Smoothed Temporal Conditional Probability
Given recent history $E_{\text{hist}}$ observed at stage $L_{\text{tgt}}$ over recent decoding steps:

$$P_{\text{temporal}}(e_j^{(L_{\text{tgt}})} \mid e_k^{(L_{\text{tgt}})}) = \frac{C_{\text{temporal}}(L_{\text{tgt}}, e_k, e_j) + \lambda \cdot P_{\text{marginal}}(e_j^{(L_{\text{tgt}})})}{\sum_{e'=0}^{M-1} C_{\text{temporal}}(L_{\text{tgt}}, e_k, e') + \lambda}$$

The aggregated temporal score over active history entries is:

$$S_{\text{temporal}}(e_j^{(L_{\text{tgt}})}) = \frac{1}{|E_{\text{hist}}|} \sum_{e_k \in E_{\text{hist}}} P_{\text{temporal}}(e_j^{(L_{\text{tgt}})} \mid e_k^{(L_{\text{tgt}})})$$

If $E_{\text{hist}} = \emptyset$ (e.g., initial token cold start), $S_{\text{temporal}}(e_j) = P_{\text{marginal}}(e_j)$.

### 3.4 Composite Scoring and Ranking
The overall score $S(e_j)$ combines spatial and temporal components via weighting parameter $\alpha \in [0, 1]$:

$$S(e_j^{(L_{\text{tgt}})}) = \alpha \cdot S_{\text{spatial}}(e_j^{(L_{\text{tgt}})}) + (1 - \alpha) \cdot S_{\text{temporal}}(e_j^{(L_{\text{tgt}})})$$

Candidates are sorted descending by $S(e_j)$, and the top $B$ experts are selected for prefetching.

---

## 4. Fundamental Data Structures

The predictor maintains four core data structures:

1. **`stage_marginal_counts`** (`Dict[Tuple[int, int], int]` in Python; flat 2D array in C++):
   - Dimension: `[num_stages, num_experts]`
   - Purpose: Tracks the cumulative occurrence of each expert at each stage.
2. **`spatial_counts`** (`Dict[Tuple[int, int, int, int], int]` in Python; sparse/dense 4D array in C++):
   - Keys: `(src_stage, src_expert, tgt_stage, tgt_expert)`
   - Purpose: Records observed pairwise transitions across stage pairs separated by lookahead horizon $H$.
3. **`temporal_counts`** (`Dict[Tuple[int, int, int], int]` in Python; flat 3D array in C++):
   - Keys: `(stage, prev_expert, next_expert)`
   - Purpose: Records successive token-to-token expert transitions within the same stage.
4. **`recent_experts` (Request-Local History Buffer)**:
   - Dimension: `[num_stages, history_capacity]`
   - Purpose: Bounded queue of expert activations from recent decoding steps for the active request. Flushed to empty at request boundaries.

---

## 5. Hyperparameter Roles & Defaults

| Parameter | Default Value | Role & Theoretical Interpretation |
| :--- | :---: | :--- |
| **$\alpha$ (alpha)** | `0.5` | Balances inter-layer spatial correlation ($\alpha = 1.0$) against intra-layer temporal persistence ($\alpha = 0.0$). Standard configuration assigns equal weight to spatial and temporal evidence. |
| **$\lambda$ (lambda)** | `1.0` | Dirichlet prior pseudocount. Determines how rapidly empirical transition frequencies dominate the marginal prior. When count evidence is zero, predictive distribution defaults exactly to $P_{\text{marginal}}$. |
| **$\epsilon$ (epsilon)** | `1e-4` | Laplace smoothing parameter for marginal counts. Prevents zero-division and provides uniform baseline regularization for unobserved experts. |
| **History Capacity** | `8` | Maximum number of past experts retained in `recent_experts` per stage during autoregressive generation. |

---

## 6. Offline Profiling via `fit_profile()`

CSTP profiles routing behavior offline using training trace splits:
1. Iterates through training prompts token-by-token and stage-by-stage.
2. Increments `stage_marginal_counts[stage, expert]` for each activated expert.
3. For every pair of stages separated by lookahead $H$, increments `spatial_counts[src_stage, src_expert, tgt_stage, tgt_expert]`.
4. Increments `temporal_counts[stage, prev_expert, curr_expert]` between consecutive tokens.
5. In Phase 1A, profiling fraction sweeps ($25\%, 50\%, 75\%, 100\%$) evaluated the sample efficiency and scaling laws of this profiling phase across 3 distinct random seeds (`42, 123, 999`).

---

## 7. Evaluation Regimes: Frozen vs. Online Within-Request

CSTP supports two strictly defined evaluation policies:

1. **`frozen` (Static Offline Deployment):**
   - Count matrices (`stage_marginal_counts`, `spatial_counts`, `temporal_counts`) are frozen after `fit_profile()`.
   - `recent_experts` updates dynamically during request decoding and clears at request boundaries.
   - Zero mutation of transition counters occurs during evaluation.
2. **`online_within_request` (Primary Causal Standard):**
   - Predictor begins each test request initialized from the offline-profiled count tables.
   - As each token is decoded, observed routing decisions update `spatial_counts` and `temporal_counts` causally (`Predict -> Score -> Update`).
   - At request/prompt boundaries, count tables are reset to their original offline-profiled state.
   - **Guarantees:** Strict request-order invariance, multi-tenant safety, zero cross-request distribution contamination.

---

## 8. Prediction Horizons: $H=1$ vs. $H=2$

- **$H=1$ (Adjacent Layer):** Target stage is $L + 1$. Prefetches experts for the immediately adjacent MoE layer.
- **$H=2$ (Skip Layer):** Target stage is $L + 2$. Prefetches experts two layers in advance, providing double the latency hiding window at the cost of slight spatial correlation decay.

---

## 9. Certified Results Summary (Phase 1A Reproduction)

Authoritative performance across authoritative MoE workloads under primary standard ($B = K = 8$, `online_within_request`, Seed 42):

| Workload / Model | Stages | Profiling % | Horizon | Test Opps | Frozen Recall@8 (%) | Online Recall@8 (%) | Online Gain (pp) | Hit Rate (%) [Hits $\ge 1$] | Exact Match (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MoE Anywhere** (Sage-A1.5) | 40 | 25% | $H=1$ | 16,800 | 43.83% | 44.68% | +0.85 pp | 96.54% | 0.08% |
| **MoE Anywhere** (Sage-A1.5) | 40 | 50% | $H=1$ | 16,800 | 44.53% | 44.85% | +0.32 pp | 96.64% | 0.08% |
| **MoE Anywhere** (Sage-A1.5) | 40 | 75% | $H=1$ | 16,800 | 45.27% | 45.48% | +0.21 pp | 96.81% | 0.09% |
| **MoE Anywhere** (Sage-A1.5) | 40 | 100% | $H=1$ | 16,800 | **45.37%** | **45.51%** | +0.14 pp | **96.67%** | 0.08% |
| **MoE Anywhere** (Sage-A1.5) | 40 | 100% | $H=2$ | 16,800 | 44.13% | 44.24% | +0.11 pp | 95.84% | 0.07% |
| **MoE Suite** (GLM-5.2) | 75 | 25% | $H=1$ | 7,500 | 15.05% | 15.48% | +0.43 pp | 81.33% | 0.00% |
| **MoE Suite** (GLM-5.2) | 75 | 50% | $H=1$ | 7,500 | 14.95% | 15.21% | +0.26 pp | 81.27% | 0.00% |
| **MoE Suite** (GLM-5.2) | 75 | 75% | $H=1$ | 7,500 | 18.03% | 18.14% | +0.11 pp | 85.57% | 0.00% |
| **MoE Suite** (GLM-5.2) | 75 | 100% | $H=1$ | 7,500 | 16.47% | 16.59% | +0.12 pp | 82.59% | 0.00% |
| **MoE Suite** (GLM-5.2) | 75 | 100% | $H=2$ | 7,500 | 16.27% | 16.36% | +0.09 pp | 82.20% | 0.00% |

> [!NOTE]
> All metrics derived strictly from [`results/csv/corrected_aggregate_metrics.csv`](file:///home/mohman/ca_project/mohman/point_1/cstp/results/csv/corrected_aggregate_metrics.csv). Exact match is defined as retrieving all 8 target experts correctly in an opportunity ($\mathbb{I}(|P_i \cap G_i| = 8)$).

---

## 10. Reproduction of Phase 1A Results

To inspect or reproduce the certified Phase 1A outputs using the frozen artifacts:

```bash
# 1. Run unit and certification validation test suite
python3 -m unittest mohman/point_1/cstp/tests/test_phase1a_certified_correction.py

# 2. Run standalone CSTP unit test suite
python3 -m unittest mohman/point_1/cstp/tests/test_cstp.py

# 3. View the full certified analysis report
cat mohman/point_1/cstp/reports/PHASE1A_CORRECTED_FINAL_REPORT.md
```

---

## 11. Python vs. C++ Implementation Comparison

| Dimension | Python Implementation ([`src/cstp_predictor.py`](file:///home/mohman/ca_project/mohman/point_1/cstp/src/cstp_predictor.py)) | C++ Implementation ([`src/cstp_predictor.cpp`](file:///home/mohman/ca_project/mohman/point_1/cstp/src/cstp_predictor.cpp)) |
| :--- | :--- | :--- |
| **Execution Environment** | Offline evaluation harness, unit tests, analytical sweeps | Cycle-accurate hardware simulator execution (`LLMSimulator`) |
| **Data Representation** | Python `dict` and `numpy.ndarray` with dynamic hashing | Contiguous `std::vector<uint32_t>` and flat 1D/2D arrays |
| **Scoring Arithmetic** | Double-precision IEEE 754 floating point | Single-precision float / fixed-point integer accumulation |
| **Memory Footprint** | Dynamic sparse allocation per observed transition key | Static pre-allocated matrix sized to `[stages, experts]` |
| **State Reset Mechanics** | In-memory deepcopy / state dictionary replacement | Fast `std::fill` / `memcpy` from baseline snapshot |

---

## 12. Known Limitations & Scientific Caveats

1. **Count-Mass Saturation Effect:**
   As offline profiling increases from 25% to 100%, accumulated transition counts in `spatial_counts` and `temporal_counts` grow from $\sim 10^5$ to $> 10^6$. Because the denominator of Dirichlet smoothing scales with total count mass, online single-token observations provide diminishing marginal probability adjustments (+0.85 pp gain at 25% drops to +0.14 pp at 100%).
2. **MoE Suite Statistical Scope ($N=2$ Tasks):**
   MoE Suite routing traces contain only two test tasks (`gsm8k` and `alpaca`). In accordance with rigorous scientific standards, all MoE Suite aggregate figures and confidence intervals are certified as `DESCRIPTIVE_ONLY_N2_TASKS`.
3. **Cross-Task Reversal on MoE Suite:**
   On MoE Suite, 75% profiling achieves higher recall (18.03%) than 100% profiling (16.47%). This descriptive discrepancy arises because the 100% training split introduces additional out-of-domain tasks that dilute the specific spatial transitions relevant to `gsm8k` and `alpaca`.

