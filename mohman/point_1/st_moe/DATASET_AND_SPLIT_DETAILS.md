# MoE Routing Trace Datasets and Partitioning Details

**Deliverable:** Point 1 — Paper-Faithful ST-MoE Baseline  
**Configuration Reference:** [`mohman/point_1/st_moe/configs/experiment_manifest.json`](file:///home/mohman/ca_project/mohman/point_1/st_moe/configs/experiment_manifest.json)  
**Dataset Inventory Source:** [`results/phase2_stmoe_reproduction/dataset_inventory.csv`](file:///home/mohman/ca_project/results/phase2_stmoe_reproduction/dataset_inventory.csv)

---

## 1. Overview of Evaluated Workloads

To benchmark expert prediction rigorously across diverse real-world models, we evaluate four MoE routing datasets categorized into **Authoritative Production Models** and **Diagnostic / Synthetic Workloads**:

| Dataset Identifier | Classification | Source Model | MoE Stages | Routed Experts | Top-$k$ | Test Units | Eval Tokens | Total Test Opportunities |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`moe_anywhere`** | **Authoritative** | InternScience/Sage-A1.5 | 40 | 256 | 8 | 14 prompts | 30 | **16,800** |
| **`moe_suite`** | **Authoritative** | zai-org/GLM-5.2 | 75 | 256 | 8 | 2 tasks | 50 | **7,500** |
| **`duplex_a100`** | Diagnostic | Duplex Synthetic A100 | 8 | 16 | 2 | 8 sequences | 20 | **1,280** |
| **`correlated_synthetic`** | Diagnostic | Markov Correlation Engine | 16 | 32 | 8 | 4 sequences | 20 | **1,140** |
| *`deepseek_random`* | *EXCLUDED* | DeepSeek-V3 (Mini) | 60 | 256 | 8 | 1 prompt | 5 | *Invalid (Data Leak)* |

---

## 2. Authoritative Production Datasets

### 2.1 MoE Anywhere (`moe_anywhere`)
- **Model Architecture:** InternScience/Sage-A1.5, featuring 40 MoE layers with 256 routed experts per layer and Top-8 gating.
- **Trace Source:** High-fidelity execution traces from `carlosfelipe/moe-anywhere-routing-traces` (`e1-sage-a1.5.parquet`), containing 74 diverse real-world prompts across 10 evaluation tasks (totaling 19,018 token transitions).
- **Partitioning Strategy (`deterministic_prompt_disjoint`):**
  - **Training / Profiling:** 52 prompts (13,364 tokens) used exclusively for offline CCT calibration.
  - **Validation:** 8 prompts (2,056 tokens) used for hyperparameter verification.
  - **Test:** 14 prompts (3,598 tokens).
- **Authoritative Test Volume:** Evaluated over 30 tokens per test prompt across all 40 MoE stages:
  $$\text{Opportunities} = 14 \text{ prompts} \times 30 \text{ tokens} \times 40 \text{ stages} = \mathbf{16,800 \text{ opportunities}}$$

### 2.2 MoE Suite (`moe_suite`)
- **Model Architecture:** zai-org/GLM-5.2, an ultra-deep architecture featuring 75 MoE layers with 256 routed experts and Top-8 gating.
- **Trace Source:** Official benchmarks from `mistrjirka/Predicting-Experts-for-MOE-Suite` (`token_cube` configuration) spanning 12 distinct NLP and reasoning tasks (8,191 tokens per task).
- **Partitioning Strategy (`official_task_disjoint`):**
  - **Training / Profiling:** 8 tasks (65,528 tokens) used for offline CCT profiling.
  - **Validation:** 2 tasks (16,382 tokens).
  - **Test:** 2 tasks (16,382 tokens).
- **Authoritative Test Volume:** Evaluated over 50 tokens per test task across all 75 MoE stages:
  $$\text{Opportunities} = 2 \text{ tasks} \times 50 \text{ tokens} \times 75 \text{ stages} = \mathbf{7,500 \text{ opportunities}}$$

---

## 3. Diagnostic Traces

### 3.1 Duplex A100 (`duplex_a100`)
- **Characteristics:** Lightweight 8-layer, 16-expert architecture with narrow Top-2 routing ($K=2$).
- **Role:** Serves as a fast-turnaround diagnostic harness for sanity-checking CCT counters, boundary conditions, and hardware prefetch interaction.

### 3.2 Correlated Synthetic (`correlated_synthetic`)
- **Characteristics:** 16 layers, 32 experts, Top-8 routing generated with controlled synthetic Markov transition matrices.
- **Role:** Verifies theoretical upper bounds when expert transition matrices possess known ground-truth correlation entropy.

---

## 4. Exclusion of Contaminated Dataset (`deepseek_random`)

During our Phase 0 and Phase 2 audits, a trace named `deepseek_random` claimed to represent DeepSeek-V3 routing with 98.3% prediction recall. A granular code and data audit revealed a critical experimental defect:
1. The raw trace file contained **only 1 single prompt consisting of 5 tokens total**.
2. When passed to the split utility, the single prompt was mirrored into both the profiling and evaluation sets.
3. The predictor achieved 98.3% recall simply by memorizing 5 static token transitions that were identical in training and testing.
4. **Action:** `deepseek_random` was immediately and permanently disqualified from the benchmark suite to maintain scientific credibility.

---

## 5. Formal Definition of a Prediction Opportunity

To ensure complete mathematical consistency across all scripts and tables, a **prediction opportunity** is formally defined as:

$$\mathcal{O} = \langle \text{Request ID } r, \text{ Token Index } t, \text{ Target MoE Stage } L_{\text{tgt}} \rangle$$

For each opportunity:
- **Available Context:** Spatial routing outputs from layer $L_{\text{src}} = L_{\text{tgt}} - H$ (for $L_{\text{tgt}} \ge H$), combined with the request-local History Table up to step $(t, L_{\text{tgt}} - 1)$.
- **Ground Truth Target ($G$):** The set of actual routed experts selected by the gating network:
  $$G = \{e_1, e_2, \dots, e_K\}, \quad |G| = K = 8$$
- **Predicted Output ($P$):** The candidate set emitted by ST-MoE:
  - Fixed-Budget: $P = \text{TopB}(\text{Score}(e)), \quad |P| = B$
  - Native Threshold: $P = \{e \mid \text{Score}(e) \ge \tau\}, \quad |P| = \text{Variable}$

