# Controlled Experiment Design — LRU vs FA-LRU

## MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference

---

## 1. Research Questions

This experimental campaign investigates four core research questions:
- **RQ1 (Cache Hit Rate):** Does Frequency-Aware LRU (FA-LRU) significantly increase HBM hit rates compared to standard LRU by retaining heavy-hitter experts?
- **RQ2 (DDR5 Traffic & Churn):** Does FA-LRU reduce redundant DDR5 expert reload traffic (evicting and reloading the same expert across consecutive tokens)?
- **RQ3 (Latency & Stall Reduction):** Does the higher hit rate translate into measurable reductions in exposed GPU memory stall time and end-to-end token latency?
- **RQ4 (Prefetch Interaction):** How does FA-LRU interact with speculative prefetchers (ST-MoE and CSTP)? Does it prevent speculative transfers from prematurely evicting hot demand experts, or does it improve prefetch timeliness?

---

## 2. Controlled Experiment Matrix

We design a 6-way controlled factorial experiment ($2 \text{ cache policies} \times 3 \text{ prefetching strategies}$):

| Experiment ID | Prefetch Strategy | Eviction Policy | Description | Primary Objective |
| :--- | :--- | :--- | :--- | :--- |
| **E1** | None (Baseline) | Standard LRU | Demand-only loading under baseline LRU eviction | Reference baseline |
| **E2** | None (Baseline) | FA-LRU | Demand-only loading with Frequency-Aware LRU | Isolate pure cache policy gain |
| **E3** | ST-MoE ($H=1, B=4$) | Standard LRU | ST-MoE lookahead prefetching with standard LRU | Prefetching baseline |
| **E4** | ST-MoE ($H=1, B=4$) | FA-LRU | ST-MoE lookahead prefetching with FA-LRU | Evaluate prefetch + FA-LRU synergy |
| **E5** | CSTP ($H=1, B=4$) | Standard LRU | CSTP confidence-filtered prefetching with LRU | Prefetching baseline |
| **E6** | CSTP ($H=1, B=4$) | FA-LRU | CSTP confidence-filtered prefetching with FA-LRU | Evaluate prefetch + FA-LRU synergy |

---

## 3. Representative Workloads & Model Configurations

To ensure direct continuity with Point 3 without redundant simulation runs, we select four representative workloads spanning batch-size and sequence-length axes:

| Workload ID | Batch Size ($B$) | Output Tokens ($T$) | MoE Stages ($L$) | Total Routed Experts | Trace Source | Routing Mode |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **W1** | 1 | 10 | 40 | 10,240 | MoE-Anywhere Prompts 60–67 | Real Replay |
| **W2** | 2 | 10 | 40 | 10,240 | MoE-Anywhere Prompts 60–67 | Real Replay |
| **W5** | 8 | 10 | 40 | 10,240 | MoE-Anywhere Prompts 60–67 | Real Replay |
| **W7** | 16 | 20 | 40 | 10,240 | MoE-Anywhere Prompts 60–67 | Real Replay |

### Hardware Platform Configuration
- **Accelerator:** NVIDIA H100 GPU architecture (compute peak: 989.4 TFLOPS FP16/BF16).
- **Off-Chip Memory:** DDR5-4800 dual-channel subsystem calibrated to 38.4 GB/s effective bandwidth.
- **Interconnect:** PCIe Gen5 $\times16$ / NVLink 3 interface.
- **Clock Continuity:** Enabled (`CLOCK_CONTINUITY=1`).
- **Trace Directory:** Isolated per-experiment directories under `mohman/point_4/experiments/`.

---

## 4. HBM Capacity Sweep

Cache policy effectiveness depends strongly on memory pressure. We evaluate three distinct capacity tiers:
1. **Tier 1 — High Pressure (25 GB):**
   - Capacity: $25,000,000,000 \text{ bytes}$.
   - Max Resident Experts: 283 experts (~2.76% of model).
   - *Purpose:* Evaluates policy resilience under severe capacity constraints where evictions occur on almost every token.
2. **Tier 2 — Standard Pressure (35 GB):**
   - Capacity: $35,000,000,000 \text{ bytes}$.
   - Max Resident Experts: 397 experts (~3.88% of model).
   - *Purpose:* Primary comparison point directly matching the Point 3 large-workload campaign.
3. **Tier 3 — Moderate Pressure (50 GB):**
   - Capacity: $50,000,000,000 \text{ bytes}$.
   - Max Resident Experts: 567 experts (~5.54% of model).
   - *Purpose:* Tests whether frequency awareness continues to provide benefits as working set retention expands.

---

## 5. Metrics Specification & Formulations

Every metric is mathematically defined to ensure reproducible aggregation:

### 5.1 Cache & Memory Performance Metrics
- **Overall HBM Hit Rate ($H_{\text{overall}}$):**
  $$H_{\text{overall}} = \frac{N_{\text{resident\_hits}} + N_{\text{timely\_prefetch\_hits}}}{N_{\text{actual\_demands}}}$$
- **Unique HBM Hit Rate ($H_{\text{unique}}$):**
  $$H_{\text{unique}} = \frac{U_{\text{resident\_hits}} + U_{\text{timely\_prefetch\_hits}}}{U_{\text{actual\_demands}}}$$
- **Total Eviction Count ($E_{\text{total}}$):**
  $$E_{\text{total}} = N_{\text{capacity\_evictions}} + N_{\text{speculative\_evictions}}$$
- **Expert Reload Count ($R_{\text{reloads}}$):**
  Number of demand misses where the requested `{layer_id, expert_id}` was previously resident and subsequently evicted.
- **Total DDR5 Memory Traffic ($M_{\text{DDR5}}$):**
  $$M_{\text{DDR5}} = M_{\text{demand}} + M_{\text{speculative}} \quad [\text{GB}]$$

### 5.2 Latency and Stall Metrics
- **Exposed Memory Stall Time ($S_{\text{exposed}}$):**
  Aggregate simulated time (ns) where GPU computation is completely blocked waiting for DDR5 transfers to arrive in HBM.
- **Aggregate Demand Wait Time ($W_{\text{demand}}$):**
  Sum of latency delays experienced by all individual expert demands:
  $$W_{\text{demand}} = \sum_{i} (\text{ready\_ts}_i - \text{demand\_ts}_i) \quad [\text{ns}]$$
- **Total End-to-End Execution Time ($T_{\text{exec}}$):**
  Total simulated time (ns) required to finish the inference workload.
- **Mean Token Latency ($L_{\text{token}}$):**
  $$L_{\text{token}} = \frac{T_{\text{exec}}}{N_{\text{tokens}}} \quad [\text{ns/tok}]$$

### 5.3 Prefetch Effectiveness Metrics
- **Prefetch Coverage ($C_{\text{cov}}$):**
  $$C_{\text{cov}} = \frac{N_{\text{timely\_hits}} + N_{\text{late\_attachments}}}{N_{\text{actual\_demands}}}$$
- **Timely Prefetch Rate ($R_{\text{timely}}$):**
  $$R_{\text{timely}} = \frac{N_{\text{timely\_hits}}}{N_{\text{dispatched\_prefetches}}}$$
- **Unused Prefetch Count ($U_{\text{unused}}$):**
  Number of prefetches completed and admitted to HBM that were evicted without ever being demanded.

---

## 6. Required Publication Plots

The experimental results will be consolidated into 10 publication-quality dual-format (PDF and PNG) figures:

| Fig # | Plot Title | X-Axis | Y-Axis | Comparison Groups | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fig 1** | HBM Hit Rate vs Batch Size | Batch Size ($B$) | HBM Hit Rate (%) | LRU vs FA-LRU across E1–E6 | Direct hit rate gains from frequency tracking |
| **Fig 2** | Demand Miss Count Comparison | Workload ID | Demand Miss Count | LRU vs FA-LRU | Absolute reduction in off-chip DDR5 requests |
| **Fig 3** | Cache Evictions and Expert Reloads | Workload ID | Event Counts | Evictions & Reloads | Demonstrates suppression of cache thrashing |
| **Fig 4** | HBM Working Set Occupancy Over Time | Time (ns) | Resident Bytes (GB) | LRU vs FA-LRU | Illustrates stability of heavy-hitter retention |
| **Fig 5** | Exposed GPU Memory Stall Reduction | Batch Size ($B$) | Exposed Stall (ms) | LRU vs FA-LRU | Direct compute acceleration from fewer misses |
| **Fig 6** | Total DDR5 Memory Traffic Breakdown | Workload ID | Traffic (GB) | Demand vs Speculative bytes | Quantifies off-chip bus bandwidth conservation |
| **Fig 7** | Mean Token Latency vs Batch Size | Batch Size ($B$) | Latency (ms/tok) | LRU vs FA-LRU across E1–E6 | End-to-end user-perceived speedup |
| **Fig 8** | Prefetch Usefulness & Unused Prefetches | Workload ID | Transfer Count | Timely vs Unused transfers | Checks if FA-LRU protects prefetched experts |
| **Fig 9** | Capacity Sensitivity (25G, 35G, 50G) | Capacity (GB) | HBM Hit Rate (%) | LRU vs FA-LRU | Evaluates policy robustness across cache tiers |
| **Fig 10** | Consolidated Policy Performance Radar | 6 Key Axes | Normalized Score [0, 1] | LRU Baseline vs FA-LRU Best | Executive summary radar chart |

