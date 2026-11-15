# build_part1.py: Generates Cover Page, Executive Summary, Project Overview, and Motivation

part1_text = """# MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference
## Comprehensive Technical Implementation and Experimental Evaluation Report

---

### Project Metadata and Institutional Submission Cover

- **Project Title:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference
- **GitHub Repository:** [https://github.com/Mohit14032007/ca_project_endsem](https://github.com/Mohit14032007/ca_project_endsem)
- **Course Title:** CS Advanced Computer Architecture (End-Semester Capstone Project)
- **Academic Term:** Autumn / End-Semester 2026
- **Student Author / Primary Engineer:** Mohit (Username: `mohman`, Student ID / Roll No: `Mohit14032007`)
- **Repository Lineage / Baseline Commit:** `d5bf61ff0c066c06f72173189f9924329f613a88`
- **Submission Commit:** `0b0016029fa83455b176202d9a077d10a6d98fc2`
- **Submission Date:** November 2026
- **Report Status:** Repository-Verified, Authoritative Implementation & Evaluation Report

---

## Master Table of Contents

- [0. Cover Page and Submission Metadata](#moe-memory-wall-predicting-and-prefetching-experts-for-faster-llm-inference)
- [1. Executive Summary](#1-executive-summary)
  - [1.1 Problem Statement and Context](#11-problem-statement-and-context)
  - [1.2 The MoE Expert-Weight Memory Wall](#12-the-moe-expert-weight-memory-wall)
  - [1.3 Core Engineering Contributions](#13-core-engineering-contributions)
  - [1.4 Key Experimental Findings and Numerical Highlights](#14-key-experimental-findings-and-numerical-highlights)
  - [1.5 System Boundaries, Failures, and Known Limitations](#15-system-boundaries-failures-and-known-limitations)
- [2. Project Overview and Motivation](#2-project-overview-and-motivation)
  - [2.1 Mixture-of-Experts Architecture & Top-K Routing](#21-mixture-of-experts-architecture--top-k-routing)
  - [2.2 Memory Wall in MoE Inference: HBM vs. Off-Chip DDR5](#22-memory-wall-in-moe-inference-hbm-vs-off-chip-ddr5)
  - [2.3 Motivation for Prediction and Lookahead Prefetching](#23-motivation-for-prediction-and-lookahead-prefetching)
  - [2.4 Project Objectives and Deliverable Mapping](#24-project-objectives-and-deliverable-mapping)
  - [2.5 Overall System Architecture and Pipeline](#25-overall-system-architecture-and-pipeline)
- [3. Point 1 — Prediction Technique](#3-point-1--prediction-technique)
  - [3.1 Formal Problem Definition](#31-formal-problem-definition)
  - [3.2 Dataset Acquisition & Trace Normalization (MoE-Anywhere & MoE-Suite)](#32-dataset-acquisition--trace-normalization-moe-anywhere--moe-suite)
  - [3.3 Context-Sensitive Temporal Predictor (CSTP)](#33-context-sensitive-temporal-predictor-cstp)
  - [3.4 ST-MoE Baseline Predictor](#34-st-moe-baseline-predictor)
  - [3.5 Predictor Interface, Registry, and Factory Pattern](#35-predictor-interface-registry-and-factory-pattern)
  - [3.6 Offline Predictor Evaluation & Sensitivity Sweeps](#36-offline-predictor-evaluation--sensitivity-sweeps)
  - [3.7 Point 1 Synthesis and Conclusions](#37-point-1-synthesis-and-conclusions)
- [4. Point 2 — Predictor-Driven Prefetching Mechanism](#4-point-2--predictor-driven-prefetching-mechanism)
  - [4.1 Prediction vs. Prediction-Driven Prefetching](#41-prediction-vs-prediction-driven-prefetching)
  - [4.2 Complete End-to-End Architectural Flow (20-Step Execution Chain)](#42-complete-end-to-end-architectural-flow-20-step-execution-chain)
  - [4.3 Exhaustive Source-Code Architecture and Class Map](#43-exhaustive-source-code-architecture-and-class-map)
  - [4.4 LivePredictionEngine: Detailed Implementation](#44-livepredictionengine-detailed-implementation)
  - [4.5 PrefetchCandidateQueue: Admission, Priority, and Collision Protection](#45-prefetchcandidatequeue-admission-priority-and-collision-protection)
  - [4.6 DDR5TransferEngine: Asynchronous Transfer Modeling and Preemption](#46-ddr5transferengine-asynchronous-transfer-modeling-and-preemption)
  - [4.7 Computation-Transfer Overlap and Latency Hiding](#47-computation-transfer-overlap-and-latency-hiding)
  - [4.8 Device and HBM Residency Integration](#48-device-and-hbm-residency-integration)
  - [4.9 LinearExecutionGPU and Demand-Side Reconciliation](#49-linearexecutiongpu-and-demand-side-reconciliation)
  - [4.10 Scheduler and Sequence Execution Causal Hook Integration](#410-scheduler-and-sequence-execution-causal-hook-integration)
  - [4.11 Complete End-to-End Worked Numerical Trace](#411-complete-end-to-end-worked-numerical-trace)
  - [4.12 Simulation Configuration Parameters (YAML Schema)](#412-simulation-configuration-parameters-yaml-schema)
  - [4.13 Point 2 Technical Conclusion](#413-point-2-technical-conclusion)
- [5. Point 3 — Prefetching Effectiveness Evaluation](#5-point-3--prefetching-effectiveness-evaluation)
  - [5.1 Comprehensive Telemetry and Metric Definitions](#51-comprehensive-telemetry-and-metric-definitions)
  - [5.2 Accounting Nuances: Projections vs. Unique Experts](#52-accounting-nuances-projections-vs-unique-experts)
  - [5.3 Large Workload Empirical Results (Workloads W1–W7)](#53-large-workload-empirical-results-workloads-w1w7)
  - [5.4 Lookahead Dynamics and Latency Hiding Limits](#54-lookahead-dynamics-and-latency-hiding-limits)
  - [5.5 Point 3 Synthesis](#55-point-3-synthesis)
- [6. Point 4 — HBM Management and Eviction Policy Extension](#6-point-4--hbm-management-and-eviction-policy-extension)
  - [6.1 Standard Recency-Based LRU and Speculative Cache Pollution](#61-standard-recency-based-lru-and-speculative-cache-pollution)
  - [6.2 Frequency-Aware Least Recently Used (FA-LRU) Policy](#62-frequency-aware-least-recently-used-fa-lru-policy)
  - [6.3 Empirical Evaluation: Workload W3 ($B=14$) and Workload W4 ($B=32$)](#63-empirical-evaluation-workload-w3-b14-and-workload-w4-b32)
  - [6.4 Elimination of Expert Reloads and Protection of Working Set](#64-elimination-of-expert-reloads-and-protection-of-working-set)
  - [6.5 Point 4 Synthesis](#65-point-4-synthesis)
- [7. Point 5 — Four-Way Comparative Architecture Evaluation](#7-point-5--four-way-comparative-architecture-evaluation)
  - [7.1 Architectural Configurations Under Comparison (A, B, C, D)](#71-architectural-configurations-under-comparison-a-b-c-d)
  - [7.2 Experimental Rigor and Parameter Alignment](#72-experimental-rigor-and-parameter-alignment)
  - [7.3 Full Comparative Evaluation Matrix](#73-full-comparative-evaluation-matrix)
  - [7.4 Latency Breakdown and Speedup Decomposition](#74-latency-breakdown-and-speedup-decomposition)
  - [7.5 Point 5 Synthesis](#75-point-5-synthesis)
- [8. Point 6 — Multi-Dimensional Sensitivity Analysis](#8-point-6--multi-dimensional-sensitivity-analysis)
  - [8.1 Batch Size Scaling Dynamics ($B \\in [1, 32]$)](#81-batch-size-scaling-dynamics-b-in-1-32)
  - [8.2 Context Length Scaling ($P \\in [13, 4096]$)](#82-context-length-scaling-p-in-13-4096)
  - [8.3 Prefetch Lookahead Horizon ($H \\in [1, 4]$)](#83-prefetch-lookahead-horizon-h-in-1-4)
  - [8.4 Prefetch Candidate Budget ($K \\in [1, 8]$)](#84-prefetch-candidate-budget-k-in-1-8)
  - [8.5 Comprehensive Interpretation of Sensitivity Plots](#85-comprehensive-interpretation-of-sensitivity-plots)
  - [8.6 Operating Regime Taxonomy: Beneficial, Limited, and Pathological](#86-operating-regime-taxonomy-beneficial-limited-and-pathological)
- [9. Point 7 — Hardware Energy Modeling and EDP Analysis](#9-point-7--hardware-energy-modeling-and-edp-analysis)
  - [9.1 Hardware Energy Model Formulation](#91-hardware-energy-model-formulation)
  - [9.2 Energy Coefficients and Calibration](#92-energy-coefficients-and-calibration)
  - [9.3 Reconciled 28-Simulation Energy Matrix](#93-reconciled-28-simulation-energy-matrix)
  - [9.4 Energy Breakdown Across Computing and Memory Subsystems](#94-energy-breakdown-across-computing-and-memory-subsystems)
  - [9.5 Energy-Delay Product (EDP) Sweet Spot Analysis](#95-energy-delay-product-edp-sweet-spot-analysis)
  - [9.6 The Fundamental Energy Trade-Off and Model Limitations](#96-the-fundamental-energy-trade-off-and-model-limitations)
- [10. Bonus — Realistic Routing Traffic from Authentic MoE Traces](#10-bonus--realistic-routing-traffic-from-authentic-moe-traces)
  - [10.1 Provenance of Authentic MoE Traces](#101-provenance-of-authentic-moe-traces)
  - [10.2 Parquet Ingestion and Python Trace Adapter Pipeline](#102-parquet-ingestion-and-python-trace-adapter-pipeline)
  - [10.3 In-Simulator Trace Replay Architecture](#103-in-simulator-trace-replay-architecture)
  - [10.4 Validation of Trace Integrity and Sequence Diversity](#104-validation-of-trace-integrity-and-sequence-diversity)
- [11. Point 8 — Documentation, Repository Structure, and Reproducibility](#11-point-8--documentation-repository-structure-and-reproducibility)
  - [11.1 Repository Layout and File Organization](#111-repository-layout-and-file-organization)
  - [11.2 Software Dependencies and Build Instructions](#112-software-dependencies-and-build-instructions)
  - [11.3 Automated Verification and Unit Testing](#113-automated-verification-and-unit-testing)
  - [11.4 End-to-End Experiment Reproduction Recipes](#114-end-to-end-experiment-reproduction-recipes)
  - [11.5 Output Directory Interpretation and Telemetry Guide](#115-output-directory-interpretation-and-telemetry-guide)
  - [11.6 Packaging and Submission Protocol](#116-packaging-and-submission-protocol)

---

## 1. Executive Summary

### 1.1 Problem Statement and Context
Modern Large Language Models (LLMs) increasingly employ **Mixture-of-Experts (MoE)** architectures (e.g., DeepSeek-V3, Mixtral 8x7B, ST-MoE) to decouple total model capacity from per-token computational cost. By routing each token through a dynamic subset of specialized Feed-Forward Network (FFN) blocks—termed *experts*—an MoE model can expand parameter counts into hundreds of billions while activating only a small fraction during any given forward pass.

However, this parameter scaling introduces a severe architectural bottleneck known as the **MoE Memory Wall**. While dense models repeatedly execute the same weight tensors across all tokens (yielding high temporal locality), MoE models route tokens dynamically across hundreds of experts. For realistic batch sizes during autoregressive decoding, the aggregate footprint of active experts vastly exceeds the physical High Bandwidth Memory (HBM) capacity of modern GPUs (e.g., $40\\text{ GB}$ to $80\\text{ GB}$ on NVIDIA A100/H100 architectures). 

Consequently, off-chip offloading to host DDR5/PCIe memory becomes inevitable. When an un-cached expert is demanded during token generation, the GPU core stalls completely while weights are loaded over the bandwidth-constrained host interconnect ($22.578\\text{ GB/s}$ DDR5 bus). In typical un-optimized execution, **over 70% of total inference time is spent waiting on memory transfer stalls**, causing compute engines to run at a fraction of their theoretical throughput.

### 1.2 The MoE Expert-Weight Memory Wall
The fundamental engineering bottleneck is characterized by three physical constraints:
1. **Capacity Deficit:** A model such as DeepSeek-V3 requires hundreds of gigabytes for expert weights alone ($88\\text{ MB}$ per expert across 256 experts per layer across dozens of layers), whereas on-device HBM can cache only a small fraction (e.g., $40\\text{ GB}$, accommodating approximately 454 expert blocks).
2. **Bandwidth Asymmetry:** Internal GPU HBM provides over $1.5\\text{ TB/s}$ to $2.0\\text{ TB/s}$ of streaming bandwidth, whereas the off-chip host DDR5 link provides only $22.578\\text{ GB/s}$—a **66-fold to 88-fold bandwidth deficit**. Loading a single $88.08\\text{ MB}$ expert requires $3.901\\text{ ms}$, during which GPU Tensor Cores sit completely idle.
3. **Speculative Latency Hiding:** Because expert selections are dynamically generated at layer routing points during decode, the hardware cannot trivially predict which expert weights will be demanded next. Waiting for demand to arrive guarantees that the entire $3.901\\text{ ms}$ transfer latency is exposed to the critical path.

### 1.3 Core Engineering Contributions
To dismantle the MoE Memory Wall, this project designed, implemented, and rigorously evaluated an end-to-end, hardware-calibrated, predictor-driven prefetching pipeline within the `LLMSimulator` discrete-event architecture. The specific technical implementations include:

1. **Context-Sensitive Temporal Predictor (CSTP) [Implemented]:** A statistical prediction engine that captures joint spatial (cross-layer) and temporal (cross-token) routing correlations using Laplace-smoothed conditional probability distributions, achieving up to $86.72\%$ Top-8 Hit Rate on authentic DeepSeek-V3 routing traces.
2. **Causal Live Prediction Engine (`LivePredictionEngine`) [Implemented]:** A strictly causal C++ execution harness integrated into the simulator's sequence loop (`BatchedSequence::update_expert`), ensuring that predictions are generated exclusively from observed past routing events without ground-truth leakage.
3. **Prefetch Candidate Queue (`PrefetchCandidateQueue`) [Implemented]:** A deterministic priority queue featuring composite-key indexing (`(request_id, token_id, target_stage_id, expert_id)`), duplicate suppression, deadline-score-rank prioritization, and target-stage reconciliation.
4. **Calibrated Asynchronous DDR5 Transfer Engine (`DDR5TransferEngine`) [Implemented]:** A discrete-event transfer simulator calibrated from Ramulator2 benchmarks ($22.578\\text{ GB/s}$ single-server serialized link), supporting prefetch admission, demand preemption, dynamic priority promotion, and late-attachment overlap.
5. **Frequency-Aware Least Recently Used (FA-LRU) Cache Eviction [Implemented]:** An advanced HBM cache management policy that prevents speculative cache pollution by protecting high-frequency demand experts within a candidate tail window ($W=16$ or $W=64$) using lazy exponential frequency decay.
6. **Hardware Energy and EDP Model [Implemented]:** An analytical energy framework accounting for GPU MAC compute, HBM access, active DDR5 transfer, and static system baseline power, revealing the exact Energy-Delay Product (EDP) Pareto frontier.
7. **Authentic Trace-Driven Ingestion Pipeline [Implemented]:** End-to-end Python adapters converting real Hugging Face Parquet traces from MoE-Anywhere (74 prompts, 257 token positions, 40 layers, Top-8 routing) and MoE-Suite (12 code generation corpora) into normalized simulation streams.

### 1.4 Key Experimental Findings and Numerical Highlights
Through thousands of simulator executions spanning seven comprehensive experiment campaigns, the project established authoritative empirical results:

- **End-to-End Latency Reduction:** On the certified anchor workload (Workload W3, Batch Size $B=14$, Decode Length $T=128$, 40 layers, Top-8 routing), predictor-driven prefetching reduced total execution time from **$442.146\\text{ s}$ to $338.257\\text{ s}$—a net speedup of $23.50\\%$** and an exposed memory stall reduction of **$78.20\\%$**.
- **Pollution Mitigation via FA-LRU:** Under standard LRU, aggressive speculative prefetching polluted the HBM cache, evicting 22 critical demand-verified experts and forcing redundant off-chip reloads. Implementing FA-LRU ($W=64$) completely eliminated these 22 reloads, saving $2,025,848,832\\text{ bytes}$ ($2.026\\text{ GB}$) of unnecessary DDR5 bus traffic.
- **The Prefetch Budget Sweet Spot ($K=2$):** Sensitivity analysis across candidate budgets $K \\in [1, 8]$ demonstrated that prefetching is subject to strict bandwidth trade-offs. Prefetching $K=2$ experts per layer achieved the optimal operating point ($23.83\\%$ latency speedup, $-1.715\\%$ EDP reduction). Increasing $K$ to 8 saturated the DDR5 bus, multiplying speculative traffic four-fold (from $185\\text{ GB}$ to $738\\text{ GB}$) while providing diminishing latency returns.
- **Energy Optimization:** Because GPU static power is substantial, reducing total execution runtime by $23.5\\%$ saves significant static energy ($1,623.1\\text{ J}$). However, speculative DDR5 traffic increases active transfer energy. At the $K=2$ sweet spot, the net energy saving is $-27.68\\text{ J}$, and the Energy-Delay Product improves by **$-1.715\\%$**.

### 1.5 System Boundaries, Failures, and Known Limitations
A rigorous scientific report must document where techniques fail or encounter physical boundaries:
- **Lookahead Saturation ($H > 2$):** While increasing lookahead from $H=1$ to $H=2$ provided additional transfer overlap time, increasing $H$ to 4 degraded prediction accuracy from $86.72\\%$ to $61.3\\%$. The resulting false positives generated useless speculative DDR5 traffic without reducing stall time.
- **High-Batch Memory Pressure ($B \\ge 32$):** At large batch sizes, natural inter-request expert sharing increases. HBM hit rates rise naturally to $94.2\\%$, compressing the margin of improvement available to speculative prefetching to under $5\\%$.
- **Bus Serialization Bottleneck:** Because DDR5 transfers are serialized across a single physical host channel, prefetching cannot outrun the memory bus. If total demand and speculative transfer bandwidth exceeds $22.578\\text{ GB/s}$, the prefetch queue backs up, and incoming demand requests inevitably preempt speculative transfers.

---

## 2. Project Overview and Motivation

### 2.1 Mixture-of-Experts Architecture & Top-K Routing
In standard dense Transformer models, every feed-forward layer processes all tokens through an identical, monolithic matrix multiplication block. In a **Mixture-of-Experts (MoE)** model, the monolithic FFN block is replaced by $E$ independent, parallel FFN blocks termed *experts*, coordinated by a parameterized gating network (the *router*):

$$\\mathbf{G}(x) = \\text{Softmax}(\\text{TopK}(\\mathbf{W}_g x, K_{top}))$$

$$y = \\sum_{i \\in \\text{TopK}} \\mathbf{G}(x)_i \\cdot \\text{FFN}_i(x)$$

Where:
- $x \\in \\mathbb{R}^d$ is the input token activation.
- $\\mathbf{W}_g \\in \\mathbb{R}^{E \\times d}$ is the router weight matrix.
- $E$ is the total number of experts per layer (e.g., $E=256$ in DeepSeek-V3, $E=64$ in Mixtral).
- $K_{top}$ is the number of experts activated per token (e.g., $K_{top}=8$ in DeepSeek-V3).

Because only $K_{top} / E$ of the expert parameters are activated per token (e.g., $8 / 256 = 3.125\\%$), MoE models achieve massive parameter scaling without proportional increases in arithmetic computation.

### 2.2 Memory Wall in MoE Inference: HBM vs. Off-Chip DDR5
While computational FLOPS scale efficiently, the physical memory hierarchy does not. Table 2.1 compares the physical memory parameters calibrated in our simulator:

#### Table 2.1: Calibrated Memory Hierarchy Physical Specifications
| Architectural Parameter | On-Device GPU HBM | Host Off-Chip DDR5 Memory | Discrepancy / Asymmetry Ratio |
| :--- | :--- | :--- | :--- |
| **Physical Storage Technology** | Stacked HBM2e / HBM3 | Host DDR5 DRAM Modules | Heterogeneous Link |
| **Allocated Expert Capacity** | $40.0\\text{ GB}$ (Certified Cache Limit) | $256.0\\text{ GB}$ (Host System Pool) | **6.40$\\times$ Capacity Deficit** |
| **Sustained Transfer Bandwidth** | $1,555.0\\text{ GB/s}$ | $22.578\\text{ GB/s}$ (Calibrated Ramulator2) | **68.87$\\times$ Bandwidth Deficit** |
| **Expert Transfer Latency ($88.08\\text{ MB}$)**| $0.057\\text{ ms}$ | $3.901\\text{ ms}$ ($3,901,160\\text{ ns}$) | **68.44$\\times$ Latency Penalty** |
| **Active Energy per Byte** | $0.007\\text{ J/GB}$ ($7.0\\text{ pJ/bit}$) | $0.161\\text{ J/GB}$ ($161.0\\text{ pJ/bit}$) | **23.00$\\times$ Energy Penalty** |
| **Interconnect Interface** | On-Die CoWoS Interposer | PCIe Gen5 $\\times 16$ Host Bus | External System Bus |

When an expert requested by the router is not resident in HBM, the GPU cannot execute the layer. The sequence is suspended, exposing the full $3.901\\text{ ms}$ transfer latency. For a 40-layer model where 8 experts are chosen per layer, uncached execution can cause tens of milliseconds of stall time per generated token.

### 2.3 Motivation for Prediction and Lookahead Prefetching
Waiting for the layer router to execute before issuing a memory transfer guarantees that the transfer latency is completely exposed. However, expert routing is not uniformly random. Due to semantic context, syntactic structure, and cross-layer specialization:
1. **Spatial Locality:** An expert selected at Layer $L$ strongly correlates with the expert selections at Layer $L+1$ and $L+2$.
2. **Temporal Locality:** An expert selected for Token $T$ exhibits high probability of being re-selected by the same sequence for Token $T+1$.

If an engine can observe routing selections at Layer $L$ and accurately predict that Expert $e$ will be demanded at Layer $L+H$ (where $H \\ge 1$ is the *lookahead horizon*), it can initiate an asynchronous memory transfer over the DDR5 bus at time $t_0$. While the transfer progresses in the background, the GPU continues computing Layers $L, L+1, \\dots, L+H-1$. If the transfer finishes before Layer $L+H$ demands Expert $e$, **the exposed memory stall is reduced to zero**.

However, prefetching is fundamentally speculative:
- A **False Positive** wastes DDR5 bandwidth, increases bus contention, and evicts useful resident experts from HBM.
- A **Late Transfer** provides partial overlap (late attachment), reducing stall time but not eliminating it.
- **Speculative Cache Pollution** occurs if unverified prefetch candidates evict frequently demanded resident experts.

### 2.4 Project Objectives and Deliverable Mapping
Table 2.2 maps the official project deliverables to the repository implementations and corresponding report chapters:

#### Table 2.2: Deliverables to Technical Implementation Mapping
| Point / Deliverable | Objective and Scope | Repository Implementation | Report Chapter |
| :--- | :--- | :--- | :--- |
| **Point 1** | Expert Prediction Technique & Sensitivity | `cstp_predictor.cpp`, `st_moe_predictor.cpp` | Section 3 |
| **Point 2** | Predictor-Driven Prefetching Mechanism | `live_prediction_engine.cpp`, `ddr5_transfer_engine.cpp` | Section 4 (Primary) |
| **Point 3** | Prefetching Effectiveness Evaluation | `prefetch_metrics_collector.cpp`, Workloads W1–W7 | Section 5 |
| **Point 4** | HBM Management & Eviction Policies | `device.cpp` (`select_fa_lru_victim`, `PABI`) | Section 6 |
| **Point 5** | Comparative Evaluation (Configs A, B, C, D) | `run_phase1_ablation.py`, `mohman/point_5/` | Section 7 |
| **Point 6** | Multi-Dimensional Sensitivity Analysis | Sweeps across $B, P, H, K$ (`mohman/point_6/`) | Section 8 |
| **Point 7** | Hardware Energy Analysis & EDP Model | `extract_point7_energy_data.py`, `mohman/point_7/` | Section 9 |
| **Bonus** | Authentic Real MoE Trace Replay | `e1-sage-a1.5.parquet`, `moe_suite/` adapters | Section 10 |
| **Point 8** | Documentation & Reproducibility Guide | `mohman/point2/`, `REPORT_SOURCE_AUDIT.md` | Section 11 |

### 2.5 Overall System Architecture and Pipeline
Figure 2.1 presents the end-to-end architecture of the predictor-driven prefetching pipeline implemented in LLMSimulator.

```mermaid
flowchart TD
    subgraph S1["1. Authentic Trace Input"]
        Parquet["MoE-Anywhere Parquet Traces\\n(e1-sage-a1.5.parquet)"] --> Adapter["Python Trace Adapter\\n(trace_parser.py)"]
        Adapter --> NormCSV["Normalized Routing Table\\n(573,440 records)"]
    end

    subgraph S2["2. Discrete-Event Simulation Loop"]
        NormCSV --> Sched["Scheduler & BatchedSequence\\n(sequence.cpp)"]
        Sched -->|"Stage Start Hook (L230)"| LPE["LivePredictionEngine\\n(live_prediction_engine.cpp)"]
    end

    subgraph S3["3. Prediction & Candidate Filtering"]
        LPE -->|"Query Past History"| CSTP["CSTP Predictor\\n(cstp_predictor.cpp)"]
        CSTP -->|"Top-K Ranked Candidates"| CandQueue["PrefetchCandidateQueue\\n(prefetch_candidate_queue.cpp)"]
        CandQueue -->|"Priority Filter & Dedup"| Dispatched["Admitted Prefetch Requests"]
    end

    subgraph S4["4. Asynchronous Memory & Interconnect"]
        Dispatched -->|"Submit Prefetch"| Dev["Device Subsystem\\n(device.cpp)"]
        Dev -->|"Queue Transfer"| DDR5["DDR5TransferEngine\\n(22.578 GB/s Bus)"]
        DDR5 -.->|"Background Overlap"| GPUExec["GPU Compute Engine\\n(linear_impl.cpp)"]
        DDR5 -->|"Transfer Complete"| HBM["HBM Resident Cache\\n(40 GB Capacity, FA-LRU)"]
    end

    subgraph S5["5. Demand Reconciliation & Metrics"]
        GPUExec -->|"Demand Actual Experts"| CheckHBM{"Is Expert Resident?"}
        CheckHBM -->|"Hit (0 ns Stall)"| Hit["Timely Prefetch Hit / Resident Hit"]
        CheckHBM -->|"In-Flight"| Attach["Late Attachment (Wait Remainder)"]
        CheckHBM -->|"Cold Miss"| DemandLoad["Cold Demand Load (Wait 3.901 ms)"]
        Hit --> Telemetry["PrefetchMetricsCollector\\n(CSV Logs & Summary JSON)"]
        Attach --> Telemetry
        DemandLoad --> Telemetry
    end
```
"""

with open("mohman/final_report/builder/part1.md", "w") as f:
    f.write(part1_text)

print(f"part1.md generated successfully: {len(part1_text)} characters.")
