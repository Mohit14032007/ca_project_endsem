# Phase C: Threats to Validity, Limitations & Scope Boundaries

## 1. Threats to Validity

### A. Internal Validity (Measurement & Causal Rigor)
To ensure internal validity and eliminate experimental artifacts:
1. **Strict Temporal Causality**: Predictions are generated strictly prior to target stage execution. Ground-truth routing labels are never revealed to the predictor or passed to its state update until after the prediction set has been logged.
2. **Zero Split Leakage**: Unit-level partitioning (by request ID for Duplex and Synthetic, by prompt ID for MoE Anywhere, and by task corpus for MoE Suite) guarantees that test evaluations never encounter previously profiled sequence instances.
3. **State Isolation**: Predictors are deep-copied per budget sweep and re-instantiated per regime, preventing cross-regime memory pollution.
4. **Code Quality**: Strict zero-ternary policy enforced via automated AST parsers on all test and analysis scripts.

### B. External Validity (Trace Generalizability)
Our workload suite includes two production LLM routing datasets (`moe_anywhere` and `moe_suite`), one hardware simulator trace (`duplex_a100`), and two controlled synthetic benchmarks. While these span 8 to 256 experts and 32 to 75 layers:
- **Batching & Concurrency**: The evaluated traces reflect sequential token-by-token generation for single-request or isolated-stream contexts. Highly batched multi-tenant production engines (e.g., continuous batching with 64–256 concurrent requests) may exhibit different temporal locality dynamics.
- **Routing Algorithm Variations**: The evaluated production models employ Top-$K$ gating with load-balancing losses. Alternative architectures, such as Soft MoE or expert-choice routing, would require adapter modifications.

### C. Construct Validity (Evaluation Metrics)
We employ Macro Recall@B, Macro Precision@B, F1@B, and Hit Rate as primary proxies for prefetching efficacy:
- **Recall@B** represents the fraction of required experts that would already reside in GPU HBM cache when the stage executes.
- **Precision@B** measures prefetch bandwidth efficiency, identifying the fraction of prefetched bytes that were actually utilized.
- **Limitation**: Recall alone does not model DRAM bus contention or queueing delay in the memory controller.

---

## 2. Explicit Scope Boundaries (Phase C vs Future Phases)

### What Phase C Establishes:
1. Validates the mathematical and empirical accuracy of expert prediction across 5 workloads.
2. Proves that online adaptation is essential to prevent catastrophic collapse under distribution drift.
3. Accurately micro-benchmarks the query latency, update latency, and memory footprint of CSTP vs. ST-MoE.
4. Demonstrates the viability of a CSTP+ST-MoE hybrid.

### What Phase C Does NOT Claim:
1. **No End-to-End Latency Speedup**: Phase C does **not** implement asynchronous off-chip DRAM prefetching in the C++ simulator engine. We make no claims regarding wall-clock inference speedup or token generation time reductions.
2. **No Hardware Controller Simulation**: Memory controller arbitration, PCIe/NVLink saturation, and DDR5 bank conflicts during concurrent prefetch and execution are outside the scope of Phase C.

---

## 3. Honest Algorithmic Assessment & Negative Findings

In the interest of scientific rigor, we explicitly acknowledge where CSTP fell short of initial expectations:

1. **Computational & Memory Overhead**:
   - CSTP consumes **58 MB** of memory for DeepSeek-V3, compared to just **1.4 MB** for ST-MoE Paper (a 41x disparity).
   - CSTP's query latency is **115.6 $\mu\text{s}$**, approximately 3.0x slower than ST-MoE's **39.0 $\mu\text{s}$**.
2. **Inertia in Cross-Task Shift**:
   - On disjoint tasks (`moe_suite`), ST-MoE's bounded 2-bit counters allowed it to adapt significantly faster (+45.9% online gain) than CSTP (+0.7% gain). CSTP's unbounded float64 count matrices act as high-inertia anchors that slow adaptation across radical task shifts.
3. **Where CSTP Proves Superior**:
   - CSTP provides continuous, smooth Bayesian posterior probabilities that enable fine-grained thresholding, deterministic tie-breaking, and principled temperature scaling.
   - When ensembled with ST-MoE (`cstp_st_moe_hybrid`), it achieves the highest overall recall across all workloads.
