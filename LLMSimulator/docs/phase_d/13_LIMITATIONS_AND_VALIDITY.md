# Phase D: Limitations, Scope Boundaries, and Threats to Validity

**Document:** `docs/phase_d/13_LIMITATIONS_AND_VALIDITY.md`  
**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Repository:** `LLMSimulator` (`/home/mohman/ca_project/LLMSimulator`)  
**Commit:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Author:** ML Systems Research & Reproducibility Team  

---

## 1. Architectural Scope Boundary & Non-Claims

A core principle of scientific integrity in systems research is defining the boundary between what was experimentally proven and what lies beyond project scope:

1. **Prediction Utility vs Hardware Speedup:**
   - **What Phase D Evaluated:** Phase D evaluated algorithmic expert prediction accuracy (Recall, Precision, F1, Hit Rate), online adaptability, distribution shift resilience, host-side compute latency, and memory footprints.
   - **What Phase D Did NOT Evaluate:** Phase D did NOT implement asynchronous DRAM-to-HBM prefetching in the C++ `LLMSimulator` engine, did NOT simulate DDR5 memory bus contention during prefetch transfers, and did NOT measure end-to-end inference wall-clock speedup (tokens/sec).
   - **Explicit Non-Claim:** We make **no claim of hardware-level speedup** or reduced inference wall-clock latency in Phase D. High prediction recall is a *necessary prerequisite* for effective prefetching, but real speedup depends on PCIe/CXL bandwidth, DRAM channel utilization, and overlap windows.

---

## 2. Threats to Internal and External Validity

### 2.1. Trace Memorization in Single-Request Benchmarks
- **Observation:** In `deepseek_random`, the simulator trace contained only 1 generated request spanning 5 tokens. When split into train/test, the split adapter defaulted to $train = test = [1]$.
- **Threat:** Offline profiling on the same request yields artificial 99.7% recall.
- **Mitigation:** In Phase D, we established the Cold-Start regime as the authoritative ground truth for `deepseek_random`. Cold-start evaluation demonstrated that unprofiled models achieve ~3.1% recall, exactly matching uniform random guessing ($8/256 = 0.03125$).

### 2.2. Multi-Horizon Quality Decay
- **Observation:** As the prediction horizon increases from $H=1$ stage ahead to $H=2$ stages ahead (as evaluated on `moe_anywhere` and `moe_suite`), prediction recall degrades by approximately $15\%\text{--}25\%$.
- **Threat:** Prefetching further in advance to hide longer DRAM transfer latencies incurs higher prefetch inaccuracy and memory cache pollution.
- **Mitigation:** ACSH tracks horizon-specific reliability $R(L, H, t)$, allowing dynamic budget expansion ($B > K$) when predicting deeper horizons.

### 2.3. Multi-Task Negative Transfer & Profiling Inertia
- **Observation:** On multi-task benchmarks (`moe_suite`), accumulating millions of routing records across diverse training tasks severely degraded test performance on unseen tasks (Recall dropping from 0.27 to 0.16).
- **Threat:** Relying exclusively on offline-profiled priors hurts generalist LLMs serving multi-turn, multi-topic queries.
- **Mitigation:** ACSH dynamically downweights high-inertia models via autonomous EMA tracking. However, incorporating explicit count decay or rolling window bounds into CSTP's count matrices remains an important area for future optimization.

### 2.4. Static Routing Assumption
- **Observation:** All evaluated traces employed fixed Top-$K$ routing ($K=8$ or $K=2$).
- **Threat:** Emerging MoE architectures (such as DeepSeek-V3 dynamic thresholding or Mixture-of-Depths) dynamically route 0 to $K_{\max}$ experts per token.
- **Mitigation:** Future work will extend the common `BasePredictor` interface to accept dynamic budget requests conditioned on router activation entropy.

