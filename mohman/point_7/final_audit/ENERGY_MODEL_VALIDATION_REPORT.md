# Energy Model Validation Report: Physical Validity, Boundary Assessment, and Modeling Assumptions

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 7 — Energy Analysis  
**Document:** Rigorous Physical Validation and Academic Boundary Assessment  
**Author:** Mohman / Advanced Agentic Coding Assistant  
**Date:** November 11, 2026  

---

## 1. Executive Summary

A critical standard of academic computer architecture research is rigorous distinction between what is physically measured, what is simulated via microarchitectural event counters, and what is analytically estimated via post-hoc models. This document formally validates the physical integrity, scope, and limitations of the Point 7 Energy Analysis.

### Official Evaluation Verdict:
The Point 7 Energy Model is classified as **SCIENTIFICALLY DEFENSIBLE UNDER EXPLICIT ANALYTICAL EXTENSIONS**.

The on-chip GPU HBM2e memory access and Tensor Core MAC compute energies are cycle-synchronized and tracked via microarchitectural event counters. The off-chip host DDR5 bus active transfer energy ($0.1\text{ nJ/byte}$) and static background DRAM power ($4.5\text{ W}$) are analytical extensions grounded in project design specifications (`PHASE2D_EXECUTION_RECOMMENDATION.md` and `MASTER_RESEARCH_EXPERIMENT_PLAN.md`).

---

## 2. Granular Evaluation of Energy Model Components

### 2.1. Off-Chip DDR5 Active Transfer Energy Model ($k_{\text{DDR5}} = 0.1\text{ nJ/byte}$)

1. **Origin and Provenance:** Formally established in `docs/final_campaign/PHASE2D_EXECUTION_RECOMMENDATION.md` Section 2.1 Condition 3 as the post-hoc accounting standard:
   $$E_{\text{DDR5}} = \text{Bytes} \times 8\text{ bits/byte} \times 12.5\text{ pJ/bit} = \text{Bytes} \times 100\text{ pJ/byte} = \text{Bytes} \times 0.1\text{ nJ/byte}$$
2. **Physical Scope:** This coefficient models the physical I/O signaling energy of the DDR5 bus interface, including host memory controller PHY drivers, transmission line parasitics, on-die termination (ODT), and receiver circuits operating at DDR5-4800 / DDR5-5600 signaling rates.
3. **Subsystem Level:** It is an analytical bus/PHY transfer energy model, not a complete DRAM array power simulator. It does not dynamically model internal DDR5 bank activations, precharge cycles, or bank conflict states.
4. **Demand vs. Speculative Uniformity:** The coefficient applies uniformly ($0.1\text{ nJ/byte}$) to both demand and speculative transfers because both travel across the same physical DDR5 bus channels.
5. **Transfer Counter Accuracy:** Byte counts are tracked directly by the `PrefetchMetricsCollector` and `Device` transfer queues. Only completed transfers are charged energy; canceled or inflight requests that did not finish before termination are excluded.

### 2.2. Static DRAM Background Power Model ($P_{\text{static}} = 4.5\text{ W}$)

1. **Origin and Provenance:** Formally established in `docs/phase_e/MASTER_RESEARCH_EXPERIMENT_PLAN.md` Section 9.1:
   > *"Background static power: $4.5\text{ W}$ per dual-rank DIMM."*
2. **Physical Scope:** Models continuous background standby and auto-refresh power of a server dual-rank registered DIMM channel. Dual-rank DDR5 DIMMs draw steady-state standby power ($I_{2N}, I_{3N}$) to maintain register buffer state and DRAM capacitor charge.
3. **Execution Interval Grounding:** Static energy is calculated as:
   $$E_{\text{static}} = P_{\text{static}} \times t_{\text{sim}}$$
   Where $t_{\text{sim}}$ is the total simulated device execution time. Static energy savings occur strictly when simulated execution time decreases, avoiding the common error of conflating exposed memory stall reduction with total execution duration reduction.

### 2.3. On-Chip GPU HBM2e Memory Model ($k_{\text{ACT}} = 0.909\text{ nJ}$, $k_{\text{RD/WR}} = 3.48\text{ pJ/bit}$)

1. **Implementation:** Fully native in C++ simulator (`LLMSimulator/src/dram/power.h` and `src/hardware/device.cpp`).
2. **Provenance:** Keckler et al., *"Fine-Grained DRAM Architecture for High-Bandwidth Memory,"* IEEE/ACM MICRO 2017.
3. **Microarchitectural Grounding:** Memory requests are broken into column accesses and pseudo-channel commands. Row activations and column read/write bursts are tracked by cycle-accurate event counters.
4. **Physical Scope:** Accurately accounts for local GPU HBM2e DRAM bank operational energy, column decoders, TSVs, and silicon interposer I/O.

### 2.4. GPU MAC Compute Model ($k_{\text{MAC}} = 0.23\text{ pJ/FLOP}$)

1. **Implementation:** Fully native in C++ (`src/module/module_graph.cpp`).
2. **Provenance:** Standard Tensor Core dynamic energy modeling ($0.46\text{ pJ}$ per MAC = $0.23\text{ pJ/FLOP}$).
3. **Scope and Limitation:** Models dynamic switching energy of matrix multiply-accumulate arithmetic units. It does **not** model GPU streaming multiprocessor (SM) static idle leakage, instruction cache fetches, register file leakage, or cooling fan power.

---

## 3. Workload and Architectural Scope Limitations

1. **Decode-Phase Focus vs. Prefill Phase:**
   The evaluation traces originate from authentic MoE-Anywhere token generation recordings. The sensitivity sweep across context lengths ($P \in \{128, 512, 2048, 4096\}$) evaluates decoding-phase attention KV cache overhead and compute scaling while replaying identical generation routing traces. It does not simulate the initial prompt prefill phase.
2. **Dedicated Expert-Cache Partitioning:**
   In the modeled hardware configuration, the GPU HBM expert cache (35 GiB) operates as a dedicated physical memory partition, strictly decoupled from the KV cache partition. Consequently, increasing $P$ increases attention compute and attention HBM memory traffic, but does not evict expert weights from the cache.
3. **Workload Specificity:**
   All 28 experiments evaluate DeepSeek-V3 / Mixtral-style top-8 routing over 40 layers. While the mathematical relationships (linear bus energy scaling, concurrency amortization, conservative budget sweet spots) are architecturally universal, the exact optimal numerical budget ($K=2$) is calibrated to this model's routing entropy and cache capacity.

---

## 4. Scientific Terminology Standards

To maintain academic integrity in published deliverables:
- **DO NOT USE:** "Physically measured GPU power," "Silicon-validated energy," or "Cycle-accurate DDR5 power."
- **ALWAYS USE:** "Modeled system total energy," "Analytical DDR5 bus transfer model," "Cycle-synchronized HBM event counters," and "Simulator-derived operational energy."

