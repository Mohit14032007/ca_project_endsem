# Point 7 — Phase A: Energy Model Validity & Classification Report

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 7 — Energy Analysis  
**Author:** Antigravity Engineering  
**Date:** November 11, 2026  
**Status:** Authoritative Assessment  

---

## 1. Official Validity Classification

Following a rigorous source-code audit across `LLMSimulator` and all project documentation, the available energy model is classified as:

### **CLASSIFICATION: PARTIALLY USABLE**

---

## 2. Technical Justification & Source-Code Evidence

### 2.1 Why Not "Fully Usable"?
The model cannot be classified as *fully usable* because:
1. **Lack of Native Off-Chip DDR5 Bus PHY Energy:**
   - As documented in `LLMSimulator/docs/final_campaign/PHASE2D_METRIC_VALIDATION.csv` (line 28):
     ```csv
     ddr5_energy_nj,Energy_Consumption,DDR5 bus transfer energy,total_tokens,N/A,N/A,NOT IMPLEMENTED,Simulator lacks physical DDR5 bus pJ/bit model
     ```
   - In `LLMSimulator/src/hardware/linear_impl.cpp` (lines 102–146), when off-chip transfers execute via `use_event_driven_dram = true`, only `weight_status.memory_duration` is updated. Off-chip transfers do not increment DRAM command counters (`act_count`, `read_count`) in `status.act_energy` or `status.read_energy`.
   - Consequently, `Cluster::getTotalEnergy()` captures on-chip GPU HBM DRAM and compute MAC energy, but omits the transmission energy expended across the off-chip DDR5 bus.
2. **Lack of Host CPU Predictor Energy:**
   - Documented in `PHASE2D_METRIC_VALIDATION.csv` (line 29) as `NOT IMPLEMENTED`. Predictors execute on the host CPU and are outside the GPU simulation boundary.
3. **Lack of Time-Dependent Static / Leakage Power:**
   - Native `status.total_energy` accumulates only dynamic energy. Idle time during memory stalls incurs zero energy cost in native simulator telemetry.

### 2.2 Why Not "Not Usable"?
The model cannot be classified as *not usable* because:
1. **Fully Calibrated On-Chip HBM2e Memory Model:**
   - In `LLMSimulator/src/dram/power.h` (lines 10–35), HBM2e DRAM row activation (`kACT_energy_j_ = 0.909 nJ`) and read/write burst energy (`kREAD_energy_j_ = 0.891 nJ`, reflecting $3.48\text{ pJ/bit}$) are fully implemented, calibrated from the published MICRO 2017 research baseline.
   - Scaled accurately across pseudo-channels in `Device::initializeDRAM` (lines 367–384).
2. **Fully Calibrated GPU Tensor Core Compute Model:**
   - Floating-point matrix arithmetic operations are tracked per kernel in `TopModuleGraph::update_status` using $0.23\text{ pJ/FLOP}$ ($0.46\text{ pJ/MAC}$), correctly scaling with model layers and context lengths.
3. **Exact Byte-Level Off-Chip DDR5 Traffic Accounting:**
   - While physical bus transmission energy is not accumulated in C++ `total_energy`, `DDR5TransferEngine` and `PrefetchMetricsCollector` track demand bytes and speculative bytes with exact single-byte precision.
   - This enables an authoritative, documented post-hoc analytical translation using established project coefficients ($12.5\text{ pJ/bit} = 100\text{ pJ/byte}$).

---

## 3. Operational Protocol for Point 7 Deliverables

To maintain strict scientific integrity and avoid fabricating unmodeled phenomena:

1. **Dual-Metric Energy Reporting:**
   - **Native Simulator Energy ($E_{\text{native}}$):** The exact, unmodified output of `Cluster::getTotalEnergy()`, representing on-chip GPU HBM2e DRAM and Tensor Core compute ($E_{\text{HBM}} + E_{\text{comp}}$).
   - **Total Modeled System Energy ($E_{\text{system}}$):** The unified total combining native energy, analytical off-chip DDR5 bus energy ($100\text{ pJ/byte}$), and static background power ($4.5\text{ W} \times t$).
2. **Explicit Labeling:** Every table, figure, and report section must clearly state whether metrics reflect native on-chip energy or extended system energy.
3. **Metric Attribution:** All energy coefficients ($0.23\text{ pJ/FLOP}$, $3.48\text{ pJ/bit}$, $100\text{ pJ/byte}$, $4.5\text{ W}$) are cited directly to their authoritative project documentation and primary literature sources. Zero arbitrary constants are invented.

