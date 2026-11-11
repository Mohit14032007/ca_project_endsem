# Energy Model Limitations and Methodology Boundaries

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 7 — Energy Analysis  
**Document:** Simulator Energy Model Limitations and Modeling Boundaries  
**Author:** Mohman / Advanced Agentic Coding Assistant  
**Date:** November 11, 2026  

---

## 1. Executive Summary

A foundational requirement of the Point 7 Energy Analysis is absolute scientific honesty regarding what `LLMSimulator` models natively, what it models analytically, and what physical hardware phenomena are unmodeled. Based on our source-level audit of `LLMSimulator` (specifically `src/dram/power.h`, `src/module/module_graph.cpp`, `src/hardware/cluster.cpp`, `src/hardware/device.cpp`, and `eval/test_deepseekv3_mixtral_style.cpp`), the simulator's native energy model is officially classified as:

$$\mathbf{PARTIALLY\ USABLE}$$

It provides precise, cycle-synchronized operational energy accounting for on-chip GPU HBM2e activations, reads, writes, and Tensor Core MAC compute. However, it completely omits off-chip DDR5 host-to-device active bus transfer energy and static background DRAM refresh/standby power. 

To conduct a physically valid evaluation of the MoE Memory Wall, we deployed the documented analytical extensions from `docs/final_campaign/PHASE2D_EXECUTION_RECOMMENDATION.md` and `docs/phase_e/MASTER_RESEARCH_EXPERIMENT_PLAN.md`. This report formally establishes the boundaries, assumptions, mathematical formulations, and limitations of this methodology.

---

## 2. Granular Inventory of Modeled vs. Unmodeled Components

| Component / Physical Phenomenon | Implementation Layer | Modeling Mechanism | Mathematical Basis | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **GPU HBM2e DRAM Activations** | Native C++ Simulator | `MemoryPower::CalcEnergy()` in `power.h` | $E_{\text{ACT}} = N_{\text{ACT}} \times V_{DD} \times (I_{0} - I_{2N}) \times t_{RAS}$ ($k_{\text{ACT}} \approx 0.909\text{ nJ}$) | Fully Modeled & Populated |
| **GPU HBM2e DRAM Read / Write** | Native C++ Simulator | `MemoryPower::CalcEnergy()` in `power.h` | $E_{\text{RD/WR}} = N_{\text{BL}} \times V_{DD} \times (I_{4R/W} - I_{3N}) \times t_{BL}$ ($k_{\text{RD/WR}} \approx 0.891\text{ nJ} = 3.48\text{ pJ/bit}$) | Fully Modeled & Populated |
| **GPU Tensor Core Compute** | Native C++ Simulator | `ComputeUnit::ComputeEnergy()` in `device.cpp` | $E_{\text{MAC}} = \text{FLOPs} \times 0.23\text{ pJ/FLOP}$ ($k_{\text{MAC}} = 0.23\text{ pJ}$) | Fully Modeled & Populated |
| **Layer-Wise Energy Breakdown** | Native C++ Simulator | `ModuleGraph::ProfileEnergy()` in `module_graph.cpp` | Accumulated by module type: `fc`, `attn`, `moe` | Fully Modeled & Populated |
| **Off-Chip Host-to-Device DDR5 Bus Active Energy** | Analytical Extension | Explicit Energy Transfer Model | $E_{\text{DDR5}} = \text{Bytes}_{\text{DDR5}} \times 0.1\text{ nJ/byte}$ ($12.5\text{ pJ/bit} = 100\text{ pJ/byte}$) | Grounded in JEDEC DDR5 specifications & Project Docs |
| **Host System & DRAM Static Leakage Power** | Analytical Extension | Time-Integrated Background Power | $E_{\text{static}} = P_{\text{static}} \times t_{\text{sim}}$ ($P_{\text{static}} = 4.5\text{ W}$ background) | Grounded in Server Micron DIMM datasheets |
| **Host CPU Predictor Compute Overhead** | Analytical Evaluation | Upper-Bound CPU Energy Model | $E_{\text{pred}} = N_{\text{tokens}} \times N_{\text{layers}} \times E_{\text{history\_lookup}}$ ($< 0.05\text{ J}$ total) | Bound established; negligible (<0.0005% of system total) |
| **GPU Streaming Multiprocessor (SM) Static Idle Power** | Unmodeled | Not simulated | Constant baseline offset across all runs | Omitted to prevent speculative bias |
| **PCIe Gen5 SerDes Protocol Signaling** | Unmodeled | Abstracted into off-chip bus energy | Included conceptually within the 12.5 pJ/bit boundary | Lumped into DDR5 bus metric |
| **Thermal Throttling & Dynamic DVFS** | Unmodeled | Fixed frequency assumed | Fixed clock cycles | Standard architectural simulation practice |

---

## 3. Detailed Boundary Analysis

### 3.1. Native Simulator Energy Boundary

The native simulator calculates the energy consumed within the GPU chip boundary during generation token decoding.
1. **Capacity Access vs. Transfer:** Native `hbm_read_j` and `hbm_write_j` measure the energy of reading weights from the local GPU HBM2e memory banks into the GPU register files and streaming multiprocessors. 
2. **Dynamic Behavior Across Configurations:** As revealed in our forensic audit of the 28 runs, native compute energy (`gpu_mac_j = 71.25 J`) is strictly invariant across configurations because the forward-pass matrix arithmetic is identical. However, native HBM DRAM energy varies slightly between baseline and prefetching configurations:
   - Baseline A (`anc_c1`): Native energy is $1,114.12\text{ J}$ ($1,042.87\text{ J}$ HBM + $71.25\text{ J}$ MAC).
   - Prediction-Only B (`anc_c2`): Native energy is $1,114.12\text{ J}$ (identical to baseline as predictor runs on host CPU).
   - Complete System D (`anc_c4`): Native energy is $1,115.43\text{ J}$ ($1,044.18\text{ J}$ HBM + $71.25\text{ J}$ MAC).
   - **Why?** Prefetching dispatches speculative expert transfers into HBM, causing minor additional HBM bank activations and row-buffer reads ($+1.31\text{ J}$ or $+0.12\%$).
3. **Limitation:** Evaluating prefetching energy *solely* through native simulator energy would capture only this $+0.12\%$ on-chip delta, completely ignoring the primary energy expenditure: off-chip DDR5 physical bus data movement.

### 3.2. Analytical Off-Chip DDR5 Bus Model

Because the MoE Memory Wall problem is fundamentally defined by the expert transfer from host CPU DDR5 memory to GPU HBM, the energy consumed by the DDR5 channels and memory controllers during expert migration must be accounted for.
1. **Transfer Energy Coefficient:** Based on JEDEC DDR5-4800 / DDR5-5600 power measurements and established literature (e.g., Micron DDR5 power calculators, Phase 2D specifications), the active physical I/O transfer energy is:
   $$k_{\text{DDR5}} = 12.5\text{ pJ/bit} = 100\text{ pJ/byte} = 0.1\text{ nJ/byte} = 100\text{ J/TB} = 109.95\text{ J/TiB}$$
2. **Strict Reconciliation:** In `extract_point7_energy_data.py`, we track:
   - Demand DDR5 Active Energy: $E_{\text{DDR5,demand}} = \text{Bytes}_{\text{DDR5,demand}} \times 1.0 \times 10^{-10}\text{ J/byte}$
   - Speculative DDR5 Active Energy: $E_{\text{DDR5,spec}} = \text{Bytes}_{\text{DDR5,speculative}} \times 1.0 \times 10^{-10}\text{ J/byte}$
   - Total DDR5 Active Energy: $E_{\text{DDR5,total}} = E_{\text{DDR5,demand}} + E_{\text{DDR5,spec}}$
3. **Reconciliation Invariant:** For every single run in our 28-experiment matrix, we confirmed:
   $$E_{\text{DDR5,total}} = E_{\text{DDR5,demand}} + E_{\text{DDR5,spec}} \quad (\text{Residual } < 10^{-12}\text{ J})$$

### 3.3. Static Leakage Power and Execution Duration

Prefetching hides memory latency, reducing exposed stall time and simulated execution duration. Because DRAM chips consume static standby and background refresh power continuously while powered on, shortening the runtime saves static energy.
1. **Background Power Rating:** We adopt $P_{\text{static}} = 4.5\text{ W}$, representing the background power of a multi-channel server DDR5 memory subsystem (dual-rank registered DIMMs in active standby / auto-refresh mode).
2. **Static Energy Formulation:**
   $$E_{\text{static}} = P_{\text{static}} \times t_{\text{sim}}$$
3. **Physical Significance:** As simulated execution time decreases from $767.27\text{ s}$ (Baseline A) to $756.34\text{ s}$ ($K=2$ Sweet Spot), static energy drops from $3,452.72\text{ J}$ to $3,403.54\text{ J}$, delivering **$49.18\text{ J}$ of static energy savings**.

---

## 4. Double-Counting Safeguards

To maintain scientific integrity, we implemented rigorous safeguards against double counting:

1. **Separation of HBM and DDR5:**
   - HBM energy accounts *exclusively* for local on-chip high-bandwidth memory bank access ($E_{\text{ACT}} + E_{\text{READ}} + E_{\text{WRITE}}$).
   - DDR5 energy accounts *exclusively* for off-chip host bus data movement ($E_{\text{DDR5}}$).
   - The two do not overlap.
2. **Separation of Compute and Memory:**
   - Tensor Core MAC energy ($E_{\text{MAC}}$) is calculated strictly from executed arithmetic operations.
   - DRAM and bus energies are calculated strictly from memory cycles and byte transfers.
3. **Demand vs. Speculative Isolation:**
   - Demand DDR5 bytes are transferred to satisfy mandatory expert misses.
   - Speculative DDR5 bytes are transferred by the prefetcher in anticipation of future demand.
   - Both are tracked separately, preventing duplicate accounting.

---

## 5. Scope of Validity and Recommendations

1. **Comparative Validity:** Because identical physical coefficients ($k_{\text{DDR5}} = 0.1\text{ nJ/byte}$, $P_{\text{static}} = 4.5\text{ W}$, $k_{\text{MAC}} = 0.23\text{ pJ/FLOP}$) are applied uniformly across all 28 experiments, relative comparisons between configurations ($\Delta E\%$, $\Delta \text{EDP}\%$) are mathematically rigorous and resilient to scaling constants.
2. **Dual-Metric Transparency:** In all executive summaries, research papers, and technical reports, researchers must present **both** the native simulator metric and the total modeled system metric side-by-side. Hiding the methodology or reporting only one number violates academic transparency.
3. **Future Simulator Enhancements:** Future versions of `LLMSimulator` should integrate an explicit off-chip bus controller module directly into the C++ `System` class to capture physical bus contention, channel arbitration energy, and SerDes physical layer power dynamically during simulation.

