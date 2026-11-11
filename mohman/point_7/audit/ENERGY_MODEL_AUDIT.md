# Point 7 — Phase A: Comprehensive Energy and Power Model Audit

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 7 — Energy Analysis  
**Author:** Antigravity Engineering  
**Date:** November 11, 2026  
**Status:** Complete Source-Level Audit  

---

## 1. Executive Summary

This audit performs an exhaustive, line-by-line inspection of the `LLMSimulator` codebase to identify all energy models, power models, hardware coefficients, accounting accumulators, and exported telemetry. The objective is to determine exactly what physical components are modeled, what events trigger energy accounting, what units are used, and whether the model can validly evaluate the energy trade-offs of asynchronous expert prefetching and Frequency-Aware LRU (FA-LRU) cache replacement.

---

## 2. Source-Level Inventory of Energy Components

The repository implements energy tracking across four primary modules:

1. **Hardware Configuration and Energy Coefficients:**  
   `LLMSimulator/src/dram/power.h`  
   Defines `struct DramEnergy` and hardware-specific energy coefficients for GPU HBM2e, Logic PIM, and Bank-Group PIM.
2. **Device-Level Energy Initialization:**  
   `LLMSimulator/src/hardware/device.cpp` (lines 367–384), `src/hardware/device.h`  
   Scales baseline DRAM energy coefficients by pseudo-channel concurrency (`num_pseudo_ch = num_cube * num_channel / 2`).
3. **Module-Graph Energy Accumulation:**  
   `LLMSimulator/src/module/module_graph.cpp` (lines 254–276, 327–333), `src/module/timeboard.h`  
   Accumulates operational energy per tensor kernel (`act_energy`, `read_energy`, `write_energy`, `mac_energy`) from executed commands and FLOPs.
4. **Cluster-Wide Energy Aggregation & Layer Breakdown:**  
   `LLMSimulator/src/hardware/cluster.cpp` (lines 358–368, 960–1113), `src/hardware/stat.h`  
   Aggregates total energy across devices (`Cluster::getTotalEnergy()`) and computes layer-type breakdowns (`FC_DRAM`, `FC_COMP`, `Attn_DRAM`, `Attn_COMP`, `MoE_DRAM`, `MoE_COMP`).
5. **Telemetry Export in Decode Harness:**  
   `LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp` (lines 481–482, 556–564, 624–638)  
   Exports 14 separate energy metrics per decoding step into the summary decode CSV (`deepseekV3_*_decode.csv`).

---

## 3. Systematic 14-Point Architectural Energy Audit

### 1. File Path and Source Locations
- `src/dram/power.h`: Lines 10–46 (Energy structure and hardware presets).
- `src/hardware/device.cpp`: Lines 367–384 (`Device::initializeDRAM`).
- `src/module/module_graph.cpp`: Lines 254–276 (`TopModuleGraph::update_status`), Lines 327–333 (`TopModuleGraph::getDeviceEnergy`).
- `src/hardware/cluster.cpp`: Lines 358–368 (`Cluster::getTotalEnergy`), Lines 964–1101 (`Cluster::setTimeBreakDown`).
- `src/hardware/stat.h`: Lines 53–67 (Stat structure energy fields).
- `eval/test_deepseekv3_mixtral_style.cpp`: Lines 556–564, Lines 624–638 (CSV export loop).

### 2. Physical Components Modeled
- **Modeled:**
  - **GPU On-Chip HBM (HBM2e):** Row activation energy (`kACT_energy_j_`), memory read burst energy (`kREAD_energy_j_`), and memory write burst energy (`kWRITE_energy_j_`).
  - **GPU Tensor Core Compute:** Matrix-multiply-accumulate (MAC) arithmetic energy (`kMAC_energy_j_`).
  - **Subsystem Breakdown:** Fully Connected layers (`FC_DRAM`, `FC_COMP`), Attention layers (`Attn_DRAM`, `Attn_COMP`), and MoE Expert layers (`MoE_DRAM`, `MoE_COMP`).
- **NOT Modeled in Native Simulator:**
  - **Off-Chip DDR5 Physical Bus & PHY Transfer Energy:** Documented as `NOT IMPLEMENTED` in `docs/final_campaign/PHASE2D_METRIC_VALIDATION.csv` (line 28). Handled via documented analytical post-hoc models.
  - **Host CPU Predictor Execution Energy:** Documented as `NOT IMPLEMENTED` in `docs/final_campaign/PHASE2D_METRIC_VALIDATION.csv` (line 29). Predictor operates on host CPU outside GPU boundary.
  - **Static / Leakage Power Over Time:** The simulator does not integrate a $P_{\text{static}} \times t$ term into `stat.total_energy`.

### 3. Triggering Events for Energy Charges
- **HBM Access:** Charged whenever a tensor operation reads or writes data through `device->dram_interface` or `run_ideal`. Read and write counts are incremented by 32-byte burst granularity.
- **Compute (FLOPs):** Charged whenever a kernel runs matrix multiplication, proportional to total FLOPs ($2 \times \text{MACs}$).
- **Expert Cache Hits:** When an expert is resident in HBM, `linear_impl.cpp` executes `issueRamulatorBatch` with `MemoryTarget::HBM`, charging HBM read burst and row activation energy.
- **Expert Cache Misses:** When using `use_event_driven_dram = true`, demand misses trigger asynchronous transfers via `ddr5_transfer_engine`. In `linear_impl.cpp`, off-chip transfers set `weight_status.memory_duration`, but do not increment HBM read counters for the transfer itself. Once resident, subsequent forward compute reads the weights from HBM.

### 4. Dynamic vs. Static Energy
- **Native Implementation:** Purely **dynamic energy**. Energy accumulates strictly when operations execute. If a GPU stalls for 500 seconds, zero idle/static energy is accumulated into native `stat.total_energy`.
- **Analytical Extension:** Documented in `docs/phase_e/MASTER_RESEARCH_EXPERIMENT_PLAN.md` (Section 9.1): provides a baseline background static power specification ($4.5\text{ W}$ per dual-rank DIMM) to model time-dependent energy.

### 5. Units of Measurement
- Native internal accumulator type: `using energy_nJ = double` (`src/common/type.h`, line 10).
- All energy variables in `power.h`, `module_graph.cpp`, `cluster.cpp`, and exported CSV columns are denominated in **nanojoules (nJ)** ($10^{-9}\text{ Joules}$).
- Standard conversion: $1\text{ J} = 10^9\text{ nJ}$, $1\text{ mJ} = 10^6\text{ nJ}$.

### 6. Coefficients and Theoretical Sources
- Documented in `src/dram/power.h` (citing 2017 IEEE/ACM MICRO paper *Fine-Grained DRAM*, Keckler et al.):
  - `kACT_energy_j_ = 0.909 nJ` (row activation).
  - `Cell (RD/WRT) energy: 0.44 pJ/bit`.
  - `RD/WR Energy (column decoder to BG MUX): 1.01 pJ/bit`.
  - `RD/WR Energy (BG Mux to GIO Mux): 1.23 pJ/bit`.
  - `TSV energy: 0.5 pJ/bit`.
  - `Silicon interposer IO energy: 0.3 pJ/bit`.
  - **Total HBM2e Read/Write Energy:** $0.44 + 1.01 + 1.23 + 0.5 + 0.3 = 3.48\text{ pJ/bit}$.
  - At 32-byte (256-bit) burst granularity: $3.48\text{ pJ/bit} \times 256\text{ bits} / 1000 = \mathbf{0.8912\text{ nJ}}$ per burst (`kREAD_energy_j_ = 0.891`).
  - **Compute MAC Energy:** $0.46\text{ nJ} / 2 / 1000 = \mathbf{0.00023\text{ nJ/FLOP}} = \mathbf{0.23\text{ pJ/FLOP}}$ (`kMAC_energy_j_`).
- Scaled by `Device::initializeDRAM`: multiplied by `num_pseudo_ch` (for A100/GPU: $4 \times 8 / 2 = 16$).

### 7. Dependence on Workload Metrics
- `act_energy`: Proportional to number of DRAM row activation commands (`act_count`).
- `read_energy`: Proportional to number of 32-byte read bursts (`read_count`).
- `write_energy`: Proportional to number of 32-byte write bursts (`write_count`).
- `mac_energy`: Proportional to total executed floating-point operations (`flops`).
- `moe_dram`: Directly reflects HBM reads required by expert projections ($W_1, W_3, W_2$).

### 8. Treatment of Speculative Prefetch Transfers
- In native C++ `total_energy`, speculative DDR5 transfers are **NOT directly charged**, because `total_energy` only measures on-chip GPU HBM DRAM and compute MAC energy.
- However, byte-level speculative volume is tracked exactly in `prefetch_summary.json` (`speculative_ddr5_bytes`), `ddr5_transfer_engine.h` (`total_bytes_transferred`), and `prefetch_transfer_events.csv`.
- Under the calibrated analytical DDR5 model ($100\text{ pJ/byte} = 12.5\text{ pJ/bit}$), speculative DDR5 energy is directly proportional to speculative byte volume:
  $$E_{\text{speculative}} = \text{speculative\_ddr5\_bytes} \times 100\text{ pJ/byte}$$

### 9. Treatment of Cancelled Transfers
- In `DDR5TransferEngine`, transfers cancelled before dispatch transmit 0 bytes.
- Active transfers preempted by demands preserve progress (`transferred_bytes`); energy is charged strictly for transmitted bytes.

### 10. In-Flight Promotion to Demand
- In `DDR5TransferEngine::promote_transfer_to_demand`, an active or queued prefetch is modified in place (`transfer_class = DEMAND`).
- No duplicate transfer is scheduled. The bytes transferred are counted exactly once, preventing double counting.

### 11. Predictor Execution Overhead
- Predictor algorithms (CSTP, ST-MoE) execute as host software modules during token scheduling.
- The simulator does not model host CPU execution energy or PCIe control packet energy. In Configuration B (prediction-only shadow mode), GPU energy is byte-for-byte identical to Baseline A.

### 12. Idle / Background Energy
- Native simulator telemetry records zero idle energy during memory stall periods.
- When exposed memory stalls decrease by $10.94\text{ seconds}$ (as in $K=2$), native dynamic energy does not capture static power savings. This must be evaluated using the analytical static power model ($P_{\text{static}} \times \Delta t$).

### 13. Accumulation Granularity
- Energy is calculated at kernel granularity in `TopModuleGraph::update_status`.
- Stored per decoding step in `Stat` structures and exported per token step in `summary_csv` (128 rows for $T=128$).
- Total run energy is the exact sum of step records.

### 14. Consistency Across Configurations
- All four configurations (A, B, C, D) utilize identical compute kernel models, HBM coefficient matrices, and accumulation logic.
- Differences in energy stem strictly from cache residency dynamics (more resident hits reading from HBM) and off-chip DDR5 byte traffic.

---

## 4. Conclusion

The native `LLMSimulator` energy model is **authoritative and accurate for on-chip GPU HBM2e memory and compute**, but relies on post-hoc analytical translation for **off-chip DDR5 physical bus energy** and **static background power**. A scientifically valid evaluation must report both native on-chip energy and total extended system energy.

