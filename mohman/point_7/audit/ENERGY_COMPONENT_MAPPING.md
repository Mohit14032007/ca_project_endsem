# Point 7 — Phase A: Energy Component Mapping & Accounting Framework

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** Point 7 — Energy Analysis  
**Author:** Antigravity Engineering  
**Date:** November 11, 2026  
**Status:** Complete Architectural Mapping  

---

## 1. Component Mapping Architecture

To provide an unambiguous, mathematically sound evaluation, system energy is decomposed into clearly delineated physical components:

```
+-------------------------------------------------------------------------------------------------------+
|                                    TOTAL SYSTEM ENERGY DECOMPOSITION                                  |
+---------------------------------------------------+---------------------------------------------------+
|               NATIVE SIMULATOR ENERGY             |                 ANALYTICAL EXTENSIONS             |
|                  (On-Chip GPU/HBM)                |              (Off-Chip DDR5 & Static)             |
+-------------------------+-------------------------+-------------------------+-------------------------+
|     GPU Compute MAC     |     On-Chip HBM DRAM    |     DDR5 Bus Transfer   |     Static Background   |
| (Tensor Core Arithmetic)|   (Activations & Hits)  |  (Demand + Speculative) |  (Idle Leakage Power)   |
|   kMAC = 0.23 pJ/FLOP   |   kREAD = 3.48 pJ/bit   |  kBUS = 100 pJ/byte     |   Pstatic = 4.5 W       |
+-------------------------+-------------------------+-------------------------+-------------------------+
| - fc_comp               | - act_energy            | - E_DDR5_demand         | - E_static_dram         |
| - attn_comp             | - read_energy           | - E_DDR5_speculative    | - E_static_gpu          |
| - moe_comp              | - write_energy          | - E_DDR5_total          |                         |
+-------------------------+-------------------------+-------------------------+-------------------------+
```

---

## 2. Component Specifications and Formulations

### 2.1 Component 1: GPU Compute Energy ($E_{\text{comp}}$)
* **Physical Origin:** Dynamic switching power consumed by NVIDIA A100 Tensor Cores executing FP16 matrix-multiply-accumulate (MAC) arithmetic operations.
* **Simulator Counter:** `status.mac_energy` (in nanojoules), broken down into `stat.fc_comp`, `stat.attn_comp`, and `stat.moe_comp`.
* **Governing Formula:**
  $$E_{\text{comp}} = \sum_{\text{operations}} \text{FLOPs} \times k_{\text{MAC}}$$
  where $k_{\text{MAC}} = 0.00023\text{ nJ/FLOP} = 0.23\text{ pJ/FLOP} = 0.46\text{ pJ/MAC}$.
* **Invariance:** Constant across configurations for a fixed batch size and context length, as prefetching and eviction policies do not alter the number of required model FLOPs.

### 2.2 Component 2: On-Chip HBM DRAM Energy ($E_{\text{HBM}}$)
* **Physical Origin:** Row activation, column decoding, TSV traversal, and silicon interposer I/O within the 80GB HBM2e memory stack (2,039 GB/s).
* **Simulator Counters:**
  * `stat.act_energy`: Row activations ($k_{\text{ACT}} = 0.909\text{ nJ}$).
  * `stat.read_energy`: 32-byte burst reads ($k_{\text{READ}} = 0.891\text{ nJ}$).
  * `stat.write_energy`: 32-byte burst writes ($k_{\text{WRITE}} = 0.891\text{ nJ}$).
  * Layer breakdown: `stat.fc_dram`, `stat.attn_dram`, `stat.moe_dram`.
* **Governing Formula:**
  $$E_{\text{HBM}} = E_{\text{act}} + E_{\text{read}} + E_{\text{write}}$$
* **Prefetch Sensitivity:** When asynchronous prefetching successfully places experts in HBM prior to demand, the subsequent forward pass accesses weights directly from HBM, slightly increasing $E_{\text{read}}$ (proportional to resident hits).

### 2.3 Component 3: Off-Chip DDR5 Bus Transfer Energy ($E_{\text{DDR5}}$)
* **Physical Origin:** Host memory controller PHY, PCB trace transmission line signaling, and DDR5 DIMM DRAM bank operations across the 64 GB/s off-chip interconnect.
* **Simulator Telemetry:** Tracked to the exact byte by `DDR5TransferEngine` and `PrefetchMetricsCollector`:
  * $D_{\text{bytes}}$: `demand_ddr5_bytes` (unbuffered demand expert fetches).
  * $S_{\text{bytes}}$: `speculative_ddr5_bytes` (speculatively dispatched prefetches).
  * $T_{\text{bytes}}$: `total_ddr5_bytes` ($T_{\text{bytes}} = D_{\text{bytes}} + S_{\text{bytes}}$).
* **Governing Analytical Formula (Post-Hoc Extension):**
  Documented in `docs/final_campaign/PHASE2D_EXECUTION_RECOMMENDATION.md` (Condition 3):
  $$E_{\text{DDR5}} = \text{Transferred Bytes} \times 8\text{ bits/byte} \times 12.5\text{ pJ/bit} = \text{Transferred Bytes} \times 100\text{ pJ/byte}$$
  where:
  $$E_{\text{DDR5,demand}} = D_{\text{bytes}} \times 100\text{ pJ/byte} = D_{\text{bytes}} \times 0.1\text{ nJ/byte}$$
  $$E_{\text{DDR5,speculative}} = S_{\text{bytes}} \times 100\text{ pJ/byte} = S_{\text{bytes}} \times 0.1\text{ nJ/byte}$$
  $$E_{\text{DDR5,total}} = E_{\text{DDR5,demand}} + E_{\text{DDR5,speculative}}$$

### 2.4 Component 4: Static Background / Idle Energy ($E_{\text{static}}$)
* **Physical Origin:** Transistor leakage currents in DDR5 DIMMs and GPU silicon while waiting during exposed memory stall periods.
* **Governing Analytical Formula:**
  Documented in `docs/phase_e/MASTER_RESEARCH_EXPERIMENT_PLAN.md` (Section 9.1):
  $$E_{\text{static}} = P_{\text{background}} \times t_{\text{execution}}$$
  where $P_{\text{background}} = 4.5\text{ W}$ per dual-rank DIMM channel.
* **Prefetch Sensitivity:** Prefetching reduces exposed memory stalls (by up to $10.94\text{ seconds}$ at $B=32, K=2$), yielding direct static power savings:
  $$\Delta E_{\text{static}} = P_{\text{background}} \times (t_{\text{baseline}} - t_{\text{prefetch}})$$

### 2.5 Component 5: Host CPU Predictor Energy ($E_{\text{pred}}$)
* **Physical Origin:** Host x86 CPU cores calculating CSTP co-occurrence frequencies and top-K candidate selections.
* **Simulator Status:** `NOT IMPLEMENTED` in GPU simulator. Software operates on host CPU outside GPU telemetry boundary.
* **Reporting Standard:** Quantified as out-of-boundary software overhead. In Configuration B (prediction-only shadow mode), GPU energy is identical to Baseline A.

---

## 3. Strict Double-Counting Safeguards

To prevent artificial inflation or deflation of energy results:

1. **In-Flight Promotion:** When a demand request attaches to an in-flight speculative prefetch (`DDR5TransferEngine::promote_transfer_to_demand`), the transfer is modified in place. Its bytes are charged strictly once under speculative bytes; zero duplicate demand bytes are recorded.
2. **Cancelled Prefetches:** Transfers cancelled before dispatch transmit 0 bytes and incur 0 energy.
3. **Physical Conservation:** Across all 26 evaluations, verified that:
   $$\text{Total DDR5 Bytes} = \text{Demand DDR5 Bytes} + \text{Speculative DDR5 Bytes}$$
   $$E_{\text{DDR5,total}} = E_{\text{DDR5,demand}} + E_{\text{DDR5,speculative}}$$
4. **HBM Cache Reads vs. DDR5 Misses:** HBM reads are charged only when weights are accessed by the GPU compute kernels. The physical transfer from host memory to GPU HBM is charged under DDR5, avoiding double-counting between the interconnect and on-chip memory.

---

## 4. Derived Energy Performance Metrics

1. **Total Modeled System Energy:**
   $$E_{\text{total}} = E_{\text{HBM}} + E_{\text{comp}} + E_{\text{DDR5,total}} + E_{\text{static}}$$
2. **Energy per Generated Token:**
   $$E_{\text{token}} = \frac{E_{\text{total}}}{B \times T} \quad \left[\frac{\text{Joules}}{\text{token}}\right]$$
3. **Energy-Delay Product (EDP):**
   $$\text{EDP} = E_{\text{total}} \times t_{\text{execution}} \quad [\text{Joules} \cdot \text{seconds}]$$
4. **Energy Saving / Overhead Relative to Baseline:**
   $$\Delta E (\%) = \frac{E_{\text{proposed}} - E_{\text{baseline}}}{E_{\text{baseline}}} \times 100\%$$

