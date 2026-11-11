# Master YAML Configuration Architecture & Specification

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Classification:** Research Specification & Schema Design (Planning Only)  
**Date:** November 2026  

---

## 1. Architectural Overview & Design Objectives

This document establishes the unified configuration schema for all subsequent experimental phases (Phase 1 through Phase 7). The configuration framework bridges:
1. **Model Topologies:** Supporting 57-stage DeepSeek-V3 synthetic baselines, 75-stage MoE Suite replays, and 40-stage MoE Anywhere replays.
2. **Hardware Hierarchies:** Parameterizing unconstrained (80 GB) and capacity-constrained (2–16 GiB) HBM, DDR5 off-chip memory systems, and execution cache bypass flags.
3. **Algorithm Integration:** Incorporating offline and in-simulator predictors (CSTP, ST-MoE, ACSH) and asynchronous prefetch engines without code rewrites.

> [!IMPORTANT]
> In accordance with planning restrictions, **no existing YAML files are modified or overwritten**. All templates specified herein define the target schemas for future execution phases.

---

## 2. Parameter Semantics & Section Specifications

### 2.1 The `model` Section
```yaml
model:
  name: "deepseekV3"          # Base architecture template
  num_layers: 60              # Total Transformer layers (Dense + MoE)
  first_k_dense: 3            # Initial dense layers before MoE stages
  num_routed_expert: 256      # Number of routed experts per stage
  top_k: 8                    # Experts selected per token per stage
  num_shared_expert: 1        # Dedicated shared experts
  hidden_size: 7168           # Model hidden dimension
  intermediate_size: 2048     # Expert FFN intermediate dimension
  input_len: 128              # KV-cache prefill context length
  output_len: 512             # Target sequence generation limit
  precision_byte: 2           # FP16 (2 bytes per parameter)
```
- **Layer Mapping Formula:**
  $$\text{moe\_layers} = \text{model.num\_layers} - \text{model.first\_k\_dense}$$
  - For DeepSeek-V3 Full Depth (57 MoE stages): `num_layers: 60`, `first_k_dense: 3` $\to \mathbf{57\text{ MoE stages}}$.
  - For MoE Suite Full Depth (75 MoE stages): `num_layers: 78`, `first_k_dense: 3` $\to \mathbf{75\text{ MoE stages}}$.
  - For MoE Anywhere Full Depth (40 MoE stages): `num_layers: 43`, `first_k_dense: 3` $\to \mathbf{40\text{ MoE stages}}$.

### 2.2 The `system` Section
```yaml
system:
  processor_type: "GPU"
  compute_peak_flops: 312000000000000 # 312 TFLOPS (A100 FP16 Tensor Core)
  hbm_capacity_bytes: 80000000000     # Default 80 GB reference
  hbm_policy: "LRU"                   # LRU, LFU_Frequency, or Prediction_Aware
  optimization:
    bypass_offchip_execution_cache: true # Enforce cycle-accurate Ramulator simulation
    use_ramulator: true
  dram:
    config_path: "configs/A100_HBM2.yaml"
    scale_factor: 1.0
  offchip_dram:
    config_path: "src/dram/ramulator2/build/DDR5_32GB_x16_2ch.yaml"
    scale_factor: 0.625
```
- **Execution Cache Semantics:**
  - `bypass_offchip_execution_cache: true`: Forces Ramulator to simulate all 64-byte bursts for every off-chip transfer. Mandatory for prefetch bus contention evaluation.
  - `bypass_offchip_execution_cache: false`: Enables memoized cycle reuse. Suitable for rapid screening baselines with identical un-contended burst latencies.

### 2.3 The `scheduler` & `simulation` Sections
```yaml
scheduler:
  max_batch_size: 1           # Concurrent active sequences (1, 2, 4, 8)
  injection_rate: 0           # Closed-loop batch injection
  trace_file: ""              # Path to CSV replay trace (empty for DS-SYN)

simulation:
  iter: 1                     # Decoding steps per sequence (1=cold/warm, 2=LRU)
  warm_hbm_preload: false     # Preload trace working set prior to cycle counting
```

### 2.4 New Extension: The `predictor` Section (Phases 3–7)
```yaml
predictor:
  name: "acsh"                # none, random, frequency, cstp, st_moe, acsh
  lookahead_horizon: 2        # Inter-layer spatial lookahead (H in [1, 2, 3, 4])
  budget_kp: 8                # Prediction candidate budget (Kp in [2, 4, 8, 16])
  online_adaptation: true     # Enable online state updates during generation
  confidence_threshold: 0.65  # Confidence gating threshold for hybrid switching
```

### 2.5 New Extension: The `prefetch` Section (Phases 4–7)
```yaml
prefetch:
  enabled: true               # Enable asynchronous prefetch engine
  mode: "asynchronous"        # asynchronous vs synchronous
  coalescing: true            # In-flight duplicate coalescing
  pinning: true               # Pin prefetched experts in HBM until consumption
  max_outstanding_requests: 32 # Maximum concurrent DDR5 prefetch transactions
```

---

## 3. Reference YAML Templates for Experimental Phases

### Template 1: DeepSeek-V3 Full-Depth Synthetic Baseline (`DS-SYN`, 57 MoE Stages)
```yaml
# config/phase1/ds_syn_57l_t512_cold.yaml
model:
  name: "deepseekV3"
  num_layers: 60
  first_k_dense: 3
  num_routed_expert: 256
  top_k: 8
  input_len: 128
  output_len: 512
  precision_byte: 2

system:
  processor_type: "GPU"
  hbm_capacity_bytes: 80000000000
  hbm_policy: "LRU"
  optimization:
    bypass_offchip_execution_cache: true
    use_ramulator: true
  offchip_dram:
    config_path: "src/dram/ramulator2/build/DDR5_32GB_x16_2ch.yaml"

scheduler:
  max_batch_size: 1
  injection_rate: 0
  trace_file: ""

simulation:
  iter: 1
  warm_hbm_preload: false

log:
  output_directory: "results/phase_e/phase1_baselines/ds_syn_t512_cold"
```

### Template 2: MoE Suite Full-Depth Real Replay Baseline (`MS`, 75 MoE Stages)
```yaml
# config/phase1/ms_task12_75l_t512_cold.yaml
model:
  name: "deepseekV3"
  num_layers: 78              # 3 Dense + 75 MoE stages
  first_k_dense: 3
  num_routed_expert: 256
  top_k: 8
  input_len: 128
  output_len: 512
  precision_byte: 2

system:
  processor_type: "GPU"
  hbm_capacity_bytes: 80000000000
  hbm_policy: "LRU"
  optimization:
    bypass_offchip_execution_cache: true
    use_ramulator: true
  offchip_dram:
    config_path: "src/dram/ramulator2/build/DDR5_32GB_x16_2ch.yaml"

scheduler:
  max_batch_size: 1
  injection_rate: 0
  trace_file: "results/phase_e/exported_traces/moe_suite_task12_full75_t512.csv"

simulation:
  iter: 1
  warm_hbm_preload: false

log:
  output_directory: "results/phase_e/phase1_baselines/ms_task12_t512_cold"
```

### Template 3: ACSH Predictor + Asynchronous Prefetching (`Phase 4`)
```yaml
# config/phase4/ms_task12_acsh_prefetch_h2_kp8.yaml
model:
  name: "deepseekV3"
  num_layers: 78
  first_k_dense: 3
  num_routed_expert: 256
  top_k: 8
  input_len: 128
  output_len: 512
  precision_byte: 2

system:
  processor_type: "GPU"
  hbm_capacity_bytes: 80000000000
  hbm_policy: "LRU"
  optimization:
    bypass_offchip_execution_cache: true
    use_ramulator: true
  offchip_dram:
    config_path: "src/dram/ramulator2/build/DDR5_32GB_x16_2ch.yaml"

scheduler:
  max_batch_size: 1
  injection_rate: 0
  trace_file: "results/phase_e/exported_traces/moe_suite_task12_full75_t512.csv"

simulation:
  iter: 1
  warm_hbm_preload: false

predictor:
  name: "acsh"
  lookahead_horizon: 2
  budget_kp: 8
  online_adaptation: true
  confidence_threshold: 0.65

prefetch:
  enabled: true
  mode: "asynchronous"
  coalescing: true
  pinning: true
  max_outstanding_requests: 32

log:
  output_directory: "results/phase_e/phase4_prefetch/ms_acsh_h2_kp8"
```

### Template 4: Constrained HBM (4 GiB) + Prediction-Aware Eviction (`Phase 5`)
```yaml
# config/phase5/ms_task12_acsh_predaware_4gb.yaml
model:
  name: "deepseekV3"
  num_layers: 78
  first_k_dense: 3
  num_routed_expert: 256
  top_k: 8
  input_len: 128
  output_len: 512
  precision_byte: 2

system:
  processor_type: "GPU"
  hbm_capacity_bytes: 4294967296     # Constrained to 4 GiB (Binary IEC)
  hbm_policy: "Prediction_Aware"      # Evicts unpredicted experts first
  optimization:
    bypass_offchip_execution_cache: true
    use_ramulator: true
  offchip_dram:
    config_path: "src/dram/ramulator2/build/DDR5_32GB_x16_2ch.yaml"

scheduler:
  max_batch_size: 1
  injection_rate: 0
  trace_file: "results/phase_e/exported_traces/moe_suite_task12_full75_t512.csv"

simulation:
  iter: 2
  warm_hbm_preload: false

predictor:
  name: "acsh"
  lookahead_horizon: 2
  budget_kp: 8

prefetch:
  enabled: true
  mode: "asynchronous"
  coalescing: true
  pinning: true

log:
  output_directory: "results/phase_e/phase5_hbm/ms_acsh_predaware_4gb"
```

---

## 4. Environment Variables & Safety Overrides

To maintain absolute test harness integrity, the following runtime environment overrides are supported:
- `BYPASS_OFFCHIP_CACHE=1`: Enforces cycle-accurate Ramulator simulation regardless of YAML default.
- `DEBUG_CACHE_CAPACITY=<bytes>`: Dynamically overrides HBM capacity for constrained sweeps without editing YAML files.
- `TRACE_DIR=<path>`: Redirects internal log writing (`expert_routing.csv`, `expert_cache.csv`, `expert_load.csv`) to designated run directories.
- `PREFETCHER_DISABLE=1`: Hardware kill-switch that disables asynchronous prefetch issue, allowing instant fallback to pure demand loading.

