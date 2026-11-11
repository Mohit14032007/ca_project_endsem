# E1-A: Baseline Characterization Audit — Synthetic DeepSeek Workload

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** E1-A Read-Only Baseline Architecture Audit  
**Date:** November 2026  
**Status:** COMPLETE (READ-ONLY AUDIT)  
**Host Target:** `x86_64` Linux  
**Repository Path:** `/home/mohman/ca_project/LLMSimulator`  
**Authoritative Binary:** `build/run_deepseekv3_mixtral_style`  

---

## 1. Executive Summary

This document establishes the comprehensive pre-experiment audit for **Phase E1-A: Synthetic DeepSeek Workload — Baseline Characterization**.

The objective of Phase E1-A is to characterize the hardware performance of the existing demand-loading memory subsystem in `LLMSimulator` prior to implementing any asynchronous prefetching or predictor integration. This evaluation operates strictly under **NO-PREFETCH** demand paging (`bypass_offchip_execution_cache = true`), measuring DDR5 expert-loading latency, HBM3 cache hit/miss behavior, memory stalls, DDR5 bandwidth utilization, and execution throughput across batch sizes ($B \in \{1, 2, 4, 8\}$), sequence lengths ($L \in \{128, 256, 512, 1024\}$), and HBM initial states (Cold, Warm, Natural LRU).

### Critical Finding on Routing Fidelity
> [!IMPORTANT]
> **Synthetic Routing vs Neural Routing:** The existing DeepSeek workload in `LLMSimulator` utilizes a synthetic pseudo-random / Zipfian expert generator (`Scheduler::getRandomExpert` / `Scheduler::getZipfianRandomExpert`). It does **not** evaluate genuine DeepSeek neural routing logits or routing traces. This baseline characterizes the memory and bus architecture under synthetic demand loads and must **never** be cited as actual DeepSeek-V3 neural traffic. Genuine trace replay is reserved for Phase E1-B.

---

## 2. Workload Configuration Audit

### 2.1 Model Architecture Parameters

The DeepSeek-V3 architecture configurations are defined in [`src/model/model_config.h`](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h). Two primary configurations exist in the source code:

| Parameter | `deepseekV3` (Full Scale) | `debugV3` (Fast Debug) | Description |
|---|---|---|---|
| Hidden Dimension ($d_{\text{model}}$) | 7168 | 7168 | Model hidden representation width |
| Number of Attention Heads | 128 | 128 | Standard multi-head / MLA query heads |
| Total Layers ($N_{\text{layers}}$) | 60 | 10 | Total transformer layers |
| Dense Layers ($N_{\text{dense}}$) | 3 | 3 | First $k$ dense layers (`first_k_dense = 3`) |
| MoE Layers ($N_{\text{moe}}$) | 57 | 7 | Layers containing Mixture-of-Experts |
| Routed Experts per Layer ($E$) | 256 | 256 | Total routed expert inventory per MoE layer |
| Top-K Routing ($K$) | 8 | 8 | Active routed experts selected per token |
| Shared Experts per Layer ($E_{\text{shared}}$) | 1 | 1 | Always-active shared expert |
| Intermediate Dimension ($d_{\text{ffn}}$) | 2048 | 2048 | Expert feed-forward inner dimension |
| FFN Projections (`ffn_way`) | 3 | 3 | Gate, Up, and Down projections ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) |
| Weight Precision | FP16 (2 bytes) | FP16 (2 bytes) | Precision per parameter (`precision_byte = 2`) |

### 2.2 Expert Weight Footprint and Total Inventory

For each expert in DeepSeek-V3:
- $W_{\text{gate}} \in \mathbb{R}^{7168 \times 2048} \implies 14,680,064\text{ parameters}$
- $W_{\text{up}} \in \mathbb{R}^{7168 \times 2048} \implies 14,680,064\text{ parameters}$
- $W_{\text{down}} \in \mathbb{R}^{2048 \times 7168} \implies 14,680,064\text{ parameters}$
- **Total parameters per expert:** $3 \times 14,680,064 = 44,040,192\text{ parameters}$
- **Expert weight footprint (FP16):**
  $$\text{Size}_{\text{expert}} = 44,040,192 \times 2\text{ bytes} = 88,080,384\text{ bytes} \approx 88.08\text{ MB}\;(84.00\text{ MiB})$$

Total memory footprint across all layers:
- **Full Model (`deepseekV3`, 57 MoE layers):**
  - Total routed experts: $57 \times 256 = 14,592\text{ experts}$
  - Routed expert footprint: $14,592 \times 88,080,384\text{ bytes} = 1,285,268,963,328\text{ bytes} \approx 1.285\text{ TB}\;(1.169\text{ TiB})$
  - Shared expert footprint: $57 \times 88,080,384\text{ bytes} = 5,020,581,888\text{ bytes} \approx 5.02\text{ GB}$
  - Total expert weight footprint: $\approx 1.290\text{ TB}$
- **Debug Model (`debugV3`, 7 MoE layers):**
  - Total routed experts: $7 \times 256 = 1,792\text{ experts}$
  - Total footprint: $1,792 \times 88,080,384\text{ bytes} = 157,840,047,360\text{ bytes} \approx 157.84\text{ GB}$
- **Bounded Smoke Model (`num_layers = 5`, 2 MoE layers):**
  - Total routed experts: $2 \times 256 = 512\text{ experts}$
  - Total footprint: $512 \times 88,080,384\text{ bytes} = 45,097,156,608\text{ bytes} \approx 45.10\text{ GB}$

### 2.3 Batch Size and Sequence Length Parameters

- **Batch Size ($B$):** Configured via `serving: max_batch_size: <B>`. Controls the number of active sequences populated into the running queue via `scheduler->pushDummySeq(input_len, output_len)`.
- **Input Context Length ($L$):** Configured via `simulation: input_len: <L>`. Determines the context prompt length, setting attention KV cache allocations and activation tensor dimensions.
- **Decoding Steps ($I$):** Configured via `simulation: iter: <I>`. Governs the number of autoregressive sequential token generation steps executed in the decoding loop.

### 2.4 Routing Generation Mechanism and Random Seed

In [`src/scheduler/sequence.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L210-L225):
```cpp
std::set<int> expert_list;
if (scheduler->model_config.skewness > 0) {
  expert_list = scheduler->getZipfianRandomExpert(skewness_weight, top_k);
} else {
  expert_list = scheduler->getRandomExpert(top_k);
}
```
In [`src/scheduler/scheduler.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L141-L151):
```cpp
std::set<int> Scheduler::getRandomExpert(int top_k) {
  static unsigned int seed = 777;
  static std::mt19937 generator(seed);
  static std::uniform_int_distribution<int> distribution(0, model_config.num_routed_expert - 1);
  std::set<int> route;
  while (route.size() < top_k) {
    int seq_id = (distribution(generator) % model_config.num_routed_expert);
    route.insert(seq_id);
  }
  return route;
}
```
- **Reproducibility:** The RNG seed is hardcoded as `static unsigned int seed = 777` in `Scheduler::getRandomExpert` and `Scheduler::getRandomExpertSeqId`.
- **Determinism:** Each simulation run starting from process initialization generates the exact same sequence of pseudo-random expert routing IDs.
- **Routing Cache:** Within a simulation, `cached_routing[{token_id, cur_layer}]` memoizes the selected experts for that token and layer, ensuring consistency if a token is re-evaluated.

---

## 3. Memory-System Configuration Audit

### 3.1 Dual-DRAM Physical Topology

The simulator models a heterogeneous dual-memory hierarchy: high-bandwidth on-chip HBM3 acting as an active expert execution cache, backed by large-capacity off-chip DDR5 memory storing the complete MoE parameter inventory.

```
+-----------------------------------------------------------------------------------+
|                                 GPU Accelerator                                   |
|                                                                                   |
|  +--------------------+       +------------------------------------------------+  |
|  | Tensor Core Compute| <---> | On-Chip HBM3 (80 GB Local Expert Cache)       |  |
|  | (312 TFLOPS FP16)  |       | 32 Channels, 2.04 TB/s Peak Bandwidth          |  |
|  +--------------------+       +------------------------------------------------+  |
+----------------------------------------^------------------------------------------+
                                         |
                                         | PCIe / CXL / Off-Chip Bus
                                         | Demand Transfer on Cache Miss (~3.90 ms)
                                         v
+-----------------------------------------------------------------------------------+
| Off-Chip DDR5 Backing Store (2.75 TB Capacity Across 10,485,760 Rows)            |
| Preset: DDR5_32Gb_x16, 2 Channels, 2 Ranks, DDR5-3200BN (51.2 GB/s Peak)        |
+-----------------------------------------------------------------------------------+
```

### 3.2 Memory Configuration Parameters

From [`src/dram/memory_config.h`](file:///home/mohman/ca_project/LLMSimulator/src/dram/memory_config.h) and YAML configuration snapshots:

| Parameter | On-Chip HBM3 (`hbm3_80GB`) | Off-Chip DDR5 (`ddr5_32gb_x16_2ch`) |
|---|---|---|
| Memory Class | `MemoryType::HBM` | `MemoryType::DDR5` |
| Cubes | 5 | 1 |
| Channels | 32 pseudo-channels | 2 channels |
| Ranks per Channel | 2 | 2 |
| Bankgroups per Rank | 4 | 4 |
| Banks per Bankgroup | 4 | 4 |
| Rows | 16,384 | 10,485,760 ($2^{17} \times 80$) |
| Columns | 32 | 64 |
| Burst Granularity | 32 bytes | 64 bytes |
| Total Addressable Capacity | $80\text{ GB}\;(80,000,000,000\text{ B})$ | $2,748,779,069,440\text{ B}\;(\approx 2.75\text{ TB})$ |
| Memory Timing Preset | `HBM3` | `DDR5_3200BN` (1600 MHz clock, 3200 MT/s) |
| Peak Theoretical Bandwidth | $2,039\text{ GB/s}\;(2.04\text{ TB/s})$ | $51.2\text{ GB/s}\;(2 \times 8\text{ B} \times 3200\text{ MT/s})$ |
| Address Mapping Scheme | `RoBaRaCoCh` | `RoBaRaCoCh` (Row-Bank-Rank-Column-Channel) |

### 3.3 HBM Residency Capacity and Working Set

- **Cache Budget:** $80\text{ GB} = 80,000,000,000\text{ bytes}$ (configured via `expert_cache_capacity_bytes`).
- **Expert Size:** $88,080,384\text{ bytes}$ ($88.08\text{ MB}$).
- **Maximum Resident Experts in HBM:**
  $$N_{\text{resident}}^{\max} = \left\lfloor \frac{80,000,000,000}{88,080,384} \right\rfloor = 908\text{ experts}$$
- **Full Working Set Capacity Analysis:**
  - 1 token decoding step in full DeepSeek-V3 activates $57\text{ layers} \times 8\text{ experts} = 456\text{ experts}$.
  - Memory required for 1 step: $456 \times 88,080,384 = 40,164,655,104\text{ bytes} \approx 40.16\text{ GB}$ ($50.2\%$ of HBM).
  - 2 token steps with distinct experts require $912\text{ experts} \approx 80.33\text{ GB}$, exceeding the $908$-expert limit.
  - Therefore, **LRU evictions naturally begin occurring on Step 2 or Step 3** under random expert selection.
  - An all-resident configuration ($14,592\text{ experts} = 1.285\text{ TB}$) exceeds HBM capacity by $16.1\times$ and is physically **infeasible** on a single 80 GB device.

### 3.4 LRU Cache Implementation

In [`src/hardware/device.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L275-L335):
- **Data Structures:**
  - `std::list<std::tuple<int, int, long long>> lru_expert_list;` — Doubly-linked list ordered from LRU (front) to MRU (back). Stores `(layer_id, expert_id, expert_size)`.
  - `std::map<std::pair<int, int>, ...> resident_experts;` — $O(1)$ lookup iterator mapping `(layer_id, expert_id)` to list node.
- **Lookup (`is_expert_resident`):** Probes `resident_experts`. On HIT, splices the node to the back (MRU) and logs `HIT` to `expert_cache.csv`. On MISS, logs `MISS`.
- **Insertion (`mark_expert_resident`):** While `resident_bytes + expert_size > cache_capacity_bytes`, pops the front element (LRU), removes it from `resident_experts`, decrements `resident_bytes`, and logs `EVICT`. Then appends the new expert to MRU, increments `resident_bytes`, and logs `INSERT`.

---

## 4. Execution and Timing Audit

### 4.1 Latency and Duration Metrics

The simulator distinguishes between simulated hardware time and host wall-clock time:

1. **Expert-Loading Latency (Simulated):**
   - When an expert misses HBM, tensors are batched and dispatched to Ramulator2 via `issueRamulatorBatch`.
   - Simulated DDR5 loading latency for 88.08 MB across 2 DDR5 channels is $\approx 3,900,000\text{ ns}\;(3.90\text{ ms})$.
   - On cache HIT, memory target is HBM, incurring minimal on-chip delay ($\approx 11.4\,\mu\text{s}$).
2. **GPU Compute Duration (Simulated):**
   - Modeled analytically based on FP16 tensor core throughput ($312\text{ TFLOPS}$).
   - For batch size 1 ($M=1, K=7168, N=2048$), compute duration is $\approx 1.2\,\mu\text{s}$.
3. **Memory Stall Duration (Simulated):**
   - Because demand paging is synchronous, the GPU pipeline stalls during the entire DDR5 transfer duration:
     $$T_{\text{stall}} = T_{\text{DDR5\_transfer}} \approx 3.90\text{ ms per cold expert}$$
   - Fraction of execution time spent stalling:
     $$\text{Stall Fraction} = \frac{T_{\text{stall}}}{T_{\text{compute}} + T_{\text{stall}}} = \frac{3900\,\mu\text{s}}{1.2\,\mu\text{s} + 3900\,\mu\text{s}} \approx 99.97\%$$
4. **Host Wall-Clock Execution Time:**
   - Measured using `std::chrono::high_resolution_clock::now()` on the host CPU.
   - Cycle-accurate simulation of 88 MB in Ramulator2 consumes **$\sim 9.2 - 10.1$ seconds of host CPU wall-clock time per cold expert**.
5. **Throughput:**
   - Simulated Throughput: $\text{TPS}_{\text{sim}} = \frac{\text{process\_tokens}}{\text{step\_device\_time\_seconds}}$ (tokens/sec in simulated time).
   - Host Throughput: $\text{TPS}_{\text{host}} = \frac{\text{process\_tokens}}{\text{step\_host\_seconds}}$ (tokens/sec in host wall-clock time).

---

## 5. Execution-Cache Configuration Audit

### 5.1 Primary Baseline Requirement: `bypass_offchip_execution_cache = true`

In [`src/hardware/layer_impl.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/layer_impl.cpp#L19-L35):
- When `bypass_offchip_execution_cache = false` (legacy shortcut), transfers with identical `(LayerType, ProcessorType, DRAMRequestType, size, MemoryTarget)` reuse the memoized `ExecStatus` from the first Ramulator simulation, bypassing subsequent DRAM simulation.
- When `bypass_offchip_execution_cache = true`, **every individual DRAM request is simulated through Ramulator2**, capturing true bus traffic, memory controller queuing, and bank contention.

### 5.2 Independence of HBM Caching vs Off-Chip Execution Cache

> [!NOTE]
> **Architectural Independence:**
> - **HBM LRU Cache (`Device`):** Models the physical 80 GB high-bandwidth memory on the GPU. Tracks whether an expert is resident on-chip.
> - **Off-Chip Execution Cache (`Cluster`):** A simulator memoization acceleration table that shortcuts Ramulator2 invocations.
> 
> Bypassing the execution cache (`bypass_offchip_execution_cache = true`) does **not** disable HBM caching. When an expert is resident in HBM, `weight_target` is `MemoryTarget::HBM`, completely avoiding DDR5. Only DDR5 misses enter the off-chip transfer path where the execution cache bypass is applied.

### 5.3 Environment Override Validation

In [`src/hardware/hardware_config.h`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/hardware_config.h#L155-L162):
```cpp
void apply_environment_overrides() {
  const char* bypass_cache = std::getenv("BYPASS_OFFCHIP_EXECUTION_CACHE");
  if (bypass_cache) {
    std::string val(bypass_cache);
    if (val == "1" || val == "true" || val == "on" || val == "yes") {
      bypass_offchip_execution_cache = true;
    } else if (val == "0" || val == "false" || val == "off" || val == "no") {
      bypass_offchip_execution_cache = false;
    }
  }
}
```
Precedence order:
1. Environment variable `BYPASS_OFFCHIP_EXECUTION_CACHE` (highest).
2. YAML configuration `system.optimization.bypass_offchip_execution_cache`.
3. C++ default (`false`).

For all Phase E1-A runs, both the YAML configuration will set `bypass_offchip_execution_cache: true` and the environment variable `BYPASS_OFFCHIP_EXECUTION_CACHE=1` will be exported to guarantee 100% enforcement.

---

## 6. Feasibility and Workload Matrix

### 6.1 Feasibility Analysis and Host Runtime Constraints

Detailed host runtime measurement across existing verified runs in the repository:
- **Smoke test** (`num_layers = 5`, 2 MoE layers, $2 \times 8 \times 2 = 32$ expert loads): **323.98 seconds (5.4 minutes)** $\implies 10.12\text{ s/expert}$.
- **Pilot 1-step** (`num_layers = 60`, 57 MoE layers, $57 \times 8 \times 1 = 456$ expert loads): **4,685 seconds (78.1 minutes / 1.3 hours)** $\implies 10.27\text{ s/expert}$.
- **Full 5-step** (`num_layers = 60`, 57 MoE layers, $456 \times 5 = 2,280$ expert loads): **21,159 seconds (5.87 hours)** $\implies 9.28\text{ s/expert}$.

Strict Constraint: *"Do not execute long-running simulations."*
Running 128 sequential decoding steps of 60 layers would require $\sim 149\text{ hours}$ (6.2 days), which is completely infeasible.

### 6.2 Supported Workload Configurations

To maintain exact DeepSeek-V3 model semantics (256 routed experts, top-8 routing, 7168 hidden dimension, 2048 intermediate dimension, 88.08 MB expert size, 80 GB HBM, 2.75 TB DDR5) while completing execution in bounded time:
1. **Full-Depth Characterization Baseline:** Anchor to the authoritative 1-step and 5-step full 60-layer DeepSeek runs (`pilot_1step` and `full_run_5step_20261011`).
2. **Parametric Scaling Matrix:** Execute batch size scaling ($B \in \{1, 2, 4, 8\}$) and sequence length scaling ($L \in \{128, 256, 512, 1024\}$) under controlled iteration stepping ($1$ decoding step per configuration) on the verified model configuration.
3. **HBM Initial-State Matrix:** Compare Cold HBM, Warm HBM, and Natural LRU under identical routing seeds.

### 6.3 Proposed Experiment Matrix

| Experiment ID | Category | Batch Size ($B$) | Sequence Length ($L$) | Total Layers | MoE Layers | Steps ($I$) | HBM Initial State |
|---|---|---|---|---|---|---|---|
| `exp_b1_l128` | Batch Sweep / Combined | 1 | 128 | 5 | 2 | 1 | Cold |
| `exp_b2_l128` | Batch Sweep | 2 | 128 | 5 | 2 | 1 | Cold |
| `exp_b4_l128` | Batch Sweep | 4 | 128 | 5 | 2 | 1 | Cold |
| `exp_b8_l128` | Batch Sweep | 8 | 128 | 5 | 2 | 1 | Cold |
| `exp_b1_l256` | SeqLen Sweep | 1 | 256 | 5 | 2 | 1 | Cold |
| `exp_b1_l512` | SeqLen Sweep | 1 | 512 | 5 | 2 | 1 | Cold |
| `exp_b1_l1024`| SeqLen Sweep | 1 | 1024 | 5 | 2 | 1 | Cold |
| `exp_b2_l256` | Combined Scaling | 2 | 256 | 5 | 2 | 1 | Cold |
| `exp_b4_l512` | Combined Scaling | 4 | 512 | 5 | 2 | 1 | Cold |
| `exp_b8_l512` | Combined Scaling | 8 | 512 | 5 | 2 | 1 | Cold |
| `exp_b8_l1024`| Combined Scaling | 8 | 1024 | 5 | 2 | 1 | Cold |
| `exp_cold` | HBM State | 1 | 128 | 5 | 2 | 2 | Cold (Step 0) |
| `exp_warm` | HBM State | 1 | 128 | 5 | 2 | 2 | Warm Replay (Step 1) |
| `exp_lru` | HBM State | 1 | 128 | 5 | 2 | 5 | Natural LRU |

---

## 7. Metric Definitions and Logging Locations

| Metric Group | Metric Name | Definition / Formula | Source File / Telemetry Path |
|---|---|---|---|
| **Performance** | Simulated Device Time ($T_{\text{sim}}$) | Total simulated hardware elapsed time | `device->status.device_time`, `Stat.latency` (ns) |
| | Host Wall-Clock Time ($T_{\text{host}}$) | Wall-clock execution time on host CPU | `std::chrono::high_resolution_clock` (s) |
| | Average Token Latency | $\frac{T_{\text{sim}}}{I}$ (ns / token) | Calculated from `summary_csv` |
| | Simulated Throughput | $\frac{\text{Total Processed Tokens}}{T_{\text{sim\_sec}}}$ (tokens/s) | Calculated from `summary_csv` |
| | Host Throughput | $\frac{\text{Total Processed Tokens}}{T_{\text{host\_sec}}}$ (tokens/s) | Calculated from `summary_csv` |
| **HBM Cache** | HBM Capacity | Total cache budget: $80,000,000,000\text{ B}$ | `device->cache_capacity_bytes` |
| | Resident Experts | Number of expert weights resident in HBM | `device->lru_expert_list.size()` |
| | Resident Memory | Total bytes resident in HBM | `device->resident_bytes` |
| | HBM Hits / Misses | Count of cache probe actions | `expert_cache.csv` (`HIT`, `MISS`) |
| | HBM Hit Rate | $\frac{\text{Hits}}{\text{Hits} + \text{Misses}}$ | Derived from `expert_cache.csv` |
| | Expert Evictions | Number of LRU evictions | `expert_cache.csv` (`EVICT`) |
| **DDR5 DRAM** | Bytes Transferred | Total expert weight bytes read from DDR5 | `expert_load.csv` (`size_bytes` where miss) |
| | DRAM Requests | Total 64-byte bursts sent to Ramulator2 | Derived: $\frac{\text{Total Bytes}}{64}$ |
| | DDR5 Simulated Time | Simulated off-chip memory clock delta | `device->offchip_dram_interface->time` (ns) |
| | DDR5 Bandwidth Util | $\frac{\text{Bytes Transferred}}{T_{\text{DDR5\_sim\_sec}} \times 51.2\text{ GB/s}}$ | Derived from simulated DDR5 time |
| **GPU Stalls** | Memory Stall Time | Exposed GPU wait time for DDR5 expert loads | `expert_load.csv` (`duration` where miss) |
| | Memory Stall Fraction | $\frac{T_{\text{stall}}}{T_{\text{sim}}}$ | Derived: $\approx 99.97\%$ |

---

## 8. Known Limitations

1. **Synthetic Routing:** As documented, expert selection uses `Scheduler::getRandomExpert` (uniform random from $[0, 255]$ with seed 777), not neural-router gate activations.
2. **Synchronous Demand Stalls:** The current simulator has no prefetch queue or asynchronous staging pipeline. Every cold expert load halts the GPU until transfer completes.
3. **Ramulator2 Simulation Overhead:** Cycle-accurate simulation of 88 MB per expert imposes $\sim 10$ seconds of host CPU runtime per cold expert, limiting high-iteration multi-layer sweeps.

---

## 9. Exact Execution Commands

### 9.1 Verification of Executable and Config Validation
```bash
cd /home/mohman/ca_project/LLMSimulator
./build/run_deepseekv3_mixtral_style reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/configs/deepseekv3_smoke_test.yaml --validate-config
```

### 9.2 Running an Experiment Profile
```bash
export BYPASS_OFFCHIP_EXECUTION_CACHE=1
./build/run_deepseekv3_mixtral_style <path_to_config.yaml>
```
All outputs are written to the strictly isolated directories configured in the YAML:
- `summary_csv`: `<output_directory>/`
- `expert_routing.csv`, `expert_load.csv`, `expert_cache.csv`: `<traces_directory>/`
- `completion_status.txt`: `<output_directory>/../logs/`

