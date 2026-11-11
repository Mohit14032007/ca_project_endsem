# Phase E1-C: Forensic Audit of YAML Configuration and Execution Semantics

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** E1-C Consolidated Forensic Audit  
**Status:** **AUDIT COMPLETE — VERIFIED**  
**Repository Path:** `/home/mohman/ca_project/LLMSimulator`  
**Target Executable:** `build/run_deepseekv3_mixtral_style`  
**Audit Scope:** Rigorous code-to-hardware semantic tracing of configuration parameters across Phase E1-A and Phase E1-B.

---

## 1. Executive Overview

This document provides a line-by-line source-code audit tracing how YAML configuration keys are ingested, parsed, and translated into hardware and scheduler operations within `LLMSimulator`. 

Every key investigated has been verified against the C++ source code (`eval/test_deepseekv3_mixtral_style.cpp`, `src/scheduler/scheduler.cpp`, `src/scheduler/sequence.cpp`, `src/model/llm.cpp`, `src/hardware/layer_impl.cpp`, and `src/hardware/device.cpp`).

---

## 2. Parameter Tracing and Source Verification

### 2.1 `simulation.iter`
- **YAML Definition:** `simulation.iter` (Integer, e.g. 1, 2)
- **C++ Parser:** [`eval/test_deepseekv3_mixtral_style.cpp:120`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L120)
  ```cpp
  int iter = config["simulation"]["iter"].as<int>();
  ```
- **Execution Path:**
  1. *Trace Validation:* In Profile B (`use_hitting_queue == false`), `iter` is passed to `Scheduler::loadRealTrace` as `iter_tokens` ([`eval/test_deepseekv3_mixtral_style.cpp:323`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L323)), asserting that the trace file possesses at least `iter` valid tokens per sequence.
  2. *Warm Preload Scope:* In warm replay mode ([`eval/test_deepseekv3_mixtral_style.cpp:344`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L344)):
     ```cpp
     for (int t = 0; t < iter; t++) { ... }
     ```
     `iter` dictates exactly how many forward decoding steps of expert working sets are pre-marked resident in HBM before simulation begins.
  3. *Execution Loop Control:* Controls the primary decoding loop ([`eval/test_deepseekv3_mixtral_style.cpp:371`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L371)):
     ```cpp
     for (int step = 0; step < iter; step++) {
       auto metadata = scheduler->setMetadata();
       cluster->run(metadata);
       ...
       scheduler->updateScheduler(step_device_time);
       scheduler->fillSequenceQueue(step_device_time, cumulative_device_time);
       scheduler->fillRunningQueue();
     }
     ```
- **Answers to Forensic Questions:**
  - *Is it the number of simulation iterations / experiment repetitions?* **NO.** It is NOT statistical repetitions. It is the number of **sequential auto-regressive decoding steps (tokens generated)**.
  - *Does it control request injection?* In legacy Profile A (`use_hitting_queue == true`), it sized the Poisson request table (`iter * 100`). In Profile B (E1-A and E1-B), initial requests are created explicitly before the loop; `iter` strictly bounds the step loop.
  - *Does it interact with `output_len`?* Yes. If `iter >= output_len`, sequences reach `total_len` and are popped from the queue.
  - *Does it reset the RNG or routing state?* **NO.** The RNG in `Scheduler::getRandomExpert` is a static generator (`static std::mt19937 generator(seed)`) that is never re-seeded across steps.
- **Verification Status:** **VERIFIED**

---

### 2.2 `model.num_layers`
- **YAML Definition:** `model.num_layers` (Integer, configured as 5 in E1-A and E1-B)
- **C++ Parser:** [`eval/test_deepseekv3_mixtral_style.cpp:215`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L215)
  ```cpp
  ModelConfig model_config = (model_name == "debugV3") ? debugV3 : deepseekV3;
  if (config["model"]["num_layers"]) {
    model_config.num_layers = config["model"]["num_layers"].as<int>();
  }
  ```
- **Execution Path:**
  1. *Model Architecture Instantiation:* In [`src/model/llm.cpp:43-54`](file:///home/mohman/ca_project/LLMSimulator/src/model/llm.cpp#L43-L54):
     ```cpp
     for (int layer = 0; layer < model_config.first_k_dense; layer++) {
       auto decoder = Decoder::Create(..., "decoder_" + std::to_string(layer), ...);
       add_module(decoder);
     }
     for (int layer = model_config.first_k_dense; layer < model_config.num_layers; layer++) {
       auto moe_decoder = MoEDecoder::Create(..., "MoE_decoder_" + std::to_string(layer), ...);
       add_module(moe_decoder);
     }
     ```
  2. *Logical MoE Stage Calculation:* DeepSeek-V3 has `first_k_dense = 3` ([`src/model/model_config.h:108`](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h#L108)).
     When `num_layers = 5`:
     $$\text{MoE Stages} = \text{num\_layers} - \text{first\_k\_dense} = 5 - 3 = 2$$
     Layers 0, 1, and 2 are standard dense MLA/FFN Transformer decoders.
     Layers 3 and 4 are MoE decoders containing 256 routed experts + 1 shared expert.
  3. *Scheduler Stage Indexing:* In [`src/scheduler/sequence.cpp:158`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L158):
     `num_layer = scheduler->model_config.num_layers - scheduler->model_config.first_k_dense;`
     Logical MoE stage IDs within the MoE subsystem map to index $0$ (Layer 3) and index $1$ (Layer 4).
- **Answers to Forensic Questions:**
  - *Does `num_layers: 5` mean 5 MoE stages?* **NO.** It means 5 total Transformer layers, comprising **3 dense layers and 2 MoE stages**.
  - *Is the model truncated?* Yes. Full DeepSeek-V3 comprises 60 Transformer layers (3 dense + 57 MoE stages). The test harness instantiates a 5-layer truncation (3 dense + 2 MoE stages) to allow cycle-accurate Ramulator DRAM simulation within viable compute budgets while exercising both dense and MoE layer types.
- **Verification Status:** **VERIFIED**

---

### 2.3 `simulation.input_len` and `simulation.output_len`
- **YAML Definitions:** `simulation.input_len` (e.g. 128, 256, 512), `simulation.output_len` (e.g. 2, 3)
- **C++ Parser:** [`eval/test_deepseekv3_mixtral_style.cpp:117-118`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L117-L118)
- **Execution Path:**
  1. *Sequence Instantiation:* In [`src/scheduler/sequence.cpp:14`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L14):
     ```cpp
     total_len = input_len + output_len - 1;
     ```
  2. *Decode-Mode Offset:* In [`eval/test_deepseekv3_mixtral_style.cpp:327`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L327):
     ```cpp
     if (system_config.decode_mode) {
       new_seq->current_len = input_len;
     }
     ```
- **Answers to Forensic Questions:**
  - *Is the input context actually processed through prefill?* **NO.** `system_config.prefill_mode = false` and `system_config.decode_mode = true`. The `input_len` parameter is used exclusively to establish the starting KV-cache history length for Attention operations (`stat.average_seq_len = input_len + step + 1`).
  - *What does `output_len` control?* It bounds sequence lifetime. A sequence terminates when `current_len == total_len`. In single-step experiments, `output_len = 2` ensures the sequence remains active during step 0.
  - *How many routing events occur per generated token?*
    $$\text{Routing Events per Token} = \text{MoE Stages} \times \text{TopK} = 2 \times 8 = 16$$
    Across an entire batch of size $B$:
    $$\text{Total Routing Decisions per Step} = B \times 16$$
    - For $B=1$: 16 routing decisions.
    - For $B=2$: 32 routing decisions.
    - For $B=4$: 64 routing decisions.
    - For $B=8$: 128 routing decisions.
- **Verification Status:** **VERIFIED**

---

### 2.4 `serving.max_batch_size` and `simulation.injection_rate`
- **YAML Definitions:** `serving.max_batch_size` (Integer: 1, 2, 4, 8), `simulation.injection_rate` (Float: 0)
- **C++ Parser:** [`eval/test_deepseekv3_mixtral_style.cpp:109`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L109)
- **Execution Path:**
  1. *Initial Sequence Creation:* In [`eval/test_deepseekv3_mixtral_style.cpp:324`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L324):
     ```cpp
     for (int b = 0; b < max_batch_size; b++) {
       Sequence::Ptr new_seq = Sequence::Create(b, input_len, output_len);
       scheduler->sequence_queue.push_back(new_seq);
     }
     ```
     `max_batch_size` directly creates $B$ distinct sequence instances.
  2. *Closed-Loop Serving via `injection_rate: 0`:* In [`src/scheduler/scheduler.cpp:472`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L472):
     ```cpp
     if (system_config.use_inject_rate) {
       ... // Poisson arrival process
     } else {
       num_request_to_inject = total_batch_size - getBatchSize();
     }
     pushSeq(num_request_to_inject);
     ```
     An `injection_rate` of 0 indicates that Poisson request arrival is disabled (`system_config.use_inject_rate = false`). The scheduler operates in **closed-loop mode**, immediately refilling the batch queue up to `max_batch_size` whenever a sequence completes.
- **Answers to Forensic Questions:**
  - *Does `max_batch_size` create requests or merely limit them?* In the test harness, it directly instantiates the initial active request pool ($B$ sequences).
  - *How were B=1, 2, 4, 8 produced?* Explicitly configured via `serving.max_batch_size: B`, instantiating $B$ concurrent sequence objects in `scheduler->running_queue`.
- **Verification Status:** **VERIFIED**

---

## 3. Comprehensive YAML Parameter Reference Table

| Parameter | E1-A Value | E1-B Value | Source-Code Implementation & Meaning | Actual Observed Effect | Status |
|---|---|---|---|---|---|
| `simulation.iter` | 1 (2 for LRU) | 1 (2 for LRU) | `test_deepseekv3_mixtral_style.cpp:120, 371`: Loop bound for sequential decoding steps. | Executes exactly $N$ sequential decoding tokens per active sequence. | **VERIFIED** |
| `model.model_name` | `deepseekV3` | `deepseekV3` | `test_deepseekv3_mixtral_style.cpp:213`: Selects DeepSeek-V3 architectural preset. | Configures $d_{\text{model}}=7168$, $E=256$, Top-8, 1 shared expert, MLA attention. | **VERIFIED** |
| `model.num_layers` | 5 | 5 | `test_deepseekv3_mixtral_style.cpp:215`: Total Transformer layers. | Instantiates 3 dense MLA/FFN decoders and 2 MoE decoders. | **VERIFIED** |
| `serving.max_batch_size` | 1, 2, 4, 8 | 1, 2, 4 | `test_deepseekv3_mixtral_style.cpp:109, 324`: Instantiates concurrent sequences. | Populates running queue with $B$ concurrent request streams. | **VERIFIED** |
| `serving.max_process_token` | 0 | 0 | `test_deepseekv3_mixtral_style.cpp:111`: Token budget threshold. | Defaults to $65536 \times 8 = 524,288$ tokens (unconstrained batch execution). | **VERIFIED** |
| `simulation.input_len` | 128, 256, 512, 1024 | 128, 256, 512 | `test_deepseekv3_mixtral_style.cpp:117`: Initial sequence context offset. | Sizes KV cache history; does not trigger prefill forward pass. | **VERIFIED** |
| `simulation.output_len` | 2 (3 for LRU) | 2 (3 for LRU) | `test_deepseekv3_mixtral_style.cpp:118`: Total generation length threshold. | Keeps sequence alive across the $N$ decoding steps. | **VERIFIED** |
| `simulation.injection_rate` | 0 | 0 | `scheduler.cpp:457`: Disables Poisson arrival process. | Operates in closed-loop batch serving mode (`use_inject_rate = false`). | **VERIFIED** |
| `simulation.warm_hbm_preload` | `false` / `true` | `false` / `true` | `test_deepseekv3_mixtral_style.cpp:339`: Pre-populates HBM cache metadata. | Cold: 0 resident experts; Warm: 16 (or 20) resident experts marked before step 0. | **VERIFIED** |
| `simulation.trace_file` | `""` | `exported_traces/*.csv` | `test_deepseekv3_mixtral_style.cpp:322`: File path to real MoE trace. | When populated, loads genuine router expert selections into `SequenceInfo`. | **VERIFIED** |
| `system.optimization.expert_cache_capacity_bytes` | 80000000000 | 80000000000 | `device.h:127`: Maximum resident bytes in HBM LRU tracker. | Allocates 80 GB capacity budget ($80 \times 10^9$ bytes = 74.51 GiB). | **VERIFIED** |
| `system.optimization.hbm_dram_cfg_path` | `dram_config_HBM3_80GB.yaml` | `dram_config_HBM3_80GB.yaml` | `test_deepseekv3_mixtral_style.cpp:184`: Memory configuration for local HBM. | Configures 5 HBM3 cubes, 32 pseudo-channels, 2,039 GB/s peak bandwidth. | **VERIFIED** |
| `system.optimization.offchip_dram_cfg_path` | `dram_config_DDR5_retry.yaml` | `dram_config_DDR5_retry.yaml` | `test_deepseekv3_mixtral_style.cpp:183`: Memory configuration for off-chip DDR5. | Configures 2-channel DDR5-3200BN, 51.2 GB/s peak bandwidth. | **VERIFIED** |
| `system.optimization.use_ramulator` | `on` | `on` | `test_deepseekv3_mixtral_style.cpp:182`: Enables Ramulator cycle-level simulation. | Passes memory requests to Ramulator2 engine. | **VERIFIED** |
| `system.optimization.bypass_offchip_execution_cache` | `true` | `true` | `test_deepseekv3_mixtral_style.cpp:200`, `layer_impl.cpp:80`: Disables memoization. | Enforces full cycle-accurate DRAM simulation for every DDR5 access. | **VERIFIED** |
| `log.export_gantt` | `false` | `false` | `test_deepseekv3_mixtral_style.cpp:137`: Disables trace visualization dump. | Suppresses multi-gigabyte raw Gantt log creation. | **VERIFIED** |

