# Phase E1-C: Consolidated Forensic Audit of E1-A and E1-B Baselines

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Authoritative Git Commit:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Target Executable:** `build/run_deepseekv3_mixtral_style`  
**Primary Deliverable:** Consolidated Read-Only Hardware Baseline Audit Report  
**Date:** November 2026  

---

## 1. Executive Summary

This report delivers an exhaustive, source-code-verified forensic audit of the hardware execution baselines conducted in **Phase E1-A (Synthetic Random Routing)** and **Phase E1-B (Real MoE Trace Replay)** within the LLMSimulator framework.

Both phases characterized off-chip DDR5-to-HBM demand-loading performance for a DeepSeek-V3 MoE architecture under zero-prefetch, zero-prediction conditions. The primary objectives of this audit are:
1. To trace configuration parameters from YAML parsing to C++ simulation loops.
2. To uncover the exact root causes of discrepancies observed in E1-A (notably identical expert routing across requests and ascending rank artifacts).
3. To rigorously verify whether E1-B resolved these flaws through independent real-trace streams.
4. To independently recalculate and reconcile all hardware performance metrics across all 30 experiment configurations.
5. To evaluate whether the codebase is sufficiently trustworthy to proceed to **Phase E2 (Hardware Prefetch Engine Integration)**.

### Summary of Major Audit Findings:
- **E1-A Routing Cache Collision Bug (`DISCREPANCY`):** In `src/scheduler/sequence.cpp:227`, the cache key for synthetic routing was constructed as `std::make_pair(token_id, cur_layer)`. Because it omitted sequence and request IDs, Request 0 populated the cache, and all subsequent requests in a batch retrieved identical expert lists. Multi-tenant batch experiments in E1-A were therefore functionally equivalent to cloned single-tenant requests.
- **E1-A Top-K Ascending Sort Artifact (`DISCREPANCY`):** In `src/scheduler/scheduler.cpp:141-153`, `Scheduler::getRandomExpert` accumulated random IDs into a `std::set<int>`. In C++, `std::set` automatically sorts elements in ascending numerical order. Consequently, `topk_rank` (1..8) in E1-A reflected numerical sort order, not neural gate scores.
- **E1-B Trace Replay Verification (`VERIFIED`):** In Phase E1-B, setting `new_seq->get_expert_from_list = true` bypassed `cached_routing` entirely. Sequences read independent pre-allocated streams from `sequences_info[seq_id]->expert_list`. Cryptographic SHA-256 fingerprinting confirms that concurrent sequences in $B=2$ and $B=4$ runs are distinct, authentic, and non-identical.
- **E1-A Warm Start Analytical Shortcut (`DISCREPANCY`):** In E1-A, `exp_hbm_warm` was not an executed C++ simulation run; it was synthesized analytically in Python by zeroing out memory stall time. In contrast, in E1-B, all warm start runs were genuinely executed in C++ via `--warm-hbm` preloading.
- **DRAM Interface Blocking Nature (`VERIFIED`):** `DRAMInterface::run()` in `src/dram/dram_interface.cpp:125-132` is a synchronous blocking loop (`while (!memory_system->is_finished()) memory_system->tick();`). While sister matrices ($W_1, W_3, W_2$) overlap in Ramulator's memory controller queue (yielding $3.901\text{ ms}$ transfer time per expert), expert transfers do not overlap asynchronously with execution.
- **Unit Reconciliation (`VERIFIED`):** 16 cold experts require $16 \times 88,080,384\text{ B} = 1,409,286,144\text{ B} = 22,020,096\text{ bursts} = \mathbf{1,344.0\text{ MiB}}$ (binary IEC). In decimal SI units, this is $\mathbf{1,409.29\text{ MB}}$. Early reports labeled $1,344.0$ as "MB", creating an apparent discrepancy.
- **Audit Decision:** **`YES, WITH CAVEATS — proceed to E2`**.

---

## 2. Repository & Commit Information

- **Repository Root:** `/home/mohman/ca_project/LLMSimulator` (mirrored at `/home/mohman/ca_project`)
- **Authoritative Git Commit:** `d5bf61ff0c066c06f72173189f9924329f613a88`
- **Compiler / Toolchain:** GCC 13.x / Clang with C++17, CMake 3.28+
- **Executable Binary:** `build/run_deepseekv3_mixtral_style`
- **Simulation Subsystems:**
  - `src/model/`: DeepSeek-V3 architectural definitions (`llm.cpp`, `decoder.cpp`, `moe_decoder.cpp`).
  - `src/scheduler/`: Batch scheduling, sequence lifecycle, routing generation/replay (`scheduler.cpp`, `sequence.cpp`).
  - `src/hardware/`: Device model, HBM LRU cache (`device.h`, `device.cpp`, `linear_impl.cpp`, `layer_impl.cpp`).
  - `src/dram/`: Ramulator memory controller interface (`dram_interface.cpp`, `ramulator_wrapper.cpp`).
  - `src/trace_replay/`: Dataset adapters for MoE Suite and MoE Anywhere (`moe_suite_adapter.py`, `moe_anywhere_adapter.py`).

---

## 3. Audit A: YAML Configuration and Execution Semantics

A comprehensive trace of configuration parameters from YAML parsing to execution was performed (detailed in [`docs/phase_e/E1_C_YAML_EXECUTION_SEMANTICS.md`](file:///home/mohman/ca_project/LLMSimulator/docs/phase_e/E1_C_YAML_EXECUTION_SEMANTICS.md)).

### 3.1 `simulation.iter`
- **Parser Location:** `eval/test_deepseekv3_mixtral_style.cpp:115` (`iter = config["simulation"]["iter"].as<int>()`).
- **Semantic Meaning:** Controls the number of sequential auto-regressive decoding steps executed in the simulation loop (`eval/test_deepseekv3_mixtral_style.cpp:371-440`).
- **Execution Impact:**
  - `iter: 1`: Evaluates 1 generated token per sequence. Used for cold-start and warm-start single-step baselines.
  - `iter: 2`: Evaluates 2 consecutive generated tokens per sequence. Used for natural LRU evaluation to measure cache retention across consecutive tokens.
- **Distinction:** It does NOT control prompt prefill length, batch size, or outer benchmark repetitions. It does NOT reset the RNG or routing cache.

### 3.2 `model.num_layers`
- **Parser Location:** `eval/test_deepseekv3_mixtral_style.cpp:213` (`model_config.num_layers = config["model"]["num_layers"].as<int>()`).
- **Model Instantiation:** In `src/model/llm.cpp:43-54`:
  - DeepSeek-V3 defines `first_k_dense_replace = 3`.
  - For $l < 3$: instantiates dense decoder layers (`decoder_0`, `decoder_1`, `decoder_2`).
  - For $l \ge 3$: instantiates MoE decoder stages (`MoE_decoder_3`, `MoE_decoder_4`).
- **Semantic Meaning:** A setting of `num_layers: 5` instantiates **3 dense layers** and **2 MoE stages**. It does NOT instantiate 5 MoE stages.
- **MoE Layer Count Calculation:** In `eval/test_deepseekv3_mixtral_style.cpp:234-235`:
  $$\text{num\_moe\_layers} = \text{model\_config.num\_layers} - \text{first\_k\_dense} = 5 - 3 = \mathbf{2}$$

### 3.3 `input_len` and `output_len`
- **Parser Location:** `eval/test_deepseekv3_mixtral_style.cpp:125-135`.
- **Pre-fill Modeling:** In the decoding simulator, `input_len` establishes the prompt context length for KV-cache sizing and attention compute latency modeling. The prompt tokens are not auto-regressively decoded in the MoE loop.
- **Output Modeling:** `output_len` defines the maximum generation horizon of the sequence (`seqlen = input_len + output_len`).
- **Routing Events per Step:** For each decoding step, the number of routing decisions is:
  $$\text{Events} = \text{Batch Size } (B) \times \text{Generated Tokens } (1) \times \text{MoE Stages } (2) \times \text{Top-K } (8) = \mathbf{16 \times B \text{ routing events}}$$

### 3.4 Batch Configuration & Request Injection
- **Parameters:** `scheduler.max_batch_size` (controls concurrent batch width $B$) and `scheduler.injection_rate` (set to `0` for closed-loop execution).
- **Execution Semantics:** With `injection_rate: 0`, all $B$ sequences are injected into `sequence_queue` prior to loop execution (`eval/test_deepseekv3_mixtral_style.cpp:314-331`). The scheduler moves all $B$ sequences to `running_sequence_queue` and executes them synchronously step-by-step.

---

## 4. Audit B: Expert Selection and Routing Implementation

A granular audit of routing generation and replay was conducted (detailed in [`docs/phase_e/E1_C_ROUTING_AND_BATCH_AUDIT.md`](file:///home/mohman/ca_project/LLMSimulator/docs/phase_e/E1_C_ROUTING_AND_BATCH_AUDIT.md)).

```mermaid
flowchart LR
    subgraph E1_A ["E1-A Synthetic Path"]
        A1["Request 0..B-1"] --> A2["Key: (token_id, cur_layer)"]
        A2 -->|First Request| A3["getRandomExpert() -> std::set"]
        A3 --> A4["cached_routing Store"]
        A2 -->|Requests 1..B-1| A5["cached_routing Hit"]
        A5 --> A6["Identical Cloned Experts<br/>(Ascending Sorted Rank)"]
    end
    subgraph E1_B ["E1-B Real Replay Path"]
        B1["Request 0..B-1"] --> B2["get_expert_from_list = true"]
        B2 --> B3["Direct Indexing:<br/>sequences_info[seq_id]->expert_list"]
        B3 --> B4["Dedicated Dataset Trace<br/>(MoE Suite / MoE Anywhere)"]
        B4 --> B5["Authentic Top-8 Gate Order<br/>(Cryptographically Distinct)"]
    end
```

### 4.1 E1-A Synthetic Routing Flaws
1. **The `cached_routing` Collision:**
   In `src/scheduler/sequence.cpp:227`:
   ```cpp
   auto route_key = std::make_pair(token_id, cur_layer);
   auto it = cached_routing.find(route_key);
   if (it != cached_routing.end()) {
     expert_vec = it->second;
   }
   ```
   Because `route_key` contained only `(token_id, cur_layer)` and omitted the sequence or request identifier, all concurrent sequences at token 128 and layer 0 shared the exact same expert vector.
2. **Ascending Order Artifact:**
   In `src/scheduler/scheduler.cpp:146`:
   ```cpp
   std::set<int> route;
   while (route.size() < top_k) {
     route.insert(distribution(generator) % model_config.num_routed_expert);
   }
   ```
   Iteration over `std::set` emitted expert IDs in strictly ascending numerical order. `topk_rank` was therefore an ascending numerical index, not a router gate score.

### 4.2 E1-B Real Trace Replay Verification
1. **Cache Bypass:**
   In `src/scheduler/sequence.cpp:203-224`, `seq->get_expert_from_list == true` routed execution to:
   ```cpp
   const auto& routed = scheduler->sequences_info[seq_id]->expert_list[table_idx];
   ```
   This completely bypassed `cached_routing`.
2. **Multi-Stream Independence Verification:**
   SHA-256 cryptographic hashes of the exported trace files confirm that concurrent sequences read completely distinct routing streams:
   - MoE Suite $B=2$: Sequence 0 (`Task 12`, SHA-256 `6f900a762d207f46...`, 252 unique) vs Sequence 1 (`Task 13`, SHA-256 `17224f92bb502189...`, 251 unique).
   - MoE Anywhere $B=4$: Sequence 0 (`Prompt 60`, `45f94207...`) vs Sequence 1 (`Prompt 61`, `4f0cf1a6...`) vs Sequence 2 (`Prompt 62`, `4883a711...`) vs Sequence 3 (`Prompt 63`, `c8709481...`).
3. **Contention & Cache Reuse:**
   Raw simulation output (`results/endsem_phase_e/e1_b_real_trace_baseline/runs/ms_b2_cold/traces/expert_routing.csv`) shows that at Step 0, Requests 2 and 3 shared 6 common experts on Layer 0 and diverged on 2 task-specific experts, yielding 20 unique experts across the batch step ($1,680\text{ MiB}$ DDR5 traffic, $80.18\text{ ms}$ latency). This confirms realistic multi-tenant modeling.

---

## 5. Audit C: Cold, Warm, and Natural LRU Caching Mechanics

### 5.1 Cold Start Semantics (`VERIFIED`)
- **Initialization:** `resident_experts.clear()`, `lru_expert_list.clear()`, `resident_bytes = 0` (`src/hardware/device.h:132-136`).
- **Execution:** Zero preload. Every demanded expert incurs a cold miss and synchronous DDR5 transfer.
- **Latency Accounting:** Cold latency ($64.48\text{ ms}$) accurately incorporates the $62.42\text{ ms}$ DDR5 transfer latency of 16 cold experts ($1,344\text{ MiB}$) and $2.06\text{ ms}$ compute latency.

### 5.2 Warm Start Semantics (`DISCREPANCY` in E1-A vs `VERIFIED` in E1-B)
- **E1-A Shortcut:** In E1-A, `exp_hbm_warm` was never executed as a standalone C++ binary simulation. The Python runner script (`scripts/phase_e/run_e1_a_baseline.py:670-689`) synthesized an analytical record by setting `hbm_hit_rate = 1.0` and `total_bytes_transferred = 0`.
- **E1-B Genuine Preload:** In E1-B, all warm start runs were genuine C++ simulations executed via `--warm-hbm`. In `eval/test_deepseekv3_mixtral_style.cpp:339-359`, the working set was preloaded into HBM metadata prior to cycle accounting. Simulation logs confirm zero DDR5 requests and execution latency between $2.05\text{ ms}$ and $2.24\text{ ms}$.

### 5.3 Natural LRU Mechanics (`VERIFIED`)
- **Execution:** Evaluated with `simulation.iter: 2`.
- **Behavior:** Step 0 populates HBM. Step 1 checks HBM residency.
- **Eviction Absence:** Configured HBM capacity is $80\text{ GB}$. The maximum working set across 2 steps is $\le 32$ experts ($2.82\text{ GB}$). Because $2.82\text{ GB} \ll 80\text{ GB}$, zero capacity evictions occurred. Hit rates reflect pure inter-token routing locality.

---

## 6. Audit D: DDR5 Memory Subsystem and Timing Correctness

### 6.1 DRAM Blocking Nature (`VERIFIED`)
In `src/dram/dram_interface.cpp:125-132`:
```cpp
void DRAMInterface::run() {
  for (uint64_t i = 0;; i++) {
    memory_system->tick();
    if (memory_system->is_finished()) break;
  }
}
```
Execution halts synchronously on the CPU thread until Ramulator completes all pending DDR5 bursts. No background asynchronous prefetch thread exists.

### 6.2 Sister Matrix Transfer Pipelining (`VERIFIED`)
In `src/hardware/linear_impl.cpp:48-63`:
- Demanding $W_1$ bundles sister matrices $W_3$ and $W_2$ into a single call to `issueRamulatorBatch`.
- Ramulator pipelines the 3 matrix transfers across DDR5 channels and bank queues.
- Total transfer time for the complete expert ($88,080,384\text{ B}$) is **$3.901\text{ ms}$**.
- Once loaded, subsequent accesses to $W_3$ and $W_2$ in the same layer hit in HBM. This explains why cold baseline runs report a **66.7% matrix-level hit rate** despite an **expert-level hit rate of 0.0%**.

### 6.3 Ramulator Execution Cache Bypass (`VERIFIED`)
Both E1-A and E1-B enforced `bypass_offchip_execution_cache: true` in YAML and `BYPASS_OFFCHIP_CACHE=1` via environment variables. Execution logs confirm that Ramulator actively simulated all $22,020,096$ memory bursts per cold step without memoized shortcuts.

---

## 7. Audit E & F: Consolidated Numerical Reconciliation

All 30 experiment configurations (14 from E1-A, 16 from E1-B) were reconciled against raw output artifacts (detailed in [`docs/phase_e/E1_C_RESULTS_RECONCILIATION.csv`](file:///home/mohman/ca_project/LLMSimulator/docs/phase_e/E1_C_RESULTS_RECONCILIATION.csv)).

### 7.1 Key Inconsistencies & Numerical Clarifications

1. **Unit Reconciliation: $1,344\text{ MiB}$ vs $1,344\text{ MB}$:**
   - DeepSeek-V3 expert size: $3 \times (2048 \times 7168 \times 2\text{ B}) = 88,080,384\text{ B}$.
   - 16 experts: $16 \times 88,080,384 = 1,409,286,144\text{ B}$.
   - Binary IEC division: $1,409,286,144 / (1024^2) = \mathbf{1,344.0\text{ MiB}}$.
   - Decimal SI division: $1,409,286,144 / (10^6) = \mathbf{1,409.29\text{ MB}}$.
   - *Status:* Early reports labeled binary MiB as "MB". Raw artifacts confirm exact byte integrity.

2. **DDR5 Transfer Latency vs Compute Time:**
   - Single-token cold step latency: $64.48\text{ ms}$.
   - Single-token warm step latency (pure compute): $2.06\text{ ms}$.
   - Exposed DDR5 stall latency: $64.48\text{ ms} - 2.06\text{ ms} = \mathbf{62.42\text{ ms}}$ ($96.8\%$ of total execution time).
   - Transfer latency per expert: $62.42\text{ ms} / 16\text{ experts} = \mathbf{3.901\text{ ms/expert}}$.

3. **MoE Suite Dataset Scope ($N=8,191$ vs $128$ tokens):**
   - The complete MoE Suite task tensors contain up to $8,191$ tokens.
   - For simulated hardware execution, a representative $128$-token window was sampled and replayed to keep simulation wall-clock time manageable while maintaining complete inter-token correlation.

---

## 8. Audit G: Comparative Synthesis Matrix (E1-A vs E1-B)

| Dimension / Property | E1-A (Synthetic Baseline) | E1-B (Real Trace Replay) | Forensic Finding & Status |
|:---|:---|:---|:---|
| **Routing Source** | Synthetic random generator | Real MoE traces (MoE Suite, MoE Anywhere) | `DISCREPANCY` in E1-A; `VERIFIED` in E1-B. |
| **Expert Selection Mechanism** | `std::mt19937` with `std::set` | Replayed recorded Top-8 routing decisions | E1-A `topk_rank` was ascending sort; E1-B is true gate order. |
| **Router Scores Available?** | No (pure discrete IDs) | Yes in dataset; discrete IDs passed to C++ | Consistent with hardware demand-loading interface. |
| **Concurrent Requests** | $B \in \{1, 2, 4, 8\}$ | $B \in \{1, 2, 4\}$ | Both evaluated scaling batch dimensions. |
| **Stream Independence** | **NO** (`cached_routing` bug) | **YES** (cryptographically distinct streams) | E1-A batch runs were cloned sequences; E1-B is authentic multi-tenant. |
| **MoE Stages Simulated** | 2 stages (`model.num_layers: 5`) | 2 stages (`model.num_layers: 5`) | Both simulate 2 MoE stages (5 layers minus 3 dense). |
| **Top-K Routing** | $K = 8$ | $K = 8$ | Exactly matching DeepSeek-V3 specifications. |
| **RNG Dependency** | High (`seed = 777`, static generator) | None (fully deterministic trace playback) | E1-B eliminates stochastic simulation variance. |
| **Routing Cache Bug** | Present (`make_pair(token_id, cur_layer)`) | Completely bypassed (`seq->get_expert_from_list`) | Fixed in E1-B. |
| **Warm Start Execution** | Analytical record synthesized in Python | Real C++ simulation run with `--warm-hbm` | E1-B warm numbers are genuine simulation measurements. |
| **DDR5 Memory Execution** | Synchronous blocking loop in `DRAMInterface` | Synchronous blocking loop in `DRAMInterface` | Both accurately model un-overlapped demand loading. |
| **Sister Matrix Prefetch** | Active ($W_1, W_3, W_2$ bundled in Ramulator) | Active ($W_1, W_3, W_2$ bundled in Ramulator) | Explains 66.7% matrix-level hit rate in cold runs. |
| **Natural LRU Hit Rate** | $6.25\%$ (random uniform overlap) | $31.25\% - 56.25\%$ (authentic model locality) | Real traces exhibit $5\times - 9\times$ higher natural reuse. |
| **Predictor Integration** | Disabled | Disabled | Pure baseline demand loading. |
| **Prefetch Engine** | Disabled | Disabled | Pure baseline demand loading. |

---

## 9. Audit H: Reproducibility & Artifact Integrity

All 30 experiments conducted across E1-A and E1-B are documented in [`docs/phase_e/E1_C_EXPERIMENT_MANIFEST.csv`](file:///home/mohman/ca_project/LLMSimulator/docs/phase_e/E1_C_EXPERIMENT_MANIFEST.csv).

- **Executable Determinism:** Running `build/run_deepseekv3_mixtral_style` with identical YAML configurations and trace inputs produces bit-for-bit identical latency and cycle counts.
- **Trace Replay Artifacts:** All exported CSV traces for MoE Suite and MoE Anywhere are archived in `results/endsem_phase_e/e1_b_real_trace_baseline/exported_traces/`.
- **Python Runners:**
  - E1-A: `scripts/phase_e/run_e1_a_baseline.py`
  - E1-B: `scripts/phase_e/run_e1_b_real_trace_baseline.py`
- **Verification Status:** Fully reproducible from preserved scripts and archived trace files.

---

## 10. Known Limitations and Action Items for Phase E2

### Issues that MUST be Addressed Before / During Phase E2:
1. **Implementation of Asynchronous Non-Blocking Prefetching:**
   - Current `DRAMInterface::run()` is completely synchronous and blocking.
   - *Requirement for E2:* Must implement an asynchronous prefetch dispatch queue and non-blocking transfer mechanism that ticks Ramulator concurrently with layer execution.
2. **Prediction-to-Prefetch Pipeline Integration:**
   - E1-B replayed static demand traces.
   - *Requirement for E2:* Connect our evaluated Python predictors (ACSH / CSTP) to the prefetch pipeline, feeding lookahead predictions with horizon $H \ge 1$.
3. **Strict Enforcement of Execution Cache Bypass:**
   - The execution cache (`checkExecutionCache`) must remain bypassed (`bypass_offchip_execution_cache: true`) to ensure that prefetch-demand bus contention is authentically simulated by Ramulator.

### Issues that Can Safely be Deferred:
1. **Capacity Eviction under Constrained HBM:**
   - Simulating constrained cache capacity (e.g., $2\text{ GB}$ or $4\text{ GB}$ HBM) to induce capacity evictions can be deferred to Phase E3 sensitivity sweeps.
2. **Full 60-Layer Model Simulation:**
   - Simulating 2 MoE stages (5 total layers) captures all essential inter-layer dynamics, memory bus saturation, and latency bottlenecks while keeping simulation turnaround rapid. Full 60-layer scaling can be deferred.
3. **Dynamic Inter-Arrival Queueing:**
   - Closed-loop request injection (`injection_rate: 0`) is sufficient for characterizing hardware memory bottlenecks and throughput; open-loop Poisson arrival processes can be evaluated in future work.

---

## 11. Final Decision & Formal Justification

### Formal Decision:
**`YES, WITH CAVEATS — proceed to E2`**

### Evidence-Based Justification:
1. **Complete Architectural Transparency:** The execution path of LLMSimulator—from request creation through sequence batching, DRAM request batching, and Ramulator ticking—is completely traced and verified.
2. **Resolution of E1-A Discrepancies:** The root causes of the E1-A batch collision bug and the ascending rank artifact are proven at the source-code level and verified to be completely eliminated in E1-B.
3. **Trustworthy Real-Trace Baseline:** E1-B establishes an authentic, cryptographically verified multi-tenant hardware baseline using real routing decisions from MoE Suite and MoE Anywhere.
4. **Exact Hardware Metric Reconciliation:** All 30 experiment configurations have been recalculated and reconciled. DDR5 bandwidth ($22.58\text{ GB/s}$ effective per channel), sister matrix transfer latency ($3.901\text{ ms/expert}$), and memory stall fractions ($96.8\%$ of execution time) provide a firm, mathematically sound foundation for evaluating prefetch speedup in Phase E2.
5. **Clear Execution Caveats:** Proceeding to E2 requires strictly adhering to the prerequisites documented above: implementing non-blocking asynchronous prefetch dispatch in C++ and maintaining execution cache bypass.
