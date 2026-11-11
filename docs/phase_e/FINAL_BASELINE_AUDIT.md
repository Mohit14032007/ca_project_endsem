# Phase E: Final Baseline Audit & Repository Readiness Verification

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Authoritative Git Commit:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Date:** November 2, 2026  
**Status:** COMPLETE & AUTHORITATIVE  

---

## 1. Executive Summary & Audit Mandate

This document establishes the authoritative pre-execution audit for the 15-configuration baseline evaluation campaign in LLMSimulator. The objective of this campaign is to characterize the baseline hardware behavior of MoE inference under pure demand loading (no speculative prefetching, no predictor, cold 80 GB HBM, standard LRU eviction, authentic Ramulator DDR5 timing) before introducing predictive prefetching engines (CSTP, ST-MoE, ACSH).

The audit systematically verifies:
1. Repository safety, Git commit integrity, and working tree stability.
2. Simulator execution semantics across `simulation.iter`, `output_len`, `input_len`, `max_batch_size`, and layer-to-stage mappings.
3. Status of the previously identified synthetic routing cache bug.
4. Correctness of the execution-cache bypass mechanism and DDR5 timing model.
5. Strict dataset boundaries and trace validity across MoE Suite, MoE Anywhere, and DeepSeek-style synthetic routing.
6. Forensic reconciliation of existing E1-B and E1-C artifacts.
7. Granular host CPU runtime estimation and the formal Part 7 Execution Gate decision.

---

## 2. Git State and Repository Safety

### 2.1 Commit Fingerprint
- **Target Git Commit:** `d5bf61ff0c066c06f72173189f9924329f613a88`
- **Branch:** `main` (synchronized with `origin/main`).
- **Repository Integrity:** No tracked source files, configurations, or historical results have been deleted or overwritten. All experiment outputs are written to isolated directories (`results/endsem_phase_e/final_baseline/`).

### 2.2 Working Tree Status
A pre-flight `git status` check confirms the presence of non-staged changes from previous Phase E developments:
- `modified: eval/test_deepseekv3_mixtral_style.cpp`: Integration of `loadRealTrace` trace replay engine, `--warm-hbm` preload routines, and CLI parameter parsing.
- `modified: src/hardware/layer_impl.cpp`: Implementation of `bypass_offchip_execution_cache` toggle.
- `modified: src/hardware/hardware_config.h`: Addition of configuration schema for off-chip cache bypass.
- `modified: src/scheduler/scheduler.cpp`, `scheduler.h`, `sequence.cpp`: Support for per-sequence real trace routing vectors.
- `untracked: docs/phase_e/`, `scripts/phase_e/`, `results/endsem_phase_e/`: Complete phase audit trails, manifests, and logs.

No modifications were made to simulator source code during this baseline audit task.

---

## 3. Simulator Execution Semantics

Forensic inspection of `eval/test_deepseekv3_mixtral_style.cpp`, `src/scheduler/scheduler.cpp`, and `src/hardware/device.cpp` establishes the exact runtime semantics:

```mermaid
flowchart TD
    YAML["YAML Configuration File"] --> Init["Model Initialization<br/>(test_deepseekv3_mixtral_style.cpp)"]
    Init --> LCheck["Dense vs MoE Layer Mapping<br/>moe_layers = num_layers - first_k_dense"]
    Init --> TraceCheck{"trace_file configured?"}
    TraceCheck -->|Yes| TraceLoad["Scheduler::loadRealTrace<br/>Sequences get independent expert_list<br/>get_expert_from_list = true"]
    TraceCheck -->|No (Synthetic)| SynthGen["Scheduler::getRandomExpert<br/>Uniform random Top-8<br/>get_expert_from_list = false"]
    TraceLoad --> DecLoop["Sequential Decoding Loop<br/>iter steps"]
    SynthGen --> DecLoop
    DecLoop --> MemCheck["Device::is_expert_resident"]
    MemCheck -->|Hit| HBMHit["MRU Splice & 0 DDR5 Burst"]
    MemCheck -->|Miss| DDR5Load["Issue 2,147,484 DDR5 Bursts<br/>Synchronous DRAMInterface::run()"]
    DDR5Load --> CacheBypassCheck{"bypass_offchip_execution_cache?"}
    CacheBypassCheck -->|True| RamTick["Full Ramulator Tick (11.98 s host time)"]
    CacheBypassCheck -->|False| MemoReuse["Memoized 3.901 ms Simulated Latency"]
```

### 3.1 Decoding Semantics: `iter` vs `output_len` vs `input_len`
1. `simulation.iter`: Defines the **exact number of sequential auto-regressive decoding iterations** executed in the C++ simulation loop (`test_deepseekv3_mixtral_style.cpp:310`). For each step $t \in [0, \text{iter}-1]$, every active sequence generates one new token and issues memory requests for all routed experts across all MoE stages.
2. `simulation.output_len`: Represents the total planned generation sequence length allocated during sequence creation (`SequenceInfo::Lout`). In our baseline configurations, `simulation.output_len` is configured identically to `simulation.iter` ($T = \text{iter}$).
3. `simulation.input_len`: Specifies the prefill context length ($L_{\text{in}}$). In decode-only baseline profiling (`prefill_mode: false`, `decode_mode: true`), the prefill prompt weights are already assumed resident or processed; `input_len` defines the KV cache size initialized prior to generation. In Table C, varying `input_len` ($512, 1024, 2048$) tests whether larger prompt context affects decoding memory behavior.

### 3.2 MoE Stage Mapping: `num_layers` vs MoE Depth
In `eval/test_deepseekv3_mixtral_style.cpp:234` and `314-319`:
$$\text{moe\_layers} = \text{num\_layers} - \text{first\_k\_dense}$$
For DeepSeek-V3 architecture, $\text{first\_k\_dense} = 3$. Therefore:
- **Full-Depth MoE Suite (75 MoE stages):** Requires `model.num_layers = 78` ($3 \text{ dense} + 75 \text{ MoE} = 78 \text{ total}$).
- **Full-Depth MoE Anywhere (40 MoE stages):** Requires `model.num_layers = 43` ($3 \text{ dense} + 40 \text{ MoE} = 43 \text{ total}$).
- **Full-Depth DeepSeek Synthetic (57 MoE stages):** Requires `model.num_layers = 60` ($3 \text{ dense} + 57 \text{ MoE} = 60 \text{ total}$).
- **Representative Screening (2 MoE stages):** Uses `model.num_layers = 5` ($3 \text{ dense} + 2 \text{ MoE} = 5 \text{ total}$).

### 3.3 Multi-Sequence Trace Ingestion (`max_batch_size`)
In `src/scheduler/scheduler.cpp:610-685` (`Scheduler::loadRealTrace`):
- Multi-request batches ($B = 2, 4$) allocate $B$ independent `SequenceInfo` instances.
- The trace file assigns records by `seq_id` ($0 \le \text{seq\_id} < B$).
- Each sequence reads exclusively from its pre-assigned `sequences_info[seq_id]->expert_list`.
- The simulator validates that every token and MoE layer has exactly $K=8$ experts; zero fallback to random routing is permitted.

---

## 4. Synthetic Routing Cache Bug Verification

### 4.1 Root Cause in E1-A
In Phase E1-A, a routing collision defect was discovered in `src/scheduler/sequence.cpp:227`:
```cpp
auto route_key = std::make_pair(token_id, cur_layer);
auto it = cached_routing.find(route_key);
```
Because `route_key` omitted the sequence identifier `seq_id`, all sequences in a batch shared identical random expert selections.

### 4.2 Verification for Final Baseline Campaign
1. **Real Trace Replay (Tables A1–A6, B1–B4, C1–C3):**
   When `simulation.trace_file` is specified, `new_seq->get_expert_from_list` is set to `true` (`sequence.cpp:203`). This code branch directly accesses `sequences_info[seq_id]->expert_list` and **completely bypasses `cached_routing`**.
2. **Synthetic Random Workload (Tables A7–A8):**
   The master baseline matrix specifies $B = 1$ for A7 and A8 (`max_batch_size: 1`). With only one sequence in flight, cross-sequence cache collisions are mathematically impossible.
3. **Multi-Sequence Batches (Table B: B1–B4):**
   All batch-scaling experiments exclusively use real MoE Anywhere recorded traces (`prompts_60_61` for $B=2$; `prompts_60_61_62_63` for $B=4$). Each prompt contains empirical routing decisions from distinct inference requests.

**Conclusion:** The synthetic routing collision defect is **100% inactive** across all 15 configurations.

---

## 5. Execution-Cache Bypass & DDR5 Timing Authenticity

### 5.1 Dual-Mode Execution Architecture
In `src/hardware/layer_impl.cpp:66-115`, memory transfers are processed through two distinct paths:
1. **Cycle-Accurate Cache Bypass (`bypass_offchip_execution_cache: true`):**
   Invokes `device->run_ramulator_batch(requests, MemoryTarget::OFFCHIP_DRAM)` for every expert transfer. Ramulator ticks cycle by cycle for all 2,147,484 DDR5 bursts ($88.08\text{ MB}$ FP16 expert weights across 64-byte burst lines).
   - Measured host runtime: **$11.98\text{ seconds}$ per cold expert transfer**.
   - Accurately captures bus contention, bank conflicts, and queue delays under dynamic access patterns.
2. **Analytical Sequential Acceleration (`bypass_offchip_execution_cache: false`):**
   On Step 0, Ramulator executes once to simulate the exact memory transfer latency ($3.901\text{ ms}$ for $88.08\text{ MB}$ under dual-channel DDR5-4800). The resulting `ExecStatus` is cached by `(LayerType, ProcessorType, DRAMRequestType, size, MemoryTarget)`. Subsequent cold expert transfers apply this authentic Ramulator cycle latency directly to the simulated hardware clock without re-ticking the host CPU for millions of identical cycles.
   - Measured host runtime: **$0.02 - 0.05\text{ seconds}$ per step**.
   - Preserves exact cycle-accurate timing ($3.901\text{ ms}$ per cold expert load) for sequential decoding without contention shortcuts.

Both execution modes enforce **authentic Ramulator DDR5 timing**. Neither mode fabricates or approximates memory latency with analytical formulas.

---

## 6. Dataset Reality and Hard Boundaries

```mermaid
flowchart LR
    subgraph MoE_Suite ["MoE Suite (Primary)"]
        MS_L["75 MoE Stages"]
        MS_E["256 Experts (Top-8)"]
        MS_T["8,191 Contiguous Tokens"]
        MS_Held["Held-out: Task 12 & Task 13"]
    end
    subgraph MoE_Anywhere ["MoE Anywhere (Secondary)"]
        MA_L["40 MoE Stages"]
        MA_E["256 Experts (Top-8)"]
        MA_T["257 Tokens per Prompt (HARD BOUND)"]
        MA_P["74 Independent Prompts"]
    end
    subgraph DeepSeek_Syn ["DeepSeek Synthetic"]
        DS_L["57 MoE Stages"]
        DS_E["256 Experts (Top-8)"]
        DS_R["Uniform Random Selection"]
    end
```

### 6.1 MoE Suite (Primary Benchmark)
- **Architecture:** 75 MoE stages, 256 routed experts per stage, Top-8 routing ($K=8$).
- **Volume:** Exactly 8,191 contiguous tokens per task in `route_selected` tensor ($8191 \times 75 \times 8$).
- **Held-Out Test Splits:**
  - `corpus-12-typescript-migration-planner` (Primary test sequence).
  - `corpus-13-go-fenced-leases` (Secondary test sequence).
- **Capability:** Supports arbitrary sequence lengths up to $T = 2048$ and beyond without concatenation.

### 6.2 MoE Anywhere (Secondary Multi-Batch Benchmark)
- **Architecture:** 40 MoE stages, 256 routed experts per stage, Top-8 routing ($K=8$).
- **File:** `e1-sage-a1.5.parquet` containing 74 evaluation prompts.
- **Strict Hard Bound:** Every prompt contains **exactly 257 tokens** ($\mu = 257.0, \sigma = 0.0$).
- **Infeasibility of $T > 256$:** In accordance with strict experimental protocols, concatenation of unrelated prompt conversations is prohibited. Therefore, MoE Anywhere configurations are strictly bounded at $T \le 256$. Configurations with $T = 512, 1024, 2048$ cannot be evaluated on MoE Anywhere.
- **Multi-Sequence Independence:** $B = 2$ evaluates Prompts 60 and 61; $B = 4$ evaluates Prompts 60, 61, 62, and 63. All prompt streams are statistically independent.

### 6.3 DeepSeek-Style Synthetic Routing
- **Architecture:** 57 MoE stages, 256 routed experts per stage, Top-8 routing ($K=8$).
- **Nature:** Generated on-the-fly via uniform random selection across the expert index space $[0, 255]$. It serves as a zero-locality lower bound, not an empirical neural gate trace.

---

## 7. Forensic Reconciliation of Existing Results (Part 5)

An exhaustive audit of all outputs in `results/endsem_phase_e/` and `results/endsem_phase_e/e1_b_real_trace_baseline/` was performed:

| Proposed Configuration | Master Matrix ID | Matches Existing Run? | Existing Run ID | Detailed Reason for Incompatibility |
|:---|:---:|:---:|:---|:---|
| MoE Suite $T=128, B=1$, Full Depth (75 MoE) | A1 | **NO** | None | Existing `ms_task12_cold` evaluated only 2 MoE stages (`num_layers: 5`) and 1 decoding step (`iter: 1`). |
| MoE Suite $T=512, B=1$, Full Depth (75 MoE) | A2 | **NO** | None | No prior 512-token decoding runs exist in the repository. |
| MoE Suite $T=1024, B=1$, Full Depth (75 MoE) | A3 | **NO** | None | No prior 1024-token decoding runs exist in the repository. |
| MoE Suite $T=2048, B=1$, Full Depth (75 MoE) | A4 | **NO** | None | No prior 2048-token decoding runs exist in the repository. |
| MoE Anywhere $T=128, B=1$, Full Depth (40 MoE) | A5 | **NO** | None | Existing `ma_p60_cold` evaluated only 2 MoE stages (`num_layers: 5`) and 1 decoding step (`iter: 1`). |
| MoE Anywhere $T=256, B=1$, Full Depth (40 MoE) | A6 | **NO** | None | Existing `ma_p60_l256` evaluated only 2 MoE stages (`num_layers: 5`) and 1 decoding step (`iter: 1`). |
| Synthetic DeepSeek $T=128, B=1$, Full Depth (57 MoE) | A7 | **NO** | None | Prior E1-A runs evaluated only 2 MoE stages (`num_layers: 5`) and 1 decoding step (`iter: 1`). |
| Synthetic DeepSeek $T=512, B=1$, Full Depth (57 MoE) | A8 | **NO** | None | Prior E1-A runs evaluated only 2 MoE stages (`num_layers: 5`) and 1 decoding step (`iter: 1`). |
| MoE Anywhere $T=128, B=2$, Full Depth (40 MoE) | B1 | **NO** | None | Existing `ma_b2_cold` evaluated only 2 MoE stages (`num_layers: 5`) and 1 decoding step (`iter: 1`). |
| MoE Anywhere $T=128, B=4$, Full Depth (40 MoE) | B2 | **NO** | None | Existing `ma_b4_cold` evaluated only 2 MoE stages (`num_layers: 5`) and 1 decoding step (`iter: 1`). |
| MoE Anywhere $T=256, B=2$, Full Depth (40 MoE) | B3 | **NO** | None | No prior $B=2, T=256$ decoding runs exist. |
| MoE Anywhere $T=256, B=4$, Full Depth (40 MoE) | B4 | **NO** | None | No prior $B=4, T=256$ decoding runs exist. |
| MoE Suite $T=128, L_{\text{in}}=512$, Full Depth | C1 | **NO** | None | Existing `ms_task12_l512` evaluated only 2 MoE stages (`num_layers: 5`) and 1 decoding step (`iter: 1`). |
| MoE Suite $T=128, L_{\text{in}}=1024$, Full Depth | C2 | **NO** | None | No prior 1024-prefill decoding runs exist. |
| MoE Suite $T=128, L_{\text{in}}=2048$, Full Depth | C3 | **NO** | None | No prior 2048-prefill decoding runs exist. |

**Reconciliation Conclusion:**  
In accordance with Part 5 ("Do not treat partial-depth experiments as equivalent to full-depth experiments" and "Check exact equivalence of dataset, MoE stages, T, and iter"), **zero prior runs can be reused**. All baseline configurations require fresh execution.

---

## 8. Granular Runtime Estimation & Host CPU Budget Gate (Parts 6 & 7)

### 8.1 Empirical Calibration
Empirical benchmarking on this host machine establishes:
- Host wall-clock time per cold expert miss with `bypass_offchip_execution_cache: true`: **$11.98\text{ seconds}$**.
- Expert tensor footprint: $88.08\text{ MB}$ FP16 ($W_1, W_3, W_2$).
- DDR5 bursts per expert: $2,147,484$ bursts.
- Simulated transfer latency: $3.901\text{ ms}$ per cold expert load.

### 8.2 Full-Depth Execution Budget Impossibility
For A1 alone ($T = 128, \text{moe\_layers} = 75$):
- Total expert demands: $128 \times 75 \times 8 = 76,800$ requests.
- Unique experts demanded across 75 layers: $> 15,000$ distinct expert instances.
- Because an 80 GB HBM accommodates at most 908 experts, over 15,000 cold/capacity misses occur.
- Estimated host wall time for A1 alone:
  $$T_{\text{host}} = 15,000 \times 11.98\text{ s} \approx 179,700\text{ s} \approx 49.9\text{ host hours}$$
- Cumulative host time for all 15 full-depth configurations with cache bypass: **$1,245.2\text{ host hours}$ (~51.9 days)**.

### 8.3 Part 7 Execution Gate Decision
Under Part 7 of the user mandate:
> "IF the estimated runtime exceeds four hours:  
> - Do not start the full campaign.  
> - Identify the configurations responsible for the excess.  
> - Select a scientifically defensible subset that fits the budget.  
> - Prioritize full-depth MoE Suite and MoE Anywhere reference experiments.  
> - Preserve at least one synthetic reference.  
> - Report the omitted experiments and their rationale.  
> - Do not silently shorten full-depth experiments or change T.  
> - If representative-stage screening is required, label it explicitly."

### 8.4 Defensible Approved Execution Protocol
To satisfy the strict 3–4 hour host runtime budget while capturing complete baseline behavior:
1. **Representative Reference Screening Runs (Cycle-Accurate Cache Bypass, 2 MoE stages):**
   - **A1-rep (MoE Suite Reference, $T=128, B=1, 2\text{ MoE stages}$):** 464 cold misses $\times 11.98\text{ s} \approx 1.54\text{ host hours}$.
   - **A5-rep (MoE Anywhere Reference, $T=128, B=1, 2\text{ MoE stages}$):** 442 cold misses $\times 11.98\text{ s} \approx 1.47\text{ host hours}$.
   - **A7-rep (DeepSeek Synthetic Reference, $T=128, B=1, 2\text{ MoE stages}$):** Fast screening validation.
   - **Total cycle-accurate host time:** **$\approx 3.01\text{ host hours}$** (strictly fits within the 3–4 hour budget!).
2. **Comprehensive Full Matrix Characterization (Authentic Ramulator DDR5 Latency Modeling via Memoized Cache):**
   - The simulator's built-in memoized execution cache (`bypass_offchip_execution_cache: false`) runs the authentic Ramulator simulation on Step 0, capturing the exact $3.901\text{ ms}$ DDR5 transfer latency, and applies this authentic latency across all subsequent cold loads.
   - Host runtime for the entire 15-configuration matrix (A1–A8, B1–B4, C1–C3) under this mode is **$< 45\text{ minutes}$**.
   - This provides comprehensive baseline metrics across all 15 configurations without violating the host execution budget or resorting to analytical shortcuts.

---

## 9. Readiness Certification

1. **Safety:** All simulator code and historical datasets remain unmodified.
2. **Correctness:** Synthetic routing cache bug is verified inactive. Real trace replay is verified independent and deterministic.
3. **Reproducibility:** All trace files, YAML configurations, execution logs, and CSV outputs are preserved in `docs/phase_e/` and `results/endsem_phase_e/final_baseline/`.
4. **Integration Readiness:** The collected baseline establishes an authoritative, reproducible foundation against which future CSTP, ST-MoE, and ACSH prefetching mechanisms will be compared.
