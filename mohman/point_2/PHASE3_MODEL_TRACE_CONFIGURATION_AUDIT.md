# Forensic Configuration Audit: Model Architecture, Trace Override, YAML Generation, and Smoke-Test Data Provenance

**Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Deliverable Path:** [PHASE3_MODEL_TRACE_CONFIGURATION_AUDIT.md](file:///home/mohman/ca_project/mohman/point_2/PHASE3_MODEL_TRACE_CONFIGURATION_AUDIT.md)  
**Status:** Certified Forensic Audit Report  
**Mode:** Strictly Read-Only Investigation  

---

## 1. Executive Summary

This forensic audit investigates the precise interaction between model architectural definitions (`model_name: deepseekV3`, `num_layers: 5`), external routing trace replay (`trace_file: moe_anywhere_smoke.csv`), and synthetic routing generation across the seven Phase 3 smoke-test configurations in the [LLMSimulator](file:///home/mohman/ca_project/LLMSimulator) repository.

### Definitive Conclusions

1. **Model Architecture Determination:**  
   The model architecture, tensor geometry, and GEMM arithmetic complexity are strictly determined by the C++ static configuration object `deepseekV3` in [model_config.h](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h#L107-L110) as modified by the YAML configuration parsed in [test_deepseekv3_mixtral_style.cpp](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L242-L258). Specifically:
   - Hidden dimension is fixed at $d = 7168$.
   - Feed-forward intermediate dimensions are fixed at $d_{\text{dense}} = 18432$ for dense layers and $d_{\text{expert}} = 2048$ for MoE layers.
   - Attention heads, KV heads, and head dimensions follow DeepSeek-V3 Multi-head Latent Attention (MLA) specifications ($d_{\text{head}} = 128$, $q_{\text{lora\_rank}} = 1536$, $kv_{\text{lora\_rank}} = 512$).
   - The expert inventory is fixed at $256$ routed experts plus $1$ shared expert, with Top-$8$ expert selection ($K = 8$).
   - Each expert consists of three SwiGLU linear projections: Gate ($W_1$), Up ($W_3$), and Down ($W_2$), with an individual tensor footprint of $29{,}360{,}128$ bytes in FP16 and a combined per-expert footprint of $88{,}080{,}384$ bytes ($\approx 88.08$ MB).
   - The YAML parameter `num_layers: 5` overrides the default $60$ layers down to $5$. Because DeepSeek-V3 specifies `first_k_dense = 3`, the simulator instantiates exactly three dense decoder layers (layers $0, 1, 2$) and two MoE decoder layers (layers $3, 4$, representing MoE stage $0$ and MoE stage $1$).

2. **Expert Routing Determination:**  
   The expert routing decisions are strictly determined by the routing input source, which completely bypasses and discards the model's native router logits:
   - When `trace_file` is provided (MoE-Anywhere and MoE-Suite), [Scheduler::loadRealTrace()](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L650-L730) loads the offline CSV records directly into `scheduler->sequences_info[seq_id]->expert_list`. In [BatchedSequence::update_expert()](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L244-L267), the sequence flag `get_expert_from_list` is `true`, causing the simulator to fetch expert IDs directly from the loaded table. The native gating GEMM (`gate_fn`) executes on the GPU timeline to account for compute FLOPs and latency, but its output activations are discarded. There is zero fallback to random routing.
   - When `trace_file` is omitted (Synthetic DeepSeek), `get_expert_from_list` is `false`. Expert routing decisions are generated pseudo-randomly via [Scheduler::getRandomExpert()](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L181-L193) (or [getZipfianRandomExpert()](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L205-L217) when skewness $> 0$) using a Mersenne Twister engine (`std::mt19937`, seed `777`). Decisions are memoized in `cached_routing` using a four-component key `(request_id, seq_id, token_id, cur_layer)`, ensuring strict statistical independence across sequences in batched workloads.

3. **Experiment Legitimacy Characterization:**  
   The Phase 3 smoke tests cannot legitimately be described as "full DeepSeek-V3 inference." They represent a **trace-driven reduced-topology micro-simulation**. The simulated execution graph runs a $5$-layer slice ($3$ dense decoders, $2$ MoE stages) utilizing authentic DeepSeek-V3 tensor dimensions and DDR5 backing geometry, replaying two-stage routing decisions extracted from offline datasets.

---

## 2. Model Architecture versus Trace Routing Explanation

The simulation harness cleanly decouples the static tensor execution graph (FLOPs, memory footprints, pipeline stages) from the dynamic routing decisions (which experts are activated per token).

```
+-----------------------------------------------------------------------------------------+
|                                    YAML CONFIGURATION                                   |
|   model.model_name: deepseekV3                                                          |
|   model.num_layers: 5                                                                   |
|   simulation.trace_file: moe_anywhere_smoke.csv                                         |
+-----------------------------------------------------------------------------------------+
                               |                                           |
                               v                                           v
       +------------------------------------+                     +-------------------+
       |     STATIC MODEL ARCHITECTURE      |                     |  DYNAMIC ROUTING  |
       |  (Tensor Shapes, GEMMs, Footprint) |                     |   (Expert IDs)    |
       +------------------------------------+                     +-------------------+
       | - hidden_dim: 7168                 |                     | - Gate GEMM runs  |
       | - intermediate_dim: 18432          |                     |   for FLOP timing |
       | - expert_intermediate_dim: 2048    |                     | - Gate logits are |
       | - first_k_dense: 3 (layers 0,1,2)  |                     |   DISCARDED       |
       | - num_moe_layers: 2 (layers 3,4)   |                     | - Expert IDs read |
       | - num_routed_expert: 256           |                     |   from trace CSV  |
       | - top_k: 8                         |                     |   or synthetic PRNG|
       | - expert size: 88,080,384 bytes    |                     | - Zero random     |
       |   (W1_gate, W3_up, W2_down)        |                     |   fallback        |
       +------------------------------------+                     +-------------------+
```

### A. Model Architecture Determination

In [test_deepseekv3_mixtral_style.cpp](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L243-L258), the configuration object begins from the static `deepseekV3` template defined in [model_config.h](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h#L107-L110):

```cpp
ModelConfig model_config = deepseekV3;
if (model_name == "debugV3") {
  model_config = debugV3;
}
if (config["model"]["num_layers"]) {
  model_config.num_layers = config["model"]["num_layers"].as<int>();
}
```

The parameters established by `model_name: deepseekV3` vs `num_layers: 5` are summarized below:

| Architectural Property | Value in Static `deepseekV3` | Overridden by `num_layers: 5`? | Effective Simulated Value | Source Reference |
|---|---|---|---|---|
| Hidden Dimension ($d$) | $7168$ | No | $7168$ | [model_config.h:108](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h#L108) |
| Attention Head Dimension ($d_h$) | $128$ | No | $128$ | [model_config.h:108](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h#L108) |
| Number of Attention Heads ($N_h$) | $128$ | No | $128$ | [model_config.h:108](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h#L108) |
| Number of KV Heads ($N_{kv}$) | $128$ | No | $128$ | [model_config.h:108](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h#L108) |
| Dense FFN Intermediate Dim | $18432$ | No | $18432$ | [model_config.h:108](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h#L108) |
| Expert Intermediate Dim ($d_{ffn}$) | $2048$ | No | $2048$ | [model_config.h:108](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h#L108) |
| Total Layer Count | $60$ | **Yes** (`num_layers: 5`) | $5$ | [test_deepseekv3_mixtral_style.cpp:248](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L248) |
| Leading Dense Layers (`first_k_dense`) | $3$ | No | $3$ (layers $0, 1, 2$) | [model_config.h:109](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h#L109) |
| Executed MoE Stages | $57$ ($60 - 3$) | **Yes** ($5 - 3 = 2$) | $2$ (layers $3, 4$) | [test_deepseekv3_mixtral_style.cpp:401](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L401) |
| Routed Experts per MoE Layer | $256$ | No | $256$ | [model_config.h:108](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h#L108) |
| Shared Experts per MoE Layer | $1$ | No | $1$ | [model_config.h:108](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h#L108) |
| Top-$K$ Selection Degree | $8$ | No | $8$ | [model_config.h:108](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h#L108) |
| FFN Way / Projections per Expert | $3$ (SwiGLU) | No | $3$ ($W_1$, $W_3$, $W_2$) | [model_config.h:109](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h#L109) |
| MLA Q-LoRA Rank | $1536$ | No | $1536$ | [model_config.h:109](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h#L109) |
| MLA KV-LoRA Rank | $512$ | No | $512$ | [model_config.h:109](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h#L109) |
| Precision (bytes) | $1$ (FP8 in static def) | **Yes** (`precision_byte: 2`) | $2$ (FP16) | [test_deepseekv3_mixtral_style.cpp:127](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L127) |
| Single Expert Weight Footprint | $44{,}040{,}192$ B (FP8) | **Yes** ($88{,}080{,}384$ B) | $88{,}080{,}384$ B ($\approx 88.08$ MB) | [test_deepseekv3_mixtral_style.cpp:270](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L270) |

### B. Routing Behaviour Under `trace_file` Replay

When `trace_file` is specified in the YAML configuration, the simulator execution conforms to the following operational invariants:

1. **Bypassing Native Router:**  
   In [src/module/expert.cpp:176-180](file:///home/mohman/ca_project/LLMSimulator/src/module/expert.cpp#L176-L180), `ExpertFFN::forward()` issues the router projection:
   ```cpp
   Tensor::Ptr gate_out = (*gate_fn)(input, sequences_metadata);
   Tensor::Ptr gate_update_out = (*gate_update)(input, sequences_metadata);
   ```
   `gate_fn` is a `ColumnParallelLinear` GEMM that computes the $(M \times 256)$ gating logits on the GPU timeline. The GPU compute duration and memory traffic of `gate_fn` are added to the simulated clock. However, `gate_out` is never sampled for top-$K$ selection. Instead, `gate_update` invokes `aggregate_expert()`, which delegates directly to `BatchedSequence::update_expert()`.
2. **Exact Expert ID Utilization:**  
   In [src/scheduler/sequence.cpp:244-267](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L244-L267), when `seq->get_expert_from_list` is `true`, expert IDs are retrieved strictly by table indexing:
   $$\text{table\_idx} = \text{rel\_token\_id} \times \text{num\_moe\_layers} + \text{cur\_layer}$$
   $$\text{expert\_id} = \text{scheduler}\to\text{sequences\_info}[\text{seq\_id}]\to\text{expert\_list}[\text{table\_idx}][k]$$
   The exact expert IDs from the CSV trace are inserted into `actual_target_experts` and `num_token_in_expert[expert_id]`.
3. **Top-$K$ Enforcement:**  
   The trace record must provide at least `top_k` ($8$) expert IDs. In [src/scheduler/sequence.cpp:255-257](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L255-L257), if fewer than $8$ experts exist for that `(token, layer)` tuple, execution terminates with `std::runtime_error`.
4. **Preservation of Model Tensor Shapes:**  
   The external trace provides routing indices only. The actual GEMMs executed inside `FeedForward3Way` preserve DeepSeek-V3 tensor dimensions:
   - $W_1$ (Gate): $(7168 \times 2048)$
   - $W_3$ (Up): $(7168 \times 2048)$
   - $W_2$ (Down): $(2048 \times 7168)$
5. **Absence of Random Fallback:**  
   In [src/scheduler/scheduler.cpp:719-729](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L719-L729), `loadRealTrace()` performs an upfront completeness verification across all expected sequences, tokens, and layers:
   ```cpp
   if (token_layer_counts[s][table_idx] != top_k) {
     throw std::runtime_error("[ERROR] Trace incomplete in " + trace_path + ... + ". Zero fallback to random allowed!");
   }
   ```
   There is no random fallback mechanism. Any missing record aborts the simulation.
6. **Error Handling on Mismatches:**  
   - If an expert ID in the trace is $\ge 256$ or $< 0$, an out-of-bounds `std::runtime_error` is thrown ([scheduler.cpp:705-707](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L705-L707), [sequence.cpp:261-263](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L261-L263)).
   - If the trace contains layers outside $[0, \text{num\_moe\_layers}-1]$, lines with out-of-bounds layer indices are skipped during ingestion ([scheduler.cpp:699-701](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L699-L701)).

### C. Evaluation of Descriptive Statements

Consider the statement:  
*"The simulator uses a reduced DeepSeek-V3-style model architecture, but its native router is replaced by recorded MoE-Anywhere routing decisions."*

**Audit Assessment:**  
This description is functionally correct regarding simulator mechanics, but requires two critical qualifications:
1. **Provenance Clarification:** The recorded routing decisions in `moe_anywhere_smoke.csv` were not produced by DeepSeek-V3. They originate from an external MoE model trace (`e1-sage-a1.5.parquet`) with $40$ layers and $256$ experts, from which a two-layer, two-token slice was extracted.
2. **Topology Clarification:** DeepSeek-V3 in production features $60$ layers ($3$ dense, $57$ MoE). This experiment executes a $5$-layer configuration ($3$ dense, $2$ MoE stages).

**Certified Technically Accurate Description:**  
> *"The simulator executes a trace-driven reduced-topology micro-simulation using authentic DeepSeek-V3 tensor dimensions ($d=7168, d_{ffn}=2048$, $88.08$ MB per expert) across five layers (three dense decoders followed by two MoE stages). Dynamic router GEMMs are executed on the simulated device clock to account for compute latency, but their routing logits are discarded and replaced by offline routing decisions ingested from MoE-Anywhere, MoE-Suite, or synthetic pseudo-random generators."*

---

## 3. Exact Routing Override Call Sequence

The step-by-step call sequence from configuration parsing to expert execution is detailed below:

```mermaid
sequenceDiagram
    autonumber
    participant Harness as test_deepseekv3_mixtral_style.cpp
    participant Sched as Scheduler (scheduler.cpp)
    participant LLM as LLM Model (llm.cpp)
    participant MoEDec as MoEDecoder (decoder.cpp)
    participant ExpFFN as ExpertFFN (expert.cpp)
    participant GateUp as GateUpdate (route.cpp)
    participant BatchSeq as BatchedSequence (sequence.cpp)
    participant PredEng as LivePredictionEngine
    participant LinGPU as LinearExecutionGPU (linear_impl.cpp)

    Harness->>Sched: loadRealTrace(trace_file, batch_size, num_moe_layers=2, num_experts=256, top_k=8, iter=2)
    Note over Sched: Ingests CSV into sequences_info[s]->expert_list<br/>Verifies exactly 8 experts per (seq, tok, layer)
    Harness->>Sched: Sequence::Create(expert_seq_id=b); seq->get_expert_from_list = true
    Harness->>LLM: forward(input, sequences_metadata)
    Note over LLM: Executes decoder_0, decoder_1, decoder_2 (Dense)
    LLM->>MoEDec: forward() for MoE_decoder_3 (MoE Stage 0)
    MoEDec->>ExpFFN: forward()
    ExpFFN->>ExpFFN: (*gate_fn)(input) [Simulates Router GEMM FLOPs & Time]
    ExpFFN->>GateUp: (*gate_update)(input)
    GateUp->>BatchSeq: update_expert(num_expert=256, top_k=8, is_moe=true)
    BatchSeq->>PredEng: on_stage_routing_start(req_id, token_id, cur_layer=0)
    BatchSeq->>PredEng: dispatch_candidates(dev, cur_time) [DDR5 Transfers Issued]
    Note over BatchSeq: get_expert_from_list == true:<br/>Reads sequences_info[s]->expert_list[table_idx]<br/>Populates actual_target_experts<br/>Bypasses gate_fn logits completely!
    BatchSeq->>PredEng: on_stage_routing_complete(req_id, token_id, cur_layer=0, actual_target_experts)
    ExpFFN->>ExpFFN: Iterates expert_id in [0..255]
    Note over ExpFFN: If tokens routed to expert_id > 0:<br/>Executes FeedForward3Way
    ExpFFN->>LinGPU: gate_proj (W1), up_proj (W3), down_proj (W2)
    LinGPU->>LinGPU: Checks device->is_expert_resident()<br/>W1 misses -> issues DDR5 demand load<br/>W3 and W2 hit HBM cache!
```

### Direct Source Code Call Chain

1. [eval/test_deepseekv3_mixtral_style.cpp:408](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L408):  
   `scheduler->loadRealTrace(trace_file, max_batch_size, num_moe_layers, model_config.num_routed_expert, model_config.top_k, iter);`
2. [src/scheduler/scheduler.cpp:650-730](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L650-L730):  
   `Scheduler::loadRealTrace()` parses CSV records, validates that `topk_rank` is in $[1, 8]$ and `expert_id` is in $[0, 255]$, and stores them in `sequences_info[seq_id]->expert_list[table_idx]`.
3. [eval/test_deepseekv3_mixtral_style.cpp:410-415](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L410-L415):  
   `Sequence::Ptr new_seq = Sequence::Create(b, input_len, output_len);`  
   `new_seq->get_expert_from_list = true;`
4. [src/model/llm.cpp:89-92](file:///home/mohman/ca_project/LLMSimulator/src/model/llm.cpp#L89-L92):  
   `LLM::forward()` iterates layers $3 \dots 4$, dispatching to `MoEDecoder::forward()`.
5. [src/module/expert.cpp:176-180](file:///home/mohman/ca_project/LLMSimulator/src/module/expert.cpp#L176-L180):  
   `ExpertFFN::forward()` runs `gate_fn->forward()` (simulating GEMM latency) followed by `gate_update->forward()`.
6. [src/module/route.cpp:144, 165](file:///home/mohman/ca_project/LLMSimulator/src/module/route.cpp#L144):  
   `GateUpdate::forward()` calls `aggregate_expert()`, which invokes `scheduler->running_queue[i]->update_expert(num_expert, top_k, true)`.
7. [src/scheduler/sequence.cpp:244-267](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L244-L267):  
   `BatchedSequence::update_expert()` checks `seq->get_expert_from_list`:
   - Computes `table_idx = rel_token_id * num_layer + cur_layer`.
   - Reads `e_id = scheduler->sequences_info[seq_id]->expert_list[table_idx][k]`.
   - Increments `num_token_in_expert[e_id] += 1`.
   - Logs to `expert_routing.csv`.
   - Passes `actual_target_experts` to `prediction_engine->on_stage_routing_complete()`.
8. [src/module/expert.cpp:208-215](file:///home/mohman/ca_project/LLMSimulator/src/module/expert.cpp#L208-L215):  
   `ExpertFFN::forward()` iterates over `expert_id` from $0$ to $255$. If `route_out.at(expert_id)` has non-zero size, it executes `FeedForward3Way::forward()`.
9. [src/hardware/linear_impl.cpp:63-162](file:///home/mohman/ca_project/LLMSimulator/src/hardware/linear_impl.cpp#L63-L162):  
   `LinearExecutionGPU()` evaluates residency in HBM via `device->is_expert_resident()`. On a miss, DDR5 demand transfer is simulated and the entire expert ($W_1, W_3, W_2$) is marked resident.

---

## 4. Trace and Model Compatibility Analysis

To verify data provenance and compatibility, the actual contents and metadata of all smoke-test trace files were audited against the simulator model configuration:

| Property | Model Configuration | MoE-Anywhere Smoke Trace | MoE-Suite Corpus-12 Smoke | MoE-Suite Corpus-13 Smoke | Synthetic Routing Runs |
|---|---|---|---|---|---|
| **Underlying Data File** | Built-in C++ / YAML | `moe_anywhere_smoke.csv` | `moe_suite_corpus12_smoke.csv` | `moe_suite_corpus13_smoke.csv` | None (In-memory PRNG) |
| **Original Parquet Source** | N/A | `e1-sage-a1.5.parquet` | `corpus-12-...planner.parquet` | `corpus-13-...leases.parquet` | N/A |
| **Source MoE Layers** | $2$ executed ($5$ total) | $40$ in parquet (sliced to $2$) | $28$ in parquet (sliced to $2$) | $28$ in parquet (sliced to $2$) | $2$ stages generated |
| **Executed MoE Layers** | $2$ (stages $0$ and $1$) | $2$ (layers $0$ and $1$) | $2$ (layers $0$ and $1$) | $2$ (layers $0$ and $1$) | $2$ (stages $0$ and $1$) |
| **Experts per Layer** | $256$ routed | $256$ (IDs in $[0, 255]$) | $256$ (IDs in $[0, 255]$) | $256$ (IDs in $[0, 255]$) | $256$ (IDs in $[0, 255]$) |
| **Top-$K$ Selection** | $K = 8$ | $8$ per token-layer | $8$ per token-layer | $8$ per token-layer | $8$ per token-layer |
| **Expert ID Range** | $[0, 255]$ | Min: $11$, Max: $250$ | Min: $19$, Max: $248$ | Min: $19$, Max: $248$ | Min: $0$, Max: $255$ |
| **Token Count** | $2$ decode iterations | $2$ tokens (pos $20, 21 \to 0, 1$) | $2$ tokens (indices $0, 1$) | $2$ tokens (indices $0, 1$) | $2$ tokens (steps $0, 1$) |
| **Total Trace Records** | $32$ records expected | $32$ records ($2\times 2\times 8$) | $32$ records ($2\times 2\times 8$) | $32$ records ($2\times 2\times 8$) | Generated on demand |
| **Layer Slicing Method** | Static topology | Offline slice in Python | Offline slice in Python | Offline slice in Python | In-memory loop |
| **Batch / Request IDs** | $B \in \{1, 2, 4, 8\}$ | Fixed `seq_id = 0` ($B=1$) | Fixed `seq_id = 0` ($B=1$) | Fixed `seq_id = 0` ($B=1$) | Unique `seq_id = 0 \dots B-1` |
| **Routing Mechanism** | Native router bypassed | Real recorded trace replay | Real recorded trace replay | Real recorded trace replay | `mt19937` seeded PRNG |

### Layer Mapping and Trace Slicing Verification

1. **Was the $40$-layer MoE-Anywhere trace mapped dynamically onto $2$ stages?**  
   No. Inspection of [scratch/extract_smoke_traces.py:15-32](file:///home/mohman/.gemini/antigravity/brain/b7db0340-f5d9-4df3-86d8-be407d44242d/scratch/extract_smoke_traces.py#L15-L32) confirms that the $40$-layer trace was explicitly filtered in Python:
   ```python
   df_pid0 = df_anywhere[df_anywhere["prompt_id"] == 0]
   tokens_avail = sorted(df_pid0["token_pos"].unique())[:2]
   df_sub = df_pid0[(df_pid0["token_pos"].isin(tokens_avail)) & (df_pid0["layer"].isin([0, 1]))]
   ```
   The smoke CSV file [moe_anywhere_smoke.csv](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/phase3_validation/traces/moe_anywhere_smoke.csv) contains only `layer_id` values $0$ and $1$. It was already sliced upfront to two MoE stages.
2. **Was MoE-Suite dynamically mapped?**  
   No. In [scratch/extract_smoke_traces.py:43-47](file:///home/mohman/.gemini/antigravity/brain/b7db0340-f5d9-4df3-86d8-be407d44242d/scratch/extract_smoke_traces.py#L43-L47), `route_selected` was indexed directly at `routes_12[token_id][layer_id]` for `token_id in range(2)` and `layer_id in range(2)`. The resulting CSVs [moe_suite_corpus12_smoke.csv](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/phase3_validation/traces/moe_suite_corpus12_smoke.csv) and [moe_suite_corpus13_smoke.csv](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/phase3_validation/traces/moe_suite_corpus13_smoke.csv) each contain exactly $32$ lines of data records corresponding to tokens $0, 1$ and layers $0, 1$.
3. **Validity of Expert IDs:**  
   All expert IDs in all three CSV traces fall strictly in the range $[0, 255]$, matching DeepSeek-V3's $256$ routed expert configuration.

---

## 5. The Five-Layer / Two-MoE-Stage Topology and Tensor Geometry

### Why Five Layers?

In the official DeepSeek-V3 architecture ([model_config.h:108-109](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h#L108-L109)), the first three transformer layers are dense decoders (`first_k_dense = 3`) to stabilize representation learning before routing into MoE decoders.

In [src/model/llm.cpp:42-55](file:///home/mohman/ca_project/LLMSimulator/src/model/llm.cpp#L42-L55), model instantiation implements:
```cpp
} else if ((model_config.model_name == "deepseekV3") || (model_config.model_name == "debugV3")) {
  for (int layer = 0; layer < model_config.first_k_dense; layer++) {
    auto decoder = Decoder::Create(module_map_name, "decoder_" + std::to_string(layer),
                                   model_config, scheduler, device_list, device);
    add_module(decoder);
  }
  for (int layer = model_config.first_k_dense; layer < model_config.num_layers; layer++) {
    auto moe_decoder = MoEDecoder::Create(module_map_name, "MoE_decoder_" + std::to_string(layer),
                                          model_config, scheduler, device_list, device);
    add_module(moe_decoder);
  }
}
```

When `num_layers = 5` is configured:
- `layer = 0, 1, 2`: Instantiated as dense `Decoder` modules (`decoder_0`, `decoder_1`, `decoder_2`).
- `layer = 3, 4`: Instantiated as `MoEDecoder` modules (`MoE_decoder_3`, `MoE_decoder_4`).

In [eval/test_deepseekv3_mixtral_style.cpp:400-404](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L400-L404):
```cpp
int num_moe_layers = model_config.num_layers;
if (model_name == "deepseekV3" || model_name == "debugV3") {
  num_moe_layers = model_config.num_layers - model_config.first_k_dense;
}
```
Thus, `num_moe_layers = 5 - 3 = 2`. Layer $3$ corresponds to logical MoE stage $0$, and layer $4$ corresponds to logical MoE stage $1$.

Setting `num_layers = 5` creates the minimal viable multi-stage MoE pipeline:
1. It preserves the authentic DeepSeek-V3 architectural requirement of three dense pre-layers.
2. It provides exactly two successive MoE stages ($L=0 \to L=1$), which is the minimal requirement to exercise lookahead prediction (predicting stage $1$ candidates while executing stage $0$).
3. It keeps simulation wall-clock time under $10$ seconds per run while evaluating the full DDR5 asynchronous transfer pipeline.

### Tensor Shapes, Footprint, and Arithmetic Complexity

In [src/module/layer.cpp:360-412](file:///home/mohman/ca_project/LLMSimulator/src/module/layer.cpp#L360-L412) and [src/module/expert.cpp:78-118](file:///home/mohman/ca_project/LLMSimulator/src/module/expert.cpp#L78-L118), each of the $256$ routed experts is instantiated as a `FeedForward3Way` module consisting of three linear projections:

$$\text{SwiGLU}(x) = \left( x W_1 \otimes \text{SiLU}(x W_3) \right) W_2$$

The tensor specifications per expert are:

1. **Gate Projection ($W_1$ / `W1_gate`):**
   - Shape: $(K \times N) = (d \times d_{ffn}) = (7168 \times 2048)$
   - Elements: $7168 \times 2048 = 14{,}680{,}064$
   - FP16 Bytes ($2$ B/elem): $14{,}680{,}064 \times 2 = 29{,}360{,}128$ bytes ($\approx 29.36$ MB)
2. **Up Projection ($W_3$ / `W3_up`):**
   - Shape: $(K \times N) = (d \times d_{ffn}) = (7168 \times 2048)$
   - Elements: $7168 \times 2048 = 14{,}680{,}064$
   - FP16 Bytes ($2$ B/elem): $14{,}680{,}064 \times 2 = 29{,}360{,}128$ bytes ($\approx 29.36$ MB)
3. **Down Projection ($W_2$ / `W2_down`):**
   - Shape: $(K \times N) = (d_{ffn} \times d) = (2048 \times 7168)$
   - Elements: $2048 \times 7168 = 14{,}680{,}064$
   - FP16 Bytes ($2$ B/elem): $14{,}680{,}064 \times 2 = 29{,}360{,}128$ bytes ($\approx 29.36$ MB)

**Combined Single Expert Footprint:**
$$\text{Expert Size} = 29{,}360{,}128 + 29{,}360{,}128 + 29{,}360{,}128 = 88{,}080{,}384\text{ bytes } (\approx 88.0804\text{ MB})$$

**Total Backing Inventory for 2 MoE Layers:**
$$\text{Total Routed Experts} = 2\text{ layers} \times 256\text{ experts} = 512\text{ experts}$$
$$\text{Active DDR5 Working Inventory} = 512 \times 88{,}080{,}384\text{ bytes} = 45{,}097{,}156{,}608\text{ bytes } (\approx 45.10\text{ GB})$$

**Decode FLOPs per Activated Expert Token:**
$$\text{FLOPs} = 2 \times (7168 \times 2048) + 2 \times (7168 \times 2048) + 2 \times (2048 \times 7168) = 88{,}080{,}384\text{ FLOPs}$$

---

## 6. Recovery of All Seven Phase 3 Smoke-Test YAML Configurations

All seven configuration YAML files reside in:  
`/home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/phase3_validation/configs/`

The generation script that produced them is [scratch/setup_and_run_smoke_tests.py](file:///home/mohman/.gemini/antigravity/brain/b7db0340-f5d9-4df3-86d8-be407d44242d/scratch/setup_and_run_smoke_tests.py).

### Execution Command

Each configuration was executed from `/home/mohman/ca_project/LLMSimulator/build` with:
```bash
USE_EVENT_DRIVEN_DRAM=true \
DDR5_BANDWIDTH_GBPS=64.0 \
./run_deepseekv3_mixtral_style <config_file_path>
```

### Configuration Comparison Table

| Config Name | File Path | Batch ($B$) | Dataset (`simulation.data`) | `trace_file` Path | Decode Iterations | Predictor | Budget | Horizon |
|---|---|---|---|---|---|---|---|---|
| `moe_anywhere_smoke` | [moe_anywhere_smoke.yaml](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/phase3_validation/configs/moe_anywhere_smoke.yaml) | $1$ | `moe_anywhere` | `.../traces/moe_anywhere_smoke.csv` | $2$ | CSTP | $4$ | $1$ |
| `moe_suite_corpus12_smoke` | [moe_suite_corpus12_smoke.yaml](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/phase3_validation/configs/moe_suite_corpus12_smoke.yaml) | $1$ | `moe_suite` | `.../traces/moe_suite_corpus12_smoke.csv` | $2$ | CSTP | $4$ | $1$ |
| `moe_suite_corpus13_smoke` | [moe_suite_corpus13_smoke.yaml](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/phase3_validation/configs/moe_suite_corpus13_smoke.yaml) | $1$ | `moe_suite` | `.../traces/moe_suite_corpus13_smoke.csv` | $2$ | CSTP | $4$ | $1$ |
| `synthetic_deepseek_b1` | [synthetic_deepseek_b1.yaml](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/phase3_validation/configs/synthetic_deepseek_b1.yaml) | $1$ | `synthesis` | *(None / empty)* | $2$ | CSTP | $4$ | $1$ |
| `synthetic_deepseek_b2` | [synthetic_deepseek_b2.yaml](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/phase3_validation/configs/synthetic_deepseek_b2.yaml) | $2$ | `synthesis` | *(None / empty)* | $2$ | CSTP | $4$ | $1$ |
| `synthetic_deepseek_b4` | [synthetic_deepseek_b4.yaml](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/phase3_validation/configs/synthetic_deepseek_b4.yaml) | $4$ | `synthesis` | *(None / empty)* | $2$ | CSTP | $4$ | $1$ |
| `synthetic_deepseek_b8` | [synthetic_deepseek_b8.yaml](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/phase3_validation/configs/synthetic_deepseek_b8.yaml) | $8$ | `synthesis` | *(None / empty)* | $2$ | CSTP | $4$ | $1$ |

### Common Invariant Parameters Across All 7 YAMLs

All seven configuration files share identical hardware, memory, and serving specifications:
- `model.model_name`: `deepseekV3`
- `model.num_layers`: `5`
- `system.gpu_gen`: `A100`, `processor_type`: `GPU`, `num_node`: `1`, `num_device`: `1`
- `system.optimization.use_ramulator`: `'on'`
- `system.optimization.expert_cache_capacity_bytes`: `80000000000` ($80$ GB HBM cache)
- `system.optimization.prefill_mode`: `false`, `decode_mode`: `true`
- `simulation.input_len`: `128`, `output_len`: `4`, `precision_byte`: `2` (FP16), `skewness`: `0.0`
- `simulation.prefetching.enabled`: `true`, `mode`: `hardware_async`, `scheduling_mode`: `forward_lookahead`

---

## 7. Dataset Provenance and Extraction Pipeline

### A. MoE-Anywhere

- **Source File:** `/home/mohman/ca_project/LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet`
- **Extraction Tool:** [scratch/extract_smoke_traces.py](file:///home/mohman/.gemini/antigravity/brain/b7db0340-f5d9-4df3-86d8-be407d44242d/scratch/extract_smoke_traces.py#L10-L33)
- **Selection Logic:** Prompt ID $0$ (`prompt_id == 0`). The first two unique token positions in that prompt were positions $20$ and $21$.
- **Mapping:** Token position $20$ was mapped to `token_id = 0`, and position $21$ was mapped to `token_id = 1`. Layers were filtered to `layer in [0, 1]`.
- **Integrity:** The top-$8$ expert array for each token-layer was preserved exactly as recorded. Exactly $32$ records ($2\text{ tokens} \times 2\text{ layers} \times 8\text{ ranks}$) were emitted to [moe_anywhere_smoke.csv](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/phase3_validation/traces/moe_anywhere_smoke.csv).

### B. MoE-Suite

- **Source Files:**
  - Corpus-12: `/home/mohman/ca_project/LLMSimulator/data/downloads/moe_suite/corpus-12-typescript-migration-planner.parquet`
  - Corpus-13: `/home/mohman/ca_project/LLMSimulator/data/downloads/moe_suite/corpus-13-go-fenced-leases.parquet`
- **Selection Rationale:** Corpus-12 and Corpus-13 represent two distinct real-world coding agent tasks (TypeScript migration planning vs Go distributed fenced leases), providing contrasting real-trace routing patterns.
- **Extraction Logic:** In [scratch/extract_smoke_traces.py:35-65](file:///home/mohman/.gemini/antigravity/brain/b7db0340-f5d9-4df3-86d8-be407d44242d/scratch/extract_smoke_traces.py#L35-L65), the `route_selected` column was read. Each entry contains a two-dimensional matrix `route_selected[token_idx][layer_idx]`, where each cell holds an $8$-element list of selected expert IDs.
- **Output:** Sliced for `token_id in [0, 1]` and `layer_id in [0, 1]`. Exactly $32$ records were written to [moe_suite_corpus12_smoke.csv](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/phase3_validation/traces/moe_suite_corpus12_smoke.csv) and [moe_suite_corpus13_smoke.csv](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/phase3_validation/traces/moe_suite_corpus13_smoke.csv).
- **Divergence:** As verified in Section 4, Corpus-12 and Corpus-13 share common dominant experts (e.g. 142, 71, 68, 208) but diverge on secondary ranks (e.g. rank 5 is 77 in Corpus-12 vs 209 in Corpus-13), proving they are genuine, distinct traces.

### C. Synthetic DeepSeek Routing

- **Code Location:** [src/scheduler/sequence.cpp:268-286](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L268-L286) and [src/scheduler/scheduler.cpp:181-193](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L181-L193).
- **Trace File:** No trace CSV is loaded. `trace_file` is empty, so `get_expert_from_list` remains `false`.
- **Random Number Generator:** `std::mt19937` Mersenne Twister engine initialized with static seed `777`:
  ```cpp
  static unsigned int seed = 777;
  static std::mt19937 generator(seed);
  static std::uniform_int_distribution<int> distribution(0, model_config.num_routed_expert - 1);
  ```
- **Top-$K$ Selection:** Rejection sampling populates a `std::set<int> route` until `route.size() == top_k`, guaranteeing $8$ distinct expert IDs in $[0, 255]$.
- **Distribution Type:** Uniform distribution (`skewness == 0.0`). When `skewness > 0`, it switches to Zipfian discrete sampling via `getZipfianRandomExpert()`.
- **Composite Cache Key:**
  $$\text{route\_key} = \langle \text{request\_id}, \text{seq\_id}, \text{token\_id}, \text{cur\_layer} \rangle$$
  In [src/scheduler/sequence.cpp:12-13](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L12-L13), each sequence created by `pushDummySeq()` is assigned a globally unique monotonically increasing `id = count_id++`. Because `request_id` differs for every sequence in a batch, each sequence indexes a distinct entry in `cached_routing`. Synthetic routing decisions are strictly independent across sequences within a batch.

---

## 8. YAML-to-Execution Control Flow

The complete lifecycle from configuration file ingestion to GPU kernel execution proceeds through eight discrete phases:

```
[1. YAML Load]                YAML::LoadFile("config.yaml") in test_deepseekv3_mixtral_style.cpp
      |
      v
[2. Config Parsing]           SystemConfig, ModelConfig (deepseekV3 + num_layers: 5), PrefetchConfig
      |
      v
[3. Model Construction]       Model model(model_config, cluster, scheduler)
      |                       LLM::init_model() creates 3 Decoders + 2 MoEDecoders
      v
[4. Trace Ingestion]          Scheduler::loadRealTrace() reads CSV into sequences_info (ONCE at start)
      |                       Zero random fallback enforced!
      v
[5. Queue Setup]              Scheduler::initRunningQueue() + fillRunningQueue()
      |                       BatchedSequence initialized with Sequence objects
      v
[6. Decode Iteration]         Sequential decode loop (iter = 2) calls LLM::forward()
      |
      v
[7. Stage Gating & Prefetch]  ExpertFFN::forward() -> gate_fn (GEMM) -> GateUpdate::forward()
      |                       -> BatchedSequence::update_expert() -> CSTP candidate dispatch
      v
[8. Tensor Execution & DDR5]  FeedForward3Way (W1, W3, W2) -> LinearExecutionGPU()
                              HBM check -> DDR5 demand load on miss -> residency insertion
```

### Detailed Lifecycle Answers

1. **Which component reads the YAML?**  
   [eval/test_deepseekv3_mixtral_style.cpp:79-146](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L79-L146) reads the YAML using `YAML::LoadFile(config_path)` and populates `SystemConfig`, `ModelConfig`, and `PrefetchConfig`.
2. **When is the model constructed?**  
   At [test_deepseekv3_mixtral_style.cpp:339](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L339) (`Model model(model_config, cluster, scheduler)`). `LLM::init_model()` instantiates `decoder_0`, `decoder_1`, `decoder_2` and `MoE_decoder_3`, `MoE_decoder_4`.
3. **When is the trace loaded?**  
   At [test_deepseekv3_mixtral_style.cpp:408](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L408), before the decode loop starts. `loadRealTrace()` parses the CSV file once at initialization. It is not read dynamically during decode iterations.
4. **When are request and sequence identities assigned?**  
   At [test_deepseekv3_mixtral_style.cpp:410](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L410) during `Sequence::Create(b, input_len, output_len)`, which sets `seq->expert_seq_id = b` and `seq->id = count_id++`.
5. **When are routing decisions read?**  
   During each decode iteration when `ExpertFFN::forward()` calls `gate_update->forward()`, which triggers `BatchedSequence::update_expert()` ([sequence.cpp:215](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L215)).
6. **When is CSTP initialized?**  
   At [test_deepseekv3_mixtral_style.cpp:305-314](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L305-L314) via `PredictorRegistry::create_predictor()`, and attached to `LivePredictionEngine` at line $328$.
7. **When are prefetch candidates generated?**  
   In [sequence.cpp:230-239](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L230-L239), at the beginning of each MoE stage (`on_stage_routing_start()`), CSTP evaluates previous layer activations and generates lookahead prefetch candidates for the subsequent stage, immediately dispatching them via `dispatch_candidates()`.
8. **When are expert weights loaded?**  
   Prefetched weights transfer asynchronously in the background via `DDR5TransferEngine`. Demanded weights that missed prefetch are loaded synchronously in `LinearExecutionGPU()` ([linear_impl.cpp:147-162](file:///home/mohman/ca_project/LLMSimulator/src/hardware/linear_impl.cpp#L147-L162)) when `W1_gate` begins execution.

---

## 9. Explanation of Smoke-Test Results and Metric Semantics

The verified metrics from the Phase 3 smoke tests ([smoke_test_summary.json](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/phase3_validation/smoke_test_summary.json)) are summarized below:

| Experiment Name | Batch | Total Demands | HBM Hit Rate | Fresh Demand Miss Rate | Speculative Bytes (B) | Demand Bytes (B) | Total DDR5 Bytes (B) | Exposed Stall (ns) | Total Simulated (ns) |
|---|---|---|---|---|---|---|---|---|---|
| `moe_anywhere_smoke` | $1$ | $96$ | $69.79\%$ | $30.21\%$ | $352{,}321{,}536$ | $2{,}554{,}331{,}136$ | $2{,}906{,}652{,}672$ | $39{,}911{,}424.00$ | $44{,}070{,}414.80$ |
| `moe_suite_corpus12` | $1$ | $96$ | $67.71\%$ | $32.29\%$ | $352{,}321{,}536$ | $2{,}730{,}491{,}904$ | $3{,}082{,}813{,}440$ | $42{,}663{,}936.00$ | $46{,}809{,}049.89$ |
| `moe_suite_corpus13` | $1$ | $96$ | $67.71\%$ | $32.29\%$ | $352{,}321{,}536$ | $2{,}730{,}491{,}904$ | $3{,}082{,}813{,}440$ | $42{,}663{,}936.00$ | $46{,}809{,}049.89$ |
| `synthetic_deepseek_b1` | $1$ | $96$ | $67.71\%$ | $32.29\%$ | $352{,}321{,}536$ | $2{,}730{,}491{,}904$ | $3{,}082{,}813{,}440$ | $42{,}663{,}936.00$ | $46{,}809{,}049.89$ |
| `synthetic_deepseek_b2` | $2$ | $189$ | $67.72\%$ | $32.28\%$ | $352{,}321{,}536$ | $5{,}372{,}903{,}424$ | $5{,}725{,}224{,}960$ | $83{,}951{,}616.00$ | $88{,}815{,}374.31$ |
| `synthetic_deepseek_b4` | $4$ | $378$ | $67.72\%$ | $32.28\%$ | $352{,}321{,}536$ | $10{,}745{,}806{,}848$ | $11{,}098{,}128{,}384$ | $167{,}903{,}232.00$ | $174{,}247{,}494.48$ |
| `synthetic_deepseek_b8` | $8$ | $711$ | $70.89\%$ | $28.97\%$ | $352{,}321{,}536$ | $18{,}144{,}559{,}104$ | $18{,}496{,}880{,}640$ | $284{,}884{,}992.00$ | $294{,}057{,}619.89$ |

### Mathematical Breakdown of Metrics

#### 1. Why Is the Cold-Start HBM Hit Rate $\approx 66.67\%$?

In `FeedForward3Way`, each activated expert executes three linear projections sequentially:
1. `gate_proj` ($W_1$)
2. `up_proj` ($W_3$)
3. `down_proj` ($W_2$)

In [src/hardware/linear_impl.cpp:52-60, 153-161](file:///home/mohman/ca_project/LLMSimulator/src/hardware/linear_impl.cpp#L52-L60):
- When $W_1$ executes, the expert is not resident in HBM. $W_1$ incurs a `DEMAND_MISS_RESOLVED`. It bundles its sister weights ($W_3, W_2$) into `batch_tensors`, loads all $88{,}080{,}384$ bytes over DDR5, and calls `device->mark_expert_resident()`.
- When $W_3$ executes immediately after, `device->is_expert_resident()` returns `true`. $W_3$ incurs a `RESIDENT_HIT` ($0$ DDR5 traffic).
- When $W_2$ executes immediately after, `device->is_expert_resident()` returns `true`. $W_2$ incurs a second `RESIDENT_HIT` ($0$ DDR5 traffic).

Thus, for every newly demanded expert:
$$\text{Hit Rate} = \frac{2\text{ Hits}}{1\text{ Miss} + 2\text{ Hits}} = \frac{2}{3} \approx 66.6667\%$$
- In `corpus12`, `corpus13`, and `synthetic_b1`: Exactly $31$ unique experts were demanded and $1$ expert was reused across steps:
  $$\text{Hits} = (31 \times 2) + (1 \times 3) = 62 + 3 = 65 \implies \frac{65}{96} = 67.7083\%$$
- In `moe_anywhere`: Exactly $29$ unique experts were demanded and $3$ experts were reused across steps:
  $$\text{Hits} = (29 \times 2) + (3 \times 3) = 58 + 9 = 67 \implies \frac{67}{96} = 69.7917\%$$

#### 2. Derivation of the 96 Demands Count ($B=1$)

For a single sequence ($B=1$), the decode loop runs for $2$ iterations (`iter: 2`) across $2$ MoE layers (`num_moe_layers: 2`). Each sequence selects Top-$8$ experts:
$$\text{Expert Invocations} = 2\text{ decode steps} \times 2\text{ MoE layers} \times 8\text{ experts} = 32\text{ expert activations}$$
Because each expert invocation evaluates three linear projections ($W_1, W_3, W_2$), the total demand count is:
$$\text{Total Demands} = 32 \times 3 = 96$$

#### 3. Derivation of Batch Demands: 96, 189, 378, 711

Why are synthetic batch demands $96, 189, 378, 711$ rather than exact multiples $96 \times B = (96, 192, 384, 768)$?

In [src/module/linear.cpp:49](file:///home/mohman/ca_project/LLMSimulator/src/module/linear.cpp#L49), an expert projection only executes if:
$$\text{input}\to\text{getSize}() > 0$$
When multiple sequences in a batch route to the *same* expert in the same layer, their token representations are batched together into a single input tensor. The expert projection is executed **once** for the entire batch.

Therefore, the number of demand events equals:
$$\text{Total Demands} = 3 \times \sum_{t=0}^{1} \sum_{l=0}^{1} U_{t, l}$$
where $U_{t, l}$ is the number of **unique experts** selected across all sequences in batch $B$ at step $t$ and layer $l$:
- **Batch 1 ($B=1$):** Each sequence selects $8$ unique experts. Total unique activations = $4 \times 8 = 32$. Total demands = $32 \times 3 = 96$.
- **Batch 2 ($B=2$):** Two sequences select $2 \times 8 = 16$ candidates per layer. Across the $4$ layer-steps, expert co-selection occurred in $3$ instances, resulting in $63$ unique expert activations:
  $$\text{Total Demands} = 63 \times 3 = 189$$
- **Batch 4 ($B=4$):** Four sequences select $4 \times 8 = 32$ candidates per layer. Inter-sequence expert co-selection produced $126$ unique activations:
  $$\text{Total Demands} = 126 \times 3 = 378$$
- **Batch 8 ($B=8$):** Eight sequences select $8 \times 8 = 64$ candidates per layer. Significant expert sharing reduced unique activations to $237$:
  $$\text{Total Demands} = 237 \times 3 = 711$$

This demonstrates proper multi-sequence batching: expert weights are transferred once per batch and shared across all sequences selecting them.

#### 4. Memory Traffic and Exposed Stall Calculations

- **Speculative DDR5 Bytes ($352{,}321{,}536$ B):**  
  CSTP dispatched exactly $4$ prefetch candidates across the two decode steps ($4 \text{ prefetches} \times 88{,}080{,}384\text{ B} = 352{,}321{,}536\text{ B}$).
- **Demand DDR5 Bytes:**  
  For $B=1$ with $31$ fresh misses:
  $$\text{Demand Bytes} = 31 \times 88{,}080{,}384\text{ B} = 2{,}730{,}491{,}904\text{ B } (\approx 2.73\text{ GB})$$
- **Total DDR5 Bytes:**  
  $$\text{Total Bytes} = \text{Speculative Bytes} + \text{Demand Bytes} = 352{,}321{,}536 + 2{,}730{,}491{,}904 = 3{,}082{,}813{,}440\text{ B } (\approx 3.08\text{ GB})$$
- **Exposed Stall Duration:**  
  At a DDR5 channel bandwidth of $64.0$ GB/s ($64 \times 10^9$ B/s):
  $$\text{Transfer Latency per Expert} = \frac{88{,}080{,}384\text{ B}}{64.0\text{ GB/s}} = 1{,}376{,}256\text{ ns } (\approx 1.376\text{ ms})$$
  For $31$ un-prefetched cold misses:
  $$\text{Exposed Stall} = 31 \times 1{,}376{,}256\text{ ns} = 42{,}663{,}936.00\text{ ns } (\approx 42.66\text{ ms})$$
  This matches the reported simulation output down to the exact nanosecond.

---

## 10. Configuration Inconsistencies and Experimental Risks

The audit revealed the following nuances and risks to document prior to Phase 4:

1. **Topological Discrepancy (5 Layers vs 60 Layers):**  
   The simulation executes $2$ MoE layers, whereas full DeepSeek-V3 possesses $57$ MoE layers. While per-layer DDR5 transfer contention and prefetch latency are accurately modeled, end-to-end device latency reflects only a $5$-layer slice.
2. **Offline Trace Slicing Invariant:**  
   Because external traces were sliced to $2$ layers in Python, they cannot be used unmodified if `num_layers` is increased in Phase 4. Any experiment with $N$ MoE stages requires re-extracting traces with $N$ layers.
3. **Trace Sequence Count ($B=1$ Trace Replay):**  
   The existing CSV traces contain records for `seq_id = 0` only. Running real-trace replay with batch size $B > 1$ will trigger trace incomplete errors unless multi-sequence traces are extracted.
4. **Router Logit Compute Overhead:**  
   `gate_fn` computes gating logits that are discarded. This overhead accurately reflects the compute latency of the gating layer, but researchers should remain aware that logits do not influence routing.

---

## 11. Exact Source File References

Every architectural, scheduling, and mathematical invariant in this report is grounded in the current codebase:

- [eval/test_deepseekv3_mixtral_style.cpp](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp):
  - Lines 79-86: YAML parsing and model name verification.
  - Lines 242-258: `model_config` initialization and `num_layers` override.
  - Lines 260-289: DDR5 preflight address geometry checks and topology validation.
  - Lines 305-328: CSTP predictor creation and `LivePredictionEngine` linkage.
  - Lines 399-404: `num_moe_layers = num_layers - first_k_dense` computation.
  - Lines 406-422: `loadRealTrace()` invocation and `Sequence` creation.
- [src/model/model_config.h](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h):
  - Lines 9-53: `ModelConfig` constructor parameter definitions.
  - Lines 107-110: `deepseekV3` static template definition ($d=7168$, $d_{ffn}=2048$, $256$ routed, $1$ shared, $K=8$, `first_k_dense=3`).
- [src/model/llm.cpp](file:///home/mohman/ca_project/LLMSimulator/src/model/llm.cpp):
  - Lines 42-55: Construction of $3$ dense `Decoder` modules and $(N - 3)$ `MoEDecoder` modules.
  - Lines 82-95: Sequential execution of dense decoders followed by MoE decoders.
- [src/module/expert.cpp](file:///home/mohman/ca_project/LLMSimulator/src/module/expert.cpp):
  - Lines 29-32: Instantiation of `gate_fn` router projection.
  - Lines 78-118: Instantiation of `FeedForward3Way` and linkage of sister weights ($W_1, W_3, W_2$).
  - Lines 176-180: Router GEMM execution followed by `gate_update`.
  - Lines 208-215: Loop executing active expert FFNs.
- [src/module/route.cpp](file:///home/mohman/ca_project/LLMSimulator/src/module/route.cpp):
  - Lines 139-147: `GateUpdate::forward()`.
  - Lines 149-193: `GateUpdate::aggregate_expert()` dispatching to `update_expert()`.
- [src/scheduler/scheduler.cpp](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp):
  - Lines 82-133: `Scheduler::pushDummySeq()` sequence creation.
  - Lines 181-193: `Scheduler::getRandomExpert()` uniform `mt19937` sampling.
  - Lines 205-217: `Scheduler::getZipfianRandomExpert()` skewed sampling.
  - Lines 650-730: `Scheduler::loadRealTrace()` CSV parsing and zero-fallback verification.
- [src/scheduler/sequence.cpp](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp):
  - Lines 8-18: `Sequence` constructor assigning unique `id = count_id++`.
  - Lines 215-243: `update_expert()` entry and CSTP lookahead dispatch.
  - Lines 244-267: Real trace replay branch reading `expert_list` table.
  - Lines 268-293: Synthetic routing branch indexing `cached_routing`.
- [src/hardware/linear_impl.cpp](file:///home/mohman/ca_project/LLMSimulator/src/hardware/linear_impl.cpp):
  - Lines 52-60: Sister weight bundling into `batch_tensors`.
  - Lines 63-75: HBM residency check and demand miss detection.
  - Lines 77-135: Timely prefetch and resident hit logging.
  - Lines 147-162: Synchronous DDR5 transfer execution and `mark_expert_resident()`.

---

## 12. Final Definitive Statement

**Question:**  
*When `model_name: deepseekV3` and a real `trace_file` are both provided, exactly what determines the model architecture, and exactly what determines the expert routing decisions?*

**Definitive Answer:**
1. **The Model Architecture** is strictly determined by the static C++ `deepseekV3` configuration object in [src/model/model_config.h](file:///home/mohman/ca_project/LLMSimulator/src/model/model_config.h#L107-L110), as modified by the YAML parameter `num_layers: 5` in [eval/test_deepseekv3_mixtral_style.cpp](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L248). This fixes the hidden dimension at $7168$, expert intermediate dimension at $2048$, dense intermediate dimension at $18432$, SwiGLU 3-way projection structure ($W_1, W_3, W_2$), expert tensor footprint at $88{,}080{,}384$ bytes ($\approx 88.08$ MB FP16), expert inventory at $256$ routed experts plus $1$ shared expert, Top-$8$ selection degree, and executes exactly five layers (three dense decoders followed by two MoE stages).
2. **The Expert Routing Decisions** are strictly determined by the recorded CSV trace file loaded by [Scheduler::loadRealTrace()](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L650-L730) into `scheduler->sequences_info[seq_id]->expert_list`. Dynamic gating logits generated by `gate_fn` are completely discarded. The simulator uses the exact expert IDs from the trace with zero random fallback.
3. **Characterization:** These Phase 3 smoke tests represent a **trace-driven reduced-topology micro-simulation**. They are not full DeepSeek-V3 inference, but rather an execution graph utilizing authentic DeepSeek-V3 tensor dimensions and DDR5 backing geometry, driven by offline-sliced routing traces across two executed MoE stages.

