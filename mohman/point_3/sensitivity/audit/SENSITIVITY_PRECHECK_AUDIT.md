# Phase 4C — Pre-Experiment Audit: Batch-Size and Prediction-Horizon Sensitivity Evaluation

**Repository:** [`/home/mohman/ca_project/LLMSimulator`](file:///home/mohman/ca_project/LLMSimulator)  
**Git Commit:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Working Branch:** `main`  
**Execution Timestamp:** 2026-11-10  
**Audit Deliverable Path:** [`/home/mohman/ca_project/mohman/point_3/sensitivity/audit/SENSITIVITY_PRECHECK_AUDIT.md`](file:///home/mohman/ca_project/mohman/point_3/sensitivity/audit/SENSITIVITY_PRECHECK_AUDIT.md)  
**Status:** **AUDIT CERTIFIED — READY FOR CAMPAIGN EXECUTION**

---

## 1. Executive Summary

Prior to initiating the Phase 4C experimental campaign evaluating Batch-Size sensitivity ($B \in \{1, 2, 4, 8\}$) and Prediction-Horizon sensitivity ($H \in \{1, 2\}$), a comprehensive, forensic pre-experiment audit was conducted on the current checkout of `LLMSimulator`.

This audit establishes five prerequisites:
1. **Batch Execution Mechanics:** `serving.max_batch_size` directly instantiates $B$ independent sequence instances with unique monotonically assigned request IDs (`Sequence::id`) and sequence slot IDs (`Sequence::expert_seq_id`). Each sequence is assigned its own dedicated row in `Scheduler::sequences_info`.
2. **Cross-Sequence GEMM Aggregation ($M$ Dimension):** In [`src/module/route.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/module/route.cpp#L231-L236), the expert activation tensor batch dimension $M$ dynamically equals `sequences_metadata->num_token_in_expert[expert_idx]`. When multiple concurrent sequences route to the same expert, $M$ aggregates across sequences ($1 \le M \le B$), and the expert weights are demanded and executed exactly once for the batch.
3. **Three-Stage Topology Verification:** In [`src/model/llm.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/model/llm.cpp#L43-L54), `num_layers: 6` with `first_k_dense: 3` constructs exactly 3 dense prefix layers (`decoder_0`, `decoder_1`, `decoder_2`) followed by 3 MoE decoder layers (`MoE_decoder_3`, `MoE_decoder_4`, `MoE_decoder_5`). The routed stage count is $N_{\text{stages}} = 6 - 3 = 3$ (Stage 0, Stage 1, Stage 2). Intra-token lookahead with horizon $H=2$ evaluates `target_stage = 0 + 2 = 2 < 3`, targeting a topologically valid stage.
4. **Data Provenance & Sequence Availability:** The held-out test split of MoE-Anywhere (`e1-sage-a1.5.parquet`, prompts 60..73) provides 14 distinct test prompts. Prompts 60 through 67 each contain 257 unique tokens across 40 layers, providing complete, uncorrupted Top-8 real routing decisions for all $B \in \{1, 2, 4, 8\}$ runs. MoE-Suite Task 12 (`corpus-12-typescript-migration-planner.parquet`) contains 8,191 tokens across 75 layers, fully providing independent sequences for confirmation runs.
5. **Regression Baseline Integrity:** All four existing regression suites pass with zero failures:
   - `test_routing_ground_truth`: 10/10 passed.
   - `test_phase4b_effectiveness_metrics`: 10/10 passed.
   - `test_prefetch_metrics`: 15/15 passed.
   - `test_synthetic_routing_diversity`: 8/8 passed.

---

## 2. Forensic Review of Pre-Existing Routing Audit Artifacts

The following audit documents produced during the Phase 4B remediation were reviewed against the active codebase:

- [`ROUTING_GROUND_TRUTH_AUDIT.md`](file:///home/mohman/ca_project/mohman/point_3/routing_audit/ROUTING_GROUND_TRUTH_AUDIT.md): Confirmed all 10 trace-file loading criteria, scheduler routing dataflows, synthetic generator invariants, and dataset-specific routing requirements.
- [`ROUTING_PIPELINE_SOURCE_MAP.md`](file:///home/mohman/ca_project/mohman/point_3/routing_audit/ROUTING_PIPELINE_SOURCE_MAP.md): Verified the call tree from `Cluster::run` to `LLM::forward`, `MoEDecoder::forward`, `Route::forward`, `Linear::forward`, and `Device::execution`.
- [`ROUTING_FIX_REPORT.md`](file:///home/mohman/ca_project/mohman/point_3/routing_audit/ROUTING_FIX_REPORT.md): Verified the remediation of:
  - `total_len = input_len + output_len` in [`src/scheduler/sequence.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L16).
  - Routed MoE layer count `num_layer = model_config.num_layers - model_config.first_k_dense` in [`src/module/linear.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/module/linear.cpp#L69).
  - Multi-sequence synthetic cache lookup using the 4-component tuple `(seq->id, seq->expert_seq_id, seq->current_len, cur_layer)`.
  - Strict isolation in [`eval/test_deepseekv3_mixtral_style.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L418-L440) preventing dummy synthetic injection during real trace replay.
- [`ROUTING_REGRESSION_RESULTS.md`](file:///home/mohman/ca_project/mohman/point_3/routing_audit/ROUTING_REGRESSION_RESULTS.md): Verified that 12 controlled simulations produced 100% exact trace matches against ground-truth files with zero random fallback.
- [`CORE_SMOKE_VALIDATION.csv`](file:///home/mohman/ca_project/mohman/point_3/routing_audit/CORE_SMOKE_VALIDATION.csv): Verified quantitative baseline, ST-MoE, and CSTP results across 2-stage and 3-stage topologies.

---

## 3. Audit of Batch-Size Execution Mechanics (Research Question A)

### 3.1 Traceability of `serving.max_batch_size`
In [`eval/test_deepseekv3_mixtral_style.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L117):
```cpp
int max_batch_size = config["serving"]["max_batch_size"].as<int>();
Scheduler::Ptr scheduler = Scheduler::Create(system_config, model_config, "none", max_batch_size, 8192, max_process_token);
```
In [`src/scheduler/scheduler.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L25-L35):
`total_batch_size` is initialized to `max_batch_size`. In `fillRunningQueue(time_ns time)`:
```cpp
int num_empty_seq = total_batch_size - getBatchSize();
num_empty_seq = std::min(num_empty_seq, int(sequence_queue.size()));
```
When `max_batch_size` sequences are pushed to `sequence_queue`, `fillRunningQueue()` moves all $B$ sequences into `running_queue[0]`.

### 3.2 Sequence Identity & Trace Assignment
In [`eval/test_deepseekv3_mixtral_style.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L428-L440):
```cpp
scheduler->loadRealTrace(trace_file, max_batch_size, num_moe_layers, model_config.num_routed_expert, model_config.top_k, iter);
for (int b = 0; b < max_batch_size; b++) {
    Sequence::Ptr new_seq = Sequence::Create(b, input_len, seq_out_len);
    new_seq->current_len = input_len;
    new_seq->get_expert_from_list = true;
    scheduler->sequence_queue.push_back(new_seq);
}
```
Each sequence `b` possesses:
- `expert_seq_id = b`: The exact sequence index in the trace file ($0 \le b < B$).
- `id`: Monotonically increasing unique request ID assigned by global atomic counter `count_id`.

In [`Scheduler::loadRealTrace`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L663-L735):
```cpp
sequences_info.resize(expected_seqs);
for (int s = 0; s < expected_seqs; s++) {
    sequences_info[s] = SequenceInfo::Ptr(new SequenceInfo);
    sequences_info[s]->expert_list.resize((size_t)iter_tokens * num_layer);
}
```
The loader reads records from the trace CSV where `seq_id` corresponds to $s$. The loader validates that every sequence $s \in [0, B-1]$, token $t \in [0, \text{iter}-1]$, and layer $l \in [0, \text{num\_layer}-1]$ has exactly `top_k` (8) records. If any sequence is missing records, the loader throws a runtime error. It never copies sequence 0 into sequence 1, nor does it generate random filler.

### 3.3 GEMM Dimension $M$ Aggregation Across Sequences
In [`src/scheduler/sequence.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L217-L267), `BatchedSequence::update_expert()` executes for each layer:
```cpp
for (auto seq : sequence) {
    int seq_id = seq->expert_seq_id;
    int token_id = seq->current_len;
    size_t table_idx = (size_t)rel_token_id * num_layer + cur_layer;
    const auto& routed = scheduler->sequences_info[seq_id]->expert_list[table_idx];
    for (int k = 0; k < top_k; k++) {
        int e_id = routed[k];
        num_token_in_expert[e_id] += 1;
        routing_file << request_id << "," << token_id << "," << cur_layer << "," << (k + 1) << "," << e_id << "\n";
    }
}
```
For a batch of size $B$, the sum across all experts is:
$$\sum_{e=0}^{N-1} \text{num\_token\_in\_expert}[e] = B \times K$$
- For $B=1, K=8$: 8 token-expert assignments.
- For $B=2, K=8$: 16 token-expert assignments.
- For $B=4, K=8$: 32 token-expert assignments.
- For $B=8, K=8$: 64 token-expert assignments.

In [`src/module/route.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/module/route.cpp#L231-L236):
```cpp
for (int expert_idx = 0; expert_idx < num_expert; expert_idx++) {
    std::vector<int> shape = {
        sequences_metadata->num_token_in_expert[expert_idx],
        input[0]->shape[1]};
    Tensor::Ptr output =
        get_activation("expert_input_" + std::to_string(expert_idx), shape);
    output_vec.push_back(output);
```
The input activation tensor to each expert GEMM has shape $\{M, K_{\text{dim}}\}$, where:
$$M = \text{num\_token\_in\_expert}[\text{expert\_idx}]$$
If two sequences in the batch select the same expert, $M=2$. If eight sequences select the same expert, $M=8$. If zero sequences select the expert, $M=0$, and `Linear::forward()` skips execution.

### 3.4 Measured Cross-Sequence Expert Sharing
Analysis of the MoE-Anywhere held-out test prompts (prompts 60 through 67) reveals the following measured distinct expert counts across an 8-sequence batch ($B=8$):

| RelToken | MoE Stage 0 (Distinct / 64) | MoE Stage 1 (Distinct / 64) | MoE Stage 2 (Distinct / 64) | Mean Across Stages |
|---|---|---|---|---|
| **Token 0** | 9 distinct | 13 distinct | 11 distinct | 11.0 |
| **Token 1** | 11 distinct | 10 distinct | 9 distinct | 10.0 |
| **Token 2** | 10 distinct | 9 distinct | 9 distinct | 9.3 |
| **Token 3** | 14 distinct | 11 distinct | 10 distinct | 11.7 |

**Core Finding:** Across 64 total token-expert assignments ($8 \times 8$), only 9 to 14 unique experts are demanded per stage. The demand footprint grows sub-linearly with batch size, resulting in significant expert reuse and GEMM $M$ values up to 8.

---

## 4. Audit of Prediction-Horizon Mechanics (Research Question B)

### 4.1 Topology Construction and Stage Mapping
In [`src/model/llm.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/model/llm.cpp#L42-L54):
```cpp
} else if ((model_config.model_name == "deepseekV3") || (model_config.model_name == "debugV3")) {
    for (int layer = 0; layer < model_config.first_k_dense; layer++) {
        auto decoder = Decoder::Create(module_map_name, "decoder_" + std::to_string(layer), ...);
        add_module(decoder);
    }
    for (int layer = model_config.first_k_dense; layer < model_config.num_layers; layer++) {
        auto moe_decoder = MoEDecoder::Create(module_map_name, "MoE_decoder_" + std::to_string(layer), ...);
        add_module(moe_decoder);
    }
}
```
When configured with:
```yaml
model:
  model_name: deepseekV3
  num_layers: 6
```
And system preset DeepSeek-V3 with `first_k_dense: 3`:
- Layers 0, 1, 2: Dense attention and FFN modules (`decoder_0`, `decoder_1`, `decoder_2`).
- Layers 3, 4, 5: MoE attention and Routed MoE FFN modules (`MoE_decoder_3`, `MoE_decoder_4`, `MoE_decoder_5`).
- The routed MoE stage count is:
$$N_{\text{stages}} = \text{num\_layers} - \text{first\_k\_dense} = 6 - 3 = 3$$
- The logical stage IDs are strictly:
$$\text{Stage 0 } (\text{Layer 3}), \quad \text{Stage 1 } (\text{Layer 4}), \quad \text{Stage 2 } (\text{Layer 5})$$

### 4.2 Prediction Opportunity Counting & Denominators
In [`src/scheduler/sequence.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L320-L322):
`cur_layer` advances sequentially: $0 \to 1 \to 2 \to 0$.

For lookahead scheduling with horizon $H$:
- **Horizon $H=1$:**
  - Source Stage 0 $\to$ Target Stage 1 (`target = 0 + 1 = 1 < 3`): Valid transition.
  - Source Stage 1 $\to$ Target Stage 2 (`target = 1 + 1 = 2 < 3`): Valid transition.
  - Source Stage 2: `target = 2 + 1 = 3 \ge 3` (out of intra-token bounds, dropped).
  - **Opportunities per sequence per token:** Exactly 2 prediction events.
  - **For $B=1$, 4 tokens:** $4 \times 2 = 8$ total prediction opportunities.
- **Horizon $H=2$:**
  - Source Stage 0 $\to$ Target Stage 2 (`target = 0 + 2 = 2 < 3`): Valid transition.
  - Source Stage 1: `target = 1 + 2 = 3 \ge 3` (out of bounds, dropped).
  - Source Stage 2: `target = 2 + 2 = 4 \ge 3` (out of bounds, dropped).
  - **Opportunities per sequence per token:** Exactly 1 prediction event.
  - **For $B=1$, 4 tokens:** $4 \times 1 = 4$ total prediction opportunities.

**Accounting Mandate:** Metric summaries must never compare raw event counts between $H=1$ and $H=2$ without normalizing by the respective transition denominators (8 for $H=1$, 4 for $H=2$).

---

## 5. Dataset Provenance and Trace Verification

### 5.1 MoE-Anywhere
- **Source File:** [`LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet`](file:///home/mohman/ca_project/LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet)
- **Total Prompts:** 74 prompts (0 through 73).
- **Held-Out Test Split:** Prompts 60 through 73 (14 prompts).
- **Prompt Validation for Batch Sizes:**
  - $B=1$: Prompt 60 (257 unique tokens, 40 layers).
  - $B=2$: Prompts 60, 61 (257 unique tokens each).
  - $B=4$: Prompts 60, 61, 62, 63 (257 unique tokens each).
  - $B=8$: Prompts 60, 61, 62, 63, 64, 65, 66, 67 (257 unique tokens each).
- **Exporter Script:** [`LLMSimulator/scripts/phase_e/export_trace_for_simulator.py`](file:///home/mohman/ca_project/LLMSimulator/scripts/phase_e/export_trace_for_simulator.py)
  - Exports exact recorded Top-8 expert IDs per sequence, token, and layer.
  - Assigns monotonic `seq_idx` matching sequence slots.
  - Enforces zero ternary operators and strict split isolation.

### 5.2 MoE-Suite
- **Source File:** [`LLMSimulator/data/downloads/moe_suite/corpus-12-typescript-migration-planner.parquet`](file:///home/mohman/ca_project/LLMSimulator/data/downloads/moe_suite/corpus-12-typescript-migration-planner.parquet)
- **Trace Length:** 8,191 tokens across 75 layers with Top-8 expert selections.
- **Role in Phase 4C:** Dedicated confirmation dataset for Experiment C ($B=1$ Baseline, ST-MoE $H=1$, CSTP $H=1$).

---

## 6. Empirical Multi-Sequence Diagnostic Test Results

To empirically demonstrate that batch-size scaling functions correctly prior to full campaign execution, diagnostic runs were conducted with the simulator binary on three-stage topologies:

### 6.1 Diagnostic $B=2$ CSTP $H=1$
- **Input:** 2 sequences, 4 tokens, 3 MoE stages, Top-8 routing.
- **Trace Records Ingested:** 192 (100% verified, 0 fallback to random).
- **Live Prediction Engine Summary:**
  - `Total Predictions: 16 | Total Updates: 16 | Invalid/Skipped: 0`
  - `Precision@B: 0.4219 | Recall@B: 0.2109 | Hit Rate: 0.7500`
- **Output Record Counts:**
  - `expert_routing.csv`: 193 lines (192 records + header).
  - `expert_load.csv`: 733 lines.
  - `expert_cache.csv`: 436 lines.
- **Diagnostic Verification:** [`verify_routing_pipeline.py`](file:///home/mohman/ca_project/mohman/point_3/routing_audit/scripts/verify_routing_pipeline.py) confirmed 100% `EXACT_MATCH` across all 24 sequence-token-stage routing tuples.

### 6.2 Diagnostic $B=4$ Baseline
- **Trace Records Ingested:** 384 (4 sequences $\times$ 4 tokens $\times$ 3 stages $\times$ 8 experts).
- **Output Record Counts:**
  - `expert_routing.csv`: 385 lines.
  - `expert_load.csv`: 823 lines.
  - `expert_cache.csv`: 481 lines.
- **Result:** All 4 sequences completed 4 decode tokens with 100% trace fidelity.

### 6.3 Diagnostic $B=8$ Baseline
- **Trace Records Ingested:** 768 (8 sequences $\times$ 4 tokens $\times$ 3 stages $\times$ 8 experts).
- **Host Runtime:** 3.05 seconds.
- **Output Record Counts:**
  - `expert_routing.csv`: 769 lines.
  - `expert_load.csv`: 913 lines.
  - `expert_cache.csv`: 535 lines.
- **Result:** Monotonic distinct request IDs [8, 9, 10, 11, 12, 13, 14, 15] successfully executed across all 4 decode steps.

---

## 7. Experimental Controls and Invariant Hardware Parameters

All Phase 4C sensitivity experiments will enforce identical simulation parameters:

| Parameter | Configuration Value | Verification Source |
|---|---|---|
| **GPU Architecture** | NVIDIA A100-SXM4-80GB | Preset `A100` |
| **Model Geometry** | DeepSeek-V3 Mixtral-style | `model_name: deepseekV3` |
| **Layer Configuration** | `num_layers: 6` (3 dense prefix, 3 MoE stages) | `src/model/llm.cpp` |
| **Expert Count & Top-K** | 256 routed experts, Top-8 | `num_routed_expert: 256, top_k: 8` |
| **Precision** | FP16 (2 bytes per parameter) | `precision_byte: 2` |
| **Expert Footprint** | 88,080,384 bytes (~88.08 MB) | Validated across all runs |
| **Local HBM Budget** | 85,899,345,920 bytes (80.00 GB) | `expert_cache_capacity_bytes` |
| **DDR5 Memory Model** | Event-Driven Ramulator, 64.0 GB/s bandwidth | `USE_EVENT_DRIVEN_DRAM=true` |
| **Workload Dimensions** | $L_{\text{in}} = 128, L_{\text{out}} = 4, \text{iter} = 4$ | `input_len: 128, output_len: 4, iter: 4` |
| **Candidate Budget** | $B_{\text{cand}} = 4$ | `budget: 4` |
| **HBM Cache Eviction** | LRU with pinning protection | Invariant across runs |
| **DDR5 Channel Priority** | Priority 1 (Demands), Priority 0 (Prefetches) | Invariant across runs |
| **Execution Mode** | Decode-only sequential loop | `decode_mode: true, prefill_mode: false` |

---

## 8. Identified Constraints & Pre-Experiment Findings

1. **DDR5 Bus Contention During Cold Start:**  
   In Step 0, cold demand misses arrive at Priority 1 and monopolize the DDR5 bus. Because speculative prefetches execute at Priority 0, they complete after Step 0 demands have already stalled the device. However, as batch size increases, cross-sequence sharing increases the proportion of compute time relative to transfer time ($M$ increases), potentially creating expanded overlap windows for subsequent tokens.
2. **HBM Cache Size vs Sequence Length:**  
   Because the HBM cache is 80 GB, once an expert is demanded in Step 0, it remains resident in HBM for the duration of the 4-token sequence. Prefetching benefits can only manifest for fresh, unseen expert demands that are predicted ahead of their initial demand.
3. **Execution Runtime Scaling:**  
   Host execution runtime scales minimally with batch size (3.05 seconds for $B=8$ vs 2.66 seconds for $B=1$). The complete 20-run sensitivity campaign (12 Batch runs + 5 Horizon runs + 3 Suite runs) will execute in under 3 minutes total host runtime.

---

## 9. Pre-Experiment Certification

The pre-experiment audit is complete:
- Batch size scaling is functional, verified, and backed by independent real traces.
- Real trace ingestion is exact and free of random fallback.
- GEMM dimension $M$ correctly reflects cross-sequence expert aggregation.
- The 3-stage topology correctly enables both $H=1$ and $H=2$ intra-token lookahead.
- All 4 regression test suites are passing.

The repository is certified and ready for Phase 4C experiment execution.

