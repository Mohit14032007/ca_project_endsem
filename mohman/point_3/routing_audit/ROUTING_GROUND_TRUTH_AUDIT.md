# Phase A: Routing Ground-Truth Forensic Audit Report

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Deliverable Path:** [ROUTING_GROUND_TRUTH_AUDIT.md](file:///home/mohman/ca_project/mohman/point_3/routing_audit/ROUTING_GROUND_TRUTH_AUDIT.md)  
**Supporting Map:** [ROUTING_PIPELINE_SOURCE_MAP.md](file:///home/mohman/ca_project/mohman/point_3/routing_audit/ROUTING_PIPELINE_SOURCE_MAP.md)  
**Status:** Certified Read-Only Forensic Audit  

---

## 1. Executive Summary

This audit rigorously establishes the integrity and provenance of expert routing decisions in [LLMSimulator](file:///home/mohman/ca_project/LLMSimulator). We examined whether real routing traces (MoE-Anywhere and MoE-Suite) and synthetic routing traces are strictly distinguished, correctly loaded, and consumed without corruption or accidental fallback to random generation.

### Primary Audit Verdicts

1. **Real Traces Genuinely Ingested:** In real-trace replay mode, expert IDs are loaded line-by-line from the specified CSV into `sequences_info[s]->expert_list`. Every record is validated against bounds and Top-K counts. If any entry is missing, malformed, or incomplete, an exception is thrown immediately. Zero fallback to random generation occurs.
2. **Scheduler Dispatch Fidelity:** During MoE stage execution, `BatchedSequence::update_expert()` reads the exact Top-K expert IDs from `sequences_info[s]->expert_list` (in real mode) or from `cached_routing` (in synthetic mode). These identical IDs populate `num_token_in_expert`, which dictates the activation tensor sizes in `Route::forward()` and triggers linear GEMM layers in `LinearExecutionGPU()`.
3. **Synthetic Independence Preserved:** Synthetic routing constructs a 4-tuple cache key `(request_id, seq_id, token_id, cur_layer)`. Distinct sequences within a batch never collide or overwrite each other's routing cache.
4. **Topology Lookahead Horizon Boundary Identified:** In a 5-layer DeepSeek-V3 configuration with 3 dense prefix layers, there are only 2 MoE stages (Stage 0 and Stage 1). Forward lookahead with $H=2$ evaluates Stage $0 + 2 = 2 \ge \text{num\_stages}$, which correctly yields 0 predictions. This architectural constraint accounts for the absence of prefetch dispatches under $H=2$.
5. **Cold-Start Bus Monopoly Identified:** Speculative prefetch transfers are assigned Priority 0, whereas demand miss transfers are assigned Priority 1. During cold start, 16 demand misses monopolize the DDR5 channel for $22.02$ ms, causing speculative transfers dispatched in Step 0 to finish after their target demand deadline.

---

## 2. Section A.1: Trace-File Loading Forensic Audit

We conducted a line-by-line inspection of [Scheduler::loadRealTrace()](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L650-L735) in `src/scheduler/scheduler.cpp`. Below are the verified answers to the ten mandatory audit criteria:

### 1. Is the YAML `trace_file` actually read?
**Yes.** In [test_deepseekv3_mixtral_style.cpp](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L133-L136), the simulation configuration parser extracts the path:
```cpp
std::string trace_file = trace_file_cli;
if (trace_file.empty() && config["simulation"]["trace_file"]) {
  trace_file = config["simulation"]["trace_file"].as<std::string>();
}
```
If present, the runner invokes `scheduler->loadRealTrace(trace_file, ...)` at line 416.

### 2. Is the file opened successfully?
**Yes.** In [src/scheduler/scheduler.cpp](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L652-L655):
```cpp
std::ifstream f(trace_path);
if (!f.is_open()) {
  throw std::runtime_error("[ERROR] Failed to open real MoE trace file: " + trace_path);
}
```
If the file cannot be opened, execution halts immediately with a `std::runtime_error`.

### 3. Which parser reads the CSV?
**Custom streaming parser in `Scheduler::loadRealTrace()`.** Lines 676–692 read the file line-by-line using `std::getline(f, line)`, tokenizing commas via `std::stringstream` and parsing fields with `std::stoi()`.

### 4. What is the expected CSV schema?
The CSV schema is strictly:
```csv
seq_id,token_id,layer_id,topk_rank,expert_id
```
The first line is consumed as a header (`std::getline(f, header)` at line 658).

### 5. How are request IDs, sequence IDs, token positions, layer IDs, and expert IDs mapped?
- `seq_id`: Index into `scheduler->sequences_info[seq_id]`.
- `token_id`: Relative decode token index $t \in [0, \text{iter\_tokens})$.
- `layer_id`: MoE stage index $l \in [0, \text{num\_layer})$.
- `table_idx`: Mapped via flattened 1D index: `(size_t)token_id * num_layer + layer_id`.
- `topk_rank`: 1-based rank index $r \in [1, \text{top\_k}]$.
- `expert_id`: Stored at `sequences_info[seq_id]->expert_list[table_idx][topk_rank - 1] = expert_id`.

### 6. Are all Top-K expert IDs preserved?
**Yes.** The vector `exp_vec` is resized to accommodate `topk_rank`, storing each expert at index `topk_rank - 1`. All $K$ routed experts per stage are preserved.

### 7. Are the expert IDs sorted, transformed, truncated, or regenerated?
**No.** Expert IDs are stored directly as parsed from the CSV. They are neither truncated, transformed, nor re-hashed. When consumed in [sequence.cpp](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L259-L264), they are retrieved in order $k = 0, \dots, \text{top\_k} - 1$.

### 8. Does the loader silently fall back to synthetic/random routing if loading fails?
**No.** If file opening fails, if headers are absent, or if any record is invalid, the loader throws a fatal exception. Line 725 explicitly enforces:
```cpp
throw std::runtime_error("[ERROR] Trace incomplete in " + trace_path + 
  ": sequence " + std::to_string(s) + ", token " + std::to_string(t) + 
  ", layer " + std::to_string(l) + " has " + std::to_string(token_layer_counts[s][table_idx]) + 
  " experts (expected " + std::to_string(top_k) + "). Zero fallback to random allowed!");
```

### 9. Are incomplete or missing trace entries rejected?
**Yes.** A tracking matrix `token_layer_counts[s][table_idx]` counts records per sequence, token, and layer. After EOF, lines 720–729 assert that every `token_layer_counts[s][table_idx] == top_k`. Any discrepancy causes an immediate fatal abort.

### 10. Is the trace consumed in the correct order?
**Yes.** In [sequence.cpp](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L248-L254), during decode step `step`:
```cpp
int rel_token_id = token_id - seq->input_len;
if (rel_token_id < 0) rel_token_id = 0;
size_t table_idx = (size_t)rel_token_id * num_layer + cur_layer;
```
For `step = 0`, `rel_token_id = 0`, accessing `table_idx = 0 * num_layer + cur_layer`.
For `step = 1`, `rel_token_id = 1`, accessing `table_idx = 1 * num_layer + cur_layer`.
This matches the exact layout generated during `Scheduler::loadRealTrace()`.

---

## 3. Section A.2: Scheduler Routing and Dispatch Audit

We audited the complete chain between scheduler routing and GPU GEMM execution:

1. **Selection Function:** Actual expert selection occurs in `BatchedSequence::update_expert()` ([sequence.cpp:165](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L165)).
2. **Routing Source Discrimination:** If `seq->get_expert_from_list` is `true`, IDs are read from `scheduler->sequences_info[seq_id]->expert_list[table_idx]`. If `false`, IDs are generated or read from `cached_routing`.
3. **Router Gate Output:** In the simulation timing model, the gate projection linear layer `gate_fn` computes compute and DRAM latencies for router inference ([expert.cpp:179](file:///home/mohman/ca_project/LLMSimulator/src/module/expert.cpp#L179)). The architectural routing decisions (which experts are active) are populated into `sequences_metadata->num_token_in_expert` via `GateUpdate::aggregate_expert()`.
4. **No Overwrite:** Once `update_expert()` writes `actual_target_experts` into `num_token_in_expert`, no subsequent routine modifies the active expert set for that stage.
5. **Exact Top-K Preservation:** All $K$ expert IDs are added to `actual_target_experts` and written to `expert_routing.csv`.
6. **GEMM Aggregation & Demand Loading Consistency:**
   - In `Route::forward()` ([route.cpp:231](file:///home/mohman/ca_project/LLMSimulator/src/module/route.cpp#L231)), each expert $i$ receives an input tensor with dimension `shape[0] = sequences_metadata->num_token_in_expert[i]`.
   - In `Linear::forward()` ([linear.cpp:49](file:///home/mohman/ca_project/LLMSimulator/src/module/linear.cpp#L49)), an expert weight is only evaluated if `A->is_expert_weight && input->getSize() > 0`.
   - If an expert was not selected, `input->getSize() == 0`, and the linear layer returns without issuing memory requests.
   - If an expert was selected, `LinearExecutionGPU()` ([linear_impl.cpp:64](file:///home/mohman/ca_project/LLMSimulator/src/hardware/linear_impl.cpp#L64)) checks HBM residency and triggers DDR5 demand transfers on misses.

---

## 4. Section A.3: Synthetic Routing Forensic Audit

We re-audited the synthetic routing subsystem to verify that the batch-diversity bug fix from `mohman/point_2/` remains active and correct:

1. **Generator Invocation:** When `seq->get_expert_from_list` is `false`, `BatchedSequence::update_expert()` executes the synthetic branch ([sequence.cpp:268](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L268)).
2. **Cache Key Composite:** The cache key is constructed as:
   ```cpp
   auto route_key = std::make_tuple(request_id, seq_id, token_id, cur_layer);
   ```
   Including `seq_id` and `request_id` guarantees that sequence 0 and sequence 1 at the same token position and layer generate distinct cache entries.
3. **Top-K Count:** `scheduler->getRandomExpert(top_k)` generates a `std::set<int>` of size $K$.
4. **Expert Range Bounds:** The generator selects randomly within $[0, \text{num\_routed\_expert} - 1]$.
5. **Zero Trace Contamination:** When `get_expert_from_list` is `false`, `sequences_info` is never queried or accessed.

---

## 5. Section A.4: Routing-Source Decision Table

The table below documents the verified source of truth across all operating modes:

| Mode | Expected Routing Source | Actual Routing Source | Exact Defining File & Line | Verified Status |
| :--- | :--- | :--- | :--- | :--- |
| **Synthetic** | Synthetic Generator (`getRandomExpert` / `cached_routing`) | `cached_routing` keyed on `(request_id, seq_id, token_id, cur_layer)` | [sequence.cpp:269-286](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L269-L286) | **VERIFIED** |
| **MoE-Anywhere** | Trace CSV (`moe_anywhere_*.csv`) | `sequences_info[s]->expert_list[table_idx]` | [scheduler.cpp:650-735](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L650-L735), [sequence.cpp:244-267](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L244-L267) | **VERIFIED** |
| **MoE-Suite Task 12** | Trace CSV (`moe_suite_task12_*.csv`) | `sequences_info[s]->expert_list[table_idx]` | [scheduler.cpp:650-735](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L650-L735), [sequence.cpp:244-267](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L244-L267) | **VERIFIED** |
| **MoE-Suite Task 13** | Trace CSV (`moe_suite_task13_*.csv`) | `sequences_info[s]->expert_list[table_idx]` | [scheduler.cpp:650-735](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L650-L735), [sequence.cpp:244-267](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L244-L267) | **VERIFIED** |

---

## 6. Identified Discrepancies and Code Gaps

During this audit, we identified two specific discrepancies that require surgical remediation in Phase E:

1. **Layer Sizing Bug in `Linear::forward`:**
   In [src/module/linear.cpp:67-70](file:///home/mohman/ca_project/LLMSimulator/src/module/linear.cpp#L67-L70):
   ```cpp
   int num_layer = sequences_metadata->scheduler->model_config.num_layers;
   if (sequences_metadata->scheduler->model_config.expert_freq != 0) {
     num_layer = sequences_metadata->scheduler->model_config.num_layers / sequences_metadata->scheduler->model_config.expert_freq;
   }
   ```
   For DeepSeek-V3, `model_config.num_layers` is 5 and `first_k_dense` is 3, meaning `num_moe_layers = 2`. Because `linear.cpp` does not account for `first_k_dense`, it uses `num_layer = 5` instead of `2`. For `token_id >= 1`, `(rel_tok * 5 + layer_id)` exceeds `expert_list.size()`, causing demand metadata to default to sequence 0.
   Furthermore, in synthetic mode (`!s->get_expert_from_list`), `linear.cpp` lacks a lookup into `cached_routing`, defaulting demands to sequence 0.

2. **Runner Mode Isolation Enforcement:**
   In [eval/test_deepseekv3_mixtral_style.cpp:414](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L414):
   The branching currently checks `if (!trace_file.empty())`. If `data: synthetic` is specified in YAML but a `trace_file` path exists, it would load the trace file. Conversely, if a real dataset name is specified but `trace_file` is empty, it silently falls back to synthetic dummy sequences.
   *Remediation:* Explicitly check `data_name == "synthetic"` to force pure synthetic generation, and throw an immediate exception if a real dataset is specified without a valid `trace_file`.

