# Phase E1-B: Read-Only Audit of Real MoE Trace Datasets and Simulator Integration

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** E1-B Real MoE Trace Replay Baseline  
**Date:** November 2026  
**Status:** COMPLETE  
**Repository Path:** `/home/mohman/ca_project/LLMSimulator`  
**Referenced Datasets:**
1. MoE Suite: `LLMSimulator/data/downloads/moe_suite/*.parquet` (Primary held-out benchmark)
2. MoE Anywhere: `LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet` (Secondary benchmark)

---

## 1. Executive Summary

This read-only audit establishes the architectural mapping required to replay real MoE routing traces inside `LLMSimulator`. In Phase E1-A, we characterized a synthetic baseline using uniform pseudo-random expert routing (`Scheduler::getRandomExpert`, seed `777`). Phase E1-B replaces this synthetic generator with actual recorded $\text{Top-}K$ expert routing decisions from two verified production datasets: **MoE Suite** and **MoE Anywhere**.

This audit confirms that:
1. Both datasets utilize $E=256$ routed experts and $\text{Top-}8$ routing ($K=8$), matching the DeepSeek-V3 geometry configured in `LLMSimulator`.
2. The C++ simulator natively contains the data structures (`SequenceInfo`, `expert_list`, `seq->get_expert_from_list`) to execute externally specified routing sequences.
3. Batch replay can preserve completely independent token streams by binding each sequence in a batch to a distinct trace sequence in `scheduler->sequences_info`.
4. No changes to the DRAM timing model, HBM cache capacity, or LRU eviction algorithms are required.

---

## 2. Dataset Schemas and Routing Dimensions

### 2.1 MoE Suite (Primary Benchmark)
- **Origin / Reference:** `mistrjirka/Predicting-Experts-for-MOE-Suite` (token cube representation).
- **Model Topology:** 75 MoE stages (Transformer layers 3..77), 256 routed experts per stage ($E=256$), $\text{Top-}8$ routing ($K=8$).
- **File Structure:** 12 task-specific Parquet files in `data/downloads/moe_suite/`. Each file contains $8,191$ sequential tokens for a distinct coding task.
- **Key Columns:**
  - `route_selected`: Nested list of shape $(N_{\text{tokens}}, 75, 8)$, containing exact $\text{Top-}8$ expert IDs in $[0, 255]$ for each of the 75 MoE stages.
  - `task_id`: String task identifier (e.g., `corpus-12-typescript-migration-planner`).
  - `token_id`, `generated_position`, `context_position`: Chronological token position markers.
- **Official Task-Disjoint Splits:**
  - **Train (8 tasks):** `corpus-02` through `corpus-09` ($65,528$ tokens).
  - **Validation (2 tasks):** `corpus-10-cuda-segmented-reduction`, `corpus-11-cpp-bounded-channel` ($16,382$ tokens).
  - **Test (2 tasks):** `corpus-12-typescript-migration-planner`, `corpus-13-go-fenced-leases` ($16,382$ tokens).
- **Evaluation Mandate:** Only the two official held-out test tasks are used for the primary replay baseline, evaluated and reported separately.

### 2.2 MoE Anywhere (Secondary Benchmark)
- **Origin / Reference:** `carlosfelipe/moe-anywhere-routing-traces` (`e1-sage-a1.5.parquet`).
- **Model Topology:** InternScience/Sage-A1.5, 40 MoE layers ($L=40$, layers 0..39), 256 routed experts per layer ($E=256$), $\text{Top-}8$ routing ($K=8$).
- **File Structure:** A single consolidated Parquet file containing $760,720$ records across 74 unique prompts.
- **Key Columns:**
  - `prompt_id`: Integer prompt identifier ($0 \le \text{prompt\_id} < 74$).
  - `task`: High-level domain task string (e.g., `"code"`, `"math"`, `"general"`).
  - `token_pos`: Chronological token position within the sequence ($0 \le \text{token\_pos} < L_{\text{seq}}$).
  - `layer`: MoE layer index ($0 \le \text{layer} < 40$).
  - `experts`: List of 8 selected expert IDs in $[0, 255]$.
  - `weights`: List of 8 router softmax weights corresponding to selected experts.
- **Prompt-Disjoint Splits:**
  - **Train (52 prompts):** Prompts 0 to 51.
  - **Validation (8 prompts):** Prompts 52 to 59.
  - **Test (14 prompts):** Prompts 60 to 73.

---

## 3. ID Representation and Coordinate Mapping

| Conceptual Dimension | MoE Suite Representation | MoE Anywhere Representation | LLMSimulator C++ Target |
|---|---|---|---|
| **Request / Sequence ID** | Task string (e.g. `corpus-12`) | Integer `prompt_id` (e.g. 60) | `Sequence::id`, `seq->expert_seq_id` |
| **Token Position** | Row index $t \in [0, N_{\text{tokens}}-1]$ | Column `token_pos` | `token_id` (in decode loop: $L_{\text{in}} + \text{step}$) |
| **MoE Layer ID** | Axis 1 index $l \in [0, 74]$ | Column `layer` $l \in [0, 39]$ | `cur_layer` ($0 \le l < \text{num\_layers}$) |
| **Top-K Rank** | Axis 2 index $k \in [0, 7]$ | Array index $k \in [0, 7]$ | `topk_rank` ($1 \le k \le 8$) |
| **Expert ID** | Integer in $[0, 255]$ | Integer in $[0, 255]$ | `e_id` (in $[0, 255]$) |

### 3.1 Mapping to Simulator Execution
In `LLMSimulator`, the sequence scheduling and routing consumption operate as follows:
1. `SequenceInfo` holds the sequence metadata:
   ```cpp
   struct SequenceInfo {
     int Lin;
     int Lout;
     int cur;
     std::vector<std::vector<int>> expert_list;
   };
   ```
2. The 2D table `expert_list` stores the $\text{Top-}K$ vector for each `(token_id, cur_layer)` pair:
   $$\text{Table Index} = \text{token\_id} \times \text{num\_layer} + \text{cur\_layer}$$
3. During decoding, `BatchedSequence::update_expert` executes:
   ```cpp
   if (seq->get_expert_from_list) {
     for (int k = 0; k < top_k; k++) {
       int e_id = scheduler->sequences_info[seq_id]->expert_list[token_id * num_layer + cur_layer][k];
       num_token_in_expert[e_id] += 1;
       routing_file << request_id << "," << token_id << "," << cur_layer << "," << (k + 1) << "," << e_id << "\n";
     }
   }
   ```
4. In `Linear::forward`, any expert with `num_token_in_expert[e_id] > 0` is executed on the simulated GPU.

---

## 4. Architectural Feasibility Analysis

### 4.1 Support for Injecting Externally Specified Expert IDs
- **Status:** **Supported natively in core data structures, but requires an updated loader.**
- **Details:** The core structures (`SequenceInfo`, `expert_list`, `seq->get_expert_from_list = true`) already exist and were designed for pre-recorded trace replay. However, the legacy file reader in `Scheduler::initializeInput` (`src/scheduler/scheduler.cpp`: lines 550–590) hardcodes a legacy text format with only two experts per token (`top_1, top_2`).
- **Solution:** Implement a modern, lightweight trace reader that parses recorded traces into `scheduler->sequences_info` with arbitrary $K$ ($K=8$).

### 4.2 Preserving Independent Token Streams in Batch Replay
- **Status:** **Fully Supported.**
- **Root Cause of E1-A Artifact:** In E1-A, all batch elements received the same experts because `cached_routing` was keyed only by `(token_id, cur_layer)`.
- **Trace Replay Mechanism:** In trace replay mode, each sequence in `running_queue` is assigned a distinct `expert_seq_id`:
  - Batch Sequence 0 $\implies$ `expert_seq_id = 0` (e.g., Task 12 or Prompt 60).
  - Batch Sequence 1 $\implies$ `expert_seq_id = 1` (e.g., Task 13 or Prompt 61).
  - Batch Sequence 2 $\implies$ `expert_seq_id = 2` (e.g., Prompt 62).
  - Batch Sequence 3 $\implies$ `expert_seq_id = 3` (e.g., Prompt 63).
- Each sequence indexes `scheduler->sequences_info[seq_id]` independently. Batch elements route to their genuine, independent expert sets, enabling accurate measurement of batch memory scaling and cache competition.

### 4.3 Preservation of Chronological Token and Layer Ordering
- **Status:** **Guaranteed by Simulator Execution Flow.**
- In `test_deepseekv3_mixtral_style.cpp`:
  - The outer loop steps chronologically: $\text{step} = 0, 1, 2, \dots, N_{\text{tokens}}-1$.
  - Within each step, `cluster->run(metadata)` iterates through all layers sequentially: $\text{cur\_layer} = 0, 1, \dots, \text{num\_layers}-1$.
  - Therefore, token ordering and intra-token layer ordering are strictly preserved.

### 4.4 Compatibility with DRAM and HBM Subsystems
- **Status:** **100% Compatible without Architectural Changes.**
- **Memory Footprint:** Each expert is $88,080,384\text{ bytes}$ ($88.08\text{ MB}$).
- **Burst Sizing:** Ramulator2 fetches experts using 64-byte bursts ($1,376,256\text{ bursts}$ per expert).
- **HBM Cache Management:** `Device::is_expert_resident` checks whether `(cur_layer, expert_id)` is in HBM. On a miss, DDR5 demand transfer is simulated via `issueRamulatorBatch`, followed by `Device::mark_expert_resident` inserting into the 80 GB LRU cache.
- The memory hierarchy functions identically regardless of whether the expert IDs originate from PRNG or real trace replay.

---

## 5. Trace Replay Constraints and Invariants

1. **Strictly Held-Out Evaluation:**
   - MoE Suite: Use ONLY test tasks (`corpus-12`, `corpus-13`). Never mix or leak training tasks.
   - MoE Anywhere: Use ONLY test prompts (prompts 60..73).
2. **No Random Fallback:** If a trace record is missing or corrupt, execution must terminate immediately with an explicit error.
3. **Bounded Layer Replay:** Full 75-stage MoE Suite replay over 128 tokens would require $75 \times 8 \times 128 = 76,800$ expert loads ($>150\text{ hours}$ in Ramulator2). To maintain reproducibility within tractable host runtimes, experiments will evaluate a representative MoE stage block (e.g., 2 MoE stages) replayed across the full token sequence length ($L \in \{128, 256, 512\}$), exactly mirroring the E1-A evaluation methodology.
4. **Zero Ternary Operators:** All newly written Python code enforces `ast.IfExp == 0`.

---

## 6. Audit Conclusion
The existing simulator codebase is fully capable of executing real MoE trace replays. Phase 2 will implement the minimum real-trace replay adapter to populate `scheduler->sequences_info` directly from the Parquet datasets.

