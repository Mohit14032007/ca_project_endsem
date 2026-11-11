# Phase E1-A: Numerical Reconciliation and Architectural Audit

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Phase:** E1-A Follow-up Audit & E1-B Handover  
**Date:** November 2026  
**Status:** COMPLETE  
**Repository Path:** `/home/mohman/ca_project/LLMSimulator`  
**Referenced Artifacts:**
- `docs/phase_e/E1_A_BASELINE_AUDIT.md`
- `docs/phase_e/E1_A_SYNTHETIC_BASELINE_RESULTS.md`
- `results/endsem_phase_e/e1_a_synthetic_baseline/`

---

## 1. Objective and Audit Scope

This document provides a formal forensic audit and numerical reconciliation of the Phase E1-A synthetic baseline characterization. Before implementing real MoE trace replay in Phase E1-B, four critical discrepancies and architectural assumptions in E1-A must be clarified and mathematically resolved:

1. **DDR5 Traffic Numerical Units (MB vs MiB vs Bytes):** Resolving the relationship between expert dimensions, DRAM bursts, and reported memory traffic numbers.
2. **HBM Cache Hit Rate Semantics (Matrix-Level vs Expert-Level):** Distinguishing sub-expert tensor accesses from true expert-level demand routing hits.
3. **Batch Routing Behavior and Working-Set Sharing:** Investigating why throughput scaled linearly with batch size and determining whether batch elements received independent routing decisions.
4. **Scope of the Baseline Model Geometry:** Explicitly defining the 2-MoE-layer representative block evaluation versus full-scale DeepSeek-V3.

In accordance with scientific integrity and project execution rules, **historical E1-A results are preserved as originally produced**, while this reconciliation provides the authoritative, mathematically rigorous interpretation.

---

## 2. Investigation A: DDR5 Traffic and Unit Conversions

### 2.1 Expert Dimensions and Byte Counting
In the DeepSeek-V3 model configuration (`config.yaml`), the architectural dimensions are:
- Model Dimension ($d_{\text{model}}$): $7168$
- Feed-Forward Hidden Dimension ($d_{\text{ffn}}$): $2048$
- Numerical Precision: FP16 ($2\text{ bytes}$ per parameter)

Each routed expert consists of three linear projection weight matrices:
1. Gate projection ($W_{\text{gate}}$): $[d_{\text{model}} \times d_{\text{ffn}}] = 7168 \times 2048 = 14,680,064\text{ parameters} \implies 29,360,128\text{ bytes}$
2. Up projection ($W_{\text{up}}$): $[d_{\text{model}} \times d_{\text{ffn}}] = 7168 \times 2048 = 14,680,064\text{ parameters} \implies 29,360,128\text{ bytes}$
3. Down projection ($W_{\text{down}}$): $[d_{\text{ffn}} \times d_{\text{model}}] = 2048 \times 7168 = 14,680,064\text{ parameters} \implies 29,360,128\text{ bytes}$

$$\text{Bytes per Expert} = 3 \times (7168 \times 2048 \times 2\text{ B}) = 88,080,384\text{ bytes}$$

### 2.2 DRAM Burst Granularity and Total Transferred Bytes
In `LLMSimulator` with Ramulator2:
- Off-chip DDR5 burst size = $64\text{ bytes}$.
- Bursts required per expert:
  $$\text{Bursts per Expert} = \frac{88,080,384\text{ bytes}}{64\text{ bytes/burst}} = 1,376,256\text{ bursts}$$

In the E1-A baseline experiment, each decoding step executes across $2$ MoE layers with $\text{Top-}8$ selection. Under cold demand loading (where no experts are initially resident in HBM), all demanded experts must be fetched from DDR5:
$$\text{Total Demanded Experts} = 2\text{ layers} \times 8\text{ experts/layer} = 16\text{ experts}$$
$$\text{Total Bursts} = 16 \times 1,376,256 = 22,020,096\text{ bursts}$$
$$\text{Total Transferred Bytes} = 22,020,096 \times 64\text{ B} = 1,409,286,144\text{ bytes}$$

### 2.3 Numerical Reconciliation (MB vs MiB)
In the E1-A summary tables, the traffic is reported as **`1344.0 MB`**.
Performing the unit conversions:

| Unit Standard | Unit Name | Calculation | Result |
|---|---|---|---|
| **Raw Bytes** | Bytes ($\text{B}$) | $16 \times 88,080,384$ | **$1,409,286,144\text{ B}$** |
| **Binary IEC** | Mebibytes ($\text{MiB}$) | $1,409,286,144 / (1024^2)$ | **$1,344.0\text{ MiB}$** |
| **Binary IEC** | Gibibytes ($\text{GiB}$) | $1,409,286,144 / (1024^3)$ | **$1.3125\text{ GiB}$** |
| **Decimal SI** | Megabytes ($\text{MB}$) | $1,409,286,144 / (1000^2)$ | **$1,409.286144\text{ MB}$** |
| **Decimal SI** | Gigabytes ($\text{GB}$) | $1,409,286,144 / (1000^3)$ | **$1.409286144\text{ GB}$** |

### 2.4 Conclusion & Source of Discrepancy
The discrepancy arose because the legacy reporting scripts in `run_e1_a_baseline.py` and `LLMSimulator` computed:
$$\text{Traffic} = \frac{\text{Total Bytes}}{1024 \times 1024} = 1344.0$$
and attached the string label **`"MB"`** instead of **`"MiB"`**.
- **Correct Interpretation:** The physical DDR5 traffic is exactly **$1,409,286,144\text{ bytes}$**, which corresponds to **$1,344.0\text{ MiB}$** (binary) or **$1,409.29\text{ MB}$** (decimal).
- **Policy for Phase E1-B:** All Phase E1-B CSVs and documentation must report exact byte counts and explicitly distinguish binary units ($\text{MiB}, \text{GiB}$) from decimal units ($\text{MB}, \text{GB}$).

---

## 3. Investigation B: HBM Hit-Rate Semantics

### 3.1 Trace of Cache Access in `linear.cpp` and `linear_impl.cpp`
In `src/module/linear.cpp` (lines 48–56):
```cpp
A->weight_target = MemoryTarget::HBM;
A->load_sisters = false;
if (A->is_expert_weight && input->getSize() > 0) {
  int layer_id = sequences_metadata->cur_layer;
  A->expert_layer_id = layer_id;
  if (!device->config.offchip_dram_cfg_path.empty()) {
    if (!device->is_expert_resident(layer_id, A->expert_id)) {
      A->weight_target = MemoryTarget::OFFCHIP_DRAM;
      A->load_sisters = true;
    }
  }
}
```

When an expert is executed, the simulator processes its three constituent matrices sequentially:
1. **Matrix 1 ($W_{\text{gate}}$):**
   - Probes `device->is_expert_resident(layer_id, expert_id)`.
   - On cold start, returns `false` $\implies$ **MISS**.
   - Sets `A->weight_target = OFFCHIP_DRAM` and `A->load_sisters = true`.
   - In `linear_impl.cpp` (lines 50–56), `load_sisters` causes all 3 matrices ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) to be packed into `batch_tensors` and fetched from DDR5 via `issueRamulatorBatch`.
   - At line 70, `device->mark_expert_resident(...)` immediately installs the expert into HBM.
   - Logs $W_{\text{gate}}$ to `expert_load.csv` with `cache_hit_or_miss = "miss"`.
2. **Matrix 2 ($W_{\text{up}}$):**
   - Probes `device->is_expert_resident(layer_id, expert_id)`.
   - Because $W_{\text{gate}}$ previously registered the expert, this returns `true` $\implies$ **HIT**.
   - Sets `A->weight_target = HBM`.
   - Logs $W_{\text{up}}$ to `expert_load.csv` with `cache_hit_or_miss = "hit"`.
3. **Matrix 3 ($W_{\text{down}}$):**
   - Probes `device->is_expert_resident(layer_id, expert_id)`.
   - Returns `true` $\implies$ **HIT**.
   - Logs $W_{\text{down}}$ to `expert_load.csv` with `cache_hit_or_miss = "hit"`.

### 3.2 Matrix-Level vs Expert-Level Hit Rates

Under Cold Demand execution:
- Total Matrix Accesses Logged = $16\text{ experts} \times 3\text{ matrices} = 48\text{ accesses}$.
- Matrix Hits = $16 \times 2 = 32\text{ hits}$.
- Matrix Misses = $16 \times 1 = 16\text{ misses}$.
$$\text{Matrix-Level Hit Rate} = \frac{32}{48} = 66.6667\% \approx 66.7\%$$

However, at the semantic level of **Expert Routing Demands**:
- Total Expert Demands = $16$.
- Resident Experts in HBM at time of demand = $0$.
- Non-Resident Cold Experts = $16$.
$$\text{Expert-Level Hit Rate} = \frac{0}{16} = 0.0\%$$

### 3.3 Conclusion & Reporting Mandate
The $66.7\%$ hit rate reported in E1-A is strictly a **sub-expert matrix-level access hit rate**, reflecting that the prefetching of sister weights ($W_{\text{up}}, W_{\text{down}}$) upon a $W_{\text{gate}}$ miss converts subsequent intra-expert accesses into hits.
- **True Expert Cache Performance:** The cold-start expert demand hit rate was **$0.0\%$**.
- **Warm Cache Performance:** In Warm HBM replay, both matrix-level and expert-level hit rates are **$100.0\%$**.
- **Phase E1-B Mandate:** All subsequent reports must explicitly distinguish and separately tabulate:
  1. `Expert Hit Rate` (denominator: total routed expert selections, $N_{\text{demands}}$).
  2. `Matrix Hit Rate` (denominator: total tensor projections, $3 \times N_{\text{demands}}$).

---

## 4. Investigation C: Batch Routing Behavior and Throughput Scaling

### 4.1 Inspection of Batch Generation and Routing Logic
In `src/scheduler/sequence.cpp` (lines 197–233):
```cpp
for (int token_id = current_len;
     token_id < current_len + seq->num_process_token; token_id++) {
  if (seq->get_expert_from_list) {
    // Replay from sequences_info...
  } else {
    std::vector<int> expert_vec;
    auto route_key = std::make_pair(token_id, cur_layer);
    auto it = cached_routing.find(route_key);
    if (it != cached_routing.end()) {
      expert_vec = it->second;
    } else {
      std::set<int> expert_list;
      if (scheduler->model_config.skewness > 0) {
        expert_list = scheduler->getZipfianRandomExpert(skewness_weight, top_k);
      } else {
        expert_list = scheduler->getRandomExpert(top_k);
      }
      for (auto idx : expert_list) {
        expert_vec.push_back(idx);
      }
      cached_routing[route_key] = expert_vec;
    }
    for (auto idx : expert_vec) {
      num_token_in_expert[idx] += 1;
      routing_file << request_id << "," << token_id << "," << cur_layer << "," << k_rank << "," << idx << "\n";
    }
  }
}
```

In `test_deepseekv3_mixtral_style.cpp`, the decoding loop creates $B$ sequences via `scheduler->pushDummySeq(input_len, output_len)`.
1. All sequences initialized in the batch have `seq->current_len = input_len`.
2. At step 0, for sequence $0$:
   - `token_id = input_len`.
   - `route_key = (input_len, cur_layer)`.
   - Not in `cached_routing`. It calls `getRandomExpert(8)` and stores the 8 expert IDs into `cached_routing[route_key]`.
3. For sequences $1, 2, \dots, B-1$ in the same batch:
   - `token_id` is also `input_len`.
   - `route_key` is identical: `(input_len, cur_layer)`.
   - `cached_routing.find(route_key)` succeeds!
   - Every subsequent batch sequence receives the **identical 8 experts**.

### 4.2 Impact on Memory Demands and Linear Throughput
Because every sequence in the batch was assigned the exact same routed experts:
- Total unique experts demanded across the entire batch:
  $$\text{Unique Experts}(B=1) = \text{Unique Experts}(B=2) = \text{Unique Experts}(B=4) = \text{Unique Experts}(B=8) = 16\text{ experts}$$
- Total DDR5 traffic across all batch sizes: exactly **$1,409,286,144\text{ bytes}$ ($1,344.0\text{ MiB}$)**.
- Total DDR5 stall time: exactly **$62.42\text{ ms}$** across all $B$.
- Because $62.42\text{ ms}$ accounts for $96.8\%$ of step time, total step latency rose by only $0.04\%$ ($64.48\text{ ms} \to 64.51\text{ ms}$).
- Therefore, throughput scaled by:
  $$\text{Throughput} = \frac{B}{T_{\text{step}}} \implies \frac{1}{64.48\text{ ms}} = 15.51\text{ tok/s} \quad \longrightarrow \quad \frac{8}{64.51\text{ ms}} = 124.01\text{ tok/s} \quad (8.0\times)$$

### 4.3 Conclusion
The observed linear throughput scaling in E1-A was an artifact of **complete working-set sharing across batch elements** due to the `cached_routing` dictionary keying solely on `(token_id, cur_layer)`.
- In real multi-user LLM serving, independent prompt streams route to **different experts**, which expands the expert working set as $B$ increases until HBM cache thrashing or DDR5 saturation occurs.
- **Phase E1-B Mandate:** Batch replay must preserve **independent token streams** with distinct prompt/task sequences, so that batching evaluates true multi-sequence expert diversity and cache competition.

---

## 5. Investigation D: Scope of the Baseline Model Geometry

### 5.1 Architecture Differences: Full DeepSeek-V3 vs 2-Layer Block
Full DeepSeek-V3 architecture specifies:
- 61 total Transformer layers (1 dense attention layer + 60 MoE layers, or 3 dense + 57 MoE layers).
- 256 routed experts + 1 shared expert per MoE layer.
- $\text{Top-}8$ routing.
- Total expert footprint: $57 \times 256 \times 88.08\text{ MB} = 1.285\text{ TB}$.
- Per-step demanded experts: $57 \times 8 = 456\text{ experts}$ ($40.16\text{ GB}$ per token).

### 5.2 Rationale for Bounded 2-MoE-Layer Evaluation
Simulating a full 60-layer DeepSeek-V3 decoding step in Ramulator2:
- 456 experts $\times 1,376,256\text{ bursts} = 627,572,736\text{ DRAM requests}$.
- At an emulation throughput of $\sim 120,000\text{ bursts/second}$ on the host CPU, a single decoding step takes **$\sim 1.45\text{ hours}$**.
- Running 14 benchmark runs would require **$>20\text{ hours}$** of continuous host simulation.

To enable comprehensive architectural exploration within a practical timeframe ($\sim 45\text{ minutes}$ total host runtime), E1-A configured a representative **2-MoE-layer block**:
- Tested $16$ cold expert transfers ($22,020,096$ DRAM bursts, $1.34\text{ GB}$).
- Exercised the exact same A100 HBM3 cache, DDR5-3200 timing parameters, and `RoBaRaCoCh` DRAM address mapping.
- Validated LRU eviction, burst batching, and cache hits.

### 5.3 Formal Scope Statement
> **Phase E1-A Scope:** Phase E1-A measures the cycle-accurate hardware performance of a **representative two-MoE-layer execution block** ($K=8$, 16 expert demands per step). It does not represent full-model 60-layer DeepSeek-V3 end-to-end inference. All metrics must be understood as per-block performance counters.

---

## 6. Summary of Numerical Reconciliations

| Metric / Dimension | E1-A Historical Report | Reconciled Value | Reason / Explanation |
|---|---|---|---|
| **DDR5 Transferred Bytes** | 1344.0 MB | **$1,409,286,144\text{ bytes}$** | Exact DRAM burst count ($22,020,096 \times 64\text{ B}$). |
| **DDR5 Traffic (Binary)** | 1344.0 MB | **$1,344.0\text{ MiB}$** ($1.3125\text{ GiB}$) | Calculated with divisor $1024^2$, mislabeled as "MB". |
| **DDR5 Traffic (Decimal)** | 1344.0 MB | **$1,409.29\text{ MB}$** ($1.4093\text{ GB}$) | Calculated with divisor $1000^2$. |
| **Cold Cache Hit Rate** | 66.7% | **$66.7\%$ Matrix / $0.0\%$ Expert** | Counts sub-expert matrix hits ($W_{\text{up}}, W_{\text{down}}$); expert hit rate was $0\%$. |
| **Warm Cache Hit Rate** | 100.0% | **$100.0\%$ Matrix / $100.0\%$ Expert** | All experts resident; zero DDR5 demand traffic. |
| **Batch Working Set** | Identical ($16$ experts) | **Shared due to `cached_routing`** | Key `(token_id, cur_layer)` assigned same random experts to all batch elements. |
| **Model Scope** | DeepSeek-V3 Preset | **2-MoE-Layer Representative Block** | Preserved exact DDR5/HBM geometry while maintaining tractable host emulation times. |

---

## 7. Deliverable Sign-off
This reconciliation document resolves all numerical and semantic ambiguities from Phase E1-A and establishes the required invariants and metric reporting standards for Phase E1-B real-trace replay.

