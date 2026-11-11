# Phase 1 Model Configuration & Provenance Verification

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Campaign Stage:** Stage 2B.1 — Phase 1 Results Reconciliation Audit  
**Authoritative Git Baseline Commit:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Deliverable File:** `docs/final_campaign/phase1_reconciliation/PHASE1_MODEL_CONFIG_VERIFICATION.md`  
**Audit Classification:** READ-ONLY Verification  
**Date:** November 2026  

---

## 1. Audit Objective & Scope

This audit formally verifies the effective model parameters, architecture provenance, and configuration overrides for Workloads **A1** and **B2** evaluated during Phase 1. 

Specifically, this audit resolves whether:
1. Setting `model_name: deepseekV3` in YAML files is a functional architecture selector or purely a descriptive label.
2. The simulated model dimensions for A1 and B2 match the expected experimental specifications from the Phase E baseline.
3. The names "GLM-5.2" and "Sage-A1.5" in historical reports represent distinct hardcoded C++ structs or benchmark trace sources executed on the DeepSeek-V3 execution template.

---

## 2. Source Code Architecture & Configuration Tracing

### 2.1 The Model Parameter Template (`src/model/model_config.h`)
In `src/model/model_config.h` (lines 107–110), the `deepseekV3` configuration struct is defined with the following baseline parameters:

```cpp
static ModelConfig deepseekV3 =
    ModelConfig(7168, 128, 60, 128, 128, 131072, 18432, 2048, 1, 1, 256, 1, 1, 8,
                3, 3, 1536, 512, 128, 64, 129280, true, true, 0.0, "deepseekV3");
```

Parameter-by-parameter mapping against the `ModelConfig` constructor (`model_config.h:20-52`):
- `hidden_dim`: **7,168**
- `head_dim`: **128**
- `num_layers`: **60** (overridden by YAML configuration)
- `num_heads`: **128**
- `num_kv_heads`: **128**
- `max_seq_len`: **131,072**
- `intermediate_dim`: **18,432** (dense FFN intermediate dimension)
- `expert_intermediate_dim`: **2,048** (MoE expert projection dimension)
- `activation_factor`: **1**
- `precision_byte`: **1** (overridden by YAML `precision_byte: 2` for FP16)
- `num_routed_expert`: **256**
- `num_shared_expert`: **1**
- `expert_freq`: **1** (every MoE layer contains routed experts)
- `top_k`: **8** (each token activates 8 routed experts)
- `ffn_way`: **3** (`FeedForward3Way`: gate, up, and down projection matrices)
- `first_k_dense`: **3** (first 3 transformer stages are dense layers)
- `q_lora_rank`: **1,536** (MLA query compression rank)
- `kv_lora_rank`: **512** (MLA KV compression rank)
- `qk_nope_head_dim`: **128**
- `qk_rope_head_dim`: **64**
- `n_vocab`: **129,280**
- `compressed_kv`: `true`
- `use_absorb`: `true`
- `model_name`: `"deepseekV3"`

### 2.2 Configuration Instantiation in Harness (`eval/test_deepseekv3_mixtral_style.cpp`)
In `eval/test_deepseekv3_mixtral_style.cpp` (lines 232–250):
```cpp
  ModelConfig model_config = deepseekV3;
  if (model_name == "debugV3") {
    model_config = debugV3;
  }
  if (config["model"]["num_layers"]) {
    model_config.num_layers = config["model"]["num_layers"].as<int>();
  }
  model_config.e_tp_dg = e_tp_dg;
  model_config.ne_tp_dg = ne_tp_dg;
  model_config.precision_byte = precision_byte;
  model_config.input_len = input_len;
  model_config.output_len = output_len;
  model_config.skewness = skewness;
  model_config.dataset = data_name;
```

**Key Discovery:** The executable `run_deepseekv3_mixtral_style` exclusively initializes `model_config` from `deepseekV3` (or `debugV3` for test mocks). There is no hardcoded struct named `glm_5_2` or `sage_a1_5` in `src/model/model_config.h`.

Instead, the simulator uses the **DeepSeek-V3 architectural template**, overriding `num_layers` to match the exact layer depth of the target workload trace.

---

## 3. Workload Verification & Parameter Breakdown

### 3.1 Workload A1 Verification

| Architectural Attribute | Specification in Phase 1 Harness | Authoritative Baseline (`A1.yaml`) | Status |
| :--- | :--- | :--- | :--- |
| **Model Config Source** | `deepseekV3` template | `deepseekV3` template | **VERIFIED** |
| **Total Layers (`num_layers`)** | 78 layers | 78 layers | **VERIFIED** |
| **Dense Stages (`first_k_dense`)** | 3 dense layers | 3 dense layers | **VERIFIED** |
| **MoE Stages (`moe_layers`)** | 75 MoE layers ($78 - 3$) | 75 MoE layers ($78 - 3$) | **VERIFIED** |
| **Routed Experts per Layer** | 256 | 256 | **VERIFIED** |
| **Routing Top-K** | 8 | 8 | **VERIFIED** |
| **Shared Experts per Layer** | 1 | 1 | **VERIFIED** |
| **Hidden Dimension ($d_{\text{model}}$)** | 7,168 | 7,168 | **VERIFIED** |
| **Expert Intermediate Dimension ($d_{\text{ffn}}$)** | 2,048 | 2,048 | **VERIFIED** |
| **Precision** | FP16 (2 bytes) | FP16 (2 bytes) | **VERIFIED** |
| **Expert Weight Breakdown** | - $W_{1,\text{gate}}: 7,168 \times 2,048 \times 2 = 29,360,128\text{ B}$<br/>- $W_{3,\text{up}}: 7,168 \times 2,048 \times 2 = 29,360,128\text{ B}$<br/>- $W_{2,\text{down}}: 2,048 \times 7,168 \times 2 = 29,360,128\text{ B}$ | Same ($3 \times 29.36\text{ MB}$) | **VERIFIED** |
| **Total Single Expert Footprint** | **88,080,384 bytes (~88.08 MB)** | **88,080,384 bytes (~88.08 MB)** | **VERIFIED** |
| **Batch Size ($B$)** | 1 | 1 | **VERIFIED** |
| **Decoding Steps (`iter`)** | 128 tokens | 128 tokens | **VERIFIED** |
| **Input Context Length ($L_{\text{in}}$)** | 128 | 128 | **VERIFIED** |
| **Replayed Trace File** | `results/endsem_phase_e/final_baseline/traces/ms_task12_t128_l75.csv` | Same | **VERIFIED** |
| **Trace SHA256 Checksum** | `5a635dece5747e78be7f600082c888a7f9e9d2ee50e1ae829d338f75f235ea1c` | Same | **VERIFIED** |

**Classification for Workload A1:** **VERIFIED**  
*Provenance Resolution:* In earlier research reports, Workload A1 was designated as "GLM-5.2" because the routing trace was generated from MoE Suite Task 12 (GLM-5.2 benchmark). In the simulation engine, this workload is simulated via the DeepSeek-V3 parameter template scaled to 78 total layers (75 MoE layers).

---

### 3.2 Workload B2 Verification

| Architectural Attribute | Specification in Phase 1 Harness | Authoritative Baseline (`B2.yaml`) | Status |
| :--- | :--- | :--- | :--- |
| **Model Config Source** | `deepseekV3` template | `deepseekV3` template | **VERIFIED** |
| **Total Layers (`num_layers`)** | 43 layers | 43 layers | **VERIFIED** |
| **Dense Stages (`first_k_dense`)** | 3 dense layers | 3 dense layers | **VERIFIED** |
| **MoE Stages (`moe_layers`)** | 40 MoE layers ($43 - 3$) | 40 MoE layers ($43 - 3$) | **VERIFIED** |
| **Routed Experts per Layer** | 256 | 256 | **VERIFIED** |
| **Routing Top-K** | 8 | 8 | **VERIFIED** |
| **Shared Experts per Layer** | 1 | 1 | **VERIFIED** |
| **Hidden Dimension ($d_{\text{model}}$)** | 7,168 | 7,168 | **VERIFIED** |
| **Expert Intermediate Dimension ($d_{\text{ffn}}$)** | 2,048 | 2,048 | **VERIFIED** |
| **Precision** | FP16 (2 bytes) | FP16 (2 bytes) | **VERIFIED** |
| **Expert Weight Breakdown** | - $W_{1,\text{gate}}: 7,168 \times 2,048 \times 2 = 29,360,128\text{ B}$<br/>- $W_{3,\text{up}}: 7,168 \times 2,048 \times 2 = 29,360,128\text{ B}$<br/>- $W_{2,\text{down}}: 2,048 \times 7,168 \times 2 = 29,360,128\text{ B}$ | Same ($3 \times 29.36\text{ MB}$) | **VERIFIED** |
| **Total Single Expert Footprint** | **88,080,384 bytes (~88.08 MB)** | **88,080,384 bytes (~88.08 MB)** | **VERIFIED** |
| **Batch Size ($B$)** | 4 concurrent request streams | 4 concurrent request streams | **VERIFIED** |
| **Decoding Steps (`iter`)** | 128 tokens | 128 tokens | **VERIFIED** |
| **Input Context Length ($L_{\text{in}}$)** | 128 | 128 | **VERIFIED** |
| **Replayed Trace File** | `results/endsem_phase_e/final_baseline/traces/ma_b4_t128_l40.csv` | Same | **VERIFIED** |
| **Trace SHA256 Checksum** | `d53226b68966d79ecbbab4edab99f6f7f5506edb1bc97f62fd06d2a0365e86a0` | Same | **VERIFIED** |

**Classification for Workload B2:** **VERIFIED**  
*Provenance Resolution:* In earlier research reports, Workload B2 was designated as "Sage-A1.5" because the routing trace was generated from the MoE Anywhere benchmark (Sage-A1.5, batch concurrency $B=4$). In the simulation engine, this workload is simulated via the DeepSeek-V3 parameter template scaled to 43 total layers (40 MoE layers).

---

## 4. Architectural Geometry & Memory Bounds Check

During the pre-flight checks of `test_deepseekv3_mixtral_style.cpp` (lines 252–276), the DDR5 physical geometry mapping was validated against the exact layer counts:

1. **Workload A1 (75 MoE stages):**
   - Total expert footprint: $75 \times 256 \times 88,080,384 = 1,691,143,372,800\text{ bytes} \approx 1.691\text{ TB}$.
   - DDR5 Capacity: $2.75\text{ TB}$ ($10,485,760\text{ rows}$).
   - Maximum production row index: $6,451,200$ (Row utilization: $61.52\% < 100\%$).
2. **Workload B2 (40 MoE stages):**
   - Total expert footprint: $40 \times 256 \times 88,080,384 = 901,943,132,160\text{ bytes} \approx 0.902\text{ TB}$.
   - Maximum production row index: $3,440,640$ (Row utilization: $32.81\% < 100\%$).

Both workloads fall strictly within DDR5 backing bounds and obey identical single-server rank interleaving geometry.

---

## 5. Audit Conclusion

The effective model configurations executed in Phase 1 for Workloads A1 and B2 are **100% consistent and verified** with the authoritative baseline anchor `d5bf61ff0c066c06f72173189f9924329f613a88`. The `model_name: deepseekV3` parameter correctly invokes the unified MoE hardware execution model across both workloads.
