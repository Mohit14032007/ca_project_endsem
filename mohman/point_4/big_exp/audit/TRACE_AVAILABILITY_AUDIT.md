# Phase A.1: Trace Availability and Provenance Audit

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Campaign:** Big Experiment: LRU vs. FA-LRU ($W=64$) Under Large MoE Workloads  
**Evaluation Target:** Authentic Generation Horizon $T=128$ for Workloads W3 ($B=14$) and W4 ($B=32$)  
**Status:** AUDIT PASSED (100% Authentic, Zero Synthetic Fallback)  
**Date:** 2026-11-10  

---

## 1. Executive Summary

This audit establishes the provenance, integrity, and availability of the real-world routing traces required for the Big Experiment evaluating Standard LRU versus Frequency-Aware LRU ($W=64$). The target experimental workloads require:
- **Workload W3:** Batch size $B=14$, generation length $T=128$, active layers $L=40$, Top-$K=8$.
- **Workload W4:** Batch size $B=32$, generation length $T=128$, active layers $L=40$, Top-$K=8$.

Our pre-experiment forensic inspection of the underlying MoE trace dataset confirms that **authentic generation horizons of $T=128$ tokens are fully available and contiguous** for all required sequences, requiring zero synthetic repetition, zero token extrapolation, and zero artificial routing generation.

---

## 2. Dataset Source and Structural Geometry

The routing traces are extracted directly from the authentic MoE-Anywhere trace repository:
- **Archive File Path:** [`LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet`](file:///home/mohman/ca_project/LLMSimulator/data/downloads/moe_anywhere/e1-sage-a1.5.parquet)
- **Model Architecture:** DeepSeek-V3 Mixtral-style MoE (43 layers: 3 dense, 40 MoE stages, 256 routed experts per stage, Top-$K=8$).
- **Total Dataset Size:** 760,720 routing records covering 74 independent prompts (Prompt IDs 0 through 73).
- **Available Generation Steps Per Prompt:** Exactly 257 contiguous token generation steps are recorded for every single prompt in the dataset.
- **Maximum Valid Horizon:** Because 257 contiguous token steps are recorded, the requested target of $T=128$ tokens is fully covered ($128 \le 257$) with zero truncation.

---

## 3. Workload Composition and Dataset Split Provenance

In strict compliance with experimental standards, **no prompts are ever duplicated** to assemble batch dimensions. Every sequence in a batch corresponds to an authentic, independent conversation prompt.

### 3.1 Workload W3 ($B=14, T=128, L=40, K=8$)
- **Batch Size:** 14 concurrent sequences.
- **Prompt IDs:** Prompts 60 through 73.
- **Split Partition:** **100% Held-Out Test Split.** Zero contamination from training or validation partitions.
- **Task Domain Diversity:**
  - `tools` (3 prompts: 60, 61, 62)
  - `summarize` (3 prompts: 63, 64, 65)
  - `translate` (4 prompts: 66, 67, 68, 69)
  - `creative` (4 prompts: 70, 71, 72, 73)
- **Contiguity Check:** Verified strictly contiguous across all 128 generation tokens ($token\_pos_{i+1} = token\_pos_i + 1$).
- **Total Routing Records:** $14 \text{ seqs} \times 128 \text{ tokens} \times 40 \text{ stages} \times 8 \text{ experts} = 573,440 \text{ records}$.

### 3.2 Workload W4 ($B=32, T=128, L=40, K=8$)
- **Batch Size:** 32 concurrent sequences.
- **Prompt IDs:** 32 unique, non-overlapping prompts:
  - Prompts 60–73 (14 prompts from held-out **Test** split)
  - Prompts 52–59 (8 prompts from **Validation** split)
  - Prompts 0–9 (10 prompts from **Train** split)
- **Contiguity Check:** Verified strictly contiguous across all 128 generation tokens for all 32 prompts.
- **Total Routing Records:** $32 \text{ seqs} \times 128 \text{ tokens} \times 40 \text{ stages} \times 8 \text{ experts} = 1,310,720 \text{ records}$.

### Full Disclosure of W4 Split Composition & Limitations
The MoE-Anywhere dataset contains exactly 14 held-out test prompts. To evaluate a realistic serving batch size of $B=32$ without duplicating prompts, W4 incorporates 8 validation prompts and 10 training prompts. 

**Scientific Implications:**
1. **Cache Policy Evaluation Validity:** Because this is an HBM cache management experiment (measuring expert residency, eviction victim selection, and bus traffic), cross-split prompt mixing does not compromise experimental validity. Cache dynamics depend solely on the spatial and temporal reuse patterns of expert parameters, not model training convergence.
2. **Predictor Evaluation Context:** For configurations utilizing CSTP prefetching (C3 and C4), the CSTP predictor operates in online tracking mode ($H=1, K=4$) on the ingested sequence history. Both LRU and FA-LRU are evaluated on the exact same mixed trace, guaranteeing fair paired comparisons.

---

## 4. Sequence Mapping and Token Horizon Verification

The table below summarizes the audited sequence parameters from [`mohman/point_4/big_exp/traces/trace_manifest.csv`](file:///home/mohman/ca_project/mohman/point_4/big_exp/traces/trace_manifest.csv):

| Workload | Sequence Range | Prompt Range | Splits Included | Prefill Start Token | Horizon Used ($T$) | Records Exported |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **W3** | `seq_0` – `seq_13` | 60 – 73 | Test (14) | 17 – 44 | 128 tokens | 573,440 |
| **W4** | `seq_0` – `seq_31` | 60–73, 52–59, 0–9 | Test (14), Val (8), Train (10) | 17 – 44 | 128 tokens | 1,310,720 |

---

## 5. Audit Conclusion

The trace availability audit passes with zero exceptions:
- Authentic $T=128$ token horizons are confirmed available.
- Zero synthetic tokens or random fallbacks are utilized.
- All 1,884,160 total routing records across both workloads have been exported and certified.

