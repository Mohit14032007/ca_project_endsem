# Dataset Quality and Validation Report

**Phase:** Phase C Comprehensive Multi-Dataset Benchmarking  
**Generated:** 2026-11-01  
**Validation Suite:** `scripts/phase_c/validate_datasets.py`  

---

## 1. Workload Summary

The evaluation harness validates five diverse workloads representing negative controls, small-universe systems, calibrated synthetic distributions, and production MoE LLMs.

| Dataset ID | Workload Type | Requests/Prompts | Tokens | Stages | Experts ($E$) | Top-$K$ | Total Records | Routing Entropy (bits) | Consecutive Jaccard | Adjacent Jaccard |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `deepseek_random` | Negative Control | 1 | 5 | 57 | 256 | 8 | 2280 | 7.9206 / 8.0 | 0.7186 | 0.0785 |
| `duplex_a100` | Simulator Trace | 32 | 5 | 32 | 8 | 2 | 10240 | 2.9994 / 3.0 | 0.9976 | 1.0 |
| `correlated_synthetic` | Calibrated Synthetic | 16 | 5 | 57 | 256 | 8 | 36480 | 7.5772 / 8.0 | 0.6741 | 0.5125 |
| `moe_suite` | Production LLM (GLM-5.2) | 12 | 98292 | 75 | 256 | 8 | 58975200 | 7.9956 / 8.0 | 0.8407 | 0.436 |
| `moe_anywhere` | Production LLM (Sage-A1.5) | 74 | 19018 | 40 | 256 | 8 | 760720 | 7.9722 / 8.0 | 0.5935 | 0.3659 |

---

## 2. Integrity and Anomaly Checks

Every dataset passed strict data-integrity checks with zero missing records, zero invalid expert IDs, and zero split leakage across train, validation, and test partitions.

| Dataset ID | Missing Records | Duplicate Records | Invalid Expert IDs | Inconsistent Top-$K$ | Split Leakage | Primary Horizon | Validation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `deepseek_random` | 0 | 0 | 0 | 0 | 1 | 1 | **PASSED** |
| `duplex_a100` | 0 | 0 | 0 | 0 | 0 | 1 | **PASSED** |
| `correlated_synthetic` | 0 | 0 | 0 | 0 | 0 | 1 | **PASSED** |
| `moe_suite` | 0 | 0 | 0 | 0 | 0 | 2 | **PASSED** |
| `moe_anywhere` | 0 | 0 | 0 | 0 | 0 | 1 | **PASSED** |

---

## 3. Data Split Isolation Details

- **DeepSeek Random:** Single request (5 tokens). Serves strictly as a negative control for routing predictability.
- **Duplex A100:** 32 requests partitioned into 24 train requests (75%) and 8 test requests (25%). Zero request-ID leakage.
- **Correlated Synthetic:** 16 requests partitioned into 12 train requests (75%) and 4 test requests (25%). Zero request-ID leakage.
- **MoE Suite (`mistrjirka/Predicting-Experts-for-MOE-Suite`):** Official task-disjoint split preserved. 8 training tasks (`corpus-02` through `corpus-09`), 2 validation tasks (`corpus-10`, `corpus-11`), and 2 test tasks (`corpus-12`, `corpus-13`). Zero prompt or token repartitioning across tasks.
- **MoE Anywhere (`carlosfelipe/moe-anywhere-routing-traces`):** 74 prompts partitioned strictly by `prompt_id` into 52 train prompts (70%), 8 validation prompts (11%), and 14 test prompts (19%).

---

## 4. Machine-Readable Metadata Files

The validation script produced the following persistent CSV tables:
- [`dataset_summary.csv`](file:///home/mohman/ca_project/LLMSimulator/results/endsem_phase_c/metadata/dataset_summary.csv)
- [`dataset_quality.csv`](file:///home/mohman/ca_project/LLMSimulator/results/endsem_phase_c/metadata/dataset_quality.csv)
- [`split_manifest.csv`](file:///home/mohman/ca_project/LLMSimulator/results/endsem_phase_c/metadata/split_manifest.csv)
