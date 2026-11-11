# Comparative Study Scientific Integrity & Certification Audit

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Deliverable:** Point 1 — Comparative Study Certification  
**Path:** [`mohman/point_1/stmoe_cstp/validation/comparative_integrity_report.md`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/validation/comparative_integrity_report.md)  
**Execution Timestamp:** 2026-11-09  
**Regression Test Suite:** [`mohman/point_1/cstp/tests/test_phase1a_certified_correction.py`](file:///home/mohman/ca_project/mohman/point_1/cstp/tests/test_phase1a_certified_correction.py)  

---

## 1. Executive Summary & Verification Clearance

This document certifies that all comparative empirical results presented between CSTP and ST-MoE are derived from mathematically verified, causally isolated, and statistically robust artifacts.

Zero experiments, benchmarks, evaluations, or profiling workloads were executed during this packaging process. All reported numbers originate directly from the frozen certified datasets generated during Phase 1A.

---

## 2. Itemized Verification of Resolved Methodological Issues

### 2.1 Complete Untruncated Event Streaming
- **Prior Issue:** Earlier uncertified iterations truncated per-event CSV exports to the first 100 events per configuration.
- **Certified Solution:** The certified pipeline streams and preserves all **2,332,800** individual causal prediction events (`corrected_per_event_metrics.csv`, 222 MB).
- **Verification:** Unit test `test_01_event_counts_match_declared_opportunities` passed, confirming exactly:
  - MoE Anywhere: $16,800 \text{ opps} \times 96 \text{ configs} = 1,612,800$ rows.
  - MoE Suite: $7,500 \text{ opps} \times 96 \text{ configs} = 720,000$ rows.

### 2.2 Strict Dynamic Denominators
- **Prior Issue:** Earlier iterations hardcoded precision and recall denominators to 8.
- **Certified Solution:** Denominators are dynamically computed for every opportunity:
  $$\text{Recall}_i = \frac{|P_i \cap G_i|}{|G_i|}, \quad \text{Precision}_i = \frac{|P_i \cap G_i|}{|P_i|}$$
- **Verification:** Unit test `test_02_metrics_dynamic_denominators` verified across all event records that $|P_i| = 8$ and $|G_i| = 8$, mathematically ensuring that precision equals recall for matched budget $B = K = 8$.

### 2.3 MoE Suite Statistical Limitation ($N=2$ Tasks)
- **Prior Issue:** Reporting MoE Suite confidence intervals as strong inferential evidence despite having only two test tasks.
- **Certified Solution:** All GLM-5.2/MoE Suite comparisons are formally categorized and labeled as **`DESCRIPTIVE_ONLY_N2_TASKS`** in all CSVs, plots, and tables. No inferential generalization claims are made from this trace.
- **Verification:** Unit test `test_05_moe_suite_explicitly_tagged_descriptive` passed with 100% compliance.

### 2.4 Multi-Seed Hierarchical Bootstrap & Valid Two-Sided ASL
- **Prior Issue:** Bootstrap resampling evaluated only Seed 42 and reported pseudo-$p$-values.
- **Certified Solution:** A two-level hierarchical bootstrap was implemented:
  - Stage 1: Resample profiling seeds from $\{42, 123, 999\}$.
  - Stage 2: Resample prompt units with replacement.
  - Replaced informal pseudo-$p$-values with valid bootstrap Achieved Significance Levels: $\text{ASL} = 2 \times \min(p_{\text{boot}}, 1 - p_{\text{boot}})$.
- **Verification:** Unit tests `test_03_hierarchical_bootstrap_uses_all_three_seeds` and `test_04_p_value_greater_than_zero_renamed` passed.

### 2.5 Macro F1 Mathematical Precision
- **Prior Issue:** Conflating macro F1 with the harmonic mean of macro precision and macro recall.
- **Certified Solution:** Macro F1 is computed by calculating $F1_u = \frac{2 P_u R_u}{P_u + R_u}$ on each individual prompt unit $u$, and then averaging across prompts: $\frac{1}{U} \sum_u F1_u$.

---

## 3. Preservation of Historical and Baseline Repositories

Lightweight cryptographic hashing verified that:
1. **`mohman/point_1/st_moe/`:** All 37 existing files remain 100% bit-for-bit identical to baseline hashes recorded at audit start. Zero modifications occurred.
2. **`results/phase1a_cstp_profiling/csv/`:** All 7 historical CSVs remain 100% bit-for-bit identical to baseline hashes.
3. **Copied Deliverables:** Every file copied into `mohman/point_1/cstp/` and `mohman/point_1/stmoe_cstp/` matched its source SHA-256 hash and byte size.

