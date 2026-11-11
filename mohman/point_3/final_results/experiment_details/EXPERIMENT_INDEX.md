# Point 3 — Comprehensive Experiment Index

**Repository:** [`/home/mohman/ca_project/LLMSimulator`](file:///home/mohman/ca_project/LLMSimulator)
**Deliverable Path:** [`mohman/point_3/final_results/experiment_details/EXPERIMENT_INDEX.md`](file:///home/mohman/ca_project/mohman/point_3/final_results/experiment_details/EXPERIMENT_INDEX.md)
**Status:** Complete and Independently Verified (73 Execution Runs)

This document catalogs all 73 simulation experiments executed for **Point 3: Prefetching Effectiveness** across three campaigns:
1. **Extended Large-Workload Campaign** (27 runs, Workloads W1 to W7, 40 stages, 35 GB HBM)
2. **Original 20-Run Campaign** (20 runs, B=1 to 8, 3 stages, 80 GB HBM)
3. **Phase 2 40-Stage Scaling Campaign** (26 runs, B=1 to 8, 40 stages, 30 GB vs 80 GB HBM)

---

## 1. Extended Large-Workload Campaign (27 Runs)

| Run ID | Workload | Batch ($B$) | Length ($T$) | Stages | Experts | Top-$K$ | HBM (GB) | Predictor | Horizon ($H$) | Budget ($C$) | Max Out | Config Link | Exit Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `w1_baseline` | **W1** | 14 | 4 | 40 | 256 | 8 | 35.0 | Baseline | 0 | 0 | 4 | [`w1_baseline.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/baseline/w1_baseline.yaml) | SUCCESS (0) |
| `w1_stmoe_h2` | **W1** | 14 | 4 | 40 | 256 | 8 | 35.0 | ST-MoE | 2 | 4 | 4 | [`w1_stmoe_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/w1_stmoe_h2.yaml) | SUCCESS (0) |
| `w1_cstp_h2` | **W1** | 14 | 4 | 40 | 256 | 8 | 35.0 | CSTP | 2 | 4 | 4 | [`w1_cstp_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/w1_cstp_h2.yaml) | SUCCESS (0) |
| `w1_stmoe_h1` | **W1** | 14 | 4 | 40 | 256 | 8 | 35.0 | ST-MoE | 1 | 4 | 4 | [`w1_stmoe_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/w1_stmoe_h1.yaml) | SUCCESS (0) |
| `w1_cstp_h1` | **W1** | 14 | 4 | 40 | 256 | 8 | 35.0 | CSTP | 1 | 4 | 4 | [`w1_cstp_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/w1_cstp_h1.yaml) | SUCCESS (0) |
| `w2_baseline` | **W2** | 14 | 16 | 40 | 256 | 8 | 35.0 | Baseline | 0 | 0 | 4 | [`w2_baseline.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/baseline/w2_baseline.yaml) | SUCCESS (0) |
| `w2_stmoe_h2` | **W2** | 14 | 16 | 40 | 256 | 8 | 35.0 | ST-MoE | 2 | 4 | 4 | [`w2_stmoe_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/w2_stmoe_h2.yaml) | SUCCESS (0) |
| `w2_cstp_h2` | **W2** | 14 | 16 | 40 | 256 | 8 | 35.0 | CSTP | 2 | 4 | 4 | [`w2_cstp_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/w2_cstp_h2.yaml) | SUCCESS (0) |
| `w2_stmoe_h1` | **W2** | 14 | 16 | 40 | 256 | 8 | 35.0 | ST-MoE | 1 | 4 | 4 | [`w2_stmoe_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/w2_stmoe_h1.yaml) | SUCCESS (0) |
| `w2_cstp_h1` | **W2** | 14 | 16 | 40 | 256 | 8 | 35.0 | CSTP | 1 | 4 | 4 | [`w2_cstp_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/w2_cstp_h1.yaml) | SUCCESS (0) |
| `w3_baseline` | **W3** | 32 | 16 | 40 | 256 | 8 | 35.0 | Baseline | 0 | 0 | 4 | [`w3_baseline.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/baseline/w3_baseline.yaml) | SUCCESS (0) |
| `w3_stmoe_h2` | **W3** | 32 | 16 | 40 | 256 | 8 | 35.0 | ST-MoE | 2 | 4 | 4 | [`w3_stmoe_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/w3_stmoe_h2.yaml) | SUCCESS (0) |
| `w3_cstp_h2` | **W3** | 32 | 16 | 40 | 256 | 8 | 35.0 | CSTP | 2 | 4 | 4 | [`w3_cstp_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/w3_cstp_h2.yaml) | SUCCESS (0) |
| `w4_baseline` | **W4** | 48 | 16 | 40 | 256 | 8 | 35.0 | Baseline | 0 | 0 | 4 | [`w4_baseline.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/baseline/w4_baseline.yaml) | SUCCESS (0) |
| `w4_stmoe_h2` | **W4** | 48 | 16 | 40 | 256 | 8 | 35.0 | ST-MoE | 2 | 4 | 4 | [`w4_stmoe_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/w4_stmoe_h2.yaml) | SUCCESS (0) |
| `w4_cstp_h2` | **W4** | 48 | 16 | 40 | 256 | 8 | 35.0 | CSTP | 2 | 4 | 4 | [`w4_cstp_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/w4_cstp_h2.yaml) | SUCCESS (0) |
| `w5_baseline` | **W5** | 70 | 16 | 40 | 256 | 8 | 35.0 | Baseline | 0 | 0 | 4 | [`w5_baseline.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/baseline/w5_baseline.yaml) | SUCCESS (0) |
| `w5_stmoe_h2` | **W5** | 70 | 16 | 40 | 256 | 8 | 35.0 | ST-MoE | 2 | 4 | 4 | [`w5_stmoe_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/w5_stmoe_h2.yaml) | SUCCESS (0) |
| `w5_cstp_h2` | **W5** | 70 | 16 | 40 | 256 | 8 | 35.0 | CSTP | 2 | 4 | 4 | [`w5_cstp_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/w5_cstp_h2.yaml) | SUCCESS (0) |
| `w5_stmoe_h1` | **W5** | 70 | 16 | 40 | 256 | 8 | 35.0 | ST-MoE | 1 | 4 | 4 | [`w5_stmoe_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/w5_stmoe_h1.yaml) | SUCCESS (0) |
| `w5_cstp_h1` | **W5** | 70 | 16 | 40 | 256 | 8 | 35.0 | CSTP | 1 | 4 | 4 | [`w5_cstp_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/w5_cstp_h1.yaml) | SUCCESS (0) |
| `w6_baseline` | **W6** | 70 | 32 | 40 | 256 | 8 | 35.0 | Baseline | 0 | 0 | 4 | [`w6_baseline.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/baseline/w6_baseline.yaml) | SUCCESS (0) |
| `w6_stmoe_h2` | **W6** | 70 | 32 | 40 | 256 | 8 | 35.0 | ST-MoE | 2 | 4 | 4 | [`w6_stmoe_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/w6_stmoe_h2.yaml) | SUCCESS (0) |
| `w6_cstp_h2` | **W6** | 70 | 32 | 40 | 256 | 8 | 35.0 | CSTP | 2 | 4 | 4 | [`w6_cstp_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/w6_cstp_h2.yaml) | SUCCESS (0) |
| `w7_baseline` | **W7** | 70 | 64 | 40 | 256 | 8 | 35.0 | Baseline | 0 | 0 | 4 | [`w7_baseline.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/baseline/w7_baseline.yaml) | SUCCESS (0) |
| `w7_stmoe_h2` | **W7** | 70 | 64 | 40 | 256 | 8 | 35.0 | ST-MoE | 2 | 4 | 4 | [`w7_stmoe_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/w7_stmoe_h2.yaml) | SUCCESS (0) |
| `w7_cstp_h2` | **W7** | 70 | 64 | 40 | 256 | 8 | 35.0 | CSTP | 2 | 4 | 4 | [`w7_cstp_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/w7_cstp_h2.yaml) | SUCCESS (0) |

---

## 2. Original Sensitivity Campaign (20 Runs)

| Run ID | Workload | Batch ($B$) | Length ($T$) | Stages | Experts | Top-$K$ | HBM (GB) | Predictor | Horizon ($H$) | Budget ($C$) | Max Out | Config Link | Exit Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `baseline_b1` | **ORIG_B1** | 1 | 4 | 3 | 256 | 8 | 80.0 | Baseline | 0 | 0 | 4 | [`baseline_b1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/baseline/baseline_b1.yaml) | SUCCESS (0) |
| `baseline_b2` | **ORIG_B2** | 2 | 4 | 3 | 256 | 8 | 80.0 | Baseline | 0 | 0 | 4 | [`baseline_b2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/baseline/baseline_b2.yaml) | SUCCESS (0) |
| `baseline_b4` | **ORIG_B4** | 4 | 4 | 3 | 256 | 8 | 80.0 | Baseline | 0 | 0 | 4 | [`baseline_b4.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/baseline/baseline_b4.yaml) | SUCCESS (0) |
| `baseline_b8` | **ORIG_B8** | 8 | 4 | 3 | 256 | 8 | 80.0 | Baseline | 0 | 0 | 4 | [`baseline_b8.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/baseline/baseline_b8.yaml) | SUCCESS (0) |
| `stmoe_b1_h1` | **ORIG_B1** | 1 | 4 | 3 | 256 | 8 | 80.0 | ST-MoE | 1 | 4 | 4 | [`stmoe_b1_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/stmoe_b1_h1.yaml) | SUCCESS (0) |
| `stmoe_b2_h1` | **ORIG_B2** | 2 | 4 | 3 | 256 | 8 | 80.0 | ST-MoE | 1 | 4 | 4 | [`stmoe_b2_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/stmoe_b2_h1.yaml) | SUCCESS (0) |
| `stmoe_b4_h1` | **ORIG_B4** | 4 | 4 | 3 | 256 | 8 | 80.0 | ST-MoE | 1 | 4 | 4 | [`stmoe_b4_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/stmoe_b4_h1.yaml) | SUCCESS (0) |
| `stmoe_b8_h1` | **ORIG_B8** | 8 | 4 | 3 | 256 | 8 | 80.0 | ST-MoE | 1 | 4 | 4 | [`stmoe_b8_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/stmoe_b8_h1.yaml) | SUCCESS (0) |
| `stmoe_b1_h2` | **ORIG_B1** | 1 | 4 | 3 | 256 | 8 | 80.0 | ST-MoE | 2 | 4 | 4 | [`stmoe_b1_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/stmoe_b1_h2.yaml) | SUCCESS (0) |
| `stmoe_b2_h2` | **ORIG_B2** | 2 | 4 | 3 | 256 | 8 | 80.0 | ST-MoE | 2 | 4 | 4 | [`stmoe_b2_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/stmoe_b2_h2.yaml) | SUCCESS (0) |
| `stmoe_b4_h2` | **ORIG_B4** | 4 | 4 | 3 | 256 | 8 | 80.0 | ST-MoE | 2 | 4 | 4 | [`stmoe_b4_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/stmoe_b4_h2.yaml) | SUCCESS (0) |
| `stmoe_b8_h2` | **ORIG_B8** | 8 | 4 | 3 | 256 | 8 | 80.0 | ST-MoE | 2 | 4 | 4 | [`stmoe_b8_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/stmoe_b8_h2.yaml) | SUCCESS (0) |
| `cstp_b1_h1` | **ORIG_B1** | 1 | 4 | 3 | 256 | 8 | 80.0 | CSTP | 1 | 4 | 4 | [`cstp_b1_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/cstp_b1_h1.yaml) | SUCCESS (0) |
| `cstp_b2_h1` | **ORIG_B2** | 2 | 4 | 3 | 256 | 8 | 80.0 | CSTP | 1 | 4 | 4 | [`cstp_b2_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/cstp_b2_h1.yaml) | SUCCESS (0) |
| `cstp_b4_h1` | **ORIG_B4** | 4 | 4 | 3 | 256 | 8 | 80.0 | CSTP | 1 | 4 | 4 | [`cstp_b4_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/cstp_b4_h1.yaml) | SUCCESS (0) |
| `cstp_b8_h1` | **ORIG_B8** | 8 | 4 | 3 | 256 | 8 | 80.0 | CSTP | 1 | 4 | 4 | [`cstp_b8_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/cstp_b8_h1.yaml) | SUCCESS (0) |
| `cstp_b1_h2` | **ORIG_B1** | 1 | 4 | 3 | 256 | 8 | 80.0 | CSTP | 2 | 4 | 4 | [`cstp_b1_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/cstp_b1_h2.yaml) | SUCCESS (0) |
| `cstp_b2_h2` | **ORIG_B2** | 2 | 4 | 3 | 256 | 8 | 80.0 | CSTP | 2 | 4 | 4 | [`cstp_b2_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/cstp_b2_h2.yaml) | SUCCESS (0) |
| `cstp_b4_h2` | **ORIG_B4** | 4 | 4 | 3 | 256 | 8 | 80.0 | CSTP | 2 | 4 | 4 | [`cstp_b4_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/cstp_b4_h2.yaml) | SUCCESS (0) |
| `cstp_b8_h2` | **ORIG_B8** | 8 | 4 | 3 | 256 | 8 | 80.0 | CSTP | 2 | 4 | 4 | [`cstp_b8_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/cstp_b8_h2.yaml) | SUCCESS (0) |

---

## 3. Phase 2 Multi-Stage Scaling Campaign (26 Runs)

| Run ID | Workload | Batch ($B$) | Length ($T$) | Stages | Experts | Top-$K$ | HBM (GB) | Predictor | Horizon ($H$) | Budget ($C$) | Max Out | Config Link | Exit Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `phase2_baseline_b1` | **P2_B1** | 1 | 4 | 40 | 256 | 8 | 35.0 | Baseline | 0 | 0 | 4 | [`phase2_baseline_b1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/baseline/phase2_baseline_b1.yaml) | SUCCESS (0) |
| `phase2_baseline_b2` | **P2_B2** | 2 | 4 | 40 | 256 | 8 | 35.0 | Baseline | 0 | 0 | 4 | [`phase2_baseline_b2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/baseline/phase2_baseline_b2.yaml) | SUCCESS (0) |
| `phase2_baseline_b4` | **P2_B4** | 4 | 4 | 40 | 256 | 8 | 35.0 | Baseline | 0 | 0 | 4 | [`phase2_baseline_b4.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/baseline/phase2_baseline_b4.yaml) | SUCCESS (0) |
| `phase2_baseline_b8` | **P2_B8** | 8 | 4 | 40 | 256 | 8 | 35.0 | Baseline | 0 | 0 | 4 | [`phase2_baseline_b8.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/baseline/phase2_baseline_b8.yaml) | SUCCESS (0) |
| `phase2_stmoe_b1_h1` | **P2_B1** | 1 | 4 | 40 | 256 | 8 | 35.0 | ST-MoE | 1 | 4 | 4 | [`phase2_stmoe_b1_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/phase2_stmoe_b1_h1.yaml) | SUCCESS (0) |
| `phase2_stmoe_b2_h1` | **P2_B2** | 2 | 4 | 40 | 256 | 8 | 35.0 | ST-MoE | 1 | 4 | 4 | [`phase2_stmoe_b2_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/phase2_stmoe_b2_h1.yaml) | SUCCESS (0) |
| `phase2_stmoe_b4_h1` | **P2_B4** | 4 | 4 | 40 | 256 | 8 | 35.0 | ST-MoE | 1 | 4 | 4 | [`phase2_stmoe_b4_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/phase2_stmoe_b4_h1.yaml) | SUCCESS (0) |
| `phase2_stmoe_b8_h1` | **P2_B8** | 8 | 4 | 40 | 256 | 8 | 35.0 | ST-MoE | 1 | 4 | 4 | [`phase2_stmoe_b8_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/phase2_stmoe_b8_h1.yaml) | SUCCESS (0) |
| `phase2_stmoe_b1_h2` | **P2_B1** | 1 | 4 | 40 | 256 | 8 | 35.0 | ST-MoE | 2 | 4 | 4 | [`phase2_stmoe_b1_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/phase2_stmoe_b1_h2.yaml) | SUCCESS (0) |
| `phase2_stmoe_b2_h2` | **P2_B2** | 2 | 4 | 40 | 256 | 8 | 35.0 | ST-MoE | 2 | 4 | 4 | [`phase2_stmoe_b2_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/phase2_stmoe_b2_h2.yaml) | SUCCESS (0) |
| `phase2_stmoe_b4_h2` | **P2_B4** | 4 | 4 | 40 | 256 | 8 | 35.0 | ST-MoE | 2 | 4 | 4 | [`phase2_stmoe_b4_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/phase2_stmoe_b4_h2.yaml) | SUCCESS (0) |
| `phase2_stmoe_b8_h2` | **P2_B8** | 8 | 4 | 40 | 256 | 8 | 35.0 | ST-MoE | 2 | 4 | 4 | [`phase2_stmoe_b8_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/phase2_stmoe_b8_h2.yaml) | SUCCESS (0) |
| `phase2_cstp_b1_h1` | **P2_B1** | 1 | 4 | 40 | 256 | 8 | 35.0 | CSTP | 1 | 4 | 4 | [`phase2_cstp_b1_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/phase2_cstp_b1_h1.yaml) | SUCCESS (0) |
| `phase2_cstp_b2_h1` | **P2_B2** | 2 | 4 | 40 | 256 | 8 | 35.0 | CSTP | 1 | 4 | 4 | [`phase2_cstp_b2_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/phase2_cstp_b2_h1.yaml) | SUCCESS (0) |
| `phase2_cstp_b4_h1` | **P2_B4** | 4 | 4 | 40 | 256 | 8 | 35.0 | CSTP | 1 | 4 | 4 | [`phase2_cstp_b4_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/phase2_cstp_b4_h1.yaml) | SUCCESS (0) |
| `phase2_cstp_b8_h1` | **P2_B8** | 8 | 4 | 40 | 256 | 8 | 35.0 | CSTP | 1 | 4 | 4 | [`phase2_cstp_b8_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/phase2_cstp_b8_h1.yaml) | SUCCESS (0) |
| `phase2_cstp_b1_h2` | **P2_B1** | 1 | 4 | 40 | 256 | 8 | 35.0 | CSTP | 2 | 4 | 4 | [`phase2_cstp_b1_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/phase2_cstp_b1_h2.yaml) | SUCCESS (0) |
| `phase2_cstp_b2_h2` | **P2_B2** | 2 | 4 | 40 | 256 | 8 | 35.0 | CSTP | 2 | 4 | 4 | [`phase2_cstp_b2_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/phase2_cstp_b2_h2.yaml) | SUCCESS (0) |
| `phase2_cstp_b4_h2` | **P2_B4** | 4 | 4 | 40 | 256 | 8 | 35.0 | CSTP | 2 | 4 | 4 | [`phase2_cstp_b4_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/phase2_cstp_b4_h2.yaml) | SUCCESS (0) |
| `phase2_cstp_b8_h2` | **P2_B8** | 8 | 4 | 40 | 256 | 8 | 35.0 | CSTP | 2 | 4 | 4 | [`phase2_cstp_b8_h2.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/phase2_cstp_b8_h2.yaml) | SUCCESS (0) |
| `phase2_ref80g_baseline_b1` | **P2_B1** | 1 | 4 | 40 | 256 | 8 | 80.0 | Baseline | 0 | 0 | 4 | [`phase2_ref80g_baseline_b1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/baseline/phase2_ref80g_baseline_b1.yaml) | SUCCESS (0) |
| `phase2_ref80g_stmoe_b1_h1` | **P2_B1** | 1 | 4 | 40 | 256 | 8 | 80.0 | ST-MoE | 1 | 4 | 4 | [`phase2_ref80g_stmoe_b1_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/phase2_ref80g_stmoe_b1_h1.yaml) | SUCCESS (0) |
| `phase2_ref80g_cstp_b1_h1` | **P2_B1** | 1 | 4 | 40 | 256 | 8 | 80.0 | CSTP | 1 | 4 | 4 | [`phase2_ref80g_cstp_b1_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/phase2_ref80g_cstp_b1_h1.yaml) | SUCCESS (0) |
| `phase2_t8_baseline_b1` | **P2_B1** | 1 | 8 | 40 | 256 | 8 | 35.0 | Baseline | 0 | 0 | 4 | [`phase2_t8_baseline_b1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/baseline/phase2_t8_baseline_b1.yaml) | SUCCESS (0) |
| `phase2_t8_stmoe_b1_h1` | **P2_B1** | 1 | 8 | 40 | 256 | 8 | 35.0 | ST-MoE | 1 | 4 | 4 | [`phase2_t8_stmoe_b1_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/st_moe/phase2_t8_stmoe_b1_h1.yaml) | SUCCESS (0) |
| `phase2_t8_cstp_b1_h1` | **P2_B1** | 1 | 8 | 40 | 256 | 8 | 35.0 | CSTP | 1 | 4 | 4 | [`phase2_t8_cstp_b1_h1.yaml`](file:///home/mohman/ca_project/mohman/point_3/final_results/configs/cstp/phase2_t8_cstp_b1_h1.yaml) | SUCCESS (0) |

---

