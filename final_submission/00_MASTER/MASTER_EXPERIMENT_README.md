# Master Experiment Reference Manual
**MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference**

---

## 1. Hardware & Simulation Environment

### 1.1 Simulated Hardware Specifications
- **GPU Architecture:** NVIDIA A100 Tensor Core GPU equivalent
- **On-Chip Memory (HBM):** 80 GB HBM2e (2.0 TB/s theoretical bandwidth, modeled via cycle-accurate Ramulator2 HBM3/HBM2e timing configuration `dram_config_HBM3_80GB.yaml`)
- **Off-Chip Parameter Memory (DRAM):** 2-channel calibrated DDR5-4800 interface
  - **Calibrated Bus Bandwidth:** `22.578 GB/s` (empirical transfer rate verified in `calibrate_ddr5_transfers`)
  - **Expert Parameter Size:** `88,080,384 bytes` (84 MiB binary / 88.080384 MB decimal, FP16 precision)
  - **Isolated Single-Expert Transfer Latency:**
    $$\tau_{\text{xfer}} = \frac{88,080,384\text{ bytes}}{22.578 \times 10^9\text{ bytes/sec}} = 3.90116\text{ ms} = 3,901,160\text{ ns}$$
- **Model Topology (Workload A1, GLM-5.2):**
  - Total Layers: 78 (3 dense self-attention layers + 75 MoE layers)
  - Total Experts per MoE Layer: 256 experts
  - Top-K Routing: 8 active experts per token per MoE stage
  - Single-Token MoE Lookups: $75 \times 8 = 600$ lookups/token
  - 4-Token Workload A1 Total Lookups: $600 \times 4 = 2,400$ expert activations (7,200 total weight tensor evaluations across gates/projections)

### 1.2 Host Benchmark Environment
- **Host CPU:** 11th Gen Intel(R) Core(TM) i3-1115G4 @ 3.00GHz (2 physical cores / 4 threads)
- **Host RAM:** 8.0 GB Physical RAM
- **OS:** Linux 6.6.137+ x86_64 Ubuntu-compatible
- **Toolchain:** GCC 13.3.0, CMake 3.28.3, Python 3.12.3

---

## 2. Historical Baseline Experiments (19 Runs)

Location: `LLMSimulator/results/endsem_phase_e/final_baseline/runs/`

| Exp ID | Workload | Dataset / Trace | Model Architecture | Batch | Context ($L_{\text{in}}$) | Tokens ($T$) | Sim Time (ms) | HBM Hit Rate | Exposed Stall (ms) | Output Directory |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A1** | MoE-Suite Task 12 | `ms_task12_t128_l75.csv` | GLM-5.2 (75 MoE, 256 exp) | 1 | 128 | 128 | 248,926.83 | 72.73% | 245,074.77 | `runs/A1/` |
| **A2** | MoE-Suite Task 12 | `ms_task12_t512_l75.csv` | GLM-5.2 (75 MoE, 256 exp) | 1 | 128 | 512 | 1,033,760.00 | 71.67% | 1,626,543.58 | `runs/A2/` |
| **A3** | MoE-Suite Task 12 | `ms_task12_t1024_l75.csv` | GLM-5.2 (75 MoE, 256 exp) | 1 | 128 | 1024 | 2,097,890.00 | 71.25% | 3,274,747.27 | `runs/A3/` |
| **A4** | MoE-Suite Task 12 | `ms_task12_t2048_l75.csv` | GLM-5.2 (75 MoE, 256 exp) | 1 | 128 | 2048 | 4,222,330.00 | 71.06% | 6,568,380.81 | `runs/A4/` |
| **A5** | MoE-Anywhere P60 | `ma_p60_t128_l40.csv` | Sage-A1.5 (40 MoE, 256 exp) | 1 | 128 | 128 | 83,416.90 | 83.07% | 177,418.18 | `runs/A5/` |
| **A6** | MoE-Anywhere P60 | `ma_p60_t256_l40.csv` | Sage-A1.5 (40 MoE, 256 exp) | 1 | 128 | 256 | 167,636.00 | 82.99% | 355,409.35 | `runs/A6/` |
| **A7** | DeepSeek Synthetic | Random Seed 777 | DeepSeek-V3 (57 MoE, 256 exp) | 1 | 128 | 128 | 216,780.00 | 68.69% | 322,882.02 | `runs/A7/` |
| **A8** | DeepSeek Synthetic | Random Seed 777 | DeepSeek-V3 (57 MoE, 256 exp) | 1 | 128 | 512 | 866,336.00 | 68.72% | 1,291,528.08 | `runs/A8/` |
| **B1** | MoE-Anywhere P60 | Multi-user batch 2 | Sage-A1.5 (40 MoE, 256 exp) | 2 | 128 | 128 | 166,833.80 | 82.95% | 354,836.36 | `runs/B1/` |
| **B2** | MoE-Anywhere P60 | Multi-user batch 4 | Sage-A1.5 (40 MoE, 256 exp) | 4 | 128 | 128 | 333,667.60 | 82.80% | 709,672.72 | `runs/B2/` |
| **B3** | MoE-Anywhere P60 | Multi-user batch 2 | Sage-A1.5 (40 MoE, 256 exp) | 2 | 128 | 256 | 335,272.00 | 82.90% | 710,818.70 | `runs/B3/` |
| **B4** | MoE-Anywhere P60 | Multi-user batch 4 | Sage-A1.5 (40 MoE, 256 exp) | 4 | 128 | 256 | 1,092,650.00 | 67.29% | 1,531,237.30 | `runs/B4/` |
| **C1** | MoE-Suite Task 12 | `ms_task12_t128_l75.csv` | GLM-5.2 (75 MoE, 256 exp) | 1 | 512 | 128 | 248,929.64 | 72.73% | 245,074.77 | `runs/C1/` |
| **C2** | MoE-Suite Task 12 | `ms_task12_t128_l75.csv` | GLM-5.2 (75 MoE, 256 exp) | 1 | 1024 | 128 | 248,933.62 | 72.73% | 245,074.77 | `runs/C2/` |
| **C3** | MoE-Suite Task 12 | `ms_task12_t128_l75.csv` | GLM-5.2 (75 MoE, 256 exp) | 1 | 2048 | 128 | 248,941.61 | 72.73% | 245,074.77 | `runs/C3/` |
| **DS_FULL_PASS1** | DeepSeek Cold | Cold start tick | DeepSeek-V3 Full (57 MoE) | 1 | 2048 | 1 | 1,801.05 | 0.00% | 1,778.91 | `pilot_1step/` |
| **DS_FULL_PASS2** | DeepSeek Warm | Warm replay tick | DeepSeek-V3 Full (57 MoE) | 1 | 2048 | 1 | 22.14 | 100.00% | 0.00 | `pilot_1step/` |
| **DS_PILOT_1STEP** | DeepSeek Pilot | Cycle-accurate tick | DeepSeek-V3 Full (57 MoE) | 1 | 2048 | 1 | 1,801.05 | 0.00% | 1,778.91 | `pilot_1step/` |
| **DS_FULL_5STEP** | DeepSeek 5-Step | Autoregressive decode | DeepSeek-V3 Full (57 MoE) | 1 | 2048 | 5 | 8,713.53 | 3.26% | 8,601.93 | `full_run_5step/` |

---

## 3. Component Ablation Experiments (9 Configurations)

Location: `results/final_comparative_campaign/ablation/` and `results/final_comparative_campaign/tier1/`

| Config ID | Description | Predictor | Prefetch | Eviction Policy | PABI Quarantine | Belady Pinning | Sim Time (ms) | Exposed Stall (ms) | Stall Delta (ms) | Speedup (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B** | Demand-only baseline | None | Off | LRU | No | No | 248,918.55 | 245,074.77 | 0.00 | Baseline |
| **P-S** | ST-MoE Prediction Only | ST-MoE | Off | LRU | No | No | 248,918.55 | 245,074.77 | 0.00 | 0.00% |
| **P-C** | CSTP Prediction Only | CSTP | Off | LRU | No | No | 248,918.55 | 245,074.77 | 0.00 | 0.00% |
| **PF-S** | ST-MoE Prefetch (LRU) | ST-MoE | On | LRU | No | No | 248,306.29 | 244,460.74 | +614.03 | 0.25% |
| **PF-C** | CSTP Prefetch (LRU) | CSTP | On | LRU | No | No | 247,975.39 | 244,128.85 | +945.92 | 0.38% |
| **QUAR-S** | ST-MoE + PABI Quarantine | ST-MoE | On | PABI | Yes (20% cap) | No | 248,394.62 | 244,549.32 | +525.45 | 0.21% |
| **QUAR-C** | CSTP + PABI Quarantine | CSTP | On | PABI | Yes (20% cap) | No | 248,083.52 | 244,237.30 | +837.47 | 0.34% |
| **FULL-S** | ST-MoE + PABI + Pinning | ST-MoE | On | PABI | Yes (20% cap) | Yes (max 4) | 248,394.62 | 244,549.32 | +525.45 | 0.21% |
| **FULL-C** | CSTP + PABI + Pinning | CSTP | On | PABI | Yes (20% cap) | Yes (max 4) | 248,083.52 | 244,237.30 | +837.47 | 0.34% |

**Ablation Attribution Insights:**
1. **Prediction Overhead ($P - B = 0.00\text{ ms}$):** The prediction engine adds $< 0.16\%$ host wall-clock time overhead and $0.00\text{ ms}$ simulated device stall overhead.
2. **Predictor Quality ($PF\text{-}C \text{ vs } PF\text{-}S$):** CSTP saves an additional $331.89\text{ ms}$ of memory stall over ST-MoE due to its multi-horizon spatio-temporal transition probabilities.
3. **PABI Quarantine ($QUAR \text{ vs } PF$):** PABI slightly increases stall by $88.58\text{ ms}$ because speculative experts are confined to a 20% partition rather than the entire 80 GB cache, but in return it eliminates demand-verified cache pollution.
4. **Pinning ($FULL \text{ vs } QUAR = 0.00\text{ ms}$):** In Workload A1 with an 80 GB HBM capacity, 35,563 pin intents were created and cleanly expired, but because speculative evictions were rare, pinning prevented no evictions that would have become hits.

---

## 4. Phase 2F Synthetic Compute-Window Campaign (15 Runs)

Location: `results/final_comparative_campaign/phase2f/`  
Workload: Workload A1 (4 decoding tokens, 75 MoE stages, batch size 1)  
Parameter: **Synthetic On-Chip Execution-Window Multiplier ($M \in \{1.0, 4.0, 8.0, 12.0, 16.0\}$)**

| Run ID | Multiplier ($M$) | Window ($W_{H=2}$) | Predictor | Sim Time (ms) | Exposed Stall (ms) | Stall Reduction (ms) | Timely Hits | Late Attachments | Speculative Traffic (MiB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B-M1** | $1.0\times$ | $0.366\text{ ms}$ | None | 8003.70 | 7884.24 | Baseline | 0 | 0 | 0.0 |
| **FWD-S-H2-M1** | $1.0\times$ | $0.366\text{ ms}$ | ST-MoE | 8003.70 | 7884.24 | 0.00 | 0 | 59 | 33,180.0 |
| **FWD-C-H2-M1** | $1.0\times$ | $0.366\text{ ms}$ | CSTP | 8003.70 | 7884.24 | 0.00 | 0 | 68 | 38,388.0 |
| **B-M4** | $4.0\times$ | $1.465\text{ ms}$ | None | 8323.86 | 7884.24 | Baseline | 0 | 0 | 0.0 |
| **FWD-S-H2-M4** | $4.0\times$ | $1.465\text{ ms}$ | ST-MoE | 8320.01 | 7880.34 | **+3.90** | **1** | 52 | 33,180.0 |
| **FWD-C-H2-M4** | $4.0\times$ | $1.465\text{ ms}$ | CSTP | 8320.01 | 7880.34 | **+3.90** | **1** | 61 | 38,388.0 |
| **B-M8** | $8.0\times$ | $2.931\text{ ms}$ | None | 8755.98 | 7884.24 | Baseline | 0 | 0 | 0.0 |
| **FWD-S-H2-M8** | $8.0\times$ | $2.931\text{ ms}$ | ST-MoE | 8752.17 | 7880.34 | **+3.90** | **1** | 48 | 33,180.0 |
| **FWD-C-H2-M8** | $8.0\times$ | $2.931\text{ ms}$ | CSTP | 8752.17 | 7880.34 | **+3.90** | **1** | 57 | 38,388.0 |
| **B-M12** | $12.0\times$ | $4.396\text{ ms}$ | None | 9194.20 | 7884.24 | Baseline | 0 | 0 | 0.0 |
| **FWD-S-H2-M12** | $12.0\times$ | $4.396\text{ ms}$ | ST-MoE | 9186.64 | 7876.44 | **+7.80** | **2** | 43 | 33,180.0 |
| **FWD-C-H2-M12** | $12.0\times$ | $4.396\text{ ms}$ | CSTP | 9186.64 | 7876.44 | **+7.80** | **2** | 52 | 38,388.0 |
| **B-M16** | $16.0\times$ | $5.861\text{ ms}$ | None | 9638.50 | 7884.24 | Baseline | 0 | 0 | 0.0 |
| **FWD-S-H2-M16** | $16.0\times$ | $5.861\text{ ms}$ | ST-MoE | 9625.96 | 7871.39 | **+12.86** | **2** | 31 | 33,180.0 |
| **FWD-C-H2-M16** | $16.0\times$ | $5.861\text{ ms}$ | CSTP | 9631.02 | 7876.44 | **+7.80** | **2** | 48 | 38,388.0 |

---

## 5. Candidate-Budget Sensitivity Campaign (8 Runs)

Location: `results/final_comparative_campaign/sensitivity_budget/`  
Workload: Workload A1, Forward Lookahead $H=2$

| Run ID | Multiplier ($M$) | Predictor | Budget ($B$) | Avg Queue Wait (ms) | Speculative Traffic (MiB) | Timely Hits | Late Attachments | Stall Reduction (ms) | Empirical Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **FWD-S-H2-B1-M1** | $1.0\times$ | ST-MoE | **1** | 3,027.18 | **7,392.0** | 0 | 10 | 0.00 | $77.7\%$ traffic reduction |
| **FWD-S-H2-B2-M1** | $1.0\times$ | ST-MoE | **2** | 2,785.98 | 15,456.0 | **1** | 23 | **+3.90** | **Timely hit at nominal $M=1$** |
| **FWD-S-H2-B4-M1** | $1.0\times$ | ST-MoE | **4** | 2,411.11 | 33,180.0 | 0 | 59 | 0.00 | FIFO serialization |
| **FWD-C-H2-B1-M1** | $1.0\times$ | CSTP | **1** | 3,261.48 | **8,064.0** | 0 | 7 | 0.00 | $79.0\%$ traffic reduction |
| **FWD-C-H2-B2-M1** | $1.0\times$ | CSTP | **2** | 2,513.25 | 17,976.0 | **1** | 29 | **+3.90** | **Timely hit at nominal $M=1$** |
| **FWD-C-H2-B4-M1** | $1.0\times$ | CSTP | **4** | 2,197.47 | 38,388.0 | 0 | 68 | 0.00 | High traffic, zero hit |
| **FWD-S-H2-B1-M12** | $12.0\times$ | ST-MoE | **1** | **0.00** | **7,392.0** | **2** | **0** | **+7.80** | **Zero queue delay** |
| **FWD-S-H2-B2-M12** | $12.0\times$ | ST-MoE | **2** | 714.87 | 15,456.0 | **2** | 5 | **+7.80** | Low queue delay |
| **FWD-C-H2-B1-M12** | $12.0\times$ | CSTP | **1** | **0.00** | **8,064.0** | **2** | **0** | **+7.80** | **Zero queue delay** |
| **FWD-C-H2-B2-M12** | $12.0\times$ | CSTP | **2** | 624.41 | 17,976.0 | **2** | 16 | **+18.82** | **Peak stall reduction** |

---

## 6. Real Recorded Routing Traces Provenance

| Trace Name | Model | MoE Stages | Experts/Stage | Active Top-K | Sequence Length ($T$) | Source Repository | Checksum (SHA-256) |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| `ms_task12_t128_l75.csv` | GLM-5.2-MoE | 75 | 256 | 8 | 128 | MoE-Suite (Task 12) | `45da8c85ad05096ba181c5be4c0506e7464a9386377e80277df60662df94d658` |
| `ma_p60_t128_l40.csv` | Sage-A1.5 | 40 | 256 | 8 | 128 | MoE-Anywhere (Prompt 60) | `e2a4f40f28383a54b38d7cb1466a935392cf99a896d859e99a800e84b8109d94` |
| `synthetic_random` | DeepSeek-V3 | 57 | 256 | 8 | 128 | MT19937 Seed 777 Control | Pseudo-random in-memory generation |
