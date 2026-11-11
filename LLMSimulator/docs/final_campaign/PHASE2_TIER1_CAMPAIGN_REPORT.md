# Phase 2 Tier 1: Core Matrix Comparative Evaluation Report

## Executive Summary

This report presents the complete experimental results for the **Phase 2 Tier 1 Core Matrix** campaign.
The campaign evaluated **8 representative workloads** across **3 configurations**:
1. **Configuration B**: Demand-only Baseline with LRU eviction (pure reactive loading).
2. **Configuration FULL-S**: ST-MoE Predictor + Asynchronous Prefetching + PABI Speculative Quarantine (20%) + Belády-Style Stage Pinning (max 4).
3. **Configuration FULL-C**: CSTP Predictor + Asynchronous Prefetching + PABI Speculative Quarantine (20%) + Belády-Style Stage Pinning (max 4).

All 24 experimental runs executed under identical calibrated hardware parameters (DDR5-4800, 22.578 GB/s bandwidth, 3.90116 ms physical transfer latency per 88.08 MB expert).
All metrics were reconciled using the verified filter (`weight_tensor == "all_weights"`), strictly avoiding double-counting and separating physical DDR5 service time from exposed critical-path memory stall.

### Key Campaign Results

### Table 1: End-to-End Performance Summary Across All 24 Runs

| Workload | Batch | Tokens | Config | Device Time (ms) | Speedup vs Base | Exposed Stall (ms) | Stall Red. (%) | Expert Hit Rate (%) | Matrix Hit Rate (%) | Physical Transfers | Total Traffic (MB) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| A1 | 1 | 128 | B | 248919.0 | 1.000x | 245074.8 | 0.00% | 72.73% | 47.07% | 62,821 | 5,276,964.0 |
| A1 | 1 | 128 | FULL-S | 248395.0 | 1.002x | 244549.3 | 0.21% | 72.79% | 47.14% | 63,621 | 5,344,164.0 |
| A1 | 1 | 128 | FULL-C | 248084.0 | 1.003x | 244237.3 | 0.34% | 72.83% | 47.18% | 63,542 | 5,337,528.0 |
| A5 | 1 | 128 | B | 83414.1 | 1.000x | 81151.9 | 0.00% | 83.07% | 62.06% | 20,802 | 1,747,368.0 |
| A5 | 1 | 128 | FULL-S | 83465.8 | 0.999x | 81203.7 | -0.06% | 83.06% | 62.04% | 21,344 | 1,792,896.0 |
| A5 | 1 | 128 | FULL-C | 83512.5 | 0.999x | 81250.6 | -0.12% | 83.05% | 62.02% | 21,356 | 1,793,904.0 |
| A6 | 1 | 256 | B | 167630.0 | 1.000x | 163107.5 | 0.00% | 82.99% | 61.92% | 41,810 | 3,512,040.0 |
| A6 | 1 | 256 | FULL-S | 168071.0 | 0.997x | 163549.4 | -0.27% | 82.94% | 61.84% | 42,981 | 3,610,404.0 |
| A6 | 1 | 256 | FULL-C | 168193.0 | 0.997x | 163672.3 | -0.35% | 82.93% | 61.82% | 43,012 | 3,613,008.0 |
| A7 | 1 | 128 | B | 216773.0 | 1.000x | 213908.4 | 0.00% | 68.69% | 42.24% | 54,832 | 4,605,888.0 |
| A7 | 1 | 128 | FULL-S | 219257.0 | 0.989x | 216400.4 | -1.16% | 68.32% | 41.82% | 56,153 | 4,716,852.0 |
| A7 | 1 | 128 | FULL-C | 219814.0 | 0.986x | 216958.4 | -1.43% | 68.24% | 41.73% | 56,295 | 4,728,780.0 |
| B1 | 2 | 128 | B | 177845.0 | 1.000x | 174779.8 | 0.00% | 79.34% | 56.15% | 44,802 | 3,763,368.0 |
| B1 | 2 | 128 | FULL-S | 177148.0 | 1.004x | 174081.4 | 0.40% | 79.43% | 56.27% | 45,358 | 3,810,072.0 |
| B1 | 2 | 128 | FULL-C | 176950.0 | 1.005x | 173882.5 | 0.51% | 79.45% | 56.31% | 45,308 | 3,805,872.0 |
| B2 | 4 | 128 | B | 506117.0 | 1.000x | 501888.1 | 0.00% | 67.99% | 41.45% | 128,651 | 10,806,684.0 |
| B2 | 4 | 128 | FULL-S | 506147.0 | 1.000x | 501918.3 | -0.01% | 67.98% | 41.44% | 129,692 | 10,894,128.0 |
| B2 | 4 | 128 | FULL-C | 506397.0 | 0.999x | 502169.1 | -0.06% | 67.97% | 41.43% | 129,756 | 10,899,504.0 |
| B3 | 2 | 256 | B | 381591.0 | 1.000x | 375303.3 | 0.00% | 78.81% | 55.36% | 96,203 | 8,081,052.0 |
| B3 | 2 | 256 | FULL-S | 379739.0 | 1.005x | 373446.2 | 0.49% | 78.92% | 55.51% | 97,239 | 8,168,076.0 |
| B3 | 2 | 256 | FULL-C | 379093.0 | 1.007x | 372798.8 | 0.67% | 78.95% | 55.57% | 97,073 | 8,154,132.0 |
| C1 | 1 | 128 | B | 248921.0 | 1.000x | 245074.8 | 0.00% | 72.73% | 47.07% | 62,821 | 5,276,964.0 |
| C1 | 1 | 128 | FULL-S | 248396.0 | 1.002x | 244547.5 | 0.22% | 72.79% | 47.14% | 63,622 | 5,344,248.0 |
| C1 | 1 | 128 | FULL-C | 248085.0 | 1.003x | 244235.7 | 0.34% | 72.83% | 47.18% | 63,542 | 5,337,528.0 |

---

## Per-Workload Performance & Comparative Analysis

### Workload A1: MoE_Suite_T128_B1_FullDepth
- **Model**: DeepSeek-V3 unified architecture (78 layers, 75 MoE layers).
- **Parameters**: Batch Size = 1, Context Length = 128, Tokens = 128.
- **Routing Source**: `results/endsem_phase_e/final_baseline/traces/ms_task12_t128_l75.csv` (expected records: 76,801).

| Metric | Configuration B (Baseline) | Configuration FULL-S (ST-MoE) | Configuration FULL-C (CSTP) |
| :--- | :---: | :---: | :---: |
| Simulated Device Time (ms) | 248919.0 | 248395.0 | 248084.0 |
| Simulated Speedup vs Baseline | 1.000x | 1.002x | 1.003x |
| Per-Token Latency (ms) | 1944.68 | 1940.58 | 1938.15 |
| Throughput (tokens/s) | 0.51 | 0.52 | 0.52 |
| Exposed Memory Stall (ms) | 245074.8 | 244549.3 | 244237.3 |
| Memory Stall Reduction (%) | 0.00% | 0.21% | 0.34% |
| Physical DDR5 Service Time (ms) | 245074.8 | 248195.7 | 247887.5 |
| True Expert HBM Hit Rate (%) | 72.73% | 72.79% | 72.83% |
| Matrix-Level HBM Hit Rate (%) | 47.07% | 47.14% | 47.18% |
| Demand Transfers | 62,821 | 62,687 | 62,607 |
| Prefetch Transfers Completed | 0 | 934 | 935 |
| Total Physical Transfers | 62,821 | 63,621 | 63,542 |
| Total DDR5 Traffic (MB) | 5,276,964.0 | 5,344,164.0 | 5,337,528.0 |
| Prefetch Dispatched | 0 | 10,929 | 6,406 |
| Prefetch Useful Hits | 0 | 0 | 0 |
| Demand In-Flight Attachments | 0 | 6,292 | 3,671 |
| Speculative Quarantine Insertions | 0 | 12 | 12 |
| Speculative Promotions | 0 | 0 | 0 |
| Pin Intents Created | 0 | 35,563 | 33,584 |
| Pin Expirations | 0 | 35,563 | 33,584 |
| Predictor Precision (%) | N/A | 30.33% | 30.17% |
| Predictor Recall (%) | N/A | 15.16% | 15.08% |
| Predictor F1 Score | N/A | 0.2022 | 0.2011 |
| Host Runtime (seconds) | 193.8 | 309.6 | 252.5 |

### Workload A5: MoE_Anywhere_T128_B1_FullDepth
- **Model**: DeepSeek-V3 unified architecture (43 layers, 40 MoE layers).
- **Parameters**: Batch Size = 1, Context Length = 128, Tokens = 128.
- **Routing Source**: `results/endsem_phase_e/final_baseline/traces/ma_p60_t128_l40.csv` (expected records: 40,961).

| Metric | Configuration B (Baseline) | Configuration FULL-S (ST-MoE) | Configuration FULL-C (CSTP) |
| :--- | :---: | :---: | :---: |
| Simulated Device Time (ms) | 83414.1 | 83465.8 | 83512.5 |
| Simulated Speedup vs Baseline | 1.000x | 0.999x | 0.999x |
| Per-Token Latency (ms) | 651.67 | 652.08 | 652.44 |
| Throughput (tokens/s) | 1.53 | 1.53 | 1.53 |
| Exposed Memory Stall (ms) | 81151.9 | 81203.7 | 81250.6 |
| Memory Stall Reduction (%) | 0.00% | -0.06% | -0.12% |
| Physical DDR5 Service Time (ms) | 81151.9 | 83266.4 | 83313.2 |
| True Expert HBM Hit Rate (%) | 83.07% | 83.06% | 83.05% |
| Matrix-Level HBM Hit Rate (%) | 62.06% | 62.04% | 62.02% |
| Demand Transfers | 20,802 | 20,816 | 20,828 |
| Prefetch Transfers Completed | 0 | 528 | 528 |
| Total Physical Transfers | 20,802 | 21,344 | 21,356 |
| Total DDR5 Traffic (MB) | 1,747,368.0 | 1,792,896.0 | 1,793,904.0 |
| Prefetch Dispatched | 0 | 3,731 | 2,306 |
| Prefetch Useful Hits | 0 | 0 | 0 |
| Demand In-Flight Attachments | 0 | 1,666 | 985 |
| Speculative Quarantine Insertions | 0 | 20 | 20 |
| Speculative Promotions | 0 | 0 | 0 |
| Pin Intents Created | 0 | 17,573 | 16,832 |
| Pin Expirations | 0 | 17,573 | 16,831 |
| Predictor Precision (%) | N/A | 49.47% | 48.84% |
| Predictor Recall (%) | N/A | 24.74% | 24.42% |
| Predictor F1 Score | N/A | 0.3298 | 0.3256 |
| Host Runtime (seconds) | 96.9 | 126.1 | 112.4 |

### Workload A6: MoE_Anywhere_T256_B1_FullDepth
- **Model**: DeepSeek-V3 unified architecture (43 layers, 40 MoE layers).
- **Parameters**: Batch Size = 1, Context Length = 128, Tokens = 256.
- **Routing Source**: `results/endsem_phase_e/final_baseline/traces/ma_p60_t256_l40.csv` (expected records: 81,921).

| Metric | Configuration B (Baseline) | Configuration FULL-S (ST-MoE) | Configuration FULL-C (CSTP) |
| :--- | :---: | :---: | :---: |
| Simulated Device Time (ms) | 167630.0 | 168071.0 | 168193.0 |
| Simulated Speedup vs Baseline | 1.000x | 0.997x | 0.997x |
| Per-Token Latency (ms) | 654.81 | 656.53 | 657.01 |
| Throughput (tokens/s) | 1.53 | 1.52 | 1.52 |
| Exposed Memory Stall (ms) | 163107.5 | 163549.4 | 163672.3 |
| Memory Stall Reduction (%) | 0.00% | -0.27% | -0.35% |
| Physical DDR5 Service Time (ms) | 163107.5 | 167675.7 | 167796.7 |
| True Expert HBM Hit Rate (%) | 82.99% | 82.94% | 82.93% |
| Matrix-Level HBM Hit Rate (%) | 61.92% | 61.84% | 61.82% |
| Demand Transfers | 41,810 | 41,924 | 41,956 |
| Prefetch Transfers Completed | 0 | 1,057 | 1,056 |
| Total Physical Transfers | 41,810 | 42,981 | 43,012 |
| Total DDR5 Traffic (MB) | 3,512,040.0 | 3,610,404.0 | 3,613,008.0 |
| Prefetch Dispatched | 0 | 6,696 | 4,353 |
| Prefetch Useful Hits | 0 | 0 | 0 |
| Demand In-Flight Attachments | 0 | 3,829 | 2,287 |
| Speculative Quarantine Insertions | 0 | 20 | 20 |
| Speculative Promotions | 0 | 0 | 0 |
| Pin Intents Created | 0 | 35,105 | 33,479 |
| Pin Expirations | 0 | 35,105 | 33,478 |
| Predictor Precision (%) | N/A | 50.29% | 49.40% |
| Predictor Recall (%) | N/A | 25.14% | 24.70% |
| Predictor F1 Score | N/A | 0.3352 | 0.3293 |
| Host Runtime (seconds) | 206.3 | 269.6 | 238.4 |

### Workload A7: DeepSeek_Synthetic_T128_B1_FullDepth
- **Model**: DeepSeek-V3 unified architecture (60 layers, 57 MoE layers).
- **Parameters**: Batch Size = 1, Context Length = 128, Tokens = 128.
- **Routing Source**: `None` (expected records: 58,369).

| Metric | Configuration B (Baseline) | Configuration FULL-S (ST-MoE) | Configuration FULL-C (CSTP) |
| :--- | :---: | :---: | :---: |
| Simulated Device Time (ms) | 216773.0 | 219257.0 | 219814.0 |
| Simulated Speedup vs Baseline | 1.000x | 0.989x | 0.986x |
| Per-Token Latency (ms) | 1693.54 | 1712.95 | 1717.30 |
| Throughput (tokens/s) | 0.59 | 0.58 | 0.58 |
| Exposed Memory Stall (ms) | 213908.4 | 216400.4 | 216958.4 |
| Memory Stall Reduction (%) | 0.00% | -1.16% | -1.43% |
| Physical DDR5 Service Time (ms) | 213908.4 | 219061.8 | 219615.8 |
| True Expert HBM Hit Rate (%) | 68.69% | 68.32% | 68.24% |
| Matrix-Level HBM Hit Rate (%) | 42.24% | 41.82% | 41.73% |
| Demand Transfers | 54,832 | 55,471 | 55,615 |
| Prefetch Transfers Completed | 0 | 682 | 680 |
| Total Physical Transfers | 54,832 | 56,153 | 56,295 |
| Total DDR5 Traffic (MB) | 4,605,888.0 | 4,716,852.0 | 4,728,780.0 |
| Prefetch Dispatched | 0 | 15,187 | 11,793 |
| Prefetch Useful Hits | 0 | 0 | 0 |
| Demand In-Flight Attachments | 0 | 11,006 | 8,239 |
| Speculative Quarantine Insertions | 0 | 10 | 10 |
| Speculative Promotions | 0 | 0 | 0 |
| Pin Intents Created | 0 | 27,168 | 26,987 |
| Pin Expirations | 0 | 27,168 | 26,987 |
| Predictor Precision (%) | N/A | 3.13% | 4.05% |
| Predictor Recall (%) | N/A | 1.56% | 2.03% |
| Predictor F1 Score | N/A | 0.0209 | 0.0270 |
| Host Runtime (seconds) | 150.3 | 235.7 | 224.9 |

### Workload B1: MoE_Anywhere_T128_B2_FullDepth
- **Model**: DeepSeek-V3 unified architecture (43 layers, 40 MoE layers).
- **Parameters**: Batch Size = 2, Context Length = 128, Tokens = 128.
- **Routing Source**: `results/endsem_phase_e/final_baseline/traces/ma_b2_t128_l40.csv` (expected records: 81,921).

| Metric | Configuration B (Baseline) | Configuration FULL-S (ST-MoE) | Configuration FULL-C (CSTP) |
| :--- | :---: | :---: | :---: |
| Simulated Device Time (ms) | 177845.0 | 177148.0 | 176950.0 |
| Simulated Speedup vs Baseline | 1.000x | 1.004x | 1.005x |
| Per-Token Latency (ms) | 1389.41 | 1383.97 | 1382.42 |
| Throughput (tokens/s) | 1.44 | 1.45 | 1.45 |
| Exposed Memory Stall (ms) | 174779.8 | 174081.4 | 173882.5 |
| Memory Stall Reduction (%) | 0.00% | 0.40% | 0.51% |
| Physical DDR5 Service Time (ms) | 174779.8 | 176948.8 | 176753.7 |
| True Expert HBM Hit Rate (%) | 79.34% | 79.43% | 79.45% |
| Matrix-Level HBM Hit Rate (%) | 56.15% | 56.27% | 56.31% |
| Demand Transfers | 44,802 | 44,623 | 44,572 |
| Prefetch Transfers Completed | 0 | 735 | 736 |
| Total Physical Transfers | 44,802 | 45,358 | 45,308 |
| Total DDR5 Traffic (MB) | 3,763,368.0 | 3,810,072.0 | 3,805,872.0 |
| Prefetch Dispatched | 0 | 6,088 | 3,602 |
| Prefetch Useful Hits | 0 | 0 | 4 |
| Demand In-Flight Attachments | 0 | 3,511 | 2,000 |
| Speculative Quarantine Insertions | 0 | 14 | 14 |
| Speculative Promotions | 0 | 0 | 4 |
| Pin Intents Created | 0 | 34,095 | 31,549 |
| Pin Expirations | 0 | 28,354 | 24,411 |
| Predictor Precision (%) | N/A | 52.16% | 51.66% |
| Predictor Recall (%) | N/A | 26.08% | 25.83% |
| Predictor F1 Score | N/A | 0.3477 | 0.3444 |
| Host Runtime (seconds) | 105.9 | 151.9 | 131.2 |

### Workload B2: MoE_Anywhere_T128_B4_FullDepth
- **Model**: DeepSeek-V3 unified architecture (43 layers, 40 MoE layers).
- **Parameters**: Batch Size = 4, Context Length = 128, Tokens = 128.
- **Routing Source**: `results/endsem_phase_e/final_baseline/traces/ma_b4_t128_l40.csv` (expected records: 163,841).

| Metric | Configuration B (Baseline) | Configuration FULL-S (ST-MoE) | Configuration FULL-C (CSTP) |
| :--- | :---: | :---: | :---: |
| Simulated Device Time (ms) | 506117.0 | 506147.0 | 506397.0 |
| Simulated Speedup vs Baseline | 1.000x | 1.000x | 0.999x |
| Per-Token Latency (ms) | 3954.04 | 3954.27 | 3956.23 |
| Throughput (tokens/s) | 1.01 | 1.01 | 1.01 |
| Exposed Memory Stall (ms) | 501888.1 | 501918.3 | 502169.1 |
| Memory Stall Reduction (%) | 0.00% | -0.01% | -0.06% |
| Physical DDR5 Service Time (ms) | 501888.1 | 505949.2 | 506198.9 |
| True Expert HBM Hit Rate (%) | 67.99% | 67.98% | 67.97% |
| Matrix-Level HBM Hit Rate (%) | 41.45% | 41.44% | 41.43% |
| Demand Transfers | 128,651 | 128,661 | 128,724 |
| Prefetch Transfers Completed | 0 | 1,031 | 1,032 |
| Total Physical Transfers | 128,651 | 129,692 | 129,756 |
| Total DDR5 Traffic (MB) | 10,806,684.0 | 10,894,128.0 | 10,899,504.0 |
| Prefetch Dispatched | 0 | 12,771 | 8,073 |
| Prefetch Useful Hits | 0 | 0 | 2 |
| Demand In-Flight Attachments | 0 | 9,558 | 5,894 |
| Speculative Quarantine Insertions | 0 | 8 | 8 |
| Speculative Promotions | 0 | 0 | 2 |
| Pin Intents Created | 0 | 71,478 | 71,006 |
| Pin Expirations | 0 | 53,332 | 46,166 |
| Predictor Precision (%) | N/A | 54.01% | 54.20% |
| Predictor Recall (%) | N/A | 27.01% | 27.10% |
| Predictor F1 Score | N/A | 0.3601 | 0.3614 |
| Host Runtime (seconds) | 113.7 | 238.8 | 171.1 |

### Workload B3: MoE_Anywhere_T256_B2_FullDepth
- **Model**: DeepSeek-V3 unified architecture (43 layers, 40 MoE layers).
- **Parameters**: Batch Size = 2, Context Length = 128, Tokens = 256.
- **Routing Source**: `results/endsem_phase_e/final_baseline/traces/ma_b2_t256_l40.csv` (expected records: 163,841).

| Metric | Configuration B (Baseline) | Configuration FULL-S (ST-MoE) | Configuration FULL-C (CSTP) |
| :--- | :---: | :---: | :---: |
| Simulated Device Time (ms) | 381591.0 | 379739.0 | 379093.0 |
| Simulated Speedup vs Baseline | 1.000x | 1.005x | 1.007x |
| Per-Token Latency (ms) | 1490.59 | 1483.36 | 1480.83 |
| Throughput (tokens/s) | 1.34 | 1.35 | 1.35 |
| Exposed Memory Stall (ms) | 375303.3 | 373446.2 | 372798.8 |
| Memory Stall Reduction (%) | 0.00% | 0.49% | 0.67% |
| Physical DDR5 Service Time (ms) | 375303.3 | 379344.9 | 378697.3 |
| True Expert HBM Hit Rate (%) | 78.81% | 78.92% | 78.95% |
| Matrix-Level HBM Hit Rate (%) | 55.36% | 55.51% | 55.57% |
| Demand Transfers | 96,203 | 95,727 | 95,561 |
| Prefetch Transfers Completed | 0 | 1,512 | 1,512 |
| Total Physical Transfers | 96,203 | 97,239 | 97,073 |
| Total DDR5 Traffic (MB) | 8,081,052.0 | 8,168,076.0 | 8,154,132.0 |
| Prefetch Dispatched | 0 | 12,064 | 7,358 |
| Prefetch Useful Hits | 0 | 0 | 4 |
| Demand In-Flight Attachments | 0 | 8,571 | 4,888 |
| Speculative Quarantine Insertions | 0 | 14 | 14 |
| Speculative Promotions | 0 | 0 | 4 |
| Pin Intents Created | 0 | 68,745 | 62,067 |
| Pin Expirations | 0 | 60,085 | 50,572 |
| Predictor Precision (%) | N/A | 51.29% | 51.18% |
| Predictor Recall (%) | N/A | 25.64% | 25.59% |
| Predictor F1 Score | N/A | 0.3419 | 0.3412 |
| Host Runtime (seconds) | 226.2 | 352.5 | 293.9 |

### Workload C1: MoE_Suite_Lin512_T128_B1_FullDepth
- **Model**: DeepSeek-V3 unified architecture (78 layers, 75 MoE layers).
- **Parameters**: Batch Size = 1, Context Length = 512, Tokens = 128.
- **Routing Source**: `results/endsem_phase_e/final_baseline/traces/ms_task12_t128_l75.csv` (expected records: 76,801).

| Metric | Configuration B (Baseline) | Configuration FULL-S (ST-MoE) | Configuration FULL-C (CSTP) |
| :--- | :---: | :---: | :---: |
| Simulated Device Time (ms) | 248921.0 | 248396.0 | 248085.0 |
| Simulated Speedup vs Baseline | 1.000x | 1.002x | 1.003x |
| Per-Token Latency (ms) | 1944.70 | 1940.59 | 1938.16 |
| Throughput (tokens/s) | 0.51 | 0.52 | 0.52 |
| Exposed Memory Stall (ms) | 245074.8 | 244547.5 | 244235.7 |
| Memory Stall Reduction (%) | 0.00% | 0.22% | 0.34% |
| Physical DDR5 Service Time (ms) | 245074.8 | 248199.6 | 247887.5 |
| True Expert HBM Hit Rate (%) | 72.73% | 72.79% | 72.83% |
| Matrix-Level HBM Hit Rate (%) | 47.07% | 47.14% | 47.18% |
| Demand Transfers | 62,821 | 62,687 | 62,607 |
| Prefetch Transfers Completed | 0 | 935 | 935 |
| Total Physical Transfers | 62,821 | 63,622 | 63,542 |
| Total DDR5 Traffic (MB) | 5,276,964.0 | 5,344,248.0 | 5,337,528.0 |
| Prefetch Dispatched | 0 | 10,929 | 6,406 |
| Prefetch Useful Hits | 0 | 0 | 0 |
| Demand In-Flight Attachments | 0 | 6,292 | 3,671 |
| Speculative Quarantine Insertions | 0 | 12 | 12 |
| Speculative Promotions | 0 | 0 | 0 |
| Pin Intents Created | 0 | 35,563 | 33,584 |
| Pin Expirations | 0 | 35,563 | 33,584 |
| Predictor Precision (%) | N/A | 30.33% | 30.17% |
| Predictor Recall (%) | N/A | 15.16% | 15.08% |
| Predictor F1 Score | N/A | 0.2022 | 0.2011 |
| Host Runtime (seconds) | 211.4 | 310.2 | 259.4 |

---

## FULL-S (ST-MoE) vs FULL-C (CSTP) Head-to-Head Comparison

| Workload | ST-MoE F1 | CSTP F1 | Winner (F1) | ST-MoE Speedup | CSTP Speedup | Winner (Speedup) | ST-MoE Useful Hits | CSTP Useful Hits |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| A1 | 0.2022 | 0.2011 | ST-MoE | 1.002x | 1.003x | CSTP | 0 | 0 |
| A5 | 0.3298 | 0.3256 | ST-MoE | 0.999x | 0.999x | ST-MoE | 0 | 0 |
| A6 | 0.3352 | 0.3293 | ST-MoE | 0.997x | 0.997x | ST-MoE | 0 | 0 |
| A7 | 0.0209 | 0.0270 | CSTP | 0.989x | 0.986x | ST-MoE | 0 | 0 |
| B1 | 0.3477 | 0.3444 | ST-MoE | 1.004x | 1.005x | CSTP | 0 | 4 |
| B2 | 0.3601 | 0.3614 | CSTP | 1.000x | 0.999x | ST-MoE | 0 | 2 |
| B3 | 0.3419 | 0.3412 | ST-MoE | 1.005x | 1.007x | CSTP | 0 | 4 |
| C1 | 0.2022 | 0.2011 | ST-MoE | 1.002x | 1.003x | CSTP | 0 | 0 |

---

## DDR5 Service Time vs Critical-Path Stall Decoupling

A key finding of this campaign is the strict separation between **Physical DDR5 Service Time** (the total time the DDR5 physical bus spent transmitting experts) and **Exposed Memory Stall** (the time the compute engine spent halted waiting on expert weights).

| Workload | Config | Physical Transfers | DDR5 Service Time (ms) | Exposed Stall (ms) | Overlap Ratio (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| A1 | B | 62,821 | 245074.8 | 245074.8 | 0.00% |
| A1 | FULL-S | 63,621 | 248195.7 | 244549.3 | 1.47% |
| A1 | FULL-C | 63,542 | 247887.5 | 244237.3 | 1.47% |
| A5 | B | 20,802 | 81151.9 | 81151.9 | 0.00% |
| A5 | FULL-S | 21,344 | 83266.4 | 81203.7 | 2.48% |
| A5 | FULL-C | 21,356 | 83313.2 | 81250.6 | 2.48% |
| A6 | B | 41,810 | 163107.5 | 163107.5 | 0.00% |
| A6 | FULL-S | 42,981 | 167675.7 | 163549.4 | 2.46% |
| A6 | FULL-C | 43,012 | 167796.7 | 163672.3 | 2.46% |
| A7 | B | 54,832 | 213908.4 | 213908.4 | 0.00% |
| A7 | FULL-S | 56,153 | 219061.8 | 216400.4 | 1.21% |
| A7 | FULL-C | 56,295 | 219615.8 | 216958.4 | 1.21% |
| B1 | B | 44,802 | 174779.8 | 174779.8 | 0.00% |
| B1 | FULL-S | 45,358 | 176948.8 | 174081.4 | 1.62% |
| B1 | FULL-C | 45,308 | 176753.7 | 173882.5 | 1.62% |
| B2 | B | 128,651 | 501888.1 | 501888.1 | 0.00% |
| B2 | FULL-S | 129,692 | 505949.2 | 501918.3 | 0.80% |
| B2 | FULL-C | 129,756 | 506198.9 | 502169.1 | 0.80% |
| B3 | B | 96,203 | 375303.3 | 375303.3 | 0.00% |
| B3 | FULL-S | 97,239 | 379344.9 | 373446.2 | 1.55% |
| B3 | FULL-C | 97,073 | 378697.3 | 372798.8 | 1.56% |
| C1 | B | 62,821 | 245074.8 | 245074.8 | 0.00% |
| C1 | FULL-S | 63,622 | 248199.6 | 244547.5 | 1.47% |
| C1 | FULL-C | 63,542 | 247887.5 | 244235.7 | 1.47% |

---

## Campaign Integrity & Provenance Verification

- **Authoritative Git Anchor**: `d5bf61ff0c066c06f72173189f9924329f613a88`
- **Calibrated DDR5 Bandwidth**: `22.578 GB/s` (3.90116 ms per expert)
- **Total Experiments Attempted**: 24
- **Total Experiments Completed**: 24
- **Total Experiments Validated**: 24
- **Failed Experiments**: 0
- **Total Host Runtime**: 4982.63 seconds (1.38 hours)

### Artifact Checksum Table

| File | SHA256 Checksum |
| :--- | :--- |
| `tier1_combined_results.csv` | `ddd03dd30b65ecddadb29988eabcf38b46d95d91895e8b071bcccd40b879f394` |
| `tier1_prediction_metrics.csv` | `20d926fb0ea756d1bcd9498bce67711ac75116b55402a5dccc142a8fd2922d3d` |
| `tier1_prefetch_lifecycle.csv` | `c29d71243bdda1fe7d09bcb1df8e40741fee76092e281e37221eabea8a740663` |
| `tier1_hbm_metrics.csv` | `a05008a1bc1219a23fc8a7fb20d99bbf32abab99d2ace2ba29fd7db2d8bb0d6c` |
| `tier1_ddr5_metrics.csv` | `fe2af9b096df8872e778e3e236f906a1f8a1a9e4c2083576124005313f6d6293` |
| `tier1_performance_metrics.csv` | `22a4c1685074136fbf5321437a6a88f03af61f991fbf1d290ecef0cc5ea8ea0d` |

