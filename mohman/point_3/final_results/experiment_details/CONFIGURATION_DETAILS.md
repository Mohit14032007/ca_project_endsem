# Point 3 — Simulation and Architectural Configuration Details

**Repository:** [`/home/mohman/ca_project/LLMSimulator`](file:///home/mohman/ca_project/LLMSimulator)  
**Deliverable Path:** [`mohman/point_3/final_results/experiment_details/CONFIGURATION_DETAILS.md`](file:///home/mohman/ca_project/mohman/point_3/final_results/experiment_details/CONFIGURATION_DETAILS.md)  
**Status:** Complete Technical Specification  

---

## 1. DeepSeek-V3 Model Hyperparameters

The model simulated throughout the Point 3 evaluation is the DeepSeek-V3 Mixture-of-Experts architecture:

```
+----------------------------------------------------------------------------------------+
|                             DEEPSEEK-V3 ARCHITECTURAL PROFILE                          |
+------------------------------------+---------------------------------------------------+
| Parameter                          | Specification / Value                             |
+------------------------------------+---------------------------------------------------+
| Total Transformer Layers           | 61 Layers                                         |
| Initial Dense Layers               | 1 Layer (First-K Dense = 1)                       |
| MoE Layers (Stages)                | 40 Stages (Layers 1 to 40)                        |
| Remaining Non-MoE / Final Layers   | 20 Layers                                         |
| Routed Experts per Stage           | 256 Experts                                       |
| Shared Experts per Stage           | 1 Expert (Always resident, executed per token)   |
| Routing Policy                     | Top-K = 8 Routed Experts per Token                |
| Model Embedding Dimension (d_model)| 7,168                                             |
| Expert Intermediate Dim (d_expert) | 2,048                                             |
| Expert Projection Matrices         | Gate (W1), Up (W3), Down (W2)                     |
| Numeric Precision                  | FP16 / BF16 (2 bytes per parameter)               |
| Gate Matrix Footprint (W1)         | 7,168 x 2,048 x 2 = 29,360,128 bytes (~29.36 MB)  |
| Up Matrix Footprint (W3)           | 7,168 x 2,048 x 2 = 29,360,128 bytes (~29.36 MB)  |
| Down Matrix Footprint (W2)         | 2,048 x 7,168 x 2 = 29,360,128 bytes (~29.36 MB)  |
| Single Expert Footprint (W1+W3+W2) | 88,080,384 bytes (83.999 MB ~ 88.08 MB)          |
| Full Model Routed Parameter Size   | 40 stages x 256 x 88.08 MB = 901.94 GB            |
+------------------------------------+---------------------------------------------------+
```

---

## 2. Hardware and Interconnect Specifications

The execution engine models an NVIDIA A100-class accelerator connected to host DRAM over a dedicated off-chip interconnect channel:

```
+----------------------------------------------------------------------------------------+
|                             HARDWARE SIMULATION PARAMETERS                             |
+------------------------------------+---------------------------------------------------+
| Hardware Subsystem                 | Simulation Value / Baseline                       |
+------------------------------------+---------------------------------------------------+
| Accelerator Architecture           | NVIDIA A100-SXM4-80GB                             |
| FP16 Tensor Core Compute Capacity  | 312 TFLOPS                                        |
| Inter-Stage GEMM Computation Time  | 20.0 to 50.0 us (Scaled by dynamic batch M)      |
| Host DRAM Type                     | DDR5-4800 / DDR5-5600 Off-Chip Memory            |
| Host-to-Device Interconnect Bus    | Dedicated PCIe Gen4 x16 / DDR5 Channel            |
| Physical Interconnect Bandwidth    | 64.0 GB/s (0.064 bytes/ns)                        |
| Nominal Expert Transfer Latency    | 88,080,384 B / (64 x 10^9 B/s) = 1,376,256 ns     |
| Exact DDR5 Bus Service Time        | 1.376256 milliseconds per expert                 |
| Large-Workload HBM Expert Cache    | 35.0 GB (Accommodates ~397 expert slots)          |
| Original Campaign HBM Expert Cache | 80.0 GB (Accommodates ~908 expert slots)          |
| HBM Replacement Policy             | Least Recently Used (LRU) with Speculative Buffer |
| Off-Chip Bus Arbitration           | Priority-driven FIFO with Demand Preemption       |
+------------------------------------+---------------------------------------------------+
```

---

## 3. Predictor Configuration and Hyperparameters

Both predictors operate in **Online-Adaptive Mode from an un-profiled cold start** (`online_adaptation: true`). No offline weights or external precomputed profile tables are loaded.

### 3.1 ST-MoE Predictor Configuration
- **Algorithm Class:** Correlation-based Cross-Stage Transition Table (CCT).
- **Correlation Horizons:** $H = 1$ (Stage $s \to s+1$) and $H = 2$ (Stage $s \to s+2$).
- **Transition Registers:** 2-bit saturating counters clamped to $[0, 3]$.
- **History Table Bonus:** $+2.0$ added to the confidence score if the candidate was activated by the same request at the target stage on previous tokens.
- **Confidence Threshold ($\tau$):** $\tau = 1.0$ (Candidates with score $\ge 1.0$ admitted).
- **Candidate Budget ($C$):** $C = 4$ top-ranked candidates per stage dispatch.

### 3.2 CSTP Predictor Configuration
- **Algorithm Class:** Contextual Spatio-Temporal Predictor with Dirichlet Smoothing.
- **Prior Pseudo-Count ($\epsilon$):** $\epsilon = 0.01$ (Uniform marginal prior $P_0(v) = 1/256$).
- **Smoothing Coefficient ($\lambda$):** $\lambda = 1.0$ (Laplace spatial co-activation smoothing).
- **Spatio-Temporal Balance ($\alpha$):** $\alpha = 0.7$ (70% weight on spatial co-activation, 30% weight on temporal sequential transitions).
- **Probability Normalization:** Full Dirichlet posterior normalization over the 256-expert inventory.
- **Candidate Budget ($C$):** $C = 4$ highest-probability candidates admitted per stage dispatch.

---

## 4. Transfer Queue and Scheduling Parameters

- **`PrefetchCandidateQueue`:**
  - Filtering: Candidates already resident in HBM or actively in-flight on the bus are deduplicated and discarded.
  - Ranking: Candidates sorted descending by score. Ties broken deterministically by ascending expert ID.
  - Admission Ceiling: Exactly $\min(C, \text{valid\_candidates})$ transferred per stage.
- **`DDR5TransferEngine`:**
  - Demand Queue: High priority, FIFO.
  - Prefetch Queue: Low priority, FIFO.
  - Preemption Policy: When a demand transfer arrives while an active prefetch is transmitting, the prefetch is preempted immediately. Its partial byte progress is preserved, and the bus is ceded to the demand load.
  - Late Attachment: When a demand targets an expert currently transmitting as a prefetch, the transfer is promoted to demand priority in place, and the compute kernel waits only for the remaining bus transfer time.

