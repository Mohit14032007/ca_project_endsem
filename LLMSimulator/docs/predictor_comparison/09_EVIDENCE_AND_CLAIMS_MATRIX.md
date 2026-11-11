# Evidence and Scientific Claims Matrix

**Purpose**: Defines strictly what can and cannot be claimed at the current stage of the project based on verifiable source code and empirical measurements.

---

## 1. Evidence and Claims Evaluation Table

| Research Question | Evidence Currently Available | Can We Claim It? | Status Label | Missing Work Required |
| :--- | :--- | :---: | :---: | :--- |
| **1. How does ST-MoE score candidates?** | Implemented in `st_moe_paper.py`; unit-tested with hand calculation in `test_predictor_hand_example.py`. | **YES** | **Verified from source code** | None. Algorithm 2 and Equation 1 are completely verified. |
| **2. How does CSTP score candidates?** | Implemented in `cstp_predictor.py`; unit-tested in `test_cstp.py`. Continuous composite probability $\alpha S_{\text{sp}} + (1-\alpha) S_{\text{temp}}$. | **YES** | **Verified from source code** | None. Formulations and code are verified. |
| **3. Are CSTP spatial statistics offline-trained?** | `fit_profile()` runs on the training split prior to evaluation; tensors are frozen during inference (`online_adaptation=False`). | **YES** | **Verified from source code** | Phase C dataset scaling across diverse traces. |
| **4. Is CSTP temporal history online?** | `history_table` updates dynamically after each token's routing is revealed. | **YES** | **Verified from source code** | None. |
| **5. Is ST-MoE CCT pre-trained?** | `build_cct_from_profile()` builds CCT from training traces; counters adapt online during evaluation. | **YES** | **Verified from source code** | None. |
| **6. Is ST-MoE HT online?** | `history_table` updates dynamically per token per request during inference. | **YES** | **Verified from source code** | None. |
| **7. Is evaluation causal?** | `predict()` executes before `update()` in `evaluation.py:226,247`; tested by label-mutation tests. | **YES** | **Verified from source code** | None. Causality is strictly enforced. |
| **8. Is train/test separation valid?** | `split_by_request()` isolates sequences by request ID; single-request traces trigger temporal split with limitation warning. | **YES** | **Verified from source code** | Evaluate on multi-request traces in Phase C. |
| **9. What is prediction accuracy?** | Phase 1 benchmark measured across synthetic and PRNG traces; real trace evaluation scheduled for Phase C. | **PARTIAL** | **Supported but incomplete** | Run Phase C evaluation on real dataset traces (`Predicting-Experts-for-MOE-Suite` and `moe-anywhere`). |
| **10. What is prediction overhead?** | Measured via CPU benchmark (`scripts/benchmark_predictor_latency.py`): ST-MoE $\approx 62.6\ \mu\text{s}$, CSTP $\approx 197.5\ \mu\text{s}$. | **YES** | **Verified from experiment output** | Profile in C++ simulation loop if EPU is ported to C++. |
| **11. Is transfer asynchronous?** | C++ audit reveals `DRAMInterface::run()` is a synchronous, blocking while-loop. | **NO** | **Not implemented** | Implement asynchronous prefetch queue in future phases. |
| **12. Is computation-transfer overlap modeled?** | Current simulator executes memory loads and compute sequentially ($T_{\text{overlap}} = 0$). | **NO** | **Not implemented** | Implement discrete-event overlap timing model. |
| **13. What is end-to-end speedup?** | Cannot claim speedup because memory prefetching is not yet asynchronous in C++. | **NO** | **Not measured** | Connect offline prediction decisions to non-blocking simulator DRAM path. |

---

## 2. Definitive Answers to the 9 Core Questions

### 1. Are our predictors offline-trained, online-adaptive, or a mixture?
- **CSTP**: A **mixture**. Spatial co-activations, temporal transitions, and marginal popularity priors are **offline-profiled** on training records and remain **frozen** during evaluation. However, the sequence-specific **temporal history is online** (rolling $t-1$ cache).
- **ST-MoE Paper**: A **mixture**. CCT candidate sets are **offline-initialized** from profiling records, but the 2-bit counters **adapt online** via Algorithm 3 increment/decrement/replacement, while the History Table tracks previous-token routing online.

### 2. What does CSTP know before the first inference token?
- The empirical marginal popularity of each expert at each stage ($P_{\text{prior}}$).
- The smoothed conditional transition probabilities $P(v \mid u)$ across adjacent layers.
- The smoothed token-to-token transition probabilities $P(v \mid w)$.
- **It knows zero token history for the upcoming request** (HT is empty).

### 3. What does ST-MoE know before the first inference token?
- The Top-$K$ most correlated destination candidates for each source expert at each layer transition, stored in the CCT with initial confidence $2$ (binary $10$).
- **It knows zero token history for the upcoming request** (HT is empty).

### 4. What does each predictor learn from the current request?
- **Both predictors learn the actual experts activated at token $t-1$**, storing them in `history_table[request_id][stage]` to guide prediction for token $t$.
- **ST-MoE additionally adjusts its 2-bit CCT counters**: reinforcing candidates that matched the token's routing and demoting or replacing candidates that missed.
- **CSTP leaves its transition tensors frozen** (by default), learning only the temporal anchor point for the next token.

### 5. What gets reset at the start of a request?
- **Only the temporal History Table for that specific `request_id`** (`del self.history_table[request_id]`).
- Learned spatial transition tensors, CCT dictionaries, and marginal priors are **preserved**.

### 6. Is our current accuracy evaluation causal?
- **Yes, 100% causal**. Prediction inputs contain only current layer selections and previous-token history. The target layer's ground-truth selections are revealed only **after** the prediction is recorded.

### 7. How much CPU time does each prediction take?
- **ST-MoE Paper**: Mean $= \mathbf{62.60\ \mu\text{s}}$ (P95 $= 99.91\ \mu\text{s}$).
- **CSTP (Direct)**: Mean $= \mathbf{197.46\ \mu\text{s}}$ (P95 $= 286.93\ \mu\text{s}$).
- **CSTP (Wrapper)**: Mean $= \mathbf{217.82\ \mu\text{s}}$ (P95 $= 410.89\ \mu\text{s}$).
- **Comparison to GPU Stage Computation**: A single transformer MoE execution stage takes $\approx 15\text{ to }30\text{ ms}$. Both predictors execute in $< 0.3\text{ ms}$, representing less than $1\%$ of the available computation window.

### 8. Can the current simulator actually overlap transfer with GPU computation?
- **No**. The existing simulator memory path is strictly synchronous and blocking: `issueRamulatorBatch` runs Ramulator to completion before returning to linear projection computation.

### 9. What must be implemented before we can claim latency improvement?
1. First, establish prediction quality and prefetch utility (Hit Rate, Precision, Recall, Coverage) on real datasets (Phase C).
2. Implement an asynchronous prefetching event model in C++ that issues DRAM transfers in parallel with attention/computation.
3. Model transfer completion timestamps, late demand stalls, and HBM buffer residency.

