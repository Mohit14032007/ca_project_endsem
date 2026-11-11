# ST-MoE Evaluation Methodology and Metric Formulations

**Deliverable:** Point 1 — Paper-Faithful ST-MoE Baseline  
**Harness Source:** [`LLMSimulator/src/predictor/evaluation.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/evaluation.py)  
**Verification Suite:** [`LLMSimulator/tests/test_evaluation_semantics.py`](file:///home/mohman/ca_project/LLMSimulator/tests/test_evaluation_semantics.py)

---

## 1. Mathematical Metric Formulations

Let $N$ denote the total number of evaluated prediction opportunities in a test campaign. For each opportunity $i \in \{1, \dots, N\}$, let:
- $G_i \subset \{1, \dots, E\}$ denote the ground-truth set of routed experts activated by the gating network ($|G_i| = K = 8$).
- $P_i \subset \{1, \dots, E\}$ denote the set of candidate experts predicted by ST-MoE ($|P_i| = B$ under fixed budget; variable under thresholding).

### 1.1 Micro-Averaged Recall
Recall measures the proportion of actual executed experts successfully forecasted by the predictor:

$$\text{Recall} = \frac{\sum_{i=1}^N |P_i \cap G_i|}{\sum_{i=1}^N |G_i|} = \frac{1}{N \cdot K} \sum_{i=1}^N |P_i \cap G_i|$$

### 1.2 Micro-Averaged Precision
Precision measures the fraction of prefetch candidates that were actually executed by the model:

$$\text{Precision} = \frac{\sum_{i=1}^N |P_i \cap G_i|}{\sum_{i=1}^N |P_i|}$$

### 1.3 Micro-Averaged F1 Score
The harmonic mean of precision and recall:

$$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

> [!IMPORTANT]
> **Mathematical Equivalence at $B = K$:**  
> When the prediction budget is constrained to match the model routing degree ($B = K = 8$), every opportunity predicts exactly 8 experts ($|P_i| = 8$) and executes exactly 8 experts ($|G_i| = 8$). Consequently:
> $$\sum_{i=1}^N |P_i| = \sum_{i=1}^N |G_i| = 8N \implies \text{Recall@8} \equiv \text{Precision@8} \equiv \text{F1@8}$$

---

### 1.4 Hit Rate (Any-Hit Opportunity Coverage)
Hit Rate quantifies the fraction of prediction opportunities where **at least one** predicted expert was correct:

$$\text{Hit Rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(|P_i \cap G_i| \ge 1\right)$$

Hit Rate is a standard cache-line coverage metric. If a prefetcher achieves 95% hit rate, the compute core finds at least one required expert already in memory on 95% of execution steps, avoiding total pipeline stall even if other experts must still be fetched.

---

### 1.5 Cold-Start vs. Steady-State Recall
To assess how rapidly the predictor adapts to a new prompt:
- **Cold-Start Opportunities ($\mathcal{O}_{\text{cold}}$):** Evaluated over the first 5 decoding steps of each request ($t \in \{0, 1, 2, 3, 4\}$), where the History Table is empty or sparse:
  $$\text{Recall}_{\text{cold}} = \frac{\sum_{i \in \mathcal{O}_{\text{cold}}} |P_i \cap G_i|}{\sum_{i \in \mathcal{O}_{\text{cold}}} |G_i|}$$
- **Steady-State Opportunities ($\mathcal{O}_{\text{steady}}$):** Evaluated over all subsequent steps ($t \ge 5$), where temporal history has populated:
  $$\text{Recall}_{\text{steady}} = \frac{\sum_{i \in \mathcal{O}_{\text{steady}}} |P_i \cap G_i|}{\sum_{i \in \mathcal{O}_{\text{steady}}} |G_i|}$$

---

### 1.6 Exact Set Accuracy
The strict probability that the predicted set perfectly matches the true routed set:

$$\text{Exact Set Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(P_i = G_i\right)$$

Because choosing 8 experts out of 256 yields $\binom{256}{8} \approx 4.38 \times 10^{14}$ combinations, exact set accuracy is naturally low ($0.0\%$ to $0.9\%$) in training-free predictors.

---

### 1.7 Mean Predicted Set Size (Threshold Mode)
Under native thresholding, the number of emitted candidates varies dynamically:

$$\overline{|P|} = \frac{1}{N} \sum_{i=1}^N |P_i|$$

This metric directly indicates whether a thresholding policy induces bandwidth explosion on the accelerator interconnect.

---

## 2. Evaluation Policies and Adaptation Protocols

Phase 2B formalized three mutually exclusive evaluation regimes:

| Evaluation Policy | Initial CCT State | Adaptation Scope | Request-Order Invariant? | Multi-Tenant Clean? |
| :--- | :--- | :--- | :---: | :---: |
| **`frozen`** | Offline Profiling | None (Static) | Yes | Yes |
| **`online_within_request`** | Offline Profiling | Intra-request only (Resets per request) | **Yes** | **Yes** |
| **`online_across_requests`** | Offline Profiling | Continuous lifelong adaptation | No | No |

### Why `online_within_request` is the Primary Standard
1. **Scientific Reproducibility:** Because the predictor is reset to its calibrated profiling state prior to each test prompt, the resulting recall is completely invariant to prompt ordering.
2. **Production Realism:** In cloud LLM serving, user prompts arrive concurrently or sequentially from disjoint tasks. Allowing CCT updates to bleed across unrelated users causes synthetic drift and cache poisoning.
3. **Controlled Comparison:** It isolates the true intra-prompt semantic learning capability of ST-MoE from long-term memory accumulation, establishing a fair benchmark for evaluating our proposed CSTP technique.

