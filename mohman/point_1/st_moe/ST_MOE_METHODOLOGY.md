# ST-MoE Architectural Methodology and Theoretical Framework

**Deliverable:** Point 1 — Paper-Faithful ST-MoE Baseline  
**Reference Paper:** Zhao et al., *"A Spatio-Temporal Expert Prefetching Framework for Efficient MoE-based LLM Inference"*, arXiv:2606.15453v1  
**Implementation Source:** [`LLMSimulator/src/predictor/st_moe_paper.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/st_moe_paper.py)

---

## 1. High-Level Architectural Motivation

Mixture-of-Experts (MoE) architectures dramatically scale parameter count while maintaining constant per-token compute FLOPs by activating only a small subset of experts (Top-$k$) at each layer. However, during autoregressive token generation, the routing decisions of layer $L$ are computed dynamically by a gating network from hidden activations:

$$G(x) = \text{TopK}(\text{Softmax}(W_g x), K)$$

Because expert parameters in massive models (e.g., 256 experts per layer across 40–75 layers) vastly exceed high-bandwidth GPU memory (HBM) capacity, unselected experts must either remain in host DRAM or be dynamically transferred across PCIe. Waiting for gating computation to finish before fetching expert weights incurs severe memory stall stalls on the critical execution path.

The **ST-MoE** framework addresses this memory wall by prefetching experts $H$ stages in advance ($H \in \{1, 2\}$) without retraining the model. It exploits two intrinsic properties of transformer routing traces:
1. **Spatial Correlation:** If expert $e_i$ is activated at source layer $L_{\text{src}}$, certain experts $\{e_j\}$ at downstream layer $L_{\text{tgt}} = L_{\text{src}} + H$ exhibit elevated conditional activation probabilities $P(e_j^{(L+H)} \mid e_i^{(L)})$.
2. **Temporal Correlation:** Tokens within the same request or sentence reuse a recurring semantic cluster of experts across successive decoding steps.

---

## 2. Core Predictor Components

```
                   +---------------------------------------------+
                   |           Current Token Execution           |
                   | Source Experts at Layer L_src: {e_1, ...}   |
                   +----------------------+----------------------+
                                          |
                +-------------------------+-------------------------+
                |                                                   |
                v                                                   v
    +-----------------------+                           +-----------------------+
    | Spatial Correlation   |                           | Temporal Correlation  |
    | Correlation Confidence|                           | History Table (HT)    |
    | Table (CCT)           |                           | (Request-Local Buffer)|
    | 2-bit Counters (0..3) |                           | Recent Expert Recency |
    +-----------+-----------+                           +-----------+-----------+
                |                                                   |
                | CCT Confidence Values                             | HT Recency Values
                +-------------------------+-------------------------+
                                          |
                                          v
                   +---------------------------------------------+
                   | Dynamic Scoring (Equation 1)                |
                   | Score(e) = CCT_conf(e) + HT_conf(e)         |
                   +----------------------+----------------------+
                                          |
                                          v
                   +---------------------------------------------+
                   | Prefetch Set Selection                      |
                   | Fixed Budget: Top-B Candidates by Score     |
                   | Native Threshold: Filter Score >= tau       |
                   +---------------------------------------------+
```

### 2.1 The Correlation Confidence Table (CCT)
The CCT is a structured spatial correlation cache organized by target layer and source expert:

$$\text{CCT}[L_{\text{tgt}}][e_{\text{src}}] \to \text{Dict}[e_{\text{cand}}, \text{Counter}]$$

Each source expert tracks up to $C = 8$ candidate experts at downstream layer $L_{\text{tgt}}$. Every candidate entry is governed by a **2-bit saturating counter**:

$$\text{Counter} \in \{0, 1, 2, 3\}$$

The states represent:
- `3` (`11_2`): **Strongly Active** — High historical co-occurrence.
- `2` (`10_2`): **Weakly Active** — Moderate historical co-occurrence.
- `1` (`01_2`): **Weakly Inactive** — Newly inserted candidate or decaying correlation.
- `0` (`00_2`): **Strongly Inactive** — Prime candidate for eviction.

#### Offline Profiling (Algorithm 1)
Prior to serving, the CCT is calibrated on a sequence of offline routing traces:
1. For every observed transition $(L_{\text{src}}, e_i) \to (L_{\text{tgt}}, e_j)$:
   - If $e_j$ exists in $\text{CCT}[L_{\text{tgt}}][e_i]$, its counter increments: $c = \min(3, c + 1)$.
   - If $e_j$ is absent and the table has capacity ($< 8$), $e_j$ is inserted with initial counter $c = 1$.
   - If capacity is saturated, the entry with the lowest counter is replaced if its counter is $\le 1$.
2. For candidate experts in the table that were *not* activated in layer $L_{\text{tgt}}$, their counters decrement: $c = \max(0, c - 1)$.

#### Online Dynamic Update (Algorithm 3)
During online inference, the CCT adapts continuously to incoming tokens. Once the actual ground truth routing decisions for layer $L_{\text{tgt}}$ become known, the update rule executes causally:
- Co-occurring candidate counters increment towards 3.
- Absent candidate counters decrement towards 0.
- Newly observed recurring correlations replace inactive entries ($c = 0$).

---

### 2.2 The History Table (HT)
The History Table captures temporal locality. In transformer decoding, consecutive tokens within a prompt frequently share thematic and syntactic domains, causing repetitive expert activation.

The HT maintains a sliding buffer of the most recently activated experts within the active request. For an expert $e$, its temporal score is determined by recency:

$$\text{HT}(e) \in \{0, 1, 2, 3\}$$

where actively referenced experts receive maximum confidence ($3$), decaying over subsequent decoding steps if not re-activated. Crucially, the HT is strictly **request-local**; when a prompt finishes, the HT is completely flushed to prevent cross-request contamination.

---

## 3. Dynamic Prediction Scoring (Equation 1)

The paper defines the combined prediction score for candidate expert $e$ at downstream layer $L_{\text{tgt}}$ via Equation 1:

$$\text{Score}(e) = \text{CCT}(e) + \text{HT}(e)$$

where:
- $\text{CCT}(e) = \max_{e_{\text{src}} \in G(L_{\text{src}})} \text{Counter}(e_{\text{src}} \to e)$ is the maximum spatial confidence among all active source experts at layer $L_{\text{src}}$.
- $\text{HT}(e)$ is the temporal recency confidence of expert $e$ in the History Table.

### Dual Mathematical Interpretations of Equation 1
In our Phase 2A audit, we identified an ambiguity between the mathematical literal in Equation 1 and the accompanying prose description in Algorithm 2:

```
+-----------------------------------+-----------------------------------+
|  1. Confidence Additive (Literal) | 2. Indicator Intersection (Prose) |
+-----------------------------------+-----------------------------------+
| Score(e) = CCT(e) + HT(e)         | Score(e) = I(CCT >= 1) + I(HT >= 1)|
| Score range: [0, 6]               | Score range: {0, 1, 2}            |
| Threshold tau = 2 admits:         | Threshold tau = 2 admits:         |
|   - CCT >= 2 (Spatial only)       |   - Strictly requires BOTH        |
|   - HT >= 2 (Temporal only)       |     CCT >= 1 AND HT >= 1          |
|   - CCT >= 1 AND HT >= 1          |     (Intersection of tables)      |
+-----------------------------------+-----------------------------------+
```

Our implementation in [`LLMSimulator/src/predictor/st_moe_paper.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/st_moe_paper.py) explicitly supports both modes via the `--scoring-mode` flag:
1. `confidence_additive`: Authoritative mathematical reproduction of Equation 1.
2. `indicator_intersection`: Ablation evaluating the strict intersection hypothesis.

---

## 4. Prefetch Selection: Bounded Budget vs. Native Thresholding

The paper suggests a native thresholding rule: admit any candidate expert whose $\text{Score}(e) \ge \tau = 2$.

### The Pathological Set-Size Dilation of Native Thresholding
In practical hardware systems, DRAM-to-HBM bus bandwidth and prefetch buffer capacity are strictly bounded. Our empirical experiments reveal a critical flaw in unbounded thresholding:

1. Under `confidence_additive`, any candidate with $\text{CCT} \ge 2$ qualifies regardless of temporal history. Because 8 source experts each vote on up to 8 candidates, the union of qualified candidates dilates dramatically:
   - On `MoE Anywhere` ($K=8$), the mean predicted set size is **$|P| = 32.02$ experts** (frozen) and **$20.88$ experts** (online).
   - On `MoE Suite` ($K=8$), the mean predicted set size is **$|P| = 44.11$ experts** (frozen) and **$27.93$ experts** (online).
2. Fetching 32 experts when the model only executes 8 causes a **$4\times$ memory bandwidth amplification**, diluting precision down to $18.9\%$–$27.8\%$ and causing massive bus contention.

### The Fixed-Budget Solution ($B \le K$)
To provide actionable prefetch candidates for real accelerator memory hierarchies, we implement **fixed-budget ranking**:
- All candidates are scored via Equation 1.
- Candidates are sorted in descending order of score.
- The top $B$ candidates ($B \in \{1, 2, 4, 6, 8\}$) are selected:

$$P = \text{TopB}_{e}(\text{Score}(e))$$

When $B = K = 8$, the prefetcher fetches exactly the number of experts the hardware pipeline can execute, bounding bandwidth overhead to $1.0\times$ while maximizing expert hit rate.

