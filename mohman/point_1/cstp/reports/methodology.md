# CSTP Theoretical Methodology & Formal Foundations

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Deliverable:** Point 1 — CSTP Methodology  
**Path:** [`mohman/point_1/cstp/reports/methodology.md`](file:///home/mohman/ca_project/mohman/point_1/cstp/reports/methodology.md)  
**Implementation:** [`src/cstp_predictor.py`](file:///home/mohman/ca_project/mohman/point_1/cstp/src/cstp_predictor.py) & [`src/cstp_predictor.cpp`](file:///home/mohman/ca_project/mohman/point_1/cstp/src/cstp_predictor.cpp)  

---

## 1. Mathematical Derivations

### 1.1 Mixture-of-Experts System Model
Consider an autoregressive transformer composed of $L_{\text{total}}$ decoder layers, where $S$ layers are MoE layers (stages $0, \dots, S-1$). At each MoE stage $s$, an input activation vector $x_t \in \mathbb{R}^d$ for decoding step $t$ is routed to a sparse subset of $K$ experts from a pool of $M$ experts:

$$E_t^{(s)} = \text{TopK}\left(\text{Softmax}(W_g^{(s)} x_t), K\right) \subset \{0, \dots, M-1\}$$

In our target architectures:
- **MoE Anywhere (Sage-A1.5):** $S = 40$, $M = 256$, $K = 8$.
- **MoE Suite (GLM-5.2):** $S = 75$, $M = 256$, $K = 8$.

Because $M \times \text{Size}(\text{Expert})$ vastly exceeds high-bandwidth GPU memory (HBM), unselected experts reside in host DRAM. The predictor must forecast $E_t^{(s+H)}$ when token $t$ is executing at stage $s$, where $H \in \{1, 2\}$ is the lookahead horizon.

---

### 1.2 Bayesian Decomposition of Expert Activation Probability

Let $Y_j \in \{0, 1\}$ denote the binary indicator that candidate expert $e_j$ is activated at stage $L_{\text{tgt}} = s + H$. The posterior probability of activation given observed source execution $E_{\text{src}} = E_t^{(s)}$ and historical execution context $\mathcal{H}_t = \{E_{\tau}^{(s+H)}\}_{\tau < t}$ is decomposed into spatial and temporal conditional likelihoods:

$$P(Y_j = 1 \mid E_{\text{src}}, \mathcal{H}_t) = \alpha P_{\text{spatial}}(Y_j = 1 \mid E_{\text{src}}) + (1 - \alpha) P_{\text{temporal}}(Y_j = 1 \mid \mathcal{H}_t)$$

where $\alpha \in [0, 1]$ represents the convex mixture weight.

---

### 1.3 Stage-Specific Marginal Expert Frequency Prior

Expert popularity distributions in MoE models are non-uniform, exhibiting heavy-tailed activation skew. To capture baseline expert frequencies independent of transient token context, CSTP computes stage-specific marginal priors:

$$P_{\text{marginal}}(e_j^{(L_{\text{tgt}})}) = \frac{C_{\text{marginal}}(L_{\text{tgt}}, e_j) + \epsilon}{\sum_{e'=0}^{M-1} C_{\text{marginal}}(L_{\text{tgt}}, e') + M \cdot \epsilon}$$

- $C_{\text{marginal}}(L, e)$ is the cumulative activation count of expert $e$ at layer $L$.
- $\epsilon > 0$ is a uniform Dirichlet/Laplace hyperparameter (default `1e-4`).

---

### 1.4 Spatial Transition Estimator with Empirical Prior Shrinkage

For a given source expert $e_i \in E_{\text{src}}$ at $L_{\text{src}}$, the conditional probability that expert $e_j$ activates at $L_{\text{tgt}}$ is modeled as a categorical distribution over $M$ outcomes. When sample counts are finite, maximum likelihood estimation $\frac{C(e_i, e_j)}{\sum C(e_i, e')}$ suffers from severe variance and over-allocation to spurious co-occurrences.

CSTP regularizes conditional probabilities by shrinking them toward the stage-specific marginal prior $P_{\text{marginal}}(e_j^{(L_{\text{tgt}})})$ using pseudocount parameter $\lambda > 0$:

$$P_{\text{spatial}}(e_j^{(L_{\text{tgt}})} \mid e_i^{(L_{\text{src}})}) = \frac{C_{\text{spatial}}(L_{\text{src}}, e_i, L_{\text{tgt}}, e_j) + \lambda \cdot P_{\text{marginal}}(e_j^{(L_{\text{tgt}})})}{\sum_{e'=0}^{M-1} C_{\text{spatial}}(L_{\text{src}}, e_i, L_{\text{tgt}}, e') + \lambda}$$

When the source expert $e_i$ has never been observed in training ($C_{\text{spatial}} = 0$), the posterior reverts gracefully to $P_{\text{marginal}}(e_j^{(L_{\text{tgt}})})$.

Aggregating over all $K$ activated source experts in $E_{\text{src}}$:

$$S_{\text{spatial}}(e_j^{(L_{\text{tgt}})}) = \frac{1}{|E_{\text{src}}|} \sum_{e_i \in E_{\text{src}}} P_{\text{spatial}}(e_j^{(L_{\text{tgt}})} \mid e_i^{(L_{\text{src}})})$$

---

### 1.5 Temporal Transition Estimator

Successive tokens in autoregressive generation exhibit semantic topic locality, causing recurring activations of domain-specific expert clusters. CSTP tracks this via first-order Markovian transitions across consecutive decoding tokens at the same layer:

$$P_{\text{temporal}}(e_j^{(L_{\text{tgt}})} \mid e_k^{(L_{\text{tgt}})}) = \frac{C_{\text{temporal}}(L_{\text{tgt}}, e_k, e_j) + \lambda \cdot P_{\text{marginal}}(e_j^{(L_{\text{tgt}})})}{\sum_{e'=0}^{M-1} C_{\text{temporal}}(L_{\text{tgt}}, e_k, e') + \lambda}$$

Aggregating over recent history buffer $E_{\text{hist}}$ (capacity $C_h = 8$):

$$S_{\text{temporal}}(e_j^{(L_{\text{tgt}})}) = \frac{1}{|E_{\text{hist}}|} \sum_{e_k \in E_{\text{hist}}} P_{\text{temporal}}(e_j^{(L_{\text{tgt}})} \mid e_k^{(L_{\text{tgt}})})$$

If $E_{\text{hist}} = \emptyset$ (tokens $0 \dots 4$ during prompt prefill or cold start):

$$S_{\text{temporal}}(e_j^{(L_{\text{tgt}})}) = P_{\text{marginal}}(e_j^{(L_{\text{tgt}})})$$

---

### 1.6 Composite Scoring and Bounded Prefetch Selection

The composite candidate score is:

$$S(e_j^{(L_{\text{tgt}})}) = \alpha S_{\text{spatial}}(e_j^{(L_{\text{tgt}})}) + (1 - \alpha) S_{\text{temporal}}(e_j^{(L_{\text{tgt}})})$$

Candidate experts are sorted in descending order of $S(e_j)$, and the top $B$ unique experts are selected:

$$\hat{E}_t^{(L_{\text{tgt}})} = \arg\max_{E \subset \{0, \dots, M-1\}, |E| = B} \sum_{e \in E} S(e)$$

---

## 2. Algorithmic Workflow & Causal Invariants

```
Algorithm: CSTP Evaluation Step (online_within_request)
Input: Source layer L_src, Active source experts E_src, Lookahead horizon H,
       Active request history buffer H_buf, Profiled state snapshot S_0
Output: Predicted expert set P_tgt of size B

1. L_tgt <- L_src + H
2. Retrieve E_hist <- H_buf[L_tgt]
3. For each candidate expert e_j in 0 ... M-1:
     Compute P_marginal(e_j) from stage_marginal_counts[L_tgt]
     Compute S_spatial(e_j) from spatial_counts[L_src, E_src, L_tgt, e_j]
     If E_hist is not empty:
         Compute S_temporal(e_j) from temporal_counts[L_tgt, E_hist, e_j]
     Else:
         S_temporal(e_j) <- P_marginal(e_j)
     S(e_j) <- alpha * S_spatial(e_j) + (1 - alpha) * S_temporal(e_j)
4. P_tgt <- TopB_Indices(S, B)
5. [Causal Boundary] Yield P_tgt for evaluation scoring against Ground Truth G_tgt
6. [Post-Scoring Online Adaptation]
     For each e_i in E_src, each e_j in G_tgt:
         Increment spatial_counts[L_src, e_i, L_tgt, e_j]
     For each e_k in E_hist, each e_j in G_tgt:
         Increment temporal_counts[L_tgt, e_k, e_j]
     For each e_j in G_tgt:
         Increment stage_marginal_counts[L_tgt, e_j]
     Update H_buf[L_tgt] with G_tgt
7. If EndOfRequest:
     Restore stage_marginal_counts, spatial_counts, temporal_counts <- S_0
     Clear H_buf
```

### Strict Causal Invariants
1. **Zero Future Peeking:** Predictions for token $t$ layer $L_{\text{tgt}}$ use only routing decisions from $L \le L_{\text{src}}$ at token $t$, and history up to token $t-1$.
2. **Order-Invariance Across Requests:** Because count tables restore to $S_0$ at request boundaries, evaluation of request $R_b$ is numerically identical whether evaluated before or after request $R_a$.

