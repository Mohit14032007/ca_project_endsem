# Metric Definitions & Mathematical Formulations

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Deliverable:** Point 1 — Comparative Study Methodology  
**Path:** [`mohman/point_1/stmoe_cstp/methodology/metrics_definitions.md`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/methodology/metrics_definitions.md)  

---

## 1. Primary Metrics (Event-Level)

For each prediction opportunity $i \in \{1, \dots, N\}$, let $P_i \subset \{0, \dots, M-1\}$ denote the predicted expert set ($|P_i| = B$), and $G_i \subset \{0, \dots, M-1\}$ denote the true routed expert set ($|G_i| = K$). The number of correct hits is:

$$H_i = |P_i \cap G_i|$$

### 1.1 Micro-Averaged Recall@B
The global proportion of ground-truth expert activations successfully retrieved:

$$\text{Recall}_{\text{micro}} = \frac{\sum_{i=1}^N H_i}{\sum_{i=1}^N |G_i|}$$

### 1.2 Micro-Averaged Precision@B
The global proportion of predicted experts that were actually executed:

$$\text{Precision}_{\text{micro}} = \frac{\sum_{i=1}^N H_i}{\sum_{i=1}^N |P_i|}$$

*When $B = K$ (as in our primary standard $B = K = 8$), $|P_i| = |G_i|$ for all $i$, so $\text{Recall}_{\text{micro}} \equiv \text{Precision}_{\text{micro}} \equiv \text{F1}_{\text{micro}}$.*

### 1.3 Hit Rate ($\mathbb{P}(\text{Hits} \ge 1)$)
The fraction of opportunities where the prefetcher retrieved at least one correct expert:

$$\text{Hit Rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(H_i \ge 1)$$

### 1.4 Exact Set Accuracy
The fraction of opportunities where the entire Top-$K$ set was predicted perfectly:

$$\text{Exact Set Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(H_i = K)$$

---

## 2. Macro vs. Micro Aggregation

- **Micro Aggregation:** Sums hits across all opportunities globally before dividing by total opportunities. Gives equal weight to every token event.
- **Macro Aggregation:** Computes metrics independently per prompt sequence $u$, and averages across all $U$ prompt units:
  $$\text{Recall}_{\text{macro}} = \frac{1}{U} \sum_{u=1}^U \text{Recall}_u$$
- **Macro F1:** Computed per prompt as $F1_u = \frac{2 \cdot P_u \cdot R_u}{P_u + R_u}$, then averaged: $\frac{1}{U} \sum_u F1_u$. *Note: This is strictly distinguished from the harmonic mean of macro precision and macro recall.*

---

## 3. Split Metrics: Cold-Start vs. Warm-Token

- **Cold-Start Recall:** Micro-recall evaluated strictly over initial prompt tokens ($t < 5$ opportunities per sequence) where temporal history buffers are partially unpopulated.
- **Warm-Token Recall:** Micro-recall evaluated over steady-state generation tokens ($t \ge 5$).

