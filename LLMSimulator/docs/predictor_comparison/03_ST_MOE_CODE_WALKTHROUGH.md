# ST-MoE Paper-Faithful Code Walkthrough and Internal Data-Structure Audit

**Implementation Source File**: [`src/predictor/st_moe_paper.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/st_moe_paper.py)  
**Reference Paper**: Yingnan Zhao et al., *A Spatio-Temporal Expert Prefetching Framework for Efficient MoE-based LLM Inference*, arXiv:2606.15453v1 [cs.AR], 13 Jun 2026.

---

## 1. Executive Summary

`STMoEPaperPredictor` faithfully implements the spatio-temporal expert prediction framework described by Zhao et al. (2026). It maintains:
1. **Cross-Layer Correlation Table (CCT)**: An array of lookup dictionaries tracking spatial correlations between adjacent layers ($L \to L+1$) using 2-bit saturating counters ($00, 01, 10, 11$) initialized to decimal $2$.
2. **History Table (HT)**: A request-isolated temporal store recording actual expert selections from the immediately preceding token ($t-1$) for each layer, conferring an implicit temporal confidence bonus of $+2$.
3. **Algorithm 2 Prediction Function & Equation 1 Aggregation**: Linearly accumulates multi-source spatial confidence and temporal confidence into integer scores, applying the paper's confidence threshold ($\ge 2$) and budget truncation.
4. **Algorithm 3 Table Update Function**: Incrementing matching candidates, decrementing mismatching candidates, and triggering replacement for zero-confidence slots.

---

## 2. Deep Dive: Core Data Structures

### 2.1 Cross-Layer Correlation Table (CCT)

#### Python Representation
```python
self.cct: Dict[int, Dict[int, Dict[int, int]]] = {}
```
- **Level 1 Key**: `source_stage_id` ($\text{int} \in [0, \text{num\_moe\_stages}-2]$). Represents the transition from layer $L$ to layer $L+1$.
- **Level 2 Key**: `source_expert_id` ($\text{int} \in [0, \text{num\_routed\_experts}-1]$). The active expert at layer $L$.
- **Level 3 Mapping**: `dest_expert_id: int -> counter: int`. Stores up to `self.cct_capacity` (Top-$K$) destination expert candidates and their 2-bit confidence values ($0, 1, 2, 3$).

#### 2-Bit Confidence Counter States
| Binary State | Decimal Value | Meaning in Paper | Operational Behavior |
| :---: | :---: | :--- | :--- |
| `00` | 0 | Strongly Not Preferred | **Eligible for replacement** upon next mismatch |
| `01` | 1 | Weakly Not Preferred | Below prefetch threshold; retained in table |
| `10` | 2 | Weakly Preferred | **Initial state** for all new/replaced entries; meets threshold |
| `11` | 3 | Strongly Preferred | Saturated maximum; requires 2 consecutive misses to fall below threshold |

#### Why Initialized to Decimal 2?
Algorithm 1 (Line 15) and Section 4.2 of the paper explicitly specify:
> "All entries are initialized to 10, so each stored expert is initially treated as a preferred candidate."
By initializing to decimal 2 (`10`), any candidate identified during offline profiling is immediately eligible for prefetching on its first inference activation, requiring an observed mismatch before being demoted.

#### Saturating Counter Dynamics (Algorithm 3)
When actual target-layer routing $F_j$ is revealed:
- **Match ($k \in F_j$)**:
  ```python
  if conf < 3:
      dest_map[k] = conf + 1
  else:
      dest_map[k] = 3
  ```
- **Mismatch ($k \notin F_j$)**:
  ```python
  if conf > 0:
      dest_map[k] = conf - 1
  else:
      # Replacement triggered
  ```

#### Replacement Policy (Algorithm 3, Lines 11-14)
When a stored candidate $k$ has counter $0$ upon mismatch:
1. `prefetch_set = set(dest_map.keys())`
2. Search for candidate $g \in (F_j \setminus \text{prefetch\_set})$.
3. If found, candidate $k$ is deleted, candidate $g$ is inserted, and its counter is initialized to decimal $2$ (`10`).
4. If multiple candidates are eligible, the lowest expert ID is selected deterministically.

#### Concrete CCT State Example
```python
self.cct[0][0] = {
    1: 3,  # Strongly preferred (counter=3)
    2: 2   # Preferred (counter=2)
}
self.cct[0][1] = {
    2: 3,  # Strongly preferred (counter=3)
    4: 2   # Preferred (counter=2)
}
```

---

### 2.2 History Table (HT)

#### Python Representation
```python
self.history_table: Dict[int, Dict[int, List[int]]] = {}
```
- **Level 1 Key**: `request_id` ($\text{int}$). Ensures 100% request isolation between concurrent or interleaved sequences.
- **Level 2 Key**: `stage_id` ($\text{int} \in [0, \text{num\_moe\_stages}-1]$).
- **Stored Value**: `List[int]` containing the actual Top-$K$ expert IDs activated by decoding token $t-1$ at `stage_id`.

#### HT Operational Properties
1. **Temporal Confidence Bonus**: Section 4.2 states that HT entries carry implicit confidence $2$ (binary $10$). In Algorithm 2 and Equation 1, each candidate in HT contributes $+2$ to its combined confidence score.
2. **Cold-Start Behavior**: For the first token of a sequence ($t=0$), `self.history_table[request_id]` is empty. The temporal score for all candidates is $0$, and predictions rely exclusively on spatial CCT lookup.
3. **Request Reset**: Calling `reset(request_id)` executes `del self.history_table[request_id]`, ensuring no state leaks to subsequent sequences.
4. **Update Timing**: HT is overwritten strictly **after** the gating network executes for that stage. Predicted experts are never stored in HT.

---

## 3. Prediction Algorithm & Equation 1 vs. Algorithm 2 Pseudocode

### 3.1 Trace of `predict()` Execution (Step-by-Step)

```python
def compute_confidence_scores(self, request_id, target_stage_id, current_source_experts):
    # 1. Initialize all E candidates to 0
    conf_score = {exp_id: 0 for exp_id in range(self.num_routed_experts)}

    # 2. Accumulate spatial scores from active source experts (Alg 2, Lines 3-5)
    if target_stage_id > 0 and len(current_source_experts) > 0:
        source_stage = target_stage_id - 1
        stage_cct = self.cct.get(source_stage, {})
        for j in current_source_experts:
            dest_map = stage_cct.get(j, {})
            for k, counter in dest_map.items():
                conf_score[k] = conf_score[k] + counter

    # 3. Add temporal contribution from History Table (Equation 1)
    req_hist = self.history_table.get(request_id, {})
    ht_experts = req_hist.get(target_stage_id, [])
    for h in ht_experts:
        if self.literal_alg2_overwrite:
            conf_score[h] = 2  # Literal Alg 2 Lines 7-9 overwrite
        else:
            conf_score[h] = conf_score[h] + 2  # Equation 1 addition

    return conf_score
```

### 3.2 Analysis of the Paper's Discrepancy: Equation 1 vs. Algorithm 2

The paper contains an internal discrepancy between its text/formula and its pseudocode:

1. **Equation (1) and Section 4.2 Text**:
   $$\text{Score}(e) = \begin{cases} \text{CCT}(e) + \text{HT}(e), & \text{if } e \in \text{CCT} \text{ and } e \in \text{HT} \\ \text{CCT}(e), & \text{if } e \in \text{CCT} \text{ and } e \notin \text{HT} \\ \text{HT}(e), & \text{if } e \notin \text{CCT} \text{ and } e \in \text{HT} \end{cases}$$
   The text states: *"The candidate experts from the two sources are merged, and their confidence scores are aggregated. If an expert appears in both the CCT and HT, its final confidence score is computed by combining the two scores, as shown in Eq. 1."*

2. **Algorithm 2 Pseudocode Lines 6-9**:
   ```text
   6: for all expert h in HT do
   7:     ConfScore[h] <- 1
   8: for all expert h in HT do
   9:     ConfScore[h] <- ConfScore[h] + 1
   ```
   If Line 7 is executed as a literal variable assignment (`ConfScore[h] = 1`), it obliterates any previously accumulated spatial score from Line 5! This contradicts Equation (1).

#### How Our Implementation Handles This
- **Primary Mode (`literal_alg2_overwrite=False`)**: Follows Equation 1: $\text{ConfScore}[h] \leftarrow \text{ConfScore}[h] + 2$. When an expert is nominated by both CCT and HT, its scores sum additively.
- **Literal Pseudocode Toggle (`literal_alg2_overwrite=True`)**: Configurable flag setting `conf_score[h] = 2` to allow empirical verification of the literal pseudocode.
- Both modes are tested and documented. Default execution adheres to Equation (1).

### 3.3 Threshold Filtering vs. Fixed-Budget Selection
- **Paper Threshold Selection (`predict_threshold`)**:
  $$\text{PrefetchSet} = \{ k \mid \text{ConfScore}[k] \ge 2 \}$$
- **Budget-Limited Prediction (`predict`)**:
  Sorts all $E$ candidates by `(-conf_score[idx], idx)` and returns the top $B$ elements.

---

## 4. Method-by-Method Source Walkthrough

| Method | Source Lines | Inputs | Outputs | Structures Read | Structures Modified | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `__init__` | [L21-L54](file:///home/mohman/ca_project/LLMSimulator/src/predictor/st_moe_paper.py#L21-L54) | Dimensions, capacities, threshold | None | None | `self.cct`, `self.history_table` | Allocates empty CCT lookup levels and HT dictionary |
| `reset` | [L56-L61](file:///home/mohman/ca_project/LLMSimulator/src/predictor/st_moe_paper.py#L56-L61) | `request_id` | None | None | `self.history_table` | Deletes `history_table[request_id]` |
| `build_cct_from_profile` | [L93-L146](file:///home/mohman/ca_project/LLMSimulator/src/predictor/st_moe_paper.py#L93-L146) | Training routing records | None | Input records | `self.cct` | Algorithm 1: Co-activation profiling and initial CCT construction (conf=2) |
| `compute_confidence_scores` | [L148-L188](file:///home/mohman/ca_project/LLMSimulator/src/predictor/st_moe_paper.py#L148-L188) | `req_id`, `target_stage`, `source_experts` | `Dict[int, int]` | `self.cct`, `self.history_table` | None | Pure read: Algorithm 2 & Eq 1 score calculation |
| `predict_threshold` | [L190-L208](file:///home/mohman/ca_project/LLMSimulator/src/predictor/st_moe_paper.py#L190-L208) | `req_id`, `target_stage`, `source_experts` | `List[int]` | `self.cct`, `self.history_table` | None | Filters candidates with ConfScore >= threshold |
| `predict` | [L210-L242](file:///home/mohman/ca_project/LLMSimulator/src/predictor/st_moe_paper.py#L210-L242) | `req_id`, `target_stage`, `source_experts`, `budget` | `List[int]` | `self.cct`, `self.history_table` | None | Standardized top-B ranking with deterministic tie-breaking |
| `update` | [L244-L313](file:///home/mohman/ca_project/LLMSimulator/src/predictor/st_moe_paper.py#L244-L313) | `req_id`, `target_stage`, `source_experts`, `actual_target_experts` | None | `actual_target_experts` | `self.cct`, `self.history_table` | Algorithm 3: Increments/decrements CCT counters, handles replacement, updates HT |

