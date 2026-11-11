# ST-MoE Paper Fidelity, Algorithmic Analysis, and Implementation Specification

**Reference Paper**:  
Yingnan Zhao, Razvan Bunescu, Ahmed Louri, Avinash Karanth, and Ke Wang.  
*A Spatio-Temporal Expert Prefetching Framework for Efficient MoE-based LLM Inference*, arXiv:2606.15453v1 [cs.AR], 13 Jun 2026.

---

## 1. Overview and Core Philosophy

In Mixture-of-Experts (MoE) LLMs, each layer dynamically routes tokens to a subset of $K$ experts out of $E$ total candidates (e.g., $K=8$ out of $E=256$ in DeepSeek-V3; $K=6$ out of $E=64$ in Qwen 2.0). Loading expert weights on-demand from off-chip DRAM introduces severe memory bandwidth bottlenecks and execution stalls.

ST-MoE proposes to proactively prefetch experts for layer $L+1$ while layer $L$ is executing computation and attention. The framework relies on two empirical observations:
1. **Spatial Correlation (Cross-Layer)**: Gating decisions across adjacent transformer layers are statistically dependent ($p < 0.01$) because representations are progressively refined while maintaining semantic properties.
2. **Temporal Correlation (Cross-Token)**: In autoregressive generation, consecutive decoding tokens share semantic context, leading to high overlap in expert activations within the same layer ($>2\times$ higher than uniform random expectation).

ST-MoE structures this knowledge into two lookup structures:
- **Cross-Layer Correlation Table (CCT)**: Tracks spatial cross-layer transitions.
- **History Table (HT)**: Tracks temporal consecutive-token activations.

---

## 2. Cross-Layer Correlation Table (CCT)

### 2.1 CCT Structure and Capacity
- The CCT maps each source layer $L$ and source expert $u \in \{0 \dots E-1\}$ to a bounded set of candidate destination experts $v \in \{0 \dots E-1\}$ at target layer $L+1$.
- **Capacity**: For each source expert, the CCT stores at most $C$ candidate experts.
  - The paper specifies $C = \text{Top-}K$ (Section 4.2: "ST-MoE then selects the Top-K correlated experts in the next layer for each expert in the current layer and stores them in the corresponding CCT entry, where K is set to match the number of activated experts in the target LLM").
  - In Section 5.2 (Table 3 discussion), for DeepSeek with 256 experts, "the EPU stores a 256-entry CCT, where each entry contains 8 candidates of 10 bits, including 8 bits for the expert candidate index and 2 bits for the confidence score."

### 2.2 2-Bit Saturating Confidence Counters
Each candidate in the CCT is associated with a 2-bit confidence counter:

| Binary State | Decimal Value | Semantic Meaning |
| :---: | :---: | :--- |
| `00` | 0 | Strongly not preferred |
| `01` | 1 | Weakly not preferred |
| `10` | 2 | Weakly preferred |
| `11` | 3 | Strongly preferred |

### 2.3 CCT Initialization
- **Paper Requirement**: Algorithm 1 (Line 15) and Section 4.2 state:
  $$\text{CCT}[j][*].\text{ConfScore} \leftarrow 10 \quad (\text{decimal } 2)$$
  "All entries are initialized to 10, so each stored expert is initially treated as a preferred candidate."
- **Legacy Implementation Deviation**: In `st_moe_predictor.py`, new candidate insertions were initialized to decimal `1` (`dest_map[v_star] = 1`). `st_moe_paper.py` faithfully initializes to decimal `2`.

### 2.4 CCT Profiling (Algorithm 1)
Algorithm 1 constructs the initial CCT offline from profiling tokens:
1. Initialize a co-activation matrix $\text{Co-matrix}[E][E] \leftarrow 0$.
2. For each profiling token $p \in \{1 \dots P\}$:
   - Let $E_i$ be the set of selected experts at layer $i$.
   - Let $F_i$ be the set of selected experts at layer $i+1$.
   - For all $j \in E_i$ and $k \in F_i$:
     $$\text{Co-matrix}[j][k] \leftarrow \text{Co-matrix}[j][k] + 1$$
3. For each source expert $j \in \{0 \dots E-1\}$:
   - Identify the Top-$K$ experts in $\text{Co-matrix}[j][*]$.
   - Store these in $\text{CCT}[j][*]$.
   - Initialize their confidence scores to binary `10` (decimal 2).

---

## 3. History Table (HT)

### 3.1 Structure and Maintenance
- The History Table stores the actual expert selections from the immediately preceding decoding token ($t-1$) for each MoE layer.
- **Capacity**: For each MoE layer, the HT stores $K$ experts.
- **Request Isolation**: The HT tracks state on a per-request/sequence basis. Request $A$ and Request $B$ maintain independent HTs.
- **Update Timing**: The HT is updated strictly **after** the gating network reveals the actual expert selections for that layer.
- **Temporal Confidence**: Section 4.2 states: "For consistency with the CCT-based prediction logic, each stored expert is also associated with a 2-bit confidence score. Unlike the CCT, however, these scores are not updated during execution and are initialized to 10." (decimal 2).

### 3.2 Cold-Start Behavior
At the first token of a sequence ($t=0$), the HT has no previous-token history. The temporal candidates set is empty ($\emptyset$). In this scenario, predictions rely entirely on the spatial CCT.

---

## 4. Prediction Algorithm (Algorithm 2 & Equation 1)

### 4.1 Paper Algorithmic Specification

Algorithm 2 defines the prediction function:
```text
Algorithm 2 Expert Predict Function
Input: t: current token index, E_i: experts selected at layer i for token t,
       CCT: cross-layer correlation table, HT: history table
Output: PrefetchSet: experts to prefetch for layer (i + 1)
1: Initialize:
2:   ConfScore[E] <- 0 for all experts E
3: for all expert j in E_i do
4:     for all candidate expert k in CCT[j] do
5:         ConfScore[k] <- ConfScore[k] + CCT[j][k]
6: for all expert h in HT do
7:     ConfScore[h] <- 1
8: for all expert h in HT do
9:     ConfScore[h] <- ConfScore[h] + 1
10: PrefetchSet <- { k | ConfScore[k] >= 2 }
11: return PrefetchSet
```

### 4.2 Equation 1 and the Pseudocode Discrepancy
Section 4.2 describes Equation (1):
$$\text{Score}(e) = \begin{cases} \text{CCT}(e) + \text{HT}(e), & \text{if } e \in \text{CCT} \text{ and } e \in \text{HT} \\ \text{CCT}(e), & \text{if } e \in \text{CCT} \text{ and } e \notin \text{HT} \\ \text{HT}(e), & \text{if } e \notin \text{CCT} \text{ and } e \in \text{HT} \end{cases}$$
where $\text{HT}(e) = 2$ for all $e \in \text{HT}$.

#### Critical Analysis of Algorithm 2 Lines 6–9:
- In the published paper's Algorithm 2:
  - Line 7 writes: `ConfScore[h] <- 1`
  - Line 9 writes: `ConfScore[h] <- ConfScore[h] + 1`
- If Line 7 were interpreted as an assignment (`= 1`), it would overwrite any accumulated spatial score $\text{ConfScore}[h]$ that was added in Line 5! This directly contradicts Equation (1) and the accompanying text: *"If an expert appears in both the CCT and HT, its final confidence score is computed by combining the two scores, as shown in Eq. 1; otherwise, its score remains unchanged."*
- **Faithful Implementation Interpretation**: In our paper-faithful implementation, we follow Equation (1) and the paper text:
  - For any candidate $h \in \text{HT}$, the temporal contribution of $+2$ is added to the accumulated spatial score:
    $$\text{ConfScore}[h] \leftarrow \text{ConfScore}[h] + 2$$
  - If an expert appears only in CCT, its score is its accumulated CCT confidence.
  - If an expert appears only in HT, its score is $0 + 2 = 2$.
  - If an expert appears in neither, its score is 0.
- Both interpretations (Equation 1 additive combination vs. literal overwrite) are documented and supported via an explicit configuration parameter `literal_alg2_overwrite = False` (defaulting to Equation 1 fidelity).

### 4.3 Multi-Source Spatial Accumulation
Notice Algorithm 2 Lines 3–5:
If two currently active source experts $j_1$ and $j_2$ both contain candidate $k$ in their CCT entries (with confidences $c_1$ and $c_2$), their contributions accumulate:
$$\text{ConfScore}[k] = c_1 + c_2$$
- In contrast, legacy `st_moe_predictor.py` took `max(c1, c2)` rather than the sum.

### 4.4 Thresholding vs. Fixed Budget
- **Paper Threshold Set**: Algorithm 2 filters by threshold:
  $$\text{PrefetchSet}_{\text{threshold}} = \{ k \mid \text{ConfScore}[k] \ge 2 \}$$
  Because any candidate in HT has score $\ge 2$, and any CCT candidate with confidence $\ge 2$ has score $\ge 2$, this set can contain between 0 and $K_{\text{spatial}} + K_{\text{temporal}}$ experts.
- **Fixed Budget Selection**: In real system prefetching under bandwidth and buffer constraints, the system must select at most $B$ candidates.
  - Candidates are sorted by $\text{ConfScore}[k]$ descending.
  - Ties are broken deterministically by expert index ascending.
  - The top $B$ candidates are returned.

---

## 5. Table Update Algorithm (Algorithm 3)

Algorithm 3 specifies the online update procedure executed when the actual target layer selections $F_j$ are revealed:
```text
Algorithm 3 Table Update Function
Input: E_i: experts selected at layer i, F_j: experts selected at layer (i + 1),
       CCT: cross-layer correlation table
Output: Updated CCT
1: for all expert j in E_i do
2:     PrefetchSet <- experts stored in CCT[j][*]
3:     for all (k, ConfScore) in CCT[j][*] do
4:         if k in F_j then
5:             ConfScore <- min(ConfScore + 1, 3)
6:         else
7:             if ConfScore > 0 then
8:                 ConfScore <- ConfScore - 1
9:             else
10:                g <- any expert in (F_j \ PrefetchSet)
11:                k <- g
12:                ConfScore <- 10 (decimal 2)
13: return CCT
```

### 5.1 Saturating Counter Dynamics
- **Match ($k \in F_j$)**: Counter increments: $\min(\text{ConfScore} + 1, 3)$.
  Transitions: $0 \to 1$, $1 \to 2$, $2 \to 3$, $3 \to 3$.
- **Mismatch ($k \notin F_j$)**: Counter decrements if $> 0$:
  Transitions: $3 \to 2$, $2 \to 1$, $1 \to 0$.
- **Replacement Trigger**: If a candidate has $\text{ConfScore} == 0$ upon a mismatch, it becomes eligible for replacement.
  - An expert $g \in (F_j \setminus \text{PrefetchSet})$ is selected.
  - The slot $k$ is replaced by $g$.
  - The new candidate's confidence is initialized to binary `10` (decimal 2).
  - Deterministic tie-breaking: If multiple candidates $g$ are eligible, the one with the lowest expert ID is chosen.

---

## 6. Comprehensive Comparison: `st_moe_paper` vs. `st_moe_legacy` vs. `cstp`

| Dimension | Paper-Faithful ST-MoE (`st_moe_paper`) | Legacy ST-MoE (`st_moe_legacy`) | Proposed CSTP (`cstp`) |
| :--- | :--- | :--- | :--- |
| **Scoring Formulation** | Integer score aggregation from CCT counters and HT: $\text{Score}(e) = \text{CCT}(e) + \text{HT}(e)$ | Tiered priority fallback (Spatial $\ge 2$ first, then HT, then Spatial $< 2$) | Dirichlet-smoothed continuous composite probability: $\alpha S_{\text{spatial}} + (1-\alpha) S_{\text{temporal}}$ |
| **Multi-Source Spatial** | Additive sum: $\sum_{u \in E_i} \text{CCT}[u][k]$ | Maximum counter: $\max_{u \in E_i} \text{CCT}[u][k]$ | Averaged conditional transition probability with Dirichlet smoothing |
| **CCT Init Confidence** | Binary `10` (Decimal 2) | Decimal 1 | Continuous prior from profiling |
| **Replacement Init** | Decimal 2 | Decimal 1 | Continuous counts updated |
| **HT Representation** | Top-$K$ previous token routing with confidence 2 | Set of previous token routing | Normalized probability distribution over all $E$ |
| **Vocabulary Coverage** | Bounded top-$K$ candidates per source expert | Bounded top-$K$ candidates per source expert | Full vocabulary evaluation ($E$ experts) |
| **Target Prior Awareness** | None (only tracks observed pairs) | Fallback marginal prior list | Explicit empirical marginal prior with smoothing $\epsilon$ |
| **CCT Replacement** | When ConfScore == 0, replaces with $g \in (F_j \setminus \text{PrefetchSet})$ | Searches for any zero-counter entry | Continuous counts naturally shift probability mass |

---

## 7. Deterministic Hand-Calculated Verification Example

To guarantee 100% mathematical fidelity and reproducibility, we construct a fully specified numerical example that can be calculated by hand.

### 7.1 Setup Specifications
- **Model Topology**:
  - Source Layer $L$, Target Layer $L+1$.
  - Expert vocabulary size $E = 6$, indexed $\{0, 1, 2, 3, 4, 5\}$.
  - Top-$K = 2$.
- **Current Observation at Layer $L$ (Token $t$)**:
  - $E_L = \{0, 1\}$.
- **Cross-Layer Correlation Table (CCT)** at Layer $L \to L+1$:
  - For Source Expert $0$:
    - Destination Expert $1$: Confidence $= 3$
    - Destination Expert $2$: Confidence $= 2$
  - For Source Expert $1$:
    - Destination Expert $2$: Confidence $= 3$
    - Destination Expert $4$: Confidence $= 2$
  - Other source experts $\{2, 3, 4, 5\}$ are not currently active.
- **History Table (HT)** at Target Layer $L+1$ from Token $t-1$:
  - $HT = \{2, 3\}$
  - By Section 4.2 / Algorithm 2, each expert in HT has temporal confidence $\text{HT}(e) = 2$.

---

### 7.2 Step-by-Step Numerical Walkthrough

#### Step 1: Initialize Candidate Confidence Scores
$$\text{ConfScore}[v] = 0 \quad \text{for all } v \in \{0, 1, 2, 3, 4, 5\}$$

#### Step 2: Accumulate Spatial Scores (Algorithm 2, Lines 3–5)
- From source expert $j = 0$:
  - Candidate $k = 1$: $\text{ConfScore}[1] \leftarrow 0 + 3 = 3$
  - Candidate $k = 2$: $\text{ConfScore}[2] \leftarrow 0 + 2 = 2$
- From source expert $j = 1$:
  - Candidate $k = 2$: $\text{ConfScore}[2] \leftarrow 2 + 3 = 5$  *(Both source experts 0 and 1 nominate destination 2!)*
  - Candidate $k = 4$: $\text{ConfScore}[4] \leftarrow 0 + 2 = 2$

Intermediate Spatial Scores:
$$\text{ConfScore} = [0: 0,\; 1: 3,\; 2: 5,\; 3: 0,\; 4: 2,\; 5: 0]$$

#### Step 3: Apply Temporal Contributions from HT (Algorithm 2 & Equation 1)
$HT = \{2, 3\}$, with temporal confidence $+2$ each:
- For $h = 2$: $2 \in \text{CCT}$ and $2 \in \text{HT} \implies \text{ConfScore}[2] \leftarrow 5 + 2 = 7$
- For $h = 3$: $3 \notin \text{CCT}$ and $3 \in \text{HT} \implies \text{ConfScore}[3] \leftarrow 0 + 2 = 2$

Final Combined Scores:
| Expert ID | Spatial Contribution | Temporal Contribution | Final Combined Score |
| :---: | :---: | :---: | :---: |
| **0** | 0 | 0 | **0** |
| **1** | 3 | 0 | **3** |
| **2** | 5 | 2 | **7** |
| **3** | 0 | 2 | **2** |
| **4** | 2 | 0 | **2** |
| **5** | 0 | 0 | **0** |

#### Step 4: Evaluate Paper Threshold Criterion ($\text{ConfScore}[v] \ge 2$)
$$\text{PrefetchSet}_{\text{threshold}} = \{ v \mid \text{ConfScore}[v] \ge 2 \} = \{1, 2, 3, 4\}$$

#### Step 5: Ranking and Budget-Limited Selection
Sort candidates by score descending; break ties by expert ID ascending:
1. **Expert 2**: Score $7$
2. **Expert 1**: Score $3$
3. **Expert 3**: Score $2$ (Tied with 4; ID $3 < 4$)
4. **Expert 4**: Score $2$ (ID $4$)
5. **Expert 0**: Score $0$ (ID $0 < 5$)
6. **Expert 5**: Score $0$ (ID $5$)

Selection under different prefetch budgets $B$:
- **Budget $B = 1$**: `[2]`
- **Budget $B = 2$**: `[2, 1]`
- **Budget $B = 3$**: `[2, 1, 3]`
- **Budget $B = 4$**: `[2, 1, 3, 4]`

---

### 7.3 Step-by-Step Online Table Update (Algorithm 3)
Now assume the actual routing for target layer $L+1$ is revealed as:
$$F_j = \{2, 5\}$$

#### Updating Source Expert $0$ CCT:
- Stored candidates: $\{1: 3, 2: 2\}$.
- Candidate $1$: $1 \notin F_j \implies \text{ConfScore} = 3 - 1 = 2$.
- Candidate $2$: $2 \in F_j \implies \text{ConfScore} = \min(2 + 1, 3) = 3$.
- Updated `CCT[0]`: $\{1: 2, 2: 3\}$.

#### Updating Source Expert $1$ CCT:
- Stored candidates: $\{2: 3, 4: 2\}$.
- Candidate $2$: $2 \in F_j \implies \text{ConfScore} = \min(3 + 1, 3) = 3$.
- Candidate $4$: $4 \notin F_j \implies \text{ConfScore} = 2 - 1 = 1$.
- Updated `CCT[1]`: $\{2: 3, 4: 1\}$.

#### Updating History Table:
- $\text{HT}_{L+1} \leftarrow \{2, 5\}$.

This exact test case is validated programmatically in `test_predictor_hand_example.py`.

---

## 8. Unresolved Paper Ambiguities and Implementation Decisions

1. **Algorithm 2 Line 7 vs. Equation 1**:
   - *Ambiguity*: Algorithm 2 line 7 states `ConfScore[h] <- 1` which literally overwrites accumulated spatial scores.
   - *Resolution*: As detailed in Section 4.2, Equation 1 and the paper text state that CCT and HT scores are combined ($CCT(e) + HT(e)$). We implement Equation 1 as primary, and provide a configurable toggle for the literal overwrite.
2. **Deterministic Tie-Breaking**:
   - *Ambiguity*: The paper does not specify how to order candidates with identical confidence scores when enforcing a budget $B$.
   - *Resolution*: We enforce deterministic ascending expert ID ordering.
3. **CCT Replacement Selection**:
   - *Ambiguity*: Algorithm 3 line 10 specifies "$g \leftarrow \text{any expert in } (F_j \setminus \text{PrefetchSet})$".
   - *Resolution*: When multiple un-prefetched actual experts exist, we select the lowest expert ID deterministically.

---

## 9. Comprehensive Test Coverage and Verification Matrix

The paper-faithful ST-MoE implementation is validated across 24 dedicated unit tests in `tests/predictor_comparison/`:

| Verification Area | Test File | Test Method | Tested Functionality |
| :--- | :--- | :--- | :--- |
| **CCT Init** | `test_st_moe_paper.py` | `test_cct_initialization_to_decimal_two` | ConfScore initialized to decimal 2 (10) |
| **Counter Dynamics** | `test_st_moe_paper.py` | `test_cct_saturating_counter_increment_and_ceiling` | Increment saturates at 3 (11) |
| **Counter Dynamics** | `test_st_moe_paper.py` | `test_cct_saturating_counter_decrement_and_floor` | Decrement floors at 0 (00) |
| **Replacement** | `test_st_moe_paper.py` | `test_cct_zero_confidence_replacement_eligibility` | Replacement triggers on 0; new candidate initialized to 2 |
| **HT Timing** | `test_st_moe_paper.py` | `test_history_table_isolation_and_timing` | HT updated post-routing per-request and per-stage |
| **Multi-Source** | `test_st_moe_paper.py` | `test_multi_source_spatial_score_accumulation` | Spatial score summation across source experts (Alg 2) |
| **Eq 1 Aggregation** | `test_st_moe_paper.py` | `test_equation1_spatio_temporal_combination` | Joint CCT(e) + HT(e) additive scoring |
| **Thresholding** | `test_st_moe_paper.py` | `test_predict_threshold_filtering` | PrefetchSet filters by ConfScore >= 2 |
| **Budget Policy** | `test_st_moe_paper.py` | `test_budget_limited_prediction` | Deterministic ranking and top-B truncation |
| **Cold-Start** | `test_st_moe_paper.py` | `test_cold_start_prediction_without_history` | Spatial fallback when HT is empty |
| **Hand Example** | `test_predictor_hand_example.py` | `test_hand_calculated_confidence_scores` | Exact verification of all 6 candidate scores |
| **Hand Example** | `test_predictor_hand_example.py` | `test_hand_calculated_threshold_set` | Exact verification of threshold set {1, 2, 3, 4} |
| **Hand Example** | `test_predictor_hand_example.py` | `test_hand_calculated_budget_selections` | Exact verification of budget subsets (B=1,2,3,4) |
| **Hand Example** | `test_predictor_hand_example.py` | `test_hand_calculated_algorithm3_update` | Exact verification of CCT counters & HT update |
| **Polymorphism** | `test_predictor_interface.py` | `test_all_classes_inherit_from_base_predictor` | BasePredictor inheritance across all wrappers |
| **Interface Budget** | `test_predictor_interface.py` | `test_predict_returns_valid_expert_ids_and_respects_budget` | Budget constraint adherence |
| **Boundary Budgets**| `test_predictor_interface.py` | `test_zero_and_negative_budgets` | Clean empty list return on budget <= 0 |
| **CSTP Invariance** | `test_predictor_interface.py` | `test_cstp_mathematical_preservation_in_wrapper` | Wrapper preserves CSTP scores bit-for-bit |
| **Frequency Baseline**| `test_predictor_interface.py` | `test_frequency_predictor_profiling_and_ranking` | Frequency ranking baseline verification |
| **Request Isolation** | `test_predictor_request_isolation.py` | `test_st_moe_paper_and_legacy_request_isolation` | Independent temporal state across interleaved requests |
| **Request Isolation** | `test_predictor_request_isolation.py` | `test_cstp_request_isolation` | CSTP temporal isolation across sequences |
| **Causality** | `test_predictor_causality.py` | `test_future_label_mutation_invariance` | Future label mutation has 0 effect on predictions |
| **Idempotence** | `test_predictor_causality.py` | `test_idempotent_repeated_prediction` | Calling predict() repeatedly does not mutate state |
| **Update Sequence** | `test_predictor_causality.py` | `test_update_executes_only_post_prediction` | Internal state updates only occur upon update() |
