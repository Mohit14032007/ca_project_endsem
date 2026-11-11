# ST-MoE Source Reference & Interface Specification

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Deliverable:** Point 1 — Comparative Study Reference  
**Path:** [`mohman/point_1/stmoe_cstp/source_references/ST_MoE_source_reference.md`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/source_references/ST_MoE_source_reference.md)  
**Authoritative Git Anchor:** `d5bf61ff0c066c06f72173189f9924329f613a88`  
**Paper Reference:** Zhao et al., *"A Spatio-Temporal Expert Prefetching Framework for Efficient MoE-based LLM Inference"*, arXiv:2606.15453v1  

---

## 1. Primary Source Locations

- **Paper-Faithful Python Predictor:** [`LLMSimulator/src/predictor/st_moe_paper.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/st_moe_paper.py)
- **C++ Production Predictor:** [`LLMSimulator/src/prefetch/st_moe_predictor.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/st_moe_predictor.cpp)
- **C++ Class Header:** [`LLMSimulator/src/prefetch/st_moe_predictor.h`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/st_moe_predictor.h)
- **ST-MoE Dedicated Results Archive:** [`mohman/point_1/st_moe/`](file:///home/mohman/ca_project/mohman/point_1/st_moe/)

---

## 2. Core Python Class API (`STMoEPaperPredictor`)

```python
class STMoEPaperPredictor(BasePredictor):
    def __init__(
        self,
        name: str = "STMoEPaperPredictor",
        cct_capacity: int = 8,
        confidence_bits: int = 2,
        confidence_threshold: int = 2,
        history_capacity: int = 8,
        num_experts: int = 256,
        scoring_mode: str = "confidence_additive",  # Equation 1: CCT + HT
    ):
        ...

    def fit_profile(self, training_records: List[Dict]) -> None:
        """Constructs offline Correlation Confidence Table (CCT) using 2-bit counters."""
        ...

    def predict(
        self,
        source_stage: int,
        active_source_experts: List[int],
        target_stage: int,
        budget: int = 8,
    ) -> List[int]:
        """Calculates dynamic composite score Score(e) = CCT(e) + HT(e) and ranks candidates."""
        ...

    def update_online(
        self,
        source_stage: int,
        active_source_experts: List[int],
        target_stage: int,
        ground_truth_experts: List[int],
    ) -> None:
        """Promotes/demotes CCT 2-bit saturating counters and updates History Table."""
        ...

    def reset_request_state(self) -> None:
        """Clears History Table. In online_within_request, restores CCT snapshot."""
        ...
```

---

## 3. Mathematical Scoring Formulation (Equation 1)

For every candidate expert $e \in \{0, \dots, M-1\}$ at target stage $L_{\text{tgt}}$:

$$\text{Score}(e) = \text{CCT}_{\text{conf}}(e) + \text{HT}_{\text{conf}}(e)$$

where:
- $\text{CCT}_{\text{conf}}(e) = \sum_{e_i \in E_{\text{src}}} \text{Counter}(e_i \to e) \in \{0, 1, 2, 3\}$.
- $\text{HT}_{\text{conf}}(e) \in \{0, 1, 2\}$ tracks recency of expert $e$ in recent decoding steps at $L_{\text{tgt}}$.
- In the fixed-budget regime ($B = K = 8$), the top 8 experts by $\text{Score}(e)$ are selected.

