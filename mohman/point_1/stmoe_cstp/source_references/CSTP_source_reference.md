# CSTP Source Reference & Interface Specification

**Project:** MoE Memory Wall: Predicting and Prefetching Experts for Faster LLM Inference  
**Deliverable:** Point 1 — Comparative Study Reference  
**Path:** [`mohman/point_1/stmoe_cstp/source_references/CSTP_source_reference.md`](file:///home/mohman/ca_project/mohman/point_1/stmoe_cstp/source_references/CSTP_source_reference.md)  
**Authoritative Git Anchor:** `d5bf61ff0c066c06f72173189f9924329f613a88`  

---

## 1. Primary Source Locations

- **Python Reference Predictor:** [`mohman/point_1/cstp/src/cstp_predictor.py`](file:///home/mohman/ca_project/mohman/point_1/cstp/src/cstp_predictor.py) (original: [`LLMSimulator/src/predictor/cstp_predictor.py`](file:///home/mohman/ca_project/LLMSimulator/src/predictor/cstp_predictor.py))
- **C++ Production Predictor:** [`mohman/point_1/cstp/src/cstp_predictor.cpp`](file:///home/mohman/ca_project/mohman/point_1/cstp/src/cstp_predictor.cpp) (original: [`LLMSimulator/src/prefetch/cstp_predictor.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/cstp_predictor.cpp))
- **C++ Class Header:** [`mohman/point_1/cstp/src/cstp_predictor.h`](file:///home/mohman/ca_project/mohman/point_1/cstp/src/cstp_predictor.h)
- **Base Interface:** [`mohman/point_1/cstp/src/base_predictor.py`](file:///home/mohman/ca_project/mohman/point_1/cstp/src/base_predictor.py)
- **Causal Evaluator:** [`mohman/point_1/cstp/src/evaluation.py`](file:///home/mohman/ca_project/mohman/point_1/cstp/src/evaluation.py)

---

## 2. Core Python Class API (`CSTPPredictor`)

```python
class CSTPPredictor(BasePredictor):
    def __init__(
        self,
        name: str = "CSTPPredictor",
        alpha: float = 0.5,
        smoothing_lambda: float = 1.0,
        prior_epsilon: float = 1e-4,
        history_capacity: int = 8,
        num_experts: int = 256,
        num_stages: int = 40,
    ):
        ...

    def fit_profile(self, training_records: List[Dict]) -> None:
        """Accumulates offline marginal, spatial, and temporal count matrices."""
        ...

    def predict(
        self,
        source_stage: int,
        active_source_experts: List[int],
        target_stage: int,
        budget: int = 8,
    ) -> List[int]:
        """Calculates composite spatial-temporal probabilistic scores and returns Top-B candidates."""
        ...

    def update_online(
        self,
        source_stage: int,
        active_source_experts: List[int],
        target_stage: int,
        ground_truth_experts: List[int],
    ) -> None:
        """Causally updates spatial, temporal, and marginal count matrices after prediction."""
        ...

    def reset_request_state(self) -> None:
        """Clears request-local history buffer. In online_within_request, also restores profiled tables."""
        ...
```

---

## 3. Core C++ Class API (`cstp_predictor.h`)

```cpp
class CSTPPredictor : public IPredictor {
public:
    CSTPPredictor(const PredictorConfig& config);
    virtual ~CSTPPredictor() = default;

    void fit_profile(const std::vector<RoutingEvent>& profile_data) override;
    std::vector<uint32_t> predict(
        uint32_t src_stage,
        const std::vector<uint32_t>& src_experts,
        uint32_t tgt_stage,
        uint32_t budget
    ) override;
    void update_online(
        uint32_t src_stage,
        const std::vector<uint32_t>& src_experts,
        uint32_t tgt_stage,
        const std::vector<uint32_t>& ground_truth
    ) override;
    void reset_request() override;
};
```

