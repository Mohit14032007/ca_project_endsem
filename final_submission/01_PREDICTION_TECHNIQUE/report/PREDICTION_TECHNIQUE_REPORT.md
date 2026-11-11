# Requirement 1: Training-Free MoE Expert Prediction Technique
**Comprehensive Evaluation of ST-MoE and CSTP Spatio-Temporal Predictors**

## 1. Mathematical Formulation

### ST-MoE Predictor
ST-MoE combines frequency-based historical prior with spatial co-activation tracking:
$$S_{\text{ST-MoE}}(e) = \alpha \cdot \text{Freq}(e) + (1 - \alpha) \cdot \sum_{e' \in \mathcal{E}_{\text{src}}} C(e', e)$$
Where:
- $\text{Freq}(e)$ is the historical access frequency of expert $e$.
- $C(e', e)$ is the co-activation count between source-stage expert $e'$ and target candidate $e$.
- $\alpha$ is the balancing hyperparameter (default $\alpha = 0.5$).
- State updates are guarded by 4-bit saturating confidence counters ($0 \le c \le 15$) with threshold $T = 2$.

### CSTP Predictor
CSTP formulates spatio-temporal expert transitions with Laplace smoothing:
$$S_{\text{CSTP}}(e) = \alpha \cdot P_{\text{spatial}}(e \mid \mathcal{E}_{\text{src}}) + (1 - \alpha) \cdot P_{\text{temporal}}(e \mid \mathcal{E}_{\text{prev}})$$
Where:
- Transition probabilities incorporate Laplace smoothing: $P(e \mid e') = \frac{N(e', e) + \lambda}{\sum_{e''} N(e', e'') + \lambda \cdot |\mathcal{E}|}$ with $\lambda = 0.01$.
- Supports multi-horizon transition modeling across $H \in \{1, 2, 3\}$.

## 2. Empirical Verification & Leakage Integrity
- Strict causal isolation is enforced in [`LivePredictionEngine::predict_for_stage()`](file:///home/mohman/ca_project/LLMSimulator/src/prefetch/live_prediction_engine.cpp#L120-L160):
  Target stage is set to $S_{\text{target}} = S_{\text{source}} + H$. The predictor observes only routing events from stage $\le S_{\text{source}}$ and has zero visibility into future routing choices.
- Unit tests [`test_phase2a_predictors`](file:///home/mohman/ca_project/LLMSimulator/eval/test_phase2a_predictors.cpp) (35 tests) and [`test_live_prediction`](file:///home/mohman/ca_project/LLMSimulator/eval/test_live_prediction.cpp) pass with 100% success rate.
