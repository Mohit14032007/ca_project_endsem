# Requirement 1 Implementation Reference
- `LLMSimulator/src/prefetch/st_moe_predictor.cpp`: ST-MoE frequency and spatial co-activation tracking
- `LLMSimulator/src/prefetch/cstp_predictor.cpp`: CSTP Laplace-smoothed spatio-temporal transition Markov model
- `LLMSimulator/src/prefetch/live_prediction_engine.cpp`: Forward lookahead prediction engine
- `LLMSimulator/eval/test_phase2a_predictors.cpp`: 35 unit tests validating scoring, counters, and boundaries
