add_library(prefetch OBJECT)

target_sources(
    prefetch PRIVATE
    prefetch_config.cpp
    none_predictor.cpp
    st_moe_predictor.cpp
    cstp_predictor.cpp
    acsh_predictor.cpp
    control_predictors.cpp
    predictor_registry.cpp
    live_prediction_engine.cpp
    prefetch_candidate_queue.cpp
    ddr5_transfer_engine.cpp
    prefetch_metrics_collector.cpp
)
