"""
Predictor package for MoE expert prediction and prefetching.
Provides base interface, paper-faithful ST-MoE, legacy ST-MoE, CSTP, and baselines.
"""

from src.predictor.base_predictor import (
    BasePredictor,
    STMoELegacyWrapper,
    CSTPWrapper,
    RandomPredictor,
    FrequencyPredictor
)
from src.predictor.st_moe_paper import STMoEPaperPredictor
from src.predictor.st_moe_predictor import STMoEPredictor
from src.predictor.cstp_predictor import CSTPPredictor
from src.predictor.trace_parser import NormalizedRoutingRecord, RoutingTraceParser
from src.predictor.evaluation import (
    EvaluationHarness,
    PredictionOpportunity,
    PredictionResultRecord
)

__version__ = "2.0.0"

__all__ = [
    "BasePredictor",
    "STMoEPaperPredictor",
    "STMoEPredictor",
    "CSTPPredictor",
    "STMoELegacyWrapper",
    "CSTPWrapper",
    "RandomPredictor",
    "FrequencyPredictor",
    "RoutingTraceParser",
    "NormalizedRoutingRecord",
    "EvaluationHarness",
    "PredictionOpportunity",
    "PredictionResultRecord"
]
