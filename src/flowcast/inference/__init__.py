"""Validated, lineage-aware inference services for frozen FlowCast models."""

from flowcast.inference.predictor import PredictionResult, Predictor
from flowcast.inference.schemas import PredictionRequest

__all__ = ["PredictionRequest", "PredictionResult", "Predictor"]
