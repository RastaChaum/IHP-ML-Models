"""Value objects for ML domain.

Value objects are immutable data carriers that represent domain concepts.
They have no identity and are compared by their attributes.
"""

from .training_data import TrainingData, TrainingDataPoint
from .prediction_request import PredictionRequest
from .prediction_result import PredictionResult
from .model_info import ModelInfo

__all__ = [
    "TrainingData",
    "TrainingDataPoint",
    "PredictionRequest",
    "PredictionResult",
    "ModelInfo",
]
