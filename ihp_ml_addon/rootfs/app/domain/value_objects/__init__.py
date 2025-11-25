"""Value objects for ML domain.

Value objects are immutable data carriers that represent domain concepts.
They have no identity and are compared by their attributes.
"""

from .device_config import DeviceConfig
from .model_info import ModelInfo
from .prediction_request import PredictionRequest
from .prediction_result import PredictionResult
from .training_data import TrainingData, TrainingDataPoint

__all__ = [
    "DeviceConfig",
    "ModelInfo",
    "PredictionRequest",
    "PredictionResult",
    "TrainingData",
    "TrainingDataPoint",
]
