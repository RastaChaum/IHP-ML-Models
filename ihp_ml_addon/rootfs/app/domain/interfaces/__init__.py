"""Domain interfaces for ML operations.

Interfaces define contracts between the domain and infrastructure layers.
The domain depends on these abstractions, not on concrete implementations.
"""

from .ml_model_trainer import IMLModelTrainer
from .ml_model_predictor import IMLModelPredictor
from .model_storage import IModelStorage

__all__ = [
    "IMLModelTrainer",
    "IMLModelPredictor",
    "IModelStorage",
]
