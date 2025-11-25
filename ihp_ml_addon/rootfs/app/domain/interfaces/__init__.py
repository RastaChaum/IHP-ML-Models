"""Domain interfaces for ML operations.

Interfaces define contracts between the domain and infrastructure layers.
The domain depends on these abstractions, not on concrete implementations.
"""

from .ha_history_reader import IHomeAssistantHistoryReader
from .ml_model_predictor import IMLModelPredictor
from .ml_model_trainer import IMLModelTrainer
from .model_storage import IModelStorage

__all__ = [
    "IHomeAssistantHistoryReader",
    "IMLModelPredictor",
    "IMLModelTrainer",
    "IModelStorage",
]
