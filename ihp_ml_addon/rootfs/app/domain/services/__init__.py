"""Domain services for ML operations.

Services contain pure business logic and operate on value objects.
"""

from .heating_prediction_service import HeatingPredictionService
from .fake_data_generator import FakeDataGenerator

__all__ = [
    "HeatingPredictionService",
    "FakeDataGenerator",
]
