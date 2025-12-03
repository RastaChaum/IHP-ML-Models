"""Domain services for ML operations.

Services contain pure business logic and operate on value objects.
"""

from .fake_data_generator import FakeDataGenerator
from .heating_prediction_service import HeatingPredictionService

__all__ = [
    "HeatingPredictionService",
    "FakeDataGenerator",
]
