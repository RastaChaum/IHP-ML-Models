"""ML model predictor interface.

Contract for making predictions with ML models.
"""

from abc import ABC, abstractmethod

from domain.value_objects import PredictionRequest, PredictionResult


class IMLModelPredictor(ABC):
    """Contract for ML prediction operations."""

    @abstractmethod
    async def predict(self, request: PredictionRequest) -> PredictionResult:
        """Make a prediction using the specified or latest model.

        Args:
            request: Prediction request with input features

        Returns:
            PredictionResult with the predicted value and metadata

        Raises:
            ModelNotFoundError: If no model is available
            PredictionError: If prediction fails
        """
        pass

    @abstractmethod
    async def has_trained_model(self) -> bool:
        """Check if a trained model is available.

        Returns:
            True if at least one trained model exists
        """
        pass
