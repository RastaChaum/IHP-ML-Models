"""Heating prediction service.

Domain service for orchestrating ML predictions for heating duration.
"""

from domain.interfaces import IMLModelPredictor, IMLModelTrainer, IModelStorage
from domain.value_objects import (
    ModelInfo,
    PredictionRequest,
    PredictionResult,
    TrainingData,
)


class HeatingPredictionService:
    """Service for heating duration predictions using ML models.

    This service orchestrates training and prediction operations
    through the provided interfaces.
    """

    def __init__(
        self,
        trainer: IMLModelTrainer,
        predictor: IMLModelPredictor,
        storage: IModelStorage,
    ) -> None:
        """Initialize the heating prediction service.

        Args:
            trainer: ML model trainer implementation
            predictor: ML model predictor implementation
            storage: Model storage implementation
        """
        self._trainer = trainer
        self._predictor = predictor
        self._storage = storage

    async def train_model(
        self, training_data: TrainingData, device_id: str | None = None
    ) -> ModelInfo:
        """Train a new heating prediction model.

        Args:
            training_data: Training data with features and labels
            device_id: Optional device ID for device-specific model

        Returns:
            Information about the trained model
        """
        return await self._trainer.train(training_data, device_id)

    async def predict_heating_duration(
        self,
        request: PredictionRequest,
    ) -> PredictionResult:
        """Predict heating duration for the given conditions.

        Args:
            request: Prediction request with environmental conditions

        Returns:
            Prediction result with estimated heating duration
        """
        return await self._predictor.predict(request)

    async def is_ready(self) -> bool:
        """Check if the service is ready to make predictions.

        Returns:
            True if a trained model is available
        """
        return await self._predictor.has_trained_model()

    async def get_model_info(self, model_id: str | None = None) -> ModelInfo | None:
        """Get information about a model.

        Args:
            model_id: Model ID or None for latest

        Returns:
            Model information or None if not found
        """
        if model_id is None:
            model_id = await self._storage.get_latest_model_id()
            if model_id is None:
                return None

        try:
            _, info = await self._storage.load_model(model_id)
            return info
        except Exception:
            return None
