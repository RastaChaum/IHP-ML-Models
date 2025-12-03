"""ML model trainer interface.

Contract for training ML models.
"""

from abc import ABC, abstractmethod

from domain.value_objects import ModelInfo, TrainingData


class IMLModelTrainer(ABC):
    """Contract for ML model training operations."""

    @abstractmethod
    async def train(
        self, training_data: TrainingData, device_id: str | None = None
    ) -> ModelInfo:
        """Train a new model with the provided data.

        Args:
            training_data: Training data containing features and labels
            device_id: Optional device/thermostat ID for model association

        Returns:
            ModelInfo with details about the trained model

        Raises:
            TrainingError: If training fails
        """
        pass

    @abstractmethod
    async def retrain(
        self,
        model_id: str,
        training_data: TrainingData,
    ) -> ModelInfo:
        """Retrain an existing model with new data.

        Args:
            model_id: Identifier of the model to retrain
            training_data: New training data

        Returns:
            ModelInfo with details about the retrained model

        Raises:
            ModelNotFoundError: If the model doesn't exist
            TrainingError: If training fails
        """
        pass
