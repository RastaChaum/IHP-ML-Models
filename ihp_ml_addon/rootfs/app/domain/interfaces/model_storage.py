"""Model storage interface.

Contract for persisting and retrieving ML models.
"""

from abc import ABC, abstractmethod
from typing import Any

from domain.value_objects import ModelInfo


class IModelStorage(ABC):
    """Contract for ML model persistence operations."""

    @abstractmethod
    async def save_model(
        self,
        model_id: str,
        model: Any,
        info: ModelInfo,
    ) -> None:
        """Save a trained model to storage.

        Args:
            model_id: Unique identifier for the model
            model: The trained model object
            info: Model metadata

        Raises:
            StorageError: If saving fails
        """
        pass

    @abstractmethod
    async def load_model(self, model_id: str) -> tuple[Any, ModelInfo]:
        """Load a model from storage.

        Args:
            model_id: Identifier of the model to load

        Returns:
            Tuple of (model object, model info)

        Raises:
            ModelNotFoundError: If model doesn't exist
            StorageError: If loading fails
        """
        pass

    @abstractmethod
    async def get_latest_model_id(self) -> str | None:
        """Get the ID of the most recently trained model.

        Returns:
            Model ID or None if no models exist
        """
        pass

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """List all available models.

        Returns:
            List of model information objects
        """
        pass

    @abstractmethod
    async def delete_model(self, model_id: str) -> None:
        """Delete a model from storage.

        Args:
            model_id: Identifier of the model to delete

        Raises:
            ModelNotFoundError: If model doesn't exist
            StorageError: If deletion fails
        """
        pass
