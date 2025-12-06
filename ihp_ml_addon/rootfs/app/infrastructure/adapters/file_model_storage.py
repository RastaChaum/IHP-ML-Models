"""File-based model storage adapter.

Infrastructure adapter that implements IModelStorage using file system.
"""

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

from domain.interfaces import IModelStorage
from domain.value_objects import ModelInfo

_LOGGER = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when a model is not found in storage."""

    pass


class StorageError(Exception):
    """Raised when a storage operation fails."""

    pass


class FileModelStorage(IModelStorage):
    """File-based implementation of model storage.

    This adapter stores models and metadata on the file system.
    """

    MODEL_FILE_SUFFIX = ".pkl"
    METADATA_FILE_SUFFIX = ".json"
    FEATURES_FILE_SUFFIX = "_features.json"
    INDEX_FILE_NAME = "models_index.json"

    def __init__(self, base_path: str | Path) -> None:
        """Initialize file-based storage.

        Args:
            base_path: Directory path for storing models
        """
        self._base_path = Path(base_path)
        self._ensure_directory_exists()

    def _ensure_directory_exists(self) -> None:
        """Create storage directory if it doesn't exist."""
        self._base_path.mkdir(parents=True, exist_ok=True)

    async def save_model(
        self,
        model_id: str,
        model: Any,
        info: ModelInfo,
    ) -> None:
        """Save a trained model to file storage.

        Args:
            model_id: Unique identifier for the model
            model: The trained model object
            info: Model metadata
        """
        try:
            # Save model object
            model_path = self._base_path / f"{model_id}{self.MODEL_FILE_SUFFIX}"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Save metadata
            metadata_path = self._base_path / f"{model_id}{self.METADATA_FILE_SUFFIX}"
            metadata = {
                "model_id": info.model_id,
                "created_at": info.created_at.isoformat(),
                "training_samples": info.training_samples,
                "feature_names": list(info.feature_names),
                "metrics": info.metrics,
                "version": info.version,
                "device_id": info.device_id,
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Save feature contract (separate file for easy access during inference)
            features_path = self._base_path / f"{model_id}{self.FEATURES_FILE_SUFFIX}"
            features_contract = {
                "model_id": info.model_id,
                "device_id": info.device_id,
                "feature_names": list(info.feature_names),
                "created_at": info.created_at.isoformat(),
            }
            with open(features_path, "w") as f:
                json.dump(features_contract, f, indent=2)

            # Update index
            await self._update_index(model_id, info.created_at, info.device_id)

            _LOGGER.info("Model saved: %s (device: %s)", model_id, info.device_id)

        except (OSError, pickle.PickleError) as e:
            raise StorageError(f"Failed to save model {model_id}: {e}") from e

    async def load_model(self, model_id: str) -> tuple[Any, ModelInfo]:
        """Load a model from file storage.

        Args:
            model_id: Identifier of the model to load

        Returns:
            Tuple of (model object, model info)
        """
        model_path = self._base_path / f"{model_id}{self.MODEL_FILE_SUFFIX}"
        metadata_path = self._base_path / f"{model_id}{self.METADATA_FILE_SUFFIX}"

        if not model_path.exists():
            raise ModelNotFoundError(f"Model not found: {model_id}")

        try:
            # Load model object
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Load metadata
            with open(metadata_path) as f:
                metadata = json.load(f)

            model_info = ModelInfo(
                model_id=metadata["model_id"],
                created_at=datetime.fromisoformat(metadata["created_at"]),
                training_samples=metadata["training_samples"],
                feature_names=tuple(metadata["feature_names"]),
                metrics=metadata["metrics"],
                version=metadata.get("version", "1.0.0"),
                device_id=metadata.get("device_id"),
            )

            _LOGGER.debug("Model loaded: %s", model_id)
            return model, model_info

        except (OSError, pickle.UnpicklingError, json.JSONDecodeError, KeyError) as e:
            raise StorageError(f"Failed to load model {model_id}: {e}") from e

    async def get_latest_model_id(self) -> str | None:
        """Get the ID of the most recently trained model.

        Returns:
            Model ID or None if no models exist
        """
        index = await self._load_index()
        if not index:
            return None

        # Sort by creation time and return the latest
        sorted_models = sorted(
            index.items(),
            key=lambda x: x[1].get("created_at", ""),
            reverse=True,
        )
        return sorted_models[0][0] if sorted_models else None

    async def get_latest_model_id_for_device(self, device_id: str) -> str | None:
        """Get the ID of the most recently trained model for a specific device.

        Args:
            device_id: Device/thermostat identifier

        Returns:
            Model ID or None if no models exist for the device
        """
        index = await self._load_index()
        if not index:
            return None

        # Filter models by device_id and sort by creation time
        device_models = [
            (model_id, data)
            for model_id, data in index.items()
            if data.get("device_id") == device_id
        ]

        if not device_models:
            return None

        sorted_models = sorted(
            device_models,
            key=lambda x: x[1].get("created_at", ""),
            reverse=True,
        )
        return sorted_models[0][0] if sorted_models else None

    async def list_models(self) -> list[ModelInfo]:
        """List all available models.

        Returns:
            List of model information objects
        """
        models = []
        index = await self._load_index()

        for model_id in index:
            try:
                _, info = await self.load_model(model_id)
                models.append(info)
            except (ModelNotFoundError, StorageError) as e:
                _LOGGER.warning("Failed to load model %s: %s", model_id, e)

        return sorted(models, key=lambda x: x.created_at, reverse=True)

    async def list_models_for_device(self, device_id: str) -> list[ModelInfo]:
        """List all available models for a specific device.

        Args:
            device_id: Device/thermostat identifier

        Returns:
            List of model information objects for the device
        """
        all_models = await self.list_models()
        return [m for m in all_models if m.device_id == device_id]

    async def load_feature_contract(self, model_id: str) -> tuple[str, ...]:
        """Load only the feature contract for a model without loading the model itself.

        Args:
            model_id: Identifier of the model

        Returns:
            Tuple of feature names

        Raises:
            ModelNotFoundError: If the model or feature contract is not found
            StorageError: If loading fails
        """
        features_path = self._base_path / f"{model_id}{self.FEATURES_FILE_SUFFIX}"
        
        if not features_path.exists():
            # Fallback to loading from metadata if features file doesn't exist
            metadata_path = self._base_path / f"{model_id}{self.METADATA_FILE_SUFFIX}"
            if not metadata_path.exists():
                raise ModelNotFoundError(f"Model not found: {model_id}")
            
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                return tuple(metadata["feature_names"])
            except (OSError, json.JSONDecodeError, KeyError) as e:
                raise StorageError(f"Failed to load feature contract for {model_id}: {e}") from e
        
        try:
            with open(features_path) as f:
                features_data = json.load(f)
            return tuple(features_data["feature_names"])
        except (OSError, json.JSONDecodeError, KeyError) as e:
            raise StorageError(f"Failed to load feature contract for {model_id}: {e}") from e

    async def delete_model(self, model_id: str) -> None:
        """Delete a model from file storage.

        Args:
            model_id: Identifier of the model to delete
        """
        model_path = self._base_path / f"{model_id}{self.MODEL_FILE_SUFFIX}"
        metadata_path = self._base_path / f"{model_id}{self.METADATA_FILE_SUFFIX}"
        features_path = self._base_path / f"{model_id}{self.FEATURES_FILE_SUFFIX}"

        if not model_path.exists():
            raise ModelNotFoundError(f"Model not found: {model_id}")

        try:
            os.remove(model_path)
            if metadata_path.exists():
                os.remove(metadata_path)
            if features_path.exists():
                os.remove(features_path)

            # Update index
            await self._remove_from_index(model_id)

            _LOGGER.info("Model deleted: %s", model_id)

        except OSError as e:
            raise StorageError(f"Failed to delete model {model_id}: {e}") from e

    async def _load_index(self) -> dict[str, dict[str, str]]:
        """Load the models index.

        Returns:
            Dictionary mapping model_id to metadata (created_at, device_id)
        """
        index_path = self._base_path / self.INDEX_FILE_NAME

        if not index_path.exists():
            return {}

        try:
            with open(index_path) as f:
                raw_index = json.load(f)
                # Handle backward compatibility with old format (string timestamps)
                converted_index = {}
                for model_id, value in raw_index.items():
                    if isinstance(value, str):
                        # Old format: just timestamp string
                        converted_index[model_id] = {"created_at": value, "device_id": None}
                    else:
                        # New format: dict with created_at and device_id
                        converted_index[model_id] = value
                return converted_index
        except (OSError, json.JSONDecodeError):
            return {}

    async def _update_index(
        self, model_id: str, created_at: datetime, device_id: str | None = None
    ) -> None:
        """Update the models index with a new model.

        Args:
            model_id: Model identifier
            created_at: Model creation timestamp
            device_id: Device identifier (optional)
        """
        index = await self._load_index()
        index[model_id] = {
            "created_at": created_at.isoformat(),
            "device_id": device_id,
        }
        await self._save_index(index)

    async def _remove_from_index(self, model_id: str) -> None:
        """Remove a model from the index.

        Args:
            model_id: Model identifier to remove
        """
        index = await self._load_index()
        if model_id in index:
            del index[model_id]
            await self._save_index(index)

    async def _save_index(self, index: dict[str, dict[str, str]]) -> None:
        """Save the models index.

        Args:
            index: Dictionary mapping model_id to metadata dict
        """
        index_path = self._base_path / self.INDEX_FILE_NAME
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
