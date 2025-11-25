"""ML Application Service.

Main application service that coordinates domain and infrastructure
for ML training and prediction use cases.
"""

import logging
from datetime import datetime, timedelta

from domain.interfaces import (
    IHomeAssistantHistoryReader,
    IMLModelPredictor,
    IMLModelTrainer,
    IModelStorage,
)
from domain.services import FakeDataGenerator, HeatingPredictionService
from domain.value_objects import (
    DeviceConfig,
    ModelInfo,
    PredictionRequest,
    PredictionResult,
    TrainingData,
)

_LOGGER = logging.getLogger(__name__)


class MLApplicationService:
    """Application service for ML operations.

    This service is the main entry point for all ML operations.
    It orchestrates domain services and infrastructure adapters.
    """

    def __init__(
        self,
        trainer: IMLModelTrainer,
        predictor: IMLModelPredictor,
        storage: IModelStorage,
        ha_history_reader: IHomeAssistantHistoryReader | None = None,
    ) -> None:
        """Initialize the ML application service.

        Args:
            trainer: ML model trainer implementation
            predictor: ML model predictor implementation
            storage: Model storage implementation
            ha_history_reader: Home Assistant history reader (optional)
        """
        self._prediction_service = HeatingPredictionService(
            trainer=trainer,
            predictor=predictor,
            storage=storage,
        )
        self._storage = storage
        self._fake_data_generator = FakeDataGenerator()
        self._ha_history_reader = ha_history_reader

    async def train_with_data(self, training_data: TrainingData) -> ModelInfo:
        """Train a model with provided training data.

        Args:
            training_data: Training data with features and labels

        Returns:
            Information about the trained model
        """
        _LOGGER.info(
            "Starting model training with %d samples",
            training_data.size,
        )
        model_info = await self._prediction_service.train_model(training_data)
        _LOGGER.info(
            "Model training completed: %s, metrics: %s",
            model_info.model_id,
            model_info.metrics,
        )
        return model_info

    async def train_with_fake_data(self, num_samples: int = 100) -> ModelInfo:
        """Train a model using generated fake data.

        This is useful for validating the technical architecture
        and initial testing.

        Args:
            num_samples: Number of fake samples to generate

        Returns:
            Information about the trained model
        """
        _LOGGER.info("Generating %d fake training samples", num_samples)
        training_data = self._fake_data_generator.generate(num_samples)
        return await self.train_with_data(training_data)

    async def train_with_device_config(self, device_config: DeviceConfig) -> ModelInfo:
        """Train a model using historical data from Home Assistant.

        This method fetches historical sensor data for the specified device
        from Home Assistant and uses it to train a new model.

        Args:
            device_config: Device configuration with sensor entity IDs

        Returns:
            Information about the trained model

        Raises:
            RuntimeError: If Home Assistant history reader is not configured
            ConnectionError: If unable to connect to Home Assistant
            ValueError: If no valid training data found
        """
        if self._ha_history_reader is None:
            raise RuntimeError(
                "Home Assistant history reader not configured. "
                "Cannot train with device configuration."
            )

        _LOGGER.info(
            "Training model for device %s using %d days of history",
            device_config.device_id,
            device_config.history_days,
        )

        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=device_config.history_days)

        # Fetch training data from Home Assistant
        training_data = await self._ha_history_reader.fetch_training_data(
            indoor_temp_entity_id=device_config.indoor_temp_entity_id,
            outdoor_temp_entity_id=device_config.outdoor_temp_entity_id,
            target_temp_entity_id=device_config.target_temp_entity_id,
            heating_state_entity_id=device_config.heating_state_entity_id,
            humidity_entity_id=device_config.humidity_entity_id,
            start_time=start_time,
            end_time=end_time,
        )

        _LOGGER.info(
            "Fetched %d training samples from Home Assistant for device %s",
            training_data.size,
            device_config.device_id,
        )

        return await self.train_with_data(training_data)

    async def is_ha_available(self) -> bool:
        """Check if Home Assistant integration is available.

        Returns:
            True if Home Assistant history reader is configured and reachable
        """
        if self._ha_history_reader is None:
            return False
        return await self._ha_history_reader.is_available()

    async def predict(self, request: PredictionRequest) -> PredictionResult:
        """Make a heating duration prediction.

        Args:
            request: Prediction request with environmental conditions

        Returns:
            Prediction result with estimated heating duration
        """
        _LOGGER.debug(
            "Predicting heating duration: indoor=%.1f, target=%.1f, outdoor=%.1f",
            request.indoor_temp,
            request.target_temp,
            request.outdoor_temp,
        )
        result = await self._prediction_service.predict_heating_duration(request)
        _LOGGER.debug(
            "Prediction result: %.1f minutes (confidence: %.2f)",
            result.predicted_duration_minutes,
            result.confidence,
        )
        return result

    async def is_ready(self) -> bool:
        """Check if the service is ready to make predictions.

        Returns:
            True if a trained model is available
        """
        return await self._prediction_service.is_ready()

    async def get_status(self) -> dict:
        """Get the current status of the ML service.

        Returns:
            Dictionary with status information
        """
        ready = await self.is_ready()
        models = await self._storage.list_models()
        ha_available = await self.is_ha_available()

        status = {
            "ready": ready,
            "model_count": len(models),
            "ha_integration_available": ha_available,
            "timestamp": datetime.now().isoformat(),
        }

        if ready:
            latest_id = await self._storage.get_latest_model_id()
            if latest_id:
                model_info = await self._prediction_service.get_model_info(latest_id)
                if model_info:
                    status["latest_model"] = {
                        "id": model_info.model_id,
                        "created_at": model_info.created_at.isoformat(),
                        "training_samples": model_info.training_samples,
                        "metrics": model_info.metrics,
                    }

        return status

    async def get_model_info(self, model_id: str | None = None) -> ModelInfo | None:
        """Get information about a specific model.

        Args:
            model_id: Model ID or None for latest

        Returns:
            Model information or None if not found
        """
        return await self._prediction_service.get_model_info(model_id)

    async def list_models(self) -> list[ModelInfo]:
        """List all available models.

        Returns:
            List of model information objects
        """
        return await self._storage.list_models()

    async def delete_model(self, model_id: str) -> None:
        """Delete a model.

        Args:
            model_id: Model ID to delete
        """
        _LOGGER.info("Deleting model: %s", model_id)
        await self._storage.delete_model(model_id)
