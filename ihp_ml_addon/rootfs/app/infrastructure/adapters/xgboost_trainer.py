"""XGBoost trainer adapter.

Infrastructure adapter that implements IMLModelTrainer using XGBoost.
"""

import uuid
import logging
from datetime import datetime
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from domain.interfaces import IMLModelTrainer, IModelStorage
from domain.value_objects import TrainingData, ModelInfo

_LOGGER = logging.getLogger(__name__)


class XGBoostTrainer(IMLModelTrainer):
    """XGBoost implementation of ML model trainer.

    This adapter uses XGBoost to train regression models for
    heating duration prediction.
    """

    FEATURE_NAMES = (
        "outdoor_temp",
        "indoor_temp",
        "target_temp",
        "temp_delta",
        "humidity",
        "hour_of_day",
        "day_of_week",
    )

    def __init__(
        self,
        storage: IModelStorage,
        hyperparams: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the XGBoost trainer.

        Args:
            storage: Model storage implementation
            hyperparams: XGBoost hyperparameters (optional)
        """
        self._storage = storage
        self._hyperparams = hyperparams or self._default_hyperparams()

    @staticmethod
    def _default_hyperparams() -> dict[str, Any]:
        """Get default XGBoost hyperparameters."""
        return {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }

    async def train(self, training_data: TrainingData) -> ModelInfo:
        """Train a new XGBoost model.

        Args:
            training_data: Training data containing features and labels

        Returns:
            ModelInfo with details about the trained model
        """
        model_id = f"xgb_{uuid.uuid4().hex[:8]}"
        _LOGGER.info("Training new XGBoost model: %s", model_id)

        # Prepare features and labels
        X, y = self._prepare_data(training_data)

        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the model
        model = xgb.XGBRegressor(**self._hyperparams)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Calculate metrics
        y_pred = model.predict(X_val)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_val, y_pred))),
            "r2": float(r2_score(y_val, y_pred)),
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
        }

        _LOGGER.info("Model %s trained with metrics: %s", model_id, metrics)

        # Create model info
        model_info = ModelInfo(
            model_id=model_id,
            created_at=datetime.now(),
            training_samples=training_data.size,
            feature_names=self.FEATURE_NAMES,
            metrics=metrics,
        )

        # Save model
        await self._storage.save_model(model_id, model, model_info)

        return model_info

    async def retrain(
        self,
        model_id: str,
        training_data: TrainingData,
    ) -> ModelInfo:
        """Retrain an existing model with new data.

        For XGBoost, we train a new model with the new data.

        Args:
            model_id: Identifier of the model to retrain
            training_data: New training data

        Returns:
            ModelInfo with details about the retrained model
        """
        _LOGGER.info("Retraining model %s with %d new samples", model_id, training_data.size)

        # Load existing model to verify it exists
        await self._storage.load_model(model_id)

        # Train new model with same ID prefix
        return await self.train(training_data)

    def _prepare_data(
        self,
        training_data: TrainingData,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training data as numpy arrays.

        Args:
            training_data: Training data value object

        Returns:
            Tuple of (features array, labels array)
        """
        features = []
        labels = []

        for dp in training_data.data_points:
            temp_delta = dp.target_temp - dp.indoor_temp
            feature_row = [
                dp.outdoor_temp,
                dp.indoor_temp,
                dp.target_temp,
                temp_delta,
                dp.humidity,
                dp.hour_of_day,
                dp.day_of_week,
            ]
            features.append(feature_row)
            labels.append(dp.heating_duration_minutes)

        return np.array(features), np.array(labels)
