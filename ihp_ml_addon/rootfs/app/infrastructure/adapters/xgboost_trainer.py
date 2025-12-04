"""XGBoost trainer adapter.

Infrastructure adapter that implements IMLModelTrainer using XGBoost.
"""

import logging
import uuid
from datetime import datetime
from typing import Any

import numpy as np
import xgboost as xgb
from domain.interfaces import IMLModelTrainer, IModelStorage
from domain.value_objects import ModelInfo, TrainingData
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

_LOGGER = logging.getLogger(__name__)


class XGBoostTrainer(IMLModelTrainer):
    """XGBoost implementation of ML model trainer.

    This adapter uses XGBoost to train regression models for
    heating duration prediction.
    """

    BASE_FEATURE_NAMES = (
        "outdoor_temp",
        "indoor_temp",
        "target_temp",
        "temp_delta",
        "humidity",
        "hour_of_day",
        # "day_of_week",
        # "week_of_month",
        # "month",
        "minutes_since_last_cycle",
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

    async def train(
        self, training_data: TrainingData, device_id: str | None = None
    ) -> ModelInfo:
        """Train a new XGBoost model.

        Args:
            training_data: Training data containing features and labels
            device_id: Optional device/thermostat ID for model association

        Returns:
            ModelInfo with details about the trained model
        """
        # Generate model_id with device prefix if device_id is provided
        if device_id:
            model_id = f"xgb_{device_id}_{uuid.uuid4().hex[:8]}"
        else:
            model_id = f"xgb_{uuid.uuid4().hex[:8]}"
        _LOGGER.info("Training new XGBoost model: %s (device: %s)", model_id, device_id)

        # Discover feature names from the training data itself
        feature_names = self._discover_feature_names(training_data)
        _LOGGER.info(
            "Discovered %d features from training data (%d base + %d adjacent room features)",
            len(feature_names),
            len(self.BASE_FEATURE_NAMES),
            len(feature_names) - len(self.BASE_FEATURE_NAMES)
        )

        # Prepare features and labels
        X, y = self._prepare_data(training_data, feature_names)

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

        # Create model info with the specific feature contract
        model_info = ModelInfo(
            model_id=model_id,
            created_at=datetime.now(),
            training_samples=training_data.size,
            feature_names=feature_names,
            metrics=metrics,
            device_id=device_id,
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

    def _discover_feature_names(self, training_data: TrainingData) -> tuple[str, ...]:
        """Discover feature names from the training data.

        Inspects the training data to determine which adjacent room features
        are present and constructs the complete feature list.

        Args:
            training_data: Training data value object

        Returns:
            Tuple of feature names (base features + discovered adjacent room features)
        """
        # Start with base features
        feature_names = list(self.BASE_FEATURE_NAMES)
        
        # Collect all adjacent room features from the data
        adjacent_zones_features = set()
        
        for dp in training_data.data_points:
            if dp.adjacent_rooms:
                for zone_name in dp.adjacent_rooms:
                    # For each zone with data, add the 4 standard features
                    for feature_type in ["current_temp", "current_humidity", 
                                        "next_target_temp", "duration_until_change"]:
                        feature_name = f"{zone_name}_{feature_type}"
                        adjacent_zones_features.add(feature_name)
        
        # Sort adjacent features for consistency
        sorted_adjacent_features = sorted(adjacent_zones_features)
        feature_names.extend(sorted_adjacent_features)
        
        if sorted_adjacent_features:
            _LOGGER.info(
                "Discovered adjacent room features: %s",
                ", ".join(sorted_adjacent_features)
            )
        
        return tuple(feature_names)

    def _prepare_data(
        self,
        training_data: TrainingData,
        feature_names: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training data as numpy arrays.

        Args:
            training_data: Training data value object
            feature_names: Expected feature names for this model

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
                dp.minutes_since_last_cycle,
            ]
            
            if len(feature_names) > len(self.BASE_FEATURE_NAMES):
                adjacent_data = dp.adjacent_rooms or {}
                
                # Process each adjacent room feature expected in feature_names
                # Known feature suffixes (from adjacency_config.py)
                feature_suffixes = [
                    "current_temp", "current_humidity", 
                    "next_target_temp", "duration_until_change"
                ]
                
                for feature_name in feature_names[len(self.BASE_FEATURE_NAMES):]:
                    # Parse feature name by finding matching suffix
                    zone_name = None
                    feature_type = None
                    
                    for suffix in feature_suffixes:
                        if feature_name.endswith(f"_{suffix}"):
                            # Extract zone name by removing the suffix
                            zone_name = feature_name[:-len(suffix)-1]  # -1 for underscore
                            feature_type = suffix
                            break
                    
                    if zone_name and feature_type:
                        # Get value from adjacent_data or use default (0.0)
                        zone_data = adjacent_data.get(zone_name, {})
                        value = zone_data.get(feature_type, 0.0)
                        feature_row.append(value)
                    else:
                        # Fallback: append 0.0 if feature name doesn't match pattern
                        _LOGGER.warning(
                            "Unable to parse adjacent room feature: %s, using default 0.0",
                            feature_name
                        )
                        feature_row.append(0.0)
            
            features.append(feature_row)
            labels.append(dp.heating_duration_minutes)

        return np.array(features), np.array(labels)
