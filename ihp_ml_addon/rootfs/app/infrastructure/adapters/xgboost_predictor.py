"""XGBoost predictor adapter.

Infrastructure adapter that implements IMLModelPredictor using XGBoost.
"""

import logging
from datetime import datetime

import numpy as np
import xgboost as xgb
from domain.interfaces import IMLModelPredictor, IModelStorage
from domain.value_objects import PredictionRequest, PredictionResult

_LOGGER = logging.getLogger(__name__)


class XGBoostPredictor(IMLModelPredictor):
    """XGBoost implementation of ML model predictor.

    This adapter uses trained XGBoost models to make
    heating duration predictions with feature contract enforcement.
    """

    def __init__(self, storage: IModelStorage) -> None:
        """Initialize the XGBoost predictor.

        Args:
            storage: Model storage implementation
        """
        self._storage = storage
        # Cache includes: model_id, model, model_info, feature_names
        self._cached_model: tuple[str, xgb.XGBRegressor, object, tuple[str, ...]] | None = None

    async def predict(self, request: PredictionRequest) -> PredictionResult:
        """Make a prediction using XGBoost.

        Args:
            request: Prediction request with input features

        Returns:
            PredictionResult with the predicted value and metadata
        """
        # Determine which model to use
        model_id = request.model_id
        if model_id is None:
            # If device_id is provided, try to find a model for that device
            if request.device_id:
                model_id = await self._storage.get_latest_model_id_for_device(request.device_id)
            if model_id is None:
                model_id = await self._storage.get_latest_model_id()
            if model_id is None:
                raise ValueError("No trained model available")

        # Load model and feature contract (use cache if available)
        model, model_info, feature_names = await self._get_model(model_id)

        # Check for missing adjacent room features
        missing_features = self._check_missing_features(request, feature_names)
        feature_mismatch = len(missing_features) > 0

        if feature_mismatch:
            _LOGGER.warning(
                "Model %s expects adjacent room features but they are missing or incomplete. "
                "Missing features: %s. Prediction will use default values (0.0) for missing data.",
                model_id,
                ", ".join(missing_features)
            )

        # Prepare features according to the model's feature contract
        features = self._prepare_features(request, feature_names)

        # Make prediction
        prediction = model.predict(features)
        predicted_value = float(prediction[0])

        # Ensure non-negative prediction
        predicted_value = max(0, predicted_value)

        # Calculate confidence based on feature importance and data proximity
        confidence = self._calculate_confidence(model, request)

        # Generate reasoning
        reasoning = (
            f"Predicted {predicted_value:.1f} minutes to heat from "
            f"{request.indoor_temp:.1f}°C to {request.target_temp:.1f}°C "
            f"(outdoor: {request.outdoor_temp:.1f}°C, humidity: {request.humidity:.0f}%)"
        )

        return PredictionResult(
            predicted_duration_minutes=predicted_value,
            confidence=confidence,
            model_id=model_id,
            timestamp=datetime.now(),
            reasoning=reasoning,
            feature_mismatch=feature_mismatch,
            missing_features=missing_features if feature_mismatch else None,
        )

    async def has_trained_model(self) -> bool:
        """Check if a trained model is available.

        Returns:
            True if at least one trained model exists
        """
        latest_id = await self._storage.get_latest_model_id()
        return latest_id is not None

    async def _get_model(self, model_id: str) -> tuple[xgb.XGBRegressor, object, tuple[str, ...]]:
        """Get model and feature contract from cache or load from storage.

        Args:
            model_id: Model identifier

        Returns:
            Tuple of (model, model_info, feature_names)
        """
        # Check cache first - avoid loading from storage if cached
        if self._cached_model is not None and self._cached_model[0] == model_id:
            return self._cached_model[1], self._cached_model[2], self._cached_model[3]

        # Load full model and info from storage
        model, model_info = await self._storage.load_model(model_id)
        
        # Get feature names from model_info (feature contract)
        feature_names = model_info.feature_names
        
        # Cache model, info, and feature contract
        self._cached_model = (model_id, model, model_info, feature_names)
        
        _LOGGER.debug(
            "Loaded model %s with %d features: %s", 
            model_id, len(feature_names), feature_names
        )
        
        return model, model_info, feature_names

    def _check_missing_features(
        self,
        request: PredictionRequest,
        feature_names: tuple[str, ...],
    ) -> list[str]:
        """Check if expected adjacent room features are missing from the request.

        Args:
            request: Prediction request
            feature_names: Expected feature names from the model's feature contract

        Returns:
            List of missing feature names (empty if all features are present)
        """
        base_feature_count = 7  # Number of base features
        
        # If model doesn't expect adjacent room features, return empty list
        if len(feature_names) <= base_feature_count:
            return []
        
        missing_features = []
        adjacent_data = request.adjacent_rooms or {}
        
        # Known feature suffixes
        feature_suffixes = [
            "current_temp", "current_humidity",
            "next_target_temp", "duration_until_change"
        ]
        
        for feature_name in feature_names[base_feature_count:]:
            # Parse feature name to extract zone and feature type
            zone_name = None
            feature_type = None
            
            for suffix in feature_suffixes:
                if feature_name.endswith(f"_{suffix}"):
                    zone_name = feature_name[:-len(suffix)-1]
                    feature_type = suffix
                    break
            
            if zone_name and feature_type:
                # Check if this feature is present in the request
                zone_data = adjacent_data.get(zone_name, {})
                if not zone_data or feature_type not in zone_data:
                    missing_features.append(feature_name)
        
        return missing_features

    def _prepare_features(
        self, 
        request: PredictionRequest, 
        feature_names: tuple[str, ...]
    ) -> np.ndarray:
        """Prepare features array for prediction according to feature contract.

        Args:
            request: Prediction request
            feature_names: Expected feature names from the model's feature contract

        Returns:
            Numpy array with features aligned to the feature contract
        """
        temp_delta = request.target_temp - request.indoor_temp
        # Default to 0 if minutes_since_last_cycle not provided
        minutes_since_last_cycle = request.minutes_since_last_cycle or 0.0
        
        # Base features (must match BASE_FEATURE_NAMES order in trainer)
        base_features = [
            request.outdoor_temp,
            request.indoor_temp,
            request.target_temp,
            temp_delta,
            request.humidity,
            request.hour_of_day,
            minutes_since_last_cycle,
        ]
        
        features = base_features.copy()
        
        # If model expects additional features, add adjacent room data
        base_feature_count = 7  # Number of base features
        if len(feature_names) > base_feature_count:
            adjacent_data = request.adjacent_rooms or {}
            
            # Process each additional feature expected by the model
            # Known feature suffixes (from adjacency_config.py)
            feature_suffixes = [
                "current_temp", "current_humidity",
                "next_target_temp", "duration_until_change"
            ]
            
            for feature_name in feature_names[base_feature_count:]:
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
                    # Get value from adjacent_data or use default imputation (0.0)
                    zone_data = adjacent_data.get(zone_name, {})
                    value = zone_data.get(feature_type, 0.0)
                    features.append(value)
                else:
                    # Fallback: append 0.0 if feature name doesn't match pattern
                    _LOGGER.warning(
                        "Unable to parse adjacent room feature name: %s, using default 0.0",
                        feature_name
                    )
                    features.append(0.0)
            
            _LOGGER.debug(
                "Prepared %d features for prediction (%d base + %d adjacent)",
                len(features), base_feature_count, len(features) - base_feature_count
            )
        
        return np.array([features])

    def _calculate_confidence(
        self,
        model: xgb.XGBRegressor,
        request: PredictionRequest,
    ) -> float:
        """Calculate confidence score for the prediction.

        This is a simplified confidence calculation.
        In production, you might use uncertainty estimation methods.

        Args:
            model: XGBoost model
            request: Prediction request

        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from model
        base_confidence = 0.8

        # Adjust based on temperature delta reasonableness
        temp_delta = request.target_temp - request.indoor_temp
        if temp_delta <= 0:
            # Already at or above target, low confidence in prediction
            return 0.5
        elif temp_delta > 10:
            # Large temperature delta, slightly lower confidence
            base_confidence -= 0.1
        elif temp_delta < 2:
            # Small delta, might be noisy
            base_confidence -= 0.05

        # Adjust for extreme outdoor temperatures
        if request.outdoor_temp < -10 or request.outdoor_temp > 35:
            base_confidence -= 0.1

        return max(0.3, min(1.0, base_confidence))
