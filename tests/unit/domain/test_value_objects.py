"""Tests for domain value objects.

These tests verify that value objects are immutable and properly validated.
"""

from datetime import datetime

import pytest
from domain.value_objects import (
    DeviceConfig,
    ModelInfo,
    PredictionRequest,
    PredictionResult,
    TrainingData,
    TrainingDataPoint,
)


class TestDeviceConfig:
    """Tests for DeviceConfig value object."""

    def test_valid_device_config_creation(self) -> None:
        """Test creating a valid device configuration."""
        config = DeviceConfig(
            device_id="ihp_salon",
            indoor_temp_entity_id="sensor.salon_temperature",
            outdoor_temp_entity_id="sensor.outdoor_temperature",
            target_temp_entity_id="climate.vtherm_salon",
            heating_state_entity_id="climate.vtherm_salon",
            humidity_entity_id="sensor.salon_humidity",
            history_days=30,
        )
        assert config.device_id == "ihp_salon"
        assert config.indoor_temp_entity_id == "sensor.salon_temperature"
        assert config.outdoor_temp_entity_id == "sensor.outdoor_temperature"
        assert config.target_temp_entity_id == "climate.vtherm_salon"
        assert config.heating_state_entity_id == "climate.vtherm_salon"
        assert config.humidity_entity_id == "sensor.salon_humidity"
        assert config.history_days == 30

    def test_device_config_without_humidity(self) -> None:
        """Test creating a device configuration without humidity sensor."""
        config = DeviceConfig(
            device_id="ihp_chambre",
            indoor_temp_entity_id="sensor.chambre_temperature",
            outdoor_temp_entity_id="sensor.outdoor_temperature",
            target_temp_entity_id="climate.vtherm_chambre",
            heating_state_entity_id="climate.vtherm_chambre",
        )
        assert config.humidity_entity_id is None
        assert config.history_days == 30  # Default value

    def test_device_config_empty_device_id_raises_error(self) -> None:
        """Test that empty device_id raises ValueError."""
        with pytest.raises(ValueError, match="device_id cannot be empty"):
            DeviceConfig(
                device_id="",
                indoor_temp_entity_id="sensor.temp",
                outdoor_temp_entity_id="sensor.outdoor",
                target_temp_entity_id="climate.vtherm",
                heating_state_entity_id="climate.vtherm",
            )

    def test_device_config_empty_indoor_temp_raises_error(self) -> None:
        """Test that empty indoor_temp_entity_id raises ValueError."""
        with pytest.raises(ValueError, match="indoor_temp_entity_id cannot be empty"):
            DeviceConfig(
                device_id="ihp_test",
                indoor_temp_entity_id="",
                outdoor_temp_entity_id="sensor.outdoor",
                target_temp_entity_id="climate.vtherm",
                heating_state_entity_id="climate.vtherm",
            )

    def test_device_config_history_days_too_low_raises_error(self) -> None:
        """Test that history_days < 1 raises ValueError."""
        with pytest.raises(ValueError, match="history_days must be at least 1"):
            DeviceConfig(
                device_id="ihp_test",
                indoor_temp_entity_id="sensor.temp",
                outdoor_temp_entity_id="sensor.outdoor",
                target_temp_entity_id="climate.vtherm",
                heating_state_entity_id="climate.vtherm",
                history_days=0,
            )

    def test_device_config_history_days_too_high_raises_error(self) -> None:
        """Test that history_days > 365 raises ValueError."""
        with pytest.raises(ValueError, match="history_days must be at most 365"):
            DeviceConfig(
                device_id="ihp_test",
                indoor_temp_entity_id="sensor.temp",
                outdoor_temp_entity_id="sensor.outdoor",
                target_temp_entity_id="climate.vtherm",
                heating_state_entity_id="climate.vtherm",
                history_days=400,
            )

    def test_device_config_is_immutable(self) -> None:
        """Test that device configuration is immutable."""
        config = DeviceConfig(
            device_id="ihp_test",
            indoor_temp_entity_id="sensor.temp",
            outdoor_temp_entity_id="sensor.outdoor",
            target_temp_entity_id="climate.vtherm",
            heating_state_entity_id="climate.vtherm",
        )
        with pytest.raises(AttributeError):
            config.device_id = "new_id"  # type: ignore


class TestTrainingDataPoint:
    """Tests for TrainingDataPoint value object."""

    def test_valid_training_data_point_creation(self) -> None:
        """Test creating a valid training data point."""
        dp = TrainingDataPoint(
            outdoor_temp=5.0,
            indoor_temp=18.0,
            target_temp=21.0,
            humidity=65.0,
            hour_of_day=7,
            day_of_week=1,
            week_of_month=2,
            month=11,
            heating_duration_minutes=45.0,
            timestamp=datetime.now(),
        )
        assert dp.outdoor_temp == 5.0
        assert dp.indoor_temp == 18.0
        assert dp.target_temp == 21.0
        assert dp.humidity == 65.0
        assert dp.hour_of_day == 7
        assert dp.day_of_week == 1
        assert dp.week_of_month == 2
        assert dp.month == 11
        assert dp.heating_duration_minutes == 45.0

    def test_training_data_point_is_immutable(self) -> None:
        """Test that TrainingDataPoint is immutable (frozen dataclass)."""
        dp = TrainingDataPoint(
            outdoor_temp=5.0,
            indoor_temp=18.0,
            target_temp=21.0,
            humidity=65.0,
            hour_of_day=7,
            day_of_week=1,
            week_of_month=2,
            month=11,
            heating_duration_minutes=45.0,
            timestamp=datetime.now(),
        )
        with pytest.raises(AttributeError):
            dp.outdoor_temp = 10.0  # type: ignore

    def test_invalid_outdoor_temp_raises_error(self) -> None:
        """Test that invalid outdoor temperature raises ValueError."""
        with pytest.raises(ValueError, match="outdoor_temp"):
            TrainingDataPoint(
                outdoor_temp=-60.0,  # Invalid: below -50
                indoor_temp=18.0,
                target_temp=21.0,
                humidity=65.0,
                hour_of_day=7,
                day_of_week=1,
                week_of_month=2,
                month=11,
                heating_duration_minutes=45.0,
                timestamp=datetime.now(),
            )

    def test_invalid_humidity_raises_error(self) -> None:
        """Test that invalid humidity raises ValueError."""
        with pytest.raises(ValueError, match="humidity"):
            TrainingDataPoint(
                outdoor_temp=5.0,
                indoor_temp=18.0,
                target_temp=21.0,
                humidity=150.0,  # Invalid: above 100
                hour_of_day=7,
                day_of_week=1,
                week_of_month=2,
                month=11,
                heating_duration_minutes=45.0,
                timestamp=datetime.now(),
            )

    def test_negative_heating_duration_raises_error(self) -> None:
        """Test that negative heating duration raises ValueError."""
        with pytest.raises(ValueError, match="heating_duration_minutes"):
            TrainingDataPoint(
                outdoor_temp=5.0,
                indoor_temp=18.0,
                target_temp=21.0,
                humidity=65.0,
                hour_of_day=7,
                day_of_week=1,
                week_of_month=2,
                month=11,
                heating_duration_minutes=-10.0,  # Invalid: negative
                timestamp=datetime.now(),
            )

    def test_invalid_week_of_month_raises_error(self) -> None:
        """Test that invalid week_of_month raises ValueError."""
        with pytest.raises(ValueError, match="week_of_month"):
            TrainingDataPoint(
                outdoor_temp=5.0,
                indoor_temp=18.0,
                target_temp=21.0,
                humidity=65.0,
                hour_of_day=7,
                day_of_week=1,
                week_of_month=0,  # Invalid: below 1
                month=11,
                heating_duration_minutes=45.0,
                timestamp=datetime.now(),
            )

    def test_invalid_month_raises_error(self) -> None:
        """Test that invalid month raises ValueError."""
        with pytest.raises(ValueError, match="month"):
            TrainingDataPoint(
                outdoor_temp=5.0,
                indoor_temp=18.0,
                target_temp=21.0,
                humidity=65.0,
                hour_of_day=7,
                day_of_week=1,
                week_of_month=2,
                month=13,  # Invalid: above 12
                heating_duration_minutes=45.0,
                timestamp=datetime.now(),
            )


class TestTrainingData:
    """Tests for TrainingData value object."""

    def test_valid_training_data_creation(self) -> None:
        """Test creating valid TrainingData."""
        dp = TrainingDataPoint(
            outdoor_temp=5.0,
            indoor_temp=18.0,
            target_temp=21.0,
            humidity=65.0,
            hour_of_day=7,
            day_of_week=1,
            week_of_month=2,
            month=11,
            heating_duration_minutes=45.0,
            timestamp=datetime.now(),
        )
        data = TrainingData.from_sequence([dp])
        assert data.size == 1
        assert dp in data.data_points

    def test_empty_training_data_raises_error(self) -> None:
        """Test that empty training data raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            TrainingData.from_sequence([])


class TestPredictionRequest:
    """Tests for PredictionRequest value object."""

    def test_valid_prediction_request_creation(self) -> None:
        """Test creating a valid prediction request."""
        req = PredictionRequest(
            outdoor_temp=5.0,
            indoor_temp=18.0,
            target_temp=21.0,
            humidity=65.0,
            hour_of_day=7,
            day_of_week=1,
            week_of_month=2,
            month=11,
        )
        assert req.outdoor_temp == 5.0
        assert req.temp_delta == 3.0
        assert req.week_of_month == 2
        assert req.month == 11

    def test_prediction_request_is_immutable(self) -> None:
        """Test that PredictionRequest is immutable."""
        req = PredictionRequest(
            outdoor_temp=5.0,
            indoor_temp=18.0,
            target_temp=21.0,
            humidity=65.0,
            hour_of_day=7,
            day_of_week=1,
            week_of_month=2,
            month=11,
        )
        with pytest.raises(AttributeError):
            req.outdoor_temp = 10.0  # type: ignore

    def test_prediction_request_with_device_id(self) -> None:
        """Test creating a prediction request with device_id."""
        req = PredictionRequest(
            outdoor_temp=5.0,
            indoor_temp=18.0,
            target_temp=21.0,
            humidity=65.0,
            hour_of_day=7,
            day_of_week=1,
            week_of_month=2,
            month=11,
            device_id="ihp_salon",
        )
        assert req.device_id == "ihp_salon"
        assert req.model_id is None


class TestPredictionResult:
    """Tests for PredictionResult value object."""

    def test_valid_prediction_result_creation(self) -> None:
        """Test creating a valid prediction result."""
        result = PredictionResult(
            predicted_duration_minutes=45.0,
            confidence=0.85,
            model_id="test_model_123",
            timestamp=datetime.now(),
            reasoning="Test reasoning",
        )
        assert result.predicted_duration_minutes == 45.0
        assert result.confidence == 0.85

    def test_invalid_confidence_raises_error(self) -> None:
        """Test that invalid confidence raises ValueError."""
        with pytest.raises(ValueError, match="confidence"):
            PredictionResult(
                predicted_duration_minutes=45.0,
                confidence=1.5,  # Invalid: above 1.0
                model_id="test_model_123",
                timestamp=datetime.now(),
                reasoning="Test reasoning",
            )


class TestModelInfo:
    """Tests for ModelInfo value object."""

    def test_valid_model_info_creation(self) -> None:
        """Test creating valid ModelInfo."""
        info = ModelInfo(
            model_id="test_model_123",
            created_at=datetime.now(),
            training_samples=100,
            feature_names=("outdoor_temp", "indoor_temp", "target_temp"),
            metrics={"rmse": 5.0, "r2": 0.85},
        )
        assert info.model_id == "test_model_123"
        assert info.training_samples == 100

    def test_empty_model_id_raises_error(self) -> None:
        """Test that empty model_id raises ValueError."""
        with pytest.raises(ValueError, match="model_id"):
            ModelInfo(
                model_id="",
                created_at=datetime.now(),
                training_samples=100,
                feature_names=("outdoor_temp",),
                metrics={"rmse": 5.0},
            )
