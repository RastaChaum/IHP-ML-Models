"""Tests for multi-room feature integration in training and prediction."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from domain.value_objects import (
    PredictionRequest,
    TrainingData,
    TrainingDataPoint,
)


@pytest.fixture
def sample_adjacency_config():
    """Create a sample adjacency configuration file."""
    config_data = {
        "zones": {
            "living_room": {"adjacent_zones": ["kitchen"]},
            "kitchen": {"adjacent_zones": ["living_room"]},
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name

    yield config_path

    Path(config_path).unlink()


@pytest.fixture
def sample_training_data_with_adjacent():
    """Create sample training data with adjacent room information."""
    data_points = []
    for i in range(20):
        adjacent_rooms = {
            "kitchen": {
                "current_temp": 20.0 + i * 0.1,
                "current_humidity": 50.0 + i * 0.5,
                "next_target_temp": 21.0,
                "duration_until_change": 30.0,
            }
        }
        data_points.append(
            TrainingDataPoint(
                outdoor_temp=5.0 + i * 0.5,
                indoor_temp=18.0 + i * 0.2,
                target_temp=21.0,
                humidity=60.0 + i * 0.3,
                hour_of_day=i % 24,
                heating_duration_minutes=30.0 + i,
                timestamp=datetime.now(),
                minutes_since_last_cycle=60.0,
                adjacent_rooms=adjacent_rooms,
            )
        )
    return TrainingData.from_sequence(data_points)


@pytest.mark.asyncio
async def test_trainer_with_adjacency_config(
    tmp_path, sample_adjacency_config, sample_training_data_with_adjacent
):
    """Test that trainer uses adjacency config to create dynamic feature lists."""
    from infrastructure.adapters import (
        AdjacencyConfig,
        FileModelStorage,
        XGBoostTrainer,
    )

    storage = FileModelStorage(tmp_path)
    adjacency_config = AdjacencyConfig(sample_adjacency_config)
    trainer = XGBoostTrainer(storage, adjacency_config=adjacency_config)

    # Train model for living_room (has adjacent kitchen)
    model_info = await trainer.train(sample_training_data_with_adjacent, device_id="living_room")

    # Check that feature names include base features + adjacent room features
    assert len(model_info.feature_names) == 7 + 4  # 7 base + 4 for kitchen
    assert "outdoor_temp" in model_info.feature_names
    assert "kitchen_current_temp" in model_info.feature_names
    assert "kitchen_current_humidity" in model_info.feature_names
    assert "kitchen_next_target_temp" in model_info.feature_names
    assert "kitchen_duration_until_change" in model_info.feature_names

    # Verify feature contract was saved
    feature_contract = await storage.load_feature_contract(model_info.model_id)
    assert feature_contract == model_info.feature_names


@pytest.mark.asyncio
async def test_trainer_without_adjacency(tmp_path, sample_training_data_with_adjacent):
    """Test that trainer works without adjacency config (base features only)."""
    from infrastructure.adapters import FileModelStorage, XGBoostTrainer

    storage = FileModelStorage(tmp_path)
    trainer = XGBoostTrainer(storage)

    # Train model without device_id
    model_info = await trainer.train(sample_training_data_with_adjacent)

    # Should only have base features
    assert len(model_info.feature_names) == 7  # Base features only
    assert "outdoor_temp" in model_info.feature_names
    assert "kitchen_current_temp" not in model_info.feature_names


@pytest.mark.asyncio
async def test_predictor_with_adjacent_rooms(
    tmp_path, sample_adjacency_config, sample_training_data_with_adjacent
):
    """Test prediction with adjacent room data."""
    from infrastructure.adapters import (
        AdjacencyConfig,
        FileModelStorage,
        XGBoostPredictor,
        XGBoostTrainer,
    )

    storage = FileModelStorage(tmp_path)
    adjacency_config = AdjacencyConfig(sample_adjacency_config)
    trainer = XGBoostTrainer(storage, adjacency_config=adjacency_config)
    predictor = XGBoostPredictor(storage)

    # Train a model with adjacent room features
    model_info = await trainer.train(sample_training_data_with_adjacent, device_id="living_room")

    # Make prediction with adjacent room data
    adjacent_rooms = {
        "kitchen": {
            "current_temp": 20.5,
            "current_humidity": 55.0,
            "next_target_temp": 21.0,
            "duration_until_change": 30.0,
        }
    }

    request = PredictionRequest(
        outdoor_temp=5.0,
        indoor_temp=18.0,
        target_temp=21.0,
        humidity=60.0,
        hour_of_day=10,
        minutes_since_last_cycle=60.0,
        device_id="living_room",
        adjacent_rooms=adjacent_rooms,
    )

    result = await predictor.predict(request)

    # Should get a valid prediction
    assert result.predicted_duration_minutes >= 0
    assert result.confidence > 0
    assert result.model_id == model_info.model_id


@pytest.mark.asyncio
async def test_predictor_missing_adjacent_data_uses_imputation(
    tmp_path, sample_adjacency_config, sample_training_data_with_adjacent
):
    """Test that predictor imputes missing adjacent room data with 0.0."""
    from infrastructure.adapters import (
        AdjacencyConfig,
        FileModelStorage,
        XGBoostPredictor,
        XGBoostTrainer,
    )

    storage = FileModelStorage(tmp_path)
    adjacency_config = AdjacencyConfig(sample_adjacency_config)
    trainer = XGBoostTrainer(storage, adjacency_config=adjacency_config)
    predictor = XGBoostPredictor(storage)

    # Train a model with adjacent room features
    model_info = await trainer.train(sample_training_data_with_adjacent, device_id="living_room")

    # Make prediction WITHOUT adjacent room data (should impute with 0.0)
    request = PredictionRequest(
        outdoor_temp=5.0,
        indoor_temp=18.0,
        target_temp=21.0,
        humidity=60.0,
        hour_of_day=10,
        minutes_since_last_cycle=60.0,
        device_id="living_room",
        # No adjacent_rooms provided
    )

    result = await predictor.predict(request)

    # Should still get a valid prediction (with imputed 0.0 values)
    assert result.predicted_duration_minutes >= 0
    assert result.confidence > 0
    assert result.model_id == model_info.model_id


@pytest.mark.asyncio
async def test_predictor_partial_adjacent_data_uses_imputation(
    tmp_path, sample_adjacency_config, sample_training_data_with_adjacent
):
    """Test that predictor imputes missing fields in adjacent room data."""
    from infrastructure.adapters import (
        AdjacencyConfig,
        FileModelStorage,
        XGBoostPredictor,
        XGBoostTrainer,
    )

    storage = FileModelStorage(tmp_path)
    adjacency_config = AdjacencyConfig(sample_adjacency_config)
    trainer = XGBoostTrainer(storage, adjacency_config=adjacency_config)
    predictor = XGBoostPredictor(storage)

    # Train a model with adjacent room features
    model_info = await trainer.train(sample_training_data_with_adjacent, device_id="living_room")

    # Make prediction with partial adjacent room data
    adjacent_rooms = {
        "kitchen": {
            "current_temp": 20.5,
            # Missing current_humidity, next_target_temp, duration_until_change
        }
    }

    request = PredictionRequest(
        outdoor_temp=5.0,
        indoor_temp=18.0,
        target_temp=21.0,
        humidity=60.0,
        hour_of_day=10,
        minutes_since_last_cycle=60.0,
        device_id="living_room",
        adjacent_rooms=adjacent_rooms,
    )

    result = await predictor.predict(request)

    # Should still get a valid prediction (with imputed 0.0 for missing fields)
    assert result.predicted_duration_minutes >= 0
    assert result.confidence > 0
    assert result.model_id == model_info.model_id


def test_training_data_point_with_adjacent_rooms():
    """Test that TrainingDataPoint accepts adjacent_rooms parameter."""
    adjacent_rooms = {
        "kitchen": {
            "current_temp": 20.5,
            "current_humidity": 55.0,
            "next_target_temp": 21.0,
            "duration_until_change": 30.0,
        }
    }

    data_point = TrainingDataPoint(
        outdoor_temp=5.0,
        indoor_temp=18.0,
        target_temp=21.0,
        humidity=60.0,
        hour_of_day=10,
        heating_duration_minutes=30.0,
        timestamp=datetime.now(),
        adjacent_rooms=adjacent_rooms,
    )

    assert data_point.adjacent_rooms == adjacent_rooms


def test_prediction_request_with_adjacent_rooms():
    """Test that PredictionRequest accepts adjacent_rooms parameter."""
    adjacent_rooms = {
        "kitchen": {
            "current_temp": 20.5,
            "current_humidity": 55.0,
            "next_target_temp": 21.0,
            "duration_until_change": 30.0,
        }
    }

    request = PredictionRequest(
        outdoor_temp=5.0,
        indoor_temp=18.0,
        target_temp=21.0,
        humidity=60.0,
        hour_of_day=10,
        device_id="living_room",
        adjacent_rooms=adjacent_rooms,
    )

    assert request.adjacent_rooms == adjacent_rooms
