"""Pytest fixtures for integration tests.

This module provides fixtures for testing the Flask API with mocked
Home Assistant responses.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from application.services import MLApplicationService
from infrastructure.adapters import (
    FileModelStorage,
    HomeAssistantHistoryReader,
    XGBoostPredictor,
    XGBoostTrainer,
)


@pytest.fixture
def temp_model_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for model storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_ha_history_data() -> list[list[Dict[str, Any]]]:
    """Generate mock Home Assistant history data for testing.
    
    Returns historical data in HA API format: list of lists, one per entity.
    Simulates 20 heating cycles with varying conditions.
    """
    base_time = datetime.now() - timedelta(days=7)
    
    indoor_temp_history = []
    outdoor_temp_history = []
    target_temp_history = []
    heating_history = []
    humidity_history = []
    
    # Simulate 20 heating cycles over 7 days with varying conditions
    for cycle in range(20):
        cycle_start = base_time + timedelta(hours=cycle * 8)
        outdoor_temp = 5.0 + (cycle % 5) * 2.0  # Vary outdoor temp
        initial_indoor = 15.0 + (cycle % 3) * 1.0  # Vary initial temp
        target = 20.0 + (cycle % 2) * 1.0  # Vary target temp
        duration_minutes = 30 + (cycle % 10) * 3  # Vary duration
        
        # Heating phase
        for i in range(duration_minutes + 1):
            timestamp = cycle_start + timedelta(minutes=i)
            progress = min(i / duration_minutes, 1.0)
            
            indoor_temp_history.append({
                "entity_id": "sensor.indoor_temp",
                "state": str(initial_indoor + (target - initial_indoor) * progress),
                "last_changed": timestamp.isoformat(),
            })
            outdoor_temp_history.append({
                "entity_id": "sensor.outdoor_temp",
                "state": str(outdoor_temp),
                "last_changed": timestamp.isoformat(),
            })
            target_temp_history.append({
                "entity_id": "sensor.target_temp",
                "state": str(target),
                "last_changed": timestamp.isoformat(),
            })
            heating_history.append({
                "entity_id": "climate.heating",
                "state": "heat" if i < duration_minutes else "off",
                "last_changed": timestamp.isoformat(),
            })
            humidity_history.append({
                "entity_id": "sensor.humidity",
                "state": str(50.0 + (cycle % 5) * 2.0),
                "last_changed": timestamp.isoformat(),
            })
        
        # Cool-down phase (30 minutes at target)
        for i in range(30):
            timestamp = cycle_start + timedelta(minutes=duration_minutes + 1 + i)
            
            indoor_temp_history.append({
                "entity_id": "sensor.indoor_temp",
                "state": str(target),
                "last_changed": timestamp.isoformat(),
            })
            outdoor_temp_history.append({
                "entity_id": "sensor.outdoor_temp",
                "state": str(outdoor_temp),
                "last_changed": timestamp.isoformat(),
            })
            target_temp_history.append({
                "entity_id": "sensor.target_temp",
                "state": str(target),
                "last_changed": timestamp.isoformat(),
            })
            heating_history.append({
                "entity_id": "climate.heating",
                "state": "off",
                "last_changed": timestamp.isoformat(),
            })
            humidity_history.append({
                "entity_id": "sensor.humidity",
                "state": str(50.0 + (cycle % 5) * 2.0),
                "last_changed": timestamp.isoformat(),
            })
    
    # HA API returns a list of lists, one per entity
    return [
        indoor_temp_history,
        outdoor_temp_history,
        target_temp_history,
        heating_history,
        humidity_history,
    ]


@pytest.fixture
def mock_ha_client(mock_ha_history_data: list[list[Dict[str, Any]]]) -> Mock:
    """Create a mock Home Assistant HTTP client.
    
    Returns a mock that simulates HA API responses.
    """
    mock_client = Mock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_ha_history_data
    mock_response.raise_for_status = Mock()
    mock_client.get.return_value = mock_response
    
    return mock_client


@pytest.fixture
def mock_ha_history_reader(mock_ha_history_data: list[list[Dict[str, Any]]]) -> HomeAssistantHistoryReader:
    """Create a HomeAssistantHistoryReader with mocked HTTP requests."""
    # Create a real reader instance
    reader = HomeAssistantHistoryReader(
        ha_url="http://localhost:8123",
        ha_token="fake_token_for_testing"
    )
    
    # Mock the requests.get call to return our mock data
    with patch('infrastructure.adapters.ha_history_reader.requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"message": "API running."}'
        mock_response.json.return_value = mock_ha_history_data
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        yield reader


@pytest.fixture
def ml_service(
    temp_model_dir: Path,
    mock_ha_history_reader: HomeAssistantHistoryReader
) -> MLApplicationService:
    """Create an MLApplicationService with mocked dependencies."""
    storage = FileModelStorage(temp_model_dir)
    trainer = XGBoostTrainer(storage)
    predictor = XGBoostPredictor(storage)
    
    service = MLApplicationService(
        trainer=trainer,
        predictor=predictor,
        storage=storage,
        ha_history_reader=mock_ha_history_reader,
    )
    
    return service


@pytest.fixture
def flask_app(ml_service: MLApplicationService, temp_model_dir: Path) -> Any:
    """Create a Flask test app with mocked services.
    
    This fixture patches the global ml_service in the server module.
    """
    # Patch the model path environment variable before importing server
    with patch.dict('os.environ', {
        'MODEL_PERSISTENCE_PATH': str(temp_model_dir),
        'SUPERVISOR_TOKEN': '',  # Disable HA integration for tests
    }):
        import infrastructure.api.server as server_module
        
        # Patch the global ml_service
        with patch.object(server_module, 'ml_service', ml_service):
            app = server_module.app
            app.config['TESTING'] = True
            yield app


@pytest.fixture
def client(flask_app: Any) -> Any:
    """Create a Flask test client."""
    return flask_app.test_client()


@pytest.fixture
def sample_training_data() -> Dict[str, Any]:
    """Sample training data payload for API requests."""
    base_time = datetime.now()
    
    return {
        "device_id": "climate.test_thermostat",
        "data_points": [
            {
                "outdoor_temp": 5.0 + i * 0.1,
                "indoor_temp": 18.0 + i * 0.2,
                "target_temp": 21.0,
                "humidity": 50.0,
                "hour_of_day": 8,
                "heating_duration_minutes": 30.0 - i * 0.5,
                "minutes_since_last_cycle": 60.0 + i * 10.0,  # Vary between cycles
                "timestamp": (base_time - timedelta(days=i)).isoformat(),
            }
            for i in range(50)
        ]
    }


@pytest.fixture
def sample_device_config() -> Dict[str, Any]:
    """Sample device configuration for training from HA history."""
    return {
        "device_id": "climate.test_thermostat",
        "indoor_temp_entity_id": "sensor.indoor_temp",
        "outdoor_temp_entity_id": "sensor.outdoor_temp",
        "target_temp_entity_id": "sensor.target_temp",
        "heating_state_entity_id": "climate.heating",
        "humidity_entity_id": "sensor.humidity",
        "history_days": 7,
    }


@pytest.fixture
def sample_prediction_request() -> Dict[str, Any]:
    """Sample prediction request payload."""
    return {
        "outdoor_temp": 5.0,
        "indoor_temp": 18.0,
        "target_temp": 21.0,
        "humidity": 50.0,
        "hour_of_day": 8,
        "minutes_since_last_cycle": 120.0,  # 2 hours since last cycle
    }
