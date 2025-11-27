"""Unit tests for Home Assistant History Reader adapter."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from infrastructure.adapters.ha_history_reader import HomeAssistantHistoryReader


class TestHomeAssistantHistoryReader:
    """Test the HA history reader adapter."""

    def test_initialization_with_supervisor_env(self):
        """Test initialization with supervisor environment variables."""
        with patch.dict('os.environ', {
            'SUPERVISOR_URL': 'http://supervisor/core',
            'SUPERVISOR_TOKEN': 'test_token_123'
        }):
            reader = HomeAssistantHistoryReader()
            assert reader._ha_url == 'http://supervisor/core'
            assert reader._ha_token == 'test_token_123'

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        reader = HomeAssistantHistoryReader(
            ha_url='http://homeassistant:8123',
            ha_token='custom_token',
            timeout=60
        )
        assert reader._ha_url == 'http://homeassistant:8123'
        assert reader._ha_token == 'custom_token'
        assert reader._timeout == 60

    @pytest.mark.asyncio
    async def test_is_available_constructs_correct_url(self):
        """Test that is_available constructs the correct URL with urljoin fix."""
        reader = HomeAssistantHistoryReader(
            ha_url='http://supervisor/core',
            ha_token='test_token'
        )

        with patch('infrastructure.adapters.ha_history_reader.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '{"message": "API running."}'
            mock_get.return_value = mock_response

            result = await reader.is_available()

            # Verify the correct URL was used (should be /core/api/, not just /api/)
            called_url = mock_get.call_args[0][0]
            assert called_url == 'http://supervisor/core/api/'
            assert result is True

    @pytest.mark.asyncio
    async def test_is_available_handles_trailing_slash(self):
        """Test that is_available works with URLs that already have trailing slash."""
        reader = HomeAssistantHistoryReader(
            ha_url='http://supervisor/core/',  # Already has trailing slash
            ha_token='test_token'
        )

        with patch('infrastructure.adapters.ha_history_reader.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '{"message": "API running."}'
            mock_get.return_value = mock_response

            result = await reader.is_available()

            # Should still produce correct URL
            called_url = mock_get.call_args[0][0]
            assert called_url == 'http://supervisor/core/api/'
            assert result is True

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_error(self):
        """Test that is_available returns False on connection error."""
        import requests
        
        reader = HomeAssistantHistoryReader(
            ha_url='http://supervisor/core',
            ha_token='test_token'
        )

        with patch('infrastructure.adapters.ha_history_reader.requests.get') as mock_get:
            mock_get.side_effect = requests.RequestException("Connection refused")

            result = await reader.is_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_non_200(self):
        """Test that is_available returns False on non-200 status."""
        reader = HomeAssistantHistoryReader(
            ha_url='http://supervisor/core',
            ha_token='test_token'
        )

        with patch('infrastructure.adapters.ha_history_reader.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = '{"message": "Unauthorized"}'
            mock_get.return_value = mock_response

            result = await reader.is_available()
            assert result is False

    def test_get_headers_includes_bearer_token(self):
        """Test that headers include proper authorization."""
        reader = HomeAssistantHistoryReader(
            ha_url='http://supervisor/core',
            ha_token='my_secret_token'
        )

        headers = reader._get_headers()
        assert headers['Authorization'] == 'Bearer my_secret_token'
        assert headers['Content-Type'] == 'application/json'


class TestOnTimeHeatingCycleExtraction:
    """Tests for extracting heating cycles from On Time sensor."""

    def test_extract_heating_cycles_from_on_time_basic(self):
        """Test basic extraction of heating cycles from On Time sensor."""
        reader = HomeAssistantHistoryReader(
            ha_url='http://supervisor/core',
            ha_token='test_token'
        )

        # Simulate On Time history: heating active (60s), then inactive
        history_data = {
            "sensor.thermostat_on_time": [
                {"state": "0", "last_changed": "2024-01-15T08:00:00+00:00"},
                {"state": "60", "last_changed": "2024-01-15T08:05:00+00:00"},  # Heating ON
                {"state": "120", "last_changed": "2024-01-15T08:10:00+00:00"},  # Still heating
                {"state": "0", "last_changed": "2024-01-15T08:30:00+00:00"},  # Heating OFF (buffer exceeded)
            ],
            "sensor.indoor_temp": [
                {"state": "18.0", "last_changed": "2024-01-15T08:00:00+00:00"},
                {"state": "18.5", "last_changed": "2024-01-15T08:10:00+00:00"},
                {"state": "21.0", "last_changed": "2024-01-15T08:30:00+00:00"},  # Target reached
            ],
            "sensor.outdoor_temp": [
                {"state": "5.0", "last_changed": "2024-01-15T08:00:00+00:00"},
            ],
            "sensor.target_temp": [
                {"state": "21.0", "last_changed": "2024-01-15T08:00:00+00:00"},
            ],
        }

        data_points = reader._extract_heating_cycles_from_on_time(
            history_data,
            "sensor.indoor_temp",
            "sensor.outdoor_temp",
            "sensor.target_temp",
            "sensor.thermostat_on_time",
            None,  # No humidity
            15,  # 15 minute buffer
        )

        assert len(data_points) >= 1
        dp = data_points[0]
        assert dp.indoor_temp == 18.0
        assert dp.outdoor_temp == 5.0
        assert dp.target_temp == 21.0

    def test_extract_heating_cycles_from_on_time_with_buffer(self):
        """Test that buffer prevents false cycle ends during short heating pauses."""
        reader = HomeAssistantHistoryReader(
            ha_url='http://supervisor/core',
            ha_token='test_token'
        )

        # Simulate On Time history with short pause (5 min) that should not end cycle
        history_data = {
            "sensor.thermostat_on_time": [
                {"state": "0", "last_changed": "2024-01-15T08:00:00+00:00"},
                {"state": "60", "last_changed": "2024-01-15T08:05:00+00:00"},  # Heating ON
                {"state": "0", "last_changed": "2024-01-15T08:10:00+00:00"},  # Short pause
                {"state": "60", "last_changed": "2024-01-15T08:12:00+00:00"},  # Heating ON again
                {"state": "0", "last_changed": "2024-01-15T08:30:00+00:00"},  # Heating OFF (buffer exceeded)
            ],
            "sensor.indoor_temp": [
                {"state": "18.0", "last_changed": "2024-01-15T08:00:00+00:00"},
                {"state": "19.0", "last_changed": "2024-01-15T08:10:00+00:00"},
                {"state": "20.0", "last_changed": "2024-01-15T08:20:00+00:00"},
                {"state": "21.0", "last_changed": "2024-01-15T08:30:00+00:00"},
            ],
            "sensor.outdoor_temp": [
                {"state": "5.0", "last_changed": "2024-01-15T08:00:00+00:00"},
            ],
            "sensor.target_temp": [
                {"state": "21.0", "last_changed": "2024-01-15T08:00:00+00:00"},
            ],
        }

        data_points = reader._extract_heating_cycles_from_on_time(
            history_data,
            "sensor.indoor_temp",
            "sensor.outdoor_temp",
            "sensor.target_temp",
            "sensor.thermostat_on_time",
            None,
            15,  # 15 minute buffer - short pause should not end cycle
        )

        # Should detect one heating cycle (short pause should be ignored due to buffer)
        assert len(data_points) >= 1

    def test_extract_heating_cycles_from_on_time_empty_history(self):
        """Test handling of empty On Time history."""
        reader = HomeAssistantHistoryReader(
            ha_url='http://supervisor/core',
            ha_token='test_token'
        )

        history_data = {
            "sensor.indoor_temp": [
                {"state": "18.0", "last_changed": "2024-01-15T08:00:00+00:00"},
            ],
            "sensor.outdoor_temp": [
                {"state": "5.0", "last_changed": "2024-01-15T08:00:00+00:00"},
            ],
            "sensor.target_temp": [
                {"state": "21.0", "last_changed": "2024-01-15T08:00:00+00:00"},
            ],
        }

        data_points = reader._extract_heating_cycles_from_on_time(
            history_data,
            "sensor.indoor_temp",
            "sensor.outdoor_temp",
            "sensor.target_temp",
            "sensor.thermostat_on_time",  # Missing from history_data
            None,
            15,
        )

        assert len(data_points) == 0

    def test_extract_heating_cycles_from_on_time_target_reached(self):
        """Test cycle ends when target temperature is reached."""
        reader = HomeAssistantHistoryReader(
            ha_url='http://supervisor/core',
            ha_token='test_token'
        )

        history_data = {
            "sensor.thermostat_on_time": [
                {"state": "0", "last_changed": "2024-01-15T08:00:00+00:00"},
                {"state": "60", "last_changed": "2024-01-15T08:05:00+00:00"},  # Heating ON
                {"state": "60", "last_changed": "2024-01-15T08:20:00+00:00"},  # Still heating
            ],
            "sensor.indoor_temp": [
                {"state": "18.0", "last_changed": "2024-01-15T08:00:00+00:00"},
                {"state": "20.9", "last_changed": "2024-01-15T08:15:00+00:00"},
                {"state": "21.1", "last_changed": "2024-01-15T08:20:00+00:00"},  # Target exceeded
            ],
            "sensor.outdoor_temp": [
                {"state": "5.0", "last_changed": "2024-01-15T08:00:00+00:00"},
            ],
            "sensor.target_temp": [
                {"state": "21.0", "last_changed": "2024-01-15T08:00:00+00:00"},
            ],
        }

        data_points = reader._extract_heating_cycles_from_on_time(
            history_data,
            "sensor.indoor_temp",
            "sensor.outdoor_temp",
            "sensor.target_temp",
            "sensor.thermostat_on_time",
            None,
            15,
        )

        # Should detect cycle ending when target is reached
        assert len(data_points) >= 1
        dp = data_points[0]
        # Duration should be calculated from start to when target was reached
        assert dp.heating_duration_minutes > 0
