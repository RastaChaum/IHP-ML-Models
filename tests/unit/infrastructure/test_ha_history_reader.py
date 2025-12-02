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
            mock_response.text = "OK"
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
            mock_response.text = "OK"
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
            mock_response.text = "Unauthorized"
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


class TestCycleSplitting:
    """Tests for heating cycle splitting functionality."""

    def test_extract_heating_cycles_without_splitting(self):
        """Test that cycles are extracted without splitting when parameter is None."""
        reader = HomeAssistantHistoryReader(
            ha_url='http://test',
            ha_token='test_token'
        )

        # Simulate a 90-minute heating cycle (heating on at 08:00, off at 09:30)
        # Temperature goes from 18°C to 20°C
        history_data = {
            "climate.thermostat": [
                {
                    "last_changed": "2024-11-25T08:00:00+00:00",
                    "state": "heat",
                    "attributes": {
                        "hvac_action": "heating",
                        "current_temperature": 18.0,
                        "temperature": 20.0,
                    }
                },
                {
                    "last_changed": "2024-11-25T09:30:00+00:00",
                    "state": "off",
                    "attributes": {
                        "hvac_action": "idle",
                        "current_temperature": 20.0,
                        "temperature": 20.0,
                    }
                },
            ],
            "sensor.outdoor": [
                {
                    "last_changed": "2024-11-25T08:00:00+00:00",
                    "state": "5.0",
                },
            ],
        }

        data_points = reader._extract_heating_cycles(
            history_data,
            indoor_temp_entity_id="climate.thermostat",
            outdoor_temp_entity_id="sensor.outdoor",
            target_temp_entity_id="climate.thermostat",
            heating_state_entity_id="climate.thermostat",
            humidity_entity_id=None,
            cycle_split_duration_minutes=None,  # No splitting
        )

        # Should create 1 data point (the full 90-minute cycle)
        assert len(data_points) == 1
        assert data_points[0].heating_duration_minutes == 90.0
        assert data_points[0].indoor_temp == 18.0
        assert data_points[0].target_temp == 20.0

    def test_extract_heating_cycles_with_splitting(self):
        """Test that long cycles are split into smaller sub-cycles."""
        reader = HomeAssistantHistoryReader(
            ha_url='http://test',
            ha_token='test_token'
        )

        # Simulate a 180-minute (3 hour) heating cycle
        # Temperature goes from 18°C to 21°C
        history_data = {
            "climate.thermostat": [
                {
                    "last_changed": "2024-11-25T08:00:00+00:00",
                    "state": "heat",
                    "attributes": {
                        "hvac_action": "heating",
                        "current_temperature": 18.0,
                        "temperature": 21.0,
                    }
                },
                {
                    "last_changed": "2024-11-25T11:00:00+00:00",
                    "state": "off",
                    "attributes": {
                        "hvac_action": "idle",
                        "current_temperature": 21.0,
                        "temperature": 21.0,
                    }
                },
            ],
            "sensor.outdoor": [
                {
                    "last_changed": "2024-11-25T08:00:00+00:00",
                    "state": "5.0",
                },
            ],
        }

        data_points = reader._extract_heating_cycles(
            history_data,
            indoor_temp_entity_id="climate.thermostat",
            outdoor_temp_entity_id="sensor.outdoor",
            target_temp_entity_id="climate.thermostat",
            heating_state_entity_id="climate.thermostat",
            humidity_entity_id=None,
            cycle_split_duration_minutes=60,  # Split into 60-minute sub-cycles
        )

        # Should create 3 data points (3 x 60-minute sub-cycles)
        assert len(data_points) == 3

        # First sub-cycle: 18°C to 19°C (1°C per 60 min = 3°C/180min)
        assert data_points[0].heating_duration_minutes == 60.0
        assert data_points[0].indoor_temp == 18.0
        assert data_points[0].target_temp == pytest.approx(19.0, abs=0.1)

        # Second sub-cycle: 19°C to 20°C
        assert data_points[1].heating_duration_minutes == 60.0
        assert data_points[1].indoor_temp == pytest.approx(19.0, abs=0.1)
        assert data_points[1].target_temp == pytest.approx(20.0, abs=0.1)

        # Third sub-cycle: 20°C to 21°C
        assert data_points[2].heating_duration_minutes == 60.0
        assert data_points[2].indoor_temp == pytest.approx(20.0, abs=0.1)
        assert data_points[2].target_temp == pytest.approx(21.0, abs=0.1)

    def test_extract_heating_cycles_with_splitting_and_remaining(self):
        """Test cycle splitting with remaining time included."""
        reader = HomeAssistantHistoryReader(
            ha_url='http://test',
            ha_token='test_token'
        )

        # Simulate a 130-minute heating cycle
        # Should split into 2 x 60-min sub-cycles + 1 x 10-min remaining
        history_data = {
            "climate.thermostat": [
                {
                    "last_changed": "2024-11-25T08:00:00+00:00",
                    "state": "heat",
                    "attributes": {
                        "hvac_action": "heating",
                        "current_temperature": 18.0,
                        "temperature": 21.0,
                    }
                },
                {
                    "last_changed": "2024-11-25T10:10:00+00:00",
                    "state": "off",
                    "attributes": {
                        "hvac_action": "idle",
                        "current_temperature": 20.5,
                        "temperature": 21.0,
                    }
                },
            ],
            "sensor.outdoor": [
                {
                    "last_changed": "2024-11-25T08:00:00+00:00",
                    "state": "5.0",
                },
            ],
        }

        data_points = reader._extract_heating_cycles(
            history_data,
            indoor_temp_entity_id="climate.thermostat",
            outdoor_temp_entity_id="sensor.outdoor",
            target_temp_entity_id="climate.thermostat",
            heating_state_entity_id="climate.thermostat",
            humidity_entity_id=None,
            cycle_split_duration_minutes=60,
        )

        # Should create 3 data points (2 x 60-min + 1 x 10-min remaining)
        assert len(data_points) == 3

        # Verify the first two are 60 minutes
        assert data_points[0].heating_duration_minutes == 60.0
        assert data_points[1].heating_duration_minutes == 60.0

        # Verify the remaining sub-cycle is 10 minutes
        assert data_points[2].heating_duration_minutes == pytest.approx(10.0, abs=0.5)

    def test_short_cycle_not_split(self):
        """Test that short cycles (< split duration) are not split."""
        reader = HomeAssistantHistoryReader(
            ha_url='http://test',
            ha_token='test_token'
        )

        # Simulate a 45-minute heating cycle (shorter than 60-minute split duration)
        history_data = {
            "climate.thermostat": [
                {
                    "last_changed": "2024-11-25T08:00:00+00:00",
                    "state": "heat",
                    "attributes": {
                        "hvac_action": "heating",
                        "current_temperature": 18.0,
                        "temperature": 20.0,
                    }
                },
                {
                    "last_changed": "2024-11-25T08:45:00+00:00",
                    "state": "off",
                    "attributes": {
                        "hvac_action": "idle",
                        "current_temperature": 20.0,
                        "temperature": 20.0,
                    }
                },
            ],
            "sensor.outdoor": [
                {
                    "last_changed": "2024-11-25T08:00:00+00:00",
                    "state": "5.0",
                },
            ],
        }

        data_points = reader._extract_heating_cycles(
            history_data,
            indoor_temp_entity_id="climate.thermostat",
            outdoor_temp_entity_id="sensor.outdoor",
            target_temp_entity_id="climate.thermostat",
            heating_state_entity_id="climate.thermostat",
            humidity_entity_id=None,
            cycle_split_duration_minutes=60,  # Split duration is 60 minutes
        )

        # Should create 1 data point (45 minutes < 60 minutes, no split)
        assert len(data_points) == 1
        assert data_points[0].heating_duration_minutes == 45.0
