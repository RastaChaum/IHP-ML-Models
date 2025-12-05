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


class TestRLExperienceExtraction:
    """Tests for RL experience extraction from historical data."""

    @pytest.mark.asyncio
    async def test_fetch_rl_experiences_requires_reward_calculator(self):
        """Test that fetch_rl_experiences raises error without reward calculator."""
        from domain.value_objects import TrainingRequest
        from datetime import datetime, timedelta

        reader = HomeAssistantHistoryReader(
            ha_url='http://test',
            ha_token='test_token',
            reward_calculator=None,  # No reward calculator
        )

        training_request = TrainingRequest(
            device_id="zone.living_room",
            indoor_temp_entity_id="sensor.indoor_temp",
            target_temp_entity_id="sensor.target_temp",
            heating_state_entity_id="binary_sensor.heating",
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now(),
        )

        with pytest.raises(ValueError, match="Reward calculator is required"):
            await reader.fetch_rl_experiences(training_request)

    @pytest.mark.asyncio
    async def test_fetch_rl_experiences_with_minimal_data(self):
        """Test RL experience extraction with minimal required data."""
        from domain.value_objects import TrainingRequest, RewardConfig
        from domain.services import HeatingRewardCalculator
        from datetime import datetime, timedelta, timezone

        # Create a mock reward calculator
        reward_calculator = HeatingRewardCalculator(config=RewardConfig())

        reader = HomeAssistantHistoryReader(
            ha_url='http://test',
            ha_token='test_token',
            reward_calculator=reward_calculator,
        )

        training_request = TrainingRequest(
            device_id="zone.living_room",
            indoor_temp_entity_id="sensor.indoor_temp",
            target_temp_entity_id="sensor.target_temp",
            heating_state_entity_id="binary_sensor.heating",
            start_time=datetime(2024, 11, 25, 8, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2024, 11, 25, 9, 0, 0, tzinfo=timezone.utc),
        )

        # Mock history data with heating cycle
        mock_history_data = {
            "sensor.indoor_temp": [
                {
                    "last_changed": "2024-11-25T08:00:00+00:00",
                    "state": "18.0",
                },
                {
                    "last_changed": "2024-11-25T08:15:00+00:00",
                    "state": "18.5",
                },
                {
                    "last_changed": "2024-11-25T08:30:00+00:00",
                    "state": "19.0",
                },
                {
                    "last_changed": "2024-11-25T08:45:00+00:00",
                    "state": "19.5",
                },
                {
                    "last_changed": "2024-11-25T09:00:00+00:00",
                    "state": "20.0",
                },
            ],
            "sensor.target_temp": [
                {
                    "last_changed": "2024-11-25T08:00:00+00:00",
                    "state": "20.0",
                },
            ],
            "binary_sensor.heating": [
                {
                    "last_changed": "2024-11-25T08:00:00+00:00",
                    "state": "on",
                },
                {
                    "last_changed": "2024-11-25T08:45:00+00:00",
                    "state": "off",
                },
            ],
        }

        # Mock the _fetch_history method
        with patch.object(reader, '_fetch_history', return_value=mock_history_data):
            experiences = await reader.fetch_rl_experiences(training_request)

            # Should have created some experiences
            assert len(experiences) > 0

            # Verify experience structure
            exp = experiences[0]
            assert exp.state is not None
            assert exp.action is not None
            assert exp.next_state is not None
            assert isinstance(exp.reward, float)
            assert isinstance(exp.done, bool)

            # Verify state and next_state are for the same device
            assert exp.state.device_id == exp.next_state.device_id == training_request.device_id

    @pytest.mark.asyncio
    async def test_construct_observation_with_all_fields(self):
        """Test observation construction with all optional fields."""
        from domain.value_objects import TrainingRequest
        from datetime import datetime, timezone

        reader = HomeAssistantHistoryReader(
            ha_url='http://test',
            ha_token='test_token',
        )

        training_request = TrainingRequest(
            device_id="zone.living_room",
            indoor_temp_entity_id="sensor.indoor_temp",
            target_temp_entity_id="sensor.target_temp",
            heating_state_entity_id="binary_sensor.heating",
            outdoor_temp_entity_id="sensor.outdoor_temp",
            indoor_humidity_entity_id="sensor.humidity",
            window_or_door_open_entity_id="binary_sensor.window",
            heating_power_entity_id="sensor.heating_power",
            heating_on_time_entity_id="sensor.heating_on_time",
            outdoor_temp_forecast_1h_entity_id="sensor.forecast_1h",
            outdoor_temp_forecast_3h_entity_id="sensor.forecast_3h",
        )

        mock_history_data = {
            "sensor.indoor_temp": [
                {"last_changed": "2024-11-25T08:00:00+00:00", "state": "19.0"},
            ],
            "sensor.target_temp": [
                {"last_changed": "2024-11-25T08:00:00+00:00", "state": "21.0"},
            ],
            "binary_sensor.heating": [
                {"last_changed": "2024-11-25T08:00:00+00:00", "state": "on"},
            ],
            "sensor.outdoor_temp": [
                {"last_changed": "2024-11-25T08:00:00+00:00", "state": "5.0"},
            ],
            "sensor.humidity": [
                {"last_changed": "2024-11-25T08:00:00+00:00", "state": "45.0"},
            ],
            "binary_sensor.window": [
                {"last_changed": "2024-11-25T08:00:00+00:00", "state": "off"},
            ],
            "sensor.heating_power": [
                {"last_changed": "2024-11-25T08:00:00+00:00", "state": "0.5"},
            ],
            "sensor.heating_on_time": [
                {"last_changed": "2024-11-25T08:00:00+00:00", "state": "300"},
            ],
            "sensor.forecast_1h": [
                {"last_changed": "2024-11-25T08:00:00+00:00", "state": "4.5"},
            ],
            "sensor.forecast_3h": [
                {"last_changed": "2024-11-25T08:00:00+00:00", "state": "4.0"},
            ],
        }

        timestamp = datetime(2024, 11, 25, 8, 0, 0, tzinfo=timezone.utc)

        observation = reader._construct_observation_at_time(
            mock_history_data,
            training_request,
            timestamp,
        )

        assert observation is not None
        assert observation.indoor_temp == 19.0
        assert observation.target_temp == 21.0
        assert observation.is_heating_on is True
        assert observation.outdoor_temp == 5.0
        assert observation.indoor_humidity == 45.0
        assert observation.window_or_door_open is False
        assert observation.heating_output_percent == 0.5
        assert observation.energy_consumption_recent_kwh == 0.5
        assert observation.time_heating_on_recent_seconds == 300
        assert observation.outdoor_temp_forecast_1h == 4.5
        assert observation.outdoor_temp_forecast_3h == 4.0
        assert observation.device_id == "zone.living_room"

    def test_infer_action_turn_on(self):
        """Test action inference when heating turns on."""
        from domain.value_objects import EntityState, RLObservation
        from domain.value_objects.rl_types import HeatingActionType
        from datetime import datetime

        reader = HomeAssistantHistoryReader(
            ha_url='http://test',
            ha_token='test_token',
        )

        # Create observations with heating off, then on
        entity = EntityState(entity_id="sensor.test", last_changed_minutes=0.0)
        timestamp = datetime(2024, 11, 25, 8, 0, 0)

        obs1 = RLObservation(
            indoor_temp=18.0,
            indoor_temp_entity=entity,
            outdoor_temp=5.0,
            outdoor_temp_entity=entity,
            indoor_humidity=50.0,
            indoor_humidity_entity=entity,
            timestamp=timestamp,
            target_temp=20.0,
            target_temp_entity=entity,
            time_until_target_minutes=0,
            current_target_achieved_percentage=50.0,
            is_heating_on=False,
            heating_output_percent=None,
            heating_output_entity=None,
            energy_consumption_recent_kwh=None,
            energy_consumption_entity=None,
            time_heating_on_recent_seconds=None,
            time_heating_on_entity=None,
            indoor_temp_change_15min=None,
            outdoor_temp_change_15min=None,
            day_of_week=0,
            hour_of_day=8,
            outdoor_temp_forecast_1h=None,
            outdoor_temp_forecast_3h=None,
            window_or_door_open=False,
            window_or_door_entity=None,
            device_id="zone.test",
        )

        obs2 = RLObservation(
            indoor_temp=18.5,
            indoor_temp_entity=entity,
            outdoor_temp=5.0,
            outdoor_temp_entity=entity,
            indoor_humidity=50.0,
            indoor_humidity_entity=entity,
            timestamp=timestamp,
            target_temp=20.0,
            target_temp_entity=entity,
            time_until_target_minutes=0,
            current_target_achieved_percentage=60.0,
            is_heating_on=True,  # Heating turned on
            heating_output_percent=None,
            heating_output_entity=None,
            energy_consumption_recent_kwh=None,
            energy_consumption_entity=None,
            time_heating_on_recent_seconds=None,
            time_heating_on_entity=None,
            indoor_temp_change_15min=None,
            outdoor_temp_change_15min=None,
            day_of_week=0,
            hour_of_day=8,
            outdoor_temp_forecast_1h=None,
            outdoor_temp_forecast_3h=None,
            window_or_door_open=False,
            window_or_door_entity=None,
            device_id="zone.test",
        )

        action = reader._infer_action(obs1, obs2)
        assert action.action_type == HeatingActionType.TURN_ON
        assert action.value == 20.0

    def test_infer_action_turn_off(self):
        """Test action inference when heating turns off."""
        from domain.value_objects import EntityState, RLObservation
        from domain.value_objects.rl_types import HeatingActionType
        from datetime import datetime

        reader = HomeAssistantHistoryReader(
            ha_url='http://test',
            ha_token='test_token',
        )

        entity = EntityState(entity_id="sensor.test", last_changed_minutes=0.0)
        timestamp = datetime(2024, 11, 25, 8, 0, 0)

        obs1 = RLObservation(
            indoor_temp=19.5,
            indoor_temp_entity=entity,
            outdoor_temp=5.0,
            outdoor_temp_entity=entity,
            indoor_humidity=50.0,
            indoor_humidity_entity=entity,
            timestamp=timestamp,
            target_temp=20.0,
            target_temp_entity=entity,
            time_until_target_minutes=0,
            current_target_achieved_percentage=90.0,
            is_heating_on=True,  # Heating on
            heating_output_percent=None,
            heating_output_entity=None,
            energy_consumption_recent_kwh=None,
            energy_consumption_entity=None,
            time_heating_on_recent_seconds=None,
            time_heating_on_entity=None,
            indoor_temp_change_15min=None,
            outdoor_temp_change_15min=None,
            day_of_week=0,
            hour_of_day=8,
            outdoor_temp_forecast_1h=None,
            outdoor_temp_forecast_3h=None,
            window_or_door_open=False,
            window_or_door_entity=None,
            device_id="zone.test",
        )

        obs2 = RLObservation(
            indoor_temp=20.0,
            indoor_temp_entity=entity,
            outdoor_temp=5.0,
            outdoor_temp_entity=entity,
            indoor_humidity=50.0,
            indoor_humidity_entity=entity,
            timestamp=timestamp,
            target_temp=20.0,
            target_temp_entity=entity,
            time_until_target_minutes=0,
            current_target_achieved_percentage=100.0,
            is_heating_on=False,  # Heating turned off
            heating_output_percent=None,
            heating_output_entity=None,
            energy_consumption_recent_kwh=None,
            energy_consumption_entity=None,
            time_heating_on_recent_seconds=None,
            time_heating_on_entity=None,
            indoor_temp_change_15min=None,
            outdoor_temp_change_15min=None,
            day_of_week=0,
            hour_of_day=8,
            outdoor_temp_forecast_1h=None,
            outdoor_temp_forecast_3h=None,
            window_or_door_open=False,
            window_or_door_entity=None,
            device_id="zone.test",
        )

        action = reader._infer_action(obs1, obs2)
        assert action.action_type == HeatingActionType.TURN_OFF
        assert action.value == 20.0

    def test_is_episode_done_target_reached(self):
        """Test episode done when target temperature is reached."""
        from domain.value_objects import EntityState, RLObservation
        from datetime import datetime

        reader = HomeAssistantHistoryReader(
            ha_url='http://test',
            ha_token='test_token',
        )

        entity = EntityState(entity_id="sensor.test", last_changed_minutes=0.0)
        timestamp = datetime(2024, 11, 25, 8, 0, 0)

        obs1 = RLObservation(
            indoor_temp=19.5,
            indoor_temp_entity=entity,
            outdoor_temp=5.0,
            outdoor_temp_entity=entity,
            indoor_humidity=50.0,
            indoor_humidity_entity=entity,
            timestamp=timestamp,
            target_temp=20.0,
            target_temp_entity=entity,
            time_until_target_minutes=0,
            current_target_achieved_percentage=90.0,
            is_heating_on=True,
            heating_output_percent=None,
            heating_output_entity=None,
            energy_consumption_recent_kwh=None,
            energy_consumption_entity=None,
            time_heating_on_recent_seconds=None,
            time_heating_on_entity=None,
            indoor_temp_change_15min=None,
            outdoor_temp_change_15min=None,
            day_of_week=0,
            hour_of_day=8,
            outdoor_temp_forecast_1h=None,
            outdoor_temp_forecast_3h=None,
            window_or_door_open=False,
            window_or_door_entity=None,
            device_id="zone.test",
        )

        obs2 = RLObservation(
            indoor_temp=20.0,  # Target reached
            indoor_temp_entity=entity,
            outdoor_temp=5.0,
            outdoor_temp_entity=entity,
            indoor_humidity=50.0,
            indoor_humidity_entity=entity,
            timestamp=timestamp,
            target_temp=20.0,
            target_temp_entity=entity,
            time_until_target_minutes=0,
            current_target_achieved_percentage=100.0,
            is_heating_on=False,
            heating_output_percent=None,
            heating_output_entity=None,
            energy_consumption_recent_kwh=None,
            energy_consumption_entity=None,
            time_heating_on_recent_seconds=None,
            time_heating_on_entity=None,
            indoor_temp_change_15min=None,
            outdoor_temp_change_15min=None,
            day_of_week=0,
            hour_of_day=8,
            outdoor_temp_forecast_1h=None,
            outdoor_temp_forecast_3h=None,
            window_or_door_open=False,
            window_or_door_entity=None,
            device_id="zone.test",
        )

        done = reader._is_episode_done(obs2, obs1)
        assert done is True  # Episode should end when target is reached

    def test_is_episode_done_target_changed(self):
        """Test episode done when target temperature changes significantly."""
        from domain.value_objects import EntityState, RLObservation
        from datetime import datetime

        reader = HomeAssistantHistoryReader(
            ha_url='http://test',
            ha_token='test_token',
        )

        entity = EntityState(entity_id="sensor.test", last_changed_minutes=0.0)
        timestamp = datetime(2024, 11, 25, 8, 0, 0)

        obs1 = RLObservation(
            indoor_temp=19.0,
            indoor_temp_entity=entity,
            outdoor_temp=5.0,
            outdoor_temp_entity=entity,
            indoor_humidity=50.0,
            indoor_humidity_entity=entity,
            timestamp=timestamp,
            target_temp=20.0,
            target_temp_entity=entity,
            time_until_target_minutes=0,
            current_target_achieved_percentage=80.0,
            is_heating_on=True,
            heating_output_percent=None,
            heating_output_entity=None,
            energy_consumption_recent_kwh=None,
            energy_consumption_entity=None,
            time_heating_on_recent_seconds=None,
            time_heating_on_entity=None,
            indoor_temp_change_15min=None,
            outdoor_temp_change_15min=None,
            day_of_week=0,
            hour_of_day=8,
            outdoor_temp_forecast_1h=None,
            outdoor_temp_forecast_3h=None,
            window_or_door_open=False,
            window_or_door_entity=None,
            device_id="zone.test",
        )

        obs2 = RLObservation(
            indoor_temp=19.5,
            indoor_temp_entity=entity,
            outdoor_temp=5.0,
            outdoor_temp_entity=entity,
            indoor_humidity=50.0,
            indoor_humidity_entity=entity,
            timestamp=timestamp,
            target_temp=22.0,  # Target changed significantly
            target_temp_entity=entity,
            time_until_target_minutes=0,
            current_target_achieved_percentage=70.0,
            is_heating_on=True,
            heating_output_percent=None,
            heating_output_entity=None,
            energy_consumption_recent_kwh=None,
            energy_consumption_entity=None,
            time_heating_on_recent_seconds=None,
            time_heating_on_entity=None,
            indoor_temp_change_15min=None,
            outdoor_temp_change_15min=None,
            day_of_week=0,
            hour_of_day=8,
            outdoor_temp_forecast_1h=None,
            outdoor_temp_forecast_3h=None,
            window_or_door_open=False,
            window_or_door_entity=None,
            device_id="zone.test",
        )

        done = reader._is_episode_done(obs2, obs1)
        assert done is True  # Episode should end when target changes significantly
