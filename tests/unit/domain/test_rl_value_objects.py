"""Unit tests for RL value objects.

Tests for RL-related value objects ensuring immutability,
validation, and correct behavior.
"""

from datetime import datetime, timedelta

import pytest

from ihp_ml_addon.rootfs.app.domain.value_objects import (
    EntityState,
    HeatingActionType,
    RLAction,
    RLExperience,
    RLObservation,
    TrainingRequest,
)


class TestHeatingActionType:
    """Tests for HeatingActionType enum."""

    def test_all_action_types_defined(self):
        """Test that all required action types are defined."""
        assert HeatingActionType.TURN_ON == "turn_on"
        assert HeatingActionType.TURN_OFF == "turn_off"
        assert HeatingActionType.SET_TARGET_TEMPERATURE == "set_target_temperature"
        assert HeatingActionType.NO_OP == "no_op"

    def test_action_type_is_string_enum(self):
        """Test that action type values are strings."""
        for action_type in HeatingActionType:
            assert isinstance(action_type.value, str)


class TestEntityState:
    """Tests for EntityState value object."""

    def test_valid_entity_state(self):
        """Test creating a valid entity state."""
        entity = EntityState(
            entity_id="sensor.temperature",
            last_changed_minutes=5.5,
        )
        assert entity.entity_id == "sensor.temperature"
        assert entity.last_changed_minutes == 5.5

    def test_entity_state_is_frozen(self):
        """Test that EntityState is immutable."""
        entity = EntityState(entity_id="sensor.temp", last_changed_minutes=1.0)
        with pytest.raises(Exception):  # dataclass frozen error
            entity.entity_id = "sensor.other"

    def test_empty_entity_id_raises_error(self):
        """Test that empty entity_id raises ValueError."""
        with pytest.raises(ValueError, match="entity_id cannot be empty"):
            EntityState(entity_id="", last_changed_minutes=1.0)

    def test_negative_last_changed_raises_error(self):
        """Test that negative last_changed_minutes raises ValueError."""
        with pytest.raises(ValueError, match="last_changed_minutes must be non-negative"):
            EntityState(entity_id="sensor.temp", last_changed_minutes=-1.0)


class TestRLObservation:
    """Tests for RLObservation value object."""

    def create_valid_observation(self, **kwargs) -> RLObservation:
        """Helper to create a valid observation with optional overrides."""
        defaults = {
            "current_temp": 20.0,
            "current_temp_entity": EntityState("sensor.indoor_temp", 0.0),
            "outdoor_temp": 5.0,
            "outdoor_temp_entity": EntityState("sensor.outdoor_temp", 0.0),
            "humidity": 65.0,
            "humidity_entity": EntityState("sensor.humidity", 0.0),
            "timestamp": datetime.now(),
            "target_temp_from_schedule": 22.0,
            "time_until_target_minutes": 30,
            "current_target_achieved_percentage": 50.0,
            "is_heating_on": True,
            "heating_output_percent": 80.0,
            "energy_consumption_recent_kwh": 0.5,
            "time_heating_on_recent_seconds": 300,
            "indoor_temp_change_15min": 0.5,
            "outdoor_temp_change_15min": -0.2,
            "day_of_week": 1,
            "hour_of_day": 14,
            "outdoor_temp_forecast_1h": 4.5,
            "outdoor_temp_forecast_3h": 4.0,
            "window_or_door_open": False,
            "window_or_door_entity": EntityState("binary_sensor.window", 0.0),
            "device_id": "ihp_salon",
        }
        defaults.update(kwargs)
        return RLObservation(**defaults)

    def test_valid_observation(self):
        """Test creating a valid RL observation."""
        obs = self.create_valid_observation()
        assert obs.current_temp == 20.0
        assert obs.outdoor_temp == 5.0
        assert obs.device_id == "ihp_salon"
        assert obs.is_heating_on is True

    def test_observation_is_frozen(self):
        """Test that RLObservation is immutable."""
        obs = self.create_valid_observation()
        with pytest.raises(Exception):  # dataclass frozen error
            obs.current_temp = 25.0

    def test_outdoor_temp_out_of_range_raises_error(self):
        """Test that outdoor temperature out of range raises ValueError."""
        with pytest.raises(ValueError, match="outdoor_temp must be between -50 and 60"):
            self.create_valid_observation(outdoor_temp=-60.0)

        with pytest.raises(ValueError, match="outdoor_temp must be between -50 and 60"):
            self.create_valid_observation(outdoor_temp=70.0)

    def test_current_temp_out_of_range_raises_error(self):
        """Test that current temperature out of range raises ValueError."""
        with pytest.raises(ValueError, match="current_temp must be between -20 and 50"):
            self.create_valid_observation(current_temp=-30.0)

        with pytest.raises(ValueError, match="current_temp must be between -20 and 50"):
            self.create_valid_observation(current_temp=60.0)

    def test_humidity_out_of_range_raises_error(self):
        """Test that humidity out of range raises ValueError."""
        with pytest.raises(ValueError, match="humidity must be between 0 and 100"):
            self.create_valid_observation(humidity=-10.0)

        with pytest.raises(ValueError, match="humidity must be between 0 and 100"):
            self.create_valid_observation(humidity=110.0)

    def test_day_of_week_out_of_range_raises_error(self):
        """Test that day_of_week out of range raises ValueError."""
        with pytest.raises(ValueError, match="day_of_week must be between 0 and 6"):
            self.create_valid_observation(day_of_week=-1)

        with pytest.raises(ValueError, match="day_of_week must be between 0 and 6"):
            self.create_valid_observation(day_of_week=7)

    def test_hour_of_day_out_of_range_raises_error(self):
        """Test that hour_of_day out of range raises ValueError."""
        with pytest.raises(ValueError, match="hour_of_day must be between 0 and 23"):
            self.create_valid_observation(hour_of_day=-1)

        with pytest.raises(ValueError, match="hour_of_day must be between 0 and 23"):
            self.create_valid_observation(hour_of_day=24)

    def test_target_achieved_percentage_out_of_range_raises_error(self):
        """Test that target achieved percentage out of range raises ValueError."""
        with pytest.raises(
            ValueError, match="current_target_achieved_percentage must be between 0 and 100"
        ):
            self.create_valid_observation(current_target_achieved_percentage=-10.0)

        with pytest.raises(
            ValueError, match="current_target_achieved_percentage must be between 0 and 100"
        ):
            self.create_valid_observation(current_target_achieved_percentage=110.0)

    def test_heating_output_percent_out_of_range_raises_error(self):
        """Test that heating output percentage out of range raises ValueError."""
        with pytest.raises(ValueError, match="heating_output_percent must be between 0 and 100"):
            self.create_valid_observation(heating_output_percent=-10.0)

        with pytest.raises(ValueError, match="heating_output_percent must be between 0 and 100"):
            self.create_valid_observation(heating_output_percent=110.0)

    def test_empty_device_id_raises_error(self):
        """Test that empty device_id raises ValueError."""
        with pytest.raises(ValueError, match="device_id cannot be empty"):
            self.create_valid_observation(device_id="")

    def test_negative_time_until_target_raises_error(self):
        """Test that negative time_until_target_minutes raises ValueError."""
        with pytest.raises(ValueError, match="time_until_target_minutes must be non-negative"):
            self.create_valid_observation(time_until_target_minutes=-5)

    def test_negative_energy_consumption_raises_error(self):
        """Test that negative energy consumption raises ValueError."""
        with pytest.raises(ValueError, match="energy_consumption_recent_kwh must be non-negative"):
            self.create_valid_observation(energy_consumption_recent_kwh=-0.5)

    def test_optional_fields_can_be_none(self):
        """Test that optional fields can be None."""
        obs = self.create_valid_observation(
            target_temp_from_schedule=None,
            time_until_target_minutes=None,
            current_target_achieved_percentage=None,
            heating_output_percent=None,
            energy_consumption_recent_kwh=None,
            time_heating_on_recent_seconds=None,
            indoor_temp_change_15min=None,
            outdoor_temp_change_15min=None,
            outdoor_temp_forecast_1h=None,
            outdoor_temp_forecast_3h=None,
            window_or_door_entity=None,
        )
        assert obs.target_temp_from_schedule is None
        assert obs.heating_output_percent is None


class TestRLAction:
    """Tests for RLAction value object."""

    def test_valid_turn_on_action(self):
        """Test creating a valid turn on action."""
        action = RLAction(
            action_type=HeatingActionType.TURN_ON,
            value=None,
            decision_timestamp=datetime.now(),
            confidence_score=0.85,
        )
        assert action.action_type == HeatingActionType.TURN_ON
        assert action.value is None
        assert action.confidence_score == 0.85

    def test_valid_set_temperature_action(self):
        """Test creating a valid set temperature action."""
        action = RLAction(
            action_type=HeatingActionType.SET_TARGET_TEMPERATURE,
            value=22.5,
            decision_timestamp=datetime.now(),
        )
        assert action.action_type == HeatingActionType.SET_TARGET_TEMPERATURE
        assert action.value == 22.5

    def test_action_is_frozen(self):
        """Test that RLAction is immutable."""
        action = RLAction(
            action_type=HeatingActionType.TURN_OFF,
            value=None,
            decision_timestamp=datetime.now(),
        )
        with pytest.raises(Exception):  # dataclass frozen error
            action.value = 25.0

    def test_set_temperature_without_value_raises_error(self):
        """Test that SET_TARGET_TEMPERATURE without value raises ValueError."""
        with pytest.raises(ValueError, match="value must be provided for SET_TARGET_TEMPERATURE"):
            RLAction(
                action_type=HeatingActionType.SET_TARGET_TEMPERATURE,
                value=None,
                decision_timestamp=datetime.now(),
            )

    def test_set_temperature_value_out_of_range_raises_error(self):
        """Test that temperature value out of range raises ValueError."""
        with pytest.raises(ValueError, match="target temperature value must be between 0 and 50"):
            RLAction(
                action_type=HeatingActionType.SET_TARGET_TEMPERATURE,
                value=-5.0,
                decision_timestamp=datetime.now(),
            )

        with pytest.raises(ValueError, match="target temperature value must be between 0 and 50"):
            RLAction(
                action_type=HeatingActionType.SET_TARGET_TEMPERATURE,
                value=60.0,
                decision_timestamp=datetime.now(),
            )

    def test_confidence_score_out_of_range_raises_error(self):
        """Test that confidence score out of range raises ValueError."""
        with pytest.raises(ValueError, match="confidence_score must be between 0.0 and 1.0"):
            RLAction(
                action_type=HeatingActionType.TURN_ON,
                value=None,
                decision_timestamp=datetime.now(),
                confidence_score=-0.1,
            )

        with pytest.raises(ValueError, match="confidence_score must be between 0.0 and 1.0"):
            RLAction(
                action_type=HeatingActionType.TURN_ON,
                value=None,
                decision_timestamp=datetime.now(),
                confidence_score=1.5,
            )


class TestRLExperience:
    """Tests for RLExperience value object."""

    def create_valid_observation(self, device_id: str = "test_device") -> RLObservation:
        """Helper to create a valid observation."""
        return RLObservation(
            current_temp=20.0,
            current_temp_entity=EntityState("sensor.temp", 0.0),
            outdoor_temp=5.0,
            outdoor_temp_entity=EntityState("sensor.outdoor", 0.0),
            humidity=65.0,
            humidity_entity=EntityState("sensor.humidity", 0.0),
            timestamp=datetime.now(),
            target_temp_from_schedule=22.0,
            time_until_target_minutes=30,
            current_target_achieved_percentage=50.0,
            is_heating_on=True,
            heating_output_percent=80.0,
            energy_consumption_recent_kwh=0.5,
            time_heating_on_recent_seconds=300,
            indoor_temp_change_15min=0.5,
            outdoor_temp_change_15min=-0.2,
            day_of_week=1,
            hour_of_day=14,
            outdoor_temp_forecast_1h=4.5,
            outdoor_temp_forecast_3h=4.0,
            window_or_door_open=False,
            window_or_door_entity=EntityState("binary_sensor.window", 0.0),
            device_id=device_id,
        )

    def test_valid_experience(self):
        """Test creating a valid RL experience."""
        state = self.create_valid_observation()
        action = RLAction(
            action_type=HeatingActionType.TURN_ON,
            value=None,
            decision_timestamp=datetime.now(),
        )
        next_state = self.create_valid_observation()

        experience = RLExperience(
            state=state,
            action=action,
            reward=1.0,
            next_state=next_state,
            done=False,
        )

        assert experience.state == state
        assert experience.action == action
        assert experience.reward == 1.0
        assert experience.next_state == next_state
        assert experience.done is False

    def test_experience_is_frozen(self):
        """Test that RLExperience is immutable."""
        state = self.create_valid_observation()
        action = RLAction(
            action_type=HeatingActionType.TURN_ON,
            value=None,
            decision_timestamp=datetime.now(),
        )
        next_state = self.create_valid_observation()

        experience = RLExperience(
            state=state,
            action=action,
            reward=1.0,
            next_state=next_state,
            done=False,
        )

        with pytest.raises(Exception):  # dataclass frozen error
            experience.reward = 2.0

    def test_mismatched_device_ids_raises_error(self):
        """Test that mismatched device IDs raise ValueError."""
        state = self.create_valid_observation(device_id="device1")
        action = RLAction(
            action_type=HeatingActionType.TURN_ON,
            value=None,
            decision_timestamp=datetime.now(),
        )
        next_state = self.create_valid_observation(device_id="device2")

        with pytest.raises(ValueError, match="state and next_state must be for the same device"):
            RLExperience(
                state=state,
                action=action,
                reward=1.0,
                next_state=next_state,
                done=False,
            )


class TestTrainingRequest:
    """Tests for TrainingRequest value object."""

    def test_valid_training_request(self):
        """Test creating a valid training request."""
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()

        request = TrainingRequest(
            device_id="ihp_salon",
            start_time=start,
            end_time=end,
            current_temp_entity_id="sensor.indoor_temp",
            outdoor_temp_entity_id="sensor.outdoor_temp",
            humidity_entity_id="sensor.humidity",
            versatile_thermostat_entity_id="climate.thermostat",
            window_or_door_open_entity_id="binary_sensor.window",
            heating_power_entity_id="sensor.power",
            behavioral_cloning_epochs=10,
            online_learning_enabled=True,
        )

        assert request.device_id == "ihp_salon"
        assert request.start_time == start
        assert request.end_time == end
        assert request.behavioral_cloning_epochs == 10
        assert request.online_learning_enabled is True

    def test_training_request_is_frozen(self):
        """Test that TrainingRequest is immutable."""
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()

        request = TrainingRequest(
            device_id="ihp_salon",
            start_time=start,
            end_time=end,
            current_temp_entity_id="sensor.indoor_temp",
            outdoor_temp_entity_id="sensor.outdoor_temp",
            humidity_entity_id="sensor.humidity",
            versatile_thermostat_entity_id="climate.thermostat",
        )

        with pytest.raises(Exception):  # dataclass frozen error
            request.device_id = "other_device"

    def test_empty_device_id_raises_error(self):
        """Test that empty device_id raises ValueError."""
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()

        with pytest.raises(ValueError, match="device_id cannot be empty"):
            TrainingRequest(
                device_id="",
                start_time=start,
                end_time=end,
                current_temp_entity_id="sensor.indoor_temp",
                outdoor_temp_entity_id="sensor.outdoor_temp",
                humidity_entity_id="sensor.humidity",
                versatile_thermostat_entity_id="climate.thermostat",
            )

    def test_empty_required_entity_id_raises_error(self):
        """Test that empty required entity IDs raise ValueError."""
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()

        with pytest.raises(ValueError, match="current_temp_entity_id cannot be empty"):
            TrainingRequest(
                device_id="ihp_salon",
                start_time=start,
                end_time=end,
                current_temp_entity_id="",
                outdoor_temp_entity_id="sensor.outdoor_temp",
                humidity_entity_id="sensor.humidity",
                versatile_thermostat_entity_id="climate.thermostat",
            )

    def test_start_time_after_end_time_raises_error(self):
        """Test that start_time after end_time raises ValueError."""
        start = datetime.now()
        end = datetime.now() - timedelta(days=30)

        with pytest.raises(ValueError, match="start_time must be before end_time"):
            TrainingRequest(
                device_id="ihp_salon",
                start_time=start,
                end_time=end,
                current_temp_entity_id="sensor.indoor_temp",
                outdoor_temp_entity_id="sensor.outdoor_temp",
                humidity_entity_id="sensor.humidity",
                versatile_thermostat_entity_id="climate.thermostat",
            )

    def test_negative_behavioral_cloning_epochs_raises_error(self):
        """Test that negative behavioral_cloning_epochs raises ValueError."""
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()

        with pytest.raises(ValueError, match="behavioral_cloning_epochs must be non-negative"):
            TrainingRequest(
                device_id="ihp_salon",
                start_time=start,
                end_time=end,
                current_temp_entity_id="sensor.indoor_temp",
                outdoor_temp_entity_id="sensor.outdoor_temp",
                humidity_entity_id="sensor.humidity",
                versatile_thermostat_entity_id="climate.thermostat",
                behavioral_cloning_epochs=-5,
            )

    def test_optional_entity_ids_can_be_none(self):
        """Test that optional entity IDs can be None."""
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()

        request = TrainingRequest(
            device_id="ihp_salon",
            start_time=start,
            end_time=end,
            current_temp_entity_id="sensor.indoor_temp",
            outdoor_temp_entity_id="sensor.outdoor_temp",
            humidity_entity_id="sensor.humidity",
            versatile_thermostat_entity_id="climate.thermostat",
            window_or_door_open_entity_id=None,
            heating_power_entity_id=None,
        )

        assert request.window_or_door_open_entity_id is None
        assert request.heating_power_entity_id is None
