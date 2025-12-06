"""Unit tests for RL domain services."""

from datetime import datetime, timezone

import pytest

from ihp_ml_addon.rootfs.app.domain.services import RLActionService, RLEpisodeService
from ihp_ml_addon.rootfs.app.domain.value_objects import (
    EntityState,
    HeatingActionType,
    RLObservation,
)


@pytest.fixture
def entity_state():
    """Create a test entity state."""
    return EntityState(entity_id="sensor.test", last_changed_minutes=0.0)


@pytest.fixture
def create_observation(entity_state):
    """Factory function to create test observations."""

    def _create(
        indoor_temp: float,
        target_temp: float,
        is_heating_on: bool,
        timestamp: datetime | None = None,
    ) -> RLObservation:
        if timestamp is None:
            timestamp = datetime(2024, 11, 25, 8, 0, 0, tzinfo=timezone.utc)

        return RLObservation(
            indoor_temp=indoor_temp,
            indoor_temp_entity=entity_state,
            outdoor_temp=5.0,
            outdoor_temp_entity=entity_state,
            indoor_humidity=50.0,
            indoor_humidity_entity=entity_state,
            timestamp=timestamp,
            target_temp=target_temp,
            target_temp_entity=entity_state,
            time_until_target_minutes=0,
            current_target_achieved_percentage=50.0,
            is_heating_on=is_heating_on,
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

    return _create


class TestRLActionService:
    """Tests for RLActionService."""

    def test_infer_action_turn_on(self, create_observation):
        """Test action inference when heating turns on."""
        service = RLActionService()

        obs1 = create_observation(18.0, 20.0, is_heating_on=False)
        obs2 = create_observation(18.5, 20.0, is_heating_on=True)

        action = service.infer_action(obs1, obs2)

        assert action.action_type == HeatingActionType.TURN_ON
        assert action.value == 20.0

    def test_infer_action_turn_off(self, create_observation):
        """Test action inference when heating turns off."""
        service = RLActionService()

        obs1 = create_observation(19.5, 20.0, is_heating_on=True)
        obs2 = create_observation(20.0, 20.0, is_heating_on=False)

        action = service.infer_action(obs1, obs2)

        assert action.action_type == HeatingActionType.TURN_OFF
        assert action.value == 20.0

    def test_infer_action_set_target_temperature_higher(self, create_observation):
        """Test action inference when target temperature increases."""
        service = RLActionService()

        obs1 = create_observation(19.0, 20.0, is_heating_on=True)
        obs2 = create_observation(19.5, 22.0, is_heating_on=True)

        action = service.infer_action(obs1, obs2)

        assert action.action_type == HeatingActionType.SET_TARGET_TEMPERATURE_HIGHER
        assert action.value == 22.0

    def test_infer_action_set_target_temperature_lower(self, create_observation):
        """Test action inference when target temperature decreases."""
        service = RLActionService()

        obs1 = create_observation(19.0, 22.0, is_heating_on=True)
        obs2 = create_observation(19.5, 20.0, is_heating_on=True)

        action = service.infer_action(obs1, obs2)

        assert action.action_type == HeatingActionType.SET_TARGET_TEMPERATURE_LOWER
        assert action.value == 20.0

    def test_infer_action_no_op(self, create_observation):
        """Test action inference when no significant change."""
        service = RLActionService()

        obs1 = create_observation(19.0, 20.0, is_heating_on=True)
        obs2 = create_observation(19.2, 20.0, is_heating_on=True)

        action = service.infer_action(obs1, obs2)

        assert action.action_type == HeatingActionType.NO_OP
        assert action.value == 20.0

    def test_infer_action_small_target_change(self, create_observation):
        """Test that small target changes (<0.1°C) result in NO_OP."""
        service = RLActionService()

        obs1 = create_observation(19.0, 20.0, is_heating_on=True)
        obs2 = create_observation(19.1, 20.05, is_heating_on=True)

        action = service.infer_action(obs1, obs2)

        assert action.action_type == HeatingActionType.NO_OP


class TestRLEpisodeService:
    """Tests for RLEpisodeService."""

    def test_episode_done_target_reached(self, create_observation):
        """Test episode termination when target is reached."""
        service = RLEpisodeService(target_tolerance_celsius=0.3)

        obs1 = create_observation(19.5, 20.0, is_heating_on=True)
        obs2 = create_observation(20.0, 20.0, is_heating_on=False)

        assert service.is_episode_done(obs2, obs1) is True

    def test_episode_done_within_tolerance(self, create_observation):
        """Test episode termination when within tolerance."""
        service = RLEpisodeService(target_tolerance_celsius=0.3)

        obs1 = create_observation(19.5, 20.0, is_heating_on=True)
        obs2 = create_observation(19.8, 20.0, is_heating_on=True)

        assert service.is_episode_done(obs2, obs1) is True

    def test_episode_done_target_changed(self, create_observation):
        """Test episode termination when target changes significantly."""
        service = RLEpisodeService(target_change_threshold_celsius=0.5)

        obs1 = create_observation(19.0, 20.0, is_heating_on=True)
        obs2 = create_observation(19.5, 21.0, is_heating_on=True)

        assert service.is_episode_done(obs2, obs1) is True

    def test_episode_continues(self, create_observation):
        """Test episode continuation when conditions not met."""
        service = RLEpisodeService(
            target_tolerance_celsius=0.3, target_change_threshold_celsius=0.5
        )

        obs1 = create_observation(18.0, 20.0, is_heating_on=True)
        obs2 = create_observation(18.5, 20.0, is_heating_on=True)

        assert service.is_episode_done(obs2, obs1) is False

    def test_episode_custom_tolerance(self, create_observation):
        """Test episode termination with custom tolerance."""
        service = RLEpisodeService(target_tolerance_celsius=0.5)

        obs1 = create_observation(19.0, 20.0, is_heating_on=True)
        obs2 = create_observation(19.6, 20.0, is_heating_on=True)

        # Within 0.5°C tolerance, should end
        assert service.is_episode_done(obs2, obs1) is True

    def test_episode_custom_change_threshold(self, create_observation):
        """Test episode termination with custom change threshold."""
        service = RLEpisodeService(target_change_threshold_celsius=1.0)

        obs1 = create_observation(19.0, 20.0, is_heating_on=True)
        obs2 = create_observation(19.5, 20.8, is_heating_on=True)

        # Change is 0.8°C, below 1.0°C threshold, should continue
        assert service.is_episode_done(obs2, obs1) is False
