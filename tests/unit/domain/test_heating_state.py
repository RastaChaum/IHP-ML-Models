"""Unit tests for HeatingState entity."""

import pytest

from ihp_ml_addon.rootfs.app.domain.entities import HeatingState


class TestHeatingState:
    """Tests for HeatingState entity."""

    def test_is_heating_when_on_and_below_target(self):
        """Test is_heating returns True when on and below target."""
        state = HeatingState(is_on=True, preset_mode="comfort", target_temp=21.0)
        assert state.is_heating(19.0) is True

    def test_is_heating_when_on_and_above_target(self):
        """Test is_heating returns False when on but above target."""
        state = HeatingState(is_on=True, preset_mode="comfort", target_temp=21.0)
        assert state.is_heating(22.0) is False

    def test_is_heating_when_off(self):
        """Test is_heating returns False when heating is off."""
        state = HeatingState(is_on=False, preset_mode="comfort", target_temp=21.0)
        assert state.is_heating(19.0) is False

    def test_is_heating_when_at_target(self):
        """Test is_heating returns False when at target temperature."""
        state = HeatingState(is_on=True, preset_mode="comfort", target_temp=21.0)
        assert state.is_heating(21.0) is False

    def test_from_ha_state_record_climate_heating(self):
        """Test creating HeatingState from climate entity when heating."""
        record = {
            "entity_id": "climate.thermostat",
            "state": "heat",
            "attributes": {
                "hvac_action": "heating",
                "hvac_mode": "heat",
                "preset_mode": "comfort",
                "temperature": 21.5,
            },
        }

        state = HeatingState.from_ha_state_record(record)
        assert state.is_on is True
        assert state.preset_mode == "comfort"
        assert state.target_temp == 21.5

    def test_from_ha_state_record_climate_idle(self):
        """Test creating HeatingState from climate entity when idle."""
        record = {
            "entity_id": "climate.thermostat",
            "state": "off",
            "attributes": {
                "hvac_action": "idle",
                "hvac_mode": "off",
                "preset_mode": "eco",
                "temperature": 18.0,
            },
        }

        state = HeatingState.from_ha_state_record(record)
        assert state.is_on is False
        assert state.preset_mode == "eco"
        assert state.target_temp == 18.0

    def test_from_ha_state_record_binary_sensor_on(self):
        """Test creating HeatingState from binary sensor when on."""
        record = {
            "entity_id": "binary_sensor.heating",
            "state": "on",
        }

        state = HeatingState.from_ha_state_record(record)
        assert state.is_on is True
        assert state.preset_mode is None
        assert state.target_temp == 0.0  # Default for binary sensors

    def test_from_ha_state_record_binary_sensor_off(self):
        """Test creating HeatingState from binary sensor when off."""
        record = {
            "entity_id": "binary_sensor.heating",
            "state": "off",
        }

        state = HeatingState.from_ha_state_record(record)
        assert state.is_on is False

    def test_from_ha_state_record_climate_missing_temperature(self):
        """Test that ValueError is raised when climate entity missing temperature."""
        record = {
            "entity_id": "climate.thermostat",
            "state": "heat",
            "attributes": {
                "hvac_action": "heating",
            },
        }

        with pytest.raises(ValueError, match="missing temperature"):
            HeatingState.from_ha_state_record(record)

    def test_from_ha_state_record_climate_hvac_action_heating(self):
        """Test climate entity with hvac_action='heating'."""
        record = {
            "entity_id": "climate.living_room",
            "state": "off",
            "attributes": {
                "hvac_action": "heating",
                "temperature": 20.0,
            },
        }

        state = HeatingState.from_ha_state_record(record)
        assert state.is_on is True

    def test_from_ha_state_record_climate_state_heating(self):
        """Test climate entity with state='heating'."""
        record = {
            "entity_id": "climate.bedroom",
            "state": "heating",
            "attributes": {
                "hvac_action": "idle",
                "temperature": 19.0,
            },
        }

        state = HeatingState.from_ha_state_record(record)
        assert state.is_on is True

    def test_from_ha_state_record_switch_on(self):
        """Test creating HeatingState from switch entity."""
        record = {
            "entity_id": "switch.heater",
            "state": "on",
        }

        state = HeatingState.from_ha_state_record(record)
        assert state.is_on is True
