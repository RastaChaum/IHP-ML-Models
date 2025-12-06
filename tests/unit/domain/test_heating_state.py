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
