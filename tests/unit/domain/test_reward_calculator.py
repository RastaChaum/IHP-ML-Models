"""Unit tests for HeatingRewardCalculator.

Tests for reward calculation logic including progress rewards,
drift penalties, overshoot penalties, and energy consumption penalties.
"""

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from ihp_ml_addon.rootfs.app.domain.services import HeatingRewardCalculator
from ihp_ml_addon.rootfs.app.domain.value_objects import (
    EntityState,
    HeatingActionType,
    RewardConfig,
    RLAction,
    RLObservation,
)


class TestRewardConfig:
    """Tests for RewardConfig value object."""

    def test_default_config(self):
        """Test creating config with default values."""
        config = RewardConfig()
        assert config.progress_reward_factor == 1.0
        assert config.drift_penalty_factor == 0.5
        assert config.overshoot_penalty_factor == 2.0
        assert config.energy_penalty_factor == 0.1
        assert config.target_achieved_reward == 10.0
        assert config.target_missed_penalty == 10.0
        assert config.target_tolerance_celsius == 0.5
        assert config.overshoot_threshold_celsius == 1.0

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = RewardConfig(
            progress_reward_factor=2.0,
            drift_penalty_factor=1.0,
            overshoot_penalty_factor=3.0,
            energy_penalty_factor=0.2,
            target_achieved_reward=20.0,
            target_missed_penalty=15.0,
            target_tolerance_celsius=0.3,
            overshoot_threshold_celsius=0.5,
        )
        assert config.progress_reward_factor == 2.0
        assert config.drift_penalty_factor == 1.0
        assert config.overshoot_penalty_factor == 3.0
        assert config.energy_penalty_factor == 0.2
        assert config.target_achieved_reward == 20.0
        assert config.target_missed_penalty == 15.0
        assert config.target_tolerance_celsius == 0.3
        assert config.overshoot_threshold_celsius == 0.5

    def test_config_is_frozen(self):
        """Test that RewardConfig is immutable."""
        config = RewardConfig()
        with pytest.raises(FrozenInstanceError):
            config.progress_reward_factor = 2.0

    def test_negative_progress_reward_factor_raises_error(self):
        """Test that negative progress_reward_factor raises ValueError."""
        with pytest.raises(ValueError, match="progress_reward_factor must be non-negative"):
            RewardConfig(progress_reward_factor=-1.0)

    def test_negative_drift_penalty_factor_raises_error(self):
        """Test that negative drift_penalty_factor raises ValueError."""
        with pytest.raises(ValueError, match="drift_penalty_factor must be non-negative"):
            RewardConfig(drift_penalty_factor=-1.0)

    def test_negative_overshoot_penalty_factor_raises_error(self):
        """Test that negative overshoot_penalty_factor raises ValueError."""
        with pytest.raises(ValueError, match="overshoot_penalty_factor must be non-negative"):
            RewardConfig(overshoot_penalty_factor=-1.0)

    def test_negative_energy_penalty_factor_raises_error(self):
        """Test that negative energy_penalty_factor raises ValueError."""
        with pytest.raises(ValueError, match="energy_penalty_factor must be non-negative"):
            RewardConfig(energy_penalty_factor=-1.0)

    def test_negative_target_achieved_reward_raises_error(self):
        """Test that negative target_achieved_reward raises ValueError."""
        with pytest.raises(ValueError, match="target_achieved_reward must be non-negative"):
            RewardConfig(target_achieved_reward=-1.0)

    def test_negative_target_missed_penalty_raises_error(self):
        """Test that negative target_missed_penalty raises ValueError."""
        with pytest.raises(ValueError, match="target_missed_penalty must be non-negative"):
            RewardConfig(target_missed_penalty=-1.0)

    def test_zero_target_tolerance_raises_error(self):
        """Test that zero target_tolerance_celsius raises ValueError."""
        with pytest.raises(ValueError, match="target_tolerance_celsius must be positive"):
            RewardConfig(target_tolerance_celsius=0.0)

    def test_negative_target_tolerance_raises_error(self):
        """Test that negative target_tolerance_celsius raises ValueError."""
        with pytest.raises(ValueError, match="target_tolerance_celsius must be positive"):
            RewardConfig(target_tolerance_celsius=-0.5)

    def test_zero_overshoot_threshold_raises_error(self):
        """Test that zero overshoot_threshold_celsius raises ValueError."""
        with pytest.raises(ValueError, match="overshoot_threshold_celsius must be positive"):
            RewardConfig(overshoot_threshold_celsius=0.0)

    def test_negative_overshoot_threshold_raises_error(self):
        """Test that negative overshoot_threshold_celsius raises ValueError."""
        with pytest.raises(ValueError, match="overshoot_threshold_celsius must be positive"):
            RewardConfig(overshoot_threshold_celsius=-0.5)


class TestHeatingRewardCalculator:
    """Tests for HeatingRewardCalculator service."""

    def create_observation(
        self,
        indoor_temp: float,
        target_temp: float,
        device_id: str = "test_device",
        energy_consumption: float | None = None,
    ) -> RLObservation:
        """Helper to create a test observation."""
        return RLObservation(
            indoor_temp=indoor_temp,
            indoor_temp_entity=EntityState("sensor.temp", 0.0),
            outdoor_temp=10.0,
            outdoor_temp_entity=EntityState("sensor.outdoor", 0.0),
            indoor_humidity=50.0,
            indoor_humidity_entity=EntityState("sensor.humidity", 0.0),
            timestamp=datetime.now(),
            target_temp=target_temp,
            target_temp_entity=EntityState("sensor.target", 0.0),
            time_until_target_minutes=30,
            current_target_achieved_percentage=None,
            is_heating_on=True,
            heating_output_percent=None,
            heating_output_entity=None,
            energy_consumption_recent_kwh=energy_consumption,
            energy_consumption_entity=(
                EntityState("sensor.energy", 0.0) if energy_consumption is not None else None
            ),
            time_heating_on_recent_seconds=None,
            time_heating_on_entity=None,
            indoor_temp_change_15min=None,
            outdoor_temp_change_15min=None,
            day_of_week=1,
            hour_of_day=14,
            outdoor_temp_forecast_1h=None,
            outdoor_temp_forecast_3h=None,
            window_or_door_open=False,
            window_or_door_entity=None,
            device_id=device_id,
        )

    def create_action(
        self, action_type: HeatingActionType = HeatingActionType.TURN_ON
    ) -> RLAction:
        """Helper to create a test action."""
        return RLAction(
            action_type=action_type,
            value=None,
            decision_timestamp=datetime.now(),
        )

    def test_initialization_with_default_config(self):
        """Test calculator initialization with default config."""
        calculator = HeatingRewardCalculator()
        assert calculator is not None

    def test_initialization_with_custom_config(self):
        """Test calculator initialization with custom config."""
        config = RewardConfig(progress_reward_factor=2.0)
        calculator = HeatingRewardCalculator(config)
        assert calculator is not None

    def test_progress_reward_when_approaching_target(self):
        """Test positive reward when temperature approaches target."""
        calculator = HeatingRewardCalculator(
            RewardConfig(
                progress_reward_factor=1.0,
                drift_penalty_factor=0.5,
                overshoot_penalty_factor=0.0,
                energy_penalty_factor=0.0,
            )
        )

        # Temperature moving from 18°C to 19°C (closer to 20°C target)
        prev_state = self.create_observation(indoor_temp=18.0, target_temp=20.0)
        curr_state = self.create_observation(indoor_temp=19.0, target_temp=20.0)
        action = self.create_action()

        reward = calculator.calculate_reward(prev_state, action, curr_state)

        # Distance improved from 2.0 to 1.0 (improvement = 1.0)
        # Reward = 1.0 * 1.0 (progress_reward_factor) = 1.0
        assert reward == pytest.approx(1.0, abs=0.01)

    def test_drift_penalty_when_moving_away_from_target(self):
        """Test negative reward (drift penalty) when temperature moves away from target."""
        calculator = HeatingRewardCalculator(
            RewardConfig(
                progress_reward_factor=1.0,
                drift_penalty_factor=0.5,
                overshoot_penalty_factor=0.0,
                energy_penalty_factor=0.0,
            )
        )

        # Temperature moving from 19°C to 18°C (farther from 20°C target)
        prev_state = self.create_observation(indoor_temp=19.0, target_temp=20.0)
        curr_state = self.create_observation(indoor_temp=18.0, target_temp=20.0)
        action = self.create_action()

        reward = calculator.calculate_reward(prev_state, action, curr_state)

        # Distance worsened from 1.0 to 2.0 (improvement = -1.0)
        # Penalty = -1.0 * 0.5 (drift_penalty_factor) = -0.5
        assert reward == pytest.approx(-0.5, abs=0.01)

    def test_overshoot_penalty_when_exceeding_target(self):
        """Test penalty when temperature exceeds target by more than threshold."""
        calculator = HeatingRewardCalculator(
            RewardConfig(
                progress_reward_factor=0.0,
                drift_penalty_factor=0.0,
                overshoot_penalty_factor=2.0,
                overshoot_threshold_celsius=1.0,
                energy_penalty_factor=0.0,
            )
        )

        # Temperature at 22.5°C with target 20°C
        # Overshoot = 22.5 - 20.0 - 1.0 = 1.5°C above threshold
        prev_state = self.create_observation(indoor_temp=22.0, target_temp=20.0)
        curr_state = self.create_observation(indoor_temp=22.5, target_temp=20.0)
        action = self.create_action()

        reward = calculator.calculate_reward(prev_state, action, curr_state)

        # Overshoot penalty = 1.5 * 2.0 = 3.0
        # Total reward = 0.0 (no progress change) - 3.0 (overshoot) = -3.0
        assert reward == pytest.approx(-3.0, abs=0.01)

    def test_no_overshoot_penalty_within_threshold(self):
        """Test no overshoot penalty when within threshold."""
        calculator = HeatingRewardCalculator(
            RewardConfig(
                progress_reward_factor=0.0,
                drift_penalty_factor=0.0,
                overshoot_penalty_factor=2.0,
                overshoot_threshold_celsius=1.0,
                energy_penalty_factor=0.0,
            )
        )

        # Temperature at 20.5°C with target 20°C (within 1.0°C threshold)
        prev_state = self.create_observation(indoor_temp=20.0, target_temp=20.0)
        curr_state = self.create_observation(indoor_temp=20.5, target_temp=20.0)
        action = self.create_action()

        reward = calculator.calculate_reward(prev_state, action, curr_state)

        # No overshoot penalty since within threshold
        assert reward == pytest.approx(0.0, abs=0.01)

    def test_energy_penalty_for_consumption(self):
        """Test penalty for energy consumption."""
        calculator = HeatingRewardCalculator(
            RewardConfig(
                progress_reward_factor=0.0,
                drift_penalty_factor=0.0,
                overshoot_penalty_factor=0.0,
                energy_penalty_factor=0.1,
            )
        )

        # Energy consumption of 0.5 kWh
        prev_state = self.create_observation(
            indoor_temp=20.0, target_temp=20.0, energy_consumption=0.5
        )
        curr_state = self.create_observation(
            indoor_temp=20.0, target_temp=20.0, energy_consumption=0.5
        )
        action = self.create_action()

        reward = calculator.calculate_reward(prev_state, action, curr_state)

        # Energy penalty = 0.5 * 0.1 = 0.05
        assert reward == pytest.approx(-0.05, abs=0.01)

    def test_no_energy_penalty_when_consumption_is_none(self):
        """Test no energy penalty when consumption data is not available."""
        calculator = HeatingRewardCalculator(
            RewardConfig(
                progress_reward_factor=0.0,
                drift_penalty_factor=0.0,
                overshoot_penalty_factor=0.0,
                energy_penalty_factor=0.1,
            )
        )

        prev_state = self.create_observation(
            indoor_temp=20.0, target_temp=20.0, energy_consumption=None
        )
        curr_state = self.create_observation(
            indoor_temp=20.0, target_temp=20.0, energy_consumption=None
        )
        action = self.create_action()

        reward = calculator.calculate_reward(prev_state, action, curr_state)

        # No energy penalty since consumption is None
        assert reward == pytest.approx(0.0, abs=0.01)

    def test_combined_reward_components(self):
        """Test reward calculation with multiple components."""
        calculator = HeatingRewardCalculator(
            RewardConfig(
                progress_reward_factor=1.0,
                drift_penalty_factor=0.5,
                overshoot_penalty_factor=2.0,
                overshoot_threshold_celsius=1.0,
                energy_penalty_factor=0.1,
            )
        )

        # Scenario: Temperature approaching target but consuming energy
        # 18°C -> 19°C (target 20°C), energy 0.3 kWh
        prev_state = self.create_observation(
            indoor_temp=18.0, target_temp=20.0, energy_consumption=0.3
        )
        curr_state = self.create_observation(
            indoor_temp=19.0, target_temp=20.0, energy_consumption=0.3
        )
        action = self.create_action()

        reward = calculator.calculate_reward(prev_state, action, curr_state)

        # Progress reward = 1.0 * 1.0 = 1.0
        # No overshoot penalty
        # Energy penalty = 0.3 * 0.1 = 0.03
        # Total = 1.0 - 0.03 = 0.97
        assert reward == pytest.approx(0.97, abs=0.01)

    def test_terminal_reward_when_target_achieved(self):
        """Test large positive terminal reward when target is achieved."""
        calculator = HeatingRewardCalculator(
            RewardConfig(
                target_achieved_reward=10.0,
                target_missed_penalty=5.0,
                energy_penalty_factor=0.1,
            )
        )

        final_state = self.create_observation(indoor_temp=20.0, target_temp=20.0)
        reward = calculator.calculate_terminal_reward(
            final_state=final_state,
            target_achieved=True,
            episode_duration_minutes=30.0,
            total_energy_consumed_kwh=1.0,
        )

        # Target achieved reward = 10.0
        # Energy penalty = 1.0 * 0.1 = 0.1
        # Total = 10.0 - 0.1 = 9.9
        assert reward == pytest.approx(9.9, abs=0.01)

    def test_terminal_reward_when_target_missed(self):
        """Test large negative terminal reward when target is missed."""
        calculator = HeatingRewardCalculator(
            RewardConfig(
                target_achieved_reward=10.0,
                target_missed_penalty=5.0,
                energy_penalty_factor=0.1,
            )
        )

        final_state = self.create_observation(indoor_temp=18.0, target_temp=20.0)
        reward = calculator.calculate_terminal_reward(
            final_state=final_state,
            target_achieved=False,
            episode_duration_minutes=30.0,
            total_energy_consumed_kwh=0.8,
        )

        # Target missed penalty = -5.0
        # Energy penalty = 0.8 * 0.1 = 0.08
        # Total = -5.0 - 0.08 = -5.08
        assert reward == pytest.approx(-5.08, abs=0.01)

    def test_terminal_reward_with_zero_energy(self):
        """Test terminal reward with zero energy consumption."""
        calculator = HeatingRewardCalculator(
            RewardConfig(
                target_achieved_reward=10.0,
                energy_penalty_factor=0.1,
            )
        )

        final_state = self.create_observation(indoor_temp=20.0, target_temp=20.0)
        reward = calculator.calculate_terminal_reward(
            final_state=final_state,
            target_achieved=True,
            episode_duration_minutes=30.0,
            total_energy_consumed_kwh=0.0,
        )

        # Target achieved reward = 10.0
        # No energy penalty
        # Total = 10.0
        assert reward == pytest.approx(10.0, abs=0.01)

    def test_reward_symmetry_for_opposite_temperature_changes(self):
        """Test that reward/penalty is symmetric for opposite temperature changes."""
        calculator = HeatingRewardCalculator(
            RewardConfig(
                progress_reward_factor=1.0,
                drift_penalty_factor=1.0,  # Same as progress for symmetry
                overshoot_penalty_factor=0.0,
                energy_penalty_factor=0.0,
            )
        )

        # Approaching target: 18°C -> 19°C (target 20°C)
        prev_state1 = self.create_observation(indoor_temp=18.0, target_temp=20.0)
        curr_state1 = self.create_observation(indoor_temp=19.0, target_temp=20.0)
        action = self.create_action()
        reward_approach = calculator.calculate_reward(prev_state1, action, curr_state1)

        # Moving away from target: 19°C -> 18°C (target 20°C)
        prev_state2 = self.create_observation(indoor_temp=19.0, target_temp=20.0)
        curr_state2 = self.create_observation(indoor_temp=18.0, target_temp=20.0)
        reward_drift = calculator.calculate_reward(prev_state2, action, curr_state2)

        # Rewards should be opposite in sign and equal in magnitude
        assert reward_approach == pytest.approx(-reward_drift, abs=0.01)
