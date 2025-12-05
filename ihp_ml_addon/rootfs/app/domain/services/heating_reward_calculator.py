"""Heating reward calculator service.

Domain service for calculating rewards in RL heating control.
"""

import logging

from domain.interfaces.reward_calculator import IRewardCalculator
from domain.value_objects import RewardConfig, RLAction, RLObservation

logger = logging.getLogger(__name__)


class HeatingRewardCalculator(IRewardCalculator):
    """Calculate rewards for heating control actions.

    This service implements dense reward shaping to guide the RL agent:
    - Progress rewards: Small positive rewards for approaching target
    - Drift penalties: Small penalties for moving away from target
    - Overshoot penalties: Penalties for exceeding target temperature
    - Energy penalties: Penalties proportional to energy consumption
    - Terminal rewards: Large rewards/penalties at episode end
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        """Initialize the reward calculator.

        Args:
            config: Reward configuration. If None, uses default values.
        """
        self._config = config or RewardConfig()
        logger.info(
            "Initialized HeatingRewardCalculator with config: "
            f"progress_factor={self._config.progress_reward_factor}, "
            f"drift_penalty={self._config.drift_penalty_factor}, "
            f"overshoot_penalty={self._config.overshoot_penalty_factor}, "
            f"energy_penalty={self._config.energy_penalty_factor}"
        )

    def calculate_reward(
        self,
        previous_state: RLObservation,
        action: RLAction,
        current_state: RLObservation,
    ) -> float:
        """Calculate the reward for a state transition.

        Implements dense reward shaping:
        1. Progress reward: Positive reward when approaching target
        2. Drift penalty: Penalty when moving away from target
        3. Overshoot penalty: Extra penalty when exceeding target
        4. Energy penalty: Penalty for energy consumption

        Args:
            previous_state: The observation state before the action
            action: The action taken
            current_state: The observation state after the action

        Returns:
            Reward value (can be positive or negative)
        """
        logger.debug(
            f"Calculating reward for device {current_state.device_id}, "
            f"action={action.action_type}, "
            f"prev_temp={previous_state.indoor_temp:.2f}, "
            f"curr_temp={current_state.indoor_temp:.2f}, "
            f"target={current_state.target_temp:.2f}"
        )

        reward = 0.0

        # 1. Progress/Drift reward component
        progress_reward = self._calculate_progress_reward(
            previous_state, current_state
        )
        reward += progress_reward
        logger.debug(f"Progress reward: {progress_reward:.3f}")

        # 2. Overshoot penalty component
        overshoot_penalty = self._calculate_overshoot_penalty(current_state)
        reward -= overshoot_penalty
        logger.debug(f"Overshoot penalty: {overshoot_penalty:.3f}")

        # 3. Energy consumption penalty
        energy_penalty = self._calculate_energy_penalty(current_state)
        reward -= energy_penalty
        logger.debug(f"Energy penalty: {energy_penalty:.3f}")

        logger.debug(f"Total intermediate reward: {reward:.3f}")
        return reward

    def calculate_terminal_reward(
        self,
        final_state: RLObservation,
        total_energy_consumed_kwh: float,
    ) -> float:
        """Calculate the terminal reward at the end of an episode.

        The target is considered achieved when:
        - Temperature is within tolerance of target_temp
        - At the scheduled time (time_until_target_minutes == 0)

        Timing penalties:
        - If time_until_target_minutes > 0: Target not reached on time (late) - higher penalty
        - If time_until_target_minutes < 0: Target reached too early - lower penalty
        - If time_until_target_minutes == 0: Perfect timing

        Args:
            final_state: The final observation state
            total_energy_consumed_kwh: Total energy consumed during episode

        Returns:
            Terminal reward value
        """
        # Determine if target temperature is achieved
        temp_diff = abs(final_state.indoor_temp - final_state.target_temp)
        temp_achieved = temp_diff <= self._config.target_tolerance_celsius

        # Determine timing (negative = early, 0 = on time, positive = late)
        time_delta = final_state.time_until_target_minutes

        logger.info(
            f"Calculating terminal reward for device {final_state.device_id}: "
            f"temp_achieved={temp_achieved}, "
            f"temp_diff={temp_diff:.2f}°C, "
            f"time_delta={time_delta:.1f}min, "
            f"energy={total_energy_consumed_kwh:.3f}kWh"
        )

        reward = 0.0

        if temp_achieved and time_delta == 0:
            # Perfect: target reached at exactly the right time
            reward += self._config.target_achieved_reward
            logger.debug(
                f"Target achieved on time reward: {self._config.target_achieved_reward:.3f}"
            )
        elif temp_achieved and time_delta < 0:
            # Target reached early - base reward minus early penalty
            reward += self._config.target_achieved_reward
            early_penalty = abs(time_delta) * self._config.early_achievement_penalty_factor
            reward -= early_penalty
            logger.debug(
                f"Target achieved early: base_reward={self._config.target_achieved_reward:.3f}, "
                f"early_penalty={early_penalty:.3f}"
            )
        elif temp_achieved and time_delta > 0:
            # Target reached late - base reward minus late penalty
            reward += self._config.target_achieved_reward
            late_penalty = time_delta * self._config.late_achievement_penalty_factor
            reward -= late_penalty
            logger.debug(
                f"Target achieved late: base_reward={self._config.target_achieved_reward:.3f}, "
                f"late_penalty={late_penalty:.3f}"
            )
        else:
            # Temperature target not achieved
            if time_delta > 0:
                # Still waiting for target - severe penalty
                missed_penalty = self._config.target_missed_penalty
                late_penalty = time_delta * self._config.late_achievement_penalty_factor
                reward -= (missed_penalty + late_penalty)
                logger.debug(
                    f"Target missed and late: missed_penalty={missed_penalty:.3f}, "
                    f"late_penalty={late_penalty:.3f}"
                )
            else:
                # Target time passed but temperature not achieved - base penalty
                reward -= self._config.target_missed_penalty
                logger.debug(
                    f"Target missed: penalty={self._config.target_missed_penalty:.3f}"
                )

        # Additional energy penalty for total consumption
        total_energy_penalty = (
            total_energy_consumed_kwh * self._config.energy_penalty_factor
        )
        reward -= total_energy_penalty
        logger.debug(f"Total energy penalty: {total_energy_penalty:.3f}")

        logger.info(f"Total terminal reward: {reward:.3f}")
        return reward

    def _calculate_progress_reward(
        self, previous_state: RLObservation, current_state: RLObservation
    ) -> float:
        """Calculate reward based on progress towards target.

        Positive reward if temperature is approaching target,
        negative reward (drift penalty) if moving away.

        Args:
            previous_state: Previous observation
            current_state: Current observation

        Returns:
            Progress reward (positive) or drift penalty (negative)
        """
        # Distance to target before and after
        prev_distance = abs(previous_state.indoor_temp - previous_state.target_temp)
        curr_distance = abs(current_state.indoor_temp - current_state.target_temp)

        # Distance improvement (positive if closer, negative if farther)
        distance_improvement = prev_distance - curr_distance

        if distance_improvement > 0:
            # Temperature is approaching target - positive reward
            reward = distance_improvement * self._config.progress_reward_factor
        else:
            # Temperature is moving away from target - drift penalty
            reward = distance_improvement * self._config.drift_penalty_factor

        return reward

    def _calculate_overshoot_penalty(self, current_state: RLObservation) -> float:
        """Calculate penalty for overshooting the target temperature.

        Args:
            current_state: Current observation

        Returns:
            Overshoot penalty (always non-negative)
        """
        # Check if temperature exceeds target by more than overshoot threshold
        overshoot = (
            current_state.indoor_temp
            - current_state.target_temp
            - self._config.overshoot_threshold_celsius
        )

        if overshoot > 0:
            # Penalty proportional to how much we overshoot
            penalty = overshoot * self._config.overshoot_penalty_factor
            return penalty

        return 0.0

    def _calculate_energy_penalty(self, current_state: RLObservation) -> float:
        """Calculate penalty for energy consumption.

        Args:
            current_state: Current observation

        Returns:
            Energy penalty (always non-negative)
        """
        # Penalty based on recent energy consumption
        if current_state.energy_consumption_recent_kwh is not None:
            penalty = (
                current_state.energy_consumption_recent_kwh
                * self._config.energy_penalty_factor
            )
            return penalty

        return 0.0
