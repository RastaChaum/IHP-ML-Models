"""Gymnasium environment for heating control.

Custom Gymnasium environment that wraps heating control as an RL environment.
"""

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from domain.value_objects import (
    RLExperience,
    RLObservation,
)
from gymnasium import spaces

_LOGGER = logging.getLogger(__name__)

# Default values for missing/optional sensor data
DEFAULT_OUTDOOR_TEMP = -50.0  # Placeholder for missing outdoor temperature
DEFAULT_HUMIDITY = 0.0  # Placeholder for missing humidity
DEFAULT_HEATING_OUTPUT = 0.0  # Placeholder for missing heating output percentage
DEFAULT_ENERGY = 0.0  # Placeholder for missing energy consumption
DEFAULT_HEATING_TIME = 0.0  # Placeholder for missing heating on time
DEFAULT_TEMP_CHANGE = 0.0  # Placeholder for missing temperature change
DEFAULT_FORECAST_TEMP = -25.0  # Placeholder for missing forecast temperature


class HeatingEnvironment(gym.Env):
    """Gymnasium environment for heating control.

    This environment wraps heating control as a standard RL environment
    that can be used with Stable-Baselines3. It uses historical experiences
    to replay the environment dynamics.

    The observation space includes all features from RLObservation.
    The action space is discrete with actions corresponding to HeatingActionType.
    """

    def __init__(self, experiences: tuple[RLExperience, ...]) -> None:
        """Initialize the heating environment.

        Args:
            experiences: Tuple of historical experiences to replay
        """
        super().__init__()

        if not experiences:
            raise ValueError("Cannot create environment with empty experiences")

        self._experiences = experiences
        self._current_index = 0
        self._episode_rewards: list[float] = []

        _LOGGER.info("Initialized HeatingEnvironment with %d experiences", len(experiences))

        # Define observation space - all continuous features from RLObservation
        # We'll use a Box space with reasonable bounds for each feature
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    -25.0,  # indoor_temp
                    DEFAULT_OUTDOOR_TEMP,  # outdoor_temp (can be None, use default as placeholder)
                    DEFAULT_HUMIDITY,  # indoor_humidity (can be None, use 0 as placeholder)
                    0.0,  # target_temp
                    -1440.0,  # time_until_target_minutes (can be negative/early)
                    0.0,  # current_target_achieved_percentage
                    0.0,  # is_heating_on (0=off, 1=on)
                    DEFAULT_HEATING_OUTPUT,  # heating_output_percent
                    DEFAULT_ENERGY,  # energy_consumption_recent_kwh
                    DEFAULT_HEATING_TIME,  # time_heating_on_recent_seconds
                    -10.0,  # indoor_temp_change_15min
                    -10.0,  # outdoor_temp_change_15min
                    0.0,  # day_of_week (0-6)
                    0.0,  # hour_of_day (0-23)
                    DEFAULT_FORECAST_TEMP,  # outdoor_temp_forecast_1h
                    DEFAULT_FORECAST_TEMP,  # outdoor_temp_forecast_3h
                    0.0,  # window_or_door_open (0=closed, 1=open)
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    50.0,  # indoor_temp
                    60.0,  # outdoor_temp
                    100.0,  # indoor_humidity
                    50.0,  # target_temp
                    1440.0,  # time_until_target_minutes (can be positive/late)
                    100.0,  # current_target_achieved_percentage
                    1.0,  # is_heating_on
                    100.0,  # heating_output_percent
                    100.0,  # energy_consumption_recent_kwh (reasonable upper bound)
                    86400.0,  # time_heating_on_recent_seconds (24 hours max)
                    10.0,  # indoor_temp_change_15min
                    10.0,  # outdoor_temp_change_15min
                    6.0,  # day_of_week
                    23.0,  # hour_of_day
                    50.0,  # outdoor_temp_forecast_1h
                    50.0,  # outdoor_temp_forecast_3h
                    1.0,  # window_or_door_open
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # Define action space - discrete actions corresponding to HeatingActionType
        # We simplify to 3 actions: TURN_OFF, TURN_ON, NO_OP
        self.action_space = spaces.Discrete(3)

        _LOGGER.debug("Observation space: %s", self.observation_space)
        _LOGGER.debug("Action space: %s", self.action_space)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to start a new episode.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            Tuple of (initial observation, info dict)
        """
        super().reset(seed=seed)

        # Start from the beginning of experiences
        self._current_index = 0
        self._episode_rewards = []

        initial_obs = self._observation_to_array(self._experiences[0].state)

        _LOGGER.debug("Environment reset. Starting at index 0.")

        return initial_obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Action to take (0=TURN_OFF, 1=TURN_ON, 2=NO_OP)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self._current_index >= len(self._experiences):
            raise RuntimeError("Episode already terminated. Call reset() first.")

        # Get current experience
        experience = self._experiences[self._current_index]

        # Get reward from experience (historical reward)
        reward = experience.reward
        self._episode_rewards.append(reward)

        # Get next observation
        next_obs = self._observation_to_array(experience.next_state)

        # Check if episode is done
        terminated = experience.done
        truncated = False  # We don't use truncation in this environment

        # Move to next experience
        self._current_index += 1

        # Check if we've exhausted all experiences
        if self._current_index >= len(self._experiences):
            terminated = True

        info = {
            "episode_reward": sum(self._episode_rewards) if terminated else None,
            "episode_length": len(self._episode_rewards),
        }

        _LOGGER.debug(
            "Step %d: action=%d, reward=%.3f, done=%s",
            self._current_index,
            action,
            reward,
            terminated,
        )

        return next_obs, reward, terminated, truncated, info

    def _observation_to_array(self, obs: RLObservation) -> np.ndarray:
        """Convert RLObservation to numpy array for the environment.

        Args:
            obs: RL observation to convert

        Returns:
            Numpy array representation of the observation
        """
        return np.array(
            [
                obs.indoor_temp,
                obs.outdoor_temp if obs.outdoor_temp is not None else DEFAULT_OUTDOOR_TEMP,
                obs.indoor_humidity if obs.indoor_humidity is not None else DEFAULT_HUMIDITY,
                obs.target_temp,
                float(obs.time_until_target_minutes),
                (
                    obs.current_target_achieved_percentage
                    if obs.current_target_achieved_percentage is not None
                    else 0.0
                ),
                1.0 if obs.is_heating_on else 0.0,
                obs.heating_output_percent if obs.heating_output_percent is not None else DEFAULT_HEATING_OUTPUT,
                (
                    obs.energy_consumption_recent_kwh
                    if obs.energy_consumption_recent_kwh is not None
                    else DEFAULT_ENERGY
                ),
                (
                    float(obs.time_heating_on_recent_seconds)
                    if obs.time_heating_on_recent_seconds is not None
                    else DEFAULT_HEATING_TIME
                ),
                obs.indoor_temp_change_15min if obs.indoor_temp_change_15min is not None else DEFAULT_TEMP_CHANGE,
                obs.outdoor_temp_change_15min if obs.outdoor_temp_change_15min is not None else DEFAULT_TEMP_CHANGE,
                float(obs.day_of_week),
                float(obs.hour_of_day),
                obs.outdoor_temp_forecast_1h if obs.outdoor_temp_forecast_1h is not None else DEFAULT_FORECAST_TEMP,
                obs.outdoor_temp_forecast_3h if obs.outdoor_temp_forecast_3h is not None else DEFAULT_FORECAST_TEMP,
                1.0 if obs.window_or_door_open else 0.0,
            ],
            dtype=np.float32,
        )
