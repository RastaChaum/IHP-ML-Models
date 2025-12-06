"""Stable-Baselines3 RL predictor adapter.

Infrastructure adapter for making predictions with trained RL models.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from domain.interfaces import IModelStorage, IRLModelPredictor
from domain.value_objects import HeatingActionType, RLAction, RLObservation
from stable_baselines3 import PPO

_LOGGER = logging.getLogger(__name__)


class StableBaselines3RLPredictor(IRLModelPredictor):
    """Stable-Baselines3 implementation of RL model predictor.

    This adapter uses trained PPO models to select actions for heating control.
    """

    def __init__(
        self,
        model_storage: IModelStorage,
        models_dir: str = "/data/models",
    ) -> None:
        """Initialize the SB3 RL predictor.

        Args:
            model_storage: Storage interface for loading model info
            models_dir: Directory where model files are stored
        """
        self._model_storage = model_storage
        self._models_dir = Path(models_dir)
        self._loaded_models: dict[str, PPO] = {}  # Cache for loaded models

        _LOGGER.info(
            "Initialized StableBaselines3RLPredictor with models_dir=%s",
            self._models_dir,
        )

    async def select_action(
        self,
        observation: RLObservation,
        model_id: str | None = None,
        explore: bool = False,
    ) -> RLAction:
        """Select an action given an observation.

        Args:
            observation: Current observation state
            model_id: Optional model identifier (uses latest for device if not specified)
            explore: Whether to use exploration (for learning) or exploitation (for deployment)

        Returns:
            RLAction to take

        Raises:
            ValueError: If no trained model is available
        """
        _LOGGER.debug(
            "Selecting action for device %s (model_id=%s, explore=%s)",
            observation.device_id,
            model_id,
            explore,
        )

        # Get model ID if not provided
        if model_id is None:
            # Load latest model for this device
            model_id = await self._model_storage.get_latest_model_id_for_device(
                observation.device_id
            )
            if model_id is None:
                raise ValueError(
                    f"No trained model found for device {observation.device_id}"
                )

        # Load model if not already cached
        if model_id not in self._loaded_models:
            model_path = self._models_dir / f"{model_id}.zip"
            if not model_path.exists():
                raise ValueError(f"Model {model_id} not found at {model_path}")

            _LOGGER.info("Loading model from %s", model_path)
            self._loaded_models[model_id] = PPO.load(str(model_path))

        model = self._loaded_models[model_id]

        # Convert observation to numpy array
        obs_array = self._observation_to_array(observation)

        # Get action from model
        action_idx, _states = model.predict(obs_array, deterministic=not explore)

        _LOGGER.debug("Model predicted action index: %d", action_idx)

        # Map action index to heating action type
        # 0=TURN_OFF, 1=TURN_ON, 2=NO_OP
        action_type = self._index_to_action_type(int(action_idx))

        # For now, use target_temp as the value
        # In a more sophisticated implementation, we could learn the setpoint
        action = RLAction(
            action_type=action_type,
            value=observation.target_temp,
            decision_timestamp=datetime.now(),
            confidence_score=None,  # PPO doesn't directly provide confidence
        )

        _LOGGER.debug(
            "Selected action: %s (value=%.1f)",
            action.action_type,
            action.value,
        )

        return action

    async def has_trained_model(self, device_id: str | None = None) -> bool:
        """Check if a trained RL model is available.

        Args:
            device_id: Optional device identifier to check for device-specific model

        Returns:
            True if at least one trained RL model exists for the device
        """
        # Check if any .zip files exist in models directory
        model_files = list(self._models_dir.glob("*.zip"))

        if device_id is None:
            return len(model_files) > 0

        # Check for device-specific models
        device_models = [f for f in model_files if f.stem.startswith(f"{device_id}_")]
        return len(device_models) > 0

    async def get_action_values(
        self,
        observation: RLObservation,
        model_id: str | None = None,
    ) -> dict[str, float]:
        """Get the Q-values or action probabilities for all possible actions.

        This method provides insight into the model's decision-making process
        by returning the values/probabilities for each action.

        Args:
            observation: Current observation state
            model_id: Optional model identifier (uses latest for device if not specified)

        Returns:
            Dictionary mapping action types to their values/probabilities

        Raises:
            ValueError: If no trained model is available
        """
        # Get model ID if not provided
        if model_id is None:
            # Load latest model for this device
            model_id = await self._model_storage.get_latest_model_id_for_device(
                observation.device_id
            )
            if model_id is None:
                raise ValueError(
                    f"No trained model found for device {observation.device_id}"
                )

        # Load model if not already cached
        if model_id not in self._loaded_models:
            model_path = self._models_dir / f"{model_id}.zip"
            if not model_path.exists():
                raise ValueError(f"Model {model_id} not found at {model_path}")

            _LOGGER.info("Loading model from %s", model_path)
            self._loaded_models[model_id] = PPO.load(str(model_path))

        model = self._loaded_models[model_id]

        # Convert observation to numpy array
        obs_array = self._observation_to_array(observation)

        # Get action probabilities from the policy
        # This is PPO-specific and uses the policy's action distribution
        obs_tensor = model.policy.obs_to_tensor(obs_array)[0]
        distribution = model.policy.get_distribution(obs_tensor)
        action_probs = distribution.distribution.probs.detach().cpu().numpy()[0]

        # Map to action types
        action_values = {
            "TURN_OFF": float(action_probs[0]),
            "TURN_ON": float(action_probs[1]),
            "NO_OP": float(action_probs[2]),
        }

        _LOGGER.debug("Action probabilities: %s", action_values)

        return action_values

    def _observation_to_array(self, obs: RLObservation) -> np.ndarray:
        """Convert RLObservation to numpy array.

        Args:
            obs: RL observation to convert

        Returns:
            Numpy array representation of the observation
        """
        return np.array(
            [
                obs.indoor_temp,
                obs.outdoor_temp if obs.outdoor_temp is not None else -50.0,
                obs.indoor_humidity if obs.indoor_humidity is not None else 0.0,
                obs.target_temp,
                float(obs.time_until_target_minutes),
                (
                    obs.current_target_achieved_percentage
                    if obs.current_target_achieved_percentage is not None
                    else 0.0
                ),
                1.0 if obs.is_heating_on else 0.0,
                obs.heating_output_percent if obs.heating_output_percent is not None else 0.0,
                (
                    obs.energy_consumption_recent_kwh
                    if obs.energy_consumption_recent_kwh is not None
                    else 0.0
                ),
                (
                    float(obs.time_heating_on_recent_seconds)
                    if obs.time_heating_on_recent_seconds is not None
                    else 0.0
                ),
                obs.indoor_temp_change_15min if obs.indoor_temp_change_15min is not None else 0.0,
                obs.outdoor_temp_change_15min if obs.outdoor_temp_change_15min is not None else 0.0,
                float(obs.day_of_week),
                float(obs.hour_of_day),
                obs.outdoor_temp_forecast_1h if obs.outdoor_temp_forecast_1h is not None else -25.0,
                obs.outdoor_temp_forecast_3h if obs.outdoor_temp_forecast_3h is not None else -25.0,
                1.0 if obs.window_or_door_open else 0.0,
            ],
            dtype=np.float32,
        )

    def _index_to_action_type(self, action_idx: int) -> HeatingActionType:
        """Map action index to HeatingActionType.

        Args:
            action_idx: Action index from model (0=TURN_OFF, 1=TURN_ON, 2=NO_OP)

        Returns:
            Corresponding HeatingActionType
        """
        if action_idx == 0:
            return HeatingActionType.TURN_OFF
        elif action_idx == 1:
            return HeatingActionType.TURN_ON
        elif action_idx == 2:
            return HeatingActionType.NO_OP
        else:
            raise ValueError(f"Invalid action index: {action_idx}")
