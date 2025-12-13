"""Stable-Baselines3 RL trainer adapter.

Infrastructure adapter for training RL models using Stable-Baselines3.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from domain.interfaces import (
    IExperienceReplayBuffer,
    IHomeAssistantHistoryReader,
    IModelStorage,
    IRLModelTrainer,
)
from domain.value_objects import ModelInfo, RLExperience, TrainingRequest
from infrastructure.adapters.gymnasium_heating_env import HeatingEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

_LOGGER = logging.getLogger(__name__)


class TrainingProgressCallback(BaseCallback):
    """Callback to log training progress."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

    def _on_step(self) -> bool:
        # Check if an episode ended
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done and "episode" in self.locals["infos"][i]:
                    episode_reward = self.locals["infos"][i]["episode"]["r"]
                    episode_length = self.locals["infos"][i]["episode"]["l"]
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    _LOGGER.info(
                        "Episode finished: reward=%.2f, length=%d",
                        episode_reward,
                        episode_length,
                    )
        return True


class StableBaselines3RLTrainer(IRLModelTrainer):
    """Stable-Baselines3 implementation of RL model trainer.

    This adapter uses the PPO (Proximal Policy Optimization) algorithm
    from Stable-Baselines3 to train RL agents for heating control.
    """

    def __init__(
        self,
        model_storage: IModelStorage,
        history_reader: IHomeAssistantHistoryReader,
        replay_buffer: IExperienceReplayBuffer | None = None,
        models_dir: str = "/data/models",
    ) -> None:
        """Initialize the SB3 RL trainer.

        Args:
            model_storage: Storage interface for persisting trained models
            history_reader: Reader interface for fetching historical data
            replay_buffer: Optional replay buffer for online learning
            models_dir: Directory for storing model files
        """
        self._model_storage = model_storage
        self._history_reader = history_reader
        self._replay_buffer = replay_buffer
        self._models_dir = Path(models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)

        _LOGGER.info(
            "Initialized StableBaselines3RLTrainer with models_dir=%s",
            self._models_dir,
        )

    def _generate_model_id(self, device_id: str) -> str:
        """Generate a unique model ID for a device.

        Args:
            device_id: Device/zone identifier

        Returns:
            Unique model identifier with timestamp
        """
        return f"{device_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    async def train_from_experiences(
        self,
        experiences: tuple[RLExperience, ...],
        device_id: str,
        use_behavioral_cloning: bool = True,
    ) -> ModelInfo:
        """Train a new RL model from a collection of experiences.

        This method trains a new RL agent using historical experiences.
        If behavioral_cloning is enabled, it uses imitation learning
        to bootstrap the policy before RL training.

        Args:
            experiences: Tuple of historical experiences
            device_id: Device/zone identifier for model association
            use_behavioral_cloning: Whether to use behavioral cloning for initialization

        Returns:
            ModelInfo with details about the trained model

        Raises:
            ValueError: If experiences are invalid or insufficient
        """
        _LOGGER.info(
            "Training RL model for device %s with %d experiences (behavioral_cloning=%s)",
            device_id,
            len(experiences),
            use_behavioral_cloning,
        )

        if not experiences:
            raise ValueError("Cannot train with empty experiences")

        # Create Gymnasium environment from experiences
        env = HeatingEnvironment(experiences)

        # Initialize PPO model
        # We use a simple MLP policy with 2 layers of 64 units each
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            tensorboard_log=None,  # Disable tensorboard for now
        )

        _LOGGER.info("PPO model initialized")

        # Train the model
        # Number of timesteps depends on the number of experiences
        total_timesteps = len(experiences) * 2  # Train for 2 full passes through the data

        callback = TrainingProgressCallback(verbose=1)

        _LOGGER.info("Starting training for %d timesteps...", total_timesteps)
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
        )

        _LOGGER.info("Training completed successfully")

        # Save the model
        model_id = self._generate_model_id(device_id)
        model_path = self._models_dir / f"{model_id}.zip"
        model.save(str(model_path))

        _LOGGER.info("Model saved to %s", model_path)

        # Store model in the domain storage
        model_info = ModelInfo(
            model_id=model_id,
            device_id=device_id,
            training_date=datetime.now(),
            model_type="RL_PPO",
            metrics={
                "num_experiences": len(experiences),
                "total_timesteps": total_timesteps,
                "avg_episode_reward": (
                    sum(callback.episode_rewards) / len(callback.episode_rewards)
                    if callback.episode_rewards
                    else 0.0
                ),
                "avg_episode_length": (
                    sum(callback.episode_lengths) / len(callback.episode_lengths)
                    if callback.episode_lengths
                    else 0
                ),
            },
        )

        # Save model info via storage interface
        await self._model_storage.save_model_info(model_info)

        _LOGGER.info("Model info saved: %s", model_info)

        return model_info

    async def train_from_request(
        self,
        training_request: TrainingRequest,
    ) -> ModelInfo:
        """Train a new RL model from a training request.

        This method is a convenience wrapper that fetches historical data
        and converts it to experiences before training.

        Args:
            training_request: Training configuration with entity IDs and time range

        Returns:
            ModelInfo with details about the trained model

        Raises:
            ValueError: If training request is invalid
            ConnectionError: If unable to fetch historical data
        """
        _LOGGER.info(
            "Training RL model from request for device %s",
            training_request.device_id,
        )

        # Fetch experiences from history reader
        experiences = await self._history_reader.fetch_rl_experiences(training_request)

        _LOGGER.info("Fetched %d experiences from history", len(experiences))

        # Store experiences in replay buffer if available
        if self._replay_buffer:
            await self._replay_buffer.add_batch(experiences)
            _LOGGER.info("Experiences added to replay buffer")

        # Train the model
        return await self.train_from_experiences(
            experiences,
            training_request.device_id,
            use_behavioral_cloning=True,
        )

    async def update_online(
        self,
        model_id: str,
        new_experiences: tuple[RLExperience, ...],
    ) -> ModelInfo:
        """Update an existing model with new experiences (online learning).

        This method performs incremental learning by updating the model
        with recent experiences without full retraining.

        Args:
            model_id: Identifier of the model to update
            new_experiences: New experiences to learn from

        Returns:
            ModelInfo with details about the updated model

        Raises:
            ValueError: If experiences are invalid
        """
        _LOGGER.info(
            "Updating model %s with %d new experiences",
            model_id,
            len(new_experiences),
        )

        if not new_experiences:
            raise ValueError("Cannot update with empty experiences")

        # Load the existing model
        model_path = self._models_dir / f"{model_id}.zip"
        if not model_path.exists():
            raise ValueError(f"Model {model_id} not found at {model_path}")

        _LOGGER.info("Loading existing model from %s", model_path)
        model = PPO.load(str(model_path))

        # Create environment from new experiences
        env = HeatingEnvironment(new_experiences)
        model.set_env(env)

        # Continue training with new experiences
        total_timesteps = len(new_experiences)
        callback = TrainingProgressCallback(verbose=1)

        _LOGGER.info("Continuing training for %d timesteps...", total_timesteps)
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False,  # Continue from previous training
            progress_bar=False,
        )

        # Save the updated model (overwrite)
        model.save(str(model_path))

        _LOGGER.info("Updated model saved to %s", model_path)

        # Load existing model info and update
        model_info = await self._model_storage.load_model_info(model_id)

        # Update metrics
        updated_metrics = model_info.metrics.copy()
        updated_metrics["num_experiences"] = (
            updated_metrics.get("num_experiences", 0) + len(new_experiences)
        )
        updated_metrics["total_timesteps"] = (
            updated_metrics.get("total_timesteps", 0) + total_timesteps
        )
        updated_metrics["last_update"] = datetime.now().isoformat()
        updated_metrics["avg_episode_reward"] = (
            sum(callback.episode_rewards) / len(callback.episode_rewards)
            if callback.episode_rewards
            else updated_metrics.get("avg_episode_reward", 0.0)
        )

        # Create updated model info
        updated_model_info = ModelInfo(
            model_id=model_id,
            device_id=model_info.device_id,
            training_date=model_info.training_date,
            model_type=model_info.model_type,
            metrics=updated_metrics,
        )

        # Save updated model info
        await self._model_storage.save_model_info(updated_model_info)

        _LOGGER.info("Updated model info saved")

        return updated_model_info

    async def retrain(
        self,
        model_id: str,
        experiences: tuple[RLExperience, ...],
    ) -> ModelInfo:
        """Fully retrain an existing model with new data.

        This method performs a complete retraining of the model,
        replacing the old policy entirely.

        Args:
            model_id: Identifier of the model to retrain
            experiences: All experiences to train on

        Returns:
            ModelInfo with details about the retrained model

        Raises:
            ValueError: If experiences are invalid
        """
        _LOGGER.info(
            "Retraining model %s with %d experiences",
            model_id,
            len(experiences),
        )

        # Load existing model info to get device_id
        model_info = await self._model_storage.load_model_info(model_id)
        device_id = model_info.device_id

        if not experiences:
            raise ValueError("Cannot retrain with empty experiences")

        # Create Gymnasium environment from experiences
        env = HeatingEnvironment(experiences)

        # Initialize new PPO model
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            tensorboard_log=None,
        )

        _LOGGER.info("PPO model initialized for retraining")

        # Train the model
        total_timesteps = len(experiences) * 2
        callback = TrainingProgressCallback(verbose=1)

        _LOGGER.info("Starting retraining for %d timesteps...", total_timesteps)
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
        )

        # Save the model with the SAME model_id
        model_path = self._models_dir / f"{model_id}.zip"
        model.save(str(model_path))

        _LOGGER.info("Retrained model saved to %s", model_path)

        # Create updated model info (keep original model_id and training_date)
        updated_model_info = ModelInfo(
            model_id=model_id,
            device_id=device_id,
            training_date=model_info.training_date,
            model_type="RL_PPO",
            metrics={
                "num_experiences": len(experiences),
                "total_timesteps": total_timesteps,
                "retrained_at": datetime.now().isoformat(),
                "avg_episode_reward": (
                    sum(callback.episode_rewards) / len(callback.episode_rewards)
                    if callback.episode_rewards
                    else 0.0
                ),
                "avg_episode_length": (
                    sum(callback.episode_lengths) / len(callback.episode_lengths)
                    if callback.episode_lengths
                    else 0
                ),
            },
        )

        # Save model info
        await self._model_storage.save_model_info(updated_model_info)

        _LOGGER.info("Model info saved after retraining")

        return updated_model_info
