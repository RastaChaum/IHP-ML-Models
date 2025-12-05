"""RL model trainer interface.

Contract for training reinforcement learning models.
"""

from abc import ABC, abstractmethod

from domain.value_objects import ModelInfo, RLExperience, TrainingRequest


class IRLModelTrainer(ABC):
    """Contract for RL model training operations.

    This interface defines how RL models are trained, including:
    - Behavioral cloning from historical data
    - Online learning from new experiences
    - Model updates and persistence
    """

    @abstractmethod
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
            TrainingError: If training fails
        """
        pass

    @abstractmethod
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
            TrainingError: If training fails
        """
        pass

    @abstractmethod
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
            ModelNotFoundError: If the model doesn't exist
            ValueError: If experiences are invalid
            TrainingError: If update fails
        """
        pass

    @abstractmethod
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
            ModelNotFoundError: If the model doesn't exist
            ValueError: If experiences are invalid
            TrainingError: If training fails
        """
        pass
