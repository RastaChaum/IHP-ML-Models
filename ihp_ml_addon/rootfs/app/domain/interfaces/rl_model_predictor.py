"""RL model predictor interface.

Contract for making predictions (action selection) with RL models.
"""

from abc import ABC, abstractmethod

from domain.value_objects import RLAction, RLObservation


class IRLModelPredictor(ABC):
    """Contract for RL model prediction/action selection operations.

    This interface defines how RL models select actions given observations.
    The predictor can operate in different modes:
    - Exploitation (greedy): Select the best action
    - Exploration: Select actions with some randomness for learning
    """

    @abstractmethod
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
            PredictionError: If action selection fails
        """
        pass

    @abstractmethod
    async def has_trained_model(self, device_id: str | None = None) -> bool:
        """Check if a trained RL model is available.

        Args:
            device_id: Optional device identifier to check for device-specific model

        Returns:
            True if at least one trained RL model exists for the device
        """
        pass

    @abstractmethod
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
            PredictionError: If value computation fails
        """
        pass
