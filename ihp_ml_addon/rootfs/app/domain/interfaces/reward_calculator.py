"""Reward calculator interface for RL.

Contract for calculating rewards in the heating control environment.
"""

from abc import ABC, abstractmethod

from domain.value_objects import RLAction, RLObservation


class IRewardCalculator(ABC):
    """Contract for reward calculation in RL.

    This interface defines how rewards are computed for heating control actions.
    Rewards should guide the agent to:
    - Reach target temperatures efficiently
    - Minimize energy consumption
    - Avoid overshooting targets
    - Maintain comfort
    """

    @abstractmethod
    def calculate_reward(
        self,
        previous_state: RLObservation,
        action: RLAction,
        current_state: RLObservation,
    ) -> float:
        """Calculate the reward for a state transition.

        This method computes the reward after taking an action in a state
        and observing the resulting next state. The reward should reflect:
        - Progress towards target temperature
        - Energy efficiency
        - Comfort maintenance
        - Penalties for overshooting

        Args:
            previous_state: The observation state before the action
            action: The action taken
            current_state: The observation state after the action

        Returns:
            Reward value (can be positive or negative)
        """
        pass

    @abstractmethod
    def calculate_terminal_reward(
        self,
        final_state: RLObservation,
        total_energy_consumed_kwh: float,
    ) -> float:
        """Calculate the terminal reward at the end of an episode.

        This method computes a final reward when an episode ends. Target achievement
        is determined internally based on:
        - Temperature within tolerance of target_temp
        - Timing relative to scheduled target time (time_until_target_minutes)

        The reward accounts for:
        - Perfect timing (time_until_target_minutes == 0)
        - Early achievement (time_until_target_minutes < 0) - lesser penalty
        - Late achievement (time_until_target_minutes > 0) - greater penalty

        Args:
            final_state: The final observation state containing temperature,
                        target, and timing information
            total_energy_consumed_kwh: Total energy consumed during episode

        Returns:
            Terminal reward value
        """
        pass
