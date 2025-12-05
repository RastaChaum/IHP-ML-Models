"""Reward configuration value object.

Configurable parameters for reward shaping in RL heating control.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Configuration for reward calculation in RL heating control.

    These parameters control the reward shaping to guide the RL agent
    towards optimal heating control behavior.

    Attributes:
        progress_reward_factor: Multiplier for temperature progress towards target.
        drift_penalty_factor: Penalty factor when temperature moves away from target.
        overshoot_penalty_factor: Penalty factor when temperature exceeds target.
        energy_penalty_factor: Penalty per kWh of energy consumed.
        target_achieved_reward: Large reward when target is reached on time.
        target_missed_penalty: Large penalty when target is missed.
        early_achievement_penalty_factor: Penalty per minute when target reached early.
        late_achievement_penalty_factor: Penalty per minute when target reached late.
        target_tolerance_celsius: Temperature tolerance for considering target achieved.
        overshoot_threshold_celsius: Temperature above target to consider overshoot.
    """

    # Progress rewards
    progress_reward_factor: float = 1.0

    # Penalties
    drift_penalty_factor: float = 0.5
    overshoot_penalty_factor: float = 2.0
    energy_penalty_factor: float = 0.1

    # Terminal rewards
    target_achieved_reward: float = 10.0
    target_missed_penalty: float = 10.0
    early_achievement_penalty_factor: float = 0.5
    late_achievement_penalty_factor: float = 1.0

    # Thresholds
    target_tolerance_celsius: float = 0.5
    overshoot_threshold_celsius: float = 1.0

    def __post_init__(self) -> None:
        """Validate reward configuration values."""
        if self.progress_reward_factor < 0:
            raise ValueError(
                f"progress_reward_factor must be non-negative, "
                f"got {self.progress_reward_factor}"
            )
        if self.drift_penalty_factor < 0:
            raise ValueError(
                f"drift_penalty_factor must be non-negative, got {self.drift_penalty_factor}"
            )
        if self.overshoot_penalty_factor < 0:
            raise ValueError(
                f"overshoot_penalty_factor must be non-negative, "
                f"got {self.overshoot_penalty_factor}"
            )
        if self.energy_penalty_factor < 0:
            raise ValueError(
                f"energy_penalty_factor must be non-negative, got {self.energy_penalty_factor}"
            )
        if self.target_achieved_reward < 0:
            raise ValueError(
                f"target_achieved_reward must be non-negative, "
                f"got {self.target_achieved_reward}"
            )
        if self.target_missed_penalty < 0:
            raise ValueError(
                f"target_missed_penalty must be non-negative, got {self.target_missed_penalty}"
            )
        if self.early_achievement_penalty_factor < 0:
            raise ValueError(
                f"early_achievement_penalty_factor must be non-negative, "
                f"got {self.early_achievement_penalty_factor}"
            )
        if self.late_achievement_penalty_factor < 0:
            raise ValueError(
                f"late_achievement_penalty_factor must be non-negative, "
                f"got {self.late_achievement_penalty_factor}"
            )
        if self.target_tolerance_celsius <= 0:
            raise ValueError(
                f"target_tolerance_celsius must be positive, "
                f"got {self.target_tolerance_celsius}"
            )
        if self.overshoot_threshold_celsius <= 0:
            raise ValueError(
                f"overshoot_threshold_celsius must be positive, "
                f"got {self.overshoot_threshold_celsius}"
            )
