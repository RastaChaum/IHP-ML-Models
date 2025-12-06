"""RL Episode termination service.

Domain service for determining when RL episodes should end.
"""

import logging

from domain.value_objects import RLObservation

logger = logging.getLogger(__name__)


class RLEpisodeService:
    """Service for determining episode termination in RL.

    This service encapsulates the business logic for deciding when
    an episode (heating cycle) should be considered complete.
    """

    def __init__(
        self,
        target_tolerance_celsius: float = 0.3,
        target_change_threshold_celsius: float = 0.5,
    ) -> None:
        """Initialize the episode service.

        Args:
            target_tolerance_celsius: Temperature tolerance for target achievement (default: 0.3°C)
            target_change_threshold_celsius: Threshold for significant target change (default: 0.5°C)
        """
        self._target_tolerance = target_tolerance_celsius
        self._target_change_threshold = target_change_threshold_celsius

        logger.info(
            "RLEpisodeService initialized: tolerance=%.2f°C, change_threshold=%.2f°C",
            self._target_tolerance,
            self._target_change_threshold,
        )

    def is_episode_done(
        self,
        current_obs: RLObservation,
        previous_obs: RLObservation,
    ) -> bool:
        """Determine if an episode should end.

        Business rules for episode termination:
        1. Target temperature changed significantly (>threshold)
        2. Target temperature is achieved (within tolerance)

        Args:
            current_obs: Current observation state
            previous_obs: Previous observation state

        Returns:
            True if episode should end, False otherwise
        """
        # Episode ends if target temperature changed significantly
        target_change = abs(current_obs.target_temp - previous_obs.target_temp)
        if target_change > self._target_change_threshold:
            logger.debug(
                "Episode ends: target changed by %.2f°C (threshold: %.2f°C)",
                target_change,
                self._target_change_threshold,
            )
            return True

        # Episode ends if target is achieved
        temp_diff = abs(current_obs.indoor_temp - current_obs.target_temp)
        if temp_diff <= self._target_tolerance:
            logger.debug(
                "Episode ends: target achieved (diff: %.2f°C, tolerance: %.2f°C)",
                temp_diff,
                self._target_tolerance,
            )
            return True

        # Episode continues
        return False
