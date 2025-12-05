"""RL Action inference service.

Domain service for inferring RL actions from observation transitions.
"""

import logging

from domain.value_objects import HeatingActionType, RLAction, RLObservation

logger = logging.getLogger(__name__)


class RLActionService:
    """Service for inferring RL actions from state transitions.

    This service encapsulates the business logic for determining which
    action was taken between two consecutive observations in the heating
    control system.
    """

    def infer_action(
        self,
        current_obs: RLObservation,
        next_obs: RLObservation,
    ) -> RLAction:
        """Infer the action taken between two observations.

        Business rules:
        - If heating state changed from OFF to ON → TURN_ON
        - If heating state changed from ON to OFF → TURN_OFF
        - If target temperature changed significantly (>0.1°C) → SET_TARGET_TEMPERATURE
        - Otherwise → NO_OP

        Args:
            current_obs: Current observation state
            next_obs: Next observation state

        Returns:
            Inferred RLAction
        """
        logger.debug(
            "Inferring action: heating %s→%s, target %.1f°C→%.1f°C",
            current_obs.is_heating_on,
            next_obs.is_heating_on,
            current_obs.target_temp,
            next_obs.target_temp,
        )

        # Infer action type based on state transition
        if not current_obs.is_heating_on and next_obs.is_heating_on:
            action_type = HeatingActionType.TURN_ON
        elif current_obs.is_heating_on and not next_obs.is_heating_on:
            action_type = HeatingActionType.TURN_OFF
        elif abs(current_obs.target_temp - next_obs.target_temp) > 0.1:
            action_type = HeatingActionType.SET_TARGET_TEMPERATURE
        else:
            action_type = HeatingActionType.NO_OP

        # Use the target temperature from the next state as the action value
        return RLAction(
            action_type=action_type,
            value=next_obs.target_temp,
            decision_timestamp=next_obs.timestamp,
            confidence_score=None,
        )
