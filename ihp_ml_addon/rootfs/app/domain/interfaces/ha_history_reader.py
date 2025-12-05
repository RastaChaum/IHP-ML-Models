"""Home Assistant history reader interface.

Contract for reading historical data from Home Assistant.
"""

from abc import ABC, abstractmethod
from datetime import datetime

from domain.value_objects import RLExperience, TrainingData, TrainingRequest


class IHomeAssistantHistoryReader(ABC):
    """Contract for reading historical data from Home Assistant.

    This interface defines how the addon retrieves historical sensor data
    from Home Assistant to prepare training data for the ML model.
    """

    @abstractmethod
    async def fetch_training_data(
        self,
        indoor_temp_entity_id: str,
        outdoor_temp_entity_id: str,
        target_temp_entity_id: str,
        heating_state_entity_id: str,
        humidity_entity_id: str | None,
        start_time: datetime,
        end_time: datetime,
        cycle_split_duration_minutes: int | None = None,
    ) -> TrainingData:
        """Fetch historical data and convert to training data.

        Args:
            indoor_temp_entity_id: Entity ID for indoor temperature sensor
            outdoor_temp_entity_id: Entity ID for outdoor temperature sensor
            target_temp_entity_id: Entity ID for target temperature
            heating_state_entity_id: Entity ID for heating state (on/off)
            humidity_entity_id: Entity ID for humidity sensor (optional)
            start_time: Start of the time range for fetching history
            end_time: End of the time range for fetching history
            cycle_split_duration_minutes: Optional duration in minutes to split
                long heating cycles into smaller sub-cycles. If None, cycles
                are not split.

        Returns:
            TrainingData with extracted heating cycles

        Raises:
            ConnectionError: If unable to connect to Home Assistant
            ValueError: If entity IDs are invalid or no data available
        """
        pass

    @abstractmethod
    async def fetch_rl_experiences(
        self,
        training_request: TrainingRequest,
    ) -> list[RLExperience]:
        """Fetch historical data and convert to RL experiences.

        This method constructs a sequence of RLExperience objects from
        historical Home Assistant data. Each experience represents a
        state transition (s, a, r, s', done) in the RL environment.

        The method:
        1. Fetches history for all entities specified in the training request
        2. Constructs RLObservation objects for each timestep
        3. Infers actions taken (based on heating state changes)
        4. Calculates rewards using the reward calculator
        5. Returns a list of RLExperience objects

        Args:
            training_request: Training configuration with entity IDs and time range

        Returns:
            List of RLExperience objects for training

        Raises:
            ConnectionError: If unable to connect to Home Assistant
            ValueError: If entity IDs are invalid or no data available
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if Home Assistant API is available.

        Returns:
            True if the addon can communicate with Home Assistant
        """
        pass
