"""Home Assistant history reader interface.

Contract for reading historical data from Home Assistant.
"""

from abc import ABC, abstractmethod
from datetime import datetime

from domain.value_objects import TrainingData


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
        on_time_entity_id: str | None = None,
        on_time_buffer_minutes: int = 15,
        use_statistics: bool = False,
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
            on_time_entity_id: Entity ID for thermostat "On Time" sensor (optional).
                If provided, this sensor is used instead of heating_state_entity_id
                to determine when heating is active.
            on_time_buffer_minutes: Buffer time in minutes for "On Time" detection.
                If heating doesn't activate for this duration, consider heating off.
                Default is 15 minutes.
            use_statistics: If True, use Home Assistant statistics API for longer
                data retention (>10 days). Requires on_time_entity_id.

        Returns:
            TrainingData with extracted heating cycles

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
