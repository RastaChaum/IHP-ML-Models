"""Device configuration value object.

Immutable data structure for IHP device configuration.
This is received from the Intelligent Heating Pilot component
to configure which sensors to use for training.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceConfig:
    """Configuration for an IHP device.

    This configuration is sent by the IHP component to tell the addon
    which sensors to use for fetching historical data and training.

    Attributes:
        device_id: Unique identifier for this IHP device
        indoor_temp_entity_id: Entity ID for indoor temperature sensor
        outdoor_temp_entity_id: Entity ID for outdoor temperature sensor
        humidity_entity_id: Entity ID for humidity sensor (optional)
        target_temp_entity_id: Entity ID for target temperature
        heating_state_entity_id: Entity ID for heating state (on/off)
        history_days: Number of days of history to fetch for training
    """

    device_id: str
    indoor_temp_entity_id: str
    outdoor_temp_entity_id: str
    target_temp_entity_id: str
    heating_state_entity_id: str
    humidity_entity_id: str | None = None
    history_days: int = 30

    def __post_init__(self) -> None:
        """Validate device configuration."""
        if not self.device_id:
            raise ValueError("device_id cannot be empty")
        if not self.indoor_temp_entity_id:
            raise ValueError("indoor_temp_entity_id cannot be empty")
        if not self.outdoor_temp_entity_id:
            raise ValueError("outdoor_temp_entity_id cannot be empty")
        if not self.target_temp_entity_id:
            raise ValueError("target_temp_entity_id cannot be empty")
        if not self.heating_state_entity_id:
            raise ValueError("heating_state_entity_id cannot be empty")
        if self.history_days < 1:
            raise ValueError(f"history_days must be at least 1, got {self.history_days}")
        if self.history_days > 365:
            raise ValueError(f"history_days must be at most 365, got {self.history_days}")
