"""Prediction request value object.

Immutable data structure for ML prediction inputs.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionRequest:
    """Request for heating duration prediction.

    Attributes:
        outdoor_temp: Outdoor temperature in °C
        indoor_temp: Current indoor temperature in °C
        target_temp: Target temperature in °C
        humidity: Relative humidity percentage (0-100)
        hour_of_day: Hour of the day (0-23)
        device_id: Device/thermostat ID for model selection (optional)
        model_id: Optional model identifier (uses latest for device if not specified)
        adjacent_rooms: Optional dict of adjacent room data, keyed by zone_id.
            Each zone should have: current_temp, current_humidity, next_target_temp, duration_until_change
    """

    outdoor_temp: float
    indoor_temp: float
    target_temp: float
    humidity: float
    hour_of_day: int
    device_id: str | None = None
    model_id: str | None = None
    minutes_since_last_cycle: float | None = None
    adjacent_rooms: dict[str, dict[str, float]] | None = None

    def __post_init__(self) -> None:
        """Validate prediction request values."""
        if not -50 <= self.outdoor_temp <= 60:
            raise ValueError(f"outdoor_temp must be between -50 and 60, got {self.outdoor_temp}")
        if not -20 <= self.indoor_temp <= 50:
            raise ValueError(f"indoor_temp must be between -20 and 50, got {self.indoor_temp}")
        if not 0 <= self.target_temp <= 50:
            raise ValueError(f"target_temp must be between 0 and 50, got {self.target_temp}")
        if not 0 <= self.humidity <= 100:
            raise ValueError(f"humidity must be between 0 and 100, got {self.humidity}")
        if not 0 <= self.hour_of_day <= 23:
            raise ValueError(f"hour_of_day must be between 0 and 23, got {self.hour_of_day}")
        if self.minutes_since_last_cycle is not None and self.minutes_since_last_cycle < 0:
            raise ValueError(f"minutes_since_last_cycle must be non-negative, got {self.minutes_since_last_cycle}")
 
    @property
    def temp_delta(self) -> float:
        """Calculate temperature difference to target."""
        return self.target_temp - self.indoor_temp
