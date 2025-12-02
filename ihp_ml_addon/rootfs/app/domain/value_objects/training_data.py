"""Training data value objects.

Immutable data structures for ML training inputs.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence


def get_week_of_month(dt: datetime) -> int:
    """Calculate the week of the month for a given datetime.

    Args:
        dt: Datetime to calculate week of month for

    Returns:
        Week number (1-5) within the month
    """
    # Calculate which week this day falls into (1-based)
    week_of_month = ((dt.day - 1) // 7) + 1
    return min(week_of_month, 5)  # Cap at 5 for months with more than 4 weeks


@dataclass(frozen=True)
class TrainingDataPoint:
    """A single training data point for heating prediction.

    Attributes:
        outdoor_temp: Outdoor temperature in °C
        indoor_temp: Current indoor temperature in °C
        target_temp: Target temperature in °C
        humidity: Relative humidity percentage (0-100)
        hour_of_day: Hour of the day (0-23)
        day_of_week: Day of week (0=Monday, 6=Sunday)
        week_of_month: Week of the month (1-5)
        month: Month of the year (1-12)
        minutes_since_last_cycle: Minutes elapsed since the previous cycle ended
        heating_duration_minutes: Label - time needed to reach target (minutes)
        timestamp: When this data point was recorded
    """

    outdoor_temp: float
    indoor_temp: float
    target_temp: float
    humidity: float
    hour_of_day: int
    # day_of_week: int
    # week_of_month: int
    # month: int
    heating_duration_minutes: float
    timestamp: datetime
    minutes_since_last_cycle: float = 0.0

    def __post_init__(self) -> None:
        """Validate data point values."""
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
        # if not 0 <= self.day_of_week <= 6:
        #     raise ValueError(f"day_of_week must be between 0 and 6, got {self.day_of_week}")
        # if not 1 <= self.week_of_month <= 5:
        #     raise ValueError(f"week_of_month must be between 1 and 5, got {self.week_of_month}")
        # if not 1 <= self.month <= 12:
        #     raise ValueError(f"month must be between 1 and 12, got {self.month}")
        if self.minutes_since_last_cycle < 0:
            raise ValueError(
                f"minutes_since_last_cycle must be non-negative, got {self.minutes_since_last_cycle}"
            )
        if self.heating_duration_minutes < 0:
            raise ValueError(
                f"heating_duration_minutes must be non-negative, got {self.heating_duration_minutes}"
            )


@dataclass(frozen=True)
class TrainingData:
    """Collection of training data points for model training.

    Attributes:
        data_points: Sequence of training data points
        model_id: Optional model identifier for updates
    """

    data_points: tuple[TrainingDataPoint, ...]
    model_id: str | None = None

    def __post_init__(self) -> None:
        """Validate training data."""
        if not self.data_points:
            raise ValueError("Training data must contain at least one data point")

    @classmethod
    def from_sequence(
        cls,
        data_points: Sequence[TrainingDataPoint],
        model_id: str | None = None,
    ) -> "TrainingData":
        """Create TrainingData from a sequence of data points."""
        return cls(data_points=tuple(data_points), model_id=model_id)

    @property
    def size(self) -> int:
        """Return the number of data points."""
        return len(self.data_points)
