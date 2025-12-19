"""Heating cycle value objects for incremental caching.

Immutable data structures representing detected heating cycles.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class HeatingCycle:
    """Represents a single detected heating cycle with its metadata.

    A heating cycle is a period where the heating system was active to raise
    the temperature from start_temp to end_temp (target).

    Attributes:
        cycle_id: Unique identifier for this cycle (format: device_id_timestamp)
        device_id: Device identifier this cycle belongs to
        start_time: When the heating cycle started
        end_time: When the heating cycle ended
        start_indoor_temp: Indoor temperature at cycle start (°C)
        end_indoor_temp: Indoor temperature at cycle end (°C)
        target_temp: Target temperature for this cycle (°C)
        outdoor_temp: Outdoor temperature during cycle start (°C)
        humidity: Humidity percentage during cycle (0-100)
        duration_minutes: Total heating duration in minutes
        hour_of_day: Hour when cycle started (0-23)
        minutes_since_last_cycle: Minutes elapsed since previous cycle ended
    """

    cycle_id: str
    device_id: str
    start_time: datetime
    end_time: datetime
    start_indoor_temp: float
    end_indoor_temp: float
    target_temp: float
    outdoor_temp: float
    humidity: float
    duration_minutes: float
    hour_of_day: int
    minutes_since_last_cycle: float = 0.0

    def __post_init__(self) -> None:
        """Validate heating cycle values."""
        if not self.cycle_id:
            raise ValueError("cycle_id cannot be empty")
        if not self.device_id:
            raise ValueError("device_id cannot be empty")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        if not -50 <= self.outdoor_temp <= 60:
            raise ValueError(f"outdoor_temp must be between -50 and 60, got {self.outdoor_temp}")
        if not -20 <= self.start_indoor_temp <= 50:
            raise ValueError(
                f"start_indoor_temp must be between -20 and 50, got {self.start_indoor_temp}"
            )
        if not -20 <= self.end_indoor_temp <= 50:
            raise ValueError(
                f"end_indoor_temp must be between -20 and 50, got {self.end_indoor_temp}"
            )
        if not 0 <= self.target_temp <= 50:
            raise ValueError(f"target_temp must be between 0 and 50, got {self.target_temp}")
        if not 0 <= self.humidity <= 100:
            raise ValueError(f"humidity must be between 0 and 100, got {self.humidity}")
        if not 0 <= self.hour_of_day <= 23:
            raise ValueError(f"hour_of_day must be between 0 and 23, got {self.hour_of_day}")
        if self.duration_minutes < 0:
            raise ValueError(
                f"duration_minutes must be non-negative, got {self.duration_minutes}"
            )
        if self.minutes_since_last_cycle < 0:
            raise ValueError(
                f"minutes_since_last_cycle must be non-negative, got {self.minutes_since_last_cycle}"
            )

    @staticmethod
    def generate_cycle_id(device_id: str, start_time: datetime) -> str:
        """Generate a unique cycle ID from device and timestamp.

        Args:
            device_id: Device identifier
            start_time: Cycle start timestamp

        Returns:
            Unique cycle ID string
        """
        timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
        return f"{device_id}_{timestamp_str}"


@dataclass(frozen=True)
class HeatingCycleCache:
    """Collection of cached heating cycles with metadata.

    Attributes:
        device_id: Device identifier for this cache
        cycles: Tuple of heating cycles (immutable)
        last_scan_time: Timestamp of the last successful scan
        retention_days: Number of days to retain cycles in cache
        cache_version: Version number for cache schema (for future migrations)
    """

    device_id: str
    cycles: tuple[HeatingCycle, ...]
    last_scan_time: datetime
    retention_days: int
    cache_version: int = 1

    def __post_init__(self) -> None:
        """Validate cache metadata."""
        if not self.device_id:
            raise ValueError("device_id cannot be empty")
        if self.retention_days < 1:
            raise ValueError(f"retention_days must be at least 1, got {self.retention_days}")
        if self.retention_days > 365:
            raise ValueError(f"retention_days must be at most 365, got {self.retention_days}")

    @property
    def size(self) -> int:
        """Return the number of cached cycles."""
        return len(self.cycles)

    @property
    def is_empty(self) -> bool:
        """Check if cache is empty."""
        return len(self.cycles) == 0

    def get_cycles_after(self, timestamp: datetime) -> tuple[HeatingCycle, ...]:
        """Get all cycles that started after a given timestamp.

        Args:
            timestamp: Cutoff timestamp

        Returns:
            Tuple of cycles that started after the timestamp
        """
        return tuple(cycle for cycle in self.cycles if cycle.start_time > timestamp)

    def get_cycles_in_range(
        self, start_time: datetime, end_time: datetime
    ) -> tuple[HeatingCycle, ...]:
        """Get all cycles within a time range.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Tuple of cycles within the time range
        """
        return tuple(
            cycle
            for cycle in self.cycles
            if start_time <= cycle.start_time <= end_time
        )
