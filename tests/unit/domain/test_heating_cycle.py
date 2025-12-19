"""Tests for heating cycle value objects.

These tests verify that heating cycle value objects are immutable and properly validated.
"""

from datetime import datetime, timedelta

import pytest
from domain.value_objects import HeatingCycle, HeatingCycleCache


class TestHeatingCycle:
    """Tests for HeatingCycle value object."""

    def test_valid_heating_cycle_creation(self) -> None:
        """Test creating a valid heating cycle."""
        start_time = datetime(2024, 1, 15, 7, 0, 0)
        end_time = datetime(2024, 1, 15, 7, 45, 0)
        
        cycle = HeatingCycle(
            cycle_id="ihp_salon_20240115_070000",
            device_id="ihp_salon",
            start_time=start_time,
            end_time=end_time,
            start_indoor_temp=18.0,
            end_indoor_temp=21.0,
            target_temp=21.0,
            outdoor_temp=5.0,
            humidity=65.0,
            duration_minutes=45.0,
            hour_of_day=7,
            minutes_since_last_cycle=120.0,
        )
        
        assert cycle.cycle_id == "ihp_salon_20240115_070000"
        assert cycle.device_id == "ihp_salon"
        assert cycle.start_time == start_time
        assert cycle.end_time == end_time
        assert cycle.start_indoor_temp == 18.0
        assert cycle.end_indoor_temp == 21.0
        assert cycle.target_temp == 21.0
        assert cycle.outdoor_temp == 5.0
        assert cycle.humidity == 65.0
        assert cycle.duration_minutes == 45.0
        assert cycle.hour_of_day == 7
        assert cycle.minutes_since_last_cycle == 120.0

    def test_heating_cycle_is_immutable(self) -> None:
        """Test that heating cycle is immutable (frozen dataclass)."""
        cycle = HeatingCycle(
            cycle_id="test_cycle",
            device_id="test_device",
            start_time=datetime(2024, 1, 15, 7, 0, 0),
            end_time=datetime(2024, 1, 15, 7, 45, 0),
            start_indoor_temp=18.0,
            end_indoor_temp=21.0,
            target_temp=21.0,
            outdoor_temp=5.0,
            humidity=65.0,
            duration_minutes=45.0,
            hour_of_day=7,
        )
        
        with pytest.raises(AttributeError):
            cycle.duration_minutes = 60.0  # type: ignore

    def test_heating_cycle_empty_cycle_id_raises_error(self) -> None:
        """Test that empty cycle_id raises ValueError."""
        with pytest.raises(ValueError, match="cycle_id cannot be empty"):
            HeatingCycle(
                cycle_id="",
                device_id="test_device",
                start_time=datetime(2024, 1, 15, 7, 0, 0),
                end_time=datetime(2024, 1, 15, 7, 45, 0),
                start_indoor_temp=18.0,
                end_indoor_temp=21.0,
                target_temp=21.0,
                outdoor_temp=5.0,
                humidity=65.0,
                duration_minutes=45.0,
                hour_of_day=7,
            )

    def test_heating_cycle_empty_device_id_raises_error(self) -> None:
        """Test that empty device_id raises ValueError."""
        with pytest.raises(ValueError, match="device_id cannot be empty"):
            HeatingCycle(
                cycle_id="test_cycle",
                device_id="",
                start_time=datetime(2024, 1, 15, 7, 0, 0),
                end_time=datetime(2024, 1, 15, 7, 45, 0),
                start_indoor_temp=18.0,
                end_indoor_temp=21.0,
                target_temp=21.0,
                outdoor_temp=5.0,
                humidity=65.0,
                duration_minutes=45.0,
                hour_of_day=7,
            )

    def test_heating_cycle_end_before_start_raises_error(self) -> None:
        """Test that end_time before start_time raises ValueError."""
        with pytest.raises(ValueError, match="end_time must be after start_time"):
            HeatingCycle(
                cycle_id="test_cycle",
                device_id="test_device",
                start_time=datetime(2024, 1, 15, 8, 0, 0),
                end_time=datetime(2024, 1, 15, 7, 0, 0),
                start_indoor_temp=18.0,
                end_indoor_temp=21.0,
                target_temp=21.0,
                outdoor_temp=5.0,
                humidity=65.0,
                duration_minutes=45.0,
                hour_of_day=7,
            )

    def test_heating_cycle_invalid_outdoor_temp_raises_error(self) -> None:
        """Test that invalid outdoor_temp raises ValueError."""
        with pytest.raises(ValueError, match="outdoor_temp must be between -50 and 60"):
            HeatingCycle(
                cycle_id="test_cycle",
                device_id="test_device",
                start_time=datetime(2024, 1, 15, 7, 0, 0),
                end_time=datetime(2024, 1, 15, 7, 45, 0),
                start_indoor_temp=18.0,
                end_indoor_temp=21.0,
                target_temp=21.0,
                outdoor_temp=70.0,  # Invalid
                humidity=65.0,
                duration_minutes=45.0,
                hour_of_day=7,
            )

    def test_heating_cycle_invalid_humidity_raises_error(self) -> None:
        """Test that invalid humidity raises ValueError."""
        with pytest.raises(ValueError, match="humidity must be between 0 and 100"):
            HeatingCycle(
                cycle_id="test_cycle",
                device_id="test_device",
                start_time=datetime(2024, 1, 15, 7, 0, 0),
                end_time=datetime(2024, 1, 15, 7, 45, 0),
                start_indoor_temp=18.0,
                end_indoor_temp=21.0,
                target_temp=21.0,
                outdoor_temp=5.0,
                humidity=150.0,  # Invalid
                duration_minutes=45.0,
                hour_of_day=7,
            )

    def test_heating_cycle_negative_duration_raises_error(self) -> None:
        """Test that negative duration_minutes raises ValueError."""
        with pytest.raises(ValueError, match="duration_minutes must be non-negative"):
            HeatingCycle(
                cycle_id="test_cycle",
                device_id="test_device",
                start_time=datetime(2024, 1, 15, 7, 0, 0),
                end_time=datetime(2024, 1, 15, 7, 45, 0),
                start_indoor_temp=18.0,
                end_indoor_temp=21.0,
                target_temp=21.0,
                outdoor_temp=5.0,
                humidity=65.0,
                duration_minutes=-10.0,  # Invalid
                hour_of_day=7,
            )

    def test_generate_cycle_id(self) -> None:
        """Test cycle ID generation from device and timestamp."""
        device_id = "ihp_salon"
        start_time = datetime(2024, 1, 15, 7, 30, 45)
        
        cycle_id = HeatingCycle.generate_cycle_id(device_id, start_time)
        
        assert cycle_id == "ihp_salon_20240115_073045"


class TestHeatingCycleCache:
    """Tests for HeatingCycleCache value object."""

    def test_valid_cache_creation(self) -> None:
        """Test creating a valid heating cycle cache."""
        cycle1 = HeatingCycle(
            cycle_id="cycle_1",
            device_id="ihp_salon",
            start_time=datetime(2024, 1, 15, 7, 0, 0),
            end_time=datetime(2024, 1, 15, 7, 45, 0),
            start_indoor_temp=18.0,
            end_indoor_temp=21.0,
            target_temp=21.0,
            outdoor_temp=5.0,
            humidity=65.0,
            duration_minutes=45.0,
            hour_of_day=7,
        )
        
        cycle2 = HeatingCycle(
            cycle_id="cycle_2",
            device_id="ihp_salon",
            start_time=datetime(2024, 1, 15, 9, 0, 0),
            end_time=datetime(2024, 1, 15, 9, 30, 0),
            start_indoor_temp=19.0,
            end_indoor_temp=21.0,
            target_temp=21.0,
            outdoor_temp=6.0,
            humidity=60.0,
            duration_minutes=30.0,
            hour_of_day=9,
        )
        
        last_scan = datetime(2024, 1, 15, 12, 0, 0)
        
        cache = HeatingCycleCache(
            device_id="ihp_salon",
            cycles=(cycle1, cycle2),
            last_scan_time=last_scan,
            retention_days=30,
        )
        
        assert cache.device_id == "ihp_salon"
        assert len(cache.cycles) == 2
        assert cache.last_scan_time == last_scan
        assert cache.retention_days == 30
        assert cache.cache_version == 1
        assert cache.size == 2
        assert not cache.is_empty

    def test_empty_cache(self) -> None:
        """Test creating an empty cache."""
        cache = HeatingCycleCache(
            device_id="ihp_salon",
            cycles=(),
            last_scan_time=datetime(2024, 1, 15, 12, 0, 0),
            retention_days=30,
        )
        
        assert cache.size == 0
        assert cache.is_empty

    def test_cache_is_immutable(self) -> None:
        """Test that cache is immutable (frozen dataclass)."""
        cache = HeatingCycleCache(
            device_id="ihp_salon",
            cycles=(),
            last_scan_time=datetime(2024, 1, 15, 12, 0, 0),
            retention_days=30,
        )
        
        with pytest.raises(AttributeError):
            cache.retention_days = 60  # type: ignore

    def test_cache_empty_device_id_raises_error(self) -> None:
        """Test that empty device_id raises ValueError."""
        with pytest.raises(ValueError, match="device_id cannot be empty"):
            HeatingCycleCache(
                device_id="",
                cycles=(),
                last_scan_time=datetime(2024, 1, 15, 12, 0, 0),
                retention_days=30,
            )

    def test_cache_invalid_retention_days_raises_error(self) -> None:
        """Test that invalid retention_days raises ValueError."""
        with pytest.raises(ValueError, match="retention_days must be at least 1"):
            HeatingCycleCache(
                device_id="ihp_salon",
                cycles=(),
                last_scan_time=datetime(2024, 1, 15, 12, 0, 0),
                retention_days=0,
            )
        
        with pytest.raises(ValueError, match="retention_days must be at most 365"):
            HeatingCycleCache(
                device_id="ihp_salon",
                cycles=(),
                last_scan_time=datetime(2024, 1, 15, 12, 0, 0),
                retention_days=400,
            )

    def test_get_cycles_after(self) -> None:
        """Test filtering cycles after a timestamp."""
        cycle1 = HeatingCycle(
            cycle_id="cycle_1",
            device_id="ihp_salon",
            start_time=datetime(2024, 1, 15, 7, 0, 0),
            end_time=datetime(2024, 1, 15, 7, 45, 0),
            start_indoor_temp=18.0,
            end_indoor_temp=21.0,
            target_temp=21.0,
            outdoor_temp=5.0,
            humidity=65.0,
            duration_minutes=45.0,
            hour_of_day=7,
        )
        
        cycle2 = HeatingCycle(
            cycle_id="cycle_2",
            device_id="ihp_salon",
            start_time=datetime(2024, 1, 15, 9, 0, 0),
            end_time=datetime(2024, 1, 15, 9, 30, 0),
            start_indoor_temp=19.0,
            end_indoor_temp=21.0,
            target_temp=21.0,
            outdoor_temp=6.0,
            humidity=60.0,
            duration_minutes=30.0,
            hour_of_day=9,
        )
        
        cache = HeatingCycleCache(
            device_id="ihp_salon",
            cycles=(cycle1, cycle2),
            last_scan_time=datetime(2024, 1, 15, 12, 0, 0),
            retention_days=30,
        )
        
        # Get cycles after 8:00
        cycles_after = cache.get_cycles_after(datetime(2024, 1, 15, 8, 0, 0))
        assert len(cycles_after) == 1
        assert cycles_after[0].cycle_id == "cycle_2"

    def test_get_cycles_in_range(self) -> None:
        """Test filtering cycles within a time range."""
        cycle1 = HeatingCycle(
            cycle_id="cycle_1",
            device_id="ihp_salon",
            start_time=datetime(2024, 1, 15, 7, 0, 0),
            end_time=datetime(2024, 1, 15, 7, 45, 0),
            start_indoor_temp=18.0,
            end_indoor_temp=21.0,
            target_temp=21.0,
            outdoor_temp=5.0,
            humidity=65.0,
            duration_minutes=45.0,
            hour_of_day=7,
        )
        
        cycle2 = HeatingCycle(
            cycle_id="cycle_2",
            device_id="ihp_salon",
            start_time=datetime(2024, 1, 15, 9, 0, 0),
            end_time=datetime(2024, 1, 15, 9, 30, 0),
            start_indoor_temp=19.0,
            end_indoor_temp=21.0,
            target_temp=21.0,
            outdoor_temp=6.0,
            humidity=60.0,
            duration_minutes=30.0,
            hour_of_day=9,
        )
        
        cycle3 = HeatingCycle(
            cycle_id="cycle_3",
            device_id="ihp_salon",
            start_time=datetime(2024, 1, 15, 11, 0, 0),
            end_time=datetime(2024, 1, 15, 11, 30, 0),
            start_indoor_temp=19.5,
            end_indoor_temp=21.0,
            target_temp=21.0,
            outdoor_temp=7.0,
            humidity=55.0,
            duration_minutes=30.0,
            hour_of_day=11,
        )
        
        cache = HeatingCycleCache(
            device_id="ihp_salon",
            cycles=(cycle1, cycle2, cycle3),
            last_scan_time=datetime(2024, 1, 15, 12, 0, 0),
            retention_days=30,
        )
        
        # Get cycles between 8:00 and 10:00
        cycles_in_range = cache.get_cycles_in_range(
            datetime(2024, 1, 15, 8, 0, 0),
            datetime(2024, 1, 15, 10, 0, 0),
        )
        assert len(cycles_in_range) == 1
        assert cycles_in_range[0].cycle_id == "cycle_2"
