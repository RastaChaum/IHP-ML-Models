"""Tests for file-based heating cycle cache.

These tests verify that the file-based cache adapter correctly persists
and retrieves heating cycles from JSON files.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from domain.value_objects import HeatingCycle, HeatingCycleCache
from infrastructure.adapters.file_cycle_cache import FileBasedCycleCache


class TestFileBasedCycleCache:
    """Tests for FileBasedCycleCache adapter."""

    @pytest.fixture
    def cache_dir(self) -> Path:
        """Create a temporary directory for cache files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def cache(self, cache_dir: Path) -> FileBasedCycleCache:
        """Create a file-based cache instance."""
        return FileBasedCycleCache(cache_dir=str(cache_dir))

    @pytest.fixture
    def sample_cycle(self) -> HeatingCycle:
        """Create a sample heating cycle for testing."""
        return HeatingCycle(
            cycle_id="ihp_salon_20240115_070000",
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
            minutes_since_last_cycle=120.0,
        )

    @pytest.fixture
    def sample_cache(self, sample_cycle: HeatingCycle) -> HeatingCycleCache:
        """Create a sample cache for testing."""
        return HeatingCycleCache(
            device_id="ihp_salon",
            cycles=(sample_cycle,),
            last_scan_time=datetime(2024, 1, 15, 12, 0, 0),
            retention_days=30,
        )

    async def test_cache_initialization(self, cache_dir: Path) -> None:
        """Test that cache initialization creates the cache directory."""
        cache = FileBasedCycleCache(cache_dir=str(cache_dir / "test_cache"))
        assert (cache_dir / "test_cache").exists()

    async def test_cache_exists_returns_false_for_nonexistent(
        self, cache: FileBasedCycleCache
    ) -> None:
        """Test that cache_exists returns False for nonexistent cache."""
        exists = await cache.cache_exists("nonexistent_device")
        assert not exists

    async def test_load_cache_returns_none_for_nonexistent(
        self, cache: FileBasedCycleCache
    ) -> None:
        """Test that load_cache returns None for nonexistent cache."""
        loaded_cache = await cache.load_cache("nonexistent_device")
        assert loaded_cache is None

    async def test_save_and_load_cache(
        self, cache: FileBasedCycleCache, sample_cache: HeatingCycleCache
    ) -> None:
        """Test saving and loading a cache."""
        # Save cache
        await cache.save_cache(sample_cache)

        # Verify cache exists
        exists = await cache.cache_exists("ihp_salon")
        assert exists

        # Load cache
        loaded_cache = await cache.load_cache("ihp_salon")

        # Verify loaded cache matches original
        assert loaded_cache is not None
        assert loaded_cache.device_id == sample_cache.device_id
        assert loaded_cache.size == sample_cache.size
        assert loaded_cache.last_scan_time == sample_cache.last_scan_time
        assert loaded_cache.retention_days == sample_cache.retention_days
        assert len(loaded_cache.cycles) == len(sample_cache.cycles)

        # Verify cycle data
        loaded_cycle = loaded_cache.cycles[0]
        original_cycle = sample_cache.cycles[0]
        assert loaded_cycle.cycle_id == original_cycle.cycle_id
        assert loaded_cycle.device_id == original_cycle.device_id
        assert loaded_cycle.start_time == original_cycle.start_time
        assert loaded_cycle.end_time == original_cycle.end_time
        assert loaded_cycle.start_indoor_temp == original_cycle.start_indoor_temp
        assert loaded_cycle.duration_minutes == original_cycle.duration_minutes

    async def test_add_cycles_creates_new_cache(
        self, cache: FileBasedCycleCache, sample_cycle: HeatingCycle
    ) -> None:
        """Test adding cycles creates a new cache if none exists."""
        new_cycles = [sample_cycle]
        last_scan = datetime(2024, 1, 15, 12, 0, 0)

        updated_cache = await cache.add_cycles(
            device_id="ihp_salon",
            new_cycles=new_cycles,
            last_scan_time=last_scan,
            retention_days=30,
        )

        assert updated_cache.device_id == "ihp_salon"
        assert updated_cache.size == 1
        assert updated_cache.last_scan_time == last_scan
        assert updated_cache.retention_days == 30

    async def test_add_cycles_merges_with_existing(
        self,
        cache: FileBasedCycleCache,
        sample_cache: HeatingCycleCache,
    ) -> None:
        """Test adding cycles merges with existing cache."""
        # Save initial cache
        await cache.save_cache(sample_cache)

        # Create a new cycle
        new_cycle = HeatingCycle(
            cycle_id="ihp_salon_20240115_090000",
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

        # Add new cycle
        updated_cache = await cache.add_cycles(
            device_id="ihp_salon",
            new_cycles=[new_cycle],
            last_scan_time=datetime(2024, 1, 15, 14, 0, 0),
            retention_days=30,
        )

        # Verify merge
        assert updated_cache.size == 2
        cycle_ids = {cycle.cycle_id for cycle in updated_cache.cycles}
        assert "ihp_salon_20240115_070000" in cycle_ids
        assert "ihp_salon_20240115_090000" in cycle_ids

    async def test_add_cycles_avoids_duplicates(
        self,
        cache: FileBasedCycleCache,
        sample_cache: HeatingCycleCache,
        sample_cycle: HeatingCycle,
    ) -> None:
        """Test that adding duplicate cycles doesn't create duplicates."""
        # Save initial cache
        await cache.save_cache(sample_cache)

        # Try to add the same cycle again
        updated_cache = await cache.add_cycles(
            device_id="ihp_salon",
            new_cycles=[sample_cycle],
            last_scan_time=datetime(2024, 1, 15, 14, 0, 0),
            retention_days=30,
        )

        # Should still have only 1 cycle
        assert updated_cache.size == 1

    async def test_add_cycles_prunes_old_cycles(
        self, cache: FileBasedCycleCache
    ) -> None:
        """Test that add_cycles prunes cycles older than retention period."""
        # Create cycles spanning 40 days
        cycles = []
        base_time = datetime(2024, 1, 1, 7, 0, 0)

        for i in range(5):
            cycle_time = base_time + timedelta(days=i * 10)
            cycle = HeatingCycle(
                cycle_id=f"ihp_salon_{i}",
                device_id="ihp_salon",
                start_time=cycle_time,
                end_time=cycle_time + timedelta(minutes=45),
                start_indoor_temp=18.0,
                end_indoor_temp=21.0,
                target_temp=21.0,
                outdoor_temp=5.0,
                humidity=65.0,
                duration_minutes=45.0,
                hour_of_day=7,
            )
            cycles.append(cycle)

        # Add cycles with 30-day retention (scan time 40 days after start)
        last_scan = base_time + timedelta(days=40)
        updated_cache = await cache.add_cycles(
            device_id="ihp_salon",
            new_cycles=cycles,
            last_scan_time=last_scan,
            retention_days=30,
        )

        # Only cycles from last 30 days should remain
        # Cycles at day 0, 10, 20, 30, 40 from base_time
        # With cutoff at day 10 (40 - 30), should have cycles at 10, 20, 30, 40
        assert updated_cache.size == 4

    async def test_prune_old_cycles(
        self, cache: FileBasedCycleCache
    ) -> None:
        """Test manual pruning of old cycles."""
        # Create cycles spanning different times
        cycles = [
            HeatingCycle(
                cycle_id="old_cycle",
                device_id="ihp_salon",
                start_time=datetime(2024, 1, 1, 7, 0, 0),
                end_time=datetime(2024, 1, 1, 7, 45, 0),
                start_indoor_temp=18.0,
                end_indoor_temp=21.0,
                target_temp=21.0,
                outdoor_temp=5.0,
                humidity=65.0,
                duration_minutes=45.0,
                hour_of_day=7,
            ),
            HeatingCycle(
                cycle_id="recent_cycle",
                device_id="ihp_salon",
                start_time=datetime(2024, 1, 30, 7, 0, 0),
                end_time=datetime(2024, 1, 30, 7, 45, 0),
                start_indoor_temp=18.0,
                end_indoor_temp=21.0,
                target_temp=21.0,
                outdoor_temp=5.0,
                humidity=65.0,
                duration_minutes=45.0,
                hour_of_day=7,
            ),
        ]

        # Save cache
        test_cache = HeatingCycleCache(
            device_id="ihp_salon",
            cycles=tuple(cycles),
            last_scan_time=datetime(2024, 1, 31, 12, 0, 0),
            retention_days=30,
        )
        await cache.save_cache(test_cache)

        # Prune with 20-day retention
        removed_count = await cache.prune_old_cycles("ihp_salon", retention_days=20)

        # Should have removed 1 old cycle
        assert removed_count == 1

        # Verify pruned cache
        pruned_cache = await cache.load_cache("ihp_salon")
        assert pruned_cache is not None
        assert pruned_cache.size == 1
        assert pruned_cache.cycles[0].cycle_id == "recent_cycle"

    async def test_clear_cache(
        self, cache: FileBasedCycleCache, sample_cache: HeatingCycleCache
    ) -> None:
        """Test clearing the cache."""
        # Save cache
        await cache.save_cache(sample_cache)
        assert await cache.cache_exists("ihp_salon")

        # Clear cache
        cleared = await cache.clear_cache("ihp_salon")
        assert cleared

        # Verify cache is gone
        assert not await cache.cache_exists("ihp_salon")

    async def test_clear_nonexistent_cache_returns_false(
        self, cache: FileBasedCycleCache
    ) -> None:
        """Test that clearing nonexistent cache returns False."""
        cleared = await cache.clear_cache("nonexistent_device")
        assert not cleared

    async def test_corrupted_cache_raises_error(
        self, cache: FileBasedCycleCache, cache_dir: Path
    ) -> None:
        """Test that loading a corrupted cache raises IOError."""
        # Create corrupted cache file
        cache_path = cache_dir / "ihp_salon_cycles.json"
        with cache_path.open("w") as f:
            f.write("{invalid json")

        # Should raise IOError
        with pytest.raises(IOError, match="Corrupted cache file"):
            await cache.load_cache("ihp_salon")

    async def test_cache_file_path_sanitization(self, cache_dir: Path) -> None:
        """Test that device_id with path separators is sanitized."""
        cache = FileBasedCycleCache(cache_dir=str(cache_dir))

        # Device ID with path separators and special characters
        malicious_id = "../../../etc/passwd"
        cache_path = cache._get_cache_path(malicious_id)

        # Should be sanitized - all special chars replaced with underscores
        # Path should be within cache_dir
        assert str(cache_path).startswith(str(cache_dir))
        # The filename should only contain safe characters (alphanumeric, _, -)
        filename = cache_path.name
        # All path separators and dots should be replaced
        assert "/" not in filename
        assert "\\" not in filename
        assert ".." not in filename
