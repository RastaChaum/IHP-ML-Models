"""File-based implementation of heating cycle cache.

This adapter persists heating cycles to JSON files on disk, providing
durable storage that survives service restarts.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from domain.interfaces import IHeatingCycleCache
from domain.value_objects import HeatingCycle, HeatingCycleCache

_LOGGER = logging.getLogger(__name__)


class FileBasedCycleCache(IHeatingCycleCache):
    """File-based implementation of heating cycle cache.

    This adapter stores cycles as JSON files in a specified directory.
    Each device has its own cache file: <cache_dir>/<device_id>_cycles.json

    The cache file format is:
    {
        "device_id": "ihp_salon",
        "last_scan_time": "2024-01-15T12:00:00",
        "retention_days": 30,
        "cache_version": 1,
        "cycles": [
            {
                "cycle_id": "...",
                "device_id": "...",
                "start_time": "...",
                ...
            }
        ]
    }
    """

    def __init__(self, cache_dir: str = "./data/cycle_cache") -> None:
        """Initialize the file-based cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        _LOGGER.info("File-based cycle cache initialized at: %s", self._cache_dir)

    def _get_cache_path(self, device_id: str) -> Path:
        """Get the cache file path for a device.

        Args:
            device_id: Device identifier

        Returns:
            Path to the cache file
        """
        # Sanitize device_id to avoid path traversal
        safe_device_id = device_id.replace("/", "_").replace("\\", "_")
        return self._cache_dir / f"{safe_device_id}_cycles.json"

    async def load_cache(self, device_id: str) -> HeatingCycleCache | None:
        """Load the cached cycles for a device.

        Args:
            device_id: Device identifier

        Returns:
            HeatingCycleCache if found, None if no cache exists

        Raises:
            IOError: If cache file is corrupted or cannot be read
        """
        _LOGGER.info("Loading cycle cache for device: %s", device_id)
        cache_path = self._get_cache_path(device_id)

        if not cache_path.exists():
            _LOGGER.info("No cache file found for device: %s", device_id)
            return None

        try:
            with cache_path.open("r") as f:
                data = json.load(f)

            # Parse cycles from JSON
            cycles = []
            for cycle_data in data.get("cycles", []):
                cycle = HeatingCycle(
                    cycle_id=cycle_data["cycle_id"],
                    device_id=cycle_data["device_id"],
                    start_time=datetime.fromisoformat(cycle_data["start_time"]),
                    end_time=datetime.fromisoformat(cycle_data["end_time"]),
                    start_indoor_temp=cycle_data["start_indoor_temp"],
                    end_indoor_temp=cycle_data["end_indoor_temp"],
                    target_temp=cycle_data["target_temp"],
                    outdoor_temp=cycle_data["outdoor_temp"],
                    humidity=cycle_data["humidity"],
                    duration_minutes=cycle_data["duration_minutes"],
                    hour_of_day=cycle_data["hour_of_day"],
                    minutes_since_last_cycle=cycle_data.get("minutes_since_last_cycle", 0.0),
                )
                cycles.append(cycle)

            cache = HeatingCycleCache(
                device_id=data["device_id"],
                cycles=tuple(cycles),
                last_scan_time=datetime.fromisoformat(data["last_scan_time"]),
                retention_days=data["retention_days"],
                cache_version=data.get("cache_version", 1),
            )

            _LOGGER.info(
                "Loaded %d cycles for device %s (last scan: %s)",
                len(cycles),
                device_id,
                cache.last_scan_time.isoformat(),
            )
            return cache

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            _LOGGER.error("Failed to load cache for device %s: %s", device_id, e)
            raise IOError(f"Corrupted cache file for device {device_id}") from e

    async def save_cache(self, cache: HeatingCycleCache) -> None:
        """Save the cache to persistent storage.

        Args:
            cache: HeatingCycleCache to persist

        Raises:
            IOError: If cache cannot be written
        """
        _LOGGER.info(
            "Saving cycle cache for device: %s (%d cycles)",
            cache.device_id,
            cache.size,
        )
        cache_path = self._get_cache_path(cache.device_id)

        try:
            # Convert cache to JSON-serializable dict
            data: dict[str, Any] = {
                "device_id": cache.device_id,
                "last_scan_time": cache.last_scan_time.isoformat(),
                "retention_days": cache.retention_days,
                "cache_version": cache.cache_version,
                "cycles": [
                    {
                        "cycle_id": cycle.cycle_id,
                        "device_id": cycle.device_id,
                        "start_time": cycle.start_time.isoformat(),
                        "end_time": cycle.end_time.isoformat(),
                        "start_indoor_temp": cycle.start_indoor_temp,
                        "end_indoor_temp": cycle.end_indoor_temp,
                        "target_temp": cycle.target_temp,
                        "outdoor_temp": cycle.outdoor_temp,
                        "humidity": cycle.humidity,
                        "duration_minutes": cycle.duration_minutes,
                        "hour_of_day": cycle.hour_of_day,
                        "minutes_since_last_cycle": cycle.minutes_since_last_cycle,
                    }
                    for cycle in cache.cycles
                ],
            }

            # Write to temp file first, then rename (atomic operation)
            temp_path = cache_path.with_suffix(".tmp")
            with temp_path.open("w") as f:
                json.dump(data, f, indent=2)

            temp_path.replace(cache_path)
            _LOGGER.debug("Cache saved successfully to: %s", cache_path)

        except (OSError, IOError) as e:
            _LOGGER.error("Failed to save cache for device %s: %s", cache.device_id, e)
            raise IOError(f"Failed to save cache for device {cache.device_id}") from e

    async def add_cycles(
        self,
        device_id: str,
        new_cycles: list[HeatingCycle],
        last_scan_time: datetime,
        retention_days: int,
    ) -> HeatingCycleCache:
        """Add new cycles to existing cache and prune old ones.

        This method:
        1. Loads existing cache (if any)
        2. Merges new cycles (avoiding duplicates by cycle_id)
        3. Prunes cycles older than retention period
        4. Updates last_scan_time
        5. Saves updated cache
        6. Returns the updated cache

        Args:
            device_id: Device identifier
            new_cycles: List of newly detected cycles
            last_scan_time: Timestamp of the scan that produced these cycles
            retention_days: Number of days to retain cycles

        Returns:
            Updated HeatingCycleCache after merge and pruning

        Raises:
            IOError: If cache cannot be read or written
        """
        _LOGGER.info(
            "Adding %d new cycles for device %s (retention: %d days)",
            len(new_cycles),
            device_id,
            retention_days,
        )

        # Load existing cache
        existing_cache = await self.load_cache(device_id)

        # Merge cycles (avoid duplicates by cycle_id)
        existing_cycles_dict = {}
        if existing_cache:
            existing_cycles_dict = {cycle.cycle_id: cycle for cycle in existing_cache.cycles}

        # Add new cycles (overwrite if cycle_id exists)
        for cycle in new_cycles:
            existing_cycles_dict[cycle.cycle_id] = cycle

        all_cycles = list(existing_cycles_dict.values())

        # Prune old cycles (older than retention period)
        cutoff_time = last_scan_time - timedelta(days=retention_days)
        pruned_cycles = [cycle for cycle in all_cycles if cycle.start_time >= cutoff_time]

        _LOGGER.info(
            "Cache merge complete: %d existing + %d new = %d total, %d after pruning",
            len(existing_cache.cycles) if existing_cache else 0,
            len(new_cycles),
            len(all_cycles),
            len(pruned_cycles),
        )

        # Sort cycles by start_time for better readability
        pruned_cycles.sort(key=lambda c: c.start_time)

        # Create updated cache
        updated_cache = HeatingCycleCache(
            device_id=device_id,
            cycles=tuple(pruned_cycles),
            last_scan_time=last_scan_time,
            retention_days=retention_days,
        )

        # Save to disk
        await self.save_cache(updated_cache)

        return updated_cache

    async def prune_old_cycles(self, device_id: str, retention_days: int) -> int:
        """Remove cycles older than the retention period.

        Args:
            device_id: Device identifier
            retention_days: Number of days to retain

        Returns:
            Number of cycles removed

        Raises:
            IOError: If cache cannot be updated
        """
        _LOGGER.info("Pruning old cycles for device %s (retention: %d days)", device_id, retention_days)

        cache = await self.load_cache(device_id)
        if cache is None:
            _LOGGER.debug("No cache found for device %s, nothing to prune", device_id)
            return 0

        cutoff_time = cache.last_scan_time - timedelta(days=retention_days)
        pruned_cycles = [cycle for cycle in cache.cycles if cycle.start_time >= cutoff_time]

        removed_count = len(cache.cycles) - len(pruned_cycles)

        if removed_count > 0:
            updated_cache = HeatingCycleCache(
                device_id=cache.device_id,
                cycles=tuple(pruned_cycles),
                last_scan_time=cache.last_scan_time,
                retention_days=retention_days,
            )
            await self.save_cache(updated_cache)
            _LOGGER.info("Pruned %d old cycles for device %s", removed_count, device_id)
        else:
            _LOGGER.debug("No cycles to prune for device %s", device_id)

        return removed_count

    async def clear_cache(self, device_id: str) -> bool:
        """Clear all cached cycles for a device.

        Args:
            device_id: Device identifier

        Returns:
            True if cache was cleared, False if no cache existed
        """
        _LOGGER.info("Clearing cycle cache for device: %s", device_id)
        cache_path = self._get_cache_path(device_id)

        if not cache_path.exists():
            _LOGGER.debug("No cache file to clear for device: %s", device_id)
            return False

        try:
            cache_path.unlink()
            _LOGGER.info("Cache cleared successfully for device: %s", device_id)
            return True
        except OSError as e:
            _LOGGER.error("Failed to clear cache for device %s: %s", device_id, e)
            raise IOError(f"Failed to clear cache for device {device_id}") from e

    async def cache_exists(self, device_id: str) -> bool:
        """Check if a cache exists for a device.

        Args:
            device_id: Device identifier

        Returns:
            True if cache exists, False otherwise
        """
        cache_path = self._get_cache_path(device_id)
        return cache_path.exists()
