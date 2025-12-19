"""Interface for heating cycle cache operations.

Abstract base class defining the contract for cache implementations.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime

from domain.value_objects.heating_cycle import HeatingCycle, HeatingCycleCache

_LOGGER = logging.getLogger(__name__)


class IHeatingCycleCache(ABC):
    """Interface for heating cycle cache storage and retrieval.

    This interface defines the contract for caching heating cycles to avoid
    re-scanning the entire Home Assistant history on every training request.
    Implementations can be file-based, database-backed, or in-memory.
    """

    @abstractmethod
    async def load_cache(self, device_id: str) -> HeatingCycleCache | None:
        """Load the cached cycles for a device.

        Args:
            device_id: Device identifier

        Returns:
            HeatingCycleCache if found, None if no cache exists

        Raises:
            IOError: If cache file is corrupted or cannot be read
        """
        pass

    @abstractmethod
    async def save_cache(self, cache: HeatingCycleCache) -> None:
        """Save the cache to persistent storage.

        Args:
            cache: HeatingCycleCache to persist

        Raises:
            IOError: If cache cannot be written
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def clear_cache(self, device_id: str) -> bool:
        """Clear all cached cycles for a device.

        Args:
            device_id: Device identifier

        Returns:
            True if cache was cleared, False if no cache existed
        """
        pass

    @abstractmethod
    async def cache_exists(self, device_id: str) -> bool:
        """Check if a cache exists for a device.

        Args:
            device_id: Device identifier

        Returns:
            True if cache exists, False otherwise
        """
        pass
