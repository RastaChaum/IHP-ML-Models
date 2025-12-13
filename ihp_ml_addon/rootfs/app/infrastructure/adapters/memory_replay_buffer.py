"""In-memory implementation of experience replay buffer.

Infrastructure adapter for storing RL experiences in memory.
"""

import logging
import random
from collections import deque

from domain.interfaces import IExperienceReplayBuffer
from domain.value_objects import RLExperience

_LOGGER = logging.getLogger(__name__)


class MemoryReplayBuffer(IExperienceReplayBuffer):
    """In-memory implementation of experience replay buffer.

    This adapter stores experiences in memory using a deque (double-ended queue)
    with a maximum capacity. When the buffer is full, the oldest experiences
    are automatically removed (FIFO).

    Attributes:
        max_capacity: Maximum number of experiences to store
    """

    def __init__(self, max_capacity: int = 10000) -> None:
        """Initialize the memory replay buffer.

        Args:
            max_capacity: Maximum number of experiences to store (default: 10000)

        Raises:
            ValueError: If max_capacity is less than 1
        """
        if max_capacity < 1:
            raise ValueError(f"max_capacity must be at least 1, got {max_capacity}")

        self._max_capacity = max_capacity
        # Store experiences grouped by device_id for efficient filtering
        self._buffers: dict[str, deque[RLExperience]] = {}
        _LOGGER.info("Initialized MemoryReplayBuffer with max_capacity=%d", max_capacity)

    async def add(self, experience: RLExperience) -> None:
        """Add a single experience to the buffer.

        Args:
            experience: The experience tuple to store

        Raises:
            ValueError: If the experience is invalid
        """
        _LOGGER.debug(
            "Adding experience for device %s (timestamp=%s)",
            experience.state.device_id,
            experience.state.timestamp,
        )

        device_id = experience.state.device_id

        # Create buffer for device if it doesn't exist
        if device_id not in self._buffers:
            self._buffers[device_id] = deque(maxlen=self._max_capacity)

        # Add experience to device-specific buffer
        self._buffers[device_id].append(experience)

        _LOGGER.debug(
            "Buffer size for device %s: %d experiences",
            device_id,
            len(self._buffers[device_id]),
        )

    async def add_batch(self, experiences: tuple[RLExperience, ...]) -> None:
        """Add multiple experiences to the buffer.

        Args:
            experiences: Tuple of experiences to store

        Raises:
            ValueError: If any experience is invalid
        """
        _LOGGER.info("Adding batch of %d experiences", len(experiences))

        for experience in experiences:
            await self.add(experience)

        _LOGGER.info(
            "Batch added successfully. Total buffer size: %d experiences",
            await self.size(),
        )

    async def sample(self, batch_size: int) -> tuple[RLExperience, ...]:
        """Sample a random batch of experiences from the buffer.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of randomly sampled experiences

        Raises:
            ValueError: If batch_size is invalid or buffer is empty
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")

        # Get all experiences from all devices
        all_experiences = []
        for buffer in self._buffers.values():
            all_experiences.extend(buffer)

        if not all_experiences:
            raise ValueError("Cannot sample from empty buffer")

        if len(all_experiences) < batch_size:
            raise ValueError(
                f"Buffer has only {len(all_experiences)} experiences, "
                f"cannot sample {batch_size}"
            )

        # Sample randomly
        sampled = random.sample(all_experiences, batch_size)

        _LOGGER.debug(
            "Sampled %d experiences from buffer (total size: %d)",
            batch_size,
            len(all_experiences),
        )

        return tuple(sampled)

    async def clear(self, device_id: str | None = None) -> None:
        """Clear the buffer.

        Args:
            device_id: If provided, only clear experiences for this device.
                      If None, clear all experiences.
        """
        if device_id is None:
            # Clear all buffers
            total_before = sum(len(buffer) for buffer in self._buffers.values())
            self._buffers.clear()
            _LOGGER.info("Cleared all buffers (%d experiences removed)", total_before)
        else:
            # Clear specific device buffer
            if device_id in self._buffers:
                count_before = len(self._buffers[device_id])
                self._buffers[device_id].clear()
                _LOGGER.info(
                    "Cleared buffer for device %s (%d experiences removed)",
                    device_id,
                    count_before,
                )
            else:
                _LOGGER.debug("No buffer found for device %s", device_id)

    async def size(self, device_id: str | None = None) -> int:
        """Get the current size of the buffer.

        Args:
            device_id: If provided, return size for this device only.
                      If None, return total size across all devices.

        Returns:
            Number of experiences currently in the buffer
        """
        if device_id is None:
            # Return total size across all devices
            return sum(len(buffer) for buffer in self._buffers.values())
        else:
            # Return size for specific device
            return len(self._buffers.get(device_id, []))

    async def is_ready(self, min_size: int, device_id: str | None = None) -> bool:
        """Check if buffer has enough experiences for training.

        Args:
            min_size: Minimum number of experiences required
            device_id: If provided, check for this device only.
                      If None, check total size across all devices.

        Returns:
            True if buffer has at least min_size experiences
        """
        current_size = await self.size(device_id)
        return current_size >= min_size

    async def get_all(self, device_id: str | None = None) -> tuple[RLExperience, ...]:
        """Get all experiences from the buffer.

        Args:
            device_id: If provided, return experiences for this device only.
                      If None, return all experiences.

        Returns:
            Tuple of all experiences in the buffer
        """
        if device_id is None:
            # Return all experiences from all devices
            all_experiences = []
            for buffer in self._buffers.values():
                all_experiences.extend(buffer)
            return tuple(all_experiences)
        else:
            # Return experiences for specific device
            if device_id in self._buffers:
                return tuple(self._buffers[device_id])
            else:
                return tuple()
