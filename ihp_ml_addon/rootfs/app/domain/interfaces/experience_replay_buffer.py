"""Experience replay buffer interface.

Contract for storing and sampling RL experiences for training.
"""

from abc import ABC, abstractmethod

from domain.value_objects import RLExperience


class IExperienceReplayBuffer(ABC):
    """Contract for experience replay buffer operations.

    The experience replay buffer stores past experiences and allows
    the RL agent to sample random batches for training, which helps
    break temporal correlations and stabilize learning.
    """

    @abstractmethod
    async def add(self, experience: RLExperience) -> None:
        """Add a single experience to the buffer.

        Args:
            experience: The experience tuple to store

        Raises:
            ValueError: If the experience is invalid
        """
        pass

    @abstractmethod
    async def add_batch(self, experiences: tuple[RLExperience, ...]) -> None:
        """Add multiple experiences to the buffer.

        Args:
            experiences: Tuple of experiences to store

        Raises:
            ValueError: If any experience is invalid
        """
        pass

    @abstractmethod
    async def sample(self, batch_size: int) -> tuple[RLExperience, ...]:
        """Sample a random batch of experiences from the buffer.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of randomly sampled experiences

        Raises:
            ValueError: If batch_size is invalid or buffer is empty
            InsufficientDataError: If buffer has fewer experiences than batch_size
        """
        pass

    @abstractmethod
    async def clear(self, device_id: str | None = None) -> None:
        """Clear the buffer.

        Args:
            device_id: If provided, only clear experiences for this device.
                      If None, clear all experiences.
        """
        pass

    @abstractmethod
    async def size(self, device_id: str | None = None) -> int:
        """Get the current size of the buffer.

        Args:
            device_id: If provided, return size for this device only.
                      If None, return total size across all devices.

        Returns:
            Number of experiences currently in the buffer
        """
        pass

    @abstractmethod
    async def is_ready(self, min_size: int, device_id: str | None = None) -> bool:
        """Check if buffer has enough experiences for training.

        Args:
            min_size: Minimum number of experiences required
            device_id: If provided, check for this device only.
                      If None, check total size across all devices.

        Returns:
            True if buffer has at least min_size experiences
        """
        pass

    @abstractmethod
    async def get_all(self, device_id: str | None = None) -> tuple[RLExperience, ...]:
        """Get all experiences from the buffer.

        Args:
            device_id: If provided, return experiences for this device only.
                      If None, return all experiences.

        Returns:
            Tuple of all experiences in the buffer
        """
        pass
