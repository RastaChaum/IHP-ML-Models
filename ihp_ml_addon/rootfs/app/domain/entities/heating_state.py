"""Heating State entity.

Domain entity representing the state of a heating system.
"""

from dataclasses import dataclass


@dataclass
class HeatingState:
    """Represents the state of a heating system.

    This entity encapsulates business rules for determining heating status
    based on various climate entity attributes and sensor states.

    Attributes:
        is_on: Whether the heating system is turned on
        preset_mode: The current preset mode (e.g., 'comfort', 'eco')
        target_temp: The target temperature setting
    """

    is_on: bool
    preset_mode: str | None
    target_temp: float

    def is_heating(self, current_temp: float) -> bool:
        """Determine if the heating is actively heating.

        The heating is considered to be actively heating if:
        1. The heating system is on (is_on = True)
        2. The current temperature is below the target temperature

        Args:
            current_temp: The current temperature in °C

        Returns:
            True if the heating is actively heating, False otherwise
        """
        return self.is_on and current_temp < self.target_temp
