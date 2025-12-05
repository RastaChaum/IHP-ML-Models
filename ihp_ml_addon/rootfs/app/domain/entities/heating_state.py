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

    @classmethod
    def from_ha_state_record(cls, state_record: dict) -> "HeatingState":
        """Create a HeatingState from a Home Assistant state record.

        This method extracts the heating state from a Home Assistant history
        record, supporting both climate entities and binary sensors/switches.

        Args:
            state_record: Home Assistant state record dictionary

        Returns:
            HeatingState instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        entity_id = state_record.get("entity_id", "")
        is_climate = entity_id.startswith("climate.")

        if is_climate:
            # Extract from climate entity
            attributes = state_record.get("attributes", {})
            hvac_action = attributes.get("hvac_action", "")
            hvac_mode = attributes.get("hvac_mode", "")
            state = state_record.get("state", "")

            # Heating is ON if hvac_action is 'heating' OR state is 'heat'/'heating'
            is_on = (
                (hvac_action and hvac_action.lower() in ("heating", "on"))
                or (state and state.lower() in ("heat", "heating"))
                or (hvac_mode and hvac_mode.lower() == "heat")
            )

            preset_mode = attributes.get("preset_mode")
            target_temp = attributes.get("temperature")

            if target_temp is None:
                raise ValueError(f"Climate entity {entity_id} missing temperature attribute")

            return cls(
                is_on=is_on,
                preset_mode=preset_mode,
                target_temp=float(target_temp),
            )
        else:
            # For binary_sensor or switch: check state only
            state = state_record.get("state", "").lower()
            is_on = state in ("on", "heat", "heating", "true", "1")

            # Binary sensors don't have preset_mode or target_temp
            # These would need to come from separate entities
            return cls(
                is_on=is_on,
                preset_mode=None,
                target_temp=0.0,  # Default, should be set from separate source
            )
