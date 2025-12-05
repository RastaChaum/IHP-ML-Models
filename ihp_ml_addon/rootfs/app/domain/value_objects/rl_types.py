"""Reinforcement Learning value objects and types.

Immutable data structures for RL operations.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class HeatingActionType(str, Enum):
    """Types of heating actions the RL agent can take.

    Attributes:
        TURN_ON: Turn heating on
        TURN_OFF: Turn heating off
        SET_TARGET_TEMPERATURE: Set a specific target temperature
        NO_OP: No operation (maintain current state)
    """

    TURN_ON = "turn_on"
    TURN_OFF = "turn_off"
    SET_TARGET_TEMPERATURE = "set_target_temperature"
    NO_OP = "no_op"


@dataclass(frozen=True)
class EntityState:
    """Represents the state of a Home Assistant entity.

    Attributes:
        entity_id: Home Assistant entity identifier
        last_changed_minutes: Minutes since the entity last changed
    """

    entity_id: str
    last_changed_minutes: float

    def __post_init__(self) -> None:
        """Validate entity state values."""
        if not self.entity_id:
            raise ValueError("entity_id cannot be empty")
        if self.last_changed_minutes < 0:
            raise ValueError(
                f"last_changed_minutes must be non-negative, got {self.last_changed_minutes}"
            )


@dataclass(frozen=True)
class RLObservation:
    """Complete observation state for the RL agent.

    This represents the full state of the environment at a given moment,
    including all sensor readings, target information, system state,
    and contextual data.

    Attributes:
        # Temperatures and Environmental Conditions
        current_temp: Current indoor temperature in °C
        current_temp_entity: Entity state for current temperature sensor
        outdoor_temp: Outdoor temperature in °C
        outdoor_temp_entity: Entity state for outdoor temperature sensor
        humidity: Relative humidity percentage (0-100)
        humidity_entity: Entity state for humidity sensor
        timestamp: When this observation was recorded

        # Target Information
        target_temp_from_schedule: Target temperature from scheduler (if available)
        time_until_target_minutes: Minutes until target should be reached
        current_target_achieved_percentage: Progress towards target (0-100%)

        # Heating System State
        is_heating_on: Whether heating is currently on
        heating_output_percent: Heating output percentage (if PWM/proportional control)
        energy_consumption_recent_kwh: Recent energy consumption in kWh
        time_heating_on_recent_seconds: Recent heating duration in seconds

        # Trends and Dynamics
        indoor_temp_change_15min: Indoor temperature change over last 15 minutes
        outdoor_temp_change_15min: Outdoor temperature change over last 15 minutes

        # Temporal Context
        day_of_week: Day of week (0=Monday, 6=Sunday)
        hour_of_day: Hour of the day (0-23)

        # Forecasts and Environmental Factors
        outdoor_temp_forecast_1h: Outdoor temperature forecast for 1 hour ahead
        outdoor_temp_forecast_3h: Outdoor temperature forecast for 3 hours ahead
        window_or_door_open: Whether any window or door is open
        window_or_door_entity: Entity state for window/door sensor

        # Identifier
        device_id: Zone/device identifier being controlled
    """

    # Temperatures and Environmental Conditions
    current_temp: float
    current_temp_entity: EntityState
    outdoor_temp: float
    outdoor_temp_entity: EntityState
    humidity: float
    humidity_entity: EntityState
    timestamp: datetime

    # Target Information
    target_temp_from_schedule: float | None
    time_until_target_minutes: int | None
    current_target_achieved_percentage: float | None

    # Heating System State
    is_heating_on: bool
    heating_output_percent: float | None
    energy_consumption_recent_kwh: float | None
    time_heating_on_recent_seconds: int | None

    # Trends and Dynamics
    indoor_temp_change_15min: float | None
    outdoor_temp_change_15min: float | None

    # Temporal Context
    day_of_week: int
    hour_of_day: int

    # Forecasts and Environmental Factors
    outdoor_temp_forecast_1h: float | None
    outdoor_temp_forecast_3h: float | None
    window_or_door_open: bool
    window_or_door_entity: EntityState | None

    # Identifier
    device_id: str

    def __post_init__(self) -> None:
        """Validate observation values."""
        # Temperature validations
        if not -50 <= self.outdoor_temp <= 60:
            raise ValueError(f"outdoor_temp must be between -50 and 60, got {self.outdoor_temp}")
        if not -20 <= self.current_temp <= 50:
            raise ValueError(f"current_temp must be between -20 and 50, got {self.current_temp}")
        if (
            self.target_temp_from_schedule is not None
            and not 0 <= self.target_temp_from_schedule <= 50
        ):
            raise ValueError(
                f"target_temp_from_schedule must be between 0 and 50, "
                f"got {self.target_temp_from_schedule}"
            )

        # Humidity validation
        if not 0 <= self.humidity <= 100:
            raise ValueError(f"humidity must be between 0 and 100, got {self.humidity}")

        # Target achievement validation
        if (
            self.current_target_achieved_percentage is not None
            and not 0 <= self.current_target_achieved_percentage <= 100
        ):
            raise ValueError(
                f"current_target_achieved_percentage must be between 0 and 100, "
                f"got {self.current_target_achieved_percentage}"
            )

        # Temporal validations
        if not 0 <= self.day_of_week <= 6:
            raise ValueError(f"day_of_week must be between 0 and 6, got {self.day_of_week}")
        if not 0 <= self.hour_of_day <= 23:
            raise ValueError(f"hour_of_day must be between 0 and 23, got {self.hour_of_day}")

        # Time validation
        if self.time_until_target_minutes is not None and self.time_until_target_minutes < 0:
            raise ValueError(
                f"time_until_target_minutes must be non-negative, "
                f"got {self.time_until_target_minutes}"
            )
        if (
            self.time_heating_on_recent_seconds is not None
            and self.time_heating_on_recent_seconds < 0
        ):
            raise ValueError(
                f"time_heating_on_recent_seconds must be non-negative, "
                f"got {self.time_heating_on_recent_seconds}"
            )

        # Energy and output validations
        if (
            self.energy_consumption_recent_kwh is not None
            and self.energy_consumption_recent_kwh < 0
        ):
            raise ValueError(
                f"energy_consumption_recent_kwh must be non-negative, "
                f"got {self.energy_consumption_recent_kwh}"
            )
        if (
            self.heating_output_percent is not None
            and not 0 <= self.heating_output_percent <= 100
        ):
            raise ValueError(
                f"heating_output_percent must be between 0 and 100, "
                f"got {self.heating_output_percent}"
            )

        # Device ID validation
        if not self.device_id:
            raise ValueError("device_id cannot be empty")

        # Forecast validations
        if (
            self.outdoor_temp_forecast_1h is not None
            and not -50 <= self.outdoor_temp_forecast_1h <= 60
        ):
            raise ValueError(
                f"outdoor_temp_forecast_1h must be between -50 and 60, "
                f"got {self.outdoor_temp_forecast_1h}"
            )
        if (
            self.outdoor_temp_forecast_3h is not None
            and not -50 <= self.outdoor_temp_forecast_3h <= 60
        ):
            raise ValueError(
                f"outdoor_temp_forecast_3h must be between -50 and 60, "
                f"got {self.outdoor_temp_forecast_3h}"
            )


@dataclass(frozen=True)
class RLAction:
    """Action taken by the RL agent.

    Attributes:
        action_type: Type of heating action to take
        value: Value associated with the action (e.g., target temperature)
        decision_timestamp: When this decision was made
        confidence_score: Optional confidence score for the action (0.0 to 1.0)
    """

    action_type: HeatingActionType
    value: float | None
    decision_timestamp: datetime
    confidence_score: float | None = None

    def __post_init__(self) -> None:
        """Validate action values."""
        # Value validation based on action type
        if self.action_type == HeatingActionType.SET_TARGET_TEMPERATURE:
            if self.value is None:
                raise ValueError(
                    "value must be provided for SET_TARGET_TEMPERATURE action"
                )
            if not 0 <= self.value <= 50:
                raise ValueError(
                    f"target temperature value must be between 0 and 50, got {self.value}"
                )
        elif self.action_type in (HeatingActionType.TURN_ON, HeatingActionType.TURN_OFF):
            # These actions typically don't need a value, but we allow it
            if self.value is not None and self.value < 0:
                raise ValueError(f"action value must be non-negative if provided, got {self.value}")

        # Confidence score validation
        if self.confidence_score is not None and not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(
                f"confidence_score must be between 0.0 and 1.0, got {self.confidence_score}"
            )


@dataclass(frozen=True)
class RLExperience:
    """A single experience tuple for reinforcement learning.

    This represents one step in the RL agent's interaction with the environment,
    following the standard (state, action, reward, next_state, done) format.

    Attributes:
        state: Current observation state
        action: Action taken in this state
        reward: Reward received after taking the action
        next_state: Resulting observation state after action
        done: Whether this experience marks the end of an episode
    """

    state: RLObservation
    action: RLAction
    reward: float
    next_state: RLObservation
    done: bool

    def __post_init__(self) -> None:
        """Validate experience values."""
        # Ensure state and next_state are for the same device
        if self.state.device_id != self.next_state.device_id:
            raise ValueError(
                f"state and next_state must be for the same device, "
                f"got {self.state.device_id} and {self.next_state.device_id}"
            )


@dataclass(frozen=True)
class TrainingRequest:
    """Request for training an RL model with historical data.

    Attributes:
        device_id: Device/zone identifier to train model for
        start_time: Start of historical period to fetch
        end_time: End of historical period to fetch

        # Entity IDs for data retrieval
        current_temp_entity_id: Entity ID for current indoor temperature
        outdoor_temp_entity_id: Entity ID for outdoor temperature
        humidity_entity_id: Entity ID for humidity sensor
        window_or_door_open_entity_id: Entity ID for window/door sensor (optional)
        heating_power_entity_id: Entity ID for heating power sensor (optional)
        heating_on_time_entity_id: Entity ID for heating on time sensor (optional)
        outdoor_temp_forecast_1h_entity_id: Entity ID for 1h forecast (optional)
        outdoor_temp_forecast_3h_entity_id: Entity ID for 3h forecast (optional)
        versatile_thermostat_entity_id: Entity ID for thermostat (target temp)

        # Training configuration
        behavioral_cloning_epochs: Number of epochs for initial behavioral cloning
        online_learning_enabled: Whether to enable online learning after initialization
    """

    device_id: str
    start_time: datetime
    end_time: datetime

    # Required entity IDs
    current_temp_entity_id: str
    outdoor_temp_entity_id: str
    humidity_entity_id: str
    versatile_thermostat_entity_id: str

    # Optional entity IDs
    window_or_door_open_entity_id: str | None = None
    heating_power_entity_id: str | None = None
    heating_on_time_entity_id: str | None = None
    outdoor_temp_forecast_1h_entity_id: str | None = None
    outdoor_temp_forecast_3h_entity_id: str | None = None

    # Training configuration
    behavioral_cloning_epochs: int = 10
    online_learning_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate training request values."""
        if not self.device_id:
            raise ValueError("device_id cannot be empty")
        if not self.current_temp_entity_id:
            raise ValueError("current_temp_entity_id cannot be empty")
        if not self.outdoor_temp_entity_id:
            raise ValueError("outdoor_temp_entity_id cannot be empty")
        if not self.humidity_entity_id:
            raise ValueError("humidity_entity_id cannot be empty")
        if not self.versatile_thermostat_entity_id:
            raise ValueError("versatile_thermostat_entity_id cannot be empty")

        if self.start_time >= self.end_time:
            raise ValueError("start_time must be before end_time")

        if self.behavioral_cloning_epochs < 0:
            raise ValueError(
                f"behavioral_cloning_epochs must be non-negative, "
                f"got {self.behavioral_cloning_epochs}"
            )
