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
        # Temperatures and Environmental Conditions (required)
        indoor_temp: Indoor temperature in °C
        indoor_temp_entity: Entity state for indoor temperature sensor
        timestamp: When this observation was recorded
        
        # Temperatures and Environmental Conditions (optional)
        outdoor_temp: Outdoor temperature in °C (optional)
        outdoor_temp_entity: Entity state for outdoor temperature sensor (optional)
        indoor_humidity: Relative humidity percentage (0-100, optional)
        indoor_humidity_entity: Entity state for humidity sensor (optional)

        # Target Information (required for RL decision-making)
        target_temp_from_schedule: Target temperature from scheduler
        target_temp_entity: Entity state for target temperature sensor
        time_until_target_minutes: Minutes until target should be reached
        current_target_achieved_percentage: Progress towards target (0-100%, optional)

        # Heating System State (required)
        is_heating_on: Whether heating is currently on
        
        # Heating System State (optional with entity states)
        heating_output_percent: Heating output percentage (if PWM/proportional control)
        heating_output_entity: Entity state for heating output sensor
        energy_consumption_recent_kwh: Recent energy consumption in kWh
        energy_consumption_entity: Entity state for energy consumption sensor
        time_heating_on_recent_seconds: Recent heating duration in seconds
        time_heating_on_entity: Entity state for heating on time sensor

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
    indoor_temp: float
    indoor_temp_entity: EntityState
    outdoor_temp: float | None
    outdoor_temp_entity: EntityState | None
    indoor_humidity: float | None
    indoor_humidity_entity: EntityState | None
    timestamp: datetime

    # Target Information (required for RL)
    target_temp: float
    target_temp_entity: EntityState
    time_until_target_minutes: int
    current_target_achieved_percentage: float | None

    # Heating System State (required)
    is_heating_on: bool
    heating_output_percent: float | None
    heating_output_entity: EntityState | None
    energy_consumption_recent_kwh: float | None
    energy_consumption_entity: EntityState | None
    time_heating_on_recent_seconds: int | None
    time_heating_on_entity: EntityState | None

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
        if self.outdoor_temp is not None and not -50 <= self.outdoor_temp <= 60:
            raise ValueError(f"outdoor_temp must be between -50 and 60, got {self.outdoor_temp}")
        if not -25 <= self.indoor_temp <= 50:
            raise ValueError(f"indoor_temp must be between -25 and 50, got {self.indoor_temp}")
        if not 0 <= self.target_temp <= 50:
            raise ValueError(
                f"target_temp_from_schedule must be between 0 and 50, "
                f"got {self.target_temp}"
            )

        # Humidity validation
        if self.indoor_humidity is not None and not 0 <= self.indoor_humidity <= 100:
            raise ValueError(f"indoor_humidity must be between 0 and 100, got {self.indoor_humidity}")

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

        # Time validation (required field)
        if self.time_until_target_minutes < 0:
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
            and not -25 <= self.outdoor_temp_forecast_1h <= 50
        ):
            raise ValueError(
                f"outdoor_temp_forecast_1h must be between -25 and 50, "
                f"got {self.outdoor_temp_forecast_1h}"
            )
        if (
            self.outdoor_temp_forecast_3h is not None
            and not -25 <= self.outdoor_temp_forecast_3h <= 50
        ):
            raise ValueError(
                f"outdoor_temp_forecast_3h must be between -25 and 50, "
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
        start_time: Start of historical period to fetch (None = use default history window)
        end_time: End of historical period to fetch (defaults to now)

        # Entity IDs for data retrieval (required)
        indoor_temp_entity_id: Entity ID for indoor temperature
        target_temp_entity_id: Entity ID for target temperature (from scheduler/thermostat)
        heating_state_entity_id: Entity ID for heating state (on/off)
        
        # Optional entity IDs
        outdoor_temp_entity_id: Entity ID for outdoor temperature (optional)
        indoor_humidity_entity_id: Entity ID for humidity sensor (optional)
        window_or_door_open_entity_id: Entity ID for window/door sensor (optional)
        heating_power_entity_id: Entity ID for heating power sensor (optional)
        heating_on_time_entity_id: Entity ID for heating on time sensor (optional)
        outdoor_temp_forecast_1h_entity_id: Entity ID for 1h forecast (optional)
        outdoor_temp_forecast_3h_entity_id: Entity ID for 3h forecast (optional)

        # Training configuration
        behavioral_cloning_epochs: Number of epochs for initial behavioral cloning
        online_learning_enabled: Whether to enable online learning after initialization
    """

    device_id: str
    
    # Required entity IDs
    indoor_temp_entity_id: str
    target_temp_entity_id: str
    heating_state_entity_id: str

    # Time range (start_time can be None for default window, end_time defaults to now)
    start_time: datetime | None = None
    end_time: datetime | None = None

    # Optional entity IDs
    outdoor_temp_entity_id: str | None = None
    indoor_humidity_entity_id: str | None = None
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
        if not self.indoor_temp_entity_id:
            raise ValueError("indoor_temp_entity_id cannot be empty")
        if not self.target_temp_entity_id:
            raise ValueError("target_temp_entity_id cannot be empty")
        if not self.heating_state_entity_id:
            raise ValueError("heating_state_entity_id cannot be empty")

        if self.end_time is None:
            object.__setattr__(self, "end_time", datetime.now())

        # Validate time range if both are provided
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.start_time >= self.end_time
        ):
            raise ValueError("start_time must be before end_time")

        if self.behavioral_cloning_epochs < 0:
            raise ValueError(
                f"behavioral_cloning_epochs must be non-negative, "
                f"got {self.behavioral_cloning_epochs}"
            )
