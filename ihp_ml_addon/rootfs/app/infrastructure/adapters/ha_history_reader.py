"""Home Assistant REST API history reader adapter.

Infrastructure adapter that implements IHomeAssistantHistoryReader
using Home Assistant's REST API.

Note: This adapter uses the synchronous requests library. While the methods
are declared async to match the interface, they run synchronously within
the Flask application context (which uses asyncio.run() for async routes).
For a production system with high concurrency, consider using aiohttp.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urljoin

import requests
from domain.interfaces import IHomeAssistantHistoryReader
from domain.interfaces.reward_calculator import IRewardCalculator
from domain.value_objects import (
    EntityState,
    HeatingActionType,
    RLAction,
    RLExperience,
    RLObservation,
    TrainingData,
    TrainingDataPoint,
    TrainingRequest,
    get_week_of_month,
)

_LOGGER = logging.getLogger(__name__)


class HomeAssistantHistoryReader(IHomeAssistantHistoryReader):
    """Home Assistant REST API implementation of history reader.

    This adapter connects to Home Assistant's REST API to fetch
    historical sensor data for training the ML model.

    Note: Uses synchronous requests library. In async context, this will
    block the current thread but is acceptable for Flask's threading model.
    """

    def __init__(
        self,
        ha_url: str | None = None,
        ha_token: str | None = None,
        timeout: int = 30,
        reward_calculator: IRewardCalculator | None = None,
    ) -> None:
        """Initialize the Home Assistant history reader.

        Args:
            ha_url: Home Assistant URL (defaults to supervisor API)
            ha_token: Long-lived access token (defaults to supervisor token)
            timeout: Request timeout in seconds
            reward_calculator: Optional reward calculator for RL experience construction
        """
        # Default to Supervisor API for addons
        self._ha_url = ha_url or os.getenv(
            "SUPERVISOR_URL", "http://supervisor/core"
        )
        self._ha_token = ha_token or os.getenv("SUPERVISOR_TOKEN", "")
        self._timeout = timeout
        self._reward_calculator = reward_calculator

        _LOGGER.info("HA History Reader initialized with URL: %s", self._ha_url)

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for Home Assistant API requests."""
        return {
            "Authorization": f"Bearer {self._ha_token}",
            "Content-Type": "application/json",
        }

    async def is_available(self) -> bool:
        """Check if Home Assistant API is available.

        Returns:
            True if the addon can communicate with Home Assistant
        """
        _LOGGER.info("=" * 60)
        _LOGGER.info("Checking Home Assistant availability")
        _LOGGER.info("Base URL: %s", self._ha_url)
        _LOGGER.info("Token configured: %s", "YES" if self._ha_token else "NO")
        _LOGGER.info("Token length: %d", len(self._ha_token) if self._ha_token else 0)
        
        try:
            # Ensure base URL ends with / for proper urljoin behavior
            base_url = self._ha_url if self._ha_url.endswith('/') else f"{self._ha_url}/"
            url = urljoin(base_url, "api/")
            _LOGGER.info("Final URL after urljoin: %s", url)
            _LOGGER.debug("Request headers: %s", {k: v[:20] + "..." if k == "Authorization" and len(v) > 20 else v for k, v in self._get_headers().items()})
            
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=self._timeout,
            )
            _LOGGER.info("Response status: %d", response.status_code)
            _LOGGER.debug("Response body: %s", response.text[:200] if response.text else "(empty)")
            _LOGGER.info("=" * 60)
            return response.status_code == 200
        except requests.RequestException as e:
            _LOGGER.error("Home Assistant API error: %s", e)
            _LOGGER.error("Error type: %s", type(e).__name__)
            _LOGGER.info("=" * 60)
            return False

    async def fetch_training_data(
        self,
        indoor_temp_entity_id: str,
        outdoor_temp_entity_id: str,
        target_temp_entity_id: str,
        heating_state_entity_id: str,
        humidity_entity_id: str | None,
        start_time: datetime,
        end_time: datetime,
        cycle_split_duration_minutes: int | None = None,
    ) -> TrainingData:
        """Fetch historical data and convert to training data.

        This method:
        1. Fetches history for all specified entities
        2. Aligns timestamps across sensors
        3. Identifies heating cycles (when heating turned on/off)
        4. Calculates heating duration for each cycle
        5. Optionally splits long cycles into smaller sub-cycles
        6. Returns TrainingData for model training

        Args:
            indoor_temp_entity_id: Entity ID for indoor temperature sensor
            outdoor_temp_entity_id: Entity ID for outdoor temperature sensor
            target_temp_entity_id: Entity ID for target temperature
            heating_state_entity_id: Entity ID for heating state (on/off)
            humidity_entity_id: Entity ID for humidity sensor (optional)
            start_time: Start of the time range for fetching history
            end_time: End of the time range for fetching history
            cycle_split_duration_minutes: Optional duration in minutes to split
                long heating cycles into smaller sub-cycles. If None, cycles
                are not split.

        Returns:
            TrainingData with extracted heating cycles
        """
        entity_ids = [
            indoor_temp_entity_id,
            outdoor_temp_entity_id,
            target_temp_entity_id,
            heating_state_entity_id,
        ]
        if humidity_entity_id:
            entity_ids.append(humidity_entity_id)

        # Fetch history for all entities
        history_data = await self._fetch_history(entity_ids, start_time, end_time)

        # Extract heating cycles and create training data points
        data_points = self._extract_heating_cycles(
            history_data,
            indoor_temp_entity_id,
            outdoor_temp_entity_id,
            target_temp_entity_id,
            heating_state_entity_id,
            humidity_entity_id,
            cycle_split_duration_minutes,
        )

        if not data_points:
            raise ValueError("No valid heating cycles found in historical data")

        _LOGGER.info(
            "Extracted %d training data points from %s to %s",
            len(data_points),
            start_time.isoformat(),
            end_time.isoformat(),
        )

        return TrainingData.from_sequence(data_points)

    async def _fetch_history(
        self,
        entity_ids: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch history data for multiple entities.

        Home Assistant limits responses to ~4000 records. This method automatically
        splits large time ranges into smaller chunks and merges the results.

        Args:
            entity_ids: List of entity IDs to fetch
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Dictionary mapping entity_id to list of state records
        """
        from datetime import timedelta

        # Calculate time range
        total_days = (end_time - start_time).days
        
        # If period is less than 7 days, fetch in one request
        if total_days <= 7:
            return await self._fetch_history_chunk(entity_ids, start_time, end_time)
        
        # Otherwise, split into weekly chunks to avoid HA API limits
        _LOGGER.info(
            "Fetching %d days of history in chunks to avoid API limits...",
            total_days,
        )
        
        chunk_size_days = 7
        result: dict[str, list[dict[str, Any]]] = {}
        
        current_start = start_time
        chunk_num = 0
        
        while current_start < end_time:
            chunk_num += 1
            current_end = min(current_start + timedelta(days=chunk_size_days), end_time)
            
            _LOGGER.debug(
                "Fetching chunk %d: %s to %s (%d days)",
                chunk_num,
                current_start.isoformat(),
                current_end.isoformat(),
                (current_end - current_start).days,
            )
            
            # Fetch this chunk
            chunk_data = await self._fetch_history_chunk(
                entity_ids,
                current_start,
                current_end,
            )
            
            # Merge with accumulated results
            for entity_id, records in chunk_data.items():
                if entity_id not in result:
                    result[entity_id] = []
                result[entity_id].extend(records)
            
            # Move to next chunk
            current_start = current_end
        
        # Sort all entities chronologically after merging
        for entity_id in result:
            result[entity_id] = sorted(
                result[entity_id],
                key=lambda x: x.get("last_changed") or x.get("last_updated") or "",
            )
            _LOGGER.info(
                "Entity %s: %d total records after merging %d chunks",
                entity_id,
                len(result[entity_id]),
                chunk_num,
            )
        
        return result

    async def _fetch_history_chunk(
        self,
        entity_ids: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch a single chunk of history data.

        Args:
            entity_ids: List of entity IDs to fetch
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Dictionary mapping entity_id to list of state records
        """
        # Format time for HA API
        start_str = start_time.isoformat()
        end_str = end_time.isoformat()

        # Build the history URL with filter_entity_id parameter
        entity_filter = ",".join(entity_ids)
        # Ensure base URL ends with / for proper urljoin behavior
        base_url = self._ha_url if self._ha_url.endswith('/') else f"{self._ha_url}/"
        url = urljoin(
            base_url,
            f"api/history/period/{start_str}?end_time={end_str}"
            f"&filter_entity_id={entity_filter}&minimal_response",
        )

        _LOGGER.debug("Fetching history chunk from: %s", url)

        try:
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=self._timeout,
            )
            response.raise_for_status()
            history_list = response.json()
        except requests.RequestException as e:
            _LOGGER.error("Failed to fetch history chunk: %s", e)
            raise ConnectionError(f"Failed to fetch history from Home Assistant: {e}") from e

        # Convert list of entity histories to dictionary and sort chronologically
        result: dict[str, list[dict[str, Any]]] = {}
        for entity_history in history_list:
            if entity_history:
                entity_id = entity_history[0].get("entity_id", "")
                if entity_id:
                    # Sort history by timestamp (chronological order)
                    sorted_history = sorted(
                        entity_history,
                        key=lambda x: x.get("last_changed") or x.get("last_updated") or "",
                    )
                    result[entity_id] = sorted_history
                    _LOGGER.debug(
                        "Entity %s: %d records from %s to %s",
                        entity_id,
                        len(sorted_history),
                        sorted_history[0].get("last_changed") if sorted_history else "N/A",
                        sorted_history[-1].get("last_changed") if sorted_history else "N/A",
                    )

        _LOGGER.debug("Fetched history for entities: %s", list(result.keys()))
        return result

    def _extract_heating_cycles(
        self,
        history_data: dict[str, list[dict[str, Any]]],
        indoor_temp_entity_id: str,
        outdoor_temp_entity_id: str,
        target_temp_entity_id: str,
        heating_state_entity_id: str,
        humidity_entity_id: str | None,
        cycle_split_duration_minutes: int | None = None,
    ) -> list[TrainingDataPoint]:
        """Extract heating cycles from historical data.

        A heating cycle is defined as:
        - START: Heating is ON AND temperature delta (target - current) > 0.2°C
        - END: Heating is OFF OR temperature delta <= 0.2°C OR current temp exceeds target

        If cycle_split_duration_minutes is provided, long heating cycles will be
        split into smaller sub-cycles. For example, a 3-hour cycle with a split
        duration of 60 minutes will be split into 3 sub-cycles of 60 minutes each.
        The target temperature for each sub-cycle is calculated by linear
        interpolation between the start and end temperatures.

        Args:
            history_data: Dictionary of entity_id -> history records
            indoor_temp_entity_id: Entity ID for indoor temperature
            outdoor_temp_entity_id: Entity ID for outdoor temperature
            target_temp_entity_id: Entity ID for target temperature
            heating_state_entity_id: Entity ID for heating state
            humidity_entity_id: Entity ID for humidity (optional)
            cycle_split_duration_minutes: Optional duration in minutes to split
                long heating cycles into smaller sub-cycles. If None, cycles
                are not split.

        Returns:
            List of TrainingDataPoint objects
        """
        # Temperature threshold for cycle detection (in °C)
        TEMP_DELTA_THRESHOLD = 0.2

        data_points: list[TrainingDataPoint] = []

        # Get heating state history
        heating_states = history_data.get(heating_state_entity_id, [])
        if not heating_states:
            _LOGGER.warning("No heating state history found for %s", heating_state_entity_id)
            return data_points

        # Get temperature histories for cycle end detection
        indoor_temp_history = history_data.get(indoor_temp_entity_id, [])
        target_temp_history = history_data.get(target_temp_entity_id, [])
        
        # Detect entity types to determine how to extract values
        # Check if entities are climate entities (with attributes) or sensors (state only)
        def is_climate_entity(entity_id: str) -> bool:
            """Check if entity is a climate entity."""
            return entity_id.startswith("climate.")

        indoor_is_climate = is_climate_entity(indoor_temp_entity_id)
        outdoor_is_climate = is_climate_entity(outdoor_temp_entity_id)
        target_is_climate = is_climate_entity(target_temp_entity_id)
        heating_is_climate = is_climate_entity(heating_state_entity_id)
        humidity_is_climate = humidity_entity_id and is_climate_entity(humidity_entity_id)

        _LOGGER.debug("Entity type detection:")
        _LOGGER.debug("  Indoor temp: %s (climate=%s)", indoor_temp_entity_id, indoor_is_climate)
        _LOGGER.debug("  Outdoor temp: %s (climate=%s)", outdoor_temp_entity_id, outdoor_is_climate)
        _LOGGER.debug("  Target temp: %s (climate=%s)", target_temp_entity_id, target_is_climate)
        _LOGGER.debug("  Heating state: %s (climate=%s)", heating_state_entity_id, heating_is_climate)
        if cycle_split_duration_minutes:
            _LOGGER.debug("  Cycle split duration: %d minutes", cycle_split_duration_minutes)

        # Detect entity types to determine how to extract values
        # Check if entities are climate entities (with attributes) or sensors (state only)
        def is_climate_entity(entity_id: str) -> bool:
            """Check if entity is a climate entity."""
            return entity_id.startswith("climate.")

        indoor_is_climate = is_climate_entity(indoor_temp_entity_id)
        outdoor_is_climate = is_climate_entity(outdoor_temp_entity_id)
        target_is_climate = is_climate_entity(target_temp_entity_id)
        heating_is_climate = is_climate_entity(heating_state_entity_id)
        humidity_is_climate = humidity_entity_id and is_climate_entity(humidity_entity_id)

        _LOGGER.debug("Entity type detection:")
        _LOGGER.debug("  Indoor temp: %s (climate=%s)", indoor_temp_entity_id, indoor_is_climate)
        _LOGGER.debug("  Outdoor temp: %s (climate=%s)", outdoor_temp_entity_id, outdoor_is_climate)
        _LOGGER.debug("  Target temp: %s (climate=%s)", target_temp_entity_id, target_is_climate)
        _LOGGER.debug("  Heating state: %s (climate=%s)", heating_state_entity_id, heating_is_climate)

        # Track heating cycles
        heating_start: datetime | None = None
        start_indoor_temp: float | None = None
        start_outdoor_temp: float | None = None
        start_humidity: float | None = None
        start_target_temp: float | None = None

        def reset_cycle() -> None:
            """Reset cycle tracking variables."""
            nonlocal heating_start, start_indoor_temp, start_outdoor_temp
            nonlocal start_humidity, start_target_temp
            heating_start = None
            start_indoor_temp = None
            start_outdoor_temp = None
            start_humidity = None
            start_target_temp = None

        def record_cycle(end_timestamp: datetime, end_temp: float | None = None) -> None:
            """Record a completed heating cycle.

            If cycle_split_duration_minutes is set and the cycle is longer than
            that duration, it will be split into multiple sub-cycles.

            Args:
                end_timestamp: The timestamp when the cycle ended
                end_temp: The indoor temperature at the end of the cycle (optional)
            """
            nonlocal data_points
            nonlocal last_cycle_end_time
            if heating_start is None:
                return

            duration_minutes = (end_timestamp - heating_start).total_seconds() / 60.0

            # Only use cycles with valid data
            if not (
                start_indoor_temp is not None
                and start_outdoor_temp is not None
                and start_target_temp is not None
                and duration_minutes > 0
                and duration_minutes < 300  # Max 5 hours for a single cycle
            ):
                return

            # Use end_temp if provided (for cycle splitting), otherwise use start_target_temp
            final_temp = end_temp if end_temp is not None else start_target_temp

            # Check if we should split this cycle
            if (
                cycle_split_duration_minutes is not None
                and duration_minutes > cycle_split_duration_minutes
                and final_temp is not None
            ):
                # Split the cycle into smaller sub-cycles
                num_sub_cycles = int(duration_minutes / cycle_split_duration_minutes)
                # Calculate remaining time after full sub-cycles
                remaining_minutes = duration_minutes - (num_sub_cycles * cycle_split_duration_minutes)

                _LOGGER.debug(
                    "Splitting %d-minute cycle into %d sub-cycles of %d minutes (remaining: %.1f min)",
                    int(duration_minutes),
                    num_sub_cycles + (1 if remaining_minutes >= 5 else 0),
                    cycle_split_duration_minutes,
                    remaining_minutes,
                )

                # Calculate temperature change per minute for linear interpolation
                temp_delta = final_temp - start_indoor_temp
                temp_per_minute = temp_delta / duration_minutes

                current_start_time = heating_start
                current_start_temp = start_indoor_temp
                # Minutes since last cycle applies to the first sub-cycle; subsequent are contiguous
                minutes_since_prev = 0.0
                if last_cycle_end_time is not None:
                    minutes_since_prev = max(
                        0.0,
                        (current_start_time - last_cycle_end_time).total_seconds() / 60.0,
                    )

                for _ in range(num_sub_cycles):
                    sub_cycle_duration = float(cycle_split_duration_minutes)
                    sub_cycle_end_time = current_start_time + timedelta(minutes=sub_cycle_duration)
                    # Calculate the temperature reached at the end of this sub-cycle
                    sub_cycle_end_temp = current_start_temp + (temp_per_minute * sub_cycle_duration)

                    try:
                        data_point = TrainingDataPoint(
                            outdoor_temp=start_outdoor_temp,
                            indoor_temp=current_start_temp,
                            target_temp=sub_cycle_end_temp,
                            humidity=start_humidity or 50.0,
                            hour_of_day=current_start_time.hour,
                            # day_of_week=current_start_time.weekday(),
                            # week_of_month=get_week_of_month(current_start_time),
                            # month=current_start_time.month,
                            minutes_since_last_cycle=minutes_since_prev,
                            heating_duration_minutes=sub_cycle_duration,
                            timestamp=current_start_time,
                        )
                        data_points.append(data_point)
                    except ValueError as e:
                        _LOGGER.debug("Skipping invalid sub-cycle data point: %s", e)

                    # Move to the next sub-cycle
                    current_start_time = sub_cycle_end_time
                    current_start_temp = sub_cycle_end_temp
                    # After the first segment, subsequent are contiguous
                    minutes_since_prev = 0.0
                    # Update last cycle end time to this sub-cycle end
                    last_cycle_end_time = sub_cycle_end_time

                # Handle remaining time if significant (>= 5 minutes)
                if remaining_minutes >= 5:
                    try:
                        data_point = TrainingDataPoint(
                            outdoor_temp=start_outdoor_temp,
                            indoor_temp=current_start_temp,
                            target_temp=final_temp,
                            humidity=start_humidity or 50.0,
                            hour_of_day=current_start_time.hour,
                            # day_of_week=current_start_time.weekday(),
                            # week_of_month=get_week_of_month(current_start_time),
                            # month=current_start_time.month,
                            minutes_since_last_cycle=0.0,
                            heating_duration_minutes=remaining_minutes,
                            timestamp=current_start_time,
                        )
                        data_points.append(data_point)
                    except ValueError as e:
                        _LOGGER.debug("Skipping invalid remaining sub-cycle data point: %s", e)
                    # Update last cycle end time to this remaining sub-cycle end
                    last_cycle_end_time = current_start_time + timedelta(minutes=remaining_minutes)
            else:
                # No splitting - record the cycle as-is (original behavior)
                try:
                    # Compute minutes since the previous cycle ended
                    minutes_since_prev = 0.0
                    if last_cycle_end_time is not None:
                        minutes_since_prev = max(
                            0.0,
                            (heating_start - last_cycle_end_time).total_seconds() / 60.0,
                        )
                    data_point = TrainingDataPoint(
                        outdoor_temp=start_outdoor_temp,
                        indoor_temp=start_indoor_temp,
                        target_temp=final_temp,
                        humidity=start_humidity or 50.0,
                        hour_of_day=heating_start.hour,
                        # day_of_week=heating_start.weekday(),
                        # week_of_month=get_week_of_month(heating_start),
                        # month=heating_start.month,
                        minutes_since_last_cycle=minutes_since_prev,
                        heating_duration_minutes=duration_minutes,
                        timestamp=heating_start,
                    )
                    data_points.append(data_point)
                except ValueError as e:
                    _LOGGER.debug("Skipping invalid data point: %s", e)
            # Update the last cycle end time to the end of the recorded cycle
            last_cycle_end_time = end_timestamp

        # Track the end time of the last recorded cycle to compute gaps
        last_cycle_end_time: datetime | None = None

        for state_record in heating_states:
            timestamp_str = state_record.get("last_changed") or state_record.get("last_updated")

            if not timestamp_str:
                continue

            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            # Determine if heating is ON
            # For climate entities: check hvac_action in attributes OR state
            is_heating = False
            if heating_is_climate:
                attributes = state_record.get("attributes", {})
                hvac_action = attributes.get("hvac_action", "")
                hvac_mode = attributes.get("hvac_mode", "")
                state = state_record.get("state", "")
                
                # Heating is ON if hvac_action is 'heating' OR state is 'heat'/'heating'
                is_heating = (
                    (hvac_action and (hvac_action.lower() in ("heating", "on"))) or
                    (state and state.lower() in ("heat", "heating")) or
                    (hvac_mode and hvac_mode.lower() == "heat")
                )
            else:
                # For binary_sensor or switch: check state
                state = state_record.get("state", "").lower()
                is_heating = state in ("on", "heat", "heating", "true", "1")

            # Get current temperatures at this timestamp
            # Use appropriate attribute names for climate entities
            if indoor_is_climate:
                current_indoor = self._get_value_at_time(
                    indoor_temp_history, timestamp, attribute_name="current_temperature"
                )
            else:
                current_indoor = self._get_value_at_time(indoor_temp_history, timestamp)

            if target_is_climate:
                current_target = self._get_value_at_time(
                    target_temp_history, timestamp, attribute_name="temperature"
                )
            else:
                current_target = self._get_value_at_time(target_temp_history, timestamp)

            if heating_start is None:
                # Not in a cycle - check if we should start one
                if is_heating and current_indoor is not None and current_target is not None:
                    temp_delta = current_target - current_indoor
                    if temp_delta > TEMP_DELTA_THRESHOLD:
                        # Start a heating cycle
                        heating_start = timestamp
                        start_indoor_temp = current_indoor
                        start_target_temp = current_target
                        
                        # Get outdoor temperature
                        if outdoor_is_climate:
                            start_outdoor_temp = self._get_value_at_time(
                                history_data.get(outdoor_temp_entity_id, []),
                                timestamp,
                                attribute_name="ext_current_temperature",
                            )
                        else:
                            start_outdoor_temp = self._get_value_at_time(
                                history_data.get(outdoor_temp_entity_id, []),
                                timestamp,
                            )
                        
                        # Get humidity
                        if humidity_entity_id:
                            if humidity_is_climate:
                                start_humidity = self._get_value_at_time(
                                    history_data.get(humidity_entity_id, []),
                                    timestamp,
                                    attribute_name="humidity",
                                )
                            else:
                                start_humidity = self._get_value_at_time(
                                    history_data.get(humidity_entity_id, []),
                                    timestamp,
                                )
                        else:
                            start_humidity = 50.0  # Default humidity
            else:
                # Currently in a cycle - check if we should end it
                cycle_ended = False
                end_reason = ""

                if not is_heating:
                    # Heating turned off
                    cycle_ended = True
                    end_reason = "heating_off"
                    start_target_temp = current_indoor  # Record final target temp as current indoor
                elif current_indoor is not None and current_target is not None:
                    temp_delta = current_target - current_indoor
                    if current_indoor > current_target:
                        # Temperature exceeded target
                        cycle_ended = True
                        end_reason = "target_exceeded"
                        start_target_temp = current_indoor  # Record final target temp as current indoor
                    elif temp_delta <= TEMP_DELTA_THRESHOLD:
                        # Target reached (within threshold)
                        cycle_ended = True
                        end_reason = "target_reached"
                        start_target_temp = current_indoor  # Record final target temp as current indoor

                if cycle_ended:
                    _LOGGER.debug(
                        "Heating cycle ended at %s (reason: %s)",
                        timestamp.isoformat(),
                        end_reason,
                    )
                    record_cycle(timestamp, end_temp=current_indoor)
                    reset_cycle()

        _LOGGER.info("Extracted %d heating cycles", len(data_points))
        return data_points

    def _get_value_at_time(
        self,
        history: list[dict[str, Any]],
        target_time: datetime,
        attribute_name: str | None = None,
    ) -> float | None:
        """Get the sensor value at or before a specific time.

        Args:
            history: List of history records for the entity
            target_time: Time to find the value for
            attribute_name: If provided, extract value from attributes (e.g., 'current_temperature')

        Returns:
            The sensor value as float, or None if not found
        """
        closest_value: float | None = None
        closest_time: datetime | None = None

        for record in history:
            timestamp_str = record.get("last_changed") or record.get("last_updated")
            if not timestamp_str:
                continue

            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            # Extract value - either from state or attributes
            value: float | None = None
            try:
                if attribute_name:
                    # For climate entities: extract from attributes
                    attributes = record.get("attributes", {})
                    if attribute_name in attributes:
                        value = float(attributes[attribute_name])
                else:
                    # For sensor entities: extract from state
                    state = record.get("state", "")
                    if state not in ("unknown", "unavailable", ""):
                        value = float(state)
            except (ValueError, TypeError, KeyError):
                continue

            # Find the closest value at or before target_time
            if value is not None and timestamp <= target_time:
                if closest_time is None or timestamp > closest_time:
                    closest_time = timestamp
                    closest_value = value

        return closest_value

    async def fetch_rl_experiences(
        self,
        training_request: TrainingRequest,
    ) -> list[RLExperience]:
        """Fetch historical data and convert to RL experiences.

        This method constructs a sequence of RLExperience objects from
        historical Home Assistant data. Each experience represents a
        state transition (s, a, r, s', done) in the RL environment.

        Args:
            training_request: Training configuration with entity IDs and time range

        Returns:
            List of RLExperience objects for training

        Raises:
            ConnectionError: If unable to connect to Home Assistant
            ValueError: If entity IDs are invalid or no data available
        """
        _LOGGER.info(
            "Fetching RL experiences for device %s from %s to %s",
            training_request.device_id,
            training_request.start_time,
            training_request.end_time,
        )

        # Validate reward calculator is available
        if self._reward_calculator is None:
            raise ValueError(
                "Reward calculator is required for RL experience construction. "
                "Please provide a reward_calculator in the constructor."
            )

        # Build list of entity IDs to fetch
        entity_ids = [
            training_request.indoor_temp_entity_id,
            training_request.target_temp_entity_id,
            training_request.heating_state_entity_id,
        ]

        # Add optional entity IDs
        if training_request.outdoor_temp_entity_id:
            entity_ids.append(training_request.outdoor_temp_entity_id)
        if training_request.indoor_humidity_entity_id:
            entity_ids.append(training_request.indoor_humidity_entity_id)
        if training_request.window_or_door_open_entity_id:
            entity_ids.append(training_request.window_or_door_open_entity_id)
        if training_request.heating_power_entity_id:
            entity_ids.append(training_request.heating_power_entity_id)
        if training_request.heating_on_time_entity_id:
            entity_ids.append(training_request.heating_on_time_entity_id)
        if training_request.outdoor_temp_forecast_1h_entity_id:
            entity_ids.append(training_request.outdoor_temp_forecast_1h_entity_id)
        if training_request.outdoor_temp_forecast_3h_entity_id:
            entity_ids.append(training_request.outdoor_temp_forecast_3h_entity_id)

        # Fetch history for all entities
        history_data = await self._fetch_history(
            entity_ids,
            training_request.start_time or datetime.now() - timedelta(days=10),
            training_request.end_time or datetime.now(),
        )

        # Extract experiences from historical data
        experiences = self._extract_rl_experiences(
            history_data,
            training_request,
        )

        if not experiences:
            raise ValueError("No valid RL experiences found in historical data")

        _LOGGER.info(
            "Extracted %d RL experiences for device %s",
            len(experiences),
            training_request.device_id,
        )

        return experiences

    def _extract_rl_experiences(
        self,
        history_data: dict[str, list[dict[str, Any]]],
        training_request: TrainingRequest,
    ) -> list[RLExperience]:
        """Extract RL experiences from historical data.

        This method creates RLExperience objects from historical state changes.
        It identifies state transitions when:
        - Heating state changes (on/off)
        - Significant temperature changes occur (>0.1°C)
        - Target temperature changes

        For each transition, it:
        1. Constructs the current state (RLObservation)
        2. Infers the action taken (from heating state change)
        3. Constructs the next state (RLObservation)
        4. Calculates the reward using the reward calculator
        5. Determines if the episode is done (target reached or timeout)

        Args:
            history_data: Dictionary of entity_id -> history records
            training_request: Training configuration

        Returns:
            List of RLExperience objects
        """
        experiences: list[RLExperience] = []

        # Get heating state history to detect transitions
        heating_states = history_data.get(training_request.heating_state_entity_id, [])
        if not heating_states:
            _LOGGER.warning(
                "No heating state history found for %s",
                training_request.heating_state_entity_id,
            )
            return experiences

        # Sample observations at regular intervals (e.g., every 5 minutes)
        # This creates a more uniform experience dataset
        sampling_interval_minutes = 5
        observations = self._sample_observations(
            history_data,
            training_request,
            sampling_interval_minutes,
        )

        _LOGGER.debug("Sampled %d observations", len(observations))

        # Create experiences from consecutive observation pairs
        for i in range(len(observations) - 1):
            current_obs = observations[i]
            next_obs = observations[i + 1]

            # Infer action based on heating state transition
            action = self._infer_action(current_obs, next_obs)

            # Calculate reward for this transition
            reward = self._reward_calculator.calculate_reward(
                previous_state=current_obs,
                action=action,
                current_state=next_obs,
            )

            # Determine if episode is done (target reached or significant time passed)
            done = self._is_episode_done(next_obs, current_obs)

            # Create experience
            try:
                experience = RLExperience(
                    state=current_obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    done=done,
                )
                experiences.append(experience)
            except ValueError as e:
                _LOGGER.debug("Skipping invalid RL experience: %s", e)

        _LOGGER.info("Created %d RL experiences", len(experiences))
        return experiences

    def _sample_observations(
        self,
        history_data: dict[str, list[dict[str, Any]]],
        training_request: TrainingRequest,
        interval_minutes: int,
    ) -> list[RLObservation]:
        """Sample observations at regular time intervals.

        This creates a uniform dataset of observations for RL training.

        Args:
            history_data: Dictionary of entity_id -> history records
            training_request: Training configuration
            interval_minutes: Sampling interval in minutes

        Returns:
            List of sampled RLObservation objects
        """
        observations: list[RLObservation] = []

        start_time = training_request.start_time or datetime.now() - timedelta(days=10)
        end_time = training_request.end_time or datetime.now()

        # Ensure timezone-aware datetimes for comparison with HA data
        import pytz
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=pytz.UTC)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=pytz.UTC)

        current_time = start_time
        while current_time <= end_time:
            # Try to construct an observation at this timestamp
            observation = self._construct_observation_at_time(
                history_data,
                training_request,
                current_time,
            )

            if observation is not None:
                observations.append(observation)

            current_time += timedelta(minutes=interval_minutes)

        return observations

    def _construct_observation_at_time(
        self,
        history_data: dict[str, list[dict[str, Any]]],
        training_request: TrainingRequest,
        timestamp: datetime,
    ) -> RLObservation | None:
        """Construct an RLObservation at a specific timestamp.

        Args:
            history_data: Dictionary of entity_id -> history records
            training_request: Training configuration
            timestamp: Target timestamp

        Returns:
            RLObservation if all required data is available, None otherwise
        """
        # Extract indoor temperature (required)
        indoor_temp = self._get_value_at_time(
            history_data.get(training_request.indoor_temp_entity_id, []),
            timestamp,
        )
        if indoor_temp is None:
            return None

        # Extract target temperature (required)
        target_temp = self._get_value_at_time(
            history_data.get(training_request.target_temp_entity_id, []),
            timestamp,
        )
        if target_temp is None:
            return None

        # Extract heating state (required)
        heating_state_record = self._get_record_at_time(
            history_data.get(training_request.heating_state_entity_id, []),
            timestamp,
        )
        if heating_state_record is None:
            return None

        is_heating_on = self._extract_heating_state(heating_state_record)

        # Extract optional fields
        outdoor_temp = None
        if training_request.outdoor_temp_entity_id:
            outdoor_temp = self._get_value_at_time(
                history_data.get(training_request.outdoor_temp_entity_id, []),
                timestamp,
            )

        indoor_humidity = None
        if training_request.indoor_humidity_entity_id:
            indoor_humidity = self._get_value_at_time(
                history_data.get(training_request.indoor_humidity_entity_id, []),
                timestamp,
            )

        window_or_door_open = False
        if training_request.window_or_door_open_entity_id:
            window_value = self._get_value_at_time(
                history_data.get(training_request.window_or_door_open_entity_id, []),
                timestamp,
            )
            window_or_door_open = window_value is not None and window_value > 0

        heating_output_percent = None
        if training_request.heating_power_entity_id:
            heating_output_percent = self._get_value_at_time(
                history_data.get(training_request.heating_power_entity_id, []),
                timestamp,
            )

        energy_consumption_recent_kwh = None
        if training_request.heating_power_entity_id:
            energy_consumption_recent_kwh = self._get_value_at_time(
                history_data.get(training_request.heating_power_entity_id, []),
                timestamp,
            )

        time_heating_on_recent_seconds = None
        if training_request.heating_on_time_entity_id:
            time_heating_on_recent_seconds_float = self._get_value_at_time(
                history_data.get(training_request.heating_on_time_entity_id, []),
                timestamp,
            )
            if time_heating_on_recent_seconds_float is not None:
                time_heating_on_recent_seconds = int(time_heating_on_recent_seconds_float)

        outdoor_temp_forecast_1h = None
        if training_request.outdoor_temp_forecast_1h_entity_id:
            outdoor_temp_forecast_1h = self._get_value_at_time(
                history_data.get(training_request.outdoor_temp_forecast_1h_entity_id, []),
                timestamp,
            )

        outdoor_temp_forecast_3h = None
        if training_request.outdoor_temp_forecast_3h_entity_id:
            outdoor_temp_forecast_3h = self._get_value_at_time(
                history_data.get(training_request.outdoor_temp_forecast_3h_entity_id, []),
                timestamp,
            )

        # Calculate temperature trends (15-minute changes)
        indoor_temp_change_15min = self._calculate_temp_change(
            history_data.get(training_request.indoor_temp_entity_id, []),
            timestamp,
            15,
        )

        outdoor_temp_change_15min = None
        if training_request.outdoor_temp_entity_id:
            outdoor_temp_change_15min = self._calculate_temp_change(
                history_data.get(training_request.outdoor_temp_entity_id, []),
                timestamp,
                15,
            )

        # Create entity states
        indoor_temp_entity = EntityState(
            entity_id=training_request.indoor_temp_entity_id,
            last_changed_minutes=0.0,  # Simplified for now
        )

        target_temp_entity = EntityState(
            entity_id=training_request.target_temp_entity_id,
            last_changed_minutes=0.0,
        )

        outdoor_temp_entity = None
        if training_request.outdoor_temp_entity_id:
            outdoor_temp_entity = EntityState(
                entity_id=training_request.outdoor_temp_entity_id,
                last_changed_minutes=0.0,
            )

        indoor_humidity_entity = None
        if training_request.indoor_humidity_entity_id:
            indoor_humidity_entity = EntityState(
                entity_id=training_request.indoor_humidity_entity_id,
                last_changed_minutes=0.0,
            )

        window_or_door_entity = None
        if training_request.window_or_door_open_entity_id:
            window_or_door_entity = EntityState(
                entity_id=training_request.window_or_door_open_entity_id,
                last_changed_minutes=0.0,
            )

        heating_output_entity = None
        if training_request.heating_power_entity_id:
            heating_output_entity = EntityState(
                entity_id=training_request.heating_power_entity_id,
                last_changed_minutes=0.0,
            )

        energy_consumption_entity = None
        if training_request.heating_power_entity_id:
            energy_consumption_entity = EntityState(
                entity_id=training_request.heating_power_entity_id,
                last_changed_minutes=0.0,
            )

        time_heating_on_entity = None
        if training_request.heating_on_time_entity_id:
            time_heating_on_entity = EntityState(
                entity_id=training_request.heating_on_time_entity_id,
                last_changed_minutes=0.0,
            )

        # Temporal context
        day_of_week = timestamp.weekday()
        hour_of_day = timestamp.hour

        # Simplified time_until_target_minutes (in real scenario, would come from scheduler)
        # For historical data, we assume target should be reached "now"
        time_until_target_minutes = 0

        # Calculate target achievement percentage
        temp_diff = abs(indoor_temp - target_temp)
        max_expected_diff = 5.0  # Maximum expected temperature difference
        current_target_achieved_percentage = max(
            0.0, min(100.0, 100.0 * (1.0 - temp_diff / max_expected_diff))
        )

        try:
            observation = RLObservation(
                indoor_temp=indoor_temp,
                indoor_temp_entity=indoor_temp_entity,
                outdoor_temp=outdoor_temp,
                outdoor_temp_entity=outdoor_temp_entity,
                indoor_humidity=indoor_humidity,
                indoor_humidity_entity=indoor_humidity_entity,
                timestamp=timestamp,
                target_temp=target_temp,
                target_temp_entity=target_temp_entity,
                time_until_target_minutes=time_until_target_minutes,
                current_target_achieved_percentage=current_target_achieved_percentage,
                is_heating_on=is_heating_on,
                heating_output_percent=heating_output_percent,
                heating_output_entity=heating_output_entity,
                energy_consumption_recent_kwh=energy_consumption_recent_kwh,
                energy_consumption_entity=energy_consumption_entity,
                time_heating_on_recent_seconds=time_heating_on_recent_seconds,
                time_heating_on_entity=time_heating_on_entity,
                indoor_temp_change_15min=indoor_temp_change_15min,
                outdoor_temp_change_15min=outdoor_temp_change_15min,
                day_of_week=day_of_week,
                hour_of_day=hour_of_day,
                outdoor_temp_forecast_1h=outdoor_temp_forecast_1h,
                outdoor_temp_forecast_3h=outdoor_temp_forecast_3h,
                window_or_door_open=window_or_door_open,
                window_or_door_entity=window_or_door_entity,
                device_id=training_request.device_id,
            )
            return observation
        except ValueError as e:
            _LOGGER.debug("Failed to construct observation at %s: %s", timestamp, e)
            return None

    def _get_record_at_time(
        self,
        history: list[dict[str, Any]],
        target_time: datetime,
    ) -> dict[str, Any] | None:
        """Get the history record at or before a specific time.

        Args:
            history: List of history records for the entity
            target_time: Time to find the record for

        Returns:
            The history record, or None if not found
        """
        closest_record: dict[str, Any] | None = None
        closest_time: datetime | None = None

        for record in history:
            timestamp_str = record.get("last_changed") or record.get("last_updated")
            if not timestamp_str:
                continue

            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            # Find the closest record at or before target_time
            if timestamp <= target_time:
                if closest_time is None or timestamp > closest_time:
                    closest_time = timestamp
                    closest_record = record

        return closest_record

    def _extract_heating_state(self, state_record: dict[str, Any]) -> bool:
        """Extract heating on/off state from a history record.

        Args:
            state_record: History record for heating state entity

        Returns:
            True if heating is on, False otherwise
        """
        # Check if entity is a climate entity
        entity_id = state_record.get("entity_id", "")
        is_climate = entity_id.startswith("climate.")

        if is_climate:
            attributes = state_record.get("attributes", {})
            hvac_action = attributes.get("hvac_action", "")
            hvac_mode = attributes.get("hvac_mode", "")
            state = state_record.get("state", "")

            # Heating is ON if hvac_action is 'heating' OR state is 'heat'/'heating'
            return (
                (hvac_action and hvac_action.lower() in ("heating", "on"))
                or (state and state.lower() in ("heat", "heating"))
                or (hvac_mode and hvac_mode.lower() == "heat")
            )
        else:
            # For binary_sensor or switch: check state
            state = state_record.get("state", "").lower()
            return state in ("on", "heat", "heating", "true", "1")

    def _calculate_temp_change(
        self,
        history: list[dict[str, Any]],
        target_time: datetime,
        minutes_back: int,
    ) -> float | None:
        """Calculate temperature change over a time period.

        Args:
            history: List of history records for the entity
            target_time: Current time
            minutes_back: How many minutes back to compare

        Returns:
            Temperature change in °C, or None if not enough data
        """
        current_temp = self._get_value_at_time(history, target_time)
        past_time = target_time - timedelta(minutes=minutes_back)
        past_temp = self._get_value_at_time(history, past_time)

        if current_temp is not None and past_temp is not None:
            return current_temp - past_temp

        return None

    def _infer_action(
        self,
        current_obs: RLObservation,
        next_obs: RLObservation,
    ) -> RLAction:
        """Infer the action taken between two observations.

        Args:
            current_obs: Current observation
            next_obs: Next observation

        Returns:
            Inferred RLAction
        """
        # Infer action based on heating state transition
        if not current_obs.is_heating_on and next_obs.is_heating_on:
            action_type = HeatingActionType.TURN_ON
        elif current_obs.is_heating_on and not next_obs.is_heating_on:
            action_type = HeatingActionType.TURN_OFF
        elif abs(current_obs.target_temp - next_obs.target_temp) > 0.1:
            action_type = HeatingActionType.SET_TARGET_TEMPERATURE
        else:
            action_type = HeatingActionType.NO_OP

        # Use the target temperature from the next state as the action value
        return RLAction(
            action_type=action_type,
            value=next_obs.target_temp,
            decision_timestamp=next_obs.timestamp,
            confidence_score=None,
        )

    def _is_episode_done(
        self,
        current_obs: RLObservation,
        previous_obs: RLObservation,
    ) -> bool:
        """Determine if an episode should end.

        An episode ends when:
        - Target temperature is reached (within tolerance)
        - Significant time has passed since target change
        - Target temperature changes significantly

        Args:
            current_obs: Current observation
            previous_obs: Previous observation

        Returns:
            True if episode should end, False otherwise
        """
        # Episode ends if target temperature changed significantly
        if abs(current_obs.target_temp - previous_obs.target_temp) > 0.5:
            return True

        # Episode ends if target is achieved
        temp_diff = abs(current_obs.indoor_temp - current_obs.target_temp)
        if temp_diff <= 0.3:  # Within 0.3°C tolerance
            return True

        # Episode continues by default
        return False
