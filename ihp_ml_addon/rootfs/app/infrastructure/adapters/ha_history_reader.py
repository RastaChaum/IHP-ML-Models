"""Home Assistant REST API history reader adapter.

Infrastructure adapter that implements IHomeAssistantHistoryReader
using Home Assistant's REST API.

Note: This adapter uses the synchronous requests library. While the methods
are declared async to match the interface, they run synchronously within
the Flask application context (which uses asyncio.run() for async routes).
For a production system with high concurrency, consider using aiohttp.

Supports two modes of operation:
1. Standard history API (default): Uses HA's history/period API for state data.
   Limited to ~10 days of precise data by default in Home Assistant.

2. Statistics API (optional): Uses HA's statistics API for longer data retention.
   Requires on_time_entity_id to detect heating state from numeric "On Time" sensor.
   See: https://data.home-assistant.io/docs/statistics/
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urljoin

import requests
from domain.interfaces import IHomeAssistantHistoryReader
from domain.value_objects import TrainingData, TrainingDataPoint, get_week_of_month

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
    ) -> None:
        """Initialize the Home Assistant history reader.

        Args:
            ha_url: Home Assistant URL (defaults to supervisor API)
            ha_token: Long-lived access token (defaults to supervisor token)
            timeout: Request timeout in seconds
        """
        # Default to Supervisor API for addons
        self._ha_url = ha_url or os.getenv(
            "SUPERVISOR_URL", "http://supervisor/core"
        )
        self._ha_token = ha_token or os.getenv("SUPERVISOR_TOKEN", "")
        self._timeout = timeout

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
        on_time_entity_id: str | None = None,
        on_time_buffer_minutes: int = 15,
        use_statistics: bool = False,
    ) -> TrainingData:
        """Fetch historical data and convert to training data.

        This method:
        1. Fetches history for all specified entities
        2. Aligns timestamps across sensors
        3. Identifies heating cycles (when heating turned on/off)
        4. Calculates heating duration for each cycle
        5. Returns TrainingData for model training

        Args:
            indoor_temp_entity_id: Entity ID for indoor temperature sensor
            outdoor_temp_entity_id: Entity ID for outdoor temperature sensor
            target_temp_entity_id: Entity ID for target temperature
            heating_state_entity_id: Entity ID for heating state (on/off)
            humidity_entity_id: Entity ID for humidity sensor (optional)
            start_time: Start of the time range for fetching history
            end_time: End of the time range for fetching history
            on_time_entity_id: Entity ID for thermostat "On Time" sensor (optional).
                If provided, this sensor is used instead of heating_state_entity_id
                to determine when heating is active.
            on_time_buffer_minutes: Buffer time in minutes for "On Time" detection.
                If heating doesn't activate for this duration, consider heating off.
                Default is 15 minutes.
            use_statistics: If True, use Home Assistant statistics API for longer
                data retention (>10 days). Requires on_time_entity_id.

        Returns:
            TrainingData with extracted heating cycles
        """
        if use_statistics:
            # Use statistics API for longer data retention
            if not on_time_entity_id:
                raise ValueError(
                    "on_time_entity_id is required when use_statistics is True"
                )
            return await self._fetch_training_data_from_statistics(
                indoor_temp_entity_id=indoor_temp_entity_id,
                outdoor_temp_entity_id=outdoor_temp_entity_id,
                target_temp_entity_id=target_temp_entity_id,
                on_time_entity_id=on_time_entity_id,
                humidity_entity_id=humidity_entity_id,
                start_time=start_time,
                end_time=end_time,
                on_time_buffer_minutes=on_time_buffer_minutes,
            )

        # Use standard history API
        entity_ids = [
            indoor_temp_entity_id,
            outdoor_temp_entity_id,
            target_temp_entity_id,
            heating_state_entity_id,
        ]
        if humidity_entity_id:
            entity_ids.append(humidity_entity_id)
        if on_time_entity_id:
            entity_ids.append(on_time_entity_id)

        # Fetch history for all entities
        history_data = await self._fetch_history(entity_ids, start_time, end_time)

        # Extract heating cycles and create training data points
        if on_time_entity_id:
            # Use On Time sensor for heating detection with buffering
            data_points = self._extract_heating_cycles_from_on_time(
                history_data,
                indoor_temp_entity_id,
                outdoor_temp_entity_id,
                target_temp_entity_id,
                on_time_entity_id,
                humidity_entity_id,
                on_time_buffer_minutes,
            )
        else:
            # Use traditional heating state entity
            data_points = self._extract_heating_cycles(
                history_data,
                indoor_temp_entity_id,
                outdoor_temp_entity_id,
                target_temp_entity_id,
                heating_state_entity_id,
                humidity_entity_id,
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
    ) -> list[TrainingDataPoint]:
        """Extract heating cycles from historical data.

        A heating cycle is defined as:
        - START: Heating is ON AND temperature delta (target - current) > 0.2°C
        - END: Heating is OFF OR temperature delta <= 0.2°C OR current temp exceeds target

        Args:
            history_data: Dictionary of entity_id -> history records
            indoor_temp_entity_id: Entity ID for indoor temperature
            outdoor_temp_entity_id: Entity ID for outdoor temperature
            target_temp_entity_id: Entity ID for target temperature
            heating_state_entity_id: Entity ID for heating state
            humidity_entity_id: Entity ID for humidity (optional)

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

        def record_cycle(end_timestamp: datetime) -> None:
            """Record a completed heating cycle."""
            nonlocal data_points
            if heating_start is None:
                return

            duration_minutes = (end_timestamp - heating_start).total_seconds() / 60.0

            # Only use cycles with valid data
            if (
                start_indoor_temp is not None
                and start_outdoor_temp is not None
                and start_target_temp is not None
                and duration_minutes > 0
                and duration_minutes < 300  # Max 5 hours for a single cycle
            ):
                try:
                    data_point = TrainingDataPoint(
                        outdoor_temp=start_outdoor_temp,
                        indoor_temp=start_indoor_temp,
                        target_temp=start_target_temp,
                        humidity=start_humidity or 50.0,
                        hour_of_day=heating_start.hour,
                        day_of_week=heating_start.weekday(),
                        week_of_month=get_week_of_month(heating_start),
                        month=heating_start.month,
                        heating_duration_minutes=duration_minutes,
                        timestamp=heating_start,
                    )
                    data_points.append(data_point)
                except ValueError as e:
                    _LOGGER.debug("Skipping invalid data point: %s", e)

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
                    record_cycle(timestamp)
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

    def _extract_heating_cycles_from_on_time(
        self,
        history_data: dict[str, list[dict[str, Any]]],
        indoor_temp_entity_id: str,
        outdoor_temp_entity_id: str,
        target_temp_entity_id: str,
        on_time_entity_id: str,
        humidity_entity_id: str | None,
        buffer_minutes: int,
    ) -> list[TrainingDataPoint]:
        """Extract heating cycles using "On Time" sensor with buffering.

        The "On Time" sensor reports seconds of heating activity. A heating cycle
        is detected when:
        - START: On Time value increases (heating is active)
        - END: On Time value stays at 0 for buffer_minutes

        This handles the fact that radiators don't heat continuously - they cycle
        on and off. The buffer prevents considering the heating as "off" during
        short pauses in the heating cycle.

        Args:
            history_data: Dictionary of entity_id -> history records
            indoor_temp_entity_id: Entity ID for indoor temperature
            outdoor_temp_entity_id: Entity ID for outdoor temperature
            target_temp_entity_id: Entity ID for target temperature
            on_time_entity_id: Entity ID for "On Time" sensor
            humidity_entity_id: Entity ID for humidity (optional)
            buffer_minutes: Buffer time to avoid false cycle ends

        Returns:
            List of TrainingDataPoint objects
        """
        # Temperature threshold for cycle detection (in °C)
        TEMP_DELTA_THRESHOLD = 0.2

        data_points: list[TrainingDataPoint] = []

        # Get On Time history
        on_time_history = history_data.get(on_time_entity_id, [])
        if not on_time_history:
            _LOGGER.warning("No On Time history found for %s", on_time_entity_id)
            return data_points

        # Get temperature histories
        indoor_temp_history = history_data.get(indoor_temp_entity_id, [])
        target_temp_history = history_data.get(target_temp_entity_id, [])

        # Detect entity types
        def is_climate_entity(entity_id: str) -> bool:
            return entity_id.startswith("climate.")

        indoor_is_climate = is_climate_entity(indoor_temp_entity_id)
        outdoor_is_climate = is_climate_entity(outdoor_temp_entity_id)
        target_is_climate = is_climate_entity(target_temp_entity_id)
        humidity_is_climate = humidity_entity_id and is_climate_entity(humidity_entity_id)

        _LOGGER.debug("Using On Time sensor for heating detection: %s", on_time_entity_id)
        _LOGGER.debug("Buffer duration: %d minutes", buffer_minutes)

        # Track heating cycles
        heating_start: datetime | None = None
        start_indoor_temp: float | None = None
        start_outdoor_temp: float | None = None
        start_humidity: float | None = None
        start_target_temp: float | None = None
        last_heating_time: datetime | None = None  # Last time heating was active

        def reset_cycle() -> None:
            nonlocal heating_start, start_indoor_temp, start_outdoor_temp
            nonlocal start_humidity, start_target_temp, last_heating_time
            heating_start = None
            start_indoor_temp = None
            start_outdoor_temp = None
            start_humidity = None
            start_target_temp = None
            last_heating_time = None

        def record_cycle(end_timestamp: datetime) -> None:
            nonlocal data_points
            if heating_start is None:
                return

            duration_minutes = (end_timestamp - heating_start).total_seconds() / 60.0

            if (
                start_indoor_temp is not None
                and start_outdoor_temp is not None
                and start_target_temp is not None
                and duration_minutes > 0
                and duration_minutes < 300
            ):
                try:
                    data_point = TrainingDataPoint(
                        outdoor_temp=start_outdoor_temp,
                        indoor_temp=start_indoor_temp,
                        target_temp=start_target_temp,
                        humidity=start_humidity or 50.0,
                        hour_of_day=heating_start.hour,
                        day_of_week=heating_start.weekday(),
                        week_of_month=get_week_of_month(heating_start),
                        month=heating_start.month,
                        heating_duration_minutes=duration_minutes,
                        timestamp=heating_start,
                    )
                    data_points.append(data_point)
                    _LOGGER.debug(
                        "Recorded cycle: start=%s, duration=%.1f min",
                        heating_start.isoformat(),
                        duration_minutes,
                    )
                except ValueError as e:
                    _LOGGER.debug("Skipping invalid data point: %s", e)

        for record in on_time_history:
            timestamp_str = record.get("last_changed") or record.get("last_updated")
            if not timestamp_str:
                continue

            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            # Get On Time value (seconds of heating)
            on_time_value: float = 0.0
            try:
                state = record.get("state", "")
                if state not in ("unknown", "unavailable", ""):
                    on_time_value = float(state)
            except (ValueError, TypeError):
                continue

            # Check if heating is active (on_time > 0 means heating was active)
            is_heating = on_time_value > 0

            # Get current temperatures
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
                        heating_start = timestamp
                        last_heating_time = timestamp
                        start_indoor_temp = current_indoor
                        start_target_temp = current_target

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
                            start_humidity = 50.0

                        _LOGGER.debug(
                            "Cycle started at %s (indoor=%.1f, target=%.1f)",
                            timestamp.isoformat(),
                            current_indoor,
                            current_target,
                        )
            else:
                # Currently in a cycle
                if is_heating:
                    # Update last heating time
                    last_heating_time = timestamp
                else:
                    # Heating is not active - check buffer
                    if last_heating_time is not None:
                        inactive_duration = (timestamp - last_heating_time).total_seconds() / 60.0
                        
                        if inactive_duration >= buffer_minutes:
                            # Buffer exceeded - end the cycle
                            _LOGGER.debug(
                                "Cycle ended at %s (buffer exceeded: %.1f min)",
                                timestamp.isoformat(),
                                inactive_duration,
                            )
                            # Use last_heating_time as end time, not current timestamp
                            record_cycle(last_heating_time)
                            reset_cycle()
                            continue

                # Check temperature conditions
                if current_indoor is not None and current_target is not None:
                    temp_delta = current_target - current_indoor
                    if current_indoor > current_target or temp_delta <= TEMP_DELTA_THRESHOLD:
                        _LOGGER.debug(
                            "Cycle ended at %s (target reached: indoor=%.1f, target=%.1f)",
                            timestamp.isoformat(),
                            current_indoor,
                            current_target,
                        )
                        record_cycle(timestamp)
                        reset_cycle()

        _LOGGER.info(
            "Extracted %d heating cycles using On Time sensor",
            len(data_points),
        )
        return data_points

    async def _fetch_training_data_from_statistics(
        self,
        indoor_temp_entity_id: str,
        outdoor_temp_entity_id: str,
        target_temp_entity_id: str,
        on_time_entity_id: str,
        humidity_entity_id: str | None,
        start_time: datetime,
        end_time: datetime,
        on_time_buffer_minutes: int,
    ) -> TrainingData:
        """Fetch training data using Home Assistant statistics API.

        This method uses the statistics API which provides aggregated numeric data
        with longer retention than the standard history API (default 10 days).
        Statistics are stored every 5 minutes with sum/mean/min/max values.

        See: https://data.home-assistant.io/docs/statistics/

        Args:
            indoor_temp_entity_id: Entity ID for indoor temperature sensor
            outdoor_temp_entity_id: Entity ID for outdoor temperature sensor
            target_temp_entity_id: Entity ID for target temperature
            on_time_entity_id: Entity ID for thermostat "On Time" sensor
            humidity_entity_id: Entity ID for humidity sensor (optional)
            start_time: Start of time range
            end_time: End of time range
            on_time_buffer_minutes: Buffer time for heating detection

        Returns:
            TrainingData with extracted heating cycles
        """
        entity_ids = [
            indoor_temp_entity_id,
            outdoor_temp_entity_id,
            target_temp_entity_id,
            on_time_entity_id,
        ]
        if humidity_entity_id:
            entity_ids.append(humidity_entity_id)

        # Fetch statistics for all entities
        statistics_data = await self._fetch_statistics(entity_ids, start_time, end_time)

        # Extract heating cycles using On Time sensor
        data_points = self._extract_heating_cycles_from_statistics(
            statistics_data,
            indoor_temp_entity_id,
            outdoor_temp_entity_id,
            target_temp_entity_id,
            on_time_entity_id,
            humidity_entity_id,
            on_time_buffer_minutes,
        )

        if not data_points:
            raise ValueError("No valid heating cycles found in statistics data")

        _LOGGER.info(
            "Extracted %d training data points from statistics (%s to %s)",
            len(data_points),
            start_time.isoformat(),
            end_time.isoformat(),
        )

        return TrainingData.from_sequence(data_points)

    async def _fetch_statistics(
        self,
        entity_ids: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch statistics data for multiple entities.

        Home Assistant stores statistics every 5 minutes with:
        - mean: Average value during the 5-minute period
        - min: Minimum value during the 5-minute period
        - max: Maximum value during the 5-minute period
        - sum: Sum of values (for measurement sensors like On Time)
        - state: Last known state value

        Args:
            entity_ids: List of entity IDs to fetch
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Dictionary mapping entity_id to list of statistics records
        """
        # Format time for HA API
        start_str = start_time.isoformat()
        end_str = end_time.isoformat()

        # Build the statistics URL
        # HA statistics API: /api/history/statistics/{start_time}
        base_url = self._ha_url if self._ha_url.endswith('/') else f"{self._ha_url}/"
        url = urljoin(
            base_url,
            f"api/history/period/{start_str}?end_time={end_str}"
            f"&filter_entity_id={','.join(entity_ids)}"
            "&significant_changes_only=false",
        )

        _LOGGER.info("Fetching statistics from: %s", url)

        try:
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=self._timeout,
            )
            response.raise_for_status()
            stats_list = response.json()
        except requests.RequestException as e:
            _LOGGER.error("Failed to fetch statistics: %s", e)
            raise ConnectionError(f"Failed to fetch statistics from Home Assistant: {e}") from e

        # Convert list to dictionary by entity_id
        result: dict[str, list[dict[str, Any]]] = {}
        for entity_stats in stats_list:
            if entity_stats:
                entity_id = entity_stats[0].get("entity_id", "")
                if entity_id:
                    sorted_stats = sorted(
                        entity_stats,
                        key=lambda x: x.get("last_changed") or x.get("last_updated") or "",
                    )
                    result[entity_id] = sorted_stats
                    _LOGGER.info(
                        "Statistics for %s: %d records",
                        entity_id,
                        len(sorted_stats),
                    )

        return result

    def _extract_heating_cycles_from_statistics(
        self,
        statistics_data: dict[str, list[dict[str, Any]]],
        indoor_temp_entity_id: str,
        outdoor_temp_entity_id: str,
        target_temp_entity_id: str,
        on_time_entity_id: str,
        humidity_entity_id: str | None,
        buffer_minutes: int,
    ) -> list[TrainingDataPoint]:
        """Extract heating cycles from statistics data using On Time sensor.

        Statistics data contains aggregated values every 5 minutes.
        For the On Time sensor (measurement type), we look at the "sum" or "state"
        field to determine heating activity.

        Args:
            statistics_data: Dictionary of entity_id -> statistics records
            indoor_temp_entity_id: Entity ID for indoor temperature
            outdoor_temp_entity_id: Entity ID for outdoor temperature
            target_temp_entity_id: Entity ID for target temperature
            on_time_entity_id: Entity ID for "On Time" sensor
            humidity_entity_id: Entity ID for humidity (optional)
            buffer_minutes: Buffer time to avoid false cycle ends

        Returns:
            List of TrainingDataPoint objects
        """
        # Statistics data format is similar to history, so we can reuse the
        # same extraction logic with minor adjustments
        return self._extract_heating_cycles_from_on_time(
            statistics_data,
            indoor_temp_entity_id,
            outdoor_temp_entity_id,
            target_temp_entity_id,
            on_time_entity_id,
            humidity_entity_id,
            buffer_minutes,
        )
