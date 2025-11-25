"""Home Assistant REST API history reader adapter.

Infrastructure adapter that implements IHomeAssistantHistoryReader
using Home Assistant's REST API.
"""

import logging
import os
from datetime import datetime
from typing import Any
from urllib.parse import urljoin

import requests

from domain.interfaces import IHomeAssistantHistoryReader
from domain.value_objects import TrainingData, TrainingDataPoint

_LOGGER = logging.getLogger(__name__)


class HomeAssistantHistoryReader(IHomeAssistantHistoryReader):
    """Home Assistant REST API implementation of history reader.

    This adapter connects to Home Assistant's REST API to fetch
    historical sensor data for training the ML model.
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
        try:
            url = urljoin(self._ha_url, "/api/")
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=self._timeout,
            )
            return response.status_code == 200
        except requests.RequestException as e:
            _LOGGER.warning("Home Assistant API not available: %s", e)
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
        url = urljoin(
            self._ha_url,
            f"/api/history/period/{start_str}?end_time={end_str}"
            f"&filter_entity_id={entity_filter}&minimal_response",
        )

        _LOGGER.debug("Fetching history from: %s", url)

        try:
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=self._timeout,
            )
            response.raise_for_status()
            history_list = response.json()
        except requests.RequestException as e:
            _LOGGER.error("Failed to fetch history: %s", e)
            raise ConnectionError(f"Failed to fetch history from Home Assistant: {e}") from e

        # Convert list of entity histories to dictionary
        result: dict[str, list[dict[str, Any]]] = {}
        for entity_history in history_list:
            if entity_history:
                entity_id = entity_history[0].get("entity_id", "")
                if entity_id:
                    result[entity_id] = entity_history

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

        This method identifies when heating started and ended,
        and calculates the duration and conditions for each cycle.

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
        data_points: list[TrainingDataPoint] = []

        # Get heating state history
        heating_states = history_data.get(heating_state_entity_id, [])
        if not heating_states:
            _LOGGER.warning("No heating state history found for %s", heating_state_entity_id)
            return data_points

        # Track heating cycles
        heating_start: datetime | None = None
        start_indoor_temp: float | None = None
        start_outdoor_temp: float | None = None
        start_humidity: float | None = None
        start_target_temp: float | None = None

        for i, state_record in enumerate(heating_states):
            state = state_record.get("state", "").lower()
            timestamp_str = state_record.get("last_changed") or state_record.get("last_updated")

            if not timestamp_str:
                continue

            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            is_heating = state in ("on", "heat", "heating", "true", "1")

            if is_heating and heating_start is None:
                # Heating cycle started
                heating_start = timestamp
                start_indoor_temp = self._get_value_at_time(
                    history_data.get(indoor_temp_entity_id, []),
                    timestamp,
                )
                start_outdoor_temp = self._get_value_at_time(
                    history_data.get(outdoor_temp_entity_id, []),
                    timestamp,
                )
                start_target_temp = self._get_value_at_time(
                    history_data.get(target_temp_entity_id, []),
                    timestamp,
                )
                if humidity_entity_id:
                    start_humidity = self._get_value_at_time(
                        history_data.get(humidity_entity_id, []),
                        timestamp,
                    )
                else:
                    start_humidity = 50.0  # Default humidity

            elif not is_heating and heating_start is not None:
                # Heating cycle ended
                duration_minutes = (timestamp - heating_start).total_seconds() / 60.0

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
                            heating_duration_minutes=duration_minutes,
                            timestamp=heating_start,
                        )
                        data_points.append(data_point)
                    except ValueError as e:
                        _LOGGER.debug("Skipping invalid data point: %s", e)

                # Reset for next cycle
                heating_start = None
                start_indoor_temp = None
                start_outdoor_temp = None
                start_humidity = None
                start_target_temp = None

        _LOGGER.info("Extracted %d heating cycles", len(data_points))
        return data_points

    def _get_value_at_time(
        self,
        history: list[dict[str, Any]],
        target_time: datetime,
    ) -> float | None:
        """Get the sensor value at or before a specific time.

        Args:
            history: List of history records for the entity
            target_time: Time to find the value for

        Returns:
            The sensor value as float, or None if not found
        """
        closest_value: float | None = None
        closest_time: datetime | None = None

        for record in history:
            state = record.get("state", "")
            timestamp_str = record.get("last_changed") or record.get("last_updated")

            if not timestamp_str or state in ("unknown", "unavailable", ""):
                continue

            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                value = float(state)
            except ValueError:
                continue

            # Find the closest value at or before target_time
            if timestamp <= target_time:
                if closest_time is None or timestamp > closest_time:
                    closest_time = timestamp
                    closest_value = value

        return closest_value
