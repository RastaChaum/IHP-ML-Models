"""End-to-end tests for Home Assistant History Reader.

These tests verify that the history reader correctly fetches and parses
real data from a Home Assistant instance.
"""

import os
from datetime import datetime, timedelta, timezone

import pytest

from ihp_ml_addon.rootfs.app.infrastructure.adapters.ha_history_reader import (
    HomeAssistantHistoryReader,
)


@pytest.fixture
def ha_config():
    """Get Home Assistant configuration from environment variables."""
    config = {
        "ha_url": os.getenv("HA_URL"),
        "ha_token": os.getenv("HA_TOKEN"),
        "indoor_temp_entity": os.getenv("HA_INDOOR_TEMP_ENTITY"),
        "outdoor_temp_entity": os.getenv("HA_OUTDOOR_TEMP_ENTITY"),
        "target_temp_entity": os.getenv("HA_TARGET_TEMP_ENTITY"),
        "heating_state_entity": os.getenv("HA_HEATING_STATE_ENTITY"),
    }

    # Check if all required environment variables are set
    if not all(config.values()):
        pytest.skip(
            "Home Assistant configuration not available. "
            "Set HA_URL, HA_TOKEN, and entity ID environment variables to run e2e tests."
        )

    return config


@pytest.fixture
def ha_reader(ha_config):
    """Create a Home Assistant history reader instance."""
    return HomeAssistantHistoryReader(
        ha_url=ha_config["ha_url"],
        ha_token=ha_config["ha_token"],
        timeout=30,
    )


@pytest.mark.asyncio
@pytest.mark.e2e
class TestHomeAssistantConnectionE2E:
    """E2E tests for Home Assistant connection."""

    async def test_ha_is_available(self, ha_reader):
        """Test that Home Assistant API is available."""
        is_available = await ha_reader.is_available()
        assert is_available is True, "Home Assistant API should be available"

    async def test_ha_connection_with_invalid_token(self):
        """Test that connection fails with invalid token."""
        reader = HomeAssistantHistoryReader(
            ha_url=os.getenv("HA_URL", "http://homeassistant:8123"),
            ha_token="invalid_token_12345",
            timeout=10,
        )

        if os.getenv("HA_URL"):  # Only run if HA_URL is set
            is_available = await reader.is_available()
            assert is_available is False, "Connection should fail with invalid token"


@pytest.mark.asyncio
@pytest.mark.e2e
class TestHistoryDataFetchingE2E:
    """E2E tests for fetching historical data from Home Assistant."""

    async def test_fetch_recent_history(self, ha_reader, ha_config):
        """Test fetching recent historical data (last 1 hour)."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)

        entity_ids = [
            ha_config["indoor_temp_entity"],
            ha_config["outdoor_temp_entity"],
        ]

        history_data = await ha_reader._fetch_history(
            entity_ids,
            start_time,
            end_time,
        )

        # Verify we got data back
        assert isinstance(history_data, dict), "Should return a dictionary"
        assert len(history_data) > 0, "Should have fetched data for at least one entity"

        # Verify indoor temperature data
        indoor_data = history_data.get(ha_config["indoor_temp_entity"])
        if indoor_data:
            assert len(indoor_data) > 0, "Should have at least one indoor temp reading"
            
            # Verify record structure
            first_record = indoor_data[0]
            assert "state" in first_record, "Record should have 'state' field"
            assert "last_changed" in first_record or "last_updated" in first_record, \
                "Record should have timestamp field"
            
            # Verify state is a valid temperature
            try:
                temp = float(first_record["state"])
                assert -50 <= temp <= 50, f"Temperature {temp} seems unrealistic"
            except ValueError:
                # State might be 'unknown' or 'unavailable'
                assert first_record["state"] in ("unknown", "unavailable"), \
                    f"Unexpected state value: {first_record['state']}"

    async def test_fetch_climate_entity_format(self, ha_reader, ha_config):
        """Test that climate entities have the expected format."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)

        entity_id = ha_config["heating_state_entity"]

        # Skip if not a climate entity
        if not entity_id.startswith("climate."):
            pytest.skip("Heating state entity is not a climate entity")

        history_data = await ha_reader._fetch_history(
            [entity_id],
            start_time,
            end_time,
        )

        climate_data = history_data.get(entity_id)
        assert climate_data is not None, f"Should have data for {entity_id}"
        assert len(climate_data) > 0, "Should have at least one climate state"

        # Verify climate entity structure
        first_record = climate_data[0]
        assert "state" in first_record, "Climate entity should have 'state'"
        assert "attributes" in first_record, "Climate entity should have 'attributes'"

        attributes = first_record["attributes"]
        
        # Verify expected climate attributes
        # Note: Not all attributes may be present in all climate entities
        expected_attrs = ["temperature", "current_temperature", "hvac_mode"]
        present_attrs = [attr for attr in expected_attrs if attr in attributes]
        
        assert len(present_attrs) > 0, \
            f"Climate entity should have at least one of: {expected_attrs}"

    async def test_fetch_binary_sensor_format(self, ha_reader, ha_config):
        """Test that binary sensor entities have the expected format."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)

        entity_id = ha_config["heating_state_entity"]

        # Skip if not a binary sensor
        if not entity_id.startswith("binary_sensor."):
            pytest.skip("Heating state entity is not a binary sensor")

        history_data = await ha_reader._fetch_history(
            [entity_id],
            start_time,
            end_time,
        )

        sensor_data = history_data.get(entity_id)
        assert sensor_data is not None, f"Should have data for {entity_id}"
        assert len(sensor_data) > 0, "Should have at least one sensor state"

        # Verify binary sensor structure
        first_record = sensor_data[0]
        assert "state" in first_record, "Binary sensor should have 'state'"
        
        # Verify state is a valid binary value
        state = first_record["state"].lower()
        assert state in ("on", "off", "unknown", "unavailable"), \
            f"Binary sensor state should be on/off/unknown/unavailable, got: {state}"

    async def test_fetch_long_history_chunking(self, ha_reader, ha_config):
        """Test that fetching >7 days of history is split into chunks."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=15)  # 15 days > 7 days

        entity_ids = [ha_config["indoor_temp_entity"]]

        history_data = await ha_reader._fetch_history(
            entity_ids,
            start_time,
            end_time,
        )

        # Verify we got data back (the method should handle chunking internally)
        indoor_data = history_data.get(ha_config["indoor_temp_entity"])
        
        if indoor_data:
            assert len(indoor_data) > 0, "Should have data despite long time range"
            
            # Verify data is chronologically sorted
            timestamps = []
            for record in indoor_data[:10]:  # Check first 10 records
                ts_str = record.get("last_changed") or record.get("last_updated")
                if ts_str:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    timestamps.append(ts)
            
            # Verify timestamps are in ascending order
            for i in range(len(timestamps) - 1):
                assert timestamps[i] <= timestamps[i + 1], \
                    "Timestamps should be in chronological order"


@pytest.mark.asyncio
@pytest.mark.e2e
class TestObservationConstructionE2E:
    """E2E tests for constructing observations from real Home Assistant data."""

    async def test_construct_observation_from_real_data(self, ha_reader, ha_config):
        """Test constructing an RLObservation from real Home Assistant data."""
        from ihp_ml_addon.rootfs.app.domain.value_objects import TrainingRequest

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)

        training_request = TrainingRequest(
            device_id="test_device",
            indoor_temp_entity_id=ha_config["indoor_temp_entity"],
            target_temp_entity_id=ha_config["target_temp_entity"],
            heating_state_entity_id=ha_config["heating_state_entity"],
            outdoor_temp_entity_id=ha_config["outdoor_temp_entity"],
            start_time=start_time,
            end_time=end_time,
        )

        # Fetch history
        entity_ids = [
            training_request.indoor_temp_entity_id,
            training_request.target_temp_entity_id,
            training_request.heating_state_entity_id,
        ]
        if training_request.outdoor_temp_entity_id:
            entity_ids.append(training_request.outdoor_temp_entity_id)

        history_data = await ha_reader._fetch_history(
            entity_ids,
            start_time,
            end_time,
        )

        # Try to construct an observation at a recent timestamp
        test_timestamp = start_time + timedelta(minutes=30)

        observation = ha_reader._construct_observation_at_time(
            history_data,
            training_request,
            test_timestamp,
        )

        # If we have sufficient data, observation should be created
        if observation is not None:
            assert observation.indoor_temp is not None, "Should have indoor temperature"
            assert observation.target_temp is not None, "Should have target temperature"
            assert observation.device_id == "test_device", "Should have correct device ID"
            assert observation.timestamp == test_timestamp, "Should have correct timestamp"
            
            # Verify temperature values are reasonable
            assert -50 <= observation.indoor_temp <= 50, \
                f"Indoor temp {observation.indoor_temp} seems unrealistic"
            assert 0 <= observation.target_temp <= 50, \
                f"Target temp {observation.target_temp} seems unrealistic"
