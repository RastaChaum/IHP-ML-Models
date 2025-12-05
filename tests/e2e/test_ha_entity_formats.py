"""E2E tests for validating Home Assistant entity response formats.

These tests verify that different types of Home Assistant entities
return data in the expected format.
"""

import os
from datetime import datetime, timedelta, timezone

import pytest

from ihp_ml_addon.rootfs.app.domain.entities import HeatingState
from ihp_ml_addon.rootfs.app.infrastructure.adapters.ha_history_reader import (
    HomeAssistantHistoryReader,
)


@pytest.fixture
def ha_config():
    """Get Home Assistant configuration from environment variables."""
    config = {
        "ha_url": os.getenv("HA_URL"),
        "ha_token": os.getenv("HA_TOKEN"),
    }

    if not all(config.values()):
        pytest.skip(
            "Home Assistant configuration not available. "
            "Set HA_URL and HA_TOKEN environment variables to run e2e tests."
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
class TestClimateEntityFormat:
    """E2E tests for climate entity format validation."""

    async def test_climate_entity_heating_state_extraction(self, ha_reader):
        """Test extracting heating state from a real climate entity."""
        climate_entity = os.getenv("HA_CLIMATE_ENTITY")
        if not climate_entity or not climate_entity.startswith("climate."):
            pytest.skip("No climate entity configured. Set HA_CLIMATE_ENTITY.")

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)

        history_data = await ha_reader._fetch_history(
            [climate_entity],
            start_time,
            end_time,
        )

        climate_data = history_data.get(climate_entity)
        assert climate_data, f"Should have data for {climate_entity}"

        # Test HeatingState extraction from multiple records
        successful_extractions = 0
        for record in climate_data[:5]:  # Test first 5 records
            try:
                heating_state = HeatingState.from_ha_state_record(record)
                
                # Verify extracted values are valid
                assert isinstance(heating_state.is_on, bool), "is_on should be boolean"
                assert heating_state.target_temp >= 0, "target_temp should be non-negative"
                
                if heating_state.preset_mode is not None:
                    assert isinstance(heating_state.preset_mode, str), \
                        "preset_mode should be string"
                
                successful_extractions += 1
            except (ValueError, KeyError) as e:
                # Some records might be incomplete, that's ok
                pass

        assert successful_extractions > 0, \
            "Should successfully extract HeatingState from at least one record"

    async def test_climate_entity_temperature_attributes(self, ha_reader):
        """Test that climate entities have required temperature attributes."""
        climate_entity = os.getenv("HA_CLIMATE_ENTITY")
        if not climate_entity or not climate_entity.startswith("climate."):
            pytest.skip("No climate entity configured. Set HA_CLIMATE_ENTITY.")

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=30)

        history_data = await ha_reader._fetch_history(
            [climate_entity],
            start_time,
            end_time,
        )

        climate_data = history_data.get(climate_entity)
        assert climate_data, f"Should have data for {climate_entity}"

        # Check that at least some records have temperature attributes
        records_with_temp = 0
        for record in climate_data:
            attributes = record.get("attributes", {})
            if "temperature" in attributes or "current_temperature" in attributes:
                records_with_temp += 1

        assert records_with_temp > 0, \
            "At least some records should have temperature attributes"


@pytest.mark.asyncio
@pytest.mark.e2e
class TestSensorEntityFormat:
    """E2E tests for sensor entity format validation."""

    async def test_temperature_sensor_format(self, ha_reader):
        """Test that temperature sensors return numeric values."""
        temp_sensor = os.getenv("HA_TEMP_SENSOR_ENTITY")
        if not temp_sensor or not temp_sensor.startswith("sensor."):
            pytest.skip("No temperature sensor configured. Set HA_TEMP_SENSOR_ENTITY.")

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)

        history_data = await ha_reader._fetch_history(
            [temp_sensor],
            start_time,
            end_time,
        )

        sensor_data = history_data.get(temp_sensor)
        assert sensor_data, f"Should have data for {temp_sensor}"

        # Test that sensor values can be parsed as floats
        valid_readings = 0
        for record in sensor_data[:10]:  # Test first 10 records
            state = record.get("state")
            if state not in ("unknown", "unavailable", ""):
                try:
                    temp_value = float(state)
                    assert -50 <= temp_value <= 50, \
                        f"Temperature {temp_value} seems unrealistic"
                    valid_readings += 1
                except ValueError:
                    pass  # Some states might be non-numeric

        assert valid_readings > 0, \
            "Should have at least one valid numeric temperature reading"

    async def test_binary_sensor_heating_state_extraction(self, ha_reader):
        """Test extracting heating state from a binary sensor."""
        binary_sensor = os.getenv("HA_BINARY_SENSOR_HEATING")
        if not binary_sensor or not binary_sensor.startswith("binary_sensor."):
            pytest.skip("No binary sensor configured. Set HA_BINARY_SENSOR_HEATING.")

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)

        history_data = await ha_reader._fetch_history(
            [binary_sensor],
            start_time,
            end_time,
        )

        sensor_data = history_data.get(binary_sensor)
        assert sensor_data, f"Should have data for {binary_sensor}"

        # Test HeatingState extraction from binary sensor records
        successful_extractions = 0
        for record in sensor_data[:5]:
            try:
                heating_state = HeatingState.from_ha_state_record(record)
                
                # Verify extracted values
                assert isinstance(heating_state.is_on, bool), "is_on should be boolean"
                
                # Binary sensors don't have preset_mode or meaningful target_temp
                assert heating_state.preset_mode is None, \
                    "Binary sensor should have None preset_mode"
                
                successful_extractions += 1
            except (ValueError, KeyError):
                pass

        assert successful_extractions > 0, \
            "Should successfully extract HeatingState from at least one record"


@pytest.mark.asyncio
@pytest.mark.e2e
class TestDataConsistency:
    """E2E tests for data consistency across multiple entities."""

    async def test_timestamp_alignment_across_entities(self, ha_reader):
        """Test that timestamps from different entities can be aligned."""
        indoor_temp = os.getenv("HA_INDOOR_TEMP_ENTITY")
        outdoor_temp = os.getenv("HA_OUTDOOR_TEMP_ENTITY")

        if not indoor_temp or not outdoor_temp:
            pytest.skip("Need both indoor and outdoor temp entities configured.")

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=2)

        history_data = await ha_reader._fetch_history(
            [indoor_temp, outdoor_temp],
            start_time,
            end_time,
        )

        indoor_data = history_data.get(indoor_temp)
        outdoor_data = history_data.get(outdoor_temp)

        assert indoor_data, f"Should have data for {indoor_temp}"
        assert outdoor_data, f"Should have data for {outdoor_temp}"

        # Test that we can find values at similar timestamps
        test_timestamp = start_time + timedelta(hours=1)

        indoor_value = ha_reader._get_value_at_time(indoor_data, test_timestamp)
        outdoor_value = ha_reader._get_value_at_time(outdoor_data, test_timestamp)

        # At least one should have a value (depending on sensor update frequency)
        assert (
            indoor_value is not None or outdoor_value is not None
        ), "Should be able to get at least one temperature reading at test timestamp"

    async def test_get_record_at_time(self, ha_reader):
        """Test that _get_record_at_time works correctly with real data."""
        entity_id = os.getenv("HA_INDOOR_TEMP_ENTITY")
        if not entity_id:
            pytest.skip("Need HA_INDOOR_TEMP_ENTITY configured.")

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)

        history_data = await ha_reader._fetch_history(
            [entity_id],
            start_time,
            end_time,
        )

        entity_data = history_data.get(entity_id)
        assert entity_data, f"Should have data for {entity_id}"

        # Test getting a record at a time in the middle
        test_timestamp = start_time + timedelta(minutes=30)
        record = ha_reader._get_record_at_time(entity_data, test_timestamp)

        if record is not None:
            # Verify record structure
            assert "state" in record or "attributes" in record, \
                "Record should have state or attributes"
            
            # Verify timestamp is before or at test_timestamp
            ts_str = record.get("last_changed") or record.get("last_updated")
            if ts_str:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                assert ts <= test_timestamp, \
                    "Record timestamp should be at or before requested time"
