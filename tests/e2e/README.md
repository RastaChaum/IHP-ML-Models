# End-to-End Tests for Home Assistant Integration

These tests verify that the IHP ML Models addon correctly reads historical data from Home Assistant and that the response format is as expected.

## Prerequisites

To run these tests, you need:

1. A running Home Assistant instance with historical data
2. A long-lived access token with permissions to read history
3. The following environment variables set:
   - `HA_URL`: URL to your Home Assistant instance (e.g., `http://homeassistant:8123`)
   - `HA_TOKEN`: Your Home Assistant long-lived access token
   - `HA_INDOOR_TEMP_ENTITY`: Entity ID for indoor temperature sensor (e.g., `sensor.indoor_temperature`)
   - `HA_OUTDOOR_TEMP_ENTITY`: Entity ID for outdoor temperature sensor (e.g., `sensor.outdoor_temperature`)
   - `HA_TARGET_TEMP_ENTITY`: Entity ID for target temperature (e.g., `climate.thermostat`)
   - `HA_HEATING_STATE_ENTITY`: Entity ID for heating state (e.g., `climate.thermostat` or `binary_sensor.heating`)

## Running the Tests

```bash
# Set environment variables
export HA_URL="http://homeassistant:8123"
export HA_TOKEN="your_long_lived_access_token"
export HA_INDOOR_TEMP_ENTITY="sensor.indoor_temperature"
export HA_OUTDOOR_TEMP_ENTITY="sensor.outdoor_temperature"
export HA_TARGET_TEMP_ENTITY="climate.thermostat"
export HA_HEATING_STATE_ENTITY="climate.thermostat"

# Run e2e tests
pytest tests/e2e/ -v

# Run with markers to skip tests if HA is not available
pytest tests/e2e/ -v -m "not requires_ha"
```

## Test Structure

- `test_ha_history_reader_e2e.py`: Tests for reading historical data from Home Assistant
- `test_ha_entity_formats.py`: Tests to verify the format of responses from different entity types

## Skipping Tests

If Home Assistant is not available, tests will be automatically skipped with appropriate messages.
