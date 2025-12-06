#!/bin/bash
# Script to run e2e tests with environment variables loaded from .env

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load .env file if it exists
if [ -f "$PROJECT_DIR/.env" ]; then
    echo "Loading environment variables from .env..."
    set -a  # automatically export all variables
    source "$PROJECT_DIR/.env"
    set +a
else
    echo "Warning: .env file not found at $PROJECT_DIR/.env"
fi

# Map SUPERVISOR_* to HA_* for tests
export HA_URL="${HA_URL:-$SUPERVISOR_URL}"
export HA_TOKEN="${HA_TOKEN:-$SUPERVISOR_TOKEN}"

# Default test entity IDs (override these in .env if needed)
export HA_INDOOR_TEMP_ENTITY="${HA_INDOOR_TEMP_ENTITY:-sensor.indoor_temperature}"
export HA_OUTDOOR_TEMP_ENTITY="${HA_OUTDOOR_TEMP_ENTITY:-sensor.outdoor_temperature}"
export HA_TARGET_TEMP_ENTITY="${HA_TARGET_TEMP_ENTITY:-climate.thermostat}"
export HA_HEATING_STATE_ENTITY="${HA_HEATING_STATE_ENTITY:-climate.thermostat}"

# Display configuration (hide token for security)
echo ""
echo "=== E2E Test Configuration ==="
echo "HA_URL: $HA_URL"
echo "HA_TOKEN: ${HA_TOKEN:0:20}... (hidden)"
echo "HA_INDOOR_TEMP_ENTITY: $HA_INDOOR_TEMP_ENTITY"
echo "HA_OUTDOOR_TEMP_ENTITY: $HA_OUTDOOR_TEMP_ENTITY"
echo "HA_TARGET_TEMP_ENTITY: $HA_TARGET_TEMP_ENTITY"
echo "HA_HEATING_STATE_ENTITY: $HA_HEATING_STATE_ENTITY"
echo "=============================="
echo ""

# Run tests with poetry
cd "$PROJECT_DIR"
poetry run pytest tests/e2e/ -v "$@"
