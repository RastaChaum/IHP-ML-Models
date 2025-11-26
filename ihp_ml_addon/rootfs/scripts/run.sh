#!/bin/bash
set -e

# Read configuration from options.json (Home Assistant add-on config) or environment
CONFIG_PATH=/data/options.json

if [ -f "$CONFIG_PATH" ]; then
    # Running as Home Assistant addon - read from options.json
    CONFIG_LOG_LEVEL=$(jq -r '.log_level // "info"' $CONFIG_PATH 2>/dev/null || echo "info")
    CONFIG_MODEL_PATH=$(jq -r '.model_persistence_path // "/data/models"' $CONFIG_PATH 2>/dev/null || echo "/data/models")
else
    # Running in development mode - use environment variables
    CONFIG_LOG_LEVEL="${LOG_LEVEL:-info}"
    CONFIG_MODEL_PATH="${MODEL_PERSISTENCE_PATH:-/data/models}"
fi

# Set environment variables (only override if not already set from environment)
export LOG_LEVEL="${LOG_LEVEL:-${CONFIG_LOG_LEVEL}}"
export MODEL_PERSISTENCE_PATH="${MODEL_PERSISTENCE_PATH:-${CONFIG_MODEL_PATH}}"

# SUPERVISOR_TOKEN and SUPERVISOR_URL should remain as-is from environment
# They are not modified here - they're either set by Docker or by Home Assistant

# Ensure model directory exists
mkdir -p "${MODEL_PERSISTENCE_PATH}"

echo "========================================"
echo "IHP ML Models Add-on Starting"
echo "========================================"
echo "Log level: ${LOG_LEVEL}"
echo "Model path: ${MODEL_PERSISTENCE_PATH}"
echo "Supervisor Token: ${SUPERVISOR_TOKEN:+***SET***}"
echo "Supervisor URL: ${SUPERVISOR_URL:-not set}"
echo "========================================"
echo ""

# Start the Flask application
cd /app
exec /opt/venv/bin/python -m infrastructure.api.server
