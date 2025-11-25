#!/bin/bash
set -e

# Read configuration from options.json (Home Assistant add-on config)
CONFIG_PATH=/data/options.json

if [ -f "$CONFIG_PATH" ]; then
    CONFIG_LOG_LEVEL=$(jq -r '.log_level // "info"' $CONFIG_PATH)
    CONFIG_MODEL_PATH=$(jq -r '.model_persistence_path // "/data/models"' $CONFIG_PATH)
else
    CONFIG_LOG_LEVEL="info"
    CONFIG_MODEL_PATH="/data/models"
fi

# Set environment variables
export LOG_LEVEL="${CONFIG_LOG_LEVEL}"
export MODEL_PERSISTENCE_PATH="${CONFIG_MODEL_PATH}"

# Ensure model directory exists
mkdir -p "${MODEL_PERSISTENCE_PATH}"

echo "[INFO] Starting IHP ML Models Add-on..."
echo "[INFO] Log level: ${LOG_LEVEL}"
echo "[INFO] Model path: ${MODEL_PERSISTENCE_PATH}"

# Start the Flask application
cd /app
exec /opt/venv/bin/python -m infrastructure.api.server
