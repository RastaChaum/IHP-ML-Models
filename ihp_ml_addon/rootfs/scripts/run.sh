#!/usr/bin/with-contenv bashio

set -e

# Read configuration
CONFIG_LOG_LEVEL=$(bashio::config 'log_level')
CONFIG_MODEL_PATH=$(bashio::config 'model_persistence_path')

# Set environment variables
export LOG_LEVEL="${CONFIG_LOG_LEVEL:-info}"
export MODEL_PERSISTENCE_PATH="${CONFIG_MODEL_PATH:-/data/models}"

# Ensure model directory exists
mkdir -p "${MODEL_PERSISTENCE_PATH}"

bashio::log.info "Starting IHP ML Models Add-on..."
bashio::log.info "Log level: ${LOG_LEVEL}"
bashio::log.info "Model path: ${MODEL_PERSISTENCE_PATH}"

# Start the Flask application
cd /app
exec python3 -m infrastructure.api.server
