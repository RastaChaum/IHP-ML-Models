#!/bin/bash
# Development script for local addon testing
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "IHP ML Models - Local Development Setup"
echo "=========================================="
echo ""

# Create necessary directories
echo "Creating test directories..."
mkdir -p test-config
mkdir -p test-data/models

# Initialize Home Assistant config if not exists
if [ ! -f "test-config/configuration.yaml" ]; then
    echo "Initializing Home Assistant test configuration..."
    cat > test-config/configuration.yaml <<EOF
# Home Assistant test configuration for IHP ML Models addon
default_config:

# Enable REST API
api:

# Enable history
history:

# Enable recorder with SQLite
recorder:
  db_url: sqlite:////config/home-assistant_v2.db
  purge_keep_days: 30

# HTTP configuration
http:
  server_host: 0.0.0.0
  server_port: 8123
  cors_allowed_origins:
    - http://localhost:5000
    - http://localhost:8123

# Logger configuration for debugging
logger:
  default: info
  logs:
    homeassistant.core: debug
    homeassistant.components.api: debug
    homeassistant.components.history: debug

# Demo platform for testing
sensor:
  - platform: demo
  - platform: random
    name: "Test Indoor Temperature"
    minimum: 18
    maximum: 24
    unit_of_measurement: "°C"
  - platform: random
    name: "Test Outdoor Temperature"
    minimum: -5
    maximum: 15
    unit_of_measurement: "°C"
  - platform: random
    name: "Test Humidity"
    minimum: 40
    maximum: 80
    unit_of_measurement: "%"

climate:
  - platform: demo
EOF
    echo "✓ Home Assistant configuration created"
fi

# Create .env file for development
if [ ! -f ".env" ]; then
    echo "Creating .env file for development..."
    cat > .env <<EOF
# Development environment variables
SUPERVISOR_TOKEN=test_token_for_development
LOG_LEVEL=debug
MODEL_PERSISTENCE_PATH=/data/models
EOF
    echo "✓ .env file created"
fi

echo ""
echo "Building and starting services..."
docker compose down
docker compose build --no-cache ihp-ml-addon
docker compose up -d

echo ""
echo "=========================================="
echo "Services started successfully!"
echo "=========================================="
echo ""
echo "Home Assistant:  http://localhost:8123"
echo "IHP ML Addon:    http://localhost:5000"
echo ""
echo "Useful commands:"
echo "  View logs:           docker compose logs -f"
echo "  View addon logs:     docker compose logs -f ihp-ml-addon"
echo "  View HA logs:        docker compose logs -f homeassistant"
echo "  Restart addon:       docker compose restart ihp-ml-addon"
echo "  Stop all:            docker compose down"
echo "  Rebuild addon:       docker compose build --no-cache ihp-ml-addon && docker compose up -d"
echo ""
echo "API endpoints:"
echo "  Health check:        curl http://localhost:5000/health"
echo "  Status:              curl http://localhost:5000/api/v1/status"
echo "  Train (fake data):   curl -X POST http://localhost:5000/api/v1/train/fake -H 'Content-Type: application/json' -d '{\"num_samples\": 100}'"
echo ""
echo "To test with Home Assistant long-lived token:"
echo "  1. Go to http://localhost:8123"
echo "  2. Create a long-lived access token in your profile"
echo "  3. Update .env with: SUPERVISOR_TOKEN=your_token"
echo "  4. Restart: docker compose restart ihp-ml-addon"
echo ""
