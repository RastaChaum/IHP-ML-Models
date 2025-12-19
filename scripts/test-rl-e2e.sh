#!/bin/bash
# End-to-End test for RL model training and prediction

set -e

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Configuration
API_URL="${API_URL:-http://localhost:5000}"
DEVICE_ID="${DEVICE_ID:-zone.living_room}"
HA_URL="${HA_URL:-http://192.168.1.100:8123}"
HA_TOKEN="${HA_TOKEN:-}"
INDOOR_TEMP_ENTITY="${HA_INDOOR_TEMP_ENTITY:-sensor.capteur_tdeg_hdeg_salle_temperature}"
OUTDOOR_TEMP_ENTITY="${HA_OUTDOOR_TEMP_ENTITY:-sensor.openweathermap_temperature}"
TARGET_TEMP_ENTITY="${HA_TARGET_TEMP_ENTITY:-climate.thermostat_salle}"
HEATING_STATE_ENTITY="${HA_HEATING_STATE_ENTITY:-climate.thermostat_salle}"

echo "=========================================="
echo "RL Model E2E Test"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  API URL: $API_URL"
echo "  Device ID: $DEVICE_ID"
echo "  HA URL: $HA_URL"
echo "  Indoor Temp Entity: $INDOOR_TEMP_ENTITY"
echo "  Outdoor Temp Entity: $OUTDOOR_TEMP_ENTITY"
echo "  Target Temp Entity: $TARGET_TEMP_ENTITY"
echo "  Heating State Entity: $HEATING_STATE_ENTITY"
echo ""

# Step 1: Check API is available
echo "Step 1: Checking API availability..."
HEALTH_RESPONSE=$(curl -s -X GET "$API_URL/health")
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo "✓ API is healthy"
else
    echo "✗ API is not responding. Make sure the server is running on $API_URL"
    exit 1
fi
echo ""

# Step 2: Train RL model
echo "Step 2: Training RL model from Home Assistant history..."
TRAIN_REQUEST=$(cat <<EOF
{
    "device_id": "$DEVICE_ID",
    "indoor_temp_entity_id": "$INDOOR_TEMP_ENTITY",
    "outdoor_temp_entity_id": "$OUTDOOR_TEMP_ENTITY",
    "target_temp_entity_id": "$TARGET_TEMP_ENTITY",
    "heating_state_entity_id": "$HEATING_STATE_ENTITY",
    "start_time": "$(date -u -d '60 days ago' +%Y-%m-%dT%H:%M:%S%z)",
    "end_time": "$(date -u +%Y-%m-%dT%H:%M:%S%z)"
}
EOF
)

echo "Sending training request..."
TRAIN_RESPONSE=$(curl -s -X POST "$API_URL/api/v1/rl/train" \
    -H "Content-Type: application/json" \
    -d "$TRAIN_REQUEST")

if echo "$TRAIN_RESPONSE" | grep -q "success"; then
    MODEL_ID=$(echo "$TRAIN_RESPONSE" | grep -o '"model_id":"[^"]*' | cut -d'"' -f4)
    echo "✓ Model trained successfully: $MODEL_ID"
    echo "Response: $TRAIN_RESPONSE"
else
    echo "✗ Training failed"
    echo "Response: $TRAIN_RESPONSE"
    exit 1
fi
echo ""

# Step 3: Get model info
echo "Step 3: Getting trained model info..."
MODEL_INFO_RESPONSE=$(curl -s -X GET "$API_URL/api/v1/rl/models/$MODEL_ID")

if echo "$MODEL_INFO_RESPONSE" | grep -q "success"; then
    echo "✓ Model info retrieved:"
    echo "$MODEL_INFO_RESPONSE" | jq '.'
else
    echo "✗ Failed to get model info"
    echo "Response: $MODEL_INFO_RESPONSE"
fi
echo ""

# Step 4: List RL models
echo "Step 4: Listing all RL models..."
LIST_RESPONSE=$(curl -s -X GET "$API_URL/api/v1/rl/models?device_id=$DEVICE_ID")

if echo "$LIST_RESPONSE" | grep -q "success"; then
    COUNT=$(echo "$LIST_RESPONSE" | grep -o '"count":[0-9]*' | cut -d':' -f2)
    echo "✓ Found $COUNT RL model(s) for device $DEVICE_ID"
else
    echo "✗ Failed to list models"
    echo "Response: $LIST_RESPONSE"
fi
echo ""

# Step 5: Make a prediction
echo "Step 5: Making a prediction with trained model..."
NOW_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
PREDICT_REQUEST=$(cat <<EOF
{
    "device_id": "$DEVICE_ID",
    "model_id": "$MODEL_ID",
    "indoor_temp": 20.0,
    "indoor_temp_timestamp": "$NOW_TS",
    "target_temp": 21.0,
    "target_temp_timestamp": "$NOW_TS",
    "outdoor_temp": 5.0,
    "outdoor_temp_timestamp": "$NOW_TS",
    "is_heating_on": true,
    "hour_of_day": 22,
    "day_of_week": 2,
    "time_until_target_minutes": 45.0,
    "current_target_achieved_percentage": 60.0
}
EOF
)

echo "Sending prediction request..."
PREDICT_RESPONSE=$(curl -s -X POST "$API_URL/api/v1/rl/predict" \
    -H "Content-Type: application/json" \
    -d "$PREDICT_REQUEST")

if echo "$PREDICT_RESPONSE" | grep -q "success"; then
    ACTION_TYPE=$(echo "$PREDICT_RESPONSE" | grep -o '"action_type":"[^"]*' | cut -d'"' -f4)
    VALUE=$(echo "$PREDICT_RESPONSE" | grep -o '"value":[0-9.]*' | cut -d':' -f2)
    echo "✓ Prediction received: $ACTION_TYPE (value=$VALUE)"
    echo "Full response: $PREDICT_RESPONSE"
else
    echo "✗ Prediction failed"
    echo "Response: $PREDICT_RESPONSE"
fi
echo ""

echo "=========================================="
echo "E2E Test Complete!"
echo "=========================================="
