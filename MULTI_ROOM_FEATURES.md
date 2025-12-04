# Multi-Room Feature Integration Guide

This document explains how to use the multi-room feature integration capability that allows models to consider environmental data from adjacent rooms when making heating predictions.

## Overview

The multi-room feature integration enhances prediction accuracy by incorporating thermal context from adjacent zones. Each model maintains a **feature contract** that defines exactly which features it expects, ensuring consistency between training and inference.

## Key Components

### 1. Adjacency Configuration

The room topology is defined in `config/adjacencies_room.json`:

```json
{
  "_description": "Room adjacency topology configuration",
  "zones": {
    "living_room": {
      "adjacent_zones": ["kitchen", "hallway"]
    },
    "bedroom": {
      "adjacent_zones": ["hallway", "bathroom"]
    }
  }
}
```

**Configuration Guidelines:**
- Each zone should list only its **physically adjacent** neighbors
- Zone IDs must match the `device_id` used during training
- Keep the topology accurate to your home layout

### 2. Adjacent Room Features

For each adjacent zone, the following **four features** are automatically added:

1. `{zone}_current_temp` - Current temperature (°C)
2. `{zone}_current_humidity` - Current humidity (%)
3. `{zone}_next_target_temp` - Next scheduled target temperature (°C)
4. `{zone}_duration_until_change` - Time until next temperature change (minutes)

## Training with Adjacent Room Data

### API Request Example

```bash
curl -X POST http://localhost:5000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "living_room",
    "data_points": [
      {
        "outdoor_temp": 5.0,
        "indoor_temp": 18.0,
        "target_temp": 21.0,
        "humidity": 65.0,
        "hour_of_day": 7,
        "heating_duration_minutes": 45.0,
        "minutes_since_last_cycle": 120.0,
        "timestamp": "2024-01-15T07:00:00",
        "adjacent_rooms": {
          "kitchen": {
            "current_temp": 19.0,
            "current_humidity": 60.0,
            "next_target_temp": 20.0,
            "duration_until_change": 30.0
          },
          "hallway": {
            "current_temp": 17.5,
            "current_humidity": 62.0,
            "next_target_temp": 18.0,
            "duration_until_change": 60.0
          }
        }
      }
    ]
  }'
```

### Python Code Example

```python
from datetime import datetime
from domain.value_objects import TrainingData, TrainingDataPoint

adjacent_rooms = {
    "kitchen": {
        "current_temp": 19.0,
        "current_humidity": 60.0,
        "next_target_temp": 20.0,
        "duration_until_change": 30.0
    }
}

data_point = TrainingDataPoint(
    outdoor_temp=5.0,
    indoor_temp=18.0,
    target_temp=21.0,
    humidity=65.0,
    hour_of_day=7,
    heating_duration_minutes=45.0,
    timestamp=datetime.now(),
    adjacent_rooms=adjacent_rooms
)

training_data = TrainingData.from_sequence([data_point])
model_info = await trainer.train(training_data, device_id="living_room")
```

## Making Predictions with Adjacent Room Data

### API Request Example

```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "outdoor_temp": 5.0,
    "indoor_temp": 18.0,
    "target_temp": 21.0,
    "humidity": 65.0,
    "hour_of_day": 7,
    "minutes_since_last_cycle": 120.0,
    "device_id": "living_room",
    "adjacent_rooms": {
      "kitchen": {
        "current_temp": 19.0,
        "current_humidity": 60.0,
        "next_target_temp": 20.0,
        "duration_until_change": 30.0
      },
      "hallway": {
        "current_temp": 17.5,
        "current_humidity": 62.0,
        "next_target_temp": 18.0,
        "duration_until_change": 60.0
      }
    }
  }'
```

### Python Code Example

```python
from domain.value_objects import PredictionRequest

adjacent_rooms = {
    "kitchen": {
        "current_temp": 19.0,
        "current_humidity": 60.0,
        "next_target_temp": 20.0,
        "duration_until_change": 30.0
    }
}

request = PredictionRequest(
    outdoor_temp=5.0,
    indoor_temp=18.0,
    target_temp=21.0,
    humidity=65.0,
    hour_of_day=7,
    minutes_since_last_cycle=120.0,
    device_id="living_room",
    adjacent_rooms=adjacent_rooms
)

result = await predictor.predict(request)
```

## Feature Contract & Model Artifacts

Each trained model generates **two artifacts**:

1. **Model File**: `{model_id}.pkl` - The trained XGBoost model
2. **Feature Contract**: `{model_id}_features.json` - Defines the exact feature order

Example feature contract:
```json
{
  "model_id": "xgb_living_room_a1b2c3d4",
  "device_id": "living_room",
  "feature_names": [
    "outdoor_temp",
    "indoor_temp",
    "target_temp",
    "temp_delta",
    "humidity",
    "hour_of_day",
    "minutes_since_last_cycle",
    "kitchen_current_temp",
    "kitchen_current_humidity",
    "kitchen_next_target_temp",
    "kitchen_duration_until_change",
    "hallway_current_temp",
    "hallway_current_humidity",
    "hallway_next_target_temp",
    "hallway_duration_until_change"
  ],
  "created_at": "2024-01-15T06:15:00"
}
```

## Missing Value Imputation

If adjacent room data is **missing or incomplete** during prediction:
- Missing features are automatically imputed with **0.0**
- The model will still make predictions (with potentially reduced accuracy)
- This ensures robustness when sensor data is temporarily unavailable

**Example**: If the API receives no `adjacent_rooms` data, all adjacent room features default to 0.0.

## Backward Compatibility

Models trained **without** adjacent room data (base features only) continue to work:
- They require only the 7 base features
- No adjacent room data is needed for predictions
- Feature contracts ensure correct handling

## Environment Variables

Configure the adjacency configuration path (optional):

```bash
export ADJACENCY_CONFIG_PATH=/path/to/adjacencies_room.json
```

If not set, defaults to `config/adjacencies_room.json` in the project root.

## Best Practices

1. **Keep Configuration Updated**: Maintain accurate room adjacency mappings
2. **Consistent Zone IDs**: Use the same zone IDs in config, training, and prediction
3. **Include All Adjacent Data**: When possible, provide complete adjacent room information
4. **Monitor Model Performance**: Track RMSE/R² metrics to evaluate multi-room impact
5. **Handle Missing Data Gracefully**: The system uses 0.0 imputation, but provide real data when available

## Troubleshooting

### Model expects adjacent room features but none provided
**Solution**: Either provide adjacent room data or train a new model without adjacency config.

### Zone not found in adjacency config
**Solution**: Add the zone to `config/adjacencies_room.json` or train without a specific `device_id`.

### Feature count mismatch
**Solution**: Ensure the adjacency configuration hasn't changed since model training. Retrain if needed.

## Performance Impact

**Expected Improvements**:
- Better prediction accuracy (5-15% RMSE reduction typical)
- More context-aware heating decisions
- Improved thermal modeling in open-plan spaces

**Trade-offs**:
- Requires more input data during prediction
- Slight increase in model complexity
- Dependency on multi-zone sensor availability

## Future Enhancements

Potential future improvements:
- Dynamic topology learning from data
- Weighted adjacency (closer rooms have more influence)
- Time-lagged adjacent room features
- Automatic missing data imputation using historical averages
