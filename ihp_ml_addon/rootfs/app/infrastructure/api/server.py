"""Flask HTTP API Server.

HTTP API for IHP Custom Component communication.
Provides endpoints for training and prediction.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from flask import Flask, Response, jsonify, request

# Add app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from application.services import MLApplicationService
from domain.value_objects import (
    DeviceConfig,
    PredictionRequest,
    TrainingData,
    TrainingDataPoint,
    get_week_of_month,
)
from infrastructure.adapters import (
    FileModelStorage,
    HomeAssistantHistoryReader,
    XGBoostPredictor,
    XGBoostTrainer,
)

# Configure logging
log_level = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
_LOGGER = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Initialize services
model_path = Path(os.getenv("MODEL_PERSISTENCE_PATH", "/data/models"))
storage = FileModelStorage(model_path)
trainer = XGBoostTrainer(storage)
predictor = XGBoostPredictor(storage)

# Initialize Home Assistant history reader if we're running as an addon
# The SUPERVISOR_TOKEN is automatically provided by Home Assistant
ha_history_reader = None
supervisor_token = os.getenv("SUPERVISOR_TOKEN")
supervisor_url = os.getenv("SUPERVISOR_URL")

_LOGGER.info("="*60)
_LOGGER.info("Home Assistant Configuration Check")
_LOGGER.info("="*60)
_LOGGER.info("SUPERVISOR_TOKEN present: %s", "YES" if supervisor_token else "NO")
if supervisor_token:
    _LOGGER.info("SUPERVISOR_TOKEN length: %d", len(supervisor_token))
    _LOGGER.info("SUPERVISOR_TOKEN preview: %s...", supervisor_token[:20] if len(supervisor_token) > 20 else supervisor_token)
_LOGGER.info("SUPERVISOR_URL: %s", supervisor_url or "(not set)")
_LOGGER.info("="*60)

if supervisor_token:
    ha_history_reader = HomeAssistantHistoryReader(
        ha_url=supervisor_url,
        ha_token=supervisor_token
    )
    _LOGGER.info("Home Assistant integration enabled")
else:
    _LOGGER.info("Running in standalone mode (no Home Assistant integration)")

ml_service = MLApplicationService(trainer, predictor, storage, ha_history_reader)


def async_route(f: Callable) -> Callable:
    """Decorator to run async functions in Flask routes.
    
    Uses asyncio.run() for proper event loop lifecycle management.
    """
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(f(*args, **kwargs))
    return wrapper


@app.route("/health", methods=["GET"])
def health_check() -> Response:
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/v1/status", methods=["GET"])
@async_route
async def get_status() -> Response:
    """Get ML service status."""
    try:
        status = await ml_service.get_status()
        return jsonify(status)
    except Exception as e:
        _LOGGER.exception("Error getting status")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/train", methods=["POST"])
@async_route
async def train_model() -> Response:
    """Train a model with provided data.

    Request body:
    {
        "device_id": str (optional - for device-specific model),
        "data_points": [
            {
                "outdoor_temp": float,
                "indoor_temp": float,
                "target_temp": float,
                "humidity": float,
                "hour_of_day": int,
                # "day_of_week": int,
                # "week_of_month": int,
                # "month": int,
                "heating_duration_minutes": float,
                "timestamp": str (ISO format)
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        data_points_raw = data.get("data_points", [])
        if not data_points_raw:
            return jsonify({"error": "No data points provided"}), 400

        device_id = data.get("device_id")

        # Parse data points
        data_points = []
        for dp in data_points_raw:
            try:
                timestamp = datetime.fromisoformat(dp["timestamp"])
            except (KeyError, ValueError):
                timestamp = datetime.now()

            # Calculate week_of_month and month from timestamp if not provided
            week_of_month = dp.get("week_of_month")
            if week_of_month is None:
                week_of_month = get_week_of_month(timestamp)

            month = dp.get("month")
            if month is None:
                month = timestamp.month

            data_points.append(TrainingDataPoint(
                outdoor_temp=float(dp["outdoor_temp"]),
                indoor_temp=float(dp["indoor_temp"]),
                target_temp=float(dp["target_temp"]),
                humidity=float(dp["humidity"]),
                hour_of_day=int(dp["hour_of_day"]),
                # day_of_week=int(dp["day_of_week"]),
                # week_of_month=int(week_of_month),
                # month=int(month),
                heating_duration_minutes=float(dp["heating_duration_minutes"]),
                timestamp=timestamp,
            ))

        training_data = TrainingData.from_sequence(data_points)
        model_info = await ml_service.train_with_data(training_data, device_id=device_id)

        return jsonify({
            "success": True,
            "model_id": model_info.model_id,
            "device_id": model_info.device_id,
            "created_at": model_info.created_at.isoformat(),
            "training_samples": model_info.training_samples,
            "metrics": model_info.metrics,
        })

    except ValueError as e:
        _LOGGER.warning("Invalid training data: %s", e)
        return jsonify({"error": f"Invalid data: {e}"}), 400
    except Exception as e:
        _LOGGER.exception("Error training model")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/train/fake", methods=["POST"])
@async_route
async def train_with_fake_data() -> Response:
    """Train a model with generated fake data.

    Request body (optional):
    {
        "num_samples": int (default: 100)
    }
    """
    try:
        data = request.get_json() or {}
        num_samples = int(data.get("num_samples", 100))

        if num_samples < 10:
            return jsonify({"error": "num_samples must be at least 10"}), 400
        if num_samples > 10000:
            return jsonify({"error": "num_samples must be at most 10000"}), 400

        model_info = await ml_service.train_with_fake_data(num_samples)

        return jsonify({
            "success": True,
            "model_id": model_info.model_id,
            "created_at": model_info.created_at.isoformat(),
            "training_samples": model_info.training_samples,
            "metrics": model_info.metrics,
        })

    except Exception as e:
        _LOGGER.exception("Error training with fake data")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/train/device", methods=["POST"])
@async_route
async def train_with_device_config() -> Response:
    """Train a model using historical data from Home Assistant.

    This endpoint receives device configuration from the IHP component
    and fetches historical sensor data from Home Assistant to train a model.

    Request body:
    {
        "device_id": str,
        "indoor_temp_entity_id": str,
        "outdoor_temp_entity_id": str,
        "target_temp_entity_id": str,
        "heating_state_entity_id": str,
        "humidity_entity_id": str (optional),
        "history_days": int (optional, default: 30),
        "cycle_split_duration_minutes": int (optional) - if set, splits long
            heating cycles into smaller sub-cycles of this duration (in minutes)
            for more training data. Must be between 10 and 300 if set.
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Check if HA integration is available
        if not await ml_service.is_ha_available():
            return jsonify({
                "error": "Home Assistant integration not available. "
                         "Ensure the addon is running with Supervisor access.",
            }), 503

        # Parse device configuration
        try:
            # Safely parse history_days with validation
            history_days_raw = data.get("history_days", 30)
            try:
                history_days = int(history_days_raw)
            except (ValueError, TypeError):
                return jsonify({
                    "error": f"Invalid history_days value: {history_days_raw}"
                }), 400

            # Safely parse cycle_split_duration_minutes (optional)
            cycle_split_raw = data.get("cycle_split_duration_minutes")
            cycle_split_duration_minutes = None
            if cycle_split_raw is not None:
                try:
                    cycle_split_duration_minutes = int(cycle_split_raw)
                except (ValueError, TypeError):
                    return jsonify({
                        "error": f"Invalid cycle_split_duration_minutes value: {cycle_split_raw}"
                    }), 400

            device_config = DeviceConfig(
                device_id=data.get("device_id", ""),
                indoor_temp_entity_id=data.get("indoor_temp_entity_id", ""),
                outdoor_temp_entity_id=data.get("outdoor_temp_entity_id", ""),
                target_temp_entity_id=data.get("target_temp_entity_id", ""),
                heating_state_entity_id=data.get("heating_state_entity_id", ""),
                humidity_entity_id=data.get("humidity_entity_id"),
                history_days=history_days,
                cycle_split_duration_minutes=cycle_split_duration_minutes,
            )
        except ValueError as e:
            return jsonify({"error": f"Invalid device configuration: {e}"}), 400

        # Train model using HA historical data
        model_info = await ml_service.train_with_device_config(device_config)

        return jsonify({
            "success": True,
            "device_id": device_config.device_id,
            "model_id": model_info.model_id,
            "created_at": model_info.created_at.isoformat(),
            "training_samples": model_info.training_samples,
            "metrics": model_info.metrics,
        })

    except ConnectionError as e:
        _LOGGER.error("Failed to connect to Home Assistant: %s", e)
        return jsonify({"error": f"Failed to connect to Home Assistant: {e}"}), 503
    except ValueError as e:
        _LOGGER.warning("Invalid training request: %s", e)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        _LOGGER.exception("Error training with device config")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/predict", methods=["POST"])
@async_route
async def predict() -> Response:
    """Make a heating duration prediction.

    Request body:
    {
        "outdoor_temp": float,
        "indoor_temp": float,
        "target_temp": float,
        "humidity": float,
        "hour_of_day": int,
        # "day_of_week": int,
        # "week_of_month": int,
        # "month": int,
        "device_id": str (optional - for device-specific model selection),
        "model_id": str (optional - for specific model selection)
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Check if model is available
        if not await ml_service.is_ready():
            return jsonify({
                "error": "No trained model available. Train a model first.",
            }), 503

        prediction_request = PredictionRequest(
            outdoor_temp=float(data["outdoor_temp"]),
            indoor_temp=float(data["indoor_temp"]),
            target_temp=float(data["target_temp"]),
            humidity=float(data["humidity"]),
            hour_of_day=int(data["hour_of_day"]),
            # day_of_week=int(data["day_of_week"]),
            # week_of_month=int(data["week_of_month"]),
            # month=int(data["month"]),
            device_id=data.get("device_id"),
            model_id=data.get("model_id"),
        )

        result = await ml_service.predict(prediction_request)

        return jsonify({
            "success": True,
            "predicted_duration_minutes": result.predicted_duration_minutes,
            "confidence": result.confidence,
            "model_id": result.model_id,
            "timestamp": result.timestamp.isoformat(),
            "reasoning": result.reasoning,
        })

    except KeyError as e:
        return jsonify({"error": f"Missing required field: {e}"}), 400
    except ValueError as e:
        _LOGGER.warning("Invalid prediction request: %s", e)
        return jsonify({"error": f"Invalid data: {e}"}), 400
    except Exception as e:
        _LOGGER.exception("Error making prediction")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/models", methods=["GET"])
@async_route
async def list_models() -> Response:
    """List all available models."""
    try:
        models = await ml_service.list_models()
        return jsonify({
            "models": [
                {
                    "model_id": m.model_id,
                    "device_id": m.device_id,
                    "created_at": m.created_at.isoformat(),
                    "training_samples": m.training_samples,
                    "metrics": m.metrics,
                    "version": m.version,
                }
                for m in models
            ]
        })
    except Exception as e:
        _LOGGER.exception("Error listing models")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/models/device/<device_id>", methods=["GET"])
@async_route
async def list_models_for_device(device_id: str) -> Response:
    """List all available models for a specific device/thermostat."""
    try:
        models = await ml_service.list_models_for_device(device_id)
        return jsonify({
            "device_id": device_id,
            "models": [
                {
                    "model_id": m.model_id,
                    "device_id": m.device_id,
                    "created_at": m.created_at.isoformat(),
                    "training_samples": m.training_samples,
                    "metrics": m.metrics,
                    "version": m.version,
                }
                for m in models
            ]
        })
    except Exception as e:
        _LOGGER.exception("Error listing models for device")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/models/<model_id>", methods=["GET"])
@async_route
async def get_model(model_id: str) -> Response:
    """Get information about a specific model."""
    try:
        model_info = await ml_service.get_model_info(model_id)
        if model_info is None:
            return jsonify({"error": "Model not found"}), 404

        return jsonify({
            "model_id": model_info.model_id,
            "device_id": model_info.device_id,
            "created_at": model_info.created_at.isoformat(),
            "training_samples": model_info.training_samples,
            "feature_names": list(model_info.feature_names),
            "metrics": model_info.metrics,
            "version": model_info.version,
        })
    except Exception as e:
        _LOGGER.exception("Error getting model info")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/models/<model_id>", methods=["DELETE"])
@async_route
async def delete_model(model_id: str) -> Response:
    """Delete a model."""
    try:
        await ml_service.delete_model(model_id)
        return jsonify({"success": True, "deleted_model_id": model_id})
    except Exception as e:
        _LOGGER.exception("Error deleting model")
        return jsonify({"error": str(e)}), 500


def main() -> None:
    """Main entry point for the server."""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "5000"))
    
    # Enable remote debugging if DEBUG_MODE is set
    debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
    if debug_mode:
        try:
            import debugpy
            debugpy.listen(("0.0.0.0", 5678))
            _LOGGER.info("🔍 Debugpy listening on port 5678 - waiting for debugger to attach...")
            _LOGGER.info("💡 In VSCode: Run 'Python: Remote Attach (Docker)' debug configuration")
            # Wait for debugger to attach
            debugpy.wait_for_client()
            _LOGGER.info("✅ Debugger attached!")
        except Exception as e:
            _LOGGER.warning("Failed to start debugpy: %s", e)

    _LOGGER.info("Starting IHP ML Models API server on %s:%d", host, port)
    _LOGGER.info("Model storage path: %s", model_path)

    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
