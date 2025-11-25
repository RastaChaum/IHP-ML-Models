"""Flask HTTP API Server.

HTTP API for IHP Custom Component communication.
Provides endpoints for training and prediction.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from functools import wraps
from typing import Callable, Any

from flask import Flask, request, jsonify, Response

# Add app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from domain.value_objects import PredictionRequest, TrainingData, TrainingDataPoint
from application.services import MLApplicationService
from infrastructure.adapters import XGBoostTrainer, XGBoostPredictor, FileModelStorage

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
ml_service = MLApplicationService(trainer, predictor, storage)


def async_route(f: Callable) -> Callable:
    """Decorator to run async functions in Flask routes."""
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
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
        "data_points": [
            {
                "outdoor_temp": float,
                "indoor_temp": float,
                "target_temp": float,
                "humidity": float,
                "hour_of_day": int,
                "day_of_week": int,
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

        # Parse data points
        data_points = []
        for dp in data_points_raw:
            try:
                timestamp = datetime.fromisoformat(dp["timestamp"])
            except (KeyError, ValueError):
                timestamp = datetime.now()

            data_points.append(TrainingDataPoint(
                outdoor_temp=float(dp["outdoor_temp"]),
                indoor_temp=float(dp["indoor_temp"]),
                target_temp=float(dp["target_temp"]),
                humidity=float(dp["humidity"]),
                hour_of_day=int(dp["hour_of_day"]),
                day_of_week=int(dp["day_of_week"]),
                heating_duration_minutes=float(dp["heating_duration_minutes"]),
                timestamp=timestamp,
            ))

        training_data = TrainingData.from_sequence(data_points)
        model_info = await ml_service.train_with_data(training_data)

        return jsonify({
            "success": True,
            "model_id": model_info.model_id,
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
        "day_of_week": int,
        "model_id": str (optional)
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
            day_of_week=int(data["day_of_week"]),
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

    _LOGGER.info("Starting IHP ML Models API server on %s:%d", host, port)
    _LOGGER.info("Model storage path: %s", model_path)

    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
