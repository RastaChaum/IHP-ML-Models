"""Integration tests for all API endpoints.

This module tests the Flask API endpoints by simulating Home Assistant
responses and validating the full request/response cycle.
"""

import json
from datetime import datetime
from typing import Any, Dict
from unittest.mock import patch

import pytest


class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_check_returns_healthy(self, client: Any) -> None:
        """Health endpoint should return healthy status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestStatusEndpoint:
    """Tests for the /api/v1/status endpoint."""
    
    def test_status_returns_service_info(self, client: Any) -> None:
        """Status endpoint should return ML service information."""
        response = client.get("/api/v1/status")
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "ready" in data
        assert "model_count" in data


class TestTrainEndpoint:
    """Tests for the /api/v1/train endpoint."""
    
    def test_train_with_valid_data(
        self, 
        client: Any, 
        sample_training_data: Dict[str, Any]
    ) -> None:
        """Training with valid data should succeed."""
        response = client.post(
            "/api/v1/train",
            json=sample_training_data,
            content_type="application/json"
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert "model_id" in data
        assert "device_id" in data
        assert data["device_id"] == "climate.test_thermostat"
        assert data["training_samples"] > 0
        assert "metrics" in data
    
    def test_train_without_device_id(
        self, 
        client: Any, 
        sample_training_data: Dict[str, Any]
    ) -> None:
        """Training without device_id should create a global model."""
        data = sample_training_data.copy()
        del data["device_id"]
        
        response = client.post(
            "/api/v1/train",
            json=data,
            content_type="application/json"
        )
        
        assert response.status_code == 200
        result = json.loads(response.data)
        assert result["success"] is True
        assert result["device_id"] is None
    
    def test_train_with_empty_data(self, client: Any) -> None:
        """Training with no data should return 400."""
        response = client.post(
            "/api/v1/train",
            json={},
            content_type="application/json"
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
    
    def test_train_with_missing_fields(
        self, 
        client: Any, 
        sample_training_data: Dict[str, Any]
    ) -> None:
        """Training with missing required fields should return error."""
        incomplete_data = sample_training_data.copy()
        # Remove required field from first data point
        del incomplete_data["data_points"][0]["outdoor_temp"]
        
        response = client.post(
            "/api/v1/train",
            json=incomplete_data,
            content_type="application/json"
        )
        
        # Should return 400 or 500 with error message
        assert response.status_code in [400, 500]
        data = json.loads(response.data)
        assert "error" in data
    
    def test_train_with_insufficient_data(self, client: Any) -> None:
        """Training with too few data points should fail."""
        response = client.post(
            "/api/v1/train",
            json={
                "data_points": [
                    {
                        "outdoor_temp": 5.0,
                        "indoor_temp": 18.0,
                        "target_temp": 21.0,
                        "humidity": 50.0,
                        "hour_of_day": 8,
                        "heating_duration_minutes": 30.0,
                        "timestamp": datetime.now().isoformat(),
                    }
                ]
            },
            content_type="application/json"
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data


class TestTrainFakeEndpoint:
    """Tests for the /api/v1/train/fake endpoint."""
    
    def test_train_with_fake_data_default_samples(self, client: Any) -> None:
        """Training with fake data using default sample count."""
        response = client.post(
            "/api/v1/train/fake",
            json={},
            content_type="application/json"
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert "model_id" in data
        assert data["training_samples"] == 100
    
    def test_train_with_fake_data_custom_samples(self, client: Any) -> None:
        """Training with fake data using custom sample count."""
        response = client.post(
            "/api/v1/train/fake",
            json={"num_samples": 200},
            content_type="application/json"
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["training_samples"] == 200
    
    def test_train_with_fake_data_too_few_samples(self, client: Any) -> None:
        """Training with too few fake samples should fail."""
        response = client.post(
            "/api/v1/train/fake",
            json={"num_samples": 5},
            content_type="application/json"
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
        assert "at least 10" in data["error"].lower()
    
    def test_train_with_fake_data_too_many_samples(self, client: Any) -> None:
        """Training with too many fake samples should fail."""
        response = client.post(
            "/api/v1/train/fake",
            json={"num_samples": 20000},
            content_type="application/json"
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
        assert "at most 10000" in data["error"].lower()


class TestTrainDeviceEndpoint:
    """Tests for the /api/v1/train/device endpoint."""
    
    def test_train_with_device_config(
        self, 
        client: Any, 
        sample_device_config: Dict[str, Any]
    ) -> None:
        """Training with device config should fetch HA history and train."""
        response = client.post(
            "/api/v1/train/device",
            json=sample_device_config,
            content_type="application/json"
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert "model_id" in data
        assert data["device_id"] == "climate.test_thermostat"
        assert data["training_samples"] > 0
    
    def test_train_with_device_config_custom_history_days(
        self, 
        client: Any, 
        sample_device_config: Dict[str, Any]
    ) -> None:
        """Training with custom history_days parameter."""
        config = sample_device_config.copy()
        config["history_days"] = 14
        
        response = client.post(
            "/api/v1/train/device",
            json=config,
            content_type="application/json"
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
    
    def test_train_with_device_config_missing_fields(
        self, 
        client: Any
    ) -> None:
        """Training with incomplete device config should fail."""
        response = client.post(
            "/api/v1/train/device",
            json={"device_id": "climate.test"},
            content_type="application/json"
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
    
    def test_train_with_device_config_invalid_history_days(
        self, 
        client: Any, 
        sample_device_config: Dict[str, Any]
    ) -> None:
        """Training with invalid history_days should fail."""
        config = sample_device_config.copy()
        config["history_days"] = "not_a_number"
        
        response = client.post(
            "/api/v1/train/device",
            json=config,
            content_type="application/json"
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
    
    def test_train_with_device_config_with_cycle_split(
        self, 
        client: Any, 
        sample_device_config: Dict[str, Any]
    ) -> None:
        """Training with cycle_split_duration_minutes parameter."""
        config = sample_device_config.copy()
        config["cycle_split_duration_minutes"] = 15
        
        response = client.post(
            "/api/v1/train/device",
            json=config,
            content_type="application/json"
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True


class TestPredictEndpoint:
    """Tests for the /api/v1/predict endpoint."""
    
    def test_predict_without_trained_model(
        self, 
        client: Any, 
        sample_prediction_request: Dict[str, Any]
    ) -> None:
        """Prediction without trained model should return 503."""
        response = client.post(
            "/api/v1/predict",
            json=sample_prediction_request,
            content_type="application/json"
        )
        
        assert response.status_code == 503
        data = json.loads(response.data)
        assert "error" in data
        assert "no trained model" in data["error"].lower()
    
    def test_predict_with_trained_model(
        self, 
        client: Any, 
        sample_training_data: Dict[str, Any],
        sample_prediction_request: Dict[str, Any]
    ) -> None:
        """Prediction with trained model should succeed."""
        # First train a model
        train_response = client.post(
            "/api/v1/train",
            json=sample_training_data,
            content_type="application/json"
        )
        assert train_response.status_code == 200
        
        # Then make prediction
        response = client.post(
            "/api/v1/predict",
            json=sample_prediction_request,
            content_type="application/json"
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert "predicted_duration_minutes" in data
        assert "confidence" in data
        assert "model_id" in data
        assert data["predicted_duration_minutes"] >= 0
    
    def test_predict_with_device_id(
        self, 
        client: Any, 
        sample_training_data: Dict[str, Any],
        sample_prediction_request: Dict[str, Any]
    ) -> None:
        """Prediction with device_id should use device-specific model."""
        # Train a device-specific model
        train_response = client.post(
            "/api/v1/train",
            json=sample_training_data,
            content_type="application/json"
        )
        assert train_response.status_code == 200
        
        # Predict with device_id
        request = sample_prediction_request.copy()
        request["device_id"] = "climate.test_thermostat"
        
        response = client.post(
            "/api/v1/predict",
            json=request,
            content_type="application/json"
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
    
    def test_predict_with_specific_model_id(
        self, 
        client: Any, 
        sample_training_data: Dict[str, Any],
        sample_prediction_request: Dict[str, Any]
    ) -> None:
        """Prediction with model_id should use that specific model."""
        # Train a model
        train_response = client.post(
            "/api/v1/train",
            json=sample_training_data,
            content_type="application/json"
        )
        assert train_response.status_code == 200
        train_data = json.loads(train_response.data)
        model_id = train_data["model_id"]
        
        # Predict with specific model_id
        request = sample_prediction_request.copy()
        request["model_id"] = model_id
        
        response = client.post(
            "/api/v1/predict",
            json=request,
            content_type="application/json"
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["model_id"] == model_id
    
    def test_predict_with_missing_fields(
        self, 
        client: Any
    ) -> None:
        """Prediction with missing required fields should fail."""
        response = client.post(
            "/api/v1/predict",
            json={"outdoor_temp": 5.0},
            content_type="application/json"
        )
        
        # Should return error (400, 500, or 503 if no model)
        assert response.status_code in [400, 500, 503]
        data = json.loads(response.data)
        assert "error" in data
    
    def test_predict_with_invalid_values(
        self, 
        client: Any, 
        sample_training_data: Dict[str, Any]
    ) -> None:
        """Prediction with invalid values should fail."""
        # Train first
        client.post(
            "/api/v1/train",
            json=sample_training_data,
            content_type="application/json"
        )
        
        # Try prediction with invalid data
        response = client.post(
            "/api/v1/predict",
            json={
                "outdoor_temp": "not_a_number",
                "indoor_temp": 18.0,
                "target_temp": 21.0,
                "humidity": 50.0,
                "hour_of_day": 8,
            },
            content_type="application/json"
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data


class TestModelsEndpoints:
    """Tests for model management endpoints."""
    
    def test_list_models_empty(self, client: Any) -> None:
        """Listing models when none exist should return empty list."""
        response = client.get("/api/v1/models")
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "models" in data
        assert len(data["models"]) == 0
    
    def test_list_models_after_training(
        self, 
        client: Any, 
        sample_training_data: Dict[str, Any]
    ) -> None:
        """Listing models after training should return the trained model."""
        # Train a model
        train_response = client.post(
            "/api/v1/train",
            json=sample_training_data,
            content_type="application/json"
        )
        assert train_response.status_code == 200
        
        # List models
        response = client.get("/api/v1/models")
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "models" in data
        assert len(data["models"]) == 1
        
        model = data["models"][0]
        assert "model_id" in model
        assert "device_id" in model
        assert "created_at" in model
        assert "training_samples" in model
        assert "metrics" in model
    
    def test_list_models_for_device(
        self, 
        client: Any, 
        sample_training_data: Dict[str, Any]
    ) -> None:
        """Listing models for a specific device."""
        # Train a device-specific model
        train_response = client.post(
            "/api/v1/train",
            json=sample_training_data,
            content_type="application/json"
        )
        assert train_response.status_code == 200
        
        # List models for device
        device_id = "climate.test_thermostat"
        response = client.get(f"/api/v1/models/device/{device_id}")
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["device_id"] == device_id
        assert "models" in data
        assert len(data["models"]) > 0
    
    def test_get_model_info(
        self, 
        client: Any, 
        sample_training_data: Dict[str, Any]
    ) -> None:
        """Getting info for a specific model."""
        # Train a model
        train_response = client.post(
            "/api/v1/train",
            json=sample_training_data,
            content_type="application/json"
        )
        assert train_response.status_code == 200
        train_data = json.loads(train_response.data)
        model_id = train_data["model_id"]
        
        # Get model info
        response = client.get(f"/api/v1/models/{model_id}")
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["model_id"] == model_id
        assert "device_id" in data
        assert "created_at" in data
        assert "training_samples" in data
        assert "feature_names" in data
        assert "metrics" in data
    
    def test_get_nonexistent_model(self, client: Any) -> None:
        """Getting info for nonexistent model should return 404."""
        response = client.get("/api/v1/models/nonexistent_model_id")
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert "error" in data
    
    def test_delete_model(
        self, 
        client: Any, 
        sample_training_data: Dict[str, Any]
    ) -> None:
        """Deleting a model should succeed."""
        # Train a model
        train_response = client.post(
            "/api/v1/train",
            json=sample_training_data,
            content_type="application/json"
        )
        assert train_response.status_code == 200
        train_data = json.loads(train_response.data)
        model_id = train_data["model_id"]
        
        # Delete the model
        response = client.delete(f"/api/v1/models/{model_id}")
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["deleted_model_id"] == model_id
        
        # Verify model is deleted
        get_response = client.get(f"/api/v1/models/{model_id}")
        assert get_response.status_code == 404
    
    def test_delete_nonexistent_model(self, client: Any) -> None:
        """Deleting nonexistent model should return error."""
        response = client.delete("/api/v1/models/nonexistent_model_id")
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert "error" in data


class TestMultipleModels:
    """Tests for scenarios with multiple models."""
    
    def test_multiple_device_models(
        self, 
        client: Any, 
        sample_training_data: Dict[str, Any]
    ) -> None:
        """Training multiple models for different devices."""
        # Train model for device 1
        data1 = sample_training_data.copy()
        data1["device_id"] = "climate.device1"
        response1 = client.post(
            "/api/v1/train",
            json=data1,
            content_type="application/json"
        )
        assert response1.status_code == 200
        
        # Train model for device 2
        data2 = sample_training_data.copy()
        data2["device_id"] = "climate.device2"
        response2 = client.post(
            "/api/v1/train",
            json=data2,
            content_type="application/json"
        )
        assert response2.status_code == 200
        
        # List all models
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data["models"]) == 2
        
        # Verify device-specific listings
        response1 = client.get("/api/v1/models/device/climate.device1")
        assert response1.status_code == 200
        data1 = json.loads(response1.data)
        assert len(data1["models"]) == 1
        assert data1["models"][0]["device_id"] == "climate.device1"


class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_invalid_json(self, client: Any) -> None:
        """Sending invalid JSON should return error."""
        response = client.post(
            "/api/v1/train",
            data="not valid json",
            content_type="application/json"
        )
        
        assert response.status_code in [400, 500]
    
    def test_missing_content_type(self, client: Any) -> None:
        """Request without content-type header should be handled."""
        response = client.post(
            "/api/v1/train",
            data='{"test": "data"}'
        )
        
        # Flask should handle this gracefully
        assert response.status_code in [400, 415, 500]
    
    def test_get_method_on_post_endpoint(self, client: Any) -> None:
        """Using wrong HTTP method should return 405."""
        response = client.get("/api/v1/train")
        
        assert response.status_code == 405
