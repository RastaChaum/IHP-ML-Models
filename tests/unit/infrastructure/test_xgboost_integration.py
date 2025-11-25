"""Integration tests for XGBoost training and prediction."""

import pytest
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime

import sys

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "ihp_ml_addon" / "rootfs" / "app"))

from domain.services import FakeDataGenerator
from domain.value_objects import PredictionRequest
from infrastructure.adapters import XGBoostTrainer, XGBoostPredictor, FileModelStorage


class TestXGBoostIntegration:
    """Integration tests for XGBoost training and prediction pipeline."""

    @pytest.fixture
    def temp_storage_path(self) -> str:
        """Create a temporary directory for model storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def storage(self, temp_storage_path: str) -> FileModelStorage:
        """Create a file model storage instance."""
        return FileModelStorage(temp_storage_path)

    @pytest.fixture
    def trainer(self, storage: FileModelStorage) -> XGBoostTrainer:
        """Create an XGBoost trainer instance."""
        return XGBoostTrainer(storage)

    @pytest.fixture
    def predictor(self, storage: FileModelStorage) -> XGBoostPredictor:
        """Create an XGBoost predictor instance."""
        return XGBoostPredictor(storage)

    @pytest.mark.asyncio
    async def test_train_with_fake_data_creates_model(
        self,
        trainer: XGBoostTrainer,
        storage: FileModelStorage,
    ) -> None:
        """Test training a model with fake data."""
        # Generate fake training data
        generator = FakeDataGenerator(seed=42)
        training_data = generator.generate(num_samples=50)
        
        # Train model
        model_info = await trainer.train(training_data)
        
        # Verify model was created
        assert model_info.model_id is not None
        assert model_info.training_samples == 50
        assert "rmse" in model_info.metrics
        assert "r2" in model_info.metrics
        
        # Verify model was saved to storage
        latest_id = await storage.get_latest_model_id()
        assert latest_id == model_info.model_id

    @pytest.mark.asyncio
    async def test_predict_with_trained_model(
        self,
        trainer: XGBoostTrainer,
        predictor: XGBoostPredictor,
    ) -> None:
        """Test making predictions with a trained model."""
        # Generate fake training data and train model
        generator = FakeDataGenerator(seed=42)
        training_data = generator.generate(num_samples=100)
        model_info = await trainer.train(training_data)
        
        # Make a prediction
        request = PredictionRequest(
            outdoor_temp=5.0,
            indoor_temp=18.0,
            target_temp=21.0,
            humidity=65.0,
            hour_of_day=7,
            day_of_week=1,
        )
        result = await predictor.predict(request)
        
        # Verify prediction
        assert result.predicted_duration_minutes >= 0
        assert 0.0 <= result.confidence <= 1.0
        assert result.model_id == model_info.model_id
        assert result.reasoning is not None

    @pytest.mark.asyncio
    async def test_predict_returns_reasonable_duration(
        self,
        trainer: XGBoostTrainer,
        predictor: XGBoostPredictor,
    ) -> None:
        """Test that predictions return reasonable heating durations."""
        # Train with more data for better predictions
        generator = FakeDataGenerator(seed=42)
        training_data = generator.generate(num_samples=200)
        await trainer.train(training_data)
        
        # Small temperature delta should result in shorter duration
        small_delta_request = PredictionRequest(
            outdoor_temp=15.0,
            indoor_temp=20.0,
            target_temp=21.0,
            humidity=50.0,
            hour_of_day=12,
            day_of_week=2,
        )
        
        # Large temperature delta should result in longer duration
        large_delta_request = PredictionRequest(
            outdoor_temp=-5.0,
            indoor_temp=15.0,
            target_temp=22.0,
            humidity=70.0,
            hour_of_day=6,
            day_of_week=0,
        )
        
        small_result = await predictor.predict(small_delta_request)
        large_result = await predictor.predict(large_delta_request)
        
        # Larger temperature delta should generally take longer
        # Note: This is a soft assertion as ML predictions can vary
        assert large_result.predicted_duration_minutes > small_result.predicted_duration_minutes

    @pytest.mark.asyncio
    async def test_predictor_has_trained_model(
        self,
        trainer: XGBoostTrainer,
        predictor: XGBoostPredictor,
    ) -> None:
        """Test has_trained_model method."""
        # Initially no model
        assert not await predictor.has_trained_model()
        
        # Train a model
        generator = FakeDataGenerator(seed=42)
        training_data = generator.generate(num_samples=50)
        await trainer.train(training_data)
        
        # Now should have a model
        assert await predictor.has_trained_model()


class TestFileModelStorage:
    """Tests for file-based model storage."""

    @pytest.fixture
    def temp_storage_path(self) -> str:
        """Create a temporary directory for model storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def storage(self, temp_storage_path: str) -> FileModelStorage:
        """Create a file model storage instance."""
        return FileModelStorage(temp_storage_path)

    @pytest.mark.asyncio
    async def test_save_and_load_model(self, storage: FileModelStorage) -> None:
        """Test saving and loading a model."""
        from domain.value_objects import ModelInfo
        
        model_id = "test_model"
        mock_model = {"type": "test", "data": [1, 2, 3]}
        model_info = ModelInfo(
            model_id=model_id,
            created_at=datetime.now(),
            training_samples=100,
            feature_names=("f1", "f2"),
            metrics={"rmse": 5.0},
        )
        
        # Save
        await storage.save_model(model_id, mock_model, model_info)
        
        # Load
        loaded_model, loaded_info = await storage.load_model(model_id)
        
        assert loaded_model == mock_model
        assert loaded_info.model_id == model_id
        assert loaded_info.training_samples == 100

    @pytest.mark.asyncio
    async def test_list_models(self, storage: FileModelStorage) -> None:
        """Test listing all models."""
        from domain.value_objects import ModelInfo
        
        # Save multiple models
        for i in range(3):
            model_id = f"model_{i}"
            model_info = ModelInfo(
                model_id=model_id,
                created_at=datetime.now(),
                training_samples=100,
                feature_names=("f1",),
                metrics={"rmse": 5.0},
            )
            await storage.save_model(model_id, {"test": i}, model_info)
        
        # List models
        models = await storage.list_models()
        assert len(models) == 3

    @pytest.mark.asyncio
    async def test_get_latest_model_id(self, storage: FileModelStorage) -> None:
        """Test getting the latest model ID."""
        from domain.value_objects import ModelInfo
        import time
        
        # Initially no models
        assert await storage.get_latest_model_id() is None
        
        # Save models
        for i in range(2):
            model_id = f"model_{i}"
            model_info = ModelInfo(
                model_id=model_id,
                created_at=datetime.now(),
                training_samples=100,
                feature_names=("f1",),
                metrics={"rmse": 5.0},
            )
            await storage.save_model(model_id, {"test": i}, model_info)
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        # Latest should be the last one saved
        latest = await storage.get_latest_model_id()
        assert latest == "model_1"

    @pytest.mark.asyncio
    async def test_delete_model(self, storage: FileModelStorage) -> None:
        """Test deleting a model."""
        from domain.value_objects import ModelInfo
        from infrastructure.adapters.file_model_storage import ModelNotFoundError
        
        model_id = "to_delete"
        model_info = ModelInfo(
            model_id=model_id,
            created_at=datetime.now(),
            training_samples=100,
            feature_names=("f1",),
            metrics={"rmse": 5.0},
        )
        
        # Save and then delete
        await storage.save_model(model_id, {"test": True}, model_info)
        await storage.delete_model(model_id)
        
        # Should not be loadable
        with pytest.raises(ModelNotFoundError):
            await storage.load_model(model_id)
