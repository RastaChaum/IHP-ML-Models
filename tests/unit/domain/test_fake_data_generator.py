"""Tests for fake data generator service."""


import pytest
from domain.services import FakeDataGenerator


class TestFakeDataGenerator:
    """Tests for FakeDataGenerator service."""

    def test_generate_creates_requested_number_of_samples(self) -> None:
        """Test that generator creates the correct number of samples."""
        generator = FakeDataGenerator(seed=42)
        data = generator.generate(num_samples=50)
        assert data.size == 50

    def test_generate_with_seed_is_reproducible(self) -> None:
        """Test that generator with same seed produces same results."""
        gen1 = FakeDataGenerator(seed=42)
        gen2 = FakeDataGenerator(seed=42)
        
        data1 = gen1.generate(num_samples=10)
        data2 = gen2.generate(num_samples=10)
        
        for dp1, dp2 in zip(data1.data_points, data2.data_points):
            assert dp1.outdoor_temp == dp2.outdoor_temp
            assert dp1.heating_duration_minutes == dp2.heating_duration_minutes

    def test_generated_data_has_valid_values(self) -> None:
        """Test that generated data points have valid values."""
        generator = FakeDataGenerator(seed=42)
        data = generator.generate(num_samples=100)
        
        for dp in data.data_points:
            # Check temperature ranges
            assert -50 <= dp.outdoor_temp <= 60
            assert -20 <= dp.indoor_temp <= 50
            assert 0 <= dp.target_temp <= 50
            
            # Check humidity range
            assert 0 <= dp.humidity <= 100
            
            # Check time values
            assert 0 <= dp.hour_of_day <= 23
            
            # Check heating duration is positive
            assert dp.heating_duration_minutes >= 0

    def test_generate_with_zero_samples_raises_error(self) -> None:
        """Test that generating zero samples raises ValueError."""
        generator = FakeDataGenerator()
        with pytest.raises(ValueError, match="at least 1"):
            generator.generate(num_samples=0)

    def test_generate_creates_realistic_heating_durations(self) -> None:
        """Test that heating durations are realistic for temperature deltas."""
        generator = FakeDataGenerator(seed=42)
        data = generator.generate(num_samples=100)
        
        # Filter points where target > indoor (actual heating needed)
        heating_points = [
            dp for dp in data.data_points
            if dp.target_temp > dp.indoor_temp
        ]
        
        if heating_points:
            # Check that larger temp deltas generally have longer heating times
            avg_duration = sum(dp.heating_duration_minutes for dp in heating_points) / len(heating_points)
            # Average should be reasonable (not too long, not too short)
            assert 5 <= avg_duration <= 300

    def test_generate_batch_creates_multiple_datasets(self) -> None:
        """Test that generate_batch creates multiple independent datasets."""
        generator = FakeDataGenerator(seed=42)
        batches = generator.generate_batch([10, 20, 30])
        
        assert len(batches) == 3
        assert batches[0].size == 10
        assert batches[1].size == 20
        assert batches[2].size == 30
