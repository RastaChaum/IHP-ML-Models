"""Fake data generator for testing and validation.

Generates synthetic training data that mimics real heating patterns.
"""

import random
from datetime import datetime, timedelta
from typing import Sequence

from domain.value_objects import TrainingData, TrainingDataPoint


class FakeDataGenerator:
    """Generator for synthetic heating training data.

    This generator creates realistic heating duration data based on:
    - Temperature difference (indoor to target)
    - Outdoor temperature effects
    - Time of day patterns
    - Day of week patterns
    - Humidity effects
    """

    # Base heating rate: °C per minute for a typical home
    BASE_HEATING_RATE = 0.05  # 3°C per hour

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the fake data generator.

        Args:
            seed: Random seed for reproducibility
        """
        self._random = random.Random(seed)

    def generate(self, num_samples: int = 100) -> TrainingData:
        """Generate synthetic training data.

        Args:
            num_samples: Number of data points to generate

        Returns:
            TrainingData with generated samples
        """
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")

        data_points: list[TrainingDataPoint] = []
        base_timestamp = datetime.now() - timedelta(days=num_samples)

        for i in range(num_samples):
            data_point = self._generate_single_point(
                timestamp=base_timestamp + timedelta(hours=i * 6),
            )
            data_points.append(data_point)

        return TrainingData.from_sequence(data_points)

    def _generate_single_point(self, timestamp: datetime) -> TrainingDataPoint:
        """Generate a single training data point.

        Args:
            timestamp: Timestamp for the data point

        Returns:
            A synthetic training data point
        """
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()

        # Generate realistic environmental conditions
        outdoor_temp = self._generate_outdoor_temp(hour_of_day, day_of_week)
        indoor_temp = self._generate_indoor_temp(outdoor_temp)
        target_temp = self._generate_target_temp(hour_of_day)
        humidity = self._generate_humidity(outdoor_temp)

        # Calculate realistic heating duration
        heating_duration = self._calculate_heating_duration(
            outdoor_temp=outdoor_temp,
            indoor_temp=indoor_temp,
            target_temp=target_temp,
            humidity=humidity,
            hour_of_day=hour_of_day,
        )

        return TrainingDataPoint(
            outdoor_temp=round(outdoor_temp, 1),
            indoor_temp=round(indoor_temp, 1),
            target_temp=round(target_temp, 1),
            humidity=round(humidity, 1),
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            heating_duration_minutes=round(heating_duration, 1),
            timestamp=timestamp,
        )

    def _generate_outdoor_temp(self, hour: int, day: int) -> float:
        """Generate outdoor temperature with realistic patterns."""
        # Base temperature varies by "season" simulation
        base_temp = self._random.uniform(0, 15)

        # Daily variation: coldest at 5-6 AM, warmest at 2-3 PM
        hour_adjustment = -5 * abs((hour - 14) / 12)

        # Add noise
        noise = self._random.gauss(0, 2)

        return base_temp + hour_adjustment + noise

    def _generate_indoor_temp(self, outdoor_temp: float) -> float:
        """Generate indoor temperature based on outdoor."""
        # Indoor temp is influenced by outdoor but more stable
        base_indoor = 18 + (outdoor_temp - 10) * 0.1
        noise = self._random.gauss(0, 1)
        return max(10, min(25, base_indoor + noise))

    def _generate_target_temp(self, hour: int) -> float:
        """Generate target temperature based on time of day."""
        # Higher targets during waking/active hours
        if 6 <= hour <= 22:
            return self._random.uniform(20, 22)
        else:
            return self._random.uniform(17, 19)

    def _generate_humidity(self, outdoor_temp: float) -> float:
        """Generate humidity based on outdoor conditions."""
        # Colder temps often have higher relative humidity
        base_humidity = 60 - outdoor_temp * 0.5
        noise = self._random.gauss(0, 10)
        return max(30, min(90, base_humidity + noise))

    def _calculate_heating_duration(
        self,
        outdoor_temp: float,
        indoor_temp: float,
        target_temp: float,
        humidity: float,
        hour_of_day: int,
    ) -> float:
        """Calculate realistic heating duration.

        This simulates how long it takes to heat from indoor_temp to target_temp
        considering environmental factors.
        """
        # Temperature difference to overcome
        temp_delta = target_temp - indoor_temp

        if temp_delta <= 0:
            return 0.0

        # Base time from temperature difference
        base_minutes = temp_delta / self.BASE_HEATING_RATE

        # Outdoor temperature factor (harder to heat when cold outside)
        outdoor_factor = 1 + max(0, (10 - outdoor_temp) * 0.03)

        # Humidity factor (high humidity slightly increases heating time)
        humidity_factor = 1 + max(0, (humidity - 50) * 0.002)

        # Time of day factor (morning heating slightly faster due to residual heat)
        if 5 <= hour_of_day <= 9:
            time_factor = 0.95
        else:
            time_factor = 1.0

        # Calculate total with some random variation
        base_duration = base_minutes * outdoor_factor * humidity_factor * time_factor
        noise = self._random.gauss(0, base_duration * 0.1)

        return max(5, base_duration + noise)

    def generate_batch(
        self,
        batch_sizes: Sequence[int],
        seeds: Sequence[int] | None = None,
    ) -> list[TrainingData]:
        """Generate multiple batches of training data.

        Args:
            batch_sizes: Size of each batch to generate
            seeds: Optional seeds for each batch

        Returns:
            List of TrainingData batches
        """
        if seeds is None:
            seeds = [self._random.randint(0, 10000) for _ in batch_sizes]

        batches = []
        for size, seed in zip(batch_sizes, seeds):
            generator = FakeDataGenerator(seed=seed)
            batches.append(generator.generate(size))

        return batches
