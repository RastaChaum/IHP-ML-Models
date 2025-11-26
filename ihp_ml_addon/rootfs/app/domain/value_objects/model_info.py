"""Model info value object.

Immutable data structure for ML model metadata.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ModelInfo:
    """Information about a trained ML model.

    Attributes:
        model_id: Unique identifier for the model
        created_at: When the model was created
        training_samples: Number of samples used for training
        feature_names: Names of features used by the model
        metrics: Training metrics (e.g., RMSE, R²)
        version: Model version string
        device_id: Device/thermostat ID this model was trained for (optional)
    """

    model_id: str
    created_at: datetime
    training_samples: int
    feature_names: tuple[str, ...]
    metrics: dict[str, float]
    version: str = "1.0.0"
    device_id: str | None = None

    def __post_init__(self) -> None:
        """Validate model info values."""
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if self.training_samples < 1:
            raise ValueError(
                f"training_samples must be at least 1, got {self.training_samples}"
            )
        if not self.feature_names:
            raise ValueError("feature_names cannot be empty")
