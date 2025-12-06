"""Prediction result value object.

Immutable data structure for ML prediction outputs.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class PredictionResult:
    """Result of heating duration prediction.

    Attributes:
        predicted_duration_minutes: Predicted heating time in minutes
        confidence: Confidence score (0.0 to 1.0)
        model_id: Identifier of the model used for prediction
        timestamp: When the prediction was made
        reasoning: Human-readable explanation of the prediction
        feature_mismatch: True if expected adjacent room features were missing
        missing_features: List of feature names that were missing (if any)
    """

    predicted_duration_minutes: float
    confidence: float
    model_id: str
    timestamp: datetime
    reasoning: str
    feature_mismatch: bool = False
    missing_features: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate prediction result values."""
        if self.predicted_duration_minutes < 0:
            raise ValueError(
                f"predicted_duration_minutes must be non-negative, "
                f"got {self.predicted_duration_minutes}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
