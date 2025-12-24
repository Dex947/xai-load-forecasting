"""Online learning for streaming load forecasting with River."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import json
from pathlib import Path

from src.logger import get_logger

logger = get_logger(__name__)

try:
    from river import linear_model, preprocessing, metrics, compose, optim
    from river.drift import ADWIN, PageHinkley

    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    # Define placeholders for type hints
    compose = None
    metrics = None
    linear_model = None
    preprocessing = None
    optim = None
    ADWIN = None
    PageHinkley = None
    logger.warning("River not installed. Install with: pip install river")


class OnlineForecaster:
    """
    Online learning forecaster using River for incremental updates.

    Supports streaming predictions without full retraining.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        l2_regularization: float = 0.0,
        drift_detector: str = "adwin",
        drift_threshold: float = 0.002,
    ):
        """
        Args:
            learning_rate: Learning rate for online updates
            l2_regularization: L2 regularization strength
            drift_detector: Drift detection method ('adwin', 'page_hinkley', 'none')
            drift_threshold: Threshold for drift detection
        """
        if not RIVER_AVAILABLE:
            raise ImportError("River required. Install with: pip install river")

        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.drift_detector_type = drift_detector
        self.drift_threshold = drift_threshold

        # Build model pipeline
        self.model = self._build_model()

        # Drift detector
        self.drift_detector = self._build_drift_detector()

        # Metrics tracking
        self.metrics = {
            "mae": metrics.MAE(),
            "rmse": metrics.RMSE(),
            "r2": metrics.R2(),
        }

        # History
        self.n_samples_seen = 0
        self.drift_events: List[Dict] = []
        self.performance_history: List[Dict] = []

        logger.info(
            f"OnlineForecaster initialized: lr={learning_rate}, "
            f"drift_detector={drift_detector}"
        )

    def _build_model(self):
        """Build River model pipeline."""
        return compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LinearRegression(
                optimizer=optim.SGD(self.learning_rate),
                l2=self.l2_regularization,
                intercept_lr=0.01,
            ),
        )

    def _build_drift_detector(self):
        """Build drift detector."""
        if self.drift_detector_type == "adwin":
            return ADWIN(delta=self.drift_threshold)
        elif self.drift_detector_type == "page_hinkley":
            return PageHinkley(threshold=self.drift_threshold)
        else:
            return None

    def predict_one(self, x: Dict[str, float]) -> float:
        """
        Predict for a single observation.

        Args:
            x: Feature dictionary

        Returns:
            Predicted value
        """
        return self.model.predict_one(x)

    def learn_one(
        self, x: Dict[str, float], y: float, sample_weight: float = 1.0
    ) -> "OnlineForecaster":
        """
        Update model with a single observation.

        Args:
            x: Feature dictionary
            y: True target value
            sample_weight: Weight for this sample

        Returns:
            Self
        """
        # Make prediction before learning
        y_pred = self.predict_one(x)

        # Update metrics
        for metric in self.metrics.values():
            metric.update(y, y_pred)

        # Check for drift
        if self.drift_detector is not None:
            error = abs(y - y_pred)
            self.drift_detector.update(error)

            if self.drift_detector.drift_detected:
                self._handle_drift(y, y_pred)

        # Learn from observation
        self.model.learn_one(x, y, w=sample_weight)
        self.n_samples_seen += 1

        return self

    def _handle_drift(self, y_true: float, y_pred: float):
        """Handle detected drift event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "sample_number": self.n_samples_seen,
            "y_true": float(y_true),
            "y_pred": float(y_pred),
            "error": float(abs(y_true - y_pred)),
            "current_mae": float(self.metrics["mae"].get()),
        }
        self.drift_events.append(event)

        logger.warning(
            f"Drift detected at sample {self.n_samples_seen}: "
            f"error={event['error']:.3f}, MAE={event['current_mae']:.3f}"
        )

        # Reset drift detector
        self.drift_detector = self._build_drift_detector()

    def fit_batch(
        self, X: pd.DataFrame, y: pd.Series, log_interval: int = 1000
    ) -> "OnlineForecaster":
        """
        Fit on batch data in streaming fashion.

        Args:
            X: Feature DataFrame
            y: Target Series
            log_interval: Log progress every N samples

        Returns:
            Self
        """
        logger.info(f"Fitting on {len(X)} samples in streaming mode...")

        for i, (idx, row) in enumerate(X.iterrows()):
            x_dict = row.to_dict()
            y_val = y.loc[idx]

            self.learn_one(x_dict, y_val)

            if (i + 1) % log_interval == 0:
                logger.info(
                    f"Processed {i + 1}/{len(X)} samples, "
                    f"MAE={self.metrics['mae'].get():.4f}"
                )

        logger.info(
            f"Fitting complete: {self.n_samples_seen} samples, "
            f"MAE={self.metrics['mae'].get():.4f}, "
            f"RMSE={self.metrics['rmse'].get():.4f}"
        )

        return self

    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict for batch of observations.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions
        """
        predictions = []
        for idx, row in X.iterrows():
            x_dict = row.to_dict()
            pred = self.predict_one(x_dict)
            predictions.append(pred)

        return np.array(predictions)

    def get_metrics(self) -> Dict[str, float]:
        """Get current metric values."""
        return {name: metric.get() for name, metric in self.metrics.items()}

    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift events."""
        return {
            "total_drift_events": len(self.drift_events),
            "samples_processed": self.n_samples_seen,
            "drift_rate": len(self.drift_events) / max(self.n_samples_seen, 1),
            "events": self.drift_events[-10:],  # Last 10 events
        }

    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = {
            "mae": metrics.MAE(),
            "rmse": metrics.RMSE(),
            "r2": metrics.R2(),
        }
        logger.info("Metrics reset")

    def save_state(self, path: str):
        """Save model state to JSON (weights only, not full model)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        state = {
            "n_samples_seen": self.n_samples_seen,
            "drift_events": self.drift_events,
            "metrics": self.get_metrics(),
            "config": {
                "learning_rate": self.learning_rate,
                "l2_regularization": self.l2_regularization,
                "drift_detector_type": self.drift_detector_type,
                "drift_threshold": self.drift_threshold,
            },
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"State saved to {path}")


class HybridForecaster:
    """
    Combines batch model (LightGBM) with online model (River).

    Uses batch model as primary, online model for real-time adjustments.
    """

    def __init__(
        self,
        batch_model: Any,
        online_weight: float = 0.2,
        adaptation_rate: float = 0.01,
    ):
        """
        Args:
            batch_model: Trained batch model (e.g., LightGBM)
            online_weight: Weight for online model predictions (0-1)
            adaptation_rate: Learning rate for online component
        """
        self.batch_model = batch_model
        self.online_weight = online_weight
        self.adaptation_rate = adaptation_rate

        # Initialize online component
        self.online_model = OnlineForecaster(
            learning_rate=adaptation_rate, drift_detector="adwin"
        )

        # Track residuals for online learning
        self.residual_history: List[float] = []

        logger.info(f"HybridForecaster initialized: online_weight={online_weight}")

    def predict_one(self, x: Dict[str, float]) -> float:
        """
        Hybrid prediction combining batch and online models.

        Args:
            x: Feature dictionary

        Returns:
            Blended prediction
        """
        # Batch model prediction
        x_df = pd.DataFrame([x])
        batch_pred = self.batch_model.predict(x_df)[0]

        # Online model predicts residual correction
        online_correction = self.online_model.predict_one(x)

        # Blend predictions
        final_pred = (1 - self.online_weight) * batch_pred + self.online_weight * (
            batch_pred + online_correction
        )

        return final_pred

    def update(self, x: Dict[str, float], y_true: float):
        """
        Update online component with new observation.

        Args:
            x: Feature dictionary
            y_true: True target value
        """
        # Get batch prediction
        x_df = pd.DataFrame([x])
        batch_pred = self.batch_model.predict(x_df)[0]

        # Compute residual
        residual = y_true - batch_pred
        self.residual_history.append(residual)

        # Train online model to predict residuals
        self.online_model.learn_one(x, residual)

    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        """Predict for batch of observations."""
        predictions = []
        for idx, row in X.iterrows():
            x_dict = row.to_dict()
            pred = self.predict_one(x_dict)
            predictions.append(pred)

        return np.array(predictions)

    def get_drift_status(self) -> Dict[str, Any]:
        """Get drift detection status from online component."""
        return self.online_model.get_drift_summary()
