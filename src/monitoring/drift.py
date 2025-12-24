"""Data drift detection for model monitoring."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
from pathlib import Path
import json
from datetime import datetime

from src.logger import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """Detects feature and prediction drift."""

    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        """
        Args:
            reference_data: Training data to use as reference distribution
            threshold: P-value threshold for drift detection (default 0.05)
        """
        self.reference_stats = self._compute_stats(reference_data)
        self.threshold = threshold
        self.feature_names = list(reference_data.columns)
        logger.info(
            f"DriftDetector initialized with {len(self.feature_names)} features"
        )

    def _compute_stats(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Compute reference statistics for each feature."""
        stats_dict = {}
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                stats_dict[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "q25": float(df[col].quantile(0.25)),
                    "q50": float(df[col].quantile(0.50)),
                    "q75": float(df[col].quantile(0.75)),
                }
        return stats_dict

    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Detect drift between reference and current data.

        Returns dict with drift status per feature.
        """
        results = {}

        for col in self.feature_names:
            if col not in current_data.columns:
                results[col] = {"status": "missing", "drift": True}
                continue

            ref_stats = self.reference_stats.get(col)
            if ref_stats is None:
                continue

            current = current_data[col].dropna()
            if len(current) < 10:
                results[col] = {"status": "insufficient_data", "drift": False}
                continue

            # Kolmogorov-Smirnov test for distribution shift
            # Generate reference samples from stored stats (approximation)
            ref_samples = np.random.normal(
                ref_stats["mean"], ref_stats["std"] + 1e-8, size=len(current)
            )

            ks_stat, p_value = stats.ks_2samp(ref_samples, current.values)

            drift_detected = p_value < self.threshold

            results[col] = {
                "status": "drift" if drift_detected else "stable",
                "drift": drift_detected,
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "current_mean": float(current.mean()),
                "reference_mean": ref_stats["mean"],
                "mean_shift": float(current.mean() - ref_stats["mean"]),
            }

        n_drifted = sum(1 for r in results.values() if r.get("drift", False))
        logger.info(f"Drift detection: {n_drifted}/{len(results)} features drifted")

        return results

    def check_prediction_drift(
        self, predictions: np.ndarray, reference_predictions: np.ndarray
    ) -> Dict:
        """Check if prediction distribution has shifted."""
        ks_stat, p_value = stats.ks_2samp(reference_predictions, predictions)

        return {
            "drift": p_value < self.threshold,
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "current_mean": float(predictions.mean()),
            "reference_mean": float(reference_predictions.mean()),
        }

    def save_reference(self, path: str):
        """Save reference statistics to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "reference_stats": self.reference_stats,
                    "threshold": self.threshold,
                    "feature_names": self.feature_names,
                    "created_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )
        logger.info(f"Reference stats saved to {path}")

    @classmethod
    def load_reference(cls, path: str) -> "DriftDetector":
        """Load reference statistics from file."""
        with open(path) as f:
            data = json.load(f)

        # Create empty detector and populate
        detector = cls.__new__(cls)
        detector.reference_stats = data["reference_stats"]
        detector.threshold = data["threshold"]
        detector.feature_names = data["feature_names"]

        logger.info(f"Reference stats loaded from {path}")
        return detector


class PerformanceMonitor:
    """Tracks model performance over time."""

    def __init__(self, alert_threshold: float = 0.3):
        """
        Args:
            alert_threshold: Relative degradation threshold for alerts (30% default)
        """
        self.alert_threshold = alert_threshold
        self.baseline_rmse: Optional[float] = None
        self.history: List[Dict] = []

    def set_baseline(self, rmse: float):
        """Set baseline RMSE from validation."""
        self.baseline_rmse = rmse
        logger.info(f"Baseline RMSE set to {rmse:.4f}")

    def log_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> Dict:
        """Log performance metrics and check for degradation."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        record = {
            "timestamp": (timestamp or datetime.now()).isoformat(),
            "rmse": float(rmse),
            "mae": float(mae),
            "n_samples": len(y_true),
        }

        # Check for degradation
        alert = False
        if self.baseline_rmse:
            degradation = (rmse - self.baseline_rmse) / self.baseline_rmse
            record["degradation"] = float(degradation)
            if degradation > self.alert_threshold:
                alert = True
                logger.warning(
                    f"Performance degradation: RMSE {rmse:.4f} "
                    f"({degradation:.1%} above baseline)"
                )

        record["alert"] = alert
        self.history.append(record)

        return record

    def get_summary(self, last_n: int = 7) -> Dict:
        """Get summary of recent performance."""
        recent = self.history[-last_n:] if self.history else []

        if not recent:
            return {"status": "no_data"}

        rmse_values = [r["rmse"] for r in recent]

        return {
            "n_records": len(recent),
            "mean_rmse": float(np.mean(rmse_values)),
            "std_rmse": float(np.std(rmse_values)),
            "min_rmse": float(np.min(rmse_values)),
            "max_rmse": float(np.max(rmse_values)),
            "n_alerts": sum(1 for r in recent if r.get("alert", False)),
            "baseline_rmse": self.baseline_rmse,
        }
