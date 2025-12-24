"""Conformal prediction for time series with guaranteed coverage."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
import joblib
from pathlib import Path

from src.logger import get_logger

logger = get_logger(__name__)

try:
    from mapie.regression import MapieTimeSeriesRegressor
    from mapie.subsample import BlockBootstrap
    from mapie.metrics import regression_coverage_score, regression_mean_width_score
    MAPIE_AVAILABLE = True
except ImportError:
    MAPIE_AVAILABLE = False
    logger.warning("MAPIE not installed. Install with: pip install mapie")


class ConformalForecaster:
    """
    Conformal prediction wrapper using MAPIE's EnbPI method.
    
    Provides distribution-free prediction intervals with guaranteed
    coverage probability for time series forecasting.
    """
    
    def __init__(
        self,
        base_model: Any,
        confidence_level: float = 0.95,
        n_blocks: int = 10,
        n_resamplings: int = 30,
    ):
        """
        Args:
            base_model: Trained sklearn-compatible model
            confidence_level: Target coverage probability (default 95%)
            n_blocks: Number of bootstrap blocks
            n_resamplings: Number of bootstrap resamplings
        """
        if not MAPIE_AVAILABLE:
            raise ImportError("MAPIE required. Install with: pip install mapie")
        
        self.base_model = base_model
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.n_blocks = n_blocks
        self.n_resamplings = n_resamplings
        
        self.mapie_model: Optional[MapieTimeSeriesRegressor] = None
        self.is_fitted = False
        
        logger.info(
            f"ConformalForecaster initialized: "
            f"confidence={confidence_level}, blocks={n_blocks}"
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ConformalForecaster":
        """
        Fit conformal predictor on training data.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Self
        """
        # Create block bootstrap for time series
        cv = BlockBootstrap(
            n_resamplings=self.n_resamplings,
            n_blocks=self.n_blocks,
            overlapping=False,
            random_state=42
        )
        
        # Wrap model in MAPIE
        self.mapie_model = MapieTimeSeriesRegressor(
            self.base_model,
            method="enbpi",
            cv=cv,
            agg_function="mean",
            n_jobs=-1
        )
        
        logger.info(f"Fitting conformal predictor on {len(X)} samples...")
        self.mapie_model.fit(X.values, y.values)
        self.is_fitted = True
        
        logger.info("Conformal predictor fitted successfully")
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        return_intervals: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate predictions with optional prediction intervals.
        
        Args:
            X: Feature matrix
            return_intervals: Whether to return prediction intervals
            
        Returns:
            Tuple of (predictions, intervals) where intervals is (n, 2) array
            with [lower, upper] bounds
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        y_pred, y_intervals = self.mapie_model.predict(
            X.values,
            alpha=self.alpha,
            ensemble=True,
            optimize_beta=True
        )
        
        if return_intervals:
            # Reshape intervals to (n, 2) format
            intervals = np.column_stack([
                y_intervals[:, 0, 0],  # Lower bound
                y_intervals[:, 1, 0]   # Upper bound
            ])
            return y_pred, intervals
        
        return y_pred, None
    
    def partial_fit(self, X: pd.DataFrame, y: pd.Series) -> "ConformalForecaster":
        """
        Update conformal scores with new observations (online update).
        
        Args:
            X: New feature observations
            y: New target values
            
        Returns:
            Self
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.mapie_model.partial_fit(X.values, y.values)
        logger.debug(f"Updated conformal scores with {len(X)} new observations")
        return self
    
    def predict_with_update(
        self,
        X: pd.DataFrame,
        y_true: Optional[pd.Series] = None,
        step_size: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with rolling updates as true values become available.
        
        Args:
            X: Feature matrix for prediction
            y_true: True values (for updating intervals)
            step_size: Number of steps between updates
            
        Returns:
            Tuple of (predictions, intervals)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_samples = len(X)
        y_pred = np.zeros(n_samples)
        intervals = np.zeros((n_samples, 2))
        
        # Initial prediction
        y_pred[:step_size], int_batch = self.predict(X.iloc[:step_size])
        intervals[:step_size] = int_batch
        
        # Rolling predictions with updates
        for i in range(step_size, n_samples, step_size):
            end_idx = min(i + step_size, n_samples)
            
            # Update with previous observations if available
            if y_true is not None:
                self.partial_fit(
                    X.iloc[i-step_size:i],
                    y_true.iloc[i-step_size:i]
                )
            
            # Predict next batch
            y_pred[i:end_idx], int_batch = self.predict(X.iloc[i:end_idx])
            intervals[i:end_idx] = int_batch
        
        return y_pred, intervals
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        intervals: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate prediction interval quality.
        
        Args:
            y_true: True values
            y_pred: Point predictions
            intervals: Prediction intervals (n, 2)
            
        Returns:
            Dict with coverage, width, and other metrics
        """
        coverage = regression_coverage_score(
            y_true, intervals[:, 0], intervals[:, 1]
        )
        
        mean_width = regression_mean_width_score(
            intervals[:, 0], intervals[:, 1]
        )
        
        # Point prediction metrics
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Interval sharpness (narrower is better)
        sharpness = np.mean(intervals[:, 1] - intervals[:, 0])
        
        # Winkler score (combines coverage and width)
        alpha = self.alpha
        lower, upper = intervals[:, 0], intervals[:, 1]
        width = upper - lower
        
        below = y_true < lower
        above = y_true > upper
        
        winkler = width.copy()
        winkler[below] += (2 / alpha) * (lower[below] - y_true[below])
        winkler[above] += (2 / alpha) * (y_true[above] - upper[above])
        winkler_score = np.mean(winkler)
        
        metrics = {
            "coverage": float(coverage),
            "target_coverage": float(self.confidence_level),
            "coverage_gap": float(abs(coverage - self.confidence_level)),
            "mean_width": float(mean_width),
            "sharpness": float(sharpness),
            "winkler_score": float(winkler_score),
            "rmse": float(rmse),
            "mae": float(mae),
        }
        
        logger.info(
            f"Evaluation: coverage={coverage:.3f} (target={self.confidence_level}), "
            f"width={mean_width:.3f}, RMSE={rmse:.3f}"
        )
        
        return metrics
    
    def save(self, path: str):
        """Save conformal forecaster to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "mapie_model": self.mapie_model,
            "base_model": self.base_model,
            "confidence_level": self.confidence_level,
            "alpha": self.alpha,
            "n_blocks": self.n_blocks,
            "n_resamplings": self.n_resamplings,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(data, path)
        logger.info(f"Conformal forecaster saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "ConformalForecaster":
        """Load conformal forecaster from file."""
        data = joblib.load(path)
        
        forecaster = cls(
            base_model=data["base_model"],
            confidence_level=data["confidence_level"],
            n_blocks=data["n_blocks"],
            n_resamplings=data["n_resamplings"],
        )
        forecaster.mapie_model = data["mapie_model"]
        forecaster.is_fitted = data["is_fitted"]
        
        logger.info(f"Conformal forecaster loaded from {path}")
        return forecaster
