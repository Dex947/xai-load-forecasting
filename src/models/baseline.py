"""Baseline models: persistence, seasonal naive, moving average."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.logger import get_logger

logger = get_logger(__name__)


class BaselineModel:
    """
    Baseline forecasting models.
    """

    def __init__(
        self,
        method: str = "persistence",
        season_length: int = 24,
        window_size: int = 168,
    ):
        """
        Initialize baseline model.

        Args:
            method: Baseline method ('persistence', 'seasonal_naive', 'moving_average')
            season_length: Season length for seasonal naive (default: 24 hours)
            window_size: Window size for moving average (default: 168 hours = 1 week)
        """
        self.method = method
        self.season_length = season_length
        self.window_size = window_size
        self.train_data = None

        logger.info(f"Baseline model initialized: {method}")

    def fit(self, train_data: pd.Series) -> "BaselineModel":
        """
        Fit baseline model (stores training data).

        Args:
            train_data: Training time series

        Returns:
            Self
        """
        self.train_data = train_data.copy()
        logger.info(f"Baseline model fitted with {len(train_data)} training samples")
        return self

    def predict(self, horizon: int) -> pd.Series:
        """
        Generate predictions.

        Args:
            horizon: Forecast horizon (number of steps)

        Returns:
            Series with predictions
        """
        if self.train_data is None:
            raise ValueError("Model must be fitted before prediction")

        logger.info(f"Generating {horizon} step forecast using {self.method}")

        if self.method == "persistence":
            predictions = self._persistence_forecast(horizon)
        elif self.method == "seasonal_naive":
            predictions = self._seasonal_naive_forecast(horizon)
        elif self.method == "moving_average":
            predictions = self._moving_average_forecast(horizon)
        else:
            raise ValueError(f"Unknown baseline method: {self.method}")

        return predictions

    def _persistence_forecast(self, horizon: int) -> pd.Series:
        """
        Persistence forecast (last observed value).
        """
        last_value = self.train_data.iloc[-1]

        # Create index for predictions
        last_timestamp = self.train_data.index[-1]
        freq = pd.infer_freq(self.train_data.index)
        if freq is None:
            freq = "H"  # Default to hourly

        pred_index = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1), periods=horizon, freq=freq
        )

        predictions = pd.Series(last_value, index=pred_index)

        return predictions

    def _seasonal_naive_forecast(self, horizon: int) -> pd.Series:
        """
        Seasonal naive forecast (same value as last season).
        """
        last_timestamp = self.train_data.index[-1]
        freq = pd.infer_freq(self.train_data.index)
        if freq is None:
            freq = "H"

        pred_index = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1), periods=horizon, freq=freq
        )

        predictions = pd.Series(index=pred_index, dtype=float)

        for i in range(horizon):
            # Look back one season
            lookback_idx = (
                len(self.train_data) - self.season_length + (i % self.season_length)
            )
            if lookback_idx >= 0 and lookback_idx < len(self.train_data):
                predictions.iloc[i] = self.train_data.iloc[lookback_idx]
            else:
                predictions.iloc[i] = self.train_data.iloc[-1]

        return predictions

    def _moving_average_forecast(self, horizon: int) -> pd.Series:
        """
        Moving average forecast.
        """
        # Calculate moving average from last window
        ma_value = self.train_data.iloc[-self.window_size :].mean()

        last_timestamp = self.train_data.index[-1]
        freq = pd.infer_freq(self.train_data.index)
        if freq is None:
            freq = "H"

        pred_index = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1), periods=horizon, freq=freq
        )

        predictions = pd.Series(ma_value, index=pred_index)

        return predictions

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """
        Evaluate predictions.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with metrics
        """
        # Align series
        common_index = y_true.index.intersection(y_pred.index)
        y_true_aligned = y_true.loc[common_index]
        y_pred_aligned = y_pred.loc[common_index]

        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned)),
            "mae": mean_absolute_error(y_true_aligned, y_pred_aligned),
            "mape": np.mean(np.abs((y_true_aligned - y_pred_aligned) / y_true_aligned))
            * 100,
            "r2": r2_score(y_true_aligned, y_pred_aligned),
            "max_error": np.max(np.abs(y_true_aligned - y_pred_aligned)),
        }

        logger.info(
            f"Baseline evaluation - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}"
        )

        return metrics


def compare_baselines(
    train_data: pd.Series,
    test_data: pd.Series,
    horizon: int = 24,
    methods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare multiple baseline methods.

    Args:
        train_data: Training time series
        test_data: Test time series
        horizon: Forecast horizon
        methods: List of baseline methods to compare

    Returns:
        DataFrame with comparison results
    """
    if methods is None:
        methods = ["persistence", "seasonal_naive", "moving_average"]

    logger.info(f"Comparing {len(methods)} baseline methods")

    results = []

    for method in methods:
        model = BaselineModel(method=method)
        model.fit(train_data)
        predictions = model.predict(horizon)

        # Align with test data
        common_index = test_data.index.intersection(predictions.index)
        if len(common_index) == 0:
            logger.warning(f"No common timestamps for {method}")
            continue

        metrics = model.evaluate(
            test_data.loc[common_index], predictions.loc[common_index]
        )

        metrics["method"] = method
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df.set_index("method")

    logger.info("Baseline comparison complete")

    return results_df
