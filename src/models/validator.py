"""
Time-series cross-validation with rolling origin.
"""

import pandas as pd
from typing import Iterator, Tuple, Optional, List, Dict
from datetime import timedelta

from src.logger import get_logger

logger = get_logger(__name__)


class RollingOriginValidator:
    """Rolling origin CV that prevents temporal leakage.
    Ensures temporal ordering: train data always comes before test data.
    Supports gap between train and test to simulate real forecasting scenarios.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size_days: int = 30,
        gap_days: int = 1,
        min_train_days: int = 365,
    ):
        """
        Initialize rolling origin validator.

        Args:
            n_splits: Number of cross-validation splits
            test_size_days: Size of test set in days
            gap_days: Gap between train and test sets in days
            min_train_days: Minimum training period in days
        """
        self.n_splits = n_splits
        self.test_size_days = test_size_days
        self.gap_days = gap_days
        self.min_train_days = min_train_days

        logger.info(
            f"Rolling origin validator initialized: {n_splits} splits, "
            f"test_size={test_size_days}d, gap={gap_days}d, min_train={min_train_days}d"
        )

    def split(
        self, df: pd.DataFrame
    ) -> Iterator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Generate train/test splits.

        Args:
            df: DataFrame with DatetimeIndex

        Yields:
            Tuple of (train_index, test_index)
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        logger.info(f"Generating {self.n_splits} rolling origin splits")

        # Calculate split points
        data_start = df.index.min()
        data_end = df.index.max()
        total_days = (data_end - data_start).days

        logger.info(f"Data range: {data_start} to {data_end} ({total_days} days)")

        # Calculate minimum required days
        min_required_days = self.min_train_days + self.gap_days + self.test_size_days

        if total_days < min_required_days:
            raise ValueError(
                f"Insufficient data: {total_days} days available, "
                f"{min_required_days} days required"
            )

        # Calculate available days for splits
        available_days = (
            total_days - self.min_train_days - self.gap_days - self.test_size_days
        )

        if available_days < 0:
            raise ValueError("Not enough data for even one split")

        # Calculate step size between splits
        if self.n_splits > 1 and available_days > 0:
            step_days = available_days / (self.n_splits - 1)
        else:
            step_days = 0

        # Generate splits
        for split_idx in range(self.n_splits):
            # Calculate train end
            train_end = data_start + timedelta(
                days=self.min_train_days + int(split_idx * step_days)
            )

            # Calculate test start and end
            test_start = train_end + timedelta(days=self.gap_days)
            test_end = test_start + timedelta(days=self.test_size_days)

            # Ensure test_end doesn't exceed data_end
            if test_end > data_end:
                test_end = data_end

            # Get indices
            train_mask = (df.index >= data_start) & (df.index <= train_end)
            test_mask = (df.index >= test_start) & (df.index <= test_end)

            train_index = df.index[train_mask]
            test_index = df.index[test_mask]

            logger.info(f"Split {split_idx + 1}/{self.n_splits}:")
            logger.info(
                f"  Train: {train_index.min()} to {train_index.max()} ({len(train_index)} samples)"
            )
            logger.info(
                f"  Test:  {test_index.min()} to {test_index.max()} ({len(test_index)} samples)"
            )

            yield train_index, test_index

    def validate_model(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str,
        model_class,
        model_params: Optional[Dict] = None,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Perform rolling origin cross-validation for a model.

        Args:
            df: DataFrame with features and target
            features: List of feature column names
            target: Target column name
            model_class: Model class to instantiate
            model_params: Parameters for model initialization
            metrics: List of metrics to compute

        Returns:
            DataFrame with validation results for each split
        """
        if metrics is None:
            metrics = ["rmse", "mae", "mape", "r2"]

        if model_params is None:
            model_params = {}

        logger.info("Starting rolling origin cross-validation")

        results = []

        for split_idx, (train_idx, test_idx) in enumerate(self.split(df)):
            logger.info(f"Validating split {split_idx + 1}/{self.n_splits}")

            # Prepare data
            X_train = df.loc[train_idx, features]
            y_train = df.loc[train_idx, target]
            X_test = df.loc[test_idx, features]
            y_test = df.loc[test_idx, target]

            # Remove rows with missing values
            train_valid = ~(X_train.isna().any(axis=1) | y_train.isna())
            test_valid = ~(X_test.isna().any(axis=1) | y_test.isna())

            X_train = X_train[train_valid]
            y_train = y_train[train_valid]
            X_test = X_test[test_valid]
            y_test = y_test[test_valid]

            if len(X_train) == 0 or len(X_test) == 0:
                logger.warning(f"Split {split_idx + 1} has no valid data, skipping")
                continue

            # Train model
            model = model_class(**model_params)
            model.fit(X_train, y_train)

            # Evaluate
            split_metrics = model.evaluate(X_test, y_test, metrics=metrics)
            split_metrics["split"] = split_idx + 1
            split_metrics["train_size"] = len(X_train)
            split_metrics["test_size"] = len(X_test)
            split_metrics["train_start"] = train_idx.min()
            split_metrics["train_end"] = train_idx.max()
            split_metrics["test_start"] = test_idx.min()
            split_metrics["test_end"] = test_idx.max()

            results.append(split_metrics)

        results_df = pd.DataFrame(results)

        # Log summary
        logger.info("Cross-validation complete")
        logger.info("Summary statistics:")
        for metric in metrics:
            if metric in results_df.columns:
                mean_val = results_df[metric].mean()
                std_val = results_df[metric].std()
                logger.info(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")

        return results_df


class ExpandingWindowValidator:
    """Expanding window CV with growing training set.
    Similar to rolling origin, but training window expands instead of sliding.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size_days: int = 30,
        gap_days: int = 1,
        min_train_days: int = 365,
    ):
        """
        Initialize expanding window validator.

        Args:
            n_splits: Number of cross-validation splits
            test_size_days: Size of test set in days
            gap_days: Gap between train and test sets in days
            min_train_days: Minimum training period in days
        """
        self.n_splits = n_splits
        self.test_size_days = test_size_days
        self.gap_days = gap_days
        self.min_train_days = min_train_days

        logger.info(f"Expanding window validator initialized: {n_splits} splits")

    def split(
        self, df: pd.DataFrame
    ) -> Iterator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Generate train/test splits with expanding window.

        Args:
            df: DataFrame with DatetimeIndex

        Yields:
            Tuple of (train_index, test_index)
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        logger.info(f"Generating {self.n_splits} expanding window splits")

        data_start = df.index.min()
        data_end = df.index.max()
        total_days = (data_end - data_start).days

        # Calculate available days for splits
        available_days = (
            total_days - self.min_train_days - self.gap_days - self.test_size_days
        )

        if available_days < 0:
            raise ValueError("Not enough data for even one split")

        # Calculate step size
        if self.n_splits > 1:
            step_days = available_days / (self.n_splits - 1)
        else:
            step_days = 0

        # Generate splits
        for split_idx in range(self.n_splits):
            # Train always starts from beginning (expanding window)
            train_start = data_start
            train_end = data_start + timedelta(
                days=self.min_train_days + int(split_idx * step_days)
            )

            # Test period
            test_start = train_end + timedelta(days=self.gap_days)
            test_end = test_start + timedelta(days=self.test_size_days)

            if test_end > data_end:
                test_end = data_end

            # Get indices
            train_mask = (df.index >= train_start) & (df.index <= train_end)
            test_mask = (df.index >= test_start) & (df.index <= test_end)

            train_index = df.index[train_mask]
            test_index = df.index[test_mask]

            logger.info(f"Split {split_idx + 1}/{self.n_splits}:")
            logger.info(
                f"  Train: {train_index.min()} to {train_index.max()} ({len(train_index)} samples)"
            )
            logger.info(
                f"  Test:  {test_index.min()} to {test_index.max()} ({len(test_index)} samples)"
            )

            yield train_index, test_index
