"""SHAP value computation and analysis."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import shap
import joblib
from pathlib import Path

from src.logger import get_logger

logger = get_logger(__name__)


class SHAPAnalyzer:
    """Computes global, local, and time-varying SHAP importance."""

    def __init__(
        self,
        model,
        X_background: pd.DataFrame,
        model_type: str = "tree",
        config: Optional[Dict] = None,
    ):
        self.model = model
        self.X_background = X_background
        self.model_type = model_type
        self.config = config or {}

        self.explainer = None
        self.shap_values = None
        self.feature_names = list(X_background.columns)

        logger.info(
            f"SHAP analyzer initialized with {len(X_background)} background samples"
        )

        # Initialize explainer
        self._initialize_explainer()

    def _initialize_explainer(self) -> None:
        logger.info("Initializing SHAP explainer")

        if self.model_type == "tree":
            # TreeExplainer for tree-based models
            self.explainer = shap.TreeExplainer(self.model, self.X_background)
        else:
            # KernelExplainer as fallback (slower)
            logger.warning(
                "Using KernelExplainer (slower). Consider using TreeExplainer for tree models."
            )

            def model_predict(X):
                return self.model.predict(X)

            self.explainer = shap.KernelExplainer(model_predict, self.X_background)

        logger.info("SHAP explainer initialized")

    def compute_shap_values(
        self, X: pd.DataFrame, check_additivity: bool = False
    ) -> np.ndarray:
        """Compute SHAP values, sampling if dataset is large."""
        logger.info(f"Computing SHAP values for {len(X)} samples")

        # Sample if too large
        sample_size = self.config.get("sample_size", 1000)
        if len(X) > sample_size:
            logger.info(f"Sampling {sample_size} samples for SHAP computation")
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X

        # Compute SHAP values
        shap_values = self.explainer.shap_values(
            X_sample, check_additivity=check_additivity
        )

        # Store for later use
        self.shap_values = shap_values
        self.X_explained = X_sample

        logger.info("SHAP values computed successfully")

        return shap_values

    def get_global_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get global feature importance from SHAP values.

        Args:
            top_n: Return top N features (None for all)

        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            raise ValueError(
                "SHAP values not computed. Call compute_shap_values() first."
            )

        logger.info("Computing global feature importance")

        # Mean absolute SHAP value for each feature
        importance = np.abs(self.shap_values).mean(axis=0)

        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        if top_n is not None:
            importance_df = importance_df.head(top_n)

        return importance_df

    def get_time_varying_importance(
        self,
        X: pd.DataFrame,
        time_column: Optional[str] = None,
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        """
        Compute time-varying feature importance.

        Args:
            X: Data with datetime index or time column
            time_column: Name of time column (uses index if None)
            aggregation: Aggregation method ('mean', 'median', 'std')

        Returns:
            DataFrame with time-varying importance
        """
        if self.shap_values is None:
            raise ValueError(
                "SHAP values not computed. Call compute_shap_values() first."
            )

        logger.info("Computing time-varying feature importance")

        # Create DataFrame with SHAP values
        shap_df = pd.DataFrame(
            self.shap_values, columns=self.feature_names, index=self.X_explained.index
        )

        # Add time features for grouping
        if isinstance(shap_df.index, pd.DatetimeIndex):
            shap_df["hour"] = shap_df.index.hour
            shap_df["day_of_week"] = shap_df.index.dayofweek
            shap_df["month"] = shap_df.index.month

        # Aggregate by time periods
        time_varying = {}

        # Hourly pattern
        if "hour" in shap_df.columns:
            hourly = shap_df.groupby("hour")[self.feature_names].agg(
                lambda x: np.abs(x).mean()
                if aggregation == "mean"
                else np.abs(x).median()
            )
            time_varying["hourly"] = hourly

        # Day of week pattern
        if "day_of_week" in shap_df.columns:
            daily = shap_df.groupby("day_of_week")[self.feature_names].agg(
                lambda x: np.abs(x).mean()
                if aggregation == "mean"
                else np.abs(x).median()
            )
            time_varying["daily"] = daily

        # Monthly pattern
        if "month" in shap_df.columns:
            monthly = shap_df.groupby("month")[self.feature_names].agg(
                lambda x: np.abs(x).mean()
                if aggregation == "mean"
                else np.abs(x).median()
            )
            time_varying["monthly"] = monthly

        logger.info(
            f"Computed time-varying importance for {len(time_varying)} time periods"
        )

        return time_varying

    def get_feature_interactions(
        self, feature1: str, feature2: str, X: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SHAP interaction values for two features.

        Args:
            feature1: First feature name
            feature2: Second feature name
            X: Data to explain (uses stored data if None)

        Returns:
            Tuple of (feature1_values, interaction_values)
        """
        if X is None:
            X = self.X_explained

        if X is None:
            raise ValueError("No data available for interaction analysis")

        logger.info(f"Computing SHAP interactions between {feature1} and {feature2}")

        # Compute interaction values (expensive operation)
        shap_interaction_values = self.explainer.shap_interaction_values(X)

        # Extract interaction for specific features
        idx1 = self.feature_names.index(feature1)
        idx2 = self.feature_names.index(feature2)

        feature1_values = X[feature1].values
        interaction_values = shap_interaction_values[:, idx1, idx2]

        return feature1_values, interaction_values

    def save_shap_values(self, file_path: str) -> None:
        """
        Save computed SHAP values to file.

        Args:
            file_path: Output file path
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values to save")

        logger.info(f"Saving SHAP values to {file_path}")

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        shap_data = {
            "shap_values": self.shap_values,
            "feature_names": self.feature_names,
            "X_explained": self.X_explained,
            "expected_value": self.explainer.expected_value,
        }

        joblib.dump(shap_data, file_path)
        logger.info("SHAP values saved successfully")

    @classmethod
    def load_shap_values(cls, file_path: str) -> Dict:
        """
        Load SHAP values from file.

        Args:
            file_path: Input file path

        Returns:
            Dictionary with SHAP data
        """
        logger.info(f"Loading SHAP values from {file_path}")

        shap_data = joblib.load(file_path)

        logger.info("SHAP values loaded successfully")

        return shap_data

    def explain_prediction(
        self, X_instance: pd.Series, top_n: int = 10
    ) -> pd.DataFrame:
        """
        Explain a single prediction.

        Args:
            X_instance: Single instance to explain
            top_n: Number of top features to return

        Returns:
            DataFrame with feature contributions
        """
        logger.info("Explaining single prediction")

        # Reshape to DataFrame if Series
        if isinstance(X_instance, pd.Series):
            X_instance = X_instance.to_frame().T

        # Compute SHAP values
        shap_values = self.explainer.shap_values(X_instance)

        # Create explanation DataFrame
        explanation = pd.DataFrame(
            {
                "feature": self.feature_names,
                "value": X_instance.iloc[0].values,
                "shap_value": shap_values[0],
                "abs_shap_value": np.abs(shap_values[0]),
            }
        ).sort_values("abs_shap_value", ascending=False)

        return explanation.head(top_n)
