"""LightGBM/XGBoost wrapper with monotonic constraint support."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import joblib
from pathlib import Path

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.logger import get_logger

logger = get_logger(__name__)


class GradientBoostingModel:
    """Gradient boosting model with monotonic constraints."""

    def __init__(
        self,
        model_type: str = "lightgbm",
        config: Optional[Dict] = None,
        monotonic_constraints: Optional[Dict[str, int]] = None,
    ):
        """Initialize with model type, config, and optional monotonic constraints."""
        self.model_type = model_type.lower()
        self.config = config or {}
        self.monotonic_constraints = monotonic_constraints or {}
        self.model = None
        self.feature_names = None
        self.feature_importance_ = None

        # Validate model type
        if self.model_type == "lightgbm" and not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")
        if self.model_type == "xgboost" and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")

        logger.info(f"Gradient boosting model initialized: {model_type}")
        if monotonic_constraints:
            logger.info(f"Monotonic constraints: {monotonic_constraints}")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: bool = True,
    ) -> "GradientBoostingModel":
        """Train model with optional validation set for early stopping."""
        logger.info(f"Training {self.model_type} model")
        logger.info(
            f"Training samples: {len(X_train)}, Features: {len(X_train.columns)}"
        )

        self.feature_names = list(X_train.columns)

        # Prepare monotonic constraints
        monotone_constraints = self._prepare_monotonic_constraints(X_train.columns)

        if self.model_type == "lightgbm":
            self._fit_lightgbm(
                X_train, y_train, X_val, y_val, monotone_constraints, verbose
            )
        elif self.model_type == "xgboost":
            self._fit_xgboost(
                X_train, y_train, X_val, y_val, monotone_constraints, verbose
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Store feature importance
        self.feature_importance_ = self._get_feature_importance()

        logger.info("Model training complete")

        return self

    def _fit_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        monotone_constraints: List[int],
        verbose: bool,
    ) -> None:
        params = self.config.get("lightgbm", {}).copy()

        # Add monotonic constraints
        if any(c != 0 for c in monotone_constraints):
            params["monotone_constraints"] = monotone_constraints
            logger.info(f"Applied monotonic constraints: {monotone_constraints}")

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")

        # Train model
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.log_evaluation(period=50 if verbose else 0),
                lgb.early_stopping(stopping_rounds=50),
            ]
            if X_val is not None
            else [lgb.log_evaluation(period=50 if verbose else 0)],
        )

    def _fit_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        monotone_constraints: List[int],
        verbose: bool,
    ) -> None:
        params = self.config.get("xgboost", {}).copy()

        # Add monotonic constraints
        if any(c != 0 for c in monotone_constraints):
            # XGBoost uses tuple format
            params["monotone_constraints"] = tuple(monotone_constraints)
            logger.info(f"Applied monotonic constraints: {monotone_constraints}")

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)

        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals.append((dval, "valid"))

        # Train model
        self.model = xgb.train(
            params,
            dtrain,
            evals=evals,
            verbose_eval=50 if verbose else False,
            early_stopping_rounds=50 if X_val is not None else None,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for input features."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        logger.info(f"Generating predictions for {len(X)} samples")

        if self.model_type == "lightgbm":
            predictions = self.model.predict(X)
        elif self.model_type == "xgboost":
            dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
            predictions = self.model.predict(dmatrix)

        return predictions

    def _prepare_monotonic_constraints(self, feature_names: List[str]) -> List[int]:
        """Map constraint dict to ordered list matching feature order."""
        constraints = []

        for feature in feature_names:
            # Check exact match first
            if feature in self.monotonic_constraints:
                constraints.append(self.monotonic_constraints[feature])
            # Check partial match (e.g., 'temperature' matches 'temperature_lag_1h')
            else:
                matched = False
                for (
                    constraint_feature,
                    constraint_value,
                ) in self.monotonic_constraints.items():
                    if constraint_feature in feature:
                        constraints.append(constraint_value)
                        matched = True
                        break

                if not matched:
                    constraints.append(0)  # No constraint

        return constraints

    def _get_feature_importance(self) -> pd.DataFrame:
        if self.model_type == "lightgbm":
            importance = self.model.feature_importance(importance_type="gain")
        elif self.model_type == "xgboost":
            importance = list(self.model.get_score(importance_type="gain").values())

        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        return importance_df

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Return feature importance DataFrame, optionally top N."""
        if self.feature_importance_ is None:
            raise ValueError("Model must be fitted first")

        if top_n is not None:
            return self.feature_importance_.head(top_n)

        return self.feature_importance_

    def evaluate(
        self, X: pd.DataFrame, y: pd.Series, metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Compute evaluation metrics (RMSE, MAE, MAPE, R², max_error)."""
        if metrics is None:
            metrics = ["rmse", "mae", "mape", "r2", "max_error"]

        predictions = self.predict(X)

        results = {}

        if "rmse" in metrics:
            results["rmse"] = np.sqrt(mean_squared_error(y, predictions))

        if "mae" in metrics:
            results["mae"] = mean_absolute_error(y, predictions)

        if "mape" in metrics:
            # Use epsilon to avoid division by zero
            eps = np.finfo(float).eps
            results["mape"] = (
                np.mean(np.abs((y - predictions) / (np.abs(y) + eps))) * 100
            )

        if "r2" in metrics:
            results["r2"] = r2_score(y, predictions)

        if "max_error" in metrics:
            results["max_error"] = np.max(np.abs(y - predictions))

        logger.info(
            f"Evaluation - RMSE: {results.get('rmse', 0):.2f}, MAE: {results.get('mae', 0):.2f}"
        )

        return results

    def save(self, file_path: str) -> None:
        """Serialize model to pickle file."""
        logger.info(f"Saving model to {file_path}")

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance_,
            "monotonic_constraints": self.monotonic_constraints,
            "config": self.config,
        }

        joblib.dump(model_data, file_path)
        logger.info("Model saved successfully")

    @classmethod
    def load(cls, file_path: str) -> "GradientBoostingModel":
        """Load model from pickle file."""
        logger.info(f"Loading model from {file_path}")

        model_data = joblib.load(file_path)

        instance = cls(
            model_type=model_data["model_type"],
            config=model_data["config"],
            monotonic_constraints=model_data["monotonic_constraints"],
        )

        instance.model = model_data["model"]
        instance.feature_names = model_data["feature_names"]
        instance.feature_importance_ = model_data["feature_importance"]

        logger.info("Model loaded successfully")

        return instance
