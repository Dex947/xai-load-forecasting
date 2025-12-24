"""Quantile regression for prediction intervals."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import joblib
from pathlib import Path

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from src.logger import get_logger

logger = get_logger(__name__)


class QuantileForecaster:
    """LightGBM quantile regression for prediction intervals."""
    
    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        config: Optional[Dict] = None
    ):
        """
        Args:
            quantiles: Quantiles to predict (e.g., [0.1, 0.5, 0.9] for 80% interval)
            config: LightGBM parameters
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM required for quantile regression")
        
        self.quantiles = sorted(quantiles)
        self.config = config or {}
        self.models: Dict[float, lgb.Booster] = {}
        self.feature_names: Optional[List[str]] = None
        
        logger.info(f"QuantileForecaster initialized with quantiles: {self.quantiles}")
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> "QuantileForecaster":
        """Train a model for each quantile."""
        self.feature_names = list(X_train.columns)
        
        base_params = {
            "objective": "quantile",
            "metric": "quantile",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "verbose": -1,
            **self.config
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        valid_names = ["train"]
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")
        
        for q in self.quantiles:
            logger.info(f"Training quantile {q:.2f} model...")
            params = {**base_params, "alpha": q}
            
            callbacks = [lgb.log_evaluation(period=0)]
            if X_val is not None:
                callbacks.append(lgb.early_stopping(stopping_rounds=50))
            
            self.models[q] = lgb.train(
                params,
                train_data,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks
            )
        
        logger.info(f"Trained {len(self.models)} quantile models")
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict all quantiles.
        
        Returns DataFrame with columns for each quantile.
        """
        if not self.models:
            raise ValueError("Model not fitted")
        
        results = {}
        for q, model in self.models.items():
            results[f"q{int(q*100):02d}"] = model.predict(X)
        
        return pd.DataFrame(results, index=X.index if hasattr(X, "index") else None)
    
    def predict_interval(
        self, 
        X: pd.DataFrame, 
        lower_q: float = 0.1, 
        upper_q: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence interval.
        
        Returns (lower, median, upper) arrays.
        """
        preds = self.predict(X)
        
        lower_col = f"q{int(lower_q*100):02d}"
        upper_col = f"q{int(upper_q*100):02d}"
        median_col = "q50"
        
        lower = preds[lower_col].values if lower_col in preds else preds.iloc[:, 0].values
        upper = preds[upper_col].values if upper_col in preds else preds.iloc[:, -1].values
        median = preds[median_col].values if median_col in preds else preds.median(axis=1).values
        
        return lower, median, upper
    
    def evaluate_coverage(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        lower_q: float = 0.1,
        upper_q: float = 0.9
    ) -> Dict[str, float]:
        """Evaluate prediction interval coverage."""
        lower, median, upper = self.predict_interval(X, lower_q, upper_q)
        
        in_interval = (y.values >= lower) & (y.values <= upper)
        coverage = in_interval.mean()
        expected_coverage = upper_q - lower_q
        
        interval_width = (upper - lower).mean()
        
        return {
            "coverage": float(coverage),
            "expected_coverage": expected_coverage,
            "coverage_gap": float(coverage - expected_coverage),
            "mean_interval_width": float(interval_width),
            "median_rmse": float(np.sqrt(((y.values - median) ** 2).mean())),
        }
    
    def save(self, path: str):
        """Save all quantile models."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "quantiles": self.quantiles,
            "models": self.models,
            "feature_names": self.feature_names,
            "config": self.config,
        }
        joblib.dump(data, path)
        logger.info(f"Quantile models saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "QuantileForecaster":
        """Load quantile models from file."""
        data = joblib.load(path)
        
        forecaster = cls(quantiles=data["quantiles"], config=data["config"])
        forecaster.models = data["models"]
        forecaster.feature_names = data["feature_names"]
        
        logger.info(f"Quantile models loaded from {path}")
        return forecaster
