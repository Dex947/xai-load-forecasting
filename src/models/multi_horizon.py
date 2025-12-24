"""Multi-horizon forecasting with direct and recursive strategies."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import joblib
from pathlib import Path

import lightgbm as lgb
from sklearn.metrics import mean_squared_error

from src.logger import get_logger

logger = get_logger(__name__)


class MultiHorizonForecaster:
    """Direct multi-horizon forecasting with separate models per horizon."""
    
    def __init__(
        self,
        horizons: List[int] = [1, 6, 12, 24, 48, 168],
        config: Optional[Dict] = None
    ):
        """
        Args:
            horizons: Forecast horizons in hours
            config: LightGBM parameters
        """
        self.horizons = sorted(horizons)
        self.config = config or {}
        self.models: Dict[int, lgb.Booster] = {}
        self.feature_names: Optional[List[str]] = None
        
        logger.info(f"MultiHorizonForecaster initialized: horizons={self.horizons}")
    
    def _create_horizon_target(
        self, 
        y: pd.Series, 
        horizon: int
    ) -> pd.Series:
        """Shift target by horizon for direct forecasting."""
        return y.shift(-horizon)
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        val_ratio: float = 0.2
    ) -> "MultiHorizonForecaster":
        """Train a model for each horizon."""
        self.feature_names = list(X.columns)
        
        base_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "verbose": -1,
            **self.config
        }
        
        # Split into train/val temporally
        split_idx = int(len(X) * (1 - val_ratio))
        
        for h in self.horizons:
            logger.info(f"Training horizon {h}h model...")
            
            # Create shifted target
            y_shifted = self._create_horizon_target(y, h)
            
            # Remove NaN rows (from shifting)
            valid_mask = ~y_shifted.isna()
            X_valid = X[valid_mask]
            y_valid = y_shifted[valid_mask]
            
            # Split
            train_end = min(split_idx, len(X_valid) - 1)
            X_train = X_valid.iloc[:train_end]
            y_train = y_valid.iloc[:train_end]
            X_val = X_valid.iloc[train_end:]
            y_val = y_valid.iloc[train_end:]
            
            if len(X_train) < 100 or len(X_val) < 10:
                logger.warning(f"Insufficient data for horizon {h}h, skipping")
                continue
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            self.models[h] = lgb.train(
                base_params,
                train_data,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=0)
                ]
            )
        
        logger.info(f"Trained {len(self.models)} horizon models")
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict all horizons."""
        if not self.models:
            raise ValueError("Model not fitted")
        
        results = {}
        for h, model in self.models.items():
            results[f"h{h}"] = model.predict(X)
        
        return pd.DataFrame(results, index=X.index if hasattr(X, "index") else None)
    
    def predict_horizon(self, X: pd.DataFrame, horizon: int) -> np.ndarray:
        """Predict specific horizon."""
        if horizon not in self.models:
            available = list(self.models.keys())
            raise ValueError(f"Horizon {horizon} not available. Available: {available}")
        
        return self.models[horizon].predict(X)
    
    def evaluate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[int, Dict[str, float]]:
        """Evaluate all horizons."""
        results = {}
        
        for h in self.models.keys():
            y_shifted = self._create_horizon_target(y, h)
            valid_mask = ~y_shifted.isna()
            
            X_valid = X[valid_mask]
            y_valid = y_shifted[valid_mask]
            
            if len(X_valid) == 0:
                continue
            
            preds = self.models[h].predict(X_valid)
            rmse = np.sqrt(mean_squared_error(y_valid, preds))
            mae = np.mean(np.abs(y_valid - preds))
            
            results[h] = {
                "rmse": float(rmse),
                "mae": float(mae),
                "n_samples": len(y_valid),
            }
        
        return results
    
    def save(self, path: str):
        """Save all horizon models."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "horizons": self.horizons,
            "models": self.models,
            "feature_names": self.feature_names,
            "config": self.config,
        }
        joblib.dump(data, path)
        logger.info(f"Multi-horizon models saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "MultiHorizonForecaster":
        """Load models from file."""
        data = joblib.load(path)
        
        forecaster = cls(horizons=data["horizons"], config=data["config"])
        forecaster.models = data["models"]
        forecaster.feature_names = data["feature_names"]
        
        logger.info(f"Multi-horizon models loaded from {path}")
        return forecaster


class RecursiveForecaster:
    """Recursive multi-step forecasting using single model."""
    
    def __init__(self, base_model, max_horizon: int = 168):
        """
        Args:
            base_model: Trained single-step model
            max_horizon: Maximum forecast horizon
        """
        self.base_model = base_model
        self.max_horizon = max_horizon
        self.feature_names = base_model.feature_names
    
    def predict(
        self, 
        X_last: pd.Series, 
        horizon: int,
        feature_builder: callable
    ) -> np.ndarray:
        """
        Recursively predict multiple steps.
        
        Args:
            X_last: Last known feature vector
            horizon: Number of steps to predict
            feature_builder: Function to build features from history
        
        Returns:
            Array of predictions
        """
        if horizon > self.max_horizon:
            raise ValueError(f"Horizon {horizon} exceeds max {self.max_horizon}")
        
        predictions = []
        current_features = X_last.copy()
        
        for step in range(horizon):
            # Predict next step
            pred = self.base_model.predict(current_features.to_frame().T)[0]
            predictions.append(pred)
            
            # Update features for next step (requires feature_builder)
            current_features = feature_builder(current_features, pred, step + 1)
        
        return np.array(predictions)
