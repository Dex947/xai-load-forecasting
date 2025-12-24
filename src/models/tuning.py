"""Hyperparameter optimization with Optuna."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from pathlib import Path
import json

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

import lightgbm as lgb
from sklearn.metrics import mean_squared_error

from src.logger import get_logger
from src.models.validator import RollingOriginValidator

logger = get_logger(__name__)


class HyperparameterTuner:
    """Optuna-based hyperparameter optimization for LightGBM."""
    
    def __init__(
        self,
        n_trials: int = 50,
        cv_splits: int = 3,
        timeout: Optional[int] = None,
        random_state: int = 42
    ):
        """
        Args:
            n_trials: Number of optimization trials
            cv_splits: Number of cross-validation splits
            timeout: Maximum optimization time in seconds
            random_state: Random seed for reproducibility
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required. Install with: pip install optuna")
        
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.timeout = timeout
        self.random_state = random_state
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict] = None
        
        logger.info(f"HyperparameterTuner initialized: {n_trials} trials, {cv_splits} CV splits")
    
    def _objective(
        self,
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: pd.Series,
        validator: RollingOriginValidator
    ) -> float:
        """Optuna objective function."""
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "n_jobs": -1,
            "random_state": self.random_state,
            
            # Tunable parameters
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        
        # Cross-validation
        cv_scores = []
        
        for train_idx, val_idx in validator.split(X):
            X_train = X.loc[train_idx]
            y_train = y.loc[train_idx]
            X_val = X.loc[val_idx]
            y_val = y.loc[val_idx]
            
            # Remove NaN rows
            train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
            val_mask = ~(X_val.isna().any(axis=1) | y_val.isna())
            
            X_train = X_train[train_mask]
            y_train = y_train[train_mask]
            X_val = X_val[val_mask]
            y_val = y_val[val_mask]
            
            if len(X_train) == 0 or len(X_val) == 0:
                continue
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            cv_scores.append(rmse)
        
        if not cv_scores:
            return float("inf")
        
        return np.mean(cv_scores)
    
    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size_days: int = 30,
        min_train_days: int = 180
    ) -> Dict:
        """
        Run hyperparameter optimization.
        
        Returns best parameters found.
        """
        logger.info("Starting hyperparameter optimization...")
        
        validator = RollingOriginValidator(
            n_splits=self.cv_splits,
            test_size_days=test_size_days,
            min_train_days=min_train_days
        )
        
        sampler = TPESampler(seed=self.random_state)
        self.study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name="lgbm_tuning"
        )
        
        self.study.optimize(
            lambda trial: self._objective(trial, X, y, validator),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        
        logger.info(f"Optimization complete. Best RMSE: {self.study.best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        return self.best_params
    
    def get_full_config(self) -> Dict:
        """Get complete LightGBM config with best params."""
        if self.best_params is None:
            raise ValueError("Run optimize() first")
        
        return {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "n_jobs": -1,
            "random_state": self.random_state,
            **self.best_params
        }
    
    def save_results(self, path: str):
        """Save optimization results."""
        if self.study is None:
            raise ValueError("No study to save")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            "best_params": self.best_params,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials),
            "trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": str(t.state),
                }
                for t in self.study.trials
            ]
        }
        
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {path}")
