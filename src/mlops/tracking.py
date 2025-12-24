"""MLflow experiment tracking for load forecasting models."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List
from pathlib import Path
from datetime import datetime
import json

from src.logger import get_logger

logger = get_logger(__name__)

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Install with: pip install mlflow")


class ExperimentTracker:
    """
    MLflow-based experiment tracking for load forecasting.
    
    Tracks parameters, metrics, artifacts, and model versions.
    """
    
    def __init__(
        self,
        experiment_name: str = "xai-load-forecasting",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """
        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local)
            artifact_location: Path for artifact storage
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow required. Install with: pip install mlflow")
        
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "mlruns"
        self.artifact_location = artifact_location
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        self.experiment = mlflow.set_experiment(experiment_name)
        self.experiment_id = self.experiment.experiment_id
        
        self.client = MlflowClient()
        self.active_run = None
        
        logger.info(
            f"ExperimentTracker initialized: experiment='{experiment_name}', "
            f"id={self.experiment_id}"
        )
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for this run
            tags: Optional tags dict
            description: Run description
            
        Returns:
            Run ID
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags,
            description=description
        )
        
        run_id = self.active_run.info.run_id
        logger.info(f"Started run: {run_name} (ID: {run_id})")
        
        return run_id
    
    def end_run(self, status: str = "FINISHED"):
        """End the active run."""
        if self.active_run:
            mlflow.end_run(status=status)
            logger.info(f"Ended run with status: {status}")
            self.active_run = None
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to active run."""
        # Flatten nested dicts
        flat_params = self._flatten_dict(params)
        
        # MLflow has 500 char limit for param values
        for key, value in flat_params.items():
            str_value = str(value)
            if len(str_value) > 500:
                str_value = str_value[:497] + "..."
            mlflow.log_param(key, str_value)
        
        logger.debug(f"Logged {len(flat_params)} parameters")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """Log metrics to active run."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                mlflow.log_metric(key, value, step=step)
        
        logger.debug(f"Logged {len(metrics)} metrics")
    
    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_name: Optional[str] = None,
        input_example: Optional[pd.DataFrame] = None
    ):
        """
        Log model to MLflow.
        
        Args:
            model: Model object to log
            artifact_path: Path within artifacts
            registered_name: Optional name for model registry
            input_example: Example input for signature inference
        """
        # Determine model flavor
        model_type = type(model).__name__
        
        if "lightgbm" in model_type.lower() or hasattr(model, "booster_"):
            mlflow.lightgbm.log_model(
                model,
                artifact_path,
                registered_model_name=registered_name,
                input_example=input_example
            )
        elif "xgboost" in model_type.lower():
            mlflow.xgboost.log_model(
                model,
                artifact_path,
                registered_model_name=registered_name,
                input_example=input_example
            )
        else:
            # Generic sklearn model
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                registered_model_name=registered_name,
                input_example=input_example
            )
        
        logger.info(f"Logged model to {artifact_path}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a local file as artifact."""
        mlflow.log_artifact(local_path, artifact_path)
        logger.debug(f"Logged artifact: {local_path}")
    
    def log_figure(self, figure, artifact_name: str):
        """Log matplotlib figure as artifact."""
        mlflow.log_figure(figure, artifact_name)
        logger.debug(f"Logged figure: {artifact_name}")
    
    def log_dict(self, data: Dict, artifact_name: str):
        """Log dictionary as JSON artifact."""
        mlflow.log_dict(data, artifact_name)
        logger.debug(f"Logged dict: {artifact_name}")
    
    def log_shap_values(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        artifact_path: str = "shap"
    ):
        """Log SHAP values and feature importance."""
        # Compute mean absolute SHAP
        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": np.abs(shap_values).mean(axis=0)
        }).sort_values("importance", ascending=False)
        
        # Log as metrics (top 10)
        for i, row in importance.head(10).iterrows():
            mlflow.log_metric(
                f"shap_importance_{row['feature']}", 
                row["importance"]
            )
        
        # Log full importance as artifact
        self.log_dict(
            importance.to_dict(orient="records"),
            f"{artifact_path}/feature_importance.json"
        )
        
        logger.info("Logged SHAP values and importance")
    
    def log_training_run(
        self,
        model: Any,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        feature_names: List[str],
        shap_values: Optional[np.ndarray] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Complete logging for a training run.
        
        Args:
            model: Trained model
            params: Training parameters
            metrics: Evaluation metrics
            feature_names: List of feature names
            shap_values: Optional SHAP values
            run_name: Run name
            tags: Optional tags
            
        Returns:
            Run ID
        """
        run_id = self.start_run(run_name=run_name, tags=tags)
        
        try:
            # Log parameters
            self.log_params(params)
            
            # Log metrics
            self.log_metrics(metrics)
            
            # Log feature names
            self.log_dict(
                {"features": feature_names},
                "features.json"
            )
            
            # Log model
            self.log_model(model, "model")
            
            # Log SHAP if provided
            if shap_values is not None:
                self.log_shap_values(shap_values, feature_names)
            
            self.end_run("FINISHED")
            
        except Exception as e:
            logger.error(f"Error during logging: {e}")
            self.end_run("FAILED")
            raise
        
        return run_id
    
    def get_best_run(
        self,
        metric: str = "rmse",
        ascending: bool = True
    ) -> Optional[Dict]:
        """
        Get the best run based on a metric.
        
        Args:
            metric: Metric name to optimize
            ascending: True for minimize, False for maximize
            
        Returns:
            Best run info dict
        """
        order = "ASC" if ascending else "DESC"
        
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1
        )
        
        if not runs:
            return None
        
        best_run = runs[0]
        return {
            "run_id": best_run.info.run_id,
            "run_name": best_run.info.run_name,
            "metrics": best_run.data.metrics,
            "params": best_run.data.params,
        }
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: List[str]
    ) -> pd.DataFrame:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: Metrics to compare
            
        Returns:
            DataFrame with comparison
        """
        results = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            row = {"run_id": run_id, "run_name": run.info.run_name}
            
            for metric in metrics:
                row[metric] = run.data.metrics.get(metric)
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def _flatten_dict(
        self,
        d: Dict,
        parent_key: str = "",
        sep: str = "."
    ) -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
