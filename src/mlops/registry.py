"""Model registry for versioning and deployment management."""

import pandas as pd
from typing import Dict, Optional, List, Any
from datetime import datetime
from pathlib import Path

from src.logger import get_logger

logger = get_logger(__name__)

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.entities.model_registry import ModelVersion
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Install with: pip install mlflow")


class ModelRegistry:
    """
    Model registry for managing model versions and deployments.
    
    Supports champion/challenger pattern for safe model updates.
    """
    
    # Stage constants
    STAGE_NONE = "None"
    STAGE_STAGING = "Staging"
    STAGE_PRODUCTION = "Production"
    STAGE_ARCHIVED = "Archived"
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Args:
            tracking_uri: MLflow tracking server URI
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow required. Install with: pip install mlflow")
        
        self.tracking_uri = tracking_uri or "mlruns"
        mlflow.set_tracking_uri(self.tracking_uri)
        
        self.client = MlflowClient()
        
        logger.info(f"ModelRegistry initialized: {self.tracking_uri}")
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model",
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> ModelVersion:
        """
        Register a model from a run.
        
        Args:
            run_id: MLflow run ID containing the model
            model_name: Name for registered model
            artifact_path: Path to model artifact in run
            description: Model description
            tags: Optional tags
            
        Returns:
            ModelVersion object
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        
        # Register model
        mv = mlflow.register_model(model_uri, model_name)
        
        # Add description if provided
        if description:
            self.client.update_model_version(
                name=model_name,
                version=mv.version,
                description=description
            )
        
        # Add tags if provided
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    name=model_name,
                    version=mv.version,
                    key=key,
                    value=value
                )
        
        logger.info(f"Registered model '{model_name}' version {mv.version}")
        return mv
    
    def get_latest_version(
        self,
        model_name: str,
        stage: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """
        Get latest version of a model.
        
        Args:
            model_name: Registered model name
            stage: Optional stage filter
            
        Returns:
            Latest ModelVersion or None
        """
        stages = [stage] if stage else None
        
        try:
            versions = self.client.get_latest_versions(model_name, stages=stages)
            if versions:
                return versions[0]
        except Exception as e:
            logger.warning(f"Could not get latest version: {e}")
        
        return None
    
    def get_production_model(self, model_name: str) -> Optional[Any]:
        """
        Load the production model.
        
        Args:
            model_name: Registered model name
            
        Returns:
            Loaded model or None
        """
        version = self.get_latest_version(model_name, self.STAGE_PRODUCTION)
        
        if version is None:
            logger.warning(f"No production model found for '{model_name}'")
            return None
        
        model_uri = f"models:/{model_name}/{self.STAGE_PRODUCTION}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        logger.info(f"Loaded production model '{model_name}' v{version.version}")
        return model
    
    def promote_to_staging(
        self,
        model_name: str,
        version: int,
        archive_existing: bool = True
    ):
        """
        Promote a model version to staging.
        
        Args:
            model_name: Registered model name
            version: Version number to promote
            archive_existing: Archive current staging model
        """
        if archive_existing:
            current = self.get_latest_version(model_name, self.STAGE_STAGING)
            if current:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=current.version,
                    stage=self.STAGE_ARCHIVED
                )
                logger.info(f"Archived staging v{current.version}")
        
        self.client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage=self.STAGE_STAGING
        )
        
        logger.info(f"Promoted '{model_name}' v{version} to Staging")
    
    def promote_to_production(
        self,
        model_name: str,
        version: int,
        archive_existing: bool = True
    ):
        """
        Promote a model version to production.
        
        Args:
            model_name: Registered model name
            version: Version number to promote
            archive_existing: Archive current production model
        """
        if archive_existing:
            current = self.get_latest_version(model_name, self.STAGE_PRODUCTION)
            if current:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=current.version,
                    stage=self.STAGE_ARCHIVED
                )
                logger.info(f"Archived production v{current.version}")
        
        self.client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage=self.STAGE_PRODUCTION
        )
        
        logger.info(f"Promoted '{model_name}' v{version} to Production")
    
    def rollback_production(self, model_name: str) -> bool:
        """
        Rollback to previous production model.
        
        Args:
            model_name: Registered model name
            
        Returns:
            True if rollback successful
        """
        # Get all versions
        versions = self.client.search_model_versions(f"name='{model_name}'")
        
        # Find archived versions (previous production)
        archived = [
            v for v in versions 
            if v.current_stage == self.STAGE_ARCHIVED
        ]
        
        if not archived:
            logger.warning("No archived versions to rollback to")
            return False
        
        # Get most recent archived version
        latest_archived = max(archived, key=lambda v: int(v.version))
        
        # Demote current production
        current = self.get_latest_version(model_name, self.STAGE_PRODUCTION)
        if current:
            self.client.transition_model_version_stage(
                name=model_name,
                version=current.version,
                stage=self.STAGE_ARCHIVED
            )
        
        # Promote archived to production
        self.client.transition_model_version_stage(
            name=model_name,
            version=latest_archived.version,
            stage=self.STAGE_PRODUCTION
        )
        
        logger.info(
            f"Rolled back '{model_name}' to v{latest_archived.version}"
        )
        return True
    
    def list_versions(self, model_name: str) -> pd.DataFrame:
        """
        List all versions of a model.
        
        Args:
            model_name: Registered model name
            
        Returns:
            DataFrame with version info
        """
        versions = self.client.search_model_versions(f"name='{model_name}'")
        
        data = []
        for v in versions:
            data.append({
                "version": v.version,
                "stage": v.current_stage,
                "status": v.status,
                "created": v.creation_timestamp,
                "description": v.description,
                "run_id": v.run_id,
            })
        
        return pd.DataFrame(data).sort_values("version", ascending=False)
    
    def delete_version(self, model_name: str, version: int):
        """Delete a specific model version."""
        self.client.delete_model_version(
            name=model_name,
            version=str(version)
        )
        logger.info(f"Deleted '{model_name}' v{version}")
    
    def compare_versions(
        self,
        model_name: str,
        version_a: int,
        version_b: int,
        metrics: List[str]
    ) -> Dict[str, Dict]:
        """
        Compare two model versions.
        
        Args:
            model_name: Registered model name
            version_a: First version
            version_b: Second version
            metrics: Metrics to compare
            
        Returns:
            Comparison dict
        """
        versions = self.client.search_model_versions(f"name='{model_name}'")
        
        results = {}
        for v in versions:
            if int(v.version) in [version_a, version_b]:
                run = self.client.get_run(v.run_id)
                results[f"v{v.version}"] = {
                    metric: run.data.metrics.get(metric)
                    for metric in metrics
                }
        
        return results
