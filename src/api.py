"""FastAPI prediction server for XAI Load Forecasting."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path

from src.models.gbm import GradientBoostingModel
from src.explainability.shap_analysis import SHAPAnalyzer
from src.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="XAI Load Forecasting API",
    description="Day-ahead load prediction with SHAP explanations",
    version="1.0.0"
)

# Global model cache
_model_cache: Dict[str, Any] = {}


class PredictionRequest(BaseModel):
    """Request body for predictions."""
    features: Dict[str, float] = Field(..., description="Feature name-value pairs")
    explain: bool = Field(False, description="Include SHAP explanation")


class PredictionResponse(BaseModel):
    """Response body for predictions."""
    prediction: float
    explanation: Optional[Dict[str, float]] = None


class BatchPredictionRequest(BaseModel):
    """Request body for batch predictions."""
    data: List[Dict[str, float]]
    explain: bool = False


class BatchPredictionResponse(BaseModel):
    """Response body for batch predictions."""
    predictions: List[float]
    explanations: Optional[List[Dict[str, float]]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    n_features: Optional[int] = None


def get_model():
    """Load or return cached model."""
    if "model" not in _model_cache:
        model_path = Path("models/artifacts/lightgbm_model.pkl")
        if not model_path.exists():
            raise HTTPException(404, "Model not found. Train a model first.")
        _model_cache["model"] = GradientBoostingModel.load(str(model_path))
        logger.info("Model loaded into cache")
    return _model_cache["model"]


def get_shap_analyzer():
    """Load or return cached SHAP analyzer."""
    if "shap" not in _model_cache:
        shap_path = Path("models/artifacts/shap_values.pkl")
        if shap_path.exists():
            shap_data = SHAPAnalyzer.load_shap_values(str(shap_path))
            _model_cache["shap_data"] = shap_data
        else:
            _model_cache["shap_data"] = None
    return _model_cache.get("shap_data")


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check API and model health."""
    try:
        model = get_model()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_type=model.model_type,
            n_features=len(model.feature_names)
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            model_loaded=False
        )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Generate a single prediction with optional SHAP explanation."""
    model = get_model()
    
    # Validate features
    missing = set(model.feature_names) - set(request.features.keys())
    if missing:
        raise HTTPException(400, f"Missing features: {missing}")
    
    # Build feature vector in correct order
    X = pd.DataFrame([{f: request.features[f] for f in model.feature_names}])
    
    prediction = float(model.predict(X)[0])
    
    explanation = None
    if request.explain:
        shap_data = get_shap_analyzer()
        if shap_data:
            # Simple approximation using global importance
            importance = pd.DataFrame({
                "feature": shap_data["feature_names"],
                "importance": np.abs(shap_data["shap_values"]).mean(axis=0)
            })
            explanation = dict(zip(importance["feature"], importance["importance"]))
    
    return PredictionResponse(prediction=prediction, explanation=explanation)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """Generate batch predictions."""
    model = get_model()
    
    # Build DataFrame
    X = pd.DataFrame(request.data)
    
    # Validate features
    missing = set(model.feature_names) - set(X.columns)
    if missing:
        raise HTTPException(400, f"Missing features: {missing}")
    
    # Reorder columns
    X = X[model.feature_names]
    
    predictions = model.predict(X).tolist()
    
    return BatchPredictionResponse(predictions=predictions)


@app.get("/features")
def list_features():
    """List required features for prediction."""
    model = get_model()
    return {"features": model.feature_names, "count": len(model.feature_names)}


@app.get("/importance")
def feature_importance(top_n: int = 20):
    """Get feature importance from trained model."""
    model = get_model()
    importance = model.get_feature_importance(top_n=top_n)
    return importance.to_dict(orient="records")
