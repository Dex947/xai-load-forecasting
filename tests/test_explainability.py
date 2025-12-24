"""Tests for explainability modules."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gbm import GradientBoostingModel
from src.explainability.shap_analysis import SHAPAnalyzer


class TestSHAPAnalyzer:
    """Tests for SHAP analysis."""

    @pytest.fixture
    def trained_model(self, train_test_split):
        """Train a model for SHAP testing."""
        X_train, X_test, y_train, y_test = train_test_split
        model = GradientBoostingModel(model_type="lightgbm")
        model.fit(X_train, y_train)
        return model, X_train, X_test

    def test_computes_shap_values(self, trained_model):
        """Verify SHAP values are computed."""
        model, X_train, X_test = trained_model
        
        analyzer = SHAPAnalyzer(
            model.model,
            X_background=X_train.sample(50, random_state=42)
        )
        shap_values = analyzer.compute_shap_values(X_test.head(20))
        
        assert shap_values is not None
        assert shap_values.shape[0] == 20
        assert shap_values.shape[1] == len(X_test.columns)

    def test_global_importance(self, trained_model):
        """Verify global importance extraction."""
        model, X_train, X_test = trained_model
        
        analyzer = SHAPAnalyzer(
            model.model,
            X_background=X_train.sample(50, random_state=42)
        )
        analyzer.compute_shap_values(X_test.head(20))
        importance = analyzer.get_global_importance()
        
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert (importance["importance"] >= 0).all()

    def test_save_and_load_shap(self, trained_model, tmp_path):
        """Verify SHAP value serialization."""
        model, X_train, X_test = trained_model
        shap_path = tmp_path / "shap.pkl"
        
        analyzer = SHAPAnalyzer(
            model.model,
            X_background=X_train.sample(50, random_state=42)
        )
        analyzer.compute_shap_values(X_test.head(20))
        analyzer.save_shap_values(str(shap_path))
        
        loaded = SHAPAnalyzer.load_shap_values(str(shap_path))
        
        assert "shap_values" in loaded
        assert "feature_names" in loaded
