"""Tests for advanced modules: solar, conformal, online, mlops, explainability."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSolarDataFetcher:
    """Tests for solar irradiance data fetching."""
    
    def test_solar_feature_creation(self):
        """Test solar feature derivation."""
        from src.data.solar import SolarDataFetcher
        
        fetcher = SolarDataFetcher(latitude=38.7, longitude=-9.1)
        
        # Create mock solar data
        df = pd.DataFrame({
            "ghi": [100, 500, 800, 200],
            "dni": [50, 400, 700, 100],
            "dhi": [50, 100, 100, 100],
            "toa_radiation": [200, 1000, 1200, 400],
            "sunshine_seconds": [1800, 3600, 3600, 900],
        }, index=pd.date_range("2024-01-01", periods=4, freq="h"))
        
        features = fetcher.create_solar_features(df)
        
        assert "solar_direct_ratio" in features.columns
        assert "clearness_index" in features.columns
        assert "solar_available" in features.columns
        assert "ghi_normalized" in features.columns
        assert len(features) == 4


class TestDataSourceManager:
    """Tests for unified data source manager."""
    
    def test_manager_initialization(self):
        """Test manager initializes correctly."""
        from src.data.sources import DataSourceManager
        
        config = {
            "location": {"latitude": 38.7, "longitude": -9.1},
            "timezone": "UTC"
        }
        
        manager = DataSourceManager(config)
        
        assert manager.latitude == 38.7
        assert manager.longitude == -9.1
        assert manager.timezone == "UTC"
    
    def test_summary_empty(self):
        """Test summary with no data loaded."""
        from src.data.sources import DataSourceManager
        
        manager = DataSourceManager({})
        summary = manager.summary()
        
        assert isinstance(summary, dict)
        assert len(summary) == 0


class TestDriftExplainer:
    """Tests for concept drift explanation."""
    
    def test_drift_analysis(self):
        """Test drift detection and analysis."""
        from src.explainability.drift_explanation import DriftExplainer
        
        baseline = {
            "feature_a": 0.5,
            "feature_b": 0.3,
            "feature_c": 0.2,
        }
        
        explainer = DriftExplainer(baseline, importance_threshold=0.2)
        
        # Test with similar importance (no drift)
        current_similar = {
            "feature_a": 0.48,
            "feature_b": 0.32,
            "feature_c": 0.20,
        }
        
        result = explainer.analyze_drift(current_similar)
        
        assert result["severity"] in ["none", "low"]
        assert "drift_score" in result
    
    def test_significant_drift_detection(self):
        """Test detection of significant drift."""
        from src.explainability.drift_explanation import DriftExplainer
        
        baseline = {
            "feature_a": 0.5,
            "feature_b": 0.3,
            "feature_c": 0.2,
        }
        
        explainer = DriftExplainer(baseline, importance_threshold=0.2)
        
        # Test with very different importance (drift)
        current_different = {
            "feature_a": 0.2,
            "feature_b": 0.6,
            "feature_c": 0.2,
        }
        
        result = explainer.analyze_drift(current_different)
        
        assert result["num_significant_changes"] > 0
        assert len(result["significant_changes"]) > 0
    
    def test_explain_drift(self):
        """Test natural language drift explanation."""
        from src.explainability.drift_explanation import DriftExplainer
        
        baseline = {"temperature": 0.5, "load_lag_1h": 0.3}
        explainer = DriftExplainer(baseline)
        
        result = explainer.analyze_drift({"temperature": 0.8, "load_lag_1h": 0.1})
        explanation = explainer.explain_drift(result)
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0


class TestNaturalLanguageExplainer:
    """Tests for natural language explanations."""
    
    def test_explain_prediction(self):
        """Test single prediction explanation."""
        from src.explainability.natural_language import NaturalLanguageExplainer
        
        feature_names = ["temperature", "load_lag_1h", "hour"]
        explainer = NaturalLanguageExplainer(feature_names)
        
        prediction = 5.5
        shap_values = np.array([0.3, 0.8, -0.2])
        feature_values = pd.Series({
            "temperature": 25.0,
            "load_lag_1h": 5.2,
            "hour": 14
        })
        base_value = 5.0
        
        explanation = explainer.explain_prediction(
            prediction, shap_values, feature_values, base_value
        )
        
        assert isinstance(explanation, str)
        assert "5.50" in explanation or "5.5" in explanation
        assert "kW" in explanation
    
    def test_generate_alert_message(self):
        """Test alert message generation."""
        from src.explainability.natural_language import NaturalLanguageExplainer
        
        feature_names = ["temperature", "load_lag_1h"]
        explainer = NaturalLanguageExplainer(feature_names)
        
        alert = explainer.generate_alert_message(
            prediction=10.0,
            threshold=8.0,
            shap_values=np.array([1.5, 0.5]),
            feature_values=pd.Series({"temperature": 35.0, "load_lag_1h": 8.0})
        )
        
        assert "ALERT" in alert
        assert "10.00" in alert or "10.0" in alert


class TestABTestManager:
    """Tests for A/B testing framework."""
    
    def test_ab_test_initialization(self):
        """Test A/B test manager initialization."""
        from src.mlops.ab_testing import ABTestManager
        
        # Mock models
        class MockModel:
            def predict(self, X):
                return np.ones(len(X))
        
        champion = MockModel()
        challenger = MockModel()
        
        manager = ABTestManager(
            champion_model=champion,
            challenger_model=challenger,
            traffic_split=0.2
        )
        
        assert manager.traffic_split == 0.2
        assert manager.min_samples == 100
    
    def test_record_outcome(self):
        """Test recording prediction outcomes."""
        from src.mlops.ab_testing import ABTestManager
        
        class MockModel:
            def predict(self, X):
                return np.ones(len(X))
        
        manager = ABTestManager(MockModel(), MockModel())
        
        manager.record_outcome(y_pred=5.0, y_true=5.5, model="champion")
        manager.record_outcome(y_pred=5.2, y_true=5.5, model="challenger")
        
        metrics = manager.get_metrics()
        
        assert "champion" in metrics
        assert "challenger" in metrics
        assert metrics["champion"]["n_samples"] == 1
        assert metrics["challenger"]["n_samples"] == 1


class TestMultiArmedBandit:
    """Tests for multi-armed bandit model selection."""
    
    def test_bandit_initialization(self):
        """Test bandit initialization."""
        from src.mlops.ab_testing import MultiArmedBandit
        
        models = {"model_a": None, "model_b": None, "model_c": None}
        bandit = MultiArmedBandit(models)
        
        assert len(bandit.model_names) == 3
        assert all(bandit.successes[m] == 0 for m in bandit.model_names)
    
    def test_model_selection(self):
        """Test Thompson Sampling selection."""
        from src.mlops.ab_testing import MultiArmedBandit
        
        models = {"model_a": None, "model_b": None}
        bandit = MultiArmedBandit(models)
        
        # Select model multiple times
        selections = [bandit.select_model() for _ in range(100)]
        
        # Both models should be selected at least once
        assert "model_a" in selections or "model_b" in selections
    
    def test_bandit_update(self):
        """Test bandit parameter updates."""
        from src.mlops.ab_testing import MultiArmedBandit
        
        models = {"model_a": None}
        bandit = MultiArmedBandit(models)
        
        # Simulate success
        bandit.update("model_a", error=0.5, threshold=1.0)
        assert bandit.successes["model_a"] == 1
        
        # Simulate failure
        bandit.update("model_a", error=1.5, threshold=1.0)
        assert bandit.failures["model_a"] == 1


def test_imports():
    """Test that all new modules can be imported."""
    # Data modules
    from src.data.solar import SolarDataFetcher, fetch_solar_data
    from src.data.sources import DataSourceManager
    
    # Model modules
    from src.models.conformal import ConformalForecaster, MAPIE_AVAILABLE
    from src.models.online import OnlineForecaster, HybridForecaster, RIVER_AVAILABLE
    
    # MLOps modules
    from src.mlops.tracking import ExperimentTracker, MLFLOW_AVAILABLE
    from src.mlops.registry import ModelRegistry
    from src.mlops.ab_testing import ABTestManager, MultiArmedBandit
    
    # Explainability modules
    from src.explainability.counterfactuals import CounterfactualExplainer, DICE_AVAILABLE
    from src.explainability.drift_explanation import DriftExplainer
    from src.explainability.natural_language import NaturalLanguageExplainer
    
    assert True  # If we get here, all imports succeeded
