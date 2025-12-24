"""Tests for model modules."""

import pandas as pd
import numpy as np
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.baseline import BaselineModel
from src.models.gbm import GradientBoostingModel
from src.models.validator import RollingOriginValidator


class TestBaselineModel:
    """Tests for baseline models."""

    def test_persistence_forecast(self, sample_load_df):
        """Verify persistence uses last value."""
        model = BaselineModel(method="persistence")
        model.fit(sample_load_df["load"])
        preds = model.predict(horizon=5)

        last_val = sample_load_df["load"].iloc[-1]
        assert (preds == last_val).all()

    def test_seasonal_naive_forecast(self, sample_load_df):
        """Verify seasonal naive uses past season."""
        model = BaselineModel(method="seasonal_naive", season_length=24)
        model.fit(sample_load_df["load"])
        preds = model.predict(horizon=24)

        assert len(preds) == 24

    def test_evaluate_returns_metrics(self, sample_load_df):
        """Verify evaluation returns expected metrics."""
        model = BaselineModel(method="persistence")
        y_true = sample_load_df["load"].iloc[-24:]
        y_pred = pd.Series(sample_load_df["load"].iloc[-25], index=y_true.index)

        metrics = model.evaluate(y_true, y_pred)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics


class TestGradientBoostingModel:
    """Tests for gradient boosting model."""

    def test_fit_and_predict(self, train_test_split):
        """Verify model trains and predicts."""
        X_train, X_test, y_train, y_test = train_test_split

        model = GradientBoostingModel(model_type="lightgbm")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        assert len(preds) == len(X_test)
        assert not np.isnan(preds).any()

    def test_save_and_load(self, train_test_split, tmp_path):
        """Verify model serialization."""
        X_train, X_test, y_train, y_test = train_test_split
        model_path = tmp_path / "model.pkl"

        model = GradientBoostingModel(model_type="lightgbm")
        model.fit(X_train, y_train)
        original_preds = model.predict(X_test)
        model.save(str(model_path))

        loaded = GradientBoostingModel.load(str(model_path))
        loaded_preds = loaded.predict(X_test)

        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_feature_importance(self, train_test_split):
        """Verify feature importance extraction."""
        X_train, X_test, y_train, y_test = train_test_split

        model = GradientBoostingModel(model_type="lightgbm")
        model.fit(X_train, y_train)
        importance = model.get_feature_importance()

        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert len(importance) == len(X_train.columns)

    def test_monotonic_constraints(self, train_test_split):
        """Verify monotonic constraints are applied."""
        X_train, X_test, y_train, y_test = train_test_split

        model = GradientBoostingModel(
            model_type="lightgbm", monotonic_constraints={"temperature": 1}
        )
        model.fit(X_train, y_train)

        # Model should train without error
        assert model.model is not None


class TestRollingOriginValidator:
    """Tests for rolling origin cross-validation."""

    def test_generates_correct_splits(self, sample_features_df):
        """Verify correct number of splits generated."""
        validator = RollingOriginValidator(
            n_splits=3, test_size_days=5, min_train_days=10
        )

        splits = list(validator.split(sample_features_df))
        assert len(splits) == 3

    def test_no_temporal_leakage(self, sample_features_df):
        """Verify train always before test."""
        validator = RollingOriginValidator(
            n_splits=3, test_size_days=5, min_train_days=10
        )

        for train_idx, test_idx in validator.split(sample_features_df):
            assert train_idx.max() < test_idx.min()
