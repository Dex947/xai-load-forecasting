"""Tests for feature engineering modules."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.temporal import TemporalFeatureEngineer
from src.features.weather import WeatherFeatureEngineer


class TestTemporalFeatureEngineer:
    """Tests for temporal feature creation."""

    def test_creates_hour_feature(self, sample_load_df):
        """Verify hour extraction from index."""
        engineer = TemporalFeatureEngineer()
        features = engineer.create_features(sample_load_df)
        
        assert "hour" in features.columns
        assert features["hour"].min() >= 0
        assert features["hour"].max() <= 23

    def test_creates_cyclical_encoding(self, sample_load_df):
        """Verify sin/cos cyclical features."""
        engineer = TemporalFeatureEngineer()
        features = engineer.create_features(sample_load_df)
        
        assert "hour_sin" in features.columns
        assert "hour_cos" in features.columns
        assert features["hour_sin"].between(-1, 1).all()
        assert features["hour_cos"].between(-1, 1).all()

    def test_lag_features_no_leakage(self, sample_load_df):
        """Verify lag features don't leak future data."""
        engineer = TemporalFeatureEngineer()
        lags = engineer.create_lag_features(
            sample_load_df, 
            target_column="load",
            lag_hours=[1, 24]
        )
        
        # First row should have NaN for lag_1h
        assert pd.isna(lags.iloc[0]["lag_1h"])
        # Row 24 should have valid lag_24h
        assert not pd.isna(lags.iloc[24]["lag_24h"])

    def test_rolling_features_shifted(self, sample_load_df):
        """Verify rolling features use shift(1) to prevent leakage."""
        engineer = TemporalFeatureEngineer()
        rolling = engineer.create_rolling_features(
            sample_load_df,
            target_column="load",
            windows=[3],
            functions=["mean"]
        )
        
        # Rolling mean at t should not include value at t
        # Due to shift(1), first value should be NaN
        assert pd.isna(rolling.iloc[0]["rolling_3h_mean"])


class TestWeatherFeatureEngineer:
    """Tests for weather feature creation."""

    def test_creates_hdd_cdd(self, merged_df):
        """Verify heating/cooling degree days."""
        engineer = WeatherFeatureEngineer()
        derived = engineer.create_derived_features(merged_df)
        
        assert "hdd" in derived.columns
        assert "cdd" in derived.columns
        # HDD and CDD should be non-negative
        assert (derived["hdd"] >= 0).all()
        assert (derived["cdd"] >= 0).all()

    def test_creates_heat_index(self, merged_df):
        """Verify heat index calculation."""
        engineer = WeatherFeatureEngineer()
        derived = engineer.create_derived_features(merged_df)
        
        assert "heat_index" in derived.columns

    def test_interaction_features(self, merged_df):
        """Verify interaction feature creation."""
        merged_df["hour"] = merged_df.index.hour
        merged_df["is_weekend"] = (merged_df.index.dayofweek >= 5).astype(int)
        
        engineer = WeatherFeatureEngineer()
        interactions = engineer.create_interaction_features(merged_df)
        
        assert "temp_x_hour" in interactions.columns
        assert "temp_x_weekend" in interactions.columns
