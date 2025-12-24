"""Pytest fixtures for XAI Load Forecasting tests."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_load_df():
    """Create sample load data for testing."""
    dates = pd.date_range(
        start="2023-01-01",
        periods=720,  # 30 days
        freq="h",
        tz="UTC"
    )
    np.random.seed(42)
    load = 5 + 3 * np.sin(np.arange(720) * 2 * np.pi / 24) + np.random.normal(0, 0.5, 720)
    return pd.DataFrame({"load": load}, index=dates)


@pytest.fixture
def sample_weather_df():
    """Create sample weather data for testing."""
    dates = pd.date_range(
        start="2023-01-01",
        periods=720,
        freq="h",
        tz="UTC"
    )
    np.random.seed(42)
    return pd.DataFrame({
        "temperature": 15 + 10 * np.sin(np.arange(720) * 2 * np.pi / 24) + np.random.normal(0, 2, 720),
        "humidity": 60 + np.random.normal(0, 10, 720),
        "wind_speed": 5 + np.random.exponential(2, 720),
        "precipitation": np.random.exponential(0.1, 720),
        "pressure": 1013 + np.random.normal(0, 5, 720),
        "cloud_cover": np.clip(50 + np.random.normal(0, 20, 720), 0, 100),
    }, index=dates)


@pytest.fixture
def merged_df(sample_load_df, sample_weather_df):
    """Merged load and weather data."""
    return sample_load_df.join(sample_weather_df)


@pytest.fixture
def sample_features_df(merged_df):
    """Create sample features for model testing."""
    df = merged_df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["load_lag_1h"] = df["load"].shift(1)
    df["load_lag_24h"] = df["load"].shift(24)
    df["rolling_24h_mean"] = df["load"].shift(1).rolling(24, min_periods=1).mean()
    return df.dropna()


@pytest.fixture
def train_test_split(sample_features_df):
    """Split features into train/test sets."""
    split_idx = int(len(sample_features_df) * 0.8)
    train = sample_features_df.iloc[:split_idx]
    test = sample_features_df.iloc[split_idx:]
    
    feature_cols = [c for c in train.columns if c != "load"]
    X_train = train[feature_cols]
    y_train = train["load"]
    X_test = test[feature_cols]
    y_test = test["load"]
    
    return X_train, X_test, y_train, y_test
