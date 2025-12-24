"""Tests for data loading module."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_load_data, load_weather_data, merge_load_weather


class TestLoadLoadData:
    """Tests for load_load_data function."""

    def test_loads_csv_correctly(self, tmp_path):
        """Verify CSV loading with proper datetime parsing."""
        csv_file = tmp_path / "load.csv"
        dates = pd.date_range("2023-01-01", periods=24, freq="h")
        df = pd.DataFrame({
            "timestamp": dates,
            "load": np.random.rand(24) * 10
        })
        df.to_csv(csv_file, index=False)
        
        result = load_load_data(str(csv_file))
        
        assert isinstance(result.index, pd.DatetimeIndex)
        assert "load" in result.columns
        assert len(result) == 24

    def test_raises_on_missing_file(self):
        """Verify FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_load_data("nonexistent.csv")

    def test_raises_on_missing_columns(self, tmp_path):
        """Verify ValueError when required columns missing."""
        csv_file = tmp_path / "bad.csv"
        pd.DataFrame({"wrong_col": [1, 2, 3]}).to_csv(csv_file, index=False)
        
        with pytest.raises(ValueError):
            load_load_data(str(csv_file))


class TestMergeLoadWeather:
    """Tests for merge_load_weather function."""

    def test_merges_on_index(self, sample_load_df, sample_weather_df):
        """Verify proper index-based merge."""
        result = merge_load_weather(sample_load_df, sample_weather_df)
        
        assert "load" in result.columns
        assert "temperature" in result.columns
        assert len(result) == len(sample_load_df)

    def test_raises_on_no_overlap(self):
        """Verify ValueError when no temporal overlap."""
        load_df = pd.DataFrame(
            {"load": [1, 2, 3]},
            index=pd.date_range("2020-01-01", periods=3, freq="h", tz="UTC")
        )
        weather_df = pd.DataFrame(
            {"temperature": [10, 11, 12]},
            index=pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
        )
        
        with pytest.raises(ValueError, match="No overlap"):
            merge_load_weather(load_df, weather_df)
