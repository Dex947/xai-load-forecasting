"""Unified data source manager for multiple data types."""

import pandas as pd
from typing import Dict, Optional, List, Any
from pathlib import Path
from datetime import datetime

from src.logger import get_logger
from src.data.loader import load_load_data, load_weather_data, merge_load_weather
from src.data.solar import SolarDataFetcher

logger = get_logger(__name__)


class DataSourceManager:
    """Manages multiple data sources and combines them into unified dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dict with data source settings
        """
        self.config = config
        self.data_sources: Dict[str, pd.DataFrame] = {}
        
        # Location for API calls
        self.latitude = config.get("location", {}).get("latitude", 38.7223)
        self.longitude = config.get("location", {}).get("longitude", -9.1393)
        self.timezone = config.get("timezone", "UTC")
        
        logger.info("DataSourceManager initialized")
    
    def load_load_data(self, path: str, **kwargs) -> pd.DataFrame:
        """Load electrical load data."""
        df = load_load_data(path, **kwargs)
        self.data_sources["load"] = df
        logger.info(f"Loaded load data: {len(df)} records")
        return df
    
    def load_weather_data(self, path: str, **kwargs) -> pd.DataFrame:
        """Load weather data from file."""
        df = load_weather_data(path, **kwargs)
        self.data_sources["weather"] = df
        logger.info(f"Loaded weather data: {len(df)} records")
        return df
    
    def fetch_solar_data(
        self,
        start_date: str,
        end_date: str,
        cache_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch solar irradiance data from API or cache.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cache_path: Optional path to cache/load data
            
        Returns:
            DataFrame with solar data
        """
        # Check cache first
        if cache_path and Path(cache_path).exists():
            df = pd.read_parquet(cache_path)
            logger.info(f"Loaded solar data from cache: {cache_path}")
            self.data_sources["solar"] = df
            return df
        
        # Fetch from API
        fetcher = SolarDataFetcher(self.latitude, self.longitude, self.timezone)
        df = fetcher.fetch(start_date, end_date)
        
        # Cache if path provided
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)
            logger.info(f"Cached solar data to: {cache_path}")
        
        self.data_sources["solar"] = df
        return df
    
    def load_grid_events(self, path: str) -> pd.DataFrame:
        """
        Load grid events (outages, topology changes) from CSV.
        
        Expected columns: timestamp, event_type, duration_hours, affected_load
        """
        if not Path(path).exists():
            logger.warning(f"Grid events file not found: {path}")
            return pd.DataFrame()
        
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df.set_index("timestamp")
        
        self.data_sources["grid_events"] = df
        logger.info(f"Loaded grid events: {len(df)} records")
        return df
    
    def merge_all_sources(
        self,
        base_source: str = "load",
        include_solar: bool = True,
        include_grid_events: bool = False
    ) -> pd.DataFrame:
        """
        Merge all loaded data sources into unified dataset.
        
        Args:
            base_source: Primary data source to use as base
            include_solar: Include solar irradiance data
            include_grid_events: Include grid event flags
            
        Returns:
            Merged DataFrame with all data sources
        """
        if base_source not in self.data_sources:
            raise ValueError(f"Base source '{base_source}' not loaded")
        
        df = self.data_sources[base_source].copy()
        
        # Merge weather if available
        if "weather" in self.data_sources:
            weather = self.data_sources["weather"]
            df = merge_load_weather(df, weather)
            logger.info("Merged weather data")
        
        # Merge solar if available and requested
        if include_solar and "solar" in self.data_sources:
            solar = self.data_sources["solar"]
            # Align indices
            common_idx = df.index.intersection(solar.index)
            df = df.loc[common_idx]
            for col in solar.columns:
                if col not in df.columns:
                    df[col] = solar.loc[common_idx, col]
            logger.info(f"Merged solar data: {len(solar.columns)} columns")
        
        # Add grid event flags if available and requested
        if include_grid_events and "grid_events" in self.data_sources:
            events = self.data_sources["grid_events"]
            df["has_grid_event"] = df.index.isin(events.index).astype(int)
            logger.info("Added grid event flags")
        
        logger.info(f"Final merged dataset: {df.shape}")
        return df
    
    def get_date_range(self) -> tuple:
        """Get common date range across all loaded sources."""
        if not self.data_sources:
            return None, None
        
        start_dates = []
        end_dates = []
        
        for name, df in self.data_sources.items():
            if hasattr(df.index, "min"):
                start_dates.append(df.index.min())
                end_dates.append(df.index.max())
        
        if not start_dates:
            return None, None
        
        return max(start_dates), min(end_dates)
    
    def summary(self) -> Dict[str, Dict]:
        """Get summary of all loaded data sources."""
        summary = {}
        for name, df in self.data_sources.items():
            summary[name] = {
                "rows": len(df),
                "columns": len(df.columns),
                "start": str(df.index.min()) if len(df) > 0 else None,
                "end": str(df.index.max()) if len(df) > 0 else None,
                "missing_pct": (df.isna().sum().sum() / df.size * 100) if df.size > 0 else 0,
            }
        return summary
