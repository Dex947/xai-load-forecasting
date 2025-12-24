"""Solar irradiance data fetching from Open-Meteo API."""

import pandas as pd
import requests
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from pathlib import Path

from src.logger import get_logger

logger = get_logger(__name__)

# Open-Meteo solar radiation variables
SOLAR_VARIABLES = [
    "shortwave_radiation",          # Global Horizontal Irradiance (GHI) W/m²
    "direct_radiation",             # Direct Normal Irradiance (DNI) W/m²
    "diffuse_radiation",            # Diffuse Horizontal Irradiance (DHI) W/m²
    "direct_normal_irradiance",     # DNI on surface perpendicular to sun
    "terrestrial_radiation",        # Top of atmosphere radiation
]

SOLAR_DERIVED = [
    "sunshine_duration",            # Seconds of sunshine per hour
]


class SolarDataFetcher:
    """Fetches solar irradiance data from Open-Meteo API."""
    
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    def __init__(
        self,
        latitude: float,
        longitude: float,
        timezone: str = "UTC"
    ):
        """
        Args:
            latitude: Location latitude
            longitude: Location longitude
            timezone: Timezone for data alignment
        """
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        
        logger.info(f"SolarDataFetcher initialized: lat={latitude}, lon={longitude}")
    
    def fetch(
        self,
        start_date: str,
        end_date: str,
        variables: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch solar irradiance data for date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            variables: Solar variables to fetch (default: all)
            
        Returns:
            DataFrame with hourly solar data
        """
        if variables is None:
            variables = SOLAR_VARIABLES + SOLAR_DERIVED
        
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
            "timezone": self.timezone,
        }
        
        logger.info(f"Fetching solar data: {start_date} to {end_date}")
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch solar data: {e}")
            raise
        
        if "hourly" not in data:
            raise ValueError("No hourly data in API response")
        
        hourly = data["hourly"]
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly["time"]),
            **{var: hourly.get(var, [None] * len(hourly["time"])) 
               for var in variables if var in hourly}
        })
        
        df = df.set_index("timestamp")
        df.index = df.index.tz_localize(self.timezone) if df.index.tz is None else df.index
        
        # Rename to standard names
        rename_map = {
            "shortwave_radiation": "ghi",
            "direct_radiation": "dni", 
            "diffuse_radiation": "dhi",
            "direct_normal_irradiance": "dni_normal",
            "terrestrial_radiation": "toa_radiation",
            "sunshine_duration": "sunshine_seconds",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        logger.info(f"Fetched {len(df)} hours of solar data with {len(df.columns)} variables")
        return df
    
    def create_solar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived solar features.
        
        Args:
            df: DataFrame with solar irradiance columns
            
        Returns:
            DataFrame with additional solar features
        """
        features = pd.DataFrame(index=df.index)
        
        # Direct/diffuse ratio (clearness indicator)
        if "dni" in df.columns and "dhi" in df.columns:
            features["solar_direct_ratio"] = df["dni"] / (df["dni"] + df["dhi"] + 1e-6)
        
        # Clearness index (GHI / TOA radiation)
        if "ghi" in df.columns and "toa_radiation" in df.columns:
            features["clearness_index"] = df["ghi"] / (df["toa_radiation"] + 1e-6)
            features["clearness_index"] = features["clearness_index"].clip(0, 1)
        
        # Solar availability (binary: significant radiation)
        if "ghi" in df.columns:
            features["solar_available"] = (df["ghi"] > 50).astype(int)
            features["ghi_normalized"] = df["ghi"] / 1000  # Normalize to 0-1 range
        
        # Sunshine fraction (if sunshine_seconds available)
        if "sunshine_seconds" in df.columns:
            features["sunshine_fraction"] = df["sunshine_seconds"] / 3600
        
        # Rolling solar features
        if "ghi" in df.columns:
            features["ghi_rolling_3h"] = df["ghi"].rolling(3, min_periods=1).mean()
            features["ghi_rolling_6h"] = df["ghi"].rolling(6, min_periods=1).mean()
            features["ghi_change_1h"] = df["ghi"].diff(1)
        
        logger.info(f"Created {len(features.columns)} solar features")
        return features


def fetch_solar_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str = "UTC",
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to fetch and optionally save solar data.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timezone: Timezone
        save_path: Optional path to save data
        
    Returns:
        DataFrame with solar irradiance data
    """
    fetcher = SolarDataFetcher(latitude, longitude, timezone)
    df = fetcher.fetch(start_date, end_date)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path)
        logger.info(f"Solar data saved to {save_path}")
    
    return df
