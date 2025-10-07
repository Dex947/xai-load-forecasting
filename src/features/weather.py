"""
Weather Feature Engineering
============================

Creates weather-based features and derived weather metrics.
Handles temperature-based degree days, heat index, and interactions.

Usage:
    from src.features.weather import WeatherFeatureEngineer
    
    engineer = WeatherFeatureEngineer(config)
    features = engineer.create_features(df)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from src.logger import get_logger

logger = get_logger(__name__)


class WeatherFeatureEngineer:
    """
    Creates weather-based features and derived metrics.
    """
    
    def __init__(self, weather_config: Optional[Dict] = None):
        """
        Initialize weather feature engineer.
        
        Args:
            weather_config: Weather feature configuration
        """
        self.weather_config = weather_config or {}
        self.derived_config = self.weather_config.get('derived_features', {})
        logger.info("Weather feature engineer initialized")
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived weather features.
        
        Args:
            df: DataFrame with base weather features
        
        Returns:
            DataFrame with derived weather features
        """
        logger.info("Creating derived weather features")
        
        features_df = pd.DataFrame(index=df.index)
        
        # Heating Degree Days (HDD)
        if self.derived_config.get('heating_degree_days', {}).get('enabled', True):
            base_temp = self.derived_config.get('heating_degree_days', {}).get('base_temp_c', 18.3)
            if 'temperature' in df.columns:
                features_df['hdd'] = np.maximum(base_temp - df['temperature'], 0)
                logger.debug(f"Created heating_degree_days (base: {base_temp}°C)")
        
        # Cooling Degree Days (CDD)
        if self.derived_config.get('cooling_degree_days', {}).get('enabled', True):
            base_temp = self.derived_config.get('cooling_degree_days', {}).get('base_temp_c', 18.3)
            if 'temperature' in df.columns:
                features_df['cdd'] = np.maximum(df['temperature'] - base_temp, 0)
                logger.debug(f"Created cooling_degree_days (base: {base_temp}°C)")
        
        # Heat Index (apparent temperature from temp + humidity)
        if self.derived_config.get('heat_index', {}).get('enabled', True):
            if 'temperature' in df.columns and 'humidity' in df.columns:
                features_df['heat_index'] = self._calculate_heat_index(
                    df['temperature'],
                    df['humidity']
                )
                logger.debug("Created heat_index")
        
        # Discomfort Index
        if self.derived_config.get('discomfort_index', {}).get('enabled', True):
            if 'temperature' in df.columns and 'humidity' in df.columns:
                features_df['discomfort_index'] = self._calculate_discomfort_index(
                    df['temperature'],
                    df['humidity']
                )
                logger.debug("Created discomfort_index")
        
        # Temperature range (requires daily aggregation)
        if self.derived_config.get('temperature_range', {}).get('enabled', True):
            if 'temperature' in df.columns:
                features_df['temp_range_24h'] = df['temperature'].rolling(
                    window=24,
                    min_periods=1
                ).apply(lambda x: x.max() - x.min())
                logger.debug("Created temperature_range_24h")
        
        # Wind chill (for cold weather)
        if self.derived_config.get('wind_chill', {}).get('enabled', False):
            if 'temperature' in df.columns and 'wind_speed' in df.columns:
                features_df['wind_chill'] = self._calculate_wind_chill(
                    df['temperature'],
                    df['wind_speed']
                )
                logger.debug("Created wind_chill")
        
        logger.info(f"Created {len(features_df.columns)} derived weather features")
        
        return features_df
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        weather_columns: List[str],
        lag_hours: List[int]
    ) -> pd.DataFrame:
        """
        Create lagged weather features.
        
        Args:
            df: DataFrame with weather columns
            weather_columns: List of weather column names
            lag_hours: List of lag hours
        
        Returns:
            DataFrame with lagged weather features
        """
        logger.info(f"Creating lagged weather features for {len(weather_columns)} columns")
        
        lag_df = pd.DataFrame(index=df.index)
        
        for col in weather_columns:
            if col not in df.columns:
                logger.warning(f"Weather column '{col}' not found, skipping")
                continue
            
            for lag in lag_hours:
                lag_col_name = f"{col}_lag_{lag}h"
                lag_df[lag_col_name] = df[col].shift(lag)
        
        logger.info(f"Created {len(lag_df.columns)} lagged weather features")
        
        return lag_df
    
    def create_interaction_features(
        self,
        df: pd.DataFrame,
        interactions_config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Create interaction features between weather and temporal variables.
        
        Args:
            df: DataFrame with weather and temporal features
            interactions_config: Interaction feature configuration
        
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features")
        
        interactions_config = interactions_config or {}
        features_df = pd.DataFrame(index=df.index)
        
        # Temperature × Hour
        if interactions_config.get('temp_hour', True):
            if 'temperature' in df.columns and 'hour' in df.columns:
                features_df['temp_x_hour'] = df['temperature'] * df['hour']
                logger.debug("Created temp_x_hour interaction")
        
        # Temperature × Weekend
        if interactions_config.get('temp_weekend', True):
            if 'temperature' in df.columns and 'is_weekend' in df.columns:
                features_df['temp_x_weekend'] = df['temperature'] * df['is_weekend']
                logger.debug("Created temp_x_weekend interaction")
        
        # Humidity × Temperature
        if interactions_config.get('humidity_temp', True):
            if 'humidity' in df.columns and 'temperature' in df.columns:
                features_df['humidity_x_temp'] = df['humidity'] * df['temperature']
                logger.debug("Created humidity_x_temp interaction")
        
        logger.info(f"Created {len(features_df.columns)} interaction features")
        
        return features_df
    
    @staticmethod
    def _calculate_heat_index(temp_c: pd.Series, humidity: pd.Series) -> pd.Series:
        """
        Calculate heat index (apparent temperature).
        
        Simplified formula for Celsius.
        
        Args:
            temp_c: Temperature in Celsius
            humidity: Relative humidity (0-100)
        
        Returns:
            Heat index in Celsius
        """
        # Convert to Fahrenheit for calculation
        temp_f = temp_c * 9/5 + 32
        
        # Rothfusz regression (valid for temp > 80°F)
        hi_f = (
            -42.379 +
            2.04901523 * temp_f +
            10.14333127 * humidity -
            0.22475541 * temp_f * humidity -
            6.83783e-3 * temp_f**2 -
            5.481717e-2 * humidity**2 +
            1.22874e-3 * temp_f**2 * humidity +
            8.5282e-4 * temp_f * humidity**2 -
            1.99e-6 * temp_f**2 * humidity**2
        )
        
        # Convert back to Celsius
        hi_c = (hi_f - 32) * 5/9
        
        # Use actual temperature if below threshold
        hi_c = np.where(temp_c < 27, temp_c, hi_c)
        
        return hi_c
    
    @staticmethod
    def _calculate_discomfort_index(temp_c: pd.Series, humidity: pd.Series) -> pd.Series:
        """
        Calculate discomfort index (Thom's formula).
        
        Args:
            temp_c: Temperature in Celsius
            humidity: Relative humidity (0-100)
        
        Returns:
            Discomfort index
        """
        di = temp_c - 0.55 * (1 - humidity / 100) * (temp_c - 14.5)
        return di
    
    @staticmethod
    def _calculate_wind_chill(temp_c: pd.Series, wind_speed_ms: pd.Series) -> pd.Series:
        """
        Calculate wind chill temperature.
        
        Valid for temperatures ≤ 10°C and wind speeds > 4.8 km/h.
        
        Args:
            temp_c: Temperature in Celsius
            wind_speed_ms: Wind speed in m/s
        
        Returns:
            Wind chill temperature in Celsius
        """
        # Convert wind speed to km/h
        wind_speed_kmh = wind_speed_ms * 3.6
        
        # Wind chill formula (Environment Canada)
        wc = (
            13.12 +
            0.6215 * temp_c -
            11.37 * wind_speed_kmh**0.16 +
            0.3965 * temp_c * wind_speed_kmh**0.16
        )
        
        # Only apply for cold temperatures and sufficient wind
        wc = np.where((temp_c <= 10) & (wind_speed_kmh > 4.8), wc, temp_c)
        
        return wc
