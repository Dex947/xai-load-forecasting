"""
Temporal Feature Engineering
=============================

Creates temporal features from datetime index.
All features respect temporal boundaries to prevent data leakage.

Usage:
    from src.features.temporal import TemporalFeatureEngineer
    
    engineer = TemporalFeatureEngineer(config)
    features = engineer.create_features(df)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

from src.logger import get_logger

logger = get_logger(__name__)


class TemporalFeatureEngineer:
    """
    Creates temporal features from datetime index.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize temporal feature engineer.
        
        Args:
            config: Configuration dictionary with feature flags
        """
        self.config = config or {}
        self.temporal_config = self.config.get('temporal', {})
        logger.info("Temporal feature engineer initialized")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all enabled temporal features.
        
        Args:
            df: DataFrame with DatetimeIndex
        
        Returns:
            DataFrame with temporal features
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        logger.info("Creating temporal features")
        
        features_df = pd.DataFrame(index=df.index)
        
        # Hour of day
        if self.temporal_config.get('hour_of_day', True):
            features_df['hour'] = df.index.hour
            logger.debug("Created hour_of_day feature")
        
        # Day of week (0=Monday, 6=Sunday)
        if self.temporal_config.get('day_of_week', True):
            features_df['day_of_week'] = df.index.dayofweek
            logger.debug("Created day_of_week feature")
        
        # Day of month
        if self.temporal_config.get('day_of_month', True):
            features_df['day_of_month'] = df.index.day
            logger.debug("Created day_of_month feature")
        
        # Day of year
        if self.temporal_config.get('day_of_year', True):
            features_df['day_of_year'] = df.index.dayofyear
            logger.debug("Created day_of_year feature")
        
        # Week of year
        if self.temporal_config.get('week_of_year', True):
            features_df['week_of_year'] = df.index.isocalendar().week
            logger.debug("Created week_of_year feature")
        
        # Month
        if self.temporal_config.get('month', True):
            features_df['month'] = df.index.month
            logger.debug("Created month feature")
        
        # Quarter
        if self.temporal_config.get('quarter', True):
            features_df['quarter'] = df.index.quarter
            logger.debug("Created quarter feature")
        
        # Is weekend
        if self.temporal_config.get('is_weekend', True):
            features_df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            logger.debug("Created is_weekend feature")
        
        # Is business hour (9 AM - 5 PM on weekdays)
        if self.temporal_config.get('is_business_hour', True):
            features_df['is_business_hour'] = (
                (df.index.hour >= 9) & 
                (df.index.hour < 17) & 
                (df.index.dayofweek < 5)
            ).astype(int)
            logger.debug("Created is_business_hour feature")
        
        # Season (meteorological seasons)
        if self.temporal_config.get('season', True):
            features_df['season'] = df.index.month.map({
                12: 0, 1: 0, 2: 0,  # Winter
                3: 1, 4: 1, 5: 1,   # Spring
                6: 2, 7: 2, 8: 2,   # Summer
                9: 3, 10: 3, 11: 3  # Fall
            })
            logger.debug("Created season feature")
        
        # Cyclical encoding for hour (sin/cos)
        if self.temporal_config.get('hour_of_day', True):
            features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
            features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
            logger.debug("Created cyclical hour features")
        
        # Cyclical encoding for day of week
        if self.temporal_config.get('day_of_week', True):
            features_df['day_of_week_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
            features_df['day_of_week_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
            logger.debug("Created cyclical day_of_week features")
        
        # Cyclical encoding for month
        if self.temporal_config.get('month', True):
            features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
            features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
            logger.debug("Created cyclical month features")
        
        logger.info(f"Created {len(features_df.columns)} temporal features")
        
        return features_df
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        lag_hours: List[int],
        prefix: str = "lag"
    ) -> pd.DataFrame:
        """
        Create lag features for target column.
        
        CRITICAL: This respects temporal boundaries - shifted values
        are only available for timestamps where they would be known.
        
        Args:
            df: DataFrame with target column
            target_column: Name of target column
            lag_hours: List of lag hours
            prefix: Prefix for lag feature names
        
        Returns:
            DataFrame with lag features
        """
        logger.info(f"Creating {len(lag_hours)} lag features for {target_column}")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        lag_df = pd.DataFrame(index=df.index)
        
        for lag in lag_hours:
            lag_col_name = f"{prefix}_{lag}h"
            lag_df[lag_col_name] = df[target_column].shift(lag)
            logger.debug(f"Created {lag_col_name}")
        
        logger.info(f"Created {len(lag_df.columns)} lag features")
        
        return lag_df
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        windows: List[int],
        functions: List[str] = ['mean', 'std', 'min', 'max']
    ) -> pd.DataFrame:
        """
        Create rolling window features.
        
        CRITICAL: Uses .shift(1) to ensure no data leakage.
        Rolling statistics only use past data.
        
        Args:
            df: DataFrame with target column
            target_column: Name of target column
            windows: List of window sizes (in hours)
            functions: List of aggregation functions
        
        Returns:
            DataFrame with rolling features
        """
        logger.info(f"Creating rolling features for {target_column}")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        rolling_df = pd.DataFrame(index=df.index)
        
        for window in windows:
            for func in functions:
                col_name = f"rolling_{window}h_{func}"
                
                # Shift by 1 to prevent data leakage
                rolling_df[col_name] = df[target_column].shift(1).rolling(
                    window=window,
                    min_periods=1
                ).agg(func)
                
                logger.debug(f"Created {col_name}")
        
        logger.info(f"Created {len(rolling_df.columns)} rolling features")
        
        return rolling_df
    
    def create_diff_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        periods: List[int] = [1, 24, 168]
    ) -> pd.DataFrame:
        """
        Create difference features (change from previous period).
        
        Args:
            df: DataFrame with target column
            target_column: Name of target column
            periods: List of periods for differencing
        
        Returns:
            DataFrame with difference features
        """
        logger.info(f"Creating difference features for {target_column}")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        diff_df = pd.DataFrame(index=df.index)
        
        for period in periods:
            col_name = f"diff_{period}h"
            diff_df[col_name] = df[target_column].diff(period)
            logger.debug(f"Created {col_name}")
        
        logger.info(f"Created {len(diff_df.columns)} difference features")
        
        return diff_df
