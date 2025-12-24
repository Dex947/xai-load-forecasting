"""
Feature engineering pipeline orchestration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from src.logger import get_logger
from src.config import load_config, load_holidays_config
from src.features.temporal import TemporalFeatureEngineer
from src.features.calendar import CalendarFeatureEngineer
from src.features.weather import WeatherFeatureEngineer

logger = get_logger(__name__)


class FeaturePipeline:
    """Coordinates temporal, calendar, weather, and interaction features."""
    
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config_obj = load_config()
            self.config = {
                'features': config_obj.features.dict(),
                'forecasting': config_obj.forecasting.dict()
            }
        else:
            self.config = config
        
        # Initialize feature engineers
        self.temporal_engineer = TemporalFeatureEngineer(
            self.config['features']
        )
        
        self.calendar_engineer = CalendarFeatureEngineer(
            calendar_config=self.config['features'].get('calendar', {}),
            holidays_config=load_holidays_config()
        )
        
        self.weather_engineer = WeatherFeatureEngineer(
            weather_config=self.config.get('weather', {})
        )
        
        logger.info("Feature pipeline initialized")
    
    def create_all_features(
        self,
        df: pd.DataFrame,
        target_column: str = 'load'
    ) -> pd.DataFrame:
        """Generate all features from raw load/weather data."""
        logger.info("Creating all features")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Start with empty feature DataFrame
        all_features = pd.DataFrame(index=df.index)
        
        # 1. Temporal features
        logger.info("Step 1/6: Creating temporal features")
        temporal_features = self.temporal_engineer.create_features(df)
        all_features = pd.concat([all_features, temporal_features], axis=1)
        
        # 2. Calendar features
        logger.info("Step 2/6: Creating calendar features")
        calendar_features = self.calendar_engineer.create_features(df)
        all_features = pd.concat([all_features, calendar_features], axis=1)
        
        # 3. Lag features (load)
        logger.info("Step 3/6: Creating lag features")
        lag_hours = self.config['features'].get('lag_hours', [1, 2, 3, 6, 12, 24, 48, 168])
        lag_features = self.temporal_engineer.create_lag_features(
            df,
            target_column=target_column,
            lag_hours=lag_hours,
            prefix=f"{target_column}_lag"
        )
        all_features = pd.concat([all_features, lag_features], axis=1)
        
        # 4. Rolling features (load)
        logger.info("Step 4/6: Creating rolling features")
        rolling_windows = self.config['features'].get('rolling_windows', [3, 6, 12, 24, 168])
        rolling_features = self.temporal_engineer.create_rolling_features(
            df,
            target_column=target_column,
            windows=rolling_windows,
            functions=['mean', 'std', 'min', 'max']
        )
        all_features = pd.concat([all_features, rolling_features], axis=1)
        
        # 5. Weather features (if available)
        weather_columns = [col for col in df.columns if col != target_column]
        if weather_columns:
            logger.info("Step 5/6: Creating weather features")
            
            # Add base weather features
            for col in weather_columns:
                if col in df.columns:
                    all_features[col] = df[col]
            
            # Derived weather features
            derived_weather = self.weather_engineer.create_derived_features(df)
            all_features = pd.concat([all_features, derived_weather], axis=1)
            
            # Weather lag features
            weather_lag_hours = self.config['features'].get('weather_lag_hours', [0, 1, 3, 6])
            if len(weather_lag_hours) > 1:  # Skip if only [0]
                weather_lag_features = self.weather_engineer.create_lag_features(
                    df,
                    weather_columns=weather_columns,
                    lag_hours=[h for h in weather_lag_hours if h > 0]
                )
                all_features = pd.concat([all_features, weather_lag_features], axis=1)
        
        # 6. Interaction features
        logger.info("Step 6/6: Creating interaction features")
        interactions_config = self.config['features'].get('interactions', {})
        interaction_features = self.weather_engineer.create_interaction_features(
            all_features,
            interactions_config=interactions_config
        )
        all_features = pd.concat([all_features, interaction_features], axis=1)
        
        # Include target if requested
        if include_target:
            all_features[target_column] = df[target_column]
        
        logger.info(f"Feature engineering complete: {len(all_features.columns)} features created")
        logger.info(f"Feature DataFrame shape: {all_features.shape}")
        
        # Log feature categories
        self._log_feature_summary(all_features)
        
        return all_features
    
    def _log_feature_summary(self, features_df: pd.DataFrame) -> None:
        """Log summary of created features by category."""
        feature_counts = {
            'temporal': 0,
            'calendar': 0,
            'lag': 0,
            'rolling': 0,
            'weather': 0,
            'interaction': 0,
            'other': 0
        }
        
        for col in features_df.columns:
            if any(x in col for x in ['hour', 'day', 'week', 'month', 'quarter', 'season']):
                feature_counts['temporal'] += 1
            elif any(x in col for x in ['holiday', 'weekend', 'business']):
                feature_counts['calendar'] += 1
            elif 'lag' in col:
                feature_counts['lag'] += 1
            elif 'rolling' in col:
                feature_counts['rolling'] += 1
            elif any(x in col for x in ['temp', 'humidity', 'wind', 'precip', 'pressure', 'hdd', 'cdd', 'heat_index']):
                feature_counts['weather'] += 1
            elif '_x_' in col:
                feature_counts['interaction'] += 1
            else:
                feature_counts['other'] += 1
        
        logger.info("Feature summary by category:")
        for category, count in feature_counts.items():
            if count > 0:
                logger.info(f"  {category}: {count}")
    
    def save_features(
        self,
        features_df: pd.DataFrame,
        file_path: str,
        format: str = 'parquet'
    ) -> None:
        """
        Save features to file.
        
        Args:
            features_df: DataFrame with features
            file_path: Output file path
            format: Output format ('parquet', 'csv')
        """
        logger.info(f"Saving features to {file_path}")
        
        file_path_obj = Path(file_path)
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'parquet':
            features_df.to_parquet(file_path, compression='snappy')
        elif format == 'csv':
            features_df.to_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Features saved: {features_df.shape}")
    
    def load_features(
        self,
        file_path: str,
        format: str = 'parquet'
    ) -> pd.DataFrame:
        """
        Load features from file.
        
        Args:
            file_path: Input file path
            format: Input format ('parquet', 'csv')
        
        Returns:
            DataFrame with features
        """
        logger.info(f"Loading features from {file_path}")
        
        if format == 'parquet':
            features_df = pd.read_parquet(file_path)
        elif format == 'csv':
            features_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Features loaded: {features_df.shape}")
        
        return features_df
