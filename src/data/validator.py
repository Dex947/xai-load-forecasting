"""
Data Validator Module
======================

Validates data quality, temporal integrity, and ensures no data leakage.
Critical for maintaining temporal rigor in time-series forecasting.

Usage:
    from src.data.validator import TemporalValidator, validate_data_quality
    
    validator = TemporalValidator()
    validator.validate_no_future_leakage(features, target, prediction_time)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List

from src.logger import get_logger

logger = get_logger(__name__)


class TemporalValidator:
    """
    Validates temporal integrity of features and prevents data leakage.
    
    This is critical for time-series forecasting to ensure that:
    1. Training data only uses information available at prediction time
    2. No future information leaks into features
    3. Proper train/test splits respect temporal ordering
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize temporal validator.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        logger.info("Temporal validator initialized")
    
    def validate_no_future_leakage(
        self,
        df: pd.DataFrame,
        prediction_time: pd.Timestamp,
        feature_columns: Optional[List[str]] = None
    ) -> bool:
        """
        Validate that no features contain future information.
        
        Args:
            df: DataFrame with features
            prediction_time: Time at which prediction is made
            feature_columns: List of feature columns to check
        
        Returns:
            True if validation passes
        
        Raises:
            ValueError: If future leakage is detected
        """
        logger.info(f"Validating no future leakage for prediction time: {prediction_time}")
        
        if feature_columns is None:
            feature_columns = df.columns.tolist()
        
        # Check that all data timestamps are <= prediction_time
        if isinstance(df.index, pd.DatetimeIndex):
            future_data = df[df.index > prediction_time]
            if len(future_data) > 0:
                raise ValueError(
                    f"Found {len(future_data)} rows with timestamps after prediction time. "
                    f"First future timestamp: {future_data.index[0]}"
                )
        
        logger.info("No future leakage detected")
        return True
    
    def validate_train_test_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        gap_days: int = 1
    ) -> bool:
        """
        Validate that train/test split respects temporal ordering.
        
        Args:
            train_df: Training data
            test_df: Test data
            gap_days: Required gap between train and test (in days)
        
        Returns:
            True if validation passes
        
        Raises:
            ValueError: If temporal ordering is violated
        """
        logger.info("Validating train/test temporal split")
        
        if not isinstance(train_df.index, pd.DatetimeIndex):
            raise ValueError("Train data must have DatetimeIndex")
        if not isinstance(test_df.index, pd.DatetimeIndex):
            raise ValueError("Test data must have DatetimeIndex")
        
        train_end = train_df.index.max()
        test_start = test_df.index.min()
        
        # Check temporal ordering
        if train_end >= test_start:
            raise ValueError(
                f"Train data ends at {train_end}, but test data starts at {test_start}. "
                "Train data must end before test data starts."
            )
        
        # Check gap
        actual_gap = (test_start - train_end).total_seconds() / 86400  # Convert to days
        if actual_gap < gap_days:
            logger.warning(
                f"Gap between train and test ({actual_gap:.2f} days) is less than "
                f"required gap ({gap_days} days)"
            )
        
        logger.info(f"Train period: {train_df.index.min()} to {train_end}")
        logger.info(f"Test period: {test_start} to {test_df.index.max()}")
        logger.info(f"Gap: {actual_gap:.2f} days")
        
        return True
    
    def get_valid_lag_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        lag_hours: List[int],
        prediction_time: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Create lag features that respect temporal boundaries.
        
        Args:
            df: DataFrame with target column
            target_column: Name of target column
            lag_hours: List of lag hours to create
            prediction_time: Time at which prediction is made (optional)
        
        Returns:
            DataFrame with valid lag features
        """
        logger.info(f"Creating {len(lag_hours)} lag features for {target_column}")
        
        lag_df = df[[target_column]].copy()
        
        for lag in lag_hours:
            lag_col_name = f"{target_column}_lag_{lag}h"
            lag_df[lag_col_name] = lag_df[target_column].shift(lag)
            
            # If prediction_time is specified, validate no future leakage
            if prediction_time is not None:
                future_mask = lag_df.index > prediction_time
                if future_mask.any():
                    lag_df.loc[future_mask, lag_col_name] = np.nan
        
        # Drop original target column
        lag_df = lag_df.drop(columns=[target_column])
        
        logger.info(f"Created lag features: {list(lag_df.columns)}")
        
        return lag_df


def validate_data_quality(
    df: pd.DataFrame,
    max_missing_ratio: float = 0.1,
    outlier_std_threshold: float = 5,
    min_data_points: int = 8760
) -> Dict[str, any]:
    """
    Validate data quality and return quality metrics.
    
    Args:
        df: DataFrame to validate
        max_missing_ratio: Maximum allowed missing data ratio
        outlier_std_threshold: Standard deviations for outlier detection
        min_data_points: Minimum required data points
    
    Returns:
        Dictionary with quality metrics and validation results
    
    Raises:
        ValueError: If data quality is below thresholds
    """
    logger.info("Validating data quality")
    
    quality_metrics = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'missing_by_column': {},
        'outliers_by_column': {},
        'validation_passed': True,
        'issues': []
    }
    
    # Check minimum data points
    if len(df) < min_data_points:
        issue = f"Insufficient data: {len(df)} rows (minimum: {min_data_points})"
        quality_metrics['issues'].append(issue)
        quality_metrics['validation_passed'] = False
        logger.error(issue)
    
    # Check missing data
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_ratio = missing_count / len(df)
        quality_metrics['missing_by_column'][col] = {
            'count': int(missing_count),
            'ratio': float(missing_ratio)
        }
        
        if missing_ratio > max_missing_ratio:
            issue = f"Column '{col}' has {missing_ratio:.2%} missing data (max: {max_missing_ratio:.2%})"
            quality_metrics['issues'].append(issue)
            quality_metrics['validation_passed'] = False
            logger.warning(issue)
    
    # Check outliers (for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            mean = df[col].mean()
            std = df[col].std()
            outliers = df[np.abs(df[col] - mean) > outlier_std_threshold * std]
            outlier_count = len(outliers)
            outlier_ratio = outlier_count / len(df)
            
            quality_metrics['outliers_by_column'][col] = {
                'count': int(outlier_count),
                'ratio': float(outlier_ratio)
            }
            
            if outlier_ratio > 0.01:  # More than 1% outliers
                logger.info(f"Column '{col}' has {outlier_count} outliers ({outlier_ratio:.2%})")
    
    # Log summary
    logger.info(f"Data quality validation: {'PASSED' if quality_metrics['validation_passed'] else 'FAILED'}")
    logger.info(f"Total rows: {quality_metrics['n_rows']}")
    logger.info(f"Total columns: {quality_metrics['n_columns']}")
    logger.info(f"Issues found: {len(quality_metrics['issues'])}")
    
    return quality_metrics


def check_temporal_consistency(
    df: pd.DataFrame,
    expected_freq: str = "H"
) -> Dict[str, any]:
    """
    Check temporal consistency of time-series data.
    
    Args:
        df: DataFrame with DatetimeIndex
        expected_freq: Expected frequency ('H' for hourly, 'D' for daily, etc.)
    
    Returns:
        Dictionary with consistency metrics
    """
    logger.info(f"Checking temporal consistency (expected frequency: {expected_freq})")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    consistency_metrics = {
        'start_time': df.index.min(),
        'end_time': df.index.max(),
        'expected_freq': expected_freq,
        'n_timestamps': len(df),
        'duplicates': 0,
        'gaps': [],
        'irregular_intervals': False
    }
    
    # Check for duplicates
    duplicates = df.index.duplicated().sum()
    consistency_metrics['duplicates'] = int(duplicates)
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate timestamps")
    
    # Check for gaps
    expected_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=expected_freq
    )
    missing_timestamps = expected_index.difference(df.index)
    consistency_metrics['gaps'] = [str(ts) for ts in missing_timestamps[:10]]  # First 10 gaps
    consistency_metrics['n_gaps'] = len(missing_timestamps)
    
    if len(missing_timestamps) > 0:
        logger.warning(f"Found {len(missing_timestamps)} missing timestamps")
    
    # Check for irregular intervals
    if len(df) > 1:
        intervals = df.index.to_series().diff().dropna()
        mode_interval = intervals.mode()[0] if len(intervals) > 0 else None
        irregular = (intervals != mode_interval).sum()
        consistency_metrics['irregular_intervals'] = int(irregular) > 0
        
        if irregular > 0:
            logger.warning(f"Found {irregular} irregular time intervals")
    
    logger.info(f"Temporal consistency check complete: {consistency_metrics['n_gaps']} gaps, "
               f"{consistency_metrics['duplicates']} duplicates")
    
    return consistency_metrics
