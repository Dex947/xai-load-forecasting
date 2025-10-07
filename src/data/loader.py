"""
Data Loader Module
==================

Handles loading of load data, weather data, and other external datasets.
Ensures proper timezone handling and data type consistency.

Usage:
    from src.data.loader import load_load_data, load_weather_data
    
    load_df = load_load_data("data/raw/load_data.csv")
    weather_df = load_weather_data("data/external/weather.csv")
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import pytz

from src.logger import get_logger
from src.config import load_config

logger = get_logger(__name__)


def load_load_data(
    file_path: str,
    date_column: str = "timestamp",
    load_column: str = "load",
    timezone: str = "UTC",
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Load electrical load data from file.
    
    Args:
        file_path: Path to load data file (CSV, Parquet, etc.)
        date_column: Name of the timestamp column
        load_column: Name of the load column
        timezone: Timezone for timestamp data
        parse_dates: Whether to parse dates automatically
    
    Returns:
        DataFrame with timestamp index and load column
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    logger.info(f"Loading load data from: {file_path}")
    
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"Load data file not found: {file_path}")
    
    # Determine file type and load accordingly
    suffix = file_path_obj.suffix.lower()
    
    try:
        if suffix == '.csv':
            df = pd.read_csv(file_path, parse_dates=[date_column] if parse_dates else None)
        elif suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, parse_dates=[date_column] if parse_dates else None)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Validate required columns
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in data")
        if load_column not in df.columns:
            raise ValueError(f"Load column '{load_column}' not found in data")
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            logger.info(f"Converting {date_column} to datetime")
            df[date_column] = pd.to_datetime(df[date_column])
        
        # Set timezone
        if df[date_column].dt.tz is None:
            logger.info(f"Localizing timestamps to {timezone}")
            df[date_column] = df[date_column].dt.tz_localize(timezone)
        else:
            logger.info(f"Converting timestamps to {timezone}")
            df[date_column] = df[date_column].dt.tz_convert(timezone)
        
        # Set timestamp as index
        df = df.set_index(date_column).sort_index()
        
        # Ensure load column is numeric
        df[load_column] = pd.to_numeric(df[load_column], errors='coerce')
        
        logger.info(f"Data range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Load statistics: mean={df[load_column].mean():.2f}, "
                   f"std={df[load_column].std():.2f}, "
                   f"min={df[load_column].min():.2f}, "
                   f"max={df[load_column].max():.2f}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading load data: {e}", exc_info=True)
        raise


def load_weather_data(
    file_path: str,
    date_column: str = "timestamp",
    timezone: str = "UTC",
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Load weather data from file.
    
    Args:
        file_path: Path to weather data file
        date_column: Name of the timestamp column
        timezone: Timezone for timestamp data
        parse_dates: Whether to parse dates automatically
    
    Returns:
        DataFrame with timestamp index and weather features
    
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    logger.info(f"Loading weather data from: {file_path}")
    
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"Weather data file not found: {file_path}")
    
    # Determine file type and load accordingly
    suffix = file_path_obj.suffix.lower()
    
    try:
        if suffix == '.csv':
            df = pd.read_csv(file_path, parse_dates=[date_column] if parse_dates else None)
        elif suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, parse_dates=[date_column] if parse_dates else None)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Validate date column
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in weather data")
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            logger.info(f"Converting {date_column} to datetime")
            df[date_column] = pd.to_datetime(df[date_column])
        
        # Set timezone
        if df[date_column].dt.tz is None:
            logger.info(f"Localizing timestamps to {timezone}")
            df[date_column] = df[date_column].dt.tz_localize(timezone)
        else:
            logger.info(f"Converting timestamps to {timezone}")
            df[date_column] = df[date_column].dt.tz_convert(timezone)
        
        # Set timestamp as index
        df = df.set_index(date_column).sort_index()
        
        logger.info(f"Weather data range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Weather features: {list(df.columns)}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading weather data: {e}", exc_info=True)
        raise


def merge_load_weather(
    load_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    how: str = "left",
    validate_alignment: bool = True
) -> pd.DataFrame:
    """
    Merge load and weather data on timestamp index.
    
    Args:
        load_df: Load data DataFrame
        weather_df: Weather data DataFrame
        how: Merge method ('left', 'inner', 'outer')
        validate_alignment: Validate timestamp alignment
    
    Returns:
        Merged DataFrame
    
    Raises:
        ValueError: If timestamp alignment is invalid
    """
    logger.info("Merging load and weather data")
    
    if validate_alignment:
        # Check timezone consistency
        if load_df.index.tz != weather_df.index.tz:
            logger.warning(f"Timezone mismatch: load={load_df.index.tz}, weather={weather_df.index.tz}")
            logger.info("Converting weather data to load data timezone")
            weather_df.index = weather_df.index.tz_convert(load_df.index.tz)
        
        # Check for overlapping time range
        load_start, load_end = load_df.index.min(), load_df.index.max()
        weather_start, weather_end = weather_df.index.min(), weather_df.index.max()
        
        if weather_end < load_start or weather_start > load_end:
            raise ValueError(
                f"No overlap between load data ({load_start} to {load_end}) "
                f"and weather data ({weather_start} to {weather_end})"
            )
        
        logger.info(f"Load data: {load_start} to {load_end}")
        logger.info(f"Weather data: {weather_start} to {weather_end}")
    
    # Merge on index
    merged_df = load_df.merge(
        weather_df,
        left_index=True,
        right_index=True,
        how=how,
        suffixes=('', '_weather')
    )
    
    logger.info(f"Merged data shape: {merged_df.shape}")
    logger.info(f"Merged data range: {merged_df.index.min()} to {merged_df.index.max()}")
    
    return merged_df


def save_processed_data(
    df: pd.DataFrame,
    file_path: str,
    format: str = "parquet",
    compression: Optional[str] = "snappy"
) -> None:
    """
    Save processed data to file.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        format: Output format ('parquet', 'csv')
        compression: Compression method
    """
    logger.info(f"Saving processed data to: {file_path}")
    
    file_path_obj = Path(file_path)
    file_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format == "parquet":
            df.to_parquet(file_path, compression=compression)
        elif format == "csv":
            df.to_csv(file_path, compression=compression)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(df)} rows to {file_path}")
    
    except Exception as e:
        logger.error(f"Error saving data: {e}", exc_info=True)
        raise
