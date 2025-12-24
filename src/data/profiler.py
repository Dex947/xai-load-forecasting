"""
Data Profiler Module
====================

Comprehensive data profiling and exploratory data analysis.
Analyzes missingness, seasonality, autocorrelation, and anomalies.

Usage:
    from src.data.profiler import DataProfiler
    
    profiler = DataProfiler(df)
    profile = profiler.generate_profile()
    profiler.plot_load_patterns()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.logger import get_logger

logger = get_logger(__name__)


class DataProfiler:
    """
    Comprehensive data profiling for load forecasting datasets.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        load_column: str = "load",
        weather_columns: Optional[List[str]] = None
    ):
        """
        Initialize data profiler.
        
        Args:
            df: DataFrame with load and weather data
            load_column: Name of load column
            weather_columns: List of weather column names
        """
        self.df = df.copy()
        self.load_column = load_column
        self.weather_columns = weather_columns or []
        
        if load_column not in df.columns:
            raise ValueError(f"Load column '{load_column}' not found in DataFrame")
        
        logger.info(f"Data profiler initialized with {len(df)} rows")
    
    def generate_profile(self) -> Dict:
        """
        Generate comprehensive data profile.
        
        Returns:
            Dictionary with profiling results
        """
        logger.info("Generating data profile")
        
        profile = {
            'basic_stats': self._basic_statistics(),
            'missing_data': self._analyze_missing_data(),
            'temporal_coverage': self._temporal_coverage(),
            'load_statistics': self._load_statistics(),
            'seasonality': self._detect_seasonality(),
            'autocorrelation': self._analyze_autocorrelation(),
            'outliers': self._detect_outliers()
        }
        
        if self.weather_columns:
            profile['weather_statistics'] = self._weather_statistics()
            profile['load_weather_correlation'] = self._load_weather_correlation()
        
        logger.info("Data profile generated successfully")
        return profile
    
    def _basic_statistics(self) -> Dict:
        """Basic dataset statistics."""
        return {
            'n_rows': len(self.df),
            'n_columns': len(self.df.columns),
            'start_date': str(self.df.index.min()),
            'end_date': str(self.df.index.max()),
            'duration_days': (self.df.index.max() - self.df.index.min()).days,
            'columns': list(self.df.columns)
        }
    
    def _analyze_missing_data(self) -> Dict:
        """Analyze missing data patterns."""
        missing_stats = {}
        
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            missing_ratio = missing_count / len(self.df)
            
            missing_stats[col] = {
                'count': int(missing_count),
                'ratio': float(missing_ratio),
                'first_missing': str(self.df[self.df[col].isna()].index[0]) if missing_count > 0 else None,
                'last_missing': str(self.df[self.df[col].isna()].index[-1]) if missing_count > 0 else None
            }
        
        return missing_stats
    
    def _temporal_coverage(self) -> Dict:
        """Analyze temporal coverage and gaps."""
        # Expected hourly timestamps
        expected_index = pd.date_range(
            start=self.df.index.min(),
            end=self.df.index.max(),
            freq='H'
        )
        
        missing_timestamps = expected_index.difference(self.df.index)
        
        return {
            'expected_timestamps': len(expected_index),
            'actual_timestamps': len(self.df),
            'missing_timestamps': len(missing_timestamps),
            'coverage_ratio': len(self.df) / len(expected_index) if len(expected_index) > 0 else 0
        }
    
    def _load_statistics(self) -> Dict:
        """Detailed load statistics."""
        load_data = self.df[self.load_column].dropna()
        
        return {
            'mean': float(load_data.mean()),
            'median': float(load_data.median()),
            'std': float(load_data.std()),
            'min': float(load_data.min()),
            'max': float(load_data.max()),
            'q25': float(load_data.quantile(0.25)),
            'q75': float(load_data.quantile(0.75)),
            'skewness': float(load_data.skew()),
            'kurtosis': float(load_data.kurtosis()),
            'cv': float(load_data.std() / load_data.mean()) if load_data.mean() != 0 else None
        }
    
    def _detect_seasonality(self) -> Dict:
        """Detect seasonality patterns."""
        logger.info("Detecting seasonality patterns")
        
        # Hourly pattern
        hourly_mean = self.df.groupby(self.df.index.hour)[self.load_column].mean()
        hourly_std = self.df.groupby(self.df.index.hour)[self.load_column].std()
        
        # Daily pattern (day of week)
        daily_mean = self.df.groupby(self.df.index.dayofweek)[self.load_column].mean()
        daily_std = self.df.groupby(self.df.index.dayofweek)[self.load_column].std()
        
        # Monthly pattern
        monthly_mean = self.df.groupby(self.df.index.month)[self.load_column].mean()
        monthly_std = self.df.groupby(self.df.index.month)[self.load_column].std()
        
        return {
            'hourly_pattern': {
                'mean': hourly_mean.to_dict(),
                'std': hourly_std.to_dict(),
                'peak_hour': int(hourly_mean.idxmax()),
                'min_hour': int(hourly_mean.idxmin())
            },
            'daily_pattern': {
                'mean': daily_mean.to_dict(),
                'std': daily_std.to_dict(),
                'peak_day': int(daily_mean.idxmax()),
                'min_day': int(daily_mean.idxmin())
            },
            'monthly_pattern': {
                'mean': monthly_mean.to_dict(),
                'std': monthly_std.to_dict(),
                'peak_month': int(monthly_mean.idxmax()),
                'min_month': int(monthly_mean.idxmin())
            }
        }
    
    def _analyze_autocorrelation(self, max_lags: int = 168) -> Dict:
        """Analyze autocorrelation structure."""
        load_data = self.df[self.load_column].dropna()
        
        # Calculate autocorrelation for key lags
        key_lags = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h to 1 week
        acf_values = {}
        
        for lag in key_lags:
            if lag < len(load_data):
                acf_values[f'lag_{lag}h'] = float(load_data.autocorr(lag=lag))
        
        return {
            'autocorrelation': acf_values,
            'max_lag_analyzed': max_lags
        }
    
    def _detect_outliers(self, std_threshold: float = 5) -> Dict:
        """Detect outliers using statistical methods."""
        load_data = self.df[self.load_column].dropna()
        
        # Z-score method
        z_scores = np.abs(stats.zscore(load_data))
        outliers_zscore = load_data[z_scores > std_threshold]
        
        # IQR method
        q1 = load_data.quantile(0.25)
        q3 = load_data.quantile(0.75)
        iqr = q3 - q1
        outliers_iqr = load_data[(load_data < q1 - 1.5 * iqr) | (load_data > q3 + 1.5 * iqr)]
        
        return {
            'zscore_method': {
                'threshold': std_threshold,
                'n_outliers': len(outliers_zscore),
                'ratio': len(outliers_zscore) / len(load_data)
            },
            'iqr_method': {
                'n_outliers': len(outliers_iqr),
                'ratio': len(outliers_iqr) / len(load_data),
                'lower_bound': float(q1 - 1.5 * iqr),
                'upper_bound': float(q3 + 1.5 * iqr)
            }
        }
    
    def _weather_statistics(self) -> Dict:
        """Weather feature statistics."""
        weather_stats = {}
        
        for col in self.weather_columns:
            if col in self.df.columns:
                data = self.df[col].dropna()
                weather_stats[col] = {
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'missing_ratio': float(self.df[col].isna().sum() / len(self.df))
                }
        
        return weather_stats
    
    def _load_weather_correlation(self) -> Dict:
        """Correlation between load and weather features."""
        correlations = {}
        
        for col in self.weather_columns:
            if col in self.df.columns:
                corr = self.df[self.load_column].corr(self.df[col])
                correlations[col] = float(corr) if not np.isnan(corr) else None
        
        return correlations
    
    def plot_load_patterns(
        self,
        save_dir: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        Plot load patterns (hourly, daily, weekly, monthly).
        
        Args:
            save_dir: Directory to save figures
            figsize: Figure size
        """
        logger.info("Plotting load patterns")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Load Patterns Analysis', fontsize=16, fontweight='bold')
        
        # Hourly pattern
        hourly_mean = self.df.groupby(self.df.index.hour)[self.load_column].mean()
        hourly_std = self.df.groupby(self.df.index.hour)[self.load_column].std()
        axes[0, 0].plot(hourly_mean.index, hourly_mean.values, marker='o', linewidth=2)
        axes[0, 0].fill_between(
            hourly_mean.index,
            hourly_mean - hourly_std,
            hourly_mean + hourly_std,
            alpha=0.3
        )
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Load')
        axes[0, 0].set_title('Hourly Load Pattern')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Daily pattern (day of week)
        daily_mean = self.df.groupby(self.df.index.dayofweek)[self.load_column].mean()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[0, 1].bar(range(7), daily_mean.values, color='steelblue')
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(day_names)
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Load')
        axes[0, 1].set_title('Daily Load Pattern')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Monthly pattern
        monthly_mean = self.df.groupby(self.df.index.month)[self.load_column].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axes[1, 0].plot(monthly_mean.index, monthly_mean.values, marker='s', linewidth=2, color='green')
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].set_xticklabels(month_names, rotation=45)
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Load')
        axes[1, 0].set_title('Monthly Load Pattern')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Weekend vs Weekday
        self.df['is_weekend'] = self.df.index.dayofweek >= 5
        weekend_hourly = self.df[self.df['is_weekend']].groupby(
            self.df[self.df['is_weekend']].index.hour
        )[self.load_column].mean()
        weekday_hourly = self.df[~self.df['is_weekend']].groupby(
            self.df[~self.df['is_weekend']].index.hour
        )[self.load_column].mean()
        
        axes[1, 1].plot(weekday_hourly.index, weekday_hourly.values, 
                       marker='o', label='Weekday', linewidth=2)
        axes[1, 1].plot(weekend_hourly.index, weekend_hourly.values, 
                       marker='s', label='Weekend', linewidth=2)
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Load')
        axes[1, 1].set_title('Weekend vs Weekday Pattern')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = Path(save_dir) / 'load_patterns.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved load patterns plot to {save_path}")
        
        plt.close()
    
    def plot_missing_data_heatmap(
        self,
        save_dir: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot missing data heatmap.
        
        Args:
            save_dir: Directory to save figure
            figsize: Figure size
        """
        logger.info("Plotting missing data heatmap")
        
        # Resample to daily for visualization
        daily_missing = self.df.isna().resample('D').sum()
        
        if daily_missing.sum().sum() == 0:
            logger.info("No missing data to visualize")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            daily_missing.T,
            cmap='YlOrRd',
            cbar_kws={'label': 'Missing Count'},
            ax=ax
        )
        ax.set_title('Missing Data Heatmap (Daily Aggregation)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Features')
        
        plt.tight_layout()
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = Path(save_dir) / 'missing_data_heatmap.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved missing data heatmap to {save_path}")
        
        plt.close()
    
    def plot_autocorrelation(
        self,
        max_lags: int = 168,
        save_dir: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5)
    ) -> None:
        """
        Plot autocorrelation and partial autocorrelation.
        
        Args:
            max_lags: Maximum number of lags
            save_dir: Directory to save figure
            figsize: Figure size
        """
        logger.info("Plotting autocorrelation")
        
        load_data = self.df[self.load_column].dropna()
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Autocorrelation Analysis', fontsize=16, fontweight='bold')
        
        # ACF
        plot_acf(load_data, lags=max_lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)')
        axes[0].set_xlabel('Lag (hours)')
        
        # PACF
        plot_pacf(load_data, lags=min(50, max_lags), ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        axes[1].set_xlabel('Lag (hours)')
        
        plt.tight_layout()
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = Path(save_dir) / 'autocorrelation.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved autocorrelation plot to {save_path}")
        
        plt.close()
    
    def plot_load_weather_scatter(
        self,
        weather_feature: str,
        save_dir: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot scatter plot of load vs weather feature.
        
        Args:
            weather_feature: Weather feature to plot
            save_dir: Directory to save figure
            figsize: Figure size
        """
        if weather_feature not in self.df.columns:
            logger.warning(f"Weather feature '{weather_feature}' not found")
            return
        
        logger.info(f"Plotting load vs {weather_feature}")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sample data for visualization if too large
        plot_df = self.df[[self.load_column, weather_feature]].dropna()
        if len(plot_df) > 10000:
            plot_df = plot_df.sample(10000, random_state=42)
        
        ax.scatter(
            plot_df[weather_feature],
            plot_df[self.load_column],
            alpha=0.3,
            s=10
        )
        
        # Add trend line
        z = np.polyfit(plot_df[weather_feature], plot_df[self.load_column], 1)
        p = np.poly1d(z)
        ax.plot(
            plot_df[weather_feature].sort_values(),
            p(plot_df[weather_feature].sort_values()),
            "r--",
            linewidth=2,
            label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}'
        )
        
        corr = plot_df[self.load_column].corr(plot_df[weather_feature])
        ax.set_xlabel(weather_feature.replace('_', ' ').title())
        ax.set_ylabel('Load')
        ax.set_title(f'Load vs {weather_feature.replace("_", " ").title()} (Correlation: {corr:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = Path(save_dir) / f'load_vs_{weather_feature}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved scatter plot to {save_path}")
        
        plt.close()
