"""Holiday and calendar feature engineering."""

import pandas as pd
from typing import Dict, Optional
from datetime import timedelta
import holidays

from src.logger import get_logger

logger = get_logger(__name__)


class CalendarFeatureEngineer:
    """Holiday flags, proximity, and special event features."""
    
    def __init__(
        self,
        calendar_config: Optional[Dict] = None,
        holidays_config: Optional[Dict] = None
    ):
        self.calendar_config = calendar_config or {}
        self.holidays_config = holidays_config or {}
        
        # Initialize holidays
        self.country = self.holidays_config.get('standard', {}).get('country', 'US')
        self.state = self.holidays_config.get('standard', {}).get('state')
        self.holiday_calendar = holidays.country_holidays(
            self.country,
            state=self.state,
            observed=self.holidays_config.get('standard', {}).get('observed', True)
        )
        
        # Custom holidays
        self.custom_holidays = set()
        for date_str in self.holidays_config.get('custom_holidays', []):
            try:
                self.custom_holidays.add(pd.to_datetime(date_str).date())
            except Exception:
                logger.warning(f"Invalid custom holiday date: {date_str}")
        
        # Special events
        self.special_events = {}
        for event in self.holidays_config.get('special_events', []):
            event_name = event.get('name')
            for date_str in event.get('dates', []):
                try:
                    date = pd.to_datetime(date_str).date()
                    self.special_events[date] = event_name
                except Exception:
                    logger.warning(f"Invalid special event date: {date_str}")
        
        logger.info(f"Calendar feature engineer initialized for {self.country}")
        logger.info(f"Custom holidays: {len(self.custom_holidays)}")
        logger.info(f"Special events: {len(self.special_events)}")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all calendar features.
        
        Args:
            df: DataFrame with DatetimeIndex
        
        Returns:
            DataFrame with calendar features
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        logger.info("Creating calendar features")
        
        features_df = pd.DataFrame(index=df.index)
        
        # Is holiday
        if self.calendar_config.get('holidays', True):
            # Convert index dates to check against holidays
            index_dates = pd.Series(df.index.date, index=df.index)
            features_df['is_holiday'] = index_dates.isin(
                self.holiday_calendar.keys()
            ).astype(int)
            
            # Add custom holidays
            features_df['is_holiday'] |= index_dates.isin(
                self.custom_holidays
            ).astype(int)
            
            logger.debug(f"Created is_holiday feature ({features_df['is_holiday'].sum()} holidays)")
        
        # Holiday proximity
        if self.calendar_config.get('holiday_proximity', True):
            proximity_config = self.holidays_config.get('proximity', {})
            days_before = proximity_config.get('days_before', 2)
            days_after = proximity_config.get('days_after', 1)
            
            features_df['days_to_holiday'] = self._calculate_days_to_holiday(
                df.index,
                days_before,
                days_after
            )
            
            features_df['is_before_holiday'] = (
                (features_df['days_to_holiday'] > 0) & 
                (features_df['days_to_holiday'] <= days_before)
            ).astype(int)
            
            features_df['is_after_holiday'] = (
                (features_df['days_to_holiday'] < 0) & 
                (features_df['days_to_holiday'] >= -days_after)
            ).astype(int)
            
            logger.debug("Created holiday proximity features")
        
        # Special events
        if self.calendar_config.get('special_events', True):
            index_dates = pd.Series(df.index.date, index=df.index)
            features_df['is_special_event'] = index_dates.isin(
                self.special_events.keys()
            ).astype(int)
            
            logger.debug(f"Created special event feature ({features_df['is_special_event'].sum()} events)")
        
        # Holiday type (major, minor, etc.)
        if self.calendar_config.get('holidays', True):
            features_df['holiday_type'] = self._get_holiday_type(df.index)
            logger.debug("Created holiday_type feature")
        
        # Weekend-adjacent holidays
        if self.holidays_config.get('weekend_adjacent', {}).get('friday_before', True):
            features_df['is_friday_before_weekend_holiday'] = self._is_friday_before_weekend_holiday(
                df.index
            ).astype(int)
        
        if self.holidays_config.get('weekend_adjacent', {}).get('monday_after', True):
            features_df['is_monday_after_weekend_holiday'] = self._is_monday_after_weekend_holiday(
                df.index
            ).astype(int)
        
        logger.info(f"Created {len(features_df.columns)} calendar features")
        
        return features_df
    
    def _calculate_days_to_holiday(
        self,
        index: pd.DatetimeIndex,
        days_before: int,
        days_after: int
    ) -> pd.Series:
        """
        Calculate days to nearest holiday.
        
        Positive values = days until next holiday
        Negative values = days since last holiday
        0 = is a holiday
        """
        days_to_holiday = pd.Series(index=index, dtype=float)
        
        # Get all holiday dates
        all_holidays = set(self.holiday_calendar.keys()) | self.custom_holidays
        
        for idx, timestamp in enumerate(index):
            date = timestamp.date()
            
            if date in all_holidays:
                days_to_holiday.iloc[idx] = 0
            else:
                # Find nearest holiday
                min_distance = float('inf')
                
                for holiday_date in all_holidays:
                    distance = (holiday_date - date).days
                    
                    if abs(distance) < abs(min_distance):
                        min_distance = distance
                
                days_to_holiday.iloc[idx] = min_distance
        
        return days_to_holiday
    
    def _get_holiday_type(self, index: pd.DatetimeIndex) -> pd.Series:
        """
        Get holiday type category (0=not holiday, 1=major, 2=minor, 3=religious, 4=regional).
        """
        holiday_type = pd.Series(0, index=index, dtype=int)
        
        categories = self.holidays_config.get('categories', {})
        major_holidays = set(categories.get('major', []))
        minor_holidays = set(categories.get('minor', []))
        religious_holidays = set(categories.get('religious', []))
        regional_holidays = set(categories.get('regional', []))
        
        for idx, timestamp in enumerate(index):
            date = timestamp.date()
            
            if date in self.holiday_calendar:
                holiday_name = self.holiday_calendar.get(date)
                
                if holiday_name in major_holidays:
                    holiday_type.iloc[idx] = 1
                elif holiday_name in minor_holidays:
                    holiday_type.iloc[idx] = 2
                elif holiday_name in religious_holidays:
                    holiday_type.iloc[idx] = 3
                elif holiday_name in regional_holidays:
                    holiday_type.iloc[idx] = 4
        
        return holiday_type
    
    def _is_friday_before_weekend_holiday(self, index: pd.DatetimeIndex) -> pd.Series:
        """Check if date is Friday before a weekend holiday."""
        is_friday_before = pd.Series(False, index=index, dtype=bool)
        
        all_holidays = set(self.holiday_calendar.keys()) | self.custom_holidays
        
        for idx, timestamp in enumerate(index):
            date = timestamp.date()
            
            # Check if Friday
            if timestamp.dayofweek == 4:  # Friday
                # Check if Saturday or Sunday is a holiday
                saturday = date + timedelta(days=1)
                sunday = date + timedelta(days=2)
                
                if saturday in all_holidays or sunday in all_holidays:
                    is_friday_before.iloc[idx] = True
        
        return is_friday_before
    
    def _is_monday_after_weekend_holiday(self, index: pd.DatetimeIndex) -> pd.Series:
        """Check if date is Monday after a weekend holiday."""
        is_monday_after = pd.Series(False, index=index, dtype=bool)
        
        all_holidays = set(self.holiday_calendar.keys()) | self.custom_holidays
        
        for idx, timestamp in enumerate(index):
            date = timestamp.date()
            
            # Check if Monday
            if timestamp.dayofweek == 0:  # Monday
                # Check if Saturday or Sunday was a holiday
                saturday = date - timedelta(days=2)
                sunday = date - timedelta(days=1)
                
                if saturday in all_holidays or sunday in all_holidays:
                    is_monday_after.iloc[idx] = True
        
        return is_monday_after
