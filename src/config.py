"""Configuration loading and validation via Pydantic."""

import yaml
from pathlib import Path
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class PathsConfig(BaseModel):
    """File path settings."""
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    data_external: str = "data/external"
    models: str = "models/artifacts"
    logs: str = "logs"
    figures: str = "docs/figures"
    notebooks: str = "notebooks"


class ForecastingConfig(BaseModel):
    """Forecast horizon and resolution."""
    horizon_hours: int = Field(24, gt=0)
    resolution_minutes: int = Field(60, gt=0)
    prediction_time: str = "00:00"
    timezone: str = "UTC"


class TemporalFeaturesConfig(BaseModel):
    """Time-based feature toggles."""
    hour_of_day: bool = True
    day_of_week: bool = True
    day_of_month: bool = True
    day_of_year: bool = True
    week_of_year: bool = True
    month: bool = True
    quarter: bool = True
    is_weekend: bool = True
    is_business_hour: bool = True
    season: bool = True


class CalendarFeaturesConfig(BaseModel):
    """Holiday/calendar feature toggles."""
    holidays: bool = True
    holiday_proximity: bool = True
    special_events: bool = True
    school_calendar: bool = False


class WeatherFeaturesConfig(BaseModel):
    """Weather variable toggles."""
    temperature: bool = True
    humidity: bool = True
    wind_speed: bool = True
    precipitation: bool = True
    cloud_cover: bool = True
    pressure: bool = True
    dew_point: bool = True
    feels_like: bool = True


class InteractionFeaturesConfig(BaseModel):
    """Cross-feature interaction toggles."""
    temp_hour: bool = True
    temp_weekend: bool = True
    humidity_temp: bool = True


class FeaturesConfig(BaseModel):
    """Feature engineering settings."""
    lag_hours: List[int] = [1, 2, 3, 6, 12, 24, 48, 168]
    rolling_windows: List[int] = [3, 6, 12, 24, 168]
    weather_lag_hours: List[int] = [0, 1, 3, 6]
    temporal: TemporalFeaturesConfig = TemporalFeaturesConfig()
    calendar: CalendarFeaturesConfig = CalendarFeaturesConfig()
    weather: WeatherFeaturesConfig = WeatherFeaturesConfig()
    interactions: InteractionFeaturesConfig = InteractionFeaturesConfig()


class ModelConfig(BaseModel):
    """Model type and hyperparameters."""
    type: str = "lightgbm"
    monotonic_constraints: Dict[str, int] = {"temperature": 1}
    lightgbm: Dict[str, Any] = {}
    xgboost: Dict[str, Any] = {}


class ValidationConfig(BaseModel):
    """Cross-validation settings."""
    method: str = "rolling_origin"
    n_splits: int = Field(5, gt=0)
    test_size_days: int = Field(30, gt=0)
    gap_days: int = Field(1, ge=0)
    min_train_days: int = Field(365, gt=0)


class SHAPConfig(BaseModel):
    """SHAP analysis settings."""
    compute_global: bool = True
    compute_local: bool = True
    compute_time_varying: bool = True
    sample_size: int = Field(1000, gt=0)
    background_size: int = Field(100, gt=0)


class CounterfactualConfig(BaseModel):
    """Counterfactual scenario settings."""
    enabled: bool = True
    n_scenarios: int = Field(5, gt=0)
    features_to_vary: List[str] = ["temperature", "humidity", "hour_of_day"]


class VisualizationsConfig(BaseModel):
    """Plot generation toggles."""
    summary_plot: bool = True
    dependence_plots: bool = True
    force_plots: bool = True
    waterfall_plots: bool = True
    time_varying_plots: bool = True


class ExplainabilityConfig(BaseModel):
    """XAI module settings."""
    shap: SHAPConfig = SHAPConfig()
    counterfactual: CounterfactualConfig = CounterfactualConfig()
    visualizations: VisualizationsConfig = VisualizationsConfig()


class DataQualityConfig(BaseModel):
    """Data validation thresholds."""
    max_missing_ratio: float = Field(0.1, ge=0, le=1)
    outlier_std_threshold: float = Field(5, gt=0)
    min_data_points: int = Field(8760, gt=0)


class LoggingConfig(BaseModel):
    """Log format and rotation."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file_rotation: bool = True
    max_bytes: int = 10485760
    backup_count: int = 5


class WeatherConfig(BaseModel):
    """Weather data source settings."""
    provider: str = "openweathermap"
    api_key_env: str = "WEATHER_API_KEY"
    cache_enabled: bool = True
    cache_duration_hours: int = 24


class HolidaysConfig(BaseModel):
    """Holiday calendar settings."""
    country: str = "US"
    state: Optional[str] = None
    custom_holidays: List[str] = []


class MetricsConfig(BaseModel):
    """Evaluation metric selection."""
    primary: str = "rmse"
    additional: List[str] = ["mae", "mape", "r2", "max_error", "quantile_loss"]


class ProjectConfig(BaseModel):
    """Project info."""
    name: str = "XAI Load Forecasting"
    version: str = "1.0.0"
    description: str = "Day-ahead feeder load forecasting with explainability"
    author: str = "AI Consultant & Data Engineer"
    created: str = "2025-10-07"


class Config(BaseModel):
    """Root config container."""
    project: ProjectConfig = ProjectConfig()
    paths: PathsConfig = PathsConfig()
    forecasting: ForecastingConfig = ForecastingConfig()
    features: FeaturesConfig = FeaturesConfig()
    model: ModelConfig = ModelConfig()
    validation: ValidationConfig = ValidationConfig()
    explainability: ExplainabilityConfig = ExplainabilityConfig()
    data_quality: DataQualityConfig = DataQualityConfig()
    logging: LoggingConfig = LoggingConfig()
    weather: WeatherConfig = WeatherConfig()
    holidays: HolidaysConfig = HolidaysConfig()
    metrics: MetricsConfig = MetricsConfig()


def load_config(config_path: str = "config/config.yaml") -> Config:
    """Load config from YAML file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Validate and create Config object
    config = Config(**config_dict)
    
    return config


def load_holidays_config(config_path: str = "config/holidays.yaml") -> Dict[str, Any]:
    """Load holidays config from YAML."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Holidays config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        holidays_config = yaml.safe_load(f)
    
    return holidays_config


def load_weather_config(config_path: str = "config/weather_config.yaml") -> Dict[str, Any]:
    """Load weather config from YAML."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Weather config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        weather_config = yaml.safe_load(f)
    
    return weather_config


def get_project_root() -> Path:
    """Return project root directory."""
    # Assuming this file is in src/, go up one level
    return Path(__file__).parent.parent


def resolve_path(relative_path: str) -> Path:
    """Convert relative path to absolute from project root."""
    return get_project_root() / relative_path
