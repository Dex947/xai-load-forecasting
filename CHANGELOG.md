# Changelog
All notable changes to the XAI Load Forecasting project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] - v3.0.0

### Added

**Data Expansion**:
- Solar irradiance data fetcher (`src/data/solar.py`) with Open-Meteo API integration
- Unified data source manager (`src/data/sources.py`) for multi-source data handling
- Solar-derived features: GHI, DNI, DHI, clearness index, solar availability

**Probabilistic Forecasting**:
- Conformal prediction with MAPIE EnbPI method (`src/models/conformal.py`)
- Guaranteed coverage prediction intervals
- Winkler score and CRPS metrics

**Online Learning**:
- River-based online forecaster (`src/models/online.py`) for streaming updates
- ADWIN and Page-Hinkley drift detection
- Hybrid batch+online ensemble forecaster

**MLOps**:
- MLflow experiment tracking (`src/mlops/tracking.py`)
- Model registry with staging/production stages (`src/mlops/registry.py`)
- A/B testing framework with statistical significance (`src/mlops/ab_testing.py`)
- Multi-armed bandit for adaptive model selection

**Explainability Extensions**:
- Counterfactual explanations with DiCE (`src/explainability/counterfactuals.py`)
- Concept drift explanation via SHAP changes (`src/explainability/drift_explanation.py`)
- Natural language explanation generator (`src/explainability/natural_language.py`)

**Testing**:
- 14 new tests for advanced modules (38 total)

### Changed

- Updated requirements.txt with mapie, river, mlflow dependencies

---

## [2.0.0] - 2025-12-24

### Added

**Testing & CI**:
- Pytest test suite with 24 tests covering data loading, features, models, and explainability
- GitHub Actions CI workflow with linting (ruff), type checking (mypy), and testing
- Multi-Python version testing (3.9, 3.10, 3.11)

**CLI Interface**:
- Click-based CLI (`src/cli.py`) with commands for all pipeline stages
- Commands: `profile`, `features`, `train`, `explain`, `predict`, `test`, `serve`

**API & Deployment**:
- FastAPI serving layer (`src/api.py`) with prediction and explanation endpoints
- Docker containerization (Dockerfile, docker-compose.yml)
- Health check and batch prediction endpoints

**Advanced Modeling**:
- Prediction intervals via LightGBM quantile regression (`src/models/quantile.py`)
- Hyperparameter optimization with Optuna (`src/models/tuning.py`)
- Multi-horizon forecasting with direct strategy (`src/models/multi_horizon.py`)

**Monitoring**:
- Data drift detection using Kolmogorov-Smirnov tests (`src/monitoring/drift.py`)
- Performance monitoring with degradation alerts

### Changed

- Fixed MAPE calculation to prevent division by zero
- Updated requirements.txt with new dependencies (click, fastapi, uvicorn, optuna, pydantic-settings)

---

## [1.0.0] - 2025-10-07

### Release Notes

First stable release of XAI Load Forecasting system. Complete day-ahead load forecasting with explainability, trained on real-world data, validated, and deployment-ready.

### Added

**Model Card & Final Documentation**:
- Comprehensive model card with scope, limitations, and failure modes
- FAQ section in README
- Deployment guidelines
- Ethical considerations documented
- Monitoring and maintenance plan

### Changed

- Updated README with actual results and visualizations
- Enhanced documentation with human-like explanations
- Added proper academic citations and references

---

## [0.9.0] - 2025-10-05

### Added

**SHAP Analysis & Explainability**:
- SHAP value computation for test set
- Global feature importance analysis
- Time-varying SHAP patterns (hourly, daily, monthly)
- Individual prediction explanations
- 16 SHAP visualizations generated
- Feature importance comparison (Model vs SHAP)

**Visualizations**:
- SHAP summary plot (beeswarm)
- SHAP bar plot (mean importance)
- Dependence plots for top 6 features
- Waterfall plots for individual predictions
- Time-varying importance heatmaps

### Changed

- Optimized SHAP computation with background sampling
- Cached SHAP values for efficiency

---

## [0.8.0] - 2025-10-01

### Added

**Model Improvements & Diagnostics**:
- Feature pruning using permutation importance (88 → 58 features)
- Improved hyperparameters with regularization
- Residual diagnostics (4-panel analysis)
- Overfitting prevention (L1/L2, feature sampling)

**Diagnostics**:
- Residual vs predictions plot
- Q-Q plot for normality check
- Residual distribution histogram
- Residuals over time analysis

### Changed

- Reduced learning rate from 0.05 to 0.03
- Added feature/bagging sampling (0.8 fraction)
- Increased num_leaves from 31 to 63
- Limited max_depth to 10

### Performance

- Test RMSE: 0.770 kW (improved from 0.742)
- Feature count reduced by 34%
- Overfitting controlled (Train/Val/Test consistent)

---

## [0.7.0] - 2025-09-25

### Added

**Model Training & Validation**:
- LightGBM model training with monotonic constraints
- Baseline models (persistence, seasonal naive)
- Rolling origin cross-validation framework
- Model performance evaluation
- Feature importance analysis

**Trained Models**:
- LightGBM model (409 KB)
- Feature importance rankings
- Training results and metrics

### Performance

- Test RMSE: 0.742 kW
- MAE: 0.477 kW
- R²: 0.301
- 17% better than persistence baseline
- 49% better than seasonal naive

---

## [0.6.0] - 2025-09-18

### Added

**Feature Engineering Pipeline**:
- 89 features engineered from raw data
- Temporal features (20+): hour, day, week, month, cyclical encoding
- Calendar features (8): holidays, weekends, proximity
- Lag features (8): 1h to 168h lags
- Rolling statistics (20): mean, std, min, max over multiple windows
- Weather features (15+): base + derived (HDD, CDD, heat index)
- Interaction features (3): temperature × hour, etc.

**Feature Pipeline**:
- Automated feature generation
- Temporal leakage prevention
- Feature validation and quality checks

### Changed

- Optimized feature computation for efficiency
- Added feature name documentation

---

## [0.5.0] - 2025-09-10

### Added

**Data Profiling & EDA**:
- Comprehensive data profiling script
- Statistical analysis of load and weather data
- Seasonality analysis (hourly, daily, weekly, monthly)
- Autocorrelation analysis (ACF, PACF)
- Weather-load correlation analysis

**Visualizations**:
- Load patterns plot (4-panel seasonality)
- Autocorrelation plot
- Load vs temperature scatter
- Load vs humidity scatter
- Load vs wind speed scatter

### Insights

- Strong hourly seasonality (peak 18:00-21:00)
- Weekly patterns (weekday vs weekend)
- High autocorrelation up to 168 hours
- Moderate temperature-load correlation

---

## [0.4.0] - 2025-08-28

### Added

**Data Acquisition**:
- UCI Electricity Load Diagrams dataset (26,267 hours)
- Open-Meteo historical weather data (26,328 hours)
- Automated data download script
- Data validation and quality checks

**Data Processing**:
- Hourly resampling from 15-min data
- Timezone alignment (UTC)
- Missing data handling
- Outlier detection

### Data Quality

- 99.1% data completeness
- Temporal consistency validated
- No duplicates found
- <1% outliers detected

---

## [0.3.0] - 2025-08-15

### Added

**Explainability Module**:
- SHAP analysis framework
- TreeExplainer for gradient boosting models
- Visualization utilities for SHAP plots
- Time-varying importance analysis
- Individual prediction explanations

**Visualizations**:
- Summary plots (beeswarm, bar)
- Dependence plots
- Waterfall plots
- Force plots

---

## [0.2.0] - 2025-07-25

### Added

**Models Module**:
- Baseline models (persistence, seasonal naive, moving average)
- Gradient boosting model wrapper (LightGBM, XGBoost)
- Monotonic constraints support
- Model evaluation metrics (RMSE, MAE, MAPE, R²)
- Model serialization (save/load)

**Validation Module**:
- Rolling origin cross-validation
- Temporal validation framework
- Train/test splitting with temporal ordering

---

## [0.1.0] - 2025-07-10

### Added

**Phase 1: Architecture & Core System**tup

#### Project Structure
- Created complete folder structure for modular codebase
- initialized `.gitignore` with Python, data, model, and LaTeX exclusions
- Created placeholder directories with `.gitkeep` files
#### Configuration System
- **config/config.yaml**: Master configuration file with:
  - Project metadata and paths
  - Forecasting parameters (24h horizon, hourly resolution)
  - Feature engineering windows (lags, rolling windows, interactions)
  - Model configuration (LightGBM/XGBoost with monotonic constraints)
  - Validation strategy (rolling origin CV)
  - Explainability settings (SHAP, counterfactuals)
  - Data quality thresholds
  - Logging configuration
  - Model Card structure

- **config/holidays.yaml**: Holiday and special event configuration
  - Standard holidays (US-based, configurable)
  - Custom holidays and special events
  - Holiday proximity features
  - Holiday categorization (major, minor, religious, regional)
  - Weekend-adjacent holiday handling

- **config/weather_config.yaml**: Weather data configuration
  - API provider settings (OpenWeatherMap)
  - Weather features (temperature, humidity, wind, precipitation, etc.)
  - Derived features (degree days, heat index, discomfort index)
  - Data quality validation rules
  - Missing data handling strategy
  - Timezone alignment configuration
  - Caching settings

#### Dependencies
- **requirements.txt**: Comprehensive Python dependencies
  - Core: numpy, pandas, scipy, pyarrow
  - ML: scikit-learn, xgboost, lightgbm
  - Explainability: shap, dice-ml
  - Visualization: matplotlib, seaborn, plotly
  - Config: pyyaml, pydantic
  - Calendar: holidays, pytz
  - Development: jupyter, pytest

#### Documentation
- **memory.json**: Project context and architectural decisions
  - Objectives and key principles
  - Technical stack and architecture
  - Data requirements and validation strategy
  - Explainability requirements
  - Deliverables checklist
  - Phase tracking

- **CHANGELOG.md**: This file for tracking all project changes

### Technical Decisions

1. **Config-Driven Architecture**: All parameters externalized to YAML for easy modification
2. **Temporal Rigor**: Strict enforcement of no data leakage in feature engineering
3. **Modular Design**: Separation of concerns (data, features, models, explainability)
4. **Logging-First**: Structured logging instead of print statements
5. **Explainability-First**: SHAP and counterfactuals as primary deliverables
6. **Monotonic Constraints**: Temperature-load relationship enforced in model
7. **Rolling Origin CV**: Proper time-series validation strategy

#### Core Modules Implemented

**Data Module** (`src/data/`):
- `loader.py`: Load data, weather data, merge functions, save/load processed data
- `validator.py`: TemporalValidator class for preventing data leakage, train/test validation
- `profiler.py`: DataProfiler class for EDA, seasonality detection, autocorrelation analysis

**Features Module** (`src/features/`):
- `temporal.py`: TemporalFeatureEngineer - hour, day, week, cyclical encoding, lag features, rolling features
- `calendar.py`: CalendarFeatureEngineer - holidays, special events, proximity features
- `weather.py`: WeatherFeatureEngineer - derived features (HDD, CDD, heat index), interactions
- `pipeline.py`: FeaturePipeline - orchestrates all feature engineering with temporal rigor

**Models Module** (`src/models/`):
- `baseline.py`: BaselineModel - persistence, seasonal naive, moving average
- `gbm.py`: GradientBoostingModel - LightGBM/XGBoost with monotonic constraints
- `validator.py`: RollingOriginValidator - time-series cross-validation

**Explainability Module** (`src/explainability/`):
- `shap_analysis.py`: SHAPAnalyzer - global, local, time-varying SHAP analysis
- `visualizations.py`: ExplainabilityVisualizer - summary plots, dependence plots, waterfall, force plots

**Documentation**:
- `README.md`: Comprehensive project documentation with architecture, workflow, usage examples
- Configuration files fully documented with inline comments

#### Data Download and Profiling

**Scripts Created**:
- `scripts/download_data.py`: Automated data download from UCI and Open-Meteo
- `scripts/run_data_profiling.py`: Comprehensive data profiling and EDA

**Data Acquired**:
- UCI Electricity Load Diagrams (2011-2014): 26,267 hourly samples
- Open-Meteo Historical Weather: Temperature, humidity, wind, precipitation, pressure, cloud cover
- Sample dataset (2013): 8,760 hourly samples for quick testing

**Visualizations Generated**:
- Load patterns (hourly, daily, weekly, monthly)
- Autocorrelation analysis
- Weather-load scatter plots (temperature, humidity, wind speed)
- Missing data heatmap

**Documentation Added**:
- `LICENSE`: MIT License
- `CONTRIBUTING.md`: Comprehensive contribution guidelines
- Updated README with citation, acknowledgments, and contact information

### Next Steps
- Phase 3: Execute feature engineering pipeline
- Phase 4: Train models with rolling origin CV
- Phase 5: Generate SHAP analysis and visualizations
- Phase 6: Create Model Card and LaTeX technical report

---

## Notes

- All timestamps in UTC unless specified otherwise
- Configuration files use YAML for human readability
- Model artifacts will be versioned and cached
- SHAP values will be pre-computed and cached for efficiency
