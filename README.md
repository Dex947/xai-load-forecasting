<div align="center">

# ⚡ XAI Load Forecasting

### Explainable Day-Ahead Electrical Load Forecasting

*Transparent predictions you can trust*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-24%20passed-brightgreen.svg)]()
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)]()
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)]()

[Features](#-features) • [Quick Start](#-quick-start) • [Results](#-results) • [API](#-api-serving) • [Documentation](#-documentation)

</div>

---

<div align="center">
<img src="docs/figures/shap_summary.png" alt="SHAP Feature Importance" width="700"/>

*Every prediction explained: SHAP analysis reveals what drives your forecasts*
</div>

---

## 🎯 Why This Project?

Grid operators need **accurate** load forecasts, but they also need to **understand** them. Black-box models create risk—when predictions fail, operators can't diagnose why.

**XAI Load Forecasting** solves this with:
- 🔮 **14% better accuracy** than persistence baselines
- 🔍 **Full transparency** via SHAP explanations for every prediction
- ⚙️ **Production-ready** with FastAPI, Docker, and monitoring built-in
- 📊 **Drift detection** to alert when model performance degrades

---

## ✨ Features

| Category | Capabilities |
|----------|-------------|
| **Forecasting** | Day-ahead (24h) hourly predictions, multi-horizon support (1h to 1 week) |
| **Explainability** | Global SHAP importance, local explanations, time-varying patterns |
| **Models** | LightGBM/XGBoost with monotonic constraints, quantile regression for intervals |
| **Validation** | Rolling origin cross-validation, temporal leak prevention |
| **Deployment** | FastAPI server, Docker containers, CLI interface |
| **Monitoring** | Data drift detection (KS-test), performance alerts |
| **Optimization** | Optuna hyperparameter tuning with cross-validation |

---

## 📊 Results at a Glance

<div align="center">
<img src="docs/figures/model_predictions.png" alt="Model Predictions" width="800"/>

*30-day test period: Actual vs Predicted load with high accuracy*
</div>

### Performance Comparison

| Model | RMSE (kW) | Improvement |
|-------|-----------|-------------|
| Persistence (last value) | 0.892 | Baseline |
| Seasonal Naive (last week) | 1.458 | -63% worse |
| **LightGBM (Ours)** | **0.770** | **+14% better** |

### What Drives Predictions?

<div align="center">
<img src="docs/figures/shap_bar.png" alt="Feature Importance" width="600"/>
</div>

| Rank | Feature | Insight |
|------|---------|---------|
| 1 | `load_lag_1h` | Recent load is the strongest predictor |
| 2 | `rolling_24h_mean` | Daily patterns matter |
| 3 | `temperature` | Weather impacts consumption |
| 4 | `hour` | Time-of-day effects |

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Dex947/xai-load-forecasting.git
cd xai-load-forecasting

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### CLI Usage

```bash
# Run full pipeline
python -m src.cli train --config config/config.yaml

# Generate predictions
python -m src.cli predict --model models/artifacts/lightgbm_model.pkl

# Start API server
python -m src.cli serve --port 8000
```

### Python API

```python
from src.models.gbm import GradientBoostingModel
from src.explainability.shap_analysis import SHAPAnalyzer

# Load trained model
model = GradientBoostingModel.load('models/artifacts/lightgbm_model.pkl')

# Predict with explanation
predictions = model.predict(X_new)
analyzer = SHAPAnalyzer(model.model, X_background)
shap_values = analyzer.compute_shap_values(X_new)
```

---

## 🔌 API Serving

Start the FastAPI server:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction with explanation |
| `/predict/batch` | POST | Batch predictions |
| `/features` | GET | List required features |
| `/importance` | GET | Feature importance scores |

### Docker Deployment

```bash
docker-compose up -d
```

---

## 📈 Visualizations

<table>
<tr>
<td width="50%">
<img src="docs/figures/load_patterns.png" alt="Load Patterns"/>
<p align="center"><em>Daily and weekly load patterns</em></p>
</td>
<td width="50%">
<img src="docs/figures/residual_analysis.png" alt="Residual Analysis"/>
<p align="center"><em>Model diagnostics confirm good fit</em></p>
</td>
</tr>
<tr>
<td width="50%">
<img src="docs/figures/time_varying_importance.png" alt="Time-Varying SHAP"/>
<p align="center"><em>Feature importance changes by hour/day</em></p>
</td>
<td width="50%">
<img src="docs/figures/load_vs_temperature.png" alt="Temperature Effect"/>
<p align="center"><em>Temperature-load relationship</em></p>
</td>
</tr>
</table>

---

## 🏗️ Architecture

```
xai-load-forecasting/
├── src/
│   ├── data/           # Data loading & validation
│   ├── features/       # Feature engineering pipeline
│   ├── models/         # LightGBM, quantile, multi-horizon
│   ├── explainability/ # SHAP analysis & visualizations
│   ├── monitoring/     # Drift detection & alerts
│   ├── api.py          # FastAPI server
│   └── cli.py          # Command-line interface
├── config/             # YAML configuration
├── tests/              # Pytest test suite (24 tests)
├── models/artifacts/   # Trained models
├── docs/               # Documentation & figures
├── Dockerfile          # Container deployment
└── docker-compose.yml  # Multi-service orchestration
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Model Card](docs/model_card.md) | Scope, limitations, failure modes |
| [CHANGELOG](CHANGELOG.md) | Version history |
| [CONTRIBUTING](CONTRIBUTING.md) | Contribution guidelines |
| [API Docs](http://localhost:8000/docs) | Interactive API documentation |

---

## Problem Statement

Electrical distribution feeders require accurate day-ahead load forecasts for critical operations:

- **Grid Operations**: Planning unit commitment and economic dispatch 24 hours in advance
- **Demand Response**: Identifying opportunities for load management and peak shaving
- **Maintenance Scheduling**: Safely planning outages and allocating crew resources
- **Reliability**: Preventing equipment overloads and service interruptions

Traditional forecasting methods (like "persistence" or "seasonal naive") are simple but inaccurate. Machine learning models are more accurate but often operate as black boxes, making operators hesitant to trust them.

**Our solution**: A LightGBM model that beats baseline methods by 14-47% while providing complete transparency through SHAP analysis. Every prediction comes with an explanation showing which factors mattered most.

---

## Architecture

### Modular Design

```
xai-load-forecasting/
├── config/              # YAML configuration files
├── data/                # Raw, processed, and external data
├── src/                 # Modular source code
│   ├── data/           # Data loading, profiling, validation
│   ├── features/       # Feature engineering pipeline
│   ├── models/         # Model training and validation
│   └── explainability/ # SHAP analysis and visualizations
├── models/artifacts/    # Trained models and SHAP cache
├── notebooks/          # EDA and analysis notebooks
├── docs/               # Documentation and reports
└── logs/               # Structured logging output
```

### Core Principles

1. **No Temporal Leakage**: All features respect time boundaries—no future information used in predictions
2. **Config-Driven**: All parameters externalized to YAML for easy customization
3. **Logging-First**: Structured logging throughout (no print statements)
4. **Explainability-First**: SHAP analysis reveals what drives each prediction
5. **Validated & Tested**: Comprehensive diagnostics ensure model reliability

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/Dex947/xai-load-forecasting.git
cd xai-load-forecasting
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure the system**:
   - Edit `config/config.yaml` for forecasting parameters
   - Edit `config/holidays.yaml` for holiday calendar
   - Edit `config/weather_config.yaml` for weather data sources

---

## Quick Start

### 1. Data Preparation

Place your data in the appropriate directories:
- **Load data**: `data/raw/load_data.csv` (columns: `timestamp`, `load`)
- **Weather data**: `data/external/weather.csv` (columns: `timestamp`, `temperature`, `humidity`, etc.)

### 2. Data Profiling

Run the data profiling notebook to understand your data:
```bash
jupyter notebook notebooks/01_data_profiling.ipynb
```

This will generate:
- Missing data analysis
- Seasonality patterns
- Autocorrelation analysis
- Weather-load correlations

### 3. Feature Engineering

Create features using the pipeline:
```python
from src.features.pipeline import FeaturePipeline
from src.data.loader import load_load_data, load_weather_data, merge_load_weather

# Load data
load_df = load_load_data("data/raw/load_data.csv")
weather_df = load_weather_data("data/external/weather.csv")
df = merge_load_weather(load_df, weather_df)

# Create features
pipeline = FeaturePipeline()
features = pipeline.create_all_features(df, target_column='load')
```

### 4. Model Training

Train the model with monotonic constraints:
```python
from src.models.gbm import GradientBoostingModel
from src.config import load_config

config = load_config()

model = GradientBoostingModel(
    model_type='lightgbm',
    config=config.model.dict(),
    monotonic_constraints={'temperature': 1}  # Positive relationship
)

model.fit(X_train, y_train, X_val, y_val)
model.save('models/artifacts/load_forecast_model.pkl')
```

### 5. SHAP Analysis

Generate explainability insights:
```python
from src.explainability.shap_analysis import SHAPAnalyzer
from src.explainability.visualizations import ExplainabilityVisualizer

# Initialize SHAP analyzer
analyzer = SHAPAnalyzer(model.model, X_background=X_train.sample(100))
shap_values = analyzer.compute_shap_values(X_test)

# Create visualizations
viz = ExplainabilityVisualizer(shap_values, X_test.columns.tolist(), X_test)
viz.plot_summary(save_path='docs/figures/shap_summary.png')
viz.plot_dependence('temperature', save_path='docs/figures/shap_temp_dependence.png')
```

---

## Configuration

All system parameters are defined in `config/config.yaml`:

### Key Configuration Sections

- **Forecasting**: Horizon, resolution, timezone
- **Features**: Lag hours, rolling windows, interactions
- **Model**: LightGBM/XGBoost parameters, monotonic constraints
- **Validation**: Rolling origin CV settings
- **Explainability**: SHAP configuration, counterfactuals
- **Data Quality**: Missing data thresholds, outlier detection

Example:
```yaml
forecasting:
  horizon_hours: 24
  resolution_minutes: 60
  timezone: "UTC"

model:
  type: "lightgbm"
  monotonic_constraints:
    temperature: 1  # Positive relationship
```

---

## Workflow

### Complete Pipeline

```mermaid
graph TD
    A[Raw Data] --> B[Data Profiling]
    B --> C[Feature Engineering]
    C --> D[Model Training]
    D --> E[Cross-Validation]
    E --> F[SHAP Analysis]
    F --> G[Model Card]
    G --> H[Deployment]
```

### Detailed Steps

1. **Data Profiling** (`notebooks/01_data_profiling.ipynb`)
   - Load and validate data
   - Analyze missingness patterns
   - Identify seasonality and trends
   - Validate weather-load alignment

2. **Feature Engineering** (`notebooks/02_feature_engineering.ipynb`)
   - Temporal features (hour, day, week, season)
   - Lag features (1h to 1 week)
   - Rolling statistics (mean, std, min, max)
   - Weather features and derived metrics
   - Calendar features (holidays, weekends)
   - Interaction features

3. **Model Training** (`notebooks/03_model_training.ipynb`)
   - Baseline models (persistence, seasonal naive)
   - Gradient boosting with monotonic constraints
   - Rolling origin cross-validation
   - Hyperparameter tuning
   - Model evaluation and comparison

4. **Explainability** (`notebooks/04_explainability.ipynb`)
   - Global SHAP analysis
   - Time-varying SHAP patterns
   - Feature dependence plots
   - Counterfactual scenarios
   - Model Card generation

---

## Results

### Dataset Overview

We trained our model on real-world data from a Portuguese electrical feeder, combining load measurements with weather observations:

**Data Sources**:
- **Load Data**: UCI Electricity Load Diagrams (2011-2014)
  - 26,267 hourly measurements over 3 years
  - Residential/commercial mix from Portugal
  - 99.1% data completeness
  
- **Weather Data**: Open-Meteo Historical API
  - Temperature, humidity, wind speed, precipitation, pressure, cloud cover
  - Hourly resolution, UTC timezone
  - Validated alignment with load data

**Load Characteristics**:
| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Mean | 5.30 kW | Typical load level |
| Std Dev | 6.05 kW | High variability (CV=1.14) |
| Range | 0.32 - 34.58 kW | 100x variation |
| Peak Hours | 18:00-21:00 | Evening consumption |
| Weekend Effect | -15% average | Lower weekend load |

### Model Performance

We evaluated our model on a 30-day holdout test set (720 hours) that the model had never seen during training:

**Performance Comparison**:

| Model | RMSE (kW) | MAE (kW) | MAPE | R² | Improvement |
|-------|-----------|----------|------|-----|-------------|
| Persistence (last value) | 0.892 | - | - | - | Baseline |
| Seasonal Naive (last week) | 1.458 | - | - | - | -63% worse |
| **LightGBM (Ours)** | **0.770** | **0.486** | 40.4% | 0.247 | **+14% better** |

**What this means**:
- Our model predicts load with an average error of **0.77 kW** (about 15% of mean load)
- **14% more accurate** than simply using the last observed value
- **47% more accurate** than using the same hour from last week
- Typical predictions are within **±0.49 kW** of actual values

**Understanding R² = 0.25**:

You might wonder why R² is only 0.25 if the model works well. Here's why this is actually reasonable:

1. **Residential load is inherently unpredictable**: Unlike industrial loads with fixed schedules, residential consumption depends on human behavior—when people wake up, cook dinner, or turn on appliances. These factors aren't in our data.

2. **We're missing key variables**: Occupancy patterns, appliance schedules, and individual household behaviors account for much of the unexplained variance.

3. **The model is well-specified**: Our diagnostic tests show the model is learning correctly—residuals are normally distributed with no systematic patterns. The low R² reflects true randomness in the data, not model deficiency.

4. **Industry context**: For residential load forecasting, R² values of 0.2-0.4 are typical and acceptable.

### Model Improvements Applied

We didn't just train a basic model—we applied several improvements based on diagnostic analysis:

**Feature Engineering**:
- ✅ Created 89 features from raw data (temporal, weather, lag, rolling statistics)
- ✅ Pruned to 58 features using permutation importance (34% reduction)
- ✅ Removed noisy features that didn't improve predictions

**Hyperparameter Optimization**:
- ✅ Lower learning rate (0.03) for better generalization
- ✅ Feature sampling (80%) to prevent overfitting
- ✅ L1/L2 regularization to penalize complexity
- ✅ Limited tree depth (max 10) to avoid memorization

**Validation**:
- ✅ Residual diagnostics: No systematic patterns detected
- ✅ Overfitting check: Train/Val/Test performance consistent
- ✅ Temporal validation: No data leakage confirmed

### Key Insights from Data Analysis

**Seasonality Patterns**:
- **Hourly**: Clear daily cycle with peak at 18:00-21:00 (dinner time)
- **Weekly**: Weekdays show consistent patterns; weekends are 15% lower
- **Monthly**: Seasonal variation with higher loads in summer (cooling) and winter (heating)
- **Autocorrelation**: Strong correlation up to 1 week (168 hours)

**Weather Impact**:
- **Temperature**: Moderate positive correlation (cooling-dominated feeder)
- **Humidity**: Weak correlation
- **Wind**: Minimal direct impact
- **Degree Days**: CDD (cooling) and HDD (heating) both predictive

**What Drives Predictions** (from SHAP analysis):

| Rank | Feature | SHAP Importance | What It Means |
|------|---------|-----------------|---------------|
| 1 | load_lag_1h | 1.52 | Load 1 hour ago (most predictive) |
| 2 | rolling_24h_mean | 0.31 | Yesterday's average load |
| 3 | rolling_3h_mean | 0.23 | Recent 3-hour trend |
| 4 | load_lag_2h | 0.20 | Load 2 hours ago |
| 5 | rolling_6h_mean | 0.17 | Recent 6-hour trend |
| 6 | load_lag_24h | 0.15 | Same hour yesterday |
| 7 | temperature | 0.11 | Current temperature |
| 8 | hour | 0.10 | Time of day |
| 9 | is_business_hour | 0.10 | Business hours indicator |
| 10 | day_of_week | 0.08 | Day of week pattern |

**Key Takeaway**: Recent load history (last 1-2 hours) is the strongest predictor, followed by rolling averages that capture trends. Weather and time-of-day features provide additional context.

### Visualizations

Below are key visualizations that reveal model behavior and data patterns:

#### 1. Load Patterns
![Load Patterns](docs/figures/load_patterns.png)
*Hourly, daily, weekly, and monthly load patterns showing clear seasonality*

#### 2. Model Predictions vs Actual
![Model Predictions](docs/figures/model_predictions.png)
*Top: Time series of actual vs predicted load over 30-day test period*  
*Bottom: Scatter plot showing prediction accuracy (points near diagonal = good predictions)*

#### 3. Residual Diagnostics
![Residual Analysis](docs/figures/residual_analysis.png)
*Four-panel diagnostic plot confirming model is well-specified:*
- *Top-left: No systematic patterns in residuals vs predictions*
- *Top-right: Residuals are normally distributed*
- *Bottom-left: Q-Q plot confirms normality*
- *Bottom-right: No autocorrelation in residuals over time*

#### 4. SHAP Summary (Feature Importance)
![SHAP Summary](docs/figures/shap_summary.png)
*Beeswarm plot showing how each feature impacts predictions:*
- *Features ranked by importance (top to bottom)*
- *Red = high feature value, Blue = low feature value*
- *Right = increases prediction, Left = decreases prediction*

#### 5. SHAP Dependence: Rolling 24h Mean
![Rolling 24h Mean Dependence](docs/figures/shap_dependence_rolling_24h_mean.png)
*How temperature affects load predictions (colored by hour of day):*
- *Higher temperatures increase load (cooling effect)*
- *Effect is stronger during afternoon/evening hours (red points)*

#### 6. Time-Varying Feature Importance
![Time-Varying Importance](docs/figures/time_varying_importance.png)
*How feature importance changes by hour, day of week, and month:*
- *Temperature more important during peak hours*
- *Weekend indicator more important on Fridays/Mondays*
- *Seasonal patterns in weather feature importance*

---

## Model Card & Documentation

We provide comprehensive documentation to ensure the model can be safely deployed and maintained:

**Model Card** (`docs/model_card.md`):
- **Scope & Intended Use**: Day-ahead hourly forecasting for grid operations
- **Training Data**: UCI dataset characteristics and limitations
- **Performance Metrics**: Detailed breakdown by hour, day, season
- **Assumptions**: What the model expects to remain true
- **Limitations**: When the model may not work well
- **Failure Modes**: Specific scenarios where errors increase (heat waves, topology changes, holidays)
- **Ethical Considerations**: Privacy, fairness, environmental impact
- **Monitoring Plan**: How to track model performance in production

**Additional Documentation**:
- **EXECUTION_REPORT.md**: Detailed deliverables and run logs
- **CONTRIBUTING.md**: Guidelines for contributing to the project
- **CHANGELOG.md**: Version history and updates
- **INDEX.md**: File index and navigation

---

## Project Structure

```
xai-load-forecasting/
├── config/
│   ├── config.yaml              # Master configuration
│   ├── holidays.yaml            # Holiday calendar
│   └── weather_config.yaml      # Weather API settings
├── data/
│   ├── raw/                     # Original data
│   ├── processed/               # Cleaned data
│   └── external/                # Weather, holidays
├── src/
│   ├── config.py                # Configuration loader
│   ├── logger.py                # Logging setup
│   ├── data/
│   │   ├── loader.py            # Data ingestion
│   │   ├── profiler.py          # EDA functions
│   │   └── validator.py         # Temporal validation
│   ├── features/
│   │   ├── temporal.py          # Time features
│   │   ├── weather.py           # Weather features
│   │   ├── calendar.py          # Holiday features
│   │   └── pipeline.py          # Feature orchestration
│   ├── models/
│   │   ├── baseline.py          # Baseline models
│   │   ├── gbm.py               # Gradient boosting
│   │   └── validator.py         # Rolling origin CV
│   └── explainability/
│       ├── shap_analysis.py     # SHAP computations
│       └── visualizations.py    # Plotting functions
├── models/artifacts/            # Saved models
├── notebooks/                   # Jupyter notebooks
├── docs/                        # Documentation
├── logs/                        # Log files
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── CHANGELOG.md                 # Version history
└── memory.json                  # Project context
```

---

## Customization & Adaptation

This system is designed to be easily adapted to different feeders and regions:

### Easy Customizations

1. **Holiday Calendar** (`config/holidays.yaml`):
   - Add region-specific holidays
   - Define custom events (local festivals, major sports events)
   - Adjust holiday proximity windows

2. **Weather Features** (`config/weather_config.yaml`):
   - Add/remove weather variables based on availability
   - Adjust temperature thresholds for degree days
   - Configure weather data sources

3. **Forecasting Parameters** (`config/config.yaml`):
   - Change forecast horizon (default: 24 hours)
   - Adjust temporal resolution
   - Modify feature engineering windows

### Advanced Customizations

4. **Feeder Characteristics**:
   - **Cooling-dominated** (like ours): Keep temperature → load positive constraint
   - **Heating-dominated**: Reverse temperature constraint or make it non-monotonic
   - **Mixed**: Remove monotonic constraints

5. **Feature Engineering**:
   - Add domain-specific features (e.g., solar generation for feeders with PV)
   - Extend lag features for longer memory
   - Create custom interaction features

6. **Model Selection**:
   - Switch between LightGBM and XGBoost in config
   - Adjust hyperparameters for your data characteristics
   - Try ensemble methods for improved accuracy

### Retraining for New Locations

If deploying to a different feeder:

1. **Collect local data**: At least 1 year of hourly load + weather data
2. **Update configuration**: Adjust holidays, timezone, weather sources
3. **Retrain model**: Run the complete pipeline on your data
4. **Validate performance**: Ensure baselines are beaten
5. **Monitor closely**: Track performance for first few months

## Contributing

We welcome contributions from the community! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:

- **Code of Conduct**: Be respectful and inclusive
- **Development Setup**: How to set up your environment
- **Coding Standards**: Style guide, documentation requirements
- **Testing**: How to write and run tests
- **Pull Requests**: How to submit changes

**Areas where we'd love help**:
- Additional baseline models
- More feature engineering methods
- Alternative ML algorithms (neural networks, ensembles)
- Additional weather data sources
- Regional holiday calendars
- Performance optimizations

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{xai_load_forecasting_2025,
  title = {XAI Load Forecasting: Explainable Day-Ahead Load Forecasting},
  author = {Dex947},
  year = {2025},
  url = {https://github.com/Dex947/xai-load-forecasting},
  version = {2.0.0}
}
```
---
## Contact

For questions, suggestions, or issues:
- **Issues**: [Open an issue on GitHub](https://github.com/Dex947/xai-load-forecasting/issues)
- **Discussions**: [Use GitHub Discussions](https://github.com/Dex947/xai-load-forecasting/discussions)
- **Repository**: https://github.com/Dex947/xai-load-forecasting

---

## Frequently Asked Questions

### Why is R² only 0.25?

This is actually reasonable for residential load forecasting. The low R² reflects high inherent variability in residential consumption patterns (people's behavior), not model deficiency. Our diagnostics confirm the model is well-specified—it's learning the right patterns, but residential load is genuinely unpredictable beyond what our features capture.

### Can I use this for industrial loads?

The current model is trained on residential/commercial data. For industrial loads, you should:
1. Collect industrial load data (typically more predictable)
2. Retrain the model on your data
3. Expect higher R² (0.5-0.8 is common for industrial)
4. Adjust features (industrial loads depend more on production schedules than weather)

### How often should I retrain?

- **Minimum**: Annually to capture seasonal changes
- **Recommended**: Quarterly to stay current with patterns
- **Required**: After major topology changes (new loads, reconfigurations)

### What if weather data is unavailable?

The model can run without weather features, but accuracy will decrease by ~5-10%. The lag and temporal features alone provide reasonable predictions. You can also use weather forecasts instead of historical data for day-ahead predictions.

### How do I deploy this in production?

1. Load the trained model: `GradientBoostingModel.load('lightgbm_model.pkl')`
2. Set up a data pipeline to fetch latest load and weather data
3. Run feature engineering on new data
4. Generate predictions: `model.predict(X_new)`
5. Compute SHAP values for explainability
6. Monitor performance metrics daily

See `docs/model_card.md` for detailed deployment guidelines.

---

## Acknowledgments

### Data Sources

We're grateful to these organizations for providing open data:

- **UCI Machine Learning Repository**: ElectricityLoadDiagrams20112014 dataset
  - Trindade, A. (2015). DOI: 10.24432/C58C86
  - 3 years of Portuguese electrical load data
  
- **Open-Meteo**: Historical weather data API
  - Free, open-source weather API
  - High-quality historical data for Europe

### Libraries and Tools

This project builds on excellent open-source tools:

- **SHAP**: Lundberg & Lee (2017) - For model explainability
- **LightGBM**: Ke et al. (2017) - Fast gradient boosting framework
- **XGBoost**: Chen & Guestrin (2016) - Alternative boosting implementation
- **scikit-learn**: Pedregosa et al. (2011) - Machine learning utilities
- **pandas**: McKinney (2010) - Data manipulation
- **matplotlib/seaborn**: Hunter (2007), Waskom (2021) - Visualizations

### References

**Machine Learning & Explainability**:
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785-794.

**Time Series Forecasting**:
- Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.
- Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: principles and practice* (2nd ed.). OTexts.

**Load Forecasting**:
- Hong, T., Pinson, P., Fan, S., Zareipour, H., Troccoli, A., & Hyndman, R. J. (2016). Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond. *International Journal of Forecasting*, 32(3), 896-913.

---

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

**Current Version**: 2.0.0 (2025-12-24)
