# Usage Guide

## CLI Commands

```bash
# Data profiling
python -m src.cli profile --config config/config.yaml

# Feature engineering
python -m src.cli features --config config/config.yaml

# Model training
python -m src.cli train --config config/config.yaml

# SHAP analysis
python -m src.cli explain --model models/artifacts/lightgbm_model.pkl

# Generate predictions
python -m src.cli predict --model models/artifacts/lightgbm_model.pkl

# Run tests
python -m src.cli test

# Start API server
python -m src.cli serve --port 8000
```

---

## Python API

### Data Loading

```python
from src.data.loader import load_load_data, load_weather_data, merge_load_weather

load_df = load_load_data("data/raw/load_data.csv")
weather_df = load_weather_data("data/external/weather.csv")
df = merge_load_weather(load_df, weather_df)
```

### Feature Engineering

```python
from src.features.pipeline import FeaturePipeline

pipeline = FeaturePipeline()
features = pipeline.create_all_features(df, target_column='load')
```

### Model Training

```python
from src.models.gbm import GradientBoostingModel
from src.config import load_config

config = load_config()
model = GradientBoostingModel(
    model_type='lightgbm',
    config=config.model.dict(),
    monotonic_constraints={'temperature': 1}
)
model.fit(X_train, y_train, X_val, y_val)
model.save('models/artifacts/model.pkl')
```

### SHAP Analysis

```python
from src.explainability.shap_analysis import SHAPAnalyzer
from src.explainability.visualizations import ExplainabilityVisualizer

analyzer = SHAPAnalyzer(model.model, X_background=X_train.sample(100))
shap_values = analyzer.compute_shap_values(X_test)

viz = ExplainabilityVisualizer(shap_values, X_test.columns.tolist(), X_test)
viz.plot_summary(save_path='docs/figures/shap_summary.png')
```

### Prediction Intervals

```python
from src.models.quantile import QuantileForecaster

forecaster = QuantileForecaster(quantiles=[0.1, 0.5, 0.9])
forecaster.fit(X_train, y_train)
intervals = forecaster.predict(X_test)
```

### Multi-Horizon Forecasting

```python
from src.models.multi_horizon import MultiHorizonForecaster

forecaster = MultiHorizonForecaster(horizons=[1, 6, 12, 24, 48, 168])
forecaster.fit(X_train, y_train)
predictions = forecaster.predict(X_test)
```

### Drift Detection

```python
from src.monitoring.drift import DriftDetector, PerformanceMonitor

detector = DriftDetector(reference_data=X_train)
drift_report = detector.detect_drift(X_new)

monitor = PerformanceMonitor(baseline_metrics={'rmse': 0.77})
monitor.update(y_true, y_pred)
alerts = monitor.check_alerts()
```

---

## Configuration

All parameters in `config/config.yaml`:

```yaml
forecasting:
  horizon_hours: 24
  resolution_minutes: 60
  timezone: "UTC"

model:
  type: "lightgbm"
  monotonic_constraints:
    temperature: 1

validation:
  method: "rolling_origin"
  n_splits: 5
  test_size_days: 30
```

---

## Docker Deployment

```bash
# Build and run
docker-compose up -d

# Check health
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {...}}'
```
