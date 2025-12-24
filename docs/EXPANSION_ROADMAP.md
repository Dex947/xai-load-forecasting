# XAI Load Forecasting - Expansion Roadmap v3.0

## Overview

This document outlines the expansion of the XAI Load Forecasting system from v2.0 to v3.0, adding:
1. **Data Expansion**: Multi-resolution, longer horizons, new data sources
2. **Probabilistic Forecasting**: Full distribution predictions with conformal prediction
3. **Online Learning**: Continuous model updates with River
4. **MLOps**: MLflow tracking, automated retraining, A/B testing
5. **Explainability Extensions**: Counterfactuals, concept drift explanation, NL summaries

---

## Phase 1: Data Expansion

### 1.1 Multi-Resolution Support
- Keep hourly as primary resolution
- Add 15-minute and daily aggregation options
- Implement resolution-aware feature engineering

### 1.2 Extended Horizons
- Current: 24h (day-ahead)
- Add: 48h, 168h (week-ahead), 720h (month-ahead)
- Use direct multi-horizon strategy (already implemented)

### 1.3 New Data Sources
| Source | Data Type | API |
|--------|-----------|-----|
| Open-Meteo | Solar irradiance (GHI, DNI, DHI) | Free |
| Open-Meteo | UV index, precipitation probability | Free |
| Calendar | School holidays, local events | holidays lib |
| Grid | Historical outages, topology changes | Manual/CSV |

### 1.4 Implementation
- `src/data/solar.py` - Solar irradiance fetcher
- `src/data/sources.py` - Unified data source manager
- Update `config/config.yaml` with new data source options

---

## Phase 2: Probabilistic Forecasting

### 2.1 Conformal Prediction with MAPIE
- Use `MapieTimeSeriesRegressor` with EnbPI method
- Provides distribution-free prediction intervals
- Guaranteed coverage at specified confidence level

### 2.2 Implementation
```python
from mapie.regression import MapieTimeSeriesRegressor
from mapie.subsample import BlockBootstrap

cv = BlockBootstrap(n_resamplings=30, n_blocks=10, overlapping=False)
mapie = MapieTimeSeriesRegressor(model, method='enbpi', cv=cv)
mapie.fit(X_train, y_train)
y_pred, y_intervals = mapie.predict(X_test, alpha=0.05)
```

### 2.3 Metrics
- Coverage score (target: 95%)
- Mean interval width (narrower = better)
- Continuous Ranked Probability Score (CRPS)

### 2.4 Files
- `src/models/conformal.py` - Conformal prediction wrapper
- `src/models/probabilistic.py` - Unified probabilistic interface

---

## Phase 3: Online Learning

### 3.1 River Integration
- Incremental learning without full retraining
- Concept drift detection built-in
- Memory-efficient for streaming data

### 3.2 Architecture
```
New Data → Drift Detection → Online Update → Model Registry
              ↓
         Alert if drift
```

### 3.3 Implementation
```python
from river import linear_model, preprocessing, metrics
from river.drift import ADWIN

model = preprocessing.StandardScaler() | linear_model.LinearRegression()
drift_detector = ADWIN()

for x, y in stream:
    y_pred = model.predict_one(x)
    model.learn_one(x, y)
    drift_detector.update(abs(y - y_pred))
    if drift_detector.drift_detected:
        trigger_alert()
```

### 3.4 Hybrid Approach
- Keep batch LightGBM as primary model
- Use River for real-time updates between retraining
- Ensemble predictions from both

### 3.5 Files
- `src/models/online.py` - Online learning wrapper
- `src/models/hybrid.py` - Batch + online ensemble

---

## Phase 4: MLOps

### 4.1 MLflow Integration
- Experiment tracking
- Model registry with versioning
- Artifact storage (models, SHAP values, plots)

### 4.2 Automated Retraining Pipeline
```
Schedule (weekly) → Data Validation → Feature Engineering
                          ↓
                    Model Training → Evaluation → Registry
                          ↓
                    A/B Comparison → Promote/Rollback
```

### 4.3 A/B Testing
- Champion/Challenger model comparison
- Statistical significance testing
- Automatic promotion based on metrics

### 4.4 Files
- `src/mlops/tracking.py` - MLflow experiment tracking
- `src/mlops/registry.py` - Model registry operations
- `src/mlops/pipeline.py` - Automated retraining pipeline
- `src/mlops/ab_testing.py` - A/B testing framework

---

## Phase 5: Explainability Extensions

### 5.1 Counterfactual Explanations (DiCE)
- "What-if" scenarios for operators
- "What would need to change for load to be X?"
- Actionable insights

### 5.2 Concept Drift Explanation
- Track SHAP importance over time
- Detect when feature importance changes significantly
- Explain WHY model performance degraded

### 5.3 Natural Language Explanations
- Generate human-readable summaries
- "Load is high because temperature increased by 5°C and it's a weekday evening"
- Template-based with SHAP values

### 5.4 Files
- `src/explainability/counterfactuals.py` - DiCE integration
- `src/explainability/drift_explanation.py` - Concept drift analysis
- `src/explainability/natural_language.py` - NL generation

---

## Dependencies to Add

```
# requirements.txt additions
mapie>=0.8.0           # Conformal prediction
river>=0.21.0          # Online learning
mlflow>=2.10.0         # Experiment tracking
prefect>=2.14.0        # Workflow orchestration (optional)
```

---

## Implementation Order

1. **Week 1**: Data expansion (solar, extended config)
2. **Week 2**: Probabilistic forecasting (MAPIE integration)
3. **Week 3**: Online learning (River integration)
4. **Week 4**: MLOps (MLflow tracking and registry)
5. **Week 5**: Explainability extensions
6. **Week 6**: Testing, documentation, deployment

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Prediction interval coverage | N/A | 95% |
| Model update latency | Manual | < 1 hour |
| Drift detection time | N/A | < 24 hours |
| Experiment reproducibility | Partial | 100% |
| Counterfactual generation | N/A | < 5 seconds |
