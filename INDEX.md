# XAI Load Forecasting - Project Index

**Quick reference guide to all project files and documentation**

---

## 📖 Documentation (Start Here)

| File | Purpose | Priority |
|------|---------|----------|
| **[README.md](README.md)** | Full documentation | ⭐⭐⭐ |
| **[CHANGELOG.md](CHANGELOG.md)** | Version history | ⭐ |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | Contribution guidelines | ⭐ |
| **[LICENSE](LICENSE)** | MIT License | ⭐ |

---

## 💻 Source Code

### Core Utilities
- `src/config.py` - Configuration management (Pydantic)
- `src/logger.py` - Structured logging framework

### Data Module (`src/data/`)
- `loader.py` - Data ingestion and merging
- `validator.py` - Temporal validation and quality checks
- `profiler.py` - EDA and statistical profiling

### Features Module (`src/features/`)
- `temporal.py` - Temporal features (hour, day, week, cyclical)
- `calendar.py` - Holiday and calendar features
- `weather.py` - Weather features and derived metrics
- `pipeline.py` - Feature orchestration

### Models Module (`src/models/`)
- `baseline.py` - Baseline models (persistence, seasonal naive)
- `gbm.py` - Gradient boosting with monotonic constraints
- `validator.py` - Rolling origin cross-validation

### Explainability Module (`src/explainability/`)
- `shap_analysis.py` - SHAP value computation
- `visualizations.py` - Explainability visualizations

---

## ⚙️ Configuration

| File | Purpose |
|------|---------|
| `config/config.yaml` | Master configuration (forecasting, features, model) |
| `config/holidays.yaml` | Holiday calendar and special events |
| `config/weather_config.yaml` | Weather data sources and validation |

---

## 🔧 Scripts

| File | Purpose | Usage |
|------|---------|-------|
| `scripts/download_data.py` | Download UCI + weather data | `python scripts/download_data.py` |
| `scripts/run_data_profiling.py` | Generate EDA and visualizations | `python scripts/run_data_profiling.py` |

---

## 📊 Data

### Raw Data (`data/raw/`)
- `load_data.csv` - Processed load data (26,267 hours, 1 MB)
- `sample_data_2013.csv` - Sample dataset for testing (8,760 hours, 603 KB)

### External Data (`data/external/`)
- `weather.csv` - Historical weather data (26,328 hours, 1.3 MB)

### Processed Data (`data/processed/`)
- Ready for feature engineering outputs

---

## 📈 Visualizations (`docs/figures/`)

1. `load_patterns.png` - Hourly, daily, weekly, monthly patterns
2. `autocorrelation.png` - ACF and PACF plots
3. `load_vs_temperature.png` - Temperature-load correlation
4. `load_vs_humidity.png` - Humidity-load correlation
5. `load_vs_wind_speed.png` - Wind-load correlation

---

## 📄 Analysis Outputs (`docs/`)

- `data_summary_statistics.csv` - Statistical summary table
- `data_profile.json` - Complete data profile (JSON)

---

## 🎯 Project Metadata

- `memory.json` - Project context and decisions
- `requirements.txt` - Python dependencies
- `.gitignore` - Git exclusions

---

## 📦 Directory Structure

```
xai-load-forecasting/
├── 📖 Documentation (9 files)
│   ├── README_FIRST.md ⭐⭐⭐
│   ├── FINAL_SUMMARY.md ⭐⭐⭐
│   ├── EXECUTION_REPORT.md ⭐⭐
│   ├── README.md ⭐⭐⭐
│   ├── PROJECT_STATUS.md
│   ├── COMPLETION_SUMMARY.md
│   ├── CHANGELOG.md
│   ├── CONTRIBUTING.md
│   └── LICENSE
│
├── 💻 Source Code (17 modules, 4,173 lines)
│   ├── src/
│   │   ├── config.py
│   │   ├── logger.py
│   │   ├── data/ (3 files)
│   │   ├── features/ (4 files)
│   │   ├── models/ (3 files)
│   │   └── explainability/ (2 files)
│   └── scripts/ (2 files)
│
├── ⚙️ Configuration (3 files)
│   └── config/
│       ├── config.yaml
│       ├── holidays.yaml
│       └── weather_config.yaml
│
├── 📊 Data (3 datasets, 2.9 MB)
│   ├── data/raw/ (2 files)
│   └── data/external/ (1 file)
│
├── 📈 Outputs
│   └── docs/
│       ├── figures/ (5 visualizations)
│       ├── data_summary_statistics.csv
│       └── data_profile.json
│
└── 🎯 Metadata
    ├── memory.json
    ├── requirements.txt
    └── .gitignore

Total: 53 files
```

---

## 🚀 Quick Commands

### View Project Status
```bash
cat README_FIRST.md
cat PROJECT_STATUS.md
```

### View Data Profile
```bash
ls docs/figures/
cat docs/data_summary_statistics.csv
```

### Run Next Phase
```python
# Phase 3: Feature Engineering
python scripts/run_feature_engineering.py

# Phase 4: Model Training
python scripts/run_model_training.py

# Phase 5: SHAP Analysis
python scripts/run_shap_analysis.py
```

---

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| Total Files | 53 |
| Python Code | 4,173 lines |
| Documentation | 2,000+ lines |
| Modules | 17 |
| Scripts | 2 |
| Config Files | 3 |
| Datasets | 3 (26,267 hours) |
| Visualizations | 5 |
| Data Size | 2.9 MB |

---

**Last Updated**: 2025-10-07  
**Version**: 1.0.0  
**Status**: Ready for Training
