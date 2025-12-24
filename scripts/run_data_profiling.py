"""Data profiling and EDA script."""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.logger import setup_logging_from_config, get_logger
from src.config import load_config
from src.data.loader import load_load_data, load_weather_data, merge_load_weather
from src.data.profiler import DataProfiler
from src.data.validator import validate_data_quality, check_temporal_consistency

# Setup logging
setup_logging_from_config()
logger = get_logger(__name__)

# Setup paths
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("DATA PROFILING AND EDA")
    logger.info("=" * 80)

    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")

        # Step 1: Load data
        logger.info("\n[Step 1/6] Loading data...")
        load_df = load_load_data(
            str(DATA_RAW / "load_data.csv"), date_column="timestamp", load_column="load"
        )

        weather_df = load_weather_data(
            str(DATA_EXTERNAL / "weather.csv"), date_column="timestamp"
        )

        logger.info(f"Load data: {load_df.shape}")
        logger.info(f"Weather data: {weather_df.shape}")

        # Step 2: Merge datasets
        logger.info("\n[Step 2/6] Merging load and weather data...")
        df = merge_load_weather(load_df, weather_df, validate_alignment=True)
        logger.info(f"Merged data: {df.shape}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

        # Step 3: Data quality validation
        logger.info("\n[Step 3/6] Validating data quality...")
        quality_metrics = validate_data_quality(
            df,
            max_missing_ratio=config.data_quality.max_missing_ratio,
            outlier_std_threshold=config.data_quality.outlier_std_threshold,
            min_data_points=config.data_quality.min_data_points,
        )

        logger.info(
            f"Data quality validation: {'PASSED' if quality_metrics['validation_passed'] else 'FAILED'}"
        )
        if quality_metrics["issues"]:
            logger.warning(f"Issues found: {len(quality_metrics['issues'])}")
            for issue in quality_metrics["issues"][:5]:  # Show first 5
                logger.warning(f"  - {issue}")

        # Step 4: Temporal consistency check
        logger.info("\n[Step 4/6] Checking temporal consistency...")
        consistency_metrics = check_temporal_consistency(df, expected_freq="h")
        logger.info("Temporal consistency:")
        logger.info(f"  Duplicates: {consistency_metrics['duplicates']}")
        logger.info(f"  Missing timestamps: {consistency_metrics['n_gaps']}")
        logger.info(
            f"  Irregular intervals: {consistency_metrics['irregular_intervals']}"
        )

        # Step 5: Comprehensive profiling
        logger.info("\n[Step 5/6] Generating comprehensive data profile...")

        weather_columns = [col for col in df.columns if col != "load"]
        profiler = DataProfiler(df, load_column="load", weather_columns=weather_columns)

        profile = profiler.generate_profile()

        # Log key statistics
        logger.info("\nLoad Statistics:")
        load_stats = profile["load_statistics"]
        logger.info(f"  Mean: {load_stats['mean']:.2f} kW")
        logger.info(f"  Std: {load_stats['std']:.2f} kW")
        logger.info(f"  Min: {load_stats['min']:.2f} kW")
        logger.info(f"  Max: {load_stats['max']:.2f} kW")
        logger.info(f"  Coefficient of Variation: {load_stats['cv']:.3f}")

        logger.info("\nSeasonality Patterns:")
        seasonality = profile["seasonality"]
        logger.info(f"  Peak hour: {seasonality['hourly_pattern']['peak_hour']}:00")
        logger.info(f"  Min hour: {seasonality['hourly_pattern']['min_hour']}:00")
        logger.info(
            f"  Peak day: {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][seasonality['daily_pattern']['peak_day']]}"
        )
        logger.info(f"  Peak month: {seasonality['monthly_pattern']['peak_month']}")

        if "load_weather_correlation" in profile:
            logger.info("\nLoad-Weather Correlations:")
            for feature, corr in profile["load_weather_correlation"].items():
                if corr is not None:
                    logger.info(f"  {feature}: {corr:.3f}")

        # Step 6: Generate visualizations
        logger.info("\n[Step 6/6] Generating visualizations...")

        # Load patterns
        logger.info("  Creating load patterns plot...")
        profiler.plot_load_patterns(save_dir=str(FIGURES_DIR))

        # Missing data heatmap
        logger.info("  Creating missing data heatmap...")
        profiler.plot_missing_data_heatmap(save_dir=str(FIGURES_DIR))

        # Autocorrelation
        logger.info("  Creating autocorrelation plot...")
        profiler.plot_autocorrelation(max_lags=168, save_dir=str(FIGURES_DIR))

        # Weather correlations
        if weather_columns:
            logger.info("  Creating weather correlation plots...")
            for weather_col in ["temperature", "humidity", "wind_speed"]:
                if weather_col in df.columns:
                    profiler.plot_load_weather_scatter(
                        weather_col, save_dir=str(FIGURES_DIR)
                    )

        # Create summary statistics table
        logger.info("\nCreating summary statistics table...")
        summary_stats = pd.DataFrame(
            {
                "Feature": ["load"] + weather_columns,
                "Mean": [df[col].mean() for col in ["load"] + weather_columns],
                "Std": [df[col].std() for col in ["load"] + weather_columns],
                "Min": [df[col].min() for col in ["load"] + weather_columns],
                "Max": [df[col].max() for col in ["load"] + weather_columns],
                "Missing %": [
                    df[col].isna().sum() / len(df) * 100
                    for col in ["load"] + weather_columns
                ],
            }
        )

        summary_file = PROJECT_ROOT / "docs" / "data_summary_statistics.csv"
        summary_stats.to_csv(summary_file, index=False)
        logger.info(f"  Summary statistics saved to: {summary_file}")

        # Save profile to JSON
        import json

        profile_file = PROJECT_ROOT / "docs" / "data_profile.json"

        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        profile_serializable = convert_to_serializable(profile)

        with open(profile_file, "w") as f:
            json.dump(profile_serializable, f, indent=2)
        logger.info(f"  Data profile saved to: {profile_file}")

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("DATA PROFILING COMPLETE")
        logger.info("=" * 80)
        logger.info("\nGenerated files:")
        logger.info(f"  Visualizations: {FIGURES_DIR}")
        logger.info(f"  Summary statistics: {summary_file}")
        logger.info(f"  Data profile: {profile_file}")
        logger.info("\nNext steps:")
        logger.info("  1. Review visualizations in docs/figures/")
        logger.info("  2. Check data quality issues in logs/")
        logger.info(
            "  3. Run feature engineering: python scripts/run_feature_engineering.py"
        )
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Error during data profiling: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
