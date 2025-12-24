"""Feature engineering script."""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime

from src.logger import setup_logging_from_config, get_logger
from src.config import load_config
from src.data.loader import load_load_data, load_weather_data, merge_load_weather
from src.features.pipeline import FeaturePipeline

# Setup logging
setup_logging_from_config()
logger = get_logger(__name__)

# Setup paths
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 80)
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Step 1: Load data
        logger.info("\n[Step 1/5] Loading data...")
        load_df = load_load_data(
            str(DATA_RAW / "load_data.csv"),
            date_column="timestamp",
            load_column="load"
        )
        
        weather_df = load_weather_data(
            str(DATA_EXTERNAL / "weather.csv"),
            date_column="timestamp"
        )
        
        logger.info(f"Load data: {load_df.shape}")
        logger.info(f"Weather data: {weather_df.shape}")
        
        # Step 2: Merge datasets
        logger.info("\n[Step 2/5] Merging load and weather data...")
        df = merge_load_weather(load_df, weather_df, validate_alignment=True)
        logger.info(f"Merged data: {df.shape}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Step 3: Initialize feature pipeline
        logger.info("\n[Step 3/5] Initializing feature pipeline...")
        pipeline = FeaturePipeline()
        logger.info("Feature pipeline initialized")
        
        # Step 4: Create all features
        logger.info("\n[Step 4/5] Creating features...")
        logger.info("This may take a few minutes...")
        
        features = pipeline.create_all_features(
            df,
            target_column='load',
            include_target=True  # Include target for training
        )
        
        logger.info(f"Feature engineering complete!")
        logger.info(f"Total features: {len(features.columns)}")
        logger.info(f"Dataset shape: {features.shape}")
        
        # Log feature counts by category
        feature_categories = {
            'temporal': len([c for c in features.columns if any(x in c for x in ['hour', 'day', 'week', 'month', 'season'])]),
            'calendar': len([c for c in features.columns if any(x in c for x in ['holiday', 'weekend', 'business'])]),
            'lag': len([c for c in features.columns if 'lag' in c]),
            'rolling': len([c for c in features.columns if 'rolling' in c]),
            'weather': len([c for c in features.columns if any(x in c for x in ['temp', 'humidity', 'wind', 'precip', 'hdd', 'cdd'])]),
            'interaction': len([c for c in features.columns if '_x_' in c])
        }
        
        logger.info("\nFeature breakdown:")
        for category, count in feature_categories.items():
            if count > 0:
                logger.info(f"  {category}: {count} features")
        
        # Step 5: Save features
        logger.info("\n[Step 5/5] Saving processed features...")
        
        # Save as parquet (efficient)
        output_file = DATA_PROCESSED / "features.parquet"
        features.to_parquet(output_file, compression='snappy')
        logger.info(f"Features saved to: {output_file}")
        
        # Also save as CSV for inspection
        output_csv = DATA_PROCESSED / "features.csv"
        features.to_csv(output_csv)
        logger.info(f"Features saved to: {output_csv}")
        
        # Save feature names
        feature_names_file = DATA_PROCESSED / "feature_names.txt"
        with open(feature_names_file, 'w') as f:
            for col in features.columns:
                f.write(f"{col}\n")
        logger.info(f"Feature names saved to: {feature_names_file}")
        
        # Data quality check
        logger.info("\nData quality check:")
        missing_pct = (features.isna().sum() / len(features) * 100).sort_values(ascending=False)
        logger.info(f"Features with >5% missing data: {(missing_pct > 5).sum()}")
        if (missing_pct > 5).any():
            logger.warning("Features with high missing data:")
            for feat, pct in missing_pct[missing_pct > 5].head(10).items():
                logger.warning(f"  {feat}: {pct:.1f}%")
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nGenerated files:")
        logger.info(f"  Features (Parquet): {output_file}")
        logger.info(f"  Features (CSV): {output_csv}")
        logger.info(f"  Feature names: {feature_names_file}")
        logger.info(f"\nDataset summary:")
        logger.info(f"  Total samples: {len(features):,}")
        logger.info(f"  Total features: {len(features.columns)}")
        logger.info(f"  Date range: {features.index.min()} to {features.index.max()}")
        logger.info(f"  Memory usage: {features.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        logger.info("\nNext steps:")
        logger.info("  1. Review feature names in data/processed/feature_names.txt")
        logger.info("  2. Run model training: python scripts/run_model_training.py")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
