"""SHAP analysis and visualization script."""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.logger import setup_logging_from_config, get_logger
from src.config import load_config
from src.models.gbm import GradientBoostingModel
from src.explainability.shap_analysis import SHAPAnalyzer
from src.explainability.visualizations import ExplainabilityVisualizer

# Setup logging
setup_logging_from_config()
logger = get_logger(__name__)

# Setup paths
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "artifacts"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("SHAP ANALYSIS PIPELINE")
    logger.info("=" * 80)

    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")

        # Step 1: Load model and data
        logger.info("\n[Step 1/5] Loading model and data...")

        # Load model
        model_file = MODELS_DIR / f"{config.model.type}_model.pkl"
        model = GradientBoostingModel.load(str(model_file))
        logger.info(f"Model loaded from: {model_file}")

        # Load features
        features_file = DATA_PROCESSED / "features.parquet"
        features_df = pd.read_parquet(features_file)
        logger.info(f"Features loaded: {features_df.shape}")

        # Load selected features
        selected_features_file = MODELS_DIR / "selected_features.txt"
        with open(selected_features_file, "r") as f:
            selected_features = [line.strip() for line in f.readlines()]
        logger.info(f"Selected features: {len(selected_features)}")

        # Prepare data
        features_df = features_df.dropna(subset=["load"])
        test_size_hours = config.validation.test_size_days * 24
        split_idx = len(features_df) - test_size_hours

        X_train = features_df.iloc[:split_idx][selected_features]
        X_test = features_df.iloc[split_idx:][selected_features]
        y_test = features_df.iloc[split_idx:]["load"]

        # Remove missing values
        test_valid = ~X_test.isna().any(axis=1)
        X_test = X_test[test_valid]
        y_test = y_test[test_valid]

        logger.info(f"Test set: {len(X_test):,} samples")

        # Step 2: Initialize SHAP analyzer
        logger.info("\n[Step 2/5] Initializing SHAP analyzer...")

        # Sample background data (100 samples for efficiency)
        X_background = X_train.sample(n=min(100, len(X_train)), random_state=42)

        analyzer = SHAPAnalyzer(
            model=model.model,
            X_background=X_background,
            model_type="tree",
            config={"sample_size": min(500, len(X_test))},
        )
        logger.info("SHAP analyzer initialized")

        # Step 3: Compute SHAP values
        logger.info("\n[Step 3/5] Computing SHAP values...")
        logger.info("This may take a few minutes...")

        shap_values = analyzer.compute_shap_values(X_test)
        logger.info(f"SHAP values computed: {shap_values.shape}")

        # Save SHAP values
        shap_file = MODELS_DIR / "shap_values.pkl"
        analyzer.save_shap_values(str(shap_file))
        logger.info(f"SHAP values saved to: {shap_file}")

        # Step 4: Generate visualizations
        logger.info("\n[Step 4/5] Generating SHAP visualizations...")

        viz = ExplainabilityVisualizer(
            shap_values=shap_values,
            feature_names=selected_features,
            X_data=analyzer.X_explained,
            expected_value=analyzer.explainer.expected_value,
        )

        # Summary plot
        logger.info("  Creating summary plot...")
        viz.plot_summary(
            max_display=20, save_path=str(FIGURES_DIR / "shap_summary.png")
        )

        # Bar plot
        logger.info("  Creating bar plot...")
        viz.plot_bar(max_display=20, save_path=str(FIGURES_DIR / "shap_bar.png"))

        # Dependence plots for top features
        logger.info("  Creating dependence plots...")
        top_features = analyzer.get_global_importance(top_n=5)
        for idx, row in top_features.iterrows():
            feature = row["feature"]
            if feature in selected_features:
                viz.plot_dependence(
                    feature=feature,
                    save_path=str(FIGURES_DIR / f"shap_dependence_{feature}.png"),
                )

        # Waterfall plot (example)
        logger.info("  Creating waterfall plot...")
        viz.plot_waterfall(
            instance_idx=0, save_path=str(FIGURES_DIR / "shap_waterfall_example.png")
        )

        # Individual prediction explanations
        logger.info("  Creating individual explanations...")
        for idx in [0, len(analyzer.X_explained) // 2, len(analyzer.X_explained) - 1]:
            viz.plot_prediction_explanation(
                instance_idx=idx,
                y_true=y_test.iloc[idx],
                y_pred=model.predict(analyzer.X_explained.iloc[[idx]])[0],
                top_n=10,
                save_path=str(FIGURES_DIR / f"shap_explanation_sample_{idx}.png"),
            )

        # Step 5: Time-varying analysis
        logger.info("\n[Step 5/5] Analyzing time-varying SHAP patterns...")

        time_varying = analyzer.get_time_varying_importance(
            X=analyzer.X_explained, aggregation="mean"
        )

        if time_varying:
            viz.plot_time_varying_importance(
                time_varying_importance=time_varying,
                top_n=10,
                save_dir=str(FIGURES_DIR),
            )
            logger.info(f"Time-varying analysis complete: {len(time_varying)} periods")

        # Global importance
        logger.info("\nGlobal Feature Importance (Top 20):")
        global_importance = analyzer.get_global_importance(top_n=20)
        for idx, row in global_importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        # Save global importance
        importance_file = MODELS_DIR / "shap_global_importance.csv"
        global_importance.to_csv(importance_file, index=False)
        logger.info(f"\nGlobal importance saved to: {importance_file}")

        # Compare with model importance
        model_importance = model.get_feature_importance()
        comparison_file = FIGURES_DIR / "feature_importance_comparison.png"
        viz.plot_feature_importance_comparison(
            model_importance=model_importance,
            shap_importance=global_importance,
            top_n=20,
            save_path=str(comparison_file),
        )
        logger.info(f"Importance comparison saved to: {comparison_file}")

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("SHAP ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info("\nGenerated files:")
        logger.info(f"  SHAP values: {shap_file}")
        logger.info(f"  Global importance: {importance_file}")
        logger.info(
            f"  Visualizations: {len(list(FIGURES_DIR.glob('shap_*.png')))} plots in {FIGURES_DIR}"
        )
        logger.info("\nKey insights:")
        logger.info(f"  Top feature: {global_importance.iloc[0]['feature']}")
        logger.info(
            f"  Top 5 features explain ~{global_importance.head(5)['importance'].sum() / global_importance['importance'].sum() * 100:.1f}% of predictions"
        )
        logger.info("\nNext steps:")
        logger.info("  1. Review SHAP visualizations in docs/figures/")
        logger.info("  2. Analyze time-varying patterns")
        logger.info("  3. Create Model Card: docs/model_card.md")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Error during SHAP analysis: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
