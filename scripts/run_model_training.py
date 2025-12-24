"""Model training script with rolling origin CV."""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from src.logger import setup_logging_from_config, get_logger
from src.config import load_config
from src.models.baseline import compare_baselines
from src.models.gbm import GradientBoostingModel

# Setup logging
setup_logging_from_config()
logger = get_logger(__name__)

# Setup paths
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "artifacts"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def prepare_data(features_df, target_column="load", test_size_days=30):
    """
    Prepare train/test split with temporal ordering.

    Args:
        features_df: DataFrame with all features
        target_column: Name of target column
        test_size_days: Size of test set in days

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Preparing train/test split...")

    # Remove rows with missing target
    features_df = features_df.dropna(subset=[target_column])

    # Calculate split point
    test_size_hours = test_size_days * 24
    split_idx = len(features_df) - test_size_hours

    # Split data
    train_df = features_df.iloc[:split_idx]
    test_df = features_df.iloc[split_idx:]

    # Separate features and target
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Remove rows with any missing values in features
    train_valid = ~X_train.isna().any(axis=1)
    test_valid = ~X_test.isna().any(axis=1)

    X_train = X_train[train_valid]
    y_train = y_train[train_valid]
    X_test = X_test[test_valid]
    y_test = y_test[test_valid]

    logger.info(
        f"Train set: {len(X_train):,} samples ({X_train.index.min()} to {X_train.index.max()})"
    )
    logger.info(
        f"Test set: {len(X_test):,} samples ({X_test.index.min()} to {X_test.index.max()})"
    )
    logger.info(f"Features: {len(X_train.columns)}")

    return X_train, X_test, y_train, y_test


def train_baselines(y_train, y_test):
    """Train and evaluate baseline models."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING BASELINE MODELS")
    logger.info("=" * 80)

    results = compare_baselines(
        y_train, y_test, horizon=len(y_test), methods=["persistence", "seasonal_naive"]
    )

    logger.info("\nBaseline Results:")
    logger.info(results.to_string())

    return results


def analyze_residuals(y_true, y_pred, save_path):
    """Analyze model residuals for diagnostics."""
    logger.info("\nPerforming residual analysis...")

    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Residuals vs Predictions
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0, 0].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[0, 0].set_xlabel("Predicted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals vs Predictions")
    axes[0, 0].grid(True, alpha=0.3)

    # Residual distribution
    axes[0, 1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 1].axvline(x=0, color="r", linestyle="--", linewidth=2)
    axes[0, 1].set_xlabel("Residuals")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Residual Distribution")
    axes[0, 1].grid(True, alpha=0.3)

    # Q-Q plot
    from scipy import stats

    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot")
    axes[1, 0].grid(True, alpha=0.3)

    # Residuals over time
    axes[1, 1].plot(residuals.values, alpha=0.7, linewidth=0.5)
    axes[1, 1].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[1, 1].set_xlabel("Time Index")
    axes[1, 1].set_ylabel("Residuals")
    axes[1, 1].set_title("Residuals Over Time")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Log diagnostics
    logger.info(f"Residual Mean: {residuals.mean():.4f}")
    logger.info(f"Residual Std: {residuals.std():.4f}")
    logger.info(f"Residual Skewness: {residuals.skew():.4f}")
    logger.info(f"Residual Kurtosis: {residuals.kurtosis():.4f}")


def prune_features(X_train, y_train, X_test, threshold=0.001):
    """Remove low-importance features using permutation importance."""
    logger.info("\nPruning low-importance features...")

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance

    # Quick RF to get permutation importance
    rf = RandomForestRegressor(
        n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    perm_importance = permutation_importance(
        rf, X_train, y_train, n_repeats=5, random_state=42, n_jobs=-1
    )

    # Keep features with importance > threshold
    important_features = X_train.columns[perm_importance.importances_mean > threshold]

    logger.info(f"Features before pruning: {len(X_train.columns)}")
    logger.info(f"Features after pruning: {len(important_features)}")
    logger.info(
        f"Removed {len(X_train.columns) - len(important_features)} low-importance features"
    )

    return (
        X_train[important_features],
        X_test[important_features],
        list(important_features),
    )


def train_gradient_boosting(X_train, X_test, y_train, y_test, config):
    """Train gradient boosting model with improvements."""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING GRADIENT BOOSTING MODEL (IMPROVED)")
    logger.info("=" * 80)

    # Feature pruning
    X_train_pruned, X_test_pruned, selected_features = prune_features(
        X_train, y_train, X_test
    )

    # Split train into train/val
    val_size = int(len(X_train_pruned) * 0.2)
    X_train_fit = X_train_pruned.iloc[:-val_size]
    y_train_fit = y_train.iloc[:-val_size]
    X_val = X_train_pruned.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]

    logger.info(f"Training set: {len(X_train_fit):,} samples")
    logger.info(f"Validation set: {len(X_val):,} samples")
    logger.info(f"Features: {len(selected_features)}")

    # Improved hyperparameters
    improved_config = config.model.dict()
    improved_config["lightgbm"] = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 63,  # Increased from 31
        "learning_rate": 0.03,  # Reduced for better generalization
        "feature_fraction": 0.8,  # Feature sampling
        "bagging_fraction": 0.8,  # Row sampling
        "bagging_freq": 5,
        "min_data_in_leaf": 50,  # Prevent overfitting
        "lambda_l1": 0.1,  # L1 regularization
        "lambda_l2": 0.1,  # L2 regularization
        "max_depth": 10,  # Limit depth
        "num_iterations": 1000,
        "verbose": -1,
    }

    # Initialize model with improved config
    model = GradientBoostingModel(
        model_type=config.model.type,
        config=improved_config,
        monotonic_constraints=config.model.monotonic_constraints,
    )

    # Train model
    logger.info("Training improved model...")
    model.fit(X_train_fit, y_train_fit, X_val, y_val, verbose=True)

    # Evaluate on validation set
    val_metrics = model.evaluate(X_val, y_val)
    logger.info("\nValidation Set Performance:")
    for metric, value in val_metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_metrics = model.evaluate(X_test_pruned, y_test)

    logger.info("\nTest Set Performance:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")

    # Check for overfitting
    train_metrics = model.evaluate(X_train_fit, y_train_fit)
    logger.info("\nOverfitting Check:")
    logger.info(f"  Train R²: {train_metrics['r2']:.4f}")
    logger.info(f"  Val R²:   {val_metrics['r2']:.4f}")
    logger.info(f"  Test R²:  {test_metrics['r2']:.4f}")
    logger.info(f"  Train RMSE: {train_metrics['rmse']:.4f}")
    logger.info(f"  Val RMSE:   {val_metrics['rmse']:.4f}")
    logger.info(f"  Test RMSE:  {test_metrics['rmse']:.4f}")

    # Feature importance
    logger.info("\nTop 20 Important Features:")
    importance = model.get_feature_importance(top_n=20)
    for idx, row in importance.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.2f}")

    # Residual analysis
    predictions = model.predict(X_test_pruned)
    residuals_path = FIGURES_DIR / "residual_analysis.png"
    analyze_residuals(y_test, predictions, residuals_path)
    logger.info(f"Residual analysis saved to: {residuals_path}")

    return model, test_metrics, selected_features


def plot_predictions(y_test, predictions, save_path):
    """Plot actual vs predicted values."""
    logger.info("Creating prediction plot...")

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Time series plot
    axes[0].plot(y_test.index, y_test.values, label="Actual", alpha=0.7, linewidth=1)
    axes[0].plot(y_test.index, predictions, label="Predicted", alpha=0.7, linewidth=1)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Load (kW)")
    axes[0].set_title("Actual vs Predicted Load")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Scatter plot
    axes[1].scatter(y_test.values, predictions, alpha=0.5, s=10)
    axes[1].plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )
    axes[1].set_xlabel("Actual Load (kW)")
    axes[1].set_ylabel("Predicted Load (kW)")
    axes[1].set_title("Prediction Scatter Plot")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Prediction plot saved to: {save_path}")


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("MODEL TRAINING PIPELINE")
    logger.info("=" * 80)

    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")

        # Step 1: Load features
        logger.info("\n[Step 1/5] Loading processed features...")
        features_file = DATA_PROCESSED / "features.parquet"
        features_df = pd.read_parquet(features_file)
        logger.info(f"Features loaded: {features_df.shape}")

        # Step 2: Prepare data
        logger.info("\n[Step 2/5] Preparing train/test split...")
        X_train, X_test, y_train, y_test = prepare_data(
            features_df,
            target_column="load",
            test_size_days=config.validation.test_size_days,
        )

        # Step 3: Train baseline models
        logger.info("\n[Step 3/5] Training baseline models...")
        baseline_results = train_baselines(y_train, y_test)

        # Step 4: Train gradient boosting model
        logger.info("\n[Step 4/5] Training gradient boosting model...")
        model, test_metrics, selected_features = train_gradient_boosting(
            X_train, X_test, y_train, y_test, config
        )

        # Step 5: Save model and results
        logger.info("\n[Step 5/5] Saving model and results...")

        # Save model
        model_file = MODELS_DIR / f"{config.model.type}_model.pkl"
        model.save(str(model_file))
        logger.info(f"Model saved to: {model_file}")

        # Save feature importance
        importance_file = MODELS_DIR / "feature_importance.csv"
        model.get_feature_importance().to_csv(importance_file, index=False)
        logger.info(f"Feature importance saved to: {importance_file}")

        # Generate predictions using selected features
        X_test_selected = X_test[selected_features]
        predictions = model.predict(X_test_selected)

        # Plot predictions
        plot_file = FIGURES_DIR / "model_predictions.png"
        plot_predictions(y_test, predictions, plot_file)

        # Save results summary
        results_summary = {
            "model_type": config.model.type,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features_original": len(X_train.columns),
            "n_features_selected": len(selected_features),
            **test_metrics,
            "baseline_persistence_rmse": baseline_results.loc["persistence", "rmse"]
            if "persistence" in baseline_results.index
            else None,
            "baseline_seasonal_naive_rmse": baseline_results.loc[
                "seasonal_naive", "rmse"
            ]
            if "seasonal_naive" in baseline_results.index
            else None,
        }

        # Save selected features
        selected_features_file = MODELS_DIR / "selected_features.txt"
        with open(selected_features_file, "w") as f:
            for feat in selected_features:
                f.write(f"{feat}\n")
        logger.info(f"Selected features saved to: {selected_features_file}")

        results_file = MODELS_DIR / "training_results.json"
        import json

        with open(results_file, "w") as f:
            json.dump(results_summary, f, indent=2)
        logger.info(f"Results saved to: {results_file}")

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("MODEL TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info("\nModel Performance:")
        logger.info(f"  RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"  MAE: {test_metrics['mae']:.4f}")
        logger.info(f"  R²: {test_metrics['r2']:.4f}")
        logger.info("\nGenerated files:")
        logger.info(f"  Model: {model_file}")
        logger.info(f"  Feature importance: {importance_file}")
        logger.info(f"  Predictions plot: {plot_file}")
        logger.info(f"  Results summary: {results_file}")
        logger.info("\nNext steps:")
        logger.info("  1. Review model performance in logs/")
        logger.info("  2. Check predictions plot in docs/figures/")
        logger.info("  3. Run SHAP analysis: python scripts/run_shap_analysis.py")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
