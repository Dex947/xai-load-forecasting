"""SHAP visualization utilities."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Optional, List, Tuple, Dict
from pathlib import Path

from src.logger import get_logger

logger = get_logger(__name__)


class ExplainabilityVisualizer:
    """SHAP summary, dependence, waterfall, and force plots."""

    def __init__(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        X_data: pd.DataFrame,
        expected_value: Optional[float] = None,
    ):
        self.shap_values = shap_values
        self.feature_names = feature_names
        self.X_data = X_data
        self.expected_value = expected_value

        logger.info(
            f"Explainability visualizer initialized with {len(feature_names)} features"
        )

    def plot_summary(
        self,
        max_display: int = 20,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> None:
        """
        Plot SHAP summary plot.

        Args:
            max_display: Maximum number of features to display
            save_path: Path to save figure
            figsize: Figure size
        """
        logger.info("Creating SHAP summary plot")

        plt.figure(figsize=figsize)
        shap.summary_plot(
            self.shap_values,
            self.X_data,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Summary plot saved to {save_path}")

        plt.close()

    def plot_bar(
        self,
        max_display: int = 20,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> None:
        """
        Plot SHAP bar plot (mean absolute SHAP values).

        Args:
            max_display: Maximum number of features to display
            save_path: Path to save figure
            figsize: Figure size
        """
        logger.info("Creating SHAP bar plot")

        plt.figure(figsize=figsize)
        shap.summary_plot(
            self.shap_values,
            self.X_data,
            feature_names=self.feature_names,
            max_display=max_display,
            plot_type="bar",
            show=False,
        )
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Bar plot saved to {save_path}")

        plt.close()

    def plot_dependence(
        self,
        feature: str,
        interaction_feature: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> None:
        """
        Plot SHAP dependence plot for a feature.

        Args:
            feature: Feature to plot
            interaction_feature: Feature to use for coloring (auto if None)
            save_path: Path to save figure
            figsize: Figure size
        """
        logger.info(f"Creating SHAP dependence plot for {feature}")

        if feature not in self.feature_names:
            raise ValueError(f"Feature '{feature}' not found")

        feature_idx = self.feature_names.index(feature)

        plt.figure(figsize=figsize)
        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            self.X_data,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False,
        )
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Dependence plot saved to {save_path}")

        plt.close()

    def plot_waterfall(
        self,
        instance_idx: int,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> None:
        """
        Plot SHAP waterfall plot for a single prediction.

        Args:
            instance_idx: Index of instance to explain
            save_path: Path to save figure
            figsize: Figure size
        """
        logger.info(f"Creating SHAP waterfall plot for instance {instance_idx}")

        plt.figure(figsize=figsize)

        # Create Explanation object
        explanation = shap.Explanation(
            values=self.shap_values[instance_idx],
            base_values=self.expected_value if self.expected_value is not None else 0,
            data=self.X_data.iloc[instance_idx].values,
            feature_names=self.feature_names,
        )

        shap.waterfall_plot(explanation, show=False)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Waterfall plot saved to {save_path}")

        plt.close()

    def plot_force(
        self,
        instance_idx: int,
        save_path: Optional[str] = None,
        matplotlib: bool = True,
    ) -> None:
        """
        Plot SHAP force plot for a single prediction.

        Args:
            instance_idx: Index of instance to explain
            save_path: Path to save figure
            matplotlib: Use matplotlib backend (True) or HTML (False)
        """
        logger.info(f"Creating SHAP force plot for instance {instance_idx}")

        if matplotlib:
            shap.force_plot(
                self.expected_value if self.expected_value is not None else 0,
                self.shap_values[instance_idx],
                self.X_data.iloc[instance_idx],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False,
            )

            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Force plot saved to {save_path}")

            plt.close()
        else:
            # HTML version
            force_plot = shap.force_plot(
                self.expected_value if self.expected_value is not None else 0,
                self.shap_values[instance_idx],
                self.X_data.iloc[instance_idx],
                feature_names=self.feature_names,
            )

            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                shap.save_html(save_path, force_plot)
                logger.info(f"Force plot saved to {save_path}")

    def plot_time_varying_importance(
        self,
        time_varying_importance: Dict[str, pd.DataFrame],
        top_n: int = 10,
        save_dir: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10),
    ) -> None:
        """
        Plot time-varying feature importance.

        Args:
            time_varying_importance: Dictionary with time-varying importance DataFrames
            top_n: Number of top features to plot
            save_dir: Directory to save figures
            figsize: Figure size
        """
        logger.info("Creating time-varying importance plots")

        n_plots = len(time_varying_importance)
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize)

        if n_plots == 1:
            axes = [axes]

        for idx, (period, importance_df) in enumerate(time_varying_importance.items()):
            # Get top features
            mean_importance = importance_df.mean(axis=0).sort_values(ascending=False)
            top_features = mean_importance.head(top_n).index

            # Plot heatmap
            sns.heatmap(
                importance_df[top_features].T,
                cmap="YlOrRd",
                cbar_kws={"label": "Mean |SHAP|"},
                ax=axes[idx],
            )
            axes[idx].set_title(
                f"Time-Varying Feature Importance ({period.capitalize()})"
            )
            axes[idx].set_xlabel(period.capitalize())
            axes[idx].set_ylabel("Features")

        plt.tight_layout()

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = Path(save_dir) / "time_varying_importance.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Time-varying importance plot saved to {save_path}")

        plt.close()

    def plot_feature_importance_comparison(
        self,
        model_importance: pd.DataFrame,
        shap_importance: pd.DataFrame,
        top_n: int = 20,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """
        Compare model feature importance with SHAP importance.

        Args:
            model_importance: Model's feature importance DataFrame
            shap_importance: SHAP-based importance DataFrame
            top_n: Number of top features to plot
            save_path: Path to save figure
            figsize: Figure size
        """
        logger.info("Creating feature importance comparison plot")

        # Merge and normalize
        comparison = model_importance.merge(
            shap_importance, on="feature", suffixes=("_model", "_shap")
        )

        # Normalize to 0-1
        comparison["importance_model_norm"] = (
            comparison["importance_model"] / comparison["importance_model"].max()
        )
        comparison["importance_shap_norm"] = (
            comparison["importance_shap"] / comparison["importance_shap"].max()
        )

        # Get top features by SHAP
        comparison = comparison.nlargest(top_n, "importance_shap_norm")

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(comparison))
        width = 0.35

        ax.barh(
            x - width / 2,
            comparison["importance_model_norm"],
            width,
            label="Model Importance",
        )
        ax.barh(
            x + width / 2,
            comparison["importance_shap_norm"],
            width,
            label="SHAP Importance",
        )

        ax.set_yticks(x)
        ax.set_yticklabels(comparison["feature"])
        ax.set_xlabel("Normalized Importance")
        ax.set_title("Feature Importance Comparison: Model vs SHAP")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Comparison plot saved to {save_path}")

        plt.close()

    def plot_prediction_explanation(
        self,
        instance_idx: int,
        y_true: float,
        y_pred: float,
        top_n: int = 10,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> None:
        """
        Plot detailed explanation for a single prediction.

        Args:
            instance_idx: Index of instance
            y_true: True value
            y_pred: Predicted value
            top_n: Number of top features to show
            save_path: Path to save figure
            figsize: Figure size
        """
        logger.info(f"Creating prediction explanation for instance {instance_idx}")

        # Get SHAP values and feature values
        shap_vals = self.shap_values[instance_idx]
        feature_vals = self.X_data.iloc[instance_idx]

        # Create explanation DataFrame
        explanation = (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "value": feature_vals.values,
                    "shap_value": shap_vals,
                    "abs_shap": np.abs(shap_vals),
                }
            )
            .sort_values("abs_shap", ascending=False)
            .head(top_n)
        )

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # SHAP values
        colors = ["red" if x < 0 else "green" for x in explanation["shap_value"]]
        ax1.barh(explanation["feature"], explanation["shap_value"], color=colors)
        ax1.set_xlabel("SHAP Value")
        ax1.set_title("Feature Contributions")
        ax1.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
        ax1.grid(True, alpha=0.3, axis="x")

        # Feature values
        ax2.barh(explanation["feature"], explanation["value"], color="steelblue")
        ax2.set_xlabel("Feature Value")
        ax2.set_title("Feature Values")
        ax2.grid(True, alpha=0.3, axis="x")

        # Add prediction info
        fig.suptitle(
            f"Prediction Explanation\nTrue: {y_true:.2f}, Predicted: {y_pred:.2f}, Error: {abs(y_true - y_pred):.2f}",
            fontsize=12,
            fontweight="bold",
        )

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Prediction explanation saved to {save_path}")

        plt.close()
