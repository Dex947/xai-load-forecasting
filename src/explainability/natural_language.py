"""Natural language explanation generation for load forecasts."""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from datetime import datetime

from src.logger import get_logger

logger = get_logger(__name__)


class NaturalLanguageExplainer:
    """
    Generates human-readable explanations for load forecasts.

    Converts SHAP values and predictions into operator-friendly text.
    """

    # Feature category mappings
    FEATURE_CATEGORIES = {
        "temporal": ["hour", "day_of_week", "month", "is_weekend", "is_business_hour"],
        "weather": [
            "temperature",
            "humidity",
            "wind",
            "pressure",
            "cloud",
            "ghi",
            "dni",
        ],
        "load_history": ["load_lag", "rolling"],
        "calendar": ["holiday", "school", "event"],
    }

    # Feature descriptions
    FEATURE_DESCRIPTIONS = {
        "load_lag_1h": "load from 1 hour ago",
        "load_lag_2h": "load from 2 hours ago",
        "load_lag_24h": "load from same hour yesterday",
        "rolling_3h_mean": "average load over past 3 hours",
        "rolling_6h_mean": "average load over past 6 hours",
        "rolling_24h_mean": "average load over past 24 hours",
        "temperature": "current temperature",
        "humidity": "humidity level",
        "hour": "time of day",
        "day_of_week": "day of the week",
        "is_weekend": "weekend indicator",
        "is_holiday": "holiday indicator",
        "is_business_hour": "business hours",
        "ghi": "solar irradiance",
        "hdd": "heating degree days",
        "cdd": "cooling degree days",
    }

    def __init__(
        self, feature_names: List[str], target_unit: str = "kW", language: str = "en"
    ):
        """
        Args:
            feature_names: List of feature names
            target_unit: Unit for predictions (e.g., 'kW', 'MW')
            language: Language for explanations
        """
        self.feature_names = feature_names
        self.target_unit = target_unit
        self.language = language

        logger.info(
            f"NaturalLanguageExplainer initialized: {len(feature_names)} features"
        )

    def explain_prediction(
        self,
        prediction: float,
        shap_values: np.ndarray,
        feature_values: pd.Series,
        base_value: float,
        timestamp: Optional[datetime] = None,
        include_details: bool = True,
    ) -> str:
        """
        Generate natural language explanation for a single prediction.

        Args:
            prediction: Predicted value
            shap_values: SHAP values for this prediction
            feature_values: Feature values for this prediction
            base_value: SHAP base value (expected value)
            timestamp: Optional timestamp for context
            include_details: Include detailed feature breakdown

        Returns:
            Human-readable explanation
        """
        # Build explanation
        lines = []

        # Header with prediction
        if timestamp:
            time_str = timestamp.strftime("%A, %B %d at %H:%M")
            lines.append(f"**Forecast for {time_str}**")

        lines.append(f"Predicted load: **{prediction:.2f} {self.target_unit}**")
        lines.append("")

        # Get top contributing features
        contributions = self._get_top_contributions(shap_values, n=5)

        # Categorize impact
        positive_impact = [(f, v) for f, v in contributions if v > 0]
        negative_impact = [(f, v) for f, v in contributions if v < 0]

        # Main drivers
        if positive_impact:
            lines.append("**Factors increasing load:**")
            for feat, impact in positive_impact[:3]:
                desc = self._describe_feature_impact(feat, impact, feature_values)
                lines.append(f"  • {desc}")
            lines.append("")

        if negative_impact:
            lines.append("**Factors decreasing load:**")
            for feat, impact in negative_impact[:3]:
                desc = self._describe_feature_impact(
                    feat, abs(impact), feature_values, negative=True
                )
                lines.append(f"  • {desc}")
            lines.append("")

        # Context sentence
        context = self._generate_context_sentence(
            prediction, base_value, positive_impact, negative_impact, feature_values
        )
        lines.append(f"*{context}*")

        if include_details:
            lines.append("")
            lines.append(self._generate_details_section(contributions, feature_values))

        return "\n".join(lines)

    def _get_top_contributions(
        self, shap_values: np.ndarray, n: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top N contributing features by absolute SHAP value."""
        contributions = list(zip(self.feature_names, shap_values))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        return contributions[:n]

    def _describe_feature_impact(
        self,
        feature: str,
        impact: float,
        feature_values: pd.Series,
        negative: bool = False,
    ) -> str:
        """Generate description for a feature's impact."""
        # Get feature value
        value = feature_values.get(feature, None)

        # Get human-readable feature name
        readable_name = self.FEATURE_DESCRIPTIONS.get(
            feature, feature.replace("_", " ")
        )

        # Build description
        direction = "decreasing" if negative else "increasing"
        impact_str = f"{abs(impact):.2f} {self.target_unit}"

        if value is not None:
            if isinstance(value, (int, float)):
                if "hour" in feature.lower():
                    value_str = f"{int(value):02d}:00"
                elif "day" in feature.lower():
                    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                    value_str = days[int(value) % 7] if 0 <= value < 7 else str(value)
                elif "temperature" in feature.lower():
                    value_str = f"{value:.1f}°C"
                elif "lag" in feature.lower() or "rolling" in feature.lower():
                    value_str = f"{value:.2f} {self.target_unit}"
                else:
                    value_str = f"{value:.2f}"

                return (
                    f"{readable_name} ({value_str}) is {direction} load by {impact_str}"
                )
            else:
                return f"{readable_name} is {direction} load by {impact_str}"

        return f"{readable_name} is {direction} load by {impact_str}"

    def _generate_context_sentence(
        self,
        prediction: float,
        base_value: float,
        positive: List[Tuple[str, float]],
        negative: List[Tuple[str, float]],
        feature_values: pd.Series,
    ) -> str:
        """Generate a contextual summary sentence."""
        diff = prediction - base_value

        if abs(diff) < 0.1:
            return "Load is near the typical average for this period."

        direction = "higher" if diff > 0 else "lower"

        # Identify main driver
        if positive and diff > 0:
            main_driver = positive[0][0]
        elif negative and diff < 0:
            main_driver = negative[0][0]
        else:
            main_driver = None

        if main_driver:
            driver_desc = self.FEATURE_DESCRIPTIONS.get(
                main_driver, main_driver.replace("_", " ")
            )
            return f"Load is {abs(diff):.2f} {self.target_unit} {direction} than average, primarily due to {driver_desc}."

        return f"Load is {abs(diff):.2f} {self.target_unit} {direction} than the typical average."

    def _generate_details_section(
        self, contributions: List[Tuple[str, float]], feature_values: pd.Series
    ) -> str:
        """Generate detailed breakdown section."""
        lines = ["**Detailed breakdown:**"]
        lines.append("| Feature | Value | Impact |")
        lines.append("|---------|-------|--------|")

        for feat, impact in contributions:
            value = feature_values.get(feat, "N/A")
            if isinstance(value, float):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)

            impact_str = f"+{impact:.3f}" if impact > 0 else f"{impact:.3f}"
            lines.append(f"| {feat} | {value_str} | {impact_str} |")

        return "\n".join(lines)

    def explain_batch(
        self,
        predictions: np.ndarray,
        shap_values: np.ndarray,
        feature_data: pd.DataFrame,
        base_value: float,
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> List[str]:
        """Generate explanations for batch of predictions."""
        explanations = []

        for i in range(len(predictions)):
            ts = timestamps[i] if timestamps is not None else None
            exp = self.explain_prediction(
                prediction=predictions[i],
                shap_values=shap_values[i],
                feature_values=feature_data.iloc[i],
                base_value=base_value,
                timestamp=ts,
                include_details=False,
            )
            explanations.append(exp)

        return explanations

    def generate_summary_report(
        self,
        predictions: np.ndarray,
        shap_values: np.ndarray,
        feature_data: pd.DataFrame,
        period_name: str = "forecast period",
    ) -> str:
        """
        Generate summary report for a forecast period.

        Args:
            predictions: Array of predictions
            shap_values: SHAP values matrix
            feature_data: Feature DataFrame
            period_name: Name for the period

        Returns:
            Summary report text
        """
        lines = [f"# Load Forecast Summary: {period_name}", ""]

        # Statistics
        lines.append("## Forecast Statistics")
        lines.append(f"- **Mean load**: {predictions.mean():.2f} {self.target_unit}")
        lines.append(f"- **Peak load**: {predictions.max():.2f} {self.target_unit}")
        lines.append(f"- **Minimum load**: {predictions.min():.2f} {self.target_unit}")
        lines.append(f"- **Std deviation**: {predictions.std():.2f} {self.target_unit}")
        lines.append("")

        # Top drivers overall
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features = sorted(
            zip(self.feature_names, mean_abs_shap), key=lambda x: -x[1]
        )[:5]

        lines.append("## Key Drivers")
        for feat, importance in top_features:
            desc = self.FEATURE_DESCRIPTIONS.get(feat, feat)
            lines.append(
                f"- **{desc}**: avg impact {importance:.3f} {self.target_unit}"
            )
        lines.append("")

        # Peak analysis
        peak_idx = predictions.argmax()
        lines.append("## Peak Load Analysis")
        lines.append(f"Peak of {predictions[peak_idx]:.2f} {self.target_unit}")

        peak_contributions = self._get_top_contributions(shap_values[peak_idx], n=3)
        lines.append("Main contributors to peak:")
        for feat, impact in peak_contributions:
            desc = self.FEATURE_DESCRIPTIONS.get(feat, feat)
            lines.append(f"  - {desc}: {impact:+.3f} {self.target_unit}")

        return "\n".join(lines)

    def generate_alert_message(
        self,
        prediction: float,
        threshold: float,
        shap_values: np.ndarray,
        feature_values: pd.Series,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """
        Generate alert message for high/low load conditions.

        Args:
            prediction: Predicted value
            threshold: Alert threshold
            shap_values: SHAP values
            feature_values: Feature values
            timestamp: Timestamp

        Returns:
            Alert message
        """
        is_high = prediction > threshold
        alert_type = "HIGH LOAD" if is_high else "LOW LOAD"

        time_str = (
            timestamp.strftime("%H:%M on %b %d") if timestamp else "upcoming period"
        )

        msg = f"⚠️ **{alert_type} ALERT** for {time_str}\n\n"
        msg += f"Predicted: {prediction:.2f} {self.target_unit} "
        msg += f"(threshold: {threshold:.2f} {self.target_unit})\n\n"

        # Top contributors
        contributions = self._get_top_contributions(shap_values, n=3)
        msg += "**Contributing factors:**\n"
        for feat, impact in contributions:
            desc = self.FEATURE_DESCRIPTIONS.get(feat, feat)
            msg += f"  • {desc}: {impact:+.2f} {self.target_unit}\n"

        return msg
