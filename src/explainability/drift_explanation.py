"""Concept drift explanation using SHAP importance changes."""

import numpy as np
from typing import Dict, Optional, List, Any
from datetime import datetime
from pathlib import Path
import json

from src.logger import get_logger

logger = get_logger(__name__)


class DriftExplainer:
    """
    Explains concept drift by analyzing SHAP importance changes over time.

    Detects when feature importance shifts and explains why model
    performance may have degraded.
    """

    def __init__(
        self,
        baseline_importance: Dict[str, float],
        importance_threshold: float = 0.2,
        top_k_features: int = 10,
    ):
        """
        Args:
            baseline_importance: Reference SHAP importance dict
            importance_threshold: Threshold for significant change (fraction)
            top_k_features: Number of top features to track
        """
        self.baseline_importance = baseline_importance
        self.importance_threshold = importance_threshold
        self.top_k_features = top_k_features

        # Normalize baseline
        total = sum(baseline_importance.values())
        self.baseline_normalized = (
            {k: v / total for k, v in baseline_importance.items()}
            if total > 0
            else baseline_importance
        )

        # History tracking
        self.importance_history: List[Dict] = []
        self.drift_events: List[Dict] = []

        logger.info(
            f"DriftExplainer initialized: {len(baseline_importance)} features, "
            f"threshold={importance_threshold}"
        )

    def compute_importance_from_shap(
        self, shap_values: np.ndarray, feature_names: List[str]
    ) -> Dict[str, float]:
        """Compute feature importance from SHAP values."""
        importance = np.abs(shap_values).mean(axis=0)
        return dict(zip(feature_names, importance))

    def analyze_drift(
        self, current_importance: Dict[str, float], timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Analyze drift between baseline and current importance.

        Args:
            current_importance: Current SHAP importance dict
            timestamp: Optional timestamp for this analysis

        Returns:
            Drift analysis results
        """
        timestamp = timestamp or datetime.now()

        # Normalize current importance
        total = sum(current_importance.values())
        current_normalized = (
            {k: v / total for k, v in current_importance.items()}
            if total > 0
            else current_importance
        )

        # Compute changes for each feature
        changes = []
        for feature in set(self.baseline_normalized.keys()) | set(
            current_normalized.keys()
        ):
            baseline_val = self.baseline_normalized.get(feature, 0)
            current_val = current_normalized.get(feature, 0)

            abs_change = current_val - baseline_val
            rel_change = abs_change / baseline_val if baseline_val > 0 else float("inf")

            changes.append(
                {
                    "feature": feature,
                    "baseline_importance": baseline_val,
                    "current_importance": current_val,
                    "absolute_change": abs_change,
                    "relative_change": rel_change,
                    "significant": abs(rel_change) > self.importance_threshold,
                }
            )

        # Sort by absolute change magnitude
        changes.sort(key=lambda x: abs(x["absolute_change"]), reverse=True)

        # Identify significant drifts
        significant_changes = [c for c in changes if c["significant"]]

        # Compute overall drift score
        drift_score = sum(abs(c["absolute_change"]) for c in changes) / 2

        # Determine drift severity
        if drift_score < 0.1:
            severity = "none"
        elif drift_score < 0.2:
            severity = "low"
        elif drift_score < 0.4:
            severity = "medium"
        else:
            severity = "high"

        result = {
            "timestamp": timestamp.isoformat(),
            "drift_score": float(drift_score),
            "severity": severity,
            "num_significant_changes": len(significant_changes),
            "top_changes": changes[: self.top_k_features],
            "significant_changes": significant_changes,
        }

        # Record in history
        self.importance_history.append(
            {
                "timestamp": timestamp.isoformat(),
                "importance": current_importance,
                "drift_score": drift_score,
            }
        )

        # Record drift event if significant
        if severity in ["medium", "high"]:
            self.drift_events.append(result)
            logger.warning(
                f"Concept drift detected: severity={severity}, "
                f"score={drift_score:.3f}, changes={len(significant_changes)}"
            )

        return result

    def explain_drift(self, drift_result: Dict[str, Any]) -> str:
        """
        Generate natural language explanation of drift.

        Args:
            drift_result: Result from analyze_drift()

        Returns:
            Human-readable explanation
        """
        severity = drift_result["severity"]
        score = drift_result["drift_score"]
        changes = drift_result["significant_changes"]

        if severity == "none":
            return (
                "No significant concept drift detected. Feature importance is stable."
            )

        explanation = (
            f"**Concept Drift Detected** (severity: {severity}, score: {score:.3f})\n\n"
        )

        if not changes:
            explanation += "Overall importance distribution has shifted, but no single feature dominates the change."
            return explanation

        # Categorize changes
        increased = [c for c in changes if c["absolute_change"] > 0]
        decreased = [c for c in changes if c["absolute_change"] < 0]

        if increased:
            explanation += "**Features with INCREASED importance:**\n"
            for c in increased[:3]:
                pct = c["relative_change"] * 100
                explanation += f"  - `{c['feature']}`: +{pct:.1f}% (now {c['current_importance']:.3f})\n"
            explanation += "\n"

        if decreased:
            explanation += "**Features with DECREASED importance:**\n"
            for c in decreased[:3]:
                pct = c["relative_change"] * 100
                explanation += f"  - `{c['feature']}`: {pct:.1f}% (now {c['current_importance']:.3f})\n"
            explanation += "\n"

        # Add interpretation
        explanation += "**Possible causes:**\n"

        for c in changes[:2]:
            feat = c["feature"]
            if "temperature" in feat.lower():
                explanation += (
                    f"  - Seasonal weather pattern change affecting `{feat}`\n"
                )
            elif "lag" in feat.lower():
                explanation += (
                    f"  - Change in load autocorrelation structure (`{feat}`)\n"
                )
            elif "hour" in feat.lower() or "day" in feat.lower():
                explanation += (
                    f"  - Shift in temporal consumption patterns (`{feat}`)\n"
                )
            elif "holiday" in feat.lower():
                explanation += f"  - Holiday/calendar effects changing (`{feat}`)\n"
            else:
                explanation += f"  - Data distribution shift in `{feat}`\n"

        explanation += "\n**Recommended actions:**\n"
        if severity == "high":
            explanation += "  - Consider retraining the model with recent data\n"
            explanation += "  - Investigate data quality issues\n"
        else:
            explanation += "  - Monitor closely over next few days\n"
            explanation += "  - Schedule model refresh if drift persists\n"

        return explanation

    def get_importance_trend(self, feature: str, window: int = 10) -> List[Dict]:
        """
        Get importance trend for a specific feature.

        Args:
            feature: Feature name
            window: Number of recent observations

        Returns:
            List of importance values over time
        """
        trend = []
        for record in self.importance_history[-window:]:
            importance = record["importance"].get(feature, 0)
            trend.append(
                {
                    "timestamp": record["timestamp"],
                    "importance": importance,
                }
            )
        return trend

    def update_baseline(self, new_importance: Dict[str, float]):
        """Update baseline importance (after retraining)."""
        self.baseline_importance = new_importance

        total = sum(new_importance.values())
        self.baseline_normalized = (
            {k: v / total for k, v in new_importance.items()}
            if total > 0
            else new_importance
        )

        logger.info("Baseline importance updated")

    def get_summary(self) -> Dict[str, Any]:
        """Get drift monitoring summary."""
        return {
            "num_observations": len(self.importance_history),
            "num_drift_events": len(self.drift_events),
            "recent_drift_scores": [
                h["drift_score"] for h in self.importance_history[-10:]
            ],
            "baseline_top_features": sorted(
                self.baseline_normalized.items(), key=lambda x: -x[1]
            )[: self.top_k_features],
        }

    def save_history(self, path: str):
        """Save drift history to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "baseline_importance": self.baseline_importance,
            "importance_history": self.importance_history,
            "drift_events": self.drift_events,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Drift history saved to {path}")

    @classmethod
    def load_history(cls, path: str) -> "DriftExplainer":
        """Load drift explainer from saved history."""
        with open(path, "r") as f:
            data = json.load(f)

        explainer = cls(baseline_importance=data["baseline_importance"])
        explainer.importance_history = data.get("importance_history", [])
        explainer.drift_events = data.get("drift_events", [])

        logger.info(f"Drift history loaded from {path}")
        return explainer
