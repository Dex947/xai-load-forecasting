"""A/B testing framework for model comparison and promotion."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime
from scipy import stats
import json
from pathlib import Path

from src.logger import get_logger

logger = get_logger(__name__)


class ABTestManager:
    """
    A/B testing framework for comparing champion vs challenger models.

    Supports statistical significance testing for safe model promotion.
    """

    def __init__(
        self,
        champion_model: Any,
        challenger_model: Any,
        traffic_split: float = 0.1,
        min_samples: int = 100,
        significance_level: float = 0.05,
    ):
        """
        Args:
            champion_model: Current production model
            challenger_model: New model to test
            traffic_split: Fraction of traffic to challenger (0-1)
            min_samples: Minimum samples before significance test
            significance_level: P-value threshold for significance
        """
        self.champion = champion_model
        self.challenger = challenger_model
        self.traffic_split = traffic_split
        self.min_samples = min_samples
        self.significance_level = significance_level

        # Results tracking
        self.champion_errors: List[float] = []
        self.challenger_errors: List[float] = []
        self.champion_predictions: List[Tuple[float, float]] = []  # (pred, actual)
        self.challenger_predictions: List[Tuple[float, float]] = []

        # Test state
        self.test_started = datetime.now()
        self.test_concluded = False
        self.winner: Optional[str] = None

        logger.info(
            f"ABTestManager initialized: split={traffic_split}, "
            f"min_samples={min_samples}"
        )

    def predict(
        self, X: pd.DataFrame, return_model: bool = False
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Route prediction to champion or challenger based on split.

        Args:
            X: Feature DataFrame
            return_model: Return which model was used

        Returns:
            Predictions and optionally model name
        """
        # Determine routing for each sample
        n_samples = len(X)
        use_challenger = np.random.random(n_samples) < self.traffic_split
        champion_mask = ~use_challenger

        predictions = np.zeros(n_samples)
        models_used = [""] * n_samples  # Pre-allocate to maintain order

        # Champion predictions
        if champion_mask.any():
            champion_preds = self.champion.predict(X[champion_mask])
            predictions[champion_mask] = champion_preds
            for i, is_champion in enumerate(champion_mask):
                if is_champion:
                    models_used[i] = "champion"

        # Challenger predictions
        if use_challenger.any():
            challenger_preds = self.challenger.predict(X[use_challenger])
            predictions[use_challenger] = challenger_preds
            for i, is_challenger in enumerate(use_challenger):
                if is_challenger:
                    models_used[i] = "challenger"

        if return_model:
            return predictions, models_used
        return predictions, None

    def record_outcome(self, y_pred: float, y_true: float, model: str):
        """
        Record prediction outcome for analysis.

        Args:
            y_pred: Predicted value
            y_true: Actual value
            model: 'champion' or 'challenger'
        """
        error = abs(y_true - y_pred)

        if model == "champion":
            self.champion_errors.append(error)
            self.champion_predictions.append((y_pred, y_true))
        else:
            self.challenger_errors.append(error)
            self.challenger_predictions.append((y_pred, y_true))

    def record_batch(self, X: pd.DataFrame, y_true: pd.Series):
        """
        Record batch of predictions and outcomes.

        Args:
            X: Feature DataFrame
            y_true: True values
        """
        predictions, models = self.predict(X, return_model=True)

        for i, (pred, actual, model) in enumerate(
            zip(predictions, y_true.values, models)
        ):
            self.record_outcome(pred, actual, model)

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get current metrics for both models."""
        metrics = {}

        if self.champion_errors:
            metrics["champion"] = {
                "n_samples": len(self.champion_errors),
                "mae": np.mean(self.champion_errors),
                "rmse": np.sqrt(np.mean(np.array(self.champion_errors) ** 2)),
                "std": np.std(self.champion_errors),
            }

        if self.challenger_errors:
            metrics["challenger"] = {
                "n_samples": len(self.challenger_errors),
                "mae": np.mean(self.challenger_errors),
                "rmse": np.sqrt(np.mean(np.array(self.challenger_errors) ** 2)),
                "std": np.std(self.challenger_errors),
            }

        return metrics

    def test_significance(self) -> Dict[str, Any]:
        """
        Perform statistical significance test.

        Uses Welch's t-test to compare mean absolute errors.

        Returns:
            Test results dict
        """
        n_champion = len(self.champion_errors)
        n_challenger = len(self.challenger_errors)

        if n_champion < self.min_samples or n_challenger < self.min_samples:
            return {
                "sufficient_data": False,
                "champion_samples": n_champion,
                "challenger_samples": n_challenger,
                "min_required": self.min_samples,
            }

        # Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(
            self.champion_errors, self.challenger_errors, equal_var=False
        )

        champion_mae = np.mean(self.champion_errors)
        challenger_mae = np.mean(self.challenger_errors)

        # Determine winner
        significant = p_value < self.significance_level
        challenger_better = challenger_mae < champion_mae

        if significant and challenger_better:
            winner = "challenger"
            recommendation = "PROMOTE challenger to production"
        elif significant and not challenger_better:
            winner = "champion"
            recommendation = "KEEP champion in production"
        else:
            winner = None
            recommendation = "CONTINUE testing (no significant difference)"

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(self.champion_errors) + np.var(self.challenger_errors)) / 2
        )
        cohens_d = (champion_mae - challenger_mae) / pooled_std if pooled_std > 0 else 0

        # Improvement percentage
        improvement_pct = (
            (champion_mae - challenger_mae) / champion_mae * 100
            if champion_mae > 0
            else 0
        )

        result = {
            "sufficient_data": True,
            "champion_mae": float(champion_mae),
            "challenger_mae": float(challenger_mae),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": significant,
            "significance_level": self.significance_level,
            "cohens_d": float(cohens_d),
            "improvement_pct": float(improvement_pct),
            "winner": winner,
            "recommendation": recommendation,
            "champion_samples": n_champion,
            "challenger_samples": n_challenger,
        }

        logger.info(
            f"Significance test: p={p_value:.4f}, "
            f"improvement={improvement_pct:.2f}%, "
            f"recommendation={recommendation}"
        )

        return result

    def conclude_test(self) -> Dict[str, Any]:
        """
        Conclude the A/B test and return final results.

        Returns:
            Final test results
        """
        results = self.test_significance()

        if results.get("sufficient_data"):
            self.test_concluded = True
            self.winner = results.get("winner")

        results["test_duration_hours"] = (
            datetime.now() - self.test_started
        ).total_seconds() / 3600

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get complete test summary."""
        metrics = self.get_metrics()
        significance = self.test_significance()

        return {
            "test_started": self.test_started.isoformat(),
            "test_concluded": self.test_concluded,
            "traffic_split": self.traffic_split,
            "metrics": metrics,
            "significance_test": significance,
            "winner": self.winner,
        }

    def save_results(self, path: str):
        """Save test results to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        results = self.get_summary()

        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj

        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=convert)

        logger.info(f"A/B test results saved to {path}")

    def should_promote_challenger(self) -> bool:
        """Check if challenger should be promoted based on test results."""
        results = self.test_significance()

        return (
            results.get("sufficient_data", False)
            and results.get("significant", False)
            and results.get("winner") == "challenger"
        )


class MultiArmedBandit:
    """
    Thompson Sampling bandit for adaptive model selection.

    Automatically routes more traffic to better-performing models.
    """

    def __init__(
        self, models: Dict[str, Any], prior_alpha: float = 1.0, prior_beta: float = 1.0
    ):
        """
        Args:
            models: Dict of model_name -> model
            prior_alpha: Beta prior alpha parameter
            prior_beta: Beta prior beta parameter
        """
        self.models = models
        self.model_names = list(models.keys())

        # Beta distribution parameters for each model
        self.alphas = {name: prior_alpha for name in self.model_names}
        self.betas = {name: prior_beta for name in self.model_names}

        # Performance tracking
        self.successes = {name: 0 for name in self.model_names}
        self.failures = {name: 0 for name in self.model_names}

        logger.info(f"MultiArmedBandit initialized with {len(models)} models")

    def select_model(self) -> str:
        """Select model using Thompson Sampling."""
        samples = {
            name: np.random.beta(self.alphas[name], self.betas[name])
            for name in self.model_names
        }
        return max(samples, key=samples.get)

    def update(self, model_name: str, error: float, threshold: float = 1.0):
        """
        Update model statistics based on prediction error.

        Args:
            model_name: Model that made prediction
            error: Absolute prediction error
            threshold: Error threshold for success/failure
        """
        if error < threshold:
            self.successes[model_name] += 1
            self.alphas[model_name] += 1
        else:
            self.failures[model_name] += 1
            self.betas[model_name] += 1

    def get_selection_probabilities(self, n_samples: int = 1000) -> Dict[str, float]:
        """Estimate selection probability for each model."""
        selections = {name: 0 for name in self.model_names}

        for _ in range(n_samples):
            selected = self.select_model()
            selections[selected] += 1

        return {name: count / n_samples for name, count in selections.items()}

    def get_summary(self) -> Dict[str, Any]:
        """Get bandit summary."""
        return {
            "models": self.model_names,
            "successes": self.successes,
            "failures": self.failures,
            "selection_probabilities": self.get_selection_probabilities(),
        }
