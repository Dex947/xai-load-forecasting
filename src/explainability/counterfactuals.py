"""Counterfactual explanations using DiCE for what-if analysis."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any, Union
from pathlib import Path

from src.logger import get_logger

logger = get_logger(__name__)

try:
    import dice_ml
    from dice_ml import Dice
    DICE_AVAILABLE = True
except ImportError:
    DICE_AVAILABLE = False
    logger.warning("DiCE not installed. Install with: pip install dice-ml")


class CounterfactualExplainer:
    """
    Generates counterfactual explanations for load forecasts.
    
    Answers: "What would need to change for the prediction to be X?"
    """
    
    def __init__(
        self,
        model: Any,
        training_data: pd.DataFrame,
        target_column: str = "load",
        continuous_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ):
        """
        Args:
            model: Trained prediction model
            training_data: Training data for feature ranges
            target_column: Name of target column
            continuous_features: List of continuous feature names
            categorical_features: List of categorical feature names
        """
        if not DICE_AVAILABLE:
            raise ImportError("DiCE required. Install with: pip install dice-ml")
        
        self.model = model
        self.target_column = target_column
        
        # Separate features and target
        if target_column in training_data.columns:
            self.feature_data = training_data.drop(columns=[target_column])
        else:
            self.feature_data = training_data
        
        # Infer feature types if not provided
        if continuous_features is None:
            continuous_features = self.feature_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()
        
        if categorical_features is None:
            categorical_features = self.feature_data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
        
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        
        # Create DiCE data object
        self.dice_data = dice_ml.Data(
            dataframe=self.feature_data,
            continuous_features=continuous_features,
            outcome_name=None  # No outcome in feature data
        )
        
        # Create DiCE model wrapper
        self.dice_model = dice_ml.Model(
            model=self._create_model_wrapper(),
            backend="sklearn"
        )
        
        # Create DiCE explainer
        self.explainer = Dice(self.dice_data, self.dice_model, method="random")
        
        logger.info(
            f"CounterfactualExplainer initialized: "
            f"{len(continuous_features)} continuous, "
            f"{len(categorical_features)} categorical features"
        )
    
    def _create_model_wrapper(self):
        """Create sklearn-compatible wrapper for the model."""
        class ModelWrapper:
            def __init__(self, model):
                self.model = model
            
            def predict(self, X):
                if hasattr(self.model, "predict"):
                    return self.model.predict(X)
                return self.model(X)
        
        return ModelWrapper(self.model)
    
    def generate_counterfactuals(
        self,
        query_instance: pd.DataFrame,
        desired_outcome: float,
        num_counterfactuals: int = 3,
        features_to_vary: Optional[List[str]] = None,
        permitted_range: Optional[Dict[str, List]] = None,
        diversity_weight: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate counterfactual explanations.
        
        Args:
            query_instance: Single row DataFrame with features
            desired_outcome: Target prediction value
            num_counterfactuals: Number of counterfactuals to generate
            features_to_vary: Features allowed to change (default: all)
            permitted_range: Allowed ranges for features
            diversity_weight: Weight for diversity in counterfactuals
            
        Returns:
            Dict with counterfactuals and analysis
        """
        if len(query_instance) != 1:
            raise ValueError("query_instance must be a single row")
        
        # Get current prediction
        current_pred = self.model.predict(query_instance)[0]
        
        # Determine direction
        if desired_outcome > current_pred:
            desired_range = [desired_outcome * 0.95, desired_outcome * 1.05]
        else:
            desired_range = [desired_outcome * 0.95, desired_outcome * 1.05]
        
        logger.info(
            f"Generating counterfactuals: current={current_pred:.2f}, "
            f"desired={desired_outcome:.2f}"
        )
        
        try:
            # Generate counterfactuals
            cf_result = self.explainer.generate_counterfactuals(
                query_instance,
                total_CFs=num_counterfactuals,
                desired_range=desired_range,
                features_to_vary=features_to_vary or "all",
                permitted_range=permitted_range,
                diversity_weight=diversity_weight
            )
            
            # Extract counterfactuals
            cf_df = cf_result.cf_examples_list[0].final_cfs_df
            
            if cf_df is None or len(cf_df) == 0:
                return {
                    "success": False,
                    "message": "No valid counterfactuals found",
                    "current_prediction": float(current_pred),
                    "desired_outcome": float(desired_outcome),
                }
            
            # Analyze changes
            changes = self._analyze_changes(query_instance, cf_df)
            
            return {
                "success": True,
                "current_prediction": float(current_pred),
                "desired_outcome": float(desired_outcome),
                "num_counterfactuals": len(cf_df),
                "counterfactuals": cf_df.to_dict(orient="records"),
                "changes": changes,
                "summary": self._generate_summary(changes, current_pred, desired_outcome),
            }
            
        except Exception as e:
            logger.error(f"Counterfactual generation failed: {e}")
            return {
                "success": False,
                "message": str(e),
                "current_prediction": float(current_pred),
                "desired_outcome": float(desired_outcome),
            }
    
    def _analyze_changes(
        self,
        original: pd.DataFrame,
        counterfactuals: pd.DataFrame
    ) -> List[Dict]:
        """Analyze what changed in each counterfactual."""
        changes = []
        
        for i, cf_row in counterfactuals.iterrows():
            row_changes = []
            
            for col in original.columns:
                if col in cf_row.index:
                    orig_val = original[col].iloc[0]
                    cf_val = cf_row[col]
                    
                    if orig_val != cf_val:
                        change = {
                            "feature": col,
                            "original": float(orig_val) if isinstance(orig_val, (int, float)) else orig_val,
                            "counterfactual": float(cf_val) if isinstance(cf_val, (int, float)) else cf_val,
                        }
                        
                        if isinstance(orig_val, (int, float)) and isinstance(cf_val, (int, float)):
                            change["delta"] = float(cf_val - orig_val)
                            change["pct_change"] = float((cf_val - orig_val) / orig_val * 100) if orig_val != 0 else 0
                        
                        row_changes.append(change)
            
            changes.append({
                "counterfactual_id": i,
                "num_changes": len(row_changes),
                "changes": row_changes,
            })
        
        return changes
    
    def _generate_summary(
        self,
        changes: List[Dict],
        current: float,
        desired: float
    ) -> str:
        """Generate natural language summary of counterfactuals."""
        if not changes:
            return "No counterfactuals could be generated."
        
        direction = "increase" if desired > current else "decrease"
        delta = abs(desired - current)
        
        # Find most common changes
        feature_counts = {}
        for cf in changes:
            for change in cf["changes"]:
                feat = change["feature"]
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
        
        if not feature_counts:
            return f"To {direction} load by {delta:.2f} kW, no simple changes were found."
        
        top_features = sorted(feature_counts.items(), key=lambda x: -x[1])[:3]
        
        summary = f"To {direction} load from {current:.2f} to {desired:.2f} kW:\n"
        for feat, count in top_features:
            summary += f"  - Adjust '{feat}' ({count}/{len(changes)} counterfactuals)\n"
        
        return summary
    
    def what_if_analysis(
        self,
        base_instance: pd.DataFrame,
        feature_changes: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Perform what-if analysis with specific feature changes.
        
        Args:
            base_instance: Original feature values
            feature_changes: Dict of feature -> new value
            
        Returns:
            Analysis results
        """
        # Get base prediction
        base_pred = self.model.predict(base_instance)[0]
        
        # Apply changes
        modified = base_instance.copy()
        for feat, value in feature_changes.items():
            if feat in modified.columns:
                modified[feat] = value
        
        # Get new prediction
        new_pred = self.model.predict(modified)[0]
        
        return {
            "base_prediction": float(base_pred),
            "modified_prediction": float(new_pred),
            "delta": float(new_pred - base_pred),
            "pct_change": float((new_pred - base_pred) / base_pred * 100) if base_pred != 0 else 0,
            "changes_applied": feature_changes,
        }
    
    def sensitivity_analysis(
        self,
        base_instance: pd.DataFrame,
        feature: str,
        values: List[float]
    ) -> pd.DataFrame:
        """
        Analyze prediction sensitivity to a single feature.
        
        Args:
            base_instance: Original feature values
            feature: Feature to vary
            values: Values to test
            
        Returns:
            DataFrame with predictions for each value
        """
        results = []
        
        for val in values:
            modified = base_instance.copy()
            modified[feature] = val
            pred = self.model.predict(modified)[0]
            
            results.append({
                feature: val,
                "prediction": pred,
            })
        
        return pd.DataFrame(results)
