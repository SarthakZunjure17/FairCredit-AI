"""
Model Factory and Pipeline Construction for Credit Scoring Benchmarks.
Project: Responsible AI Credit Scoring Research

Models:
1. Logistic Regression (Linear baseline)
2. Random Forest (Bagging ensemble baseline)
3. XGBoost (Gradient boosting baseline)

Rules:
- Standard baseline configurations without hyperparameter tuning.
- Fixed reproducible random_state = 42.
- Integrated into sklearn Pipeline with training-only preprocessing.
"""

from typing import Dict, Any
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def get_model(model_name: str, random_state: int = 42) -> BaseEstimator:
    """
    Instantiate baseline model with default/standard parameters and fixed random seed.

    Args:
        model_name: 'logistic_regression', 'random_forest', or 'xgboost'
        random_state: Fixed seed for reproducibility (default: 42)

    Returns:
        Configured scikit-learn or XGBoost classifier instance
    """
    name = model_name.lower().strip().replace(" ", "_").replace("-", "_")

    if name in ("logistic_regression", "lr"):
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state
        )

    elif name in ("random_forest", "rf"):
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1
        )

    elif name in ("xgboost", "xgb"):
        return XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )

    else:
        raise ValueError(
            f"Unsupported model_name '{model_name}'. "
            "Supported options: 'logistic_regression', 'random_forest', 'xgboost'."
        )


def build_pipeline(preprocessor: Any, model: BaseEstimator) -> Pipeline:
    """
    Construct end-to-end Pipeline ensuring preprocessor is fitted strictly on training data.
    """
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
