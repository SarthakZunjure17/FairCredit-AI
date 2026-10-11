"""
Fairlearn & Standard Fairness Mitigation Methods Module.
Project: Responsible AI Credit Scoring Research

Methods:
1. Pre-processing: Reweighing (Kamiran & Calders, 2012)
   - Computes statistical independence weights strictly on training data
   - Passes sample_weight to pipeline.fit()
2. In-processing: Exponentiated Gradient (Agarwal et al., 2018 via Fairlearn)
   - Reduction approach with EqualizedOdds constraint
   - Optimizes cost-sensitive base estimators to enforce fairness constraint
3. Post-processing: ThresholdOptimizer (Hardt et al., 2016 via Fairlearn)
   - Post-processes baseline model scores to find group-specific optimal thresholds
   - Fitted strictly on training partition using prefit baseline estimator
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer

from src.modeling.models import get_model, build_pipeline


def compute_reweighing_weights(
    y: pd.Series,
    sensitive_features: pd.Series
) -> np.ndarray:
    """
    Compute Kamiran & Calders (2012) reweighing sample weights strictly on training data.

    Formula for each observation i with sensitive group a_i and label y_i:
        W(a, y) = [P(A = a) * P(Y = y)] / P(A = a, Y = y)
                = [N(A = a) * N(Y = y)] / [N * N(A = a, Y = y)]

    Enforces statistical independence between sensitive attribute and label in the
    reweighted training distribution: P~(A = a, Y = y) = P(A = a) * P(Y = y).
    """
    y_arr = np.asarray(y)
    sens_arr = np.asarray(sensitive_features)
    n_total = len(y_arr)
    weights = np.ones(n_total, dtype=np.float64)

    unique_groups = np.unique(sens_arr)
    unique_labels = np.unique(y_arr)

    for a in unique_groups:
        n_a = np.sum(sens_arr == a)
        for y_val in unique_labels:
            n_y = np.sum(y_arr == y_val)
            mask = (sens_arr == a) & (y_arr == y_val)
            n_ay = np.sum(mask)

            if n_ay > 0:
                w = (float(n_a) * float(n_y)) / (float(n_total) * float(n_ay))
                weights[mask] = w
            else:
                weights[mask] = 1.0

    return weights


def fit_reweighing(
    base_model_name: str,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sens_train: pd.Series,
    random_state: int = 42
) -> Pipeline:
    """
    Fit pre-processing Reweighing pipeline on training partition.
    """
    # 1. Compute training weights
    sample_weights = compute_reweighing_weights(y_train, sens_train)

    # 2. Build model and pipeline
    base_model = get_model(base_model_name, random_state=random_state)
    pipeline = build_pipeline(preprocessor, base_model)

    # 3. Fit pipeline with sample weights directed to classifier step
    pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)

    return pipeline


def fit_exponentiated_gradient(
    base_model_name: str,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sens_train: pd.Series,
    random_state: int = 42,
    eps: float = 0.01,
    max_iter: int = 20
) -> Tuple[ColumnTransformer, ExponentiatedGradient]:
    """
    Fit in-processing Exponentiated Gradient with EqualizedOdds constraint.

    To strictly avoid pipeline sample_weight conflicts and prevent leakage,
    preprocessor is fitted strictly on X_train, and transformed features are passed
    to ExponentiatedGradient.
    """
    # 1. Fit preprocessor strictly on X_train
    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)

    # 2. Instantiate base model
    base_model = get_model(base_model_name, random_state=random_state)

    # 3. Instantiate and fit ExponentiatedGradient
    exp_grad = ExponentiatedGradient(
        estimator=base_model,
        constraints=EqualizedOdds(),
        eps=eps,
        max_iter=max_iter
    )
    exp_grad.fit(X_train_proc, y_train, sensitive_features=sens_train)

    return preprocessor, exp_grad


def fit_threshold_optimizer(
    fitted_baseline_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sens_train: pd.Series,
    constraints: str = "equalized_odds"
) -> ThresholdOptimizer:
    """
    Fit post-processing ThresholdOptimizer on prefit baseline model using training partition.
    """
    to = ThresholdOptimizer(
        estimator=fitted_baseline_pipeline,
        constraints=constraints,
        predict_method="predict_proba",
        prefit=True
    )
    to.fit(X_train, y_train, sensitive_features=sens_train)

    return to
