"""
SHAP Explanation Module for Credit Scoring Models.
Project: Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring

Implements:
1. Model-specific SHAP explainers:
   - Logistic Regression: LinearExplainer with training-derived background data.
   - Random Forest: TreeExplainer.
   - XGBoost: TreeExplainer (with XGBoost 3.x base_score UBJSON compatibility patch).
2. Canonical feature alignment and ordering preservation.
3. Training-only background data sampling.
4. Consistent prediction output space per model family:
   - Logistic Regression: Log-odds margin space.
   - Random Forest: Probability space P(Y=1|X) for positive default class.
   - XGBoost: Log-odds margin space.
   Both baseline and reweighed models within each model family are explained in the IDENTICAL prediction space.
"""

import io
import json
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd
import scipy.special
import shap
from shap.explainers._tree import XGBTreeModelLoader


# ---------------------------------------------------------------------------
# XGBoost 3.x UBJSON base_score Compatibility Patch for shap.TreeExplainer
# ---------------------------------------------------------------------------
_orig_xgb_loader_init = XGBTreeModelLoader.__init__

def _patched_xgb_loader_init(self, xgb_model):
    from shap.explainers._tree import _check_xgboost_version, decode_ubjson_buffer
    import xgboost as xgb

    _check_xgboost_version(xgb.__version__)
    model = xgb_model.get_booster() if hasattr(xgb_model, "get_booster") else xgb_model
    raw = model.save_raw(raw_format="ubj")
    with io.BytesIO(raw) as fd:
        jmodel = decode_ubjson_buffer(fd)

    learner = jmodel["learner"]
    learner_model_param = learner["learner_model_param"]
    objective = learner["objective"]
    booster = learner["gradient_booster"]

    n_classes = max(int(learner_model_param["num_class"]), 1)
    n_targets = max(int(learner_model_param["num_target"]), 1)
    n_targets = max(n_targets, n_classes)

    if "gbtree" in booster and "model" not in booster:
        booster = booster["gbtree"]

    iteration_indptr = np.asarray(booster["model"]["iteration_indptr"], dtype=np.int32)
    diff = np.diff(iteration_indptr)

    self.n_trees_per_iter = int(diff[0])
    self.n_targets = n_targets

    # Strip square brackets if present (e.g. '[3.1E-1]' in XGBoost 3.x)
    raw_base_score_str = str(learner_model_param["base_score"]).strip("[]")
    raw_base_score = float(raw_base_score_str)
    self.base_score = raw_base_score

    self.name_obj = objective["name"]
    self.name_gbm = booster["name"]
    if self.name_obj in ("binary:logistic", "reg:logistic"):
        self.base_score = scipy.special.logit(raw_base_score)
    else:
        self.base_score = raw_base_score

    self.num_feature = int(learner_model_param["num_feature"])
    self.num_class = int(learner_model_param["num_class"])

    trees = booster["model"]["trees"]
    self.num_trees = len(trees)
    self.node_parents = []
    self.node_cleft = []
    self.node_cright = []
    self.node_sindex = []
    self.children_default = []
    self.sum_hess = []
    self.values = []
    self.thresholds = []
    self.threshold_types = []
    self.features = []
    self.split_types = []
    self.categories = []

    feature_types = model.feature_types
    if feature_types is not None:
        cat_feature_indices = np.where(np.asarray(feature_types) == "c")[0]
        self.cat_feature_indices = None if len(cat_feature_indices) == 0 else cat_feature_indices
    else:
        self.cat_feature_indices = None

    def to_integers(data):
        return np.asanyarray(data, dtype=np.uint8)

    for i in range(self.num_trees):
        tree = trees[i]
        parents = np.asarray(tree["parents"])
        self.node_parents.append(parents)
        self.node_cleft.append(np.asarray(tree["left_children"], dtype=np.int32))
        self.node_cright.append(np.asarray(tree["right_children"], dtype=np.int32))
        self.node_sindex.append(np.asarray(tree["split_indices"], dtype=np.uint32))

        default_left = to_integers(tree["default_left"])
        default_child = np.where(default_left == 1, self.node_cleft[-1], self.node_cright[-1]).astype(np.int64)
        self.children_default.append(default_child)
        self.sum_hess.append(np.asarray(tree["sum_hessian"], dtype=np.float64))

        is_leaf = self.node_cleft[-1] == -1
        split_cond = np.asarray(tree["split_conditions"], dtype=np.float32)
        leaf_weight = np.where(is_leaf, split_cond, 0.0)
        thresholds = np.where(is_leaf, 0.0, split_cond)
        thresholds = np.where(is_leaf, 0.0, np.nextafter(thresholds, -np.float32(np.inf)))
        threshold_types = np.zeros_like(thresholds, dtype=np.int32)

        self.values.append(leaf_weight.reshape(leaf_weight.size, 1))
        self.thresholds.append(thresholds)
        self.threshold_types.append(threshold_types)
        split_idx = np.asarray(tree["split_indices"], dtype=np.int64)
        self.features.append(split_idx)
        split_types = to_integers(tree["split_type"])
        self.split_types.append(split_types)
        cat_segments = tree["categories_segments"]
        cat_sizes = tree["categories_sizes"]
        cat_nodes = tree["categories_nodes"]
        cats = tree["categories"]
        tree_categories = self.parse_categories(cat_nodes, cat_segments, cat_sizes, cats, self.node_cleft[-1])
        self.categories.append(tree_categories)

# Apply patch to enable shap.TreeExplainer on XGBoost 3.x
XGBTreeModelLoader.__init__ = _patched_xgb_loader_init


# ---------------------------------------------------------------------------
# Background Data Sampling & Cohort Utilities
# ---------------------------------------------------------------------------
def sample_background_data(
    X_train_proc: np.ndarray,
    n_samples: int = 100,
    random_state: int = 42
) -> np.ndarray:
    """
    Sample a reproducible background dataset derived strictly from training data.
    Never uses test labels or protected demographic attributes.
    """
    n_total = X_train_proc.shape[0]
    if n_total <= n_samples:
        return X_train_proc.copy()

    rng = np.random.RandomState(random_state)
    sample_indices = rng.choice(n_total, size=n_samples, replace=False)
    return X_train_proc[sample_indices].copy()


def get_model_explainer(
    model: Any,
    model_name: str,
    background_data: Optional[np.ndarray] = None
) -> shap.Explainer:
    """
    Instantiate appropriate SHAP explainer for the given model architecture.
    """
    name = model_name.lower().strip()
    if name in ("logistic_regression", "lr"):
        if background_data is None:
            raise ValueError("LinearExplainer requires background_data derived from training set.")
        return shap.LinearExplainer(model, background_data)

    elif name in ("random_forest", "rf"):
        return shap.TreeExplainer(model)

    elif name in ("xgboost", "xgb"):
        return shap.TreeExplainer(model)

    else:
        raise ValueError(f"Unsupported model architecture '{model_name}' for SHAP.")


def compute_shap_attributions(
    explainer: shap.Explainer,
    model_name: str,
    X_eval_proc: np.ndarray
) -> np.ndarray:
    """
    Compute SHAP attribution matrix (N, P) explaining the Default risk class (Y=1).
    """
    name = model_name.lower().strip()

    if name in ("logistic_regression", "lr"):
        raw_shap = explainer.shap_values(X_eval_proc)
        if isinstance(raw_shap, list):
            # Binary classification list: [class_0, class_1]
            return np.asarray(raw_shap[1], dtype=np.float64)
        elif raw_shap.ndim == 3:
            return np.asarray(raw_shap[:, :, 1], dtype=np.float64)
        else:
            return np.asarray(raw_shap, dtype=np.float64)

    elif name in ("random_forest", "rf"):
        raw_shap = explainer.shap_values(X_eval_proc, check_additivity=False)
        if isinstance(raw_shap, list):
            return np.asarray(raw_shap[1], dtype=np.float64)
        elif raw_shap.ndim == 3:
            return np.asarray(raw_shap[:, :, 1], dtype=np.float64)
        else:
            return np.asarray(raw_shap, dtype=np.float64)

    elif name in ("xgboost", "xgb"):
        raw_shap = explainer.shap_values(X_eval_proc, check_additivity=False)
        if isinstance(raw_shap, list):
            return np.asarray(raw_shap[1], dtype=np.float64)
        elif raw_shap.ndim == 3:
            return np.asarray(raw_shap[:, :, 1], dtype=np.float64)
        else:
            return np.asarray(raw_shap, dtype=np.float64)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")
