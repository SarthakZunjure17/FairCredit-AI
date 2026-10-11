"""
Performance and Fairness Evaluation Metrics Module.
Project: Responsible AI Credit Scoring Research

Metrics computed:
1. Predictive Performance:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - ROC-AUC
2. Group Fairness (via Fairlearn & explicit mathematical definitions):
   - Equalized Odds Difference (primary fairness metric: max(|delta_TPR|, |delta_FPR|))
   - Equal Opportunity Difference (|delta_TPR|)
   - Demographic Parity Difference (|delta_SelectionRate|)
   - Disparate Impact (Demographic Parity Ratio: min_SR / max_SR)
   - Disparate Impact on Approvals (Favorable outcome ratio: ApprovalRate_female / ApprovalRate_male)
   - Signed differences and subgroup breakdown rates
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from fairlearn.metrics import (
    equalized_odds_difference,
    demographic_parity_difference,
    demographic_parity_ratio,
    true_positive_rate,
    false_positive_rate,
    selection_rate,
    MetricFrame
)


def evaluate_performance(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """
    Compute standard classification performance metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
        y_prob: Predicted probability for positive class (1 = Default/Bad Risk)

    Returns:
        Dictionary of performance metrics rounded to 4 decimal places.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_prob_arr = np.asarray(y_prob)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4)
    }


def evaluate_fairness(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    sensitive_features: Union[np.ndarray, pd.Series],
    protected_attribute_name: str = "sex",
    unprivileged_group: str = "female",
    privileged_group: str = "male"
) -> Dict[str, Any]:
    """
    Compute group fairness metrics using Fairlearn protocol and subgroup rates.

    Convention:
        - Target y=1: Credit Default / Bad Risk Event.
        - Target y=0: Non-Default / Good Credit / Favorable Approval.
        - Equalized Odds Difference: max(|TPR_unprivileged - TPR_privileged|, |FPR_unprivileged - FPR_privileged|)
        - Equal Opportunity Difference: |TPR_unprivileged - TPR_privileged|
        - Demographic Parity Difference: |SelectionRate_unprivileged - SelectionRate_privileged|
        - Disparate Impact (Fairlearn ratio): min(SelectionRate) / max(SelectionRate)
        - Disparate Impact on Approvals: P(pred=0 | unprivileged) / P(pred=0 | privileged)

    Args:
        y_true: Ground truth binary target (0 or 1)
        y_pred: Predicted binary target (0 or 1)
        sensitive_features: Array or series of sensitive group values
        protected_attribute_name: Name of protected feature
        unprivileged_group: Label of unprivileged group (e.g. 'female' or '<=25')
        privileged_group: Label of privileged group (e.g. 'male' or '>25')

    Returns:
        Dictionary of fairness metrics and subgroup breakdown rates.
    """
    y_true_s = pd.Series(y_true).reset_index(drop=True)
    y_pred_s = pd.Series(y_pred).reset_index(drop=True)
    sens_s = pd.Series(sensitive_features).reset_index(drop=True)

    # Subgroup breakdown via Fairlearn MetricFrame
    mf = MetricFrame(
        metrics={
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
            "selection_rate": selection_rate
        },
        y_true=y_true_s,
        y_pred=y_pred_s,
        sensitive_features=sens_s
    )

    by_group = mf.by_group.to_dict()

    # Fairlearn standard metric calculations
    eod_diff = float(equalized_odds_difference(y_true_s, y_pred_s, sensitive_features=sens_s))
    dp_diff = float(demographic_parity_difference(y_true_s, y_pred_s, sensitive_features=sens_s))
    dp_ratio = float(demographic_parity_ratio(y_true_s, y_pred_s, sensitive_features=sens_s))

    # Equal Opportunity difference: |TPR_1 - TPR_2|
    tpr_dict = by_group.get("tpr", {})
    fpr_dict = by_group.get("fpr", {})
    sr_dict = by_group.get("selection_rate", {})

    tpr_unprivileged = tpr_dict.get(unprivileged_group, np.nan)
    tpr_privileged = tpr_dict.get(privileged_group, np.nan)
    fpr_unprivileged = fpr_dict.get(unprivileged_group, np.nan)
    fpr_privileged = fpr_dict.get(privileged_group, np.nan)
    sr_unprivileged = sr_dict.get(unprivileged_group, np.nan)
    sr_privileged = sr_dict.get(privileged_group, np.nan)

    if not np.isnan(tpr_unprivileged) and not np.isnan(tpr_privileged):
        equal_opp_diff = float(abs(tpr_unprivileged - tpr_privileged))
        equal_opp_signed = float(tpr_unprivileged - tpr_privileged)
    else:
        # Fallback across all groups
        tpr_vals = list(tpr_dict.values())
        equal_opp_diff = float(max(tpr_vals) - min(tpr_vals)) if tpr_vals else 0.0
        equal_opp_signed = equal_opp_diff

    # Demographic Parity signed
    if not np.isnan(sr_unprivileged) and not np.isnan(sr_privileged):
        dp_signed = float(sr_unprivileged - sr_privileged)
    else:
        dp_signed = dp_diff

    # Favorable outcome approval rates: P(y_pred = 0) = 1 - selection_rate
    approval_unprivileged = 1.0 - sr_unprivileged if not np.isnan(sr_unprivileged) else np.nan
    approval_privileged = 1.0 - sr_privileged if not np.isnan(sr_privileged) else np.nan
    if not np.isnan(approval_unprivileged) and not np.isnan(approval_privileged) and approval_privileged > 0:
        disparate_impact_approval = float(approval_unprivileged / approval_privileged)
    else:
        disparate_impact_approval = np.nan

    return {
        "protected_attribute": protected_attribute_name,
        "equalized_odds_difference": round(eod_diff, 4),
        "equal_opportunity_difference": round(equal_opp_diff, 4),
        "equal_opportunity_signed": round(equal_opp_signed, 4),
        "demographic_parity_difference": round(dp_diff, 4),
        "demographic_parity_signed": round(dp_signed, 4),
        "disparate_impact": round(dp_ratio, 4),
        "disparate_impact_approval_ratio": round(disparate_impact_approval, 4) if not np.isnan(disparate_impact_approval) else None,
        "unprivileged_group": unprivileged_group,
        "privileged_group": privileged_group,
        "unprivileged_selection_rate": round(sr_unprivileged, 4) if not np.isnan(sr_unprivileged) else None,
        "privileged_selection_rate": round(sr_privileged, 4) if not np.isnan(sr_privileged) else None,
        "unprivileged_tpr": round(tpr_unprivileged, 4) if not np.isnan(tpr_unprivileged) else None,
        "privileged_tpr": round(tpr_privileged, 4) if not np.isnan(tpr_privileged) else None,
        "unprivileged_fpr": round(fpr_unprivileged, 4) if not np.isnan(fpr_unprivileged) else None,
        "privileged_fpr": round(fpr_privileged, 4) if not np.isnan(fpr_privileged) else None,
    }
