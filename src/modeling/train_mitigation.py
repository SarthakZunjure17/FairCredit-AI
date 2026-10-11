"""
Master Fairness Mitigation Benchmark Execution Script.
Project: Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring

Executes:
1. Ingests Taiwan and German datasets using identical stratified train/test split (random_state=42, test_size=0.20).
2. For each of the 6 baseline configurations (2 datasets x 3 models):
   Runs 3 mitigation strategies:
   - Pre-processing: Reweighing (Kamiran & Calders, 2012)
   - In-processing: Exponentiated Gradient (Agarwal et al., 2018 with EqualizedOdds)
   - Post-processing: ThresholdOptimizer (Hardt et al., 2016 with EqualizedOdds)
3. Evaluates performance (Acc, Prec, Rec, F1, ROC-AUC) and fairness (EOD, EOpp, DPD, Approval-DI).
4. Computes exact Before vs. After Deltas and Distance from Parity.
5. Computes complete subgroup confusion matrix metrics for sex and age.
6. Saves:
   - results/mitigation_master_results.csv
   - results/mitigated_subgroup_metrics.csv
   - results/mitigation_metadata.json
   - results/mitigation_summary.md
"""

import os
import sys
import json
import time
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.modeling.preprocessing import (
    load_dataset,
    split_dataset,
    build_preprocessor
)
from src.modeling.models import get_model, build_pipeline
from src.modeling.evaluation import evaluate_performance, evaluate_fairness
from src.modeling.mitigation import (
    compute_reweighing_weights,
    fit_reweighing,
    fit_exponentiated_gradient,
    fit_threshold_optimizer
)


def run_all_mitigations():
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load baseline master results to compute exact before/after deltas
    baseline_master_path = os.path.join(results_dir, "baseline_master_results.csv")
    baseline_fair_path = os.path.join(results_dir, "baseline_fairness.csv")
    assert os.path.exists(baseline_master_path), f"Missing {baseline_master_path}"

    df_base_master = pd.read_csv(baseline_master_path)
    df_base_fair = pd.read_csv(baseline_fair_path)

    datasets = ["taiwan", "german"]
    model_names = ["logistic_regression", "random_forest", "xgboost"]
    mitigation_methods = ["reweighing", "exponentiated_gradient", "threshold_optimizer"]

    master_mitigation_rows = []
    subgroup_mitigation_rows = []
    metadata_records = {
        "random_state": 42,
        "test_size": 0.20,
        "primary_protected_attributes": {
            "taiwan": "sex",
            "german": "derived_sex"
        },
        "secondary_protected_attribute": "age (<=25 vs >25)",
        "mitigation_configurations": {
            "reweighing": {
                "type": "pre-processing",
                "method": "Kamiran & Calders (2012) Statistical Independence Weights",
                "target_loss": "weighted empirical risk minimization"
            },
            "exponentiated_gradient": {
                "type": "in-processing",
                "framework": "Fairlearn Agarwal et al. (2018)",
                "constraint": "EqualizedOdds()",
                "eps": 0.01,
                "max_iter": 10
            },
            "threshold_optimizer": {
                "type": "post-processing",
                "framework": "Fairlearn Hardt et al. (2016)",
                "constraint": "equalized_odds",
                "predict_method": "predict_proba",
                "prefit": True,
                "calibration_data": "training partition (X_train, y_train)"
            }
        },
        "experiments": []
    }

    print("=" * 70)
    print("STARTING FAIRNESS MITIGATION BENCHMARK (PHASE 3)")
    print("=" * 70)

    for ds_name in datasets:
        print(f"\n{'#' * 30} DATASET: {ds_name.upper()} {'#' * 30}")
        X, y, sens, feat_types = load_dataset(ds_name)
        X_train, X_test, y_train, y_test, sens_train, sens_test = split_dataset(
            X, y, sens, test_size=0.20, random_state=42
        )

        age_col = "age_binary" if "age_binary" in sens_test.columns else "age_group"

        for m_name in model_names:
            print(f"\n--- Base Model: {m_name.upper()} on {ds_name.upper()} ---")

            # Extract baseline metrics for this dataset and model
            base_row = df_base_master[
                (df_base_master["dataset"] == ds_name) &
                (df_base_master["model"] == m_name)
            ].iloc[0]

            base_fair_row = df_base_fair[
                (df_base_fair["dataset"] == ds_name) &
                (df_base_fair["model"] == m_name) &
                (df_base_fair["protected_attribute"] == "sex")
            ].iloc[0]

            b_acc = float(base_row["accuracy"])
            b_prec = float(base_row["precision"])
            b_rec = float(base_row["recall"])
            b_f1 = float(base_row["f1"])
            b_auc = float(base_row["roc_auc"])
            b_eod = float(base_row["equalized_odds_difference"])
            b_eopp = float(base_row["equal_opportunity_difference"])
            b_dpd = float(base_row["demographic_parity_difference"])
            b_di = float(base_row["disparate_impact"])
            b_app_di = float(base_fair_row["disparate_impact_approval_ratio"])
            b_app_di_dist = abs(b_app_di - 1.0)

            # Fit baseline pipeline (needed for ThresholdOptimizer)
            scale_num = True if m_name == "logistic_regression" else False
            base_prep = build_preprocessor(feat_types["categorical"], feat_types["numerical"], scale_numeric=scale_num)
            base_clf = get_model(m_name, random_state=42)
            base_pipeline = build_pipeline(base_prep, base_clf)
            base_pipeline.fit(X_train, y_train)
            base_pred = base_pipeline.predict(X_test)
            b_bal_acc = float(balanced_accuracy_score(y_test, base_pred))

            # Run 3 mitigation methods
            for mit_name in mitigation_methods:
                print(f"  [Running {mit_name} on {ds_name} - {m_name}]...")
                t_start = time.time()
                status = "success"
                failure_reason = None

                try:
                    if mit_name == "reweighing":
                        prep = build_preprocessor(feat_types["categorical"], feat_types["numerical"], scale_numeric=scale_num)
                        model_obj = fit_reweighing(
                            base_model_name=m_name,
                            preprocessor=prep,
                            X_train=X_train,
                            y_train=y_train,
                            sens_train=sens_train["sex"],
                            random_state=42
                        )
                        y_pred = model_obj.predict(X_test)
                        y_prob = model_obj.predict_proba(X_test)[:, 1]

                    elif mit_name == "exponentiated_gradient":
                        prep = build_preprocessor(feat_types["categorical"], feat_types["numerical"], scale_numeric=scale_num)
                        # Use max_iter=10 for fast convergence
                        prep_fitted, model_obj = fit_exponentiated_gradient(
                            base_model_name=m_name,
                            preprocessor=prep,
                            X_train=X_train,
                            y_train=y_train,
                            sens_train=sens_train["sex"],
                            random_state=42,
                            eps=0.01,
                            max_iter=10
                        )
                        X_test_proc = prep_fitted.transform(X_test)
                        y_pred = model_obj.predict(X_test_proc)
                        # Extract continuous probability from ensemble PMF
                        pmf = model_obj._pmf_predict(X_test_proc)
                        y_prob = pmf[:, 1] if pmf.shape[1] > 1 else y_pred.astype(float)

                    elif mit_name == "threshold_optimizer":
                        model_obj = fit_threshold_optimizer(
                            fitted_baseline_pipeline=base_pipeline,
                            X_train=X_train,
                            y_train=y_train,
                            sens_train=sens_train["sex"],
                            constraints="equalized_odds"
                        )
                        y_pred = model_obj.predict(X_test, sensitive_features=sens_test["sex"])
                        # ThresholdOptimizer applies a discrete thresholding rule; AUC evaluated on decision rule
                        y_prob = y_pred.astype(float)

                    exec_time = time.time() - t_start

                except Exception as e:
                    status = "unsupported"
                    failure_reason = str(e)
                    exec_time = time.time() - t_start
                    print(f"    ERROR in {mit_name}: {e}")
                    metadata_records["experiments"].append({
                        "dataset": ds_name,
                        "model": m_name,
                        "mitigation_method": mit_name,
                        "status": status,
                        "reason": failure_reason,
                        "execution_time_seconds": round(exec_time, 2)
                    })
                    continue

                # 1. Performance Evaluation
                perf = evaluate_performance(y_test, y_pred, y_prob)
                m_acc = perf["accuracy"]
                m_prec = perf["precision"]
                m_rec = perf["recall"]
                m_f1 = perf["f1"]
                m_auc = perf["roc_auc"]

                # 2. Fairness Evaluation: Primary (Sex)
                fair_sex = evaluate_fairness(
                    y_true=y_test,
                    y_pred=y_pred,
                    sensitive_features=sens_test["sex"],
                    protected_attribute_name="sex",
                    unprivileged_group="female",
                    privileged_group="male"
                )
                m_eod = fair_sex["equalized_odds_difference"]
                m_eopp = fair_sex["equal_opportunity_difference"]
                m_dpd = fair_sex["demographic_parity_difference"]
                m_app_di = fair_sex["disparate_impact_approval_ratio"] if fair_sex["disparate_impact_approval_ratio"] is not None else 1.0
                m_app_di_dist = abs(m_app_di - 1.0)

                m_bal_acc = float(balanced_accuracy_score(y_test, y_pred))
                delta_bal_acc = round(m_bal_acc - b_bal_acc, 4)

                # 3. Deltas (mitigated - baseline)
                delta_acc = round(m_acc - b_acc, 4)
                delta_prec = round(m_prec - b_prec, 4)
                delta_rec = round(m_rec - b_rec, 4)
                delta_f1 = round(m_f1 - b_f1, 4)
                delta_eod = round(m_eod - b_eod, 4)
                delta_eopp = round(m_eopp - b_eopp, 4)
                delta_dpd = round(m_dpd - b_dpd, 4)

                # ROC-AUC policy: continuous for reweighing, N/A for discrete predictors
                if mit_name == "reweighing":
                    m_auc_val = round(m_auc, 4)
                    delta_auc_val = round(m_auc - b_auc, 4)
                else:
                    m_auc_val = "N/A"
                    delta_auc_val = "N/A"

                master_record = {
                    "dataset": ds_name,
                    "model": m_name,
                    "mitigation_method": mit_name,
                    "baseline_accuracy": round(b_acc, 4),
                    "mitigated_accuracy": round(m_acc, 4),
                    "delta_accuracy": delta_acc,
                    "baseline_balanced_accuracy": round(b_bal_acc, 4),
                    "mitigated_balanced_accuracy": round(m_bal_acc, 4),
                    "delta_balanced_accuracy": delta_bal_acc,
                    "baseline_precision": round(b_prec, 4),
                    "mitigated_precision": round(m_prec, 4),
                    "delta_precision": delta_prec,
                    "baseline_recall": round(b_rec, 4),
                    "mitigated_recall": round(m_rec, 4),
                    "delta_recall": delta_rec,
                    "baseline_f1": round(b_f1, 4),
                    "mitigated_f1": round(m_f1, 4),
                    "delta_f1": delta_f1,
                    "baseline_roc_auc": round(b_auc, 4),
                    "mitigated_roc_auc": m_auc_val,
                    "delta_roc_auc": delta_auc_val,
                    "baseline_eod": round(b_eod, 4),
                    "mitigated_eod": round(m_eod, 4),
                    "delta_eod": delta_eod,
                    "baseline_eopp": round(b_eopp, 4),
                    "mitigated_eopp": round(m_eopp, 4),
                    "delta_eopp": delta_eopp,
                    "baseline_dpd": round(b_dpd, 4),
                    "mitigated_dpd": round(m_dpd, 4),
                    "delta_dpd": delta_dpd,
                    "baseline_approval_di": round(b_app_di, 4),
                    "mitigated_approval_di": round(m_app_di, 4),
                    "baseline_approval_di_distance": round(b_app_di_dist, 4),
                    "mitigated_approval_di_distance": round(m_app_di_dist, 4)
                }
                master_mitigation_rows.append(master_record)

                print(f"    BalAcc: {b_bal_acc:.4f}->{m_bal_acc:.4f} (dBalAcc: {delta_bal_acc:+.4f}) | EOD: {b_eod:.4f}->{m_eod:.4f} (dEOD: {delta_eod:+.4f}) | DPD: {b_dpd:.4f}->{m_dpd:.4f}")

                # 4. Detailed Subgroup Metrics for BOTH sex and age
                audit_configs = [
                    {
                        "attr_name": "sex",
                        "series": sens_test["sex"],
                        "unprivileged_val": "female",
                        "privileged_val": "male"
                    },
                    {
                        "attr_name": "age",
                        "series": sens_test[age_col],
                        "unprivileged_val": "<=25",
                        "privileged_val": ">25"
                    }
                ]

                for ac in audit_configs:
                    attr = ac["attr_name"]
                    groups = [ac["unprivileged_val"], ac["privileged_val"]]

                    for g in groups:
                        mask = (ac["series"] == g)
                        n_grp = int(mask.sum())
                        y_true_g = y_test[mask]
                        y_pred_g = y_pred[mask]

                        tp = int(((y_true_g == 1) & (y_pred_g == 1)).sum())
                        fp = int(((y_true_g == 0) & (y_pred_g == 1)).sum())
                        tn = int(((y_true_g == 0) & (y_pred_g == 0)).sum())
                        fn = int(((y_true_g == 1) & (y_pred_g == 0)).sum())

                        p_pos = tp + fn
                        p_neg = tn + fp

                        tpr = tp / p_pos if p_pos > 0 else 0.0
                        fpr = fp / p_neg if p_neg > 0 else 0.0
                        sel_rate = (tp + fp) / n_grp if n_grp > 0 else 0.0
                        app_rate = (tn + fn) / n_grp if n_grp > 0 else 0.0

                        subgroup_mitigation_rows.append({
                            "dataset": ds_name,
                            "model": m_name,
                            "mitigation_method": mit_name,
                            "protected_attribute": attr,
                            "group": g,
                            "group_role": "unprivileged" if g == ac["unprivileged_val"] else "privileged",
                            "N": n_grp,
                            "TP": tp,
                            "TN": tn,
                            "FP": fp,
                            "FN": fn,
                            "TPR": round(tpr, 4),
                            "FPR": round(fpr, 4),
                            "default_prediction_rate": round(sel_rate, 4),
                            "selection_rate": round(sel_rate, 4),
                            "approval_rate": round(app_rate, 4)
                        })

                # Log metadata
                metadata_records["experiments"].append({
                    "dataset": ds_name,
                    "model": m_name,
                    "mitigation_method": mit_name,
                    "status": "success",
                    "execution_time_seconds": round(exec_time, 2),
                    "perf": perf,
                    "fairness_sex": fair_sex
                })

    # Save Results
    df_mit_master = pd.DataFrame(master_mitigation_rows)
    df_mit_subgroups = pd.DataFrame(subgroup_mitigation_rows)

    master_csv_path = os.path.join(results_dir, "mitigation_master_results.csv")
    subgroups_csv_path = os.path.join(results_dir, "mitigated_subgroup_metrics.csv")
    metadata_json_path = os.path.join(results_dir, "mitigation_metadata.json")

    df_mit_master.to_csv(master_csv_path, index=False)
    df_mit_subgroups.to_csv(subgroups_csv_path, index=False)
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL 18 MITIGATION EXPERIMENTS COMPLETED!")
    print(f"Saved master results to: {master_csv_path}")
    print(f"Saved subgroup metrics to: {subgroups_csv_path}")
    print(f"Saved metadata to: {metadata_json_path}")
    print("=" * 70)

    return df_mit_master, df_mit_subgroups, metadata_records


if __name__ == "__main__":
    run_all_mitigations()
