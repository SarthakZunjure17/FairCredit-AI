"""
Master Baseline Modeling & Evaluation Orchestrator.
Project: Responsible AI Credit Scoring Research

Executes:
1. Ingests Taiwan and German canonical clean datasets.
2. Performs stratified train/test split (test_size=0.20, random_state=42).
3. Trains baseline models:
   - Logistic Regression
   - Random Forest
   - XGBoost
4. Computes performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
5. Computes fairness metrics on primary protected attribute (sex / derived sex) and secondary (age).
6. Exports results to results/baseline_*.csv and results/baseline_metadata.json.
"""

import os
import sys
import json
from typing import Dict, List, Any
import pandas as pd

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.modeling.preprocessing import (
    load_dataset,
    split_dataset,
    build_preprocessor,
    get_project_root
)
from src.modeling.models import get_model, build_pipeline
from src.modeling.evaluation import evaluate_performance, evaluate_fairness





def run_all_baselines(
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    project_root = get_project_root()
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    datasets = ["taiwan", "german"]
    model_names = ["logistic_regression", "random_forest", "xgboost"]

    performance_records = []
    fairness_records = []
    master_records = []
    metadata_records = {}

    print("=" * 70)
    print("STARTING RESPONSIBLE AI BASELINE MODELING BENCHMARK")
    print(f"Configurations: test_size={test_size}, random_state={random_state}")
    print("=" * 70)

    for ds_name in datasets:
        print(f"\n{'#' * 30} DATASET: {ds_name.upper()} {'#' * 30}")
        X, y, sens, feat_types = load_dataset(ds_name)
        X_train, X_test, y_train, y_test, sens_train, sens_test = split_dataset(
            X, y, sens, test_size=test_size, random_state=random_state
        )

        n_features = X.shape[1]
        n_train = len(X_train)
        n_test = len(X_test)

        print(f"Features: {n_features} (demographic sex excluded from X)")
        print(f"Split sizes: Train={n_train}, Test={n_test}")
        print(f"Target distribution (train): {y_train.value_counts().to_dict()}")
        print(f"Target distribution (test):  {y_test.value_counts().to_dict()}")

        metadata_records[ds_name] = {
            "random_state": random_state,
            "test_size": test_size,
            "feature_count": n_features,
            "training_rows": n_train,
            "test_rows": n_test,
            "features_in_X": list(X.columns),
            "categorical_features": feat_types["categorical"],
            "numerical_features": feat_types["numerical"]
        }

        for m_name in model_names:
            print(f"\n--- Training {m_name.upper()} on {ds_name.upper()} ---")

            # Determine whether to scale numerical features:
            # Scale for Logistic Regression, passthrough for trees
            scale_num = True if m_name == "logistic_regression" else False
            preprocessor = build_preprocessor(
                feat_types["categorical"],
                feat_types["numerical"],
                scale_numeric=scale_num
            )

            classifier = get_model(m_name, random_state=random_state)
            pipeline = build_pipeline(preprocessor, classifier)

            # Fit strictly on training partition
            pipeline.fit(X_train, y_train)

            # Predict on test partition
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            # 1. Evaluate Performance
            perf = evaluate_performance(y_test, y_pred, y_prob)
            perf_record = {
                "dataset": ds_name,
                "model": m_name,
                "training_rows": n_train,
                "test_rows": n_test,
                "feature_count": n_features,
                **perf
            }
            performance_records.append(perf_record)
            print(f"Performance: AUC={perf['roc_auc']:.4f}, F1={perf['f1']:.4f}, Acc={perf['accuracy']:.4f}, Prec={perf['precision']:.4f}, Rec={perf['recall']:.4f}")

            # 2. Evaluate Primary Fairness: Sex
            # Unprivileged: female, Privileged: male
            fairness_sex = evaluate_fairness(
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=sens_test["sex"],
                protected_attribute_name="sex",
                unprivileged_group="female",
                privileged_group="male"
            )
            fairness_record_sex = {
                "dataset": ds_name,
                "model": m_name,
                **fairness_sex
            }
            fairness_records.append(fairness_record_sex)
            print(f"Fairness (Sex): EqOddsDiff={fairness_sex['equalized_odds_difference']:.4f}, EqOppDiff={fairness_sex['equal_opportunity_difference']:.4f}, DPDiff={fairness_sex['demographic_parity_difference']:.4f}, DispImpact={fairness_sex['disparate_impact']:.4f}")

            # 3. Evaluate Secondary Fairness: Age (<=25 vs >25)
            # Unprivileged: <=25, Privileged: >25
            age_col = "age_binary" if "age_binary" in sens_test.columns else "age_group"
            fairness_age = evaluate_fairness(
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=sens_test[age_col],
                protected_attribute_name="age",
                unprivileged_group="<=25",
                privileged_group=">25"
            )
            fairness_record_age = {
                "dataset": ds_name,
                "model": m_name,
                **fairness_age
            }
            fairness_records.append(fairness_record_age)
            print(f"Fairness (Age <=25 vs >25): EqOddsDiff={fairness_age['equalized_odds_difference']:.4f}, EqOppDiff={fairness_age['equal_opportunity_difference']:.4f}, DPDiff={fairness_age['demographic_parity_difference']:.4f}, DispImpact={fairness_age['disparate_impact']:.4f}")

            # 4. Master Table Record (Primary Sex Evaluation)
            master_record = {
                "dataset": ds_name,
                "model": m_name,
                "accuracy": perf["accuracy"],
                "precision": perf["precision"],
                "recall": perf["recall"],
                "f1": perf["f1"],
                "roc_auc": perf["roc_auc"],
                "protected_attribute": "sex",
                "equalized_odds_difference": fairness_sex["equalized_odds_difference"],
                "equal_opportunity_difference": fairness_sex["equal_opportunity_difference"],
                "demographic_parity_difference": fairness_sex["demographic_parity_difference"],
                "disparate_impact": fairness_sex["disparate_impact"],
                "random_state": random_state,
                "test_size": test_size,
                "feature_count": n_features,
                "training_rows": n_train,
                "test_rows": n_test
            }
            master_records.append(master_record)

    # Convert to DataFrames and export
    df_perf = pd.DataFrame(performance_records)
    df_fair = pd.DataFrame(fairness_records)
    df_master = pd.DataFrame(master_records)

    perf_csv_path = os.path.join(results_dir, "baseline_performance.csv")
    fair_csv_path = os.path.join(results_dir, "baseline_fairness.csv")
    master_csv_path = os.path.join(results_dir, "baseline_master_results.csv")
    metadata_json_path = os.path.join(results_dir, "baseline_metadata.json")

    df_perf.to_csv(perf_csv_path, index=False)
    df_fair.to_csv(fair_csv_path, index=False)
    df_master.to_csv(master_csv_path, index=False)
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Results exported to:\n  - {perf_csv_path}\n  - {fair_csv_path}\n  - {master_csv_path}\n  - {metadata_json_path}")
    print("=" * 70)

    return {
        "performance": df_perf,
        "fairness": df_fair,
        "master": df_master,
        "metadata": metadata_records
    }


if __name__ == "__main__":
    run_all_baselines()
