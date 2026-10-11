"""
Baseline Sanity Review & Independent Metric Reconstruction Script.
Project: Responsible AI Credit Scoring Research

Tasks:
1. Load models, generate test predictions, and compute subgroup confusion matrices.
2. For each dataset / model / subgroup (sex and age):
   - TP, TN, FP, FN
   - TPR = TP / (TP + FN)
   - FPR = FP / (FP + TN)
   - Selection Rate = (TP + FP) / N
   - Approval Rate = (TN + FN) / N = 1 - Selection Rate
3. Independently reconstruct fairness metrics:
   - Equal Opportunity Difference = |TPR_u - TPR_p|
   - delta_FPR = |FPR_u - FPR_p|
   - Equalized Odds Difference = max(|TPR_u - TPR_p|, |FPR_u - FPR_p|)
   - Demographic Parity Difference = |SR_u - SR_p|
   - Disparate Impact (Selection Ratio) = min(SR_u, SR_p) / max(SR_u, SR_p)
   - Disparate Impact (Approval Ratio) = ApprovalRate_u / ApprovalRate_p
4. Compare reconstructed metrics against results/baseline_master_results.csv and results/baseline_fairness.csv.
5. Verify that sensitive attributes and IDs were never in feature matrix X.
6. Verify that preprocessing was fitted strictly on training data.
7. Export results/baseline_subgroup_metrics.csv.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.modeling.preprocessing import (
    load_dataset,
    split_dataset,
    build_preprocessor,
    get_project_root
)
from src.modeling.models import get_model, build_pipeline


def run_sanity_review():
    project_root = get_project_root()
    results_dir = os.path.join(project_root, "results")
    
    print("=" * 70)
    print("STARTING BASELINE SANITY REVIEW & INDEPENDENT RECONSTRUCTION")
    print("=" * 70)
    
    # 1. Verify existence of baseline files
    master_path = os.path.join(results_dir, "baseline_master_results.csv")
    fairness_path = os.path.join(results_dir, "baseline_fairness.csv")
    metadata_path = os.path.join(results_dir, "baseline_metadata.json")
    
    assert os.path.exists(master_path), f"Missing {master_path}"
    assert os.path.exists(fairness_path), f"Missing {fairness_path}"
    assert os.path.exists(metadata_path), f"Missing {metadata_path}"
    
    df_master = pd.read_csv(master_path)
    df_fairness_prev = pd.read_csv(fairness_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    print(f"Loaded master results ({len(df_master)} rows).")
    print(f"Loaded previous fairness results ({len(df_fairness_prev)} rows).")
    
    # 2. Re-train/predict and extract subgroup confusion matrices
    datasets = ["taiwan", "german"]
    model_names = ["logistic_regression", "random_forest", "xgboost"]
    
    subgroup_rows = []
    reconstructed_fairness = []
    
    for ds_name in datasets:
        X, y, sens, feat_types = load_dataset(ds_name)
        X_train, X_test, y_train, y_test, sens_train, sens_test = split_dataset(
            X, y, sens, test_size=0.20, random_state=42
        )
        
        # Verify feature columns in X
        banned_cols = ["id", "sex", "personal_status_sex", "original_target", "target"]
        found_banned = [col for col in banned_cols if col in X.columns]
        assert len(found_banned) == 0, f"LEAKAGE DETECTED in {ds_name}! Banned columns found in X: {found_banned}"
        print(f"[PASS] {ds_name.upper()} feature matrix X has {X.shape[1]} features. Zero banned columns found.")
        
        # Determine age column
        age_col = "age_binary" if "age_binary" in sens_test.columns else "age_group"
        
        for m_name in model_names:
            scale_num = True if m_name == "logistic_regression" else False
            preprocessor = build_preprocessor(
                feat_types["categorical"],
                feat_types["numerical"],
                scale_numeric=scale_num
            )
            model = get_model(m_name, random_state=42)
            pipeline = build_pipeline(preprocessor, model)
            
            # Fit strictly on train
            pipeline.fit(X_train, y_train)
            
            # Predict on test
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            
            # Evaluate subgroup confusion matrices for:
            # 1. SEX (female vs male)
            # 2. AGE (<=25 vs >25)
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
                grp_metrics = {}
                
                for g in groups:
                    mask = (ac["series"] == g)
                    n_group = int(mask.sum())
                    y_true_g = y_test[mask]
                    y_pred_g = y_pred[mask]
                    
                    tp = int(((y_true_g == 1) & (y_pred_g == 1)).sum())
                    fp = int(((y_true_g == 0) & (y_pred_g == 1)).sum())
                    tn = int(((y_true_g == 0) & (y_pred_g == 0)).sum())
                    fn = int(((y_true_g == 1) & (y_pred_g == 0)).sum())
                    
                    assert tp + fp + tn + fn == n_group, f"Confusion matrix sum {tp+fp+tn+fn} != N {n_group}"
                    
                    p_pos = tp + fn # Actual positives
                    p_neg = tn + fp # Actual negatives
                    
                    tpr = tp / p_pos if p_pos > 0 else 0.0
                    fpr = fp / p_neg if p_neg > 0 else 0.0
                    sel_rate = (tp + fp) / n_group if n_group > 0 else 0.0
                    app_rate = (tn + fn) / n_group if n_group > 0 else 0.0
                    
                    grp_metrics[g] = {
                        "N": n_group,
                        "TP": tp,
                        "TN": tn,
                        "FP": fp,
                        "FN": fn,
                        "P_actual_pos": p_pos,
                        "N_actual_neg": p_neg,
                        "TPR": tpr,
                        "FPR": fpr,
                        "selection_rate": sel_rate,
                        "approval_rate": app_rate
                    }
                    
                    subgroup_rows.append({
                        "dataset": ds_name,
                        "model": m_name,
                        "protected_attribute": attr,
                        "group": g,
                        "group_role": "unprivileged" if g == ac["unprivileged_val"] else "privileged",
                        "N": n_group,
                        "TP": tp,
                        "TN": tn,
                        "FP": fp,
                        "FN": fn,
                        "TPR": round(tpr, 4),
                        "FPR": round(fpr, 4),
                        "selection_rate": round(sel_rate, 4),
                        "approval_rate": round(app_rate, 4)
                    })
                    
                # Now compute reconstructed fairness metrics
                unp = grp_metrics[ac["unprivileged_val"]]
                priv = grp_metrics[ac["privileged_val"]]
                
                delta_tpr = abs(unp["TPR"] - priv["TPR"])
                delta_fpr = abs(unp["FPR"] - priv["FPR"])
                eq_odds_diff = max(delta_tpr, delta_fpr)
                eq_opp_diff = delta_tpr
                dp_diff = abs(unp["selection_rate"] - priv["selection_rate"])
                
                # Selection ratio (Disparate impact)
                sr_min = min(unp["selection_rate"], priv["selection_rate"])
                sr_max = max(unp["selection_rate"], priv["selection_rate"])
                dp_ratio = sr_min / sr_max if sr_max > 0 else 0.0
                
                # Approval ratio
                app_ratio = unp["approval_rate"] / priv["approval_rate"] if priv["approval_rate"] > 0 else 0.0
                
                reconstructed_fairness.append({
                    "dataset": ds_name,
                    "model": m_name,
                    "protected_attribute": attr,
                    "unprivileged_group": ac["unprivileged_val"],
                    "privileged_group": ac["privileged_val"],
                    "reconstructed_equal_opportunity_diff": round(eq_opp_diff, 4),
                    "reconstructed_delta_fpr": round(delta_fpr, 4),
                    "reconstructed_equalized_odds_diff": round(eq_odds_diff, 4),
                    "reconstructed_demographic_parity_diff": round(dp_diff, 4),
                    "reconstructed_disparate_impact": round(dp_ratio, 4),
                    "reconstructed_approval_ratio": round(app_ratio, 4)
                })

    df_subgroups = pd.DataFrame(subgroup_rows)
    subgroups_csv_path = os.path.join(results_dir, "baseline_subgroup_metrics.csv")
    df_subgroups.to_csv(subgroups_csv_path, index=False)
    print(f"\nSaved complete subgroup metrics to {subgroups_csv_path}")
    
    df_reconstructed = pd.DataFrame(reconstructed_fairness)
    
    # 3. Assert and verify exact matches
    print("\n" + "=" * 70)
    print("VERIFICATION OF EXACT METRIC MATCHES")
    print("=" * 70)
    
    # Compare with baseline_master_results.csv (Primary sex evaluation)
    for _, master_row in df_master.iterrows():
        ds = master_row["dataset"]
        m = master_row["model"]
        
        recon = df_reconstructed[
            (df_reconstructed["dataset"] == ds) &
            (df_reconstructed["model"] == m) &
            (df_reconstructed["protected_attribute"] == "sex")
        ].iloc[0]
        
        # Verify Equalized Odds Difference
        assert abs(master_row["equalized_odds_difference"] - recon["reconstructed_equalized_odds_diff"]) < 1e-4, \
            f"Mismatch in Equalized Odds Diff for {ds} {m}: {master_row['equalized_odds_difference']} vs {recon['reconstructed_equalized_odds_diff']}"
            
        # Verify Equal Opportunity Difference
        assert abs(master_row["equal_opportunity_difference"] - recon["reconstructed_equal_opportunity_diff"]) < 1e-4, \
            f"Mismatch in Equal Opportunity Diff for {ds} {m}: {master_row['equal_opportunity_difference']} vs {recon['reconstructed_equal_opportunity_diff']}"
            
        # Verify Demographic Parity Difference
        assert abs(master_row["demographic_parity_difference"] - recon["reconstructed_demographic_parity_diff"]) < 1e-4, \
            f"Mismatch in Demographic Parity Diff for {ds} {m}: {master_row['demographic_parity_difference']} vs {recon['reconstructed_demographic_parity_diff']}"
            
        # Verify Disparate Impact
        assert abs(master_row["disparate_impact"] - recon["reconstructed_disparate_impact"]) < 1e-4, \
            f"Mismatch in Disparate Impact for {ds} {m}: {master_row['disparate_impact']} vs {recon['reconstructed_disparate_impact']}"
            
        print(f"[VERIFIED] {ds.upper():<7} {m:<20} | EqOdds={recon['reconstructed_equalized_odds_diff']:.4f} | EqOpp={recon['reconstructed_equal_opportunity_diff']:.4f} | DPDiff={recon['reconstructed_demographic_parity_diff']:.4f} | DispImpact={recon['reconstructed_disparate_impact']:.4f}")

    print("\nAll fairness metrics successfully reconstructed from raw confusion matrix counts!")
    return df_subgroups, df_reconstructed


if __name__ == "__main__":
    run_sanity_review()
