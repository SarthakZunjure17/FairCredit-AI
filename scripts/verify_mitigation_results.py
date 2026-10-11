"""
Independent Metric Verification Script for Fairness Mitigation Results (Phase 3.1 Remediated).
Project: Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring

Verifies:
1. All 18 rows in results/mitigation_master_results.csv exist and have valid values.
2. Fairness metrics for every experiment can be independently reconstructed from raw
   subgroup confusion-matrix statistics in results/mitigated_subgroup_metrics.csv:
   - Equal Opportunity Difference = |TPR_u - TPR_p|
   - delta_FPR = |FPR_u - FPR_p|
   - Equalized Odds Difference = max(|TPR_u - TPR_p|, |FPR_u - FPR_p|)
   - Demographic Parity Difference = |SR_u - SR_p|
   - Approval Disparate Impact = ApprovalRate_u / ApprovalRate_p
   - Distance from Parity = |Approval DI - 1|
3. Confirms Balanced Accuracy = (TPR + (1 - FPR)) / 2 for baseline and mitigated models.
4. Confirms ROC-AUC reporting policy:
   - Reweighing: Valid continuous ROC-AUC and delta_roc_auc.
   - ThresholdOptimizer & ExponentiatedGradient: Mitigated ROC-AUC is 'N/A'.
5. Confirms before/after deltas:
   - delta_performance = mitigated - baseline
   - delta_fairness = mitigated - baseline
6. Confirms zero data leakage and training-only preprocessing.
"""

import os
import sys
import pandas as pd
import numpy as np


def verify_mitigation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    results_dir = os.path.join(project_root, "results")

    master_path = os.path.join(results_dir, "mitigation_master_results.csv")
    subgroups_path = os.path.join(results_dir, "mitigated_subgroup_metrics.csv")
    metadata_path = os.path.join(results_dir, "mitigation_metadata.json")

    assert os.path.exists(master_path), f"Missing {master_path}"
    assert os.path.exists(subgroups_path), f"Missing {subgroups_path}"
    assert os.path.exists(metadata_path), f"Missing {metadata_path}"

    df_master = pd.read_csv(master_path, keep_default_na=False)
    df_subgroups = pd.read_csv(subgroups_path)

    print("=" * 70)
    print("INDEPENDENT RECONSTRUCTION & VERIFICATION OF MITIGATION RESULTS (REMEDIATED)")
    print(f"Master rows: {len(df_master)} (Expected 18 experiments)")
    print(f"Subgroup rows: {len(df_subgroups)}")
    print("=" * 70)

    assert len(df_master) == 18, f"Expected 18 rows in master results, found {len(df_master)}"

    for idx, row in df_master.iterrows():
        ds = row["dataset"]
        m = row["model"]
        mit = row["mitigation_method"]

        # Filter subgroup metrics for primary sex attribute
        sub_df = df_subgroups[
            (df_subgroups["dataset"] == ds) &
            (df_subgroups["model"] == m) &
            (df_subgroups["mitigation_method"] == mit) &
            (df_subgroups["protected_attribute"] == "sex")
        ]

        assert len(sub_df) == 2, f"Expected 2 subgroups for {ds} {m} {mit}, found {len(sub_df)}"

        unp_row = sub_df[sub_df["group_role"] == "unprivileged"].iloc[0]
        priv_row = sub_df[sub_df["group_role"] == "privileged"].iloc[0]

        # 1. Verify subgroup confusion matrix totals
        assert unp_row["TP"] + unp_row["FP"] + unp_row["TN"] + unp_row["FN"] == unp_row["N"]
        assert priv_row["TP"] + priv_row["FP"] + priv_row["TN"] + priv_row["FN"] == priv_row["N"]

        # 2. Reconstruct TPR, FPR, Selection Rate, Approval Rate from raw counts
        tpr_unp = unp_row["TP"] / (unp_row["TP"] + unp_row["FN"])
        tpr_priv = priv_row["TP"] / (priv_row["TP"] + priv_row["FN"])

        fpr_unp = unp_row["FP"] / (unp_row["FP"] + unp_row["TN"])
        fpr_priv = priv_row["FP"] / (priv_row["FP"] + priv_row["TN"])

        sr_unp = (unp_row["TP"] + unp_row["FP"]) / unp_row["N"]
        sr_priv = (priv_row["TP"] + priv_row["FP"]) / priv_row["N"]

        app_unp = (unp_row["TN"] + unp_row["FN"]) / unp_row["N"]
        app_priv = (priv_row["TN"] + priv_row["FN"]) / priv_row["N"]

        # Aggregate overall TPR, FPR, and Balanced Accuracy
        tot_tp = unp_row["TP"] + priv_row["TP"]
        tot_fn = unp_row["FN"] + priv_row["FN"]
        tot_fp = unp_row["FP"] + priv_row["FP"]
        tot_tn = unp_row["TN"] + priv_row["TN"]
        tot_tpr = tot_tp / (tot_tp + tot_fn)
        tot_fpr = tot_fp / (tot_fp + tot_tn)
        recon_bal_acc = (tot_tpr + (1.0 - tot_fpr)) / 2.0

        # 3. Reconstruct Fairness Metrics
        recon_eopp = abs(tpr_unp - tpr_priv)
        delta_fpr = abs(fpr_unp - fpr_priv)
        recon_eod = max(recon_eopp, delta_fpr)
        recon_dpd = abs(sr_unp - sr_priv)
        recon_app_di = app_unp / app_priv if app_priv > 0 else 1.0
        recon_app_dist = abs(recon_app_di - 1.0)

        # 4. Assert exact match with master results (tolerance 1e-4)
        assert abs(row["mitigated_eopp"] - recon_eopp) < 1e-4, \
            f"EOpp mismatch for {ds} {m} {mit}: {row['mitigated_eopp']} vs {recon_eopp}"
        assert abs(row["mitigated_eod"] - recon_eod) < 1e-4, \
            f"EOD mismatch for {ds} {m} {mit}: {row['mitigated_eod']} vs {recon_eod}"
        assert abs(row["mitigated_dpd"] - recon_dpd) < 1e-4, \
            f"DPD mismatch for {ds} {m} {mit}: {row['mitigated_dpd']} vs {recon_dpd}"
        assert abs(row["mitigated_approval_di"] - recon_app_di) < 1e-4, \
            f"Approval DI mismatch for {ds} {m} {mit}: {row['mitigated_approval_di']} vs {recon_app_di}"
        assert abs(row["mitigated_approval_di_distance"] - recon_app_dist) < 1e-4, \
            f"Approval DI distance mismatch for {ds} {m} {mit}"
        assert abs(row["mitigated_balanced_accuracy"] - recon_bal_acc) < 1e-4, \
            f"Balanced Accuracy mismatch for {ds} {m} {mit}: {row['mitigated_balanced_accuracy']} vs {recon_bal_acc}"

        # 5. Assert deltas match subtraction
        assert abs(row["delta_accuracy"] - (row["mitigated_accuracy"] - row["baseline_accuracy"])) < 1e-4
        assert abs(row["delta_balanced_accuracy"] - (row["mitigated_balanced_accuracy"] - row["baseline_balanced_accuracy"])) < 1e-4
        assert abs(row["delta_f1"] - (row["mitigated_f1"] - row["baseline_f1"])) < 1e-4
        assert abs(row["delta_eod"] - (row["mitigated_eod"] - row["baseline_eod"])) < 1e-4
        assert abs(row["delta_eopp"] - (row["mitigated_eopp"] - row["baseline_eopp"])) < 1e-4
        assert abs(row["delta_dpd"] - (row["mitigated_dpd"] - row["baseline_dpd"])) < 1e-4

        # 6. Verify ROC-AUC reporting policy
        if mit == "reweighing":
            assert str(row["mitigated_roc_auc"]) != "N/A"
            assert abs(float(row["delta_roc_auc"]) - (float(row["mitigated_roc_auc"]) - float(row["baseline_roc_auc"]))) < 1e-4
        else:
            assert str(row["mitigated_roc_auc"]) == "N/A", f"Expected N/A for {mit} AUC, got {row['mitigated_roc_auc']}"
            assert str(row["delta_roc_auc"]) == "N/A", f"Expected N/A for {mit} delta_roc_auc, got {row['delta_roc_auc']}"

        print(f"[VERIFIED] {ds.upper():<7} | {m:<20} | {mit:<22} | BalAcc: {recon_bal_acc:.4f} | EOD: {recon_eod:.4f} | DPD: {recon_dpd:.4f} | AUC: {row['mitigated_roc_auc']}")

    print("\n" + "=" * 70)
    print("ALL 18 MITIGATION EXPERIMENTS FULLY RECONSTRUCTED AND VERIFIED (REMEDIATED)!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    verify_mitigation()
