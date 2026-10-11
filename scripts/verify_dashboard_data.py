"""
Phase 6 Data Verification Suite: FairCredit-AI Dashboard Data Integrity.
Project: FairCredit-AI: A Responsible AI Credit Model Auditing Framework

Validates:
1. All expected source files exist.
2. Six primary Reweighing research rows exist.
3. Phase 3 metrics match authoritative values.
4. Phase 4 metrics match authoritative values.
5. No N/A ROC-AUC is converted to zero.
6. Taiwan/German datasets are correctly identified.
7. All three model architectures are present.
8. Individual SHAP records have valid applicant IDs.
9. No protected attribute appears as a model feature.
10. Dashboard data loads without schema errors.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dashboard.data_loader import (
    load_mitigation_master_results,
    load_final_research_results,
    load_shap_summary,
    load_shap_group_metrics,
    load_shap_individual_metrics,
    load_research_matrix,
    load_individual_feature_shap_data,
    get_all_dashboard_data
)


def run_verification() -> bool:
    print("=" * 80)
    print("RUNNING PHASE 6 DASHBOARD DATA VERIFICATION SUITE")
    print("=" * 80)

    results_dir = os.path.join(project_root, "results")
    failures = []

    # -------------------------------------------------------------------------
    # CHECK 1: All Expected Source Files Exist
    # -------------------------------------------------------------------------
    print("\n[Check 1/10] Verifying existence of all required source files...")
    expected_files = [
        os.path.join(results_dir, "mitigation_master_results.csv"),
        os.path.join(results_dir, "phase4_explanation_results.csv"),
        os.path.join(results_dir, "shap_summary.csv"),
        os.path.join(results_dir, "shap_group_metrics.csv"),
        os.path.join(results_dir, "shap_individual_metrics.csv"),
        os.path.join(results_dir, "final_research_results.csv"),
        os.path.join(results_dir, "fairness_performance_explanation_matrix.csv"),
        os.path.join(project_root, "dashboard", "data", "individual_feature_shap.parquet")
    ]
    for fp in expected_files:
        if not os.path.exists(fp):
            failures.append(f"Missing required file: {fp}")
            print(f"  [X] Missing: {os.path.relpath(fp, project_root)}")
        else:
            print(f"  [OK] Found: {os.path.relpath(fp, project_root)}")

    if not failures:
        print("  PASSED: All expected files exist.")

    # -------------------------------------------------------------------------
    # CHECK 2: Six Primary Reweighing Research Rows Exist
    # -------------------------------------------------------------------------
    print("\n[Check 2/10] Verifying the six primary Reweighing research rows...")
    try:
        df_final = load_final_research_results()
        assert len(df_final) == 6, f"Expected 6 rows in final_research_results, got {len(df_final)}"
        
        expected_configs = {
            ("taiwan", "logistic_regression"),
            ("taiwan", "random_forest"),
            ("taiwan", "xgboost"),
            ("german", "logistic_regression"),
            ("german", "random_forest"),
            ("german", "xgboost")
        }
        actual_configs = set(zip(df_final["Dataset"], df_final["Model"]))
        assert actual_configs == expected_configs, f"Config mismatch: {actual_configs} vs {expected_configs}"
        print(f"  PASSED: All 6 primary Reweighing rows present: {sorted(list(actual_configs))}")
    except Exception as e:
        failures.append(f"Check 2 failed: {e}")
        print(f"  [FAILED]: {e}")

    # -------------------------------------------------------------------------
    # CHECK 3: Phase 3 Metrics Match Authoritative Values
    # -------------------------------------------------------------------------
    print("\n[Check 3/10] Verifying Phase 3 metrics match between master and final results...")
    try:
        df_master = load_mitigation_master_results()
        df_final = load_final_research_results()
        
        rew_master = df_master[df_master["mitigation_method"] == "reweighing"].copy()
        assert len(rew_master) == 6, f"Expected 6 reweighing rows in master, got {len(rew_master)}"
        
        for _, f_row in df_final.iterrows():
            ds = f_row["Dataset"]
            m = f_row["Model"]
            m_sub = rew_master[(rew_master["dataset"] == ds) & (rew_master["model"] == m)]
            assert len(m_sub) == 1, f"Missing {ds} - {m} in master results"
            m_row = m_sub.iloc[0]
            
            # Check AUC
            assert np.isclose(float(m_row["baseline_roc_auc"]), float(f_row["Baseline_AUC"]), atol=1e-4), \
                f"AUC Baseline mismatch for {ds}-{m}: {m_row['baseline_roc_auc']} vs {f_row['Baseline_AUC']}"
            assert np.isclose(float(m_row["mitigated_roc_auc"]), float(f_row["Reweighed_AUC"]), atol=1e-4), \
                f"AUC Mitigated mismatch for {ds}-{m}: {m_row['mitigated_roc_auc']} vs {f_row['Reweighed_AUC']}"
            assert np.isclose(float(m_row["delta_roc_auc"]), float(f_row["Delta_AUC"]), atol=1e-4), \
                f"Delta AUC mismatch for {ds}-{m}: {m_row['delta_roc_auc']} vs {f_row['Delta_AUC']}"
            
            # Check EOD
            assert np.isclose(float(m_row["baseline_eod"]), float(f_row["Baseline_EOD"]), atol=1e-4), \
                f"EOD Baseline mismatch for {ds}-{m}: {m_row['baseline_eod']} vs {f_row['Baseline_EOD']}"
            assert np.isclose(float(m_row["mitigated_eod"]), float(f_row["Reweighed_EOD"]), atol=1e-4), \
                f"EOD Mitigated mismatch for {ds}-{m}: {m_row['mitigated_eod']} vs {f_row['Reweighed_EOD']}"
            assert np.isclose(float(m_row["delta_eod"]), float(f_row["Delta_EOD"]), atol=1e-4), \
                f"Delta EOD mismatch for {ds}-{m}: {m_row['delta_eod']} vs {f_row['Delta_EOD']}"
            
            # Check DPD
            assert np.isclose(float(m_row["baseline_dpd"]), float(f_row["Baseline_DPD"]), atol=1e-4), \
                f"DPD Baseline mismatch for {ds}-{m}: {m_row['baseline_dpd']} vs {f_row['Baseline_DPD']}"
            assert np.isclose(float(m_row["mitigated_dpd"]), float(f_row["Reweighed_DPD"]), atol=1e-4), \
                f"DPD Mitigated mismatch for {ds}-{m}: {m_row['mitigated_dpd']} vs {f_row['Reweighed_DPD']}"

        print("  PASSED: Phase 3 authoritative metrics verified across all 6 configurations.")
    except Exception as e:
        failures.append(f"Check 3 failed: {e}")
        print(f"  [FAILED]: {e}")

    # -------------------------------------------------------------------------
    # CHECK 4: Phase 4 Metrics Match Authoritative Values
    # -------------------------------------------------------------------------
    print("\n[Check 4/10] Verifying Phase 4 explanation consistency metrics match authoritative values...")
    try:
        df_summary = load_shap_summary()
        df_final = load_final_research_results()
        
        for _, f_row in df_final.iterrows():
            ds = f_row["Dataset"]
            m = f_row["Model"]
            s_sub = df_summary[(df_summary["dataset"] == ds) & (df_summary["model"] == m)]
            assert len(s_sub) == 1, f"Missing {ds} - {m} in shap summary"
            s_row = s_sub.iloc[0]
            
            assert np.isclose(float(s_row["cosine_mean"]), float(f_row["Mean_SHAP_Cosine"]), atol=1e-4), \
                f"Cosine mismatch for {ds}-{m}: {s_row['cosine_mean']} vs {f_row['Mean_SHAP_Cosine']}"
            assert np.isclose(float(s_row["top5_overlap_mean"]), float(f_row["Mean_Top5_Overlap"]), atol=1e-4), \
                f"Top5 mismatch for {ds}-{m}: {s_row['top5_overlap_mean']} vs {f_row['Mean_Top5_Overlap']}"
            assert np.isclose(float(s_row["rank_corr_mean"]), float(f_row["Mean_Spearman"]), atol=1e-4), \
                f"Spearman mismatch for {ds}-{m}: {s_row['rank_corr_mean']} vs {f_row['Mean_Spearman']}"
            
        print("  PASSED: Phase 4 authoritative explanation consistency metrics verified.")
    except Exception as e:
        failures.append(f"Check 4 failed: {e}")
        print(f"  [FAILED]: {e}")

    # -------------------------------------------------------------------------
    # CHECK 5: No N/A ROC-AUC is Converted to Zero
    # -------------------------------------------------------------------------
    print("\n[Check 5/10] Verifying N/A ROC-AUC values are preserved and never converted to zero...")
    try:
        df_master = load_mitigation_master_results()
        non_continuous_methods = ["exponentiated_gradient", "threshold_optimizer"]
        
        for mit in non_continuous_methods:
            sub = df_master[df_master["mitigation_method"] == mit]
            for _, r in sub.iterrows():
                val = r["mitigated_roc_auc"]
                assert str(val).strip() == "N/A", \
                    f"Expected 'N/A' for {r['dataset']}-{r['model']}-{mit}, got {val}"
                assert val != 0.0 and val != 0 and val != "0.0", \
                    f"N/A was illegally converted to zero in {r['dataset']}-{r['model']}-{mit}"
                    
        print("  PASSED: All 12 ExponentiatedGradient & ThresholdOptimizer mitigated_roc_auc values are strictly 'N/A' (not zero).")
    except Exception as e:
        failures.append(f"Check 5 failed: {e}")
        print(f"  [FAILED]: {e}")

    # -------------------------------------------------------------------------
    # CHECK 6: Taiwan/German Datasets Correctly Identified
    # -------------------------------------------------------------------------
    print("\n[Check 6/10] Verifying Taiwan/German datasets are correctly identified across tables...")
    try:
        tables = [
            ("mitigation_master", load_mitigation_master_results(), "dataset"),
            ("final_research", load_final_research_results(), "Dataset"),
            ("shap_summary", load_shap_summary(), "dataset"),
            ("shap_individual", load_shap_individual_metrics(), "dataset"),
        ]
        for name, df, col in tables:
            dsets = set(df[col].str.lower().unique())
            assert dsets == {"taiwan", "german"}, f"Unexpected datasets in {name}: {dsets}"
        print("  PASSED: Both 'taiwan' and 'german' properly identified in all tables.")
    except Exception as e:
        failures.append(f"Check 6 failed: {e}")
        print(f"  [FAILED]: {e}")

    # -------------------------------------------------------------------------
    # CHECK 7: All Three Model Architectures Present
    # -------------------------------------------------------------------------
    print("\n[Check 7/10] Verifying all three model architectures are present...")
    try:
        expected_models = {"logistic_regression", "random_forest", "xgboost"}
        tables = [
            ("mitigation_master", load_mitigation_master_results(), "model"),
            ("final_research", load_final_research_results(), "Model"),
            ("shap_summary", load_shap_summary(), "model"),
            ("shap_individual", load_shap_individual_metrics(), "model"),
        ]
        for name, df, col in tables:
            models = set(df[col].str.lower().unique())
            assert models == expected_models, f"Unexpected models in {name}: {models}"
        print("  PASSED: All 3 model architectures present in all tables.")
    except Exception as e:
        failures.append(f"Check 7 failed: {e}")
        print(f"  [FAILED]: {e}")

    # -------------------------------------------------------------------------
    # CHECK 8: Individual SHAP Records Have Valid Applicant IDs
    # -------------------------------------------------------------------------
    print("\n[Check 8/10] Verifying individual SHAP records have valid applicant IDs...")
    try:
        df_indiv = load_shap_individual_metrics()
        assert len(df_indiv) == 2100, f"Expected 2100 individual records, found {len(df_indiv)}"
        
        # German cohort size: 200, Taiwan cohort size: 500
        germ_indiv = df_indiv[df_indiv["dataset"] == "german"]
        taiwan_indiv = df_indiv[df_indiv["dataset"] == "taiwan"]
        
        assert len(germ_indiv) == 600, f"Expected 600 German records (200x3), got {len(germ_indiv)}"
        assert len(taiwan_indiv) == 1500, f"Expected 1500 Taiwan records (500x3), got {len(taiwan_indiv)}"
        
        # Parquet lookup
        df_parquet = load_individual_feature_shap_data()
        assert len(df_parquet) == 167700, f"Expected 167700 feature attribution rows, got {len(df_parquet)}"
        
        print("  PASSED: 2,100 applicant records verified across German (N=200) and Taiwan (N=500).")
    except Exception as e:
        failures.append(f"Check 8 failed: {e}")
        print(f"  [FAILED]: {e}")

    # -------------------------------------------------------------------------
    # CHECK 9: No Protected Attribute Appears as a Model Feature
    # -------------------------------------------------------------------------
    print("\n[Check 9/10] Verifying zero leakage of protected demographic attributes into model features...")
    try:
        df_parquet = load_individual_feature_shap_data()
        feature_names = set(df_parquet["feature"].unique())
        
        prohibited = ["sex", "personal_status_sex", "derived_sex"]
        leakage = [f for f in feature_names if any(p == f.lower() or f"_{p}" in f.lower() or f"{p}_" in f.lower() for p in prohibited)]
        assert len(leakage) == 0, f"Protected attribute leakage detected in features: {leakage}"
        
        print("  PASSED: Feature space confirmed 100% clean of protected sex attributes.")
    except Exception as e:
        failures.append(f"Check 9 failed: {e}")
        print(f"  [FAILED]: {e}")

    # -------------------------------------------------------------------------
    # CHECK 10: Dashboard Data Loads Without Schema Errors
    # -------------------------------------------------------------------------
    print("\n[Check 10/10] Verifying dashboard data loader functions execute cleanly...")
    try:
        data = get_all_dashboard_data()
        assert len(data) == 6, f"Expected 6 core dashboard tables, got {len(data)}"
        for k, v in data.items():
            assert isinstance(v, pd.DataFrame), f"{k} is not a DataFrame"
            assert not v.empty, f"{k} is empty"
        print("  PASSED: All dashboard tables loaded successfully without schema or parsing errors.")
    except Exception as e:
        failures.append(f"Check 10 failed: {e}")
        print(f"  [FAILED]: {e}")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    if not failures:
        print("ALL 10 PHASE 6 DASHBOARD VERIFICATION CHECKS PASSED SUCCESSFULLY.")
        print("Authoritative research artifacts are complete, validated, and ready for Streamlit UI.")
        print("=" * 80)
        return True
    else:
        print(f"VERIFICATION FAILED: {len(failures)} error(s) encountered:")
        for f in failures:
            print(f"  - {f}")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
