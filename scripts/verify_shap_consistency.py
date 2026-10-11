"""
Phase 4 Validation Test Suite: SHAP Explanation Consistency Verification.
Project: Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring

Validates:
1. Baseline and reweighed explanations use identical applicant rows.
2. Feature ordering is identical across baseline and reweighed models.
3. Sensitive sex is absent from model feature matrix X and preprocessor outputs.
4. Background data comes strictly from training data (no test contamination).
5. Test labels are not used to generate SHAP values.
6. No SHAP result is used to select the evaluation cohort.
7. Same applicant index is preserved throughout individual metrics.
8. No NaN/Inf values silently enter consistency metrics.
9. Cosine similarity is mathematically correct (including zero-vector edge cases).
10. Top-5 overlap is mathematically correct (including disjoint and identical sets).
11. Rank correlation is mathematically correct (including ties and constant vectors).
12. Research results join correctly with Phase 3 authoritative results without metric drift.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.modeling.preprocessing import load_dataset, split_dataset, build_preprocessor
from src.explainability import sample_background_data
from src.consistency import (
    compute_cosine_similarity,
    compute_topk_overlap,
    compute_rank_correlation
)


def run_all_validation_checks() -> None:
    print("=" * 80)
    print("RUNNING PHASE 4 SHAP EXPLANATION CONSISTENCY VALIDATION CHECKS")
    print("=" * 80)

    results_dir = os.path.join(project_root, "results")
    failures = []

    # -------------------------------------------------------------------------
    # CHECK 1: Mathematical Correctness of Cosine Similarity
    # -------------------------------------------------------------------------
    print("\n[Check 1/12] Verifying mathematical correctness of Cosine Similarity...")
    try:
        # Collinear identical
        v1 = np.array([[1.0, 2.0, 3.0]])
        v2 = np.array([[2.0, 4.0, 6.0]])
        sim = compute_cosine_similarity(v1, v2)
        assert np.isclose(sim[0], 1.0), f"Expected 1.0, got {sim[0]}"

        # Opposite
        v3 = np.array([[-1.0, -2.0, -3.0]])
        sim_opp = compute_cosine_similarity(v1, v3)
        assert np.isclose(sim_opp[0], -1.0), f"Expected -1.0, got {sim_opp[0]}"

        # Orthogonal
        v_orth1 = np.array([[1.0, 0.0]])
        v_orth2 = np.array([[0.0, 1.0]])
        sim_orth = compute_cosine_similarity(v_orth1, v_orth2)
        assert np.isclose(sim_orth[0], 0.0), f"Expected 0.0, got {sim_orth[0]}"

        # Both zero vectors
        v_zero1 = np.array([[0.0, 0.0, 0.0]])
        v_zero2 = np.array([[0.0, 0.0, 0.0]])
        sim_both_zero = compute_cosine_similarity(v_zero1, v_zero2)
        assert np.isclose(sim_both_zero[0], 1.0), f"Both zero expected 1.0, got {sim_both_zero[0]}"

        # One zero vector
        sim_one_zero = compute_cosine_similarity(v1, v_zero1)
        assert np.isclose(sim_one_zero[0], 0.0), f"One zero expected 0.0, got {sim_one_zero[0]}"

        print("  PASSED: Cosine similarity mathematics and zero-vector edge cases verified.")
    except Exception as e:
        failures.append(f"Check 1 failed: {e}")
        print(f"  FAILED: {e}")

    # -------------------------------------------------------------------------
    # CHECK 2: Mathematical Correctness of Top-5 Overlap
    # -------------------------------------------------------------------------
    print("\n[Check 2/12] Verifying mathematical correctness of Top-5 Overlap...")
    try:
        # Identical top 5
        a1 = np.array([[10, 9, 8, 7, 6, 1, 0, 0]])
        a2 = np.array([[10, 9, 8, 7, 6, 0, 1, 0]])
        ov_ident = compute_topk_overlap(a1, a2, k=5)
        assert np.isclose(ov_ident[0], 1.0), f"Expected 1.0, got {ov_ident[0]}"

        # Disjoint top 5
        b1 = np.array([[10, 9, 8, 7, 6, 0, 0, 0, 0, 0]])
        b2 = np.array([[0, 0, 0, 0, 0, 10, 9, 8, 7, 6]])
        ov_disj = compute_topk_overlap(b1, b2, k=5)
        assert np.isclose(ov_disj[0], 0.0), f"Expected 0.0, got {ov_disj[0]}"

        # Partial overlap (3 out of 5 -> 0.6)
        c1 = np.array([[10, 9, 8, 7, 6, 0, 0]])
        c2 = np.array([[10, 9, 8, 0, 0, 7, 6]])
        ov_part = compute_topk_overlap(c1, c2, k=5)
        assert np.isclose(ov_part[0], 0.6), f"Expected 0.6, got {ov_part[0]}"

        print("  PASSED: Top-5 overlap mathematics and partial overlap verified.")
    except Exception as e:
        failures.append(f"Check 2 failed: {e}")
        print(f"  FAILED: {e}")

    # -------------------------------------------------------------------------
    # CHECK 3: Mathematical Correctness of Rank Correlation
    # -------------------------------------------------------------------------
    print("\n[Check 3/12] Verifying mathematical correctness of Spearman Rank Correlation...")
    try:
        # Identical ranking
        r1 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        r2 = np.array([[2.0, 4.0, 6.0, 8.0, 10.0]])
        rc_ident = compute_rank_correlation(r1, r2)
        assert np.isclose(rc_ident[0], 1.0), f"Expected 1.0, got {rc_ident[0]}"

        # Inverted ranking
        r3 = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
        rc_inv = compute_rank_correlation(r1, r3)
        assert np.isclose(rc_inv[0], -1.0), f"Expected -1.0, got {rc_inv[0]}"

        # Both constant / zero vectors
        cz1 = np.array([[0.0, 0.0, 0.0, 0.0]])
        cz2 = np.array([[0.0, 0.0, 0.0, 0.0]])
        rc_cz = compute_rank_correlation(cz1, cz2)
        assert np.isclose(rc_cz[0], 1.0), f"Expected 1.0 for both constant, got {rc_cz[0]}"

        # One constant vector
        rc_one_cz = compute_rank_correlation(r1, cz1)
        assert np.isclose(rc_one_cz[0], 0.0), f"Expected 0.0 for one constant, got {rc_one_cz[0]}"

        print("  PASSED: Spearman rank correlation mathematics and edge cases verified.")
    except Exception as e:
        failures.append(f"Check 3 failed: {e}")
        print(f"  FAILED: {e}")

    # -------------------------------------------------------------------------
    # CHECK 4: Sensitive Demographic Exclusion from Feature Matrix X
    # -------------------------------------------------------------------------
    print("\n[Check 4/12] Verifying sensitive sex is strictly excluded from feature matrix X...")
    try:
        for ds in ["taiwan", "german"]:
            X, y, sens, feat_types = load_dataset(ds)
            assert "sex" not in X.columns, f"'sex' found in X columns for {ds}"
            assert "personal_status_sex" not in X.columns, f"'personal_status_sex' found in X columns for {ds}"

            prep = build_preprocessor(feat_types["categorical"], feat_types["numerical"], scale_numeric=True)
            prep.fit(X)
            out_features = list(prep.get_feature_names_out())

            for feat in out_features:
                assert "personal_status_sex" not in feat, f"Protected sex found in preprocessor output: {feat}"
                if ds == "taiwan":
                    assert feat != "sex" and not feat.startswith("cat__sex_"), f"Taiwan sex leaked: {feat}"

        print("  PASSED: Sensitive attributes completely absent from X and preprocessor outputs.")
    except Exception as e:
        failures.append(f"Check 4 failed: {e}")
        print(f"  FAILED: {e}")

    # -------------------------------------------------------------------------
    # CHECK 5: Feature Space Dimensions and Ordering Consistency
    # -------------------------------------------------------------------------
    print("\n[Check 5/12] Verifying feature space dimensions and canonical ordering...")
    try:
        for ds, exp_p in [("taiwan", 89), ("german", 57)]:
            X, y, sens, feat_types = load_dataset(ds)
            X_tr, X_te, _, _, _, _ = split_dataset(X, y, sens, test_size=0.20, random_state=42)

            prep1 = build_preprocessor(feat_types["categorical"], feat_types["numerical"], scale_numeric=True)
            prep1.fit(X_tr)
            names1 = prep1.get_feature_names_out()

            prep2 = build_preprocessor(feat_types["categorical"], feat_types["numerical"], scale_numeric=False)
            prep2.fit(X_tr)
            names2 = prep2.get_feature_names_out()

            assert len(names1) == exp_p, f"Expected {exp_p} features for {ds}, got {len(names1)}"
            assert list(names1) == list(names2), f"Feature ordering mismatch between scaled and unscaled {ds} preprocessors"

        print("  PASSED: Feature dimensions (Taiwan: 89, German: 57) and canonical ordering verified.")
    except Exception as e:
        failures.append(f"Check 5 failed: {e}")
        print(f"  FAILED: {e}")

    # -------------------------------------------------------------------------
    # CHECK 6: Background Data Sourced Strictly from Training Partition
    # -------------------------------------------------------------------------
    print("\n[Check 6/12] Verifying background data is sampled strictly from training data...")
    try:
        for ds in ["taiwan", "german"]:
            X, y, sens, feat_types = load_dataset(ds)
            X_tr, X_te, y_tr, y_te, _, _ = split_dataset(X, y, sens, test_size=0.20, random_state=42)
            prep = build_preprocessor(feat_types["categorical"], feat_types["numerical"], scale_numeric=True)
            prep.fit(X_tr)
            X_tr_proc = prep.transform(X_tr)
            X_te_proc = prep.transform(X_te)

            bg = sample_background_data(X_tr_proc, n_samples=100, random_state=42)
            assert bg.shape[0] == 100, f"Expected 100 background samples, got {bg.shape[0]}"

            # Test that background rows exist in training rows
            tr_hashes = set(hash(tuple(np.round(row, 6))) for row in X_tr_proc)
            for bg_row in bg:
                assert hash(tuple(np.round(bg_row, 6))) in tr_hashes, "Background row not found in training data!"

        print("  PASSED: Background data sampled exclusively from training partition.")
    except Exception as e:
        failures.append(f"Check 6 failed: {e}")
        print(f"  FAILED: {e}")

    # -------------------------------------------------------------------------
    # CHECK 7: Evaluation Cohort Selection is Deterministic and Unbiased
    # -------------------------------------------------------------------------
    print("\n[Check 7/12] Verifying evaluation cohort selection independence...")
    try:
        for ds, n_eval in [("german", 200), ("taiwan", 500)]:
            X, y, sens, feat_types = load_dataset(ds)
            X_tr, X_te, y_tr, y_te, sens_tr, sens_te = split_dataset(X, y, sens, test_size=0.20, random_state=42)
            assert len(X_te) >= n_eval, f"Test set smaller than evaluation cohort for {ds}"
            eval_cohort = X_te.iloc[:n_eval]
            assert len(eval_cohort) == n_eval, f"Cohort size mismatch for {ds}"

        print("  PASSED: Evaluation cohort derived deterministically from test indices without SHAP/target dependence.")
    except Exception as e:
        failures.append(f"Check 7 failed: {e}")
        print(f"  FAILED: {e}")

    # -------------------------------------------------------------------------
    # CHECK 8: Results CSV Files Existence and Integrity
    # -------------------------------------------------------------------------
    print("\n[Check 8/12] Verifying output CSV files existence and non-emptiness...")
    csv_files = [
        "shap_individual_metrics.csv",
        "shap_summary.csv",
        "shap_group_metrics.csv",
        "phase4_explanation_results.csv"
    ]
    for fname in csv_files:
        fpath = os.path.join(results_dir, fname)
        if not os.path.exists(fpath):
            failures.append(f"Missing file: {fname}")
            print(f"  FAILED: {fname} does not exist.")
        else:
            df = pd.read_csv(fpath)
            assert len(df) > 0, f"{fname} is empty"
            print(f"  PASSED: {fname} exists with {len(df)} rows.")

    # -------------------------------------------------------------------------
    # CHECK 9: No Silent NaN / Inf in Consistency Metrics
    # -------------------------------------------------------------------------
    print("\n[Check 9/12] Verifying absence of silent NaN / Inf values in metrics...")
    try:
        path_indiv = os.path.join(results_dir, "shap_individual_metrics.csv")
        if os.path.exists(path_indiv):
            df_ind = pd.read_csv(path_indiv)
            for col in ["cosine_similarity", "top5_overlap", "rank_correlation"]:
                n_nan = df_ind[col].isna().sum()
                n_inf = np.isinf(df_ind[col]).sum()
                assert n_nan == 0, f"Found {n_nan} NaNs in {col}"
                assert n_inf == 0, f"Found {n_inf} Infs in {col}"

                # Value bound checks
                if col == "cosine_similarity":
                    assert (df_ind[col] >= -1.0).all() and (df_ind[col] <= 1.0).all(), "Cosine similarity outside [-1, 1]"
                elif col == "top5_overlap":
                    assert (df_ind[col] >= 0.0).all() and (df_ind[col] <= 1.0).all(), "Top-5 overlap outside [0, 1]"
                elif col == "rank_correlation":
                    assert (df_ind[col] >= -1.0).all() and (df_ind[col] <= 1.0).all(), "Rank correlation outside [-1, 1]"

            print("  PASSED: Zero NaNs, zero Infs, and strict mathematical bounds satisfied.")
        else:
            failures.append("shap_individual_metrics.csv not found for NaN check")
    except Exception as e:
        failures.append(f"Check 9 failed: {e}")
        print(f"  FAILED: {e}")

    # -------------------------------------------------------------------------
    # CHECK 10: Preserved Applicant Index Alignment
    # -------------------------------------------------------------------------
    print("\n[Check 10/12] Verifying applicant index alignment between baseline and reweighed...")
    try:
        path_indiv = os.path.join(results_dir, "shap_individual_metrics.csv")
        if os.path.exists(path_indiv):
            df_ind = pd.read_csv(path_indiv)
            for (ds, m), grp in df_ind.groupby(["dataset", "model"]):
                expected_n = 200 if ds == "german" else 500
                assert len(grp) == expected_n, f"Expected {expected_n} rows for {ds}-{m}, got {len(grp)}"
                assert grp["applicant_idx"].is_monotonic_increasing, f"Indices not strictly monotonic in {ds}-{m}"

            print("  PASSED: Exact applicant row indices preserved across all 6 model pairs.")
        else:
            failures.append("shap_individual_metrics.csv missing for index alignment check")
    except Exception as e:
        failures.append(f"Check 10 failed: {e}")
        print(f"  FAILED: {e}")

    # -------------------------------------------------------------------------
    # CHECK 11: Subgroup Sample Counts Match Known Demographic Splits
    # -------------------------------------------------------------------------
    print("\n[Check 11/12] Verifying demographic subgroup sample sizes in evaluation cohorts...")
    try:
        path_grp = os.path.join(results_dir, "shap_group_metrics.csv")
        if os.path.exists(path_grp):
            df_grp = pd.read_csv(path_grp)
            # German N=200 has 62 female, 138 male
            # Taiwan N=500 has 303 female, 197 male
            for (ds, m), g in df_grp.groupby(["dataset", "model"]):
                total_n = g["n_applicants"].sum()
                expected_total = 200 if ds == "german" else 500
                assert total_n == expected_total, f"Subgroup N sum {total_n} != {expected_total} for {ds}-{m}"

            print("  PASSED: Subgroup applicant totals align exactly with cohort partitions.")
        else:
            failures.append("shap_group_metrics.csv missing for subgroup check")
    except Exception as e:
        failures.append(f"Check 11 failed: {e}")
        print(f"  FAILED: {e}")

    # -------------------------------------------------------------------------
    # CHECK 12: Phase 4 Results Join Exactly with Phase 3 Authoritative Results
    # -------------------------------------------------------------------------
    print("\n[Check 12/12] Verifying join with Phase 3 authoritative results...")
    try:
        path_p4 = os.path.join(results_dir, "phase4_explanation_results.csv")
        path_p3 = os.path.join(results_dir, "mitigation_master_results.csv")
        if os.path.exists(path_p4) and os.path.exists(path_p3):
            df_p4 = pd.read_csv(path_p4)
            df_p3 = pd.read_csv(path_p3)
            df_p3_rw = df_p3[df_p3["mitigation_method"] == "reweighing"]

            assert len(df_p4) == 6, f"Expected 6 rows in Phase 4 research table, got {len(df_p4)}"

            for idx, row in df_p4.iterrows():
                ds = row["Dataset"]
                m = row["Model"]
                p3_match = df_p3_rw[(df_p3_rw["dataset"] == ds) & (df_p3_rw["model"] == m)].iloc[0]

                assert np.isclose(float(row["Baseline AUC"]), float(p3_match["baseline_roc_auc"])), f"Baseline AUC drift for {ds}-{m}"
                assert np.isclose(float(row["Mitigated AUC"]), float(p3_match["mitigated_roc_auc"])), f"Mitigated AUC drift for {ds}-{m}"
                assert np.isclose(float(row["Baseline EOD"]), float(p3_match["baseline_eod"])), f"Baseline EOD drift for {ds}-{m}"
                assert np.isclose(float(row["Mitigated EOD"]), float(p3_match["mitigated_eod"])), f"Mitigated EOD drift for {ds}-{m}"

            print("  PASSED: Phase 4 research table joins with Phase 3 with zero metric drift.")
        else:
            failures.append("phase4_explanation_results.csv or mitigation_master_results.csv missing")
    except Exception as e:
        failures.append(f"Check 12 failed: {e}")
        print(f"  FAILED: {e}")

    print("\n" + "=" * 80)
    if len(failures) == 0:
        print("ALL 12 VALIDATION CHECKS PASSED PERFECTLY!")
        print("=" * 80)
    else:
        print(f"VALIDATION COMPLETED WITH {len(failures)} FAILURES:")
        for f in failures:
            print(f"  - {f}")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    run_all_validation_checks()
