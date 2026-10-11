"""
Phase 4 Master Script: SHAP Explanation Consistency Analysis.
Project: Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring

Executes:
1. Loads canonical datasets (Taiwan and German).
2. Performs stratified train/test split (test_size=0.20, random_state=42).
3. Trains Baseline model and Mitigated model (Reweighing) for:
   - Logistic Regression
   - Random Forest
   - XGBoost
4. Draws training-derived background dataset (German: 100, Taiwan: 200).
5. Explains identical evaluation cohorts:
   - German: All N=200 test applicants.
   - Taiwan: Deterministic first N=500 test applicants (random_state=42 split).
6. Computes SHAP attributions for Y=1 (Default risk) using:
   - LinearExplainer for Logistic Regression (margin/log-odds space)
   - TreeExplainer for Random Forest (probability space)
   - TreeExplainer for XGBoost (margin/log-odds space)
7. Computes consistency metrics:
   - Cosine Similarity (with zero-vector protection)
   - Top-5 Overlap (K=5)
   - Spearman Rank Correlation (with tie/constant protection)
8. Evaluates subgroup-stratified consistency by protected attribute (sex).
9. Evaluates secondary comparable-applicant consistency via Nearest Neighbors on non-sensitive features.
10. Generates output CSVs:
    - results/shap_individual_metrics.csv
    - results/shap_summary.csv
    - results/shap_group_metrics.csv
    - results/phase4_explanation_results.csv
"""

import os
import sys
import time
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.modeling.preprocessing import (
    load_dataset,
    split_dataset,
    build_preprocessor,
    get_project_root
)
from src.modeling.models import get_model, build_pipeline
from src.modeling.mitigation import fit_reweighing
from src.explainability import (
    get_model_explainer,
    compute_shap_attributions,
    sample_background_data
)
from src.consistency import (
    compute_cosine_similarity,
    compute_topk_overlap,
    compute_rank_correlation,
    compute_bootstrap_ci,
    compute_consistency_metrics_dataframe,
    summarize_metric_distribution
)


def compute_comparable_applicant_metrics(
    phi_baseline: np.ndarray,
    phi_reweighed: np.ndarray,
    X_eval_proc: np.ndarray,
    k: int = 5
) -> Dict[str, float]:
    """
    Secondary analysis: Compare baseline explanation of applicant i with
    reweighed explanation of applicant's nearest credit neighbor j != i.
    Uses non-sensitive credit features only (Euclidean distance on standardized X).
    """
    n_samples = X_eval_proc.shape[0]
    if n_samples < 2:
        return {"mean_cosine": np.nan, "mean_top5": np.nan, "mean_rank": np.nan}

    # Find nearest neighbor j != i in non-sensitive feature space
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(X_eval_proc)
    _, indices = nn.kneighbors(X_eval_proc)
    neighbor_idx = indices[:, 1]  # index 1 is closest distinct neighbor

    phi_reweighed_neighbors = phi_reweighed[neighbor_idx]

    cos_sim = compute_cosine_similarity(phi_baseline, phi_reweighed_neighbors)
    top5_ov = compute_topk_overlap(phi_baseline, phi_reweighed_neighbors, k=k)
    rank_corr = compute_rank_correlation(phi_baseline, phi_reweighed_neighbors)

    return {
        "mean_cosine": round(float(np.mean(cos_sim)), 4),
        "mean_top5": round(float(np.mean(top5_ov)), 4),
        "mean_rank": round(float(np.mean(rank_corr)), 4)
    }


def run_phase4_shap_analysis() -> None:
    print("=" * 80)
    print("STARTING PHASE 4: SHAP EXPLANATION CONSISTENCY ANALYSIS")
    print("=" * 80)

    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    datasets = ["german", "taiwan"]  # Run fast German first, then Taiwan
    models = ["logistic_regression", "xgboost", "random_forest"]  # Fast LR & XGB first, then RF

    # Evaluation cohort policies
    cohort_sizes = {
        "german": 200,   # 100% of German test set
        "taiwan": 500    # Deterministic first 500 test set applicants
    }

    background_sizes = {
        "german": 100,
        "taiwan": 200
    }

    all_individual_records = []
    all_summary_records = []
    all_group_records = []

    for ds_name in datasets:
        print(f"\n{'#' * 30} DATASET: {ds_name.upper()} {'#' * 30}")
        X, y, sens, feat_types = load_dataset(ds_name)
        X_train, X_test, y_train, y_test, sens_train, sens_test = split_dataset(
            X, y, sens, test_size=0.20, random_state=42
        )

        n_eval = cohort_sizes[ds_name]
        n_bg = background_sizes[ds_name]

        # Deterministic evaluation cohort selection
        eval_indices = X_test.index[:n_eval]
        X_eval = X_test.iloc[:n_eval].copy()
        sens_eval = sens_test.iloc[:n_eval].copy()
        y_eval = y_test.iloc[:n_eval].copy()

        print(f"Dataset: {ds_name} | Train size: {len(X_train)} | Test size: {len(X_test)}")
        print(f"Evaluation Cohort size: {len(X_eval)} (deterministic first {n_eval} test applicants)")
        print(f"Background sample size: {n_bg} (derived strictly from training set, seed=42)")

        for m_name in models:
            print(f"\n>>> Running SHAP Analysis for {ds_name.upper()} - {m_name.upper()} <<<")
            t0 = time.time()

            scale_num = True if m_name == "logistic_regression" else False

            # 1. Fit Preprocessor strictly on training data
            preprocessor = build_preprocessor(
                feat_types["categorical"],
                feat_types["numerical"],
                scale_numeric=scale_num
            )
            preprocessor.fit(X_train)
            feature_names = preprocessor.get_feature_names_out()
            n_features = len(feature_names)

            X_train_proc = preprocessor.transform(X_train)
            X_eval_proc = preprocessor.transform(X_eval)

            # Sample background strictly from training set
            background_data = sample_background_data(
                X_train_proc,
                n_samples=n_bg,
                random_state=42
            )

            # 2. Fit Baseline Model
            print(f"  [1/4] Fitting Baseline {m_name}...")
            baseline_clf = get_model(m_name, random_state=42)
            baseline_clf.fit(X_train_proc, y_train)

            # 3. Fit Reweighed Model
            print(f"  [2/4] Fitting Reweighed {m_name}...")
            reweighed_pipeline = fit_reweighing(
                base_model_name=m_name,
                preprocessor=preprocessor,
                X_train=X_train,
                y_train=y_train,
                sens_train=sens_train["sex"],
                random_state=42
            )
            reweighed_clf = reweighed_pipeline.named_steps["classifier"]

            # 4. Generate SHAP Explanations
            print(f"  [3/4] Computing SHAP attributions for Baseline & Reweighed models...")
            exp_base = get_model_explainer(baseline_clf, m_name, background_data=background_data)
            exp_reweighed = get_model_explainer(reweighed_clf, m_name, background_data=background_data)

            phi_baseline = compute_shap_attributions(exp_base, m_name, X_eval_proc)
            phi_reweighed = compute_shap_attributions(exp_reweighed, m_name, X_eval_proc)

            print(f"        Attributions computed: phi_b shape={phi_baseline.shape}, phi_m shape={phi_reweighed.shape}")

            # 5. Compute Consistency Metrics for each individual applicant
            print(f"  [4/4] Evaluating Consistency Metrics (Cosine, Top-5, Spearman)...")
            cos_sim = compute_cosine_similarity(phi_baseline, phi_reweighed)
            top5_ov = compute_topk_overlap(phi_baseline, phi_reweighed, k=5)
            rank_corr = compute_rank_correlation(phi_baseline, phi_reweighed)

            # Individual records
            for i in range(len(X_eval)):
                app_idx = int(eval_indices[i])
                sex_val = str(sens_eval["sex"].iloc[i])
                all_individual_records.append({
                    "dataset": ds_name,
                    "model": m_name,
                    "applicant_idx": app_idx,
                    "sex": sex_val,
                    "cosine_similarity": round(float(cos_sim[i]), 4),
                    "top5_overlap": round(float(top5_ov[i]), 4),
                    "rank_correlation": round(float(rank_corr[i]), 4)
                })

            # Summary distribution
            cos_dist = summarize_metric_distribution(pd.Series(cos_sim), random_state=42)
            top5_dist = summarize_metric_distribution(pd.Series(top5_ov), random_state=42)
            rank_dist = summarize_metric_distribution(pd.Series(rank_corr), random_state=42)

            # Secondary comparable-applicant analysis
            comp_metrics = compute_comparable_applicant_metrics(
                phi_baseline, phi_reweighed, X_eval_proc, k=5
            )

            all_summary_records.append({
                "dataset": ds_name,
                "model": m_name,
                "n_applicants": len(X_eval),
                "n_features": n_features,
                "cosine_mean": cos_dist["mean"],
                "cosine_ci_95_low": cos_dist["mean_ci_95"][0],
                "cosine_ci_95_high": cos_dist["mean_ci_95"][1],
                "cosine_median": cos_dist["median"],
                "cosine_std": cos_dist["std"],
                "top5_overlap_mean": top5_dist["mean"],
                "top5_overlap_ci_95_low": top5_dist["mean_ci_95"][0],
                "top5_overlap_ci_95_high": top5_dist["mean_ci_95"][1],
                "top5_overlap_median": top5_dist["median"],
                "top5_overlap_std": top5_dist["std"],
                "rank_corr_mean": rank_dist["mean"],
                "rank_corr_ci_95_low": rank_dist["mean_ci_95"][0],
                "rank_corr_ci_95_high": rank_dist["mean_ci_95"][1],
                "rank_corr_median": rank_dist["median"],
                "rank_corr_std": rank_dist["std"],
                "comparable_nn_mean_cosine": comp_metrics["mean_cosine"],
                "comparable_nn_mean_top5": comp_metrics["mean_top5"],
                "comparable_nn_mean_rank": comp_metrics["mean_rank"]
            })

            # Subgroup stratified metrics (sex)
            df_curr_indiv = pd.DataFrame({
                "sex": sens_eval["sex"].values,
                "cosine": cos_sim,
                "top5": top5_ov,
                "rank": rank_corr
            })

            for sex_grp, grp_data in df_curr_indiv.groupby("sex"):
                grp_cos = summarize_metric_distribution(grp_data["cosine"], random_state=42)
                grp_top5 = summarize_metric_distribution(grp_data["top5"], random_state=42)
                grp_rank = summarize_metric_distribution(grp_data["rank"], random_state=42)

                all_group_records.append({
                    "dataset": ds_name,
                    "model": m_name,
                    "sex": sex_grp,
                    "n_applicants": len(grp_data),
                    "cosine_mean": grp_cos["mean"],
                    "cosine_ci_95_low": grp_cos["mean_ci_95"][0],
                    "cosine_ci_95_high": grp_cos["mean_ci_95"][1],
                    "cosine_median": grp_cos["median"],
                    "cosine_std": grp_cos["std"],
                    "top5_overlap_mean": grp_top5["mean"],
                    "top5_overlap_ci_95_low": grp_top5["mean_ci_95"][0],
                    "top5_overlap_ci_95_high": grp_top5["mean_ci_95"][1],
                    "top5_overlap_median": grp_top5["median"],
                    "top5_overlap_std": grp_top5["std"],
                    "rank_corr_mean": grp_rank["mean"],
                    "rank_corr_ci_95_low": grp_rank["mean_ci_95"][0],
                    "rank_corr_ci_95_high": grp_rank["mean_ci_95"][1],
                    "rank_corr_median": grp_rank["median"],
                    "rank_corr_std": grp_rank["std"]
                })

            elapsed = time.time() - t0
            print(f"  >>> Completed {ds_name} - {m_name} in {elapsed:.2f}s | Cosine: {cos_dist['mean']:.4f} | Top5: {top5_dist['mean']:.4f} | Rank: {rank_dist['mean']:.4f}")

    # Export CSV files
    print("\n" + "=" * 80)
    print("EXPORTING PHASE 4 RESULT CSVs")
    print("=" * 80)

    # 1. Individual metrics
    df_indiv = pd.DataFrame(all_individual_records)
    path_indiv = os.path.join(results_dir, "shap_individual_metrics.csv")
    df_indiv.to_csv(path_indiv, index=False)
    print(f"Saved: {path_indiv} ({len(df_indiv)} rows)")

    # 2. Summary metrics
    df_summary = pd.DataFrame(all_summary_records)
    path_summary = os.path.join(results_dir, "shap_summary.csv")
    df_summary.to_csv(path_summary, index=False)
    print(f"Saved: {path_summary} ({len(df_summary)} rows)")

    # 3. Group metrics
    df_group = pd.DataFrame(all_group_records)
    path_group = os.path.join(results_dir, "shap_group_metrics.csv")
    df_group.to_csv(path_group, index=False)
    print(f"Saved: {path_group} ({len(df_group)} rows)")

    # 4. Link with Phase 3 authoritative results
    path_phase3 = os.path.join(results_dir, "mitigation_master_results.csv")
    if os.path.exists(path_phase3):
        df_p3 = pd.read_csv(path_phase3)
        # Filter for Reweighing method
        df_p3_rw = df_p3[df_p3["mitigation_method"] == "reweighing"].copy()

        # Merge on dataset and model
        merged = pd.merge(
            df_p3_rw[[
                "dataset", "model", "mitigation_method",
                "baseline_roc_auc", "mitigated_roc_auc", "delta_roc_auc",
                "baseline_eod", "mitigated_eod", "delta_eod"
            ]],
            df_summary[[
                "dataset", "model",
                "cosine_mean", "top5_overlap_mean", "rank_corr_mean"
            ]],
            on=["dataset", "model"],
            how="inner"
        )

        # Rename to mandated schema:
        # Dataset, Model, Mitigation, Baseline AUC, Mitigated AUC, Delta AUC,
        # Baseline EOD, Mitigated EOD, Delta EOD, Mean SHAP Cosine, Mean Top-5 Overlap, Mean Rank Correlation
        merged_out = merged.rename(columns={
            "dataset": "Dataset",
            "model": "Model",
            "mitigation_method": "Mitigation",
            "baseline_roc_auc": "Baseline AUC",
            "mitigated_roc_auc": "Mitigated AUC",
            "delta_roc_auc": "Delta AUC",
            "baseline_eod": "Baseline EOD",
            "mitigated_eod": "Mitigated EOD",
            "delta_eod": "Delta EOD",
            "cosine_mean": "Mean SHAP Cosine",
            "top5_overlap_mean": "Mean Top-5 Overlap",
            "rank_corr_mean": "Mean Rank Correlation"
        })

        path_phase4_results = os.path.join(results_dir, "phase4_explanation_results.csv")
        merged_out.to_csv(path_phase4_results, index=False)
        print(f"Saved: {path_phase4_results} ({len(merged_out)} rows)")
        print("\nPhase 4 Explanation Results Table:")
        print(merged_out.to_string(index=False))
    else:
        print(f"WARNING: Phase 3 authoritative results not found at {path_phase3}")

    print("\nPhase 4 SHAP Analysis Successfully Completed!")


if __name__ == "__main__":
    run_phase4_shap_analysis()
