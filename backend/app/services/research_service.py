from backend.app.services.feature_dictionary import format_feature_label
import os
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

class ResearchDataService:
    def __init__(self):
        self.results_dir = os.path.join(project_root, "results")
        self.models_dir = os.path.join(project_root, "models")
        self.dashboard_data_dir = os.path.join(project_root, "dashboard", "data")
        self._cache: Dict[str, Any] = {}

    def get_manifest(self) -> Dict[str, Any]:
        if "manifest" not in self._cache:
            path = os.path.join(self.models_dir, "models_manifest.json")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Models manifest not found at {path}")
            with open(path, "r") as f:
                self._cache["manifest"] = json.load(f)
        return self._cache["manifest"]

    def get_mitigation_master(self) -> List[Dict[str, Any]]:
        if "mitigation_master" not in self._cache:
            path = os.path.join(self.results_dir, "mitigation_master_results.csv")
            # keep_default_na=False ensures 'N/A' strings are preserved
            df = pd.read_csv(path, keep_default_na=False)
            df["dataset"] = df["dataset"].str.lower().str.strip()
            df["model"] = df["model"].str.lower().str.strip()
            df["mitigation_method"] = df["mitigation_method"].str.lower().str.strip()
            self._cache["mitigation_master"] = df.to_dict(orient="records")
        return self._cache["mitigation_master"]

    def get_final_research_results(self) -> List[Dict[str, Any]]:
        if "final_research" not in self._cache:
            path = os.path.join(self.results_dir, "final_research_results.csv")
            df = pd.read_csv(path)
            df["Dataset"] = df["Dataset"].astype(str).str.lower().str.strip()
            df["Model"] = df["Model"].astype(str).str.lower().str.strip()
            df = df.where(pd.notnull(df), None)
            self._cache["final_research"] = df.to_dict(orient="records")
        return self._cache["final_research"]

    def get_shap_summary(self) -> List[Dict[str, Any]]:
        if "shap_summary" not in self._cache:
            path = os.path.join(self.results_dir, "shap_summary.csv")
            df = pd.read_csv(path)
            df["dataset"] = df["dataset"].str.lower().str.strip()
            df["model"] = df["model"].str.lower().str.strip()
            
            records = []
            for _, r in df.iterrows():
                records.append({
                    "dataset": str(r["dataset"]).lower().strip(),
                    "model": str(r["model"]).lower().strip(),
                    "mean_cosine": round(float(r["cosine_mean"]), 4),
                    "ci95_cosine": f"[{r['cosine_ci_95_low']:.4f}, {r['cosine_ci_95_high']:.4f}]",
                    "mean_top5": round(float(r["top5_overlap_mean"]), 4),
                    "ci95_top5": f"[{r['top5_overlap_ci_95_low']:.4f}, {r['top5_overlap_ci_95_high']:.4f}]",
                    "mean_rank": round(float(r["rank_corr_mean"]), 4),
                    "ci95_rank": f"[{r['rank_corr_ci_95_low']:.4f}, {r['rank_corr_ci_95_high']:.4f}]"
                })
            self._cache["shap_summary"] = records
        return self._cache["shap_summary"]

    def get_shap_group_metrics(self) -> List[Dict[str, Any]]:
        if "shap_groups" not in self._cache:
            path = os.path.join(self.results_dir, "shap_group_metrics.csv")
            df = pd.read_csv(path)
            self._cache["shap_groups"] = df.to_dict(orient="records")
        return self._cache["shap_groups"]

    def get_research_matrix(self) -> List[Dict[str, Any]]:
        """
        Builds the complete quantitative research matrix for all 18 mitigation experiments,
        joined with SHAP explanation consistency metrics from authoritative sources.
        """
        if "matrix" not in self._cache:
            mit_path = os.path.join(self.results_dir, "mitigation_master_results.csv")
            shap_path = os.path.join(self.results_dir, "shap_summary.csv")

            df_mit = pd.read_csv(mit_path, keep_default_na=False)
            df_shap = pd.read_csv(shap_path)

            df_mit["dataset"] = df_mit["dataset"].str.lower().str.strip()
            df_mit["model"] = df_mit["model"].str.lower().str.strip()
            df_mit["mitigation_method"] = df_mit["mitigation_method"].str.lower().str.strip()

            df_shap["dataset"] = df_shap["dataset"].str.lower().str.strip()
            df_shap["model"] = df_shap["model"].str.lower().str.strip()

            shap_map = {(r["dataset"], r["model"]): r for _, r in df_shap.iterrows()}

            matrix_rows = []
            for _, r in df_mit.iterrows():
                ds = str(r["dataset"]).lower().strip()
                m = str(r["model"]).lower().strip()
                mm = str(r["mitigation_method"]).lower().strip()
                shap_info = shap_map.get((ds, m)) if mm == "reweighing" else None

                matrix_rows.append({
                    "dataset": ds,
                    "model": m,
                    "mitigation_method": mm,
                    "baseline_acc": float(r["baseline_accuracy"]),
                    "mitigated_acc": float(r["mitigated_accuracy"]),
                    "delta_acc": float(r["delta_accuracy"]),
                    "baseline_auc": float(r["baseline_roc_auc"]),
                    "mitigated_auc": str(r["mitigated_roc_auc"]),
                    "delta_auc": str(r["delta_roc_auc"]),
                    "baseline_eod": float(r["baseline_eod"]),
                    "mitigated_eod": float(r["mitigated_eod"]),
                    "delta_eod": float(r["delta_eod"]),
                    "baseline_dpd": float(r["baseline_dpd"]),
                    "mitigated_dpd": float(r["mitigated_dpd"]),
                    "delta_dpd": float(r["delta_dpd"]),
                    "shap_cosine_sim": round(float(shap_info["cosine_mean"]), 4) if shap_info is not None else None,
                    "shap_top5_overlap": round(float(shap_info["top5_overlap_mean"]), 4) if shap_info is not None else None,
                    "shap_spearman_rank": round(float(shap_info["rank_corr_mean"]), 4) if shap_info is not None else None,
                })

            self._cache["matrix"] = matrix_rows
        return self._cache["matrix"]

    def get_qualitative_matrix(self) -> List[Dict[str, Any]]:
        """Returns the high-level qualitative observations matrix."""
        path = os.path.join(self.results_dir, "fairness_performance_explanation_matrix.csv")
        df = pd.read_csv(path, keep_default_na=False)
        return df.to_dict(orient="records")

    def get_individual_shap_records(
        self,
        dataset: str,
        model: str,
        applicant_idx: int = 0
    ) -> List[Dict[str, Any]]:
        parquet_path = os.path.join(self.dashboard_data_dir, "individual_feature_shap.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet artifact missing at {parquet_path}")

        if "df_individual_shap" not in self._cache:
            df = pd.read_parquet(parquet_path)
            df["dataset"] = df["dataset"].str.lower().str.strip()
            df["model"] = df["model"].str.lower().str.strip()
            self._cache["df_individual_shap"] = df

        df_all = self._cache["df_individual_shap"]
        ds = dataset.lower().strip()
        m = model.lower().strip()

        sub = df_all[
            (df_all["dataset"] == ds) &
            (df_all["model"] == m) &
            (df_all["applicant_idx"] == applicant_idx)
        ]

        if sub.empty:
            # Fallback to first available applicant index for this dataset and model
            available = df_all[(df_all["dataset"] == ds) & (df_all["model"] == m)]
            if available.empty:
                return []
            first_idx = int(available["applicant_idx"].iloc[0])
            sub = df_all[
                (df_all["dataset"] == ds) &
                (df_all["model"] == m) &
                (df_all["applicant_idx"] == first_idx)
            ]

        records = []
        for _, row in sub.iterrows():
            raw_f = str(row["feature"])
            records.append({
                "feature": format_feature_label(raw_f),
                "feature_raw": raw_f,
                "baseline_shap": float(row["baseline_shap"]),
                "reweighed_shap": float(row["reweighed_shap"]),
                "diff_shap": float(row["diff_shap"])
            })
        return records

    def get_case_study_data(self) -> Dict[str, Any]:
        """German Random Forest key finding evidence."""
        return {
            "dataset": "german",
            "model": "random_forest",
            "mitigation": "reweighing",
            "metrics": {
                "roc_auc_baseline": 0.7798,
                "roc_auc_reweighed": 0.7935,
                "roc_auc_delta": 0.0137,
                "eod_baseline": 0.1250,
                "eod_reweighed": 0.0500,
                "eod_delta": -0.0750,
                "distance_from_parity_reduction": "60.0%",
                "mean_shap_cosine_similarity": 0.9652,
                "mean_top5_overlap": 0.8140,
                "mean_spearman_rank": 0.7981
            },
            "interpretation": (
                "For German Random Forest, Reweighing reduced the Equalized Odds Difference "
                "from 0.125 to 0.050 (a 60% reduction in demographic disparity) while increasing "
                "ROC-AUC from 0.7798 to 0.7935 and maintaining a high SHAP explanation cosine similarity "
                "of 0.9652. However, this win-win behavior was configuration-specific and did not generalize "
                "uniformly to all other model architectures."
            )
        }


    def _safe_divide(self, numerator: float, denominator: float, fallback: float = 0.0) -> float:
        if denominator == 0.0 or np.isnan(denominator):
            return fallback
        return numerator / denominator

    def _compute_group_metrics(self, row: pd.Series) -> Dict[str, Any]:
        n = int(row['N'])
        tp = int(row['TP'])
        tn = int(row['TN'])
        fp = int(row['FP'])
        fn = int(row['FN'])
        tpr = round(float(row['TPR']), 4)
        fpr = round(float(row['FPR']), 4)
        sel_rate = round(float(row['selection_rate']), 4)
        app_rate = round(float(row['approval_rate']), 4)
        obs_def = round(self._safe_divide(tp + fn, n), 4)

        return {
            "group": str(row['group']),
            "group_role": str(row['group_role']),
            "sample_count": n,
            "observed_default_rate": obs_def,
            "selection_rate": sel_rate,
            "approval_rate": app_rate,
            "true_positive_rate": tpr,
            "false_positive_rate": fpr,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        }

    def _compute_fairness_disparities(self, unprivileged: Dict[str, Any], privileged: Dict[str, Any]) -> Dict[str, Any]:
        eopp = round(abs(unprivileged["true_positive_rate"] - privileged["true_positive_rate"]), 4)
        fpr_diff = round(abs(unprivileged["false_positive_rate"] - privileged["false_positive_rate"]), 4)
        eod = round(max(eopp, fpr_diff), 4)
        dpd = round(abs(unprivileged["selection_rate"] - privileged["selection_rate"]), 4)
        approval_di = round(self._safe_divide(unprivileged["approval_rate"], privileged["approval_rate"], 1.0), 4)
        di_dist = round(abs(approval_di - 1.0), 4)

        return {
            "equalized_odds_difference": eod,
            "equal_opportunity_difference": eopp,
            "demographic_parity_difference": dpd,
            "approval_disparate_impact": approval_di,
            "approval_di_distance": di_dist,
            "unprivileged_group": unprivileged["group"],
            "privileged_group": privileged["group"]
        }

    def get_fairness_audit(
        self,
        dataset: str,
        model: str,
        mitigation: str = "reweighing",
        protected_attribute: str = "sex"
    ) -> Dict[str, Any]:
        ds = dataset.lower().strip()
        m = model.lower().strip()
        mit = mitigation.lower().strip()
        pa = protected_attribute.lower().strip()

        valid_datasets = ["taiwan", "german"]
        valid_models = ["logistic_regression", "random_forest", "xgboost"]
        valid_mitigations = ["reweighing", "exponentiated_gradient", "threshold_optimizer"]
        valid_pas = ["sex", "age"]

        if ds not in valid_datasets:
            raise ValueError(f"Unsupported dataset: '{dataset}'. Supported: {valid_datasets}")
        if m not in valid_models:
            raise ValueError(f"Unsupported model: '{model}'. Supported: {valid_models}")
        if mit not in valid_mitigations:
            raise ValueError(f"Unsupported mitigation: '{mitigation}'. Supported: {valid_mitigations}")
        if pa not in valid_pas:
            raise ValueError(f"Unsupported protected attribute: '{protected_attribute}'. Supported: {valid_pas}")

        # Load baseline subgroups
        if "df_b_sub" not in self._cache:
            p = os.path.join(self.results_dir, "baseline_subgroup_metrics.csv")
            df = pd.read_csv(p)
            df['dataset'] = df['dataset'].astype(str).str.lower().str.strip()
            df['model'] = df['model'].astype(str).str.lower().str.strip()
            df['protected_attribute'] = df['protected_attribute'].astype(str).str.lower().str.strip()
            self._cache["df_b_sub"] = df
        df_b_sub = self._cache["df_b_sub"]

        # Load mitigated subgroups
        if "df_m_sub" not in self._cache:
            p = os.path.join(self.results_dir, "mitigated_subgroup_metrics.csv")
            df = pd.read_csv(p)
            df['dataset'] = df['dataset'].astype(str).str.lower().str.strip()
            df['model'] = df['model'].astype(str).str.lower().str.strip()
            df['mitigation_method'] = df['mitigation_method'].astype(str).str.lower().str.strip()
            df['protected_attribute'] = df['protected_attribute'].astype(str).str.lower().str.strip()
            self._cache["df_m_sub"] = df
        df_m_sub = self._cache["df_m_sub"]

        # Load mitigation master
        if "df_mm" not in self._cache:
            p = os.path.join(self.results_dir, "mitigation_master_results.csv")
            df = pd.read_csv(p, keep_default_na=False)
            df['dataset'] = df['dataset'].astype(str).str.lower().str.strip()
            df['model'] = df['model'].astype(str).str.lower().str.strip()
            df['mitigation_method'] = df['mitigation_method'].astype(str).str.lower().str.strip()
            self._cache["df_mm"] = df
        df_mm = self._cache["df_mm"]

        b_rows = df_b_sub[
            (df_b_sub['dataset'] == ds) &
            (df_b_sub['model'] == m) &
            (df_b_sub['protected_attribute'] == pa)
        ]
        if len(b_rows) < 2:
            raise ValueError(f"Missing baseline subgroup data for {ds}-{m}-{pa}")

        b_u_row = b_rows[b_rows['group_role'] == 'unprivileged'].iloc[0]
        b_p_row = b_rows[b_rows['group_role'] == 'privileged'].iloc[0]

        b_u_metrics = self._compute_group_metrics(b_u_row)
        b_p_metrics = self._compute_group_metrics(b_p_row)
        baseline_subgroups = [b_u_metrics, b_p_metrics]
        baseline_fairness = self._compute_fairness_disparities(b_u_metrics, b_p_metrics)

        m_rows = df_m_sub[
            (df_m_sub['dataset'] == ds) &
            (df_m_sub['model'] == m) &
            (df_m_sub['mitigation_method'] == mit) &
            (df_m_sub['protected_attribute'] == pa)
        ]
        if len(m_rows) < 2:
            raise ValueError(f"Missing mitigated subgroup data for {ds}-{m}-{mit}-{pa}")

        m_u_row = m_rows[m_rows['group_role'] == 'unprivileged'].iloc[0]
        m_p_row = m_rows[m_rows['group_role'] == 'privileged'].iloc[0]

        m_u_metrics = self._compute_group_metrics(m_u_row)
        m_p_metrics = self._compute_group_metrics(m_p_row)
        mitigated_subgroups = [m_u_metrics, m_p_metrics]
        mitigated_fairness = self._compute_fairness_disparities(m_u_metrics, m_p_metrics)

        perf_rows = df_mm[
            (df_mm['dataset'] == ds) &
            (df_mm['model'] == m) &
            (df_mm['mitigation_method'] == mit)
        ]
        perf = perf_rows.iloc[0] if not perf_rows.empty else None

        comparisons = []

        # 1. EOD
        b_eod = baseline_fairness["equalized_odds_difference"]
        m_eod = mitigated_fairness["equalized_odds_difference"]
        d_eod = round(m_eod - b_eod, 4)
        dir_eod = "improved" if d_eod < -0.0001 else ("degraded" if d_eod > 0.0001 else "unchanged")
        comparisons.append({
            "metric_name": "Equalized Odds Difference (EOD)",
            "metric_key": "eod",
            "baseline_value": b_eod,
            "mitigated_value": m_eod,
            "absolute_change": d_eod,
            "direction": dir_eod,
            "interpretation": "Disparity between subgroup TPR and FPR decreased" if dir_eod == "improved" else (
                "Disparity between subgroup TPR and FPR increased" if dir_eod == "degraded" else "Disparity unchanged"
            ),
            "parity_benchmark": "0.0000 (Exact equalized odds across groups)",
            "four_fifths_status": None
        })

        # 2. DPD
        b_dpd = baseline_fairness["demographic_parity_difference"]
        m_dpd = mitigated_fairness["demographic_parity_difference"]
        d_dpd = round(m_dpd - b_dpd, 4)
        dir_dpd = "improved" if d_dpd < -0.0001 else ("degraded" if d_dpd > 0.0001 else "unchanged")
        comparisons.append({
            "metric_name": "Demographic Parity Difference (DPD)",
            "metric_key": "dpd",
            "baseline_value": b_dpd,
            "mitigated_value": m_dpd,
            "absolute_change": d_dpd,
            "direction": dir_dpd,
            "interpretation": "Subgroup default prediction rate gap narrowed" if dir_dpd == "improved" else (
                "Subgroup default prediction rate gap widened" if dir_dpd == "degraded" else "Rate gap unchanged"
            ),
            "parity_benchmark": "0.0000 (Equal default prediction rates)",
            "four_fifths_status": None
        })

        # 3. Approval Disparate Impact
        b_di = baseline_fairness["approval_disparate_impact"]
        m_di = mitigated_fairness["approval_disparate_impact"]
        d_di = round(m_di - b_di, 4)
        b_dist = abs(b_di - 1.0)
        m_dist = abs(m_di - 1.0)
        dir_di = "improved" if m_dist < b_dist - 0.0001 else ("degraded" if m_dist > b_dist + 0.0001 else "unchanged")
        ff_pass = (0.80 <= m_di <= 1.25)
        comparisons.append({
            "metric_name": "Approval Disparate Impact (Approval DI)",
            "metric_key": "approval_di",
            "baseline_value": b_di,
            "mitigated_value": m_di,
            "absolute_change": d_di,
            "direction": dir_di,
            "interpretation": f"Approval ratio moved closer to parity (1.0000) by {abs(b_dist - m_dist):.4f}" if dir_di == "improved" else (
                f"Approval ratio moved further from parity by {abs(m_dist - b_dist):.4f}" if dir_di == "degraded" else "Ratio distance unchanged"
            ),
            "parity_benchmark": "1.0000 (Equal approval probability AR_u / AR_p)",
            "four_fifths_status": "Passes 80% screening rule (0.80 - 1.25)" if ff_pass else "Outside 80% screening window"
        })

        # 4. Accuracy
        if perf is not None:
            b_acc = round(float(perf['baseline_accuracy']), 4)
            m_acc = round(float(perf['mitigated_accuracy']), 4)
            d_acc = round(m_acc - b_acc, 4)
            dir_acc = "improved" if d_acc > 0.0001 else ("degraded" if d_acc < -0.0001 else "unchanged")
            comparisons.append({
                "metric_name": "Test Accuracy",
                "metric_key": "accuracy",
                "baseline_value": b_acc,
                "mitigated_value": m_acc,
                "absolute_change": d_acc,
                "direction": dir_acc,
                "interpretation": f"Test classification accuracy changed by {d_acc:+.4f}",
                "parity_benchmark": "1.0000 (Maximum empirical accuracy)",
                "four_fifths_status": None
            })

            # 5. ROC-AUC
            raw_m_auc = str(perf['mitigated_roc_auc']).strip()
            b_auc = round(float(perf['baseline_roc_auc']), 4)
            if raw_m_auc.upper() == "N/A":
                comparisons.append({
                    "metric_name": "ROC-AUC",
                    "metric_key": "roc_auc",
                    "baseline_value": b_auc,
                    "mitigated_value": None,
                    "absolute_change": None,
                    "direction": "not_applicable",
                    "interpretation": "N/A: ThresholdOptimizer produces discrete binary decisions, not continuous ranking probabilities.",
                    "parity_benchmark": "1.0000 (Maximum discrimination capability)",
                    "four_fifths_status": None
                })
            else:
                m_auc = round(float(raw_m_auc), 4)
                d_auc = round(m_auc - b_auc, 4)
                dir_auc = "improved" if d_auc > 0.0001 else ("degraded" if d_auc < -0.0001 else "unchanged")
                comparisons.append({
                    "metric_name": "ROC-AUC",
                    "metric_key": "roc_auc",
                    "baseline_value": b_auc,
                    "mitigated_value": m_auc,
                    "absolute_change": d_auc,
                    "direction": dir_auc,
                    "interpretation": f"ROC-AUC discrimination capability changed by {d_auc:+.4f}",
                    "parity_benchmark": "1.0000 (Maximum discrimination capability)",
                    "four_fifths_status": None
                })

        findings = []
        decreased = [c["metric_name"] for c in comparisons if c["direction"] == "improved" and c["metric_key"] in ["eod", "dpd", "approval_di"]]
        increased = [c["metric_name"] for c in comparisons if c["direction"] == "degraded" and c["metric_key"] in ["eod", "dpd", "approval_di"]]
        unchanged = [c["metric_name"] for c in comparisons if c["direction"] == "unchanged" and c["metric_key"] in ["eod", "dpd", "approval_di"]]

        if decreased:
            findings.append(f"Measured disparities decreased for: {', '.join(decreased)}.")
        if unchanged:
            findings.append(f"Measured disparities remained unchanged for: {', '.join(unchanged)}.")
        if increased:
            findings.append(f"Measured disparities increased for: {', '.join(increased)}.")

        if decreased and increased:
            findings.append(
                "Fairness metric conflict observed: One or more fairness metrics improved while another widened, "
                "highlighting the mathematical trade-off between equal opportunity/odds and demographic parity under unequal demographic base rates."
            )

        if ds == "german":
            findings.append(
                "Sample-size sensitivity: The German Credit test set contains N=200 applicants "
                f"(unprivileged '{b_u_metrics['group']}' N={b_u_metrics['sample_count']}, privileged '{b_p_metrics['group']}' N={b_p_metrics['sample_count']}). "
                "A shift in a single individual prediction alters group rates by ~1.6% to 2.3%. Observed rate changes carry wider empirical uncertainty."
            )
        else:
            findings.append(
                "Sample-size context: The Taiwan Credit test set contains N=6,000 applicants "
                f"(unprivileged '{b_u_metrics['group']}' N={b_u_metrics['sample_count']}, privileged '{b_p_metrics['group']}' N={b_p_metrics['sample_count']}), "
                "yielding tighter empirical rate stability."
            )

        findings.append(
            "Screening threshold caution: The EEOC four-fifths (80%) rule is an administrative screening heuristic [0.80, 1.25], "
            "not a legal proof of non-discrimination or a causal guarantee."
        )

        evaluation_context = {
            "dataset_name": ds,
            "dataset_display": "Statlog German Credit" if ds == "german" else "Taiwan Credit Card Default",
            "model_architecture": m,
            "mitigation_method": mit,
            "protected_attribute": pa,
            "protected_attribute_label": "Gender (Sex)" if pa == "sex" else "Age Cohort (<=25 vs >25)",
            "unprivileged_group_definition": b_u_metrics["group"],
            "privileged_group_definition": b_p_metrics["group"],
            "data_source": "Frozen research benchmark: results/baseline_subgroup_metrics.csv & results/mitigated_subgroup_metrics.csv",
            "total_test_samples": b_u_metrics["sample_count"] + b_p_metrics["sample_count"],
            "disclaimer": "Research prototype audit. Metrics represent empirical observations on held-out test splits. Not for automated credit decisions."
        }

        conventions = {
            "target_positive_class": "Y=1 indicates Credit Default / Risk Event (unfavorable ground truth)",
            "prediction_positive_class": "Y_hat=1 indicates Predicted Default / Credit Rejection",
            "approval_event": "Y_hat=0 indicates Predicted Non-Default / Credit Approval (Favorable outcome)",
            "approval_disparate_impact_ordering": "Numerator: Unprivileged group approval rate. Denominator: Privileged group approval rate.",
            "demographic_parity_formula": "Absolute difference in default prediction rates |SR_u - SR_p|",
            "equalized_odds_formula": "max(|TPR_u - TPR_p|, |FPR_u - FPR_p|)"
        }

        return {
            "dataset": ds,
            "model": m,
            "mitigation_method": mit,
            "protected_attribute": pa,
            "evaluation_context": evaluation_context,
            "conventions": conventions,
            "baseline_subgroups": baseline_subgroups,
            "mitigated_subgroups": mitigated_subgroups,
            "baseline_fairness": baseline_fairness,
            "mitigated_fairness": mitigated_fairness,
            "metric_comparisons": comparisons,
            "audit_findings": findings
        }

research_service = ResearchDataService()
