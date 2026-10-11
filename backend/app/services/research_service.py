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

research_service = ResearchDataService()
