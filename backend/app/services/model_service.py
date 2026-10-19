import os
import sys
import json
from typing import Dict, Any, List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.explainability import get_model_explainer, compute_shap_attributions
from backend.app.schemas.assessment import (
    PredictionModelOutput,
    PredictionDelta,
    AssessmentResponse,
    FeatureAttribution,
    ExplanationResponse
)

from backend.app.services.feature_dictionary import format_feature_label
class ModelInferenceService:
    def validate_features(self, dataset: str, features: Dict[str, Any]) -> None:
        if not isinstance(features, dict):
            raise ValueError("Applicant features must be provided as a key-value dictionary.")
        if len(features) == 0:
            raise ValueError("Applicant feature payload cannot be empty.")
            
        manifest = self.get_manifest()
        if dataset not in manifest["datasets"]:
            raise ValueError(f"Unsupported dataset '{dataset}'. Supported datasets: {list(manifest['datasets'].keys())}")
            
        ds_meta = manifest["datasets"][dataset]
        num_ranges = ds_meta.get("numerical_ranges", {})
        cat_values = ds_meta.get("categorical_unique_values", {})
        
        errors = []
        for k, v in features.items():
            if k in num_ranges:
                try:
                    num_val = float(v)
                except (ValueError, TypeError):
                    errors.append(f"Field '{k}' must be a valid number, got '{v}'")
                    continue
                # Physical/logical bounds validation
                if k in ("duration_months", "credit_amount", "limit_bal") and num_val <= 0:
                    errors.append(f"Field '{k}' must be strictly positive, got {num_val}")
                elif k == "age" and (num_val < 18 or num_val > 120):
                    errors.append(f"Field 'age' must be between 18 and 120 years, got {num_val}")
                elif k in ("pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6") and (num_val < -2 or num_val > 10):
                    errors.append(f"Taiwan payment status '{k}' must be between -2 and 10, got {num_val}")
            elif k in cat_values:
                allowed = set(cat_values[k])
                try:
                    f_val = float(v)
                    str_v = str(int(f_val)) if f_val.is_integer() else str(v)
                except (ValueError, TypeError):
                    str_v = str(v)

                if str_v not in allowed and str(v) not in allowed:
                    errors.append(f"Field '{k}' value '{v}' is invalid. Allowed values: {sorted(list(allowed))}")
                    
        if errors:
            raise ValueError("; ".join(errors))

    def __init__(self):
        self.models_dir = os.path.join(project_root, "models")
        self._models: Dict[str, Any] = {}
        self._backgrounds: Dict[str, Any] = {}
        self._explainers: Dict[str, Any] = {}
        self._manifest: Optional[Dict[str, Any]] = None

    def get_manifest(self) -> Dict[str, Any]:
        if self._manifest is None:
            path = os.path.join(self.models_dir, "models_manifest.json")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model manifest missing at {path}")
            with open(path, "r") as f:
                self._manifest = json.load(f)
        return self._manifest

    def _get_pipeline(self, dataset: str, model: str, variant: str):
        key = f"{dataset}_{model}_{variant}"
        if key not in self._models:
            filename = f"{dataset}_{model}_{variant}.joblib"
            path = os.path.join(self.models_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Trained model artifact {filename} not found.")
            self._models[key] = joblib.load(path)
        return self._models[key]

    def _get_background(self, dataset: str, model: str):
        key = f"{dataset}_{model}_background"
        if key not in self._backgrounds:
            filename = f"{dataset}_{model}_background.joblib"
            path = os.path.join(self.models_dir, filename)
            if os.path.exists(path):
                self._backgrounds[key] = joblib.load(path)
            else:
                self._backgrounds[key] = None
        return self._backgrounds[key]

    def _get_explainer(self, dataset: str, model: str, variant: str, pipeline):
        key = f"explainer_{dataset}_{model}_{variant}"
        if key not in self._explainers:
            clf = pipeline.named_steps["classifier"]
            bg = self._get_background(dataset, model) if model == "logistic_regression" else None
            self._explainers[key] = get_model_explainer(clf, model, background_data=bg)
        return self._explainers[key]

    def _prepare_dataframe(self, dataset: str, features: Dict[str, Any]) -> pd.DataFrame:
        manifest = self.get_manifest()
        ds_meta = manifest["datasets"][dataset]
        expected_cols = ds_meta["features_in_X"]
        
        row = {}
        for col in expected_cols:
            if col in features:
                val = features[col]
            else:
                # Fallback to median/sample default
                if col in ds_meta["numerical_features"]:
                    val = ds_meta["numerical_ranges"][col]["median"]
                else:
                    val = ds_meta["categorical_unique_values"][col][0]
            
            # Type coerce
            if col in ds_meta["numerical_features"]:
                row[col] = float(val)
            else:
                row[col] = str(val)
                
        df = pd.DataFrame([row], columns=expected_cols)
        return df

    def predict_applicant(
        self,
        dataset: str,
        model: str,
        features: Dict[str, Any]
    ) -> AssessmentResponse:
        ds = dataset.lower().strip()
        m = model.lower().strip()

        self.validate_features(ds, features)
        df_input = self._prepare_dataframe(ds, features)

        pipe_base = self._get_pipeline(ds, m, "baseline")
        pipe_rew = self._get_pipeline(ds, m, "reweighed")

        prob_base = float(pipe_base.predict_proba(df_input)[0, 1])
        pred_base = int(pipe_base.predict(df_input)[0])

        prob_rew = float(pipe_rew.predict_proba(df_input)[0, 1])
        pred_rew = int(pipe_rew.predict(df_input)[0])

        diff = round(prob_rew - prob_base, 4)
        pct_diff = round(diff * 100, 2)

        if diff > 0.0005:
            direction = "increased"
            interp = f"Reweighed model scores default probability {abs(pct_diff):.2f}% higher for this profile."
        elif diff < -0.0005:
            direction = "decreased"
            interp = f"Reweighed model scores default probability {abs(pct_diff):.2f}% lower for this profile."
        else:
            direction = "unchanged"
            interp = "Reweighing produced virtually identical estimated default probability (delta < 0.05%)."

        output_space = "Probability Space P(Y=1)" if m == "random_forest" else "Probability via Sigmoid(Margin)"

        base_out = PredictionModelOutput(
            prediction=pred_base,
            prediction_label="Default Risk (Y=1)" if pred_base == 1 else "Non-Default (Y=0)",
            default_probability=round(prob_base, 4),
            model_type=m,
            mitigation_type="baseline",
            output_space=output_space
        )

        rew_out = PredictionModelOutput(
            prediction=pred_rew,
            prediction_label="Default Risk (Y=1)" if pred_rew == 1 else "Non-Default (Y=0)",
            default_probability=round(prob_rew, 4),
            model_type=m,
            mitigation_type="reweighed",
            output_space=output_space
        )

        delta_out = PredictionDelta(
            probability_difference=diff,
            percent_difference=pct_diff,
            direction=direction,
            interpretation=interp
        )

        return AssessmentResponse(
            dataset=ds,
            model=m,
            baseline=base_out,
            reweighed=rew_out,
            delta=delta_out,
            disclaimer="Research prototype — Not for automated underwriting or individual credit denial."
        )

    def explain_applicant(
        self,
        dataset: str,
        model: str,
        features: Dict[str, Any]
    ) -> ExplanationResponse:
        ds = dataset.lower().strip()
        m = model.lower().strip()

        self.validate_features(ds, features)
        df_input = self._prepare_dataframe(ds, features)

        pipe_base = self._get_pipeline(ds, m, "baseline")
        pipe_rew = self._get_pipeline(ds, m, "reweighed")

        prob_base = float(pipe_base.predict_proba(df_input)[0, 1])
        prob_rew = float(pipe_rew.predict_proba(df_input)[0, 1])

        # Preprocess input through pipelines
        prep_base = pipe_base.named_steps["preprocessor"]
        prep_rew = pipe_rew.named_steps["preprocessor"]

        X_base_proc = prep_base.transform(df_input)
        X_rew_proc = prep_rew.transform(df_input)

        explainer_base = self._get_explainer(ds, m, "baseline", pipe_base)
        explainer_rew = self._get_explainer(ds, m, "reweighed", pipe_rew)

        phi_base = compute_shap_attributions(explainer_base, m, X_base_proc)[0]
        phi_rew = compute_shap_attributions(explainer_rew, m, X_rew_proc)[0]

        feature_names = prep_base.get_feature_names_out()

        # Combine feature attributions
        records = []
        for i, raw_f in enumerate(feature_names):
            b_val = float(phi_base[i])
            r_val = float(phi_rew[i])
            d_val = float(r_val - b_val)
            direction = "increases_risk" if b_val > 0.001 else "reduces_risk" if b_val < -0.001 else "neutral"
            records.append({
                "feature": format_feature_label(raw_f),
                "feature_raw": raw_f,
                "baseline_shap": round(b_val, 4),
                "reweighed_shap": round(r_val, 4),
                "diff_shap": round(d_val, 4),
                "abs_impact": abs(b_val) + abs(r_val),
                "direction": direction
            })

        # Rank by overall attribution magnitude and take top 10
        records.sort(key=lambda x: x["abs_impact"], reverse=True)
        top_10 = [FeatureAttribution(**r) for r in records[:10]]

        if m == "random_forest":
            output_space_desc = "Probability Space P(Y=1) (Additive to baseline default rate)"
        elif m == "logistic_regression":
            output_space_desc = "Log-Odds Margin Space (Linear logit link scale)"
        elif m == "xgboost":
            output_space_desc = "Margin Space (Raw tree ensemble sum before logistic sigmoid)"
        else:
            output_space_desc = "Continuous Output Space"

        summary_text = (
            f"Top driving features for this applicant under {m.replace('_', ' ').title()}: "
            f"'{top_10[0].feature}' (baseline SHAP: {top_10[0].baseline_shap:+.4f}) "
            f"and '{top_10[1].feature}' (baseline SHAP: {top_10[1].baseline_shap:+.4f})."
        )

        explainer_algo = "LinearExplainer" if m == "logistic_regression" else "TreeExplainer"
        return ExplanationResponse(
            dataset=ds,
            model=m,
            output_space=output_space_desc,
            explainer_method=explainer_algo,
            baseline_probability=round(prob_base, 4),
            reweighed_probability=round(prob_rew, 4),
            top_contributors=top_10,
            summary=summary_text,
            disclaimer="Attributions reflect model-specific marginal influence, not direct real-world causal factors."
        )

model_service = ModelInferenceService()
