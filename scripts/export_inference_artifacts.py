import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

project_root = r"D:\sarthak\Subjects\ERA\FairCredit AI"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.modeling.preprocessing import load_dataset, split_dataset, build_preprocessor
from src.modeling.models import get_model, build_pipeline
from src.modeling.mitigation import fit_reweighing
from src.explainability import sample_background_data

models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

datasets = ["taiwan", "german"]
model_names = ["logistic_regression", "random_forest", "xgboost"]

manifest = {
    "datasets": {},
    "configurations": []
}

print("="*70)
print("EXPORTING AND VERIFYING INFERENCE MODEL ARTIFACTS")
print("="*70)

for ds in datasets:
    X, y, sens, feat_types = load_dataset(ds)
    X_train, X_test, y_train, y_test, sens_train, sens_test = split_dataset(
        X, y, sens, test_size=0.20, random_state=42
    )
    
    sens_train_primary = sens_train["sex"]
    
    # Store schema metadata
    cat_cols = feat_types["categorical"]
    num_cols = feat_types["numerical"]
    
    cat_unique_values = {col: sorted([str(v) for v in X[col].unique().tolist()]) for col in cat_cols}
    num_ranges = {col: {"min": float(X[col].min()), "max": float(X[col].max()), "median": float(X[col].median())} for col in num_cols}
    
    # Extract synthetic demonstration applicants
    good_sample = X_test[y_test == 0].iloc[0].to_dict()
    bad_sample = X_test[y_test == 1].iloc[0].to_dict()
    
    def clean_sample(s):
        return {k: (int(v) if isinstance(v, (np.integer, int)) else float(v) if isinstance(v, (np.floating, float)) else str(v)) for k, v in s.items()}

    manifest["datasets"][ds] = {
        "features_in_X": list(X.columns),
        "categorical_features": cat_cols,
        "numerical_features": num_cols,
        "categorical_unique_values": cat_unique_values,
        "numerical_ranges": num_ranges,
        "sample_applicants": {
            "applicant_a_low_risk": clean_sample(good_sample),
            "applicant_b_higher_risk": clean_sample(bad_sample)
        }
    }

    for m_name in model_names:
        print(f"\n--- Processing {ds.upper()} : {m_name.upper()} ---")
        scale_num = (m_name == "logistic_regression")
        
        # 1. Baseline Pipeline
        preprocessor_base = build_preprocessor(cat_cols, num_cols, scale_numeric=scale_num)
        pipeline_base = build_pipeline(preprocessor_base, get_model(m_name, random_state=42))
        pipeline_base.fit(X_train, y_train)
        
        # 2. Reweighed Pipeline
        preprocessor_rew = build_preprocessor(cat_cols, num_cols, scale_numeric=scale_num)
        pipeline_rew = fit_reweighing(
            m_name,
            preprocessor_rew,
            X_train,
            y_train,
            sens_train_primary,
            random_state=42
        )
        
        # 3. Test Set Inference Validation
        y_prob_base = pipeline_base.predict_proba(X_test)[:, 1]
        y_pred_base = pipeline_base.predict(X_test)
        auc_base = roc_auc_score(y_test, y_prob_base)
        acc_base = accuracy_score(y_test, y_pred_base)
        
        y_prob_rew = pipeline_rew.predict_proba(X_test)[:, 1]
        y_pred_rew = pipeline_rew.predict(X_test)
        auc_rew = roc_auc_score(y_test, y_prob_rew)
        acc_rew = accuracy_score(y_test, y_pred_rew)
        
        print(f"  Baseline  -> ROC-AUC: {auc_base:.4f}, Accuracy: {acc_base:.4f}")
        print(f"  Reweighed -> ROC-AUC: {auc_rew:.4f}, Accuracy: {acc_rew:.4f}")
        
        # 4. Background data for LinearExplainer / SHAP
        X_train_proc = pipeline_base.named_steps["preprocessor"].transform(X_train)
        background_sample = sample_background_data(X_train_proc, n_samples=100, random_state=42)
        
        # 5. Serialize Artifacts
        base_path = os.path.join(models_dir, f"{ds}_{m_name}_baseline.joblib")
        rew_path = os.path.join(models_dir, f"{ds}_{m_name}_reweighed.joblib")
        bg_path = os.path.join(models_dir, f"{ds}_{m_name}_background.joblib")
        
        joblib.dump(pipeline_base, base_path)
        joblib.dump(pipeline_rew, rew_path)
        joblib.dump(background_sample, bg_path)
        
        manifest["configurations"].append({
            "dataset": ds,
            "model": m_name,
            "baseline_model_file": f"{ds}_{m_name}_baseline.joblib",
            "reweighed_model_file": f"{ds}_{m_name}_reweighed.joblib",
            "background_file": f"{ds}_{m_name}_background.joblib",
            "test_metrics": {
                "baseline_auc": round(float(auc_base), 4),
                "baseline_accuracy": round(float(acc_base), 4),
                "reweighed_auc": round(float(auc_rew), 4),
                "reweighed_accuracy": round(float(acc_rew), 4)
            }
        })

manifest_path = os.path.join(models_dir, "models_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print("\n"+"="*70)
print(f"SUCCESS: All 12 model pipelines & backgrounds serialized to {models_dir}")
print(f"Manifest written to {manifest_path}")
print("="*70)
