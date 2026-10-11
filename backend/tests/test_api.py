import sys
import os
import pytest
from fastapi.testclient import TestClient

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_configurations_endpoint():
    response = client.get("/api/configurations")
    assert response.status_code == 200
    data = response.json()
    assert "taiwan" in data["supported_datasets"]
    assert "german" in data["supported_datasets"]
    assert "random_forest" in data["supported_models"]
    assert "taiwan" in data["dataset_schemas"]
    assert "german" in data["dataset_schemas"]
    assert len(data["configurations"]) == 6

def test_predict_german_random_forest():
    # Load sample from configurations
    cfg_resp = client.get("/api/configurations").json()
    german_sample = cfg_resp["dataset_schemas"]["german"]["sample_applicants"]["applicant_a_low_risk"]

    payload = {
        "dataset": "german",
        "model": "random_forest",
        "features": german_sample
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["dataset"] == "german"
    assert data["model"] == "random_forest"
    assert 0.0 <= data["baseline"]["default_probability"] <= 1.0
    assert 0.0 <= data["reweighed"]["default_probability"] <= 1.0
    assert "probability_difference" in data["delta"]
    assert "interpretation" in data["delta"]

def test_predict_taiwan_xgboost():
    cfg_resp = client.get("/api/configurations").json()
    taiwan_sample = cfg_resp["dataset_schemas"]["taiwan"]["sample_applicants"]["applicant_a_low_risk"]

    payload = {
        "dataset": "taiwan",
        "model": "xgboost",
        "features": taiwan_sample
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["dataset"] == "taiwan"
    assert data["model"] == "xgboost"
    assert 0.0 <= data["baseline"]["default_probability"] <= 1.0
    assert 0.0 <= data["reweighed"]["default_probability"] <= 1.0

def test_explain_german_random_forest():
    cfg_resp = client.get("/api/configurations").json()
    german_sample = cfg_resp["dataset_schemas"]["german"]["sample_applicants"]["applicant_a_low_risk"]

    payload = {
        "dataset": "german",
        "model": "random_forest",
        "features": german_sample
    }
    response = client.post("/api/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["dataset"] == "german"
    assert data["model"] == "random_forest"
    assert len(data["top_contributors"]) > 0
    first_feature = data["top_contributors"][0]
    assert "feature" in first_feature
    assert "baseline_shap" in first_feature
    assert "reweighed_shap" in first_feature

def test_research_matrix_endpoint():
    response = client.get("/api/research/matrix")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 6

def test_research_case_study_endpoint():
    response = client.get("/api/research/case-study")
    assert response.status_code == 200
    data = response.json()
    assert data["dataset"] == "german"
    assert data["model"] == "random_forest"
    assert data["metrics"]["eod_baseline"] == 0.125
    assert data["metrics"]["eod_reweighed"] == 0.050
    assert data["metrics"]["roc_auc_reweighed"] == 0.7935
    assert data["metrics"]["mean_shap_cosine_similarity"] == 0.9652

def test_research_individual_shap():
    response = client.get("/api/research/individual-shap?dataset=german&model=random_forest&applicant_idx=0")
    assert response.status_code == 200
    data = response.json()
    assert data["dataset"] == "german"
    assert data["model"] == "random_forest"
    assert len(data["attributions"]) > 0


def test_probability_representation_and_delta_math():
    cfg_resp = client.get("/api/configurations").json()
    german_sample = cfg_resp["dataset_schemas"]["german"]["sample_applicants"]["applicant_a_low_risk"]

    payload = {
        "dataset": "german",
        "model": "random_forest",
        "features": german_sample
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    p_base = data["baseline"]["default_probability"]
    p_rew = data["reweighed"]["default_probability"]
    
    assert 0.0 <= p_base <= 1.0
    assert 0.0 <= p_rew <= 1.0
    
    expected_diff = round(p_rew - p_base, 4)
    expected_pct_diff = round(expected_diff * 100, 2)
    
    assert abs(data["delta"]["probability_difference"] - expected_diff) < 1e-4
    assert abs(data["delta"]["percent_difference"] - expected_pct_diff) < 1e-2
    assert "output_space" in data["baseline"]
    assert len(data["baseline"]["output_space"]) > 0

def test_feature_label_formatting():
    from backend.app.services.feature_dictionary import format_feature_label
    
    assert format_feature_label("cat__checking_status_A14") == "Checking Account Status: No Checking Account (A14)"
    assert format_feature_label("cat__credit_history_A34") == "Credit Payment History: Critical Account / Other Credits (A34)"
    assert format_feature_label("num__duration_months") == "Loan Duration (Months)"
    assert format_feature_label("cat__pay_0_2") == "Repayment Sep: Payment Delay 2 Months (2)"
    assert format_feature_label(None) == "Unavailable"

def test_invalid_dataset_error_handling():
    payload = {
        "dataset": "invalid_dataset_name",
        "model": "random_forest",
        "features": {}
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code in (400, 422)

def test_invalid_model_error_handling():
    payload = {
        "dataset": "german",
        "model": "unsupported_model_arch",
        "features": {}
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code in (400, 422)

def test_individual_shap_feature_formatting():
    response = client.get("/api/research/individual-shap?dataset=german&model=random_forest&applicant_idx=0")
    assert response.status_code == 200
    data = response.json()
    assert len(data["attributions"]) > 0
    first_attr = data["attributions"][0]
    assert "feature" in first_attr
    assert "feature_raw" in first_attr
    assert not first_attr["feature"].startswith("cat__")
