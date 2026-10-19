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


def test_final_research_endpoint_all_six():
    response = client.get("/api/research/final-results")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 6
    
    first = data[0]
    required_keys = [
        "Dataset", "Model", "Baseline_AUC", "Reweighed_AUC", "Delta_AUC",
        "Baseline_EOD", "Reweighed_EOD", "Delta_EOD", "Baseline_DPD", "Reweighed_DPD",
        "Baseline_Approval_DI", "Reweighed_Approval_DI", "Mean_SHAP_Cosine",
        "Cosine_CI_Lower", "Cosine_CI_Upper", "Mean_Top5_Overlap", "Mean_Spearman",
        "Explanation_Sample_Size"
    ]
    for k in required_keys:
        assert k in first, f"Missing key {k} in final-results response"

def test_final_research_filtering():
    # Filter by dataset
    resp_taiwan = client.get("/api/research/final-results?dataset=taiwan")
    assert resp_taiwan.status_code == 200
    data_taiwan = resp_taiwan.json()
    assert len(data_taiwan) == 3
    assert all(r["Dataset"] == "taiwan" for r in data_taiwan)
    
    # Filter by model
    resp_rf = client.get("/api/research/final-results?model=random_forest")
    assert resp_rf.status_code == 200
    data_rf = resp_rf.json()
    assert len(data_rf) == 2
    assert all(r["Model"] == "random_forest" for r in data_rf)
    
    # Combined filter: german random_forest
    resp_german_rf = client.get("/api/research/final-results?dataset=german&model=random_forest")
    assert resp_german_rf.status_code == 200
    data_grf = resp_german_rf.json()
    assert len(data_grf) == 1
    row = data_grf[0]
    assert row["Dataset"] == "german"
    assert row["Model"] == "random_forest"
    assert row["Baseline_AUC"] == 0.7798
    assert row["Reweighed_AUC"] == 0.7935
    assert row["Baseline_EOD"] == 0.1250
    assert row["Reweighed_EOD"] == 0.0500
    assert row["Mean_SHAP_Cosine"] == 0.9652
    assert row["Explanation_Sample_Size"] == 200

def test_final_research_invalid_filters():
    # Invalid dataset
    resp_bad_ds = client.get("/api/research/final-results?dataset=nonexistent")
    assert resp_bad_ds.status_code == 400
    assert "Invalid dataset" in resp_bad_ds.json()["detail"]
    
    # Invalid model
    resp_bad_m = client.get("/api/research/final-results?model=neural_network")
    assert resp_bad_m.status_code == 400
    assert "Invalid model" in resp_bad_m.json()["detail"]


def test_feature_validation_invalid_bounds():
    cfg_resp = client.get("/api/configurations").json()
    german_sample = dict(cfg_resp["dataset_schemas"]["german"]["sample_applicants"]["applicant_a_low_risk"])

    # Negative credit amount
    bad_amount = dict(german_sample, credit_amount=-500)
    resp = client.post("/api/predict", json={"dataset": "german", "model": "random_forest", "features": bad_amount})
    assert resp.status_code == 400
    assert "strictly positive" in resp.json()["detail"]

    # Underage applicant (< 18)
    bad_age = dict(german_sample, age=12)
    resp = client.post("/api/predict", json={"dataset": "german", "model": "random_forest", "features": bad_age})
    assert resp.status_code == 400
    assert "between 18 and 120" in resp.json()["detail"]

    # Invalid categorical value
    bad_cat = dict(german_sample, checking_status="INVALID_CODE_99")
    resp = client.post("/api/predict", json={"dataset": "german", "model": "random_forest", "features": bad_cat})
    assert resp.status_code == 400
    assert "invalid" in resp.json()["detail"].lower()

def test_explanation_to_prediction_consistency():
    cfg_resp = client.get("/api/configurations").json()
    sample = cfg_resp["dataset_schemas"]["german"]["sample_applicants"]["applicant_a_low_risk"]

    payload = {"dataset": "german", "model": "random_forest", "features": sample}
    pred_resp = client.post("/api/predict", json=payload)
    exp_resp = client.post("/api/explain", json=payload)

    assert pred_resp.status_code == 200
    assert exp_resp.status_code == 200

    pred_data = pred_resp.json()
    exp_data = exp_resp.json()

    # Exact probability consistency between prediction and explanation
    assert abs(exp_data["baseline_probability"] - pred_data["baseline"]["default_probability"]) < 1e-4
    assert abs(exp_data["reweighed_probability"] - pred_data["reweighed"]["default_probability"]) < 1e-4

    # Explainer method metadata
    assert exp_data["explainer_method"] == "TreeExplainer"
    assert pred_data["baseline"]["is_calibrated"] is False
    assert "uncalibrated" in pred_data["baseline"]["score_meaning"].lower()

def test_explain_logistic_regression_linear_explainer():
    cfg_resp = client.get("/api/configurations").json()
    sample = cfg_resp["dataset_schemas"]["taiwan"]["sample_applicants"]["applicant_a_low_risk"]

    payload = {"dataset": "taiwan", "model": "logistic_regression", "features": sample}
    exp_resp = client.post("/api/explain", json=payload)
    assert exp_resp.status_code == 200
    exp_data = exp_resp.json()

    assert exp_data["explainer_method"] == "LinearExplainer"
    assert "Margin" in exp_data["output_space"] or "Log-Odds" in exp_data["output_space"]
    assert len(exp_data["top_contributors"]) > 0
