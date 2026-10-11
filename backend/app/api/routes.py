from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query
from backend.app.schemas.assessment import (
    PredictionRequest,
    AssessmentResponse,
    ExplanationRequest,
    ExplanationResponse
)
from backend.app.schemas.research import (
    ConfigurationsResponse,
    DatasetSchemaInfo,
    ConfigurationMetadata,
    IndividualShapQueryResponse
)
from backend.app.services.model_service import model_service
from backend.app.services.research_service import research_service

router = APIRouter()

@router.get("/health")
def get_health():
    return {
        "status": "healthy",
        "service": "FairCredit-AI ML Inference & Research API",
        "version": "1.0.0",
        "framework": "Next.js + FastAPI Research Architecture",
        "disclaimer": "Research prototype — Not for automated underwriting."
    }

@router.get("/configurations", response_model=ConfigurationsResponse)
def get_configurations():
    manifest = research_service.get_manifest()
    
    dataset_schemas = {
        "taiwan": DatasetSchemaInfo(
            name="taiwan",
            title="Taiwan Credit Card Default (2005)",
            sample_size=30000,
            test_size=6000,
            default_rate="22.12%",
            protected_attribute="sex (excluded from X)",
            features_in_x=manifest["datasets"]["taiwan"]["features_in_X"],
            categorical_features=manifest["datasets"]["taiwan"]["categorical_features"],
            numerical_features=manifest["datasets"]["taiwan"]["numerical_features"],
            categorical_unique_values=manifest["datasets"]["taiwan"]["categorical_unique_values"],
            numerical_ranges=manifest["datasets"]["taiwan"]["numerical_ranges"],
            sample_applicants=manifest["datasets"]["taiwan"]["sample_applicants"]
        ),
        "german": DatasetSchemaInfo(
            name="german",
            title="Statlog German Credit (1994)",
            sample_size=1000,
            test_size=200,
            default_rate="30.00%",
            protected_attribute="derived_sex (excluded from X)",
            features_in_x=manifest["datasets"]["german"]["features_in_X"],
            categorical_features=manifest["datasets"]["german"]["categorical_features"],
            numerical_features=manifest["datasets"]["german"]["numerical_features"],
            categorical_unique_values=manifest["datasets"]["german"]["categorical_unique_values"],
            numerical_ranges=manifest["datasets"]["german"]["numerical_ranges"],
            sample_applicants=manifest["datasets"]["german"]["sample_applicants"]
        )
    }

    config_list = [ConfigurationMetadata(**c) for c in manifest["configurations"]]

    return ConfigurationsResponse(
        supported_datasets=["taiwan", "german"],
        supported_models=["logistic_regression", "random_forest", "xgboost"],
        supported_mitigations=["reweighing", "exponentiated_gradient", "threshold_optimizer"],
        configurations=config_list,
        dataset_schemas=dataset_schemas,
        disclaimer="Sex attributes are strictly excluded from predictive features X in compliance with research protocol."
    )

@router.post("/predict", response_model=AssessmentResponse)
def assess_risk(req: PredictionRequest):
    try:
        return model_service.predict_applicant(req.dataset, req.model, req.features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/explain", response_model=ExplanationResponse)
def explain_risk(req: ExplanationRequest):
    try:
        return model_service.explain_applicant(req.dataset, req.model, req.features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/research/matrix")
def get_research_matrix():
    return research_service.get_research_matrix()

@router.get("/research/mitigations")
def get_mitigation_master():
    return research_service.get_mitigation_master()

@router.get("/research/final-results")
def get_final_results():
    return research_service.get_final_research_results()

@router.get("/research/shap-summary")
def get_shap_summary():
    return research_service.get_shap_summary()

@router.get("/research/shap-groups")
def get_shap_groups():
    return research_service.get_shap_group_metrics()

@router.get("/research/case-study")
def get_case_study():
    return research_service.get_case_study_data()

@router.get("/research/individual-shap", response_model=IndividualShapQueryResponse)
def get_individual_shap(
    dataset: str = Query(..., description="taiwan or german"),
    model: str = Query(..., description="model name"),
    applicant_idx: int = Query(0, description="applicant index")
):
    try:
        recs = research_service.get_individual_shap_records(dataset, model, applicant_idx)
        return IndividualShapQueryResponse(
            dataset=dataset,
            model=model,
            applicant_idx=applicant_idx,
            attributions=recs
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
