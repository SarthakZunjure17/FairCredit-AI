from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class ResearchReportRequest(BaseModel):
    dataset: str = Field("german", description="Dataset identifier: 'taiwan' or 'german'")
    model: str = Field("random_forest", description="Model architecture: 'logistic_regression', 'random_forest', or 'xgboost'")
    mitigation: str = Field("reweighing", description="Mitigation method: 'reweighing', 'exponentiated_gradient', or 'threshold_optimizer'")
    protected_attribute: str = Field("sex", description="Protected attribute for subgroup audit: 'sex' or 'age'")

class ApplicantReportRequest(BaseModel):
    dataset: str = Field(..., description="Dataset identifier: 'taiwan' or 'german'")
    model: str = Field(..., description="Model architecture: 'logistic_regression', 'random_forest', or 'xgboost'")
    baseline: Dict[str, Any] = Field(..., description="Baseline prediction output (prediction, probability, score meaning)")
    reweighed: Dict[str, Any] = Field(..., description="Reweighed prediction output (prediction, probability, score meaning)")
    delta: Dict[str, Any] = Field(..., description="Prediction difference and interpretation")
    top_contributors: List[Dict[str, Any]] = Field(..., description="Local SHAP top feature contributions")
    output_space: str = Field("Probability Space P(Y=1)", description="Mathematical space of SHAP values")
    explainer_method: Optional[str] = Field("TreeExplainer", description="SHAP explainer algorithm")
    disclaimer: Optional[str] = Field(None, description="Research disclaimer text")
