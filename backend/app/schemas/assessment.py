from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field

DatasetType = Literal["taiwan", "german"]
ModelType = Literal["logistic_regression", "random_forest", "xgboost"]

class PredictionRequest(BaseModel):
    dataset: DatasetType = Field(..., description="Target dataset ('taiwan' or 'german')")
    model: ModelType = Field(..., description="Model architecture")
    features: Dict[str, Any] = Field(..., description="Applicant feature map matching dataset schema")

class PredictionModelOutput(BaseModel):
    prediction: int = Field(..., description="Binary classification (0=Non-default, 1=Default)")
    prediction_label: str = Field(..., description="Human-readable decision label")
    default_probability: float = Field(..., description="Estimated probability of default")
    model_type: str = Field(..., description="Model name")
    mitigation_type: str = Field(..., description="'baseline' or 'reweighed'")
    output_space: str = Field(..., description="Model prediction space")

class PredictionDelta(BaseModel):
    probability_difference: float = Field(..., description="reweighed_prob - baseline_prob")
    percent_difference: float = Field(..., description="Percentage point shift")
    direction: Literal["increased", "decreased", "unchanged"] = Field(...)
    interpretation: str = Field(..., description="Interpretation of reweighing impact on applicant")

class AssessmentResponse(BaseModel):
    dataset: str
    model: str
    baseline: PredictionModelOutput
    reweighed: PredictionModelOutput
    delta: PredictionDelta
    disclaimer: str

class ExplanationRequest(BaseModel):
    dataset: DatasetType
    model: ModelType
    features: Dict[str, Any]

class FeatureAttribution(BaseModel):
    feature: str = Field(..., description="Readable feature label")
    feature_raw: str = Field(..., description="Encoded feature name in transformer")
    baseline_shap: float = Field(..., description="Attribution under baseline model")
    reweighed_shap: float = Field(..., description="Attribution under reweighed model")
    diff_shap: float = Field(..., description="reweighed_shap - baseline_shap")
    direction: Literal["increases_risk", "reduces_risk", "neutral"]

class ExplanationResponse(BaseModel):
    dataset: str
    model: str
    output_space: str
    baseline_probability: float
    reweighed_probability: float
    top_contributors: List[FeatureAttribution]
    summary: str
    disclaimer: str
