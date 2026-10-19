from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class ConfigurationMetadata(BaseModel):
    dataset: str
    model: str
    baseline_model_file: str
    reweighed_model_file: str
    test_metrics: Dict[str, float]

class DatasetSchemaInfo(BaseModel):
    name: str
    title: str
    sample_size: int
    test_size: int
    default_rate: str
    protected_attribute: str
    features_in_x: List[str]
    categorical_features: List[str]
    numerical_features: List[str]
    categorical_unique_values: Dict[str, List[str]]
    numerical_ranges: Dict[str, Dict[str, float]]
    sample_applicants: Dict[str, Dict[str, Any]]

class ConfigurationsResponse(BaseModel):
    supported_datasets: List[str]
    supported_models: List[str]
    supported_mitigations: List[str]
    configurations: List[ConfigurationMetadata]
    dataset_schemas: Dict[str, DatasetSchemaInfo]
    disclaimer: str

class ResearchMatrixRow(BaseModel):
    dataset: str
    model: str
    mitigation_method: str
    baseline_acc: Optional[float] = None
    mitigated_acc: Optional[float] = None
    delta_acc: Optional[float] = None
    baseline_auc: Optional[float] = None
    mitigated_auc: Optional[str] = None
    delta_auc: Optional[str] = None
    baseline_eod: Optional[float] = None
    mitigated_eod: Optional[float] = None
    delta_eod: Optional[float] = None
    baseline_dpd: Optional[float] = None
    mitigated_dpd: Optional[float] = None
    delta_dpd: Optional[float] = None
    shap_cosine_sim: Optional[float] = None
    shap_top5_overlap: Optional[float] = None
    shap_spearman_rank: Optional[float] = None

class ShapSummaryRow(BaseModel):
    dataset: str
    model: str
    mean_cosine: float
    ci95_cosine: str
    mean_top5: float
    ci95_top5: str
    mean_rank: float
    ci95_rank: str

class IndividualShapQueryResponse(BaseModel):
    dataset: str
    model: str
    applicant_idx: int
    attributions: List[Dict[str, Any]]

class FinalResearchResultRow(BaseModel):
    Dataset: str
    Model: str
    Baseline_AUC: float
    Reweighed_AUC: float
    Delta_AUC: float
    Relative_AUC_Change_Percent: Optional[float] = None
    Baseline_EOD: float
    Reweighed_EOD: float
    Delta_EOD: float
    Relative_EOD_Change_Percent: Optional[float] = None
    Baseline_DPD: float
    Reweighed_DPD: float
    Delta_DPD: float
    Baseline_Approval_DI: Optional[float] = None
    Reweighed_Approval_DI: Optional[float] = None
    Mean_SHAP_Cosine: Optional[float] = None
    Median_SHAP_Cosine: Optional[float] = None
    Cosine_CI_Lower: Optional[float] = None
    Cosine_CI_Upper: Optional[float] = None
    Mean_Top5_Overlap: Optional[float] = None
    Median_Top5_Overlap: Optional[float] = None
    Mean_Spearman: Optional[float] = None
    Median_Spearman: Optional[float] = None
    Explanation_Sample_Size: Optional[int] = None
    Fairness_Direction: Optional[str] = None
    Performance_Direction: Optional[str] = None
    Explanation_Consistency_Level: Optional[str] = None
