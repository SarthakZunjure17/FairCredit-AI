export type DatasetKey = "taiwan" | "german";
export type ModelKey = "logistic_regression" | "random_forest" | "xgboost";
export type MitigationKey = "reweighing" | "exponentiated_gradient" | "threshold_optimizer";

export interface PredictionModelOutput {
  prediction: number;
  prediction_label: string;
  default_probability: number;
  model_type: string;
  mitigation_type: string;
  output_space: string;
}

export interface PredictionDelta {
  probability_difference: number;
  percent_difference: number;
  direction: "increased" | "decreased" | "unchanged";
  interpretation: string;
}

export interface AssessmentResponse {
  dataset: string;
  model: string;
  baseline: PredictionModelOutput;
  reweighed: PredictionModelOutput;
  delta: PredictionDelta;
  disclaimer: string;
}

export interface FeatureAttribution {
  feature: string;
  feature_raw: string;
  baseline_shap: number;
  reweighed_shap: number;
  diff_shap: number;
  direction: "increases_risk" | "reduces_risk" | "neutral";
}

export interface ExplanationResponse {
  dataset: string;
  model: string;
  output_space: string;
  baseline_probability: number;
  reweighed_probability: number;
  top_contributors: FeatureAttribution[];
  summary: string;
  disclaimer: string;
}

export interface DatasetSchemaInfo {
  name: string;
  title: string;
  sample_size: number;
  test_size: number;
  default_rate: string;
  protected_attribute: string;
  features_in_x: string[];
  categorical_features: string[];
  numerical_features: string[];
  categorical_unique_values: Record<string, string[]>;
  numerical_ranges: Record<string, { min: number; max: number; median: number }>;
  sample_applicants: Record<string, Record<string, any>>;
}

export interface ConfigurationMetadata {
  dataset: string;
  model: string;
  baseline_model_file: string;
  reweighed_model_file: string;
  test_metrics: {
    baseline_auc: number;
    baseline_accuracy: number;
    reweighed_auc: number;
    reweighed_accuracy: number;
  };
}

export interface ConfigurationsResponse {
  supported_datasets: string[];
  supported_models: string[];
  supported_mitigations: string[];
  configurations: ConfigurationMetadata[];
  dataset_schemas: Record<string, DatasetSchemaInfo>;
  disclaimer: string;
}

export interface CaseStudyData {
  dataset: string;
  model: string;
  mitigation: string;
  metrics: {
    roc_auc_baseline: number;
    roc_auc_reweighed: number;
    roc_auc_delta: number;
    eod_baseline: number;
    eod_reweighed: number;
    eod_delta: number;
    distance_from_parity_reduction: string;
    mean_shap_cosine_similarity: number;
    mean_top5_overlap: number;
    mean_spearman_rank: number;
  };
  interpretation: string;
}

export interface ResearchMatrixRow {
  dataset: string;
  model: string;
  mitigation_method: string;
  baseline_acc?: number;
  mitigated_acc?: number;
  delta_acc?: number;
  baseline_auc?: number;
  mitigated_auc?: string;
  delta_auc?: string;
  baseline_eod?: number;
  mitigated_eod?: number;
  delta_eod?: number;
  baseline_dpd?: number;
  mitigated_dpd?: number;
  delta_dpd?: number;
  shap_cosine_sim?: number;
  shap_top5_overlap?: number;
  shap_spearman_rank?: number;
}

export interface ShapSummaryRow {
  dataset: string;
  model: string;
  mean_cosine: number;
  ci95_cosine: string;
  mean_top5: number;
  ci95_top5: string;
  mean_rank: number;
  ci95_rank: string;
}

export interface IndividualShapResponse {
  dataset: string;
  model: string;
  applicant_idx: number;
  attributions: Array<{
    feature: string;
    feature_raw?: string;
    baseline_shap: number;
    reweighed_shap: number;
    diff_shap: number;
  }>;
}

export interface FinalResearchResultRow {
  Dataset: string;
  Model: string;
  Baseline_AUC: number;
  Reweighed_AUC: number;
  Delta_AUC: number;
  Relative_AUC_Change_Percent?: number;
  Baseline_EOD: number;
  Reweighed_EOD: number;
  Delta_EOD: number;
  Relative_EOD_Change_Percent?: number;
  Baseline_DPD: number;
  Reweighed_DPD: number;
  Delta_DPD: number;
  Baseline_Approval_DI?: number;
  Reweighed_Approval_DI?: number;
  Mean_SHAP_Cosine?: number;
  Median_SHAP_Cosine?: number;
  Cosine_CI_Lower?: number;
  Cosine_CI_Upper?: number;
  Mean_Top5_Overlap?: number;
  Median_Top5_Overlap?: number;
  Mean_Spearman?: number;
  Median_Spearman?: number;
  Explanation_Sample_Size?: number;
  Fairness_Direction?: string;
  Performance_Direction?: string;
  Explanation_Consistency_Level?: string;
}
