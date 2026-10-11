"""
FairCredit-AI Dashboard Data Loader Module.
Loads frozen research artifacts programmatically from authoritative result files.
Never hardcodes numerical values.
Preserves N/A values without converting to zero.
"""

import os
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import streamlit as st


def get_project_root() -> str:
    """Resolve project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, ".."))


@st.cache_data
def load_mitigation_master_results() -> pd.DataFrame:
    """Load results/mitigation_master_results.csv with N/A preserved."""
    root = get_project_root()
    path = os.path.join(root, "results", "mitigation_master_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Authoritative mitigation master results not found at: {path}")
    
    # keep_default_na=False ensures 'N/A' strings are preserved and not turned into NaN
    df = pd.read_csv(path, keep_default_na=False)
    # Standardize column naming
    df["dataset"] = df["dataset"].str.lower().str.strip()
    df["model"] = df["model"].str.lower().str.strip()
    df["mitigation_method"] = df["mitigation_method"].str.lower().str.strip()
    return df


@st.cache_data
def load_final_research_results() -> pd.DataFrame:
    """Load results/final_research_results.csv containing the 6 frozen Reweighing rows."""
    root = get_project_root()
    path = os.path.join(root, "results", "final_research_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Final research results not found at: {path}")
    
    df = pd.read_csv(path)
    df["Dataset"] = df["Dataset"].str.lower().str.strip()
    df["Model"] = df["Model"].str.lower().str.strip()
    return df


@st.cache_data
def load_shap_summary() -> pd.DataFrame:
    """Load results/shap_summary.csv containing explanation consistency distributions."""
    root = get_project_root()
    path = os.path.join(root, "results", "shap_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"SHAP summary not found at: {path}")
    
    df = pd.read_csv(path)
    df["dataset"] = df["dataset"].str.lower().str.strip()
    df["model"] = df["model"].str.lower().str.strip()
    return df


@st.cache_data
def load_shap_group_metrics() -> pd.DataFrame:
    """Load results/shap_group_metrics.csv containing subgroup explanation consistency."""
    root = get_project_root()
    path = os.path.join(root, "results", "shap_group_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"SHAP group metrics not found at: {path}")
    
    df = pd.read_csv(path)
    df["dataset"] = df["dataset"].str.lower().str.strip()
    df["model"] = df["model"].str.lower().str.strip()
    return df


@st.cache_data
def load_shap_individual_metrics() -> pd.DataFrame:
    """Load results/shap_individual_metrics.csv containing 2,100 applicant consistency records."""
    root = get_project_root()
    path = os.path.join(root, "results", "shap_individual_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"SHAP individual metrics not found at: {path}")
    
    df = pd.read_csv(path)
    df["dataset"] = df["dataset"].str.lower().str.strip()
    df["model"] = df["model"].str.lower().str.strip()
    return df


@st.cache_data
def load_research_matrix() -> pd.DataFrame:
    """Load results/fairness_performance_explanation_matrix.csv."""
    root = get_project_root()
    path = os.path.join(root, "results", "fairness_performance_explanation_matrix.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fairness-performance-explanation matrix not found at: {path}")
    
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_individual_feature_shap_data() -> pd.DataFrame:
    """Load pre-saved frozen individual SHAP feature attributions."""
    root = get_project_root()
    path = os.path.join(root, "dashboard", "data", "individual_feature_shap.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pre-saved individual SHAP feature data not found at: {path}")
    
    df = pd.read_parquet(path)
    df["dataset"] = df["dataset"].str.lower().str.strip()
    df["model"] = df["model"].str.lower().str.strip()
    return df


def get_applicant_feature_attributions(
    dataset: str,
    model: str,
    applicant_idx: int
) -> pd.DataFrame:
    """
    Retrieve feature-level SHAP attributions for an individual applicant.
    Returns DataFrame with columns: [feature, baseline_shap, reweighed_shap, diff_shap].
    """
    df_all = load_individual_feature_shap_data()
    ds = dataset.lower().strip()
    m = model.lower().strip()
    
    sub = df_all[
        (df_all["dataset"] == ds) &
        (df_all["model"] == m) &
        (df_all["applicant_idx"] == applicant_idx)
    ].copy()
    
    if sub.empty:
        raise ValueError(f"No SHAP feature attributions found for {ds} - {m} - applicant {applicant_idx}")
    
    return sub[["feature", "baseline_shap", "reweighed_shap", "diff_shap"]]


def get_all_dashboard_data() -> Dict[str, pd.DataFrame]:
    """Helper to load all datasets and return in a dictionary."""
    return {
        "mitigation_master": load_mitigation_master_results(),
        "final_research": load_final_research_results(),
        "shap_summary": load_shap_summary(),
        "shap_group": load_shap_group_metrics(),
        "shap_individual": load_shap_individual_metrics(),
        "matrix": load_research_matrix(),
    }
