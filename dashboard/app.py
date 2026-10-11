"""
FairCredit-AI Research Dashboard
A Responsible AI Credit Model Auditing Framework
Evaluating Fairness, Performance, and Explanation Consistency in Credit Scoring Models

RESEARCH STATUS: LOCKED & FROZEN
Visualization and Model Auditing Decision-Support Prototype.
"""

import os
import sys
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import streamlit as st

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dashboard.data_loader import (
    load_mitigation_master_results,
    load_final_research_results,
    load_shap_summary,
    load_shap_group_metrics,
    load_shap_individual_metrics,
    load_research_matrix,
    get_applicant_feature_attributions
)
from dashboard.components import (
    render_header_disclaimer,
    render_source_note,
    render_metric_card,
    plot_fairness_before_after,
    plot_explanation_consistency_summary,
    plot_individual_shap_horizontal_bars,
    PALETTE
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="FairCredit-AI Research Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, accessible, model-risk analyst theme
st.markdown(
    """
    <style>
    /* Clean font and layout */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    h1, h2, h3, h4 {
        color: #0F172A;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        background-color: #F8FAFC;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 8px;
        padding-bottom: 8px;
        font-weight: 600;
        font-size: 14px;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #2563EB !important;
        border-bottom: 3px solid #2563EB !important;
    }
    .audit-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .metric-subtext {
        font-size: 12px;
        color: #64748B;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main Application Header
render_header_disclaimer()

st.title("FairCredit-AI: Research Dashboard")
st.markdown(
    "**Evaluating Fairness, Performance, and Explanation Consistency in Credit Scoring Models**  \n"
    "*Research Prototype — Model Auditing & Responsible AI Decision Support*"
)

# Sidebar: Global Navigation & Research Scope
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/google/material-design-icons/master/png/action/account_balance/materialicons/48dp/2x/baseline_account_balance_black_48dp.png",
        width=48
    )
    st.markdown("### ⚖️ FairCredit-AI")
    st.markdown("**Research Status:** `LOCKED & FROZEN`")
    st.markdown(
        """
        ---
        **Scope of Empirical Evaluation:**
        - **Datasets (2):** Taiwan Credit Card, Statlog German
        - **Models (3):** Logistic Regression, Random Forest, XGBoost
        - **Mitigations (3):** Reweighing, Exponentiated Gradient, ThresholdOptimizer
        - **SHAP Cohort:** 6 Primary Baseline vs Reweighing pairs (N=2,100 applicant explanations)
        ---
        """
    )
    st.markdown("#### Regulatory Note")
    st.caption(
        "Designed for bank model-risk management, fair lending auditing, and algorithmic accountability. "
        "Strictly decision-support; does NOT render credit determinations."
    )

# Load data safely
try:
    df_mitigation = load_mitigation_master_results()
    df_final = load_final_research_results()
    df_shap_summary = load_shap_summary()
    df_shap_group = load_shap_group_metrics()
    df_shap_individual = load_shap_individual_metrics()
    df_matrix = load_research_matrix()
except Exception as e:
    st.error(f"Critical error loading frozen research data: {e}")
    st.stop()

# Multi-tab layout
tabs = st.tabs([
    "1. Overview",
    "2. Model Audit",
    "3. Fairness Analysis",
    "4. Explanation Consistency",
    "5. Individual Explanation",
    "6. Research Findings",
    "7. Methodology & Limitations"
])

# ==============================================================================
# TAB 1: OVERVIEW
# ==============================================================================
with tabs[0]:
    st.subheader("Framework Overview & Research Scope")
    st.markdown(
        """
        **FairCredit-AI** is a systematic model-auditing framework designed to evaluate 
        machine-learning credit-scoring models across four interconnected dimensions:
        
        1. **Predictive Performance:** Discrimination accuracy, rank ordering, and classification utility.
        2. **Group Fairness:** Disparities across legally protected demographic groups (Equalized Odds, Equal Opportunity, Demographic Parity, Disparate Impact).
        3. **Explainability:** Feature attribution profiles toward positive default risk prediction.
        4. **Explanation Consistency:** Stability and rank-preservation of individual explanations when fairness interventions alter the underlying scoring model.
        """
    )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Canonical Datasets", value="2", help="Taiwan Default (N=30,000) & Statlog German Credit (N=1,000)")
    with col2:
        st.metric(label="Model Architectures", value="3", help="Logistic Regression, Random Forest, XGBoost")
    with col3:
        st.metric(label="Fairness Mitigations", value="3", help="Reweighing (Pre), Exponentiated Gradient (In), ThresholdOptimizer (Post)")
    with col4:
        st.metric(label="Primary SHAP Pairs", value="6", help="Baseline vs Reweighing (500 Taiwan + 200 German applicants)")
        
    st.markdown("---")
    st.markdown("### Experimental Matrix & Evaluation Protocol")
    
    overview_col1, overview_col2 = st.columns(2)
    with overview_col1:
        st.markdown(
            """
            #### 📊 Evaluated Datasets & Sensitive Attributes
            - **Taiwan Credit Card Default (2005):**
              - *Sample size:* 30,000 borrowers (6,000 test set).
              - *Target:* Default payment next month (22.1% prevalence).
              - *Protected attribute:* Sex (female: 60.4%, male: 39.6%).
              - *Feature matrix:* 23 predictive financial features; protected sex is strictly excluded from feature matrix $X$.
            - **Statlog German Credit (1994):**
              - *Sample size:* 1,000 credit cases (200 test set).
              - *Target:* Credit risk default (30.0% prevalence).
              - *Protected attribute:* Derived Sex from personal status code (female: 31.0%, male: 69.0%).
              - *Feature matrix:* 20 predictive features; sensitive sex strictly excluded from $X$.
            """
        )
    with overview_col2:
        st.markdown(
            """
            #### ⚙️ Fairness Mitigation Interventions
            - **Reweighing (Pre-processing):**
              - Modifies training sample weights to decorrelate protected status and default outcome before training.
              - **Primary focus of SHAP analysis** because it produces a genuinely refitted continuous predictive scoring function.
            - **Exponentiated Gradient (In-processing):**
              - Solves a constrained min-max game yielding a randomized ensemble of predictors subject to Equalized Odds constraints.
            - **ThresholdOptimizer (Post-processing):**
              - Solves group-specific classification thresholds on baseline continuous scores to equalize TPR and FPR post-hoc.
            """
        )
        
    st.info(
        "💡 **Key Methodological Note:** The primary explanation-consistency experiment compares **Baseline models with Reweighing**. "
        "ThresholdOptimizer is a post-processing decision rule that does not alter the underlying scoring function, "
        "and ExponentiatedGradient produces a randomized predictor ensemble without a straightforward continuous scoring function "
        "under the locked evaluation protocol."
    )
    render_source_note("results/final_research_results.csv")


# ==============================================================================
# TAB 2: MODEL AUDIT
# ==============================================================================
with tabs[1]:
    st.subheader("Model Predictive Performance Audit")
    st.markdown(
        "Inspect predictive metrics across baseline and mitigated credit scoring models. "
        "Select the dataset, model architecture, and mitigation method to view before/after changes."
    )
    
    audit_col1, audit_col2, audit_col3 = st.columns(3)
    with audit_col1:
        sel_dataset_audit = st.selectbox(
            "Select Dataset:",
            options=["taiwan", "german"],
            format_func=lambda x: "Taiwan Credit Card Default" if x == "taiwan" else "Statlog German Credit",
            key="audit_dataset"
        )
    with audit_col2:
        sel_model_audit = st.selectbox(
            "Select Model Architecture:",
            options=["logistic_regression", "random_forest", "xgboost"],
            format_func=lambda x: x.replace("_", " ").title(),
            key="audit_model"
        )
    with audit_col3:
        sel_mitigation_audit = st.selectbox(
            "Select Mitigation Method:",
            options=["reweighing", "exponentiated_gradient", "threshold_optimizer"],
            format_func=lambda x: {
                "reweighing": "Reweighing (Pre-processing)",
                "exponentiated_gradient": "Exponentiated Gradient (In-processing)",
                "threshold_optimizer": "ThresholdOptimizer (Post-processing)"
            }[x],
            key="audit_mitigation"
        )
        
    # Filter row from mitigation master results
    match_row = df_mitigation[
        (df_mitigation["dataset"] == sel_dataset_audit) &
        (df_mitigation["model"] == sel_model_audit) &
        (df_mitigation["mitigation_method"] == sel_mitigation_audit)
    ]
    
    if match_row.empty:
        st.warning("No records found for the selected configuration.")
    else:
        row = match_row.iloc[0]
        st.markdown(
            f"**Audited Configuration:** `{sel_dataset_audit.title()}` | "
            f"`{sel_model_audit.replace('_', ' ').title()}` | "
            f"`{sel_mitigation_audit.replace('_', ' ').title()}`"
        )
        
        # Display 6 metric cards
        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric_card("Accuracy", row["baseline_accuracy"], row["mitigated_accuracy"], row["delta_accuracy"], higher_is_better=True)
            render_metric_card("Precision (Default Y=1)", row["baseline_precision"], row["mitigated_precision"], row["delta_precision"], higher_is_better=True)
        with c2:
            render_metric_card("Balanced Accuracy", row["baseline_balanced_accuracy"], row["mitigated_balanced_accuracy"], row["delta_balanced_accuracy"], higher_is_better=True)
            render_metric_card("Recall (Default Y=1)", row["baseline_recall"], row["mitigated_recall"], row["delta_recall"], higher_is_better=True)
        with c3:
            # ROC-AUC with explicit N/A handling
            is_na = sel_mitigation_audit in ["exponentiated_gradient", "threshold_optimizer"]
            render_metric_card(
                "ROC-AUC",
                row["baseline_roc_auc"],
                row["mitigated_roc_auc"],
                row["delta_roc_auc"],
                higher_is_better=True,
                is_na_metric=is_na,
                na_explanation="N/A — continuous ranking score not available under the locked evaluation protocol."
            )
            render_metric_card("F1-Score", row["baseline_f1"], row["mitigated_f1"], row["delta_f1"], higher_is_better=True)

        if is_na:
            st.info(
                "ℹ️ **Evaluation Protocol Notice on ROC-AUC:** "
                f"For **{sel_mitigation_audit.replace('_', ' ').title()}**, mitigated ROC-AUC is reported strictly as **N/A** "
                "because this mitigation outputs binary/step classifications rather than a continuous score suitable for unthresholded ranking evaluation. "
                "Balanced Accuracy reflects the audited performance under operational decision thresholds."
            )

    render_source_note("results/mitigation_master_results.csv")


# ==============================================================================
# TAB 3: FAIRNESS ANALYSIS
# ==============================================================================
with tabs[2]:
    st.subheader("Group Fairness & Disparity Analysis")
    st.markdown(
        """
        Evaluate group fairness metrics across protected demographic groups (Sex).  
        > **Critical Audit Principle:** *Fairness is multi-dimensional. Improvement in one fairness criterion does not guarantee improvement in another.*
        """
    )
    
    fair_col1, fair_col2, fair_col3 = st.columns(3)
    with fair_col1:
        sel_dataset_fair = st.selectbox(
            "Select Dataset:",
            options=["taiwan", "german"],
            format_func=lambda x: "Taiwan Credit Card Default" if x == "taiwan" else "Statlog German Credit",
            key="fair_dataset"
        )
    with fair_col2:
        sel_model_fair = st.selectbox(
            "Select Model Architecture:",
            options=["logistic_regression", "random_forest", "xgboost"],
            format_func=lambda x: x.replace("_", " ").title(),
            key="fair_model"
        )
    with fair_col3:
        sel_mitigation_fair = st.selectbox(
            "Select Mitigation Method:",
            options=["reweighing", "exponentiated_gradient", "threshold_optimizer"],
            format_func=lambda x: {
                "reweighing": "Reweighing (Pre-processing)",
                "exponentiated_gradient": "Exponentiated Gradient (In-processing)",
                "threshold_optimizer": "ThresholdOptimizer (Post-processing)"
            }[x],
            key="fair_mitigation"
        )
        
    fair_match = df_mitigation[
        (df_mitigation["dataset"] == sel_dataset_fair) &
        (df_mitigation["model"] == sel_model_fair) &
        (df_mitigation["mitigation_method"] == sel_mitigation_fair)
    ]
    
    if fair_match.empty:
        st.warning("No records found for the selected configuration.")
    else:
        f_row = fair_match.iloc[0]
        
        st.markdown("#### Audited Fairness Metrics")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            render_metric_card(
                "Equalized Odds Diff (EOD)",
                f_row["baseline_eod"],
                f_row["mitigated_eod"],
                f_row["delta_eod"],
                higher_is_better=False
            )
            st.caption("Interpretation: max(|ΔTPR|, |ΔFPR|). Lower absolute value = closer to equalized odds.")
            
            render_metric_card(
                "Equal Opportunity Diff (EOpp)",
                f_row["baseline_eopp"],
                f_row["mitigated_eopp"],
                f_row["delta_eopp"],
                higher_is_better=False
            )
            st.caption("Interpretation: |TPR_unprivileged - TPR_privileged|. Closer to zero = closer to equality of recall.")

        with fc2:
            render_metric_card(
                "Demographic Parity Diff (DPD)",
                f_row["baseline_dpd"],
                f_row["mitigated_dpd"],
                f_row["delta_dpd"],
                higher_is_better=False
            )
            st.caption("Interpretation: |SelectionRate_u - SelectionRate_p|. Closer to zero = closer to demographic parity.")
            
            render_metric_card(
                "Distance from Parity (|DI - 1|)",
                f_row["baseline_approval_di_distance"],
                f_row["mitigated_approval_di_distance"],
                round(float(f_row["mitigated_approval_di_distance"]) - float(f_row["baseline_approval_di_distance"]), 4),
                higher_is_better=False
            )
            st.caption("Interpretation: Absolute gap from ideal 1.0 parity ratio. Closer to zero = closer to parity.")

        with fc3:
            render_metric_card(
                "Approval Disparate Impact (DI)",
                f_row["baseline_approval_di"],
                f_row["mitigated_approval_di"],
                round(float(f_row["mitigated_approval_di"]) - float(f_row["baseline_approval_di"]), 4),
                higher_is_better=None
            )
            st.caption("Interpretation: ApprovalRate_u / ApprovalRate_p. Reference line is 1.0 (exact parity).")

        st.markdown("---")
        st.markdown("#### Visual Before vs After Fairness Comparisons")
        
        chart_col1, chart_col2, chart_col3 = st.columns(3)
        with chart_col1:
            st.markdown("**Equalized Odds Difference (EOD)**")
            st.altair_chart(
                plot_fairness_before_after("EOD", float(f_row["baseline_eod"]), float(f_row["mitigated_eod"])),
                use_container_width=True
            )
        with chart_col2:
            st.markdown("**Demographic Parity Difference (DPD)**")
            st.altair_chart(
                plot_fairness_before_after("DPD", float(f_row["baseline_dpd"]), float(f_row["mitigated_dpd"])),
                use_container_width=True
            )
        with chart_col3:
            st.markdown("**Approval Disparate Impact (Reference = 1.0)**")
            st.altair_chart(
                plot_fairness_before_after("Approval DI", float(f_row["baseline_approval_di"]), float(f_row["mitigated_approval_di"]), reference_val=1.0),
                use_container_width=True
            )

    render_source_note("results/mitigation_master_results.csv")


# ==============================================================================
# TAB 4: EXPLANATION CONSISTENCY
# ==============================================================================
with tabs[3]:
    st.subheader("Explanation Consistency Audit (Baseline vs Reweighing)")
    st.markdown(
        """
        When fairness mitigation refits model parameters, does the model explain loan decisions similarly for the same applicants?
        We evaluate three rigorous consistency metrics across identical test cohorts:
        - **Cosine Similarity:** Directional alignment of feature attribution vectors in multidimensional space.
        - **Top-5 Feature Overlap (Jaccard-style):** Proportion of top-5 most influential features shared between baseline and reweighted explanations.
        - **Spearman Rank Correlation:** Monotonic rank preservation of all feature attributions.
        """
    )
    
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        sel_ds_exp = st.selectbox(
            "Select Evaluation Dataset:",
            options=["taiwan", "german"],
            format_func=lambda x: "Taiwan Credit Card Default (N=500 applicants)" if x == "taiwan" else "Statlog German Credit (N=200 applicants)",
            key="exp_dataset"
        )
    with exp_col2:
        sel_model_exp = st.selectbox(
            "Select Model Architecture:",
            options=["logistic_regression", "random_forest", "xgboost"],
            format_func=lambda x: x.replace("_", " ").title(),
            key="exp_model"
        )
        
    sub_sum = df_shap_summary[
        (df_shap_summary["dataset"] == sel_ds_exp) &
        (df_shap_summary["model"] == sel_model_exp)
    ]
    
    if sub_sum.empty:
        st.warning("No summary records found for this configuration.")
    else:
        s_row = sub_sum.iloc[0]
        n_eval = int(s_row["n_applicants"])
        n_feats = int(s_row["n_features"])
        
        st.markdown(
            f"**Audited Cohort:** `{sel_ds_exp.title()}` | `{sel_model_exp.replace('_', ' ').title()}` | "
            f"**Cohort Size:** `N={n_eval}` | **Feature Dimensionality:** `{n_feats}`"
        )
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric(
                label="Mean SHAP Cosine Similarity",
                value=f"{float(s_row['cosine_mean']):.4f}",
                delta=f"95% CI: [{float(s_row['cosine_ci_95_low']):.4f}, {float(s_row['cosine_ci_95_high']):.4f}]",
                delta_color="off"
            )
            st.caption(
                "High cosine similarity indicates that attribution direction remained similar "
                "between the baseline and reweighted models for the evaluated applicants."
            )
        with m_col2:
            st.metric(
                label="Mean Top-5 Feature Overlap",
                value=f"{float(s_row['top5_overlap_mean']):.4f}",
                delta=f"95% CI: [{float(s_row['top5_overlap_ci_95_low']):.4f}, {float(s_row['top5_overlap_ci_95_high']):.4f}]",
                delta_color="off"
            )
            st.caption("Proportion of top-5 highest-impact decision drivers preserved between models.")
        with m_col3:
            st.metric(
                label="Mean Spearman Rank Correlation",
                value=f"{float(s_row['rank_corr_mean']):.4f}",
                delta=f"95% CI: [{float(s_row['rank_corr_ci_95_low']):.4f}, {float(s_row['rank_corr_ci_95_high']):.4f}]",
                delta_color="off"
            )
            st.caption("Global feature importance rank order preservation across all features.")

    st.markdown("---")
    st.markdown("### Cross-Model Architectural Pattern")
    st.markdown(
        "A clear architectural hierarchy emerges across the six evaluated baseline/reweighing pairs (*observed in the evaluated configurations*):  \n"
        "- **Logistic Regression:** Near-perfect consistency (Cosine > 0.998, Rank > 0.983) due to convex objective and linear parameter stability under reweighting.  \n"
        "- **XGBoost:** High consistency (Cosine 0.975 – 0.979, Rank 0.887 – 0.917) due to gradient boosting's regularized tree growth.  \n"
        "- **Random Forest:** High directional consistency (Cosine 0.938 – 0.965), but notably greater feature-ranking variation (Top-5 overlap drops to 0.792 – 0.837, Rank drops to 0.776 – 0.889) due to bagging and randomized split selections."
    )
    
    # Altair chart across all 6 pairs
    st.altair_chart(plot_explanation_consistency_summary(df_shap_summary), use_container_width=True)
    
    st.markdown("#### Complete Consistency Metrics Table Across Evaluated Configurations")
    display_summary_df = df_shap_summary[[
        "dataset", "model", "n_applicants",
        "cosine_mean", "cosine_ci_95_low", "cosine_ci_95_high",
        "top5_overlap_mean", "top5_overlap_ci_95_low", "top5_overlap_ci_95_high",
        "rank_corr_mean", "rank_corr_ci_95_low", "rank_corr_ci_95_high"
    ]].copy()
    display_summary_df.columns = [
        "Dataset", "Model", "N",
        "Cosine Mean", "Cosine 95% Low", "Cosine 95% High",
        "Top-5 Mean", "Top-5 95% Low", "Top-5 95% High",
        "Spearman Mean", "Spearman 95% Low", "Spearman 95% High"
    ]
    st.dataframe(display_summary_df.style.format({
        "Cosine Mean": "{:.4f}", "Cosine 95% Low": "{:.4f}", "Cosine 95% High": "{:.4f}",
        "Top-5 Mean": "{:.4f}", "Top-5 95% Low": "{:.4f}", "Top-5 95% High": "{:.4f}",
        "Spearman Mean": "{:.4f}", "Spearman 95% Low": "{:.4f}", "Spearman 95% High": "{:.4f}"
    }), use_container_width=True)

    render_source_note("results/shap_summary.csv")


# ==============================================================================
# TAB 5: INDIVIDUAL EXPLANATION
# ==============================================================================
with tabs[4]:
    st.subheader("Individual Applicant SHAP Explanation Audit")
    st.markdown(
        """
        Inspect feature-level SHAP attributions for an individual borrower comparing the **Baseline model** 
        against the **Reweighted model**.
        
        > **Methodological Boundary:**  
        > - **Logistic Regression:** attributions computed in **margin / log-odds space**.  
        > - **Random Forest:** attributions computed in **class-1 probability space** $P(Y=1|X)$.  
        > - **XGBoost:** attributions computed in **margin / log-odds space**.  
        > *Do NOT compare raw SHAP magnitudes between different model architectures. Comparisons are valid strictly for the same applicant, same architecture: Baseline vs Reweighted.*
        """
    )
    
    ind_c1, ind_c2, ind_c3 = st.columns(3)
    with ind_c1:
        sel_ds_ind = st.selectbox(
            "Dataset:",
            options=["taiwan", "german"],
            format_func=lambda x: "Taiwan Credit Card Default" if x == "taiwan" else "Statlog German Credit",
            key="ind_dataset"
        )
    with ind_c2:
        sel_model_ind = st.selectbox(
            "Model Architecture:",
            options=["logistic_regression", "random_forest", "xgboost"],
            format_func=lambda x: x.replace("_", " ").title(),
            key="ind_model"
        )
        
    # Filter available applicants strictly from shap_individual_metrics.csv
    valid_applicants_df = df_shap_individual[
        (df_shap_individual["dataset"] == sel_ds_ind) &
        (df_shap_individual["model"] == sel_model_ind)
    ]
    applicant_list = sorted(valid_applicants_df["applicant_idx"].unique().tolist())
    
    with ind_c3:
        sel_app_idx = st.selectbox(
            "Applicant Index (Test Cohort):",
            options=applicant_list,
            key="ind_applicant"
        )
        
    # Get applicant metrics
    app_meta = valid_applicants_df[valid_applicants_df["applicant_idx"] == sel_app_idx].iloc[0]
    
    st.markdown(
        f"**Selected Applicant Profile:** ID `{sel_app_idx}` | Protected Sex: `{str(app_meta['sex']).title()}` | "
        f"Dataset: `{sel_ds_ind.title()}` | Architecture: `{sel_model_ind.replace('_', ' ').title()}`"
    )
    
    # Metrics cards for this applicant
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        st.metric(label="Individual Cosine Similarity", value=f"{float(app_meta['cosine_similarity']):.4f}")
    with ic2:
        st.metric(label="Top-5 Feature Overlap", value=f"{float(app_meta['top5_overlap']):.4f}")
    with ic3:
        st.metric(label="Spearman Rank Correlation", value=f"{float(app_meta['rank_correlation']):.4f}")
        
    output_space_map = {
        "logistic_regression": "Margin / Log-Odds Space",
        "random_forest": "Probability Space P(Y=1|X)",
        "xgboost": "Margin / Log-Odds Space"
    }
    output_space_desc = output_space_map.get(sel_model_ind, "Prediction Space")
    st.info(f"📐 **Current Prediction Explanation Space:** **{output_space_desc}**")
    
    # Load feature attributions for this applicant
    try:
        df_feat_app = get_applicant_feature_attributions(sel_ds_ind, sel_model_ind, sel_app_idx)
        
        # Display horizontal bar chart
        fig_ind = plot_individual_shap_horizontal_bars(
            df_feat_app,
            output_space_label=output_space_desc,
            top_n=10
        )
        st.pyplot(fig_ind)
        plt.close(fig_ind)
        
        st.markdown("#### Feature Attribution Table: Top Decision Contributors")
        df_display_feats = df_feat_app.copy()
        df_display_feats["abs_baseline"] = df_display_feats["baseline_shap"].abs()
        df_display_feats = df_display_feats.sort_values(by="abs_baseline", ascending=False).head(15)
        df_display_feats["feature"] = df_display_feats["feature"].str.replace("cat__", "").str.replace("num__", "")
        
        st.dataframe(
            df_display_feats[["feature", "baseline_shap", "reweighed_shap", "diff_shap"]].style.format({
                "baseline_shap": "{:+.4f}",
                "reweighed_shap": "{:+.4f}",
                "diff_shap": "{:+.4f}"
            }),
            use_container_width=True
        )
    except Exception as err:
        st.error(f"Error loading applicant feature attributions: {err}")
        
    render_source_note("results/shap_individual_metrics.csv & dashboard/data/individual_feature_shap.parquet")


# ==============================================================================
# TAB 6: RESEARCH FINDINGS
# ==============================================================================
with tabs[5]:
    st.subheader("Synthesized Research Findings & Trade-Off Matrix")
    
    st.markdown(
        """
        ### Central Empirical Finding
        > **“Fairness mitigation through Reweighing exhibited heterogeneous effects on explanation consistency across dataset-model configurations rather than a universal fairness–explanation trade-off.”**
        
        Fairness mitigation does not universally degrade explanation stability, nor does it guarantee simultaneous gains. 
        Instead, empirical outcomes depend heavily on the model's inductive biases and dataset characteristics.
        """
    )
    
    st.markdown("### Key Observed Configurations (Case Studies)")
    
    case1, case2, case3 = st.columns(3)
    with case1:
        st.markdown(
            """
            <div class="audit-card">
                <div style="font-weight: 700; color: #16A34A; font-size: 15px;">Case 1: German Random Forest</div>
                <div style="font-size: 12px; color: #64748B; margin-bottom: 8px;">Simultaneous Multi-Objective Win</div>
                <ul style="font-size: 13px; padding-left: 18px;">
                    <li><strong>ROC-AUC:</strong> 0.7798 → 0.7935 (Δ +0.0137)</li>
                    <li><strong>EOD:</strong> 0.1250 → 0.0500 (Δ -0.0750)</li>
                    <li><strong>Cosine:</strong> 0.9652 (95% CI: 0.9624–0.9677)</li>
                </ul>
                <div style="font-size: 12px; background: #F0FDF4; border: 1px solid #BBF7D0; padding: 8px; border-radius: 4px;">
                    <strong>Interpretation:</strong> In this evaluated configuration, fairness and predictive performance improved while explanation consistency remained high.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with case2:
        st.markdown(
            """
            <div class="audit-card">
                <div style="font-weight: 700; color: #2563EB; font-size: 15px;">Case 2: Taiwan Random Forest</div>
                <div style="font-size: 12px; color: #64748B; margin-bottom: 8px;">Fairness Gain with Rank Drift</div>
                <ul style="font-size: 13px; padding-left: 18px;">
                    <li><strong>ROC-AUC:</strong> 0.7528 → 0.7548 (Δ +0.0020)</li>
                    <li><strong>EOD:</strong> 0.0298 → 0.0219 (Δ -0.0079)</li>
                    <li><strong>Cosine:</strong> 0.9380 | <strong>Top-5:</strong> 0.7924</li>
                </ul>
                <div style="font-size: 12px; background: #EFF6FF; border: 1px solid #BFDBFE; padding: 8px; border-radius: 4px;">
                    <strong>Interpretation:</strong> Fairness and AUC improved, while feature attribution rankings showed greater variation.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with case3:
        st.markdown(
            """
            <div class="audit-card">
                <div style="font-weight: 700; color: #D97706; font-size: 15px;">Case 3: Taiwan XGBoost</div>
                <div style="font-size: 12px; color: #64748B; margin-bottom: 8px;">Fairness Metric Divergence</div>
                <ul style="font-size: 13px; padding-left: 18px;">
                    <li><strong>DPD:</strong> 0.0027 → 0.0003 (Δ -0.0024)</li>
                    <li><strong>EOD:</strong> 0.0306 → 0.0436 (Δ +0.0130)</li>
                    <li><strong>Cosine:</strong> 0.9787 | <strong>Rank:</strong> 0.8871</li>
                </ul>
                <div style="font-size: 12px; background: #FFFBEB; border: 1px solid #FDE68A; padding: 8px; border-radius: 4px;">
                    <strong>Interpretation:</strong> Improvement in demographic-parity-related metrics did not translate into improved Equalized Odds.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### Frozen Multi-Objective Research Matrix")
    st.markdown(
        "Interactive view of the 6 primary evaluated baseline-vs-reweighing configurations. "
        "Allows model-risk analysts to sort and inspect trade-offs between ΔAUC, ΔEOD, and explanation consistency."
    )
    
    matrix_df = df_final[[
        "Dataset", "Model", "Delta_AUC", "Delta_EOD", "Delta_DPD",
        "Mean_SHAP_Cosine", "Mean_Top5_Overlap", "Mean_Spearman",
        "Fairness_Direction", "Performance_Direction", "Explanation_Consistency_Level"
    ]].copy()
    
    matrix_df["Dataset"] = matrix_df["Dataset"].str.title()
    matrix_df["Model"] = matrix_df["Model"].str.replace("_", " ").str.title()
    
    st.dataframe(
        matrix_df.style.format({
            "Delta_AUC": "{:+.4f}",
            "Delta_EOD": "{:+.4f}",
            "Delta_DPD": "{:+.4f}",
            "Mean_SHAP_Cosine": "{:.4f}",
            "Mean_Top5_Overlap": "{:.4f}",
            "Mean_Spearman": "{:.4f}"
        }),
        use_container_width=True
    )

    render_source_note("results/final_research_results.csv & results/fairness_performance_explanation_matrix.csv")


# ==============================================================================
# TAB 7: METHODOLOGY & LIMITATIONS
# ==============================================================================
with tabs[6]:
    st.subheader("Methodology, Protocol & Research Limitations")
    
    meth_col1, meth_col2 = st.columns(2)
    with meth_col1:
        st.markdown(
            """
            ### Three Levels of Fairness Mitigation Interventions
            The empirical audit examines fairness mitigation across three structural stages:
            
            1. **Pre-processing (Reweighing):**
               - *Intervention level:* Training data sample weighting.
               - *Mechanism:* Computes joint distribution weights $W(S, Y) = \\frac{P(S)P(Y)}{P(S, Y)}$.
               - *Scoring function:* Genuinely refitted continuous model.
               - *SHAP Consistency:* **Primary subject of Phase 4 explanation consistency analysis.**
            
            2. **In-processing (Exponentiated Gradient):**
               - *Intervention level:* Model fitting / optimization.
               - *Mechanism:* Solves a two-player zero-sum game subject to Equalized Odds constraints.
               - *Scoring function:* Outputs randomized binary predictions; continuous ranking scores are **N/A**.
               - *SHAP Consistency:* **Excluded from primary SHAP consistency** because its randomized predictor representation does not provide a straightforward continuous scoring function.
            
            3. **Post-processing (ThresholdOptimizer):**
               - *Intervention level:* Decision threshold post-processing.
               - *Mechanism:* Solves group-specific threshold rules on baseline continuous scores.
               - *Scoring function:* Operates on decision thresholds; does not alter underlying scoring function.
               - *SHAP Consistency:* **Excluded from primary SHAP consistency** because it leaves the baseline scoring function completely unchanged.
            """
        )
    with meth_col2:
        st.markdown(
            """
            ### Stated Empirical Research Limitations
            To prevent over-generalization, the findings must be interpreted within their explicit experimental boundaries:
            
            1. **Benchmark Datasets:** Limited to two canonical public credit datasets (Taiwan Default and Statlog German).
            2. **Fixed Train/Test Split:** Evaluated on deterministic 80/20 stratified splits (`random_state=42`).
            3. **Evaluation Cohorts:** Explanation consistency evaluated on deterministic cohorts ($N=500$ for Taiwan, $N=200$ for German).
            4. **German Sample Size:** German dataset is relatively small ($N=1,000$ total), creating wider confidence intervals.
            5. **Mitigation Scope for SHAP:** Primary explanation-consistency experiment focuses on Reweighing.
            6. **Post-Processing Exclusion:** ThresholdOptimizer is a decision-level rule excluded from scoring SHAP.
            7. **In-Processing Exclusion:** ExponentiatedGradient is excluded from scoring SHAP.
            8. **Synthesis Sample:** Cross-experiment matrix synthesizes six distinct model-dataset pairs.
            9. **Exploratory Association:** Observed relationships between fairness metrics and consistency cannot establish causality.
            10. **Generalizability:** Findings reflect audited configurations and may not generalize to all production portfolios or alternative credit features.
            """
        )

    st.markdown("---")
    st.markdown("### Prominent Disclaimer & Regulatory Boundaries")
    st.warning(
        "**Research Prototype — Not for Automated Credit Decisions**  \n"
        "This system is intended strictly for model auditing and responsible-AI research. "
        "It does not determine whether an individual should receive a loan or credit, "
        "nor does it replace human underwriting judgment or legal compliance verification."
    )
    render_source_note("results/FINAL_RESEARCH_FREEZE.md")
