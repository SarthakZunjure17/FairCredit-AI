"""
FairCredit-AI Dashboard Reusable UI Components and Charting Functions.
Implements restrained, accessible model-risk analytics visualizations.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import streamlit as st


# Color palette for professional financial / model-risk theme
PALETTE = {
    "baseline": "#475569",     # Slate 600
    "mitigated": "#2563EB",    # Royal Blue 600
    "positive": "#0D9488",     # Teal 600
    "negative": "#E11D48",     # Rose 600
    "neutral": "#64748B",      # Slate 500
    "highlight": "#D97706",    # Amber 600
    "reference": "#94A3B8",    # Slate 400
}


def render_header_disclaimer() -> None:
    """Render prominent regulatory and research disclaimers at top of UI."""
    st.markdown(
        """
        <div style="background-color: #F1F5F9; border-left: 5px solid #2563EB; padding: 12px 18px; border-radius: 4px; margin-bottom: 20px;">
            <div style="font-size: 13px; font-weight: 700; color: #1E293B; text-transform: uppercase; letter-spacing: 0.5px;">
                ⚖️ Research Prototype — Model Auditing & Responsible AI Decision Support
            </div>
            <div style="font-size: 12px; color: #475569; margin-top: 4px;">
                <strong>Research Prototype — Not for Automated Credit Decisions.</strong> 
                This system is intended strictly for model auditing and responsible-AI research. 
                It does not determine whether an individual should receive a loan or credit, nor does it operate as an automated compliance or underwriting engine.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_source_note(filepath: str) -> None:
    """Render subtle source traceability footnote."""
    st.caption(f"📁 **Research Data Source:** `{filepath}`")


def render_metric_card(
    title: str,
    baseline_val: Any,
    mitigated_val: Any,
    delta_val: Any,
    higher_is_better: Optional[bool] = None,
    formatter: str = ".4f",
    is_na_metric: bool = False,
    na_explanation: str = "N/A — continuous ranking score not available under the locked evaluation protocol."
) -> None:
    """Render clean model-risk metric card showing Baseline, Mitigated, and Delta."""
    with st.container():
        st.markdown(
            f"""
            <div style="border: 1px solid #E2E8F0; background-color: #FFFFFF; border-radius: 6px; padding: 14px; margin-bottom: 12px;">
                <div style="font-size: 12px; font-weight: 600; color: #64748B; text-transform: uppercase;">{title}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if is_na_metric or str(mitigated_val).strip() == "N/A":
            b_str = f"{baseline_val:{formatter}}" if isinstance(baseline_val, (int, float)) else str(baseline_val)
            st.markdown(f"**Baseline:** `{b_str}`")
            st.markdown(f"**Mitigated:** *{na_explanation}*")
            st.markdown("**Delta:** `N/A`")
            return

        # Numeric values formatting
        try:
            b_f = float(baseline_val)
            m_f = float(mitigated_val)
            d_f = float(delta_val)
            
            b_str = f"{b_f:{formatter}}"
            m_str = f"{m_f:{formatter}}"
            
            sign = "+" if d_f > 0 else ""
            d_str = f"{sign}{d_f:{formatter}}"
            
            # Delta color logic
            if higher_is_better is True:
                delta_color = "#16A34A" if d_f > 0 else ("#DC2626" if d_f < 0 else "#64748B")
            elif higher_is_better is False:
                delta_color = "#16A34A" if d_f < 0 else ("#DC2626" if d_f > 0 else "#64748B")
            else:
                delta_color = "#2563EB"

            cols = st.columns([1, 1, 1])
            with cols[0]:
                st.caption("Baseline")
                st.markdown(f"<span style='font-size: 16px; font-weight: 600; color: #334155;'>{b_str}</span>", unsafe_allow_html=True)
            with cols[1]:
                st.caption("Mitigated")
                st.markdown(f"<span style='font-size: 16px; font-weight: 600; color: #1E293B;'>{m_str}</span>", unsafe_allow_html=True)
            with cols[2]:
                st.caption("Delta (Δ)")
                st.markdown(f"<span style='font-size: 16px; font-weight: 700; color: {delta_color};'>{d_str}</span>", unsafe_allow_html=True)
        except (ValueError, TypeError):
            st.write(f"Baseline: {baseline_val} | Mitigated: {mitigated_val} | Delta: {delta_val}")


def plot_fairness_before_after(
    metric_name: str,
    baseline_val: float,
    mitigated_val: float,
    reference_val: Optional[float] = None
) -> alt.Chart:
    """Generate clean comparison bar chart for fairness metrics."""
    data = pd.DataFrame([
        {"Condition": "Baseline", "Value": baseline_val},
        {"Condition": "Mitigated", "Value": mitigated_val}
    ])
    
    chart = alt.Chart(data).mark_bar(size=45).encode(
        x=alt.X("Condition:N", axis=alt.Axis(title=None, labelAngle=0)),
        y=alt.Y("Value:Q", axis=alt.Axis(title=metric_name)),
        color=alt.Color(
            "Condition:N",
            scale=alt.Scale(domain=["Baseline", "Mitigated"], range=[PALETTE["baseline"], PALETTE["mitigated"]]),
            legend=None
        )
    ).properties(width=280, height=220)
    
    text = chart.mark_text(
        align="center",
        baseline="bottom",
        dy=-5,
        fontSize=12,
        fontWeight="bold"
    ).encode(
        text=alt.Text("Value:Q", format=".4f")
    )
    
    final_chart = chart + text
    
    if reference_val is not None:
        ref_df = pd.DataFrame([{"ref": reference_val}])
        ref_rule = alt.Chart(ref_df).mark_rule(
            strokeDash=[4, 4],
            color=PALETTE["reference"],
            strokeWidth=2
        ).encode(y="ref:Q")
        final_chart = final_chart + ref_rule
        
    return final_chart


def plot_explanation_consistency_summary(df_summary: pd.DataFrame) -> alt.Chart:
    """
    Generate Altair bar chart with 95% bootstrap confidence intervals
    across all 6 primary Baseline vs Reweighing configurations.
    """
    df = df_summary.copy()
    df["Config"] = df["dataset"].str.title() + " — " + df["model"].str.replace("_", " ").str.title()
    
    # Sort order to reflect architectural comparison
    sort_order = [
        "Taiwan — Logistic Regression", "German — Logistic Regression",
        "Taiwan — Xgboost", "German — Xgboost",
        "Taiwan — Random Forest", "German — Random Forest"
    ]
    
    bars = alt.Chart(df).mark_bar(size=25, color=PALETTE["mitigated"]).encode(
        x=alt.X("Config:N", sort=sort_order, axis=alt.Axis(title=None, labelAngle=-30)),
        y=alt.Y("cosine_mean:Q", scale=alt.Scale(domain=[0.85, 1.005]), axis=alt.Axis(title="Mean Cosine Similarity"))
    )
    
    error_bars = alt.Chart(df).mark_errorbar(extent="ci", color="#0F172A", thickness=2).encode(
        x=alt.X("Config:N", sort=sort_order),
        y=alt.Y("cosine_ci_95_low:Q", axis=alt.Axis(title="")),
        y2="cosine_ci_95_high:Q"
    )
    
    text = alt.Chart(df).mark_text(
        align="center",
        baseline="bottom",
        dy=-12,
        fontSize=11,
        fontWeight="bold"
    ).encode(
        x=alt.X("Config:N", sort=sort_order),
        y=alt.Y("cosine_mean:Q"),
        text=alt.Text("cosine_mean:Q", format=".4f")
    )
    
    return (bars + error_bars + text).properties(
        title="Mean SHAP Cosine Similarity across Evaluated Configurations (with 95% Bootstrap CIs)",
        width=700,
        height=340
    )


def plot_individual_shap_horizontal_bars(
    df_features: pd.DataFrame,
    output_space_label: str,
    top_n: int = 10
) -> plt.Figure:
    """
    Generate clean Matplotlib horizontal bar chart showing baseline vs reweighed
    SHAP contributions for top positive and negative features of a selected applicant.
    """
    # Sort by absolute baseline attribution to identify key decision drivers
    df_sorted = df_features.copy()
    df_sorted["abs_base"] = df_sorted["baseline_shap"].abs()
    df_sorted = df_sorted.sort_values(by="abs_base", ascending=False).head(top_n)
    df_sorted = df_sorted.sort_values(by="baseline_shap", ascending=True) # bottom to top
    
    y = np.arange(len(df_sorted))
    bar_height = 0.35
    
    fig, ax = plt.subplots(figsize=(10, max(5, int(len(df_sorted) * 0.45))), dpi=120)
    
    # Baseline bars
    ax.barh(
        y - bar_height / 2,
        df_sorted["baseline_shap"],
        height=bar_height,
        color=PALETTE["baseline"],
        alpha=0.85,
        label="Baseline Attribution"
    )
    
    # Reweighted bars
    ax.barh(
        y + bar_height / 2,
        df_sorted["reweighed_shap"],
        height=bar_height,
        color=PALETTE["mitigated"],
        alpha=0.85,
        label="Reweighted Attribution"
    )
    
    # Clean feature names (clean scikit-learn OneHotEncoder / StandardScaler prefixes if present)
    clean_labels = [
        f.replace("cat__", "").replace("num__", "").replace("remainder__", "")
        for f in df_sorted["feature"]
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(clean_labels, fontsize=10)
    
    ax.axvline(0, color="#64748B", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel(f"SHAP Attribution toward Default Risk ({output_space_label})", fontsize=11, fontweight="medium")
    ax.set_title(f"Individual Feature SHAP Comparison: Top {top_n} Attributions", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="best", frameon=True, fontsize=10)
    
    # Grid and spines
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    
    plt.tight_layout()
    return fig
